"""
Stage 3b — pseudo-label self-training that adapts the ROTATION HEAD (not just the angle head).

Motivation: angle-only pseudo-label-on-crop was tried 3x and is marginal/redundant with crop. The
DIAGNOSED realsense/orb bottleneck is R (gauge / foreshortened base-yaw), and the proven realsense
lever is the rotation head used as solver R_init (+0.117). But every prior self-train run kept the
rot head FROZEN and only adapted the angle head. Here we adapt the rot head (+ angle head) toward the
SOLVER's refined R on high-confidence real frames — teaching the head's real-appearance→R_init map to
land in the right basin on the foreshortened tail.

Protocol (no test contamination): per camera, contiguous split adapt=first 70% / eval=last 30%.
1. PSEUDO-GEN on adapt: frozen detector + angle/rot head + kinematic solver (return_pose=True) -> per
   frame: solver-refined angles θ* AND solver-refined rotation R* (robot->camera). Keep high-conf frames.
2. FINETUNE angle_head + rot_head: angle sin/cos+FK toward θ* (+ synth GT anti-forget on the angle head),
   AND a chordal rotation loss toward R* on the pseudo-real frames. rot head gets NO synth anti-forget by
   design (per-camera deployment) — the held-out early-stop is the never-worse safety net.
3. EVAL held-out: ADD-AUC adapted vs baseline (same frames), using the (adapted) rot head as R_init.
"""
import argparse, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE); sys.path.append(os.path.join(HERE, '../Eval'))
from model_angle import AnglePredictor, panda_forward_kinematics      # noqa
from dataset import PoseEstimationDataset                              # noqa
from solve_pose_kinematic import solve_batch                          # noqa

KP = ['link0', 'link2', 'link3', 'link4', 'link6', 'link7', 'hand']


def scale_K(camera_K, original_size, hm):
    K = camera_K.clone().float()
    for b in range(K.shape[0]):
        ow, oh = float(original_size[b][0]), float(original_size[b][1])
        K[b, 0, 0] *= hm / ow; K[b, 1, 1] *= hm / oh
        K[b, 0, 2] *= hm / ow; K[b, 1, 2] *= hm / oh
    return K


def add_auc(adds_m, thr=0.1):
    if len(adds_m) == 0:
        return 0.0
    d = 1e-5; ts = np.arange(0.0, thr, d)
    counts = (np.asarray(adds_m)[None, :] <= ts[:, None]).sum(1) / float(len(adds_m))
    return float(np.trapz(counts, dx=d) / thr)


class IdxWrap(torch.utils.data.Dataset):
    def __init__(self, ds): self.ds = ds
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        s = self.ds[i]; s['idx'] = i; return s


def build_model(args, device):
    m = AnglePredictor(args.model_name, args.image_size, head_type='mlp',
                       with_rotation=True, with_translation=args.with_translation).to(device)
    sd = torch.load(args.detector, map_location=device); sd = {k.replace('module.', ''): v for k, v in sd.items()}
    msd = m.state_dict()
    m.load_state_dict({k: v for k, v in sd.items() if k in msd and v.shape == msd[k].shape}, strict=False)
    m.angle_head.load_state_dict(torch.load(args.angle_head, map_location=device))
    m.rot_head.load_state_dict(torch.load(args.rot_head, map_location=device))
    m.freeze_detector()
    return m


def run_pipeline(m, batch, device, image_size, iters=150):
    img = batch['image'].to(device)
    K = scale_K(batch['camera_K'], batch['original_size'], image_size).to(device)
    with torch.no_grad():
        out = m(img, K)
    R_init = out.get('rot_matrix')
    refined, kp_cam, reproj, R, t = solve_batch(out['keypoints_2d'], out['confidence'], K, fix_joint7=True,
                                                iters=iters, lr=2e-2, img_size=image_size, device=device,
                                                prior_w=0.0, theta_init=out['joint_angles'],
                                                conf_gate=0.05, R_init=R_init, return_pose=True)
    return refined.detach(), kp_cam.detach(), out['confidence'], reproj.detach(), R.detach(), t.detach()


def evaluate(m, loader, device, image_size):
    m.eval(); adds = []
    for batch in tqdm(loader, desc='eval', leave=False):
        _, kp_cam, _, _, _, _ = run_pipeline(m, batch, device, image_size)
        gt3d = batch['keypoints_3d'].to(device); valid = (gt3d.abs().sum(-1) > 0)
        per_j = (kp_cam - gt3d).norm(dim=-1)
        for b in range(kp_cam.shape[0]):
            if valid[b].any():
                adds.append(float(per_j[b][valid[b]].mean().item()))
    return add_auc(adds), float(np.mean(adds) * 1000) if adds else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--real-dir', required=True)
    ap.add_argument('--synth-dir', default='../Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_train_dr')
    ap.add_argument('--detector', required=True)
    ap.add_argument('--angle-head', required=True)
    ap.add_argument('--rot-head', required=True)
    ap.add_argument('--with-translation', action='store_true', default=True,
                    help='locked rot head predicts R+t; keep True to match its checkpoint shape')
    ap.add_argument('--model-name', default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    ap.add_argument('--image-size', type=int, default=512)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--adapt-frac', type=float, default=0.7)
    ap.add_argument('--adapt-cap', type=int, default=0)
    ap.add_argument('--eval-frames', type=int, default=800)
    ap.add_argument('--conf-keep', type=float, default=0.5)
    ap.add_argument('--reproj-keep', type=float, default=0.0)
    ap.add_argument('--crop', action='store_true')
    ap.add_argument('--crop-margin', type=float, default=1.5)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--rot-lambda', type=float, default=1.0, help='weight on the chordal rot loss toward solver R*')
    ap.add_argument('--synth-ratio', type=float, default=1.0)
    ap.add_argument('--output-dir', default='./outputs_selftrain')
    args = ap.parse_args()

    device = torch.device('cuda'); assert torch.cuda.is_available()
    os.makedirs(args.output_dir, exist_ok=True)
    m = build_model(args, device)

    real_full = PoseEstimationDataset(args.real_dir, keypoint_names=KP,
                                      image_size=(args.image_size, args.image_size),
                                      heatmap_size=(args.image_size, args.image_size),
                                      augment=False, include_angles=True, sigma=2.5,
                                      crop_to_robot=args.crop, crop_margin=args.crop_margin)
    N = len(real_full.samples); cut = int(args.adapt_frac * N)
    adapt_idx = list(range(cut)); eval_idx = list(range(cut, N))
    if args.adapt_cap > 0 and len(adapt_idx) > args.adapt_cap:
        st = max(1, len(adapt_idx) // args.adapt_cap); adapt_idx = adapt_idx[::st][:args.adapt_cap]
    es = max(1, len(eval_idx) // args.eval_frames); eval_idx = eval_idx[::es][:args.eval_frames]
    print(f"real {os.path.basename(args.real_dir)}: N={N} adapt={len(adapt_idx)} eval={len(eval_idx)}", flush=True)

    real_wrap = IdxWrap(real_full)
    adapt_loader = DataLoader(Subset(real_wrap, adapt_idx), batch_size=args.batch_size,
                              shuffle=False, num_workers=8, pin_memory=True)
    eval_loader = DataLoader(Subset(real_full, eval_idx), batch_size=args.batch_size,
                             shuffle=False, num_workers=8, pin_memory=True)

    base_auc, base_add = evaluate(m, eval_loader, device, args.image_size)
    print(f"[BASELINE] held-out ADD-AUC={base_auc:.4f} mean ADD={base_add:.1f}mm", flush=True)

    # ---- PSEUDO-GEN: store solver θ* and R* on kept frames ----
    pseudo = torch.zeros(N, 7); pseudo_R = torch.zeros(N, 3, 3); keep = torch.zeros(N, dtype=torch.bool)
    m.eval()
    for batch in tqdm(adapt_loader, desc='pseudo-gen'):
        idx = batch['idx']
        refined, _, conf, reproj, R, _ = run_pipeline(m, batch, device, args.image_size)
        meanconf = conf.mean(dim=1).cpu(); reproj = reproj.cpu(); R = R.cpu()
        for b in range(refined.shape[0]):
            i = int(idx[b])
            ok = float(meanconf[b]) > args.conf_keep
            if args.reproj_keep > 0:
                ok = float(reproj[b]) < args.reproj_keep and float(meanconf[b]) > 0.3
            if ok:
                pseudo[i] = refined[b].cpu(); pseudo_R[i] = R[b]; keep[i] = True
    kept = [i for i in adapt_idx if keep[i]]
    filt = f"reproj < {args.reproj_keep}px" if args.reproj_keep > 0 else f"mean conf > {args.conf_keep}"
    print(f"[PSEUDO] kept {len(kept)}/{len(adapt_idx)} adapt frames ({filt})", flush=True)
    if len(kept) < 50:
        print("too few pseudo frames; abort"); return

    synth = PoseEstimationDataset(args.synth_dir, keypoint_names=KP,
                                  image_size=(args.image_size, args.image_size),
                                  heatmap_size=(args.image_size, args.image_size),
                                  augment=True, aug_level='strong', include_angles=True, sigma=2.5,
                                  crop_to_robot=args.crop, crop_margin=args.crop_margin)
    pseudo_loader = DataLoader(Subset(real_wrap, kept), batch_size=args.batch_size, shuffle=True,
                               num_workers=8, pin_memory=True, drop_last=True)
    synth_loader = DataLoader(synth, batch_size=args.batch_size, shuffle=True,
                              num_workers=8, pin_memory=True, drop_last=True)
    opt = torch.optim.AdamW(list(m.angle_head.parameters()) + list(m.rot_head.parameters()),
                            lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    pseudo = pseudo.to(device); pseudo_R = pseudo_R.to(device)

    best_auc = base_auc
    best_sd = ({k: v.clone() for k, v in m.angle_head.state_dict().items()},
               {k: v.clone() for k, v in m.rot_head.state_dict().items()})
    for epoch in range(args.epochs):
        m.angle_head.train(); m.rot_head.train()
        synth_it = iter(synth_loader)
        for rbatch in tqdm(pseudo_loader, desc=f'Ep{epoch} finetune', leave=False):
            img = rbatch['image'].to(device); K = scale_K(rbatch['camera_K'], rbatch['original_size'], args.image_size).to(device)
            pa = pseudo[rbatch['idx']]; pR = pseudo_R[rbatch['idx']]      # (B,7),(B,3,3)
            od = m(img, K); sc = od['sin_cos']
            pa6 = pa[:, :6]
            gt_sc = torch.stack([torch.sin(pa6), torch.cos(pa6)], dim=-1)
            loss = F.smooth_l1_loss(sc, gt_sc)
            loss = loss + 10.0 * F.mse_loss(panda_forward_kinematics(od['joint_angles']), panda_forward_kinematics(pa))
            # --- rotation pseudo-supervision: chordal distance to solver R* ---
            loss = loss + args.rot_lambda * F.mse_loss(od['rot_matrix'], pR)
            # --- synth anti-forget on the angle head (real GT angles) ---
            if args.synth_ratio > 0:
                try: sb = next(synth_it)
                except StopIteration: synth_it = iter(synth_loader); sb = next(synth_it)
                si = sb['image'].to(device); sK = scale_K(sb['camera_K'], sb['original_size'], args.image_size).to(device)
                sg = sb['angles'].to(device).clone(); sg[:, 6] = 0.0
                sod = m(si, sK); ssc = sod['sin_cos']; sg6 = sg[:, :6]
                sgt = torch.stack([torch.sin(sg6), torch.cos(sg6)], dim=-1)
                sloss = F.smooth_l1_loss(ssc, sgt) + 10.0 * F.mse_loss(
                    panda_forward_kinematics(sod['joint_angles']), panda_forward_kinematics(sg))
                loss = loss + args.synth_ratio * sloss
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        auc, add = evaluate(m, eval_loader, device, args.image_size)
        flag = ''
        if auc > best_auc:
            best_auc = auc; flag = ' *'
            best_sd = ({k: v.clone() for k, v in m.angle_head.state_dict().items()},
                       {k: v.clone() for k, v in m.rot_head.state_dict().items()})
        print(f"Ep{epoch} | held-out ADD-AUC={auc:.4f} mean ADD={add:.1f}mm (base {base_auc:.4f}){flag}", flush=True)

    torch.save(best_sd[0], os.path.join(args.output_dir, 'best_selftrain_head.pth'))
    torch.save(best_sd[1], os.path.join(args.output_dir, 'best_selftrain_rot.pth'))
    print(f"\n[RESULT] {os.path.basename(args.real_dir)}: baseline {base_auc:.4f} -> self-train {best_auc:.4f} "
          f"(delta {best_auc-base_auc:+.4f})", flush=True)


if __name__ == '__main__':
    main()
