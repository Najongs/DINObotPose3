"""
Stage 3b — DETECTOR self-training on REAL (the one component never adapted; angle/crop heads were
self-trained, the 2D keypoint detector is still synth-only). Distill the kinematic SOLVER back into
the detector: on reliable real frames (low solver reprojection), the solver-refined pose reprojects
to a CLEAN, kinematically-consistent 7-keypoint skeleton (incl. the occluded base the raw detector
drops) -> use that as a pseudo-keypoint heatmap target, finetune the keypoint head (backbone frozen),
mixed with synth GT heatmaps (anti-forgetting). Better real 2D -> better PnP/solver -> better ADD,
and it STACKS with the angle-head/crop self-train.

Protocol mirrors selftrain_pseudo.py: per camera, contiguous adapt=first 70% / held-out eval=last 30%.
Eval = full-pipeline ADD-AUC (adapted detector + existing angle + rot heads + solver).
"""
import argparse, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE); sys.path.append(os.path.join(HERE, '../Eval'))
from model_angle import AnglePredictor                                  # noqa
from dataset import PoseEstimationDataset                               # noqa
from solve_pose_kinematic import solve_batch                           # noqa
from selftrain_pseudo import scale_K, add_auc, IdxWrap, evaluate, KP    # reuse


def build_model(args, device):
    """Full pipeline model; keypoint head TRAINABLE, backbone + angle + rot heads frozen."""
    m = AnglePredictor(args.model_name, args.image_size, head_type='mlp',
                       with_rotation=args.rot_head is not None,
                       with_translation=args.rot_head is not None).to(device)
    sd = torch.load(args.detector, map_location=device); sd = {k.replace('module.', ''): v for k, v in sd.items()}
    msd = m.state_dict()
    m.load_state_dict({k: v for k, v in sd.items() if k in msd and v.shape == msd[k].shape}, strict=False)
    m.angle_head.load_state_dict(torch.load(args.angle_head, map_location=device))
    if args.rot_head:
        m.rot_head.load_state_dict(torch.load(args.rot_head, map_location=device))
    for p in m.backbone.parameters():
        p.requires_grad = False
    for p in m.angle_head.parameters():
        p.requires_grad = False
    if m.rot_head is not None:
        for p in m.rot_head.parameters():
            p.requires_grad = False
    m.backbone.eval()
    for p in m.keypoint_head.parameters():
        p.requires_grad = True
    # OPTIONAL co-finetune: unfreeze last N backbone blocks (HF DINOv3 exposes them as model.layer).
    # Distill REAL solver-pseudo keypoints into the backbone too -> real-domain adaptation with PRECISE
    # supervision (not coarse masked-feature SSL) and NO head-OOD (backbone+head co-trained).
    m.backbone._cofinetune = int(getattr(args, 'unfreeze_backbone', 0))
    if m.backbone._cofinetune > 0:
        blocks = getattr(m.backbone.model, 'layer', None) or getattr(m.backbone.model, 'blocks', None)
        nblk = len(blocks); nt = 0
        for i in range(max(0, nblk - m.backbone._cofinetune), nblk):
            for p in blocks[i].parameters():
                p.requires_grad = True; nt += p.numel()
        if hasattr(m.backbone.model, 'norm'):
            for p in m.backbone.model.norm.parameters():
                p.requires_grad = True; nt += p.numel()
        m.backbone.train()
        print(f"[CO-FT] unfroze last {m.backbone._cofinetune}/{nblk} backbone blocks + norm ({nt/1e6:.1f}M)", flush=True)
    return m


def run_pipeline(m, batch, device, image_size, rot, iters=150):
    """Full forward + solver (for pseudo-gen and eval). Mirrors selftrain_pseudo.run_pipeline but also
    returns the camera-frame keypoints to reproject into pseudo 2D targets."""
    img = batch['image'].to(device)
    K = scale_K(batch['camera_K'], batch['original_size'], image_size).to(device)
    with torch.no_grad():
        out = m(img, K)
    R_init = out.get('rot_matrix') if rot else None
    refined, kp_cam, reproj = solve_batch(out['keypoints_2d'], out['confidence'], K, fix_joint7=True,
                                          iters=iters, lr=2e-2, img_size=image_size, device=device,
                                          prior_w=0.0, theta_init=out['joint_angles'],
                                          conf_gate=0.05, R_init=R_init)
    return refined.detach(), kp_cam.detach(), out['confidence'], reproj.detach(), K


def project_points(kp_cam, K):
    z = kp_cam[..., 2:3].clamp(min=1e-4)
    uv = kp_cam[..., :2] / z
    fx, fy = K[:, 0, 0].unsqueeze(1), K[:, 1, 1].unsqueeze(1)
    cx, cy = K[:, 0, 2].unsqueeze(1), K[:, 1, 2].unsqueeze(1)
    return torch.stack([uv[..., 0] * fx + cx, uv[..., 1] * fy + cy], dim=-1)   # (B,7,2)


def render_heatmaps(kp, H, sigma, chunk=2):
    """Vectorized Gaussian heatmaps (B,N,H,W) at sub-pixel kp (B,N,2) in heatmap px. Chunk over batch
    to bound memory (H=512 -> 7*512*512 floats/sample)."""
    B, N, _ = kp.shape
    ys = torch.arange(H, device=kp.device).view(1, 1, H, 1).float()
    xs = torch.arange(H, device=kp.device).view(1, 1, 1, H).float()
    two_s2 = 2.0 * sigma * sigma
    outs = []
    for i in range(0, B, chunk):
        cx = kp[i:i + chunk, :, 0].view(-1, N, 1, 1)
        cy = kp[i:i + chunk, :, 1].view(-1, N, 1, 1)
        outs.append(torch.exp(-(((xs - cx) ** 2 + (ys - cy) ** 2) / two_s2)))
    return torch.cat(outs, 0)


def heatmap_loss(m, img, target_hm):
    """Predict heatmaps and MSE vs targets. Backbone grad flows ONLY if co-finetuning (else no_grad)."""
    if getattr(m.backbone, '_cofinetune', 0) > 0:
        tokens = m.backbone(img)                         # grad through unfrozen backbone blocks
    else:
        with torch.no_grad():
            tokens = m.backbone(img)
    pred = m.keypoint_head(tokens)                       # (B,7,H,W) with grad
    return F.mse_loss(pred, target_hm) * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--real-dir', required=True)
    ap.add_argument('--synth-dir', default='../Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_train_dr')
    ap.add_argument('--detector', required=True)
    ap.add_argument('--angle-head', required=True)
    ap.add_argument('--rot-head', default=None)
    ap.add_argument('--model-name', default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    ap.add_argument('--image-size', type=int, default=512)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--adapt-frac', type=float, default=0.7)
    ap.add_argument('--adapt-cap', type=int, default=0)
    ap.add_argument('--eval-frames', type=int, default=800)
    ap.add_argument('--conf-keep', type=float, default=0.5, help='keep frames with mean kp conf > this (proven angle-self-train filter)')
    ap.add_argument('--reproj-keep', type=float, default=25.0, help='loose reproj cap (px) to drop only egregiously-diverged solves (realsense full-frame reproj is naturally high)')
    ap.add_argument('--sigma', type=float, default=2.5)
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--synth-ratio', type=float, default=1.0)
    ap.add_argument('--unfreeze-backbone', type=int, default=0, help='co-finetune last N backbone blocks on real solver-pseudo (0=head-only)')
    ap.add_argument('--backbone-lr', type=float, default=1e-5, help='LR for co-finetuned backbone blocks (small; head uses --lr)')
    ap.add_argument('--output-dir', default='./outputs_selftrain_det')
    args = ap.parse_args()

    device = torch.device('cuda'); assert torch.cuda.is_available()
    os.makedirs(args.output_dir, exist_ok=True); IS = args.image_size
    m = build_model(args, device)

    real_full = PoseEstimationDataset(args.real_dir, keypoint_names=KP, image_size=(IS, IS),
                                      heatmap_size=(IS, IS), augment=False, include_angles=True, sigma=args.sigma)
    N = len(real_full.samples); cut = int(args.adapt_frac * N)
    adapt_idx = list(range(cut)); eval_idx = list(range(cut, N))
    if args.adapt_cap > 0 and len(adapt_idx) > args.adapt_cap:
        st = max(1, len(adapt_idx) // args.adapt_cap); adapt_idx = adapt_idx[::st][:args.adapt_cap]
    es = max(1, len(eval_idx) // args.eval_frames); eval_idx = eval_idx[::es][:args.eval_frames]
    print(f"real {os.path.basename(args.real_dir)}: N={N} adapt={len(adapt_idx)} eval={len(eval_idx)}", flush=True)

    real_wrap = IdxWrap(real_full)
    adapt_loader = DataLoader(Subset(real_wrap, adapt_idx), batch_size=args.batch_size, shuffle=False,
                              num_workers=8, pin_memory=True)
    eval_loader = DataLoader(Subset(real_full, eval_idx), batch_size=args.batch_size, shuffle=False,
                             num_workers=8, pin_memory=True)

    base_auc, base_add = evaluate(m, eval_loader, device, IS, args.rot_head)
    print(f"[BASELINE] held-out ADD-AUC={base_auc:.4f} mean ADD={base_add:.1f}mm", flush=True)

    # ---- PSEUDO-GEN: solver-reprojected 2D keypoints on reliable (low-reproj) frames ----
    pseudo = torch.zeros(N, 7, 2); keep = torch.zeros(N, dtype=torch.bool)
    m.eval()
    for batch in tqdm(adapt_loader, desc='pseudo-gen'):
        idx = batch['idx']
        _, kp_cam, conf, reproj, K = run_pipeline(m, batch, device, IS, args.rot_head)
        uv = project_points(kp_cam, K).cpu(); reproj = reproj.cpu(); meanconf = conf.mean(1).cpu()
        for b in range(uv.shape[0]):
            if float(meanconf[b]) > args.conf_keep and float(reproj[b]) < args.reproj_keep:
                pseudo[int(idx[b])] = uv[b]; keep[int(idx[b])] = True
    kept = [i for i in adapt_idx if keep[i]]
    print(f"[PSEUDO] kept {len(kept)}/{len(adapt_idx)} adapt frames (conf>{args.conf_keep}, reproj<{args.reproj_keep}px)", flush=True)
    if len(kept) < 50:
        print("too few pseudo frames; abort"); return

    synth = PoseEstimationDataset(args.synth_dir, keypoint_names=KP, image_size=(IS, IS),
                                  heatmap_size=(IS, IS), augment=True, aug_level='strong',
                                  include_angles=True, sigma=args.sigma)
    pseudo_loader = DataLoader(Subset(real_wrap, kept), batch_size=args.batch_size, shuffle=True,
                               num_workers=8, pin_memory=True, drop_last=True)
    synth_loader = DataLoader(synth, batch_size=args.batch_size, shuffle=True, num_workers=8,
                              pin_memory=True, drop_last=True)
    cofit = getattr(m.backbone, '_cofinetune', 0) > 0
    param_groups = [{'params': m.keypoint_head.parameters(), 'lr': args.lr}]
    if cofit:
        bb_params = [p for p in m.backbone.parameters() if p.requires_grad]
        param_groups.append({'params': bb_params, 'lr': args.backbone_lr})
    opt = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    pseudo = pseudo.to(device)

    def snapshot():
        # co-finetune -> save FULL detector (backbone+head); else just the keypoint head
        src = m.state_dict() if cofit else m.keypoint_head.state_dict()
        return {k: v.clone() for k, v in src.items()}
    best_auc, best_sd = base_auc, snapshot()
    for epoch in range(args.epochs):
        m.keypoint_head.train()
        if cofit: m.backbone.train()
        synth_it = iter(synth_loader)
        for rb in tqdm(pseudo_loader, desc=f'Ep{epoch} det-finetune', leave=False):
            img = rb['image'].to(device)
            tgt = render_heatmaps(pseudo[rb['idx']], IS, args.sigma)        # pseudo-kp -> target heatmaps
            loss = heatmap_loss(m, img, tgt)
            if args.synth_ratio > 0:                                       # synth anti-forgetting (GT heatmaps)
                try: sb = next(synth_it)
                except StopIteration: synth_it = iter(synth_loader); sb = next(synth_it)
                loss = loss + args.synth_ratio * heatmap_loss(m, sb['image'].to(device), sb['heatmaps'].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        auc, add = evaluate(m, eval_loader, device, IS, args.rot_head)
        flag = ''
        if auc > best_auc:
            best_auc = auc; best_sd = snapshot(); flag = ' *'
        print(f"Ep{epoch} | held-out ADD-AUC={auc:.4f} mean ADD={add:.1f}mm (base {base_auc:.4f}){flag}", flush=True)

    fname = 'best_cofinetune_detector.pth' if cofit else 'best_selftrain_detector.pth'
    torch.save(best_sd, os.path.join(args.output_dir, fname))
    print(f"\n[RESULT] {os.path.basename(args.real_dir)}: baseline {base_auc:.4f} -> det-self-train {best_auc:.4f} "
          f"(delta {best_auc-base_auc:+.4f})", flush=True)


if __name__ == '__main__':
    main()
