"""
SELF-TRAIN the robot-silhouette mask head on REAL (the v1 synth-only head was sim2real mis-placed ->
render-compare HURT -0.141). Pseudo real mask = render the kinematic SOLVER's mesh silhouette on
high-confidence real frames (good pose -> correctly-placed mask), fine-tune the mask head on real
appearance, mixed 1:1 with synth GT masks (anti-forgetting). Appearance-based -> generalizes to the
foreshortened frames render-compare needs. Output: a real-adapted mask head for render-compare.
"""
import argparse, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE); sys.path.append(os.path.join(HERE, '../Eval'))
from model_angle import AnglePredictor
from model_v4 import ViTKeypointHead, panda_forward_kinematics
from dataset import PoseEstimationDataset
from refine_eval import scale_K
from solve_pose_kinematic import solve_batch
from silhouette_mesh_probe import (load_obj_verts, robot_pointcloud, render_mesh, kabsch_batch, mesh_path,
                                   MESH_DIR, LINK_MESH)
from selftrain_pseudo import IdxWrap

KP = ['panda_link0', 'panda_link2', 'panda_link3', 'panda_link4', 'panda_link6', 'panda_link7', 'panda_hand']


def dice_loss(p, t, eps=1.0):
    p = p.flatten(1); t = t.flatten(1)
    return (1 - (2 * (p * t).sum(1) + eps) / (p.sum(1) + t.sum(1) + eps)).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--real-dir', required=True)
    ap.add_argument('--synth-dir', default='../Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_train_dr')
    ap.add_argument('--detector', required=True); ap.add_argument('--angle-head', required=True)
    ap.add_argument('--rot-head', default=None)
    ap.add_argument('--mask-head', required=True, help='v1 synth mask head to start from')
    ap.add_argument('--model-name', default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    ap.add_argument('--image-size', type=int, default=512); ap.add_argument('--out-res', type=int, default=256)
    ap.add_argument('--batch-size', type=int, default=12); ap.add_argument('--adapt-frac', type=float, default=0.7)
    ap.add_argument('--adapt-cap', type=int, default=4000); ap.add_argument('--conf-keep', type=float, default=0.55)
    ap.add_argument('--epochs', type=int, default=4); ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--iters', type=int, default=150)
    ap.add_argument('--output-dir', default='./outputs_mask/mask_selftrain')
    args = ap.parse_args()

    device = torch.device('cuda'); assert torch.cuda.is_available()
    os.makedirs(args.output_dir, exist_ok=True); S, R = args.image_size, args.out_res
    mesh_verts = [((nm, fi), torch.from_numpy(load_obj_verts(mesh_path(nm))).to(device))
                  for nm, fi in LINK_MESH]

    mp = AnglePredictor(args.model_name, S, head_type='mlp',
                        with_rotation=args.rot_head is not None, with_translation=args.rot_head is not None).to(device).eval()
    sd = torch.load(args.detector, map_location=device); sd = {k.replace('module.', ''): v for k, v in sd.items()}
    mp.load_state_dict({k: v for k, v in sd.items() if k in mp.state_dict() and v.shape == mp.state_dict()[k].shape}, strict=False)
    mp.angle_head.load_state_dict(torch.load(args.angle_head, map_location=device))
    if args.rot_head:
        mp.rot_head.load_state_dict(torch.load(args.rot_head, map_location=device))
    for p in mp.parameters():
        p.requires_grad = False

    mask_head = ViTKeypointHead(input_dim=mp.backbone.model.config.hidden_size, num_joints=1, heatmap_size=(R, R)).to(device)
    mask_head.load_state_dict(torch.load(args.mask_head, map_location=device))

    real = PoseEstimationDataset(args.real_dir, keypoint_names=KP, image_size=(S, S), heatmap_size=(S, S),
                                 augment=False, include_angles=True)
    N = len(real.samples); cut = int(args.adapt_frac * N); adapt = list(range(cut))
    if args.adapt_cap > 0 and len(adapt) > args.adapt_cap:
        st = max(1, len(adapt) // args.adapt_cap); adapt = adapt[::st][:args.adapt_cap]
    rw = IdxWrap(real)
    aloader = DataLoader(Subset(rw, adapt), batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # ---- PSEUDO-GEN: solver mesh silhouette on high-conf real frames ----
    pseudo = torch.zeros(N, R, R); keep = torch.zeros(N, dtype=torch.bool)
    for b in tqdm(aloader, desc='pseudo-mask'):
        idx = b['idx']; img = b['image'].to(device)
        K = scale_K(b['camera_K'], b['original_size'], S).to(device)
        with torch.no_grad():
            out = mp(img, K)
        R_init = out.get('rot_matrix') if args.rot_head else None
        with torch.enable_grad():                          # solve_batch does an internal loss.backward()
            theta, kp_cam, _ = solve_batch(out['keypoints_2d'], out['confidence'], K, fix_joint7=True,
                                           iters=args.iters, lr=2e-2, img_size=S, device=device, prior_w=0.0,
                                           theta_init=out['joint_angles'], conf_gate=0.05, R_init=R_init)
        with torch.no_grad():
            theta = theta.detach(); kp_cam = kp_cam.detach()
            Rm, tm = kabsch_batch(panda_forward_kinematics(theta), kp_cam)
            mask = render_mesh(robot_pointcloud(theta, mesh_verts), Rm, tm, K, R, S)
            mc = out['confidence'].mean(1)
        for j in range(img.shape[0]):
            if float(mc[j]) > args.conf_keep:
                pseudo[int(idx[j])] = (mask[j] > 0.4).float().cpu(); keep[int(idx[j])] = True
    kept = [i for i in adapt if keep[i]]
    print(f"[PSEUDO] kept {len(kept)}/{len(adapt)} high-conf real frames (conf>{args.conf_keep})", flush=True)
    if len(kept) < 50:
        print('too few; abort'); return
    pseudo = pseudo.to(device)

    synth = PoseEstimationDataset(args.synth_dir, keypoint_names=KP, image_size=(S, S), heatmap_size=(S, S),
                                  augment=False, include_angles=True)
    ploader = DataLoader(Subset(rw, kept), batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    sloader = DataLoader(synth, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    def synth_mask(b):
        ang = b['angles'].to(device).float().clone(); ang[:, 6] = 0.0
        K = scale_K(b['camera_K'], b['original_size'], S).to(device)
        Rm, tm = kabsch_batch(panda_forward_kinematics(ang), b['keypoints_3d'].to(device).float())
        with torch.no_grad():
            return (render_mesh(robot_pointcloud(ang, mesh_verts), Rm, tm, K, R, S) > 0.4).float()

    opt = torch.optim.AdamW(mask_head.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    for ep in range(args.epochs):
        mask_head.train(); sit = iter(sloader)
        for rb in tqdm(ploader, desc=f'Ep{ep} mask-selftrain', leave=False):
            img = rb['image'].to(device)
            with torch.no_grad():
                tok = mp.backbone(img)
            logit = mask_head(tok).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(logit, pseudo[rb['idx']]) + dice_loss(torch.sigmoid(logit), pseudo[rb['idx']])
            try: sb = next(sit)
            except StopIteration: sit = iter(sloader); sb = next(sit)
            simg = sb['image'].to(device)
            with torch.no_grad():
                stok = mp.backbone(simg); sm = synth_mask(sb)
            slog = mask_head(stok).squeeze(1)
            loss = loss + F.binary_cross_entropy_with_logits(slog, sm) + dice_loss(torch.sigmoid(slog), sm)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        torch.save(mask_head.state_dict(), os.path.join(args.output_dir, 'best_mask_head.pth'))
        print(f"Ep{ep} done (saved)", flush=True)
    print(f"[RESULT] mask self-train done -> {args.output_dir}/best_mask_head.pth", flush=True)


if __name__ == '__main__':
    main()
