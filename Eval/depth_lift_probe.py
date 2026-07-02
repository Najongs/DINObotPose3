"""
Ceiling probe for the DEPTH lever. The realsense failures are foreshortened poses where 2D is
degenerate for base-yaw J0 (see viz_failures). Depth would add the missing dimension. We have GT
per-keypoint depth (gt_3d[:,2]); this tests: backproject detected-2D + depth -> 3D points, then
solve pose with a WELL-POSED 3D objective (min ||R FK(theta)+t - P3d||, no reprojection
degeneracy). If realsense ADD/J0 collapse, depth is THE lever and a mono-depth model is worth it.
Also tests NOISY depth (z*(1+n)) to gauge how accurate a learned depth needs to be.

configs:
  base_2d       : current 2D-reprojection solve (pnp_drop=3)              [no depth]
  depth_oracle  : 3D solve, detected-2D + GT depth
  depth_n5/n10  : 3D solve, detected-2D + depth*(1+N(0,.05/.10))          [mono-depth realism]
"""
import argparse, math, os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

TRAIN = os.path.abspath(os.path.join(os.path.dirname(__file__), '../TRAIN'))
sys.path.append(TRAIN); sys.path.append(os.path.dirname(__file__))
from model_angle import AnglePredictor, kabsch_batch
from model_v4 import panda_forward_kinematics
from inference_4tier_eval import EvalDataset, compute_add_auc
from solve_pose_kinematic import (solve_batch, make_limits, theta_to_p, p_to_theta,
                                  rot6d_to_matrix, matrix_to_rot6d, PANDA_JOINT_MEAN)
from refine_eval import scale_K

KPN = ['panda_link0', 'panda_link2', 'panda_link3', 'panda_link4', 'panda_link6', 'panda_link7', 'panda_hand']


def solve_3d(P3d, w, theta_init, iters=200, lr=2e-2, device='cuda'):
    """Optimize (theta,R,t) to minimize w*||R FK(theta)+t - P3d||^2. P3d (B,7,3), w (B,7)."""
    B = P3d.shape[0]; dtype = torch.float32
    lo, hi = make_limits(device, dtype)
    theta0 = theta_init.to(device, dtype).clone(); theta0[:, 6] = 0.0
    theta0 = torch.max(torch.min(theta0, hi), lo)
    fk0 = panda_forward_kinematics(theta0)
    R0, t0 = kabsch_batch(fk0, P3d, w)                      # well-posed init from 3D corresp
    p = theta_to_p(theta0, lo, hi).detach().requires_grad_(True)
    d6 = matrix_to_rot6d(R0).detach().requires_grad_(True)
    t = t0.detach().requires_grad_(True)
    opt = torch.optim.Adam([p, d6, t], lr=lr)
    wn = (w / w.sum(1, keepdim=True).clamp(min=1e-6)).unsqueeze(-1)
    for _ in range(iters):
        opt.zero_grad()
        theta = p_to_theta(p, lo, hi); theta = torch.cat([theta[:, :6], torch.zeros(B, 1, device=device)], 1)
        fk = panda_forward_kinematics(theta)
        cam = torch.bmm(fk, rot6d_to_matrix(d6).transpose(1, 2)) + t.unsqueeze(1)
        loss = (wn * (cam - P3d) ** 2).sum(-1).sum(1).mean()
        loss.backward(); opt.step()
    with torch.no_grad():
        theta = p_to_theta(p, lo, hi); theta = torch.cat([theta[:, :6], torch.zeros(B, 1, device=device)], 1)
        cam = torch.bmm(panda_forward_kinematics(theta), rot6d_to_matrix(d6).transpose(1, 2)) + t.unsqueeze(1)
    return theta.detach(), cam.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--detector', required=True); ap.add_argument('--mlp-head', required=True)
    ap.add_argument('--val-dir', required=True)
    ap.add_argument('--model-name', default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    ap.add_argument('--image-size', type=int, default=512); ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--max-frames', type=int, default=600); ap.add_argument('--iters', type=int, default=200)
    args = ap.parse_args()
    device = torch.device('cuda'); S = args.image_size
    m = AnglePredictor(args.model_name, S, head_type='mlp').to(device).eval()
    sd = torch.load(args.detector, map_location=device); sd = {k.replace('module.', ''): v for k, v in sd.items()}
    m.load_state_dict({k: v for k, v in sd.items() if k in m.state_dict() and v.shape == m.state_dict()[k].shape}, strict=False)
    m.angle_head.load_state_dict(torch.load(args.mlp_head, map_location=device))

    ds = EvalDataset(args.val_dir, KPN, image_size=(S, S))
    if args.max_frames and args.max_frames < len(ds.json_files):
        stride = max(1, len(ds.json_files) // args.max_frames); ds.json_files = ds.json_files[::stride][:args.max_frames]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    res = {k: {'add': [], 'j0': []} for k in ['base_2d', 'depth_oracle', 'depth_n5', 'depth_n10']}
    g = torch.Generator(device='cpu').manual_seed(0)
    for batch in tqdm(loader, desc='depth-probe'):
        img = batch['image'].to(device); K = scale_K(batch['camera_K'], batch['original_size'], S).to(device)
        gt3d = batch['gt_3d']; found = batch['found']; gt_ang = batch['gt_angles'].numpy()
        with torch.no_grad():
            out = m(img, K)
        kp2d = out['keypoints_2d']; conf = out['confidence']; init_ang = out['joint_angles']
        B = img.shape[0]; w = found.to(device).clamp(min=1e-3)
        # backproject detected 2D + depth z -> 3D camera point
        fx = K[:, 0, 0:1]; fy = K[:, 1, 1:2]; cx = K[:, 0, 2:3]; cy = K[:, 1, 2:3]
        bx = (kp2d[:, :, 0] - cx) / fx; by = (kp2d[:, :, 1] - cy) / fy   # (B,7) bearings
        z_gt = gt3d[..., 2].to(device)                                   # (B,7) oracle depth
        def lift(z):
            return torch.stack([bx * z, by * z, z], dim=-1)             # (B,7,3)
        # base 2D solve
        th2, kc2, _ = solve_batch(kp2d, conf, K, fix_joint7=True, iters=args.iters, lr=2e-2,
                                  img_size=S, device=device, prior_w=0.0, theta_init=init_ang)
        configs = [('base_2d', th2, kc2)]
        for name, zmul in [('depth_oracle', 0.0), ('depth_n5', 0.05), ('depth_n10', 0.10)]:
            z = z_gt * (1 + (torch.randn(z_gt.shape, generator=g).to(device) * zmul if zmul > 0 else 0))
            th, kc = solve_3d(lift(z), w, init_ang, iters=args.iters, device=device)
            configs.append((name, th, kc))
        gt3d_d = gt3d.to(device)
        for name, th, kc in configs:
            thn = th.cpu().numpy()
            for b in range(B):
                f = found[b].numpy()
                if f.sum() < 4 or not np.any(gt_ang[b] != 0):
                    continue
                res[name]['add'].append(float((kc[b] - gt3d_d[b]).norm(dim=-1).cpu().numpy()[f > 0].mean()))
                d = np.arctan2(np.sin(thn[b, 0] - gt_ang[b, 0]), np.cos(thn[b, 0] - gt_ang[b, 0]))
                res[name]['j0'].append(abs(math.degrees(d)))

    print(f"\n  {os.path.basename(args.val_dir)}  depth-lift probe")
    print(f"  {'config':<14}{'ADD-AUC':>10}{'meanADD':>10}{'J0 med':>9}")
    for k in ['base_2d', 'depth_oracle', 'depth_n5', 'depth_n10']:
        a = np.array(res[k]['add']); j = np.array(res[k]['j0'])
        print(f"  {k:<14}{compute_add_auc(a):>10.4f}{a.mean()*1000:>10.1f}{np.median(j):>9.1f}")


if __name__ == '__main__':
    main()
