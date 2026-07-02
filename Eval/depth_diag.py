"""
B-diagnosis: WHERE does the per-camera ADD leak live — depth(t_z) / translation / rotation / angles?

The realsense split is -48 AUC behind RoboPEPP at the SAME 2D quality. Hypothesis: PnP depth
(t_z) is fragile at distance. This script attributes the leak by ORACLE SWAPS.

Pipeline per frame: detector -> 2D kp + conf -> mlp angle init -> kinematic solve_batch
  => theta_pred and kp_cam_pred = R_pred @ FK(theta_pred) + t_pred.
We recover (R_pred, t_pred) by Kabsch on (FK(theta_pred) -> kp_cam_pred)  [exact],
and the GT camera pose (R_gt, t_gt) by Kabsch on (FK(gt_angles) -> gt_3d) over found kp.

Then we rebuild camera-frame keypoints swapping ONE component to its GT value and measure
ADD-AUC@100mm. The swap that recovers the most AUC is the dominant leak:
  base        R_pred FK(th_pred) + t_pred           (everything predicted)
  +oracle tz  R_pred FK(th_pred) + [tx,ty, GT tz]   (depth only)
  +oracle t   R_pred FK(th_pred) + t_gt             (full translation)
  +oracle R   R_gt   FK(th_pred) + t_pred           (rotation)
  +oracle th  R_pred FK(gt_ang)  + t_pred           (joint angles)
  +oracle Rt  R_gt   FK(th_pred) + t_gt             (camera pose, pred angles)
  all-oracle  R_gt   FK(gt_ang)  + t_gt             (sanity ~ 0)
"""
import argparse, math, os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

TRAIN = os.path.abspath(os.path.join(os.path.dirname(__file__), '../TRAIN'))
sys.path.append(TRAIN); sys.path.append(os.path.dirname(__file__))
from model_angle import AnglePredictor
from model_v4 import panda_forward_kinematics
from inference_4tier_eval import EvalDataset, compute_add_auc
from solve_pose_kinematic import solve_batch
from refine_eval import scale_K

KPN = ['panda_link0', 'panda_link2', 'panda_link3', 'panda_link4', 'panda_link6', 'panda_link7', 'panda_hand']


def kabsch(P, Q, w):
    """Rigid R,t minimizing ||R P + t - Q|| (no scale). P,Q (N,3) np, w (N,) weights. -> R(3,3),t(3)."""
    w = w / (w.sum() + 1e-9)
    mp = (w[:, None] * P).sum(0); mq = (w[:, None] * Q).sum(0)
    Pc = P - mp; Qc = Q - mq
    H = (w[:, None] * Pc).T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    t = mq - R @ mp
    return R, t


def add_of(P_robot, R, t, gt3d, found):
    """mean per-frame ADD (m) of (R P + t) vs gt3d over found kp."""
    cam = (R @ P_robot.T).T + t
    e = np.linalg.norm(cam - gt3d, axis=1)
    m = found > 0
    return float(e[m].mean()) if m.any() else np.nan


def geodesic_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--detector', required=True)
    ap.add_argument('--mlp-head', required=True)
    ap.add_argument('--val-dir', required=True)
    ap.add_argument('--model-name', default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    ap.add_argument('--image-size', type=int, default=512)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--max-frames', type=int, default=800)
    ap.add_argument('--iters', type=int, default=200)
    ap.add_argument('--conf-gate', type=float, default=0.05)
    args = ap.parse_args()

    device = torch.device('cuda'); assert torch.cuda.is_available(); S = args.image_size
    m = AnglePredictor(args.model_name, S, head_type='mlp').to(device).eval()
    sd = torch.load(args.detector, map_location=device); sd = {k.replace('module.', ''): v for k, v in sd.items()}
    m.load_state_dict({k: v for k, v in sd.items() if k in m.state_dict() and v.shape == m.state_dict()[k].shape}, strict=False)
    m.angle_head.load_state_dict(torch.load(args.mlp_head, map_location=device))

    ds = EvalDataset(args.val_dir, KPN, image_size=(S, S))
    # STRIDED for a representative spread (sorted [:N] = one biased trajectory segment).
    if args.max_frames and args.max_frames < len(ds.json_files):
        stride = max(1, len(ds.json_files) // args.max_frames)
        ds.json_files = ds.json_files[::stride][:args.max_frames]
        print(f"  strided {stride} -> {len(ds.json_files)} frames")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    variants = ['base', 'oracle_tz', 'oracle_t', 'oracle_R', 'oracle_th', 'oracle_Rt', 'all_oracle']
    adds = {k: [] for k in variants}
    depth_gt, dtz, dt_xy, rot_err, ang_err = [], [], [], [], []

    for batch in tqdm(loader, desc=os.path.basename(args.val_dir)):
        img = batch['image'].to(device)
        K = scale_K(batch['camera_K'], batch['original_size'], S).to(device)
        gt3d = batch['gt_3d'].numpy()           # (B,7,3) cam-frame meters
        found = batch['found'].numpy()          # (B,7)
        gt_ang = batch['gt_angles'].numpy()     # (B,7)
        with torch.no_grad():
            out = m(img, K)
        kp2d = out['keypoints_2d']; conf = out['confidence']; init_ang = out['joint_angles']
        theta, kp_cam, _ = solve_batch(kp2d, conf, K, fix_joint7=True, iters=args.iters, lr=2e-2,
                                       img_size=S, device=device, prior_w=0.0,
                                       theta_init=init_ang, conf_gate=args.conf_gate)
        theta_np = theta.cpu().numpy(); kp_cam_np = kp_cam.cpu().numpy()
        B = img.shape[0]
        for b in range(B):
            f = found[b]
            if f.sum() < 4 or not np.any(gt_ang[b] != 0):
                continue
            fk_pred = panda_forward_kinematics(torch.from_numpy(theta_np[b:b+1]).float()).numpy()[0]  # (7,3) robot
            ga = gt_ang[b].copy(); ga[6] = 0.0
            fk_gt = panda_forward_kinematics(torch.from_numpy(ga[None]).float()).numpy()[0]
            # recover pred pose (exact) and GT pose (over found kp)
            Rp, tp = kabsch(fk_pred, kp_cam_np[b], np.ones(7))
            Rg, tg = kabsch(fk_gt, gt3d[b], f)
            # diagnostics
            depth_gt.append(tg[2]); dtz.append(abs(tp[2] - tg[2]))
            dt_xy.append(math.hypot(tp[0] - tg[0], tp[1] - tg[1]))
            rot_err.append(geodesic_deg(Rp, Rg))
            d = np.arctan2(np.sin(theta_np[b, :6] - ga[:6]), np.cos(theta_np[b, :6] - ga[:6]))
            ang_err.append(np.degrees(np.abs(d)).mean())
            # oracle swaps
            tz = tp.copy(); tz[2] = tg[2]
            adds['base'].append(add_of(fk_pred, Rp, tp, gt3d[b], f))
            adds['oracle_tz'].append(add_of(fk_pred, Rp, tz, gt3d[b], f))
            adds['oracle_t'].append(add_of(fk_pred, Rp, tg, gt3d[b], f))
            adds['oracle_R'].append(add_of(fk_pred, Rg, tp, gt3d[b], f))
            adds['oracle_th'].append(add_of(fk_gt, Rp, tp, gt3d[b], f))
            adds['oracle_Rt'].append(add_of(fk_pred, Rg, tg, gt3d[b], f))
            adds['all_oracle'].append(add_of(fk_gt, Rg, tg, gt3d[b], f))

    n = len(adds['base'])
    print(f"\n{'='*70}\n  DEPTH/POSE ATTRIBUTION  {os.path.basename(args.val_dir)}  (n={n})\n{'='*70}")
    print(f"  GT depth t_z:   mean {np.mean(depth_gt):.3f} m   median {np.median(depth_gt):.3f} m")
    print(f"  |Δt_z| (depth err):  mean {np.mean(dtz)*1000:6.1f} mm   median {np.median(dtz)*1000:6.1f} mm")
    print(f"  |Δt_xy| (lateral):   mean {np.mean(dt_xy)*1000:6.1f} mm   median {np.median(dt_xy)*1000:6.1f} mm")
    print(f"  rotation error:      mean {np.mean(rot_err):6.2f} deg  median {np.median(rot_err):6.2f} deg")
    print(f"  joint angle MAE:     mean {np.mean(ang_err):6.2f} deg")
    print(f"  {'-'*60}")
    print(f"  {'variant':<12}{'ADD-AUC@100':>13}{'meanADD mm':>13}{'medADD mm':>12}")
    base_auc = compute_add_auc(np.array(adds['base']))
    for k in variants:
        a = np.array(adds[k]); auc = compute_add_auc(a)
        dlt = f"  (+{auc-base_auc:.3f})" if k != 'base' else ""
        print(f"  {k:<12}{auc:>13.4f}{a.mean()*1000:>13.1f}{np.median(a)*1000:>12.1f}{dlt}")
    print('='*70)


if __name__ == '__main__':
    main()
