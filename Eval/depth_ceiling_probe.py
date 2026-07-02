"""
GAUGE-SAFE depth ceiling probe. The naive oracle-substitution (swap GT R/t post-hoc) is invalid
because base-yaw J0 <-> camera R <-> Kabsch t are one gauge freedom. The ONLY valid test of "would
correct depth fix the pose?" is to ADD a GT-root-depth constraint to the solver and RE-OPTIMIZE R,theta
consistently around it. Compares solver ADD-AUC: baseline vs +GT-depth-anchor, on realsense & orb.
If GT-depth gives little/no lift -> depth is NOT the lever (consistent with depth_lift_probe REJECTED);
if it gives a big lift -> a depth head (HPE k_value) is worth building.
"""
import argparse, os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

TRAIN = os.path.abspath(os.path.join(os.path.dirname(__file__), '../TRAIN'))
sys.path.append(TRAIN); sys.path.append(os.path.dirname(__file__))
from model_angle import AnglePredictor
from inference_4tier_eval import EvalDataset
from solve_pose_kinematic import solve_batch
from refine_eval import scale_K

KPN = ['panda_link0', 'panda_link2', 'panda_link3', 'panda_link4', 'panda_link6', 'panda_link7', 'panda_hand']


def add_auc(adds_m, thr=0.1):
    a = np.asarray(adds_m); d = 1e-5; ts = np.arange(0.0, thr, d)
    return float(np.trapz((a[None, :] <= ts[:, None]).sum(1) / len(a), dx=d) / thr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--detector', required=True); ap.add_argument('--mlp-head', required=True)
    ap.add_argument('--val-dir', required=True)
    ap.add_argument('--model-name', default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    ap.add_argument('--image-size', type=int, default=512); ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--max-frames', type=int, default=600); ap.add_argument('--iters', type=int, default=200)
    ap.add_argument('--depth-w', type=float, default=5.0)
    ap.add_argument('--depth-noise', type=float, default=0.0, help='fractional gaussian noise on GT root depth (sim a real predictor)')
    args = ap.parse_args()
    torch.manual_seed(0)

    device = torch.device('cuda'); S = args.image_size
    m = AnglePredictor(args.model_name, S, head_type='mlp').to(device).eval()
    sd = torch.load(args.detector, map_location=device); sd = {k.replace('module.', ''): v for k, v in sd.items()}
    m.load_state_dict({k: v for k, v in sd.items() if k in m.state_dict() and v.shape == m.state_dict()[k].shape}, strict=False)
    m.angle_head.load_state_dict(torch.load(args.mlp_head, map_location=device))

    ds = EvalDataset(args.val_dir, KPN, image_size=(S, S))
    if args.max_frames and args.max_frames < len(ds.json_files):
        stride = max(1, len(ds.json_files) // args.max_frames)
        ds.json_files = ds.json_files[::stride][:args.max_frames]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    adds_base, adds_gtz, confs = [], [], []
    for batch in tqdm(loader, desc='depth-ceil'):
        img = batch['image'].to(device)
        K = scale_K(batch['camera_K'], batch['original_size'], S).to(device)
        gt3d = batch['gt_3d'].to(device); found = batch['found'].to(device)
        with torch.no_grad():
            out = m(img, K)
        kp2d = out['keypoints_2d']; conf = out['confidence']; init = out['joint_angles']
        # GT root depth = camera-frame z of base (link0, idx 0); only frames where base is found
        base_ok = found[:, 0] > 0
        gt_tz = gt3d[:, 0, 2].clone()
        if args.depth_noise > 0.0:                       # simulate a real depth predictor's error
            gt_tz = gt_tz * (1.0 + args.depth_noise * torch.randn_like(gt_tz))
        # baseline solve
        _, kp_cam_b, _ = solve_batch(kp2d, conf, K, fix_joint7=True, iters=args.iters, lr=2e-2,
                                     img_size=S, device=device, prior_w=0.0, theta_init=init)
        # + GT-depth anchor solve
        _, kp_cam_z, _ = solve_batch(kp2d, conf, K, fix_joint7=True, iters=args.iters, lr=2e-2,
                                     img_size=S, device=device, prior_w=0.0, theta_init=init,
                                     gt_tz=gt_tz, depth_w=args.depth_w)
        f = found.bool()
        for b in range(img.shape[0]):
            if f[b].sum() < 5 or not bool(base_ok[b]):
                continue
            fb = f[b]
            adds_base.append(float((kp_cam_b[b][fb] - gt3d[b][fb]).norm(dim=-1).mean()))
            adds_gtz.append(float((kp_cam_z[b][fb] - gt3d[b][fb]).norm(dim=-1).mean()))
            confs.append(float(conf[b][fb].mean()))

    cam = os.path.basename(args.val_dir)
    print(f"\n=== DEPTH CEILING  {cam}  (n={len(adds_base)}, depth_w={args.depth_w}) ===")
    print(f"  baseline solve        ADD-AUC@100mm {add_auc(adds_base):.4f}  mean {1000*np.mean(adds_base):.1f}mm")
    print(f"  + GT root-depth anchor ADD-AUC@100mm {add_auc(adds_gtz):.4f}  mean {1000*np.mean(adds_gtz):.1f}mm")
    print(f"  Δ from perfect depth: {add_auc(adds_gtz)-add_auc(adds_base):+.4f}")
    # SELECTIVE: apply the (noisy) depth anchor ONLY to low-conf frames (the failing tail);
    # keep the baseline solve on high-conf frames (don't corrupt what already works).
    ab = np.array(adds_base); az = np.array(adds_gtz); cf = np.array(confs)
    print(f"  --- selective depth (anchor only frames with mean-conf < thr) ---")
    for thr in [0.45, 0.50, 0.55, 0.60]:
        sel = cf < thr
        merged = np.where(sel, az, ab)
        print(f"    conf<{thr}: anchored {sel.sum():>3}/{len(cf)} frames  ADD-AUC {add_auc(merged):.4f}  (Δ {add_auc(merged)-add_auc(ab):+.4f})")


if __name__ == '__main__':
    main()
