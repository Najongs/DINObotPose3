"""
B-fix test: does CONSTRAINING the solve (anchor theta to the learned init, and/or use all
keypoints) re-condition the far-camera pose basin? depth_diag proved single-component fixes
hurt realsense because (R,t,theta) are jointly under-constrained. Removing theta's DOF (anchor
it to the learned head) should make (R,t) over-determined by the 14 reprojection eqs.

Sweeps (anchor_init_w, conf_gate) and reports ADD-AUC@100mm on a camera.
"""
import argparse, os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

TRAIN = os.path.abspath(os.path.join(os.path.dirname(__file__), '../TRAIN'))
sys.path.append(TRAIN); sys.path.append(os.path.dirname(__file__))
from model_angle import AnglePredictor
from inference_4tier_eval import EvalDataset, compute_add_auc
from solve_pose_kinematic import solve_batch
from refine_eval import scale_K

KPN = ['panda_link0', 'panda_link2', 'panda_link3', 'panda_link4', 'panda_link6', 'panda_link7', 'panda_hand']


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
    # STRIDED subsample for a REPRESENTATIVE spread across the whole trajectory (the first-N
    # frames are one contiguous segment -> biased). stride = total // max_frames.
    if args.max_frames and args.max_frames < len(ds.json_files):
        stride = max(1, len(ds.json_files) // args.max_frames)
        ds.json_files = ds.json_files[::stride][:args.max_frames]
        print(f"  strided {stride} -> {len(ds.json_files)} frames spanning the set")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # cache forward outputs once; sweep only the (cheap) solver settings
    cache = []
    for batch in tqdm(loader, desc='fwd'):
        img = batch['image'].to(device)
        K = scale_K(batch['camera_K'], batch['original_size'], S).to(device)
        with torch.no_grad():
            out = m(img, K)
        cache.append((out['keypoints_2d'].cpu(), out['confidence'].cpu(), out['joint_angles'].cpu(),
                      K.cpu(), batch['gt_3d'].numpy(), batch['found'].numpy(), batch['gt_angles'].numpy()))

    # refine gate fixed at 0.05; vary how many lowest-conf keypoints to drop from the PnP init.
    # (name, pnp_drop)
    configs = [('drop0_all7 ', 0), ('drop1_top6 ', 1), ('drop2_top5 ', 2), ('drop3_top4 ', 3)]
    print(f"\n  {os.path.basename(args.val_dir)}  (n cached frames)")
    print(f"  {'config':<13}{'ADD-AUC':>10}{'meanADD':>10}{'medADD':>9}{'angMAE':>9}")
    import math
    for name, pdrop in configs:
        adds, angs = [], []
        for kp2d, conf, init_ang, K, gt3d, found, gt_ang in cache:
            theta, kp_cam, _ = solve_batch(kp2d.to(device), conf.to(device), K.to(device),
                                           fix_joint7=True, iters=args.iters, lr=2e-2, img_size=S,
                                           device=device, prior_w=0.0, theta_init=init_ang.to(device),
                                           conf_gate=0.05, anchor_init_w=0.0, min_kp=0, pnp_rel=0.0, pnp_drop=pdrop)
            theta = theta.cpu().numpy(); kp_cam = kp_cam.cpu().numpy()
            for b in range(kp_cam.shape[0]):
                f = found[b]
                if f.sum() < 4 or not np.any(gt_ang[b] != 0):
                    continue
                e = np.linalg.norm(kp_cam[b] - gt3d[b], axis=1)[f > 0]
                adds.append(float(e.mean()))
                d = np.arctan2(np.sin(theta[b, :6] - gt_ang[b, :6]), np.cos(theta[b, :6] - gt_ang[b, :6]))
                angs.append(np.degrees(np.abs(d)).mean())
        a = np.array(adds)
        print(f"  {name:<13}{compute_add_auc(a):>10.4f}{a.mean()*1000:>10.1f}{np.median(a)*1000:>9.1f}{np.mean(angs):>9.2f}")


if __name__ == '__main__':
    main()
