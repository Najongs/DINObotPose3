"""
Ceiling probe: does initializing the solver at the GT rotation fix realsense?
Failures correlate only with rot_err (+0.64) — the solver lands in a wrong rotation basin.
If an oracle R-init collapses ADD, a LEARNED rotation head is the lever. Tests:
  base        : PnP init (current)
  Rinit       : GT rotation as init, PnP t  (let solver refine t,theta around right R)
  pose_init   : GT rotation + GT t as init
GT pose = Kabsch(FK(gt_angles) -> gt_3d).
"""
import argparse, os, sys
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
from depth_diag import kabsch

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
    if args.max_frames and args.max_frames < len(ds.json_files):
        stride = max(1, len(ds.json_files) // args.max_frames)
        ds.json_files = ds.json_files[::stride][:args.max_frames]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    res = {k: [] for k in ['base', 'Rinit', 'pose_init']}
    for batch in tqdm(loader, desc='rinit'):
        img = batch['image'].to(device); K = scale_K(batch['camera_K'], batch['original_size'], S).to(device)
        gt3d = batch['gt_3d'].numpy(); found = batch['found'].numpy(); gt_ang = batch['gt_angles'].numpy()
        with torch.no_grad():
            out = m(img, K)
        kp2d = out['keypoints_2d']; conf = out['confidence']; init_ang = out['joint_angles']
        B = img.shape[0]
        # GT pose per frame via Kabsch
        Rg = np.tile(np.eye(3), (B, 1, 1)); tg = np.zeros((B, 3)); ok = np.zeros(B, bool)
        for b in range(B):
            f = found[b]
            if f.sum() < 4 or not np.any(gt_ang[b] != 0):
                continue
            ga = gt_ang[b].copy(); ga[6] = 0.0
            fk_gt = panda_forward_kinematics(torch.from_numpy(ga[None]).float()).numpy()[0]
            Rg[b], tg[b] = kabsch(fk_gt, gt3d[b], f); ok[b] = True
        Rg_t = torch.from_numpy(Rg).float(); tg_t = torch.from_numpy(tg).float()
        for name, Ri, ti in [('base', None, None), ('Rinit', Rg_t, None), ('pose_init', Rg_t, tg_t)]:
            theta, kp_cam, _ = solve_batch(kp2d, conf, K, fix_joint7=True, iters=args.iters, lr=2e-2,
                                           img_size=S, device=device, prior_w=0.0, theta_init=init_ang,
                                           R_init=Ri, t_init=ti)
            kc = kp_cam.cpu().numpy()
            for b in range(B):
                if ok[b]:
                    res[name].append(float(np.linalg.norm(kc[b] - gt3d[b], axis=1)[found[b] > 0].mean()))
    print(f"\n  {os.path.basename(args.val_dir)}  R-init ceiling probe")
    print(f"  {'config':<11}{'ADD-AUC':>10}{'meanADD':>10}{'medADD':>9}")
    for k in ['base', 'Rinit', 'pose_init']:
        a = np.array(res[k])
        print(f"  {k:<11}{compute_add_auc(a):>10.4f}{a.mean()*1000:>10.1f}{np.median(a)*1000:>9.1f}")


if __name__ == '__main__':
    main()
