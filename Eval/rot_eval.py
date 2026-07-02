"""
Evaluate the learned rotation head: predict robot->camera R from appearance, feed as the solver's
R_init, measure ADD-AUC. Also reports the LEARNED R's geodesic error vs GT (sim2real rotation
accuracy) and the realised vs oracle ceiling (rinit_probe: oracle R-init = +0.11 realsense).
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
from depth_diag import kabsch, geodesic_deg

KPN = ['panda_link0', 'panda_link2', 'panda_link3', 'panda_link4', 'panda_link6', 'panda_link7', 'panda_hand']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--detector', required=True); ap.add_argument('--mlp-head', required=True)
    ap.add_argument('--rot-head', required=True)
    ap.add_argument('--val-dir', required=True)
    ap.add_argument('--model-name', default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    ap.add_argument('--image-size', type=int, default=512); ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--max-frames', type=int, default=600); ap.add_argument('--iters', type=int, default=200)
    args = ap.parse_args()
    device = torch.device('cuda'); S = args.image_size
    m = AnglePredictor(args.model_name, S, head_type='mlp', with_rotation=True,
                       with_translation=True).to(device).eval()
    sd = torch.load(args.detector, map_location=device); sd = {k.replace('module.', ''): v for k, v in sd.items()}
    m.load_state_dict({k: v for k, v in sd.items() if k in m.state_dict() and v.shape == m.state_dict()[k].shape}, strict=False)
    m.angle_head.load_state_dict(torch.load(args.mlp_head, map_location=device))
    m.rot_head.load_state_dict(torch.load(args.rot_head, map_location=device))

    ds = EvalDataset(args.val_dir, KPN, image_size=(S, S))
    if args.max_frames and args.max_frames < len(ds.json_files):
        stride = max(1, len(ds.json_files) // args.max_frames)
        ds.json_files = ds.json_files[::stride][:args.max_frames]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    res = {'base': [], 'learned_Rinit': [], 'learned_RTinit': []}
    geo_errs, t_errs = [], []
    for batch in tqdm(loader, desc='rot-eval'):
        img = batch['image'].to(device); K = scale_K(batch['camera_K'], batch['original_size'], S).to(device)
        gt3d = batch['gt_3d'].numpy(); found = batch['found'].numpy(); gt_ang = batch['gt_angles'].numpy()
        with torch.no_grad():
            out = m(img, K)
        kp2d = out['keypoints_2d']; conf = out['confidence']; init_ang = out['joint_angles']
        Rpred = out['rot_matrix']                         # (B,3,3) learned
        Tpred = out.get('trans')                          # (B,3) learned
        B = img.shape[0]
        ok = np.zeros(B, bool)
        for b in range(B):
            f = found[b]
            if f.sum() < 4 or not np.any(gt_ang[b] != 0):
                continue
            ok[b] = True
            ga = gt_ang[b].copy(); ga[6] = 0.0
            fk_gt = panda_forward_kinematics(torch.from_numpy(ga[None]).float()).numpy()[0]
            Rg, tgt = kabsch(fk_gt, gt3d[b], f)
            geo_errs.append(geodesic_deg(Rpred[b].cpu().numpy(), Rg))
            if Tpred is not None:
                t_errs.append(float(np.linalg.norm(Tpred[b].cpu().numpy() - tgt) * 1000))
        for name, Ri, Ti in [('base', None, None), ('learned_Rinit', Rpred, None),
                             ('learned_RTinit', Rpred, Tpred)]:
            theta, kp_cam, _ = solve_batch(kp2d, conf, K, fix_joint7=True, iters=args.iters, lr=2e-2,
                                           img_size=S, device=device, prior_w=0.0, theta_init=init_ang,
                                           R_init=Ri, t_init=Ti)
            kc = kp_cam.cpu().numpy()
            for b in range(B):
                if ok[b]:
                    res[name].append(float(np.linalg.norm(kc[b] - gt3d[b], axis=1)[found[b] > 0].mean()))
    ge = np.array(geo_errs); te = np.array(t_errs)
    print(f"\n  {os.path.basename(args.val_dir)}  learned-pose eval (n={len(ge)})")
    print(f"  learned R geodesic: median {np.median(ge):.1f} deg | t-err: median {np.median(te):.1f} mm")
    print(f"  {'config':<16}{'ADD-AUC':>10}{'meanADD':>10}{'medADD':>9}")
    for k in ['base', 'learned_Rinit', 'learned_RTinit']:
        a = np.array(res[k])
        print(f"  {k:<16}{compute_add_auc(a):>10.4f}{a.mean()*1000:>10.1f}{np.median(a)*1000:>9.1f}")


if __name__ == '__main__':
    main()
