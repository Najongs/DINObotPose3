"""
Characterize WHICH realsense frames the (fixed pnp_drop=3) solver fails on, to pick the next lever.
depth_diag showed the residual gap is an under-constrained pose solve (R ~47 deg off even with good
2D). This logs per-frame ADD + candidate explanatory features, reports their correlation with ADD,
and profiles the worst vs best frames — so we can see if failures = geometric pose-ambiguity
(foreshortening / near-coplanar keypoints / arm pointing along the camera axis) vs detector/conf.

Features per frame:
  add_mm          : solver ADD (mm)            <- the target
  rot_err_deg     : Kabsch(pred) vs Kabsch(GT) rotation error
  planarity       : sqrt(lambda_min/lambda_max) of GT-3D PCA (0=coplanar -> pose-ambiguous)
  depth_spread_mm : max-min GT keypoint Z (small => low depth signal => t_z ill-posed)
  fore_axis_deg   : angle between robot principal axis and camera optical axis (small => pointing
                    AT camera => foreshortened => depth-ambiguous)
  bbox_frac       : 2D keypoint bbox area / image area (small => far/small => weak constraint)
  conf_min/mean   : detector confidence
  kp2d_err_px     : detected-vs-GT 2D error (px @ heatmap res)
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
from inference_4tier_eval import EvalDataset
from solve_pose_kinematic import solve_batch
from refine_eval import scale_K
from depth_diag import kabsch, geodesic_deg

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

    rows = []  # dict per frame
    for batch in tqdm(loader, desc='fail-diag'):
        img = batch['image'].to(device)
        K = scale_K(batch['camera_K'], batch['original_size'], S).to(device)
        gt3d = batch['gt_3d'].numpy(); found = batch['found'].numpy(); gt_ang = batch['gt_angles'].numpy()
        with torch.no_grad():
            out = m(img, K)
        kp2d = out['keypoints_2d']; conf = out['confidence']; init_ang = out['joint_angles']
        theta, kp_cam, _ = solve_batch(kp2d, conf, K, fix_joint7=True, iters=args.iters, lr=2e-2,
                                       img_size=S, device=device, prior_w=0.0, theta_init=init_ang)
        theta_np = theta.cpu().numpy(); kp_cam_np = kp_cam.cpu().numpy()
        kp2d_np = kp2d.cpu().numpy(); conf_np = conf.cpu().numpy()
        gt2d_s = batch['gt_2d'].numpy()  # px @ original; scale to S
        osz = batch['original_size'].numpy()
        for b in range(img.shape[0]):
            f = found[b]
            if f.sum() < 5 or not np.any(gt_ang[b] != 0):
                continue
            add_mm = float(np.linalg.norm(kp_cam_np[b] - gt3d[b], axis=1)[f > 0].mean() * 1000)
            ga = gt_ang[b].copy(); ga[6] = 0.0
            fk_pred = panda_forward_kinematics(torch.from_numpy(theta_np[b:b+1]).float()).numpy()[0]
            fk_gt = panda_forward_kinematics(torch.from_numpy(ga[None]).float()).numpy()[0]
            Rp, tp = kabsch(fk_pred, kp_cam_np[b], np.ones(7))
            Rg, tg = kabsch(fk_gt, gt3d[b], f)
            rot_err = geodesic_deg(Rp, Rg)
            # planarity & foreshortening from GT 3D (camera frame)
            P = gt3d[b][f > 0]
            Pc = P - P.mean(0)
            evals = np.linalg.svd(Pc, compute_uv=False) ** 2
            planarity = float(np.sqrt(evals[-1] / (evals[0] + 1e-9)))
            depth_spread = float((P[:, 2].max() - P[:, 2].min()) * 1000)
            # robot principal axis (largest PCA dir) vs camera optical axis (0,0,1)
            _, _, Vt = np.linalg.svd(Pc)
            axis = Vt[0]
            fore_axis = math.degrees(math.acos(min(1.0, abs(axis[2]))))  # 90=broadside, 0=along cam axis
            # 2D bbox fraction
            sx, sy = S / osz[b, 0], S / osz[b, 1]
            g2 = gt2d_s[b][f > 0] * np.array([sx, sy])
            bbox = (g2[:, 0].max() - g2[:, 0].min()) * (g2[:, 1].max() - g2[:, 1].min())
            bbox_frac = float(bbox / (S * S))
            kp2d_err = float(np.linalg.norm(kp2d_np[b][f > 0] - g2, axis=1).mean())
            # ---- ADD error DECOMPOSITION into mm contributed by each component ----
            # pred kp = Rp@FK(theta_p)+tp ; GT kp ~= Rg@FK(theta_g)+tg. Isolate each factor's error:
            angle_3d_mm = float(np.linalg.norm(fk_pred - fk_gt, axis=1).mean() * 1000)        # pure theta err (rot-invariant)
            rot_3d_mm = float(np.linalg.norm(fk_gt @ (Rp - Rg).T, axis=1).mean() * 1000)      # pure R err displacing the kp
            trans_mm = float(np.linalg.norm(tp - tg) * 1000)                                  # pure t err
            # ---- ORACLE-SUBSTITUTION per-frame ADD (m): swap pred R/t for GT, keep pred FK(theta) ----
            recon = (fk_pred @ Rp.T) + tp                      # ~= kp_cam (Kabsch reconstruction, consistent baseline)
            fmask = f > 0
            add_base = float(np.linalg.norm(recon - gt3d[b], axis=1)[fmask].mean())
            add_gtt  = float(np.linalg.norm((fk_pred @ Rp.T) + tg - gt3d[b], axis=1)[fmask].mean())   # GT translation
            add_gtr  = float(np.linalg.norm((fk_pred @ Rg.T) + tp - gt3d[b], axis=1)[fmask].mean())   # GT rotation
            add_gtrt = float(np.linalg.norm((fk_pred @ Rg.T) + tg - gt3d[b], axis=1)[fmask].mean())   # GT R+t (only theta pred)
            rows.append(dict(add_mm=add_mm, rot_err=rot_err, planarity=planarity,
                             depth_spread=depth_spread, fore_axis=fore_axis, bbox_frac=bbox_frac,
                             conf_min=float(conf_np[b][f > 0].min()), conf_mean=float(conf_np[b][f > 0].mean()),
                             kp2d_err=kp2d_err, angle_3d_mm=angle_3d_mm, rot_3d_mm=rot_3d_mm, trans_mm=trans_mm,
                             add_base=add_base, add_gtt=add_gtt, add_gtr=add_gtr, add_gtrt=add_gtrt))

    # ---- ORACLE CEILING: ADD-AUC@100mm when each pose component is replaced by GT ----
    def _auc(arr):
        a = np.asarray(arr); d = 1e-5; ts = np.arange(0.0, 0.1, d)
        return float(np.trapz((a[None, :] <= ts[:, None]).sum(1) / len(a), dx=d) / 0.1)
    abase = [r['add_base'] for r in rows]; agtt = [r['add_gtt'] for r in rows]
    agtr = [r['add_gtr'] for r in rows]; agtrt = [r['add_gtrt'] for r in rows]
    print(f"\n  --- ORACLE-SUBSTITUTION ADD-AUC@100mm ceiling (n={len(rows)}) ---")
    print(f"    baseline (all pred)      {_auc(abase):.4f}")
    print(f"    + GT translation t       {_auc(agtt):.4f}   (Δ {_auc(agtt)-_auc(abase):+.4f})")
    print(f"    + GT rotation R          {_auc(agtr):.4f}   (Δ {_auc(agtr)-_auc(abase):+.4f})")
    print(f"    + GT R and t (θ-only)    {_auc(agtrt):.4f}   (Δ {_auc(agtrt)-_auc(abase):+.4f})")

    keys = ['rot_err', 'angle_3d_mm', 'rot_3d_mm', 'trans_mm', 'fore_axis', 'bbox_frac', 'conf_mean', 'kp2d_err']
    # COMPONENT DECOMPOSITION: which error source dominates the ADD failure?
    a3 = np.array([r['angle_3d_mm'] for r in rows]); r3 = np.array([r['rot_3d_mm'] for r in rows]); tt = np.array([r['trans_mm'] for r in rows])
    print(f"\n  --- ADD error sources (mm), failing(>100) vs ok frames ---")
    add_tmp = np.array([r['add_mm'] for r in rows]); fl = add_tmp > 100
    for nm, v in [('angle->3d', a3), ('rotation->3d', r3), ('translation', tt), ('kp2d_px', np.array([r['kp2d_err'] for r in rows]))]:
        print(f"    {nm:<14} fail-mean {v[fl].mean():>8.1f}   ok-mean {v[~fl].mean():>8.1f}   ratio {v[fl].mean()/(v[~fl].mean()+1e-6):>5.1f}x")
    add = np.array([r['add_mm'] for r in rows])
    fail = add > 100  # ADD>100mm = ADD-AUC failure
    print(f"\n{'='*64}\n  REALSENSE FAILURE PROFILE  {os.path.basename(args.val_dir)}  (n={len(rows)})\n{'='*64}")
    print(f"  ADD: mean {add.mean():.1f}mm  median {np.median(add):.1f}mm  | fail(>100mm) {100*fail.mean():.1f}%")
    print(f"  {'feature':<14}{'corr(ADD)':>11}{'fail-mean':>11}{'ok-mean':>11}{'all-med':>10}")
    for k in keys:
        v = np.array([r[k] for r in rows])
        c = float(np.corrcoef(v, add)[0, 1])
        print(f"  {k:<14}{c:>+11.3f}{v[fail].mean():>11.2f}{v[~fail].mean():>11.2f}{np.median(v):>10.2f}")
    # worst-20 vs best-20 profile
    idx = np.argsort(-add)
    w = idx[:20]; bst = idx[-20:]
    print(f"\n  worst-20 ADD mean {add[w].mean():.0f}mm vs best-20 {add[bst].mean():.0f}mm")
    for k in keys:
        v = np.array([r[k] for r in rows])
        print(f"    {k:<14} worst {v[w].mean():>8.2f}   best {v[bst].mean():>8.2f}")


if __name__ == '__main__':
    main()
