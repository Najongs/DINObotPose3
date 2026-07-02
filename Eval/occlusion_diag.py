"""
Occlusion diagnostic: is the ADD failure tail caused by occluded / off-frame keypoints,
and does the detector KNOW (low confidence) or does it confidently hallucinate?

Per frame logs: ADD (refined), #GT keypoints off-frame, per-keypoint {off-frame?, conf, 2D px err}.
Then answers:
  (1) frame ADD vs #off-frame keypoints  (does occlusion drive the tail?)
  (2) for off-frame vs in-frame keypoints: confidence + pixel error
      -> off-frame LOW conf  = detector knows (solver can down-weight)  [good]
      -> off-frame HIGH conf + HIGH err = confidently wrong            [the bug to fix]
"""
import argparse, glob, math, os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

TRAIN = os.path.abspath(os.path.join(os.path.dirname(__file__), '../TRAIN'))
sys.path.append(TRAIN); sys.path.append(os.path.dirname(__file__))
from model_angle import AnglePredictor  # noqa
from dataset import PoseEstimationDataset
from solve_pose_kinematic import solve_batch
from refine_eval import scale_K, add_auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--detector', required=True)
    ap.add_argument('--mlp-head', required=True)
    ap.add_argument('--val-dir', required=True)
    ap.add_argument('--model-name', default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    ap.add_argument('--image-size', type=int, default=512)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--max-frames', type=int, default=400)
    ap.add_argument('--iters', type=int, default=200)
    ap.add_argument('--conf-gate', type=float, default=0.1, help='hard-reject keypoints below this conf')
    ap.add_argument('--anchor-w', type=float, default=0.0, help='anchor angles to learned init (occluded fallback)')
    ap.add_argument('--mean-fallback', action='store_true', help='set unobservable joints to mean in init')
    args = ap.parse_args()

    device = torch.device('cuda'); assert torch.cuda.is_available()
    mlp = AnglePredictor(args.model_name, args.image_size, head_type='mlp').to(device).eval()
    sd = torch.load(args.detector, map_location=device); sd = {k.replace('module.', ''): v for k, v in sd.items()}
    mlp.load_state_dict({k: v for k, v in sd.items() if k in mlp.state_dict()
                         and v.shape == mlp.state_dict()[k].shape}, strict=False)
    mlp.angle_head.load_state_dict(torch.load(args.mlp_head, map_location=device))

    ds = PoseEstimationDataset(args.val_dir, keypoint_names=['link0','link2','link3','link4','link6','link7','hand'],
                               image_size=(args.image_size, args.image_size),
                               heatmap_size=(args.image_size, args.image_size),
                               augment=False, include_angles=True, sigma=2.5)
    if args.max_frames and args.max_frames < len(ds):
        ds.samples = ds.samples[:args.max_frames]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    S = args.image_size
    rows = []          # per-frame dict
    kp_off_conf = []; kp_off_err = []; kp_in_conf = []; kp_in_err = []
    for batch in tqdm(loader, desc="occ diag"):
        img = batch['image'].to(device)
        gt3d = batch['keypoints_3d'].to(device)
        gtkp = batch['keypoints'].to(device)          # (B,7,2) px @ image-size res (GT 2D)
        K = scale_K(batch['camera_K'], batch['original_size'], S).to(device)
        with torch.no_grad():
            out = mlp(img, K)
        init = out['joint_angles']; kp2d = out['keypoints_2d']; conf = out['confidence']
        refined, kp_cam, _ = solve_batch(kp2d, conf, K, fix_joint7=True, iters=args.iters,
                                         lr=2e-2, img_size=S, device=device, prior_w=0.0, theta_init=init)
        # ----- occlusion-aware: gate occluded keypoints out of PnP + refinement -----
        init_occ = init
        if args.mean_fallback:
            # unobservable joints (no reliable downstream keypoint) -> joint mean
            MOVES = {0:[1,2,3,4,5,6], 1:[1,2,3,4,5,6], 2:[2,3,4,5,6], 3:[3,4,5,6], 4:[4,5,6], 5:[4,5,6]}
            from solve_pose_kinematic import PANDA_JOINT_MEAN
            tmean = PANDA_JOINT_MEAN.to(device); reliable = conf >= args.conf_gate
            init_occ = init.clone()
            for j, ks in MOVES.items():
                init_occ[~reliable[:, ks].any(dim=1), j] = tmean[j]
        _, kp_cam_occ, _ = solve_batch(kp2d, conf, K, fix_joint7=True, iters=args.iters,
                                       lr=2e-2, img_size=S, device=device, prior_w=0.0, theta_init=init_occ,
                                       conf_gate=args.conf_gate, anchor_init_w=args.anchor_w)
        valid = (gt3d.abs().sum(-1) > 0)
        per_j = (kp_cam - gt3d).norm(dim=-1)
        per_j_occ = (kp_cam_occ - gt3d).norm(dim=-1)
        # GT off-frame mask: GT 2D outside [0,S)
        off = ~((gtkp[..., 0] >= 0) & (gtkp[..., 0] < S) & (gtkp[..., 1] >= 0) & (gtkp[..., 1] < S))  # (B,7)
        px_err = (kp2d - gtkp).norm(dim=-1)            # (B,7) predicted-vs-GT 2D px
        for b in range(img.shape[0]):
            m = valid[b]
            add = float(per_j[b][m].mean()) if m.any() else float('nan')
            add_occ = float(per_j_occ[b][m].mean()) if m.any() else float('nan')
            rows.append({'add': add, 'add_occ': add_occ, 'noff': int(off[b].sum()),
                         'minconf': float(conf[b].min()), 'meanconf': float(conf[b].mean())})
            for j in range(7):
                (kp_off_conf if off[b, j] else kp_in_conf).append(float(conf[b, j]))
                (kp_off_err if off[b, j] else kp_in_err).append(float(px_err[b, j]))

    adds = np.array([r['add'] for r in rows]); noff = np.array([r['noff'] for r in rows])
    adds_occ = np.array([r['add_occ'] for r in rows])
    minc = np.array([r['minconf'] for r in rows])
    ok_all = ~np.isnan(adds)
    print(f"\n{'='*60}\nOCCLUSION DIAG  {os.path.basename(args.val_dir)}  N={len(rows)}\n{'='*60}")
    print(f"baseline       ADD-AUC@100mm {add_auc(adds[ok_all]):.4f} | mean {np.nanmean(adds)*1000:.1f}mm median {np.nanmedian(adds)*1000:.1f}mm")
    print(f"occ-aware(gate={args.conf_gate},anc={args.anchor_w}) ADD-AUC {add_auc(adds_occ[ok_all]):.4f} | mean {np.nanmean(adds_occ)*1000:.1f}mm median {np.nanmedian(adds_occ)*1000:.1f}mm")
    print(f"\n(1) ADD by #off-frame GT keypoints  [baseline -> occ-aware]:")
    for k in sorted(set(noff.tolist())):
        msk = (noff == k) & ok_all
        sel = adds[msk]; selo = adds_occ[msk]
        if len(sel): print(f"   noff={k}: n={len(sel):4d}  mean {sel.mean()*1000:6.1f}->{selo.mean()*1000:6.1f}mm  AUC {add_auc(sel):.3f}->{add_auc(selo):.3f}")
    # correlation
    ok = ~np.isnan(adds)
    if ok.sum() > 2:
        print(f"\n   corr(ADD, #off-frame) = {np.corrcoef(adds[ok], noff[ok])[0,1]:+.3f}")
        print(f"   corr(ADD, min-conf)   = {np.corrcoef(adds[ok], minc[ok])[0,1]:+.3f}")
    print(f"\n(2) detector behavior on off-frame vs in-frame keypoints:")
    def stat(name, c, e):
        c=np.array(c); e=np.array(e)
        if len(c)==0: print(f"   {name}: none"); return
        print(f"   {name:10s} n={len(c):5d}  conf {c.mean():.3f}  2Dpx_err {e.mean():6.1f} (median {np.median(e):.1f})")
    stat('off-frame', kp_off_conf, kp_off_err)
    stat('in-frame', kp_in_conf, kp_in_err)
    print('='*60)


if __name__ == '__main__':
    main()
