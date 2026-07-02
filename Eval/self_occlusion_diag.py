"""
Quantify self-occlusion vulnerability (user observation from viz #2798).

Per frame compute two GT-based pose-difficulty proxies (independent of the prediction):
  foreshorten = (Zmax-Zmin of keypoints) / (2D bbox diagonal px)   -- arm along camera axis
  overlap2d   = # keypoint pairs that are FAR in 3D (>0.2m) but CLOSE in 2D (<thr px)
                -- links projecting on top of each other = self-occlusion
Then bin the refined ADD (conf-gate ON) by each proxy to see how much worse self-occluded
poses are, and what fraction of the failure tail they explain.
"""
import argparse, itertools, os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

TRAIN = os.path.abspath(os.path.join(os.path.dirname(__file__), '../TRAIN'))
sys.path.append(TRAIN); sys.path.append(os.path.dirname(__file__))
from model_angle import AnglePredictor
from dataset import PoseEstimationDataset
from solve_pose_kinematic import solve_batch
from refine_eval import scale_K, add_auc, wrapped_abs_deg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--detector', required=True); ap.add_argument('--mlp-head', required=True)
    ap.add_argument('--val-dir', required=True)
    ap.add_argument('--model-name', default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    ap.add_argument('--image-size', type=int, default=512); ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--max-frames', type=int, default=800); ap.add_argument('--conf-gate', type=float, default=0.05)
    ap.add_argument('--overlap-px', type=float, default=25.0)
    args = ap.parse_args()

    device = torch.device('cuda'); assert torch.cuda.is_available(); S = args.image_size
    m = AnglePredictor(args.model_name, S, head_type='mlp').to(device).eval()
    sd = torch.load(args.detector, map_location=device); sd = {k.replace('module.', ''): v for k, v in sd.items()}
    m.load_state_dict({k: v for k, v in sd.items() if k in m.state_dict() and v.shape == m.state_dict()[k].shape}, strict=False)
    m.angle_head.load_state_dict(torch.load(args.mlp_head, map_location=device))

    ds = PoseEstimationDataset(args.val_dir, keypoint_names=['link0','link2','link3','link4','link6','link7','hand'],
                               image_size=(S, S), heatmap_size=(S, S), augment=False, include_angles=True, sigma=2.5)
    if args.max_frames and args.max_frames < len(ds): ds.samples = ds.samples[:args.max_frames]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    pairs = list(itertools.combinations(range(7), 2))
    adds = []; fore = []; ovl = []; noff = []; angerr = []; reprojs = []; minconfs = []
    for batch in tqdm(loader, desc="self-occ diag"):
        img = batch['image'].to(device); gt3d = batch['keypoints_3d'].to(device)
        gtkp = batch['keypoints'].to(device); K = scale_K(batch['camera_K'], batch['original_size'], S).to(device)
        gta = batch['angles'][:, :6].to(device)
        with torch.no_grad(): out = m(img, K)
        refined, kp_cam, reproj = solve_batch(out['keypoints_2d'], out['confidence'], K, fix_joint7=True, iters=200,
                                              lr=2e-2, img_size=S, device=device, prior_w=0.0,
                                              theta_init=out['joint_angles'], conf_gate=args.conf_gate)
        amae = wrapped_abs_deg(refined[:, :6], gta).mean(dim=1)  # (B,) deg
        mnconf = out['confidence'].min(dim=1)[0]  # (B,)
        valid = (gt3d.abs().sum(-1) > 0); per_j = (kp_cam - gt3d).norm(dim=-1)
        for b in range(img.shape[0]):
            mvalid = valid[b]
            if not mvalid.any(): continue
            adds.append(float(per_j[b][mvalid].mean())); angerr.append(float(amae[b]))
            reprojs.append(float(reproj[b])); minconfs.append(float(mnconf[b]))
            k2 = gtkp[b]; k3 = gt3d[b]
            # foreshorten
            zr = float(k3[:, 2].max() - k3[:, 2].min())
            x0, y0 = k2[:, 0].min(), k2[:, 1].min(); x1, y1 = k2[:, 0].max(), k2[:, 1].max()
            diag = float(((x1-x0)**2 + (y1-y0)**2) ** 0.5) + 1e-6
            fore.append(zr / diag * 1000)  # mm depth per px
            # overlap pairs: 3D far but 2D close
            o = 0
            for i, j in pairs:
                d3 = float((k3[i]-k3[j]).norm()); d2 = float((k2[i]-k2[j]).norm())
                if d3 > 0.2 and d2 < args.overlap_px: o += 1
            ovl.append(o)
            off = int(((k2[:,0]<0)|(k2[:,0]>=S)|(k2[:,1]<0)|(k2[:,1]>=S)).sum()); noff.append(off)

    adds = np.array(adds); fore = np.array(fore); ovl = np.array(ovl); noff = np.array(noff)
    print(f"\n{'='*64}\nSELF-OCCLUSION DIAG  {os.path.basename(args.val_dir)}  N={len(adds)}  (conf-gate {args.conf_gate})\n{'='*64}")
    print(f"overall ADD-AUC {add_auc(adds):.4f} | mean {adds.mean()*1000:.1f}mm median {np.median(adds)*1000:.1f}mm")
    print(f"\n(1) ADD by foreshortening quartile (depth-range/2D-diag, mm/px):")
    qs = np.quantile(fore, [0, .25, .5, .75, 1.0])
    for a, b in zip(qs[:-1], qs[1:]):
        s = adds[(fore >= a) & (fore <= b)]
        if len(s): print(f"   fore [{a:5.2f},{b:5.2f}]: n={len(s):4d}  mean {s.mean()*1000:6.1f}mm  median {np.median(s)*1000:6.1f}mm  AUC {add_auc(s):.3f}")
    angerr = np.array(angerr); reprojs = np.array(reprojs); minconfs = np.array(minconfs)
    print(f"\n(2) by # 2D-overlap pairs (3D>0.2m but 2D<{args.overlap_px:.0f}px = self-occlusion):")
    for k in sorted(set(ovl.tolist())):
        sel = ovl == k; s = adds[sel]
        if len(s) >= 3: print(f"   overlap={k}: n={len(s):4d}  ADD {s.mean()*1000:6.1f}mm  angleMAE {angerr[sel].mean():5.1f}deg  |  reproj {reprojs[sel].mean():6.1f}px  min-conf {minconfs[sel].mean():.3f}")
    print(f"   FLAGGABLE? overlap-frame reproj vs clean: if high, can detect+cap. clean reproj {reprojs[ovl==0].mean():.1f}px")
    print(f"\n   corr(ADD, foreshorten) = {np.corrcoef(adds, fore)[0,1]:+.3f}")
    print(f"   corr(ADD, overlap2d)   = {np.corrcoef(adds, ovl)[0,1]:+.3f}")
    print(f"   corr(ADD, off-frame)   = {np.corrcoef(adds, noff)[0,1]:+.3f}")
    # what explains the failure tail?
    tail = adds > 0.1  # ADD > 100mm = AUC-killing failures
    print(f"\n(3) Failure tail (ADD>100mm): {tail.sum()}/{len(adds)} frames ({tail.mean()*100:.1f}%)")
    if tail.sum():
        print(f"   of failures: {(ovl[tail]>0).mean()*100:.0f}% have 2D-overlap(self-occ), "
              f"{(noff[tail]>0).mean()*100:.0f}% off-frame, "
              f"{(fore[tail]>np.median(fore)).mean()*100:.0f}% above-median foreshorten")
    print('='*64)


if __name__ == '__main__':
    main()
