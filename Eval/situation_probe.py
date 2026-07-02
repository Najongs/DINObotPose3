"""
Does the EXISTING DINOv3 backbone already 'understand the situation' (occluded / self-occluded
/ about-to-fail) — so a small head suffices and no 2B VLM is needed?

Fit a tiny logistic-regression head on the FROZEN DINOv3 global feature (mean-pooled tokens,
appearance only — no keypoint geometry) to predict, per frame:
  self_occ : >=1 keypoint pair far in 3D (>0.2m) but close in 2D (<25px)  [self-occlusion]
  fail     : refined ADD > 100mm                                          [will the pose fail]
Report test AUROC, vs simple baselines (min-conf, reproj residual). High AUROC from appearance
alone = the scene understanding is already in the backbone.
"""
import argparse, itertools, os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

TRAIN = os.path.abspath(os.path.join(os.path.dirname(__file__), '../TRAIN'))
sys.path.append(TRAIN); sys.path.append(os.path.dirname(__file__))
from model_angle import AnglePredictor
from dataset import PoseEstimationDataset
from solve_pose_kinematic import solve_batch
from refine_eval import scale_K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--detector', required=True); ap.add_argument('--mlp-head', required=True)
    ap.add_argument('--val-dir', required=True)
    ap.add_argument('--model-name', default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    ap.add_argument('--image-size', type=int, default=512); ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--max-frames', type=int, default=2500); ap.add_argument('--conf-gate', type=float, default=0.05)
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
    feats = []; y_occ = []; y_fail = []; minconf = []; reprojpx = []
    for batch in tqdm(loader, desc="situation probe"):
        img = batch['image'].to(device); gt3d = batch['keypoints_3d'].to(device)
        gtkp = batch['keypoints'].to(device); K = scale_K(batch['camera_K'], batch['original_size'], S).to(device)
        with torch.no_grad():
            tokens = m.backbone(img)               # (B,Np,C) frozen DINOv3
            gfeat = tokens.mean(dim=1)             # (B,C) appearance-only global feature
            out = m(img, K)
        refined, kp_cam, reproj = solve_batch(out['keypoints_2d'], out['confidence'], K, fix_joint7=True, iters=200,
                                              lr=2e-2, img_size=S, device=device, prior_w=0.0,
                                              theta_init=out['joint_angles'], conf_gate=args.conf_gate)
        valid = (gt3d.abs().sum(-1) > 0); per_j = (kp_cam - gt3d).norm(dim=-1)
        for b in range(img.shape[0]):
            mv = valid[b]
            if not mv.any(): continue
            feats.append(gfeat[b].cpu().numpy())
            add = float(per_j[b][mv].mean()); y_fail.append(int(add > 0.1))
            k2 = gtkp[b]; k3 = gt3d[b]
            o = sum(1 for i, j in pairs if float((k3[i]-k3[j]).norm()) > 0.2 and float((k2[i]-k2[j]).norm()) < 25)
            y_occ.append(int(o >= 1))
            minconf.append(float(out['confidence'][b].min())); reprojpx.append(float(reproj[b]))

    X = np.array(feats); y_occ = np.array(y_occ); y_fail = np.array(y_fail)
    minconf = np.array(minconf); reprojpx = np.array(reprojpx)
    print(f"\n{'='*64}\nSITUATION PROBE  {os.path.basename(args.val_dir)}  N={len(X)}  feat-dim={X.shape[1]}\n{'='*64}")
    print(f"  self_occ rate {y_occ.mean()*100:.1f}%   fail(>100mm) rate {y_fail.mean()*100:.1f}%")

    def probe(name, y):
        if y.sum() < 5 or (len(y)-y.sum()) < 5:
            print(f"\n  [{name}] too few positives ({y.sum()})"); return
        Xtr, Xte, ytr, yte, mc_tr, mc_te, rp_tr, rp_te = train_test_split(
            X, y, minconf, reprojpx, test_size=0.4, random_state=0, stratify=y)
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=2000, class_weight='balanced').fit(sc.transform(Xtr), ytr)
        auc_feat = roc_auc_score(yte, clf.predict_proba(sc.transform(Xte))[:, 1])
        auc_conf = roc_auc_score(yte, -mc_te)            # low conf -> more likely
        auc_rp = roc_auc_score(yte, rp_te)               # high reproj -> more likely
        print(f"\n  [{name}] test AUROC:")
        print(f"     DINOv3 global feature (appearance only) : {auc_feat:.3f}   <-- 'situation understanding'")
        print(f"     baseline min-confidence                 : {auc_conf:.3f}")
        print(f"     baseline reproj residual                : {auc_rp:.3f}")

    probe('self_occ', y_occ)
    probe('fail>100mm', y_fail)
    print('='*64)


if __name__ == '__main__':
    main()
