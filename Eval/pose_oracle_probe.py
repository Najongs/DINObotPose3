"""Phase D ceiling probe (dino env, CPU/GPU, seconds). Decompose the crop-pose ADD error into
THETA (joint-angle) vs POSE (R,t) on the dumped crop pose. For each frame: FK(theta_solved), then
Kabsch-align FK to the GT 3D keypoints -> the BEST rigid (R,t) given the solved angles. The ADD after
that alignment = pure theta-error residual; the gap current->GT-rigid = the (R,t) error = the CEILING
for ANY pose-refinement lever (Phase D render-compare refiner, Phase C). Also splits R-only vs t-only.
If GT-rigid ~= current, theta is the bottleneck (render-compare can't help); if GT-rigid >> current,
there is large R,t headroom (Phase D worth it)."""
import argparse, os, sys
import numpy as np
import torch

TRAIN = os.path.abspath(os.path.join(os.path.dirname(__file__), '../TRAIN'))
sys.path.append(TRAIN)
from model_v4 import panda_forward_kinematics
from model_angle import kabsch_batch

ap = argparse.ArgumentParser()
ap.add_argument('--npz', required=True)
ap.add_argument('--max-frames', type=int, default=1000)
a = ap.parse_args()

d = np.load(a.npz)
theta = torch.from_numpy(d['theta']).float()
kp_cam = torch.from_numpy(d['kp_cam']).float()
gt3d = torch.from_numpy(d['gt3d']).float()
found = torch.from_numpy(d['found']).bool() if 'found' in d.files else (gt3d.abs().sum(-1) > 0)
N = min(a.max_frames, len(theta))
theta, kp_cam, gt3d, found = theta[:N], kp_cam[:N], gt3d[:N], found[:N]

fk = panda_forward_kinematics(theta)                # (N,7,3) robot frame
w = found.float()                                   # (N,7) weight valid keypoints
w = torch.where(w.sum(1, keepdim=True) > 0, w, torch.ones_like(w))

# Kabsch FK -> GT (best rigid R,t given solved angles)  => theta-only residual
R_gt, t_gt = kabsch_batch(fk, gt3d, w)
pred_rigid = torch.bmm(fk, R_gt.transpose(1, 2)) + t_gt.unsqueeze(1)
# Kabsch FK -> solved kp_cam (recovers the solved R,t exactly)
R_s, t_s = kabsch_batch(fk, kp_cam, w)
# swap-only R (keep solved t): isolate R error
pred_Ronly = torch.bmm(fk, R_gt.transpose(1, 2)) + t_s.unsqueeze(1)
# swap-only t (keep solved R): isolate t error
pred_tonly = torch.bmm(fk, R_s.transpose(1, 2)) + t_gt.unsqueeze(1)


def add_auc(pred):
    e = (pred - gt3d).norm(dim=-1)                  # (N,7)
    per = torch.stack([e[i][found[i]].mean() if found[i].any() else e[i].mean() for i in range(N)])
    er = np.sort(per.numpy())
    auc = np.mean([np.sum(er < i / 10000.0) / len(er) for i in range(1000)])
    return auc, er.mean() * 1000

for name, pred in [('current (solved R,t,theta)', kp_cam),
                   ('GT-rigid (theta-only resid)', pred_rigid),
                   ('swap R only (keep solved t)', pred_Ronly),
                   ('swap t only (keep solved R)', pred_tonly)]:
    auc, mm = add_auc(pred)
    print(f"  {name:<30}: ADD-AUC {auc:.4f}  mean {mm:5.1f}mm")
