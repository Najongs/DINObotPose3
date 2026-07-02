"""
Stage 2 — Kinematic angle + pose solver (the "decisive experiment").

Idea (see plan resilient-sparking-castle.md):
    Stop regressing joint angles. Detect 2D keypoints well (any checkpoint with a
    ViTKeypointHead -> heatmaps_2d), then SOLVE the joint angles (theta) and the
    camera pose (R, t) geometrically by minimizing a confidence-weighted
    reprojection error of the analytic forward kinematics:

        min_{theta, R, t}  sum_j  conf_j * rho( project(FK(theta)_j; R, t, K) - kp2d_j )

    Joint angles are clamped to their mechanical limits via a sigmoid
    re-parametrization; the camera rotation uses the 6D continuous representation
    (Zhou et al. 2019). Initialization comes from a single OpenCV PnP solve on
    FK(theta_mean), with optional multi-start to escape local minima.

This is the cheap, keypoint-based cousin of RoboPose's render-and-compare.

It is intentionally model-agnostic: it only requires the model forward to return
`heatmaps_2d`. Works with model.py / model_v3.py / model_v4.py checkpoints and the
2D-pretrained best_heatmap.pth.

Usage (quick decisive run on a few hundred frames):
    python Eval/solve_pose_kinematic.py \
        -p TRAIN/outputs_heatmap/best_heatmap.pth \
        -d Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_test_dr \
        --model-module model_v4 --model-class DINOv3PoseEstimatorV4 \
        --max-frames 300 -o Eval/results_kinematic
"""

import argparse
import importlib
import json
import math
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

TRAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../TRAIN'))
sys.path.append(TRAIN_DIR)
sys.path.append(os.path.dirname(__file__))

from model_v4 import panda_forward_kinematics, soft_argmax_2d, _PANDA_JOINT_LIMITS  # noqa: E402
from inference_4tier_eval import EvalDataset, compute_add_auc  # reuse loader + metric

# Hardcoded mean used only as the optimization start point for theta.
PANDA_JOINT_MEAN = torch.tensor([-5.22e-02, 2.68e-01, 6.04e-03, -2.01e+00, 1.49e-02, 1.99e+00, 0.0])

# Keypoint names exactly as expected by EvalDataset / FK keypoint order.
KEYPOINT_NAMES = ['panda_link0', 'panda_link2', 'panda_link3',
                  'panda_link4', 'panda_link6', 'panda_link7', 'panda_hand']


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def rot6d_to_matrix(d6):
    """(B, 6) 6D rotation representation -> (B, 3, 3). Zhou et al. 2019."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)  # columns are basis vectors


def matrix_to_rot6d(R):
    """(B, 3, 3) -> (B, 6): first two columns flattened."""
    return torch.cat([R[..., 0], R[..., 1]], dim=-1)


def ik_from_3d(pred_kp_3d, theta_mean, lo, hi, iters=150, lr=5e-2):
    """
    Recover joint angles from predicted robot-frame 3D keypoints by fitting FK.
    (Mirrors eval_3d_v3.optimize_ik_batch but with joint-limit clamping.)
    pred_kp_3d: (B,7,3) tensor. Returns theta (B,7) with joint7=0.
    """
    B = pred_kp_3d.shape[0]
    device = pred_kp_3d.device
    theta0 = theta_mean.unsqueeze(0).expand(B, 7).clone()
    theta0[:, 6] = 0.0
    theta0 = torch.max(torch.min(theta0, hi), lo)
    p = theta_to_p(theta0, lo, hi).clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([p], lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        theta = p_to_theta(p, lo, hi)
        theta = torch.cat([theta[:, :6], torch.zeros(B, 1, device=device)], dim=1)
        fk = panda_forward_kinematics(theta)
        loss = F.mse_loss(fk, pred_kp_3d)
        loss.backward()
        opt.step()
    with torch.no_grad():
        theta = p_to_theta(p, lo, hi)
        theta = torch.cat([theta[:, :6], torch.zeros(B, 1, device=device)], dim=1)
    return theta.detach()


def project_points(pts_robot, R, t, K):
    """
    pts_robot: (B, N, 3) robot-frame FK keypoints
    R: (B, 3, 3) robot->camera, t: (B, 3), K: (B, 3, 3)
    returns (B, N, 2) pixel coords
    """
    pts_cam = torch.bmm(pts_robot, R.transpose(1, 2)) + t.unsqueeze(1)  # (B,N,3)
    pts_img = torch.bmm(pts_cam, K.transpose(1, 2))                      # (B,N,3)
    z = pts_img[..., 2:3].clamp(min=1e-6)
    return pts_img[..., :2] / z, pts_cam


# ---------------------------------------------------------------------------
# Joint-limit re-parametrization: theta = lo + (hi-lo) * sigmoid(p)
# ---------------------------------------------------------------------------
def make_limits(device, dtype):
    lo = torch.tensor([l for l, _ in _PANDA_JOINT_LIMITS], device=device, dtype=dtype)
    hi = torch.tensor([h for _, h in _PANDA_JOINT_LIMITS], device=device, dtype=dtype)
    return lo, hi  # (7,)


def theta_to_p(theta, lo, hi):
    """Inverse sigmoid map, with safe clamping just inside the limits."""
    frac = ((theta - lo) / (hi - lo)).clamp(1e-4, 1 - 1e-4)
    return torch.log(frac / (1 - frac))


def p_to_theta(p, lo, hi):
    return lo + (hi - lo) * torch.sigmoid(p)


# ---------------------------------------------------------------------------
# PnP initialization (single solve on FK(theta_init))
# ---------------------------------------------------------------------------
def pnp_init(kp_2d, kp_3d_robot, K, conf=None, conf_gate=0.0, min_kp=6, pnp_rel=0.0, pnp_drop=3):
    """
    Confidence-ranked PnP for a robust (R, t) initialization.
    kp_2d: (B,N,2) np, kp_3d_robot: (B,N,3) np, K: (B,3,3) np, conf: (B,N) np or None.

    KEY EMPIRICAL FINDING (Eval/solve_sweep.py, strided 600 frames/cam): the cleaner the PnP
    INIT, the better the final pose — because the gradient refine uses ALL points to polish, so
    the init only has to pick the right basin, and a few HIGH-confidence points give the right
    basin while low-conf points (far-camera distance noise) reproject within tolerance but bias
    EPnP's depth. Initializing from the top-(N-pnp_drop) most-confident keypoints (pnp_drop=3 ->
    top-4) beats all-7 init on EVERY split (mean ADD-AUC 0.663 vs 0.600). Minimal sets can be
    degenerate, so we FALL BACK to progressively more points until PnP is valid.
    returns R (B,3,3), t (B,3) numpy, valid (B,) bool
    """
    B, N = kp_2d.shape[:2]
    Rs = np.tile(np.eye(3), (B, 1, 1)).astype(np.float64)
    ts = np.tile(np.array([0.0, 0.0, 1.2]), (B, 1)).astype(np.float64)
    valid = np.zeros(B, dtype=bool)
    k0 = max(4, N - pnp_drop)  # smallest (cleanest) init set; grow on degeneracy
    for b in range(B):
        order = np.argsort(-conf[b]) if conf is not None else np.arange(N)
        for k in range(k0, N + 1):                     # top-k, fall back to more pts if invalid
            sel = order[:k]
            p3 = kp_3d_robot[b][sel].astype(np.float64)
            p2 = kp_2d[b][sel].astype(np.float64)
            try:
                ok, rvec, tvec, inl = cv2.solvePnPRansac(
                    p3, p2, K[b].astype(np.float64), None,
                    iterationsCount=200, reprojectionError=5.0, flags=cv2.SOLVEPNP_EPNP)
                if ok and inl is not None and len(inl) >= 4:
                    idx = inl.flatten()
                    ok2, rvec, tvec = cv2.solvePnP(
                        p3[idx], p2[idx], K[b].astype(np.float64), None,
                        useExtrinsicGuess=True, rvec=rvec, tvec=tvec,
                        flags=cv2.SOLVEPNP_ITERATIVE)
                    R, _ = cv2.Rodrigues(rvec)
                    if np.all(np.isfinite(R)) and np.all(np.isfinite(tvec)) and tvec.flatten()[2] > 0:
                        Rs[b], ts[b], valid[b] = R, tvec.flatten(), True
                        break
            except cv2.error:
                pass
    return Rs, ts, valid


# ---------------------------------------------------------------------------
# Core optimizer (batched, all frames independent)
# ---------------------------------------------------------------------------
def solve_batch(kp_2d, conf, K, fix_joint7=True, iters=250, lr=5e-2,
                img_size=512, device='cuda', prior_w=2e-3, theta_init=None,
                conf_gate=0.0, anchor_init_w=0.0, min_kp=6, pnp_rel=0.0, pnp_drop=3,
                R_init=None, t_init=None, gt_tz=None, depth_w=0.0, return_pose=False):
    """
    kp_2d: (B,N,2) tensor, conf: (B,N) tensor, K: (B,3,3) tensor.
    theta_init: optional (B,7) tensor — learned-prediction init (refinement mode).
                If None, init from the joint mean (cold start).

    Occlusion handling (the off-frame/occluded-keypoint failure tail):
      conf_gate>0     : HARD-reject keypoints with conf<gate (weight exactly 0) so PnP and
                        the reprojection residual fully ignore hallucinated occluded points.
      anchor_init_w>0 : anchor angles to theta_init (the LEARNED prediction), not the dataset
                        mean. Joints whose keypoints are gated out keep no data constraint, so
                        this prior makes them fall back to the learned estimate instead of
                        drifting -> "when occluded, predict from the kinematic/learned prior".
    Returns theta (B,7), kp_cam (B,N,3), reproj_px (B,).
    """
    B, N = kp_2d.shape[:2]
    dtype = torch.float32
    lo, hi = make_limits(device, dtype)

    theta_mean = PANDA_JOINT_MEAN.to(device, dtype)

    # --- init theta (learned prior if given, else mean), init R,t via PnP ---
    if theta_init is not None:
        theta0 = theta_init.to(device, dtype).clone()
    else:
        theta0 = theta_mean.unsqueeze(0).expand(B, 7).clone()
    if fix_joint7:
        theta0[:, 6] = 0.0
    theta0 = torch.max(torch.min(theta0, hi), lo)  # clamp into limits
    fk0 = panda_forward_kinematics(theta0)  # (B,N,3) robot frame
    R0, t0, _ = pnp_init(kp_2d.cpu().numpy(), fk0.detach().cpu().numpy(),
                         K.cpu().numpy(), conf.cpu().numpy(), conf_gate=conf_gate,
                         min_kp=min_kp, pnp_rel=pnp_rel, pnp_drop=pnp_drop)
    R0 = torch.from_numpy(R0).to(device, dtype)
    t0 = torch.from_numpy(t0).to(device, dtype)
    # Optional learned/oracle pose init to escape the far-camera rotation-basin ambiguity
    # (the reprojection objective is degenerate; a prior on R is what pins the basin).
    if R_init is not None:
        R0 = R_init.to(device, dtype).clone()
    if t_init is not None:
        t0 = t_init.to(device, dtype).clone()

    # --- init geometry (for the per-frame divergence guard) ---
    fk_init = panda_forward_kinematics(theta0)
    R0m = rot6d_to_matrix(matrix_to_rot6d(R0))  # orthonormalized to match refine path
    uv_init, kpcam_init = project_points(fk_init, R0m, t0, K)
    reproj_init = (uv_init - kp_2d).norm(dim=-1).mean(dim=1)  # (B,)

    # --- learnable params ---
    p = theta_to_p(theta0, lo, hi).clone().detach().requires_grad_(True)
    d6 = matrix_to_rot6d(R0).clone().detach().requires_grad_(True)
    t = t0.clone().detach().requires_grad_(True)

    opt = torch.optim.Adam([p, d6, t], lr=lr)
    # base confidence weights, normalized per-frame
    base_w = conf.clamp(min=1e-3)
    if conf_gate > 0.0:
        keep = conf >= conf_gate                                   # (B,N) HARD-reject occluded kp
        if min_kp > 0:
            # floor: always keep the top-min_kp by conf so far cameras (low conf everywhere)
            # are not starved into a degenerate PnP. Only drops the genuine off-frame tail.
            rank = torch.argsort(conf, dim=1, descending=True)
            topk = torch.zeros_like(keep)
            topk.scatter_(1, rank[:, :min_kp], True)
            keep = keep | topk
        base_w = base_w * keep.to(base_w.dtype)
    base_w = base_w / base_w.sum(dim=1, keepdim=True).clamp(min=1e-6)  # (B,N)
    theta_init_p = theta0.detach().clone()  # anchor target = learned init (post-clamp)
    w = base_w.clone()
    huber_px = 8.0  # robust threshold in pixels

    for it in range(iters):
        opt.zero_grad()
        theta = p_to_theta(p, lo, hi)
        if fix_joint7:
            theta = torch.cat([theta[:, :6], torch.zeros(B, 1, device=device)], dim=1)
        fk = panda_forward_kinematics(theta)            # (B,N,3)
        R = rot6d_to_matrix(d6)
        uv, _ = project_points(fk, R, t, K)             # (B,N,2)
        resid_px = (uv - kp_2d).norm(dim=-1)            # (B,N) pixels
        # IRLS robust reweighting (Geman-McClure-ish): outliers get down-weighted.
        if it > 30:
            robust = (huber_px ** 2) / (huber_px ** 2 + resid_px.detach() ** 2)
            w = base_w * robust
            w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-6)
        # Huber on normalized pixel residual for stable LR
        res = (uv - kp_2d) / img_size
        loss_per = F.huber_loss(res, torch.zeros_like(res), delta=0.01, reduction='none').sum(-1)
        loss = (w * loss_per).sum(dim=1).mean()
        # Light prior on theta to resolve depth-ambiguous null-space directions.
        loss = loss + prior_w * ((theta[:, :6] - theta_mean[:6]) ** 2).mean()
        # GT-depth ceiling probe: anchor solved root depth t_z to GT base depth, re-solve R,theta
        # consistently around it (gauge-safe oracle test of "would correct depth fix the pose?").
        if depth_w > 0.0 and gt_tz is not None:
            loss = loss + depth_w * ((t[:, 2] - gt_tz) ** 2).mean()
        # Anchor to the LEARNED init: occluded joints (no data constraint after gating)
        # fall back to the learned prediction instead of drifting.
        if anchor_init_w > 0.0:
            loss = loss + anchor_init_w * ((theta[:, :6] - theta_init_p[:, :6]) ** 2).mean()
        loss.backward()
        opt.step()

    with torch.no_grad():
        theta = p_to_theta(p, lo, hi)
        if fix_joint7:
            theta = torch.cat([theta[:, :6], torch.zeros(B, 1, device=device)], dim=1)
        fk = panda_forward_kinematics(theta)
        R = rot6d_to_matrix(d6)
        uv, kp_cam = project_points(fk, R, t, K)
        reproj_px = (uv - kp_2d).norm(dim=-1).mean(dim=1)  # (B,)
        t_out = t.clone()

        # Per-frame divergence guard: keep refined only where it lowered reprojection,
        # else fall back to the (learned) init. Refinement is then never harmful.
        # nan reproj (degenerate minimal-PnP frame) counts as worse -> fall back to init.
        worse = ~(reproj_px < reproj_init)
        if worse.any():
            theta[worse] = theta0[worse]
            kp_cam[worse] = kpcam_init[worse]
            reproj_px[worse] = reproj_init[worse]
            R[worse] = R0m[worse]
            t_out[worse] = t0[worse]
    if return_pose:
        return theta.detach(), kp_cam.detach(), reproj_px.detach(), R.detach(), t_out.detach()
    return theta.detach(), kp_cam.detach(), reproj_px.detach()


def sample_heatmap_at(heatmaps, uv, H, W):
    """heatmaps (B,N,H,W), uv (B,N,2) px -> (B,N) bilinearly-sampled heatmap value at uv.
    Differentiable w.r.t. uv (-> w.r.t. theta,R,t). 'border' padding keeps gradient finite
    when a projection lands off-image."""
    B, N = uv.shape[:2]
    hm = heatmaps.reshape(B * N, 1, H, W)
    g = uv.reshape(B * N, 1, 1, 2).clone()
    g0 = g[..., 0] / max(W - 1, 1) * 2 - 1
    g1 = g[..., 1] / max(H - 1, 1) * 2 - 1
    grid = torch.stack([g0, g1], dim=-1)
    s = F.grid_sample(hm, grid, mode='bilinear', align_corners=True, padding_mode='border')
    return s.reshape(B, N)


def solve_batch_heatmap(heatmaps, K, fix_joint7=True, iters=200, lr=1e-2,
                        img_size=512, device='cuda', theta_init=None,
                        anchor_w=0.15, prior_w=2e-3):
    """
    Heatmap-based BIDIRECTIONAL refiner. Instead of fitting the argmax keypoints, optimize
    (theta, R, t) to MAXIMIZE the heatmap response at the reprojected FK keypoints — so the
    kinematic chain (FK) selects the heatmap mode that is structurally consistent and IGNORES
    broken argmax detections (kinematics corrects keypoints). A weak anchor to the argmax keeps
    the basin; per-frame guard keeps the init if heatmap response doesn't improve.

    heatmaps: (B,N,H,W). K: (B,3,3). theta_init: (B,N_ang+1) learned angles.
    Returns theta (B,7), kp_cam (B,N,3), response (B,).
    """
    B, N, H, W = heatmaps.shape
    dtype = torch.float32
    lo, hi = make_limits(device, dtype)
    theta_mean = PANDA_JOINT_MEAN.to(device, dtype)

    kp_argmax = soft_argmax_2d(heatmaps)                 # (B,N,2)
    conf = heatmaps.flatten(2).max(dim=2)[0]             # (B,N)
    hm_norm = heatmaps / (heatmaps.flatten(2).max(dim=2)[0].view(B, N, 1, 1) + 1e-6)

    theta0 = (theta_init.to(device, dtype).clone() if theta_init is not None
              else theta_mean.unsqueeze(0).expand(B, 7).clone())
    if fix_joint7:
        theta0[:, 6] = 0.0
    theta0 = torch.max(torch.min(theta0, hi), lo)
    fk0 = panda_forward_kinematics(theta0)
    R0, t0, _ = pnp_init(kp_argmax.cpu().numpy(), fk0.detach().cpu().numpy(),
                         K.cpu().numpy(), conf.cpu().numpy())
    R0 = torch.from_numpy(R0).to(device, dtype); t0 = torch.from_numpy(t0).to(device, dtype)

    base_w = conf.clamp(min=1e-3); base_w = base_w / base_w.sum(dim=1, keepdim=True)

    # init heatmap response (for the guard)
    with torch.no_grad():
        uv_i, kpcam_init = project_points(fk0, rot6d_to_matrix(matrix_to_rot6d(R0)), t0, K)
        resp_init = (base_w * sample_heatmap_at(hm_norm, uv_i, H, W)).sum(dim=1)

    p = theta_to_p(theta0, lo, hi).clone().detach().requires_grad_(True)
    d6 = matrix_to_rot6d(R0).clone().detach().requires_grad_(True)
    t = t0.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([p, d6, t], lr=lr)

    for _ in range(iters):
        opt.zero_grad()
        theta = p_to_theta(p, lo, hi)
        if fix_joint7:
            theta = torch.cat([theta[:, :6], torch.zeros(B, 1, device=device)], dim=1)
        fk = panda_forward_kinematics(theta)
        R = rot6d_to_matrix(d6)
        uv, _ = project_points(fk, R, t, K)
        s = sample_heatmap_at(hm_norm, uv, H, W)         # (B,N) in [0,1]
        loss_hm = -(base_w * s).sum(dim=1).mean()        # maximize heatmap response
        # weak anchor to argmax (basin), confidence-weighted, robust
        res = (uv - kp_argmax) / img_size
        anchor = (base_w * F.huber_loss(res, torch.zeros_like(res), delta=0.02,
                                        reduction='none').sum(-1)).sum(dim=1).mean()
        loss = loss_hm + anchor_w * anchor + prior_w * ((theta[:, :6] - theta_mean[:6]) ** 2).mean()
        loss.backward(); opt.step()

    with torch.no_grad():
        theta = p_to_theta(p, lo, hi)
        if fix_joint7:
            theta = torch.cat([theta[:, :6], torch.zeros(B, 1, device=device)], dim=1)
        fk = panda_forward_kinematics(theta)
        R = rot6d_to_matrix(d6)
        uv, kp_cam = project_points(fk, R, t, K)
        resp = (base_w * sample_heatmap_at(hm_norm, uv, H, W)).sum(dim=1)
        # guard: keep refined only where heatmap response improved, else fall back to init
        worse = resp < resp_init
        if worse.any():
            theta[worse] = theta0[worse]
            kp_cam[worse] = kpcam_init[worse]
            resp[worse] = resp_init[worse]
    return theta.detach(), kp_cam.detach(), resp.detach()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Kinematic angle+pose solver (Stage 2)")
    ap.add_argument("-p", "--model-path", required=True)
    ap.add_argument("-d", "--dataset-dir", required=True)
    ap.add_argument("-o", "--output-dir", default="Eval/results_kinematic")
    ap.add_argument("--model-module", default="model_v4",
                    help="python module in TRAIN/ that defines the model class")
    ap.add_argument("--model-class", default="DINOv3PoseEstimatorV4")
    ap.add_argument("--model-name", default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--iters", type=int, default=250)
    ap.add_argument("--lr", type=float, default=5e-2)
    ap.add_argument("--prior-w", type=float, default=0.0,
                    help="weight of the angle prior toward the joint mean")
    ap.add_argument("--init-mode", default="mean", choices=["mean", "ik3d", "anglehead"],
                    help="theta init: cold mean / IK from predicted 3D / learned angle head")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all frames")
    ap.add_argument("--fix-joint7", action="store_true", default=True)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # --- model ---
    mod = importlib.import_module(args.model_module)
    ModelCls = getattr(mod, args.model_class)
    model = ModelCls(dino_model_name=args.model_name,
                     heatmap_size=(args.image_size, args.image_size),
                     unfreeze_blocks=0).to(device)
    try:
        from checkpoint_compat import load_checkpoint_compat
        load_checkpoint_compat(model, args.model_path, device, is_main_process=True)
    except Exception:
        sd = torch.load(args.model_path, map_location=device)
        sd = sd.get('model_state_dict', sd.get('state_dict', sd))
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"  loaded (non-strict): {len(missing)} missing, {len(unexpected)} unexpected keys")
    model.eval()

    # --- data ---
    ds = EvalDataset(args.dataset_dir, KEYPOINT_NAMES,
                     image_size=(args.image_size, args.image_size))
    if args.max_frames and args.max_frames < len(ds):
        ds.json_files = ds.json_files[:args.max_frames]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    angle_errs = []      # per-frame (7,) deg  (REFINED)
    init_errs = []       # per-frame (7,) deg  (raw learned init)
    adds = []            # per-frame ADD (m)
    per_joint_3d = []    # per-frame (7,) m
    reproj_all = []
    kp2d_err = []        # detected-vs-GT 2D error (px @ heatmap res), per found kp

    lo_t, hi_t = make_limits(device, torch.float32)
    theta_mean_t = PANDA_JOINT_MEAN.to(device)

    for batch in tqdm(loader, desc="Solving"):
        images = batch["image"].to(device)
        gt_3d = batch["gt_3d"].numpy()         # (B,7,3) meters (cam frame, synth)
        gt_2d = batch["gt_2d"].numpy()         # (B,7,2) px @ ORIGINAL res
        found = batch["found"].numpy()         # (B,7)
        gt_angles = batch["gt_angles"].numpy() # (B,7) radians
        camera_K = batch["camera_K"].clone().float()
        orig = batch["original_size"].numpy()
        B = images.shape[0]

        # scale K to heatmap resolution
        for b in range(B):
            ow, oh = orig[b]
            sx, sy = args.image_size / ow, args.image_size / oh
            camera_K[b, 0, 0] *= sx; camera_K[b, 1, 1] *= sy
            camera_K[b, 0, 2] *= sx; camera_K[b, 1, 2] *= sy
        camera_K = camera_K.to(device)

        with torch.no_grad():
            out = model(images, camera_K=camera_K)
            hm = out["heatmaps_2d"]
            kp_2d = soft_argmax_2d(hm)                       # (B,7,2) @ heatmap res
            conf = hm.flatten(2).max(dim=2)[0]               # (B,7)

        # --- learned angle init ---
        if args.init_mode == "ik3d":
            theta_init = ik_from_3d(out["keypoints_3d"], theta_mean_t, lo_t, hi_t)
        elif args.init_mode == "anglehead":
            theta_init = out["joint_angles"]                 # (B,7) radians
        else:
            theta_init = None
        init_np = theta_init.cpu().numpy() if theta_init is not None else None

        # Detector accuracy: detected vs GT 2D (scaled to heatmap res)
        kp_2d_np = kp_2d.cpu().numpy()
        for b in range(B):
            ow, oh = orig[b]
            sx, sy = args.image_size / ow, args.image_size / oh
            gt2d_s = gt_2d[b] * np.array([sx, sy])
            for j in range(7):
                if found[b, j] > 0:
                    kp2d_err.append(float(np.linalg.norm(kp_2d_np[b, j] - gt2d_s[j])))

        theta, kp_cam, reproj = solve_batch(
            kp_2d, conf, camera_K, fix_joint7=args.fix_joint7,
            iters=args.iters, lr=args.lr, img_size=args.image_size, device=device,
            prior_w=args.prior_w, theta_init=theta_init)

        theta = theta.cpu().numpy()
        kp_cam = kp_cam.cpu().numpy()
        reproj = reproj.cpu().numpy()

        for b in range(B):
            if np.any(gt_angles[b] != 0):
                d = theta[b] - gt_angles[b]
                d = np.arctan2(np.sin(d), np.cos(d))
                angle_errs.append(np.abs(np.degrees(d)))
                if init_np is not None:
                    di = np.arctan2(np.sin(init_np[b] - gt_angles[b]),
                                    np.cos(init_np[b] - gt_angles[b]))
                    init_errs.append(np.abs(np.degrees(di)))
            if np.any(gt_3d[b] != 0):
                pj = np.linalg.norm(kp_cam[b] - gt_3d[b], axis=1)  # (7,) m
                per_joint_3d.append((pj, float(reproj[b])))  # carry reproj for validity filter
                adds.append(float(pj.mean()))
            reproj_all.append(float(reproj[b]))

    # --- report ---
    os.makedirs(args.output_dir, exist_ok=True)
    print("\n" + "=" * 64)
    print("  KINEMATIC SOLVER RESULTS (Stage 2)")
    print("=" * 64)
    metrics = {"dataset": args.dataset_dir, "checkpoint": args.model_path,
               "n_frames": len(reproj_all)}

    if angle_errs:
        ae = np.array(angle_errs)  # (M,7)  refined
        print(f"\n  [Joint Angle MAE]  ({len(ae)} frames w/ GT)  init-mode={args.init_mode}")
        if init_errs:
            ie = np.array(init_errs)
            print(f"    {'joint':<6}{'raw-init':>10}{'refined':>10}{'delta':>9}")
            for j in range(6):
                print(f"    J{j:<5}{ie[:, j].mean():>9.2f}{ae[:, j].mean():>10.2f}"
                      f"{ae[:, j].mean()-ie[:, j].mean():>+9.2f}")
            print(f"    {'MEAN6':<6}{ie[:, :6].mean():>9.2f}{ae[:, :6].mean():>10.2f}"
                  f"{ae[:, :6].mean()-ie[:, :6].mean():>+9.2f}  deg")
            metrics["init_mae_deg_mean6"] = float(ie[:, :6].mean())
        else:
            for j in range(7):
                mark = " <-- worst" if j == int(ae[:, :6].mean(0).argmax()) and j < 6 else ""
                print(f"    J{j}: {ae[:, j].mean():6.2f} deg{mark}")
            print(f"    MEAN(J0-5): {ae[:, :6].mean():.2f} deg | MEAN(all): {ae.mean():.2f} deg")
        metrics["joint_mae_deg_per"] = ae.mean(0).tolist()
        metrics["joint_mae_deg_mean6"] = float(ae[:, :6].mean())

    if adds:
        adds_arr = np.array(adds)
        reproj_per = np.array([r for _, r in per_joint_3d])
        pj_all = np.array([p for p, _ in per_joint_3d]) * 1000
        valid = reproj_per < 20.0  # PnP-valid frames (like the 4-tier harness)
        auc = compute_add_auc(adds_arr, threshold=0.1)  # AUC@100mm is outlier-robust
        print(f"\n  [3D / ADD]  ({len(adds_arr)} frames w/ GT)")
        print(f"    ADD-AUC@100mm (all): {auc:.4f}   [robust metric]")
        print(f"    Median ADD (all): {np.median(adds_arr)*1000:.2f} mm")
        if valid.any():
            va = adds_arr[valid]
            print(f"    PnP-valid frames: {valid.sum()}/{len(valid)} ({100*valid.mean():.0f}%)"
                  f" -> mean ADD {va.mean()*1000:.2f} mm, AUC {compute_add_auc(va):.4f}")
        metrics.update(add_auc_100mm=auc, add_median_mm=float(np.median(adds_arr)*1000),
                       pnp_valid_ratio=float(valid.mean()))

    if kp2d_err:
        k = np.array(kp2d_err)
        print(f"\n  [Detector 2D error vs GT]  mean={k.mean():.2f}px median={np.median(k):.2f}px"
              f"  (PCK@5px={100*(k<=5).mean():.1f}% @10px={100*(k<=10).mean():.1f}%)")
        metrics.update(kp2d_mean_px=float(k.mean()), kp2d_median_px=float(np.median(k)),
                       pck5=float((k <= 5).mean()), pck10=float((k <= 10).mean()))

    if reproj_all:
        r = np.array(reproj_all)
        print(f"\n  [Reprojection (solver fit)] mean={r.mean():.2f}px median={np.median(r):.2f}px")
        metrics["reproj_px_mean"] = float(r.mean())

    print("=" * 64)
    with open(os.path.join(args.output_dir, "metrics_kinematic.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved -> {args.output_dir}/metrics_kinematic.json")


if __name__ == "__main__":
    main()
