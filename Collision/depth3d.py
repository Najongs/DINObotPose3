"""
depth3d.py — lift the HUMAN into the robot's metric 3D frame for 3D collision.

The robot is metric-3D for free (our solver's kp_cam). The human needs depth.
Monocular depth is only relative (unknown scale+shift). KEY IDEA: we already
know the robot's true metric depth at its own pixels, so we fit an affine map
  z_metric ≈ a * d_rel + b
using the robot's projected joints as anchors (d_rel read from the depth map,
z_metric known from kp_cam). Applying it to the human's pixels puts the human in
the SAME metric camera frame as the robot — then back-projection with K gives 3D
human points ready for collision3d.

This is exactly "use the model": the pose estimate supplies the metric anchors
that resolve the monocular scale ambiguity.

- MonocularDepth: optional transformers Depth-Anything wrapper (download needed).
- align_depth_affine / backproject_mask: the metric-alignment core (no model).
- __main__: a synthetic self-test that validates alignment+backprojection to mm
  with NO model download.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


PANDA_LINKS = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]


# ----------------------------------------------------------------------------
# Metric alignment core (pure numpy — the novel part, fully testable offline)
# ----------------------------------------------------------------------------
def robot_depth_anchors(joints_3d, K, links=PANDA_LINKS, per_link=8):
    """
    Densify metric-depth anchors by sampling points ALONG the robot links
    (interpolating 3D joints), then projecting. Returns (P2d (M,2), z (M,)).
    Widens the depth span vs the 7 joints alone -> far better-conditioned fit.
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    P3 = []
    for i, j in links:
        ts = np.linspace(0, 1, per_link)
        P3.append(joints_3d[i][None] * (1 - ts)[:, None] + joints_3d[j][None] * ts[:, None])
    P3 = np.concatenate([joints_3d] + P3, axis=0)
    z = P3[:, 2]
    u = fx * P3[:, 0] / z + cx
    v = fy * P3[:, 1] / z + cy
    return np.stack([u, v], axis=1), z


def align_depth_affine(
    depth_rel: np.ndarray,
    joints_2d: np.ndarray,
    joints_z: np.ndarray,
    patch: int = 2,
) -> tuple[float, float, float]:
    """
    Fit z_metric = a*d_rel + b using the robot's projected joints as anchors.

    depth_rel : (H,W) monocular relative depth (any affine of true depth; sign ok).
    joints_2d : (K,2) projected robot joints (px).
    joints_z  : (K,)  known metric depth of those joints (m, from kp_cam[...,2]).
    patch     : half-window (px) to average depth_rel around each anchor.

    Returns (a, b, rmse_m).
    """
    H, W = depth_rel.shape
    ds, zs = [], []
    for (u, v), z in zip(joints_2d, joints_z):
        ui, vi = int(round(u)), int(round(v))
        if not (0 <= ui < W and 0 <= vi < H):
            continue
        y0, y1 = max(0, vi - patch), min(H, vi + patch + 1)
        x0, x1 = max(0, ui - patch), min(W, ui + patch + 1)
        d = float(np.median(depth_rel[y0:y1, x0:x1]))
        if np.isfinite(d):
            ds.append(d); zs.append(float(z))
    if len(ds) < 2:
        raise ValueError("need >=2 valid robot anchors to align depth")
    ds = np.asarray(ds); zs = np.asarray(zs)
    A = np.stack([ds, np.ones_like(ds)], axis=1)
    (a, b), *_ = np.linalg.lstsq(A, zs, rcond=None)
    rmse = float(np.sqrt(np.mean((A @ [a, b] - zs) ** 2)))
    return float(a), float(b), rmse


def backproject_mask(
    mask: np.ndarray,
    depth_metric: np.ndarray,
    K: np.ndarray,
    stride: int = 2,
    z_range: tuple[float, float] = (0.2, 6.0),
) -> np.ndarray:
    """
    Back-project mask pixels with a METRIC depth map into 3D camera-frame points.
    Returns (M,3) meters. `stride` subsamples for speed.
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    ys, xs = np.nonzero(mask)
    if stride > 1:
        ys, xs = ys[::stride], xs[::stride]
    z = depth_metric[ys, xs]
    ok = np.isfinite(z) & (z > z_range[0]) & (z < z_range[1])
    ys, xs, z = ys[ok], xs[ok], z[ok]
    X = (xs - cx) * z / fx
    Y = (ys - cy) * z / fy
    return np.stack([X, Y, z], axis=1)


def human_points_from_depth(
    human_mask: np.ndarray,
    depth_rel: np.ndarray,
    robot_joints_3d: np.ndarray,
    K: np.ndarray,
    links=PANDA_LINKS,
    stride: int = 2,
) -> tuple[np.ndarray, float]:
    """Full path: align relative depth to the robot (link-densified anchors),
    then back-project the human into the robot's metric camera frame."""
    j2d, jz = robot_depth_anchors(robot_joints_3d, K, links)
    a, b, rmse = align_depth_affine(depth_rel, j2d, jz)
    depth_metric = a * depth_rel + b
    pts = backproject_mask(human_mask, depth_metric, K, stride=stride)
    return pts, rmse


# ----------------------------------------------------------------------------
# Optional monocular depth model (needs a one-time weight download)
# ----------------------------------------------------------------------------
class MonocularDepth:
    """transformers Depth-Anything-V2 wrapper. Returns (H,W) relative depth."""

    def __init__(self, model="depth-anything/Depth-Anything-V2-Small-hf", device="cuda"):
        from transformers import pipeline
        self.pipe = pipeline("depth-estimation", model=model, device=0 if device == "cuda" else -1)

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        from PIL import Image
        out = self.pipe(Image.fromarray(rgb))
        return np.asarray(out["depth"], dtype=np.float32)


# ----------------------------------------------------------------------------
# Self-test (no download): make a fake relative-depth map from a known 3D scene,
# align it via robot anchors, back-project the human, check recovery to mm.
# ----------------------------------------------------------------------------
def _self_test():
    import cv2
    H, W = 480, 640
    FX = FY = 500.0; CX, CY = 320.0, 240.0
    K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], float)

    def proj(P):
        z = P[:, 2]
        return np.stack([FX * P[:, 0] / z + CX, FY * P[:, 1] / z + CY], axis=1)

    # ground-truth 3D scene
    joints = np.array([[0, .30, 1.30], [0, .15, 1.28], [.06, .02, 1.25],
                       [.13, -.10, 1.22], [.21, -.17, 1.20], [.29, -.21, 1.18],
                       [.35, -.23, 1.17]], float)
    j2d = proj(joints)
    human_center = np.array([0.25, 0.0, 1.15])

    # DENSE ground-truth metric depth map (mimics Depth-Anything's dense output):
    # background far, robot skeleton at its joint depths, human blob at 1.15 m.
    zmap = np.full((H, W), 3.0, float)
    for (i, jn) in PANDA_LINKS:
        p1, p2 = j2d[i].astype(int), j2d[jn].astype(int)
        zval = 0.5 * (joints[i, 2] + joints[jn, 2])
        cv2.line(zmap, tuple(p1), tuple(p2), float(zval), thickness=22)
    hmask = np.zeros((H, W), np.uint8)
    hc2d = proj(human_center[None])[0].astype(int)
    cv2.ellipse(hmask, tuple(hc2d), (90, 200), 0, 0, 360, 1, -1)
    hmask = hmask.astype(bool)
    zmap[hmask] = human_center[2]

    # monocular "relative" depth = unknown affine of true depth (+ mild noise),
    # inverse-depth-like to mimic a disparity output.
    scale, shift = -2.7, 4.1
    depth_rel = scale * zmap + shift + np.random.default_rng(0).normal(0, 0.02, zmap.shape)

    # ALIGN using link-densified robot anchors, then back-project the human.
    pts, rmse = human_points_from_depth(hmask, depth_rel, joints, K, stride=3)
    err = np.linalg.norm(pts.mean(0) - human_center) * 1000
    print(f"anchor-fit rmse         = {rmse*1000:.2f} mm")
    print(f"recovered human centroid= {np.round(pts.mean(0),3)}  (gt {human_center})")
    print(f"recovered human z-mean  = {pts[:,2].mean():.3f} m  (gt {human_center[2]})")
    print(f"centroid error          = {err:.1f} mm  over {len(pts)} pts")
    assert err < 80, "metric alignment should recover the human to <8cm"
    print(f"SELF-TEST PASSED: robot-anchored monocular depth alignment recovers metric human 3D "
          f"to ~{err:.0f} mm.\nNote: accuracy degrades as the human's depth leaves the robot's "
          f"anchor depth-span (extrapolation) — a metric depth model or a wider anchor set tightens it.")


if __name__ == "__main__":
    _self_test()
