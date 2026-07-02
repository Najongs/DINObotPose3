"""
run_video3d.py — METRIC 3D collision on a real frame sequence.

Ties together:
  - robot metric 3D joints (our solver's kp_cam) from a dumped npz,
  - human mask (Mask R-CNN),
  - monocular depth (Depth-Anything) aligned to the robot's metric depth,
  - collision3d.CollisionEstimator3D.

Robot npz must contain:
  kp_cam : (N,7,3) metric camera-frame joints (solver output, meters)
  K      : (3,3) or (N,3,3) intrinsics at the image resolution used here
Optional precomputed depth via --depth-dir (per-frame .npy relative depth),
otherwise Depth-Anything is downloaded/run.

Example:
  conda run -n py312 python Collision/run_video3d.py \
    --frames-dir /seq --robot-npz solved3d.npz --out Collision/run3d_out \
    --d-safe 0.15 --radius 0.07 --fps 15
"""
from __future__ import annotations

import argparse
import csv
import glob
import os

import cv2
import numpy as np

from collision3d import CollisionEstimator3D, ProbModel3D, PANDA_LINKS
from depth3d import human_points_from_depth, robot_depth_anchors


def load_frames(d):
    fs = sorted(glob.glob(os.path.join(d, "*.jpg")) + glob.glob(os.path.join(d, "*.png")))
    if not fs:
        raise SystemExit(f"no frames in {d}")
    return fs


def project(P3, K):
    z = np.clip(P3[:, 2], 1e-3, None)
    u = K[0, 0] * P3[:, 0] / z + K[0, 2]
    v = K[1, 1] * P3[:, 1] / z + K[1, 2]
    return np.stack([u, v], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--robot-npz", required=True, help="npz with kp_cam (N,7,3), K")
    ap.add_argument("--depth-dir", default=None, help="optional per-frame .npy relative depth")
    ap.add_argument("--out", default="run3d_out")
    ap.add_argument("--d-safe", type=float, default=0.15)
    ap.add_argument("--softness", type=float, default=0.08)
    ap.add_argument("--radius", type=float, default=0.07)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    frames = load_frames(args.frames_dir)
    if args.max_frames:
        frames = frames[: args.max_frames]

    z = np.load(args.robot_npz)
    kp_cam = z["kp_cam"]                       # (N,7,3)
    K_all = z["K"]
    if K_all.ndim == 2:
        K_all = np.broadcast_to(K_all, (len(kp_cam), 3, 3))

    from segmenters import HumanSegmenter
    hseg = HumanSegmenter(device=args.device)

    depth_files = None
    depth_model = None
    if args.depth_dir:
        depth_files = sorted(glob.glob(os.path.join(args.depth_dir, "*.npy")))
    else:
        from depth3d import MonocularDepth
        depth_model = MonocularDepth(device=args.device)

    est = CollisionEstimator3D(ProbModel3D(args.d_safe, args.softness),
                               horizon=args.horizon, radius=args.radius)

    img0 = cv2.imread(frames[0]); H, W = img0.shape[:2]
    vw = cv2.VideoWriter(os.path.join(args.out, "collision3d.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
    cf = open(os.path.join(args.out, "reports3d.csv"), "w", newline="")
    cw = csv.writer(cf)
    cw.writerow(["frame", "valid", "surface_dist_m", "depth_gap_m", "prob_now",
                 "prob_pred", "risk", "closing_mps_frame", "ttc_frames",
                 "human_toward_robot", "robot_toward_human", "align_rmse_mm"])

    for f, fp in enumerate(frames):
        bgr = cv2.imread(fp)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        K = np.asarray(K_all[f], float)
        J3 = np.asarray(kp_cam[f], float)

        hum = hseg.segment(rgb)
        depth_rel = np.load(depth_files[f]) if depth_files else depth_model(rgb)

        rmse = float("nan")
        if hum.any():
            HP, rmse = human_points_from_depth(hum, depth_rel, J3, K, PANDA_LINKS, stride=3)
        else:
            HP = np.zeros((0, 3))
        rep = est.step(f, J3, HP)

        # viz: robot skeleton + human mask + HUD
        vis = bgr.copy()
        vis[hum] = (0.45 * np.array([40, 40, 210]) + 0.55 * vis[hum]).astype(np.uint8)
        j2d = project(J3, K)
        for i, j in PANDA_LINKS:
            cv2.line(vis, tuple(j2d[i].astype(int)), tuple(j2d[j].astype(int)), (255, 200, 60), 3)
        rr = rep.risk
        rc = (0, int(255 * (1 - rr)), int(255 * rr))
        ttc = "inf" if not np.isfinite(rep.ttc) else f"{rep.ttc:.0f}f"
        for k, ln in enumerate([
            f"frame {f}  3D dist={rep.surface_dist*100:.1f}cm depth_gap={rep.depth_gap*100:.0f}cm",
            f"RISK={rr:.2f} (now {rep.prob_now:.2f} pred {rep.prob_pred:.2f})",
            f"closing={rep.closing_speed*100:+.1f}cm/f TTC={ttc} align_rmse={rmse:.0f}mm",
        ]):
            cv2.putText(vis, ln, (10, 24 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        rc if k == 1 else (240, 240, 240), 2, cv2.LINE_AA)
        vw.write(vis)
        cw.writerow([f, rep.valid, f"{rep.surface_dist:.4f}", f"{rep.depth_gap:.4f}",
                     f"{rep.prob_now:.4f}", f"{rep.prob_pred:.4f}", f"{rep.risk:.4f}",
                     f"{rep.closing_speed:.4f}",
                     "inf" if not np.isfinite(rep.ttc) else f"{rep.ttc:.1f}",
                     f"{rep.human_toward_robot:.4f}", f"{rep.robot_toward_human:.4f}",
                     f"{rmse:.2f}"])
        if f % 20 == 0:
            print(f"[{f}/{len(frames)}] dist={rep.surface_dist*100:6.1f}cm risk={rr:.2f}")

    vw.release(); cf.close()
    print(f"Wrote collision3d.mp4 + reports3d.csv to {args.out}/")


if __name__ == "__main__":
    main()
