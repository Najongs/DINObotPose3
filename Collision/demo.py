"""
demo.py — self-contained synthetic validation of the collision pipeline.

Builds a sequence where a Panda-like robot arm sits on the left and a human
blob walks in from the right toward the robot, then retreats. For each frame we
run CollisionEstimator and render an annotated frame:
  - robot region (blue), human region (red), overlap (magenta)
  - centroids + velocity arrows (motion vectors)
  - the boundary distance line
  - text HUD: prob_now, prob_pred, closing speed, TTC, approach flags

Outputs annotated PNGs and an mp4 into --out, and prints the per-frame report.
No external weights/data needed — validates the geometry + temporal logic.

Run:
  conda run -n py312 python Collision/demo.py --out Collision/demo_out
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np

from collision import (
    CollisionEstimator, ProbModel, keypoints_to_region, PANDA_LINKS,
)

H, W = 480, 640


def robot_keypoints(t: float) -> np.ndarray:
    """A fixed Panda-like arm on the left with a gently swaying wrist."""
    base = np.array([140, 380])
    sway = 30.0 * np.sin(t * 0.6)
    kps = np.array([
        base,                       # link0 base
        base + [0, -70],            # link2
        base + [25, -140],          # link3
        base + [55, -200],          # link4
        base + [95, -235] + [sway, 0],   # link6
        base + [140, -250] + [1.4 * sway, 0],  # link7
        base + [175, -255] + [1.7 * sway, 0],  # hand
    ], dtype=np.float64)
    return kps


def human_center(frame: int, n: int) -> np.ndarray:
    """Human enters from the right, approaches the robot, then retreats."""
    x0, x1 = 560.0, 300.0            # start far right, closest approach x=300
    half = n // 2
    if frame <= half:
        a = frame / half
    else:
        a = 1.0 - (frame - half) / (n - half)
    x = x0 + (x1 - x0) * a
    y = 250.0 + 20.0 * np.sin(frame * 0.3)
    return np.array([x, y])


def human_region(center: np.ndarray) -> np.ndarray:
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.ellipse(mask, tuple(center.astype(int)), (38, 90), 0, 0, 360, 1, -1)
    # a "head"
    cv2.circle(mask, (int(center[0]), int(center[1] - 105)), 26, 1, -1)
    return mask.astype(bool)


def draw(frame_idx, rob_mask, hum_mask, rep, kps):
    img = np.full((H, W, 3), 30, np.uint8)
    img[rob_mask] = (200, 120, 40)          # robot blue-ish (BGR)
    img[hum_mask] = (40, 40, 210)           # human red
    img[rob_mask & hum_mask] = (210, 40, 210)  # overlap magenta
    for i, j in PANDA_LINKS:
        cv2.line(img, tuple(kps[i].astype(int)), tuple(kps[j].astype(int)), (255, 255, 255), 1)

    def arrow(c, v, color):
        if c is None or v is None:
            return
        p1 = tuple(np.round(c).astype(int))
        p2 = tuple(np.round(c + v * 8).astype(int))
        cv2.arrowedLine(img, p1, p2, color, 2, tipLength=0.3)

    if rep.valid:
        cr = tuple(np.round(rep.robot_centroid).astype(int))
        ch = tuple(np.round(rep.human_centroid).astype(int))
        cv2.circle(img, cr, 4, (255, 255, 0), -1)
        cv2.circle(img, ch, 4, (0, 255, 255), -1)
        cv2.line(img, cr, ch, (120, 120, 120), 1)
        arrow(rep.robot_centroid, rep.robot_vel, (255, 255, 0))
        arrow(rep.human_centroid, rep.human_vel, (0, 255, 255))

    # risk-colored HUD
    risk = rep.risk
    rc = (0, int(255 * (1 - risk)), int(255 * risk))
    lines = [
        f"frame {frame_idx}",
        f"dist(bnd)={rep.boundary_dist:6.1f}px  centroid={rep.centroid_dist:6.1f}px",
        f"p_now={rep.prob_now:.2f}  p_pred={rep.prob_pred:.2f}  RISK={risk:.2f}",
        f"closing={rep.closing_speed:+.1f}px/f  TTC={rep.ttc:.0f}f" if np.isfinite(rep.ttc)
        else f"closing={rep.closing_speed:+.1f}px/f  TTC=inf",
        f"human->robot={rep.human_toward_robot:+.1f}  robot->human={rep.robot_toward_human:+.1f}",
    ]
    for k, ln in enumerate(lines):
        col = rc if k == 2 else (230, 230, 230)
        cv2.putText(img, ln, (10, 24 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)
    # risk bar
    cv2.rectangle(img, (W - 40, 20), (W - 20, 220), (80, 80, 80), 1)
    top = int(220 - 200 * risk)
    cv2.rectangle(img, (W - 39, top), (W - 21, 219), rc, -1)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="demo_out")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--d-safe", type=float, default=45.0)
    ap.add_argument("--softness", type=float, default=22.0)
    ap.add_argument("--horizon", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    est = CollisionEstimator(ProbModel(args.d_safe, args.softness), horizon=args.horizon)

    vw = cv2.VideoWriter(os.path.join(args.out, "collision.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), 12, (W, H))
    print(f"{'f':>3} {'dist':>7} {'p_now':>6} {'p_pred':>7} {'risk':>6} "
          f"{'close':>7} {'ttc':>6} {'h->r':>6} {'r->h':>6}")
    for f in range(args.frames):
        kps = robot_keypoints(f)
        rob = keypoints_to_region(kps, (H, W), PANDA_LINKS, thickness=20)
        hum = human_region(human_center(f, args.frames))
        rep = est.step(f, rob, hum)
        img = draw(f, rob, hum, rep, kps)
        cv2.imwrite(os.path.join(args.out, f"frame_{f:03d}.png"), img)
        vw.write(img)
        ttc = f"{rep.ttc:6.0f}" if np.isfinite(rep.ttc) else "   inf"
        print(f"{f:>3} {rep.boundary_dist:7.1f} {rep.prob_now:6.2f} {rep.prob_pred:7.2f} "
              f"{rep.risk:6.2f} {rep.closing_speed:+7.1f} {ttc} "
              f"{rep.human_toward_robot:+6.1f} {rep.robot_toward_human:+6.1f}")
    vw.release()
    print(f"\nWrote annotated frames + collision.mp4 to {args.out}/")


if __name__ == "__main__":
    main()
