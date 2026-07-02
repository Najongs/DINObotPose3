"""
demo3d.py — self-contained validation of METRIC 3D collision, and a direct
comparison against the 2D image-plane version to show why 3D matters.

Scene (camera frame, meters; x right, y down, z forward):
  - Robot: a fixed Panda-like arm at depth z~1.2 m.
  - Human: a 3D ellipsoid that
      Phase 1 (frames 0-27): passes IN FRONT of the robot in the image
        (z~0.65 m, ~0.55 m closer than the arm) — so the 2D masks OVERLAP but
        the two are far apart in depth. Ground truth: SAFE.
      Phase 2 (frames 28-59): comes to the robot's depth (z~1.2 m) and walks
        into it in x. Ground truth: real approach -> COLLISION.

For every frame we compute BOTH:
  - 2D risk (collision.CollisionEstimator on the projected masks)
  - 3D risk (collision3d.CollisionEstimator3D on the metric geometry)
and render them side by side. Phase 1 is exactly where 2D false-alarms and 3D
stays low; Phase 2 is where both fire.

Run:
  conda run -n py312 python Collision/demo3d.py --out Collision/demo3d_out
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np

from collision import CollisionEstimator, ProbModel, keypoints_to_region, PANDA_LINKS
from collision3d import CollisionEstimator3D, ProbModel3D

H, W = 480, 640
FX = FY = 500.0
CX, CY = 320.0, 240.0
K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float64)
RADIUS = 0.07  # robot link radius (m)


def project(P3: np.ndarray) -> np.ndarray:
    """(N,3) camera-frame meters -> (N,2) pixels."""
    z = np.clip(P3[:, 2], 1e-3, None)
    u = FX * P3[:, 0] / z + CX
    v = FY * P3[:, 1] / z + CY
    return np.stack([u, v], axis=1)


def robot_joints_3d(t: float) -> np.ndarray:
    sway = 0.02 * np.sin(t * 0.5)
    J = np.array([
        [0.00, 0.30, 1.30],
        [0.00, 0.15, 1.28],
        [0.06, 0.02, 1.25],
        [0.13, -0.10, 1.22],
        [0.21, -0.17, 1.20],
        [0.29, -0.21, 1.18],
        [0.35, -0.23, 1.17],
    ], dtype=np.float64)
    J[:, 0] += np.linspace(0, sway, len(J))
    return J


def human_center_3d(frame: int, n: int) -> np.ndarray:
    half = 28
    if frame < half:
        # Phase 1: in FRONT (z small), sweeping across x -> 2D overlap, 3D safe
        a = frame / (half - 1)
        x = 0.55 - 0.75 * a          # +0.55 -> -0.20
        z = 0.65
        y = 0.02
    else:
        # Phase 2: at robot depth, walking into the arm in x
        b = (frame - half) / (n - half - 1)
        x = 0.75 - 0.55 * b          # 0.75 -> 0.20 (toward arm x~0.2-0.35)
        z = 1.22
        y = 0.00
    return np.array([x, y, z])


def human_ellipsoid_points(center: np.ndarray, n=600) -> np.ndarray:
    """Sample surface points of a person-sized ellipsoid (m)."""
    rng_u = np.linspace(0, np.pi, 24)
    rng_v = np.linspace(0, 2 * np.pi, 25)
    uu, vv = np.meshgrid(rng_u, rng_v)
    rx, ry, rz = 0.20, 0.45, 0.18
    x = rx * np.sin(uu) * np.cos(vv)
    y = ry * np.cos(uu)
    z = rz * np.sin(uu) * np.sin(vv)
    P = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1) + center[None]
    return P


def human_mask_from_points(P3: np.ndarray) -> np.ndarray:
    uv = project(P3)
    uv = uv[(uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)]
    mask = np.zeros((H, W), np.uint8)
    if len(uv) >= 3:
        hull = cv2.convexHull(uv.astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 1)
    return mask.astype(bool)


def draw(f, rob_mask, hum_mask, kps2d, r2d, r3d):
    img = np.full((H, W, 3), 30, np.uint8)
    img[rob_mask] = (200, 120, 40)
    img[hum_mask] = (40, 40, 210)
    img[rob_mask & hum_mask] = (210, 40, 210)
    for i, j in PANDA_LINKS:
        cv2.line(img, tuple(kps2d[i].astype(int)), tuple(kps2d[j].astype(int)), (255, 255, 255), 1)

    def bar(x0, label, risk, col):
        cv2.rectangle(img, (x0, 250), (x0 + 22, 450), (80, 80, 80), 1)
        top = int(450 - 200 * np.clip(risk, 0, 1))
        cv2.rectangle(img, (x0 + 1, top), (x0 + 21, 449), col, -1)
        cv2.putText(img, label, (x0 - 4, 468), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

    r2 = r2d.risk
    r3 = r3d.risk
    bar(W - 120, "2D", r2, (0, int(255 * (1 - r2)), int(255 * r2)))
    bar(W - 60, "3D", r3, (0, int(255 * (1 - r3)), int(255 * r3)))

    phase = "PHASE1 (in front: 2D overlap, 3D safe)" if f < 28 else "PHASE2 (real approach)"
    lines = [
        f"frame {f}   {phase}",
        f"2D: dist={r2d.boundary_dist:5.0f}px  risk={r2:.2f}",
        f"3D: dist={r3d.surface_dist*100:6.1f}cm depth_gap={r3d.depth_gap*100:5.1f}cm risk={r3:.2f}",
        f"3D closing={r3d.closing_speed*100:+.1f}cm/f  h->r={r3d.human_toward_robot*100:+.1f} r->h={r3d.robot_toward_human*100:+.1f}",
    ]
    for k, ln in enumerate(lines):
        cv2.putText(img, ln, (10, 22 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (235, 235, 235), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="demo3d_out")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--d-safe", type=float, default=0.15, help="metric safety dist (m)")
    ap.add_argument("--softness", type=float, default=0.08)
    ap.add_argument("--horizon", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    est2d = CollisionEstimator(ProbModel(d_safe=45, softness=22), horizon=args.horizon)
    est3d = CollisionEstimator3D(ProbModel3D(args.d_safe, args.softness),
                                 horizon=args.horizon, radius=RADIUS)

    vw = cv2.VideoWriter(os.path.join(args.out, "collision3d.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), 12, (W, H))
    print(f"{'f':>3} {'2Ddist':>7} {'2Drisk':>6} | {'3Ddist_cm':>9} {'dgap_cm':>8} "
          f"{'3Drisk':>6} {'close_cm':>8} {'ttc':>5}")
    for f in range(args.frames):
        J3 = robot_joints_3d(f)
        kps2d = project(J3)
        rob_mask = keypoints_to_region(kps2d, (H, W), PANDA_LINKS, thickness=22)

        c = human_center_3d(f, args.frames)
        HP = human_ellipsoid_points(c)
        hum_mask = human_mask_from_points(HP)

        r2d = est2d.step(f, rob_mask, hum_mask)
        r3d = est3d.step(f, J3, HP)

        img = draw(f, rob_mask, hum_mask, kps2d, r2d, r3d)
        cv2.imwrite(os.path.join(args.out, f"frame_{f:03d}.png"), img)
        vw.write(img)
        ttc = f"{r3d.ttc:5.0f}" if np.isfinite(r3d.ttc) else "  inf"
        print(f"{f:>3} {r2d.boundary_dist:7.1f} {r2d.risk:6.2f} | "
              f"{r3d.surface_dist*100:9.1f} {r3d.depth_gap*100:8.1f} {r3d.risk:6.2f} "
              f"{r3d.closing_speed*100:+8.1f} {ttc}")
    vw.release()
    print(f"\nWrote frames + collision3d.mp4 to {args.out}/")
    print("Expect: PHASE1 -> 2D risk HIGH (false alarm), 3D risk LOW (depth_gap ~55cm).")
    print("        PHASE2 -> both rise as human reaches robot depth and closes in.")


if __name__ == "__main__":
    main()
