"""
run_video.py — run the collision pipeline on a real frame sequence.

Robot region source (pick one):
  --robot-kps  path.npz   : (N,K,2) projected FK keypoints from OUR pose solver
                            (the "use the model" path). Optional (N,K) 'conf'.
  --robot-mask-dir DIR    : precomputed robot masks (e.g. CtRNet DeepLabV3),
                            PNG per frame, >0 = robot.
Human region:
  torchvision Mask R-CNN person masks (auto).

Frames are read in sorted order from --frames-dir (*.jpg/*.png). Writes an
annotated mp4 + a per-frame CSV of the collision reports.

Example (robot from our solver dump):
  conda run -n py312 python Collision/run_video.py \
    --frames-dir /path/to/seq --robot-kps solved_kps.npz \
    --out Collision/run_out --d-safe 45 --softness 22 --fps 15
"""
from __future__ import annotations

import argparse
import csv
import glob
import os

import cv2
import numpy as np

from collision import CollisionEstimator, ProbModel, keypoints_to_region, PANDA_LINKS


def load_frames(d):
    fs = sorted(glob.glob(os.path.join(d, "*.jpg")) + glob.glob(os.path.join(d, "*.png")))
    if not fs:
        raise SystemExit(f"no frames in {d}")
    return fs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--robot-kps", default=None, help="npz with 'kps' (N,K,2), optional 'conf'")
    ap.add_argument("--robot-mask-dir", default=None)
    ap.add_argument("--out", default="run_out")
    ap.add_argument("--d-safe", type=float, default=45.0)
    ap.add_argument("--softness", type=float, default=22.0)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--thickness", type=int, default=20)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    if not (args.robot_kps or args.robot_mask_dir):
        raise SystemExit("provide --robot-kps or --robot-mask-dir")
    os.makedirs(args.out, exist_ok=True)

    frames = load_frames(args.frames_dir)
    if args.max_frames:
        frames = frames[: args.max_frames]

    kps_all = conf_all = None
    if args.robot_kps:
        z = np.load(args.robot_kps)
        kps_all = z["kps"]
        conf_all = z["conf"] if "conf" in z else None

    rob_mask_files = None
    if args.robot_mask_dir:
        rob_mask_files = sorted(glob.glob(os.path.join(args.robot_mask_dir, "*.png")))

    from segmenters import HumanSegmenter
    hseg = HumanSegmenter(device=args.device)

    est = CollisionEstimator(ProbModel(args.d_safe, args.softness), horizon=args.horizon)

    img0 = cv2.imread(frames[0]); H, W = img0.shape[:2]
    vw = cv2.VideoWriter(os.path.join(args.out, "collision.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
    csvf = open(os.path.join(args.out, "reports.csv"), "w", newline="")
    cw = csv.writer(csvf)
    cw.writerow(["frame", "valid", "boundary_dist", "centroid_dist", "prob_now",
                 "prob_pred", "risk", "closing_speed", "ttc",
                 "human_toward_robot", "robot_toward_human"])

    for f, fp in enumerate(frames):
        bgr = cv2.imread(fp)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        hum = hseg.segment(rgb)

        if kps_all is not None:
            kps = kps_all[f]
            conf = conf_all[f] if conf_all is not None else None
            rob = keypoints_to_region(kps, (H, W), PANDA_LINKS, args.thickness, conf)
        else:
            rob = cv2.imread(rob_mask_files[f], cv2.IMREAD_GRAYSCALE) > 0
            kps = np.zeros((7, 2))  # no skeleton to draw

        rep = est.step(f, rob, hum)
        # blend annotation over the real frame
        overlay = bgr.copy()
        overlay[rob] = (200, 120, 40)
        overlay[hum] = (40, 40, 210)
        overlay[rob & hum] = (210, 40, 210)
        vis = cv2.addWeighted(overlay, 0.45, bgr, 0.55, 0)
        vis = draw_hud(vis, f, rep)
        vw.write(vis)
        cw.writerow([f, rep.valid, f"{rep.boundary_dist:.2f}", f"{rep.centroid_dist:.2f}",
                     f"{rep.prob_now:.4f}", f"{rep.prob_pred:.4f}", f"{rep.risk:.4f}",
                     f"{rep.closing_speed:.3f}",
                     "inf" if not np.isfinite(rep.ttc) else f"{rep.ttc:.1f}",
                     f"{rep.human_toward_robot:.3f}", f"{rep.robot_toward_human:.3f}"])
        if f % 20 == 0:
            print(f"[{f}/{len(frames)}] d={rep.boundary_dist:6.1f} risk={rep.risk:.2f}")

    vw.release(); csvf.close()
    print(f"Wrote collision.mp4 + reports.csv to {args.out}/")


def draw_hud(img, f, rep):
    risk = rep.risk
    rc = (0, int(255 * (1 - risk)), int(255 * risk))
    if rep.valid:
        cr = tuple(np.round(rep.robot_centroid).astype(int))
        ch = tuple(np.round(rep.human_centroid).astype(int))
        cv2.circle(img, cr, 4, (255, 255, 0), -1)
        cv2.circle(img, ch, 4, (0, 255, 255), -1)
        cv2.line(img, cr, ch, (150, 150, 150), 1)
        if rep.robot_vel is not None:
            cv2.arrowedLine(img, cr, tuple(np.round(rep.robot_centroid + rep.robot_vel * 8).astype(int)), (255, 255, 0), 2, tipLength=0.3)
        if rep.human_vel is not None:
            cv2.arrowedLine(img, ch, tuple(np.round(rep.human_centroid + rep.human_vel * 8).astype(int)), (0, 255, 255), 2, tipLength=0.3)
    ttc = "inf" if not np.isfinite(rep.ttc) else f"{rep.ttc:.0f}f"
    lines = [
        f"frame {f}  dist={rep.boundary_dist:.0f}px",
        f"p_now={rep.prob_now:.2f} p_pred={rep.prob_pred:.2f} RISK={risk:.2f}",
        f"closing={rep.closing_speed:+.1f}px/f TTC={ttc}",
    ]
    for k, ln in enumerate(lines):
        col = rc if k == 1 else (240, 240, 240)
        cv2.putText(img, ln, (10, 24 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
    return img


if __name__ == "__main__":
    main()
