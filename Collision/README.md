# Human–Robot Collision-Probability Estimation (2D region-based)

Estimates how likely a human and the robot are to collide, from **segmentation
regions** in the image, plus a **motion-vector** predictive term over a time
sequence. Built on top of our DINOv3 pose model (the robot region can come from
the model's projected FK keypoints).

## Idea / pipeline

1. **Segmentation → regions.**
   - **Robot region**: project our pose solver's 7 FK keypoints to 2D and
     rasterize the links as thick capsules (`keypoints_to_region`) — this *uses
     the model*. Alternatively use the CtRNet DeepLabV3 robot mask.
   - **Human region**: pretrained torchvision **Mask R-CNN** person mask
     (`segmenters.HumanSegmenter`).
2. **2D distance between regions.** Boundary min-distance via a Euclidean
   distance transform (0 if overlapping) + centroid distance + overlap area
   (`region_relation`). Purely 2D / image-plane, as specified.
3. **Distance → collision probability.** Monotone logistic
   `p = sigmoid((d_safe − d) / softness)` (`ProbModel`): `p=0.5` at the safety
   distance `d_safe`, →1 on contact, →0 when far.
4. **Temporal motion-vector prediction.** Track each region's centroid over
   time, estimate its velocity by a least-squares fit over a short window
   (`MotionTrack`). From the *relative* motion:
   - `human_toward_robot`, `robot_toward_human` = each object's velocity
     component along the line connecting them (is the human walking into the
     robot? is the robot swinging toward the human?).
   - `closing_speed = (v_robot − v_human) · ê` (>0 = approaching).
   - `TTC` = boundary distance / closing speed (frames to contact).
   - **Predictive probability** `p_pred`: project the distance `horizon` frames
     ahead (`d_future = d − closing·horizon`) and map through the same logistic,
     so a fast approach raises risk *before* they are actually close, and two
     touching-but-separating objects get a *low* future risk.
   - `risk = max(p_now, p_pred)`.

All 2D and pure numpy/scipy/cv2 — no metric depth required (optionally convert
px→mm with pose `t_z`/`K` if you want metric `d_safe`).

## Files

- `collision.py` — core geometry, probability, motion tracking, `CollisionEstimator` (data-agnostic; operates on boolean masks + optional keypoints).
- `segmenters.py` — `HumanSegmenter` (Mask R-CNN person), CtRNet-mask helper.
- `demo.py` — self-contained **synthetic** validation: robot arm + a human
  walking in and out. No weights/data needed.
- `run_video.py` — run on a **real** frame folder: human via Mask R-CNN, robot
  via our solver keypoints (`--robot-kps`) or precomputed masks
  (`--robot-mask-dir`). Writes annotated mp4 + `reports.csv`.

## Run

```bash
# synthetic demo (validates geometry + temporal logic; writes annotated mp4)
conda run -n py312 python Collision/demo.py --out Collision/demo_out --frames 60

# real sequence, robot region from our pose solver's projected keypoints
conda run -n py312 python Collision/run_video.py \
  --frames-dir /path/to/seq --robot-kps solved_kps.npz \
  --out Collision/run_out --d-safe 45 --softness 22 --fps 15
```

`--robot-kps` expects an `.npz` with `kps` (N,K,2) projected FK keypoints (and
optional `conf` (N,K)); this is exactly what our solver produces per frame, so
the robot region is grounded in the pose estimate.

## Demo behaviour (validated)

Synthetic human approaches then retreats:
- `p_now` crosses 0.5 exactly at `d = d_safe` (45px). ✓
- `p_pred` fires early on fast approach (d≈170px, closing +19 → p_pred 0.79)
  and, crucially, **drops to ~0 while the two are still overlapping but moving
  apart** (frames where `closing < 0`) — the motion vector, not just distance,
  drives the prediction. ✓
- `TTC` is finite only while approaching; `human→robot` / `robot→human`
  components flip sign correctly as the human passes and reverses. ✓

## Tuning

- `d_safe`, `softness`: the danger threshold and transition sharpness (px, or mm
  if you feed metric distances).
- `horizon`: look-ahead frames for the predictive term (larger = earlier, more
  cautious warnings).
- `thickness`: robot link radius when building the region from keypoints.

## Notes / limits (2D)

- Purely 2D image-plane proximity: two objects far apart in depth but overlapping
  in the image read as "close". The 3D path below fixes exactly this.
- Velocity is in px/frame; multiply the HUD closing speed by `fps` for px/s.

---

# METRIC 3D collision (the depth-aware version)

The 2D version above is an image-plane concept. The 3D version computes a real
**metric distance in meters** and is depth-aware. It exploits our model's unique
asset: the robot's **metric 3D joints** (`kp_cam`) come straight from the pose
solver, so the robot side is 3D for free. Only the human needs depth.

## 3D pipeline

1. **Robot = 3D capsules.** Links between consecutive 3D joints, each with a
   physical radius → the arm's swept volume (`collision3d.capsule_surface_distance`).
2. **Human = 3D point cloud.** Back-project the human mask with a monocular depth
   map. Monocular depth is only *relative* (unknown scale+shift), so we **align
   it to metric using the robot as an anchor**: read the depth map at the robot's
   projected joints/links (densified along links), fit `z_metric = a·d_rel + b`
   against the known `kp_cam` depths, apply to the human's pixels
   (`depth3d.human_points_from_depth`). This resolves the monocular scale
   ambiguity *with the pose estimate* and puts human + robot in one metric frame.
3. **Collision distance** = min over human points of distance to the nearest
   robot capsule surface, in **meters** (negative = interpenetration).
4. **Probability** = logistic on the metric distance vs a metric `d_safe` (e.g.
   0.15 m).
5. **Temporal** = track 3D centroids → 3D velocity → 3D closing speed / TTC /
   direction. Now the **depth-axis** approach (human walking toward the camera
   into the robot) is visible, which the 2D version cannot see.

## 3D files

- `collision3d.py` — 3D geometry, metric probability, 3D motion tracking, `CollisionEstimator3D`.
- `depth3d.py` — robot-anchored monocular-depth→metric alignment + back-projection. `python depth3d.py` runs a **no-download self-test** (recovers a synthetic human's 3D to ~5 cm).
- `demo3d.py` — synthetic scene computing **2D and 3D risk side by side**. Phase 1: human passes *in front* of the robot (2D masks overlap → 2D false-alarms at 0.89) while 3D stays low (`depth_gap` ~58 cm). Phase 2: human reaches the robot's depth → both fire.
- `run_video3d.py` — real sequence: robot `kp_cam` npz + Mask R-CNN human + Depth-Anything (or `--depth-dir` precomputed) → metric 3D collision video + `reports3d.csv`.

## Run (3D)

```bash
# validate the metric-alignment math (no weights/data needed)
conda run -n py312 python Collision/depth3d.py

# synthetic 2D-vs-3D comparison video (shows 3D killing the depth false-alarm)
conda run -n py312 python Collision/demo3d.py --out Collision/demo3d_out

# real sequence (needs a solver dump with kp_cam + K, and a depth model)
conda run -n py312 python Collision/run_video3d.py \
  --frames-dir /seq --robot-npz solved3d.npz --out Collision/run3d_out \
  --d-safe 0.15 --radius 0.07 --fps 15
```

`solved3d.npz` = `{kp_cam: (N,7,3) meters, K: (3,3) or (N,3,3)}` — dump these
from the solver (`solve_batch(..., return_pose=True)` already returns `kp_cam`).

## 3D limits (honest)

- The robot's depth span is narrow (arm ≈ 10–20 cm deep), so aligning a monocular
  depth map from robot anchors is an **extrapolation** to the human's depth —
  accuracy ≈ 5 cm when the human is near the robot's depth and degrades farther
  away. A **metric** depth model (Depth-Anything-V2-metric) or a wider anchor set
  (floor/scene points at known depth) tightens this.
- Depth-Anything weights are a one-time download; if offline, pass precomputed
  relative-depth `.npy` via `--depth-dir`.
- Robot capsule radius (`--radius`) is a hand-set physical arm thickness; set it
  to the real link radius for calibrated metric margins.
