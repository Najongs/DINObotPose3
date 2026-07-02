"""
collision.py — 2D region-based human/robot collision-probability estimation.

Pipeline (per the task spec):
  1) Segmentation gives a *robot region* and a *human region* (2D masks).
     - Robot region can be built from OUR pose model's FK keypoints (projected
       to 2D, links rasterized as thick capsules) — this is the "use the model"
       path — or from a CtRNet segmenter mask.
     - Human region comes from a person segmenter (see segmenters.py).
  2) From the two current regions we compute how far apart they are in the image
     (boundary min-distance via a distance transform, plus centroid distance).
  3) That distance is mapped to an (approximate) collision probability with a
     monotone logistic: p = sigmoid((d_safe - d) / softness).
  4) Over a time sequence we track each region's centroid, estimate its motion
     vector (velocity), and from the *relative* motion decide whether the human
     is moving toward the robot / the robot toward the human. The closing speed
     gives a time-to-collision (TTC) and a predictive collision probability that
     projects the distance a short horizon into the future.

Everything here is pure numpy/scipy/cv2 and data-agnostic: it operates on
boolean masks (+ optional keypoints), so it works with synthetic demos, DREAM
frames with an injected human, or a real human-robot video.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

try:
    from scipy.ndimage import distance_transform_edt
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False

try:
    import cv2
    _HAS_CV2 = True
except Exception:  # pragma: no cover
    _HAS_CV2 = False


# ----------------------------------------------------------------------------
# Panda kinematic chain (the 7 keypoints our model/solver emit, SEL order).
# Links connect consecutive keypoints along the arm; used to rasterize the robot
# region from projected 2D keypoints.
# ----------------------------------------------------------------------------
PANDA_LINKS = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


# ----------------------------------------------------------------------------
# Region geometry
# ----------------------------------------------------------------------------
def mask_centroid(mask: np.ndarray) -> Optional[np.ndarray]:
    """Return (x, y) centroid of a boolean mask, or None if empty."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return np.array([xs.mean(), ys.mean()], dtype=np.float64)


def _edt(binary: np.ndarray) -> np.ndarray:
    """Euclidean distance transform: distance of each pixel to nearest True."""
    if _HAS_SCIPY:
        return distance_transform_edt(~binary)
    # Fallback (slow, rarely used): brute force via cv2 distanceTransform.
    if _HAS_CV2:
        src = (~binary).astype(np.uint8) * 255
        return cv2.distanceTransform((binary == 0).astype(np.uint8), cv2.DIST_L2, 5)
    raise RuntimeError("need scipy or cv2 for distance transform")


@dataclass
class RegionPair:
    """Static (single-frame) geometric relation between two regions."""
    boundary_dist: float          # min pixel distance between region boundaries (0 if overlapping)
    centroid_dist: float          # distance between centroids
    overlap_px: int               # number of overlapping pixels
    centroid_a: Optional[np.ndarray]
    centroid_b: Optional[np.ndarray]
    valid: bool                   # both regions non-empty


def region_relation(mask_a: np.ndarray, mask_b: np.ndarray) -> RegionPair:
    """Compute boundary distance / centroid distance / overlap for two masks."""
    ca, cb = mask_centroid(mask_a), mask_centroid(mask_b)
    if ca is None or cb is None:
        return RegionPair(np.inf, np.inf, 0, ca, cb, False)

    overlap = int(np.count_nonzero(mask_a & mask_b))
    if overlap > 0:
        bdist = 0.0
    else:
        # distance from every pixel to nearest robot(a) pixel, sampled on human(b)
        edt = _edt(mask_a)
        bdist = float(edt[mask_b].min())
    cdist = float(np.linalg.norm(ca - cb))
    return RegionPair(bdist, cdist, overlap, ca, cb, True)


# ----------------------------------------------------------------------------
# Robot region from projected FK keypoints (the "use our model" path)
# ----------------------------------------------------------------------------
def keypoints_to_region(
    kps_2d: np.ndarray,
    shape: tuple[int, int],
    links: Sequence[tuple[int, int]] = PANDA_LINKS,
    thickness: int = 18,
    conf: Optional[np.ndarray] = None,
    conf_thr: float = 0.05,
) -> np.ndarray:
    """
    Rasterize the robot's 2D occupied region from projected keypoints.

    kps_2d : (K,2) pixel coords of the projected FK keypoints (our solver output).
    shape  : (H, W) of the target mask.
    links  : pairs of keypoint indices to connect as thick capsules.
    thickness : link radius in pixels (approx. arm thickness in the image).
    conf   : optional (K,) keypoint confidences; low-conf joints are skipped.

    Returns a boolean (H, W) mask.
    """
    H, W = shape
    mask = np.zeros((H, W), dtype=np.uint8)
    if not _HAS_CV2:
        raise RuntimeError("keypoints_to_region needs cv2")
    good = np.ones(len(kps_2d), dtype=bool)
    if conf is not None:
        good = conf >= conf_thr
    for i, j in links:
        if i >= len(kps_2d) or j >= len(kps_2d):
            continue
        if not (good[i] and good[j]):
            continue
        p1 = tuple(np.round(kps_2d[i]).astype(int))
        p2 = tuple(np.round(kps_2d[j]).astype(int))
        cv2.line(mask, p1, p2, color=1, thickness=thickness, lineType=cv2.LINE_8)
    # round the joints too, so the capsule union has rounded caps
    for k, p in enumerate(kps_2d):
        if not good[k]:
            continue
        cv2.circle(mask, tuple(np.round(p).astype(int)), thickness // 2, 1, -1)
    return mask.astype(bool)


# ----------------------------------------------------------------------------
# Distance -> probability
# ----------------------------------------------------------------------------
@dataclass
class ProbModel:
    """
    Monotone distance->probability map.

    d_safe   : distance (px, or mm if metric) at which p = 0.5. Below it the
               objects are "dangerously close".
    softness : logistic width; larger = gentler transition.
    """
    d_safe: float = 40.0
    softness: float = 20.0

    def prob(self, dist: float) -> float:
        if not np.isfinite(dist):
            return 0.0
        return float(sigmoid((self.d_safe - dist) / self.softness))


# ----------------------------------------------------------------------------
# Motion tracking (per object) — centroid history -> velocity vector
# ----------------------------------------------------------------------------
@dataclass
class MotionTrack:
    """Sliding-window centroid history for one object; robust velocity."""
    window: int = 6
    hist: deque = field(default_factory=lambda: deque(maxlen=64))  # (frame_idx, centroid)

    def update(self, frame_idx: int, centroid: Optional[np.ndarray]):
        if centroid is not None:
            self.hist.append((frame_idx, centroid.copy()))

    def velocity(self) -> np.ndarray:
        """Least-squares velocity (px/frame) over the recent window."""
        if len(self.hist) < 2:
            return np.zeros(2)
        pts = list(self.hist)[-self.window:]
        t = np.array([p[0] for p in pts], dtype=np.float64)
        xy = np.stack([p[1] for p in pts], axis=0)  # (n,2)
        t = t - t.mean()
        denom = float((t * t).sum())
        if denom < 1e-9:
            return np.zeros(2)
        vx = float((t * (xy[:, 0] - xy[:, 0].mean())).sum() / denom)
        vy = float((t * (xy[:, 1] - xy[:, 1].mean())).sum() / denom)
        return np.array([vx, vy])

    @property
    def last(self) -> Optional[np.ndarray]:
        return self.hist[-1][1] if self.hist else None


# ----------------------------------------------------------------------------
# Full temporal collision predictor
# ----------------------------------------------------------------------------
@dataclass
class CollisionReport:
    frame_idx: int
    valid: bool
    boundary_dist: float
    centroid_dist: float
    prob_now: float                 # instantaneous, distance-only
    prob_pred: float                # predictive (motion-projected) probability
    risk: float                     # fused risk = max(prob_now, prob_pred)
    closing_speed: float            # px/frame, >0 = approaching
    ttc: float                      # frames to contact (inf if not closing)
    human_toward_robot: float       # component of human velocity toward robot (px/frame)
    robot_toward_human: float       # component of robot velocity toward human (px/frame)
    robot_centroid: Optional[np.ndarray] = None
    human_centroid: Optional[np.ndarray] = None
    robot_vel: Optional[np.ndarray] = None
    human_vel: Optional[np.ndarray] = None


class CollisionEstimator:
    """
    Frame-by-frame collision-probability estimator with temporal prediction.

    Call `.step(frame_idx, robot_mask, human_mask)` for each frame in order.
    Returns a CollisionReport. Maintains internal motion tracks.
    """

    def __init__(
        self,
        prob_model: Optional[ProbModel] = None,
        horizon: int = 8,             # frames to look ahead for prediction
        window: int = 6,              # velocity smoothing window
    ):
        self.pm = prob_model or ProbModel()
        self.horizon = horizon
        self.robot = MotionTrack(window=window)
        self.human = MotionTrack(window=window)

    def step(self, frame_idx: int, robot_mask: np.ndarray, human_mask: np.ndarray) -> CollisionReport:
        rel = region_relation(robot_mask, human_mask)
        self.robot.update(frame_idx, rel.centroid_a)
        self.human.update(frame_idx, rel.centroid_b)

        if not rel.valid:
            return CollisionReport(
                frame_idx, False, rel.boundary_dist, rel.centroid_dist,
                0.0, 0.0, 0.0, 0.0, np.inf, 0.0, 0.0,
                rel.centroid_a, rel.centroid_b, None, None,
            )

        v_r = self.robot.velocity()
        v_h = self.human.velocity()

        # unit vector robot -> human
        sep = rel.centroid_b - rel.centroid_a
        D = np.linalg.norm(sep)
        e = sep / D if D > 1e-6 else np.zeros(2)

        # directional components (px/frame, positive = moving toward the other)
        human_toward_robot = float(-np.dot(v_h, e))   # human moving against the away direction
        robot_toward_human = float(np.dot(v_r, e))
        # rate of change of separation = e . (v_h - v_r); closing = -that
        closing_speed = float(np.dot(v_r - v_h, e))

        # instantaneous probability from current boundary distance
        prob_now = self.pm.prob(rel.boundary_dist)

        # predictive: project the boundary distance `horizon` frames ahead using
        # the closing speed (clamped at 0). If separating, future distance grows.
        d_future = rel.boundary_dist - closing_speed * self.horizon
        d_future = max(0.0, d_future)
        prob_pred = self.pm.prob(d_future)

        # time-to-contact (boundary) in frames
        ttc = rel.boundary_dist / closing_speed if closing_speed > 1e-3 else np.inf

        risk = max(prob_now, prob_pred)

        return CollisionReport(
            frame_idx, True, rel.boundary_dist, rel.centroid_dist,
            prob_now, prob_pred, risk, closing_speed, ttc,
            human_toward_robot, robot_toward_human,
            rel.centroid_a, rel.centroid_b, v_r, v_h,
        )
