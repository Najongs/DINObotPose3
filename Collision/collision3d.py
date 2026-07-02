"""
collision3d.py — METRIC 3D human/robot collision-probability estimation.

Why 3D: our pose model already gives the robot's metric 3D joints in the camera
frame (`kp_cam`, meters) from a single image. So the robot side is fully 3D for
free. The only missing piece is the human's depth; once we have human 3D points
(camera frame, meters — e.g. from monocular depth back-projection, see
depth3d.py), collision becomes a real 3D distance instead of an image-plane one.

Geometry:
  - Robot = a set of 3D CAPSULES (link segments between consecutive 3D joints,
    each with a physical radius). This is the swept volume of the arm.
  - Human = a 3D point cloud (back-projected mask pixels) or 3D keypoints.
  - Collision distance = min over human points of (distance to nearest robot
    capsule surface) in METERS. Negative = interpenetration.
  - Probability = logistic on that metric distance vs a metric safety margin.
  - Temporal = track 3D centroids -> 3D velocity vectors -> closing speed / TTC,
    which now sees DEPTH-axis approach the 2D version was blind to.

Pure numpy. Mirrors collision.py but in metric 3D.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

PANDA_LINKS = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ----------------------------------------------------------------------------
# 3D distance primitives
# ----------------------------------------------------------------------------
def point_segment_distance(P: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from each point in P (N,3) to the segment a-b (both (3,)). -> (N,)."""
    ab = b - a
    denom = float(ab @ ab)
    if denom < 1e-12:
        return np.linalg.norm(P - a[None], axis=1)
    t = np.clip((P - a[None]) @ ab / denom, 0.0, 1.0)
    proj = a[None] + t[:, None] * ab[None]
    return np.linalg.norm(P - proj, axis=1)


def capsule_surface_distance(
    P: np.ndarray,
    joints: np.ndarray,
    links: Sequence[tuple[int, int]] = PANDA_LINKS,
    radius: float = 0.06,
) -> np.ndarray:
    """
    Signed distance (m) from each point in P (N,3) to the robot's capsule surface
    (min over links of axis-distance − radius). Negative = inside the arm volume.
    Returns (N,).
    """
    best = np.full(len(P), np.inf)
    for i, j in links:
        d = point_segment_distance(P, joints[i], joints[j]) - radius
        best = np.minimum(best, d)
    return best


@dataclass
class Relation3D:
    surface_dist: float             # min human-point -> robot-surface distance (m); <0 = penetrating
    centroid_dist: float            # 3D centroid distance (m)
    centroid_r: Optional[np.ndarray]
    centroid_h: Optional[np.ndarray]
    closest_h: Optional[np.ndarray]  # human point nearest the robot
    valid: bool


def relation_3d(
    robot_joints: np.ndarray,
    human_points: np.ndarray,
    links: Sequence[tuple[int, int]] = PANDA_LINKS,
    radius: float = 0.06,
) -> Relation3D:
    if robot_joints is None or human_points is None or len(human_points) == 0:
        return Relation3D(np.inf, np.inf, None, None, None, False)
    d = capsule_surface_distance(human_points, robot_joints, links, radius)
    k = int(np.argmin(d))
    cr = robot_joints.mean(axis=0)
    ch = human_points.mean(axis=0)
    return Relation3D(float(d[k]), float(np.linalg.norm(cr - ch)),
                      cr, ch, human_points[k].copy(), True)


@dataclass
class ProbModel3D:
    """Metric distance->probability. d_safe/softness in METERS."""
    d_safe: float = 0.15
    softness: float = 0.08

    def prob(self, dist_m: float) -> float:
        if not np.isfinite(dist_m):
            return 0.0
        return float(sigmoid((self.d_safe - dist_m) / self.softness))


@dataclass
class MotionTrack3D:
    window: int = 6
    hist: deque = field(default_factory=lambda: deque(maxlen=64))

    def update(self, frame_idx: int, c: Optional[np.ndarray]):
        if c is not None:
            self.hist.append((frame_idx, c.copy()))

    def velocity(self) -> np.ndarray:
        """Least-squares 3D velocity (m/frame) over the recent window."""
        if len(self.hist) < 2:
            return np.zeros(3)
        pts = list(self.hist)[-self.window:]
        t = np.array([p[0] for p in pts], dtype=np.float64)
        xyz = np.stack([p[1] for p in pts], axis=0)
        t = t - t.mean()
        denom = float((t * t).sum())
        if denom < 1e-9:
            return np.zeros(3)
        return np.array([float((t * (xyz[:, k] - xyz[:, k].mean())).sum() / denom)
                         for k in range(3)])


@dataclass
class Report3D:
    frame_idx: int
    valid: bool
    surface_dist: float
    centroid_dist: float
    prob_now: float
    prob_pred: float
    risk: float
    closing_speed: float            # m/frame, >0 = approaching in 3D
    ttc: float                      # frames to contact
    human_toward_robot: float       # m/frame, human 3D velocity toward robot
    robot_toward_human: float
    depth_gap: float                # |z_robot - z_human| (m) — the info 2D lacks
    centroid_r: Optional[np.ndarray] = None
    centroid_h: Optional[np.ndarray] = None
    vel_r: Optional[np.ndarray] = None
    vel_h: Optional[np.ndarray] = None


class CollisionEstimator3D:
    """Metric 3D collision estimator with motion-vector prediction."""

    def __init__(self, prob_model: Optional[ProbModel3D] = None, horizon: int = 8,
                 window: int = 6, links=PANDA_LINKS, radius: float = 0.06):
        self.pm = prob_model or ProbModel3D()
        self.horizon = horizon
        self.links = links
        self.radius = radius
        self.robot = MotionTrack3D(window=window)
        self.human = MotionTrack3D(window=window)

    def step(self, frame_idx: int, robot_joints: np.ndarray, human_points: np.ndarray) -> Report3D:
        rel = relation_3d(robot_joints, human_points, self.links, self.radius)
        self.robot.update(frame_idx, rel.centroid_r)
        self.human.update(frame_idx, rel.centroid_h)

        if not rel.valid:
            return Report3D(frame_idx, False, rel.surface_dist, rel.centroid_dist,
                            0, 0, 0, 0, np.inf, 0, 0, np.inf, None, None, None, None)

        v_r = self.robot.velocity()
        v_h = self.human.velocity()
        sep = rel.centroid_h - rel.centroid_r
        D = np.linalg.norm(sep)
        e = sep / D if D > 1e-9 else np.zeros(3)

        human_toward_robot = float(-np.dot(v_h, e))
        robot_toward_human = float(np.dot(v_r, e))
        closing_speed = float(np.dot(v_r - v_h, e))

        prob_now = self.pm.prob(rel.surface_dist)
        d_future = max(-radius_floor(self.radius), rel.surface_dist - closing_speed * self.horizon)
        prob_pred = self.pm.prob(d_future)
        ttc = rel.surface_dist / closing_speed if (closing_speed > 1e-4 and rel.surface_dist > 0) else (
            0.0 if rel.surface_dist <= 0 else np.inf)
        risk = max(prob_now, prob_pred)
        depth_gap = abs(float(rel.centroid_r[2] - rel.centroid_h[2]))

        return Report3D(frame_idx, True, rel.surface_dist, rel.centroid_dist,
                        prob_now, prob_pred, risk, closing_speed, ttc,
                        human_toward_robot, robot_toward_human, depth_gap,
                        rel.centroid_r, rel.centroid_h, v_r, v_h)


def radius_floor(r: float) -> float:
    """Clamp future distance so a projected penetration doesn't run away."""
    return r
