"""
Dataset Inference Script for DINOv3 Pose Estimation
Evaluates a trained model on a dataset with metrics:
- L2 error (px) for in-frame keypoints with AUC
- ADD (m) from selected 3D source (fk/fused), transformed to camera frame
- ADD (m) from PnP baseline (DREAM-style): 2D pred + GT_3d + K -> PnP -> ADD
"""

import argparse
import os
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional DREAM utilities (fallback to OpenCV-only path when unavailable)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../DREAM")))
try:
    import dream  # type: ignore
    DREAM_AVAILABLE = True
except Exception:
    dream = None
    DREAM_AVAILABLE = False

# Import model from TRAIN directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../TRAIN')))
from model import DINOv3PoseEstimator, panda_forward_kinematics
from checkpoint_compat import load_checkpoint_compat


def _ensure_panda_angles_7(joint_angles_tensor: torch.Tensor, fix_joint7_zero: bool = False) -> torch.Tensor:
    """Convert predicted joint angles (6/7 DoF) to 7-DoF Panda vector for FK."""
    if joint_angles_tensor.shape[1] >= 7:
        angles7 = joint_angles_tensor[:, :7].clone()
    else:
        pad = torch.zeros(
            (joint_angles_tensor.shape[0], 7 - joint_angles_tensor.shape[1]),
            device=joint_angles_tensor.device,
            dtype=joint_angles_tensor.dtype,
        )
        angles7 = torch.cat([joint_angles_tensor, pad], dim=1)
    if fix_joint7_zero and angles7.shape[1] >= 7:
        angles7[:, 6] = 0.0
    return angles7


def setup_distributed(enable_distributed: bool):
    """Initialize distributed inference if launched by torchrun."""
    if not enable_distributed:
        return False, -1, 1, -1

    if not ('RANK' in os.environ and 'WORLD_SIZE' in os.environ and 'LOCAL_RANK' in os.environ):
        print("Warning: --distributed set but torchrun env not found. Falling back to single process.")
        return False, -1, 1, -1

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])

    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    dist.init_process_group(backend=backend)
    return True, rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


class InferenceDataset(Dataset):
    """Dataset for inference on converted DREAM-format data (json contains meta.K, meta.image_path)"""

    def __init__(self, data_dir: str, keypoint_names: List[str], image_size: Tuple[int, int]=(512,512)):
        self.data_dir = Path(data_dir)
        self.keypoint_names = keypoint_names
        self.image_size = image_size
        self.is_synthetic = 'syn' in str(self.data_dir).lower()

        self.json_files = sorted(list(self.data_dir.glob("*.json")))
        if len(self.json_files) == 0:
            raise ValueError(f"No JSON files found in {data_dir}")

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

        print(f"Found {len(self.json_files)} json frames in {data_dir}")
        if self.is_synthetic:
            print(f"  Detected as SYNTHETIC data (3D GT will be converted cm -> m)")

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        json_path = self.json_files[idx]
        with open(json_path, "r") as f:
            data = json.load(f)

        # Resolve image path from meta.image_path
        img_path_str = data.get("meta", {}).get("image_path", None)
        if img_path_str is None:
            raise KeyError(f"'meta.image_path' missing in {json_path}")

        # Fix incorrect relative path: ../dataset/... should be ../../../...
        if img_path_str.startswith('../dataset/'):
            img_path_str = img_path_str.replace('../dataset/', '../../../', 1)

        img_path = (json_path.parent / img_path_str).resolve()
        if not img_path.exists():
            img_path = (self.data_dir / img_path_str).resolve()
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found for {json_path}: {img_path_str}")

        image = Image.open(img_path).convert("RGB")

        # Extract keypoints (same logic as before)
        keypoints_2d = []
        keypoints_3d = []

        if "objects" in data and len(data["objects"]) > 0:
            obj = data["objects"][0]
            if "keypoints" in obj:
                kp_dict = {kp["name"]: kp for kp in obj["keypoints"]}
                for kp_name in self.keypoint_names:
                    if kp_name in kp_dict:
                        kp = kp_dict[kp_name]
                        keypoints_2d.append(kp["projected_location"])
                        keypoints_3d.append(kp["location"])
                    else:
                        keypoints_2d.append([-999.0, -999.0])
                        keypoints_3d.append([0.0, 0.0, 0.0])

        keypoints_2d = np.array(keypoints_2d, dtype=np.float32)
        keypoints_3d = np.array(keypoints_3d, dtype=np.float32)

        # Synthetic data (DREAM sim) uses cm, convert to meters
        if self.is_synthetic:
            keypoints_3d = keypoints_3d / 100.0

        image_tensor = self.transform(image)

        # Per-sample camera intrinsics
        camera_K = np.zeros((3, 3), dtype=np.float32)
        if "meta" in data and "K" in data["meta"]:
            camera_K = np.array(data["meta"]["K"], dtype=np.float32)

        # Original image size (W, H)
        original_size = np.array([image.width, image.height], dtype=np.float32)

        # Angles (same)
        angles = np.zeros(9, dtype=np.float32)
        if "sim_state" in data and "joints" in data["sim_state"]:
            joints = data["sim_state"]["joints"]
            for i, joint in enumerate(joints[:9]):
                if "position" in joint:
                    angles[i] = joint["position"]

        return {
            "image": image_tensor,
            "keypoints": keypoints_2d,
            "keypoints_3d": keypoints_3d,
            "camera_K": camera_K,
            "original_size": original_size,
            "angles": angles,
            "image_path": str(img_path),
            "name": json_path.stem,
        }


def get_keypoints_from_heatmaps(
    heatmaps: torch.Tensor, min_confidence: float = 0.0, min_peak_logit: float = -1e9
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract keypoint coordinates from heatmaps using argmax.

    Args:
        heatmaps: (B, N, H, W) tensor

    Returns:
        keypoints: (B, N, 2) numpy array [x, y], low-confidence points as -999
        confidences: (B, N) numpy array in [0, 1]
    """
    B, N, H, W = heatmaps.shape
    heatmaps_flat = heatmaps.view(B, N, -1)
    max_indices = torch.argmax(heatmaps_flat, dim=-1)

    y = max_indices // W
    x = max_indices % W

    keypoints = torch.stack([x, y], dim=-1).float()
    confidences = torch.sigmoid(heatmaps_flat.amax(dim=-1))

    peak_logits = heatmaps_flat.amax(dim=-1)
    invalid = torch.zeros_like(confidences, dtype=torch.bool)
    if min_confidence > 0.0:
        invalid = invalid | (confidences < float(min_confidence))
    if min_peak_logit is not None:
        invalid = invalid | (peak_logits < float(min_peak_logit))
    if invalid.any():
        keypoints = keypoints.masked_fill(invalid.unsqueeze(-1), -999.0)

    return keypoints.cpu().numpy(), confidences.cpu().numpy()


def passes_pnp_spread_check(points_2d: np.ndarray, image_resolution: Tuple[int, int], min_span_px: float, min_area_ratio: float) -> bool:
    """Reject degenerate PnP point sets that are too concentrated in image space."""
    if points_2d.shape[0] < 4:
        return False
    x_span = float(np.max(points_2d[:, 0]) - np.min(points_2d[:, 0]))
    y_span = float(np.max(points_2d[:, 1]) - np.min(points_2d[:, 1]))
    bbox_area = max(0.0, x_span) * max(0.0, y_span)
    img_area = max(1.0, float(image_resolution[0] * image_resolution[1]))
    area_ratio = bbox_area / img_area
    return bool(
        (x_span >= float(min_span_px))
        and (y_span >= float(min_span_px))
        and (area_ratio >= float(min_area_ratio))
    )


def _optimize_translation_z_for_reprojection(
    robot_kpts: np.ndarray,
    pred_2d: np.ndarray,
    camera_K: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    z_min_m: float = -0.05,
    z_max_m: float = 0.05,
    z_step_m: float = 0.001,
    min_points: int = 4,
) -> Tuple[np.ndarray, Optional[float], float]:
    """Post-PnP Z-only search that minimizes 2D reprojection error."""
    valid = (
        np.isfinite(robot_kpts).all(axis=1)
        & np.isfinite(pred_2d).all(axis=1)
        & (pred_2d[:, 0] > -900.0)
        & (pred_2d[:, 1] > -900.0)
    )
    idx = np.where(valid)[0]
    if idx.shape[0] < int(min_points):
        return np.array(tvec, dtype=np.float64).reshape(3, 1), None, 0.0

    obj_valid = robot_kpts[idx].astype(np.float64)
    img_valid = pred_2d[idx].astype(np.float64)
    K = camera_K.astype(np.float64)
    rv = np.array(rvec, dtype=np.float64).reshape(3, 1)
    t_base = np.array(tvec, dtype=np.float64).reshape(3)

    step = float(z_step_m)
    if step <= 0.0:
        return t_base.reshape(3, 1), None, 0.0

    z_lo = float(z_min_m)
    z_hi = float(z_max_m)
    if z_hi < z_lo:
        z_lo, z_hi = z_hi, z_lo

    candidates = np.arange(z_lo, z_hi + 0.5 * step, step, dtype=np.float64)
    best_t = t_base.copy()
    best_err = None
    best_offset = 0.0

    for dz in candidates:
        t_try = t_base.copy()
        t_try[2] += float(dz)
        proj, _ = cv2.projectPoints(obj_valid, rv, t_try.reshape(3, 1), K, None)
        proj = proj.reshape(-1, 2)
        err = float(np.mean(np.linalg.norm(proj - img_valid, axis=1)))
        if (best_err is None) or (err < best_err):
            best_err = err
            best_t = t_try
            best_offset = float(dz)

    return best_t.reshape(3, 1), best_err, best_offset


def transform_robot_to_camera(
    robot_kpts,
    pred_2d,
    camera_K,
    apply_z_search: bool = False,
    z_search_min_m: float = -0.05,
    z_search_max_m: float = 0.05,
    z_search_step_m: float = 0.001,
):
    """
    Transform robot frame keypoints to camera frame using EPnP.

    Args:
        robot_kpts: (N, 3) keypoints in robot frame
        pred_2d: (N, 2) predicted 2D keypoints in image
        camera_K: (3, 3) camera intrinsic matrix

    Returns:
        camera_kpts: (N, 3) keypoints in camera frame (or None if PnP fails)
        proj_2d_all: (N, 2) reprojection of all robot points with solved pose (or None)
    """
    try:
        valid = (
            np.isfinite(robot_kpts).all(axis=1)
            & np.isfinite(pred_2d).all(axis=1)
            & (pred_2d[:, 0] > -900.0)
            & (pred_2d[:, 1] > -900.0)
        )
        if np.count_nonzero(valid) < 4:
            return None, None
        robot_kpts_valid = robot_kpts[valid]
        pred_2d_valid = pred_2d[valid]

        # Ensure correct data types
        robot_kpts = robot_kpts.astype(np.float64)
        pred_2d = pred_2d.astype(np.float64)
        camera_K = camera_K.astype(np.float64)

        # Solve PnP with EPnP algorithm
        success, rvec, tvec = cv2.solvePnP(
            robot_kpts_valid,
            pred_2d_valid,
            camera_K,
            None,  # No distortion
            flags=cv2.SOLVEPNP_EPNP  # Use EPnP algorithm
        )

        if not success:
            return None, None

        if apply_z_search:
            tvec, _, _ = _optimize_translation_z_for_reprojection(
                robot_kpts=robot_kpts,
                pred_2d=pred_2d,
                camera_K=camera_K,
                rvec=rvec,
                tvec=tvec,
                z_min_m=z_search_min_m,
                z_max_m=z_search_max_m,
                z_step_m=z_search_step_m,
                min_points=4,
            )

        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.flatten()

        # Transform all robot-frame keypoints with estimated pose so output shape stays (N, 3).
        camera_kpts = (R @ robot_kpts.T).T + t.reshape(1, 3)
        proj_all, _ = cv2.projectPoints(robot_kpts.astype(np.float64), rvec, tvec, camera_K, None)

        return camera_kpts, proj_all.reshape(-1, 2)

    except Exception as e:
        return None, None


def solve_pnp_epnp_iterative(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    camera_K: np.ndarray,
) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
    """OpenCV PnP fallback equivalent to DREAM-style EPnP(+iterative refine)."""
    try:
        ok, rvec, tvec = cv2.solvePnP(
            points_3d.astype(np.float64),
            points_2d.astype(np.float64),
            camera_K.astype(np.float64),
            None,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not ok:
            return False, None, None
        ok, rvec, tvec = cv2.solvePnP(
            points_3d.astype(np.float64),
            points_2d.astype(np.float64),
            camera_K.astype(np.float64),
            None,
            flags=cv2.SOLVEPNP_ITERATIVE,
            useExtrinsicGuess=True,
            rvec=rvec,
            tvec=tvec,
        )
        if not ok:
            return False, None, None
        return True, rvec, tvec
    except Exception:
        return False, None, None


def add_from_pose_rvec_tvec(
    rvec: np.ndarray,
    tvec: np.ndarray,
    points_3d: np.ndarray,
) -> float:
    """
    ADD proxy without DREAM:
    transform 3D points by solved pose and compare to original points.
    """
    R, _ = cv2.Rodrigues(np.array(rvec, dtype=np.float64).reshape(3, 1))
    t = np.array(tvec, dtype=np.float64).reshape(3)
    pts = np.array(points_3d, dtype=np.float64).reshape(-1, 3)
    pts_tf = (R @ pts.T).T + t.reshape(1, 3)
    return float(np.linalg.norm(pts_tf - pts, axis=1).mean())


def compute_keypoint_metrics(
    kp_detected: np.ndarray,
    kp_gt: np.ndarray,
    image_resolution: Tuple[int, int],
    auc_threshold: float = 20.0,
    pck_thresholds_px: Tuple[float, ...] = (2.5, 5.0, 10.0),
) -> Dict:
    """
    Compute keypoint metrics similar to DREAM.

    Args:
        kp_detected: (N, 2) detected keypoints
        kp_gt: (N, 2) ground truth keypoints
        image_resolution: (width, height)
        auc_threshold: AUC threshold in pixels

    Returns:
        metrics dictionary
    """
    num_gt_inframe = 0
    num_found_gt_inframe = 0
    kp_errors = []

    for kp_det, kp_g in zip(kp_detected, kp_gt):
        # Check if GT is in frame
        if (0.0 <= kp_g[0] <= image_resolution[0] and
            0.0 <= kp_g[1] <= image_resolution[1]):
            num_gt_inframe += 1

            # Check if detected
            if kp_det[0] > -999.0 and kp_det[1] > -999.0:
                num_found_gt_inframe += 1
                kp_errors.append(kp_det - kp_g)

    kp_errors = np.array(kp_errors)

    if len(kp_errors) > 0:
        kp_l2_errors = np.linalg.norm(kp_errors, axis=1)
        kp_l2_error_mean = np.mean(kp_l2_errors)
        kp_l2_error_median = np.median(kp_l2_errors)
        kp_l2_error_std = np.std(kp_l2_errors)
        pck_percentages = {
            f'pck@{thresh:g}px_percent': float(np.mean(kp_l2_errors <= thresh) * 100.0)
            for thresh in pck_thresholds_px
        }

        # Compute AUC
        delta_pixel = 0.01
        pck_values = np.arange(0, auc_threshold, delta_pixel)
        y_values = []

        for value in pck_values:
            valids = len(np.where(kp_l2_errors < value)[0])
            y_values.append(valids)

        kp_auc = (
            np.trapz(y_values, dx=delta_pixel) /
            float(auc_threshold) /
            float(num_gt_inframe)
        )
    else:
        kp_l2_error_mean = None
        kp_l2_error_median = None
        kp_l2_error_std = None
        kp_auc = None
        pck_percentages = {
            f'pck@{thresh:g}px_percent': None for thresh in pck_thresholds_px
        }

    out = {
        'num_gt_inframe': num_gt_inframe,
        'num_found_gt_inframe': num_found_gt_inframe,
        'l2_error_mean_px': kp_l2_error_mean,
        'l2_error_median_px': kp_l2_error_median,
        'l2_error_std_px': kp_l2_error_std,
        'l2_error_auc': kp_auc,
        'l2_error_auc_thresh_px': auc_threshold,
    }
    out.update(pck_percentages)
    return out


def compute_pnp_metrics(
    pnp_add: List[float],
    num_inframe_projs_gt: List[int],
    num_min_inframe_projs_gt_for_pnp: int = 4,
    add_auc_threshold: float = 0.1,
    pnp_magic_number: float = -999.0
) -> Dict:
    """
    Compute PnP metrics similar to DREAM.

    Args:
        pnp_add: List of ADD values
        num_inframe_projs_gt: Number of in-frame GT keypoints per sample
        num_min_inframe_projs_gt_for_pnp: Minimum keypoints for PnP
        add_auc_threshold: AUC threshold in meters
        pnp_magic_number: Magic number for failed PnP

    Returns:
        metrics dictionary
    """
    pnp_add = np.array(pnp_add)
    num_inframe_projs_gt = np.array(num_inframe_projs_gt)

    idx_pnp_found = np.where(pnp_add > pnp_magic_number)[0]
    add_pnp_found = pnp_add[idx_pnp_found]
    num_pnp_found = len(idx_pnp_found)

    if num_pnp_found > 0:
        mean_add = np.mean(add_pnp_found)
        median_add = np.median(add_pnp_found)
        std_add = np.std(add_pnp_found)
    else:
        mean_add = None
        median_add = None
        std_add = None

    num_pnp_possible = len(
        np.where(num_inframe_projs_gt >= num_min_inframe_projs_gt_for_pnp)[0]
    )
    num_pnp_not_found = num_pnp_possible - num_pnp_found

    delta_threshold = 0.00001
    add_threshold_values = np.arange(0.0, add_auc_threshold, delta_threshold)

    if num_pnp_possible > 0:
        counts = []
        for value in add_threshold_values:
            under_threshold = len(np.where(add_pnp_found <= value)[0]) / float(
                num_pnp_possible
            )
            counts.append(under_threshold)
        auc = np.trapz(counts, dx=delta_threshold) / float(add_auc_threshold)
    else:
        auc = None

    return {
        'num_pnp_found': num_pnp_found,
        'num_pnp_not_found': num_pnp_not_found,
        'num_pnp_possible': num_pnp_possible,
        'add_mean': mean_add,
        'add_median': median_add,
        'add_std': std_add,
        'add_auc': auc,
        'add_auc_thresh': add_auc_threshold,
    }

def compute_direct_add_metrics(
    pred_3d_all: np.ndarray,
    gt_3d_all: np.ndarray,
    gt_2d_all: np.ndarray,
    image_resolution: Tuple[int, int],
    add_auc_threshold: float = 0.1,
) -> Dict:
    """
    Compute ADD metrics by directly comparing model's predicted 3D keypoints
    with GT 3D keypoints (like HoRoPose / RoboPEPP evaluation).

    Args:
        pred_3d_all: (N_frames, N_kp, 3) predicted 3D keypoints from model
        gt_3d_all: (N_frames, N_kp, 3) ground truth 3D keypoints
        gt_2d_all: (N_frames, N_kp, 2) ground truth 2D keypoints (raw image coords)
        image_resolution: (W, H) image resolution used for in-frame filtering
        add_auc_threshold: AUC threshold in meters

    Returns:
        metrics dictionary
    """
    # Per-frame mean 3D distance (ADD = average of per-keypoint L2 distances)
    # Only use valid GT keypoints (not all-zero)
    frame_adds = []
    per_kp_errors = []

    img_w, img_h = image_resolution

    for pred_3d, gt_3d, gt_2d in zip(pred_3d_all, gt_3d_all, gt_2d_all):
        # Valid mask:
        # 1) GT keypoint exists in 2D annotation (not sentinel)
        # 2) GT 2D keypoint is in-frame (standard benchmark behavior)
        # 3) GT 3D keypoint is not all-zero
        has_2d = (gt_2d[:, 0] > -900.0) & (gt_2d[:, 1] > -900.0)
        in_frame = (
            (gt_2d[:, 0] >= 0.0) & (gt_2d[:, 0] <= img_w) &
            (gt_2d[:, 1] >= 0.0) & (gt_2d[:, 1] <= img_h)
        )
        has_3d = np.any(gt_3d != 0, axis=-1)
        valid = has_2d & in_frame & has_3d
        if valid.sum() == 0:
            continue

        pred_valid = pred_3d[valid]
        gt_valid = gt_3d[valid]

        # Per-keypoint L2 distance
        kp_dists = np.linalg.norm(pred_valid - gt_valid, axis=-1)  # (N_valid,)
        per_kp_errors.extend(kp_dists.tolist())

        # ADD = mean per-keypoint distance for this frame
        frame_add = np.mean(kp_dists)
        frame_adds.append(frame_add)

    frame_adds = np.array(frame_adds)
    per_kp_errors = np.array(per_kp_errors)
    n_frames = len(frame_adds)

    if n_frames == 0:
        return {
            'num_frames': 0,
            'add_mean': None,
            'add_median': None,
            'add_std': None,
            'add_auc': None,
            'add_auc_thresh': add_auc_threshold,
            'per_kp_mean': None,
            'per_kp_median': None,
        }

    # AUC computation (same method as RoboPEPP)
    delta_threshold = 0.00001
    add_threshold_values = np.arange(0.0, add_auc_threshold, delta_threshold)
    counts = []
    for value in add_threshold_values:
        under_threshold = np.mean(frame_adds <= value)
        counts.append(under_threshold)
    auc = np.trapz(counts, dx=delta_threshold) / float(add_auc_threshold)

    return {
        'num_frames': n_frames,
        'add_mean': float(np.mean(frame_adds)),
        'add_median': float(np.median(frame_adds)),
        'add_std': float(np.std(frame_adds)),
        'add_auc': float(auc),
        'add_auc_thresh': float(add_auc_threshold),
        'per_kp_mean': float(np.mean(per_kp_errors)),
        'per_kp_median': float(np.median(per_kp_errors)),
    }


def compute_robopepp_style_pnp_add_metrics(
    pred_2d_all: np.ndarray,
    pred_3d_robot_all: np.ndarray,
    gt_3d_all: np.ndarray,
    gt_2d_all: np.ndarray,
    camera_Ks: np.ndarray,
    image_resolution: Tuple[int, int],
    num_inframe_projs_gt: List[int],
    pred_conf_all: np.ndarray,
    add_auc_threshold: float = 0.1,
    pnp_magic_number: float = -999.0,
    min_points: int = 4,
    init_conf_thresh: float = 0.25,
    conf_step: float = 0.025,
    pnp_min_span_px: float = 20.0,
    pnp_min_area_ratio: float = 0.001,
    apply_z_search: bool = False,
    z_search_min_m: float = -0.05,
    z_search_max_m: float = 0.05,
    z_search_step_m: float = 0.001,
    return_raw_adds: bool = False,
) -> Dict:
    """
    RoboPEPP-style PnP ADD:
    pred_2d + pred_robot_3d(FK/source) + K -> PnP -> camera-frame pred_3d -> ADD vs GT camera 3D.
    """
    pnp_adds = []
    raw_w, raw_h = image_resolution

    for pred_2d, pred_3d_robot, gt_3d, gt_2d, sample_K, kp_conf in zip(
        pred_2d_all, pred_3d_robot_all, gt_3d_all, gt_2d_all, camera_Ks, pred_conf_all
    ):
        valid_pred2d = np.isfinite(pred_2d).all(axis=1) & (pred_2d[:, 0] > -999.0) & (pred_2d[:, 1] > -999.0)
        in_frame_gt2d = (
            np.isfinite(gt_2d).all(axis=1) &
            (gt_2d[:, 0] >= 0.0) & (gt_2d[:, 0] <= raw_w) &
            (gt_2d[:, 1] >= 0.0) & (gt_2d[:, 1] <= raw_h)
        )
        valid_gt3d = np.isfinite(gt_3d).all(axis=1) & (np.linalg.norm(gt_3d, axis=1) > 1e-8)
        base_valid = valid_pred2d & in_frame_gt2d & valid_gt3d

        thresh = init_conf_thresh
        idx = np.where(base_valid & (kp_conf > thresh))[0]
        while idx.shape[0] < min_points and thresh > -1.0:
            thresh -= conf_step
            idx = np.where(base_valid & (kp_conf > thresh))[0]
        if idx.shape[0] < min_points:
            idx = np.where(base_valid)[0]
        if idx.shape[0] < min_points:
            pnp_adds.append(pnp_magic_number)
            continue
        if not passes_pnp_spread_check(pred_2d[idx], image_resolution, pnp_min_span_px, pnp_min_area_ratio):
            pnp_adds.append(pnp_magic_number)
            continue

        try:
            success, rvec, tvec = cv2.solvePnP(
                pred_3d_robot[idx].astype(np.float64),
                pred_2d[idx].astype(np.float64),
                sample_K.astype(np.float64),
                None,
                flags=cv2.SOLVEPNP_EPNP,
            )
            if not success:
                pnp_adds.append(pnp_magic_number)
                continue

            if apply_z_search:
                tvec, _, _ = _optimize_translation_z_for_reprojection(
                    robot_kpts=pred_3d_robot,
                    pred_2d=pred_2d,
                    camera_K=sample_K,
                    rvec=rvec,
                    tvec=tvec,
                    z_min_m=z_search_min_m,
                    z_max_m=z_search_max_m,
                    z_step_m=z_search_step_m,
                    min_points=min_points,
                )

            R, _ = cv2.Rodrigues(rvec)
            t = tvec.flatten()
            pred_3d_cam = (R @ pred_3d_robot.T).T + t.reshape(1, 3)

            eval_mask = in_frame_gt2d & valid_gt3d
            if np.count_nonzero(eval_mask) == 0:
                pnp_adds.append(pnp_magic_number)
                continue

            add = float(np.linalg.norm(pred_3d_cam[eval_mask] - gt_3d[eval_mask], axis=1).mean())
            pnp_adds.append(add)
        except Exception:
            pnp_adds.append(pnp_magic_number)

    metrics = compute_pnp_metrics(
        pnp_add=pnp_adds,
        num_inframe_projs_gt=num_inframe_projs_gt,
        add_auc_threshold=add_auc_threshold,
        pnp_magic_number=pnp_magic_number,
    )
    if return_raw_adds:
        return metrics, pnp_adds
    return metrics


def collect_keypoint_l2_errors(
    kp_detected: np.ndarray,
    kp_gt: np.ndarray,
    image_resolution: Tuple[int, int],
) -> Tuple[np.ndarray, int]:
    """Collect valid in-frame 2D keypoint L2 errors and denominator used for PCK/AUC."""
    kp_l2_errors = []
    num_gt_inframe = 0
    img_w, img_h = image_resolution

    for kp_det, kp_g in zip(kp_detected, kp_gt):
        if (0.0 <= kp_g[0] <= img_w) and (0.0 <= kp_g[1] <= img_h):
            num_gt_inframe += 1
            if kp_det[0] > -999.0 and kp_det[1] > -999.0:
                kp_l2_errors.append(float(np.linalg.norm(kp_det - kp_g)))

    return np.array(kp_l2_errors, dtype=np.float64), int(num_gt_inframe)


def collect_direct_add_values(
    pred_3d_all: np.ndarray,
    gt_3d_all: np.ndarray,
    gt_2d_all: np.ndarray,
    image_resolution: Tuple[int, int],
) -> np.ndarray:
    """Collect per-frame direct ADD values (camera frame) with same validity as metric computation."""
    frame_adds = []
    img_w, img_h = image_resolution

    for pred_3d, gt_3d, gt_2d in zip(pred_3d_all, gt_3d_all, gt_2d_all):
        has_2d = (gt_2d[:, 0] > -900.0) & (gt_2d[:, 1] > -900.0)
        in_frame = (
            (gt_2d[:, 0] >= 0.0) & (gt_2d[:, 0] <= img_w) &
            (gt_2d[:, 1] >= 0.0) & (gt_2d[:, 1] <= img_h)
        )
        has_3d = np.any(gt_3d != 0, axis=-1)
        valid = has_2d & in_frame & has_3d
        if np.count_nonzero(valid) == 0:
            continue
        frame_adds.append(float(np.mean(np.linalg.norm(pred_3d[valid] - gt_3d[valid], axis=-1))))

    return np.array(frame_adds, dtype=np.float64)


def build_auc_curve(
    values: np.ndarray,
    x_max: float,
    denominator: int,
    invalid_magic: float = None,
    num_points: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build threshold-success curve used for AUC plots.
    y(x) = count(values <= x) / denominator.
    """
    if x_max <= 0.0 or denominator <= 0:
        return np.array([]), np.array([])

    arr = np.array(values, dtype=np.float64).reshape(-1)
    if invalid_magic is not None:
        arr = arr[arr > invalid_magic]
    arr = arr[np.isfinite(arr)]

    xs = np.linspace(0.0, float(x_max), int(num_points))
    if arr.size == 0:
        ys = np.zeros_like(xs)
    else:
        ys = np.array([np.count_nonzero(arr <= x) / float(denominator) for x in xs], dtype=np.float64)
    return xs, ys


def save_metric_plots(
    output_dir: Path,
    kp_errors: np.ndarray,
    kp_denom: int,
    kp_auc_threshold: float,
    pnp_adds_dream: List[float],
    pnp_adds_robopepp: List[float],
    num_pnp_possible: int,
    direct_adds: np.ndarray,
    add_auc_threshold: float,
    pred_3d_source: str,
) -> List[str]:
    """Save AUC/PCK plots and return generated file paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files = []

    # 1) PCK curve (2D)
    x_kp, y_kp = build_auc_curve(kp_errors, kp_auc_threshold, kp_denom, invalid_magic=None, num_points=600)
    if x_kp.size > 0:
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        ax.plot(x_kp, y_kp * 100.0, linewidth=2.5, color="#0072B2", label="PCK curve")
        for thr in (2.5, 5.0, 10.0):
            if thr <= kp_auc_threshold:
                val = np.count_nonzero(kp_errors <= thr) / float(kp_denom) if kp_denom > 0 else 0.0
                ax.scatter([thr], [val * 100.0], color="#D55E00", s=28, zorder=3)
        ax.set_title("2D Keypoint PCK Curve (AUC basis)")
        ax.set_xlabel("Pixel threshold (px)")
        ax.set_ylabel("PCK (%)")
        ax.set_xlim(0.0, kp_auc_threshold)
        ax.set_ylim(0.0, 100.0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right")
        fig.tight_layout()
        out_path = output_dir / "auc_curve_pck_2d.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        saved_files.append(str(out_path))

    # 2) ADD curves (camera frame): DREAM PnP + RoboPEPP PnP + Direct ADD
    x_add_dream, y_add_dream = build_auc_curve(
        np.array(pnp_adds_dream, dtype=np.float64),
        add_auc_threshold,
        num_pnp_possible,
        invalid_magic=-999.0,
        num_points=600,
    )
    x_add_robo, y_add_robo = build_auc_curve(
        np.array(pnp_adds_robopepp, dtype=np.float64),
        add_auc_threshold,
        num_pnp_possible,
        invalid_magic=-999.0,
        num_points=600,
    )
    x_add_direct, y_add_direct = build_auc_curve(
        np.array(direct_adds, dtype=np.float64),
        add_auc_threshold,
        int(len(direct_adds)),
        invalid_magic=None,
        num_points=600,
    )

    if x_add_dream.size > 0 or x_add_robo.size > 0 or x_add_direct.size > 0:
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        if x_add_robo.size > 0:
            ax.plot(x_add_robo, y_add_robo * 100.0, linewidth=2.5, color="#009E73", label="PnP ADD - RoboPEPP")
        if x_add_dream.size > 0:
            ax.plot(x_add_dream, y_add_dream * 100.0, linewidth=2.5, color="#CC79A7", label="PnP ADD - DREAM baseline")
        if x_add_direct.size > 0:
            ax.plot(x_add_direct, y_add_direct * 100.0, linewidth=2.5, color="#E69F00", label=f"Direct ADD - {pred_3d_source} source")
        ax.set_title("ADD Curves (camera-frame, AUC basis)")
        ax.set_xlabel("ADD threshold (m)")
        ax.set_ylabel("Success rate (%)")
        ax.set_xlim(0.0, add_auc_threshold)
        ax.set_ylim(0.0, 100.0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right")
        fig.tight_layout()
        out_path = output_dir / "auc_curve_add_camera_frame.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        saved_files.append(str(out_path))

    return saved_files


def load_camera_from_first_frame(dataset_dir: Path) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Load camera intrinsics K and raw image resolution (W,H) from the first frame json.
    Expected json format:
    {
      "meta": {
        "K": [[fx,0,cx],[0,fy,cy],[0,0,1]],
        "image_path": "path/to/000000.rgb.jpg"  # optional but recommended
      }
    }
    """
    json_files = sorted(dataset_dir.glob("*.json"))
    if len(json_files) == 0:
        raise FileNotFoundError(f"No json files found in {dataset_dir}")

    first_json = json_files[0]
    with open(first_json, "r") as f:
        data = json.load(f)

    if "meta" not in data or "K" not in data["meta"]:
        raise KeyError(f"'meta.K' not found in {first_json}")

    K = np.array(data["meta"]["K"], dtype=np.float32)

    # Determine raw resolution by opening the referenced image
    img_path_str = data["meta"].get("image_path", None)
    if img_path_str is None:
        raise KeyError(f"'meta.image_path' not found in {first_json} (needed to get resolution)")

    # Fix incorrect relative path: ../dataset/... should be ../../../...
    if img_path_str.startswith('../dataset/'):
        img_path_str = img_path_str.replace('../dataset/', '../../../', 1)

    # Resolve image path relative to the json file location
    img_path = (first_json.parent / img_path_str).resolve()
    if not img_path.exists():
        # fallback: try resolving relative to dataset_dir
        img_path = (dataset_dir / img_path_str).resolve()

    if not img_path.exists():
        raise FileNotFoundError(f"Image for resolution not found. Tried: {img_path}")

    with Image.open(img_path) as im:
        w, h = im.size

    return K, (w, h)

@torch.no_grad()
def run_inference(args):
    """Run inference on dataset and compute metrics"""
    is_distributed, rank, world_size, local_rank = setup_distributed(args.distributed)
    is_main_process = (not is_distributed) or (rank == 0)
    if is_main_process and not DREAM_AVAILABLE:
        print("Info: DREAM package not found. Using OpenCV fallback for PnP/ADD baseline.")

    dataset_dir = Path(args.dataset_dir)
    camera_K, raw_resolution = load_camera_from_first_frame(dataset_dir)

    if is_main_process:
        print(f"Camera intrinsics:\n{camera_K}")
        print(f"Raw resolution: {raw_resolution}")

    # Load training config from checkpoint directory
    checkpoint_dir = Path(args.model_path).parent
    config_path = checkpoint_dir / 'config.yaml'

    # Defaults
    keypoint_names = [
        'panda_link0', 'panda_link2', 'panda_link3',
        'panda_link4', 'panda_link6', 'panda_link7', 'panda_hand'
    ]
    train_config = {}

    if config_path.exists():
        import yaml
        with open(config_path, 'r') as f:
            train_config = yaml.safe_load(f)
        if 'keypoint_names' in train_config:
            keypoint_names = train_config['keypoint_names']
        if is_main_process:
            print(f"Loaded training config from {config_path}")
            print(f"  model_name: {train_config.get('model_name', 'N/A')}")
            print(f"  keypoint_names ({len(keypoint_names)}): {keypoint_names}")
    else:
        if is_main_process:
            print(f"Warning: Config not found at {config_path}, using defaults")

    # Resolve config values (CLI args override config.yaml)
    model_name = args.model_name or train_config.get('model_name', 'facebook/dinov3-vitb16-pretrain-lvd1689m')
    image_size = args.image_size or int(train_config.get('image_size', 512))
    heatmap_size = args.heatmap_size or int(train_config.get('heatmap_size', 512))
    use_joint_embedding = train_config.get('use_joint_embedding', False)
    fix_joint7_zero = args.fix_joint7_zero or bool(train_config.get('fix_joint7_zero', False))
    if is_main_process:
        print(f"  model_name: {model_name}")
        print(f"  image_size: {image_size}, heatmap_size: {heatmap_size}")
        print(f"  use_joint_embedding: {use_joint_embedding}")
        print(f"  fix_joint7_zero: {fix_joint7_zero}")
        print(f"  mode: joint_angle")

    # Create dataset
    dataset = InferenceDataset(
        data_dir=args.dataset_dir,
        keypoint_names=keypoint_names,
        image_size=(image_size, image_size)
    )

    sampler = None
    if is_distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False
        )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Load model
    if is_main_process:
        print(f"\nLoading model from {args.model_path}")
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}' if is_distributed else 'cuda')
    else:
        device = torch.device('cpu')

    use_iterative_refinement = train_config.get('use_iterative_refinement', False)
    refinement_iterations = int(train_config.get('refinement_iterations', 3))

    model = DINOv3PoseEstimator(
        dino_model_name=model_name,
        heatmap_size=(heatmap_size, heatmap_size),
        unfreeze_blocks=0,  # Not needed for inference
        use_joint_embedding=use_joint_embedding,
        use_iterative_refinement=use_iterative_refinement,
        refinement_iterations=refinement_iterations,
        fix_joint7_zero=fix_joint7_zero,
    ).to(device)

    load_checkpoint_compat(
        model=model,
        checkpoint_path=args.model_path,
        device=device,
        is_main_process=is_main_process,
    )

    model.eval()

    # Collect predictions
    all_kp_projs_detected = []
    all_kp_projs_gt = []
    all_kp_pos_gt = []
    all_n_inframe_projs_gt = []
    all_pred_3d = []  # Model's direct 3D predictions
    all_pred_3d_robot = []  # Model robot-frame 3D source (before PnP)
    all_pred_kp_conf = []  # Per-keypoint heatmap confidence (max over HxW)
    all_camera_Ks = []  # Per-sample camera intrinsics
    all_joint_angles_pred = []  # Joint angle predictions (if joint_angle mode)
    all_joint_angles_gt = []  # GT joint angles
    all_frame_names = []  # json stem per frame
    all_image_paths = []  # resolved image path per frame

    if is_main_process:
        if is_distributed:
            print(f"\nRunning distributed inference on {len(dataset)} images with world_size={world_size}...")
        else:
            print(f"\nRunning inference on {len(dataset)} images...")

    for batch in tqdm(dataloader, disable=not is_main_process):
        images = batch['image'].to(device)
        gt_keypoints = batch['keypoints'].numpy()
        gt_keypoints_3d = batch['keypoints_3d'].numpy()
        batch_camera_K = batch['camera_K'].to(device)  # (B, 3, 3) per-sample K
        batch_original_size = batch['original_size'].to(device)  # (B, 2)
        batch_original_size_np = batch_original_size.cpu().numpy()
        gt_angles = batch['angles'].numpy()  # (B, 9)
        if fix_joint7_zero and gt_angles.shape[1] >= 7:
            gt_angles = gt_angles.copy()
            gt_angles[:, 6] = 0.0
        batch_names = batch['name']
        batch_image_paths = batch['image_path']

        # Forward pass (match training-time forward inputs)
        outputs = model(
            images, camera_K=batch_camera_K, original_size=batch_original_size,
            use_refinement=use_iterative_refinement
        )
        pred_heatmaps = outputs["heatmaps_2d"]
        # Select robot-frame 3D source before camera-frame PnP transform.
        if args.pred_3d_source == 'fk':
            if fix_joint7_zero and "joint_angles" in outputs and outputs["joint_angles"] is not None:
                joint_angles_fk = _ensure_panda_angles_7(outputs["joint_angles"], fix_joint7_zero=True)
                pred_kpts_3d_tensor = panda_forward_kinematics(joint_angles_fk)
            else:
                pred_kpts_3d_tensor = outputs["keypoints_3d_fk"] if "keypoints_3d_fk" in outputs else outputs["keypoints_3d"]
        else:  # fused
            pred_kpts_3d_tensor = outputs["keypoints_3d"]
        pred_kpts_3d = pred_kpts_3d_tensor.cpu().numpy()  # (B, N_kp, 3)

        # Collect joint angles if available
        if "joint_angles" in outputs and outputs["joint_angles"] is not None:
            pred_angles = outputs["joint_angles"]
            if fix_joint7_zero and pred_angles.shape[1] >= 7:
                pred_angles = pred_angles.clone()
                pred_angles[:, 6] = 0.0
            all_joint_angles_pred.append(pred_angles.cpu().numpy())

        # Extract keypoints from heatmaps
        pred_keypoints, pred_kp_conf = get_keypoints_from_heatmaps(
            pred_heatmaps,
            min_confidence=args.kp_min_confidence,
            min_peak_logit=args.kp_min_peak_logit,
        )

        for i in range(len(pred_keypoints)):
            # Per-sample raw resolution and camera K
            sample_raw_w, sample_raw_h = batch_original_size_np[i].astype(int)
            sample_camera_K = batch_camera_K[i].cpu().numpy()

            # Scale to raw resolution
            scale_x = sample_raw_w / heatmap_size
            scale_y = sample_raw_h / heatmap_size

            # Scale predictions to raw resolution
            pred_kp_scaled = pred_keypoints[i].copy()
            pred_kp_scaled[:, 0] *= scale_x
            pred_kp_scaled[:, 1] *= scale_y

            # Transform 3D predictions from robot frame to camera frame via PnP
            pred_3d_robot_sample = pred_kpts_3d[i]
            pred_3d_sample = pred_3d_robot_sample
            if sample_camera_K is not None:
                valid_for_pnp = np.isfinite(pred_kp_scaled).all(axis=1) & (pred_kp_scaled[:, 0] > -900.0) & (pred_kp_scaled[:, 1] > -900.0)
                if np.count_nonzero(valid_for_pnp) >= 4 and (
                    not passes_pnp_spread_check(
                        pred_kp_scaled[valid_for_pnp],
                        (sample_raw_w, sample_raw_h),
                        args.pnp_min_span_px,
                        args.pnp_min_area_ratio,
                    )
                ):
                    pred_3d_camera, proj_2d_all = None, None
                else:
                    pred_3d_camera, proj_2d_all = transform_robot_to_camera(
                        pred_3d_robot_sample,
                        pred_kp_scaled,
                        sample_camera_K,
                        apply_z_search=(not args.disable_pnp_z_search),
                        z_search_min_m=args.pnp_z_search_min_m,
                        z_search_max_m=args.pnp_z_search_max_m,
                        z_search_step_m=args.pnp_z_search_step_m,
                    )
                if pred_3d_camera is not None:
                    pred_3d_sample = pred_3d_camera
                    if args.fill_invalid_2d_with_fk_reproj and proj_2d_all is not None:
                        low_conf_mask = pred_kp_conf[i] < float(args.kp_min_confidence)
                        low_reliability = low_conf_mask
                        valid_pred = np.isfinite(pred_kp_scaled).all(axis=1) & (pred_kp_scaled[:, 0] > -900.0) & (pred_kp_scaled[:, 1] > -900.0)
                        fill_mask = (~valid_pred) | low_reliability
                        pred_kp_scaled[fill_mask] = proj_2d_all[fill_mask]

            all_kp_projs_detected.append(pred_kp_scaled)
            all_kp_projs_gt.append(gt_keypoints[i])
            all_kp_pos_gt.append(gt_keypoints_3d[i])
            all_pred_3d.append(pred_3d_sample)
            all_pred_3d_robot.append(pred_3d_robot_sample)
            all_pred_kp_conf.append(pred_kp_conf[i])
            all_camera_Ks.append(sample_camera_K)
            all_joint_angles_gt.append(gt_angles[i])
            all_frame_names.append(str(batch_names[i]))
            all_image_paths.append(str(batch_image_paths[i]))

            # Count in-frame GT keypoints
            n_inframe = 0
            for kp in gt_keypoints[i]:
                if 0 <= kp[0] <= sample_raw_w and 0 <= kp[1] <= sample_raw_h:
                    n_inframe += 1
            all_n_inframe_projs_gt.append(n_inframe)

    local_payload = {
        'kp_detected': all_kp_projs_detected,
        'kp_gt': all_kp_projs_gt,
        'kp3d_gt': all_kp_pos_gt,
        'pred_3d': all_pred_3d,
        'pred_3d_robot': all_pred_3d_robot,
        'pred_kp_conf': all_pred_kp_conf,
        'camera_Ks': all_camera_Ks,
        'n_inframe': all_n_inframe_projs_gt,
        'angles_gt': all_joint_angles_gt,
        'angles_pred_chunks': all_joint_angles_pred,
        'frame_names': all_frame_names,
        'image_paths': all_image_paths,
    }

    if is_distributed:
        gathered_payloads = [None for _ in range(world_size)] if is_main_process else None
        dist.gather_object(local_payload, gathered_payloads, dst=0)
        dist.barrier()
        if not is_main_process:
            cleanup_distributed()
            return

        # Merge rank-local payloads on rank 0
        all_kp_projs_detected = []
        all_kp_projs_gt = []
        all_kp_pos_gt = []
        all_pred_3d = []
        all_pred_3d_robot = []
        all_pred_kp_conf = []
        all_camera_Ks = []
        all_n_inframe_projs_gt = []
        all_joint_angles_gt = []
        all_joint_angles_pred = []
        all_frame_names = []
        all_image_paths = []
        for payload in gathered_payloads:
            if payload is None:
                continue
            all_kp_projs_detected.extend(payload['kp_detected'])
            all_kp_projs_gt.extend(payload['kp_gt'])
            all_kp_pos_gt.extend(payload['kp3d_gt'])
            all_pred_3d.extend(payload['pred_3d'])
            all_pred_3d_robot.extend(payload.get('pred_3d_robot', []))
            all_pred_kp_conf.extend(payload.get('pred_kp_conf', []))
            all_camera_Ks.extend(payload['camera_Ks'])
            all_n_inframe_projs_gt.extend(payload['n_inframe'])
            all_joint_angles_gt.extend(payload['angles_gt'])
            all_joint_angles_pred.extend(payload['angles_pred_chunks'])
            all_frame_names.extend(payload.get('frame_names', []))
            all_image_paths.extend(payload.get('image_paths', []))

    all_kp_projs_detected = np.array(all_kp_projs_detected)
    all_kp_projs_gt = np.array(all_kp_projs_gt)
    all_kp_pos_gt = np.array(all_kp_pos_gt)
    all_pred_3d = np.array(all_pred_3d)
    all_pred_3d_robot = np.array(all_pred_3d_robot)
    all_pred_kp_conf = np.array(all_pred_kp_conf)
    all_joint_angles_gt = np.array(all_joint_angles_gt)
    if all_joint_angles_pred:
        all_joint_angles_pred = np.concatenate(all_joint_angles_pred, axis=0)
    else:
        all_joint_angles_pred = None

    # Compute keypoint metrics (2D)
    print("\nComputing keypoint metrics...")
    n_samples = len(all_kp_projs_detected)
    kp_metrics = compute_keypoint_metrics(
        all_kp_projs_detected.reshape(n_samples * len(keypoint_names), 2),
        all_kp_projs_gt.reshape(n_samples * len(keypoint_names), 2),
        raw_resolution,
        auc_threshold=args.kp_auc_threshold
    )

    # ===== Direct ADD: selected 3D source output vs GT 3D =====
    print(f"Computing Direct ADD metrics ({args.pred_3d_source} source vs GT 3D)...")
    direct_add_metrics = compute_direct_add_metrics(
        all_pred_3d,
        all_kp_pos_gt,
        all_kp_projs_gt,
        raw_resolution,
        add_auc_threshold=args.add_auc_threshold
    )

    # ===== PnP ADD (DREAM baseline): 2D pred + GT 3D + K -> PnP =====
    print("Computing PnP ADD metrics (DREAM baseline)...")
    pnp_adds = []

    for kp_det, kp_gt, kp_3d, n_inframe, sample_K in tqdm(
        zip(all_kp_projs_detected, all_kp_projs_gt, all_kp_pos_gt, all_n_inframe_projs_gt, all_camera_Ks),
        total=len(all_kp_projs_detected),
        desc="PnP solving"
    ):
        # DREAM-style filtering:
        # - valid predicted 2D
        # - GT keypoint in frame
        # - GT 3D not missing (not all zeros)
        valid_pred2d = np.isfinite(kp_det).all(axis=1) & (kp_det[:, 0] > -999.0) & (kp_det[:, 1] > -999.0)
        in_frame_gt2d = (
            np.isfinite(kp_gt).all(axis=1) &
            (kp_gt[:, 0] >= 0.0) & (kp_gt[:, 0] <= raw_resolution[0]) &
            (kp_gt[:, 1] >= 0.0) & (kp_gt[:, 1] <= raw_resolution[1])
        )
        valid_gt3d = np.isfinite(kp_3d).all(axis=1) & (np.linalg.norm(kp_3d, axis=1) > 1e-8)
        idx_good = np.where(valid_pred2d & in_frame_gt2d & valid_gt3d)[0]

        if len(idx_good) >= 4:  # Need at least 4 points for PnP
            kp_det_pnp = kp_det[idx_good]
            kp_3d_pnp = kp_3d[idx_good]
            if not passes_pnp_spread_check(
                kp_det_pnp, raw_resolution, args.pnp_min_span_px, args.pnp_min_area_ratio
            ):
                pnp_adds.append(-999.0)
                continue

            # Solve PnP using DREAM wrapper if available; otherwise OpenCV fallback.
            try:
                if DREAM_AVAILABLE:
                    success, translation, quaternion = dream.geometric_vision.solve_pnp(
                        kp_3d_pnp.astype(np.float64),
                        kp_det_pnp.astype(np.float64),
                        sample_K.astype(np.float64),
                    )

                    if success:
                        add = dream.geometric_vision.add_from_pose(
                            translation, quaternion, kp_3d_pnp, sample_K
                        )
                        pnp_adds.append(add)
                    else:
                        pnp_adds.append(-999.0)
                else:
                    success, rvec, tvec = solve_pnp_epnp_iterative(
                        kp_3d_pnp,
                        kp_det_pnp,
                        sample_K,
                    )
                    if success and rvec is not None and tvec is not None:
                        add = add_from_pose_rvec_tvec(rvec, tvec, kp_3d_pnp)
                        pnp_adds.append(add)
                    else:
                        pnp_adds.append(-999.0)
            except Exception:
                pnp_adds.append(-999.0)
        else:
            pnp_adds.append(-999.0)

    pnp_metrics = compute_pnp_metrics(
        pnp_adds,
        all_n_inframe_projs_gt,
        add_auc_threshold=args.add_auc_threshold
    )

    # ===== RoboPEPP-style PnP ADD: pred 2D + pred robot 3D + K =====
    print("Computing RoboPEPP-style PnP ADD metrics (pred 2D + pred robot 3D + K)...")
    robopepp_pnp_metrics, robopepp_pnp_adds = compute_robopepp_style_pnp_add_metrics(
        pred_2d_all=all_kp_projs_detected,
        pred_3d_robot_all=all_pred_3d_robot,
        gt_3d_all=all_kp_pos_gt,
        gt_2d_all=all_kp_projs_gt,
        camera_Ks=np.array(all_camera_Ks),
        image_resolution=raw_resolution,
        num_inframe_projs_gt=all_n_inframe_projs_gt,
        pred_conf_all=all_pred_kp_conf,
        add_auc_threshold=args.add_auc_threshold,
        init_conf_thresh=args.robopepp_pnp_init_thresh,
        conf_step=args.robopepp_pnp_conf_step,
        pnp_min_span_px=args.pnp_min_span_px,
        pnp_min_area_ratio=args.pnp_min_area_ratio,
        apply_z_search=(not args.disable_pnp_z_search),
        z_search_min_m=args.pnp_z_search_min_m,
        z_search_max_m=args.pnp_z_search_max_m,
        z_search_step_m=args.pnp_z_search_step_m,
        return_raw_adds=True,
    )

    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    print(f"\nDataset: {args.dataset_dir}")
    print(f"Model: {args.model_path}")
    print(f"3D Prediction Mode: joint_angle/{args.pred_3d_source} (robot-frame FK/source -> camera-frame via PnP)")
    print(f"Number of frames: {n_samples}")

    # 2D Keypoint metrics
    print(f"\n# L2 error (px) for in-frame keypoints (n = {kp_metrics['num_gt_inframe']}):")
    if kp_metrics['l2_error_auc'] is not None:
        print(f"#    AUC: {kp_metrics['l2_error_auc']:.5f}")
        print(f"#       AUC threshold: {kp_metrics['l2_error_auc_thresh_px']:.5f}")
        print(f"#    Mean: {kp_metrics['l2_error_mean_px']:.5f}")
        print(f"#    Median: {kp_metrics['l2_error_median_px']:.5f}")
        print(f"#    Std Dev: {kp_metrics['l2_error_std_px']:.5f}")
        print(f"#    PCK@2.5px: {kp_metrics['pck@2.5px_percent']:.2f}%")
        print(f"#    PCK@5px:   {kp_metrics['pck@5px_percent']:.2f}%")
        print(f"#    PCK@10px:  {kp_metrics['pck@10px_percent']:.2f}%")
    else:
        print("#    No valid keypoints found")

    # Direct ADD (selected 3D source)
    print(f"\n# Direct ADD (m, camera-frame) - {args.pred_3d_source} source (robot->camera transformed) vs GT 3D (n = {direct_add_metrics['num_frames']}):")
    if direct_add_metrics['add_auc'] is not None:
        print(f"#    AUC: {direct_add_metrics['add_auc']:.5f}")
        print(f"#       AUC threshold: {direct_add_metrics['add_auc_thresh']:.5f}")
        print(f"#    Mean: {direct_add_metrics['add_mean']:.5f}")
        print(f"#    Median: {direct_add_metrics['add_median']:.5f}")
        print(f"#    Std Dev: {direct_add_metrics['add_std']:.5f}")
        print(f"#    Per-keypoint Mean: {direct_add_metrics['per_kp_mean']:.5f}")
        print(f"#    Per-keypoint Median: {direct_add_metrics['per_kp_median']:.5f}")
    else:
        print("#    No valid frames for direct ADD")

    # PnP ADD (DREAM baseline)
    print(f"\n# PnP ADD (m, camera-frame) - DREAM baseline: 2D pred + GT_3D + K (n = {pnp_metrics['num_pnp_found']}):")
    if pnp_metrics['num_pnp_found'] > 0:
        print(f"#    AUC: {pnp_metrics['add_auc']:.5f}")
        print(f"#       AUC threshold: {pnp_metrics['add_auc_thresh']:.5f}")
        print(f"#    Mean: {pnp_metrics['add_mean']:.5f}")
        print(f"#    Median: {pnp_metrics['add_median']:.5f}")
        print(f"#    Std Dev: {pnp_metrics['add_std']:.5f}")
        print(f"#    PnP Success Rate: {pnp_metrics['num_pnp_found']}/{pnp_metrics['num_pnp_possible']} ({pnp_metrics['num_pnp_found']/max(1, pnp_metrics['num_pnp_possible'])*100:.1f}%)")
    else:
        print("#    No successful PnP solutions")

    # RoboPEPP-style PnP ADD
    print(f"\n# PnP ADD (m, camera-frame) - RoboPEPP-style: 2D pred + pred_3D_robot + K (n = {robopepp_pnp_metrics['num_pnp_found']}):")
    if robopepp_pnp_metrics['num_pnp_found'] > 0:
        print(f"#    AUC: {robopepp_pnp_metrics['add_auc']:.5f}")
        print(f"#       AUC threshold: {robopepp_pnp_metrics['add_auc_thresh']:.5f}")
        print(f"#    Mean: {robopepp_pnp_metrics['add_mean']:.5f}")
        print(f"#    Median: {robopepp_pnp_metrics['add_median']:.5f}")
        print(f"#    Std Dev: {robopepp_pnp_metrics['add_std']:.5f}")
        print(f"#    PnP Success Rate: {robopepp_pnp_metrics['num_pnp_found']}/{robopepp_pnp_metrics['num_pnp_possible']} ({robopepp_pnp_metrics['num_pnp_found']/max(1, robopepp_pnp_metrics['num_pnp_possible'])*100:.1f}%)")
    else:
        print("#    No successful PnP solutions")

    # Joint angle metrics (if joint_angle mode)
    joint_angle_metrics = {}
    if all_joint_angles_pred is not None:
        # Compare predicted vs GT joint angles.
        n_angle_eval = 6 if fix_joint7_zero else 7
        gt_angles_7 = all_joint_angles_gt[:, :n_angle_eval]
        pred_angles_7 = all_joint_angles_pred[:, :n_angle_eval] if all_joint_angles_pred.shape[1] >= n_angle_eval else all_joint_angles_pred

        # Only compare where GT angles are not all zero
        valid_mask = np.any(gt_angles_7 != 0, axis=-1)
        if valid_mask.sum() > 0:
            gt_valid = gt_angles_7[valid_mask]
            pred_valid = pred_angles_7[valid_mask]
            angle_errors = np.abs(pred_valid - gt_valid)  # (N_valid, 7)
            angle_errors_deg = np.degrees(angle_errors)

            joint_angle_metrics = {
                'num_frames': int(valid_mask.sum()),
                'mean_rad': float(np.mean(angle_errors)),
                'mean_deg': float(np.mean(angle_errors_deg)),
                'per_joint_mean_deg': [float(x) for x in np.mean(angle_errors_deg, axis=0)],
            }

            print(f"\n# Joint Angle Error (n = {joint_angle_metrics['num_frames']}):")
            print(f"#    Mean: {joint_angle_metrics['mean_rad']:.4f} rad ({joint_angle_metrics['mean_deg']:.2f} deg)")
            joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7'][:n_angle_eval]
            for j, (jname, jerr) in enumerate(zip(joint_names, joint_angle_metrics['per_joint_mean_deg'])):
                print(f"#      {jname}: {jerr:.2f} deg")

    print("=" * 80)

    # Save results
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Optional: save AUC/PCK plots.
        if args.save_metric_plots:
            kp_l2_errors, kp_denom = collect_keypoint_l2_errors(
                all_kp_projs_detected.reshape(n_samples * len(keypoint_names), 2),
                all_kp_projs_gt.reshape(n_samples * len(keypoint_names), 2),
                raw_resolution,
            )
            direct_add_values = collect_direct_add_values(
                all_pred_3d,
                all_kp_pos_gt,
                all_kp_projs_gt,
                raw_resolution,
            )
            metric_plot_paths = save_metric_plots(
                output_dir=output_dir,
                kp_errors=kp_l2_errors,
                kp_denom=kp_denom,
                kp_auc_threshold=args.kp_auc_threshold,
                pnp_adds_dream=pnp_adds,
                pnp_adds_robopepp=robopepp_pnp_adds,
                num_pnp_possible=int(pnp_metrics.get('num_pnp_possible', 0)),
                direct_adds=direct_add_values,
                add_auc_threshold=args.add_auc_threshold,
                pred_3d_source=args.pred_3d_source,
            )
            if metric_plot_paths:
                print("Saved metric plots:")
                for p in metric_plot_paths:
                    print(f"  - {p}")

        results = {
            'dataset': str(args.dataset_dir),
            'model': str(args.model_path),
            'num_frames': n_samples,
            'keypoint_metrics': {k: float(v) if v is not None else None for k, v in kp_metrics.items()},
            'direct_add_metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else v for k, v in direct_add_metrics.items()},
            'pnp_metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else v for k, v in pnp_metrics.items()},
            'robopepp_pnp_metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else v for k, v in robopepp_pnp_metrics.items()},
        }

        results_path = output_dir / 'eval_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)

        print(f"\nResults saved to {results_path}")

        # Optional: save per-frame 3D error report for outlier analysis
        if args.save_per_frame_errors:
            valid_mask = (
                (all_kp_projs_gt[:, :, 0] > -900.0) & (all_kp_projs_gt[:, :, 1] > -900.0) &
                (all_kp_projs_gt[:, :, 0] >= 0.0) & (all_kp_projs_gt[:, :, 0] <= raw_resolution[0]) &
                (all_kp_projs_gt[:, :, 1] >= 0.0) & (all_kp_projs_gt[:, :, 1] <= raw_resolution[1])
            )
            per_kp_err = np.linalg.norm(all_pred_3d - all_kp_pos_gt, axis=2)  # (N, num_kp)

            frame_reports = []
            for i in range(n_samples):
                valid_i = valid_mask[i]
                errs_i = per_kp_err[i]
                valid_errs_i = errs_i[valid_i]
                if valid_errs_i.size > 0:
                    mean_err = float(np.mean(valid_errs_i))
                    median_err = float(np.median(valid_errs_i))
                    max_err = float(np.max(valid_errs_i))
                    valid_idx = np.where(valid_i)[0]
                    worst_local = int(np.argmax(valid_errs_i))
                    worst_kp_idx = int(valid_idx[worst_local])
                    worst_kp_name = keypoint_names[worst_kp_idx]
                else:
                    mean_err = float('nan')
                    median_err = float('nan')
                    max_err = float('nan')
                    worst_kp_idx = -1
                    worst_kp_name = None

                per_kp_dict = {}
                for k_idx, k_name in enumerate(keypoint_names):
                    per_kp_dict[k_name] = float(errs_i[k_idx]) if valid_i[k_idx] else None

                frame_name = all_frame_names[i] if i < len(all_frame_names) else f"{i:06d}"
                frame_reports.append({
                    'frame_index': int(i),
                    'json_name': frame_name,
                    'json_path': str(Path(args.dataset_dir) / f"{frame_name}.json"),
                    'image_path': all_image_paths[i] if i < len(all_image_paths) else None,
                    'valid_keypoint_count': int(valid_i.sum()),
                    'mean_3d_error_m': mean_err,
                    'median_3d_error_m': median_err,
                    'max_3d_error_m': max_err,
                    'worst_keypoint_name': worst_kp_name,
                    'worst_keypoint_index': worst_kp_idx,
                    'per_keypoint_error_m': per_kp_dict,
                })

            # Sort by mean error (descending), NaN goes last
            frame_reports_sorted = sorted(
                frame_reports,
                key=lambda x: (np.isnan(x['mean_3d_error_m']), -x['mean_3d_error_m'] if not np.isnan(x['mean_3d_error_m']) else 0.0)
            )

            topk = max(1, int(args.outlier_topk))
            topk_reports = frame_reports_sorted[:topk]

            # Per-keypoint summary across valid points
            kp_summary = {}
            for k_idx, k_name in enumerate(keypoint_names):
                kp_valid = valid_mask[:, k_idx]
                kp_errs = per_kp_err[:, k_idx][kp_valid]
                if kp_errs.size > 0:
                    kp_summary[k_name] = {
                        'count': int(kp_errs.size),
                        'mean_m': float(np.mean(kp_errs)),
                        'median_m': float(np.median(kp_errs)),
                        'p90_m': float(np.percentile(kp_errs, 90)),
                        'p95_m': float(np.percentile(kp_errs, 95)),
                        'max_m': float(np.max(kp_errs)),
                    }
                else:
                    kp_summary[k_name] = {
                        'count': 0,
                        'mean_m': None,
                        'median_m': None,
                        'p90_m': None,
                        'p95_m': None,
                        'max_m': None,
                    }

            per_frame_path = output_dir / 'per_frame_3d_errors.json'
            outlier_topk_path = output_dir / 'outlier_topk_3d_errors.json'
            kp_summary_path = output_dir / 'per_keypoint_3d_error_summary.json'
            outlier_topk_json_names_txt = output_dir / 'outlier_topk_json_names.txt'
            outlier_topk_json_paths_txt = output_dir / 'outlier_topk_json_paths.txt'

            with open(per_frame_path, 'w') as f:
                json.dump(frame_reports_sorted, f, indent=2)
            with open(outlier_topk_path, 'w') as f:
                json.dump(topk_reports, f, indent=2)
            with open(kp_summary_path, 'w') as f:
                json.dump(kp_summary, f, indent=2)
            with open(outlier_topk_json_names_txt, 'w') as f:
                for item in topk_reports:
                    name = item.get('json_name')
                    if name:
                        f.write(f"{name}\n")
            with open(outlier_topk_json_paths_txt, 'w') as f:
                for item in topk_reports:
                    path = item.get('json_path')
                    if path:
                        f.write(f"{path}\n")

            print(f"Per-frame 3D error report saved to {per_frame_path}")
            print(f"Top-{topk} outlier report saved to {outlier_topk_path}")
            print(f"Per-keypoint 3D summary saved to {kp_summary_path}")
            print(f"Top-{topk} json-name list saved to {outlier_topk_json_names_txt}")
            print(f"Top-{topk} json-path list saved to {outlier_topk_json_paths_txt}")

    if is_distributed:
        cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description='Inference on Dataset with DREAM-style Metrics')

    # Model
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--model-name', type=str, default=None,
                        help='DINOv3 model name (auto-read from config.yaml if not specified)')
    parser.add_argument('--image-size', type=int, default=None,
                        help='Input image size (auto-read from config.yaml if not specified)')
    parser.add_argument('--heatmap-size', type=int, default=None,
                        help='Output heatmap size (auto-read from config.yaml if not specified)')

    # Dataset
    parser.add_argument('--dataset-dir', type=str, required=True,
                        help='Path to NDDS dataset directory')

    # Inference
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for inference')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--distributed', action='store_true', default=False,
                        help='Enable distributed inference (launch with torchrun)')
    parser.add_argument('--pred-3d-source', type=str, default='fk', choices=['fk', 'fused'],
                        help='Robot-frame 3D source before PnP transform: fk (recommended) or fused')
    parser.add_argument('--fix-joint7-zero', action='store_true', default=False,
                        help='RoboPEPP-style setting: force joint7=0 for FK and angle metrics')

    # Metrics
    parser.add_argument('--kp-auc-threshold', type=float, default=20.0,
                        help='AUC threshold for keypoint L2 error (pixels)')
    parser.add_argument('--add-auc-threshold', type=float, default=0.1,
                        help='AUC threshold for ADD metric (meters)')

    # Output
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results')
    parser.add_argument('--save-metric-plots', action='store_true', default=False,
                        help='Save PCK/AUC and ADD/AUC plots into output directory')
    parser.add_argument('--save-per-frame-errors', action='store_true', default=False,
                        help='Save per-frame 3D error report and top-k outliers')
    parser.add_argument('--outlier-topk', type=int, default=100,
                        help='Number of worst frames to save in outlier report')
    parser.add_argument('--robopepp-pnp-init-thresh', type=float, default=0.25,
                        help='Initial heatmap-confidence threshold for RoboPEPP-style PnP keypoint selection')
    parser.add_argument('--robopepp-pnp-conf-step', type=float, default=0.025,
                        help='Threshold decrement step when not enough keypoints for RoboPEPP-style PnP')
    parser.add_argument('--pnp-min-span-px', type=float, default=20.0,
                        help='Minimum x/y span (px) of selected 2D keypoints required for PnP')
    parser.add_argument('--pnp-min-area-ratio', type=float, default=0.001,
                        help='Minimum 2D bbox area ratio of selected points for PnP')
    parser.add_argument('--kp-min-confidence', type=float, default=0.0,
                        help='Mask predicted 2D keypoints when sigmoid(max_heatmap_logit) is below this threshold')
    parser.add_argument('--kp-min-peak-logit', type=float, default=-1e9,
                        help='Mask predicted 2D keypoints when heatmap peak logit is below this threshold')
    parser.add_argument('--fill-invalid-2d-with-fk-reproj', action='store_true',
                        help='After successful PnP, fill invalid/low-reliability 2D keypoints using FK reprojection')
    parser.add_argument('--disable-pnp-z-search', action='store_true',
                        help='Disable test-time Z-translation grid search after PnP')
    parser.add_argument('--pnp-z-search-min-m', type=float, default=-0.05,
                        help='Minimum Z offset (m) for post-PnP translation search')
    parser.add_argument('--pnp-z-search-max-m', type=float, default=0.05,
                        help='Maximum Z offset (m) for post-PnP translation search')
    parser.add_argument('--pnp-z-search-step-m', type=float, default=0.001,
                        help='Step size (m) for post-PnP Z translation search')

    args = parser.parse_args()

    run_inference(args)


if __name__ == '__main__':
    main()
