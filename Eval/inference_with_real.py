"""
Real Image Inference Script for DINOv3 Pose Estimation
- JSON 어노테이션 파일을 입력받아 이미지 경로/GT를 자동 로드
- GT vs Prediction 비교 시각화 + 정량적 메트릭 출력
"""

import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from PIL import Image as PILImage

import numpy as np
import torch
import torchvision.transforms as TVTransforms
import yaml
import cv2

# Import DREAM utilities
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../DREAM')))
import dream

# Import model from Train directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../TRAIN')))
from model import DINOv3PoseEstimator
from checkpoint_compat import load_checkpoint_compat


def _resolve_robopepp_urdf(explicit_urdf_path=None):
    if explicit_urdf_path:
        return explicit_urdf_path
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../RoboPEPP/urdfs/Panda/panda.urdf")
    )


def _load_robopepp_panda_fk_class():
    module_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../RoboPEPP/models/robot_arm.py")
    )
    spec = importlib.util.spec_from_file_location("robopepp_robot_arm", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load RoboPEPP FK module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PandaArmPytorch


def _compute_robopepp_fk_keypoints(joint_angles, device, urdf_file, fix_joint7_zero=False):
    PandaArmPytorch = _load_robopepp_panda_fk_class()
    robot = PandaArmPytorch(urdf_file, device=str(device))
    if joint_angles.shape[1] >= 7:
        joint_angles_fk = joint_angles[:, :7].clone()
    else:
        pad = torch.zeros(
            (joint_angles.shape[0], 7 - joint_angles.shape[1]),
            device=joint_angles.device,
            dtype=joint_angles.dtype,
        )
        joint_angles_fk = torch.cat([joint_angles, pad], dim=1)
    if fix_joint7_zero and joint_angles_fk.shape[1] >= 7:
        # Match RoboPEPP eval convention: use 6 predicted joints and fix joint7 to zero.
        joint_angles_fk[:, 6] = 0.0
    # Match RoboPEPP/test.py keypoint subset: link0,2,3,4,6,7,hand
    _, keypoints_3d = robot.get_joint_RT(joint_angles_fk)
    return keypoints_3d[:, [0, 2, 3, 4, 6, 7, 8], :]


def get_keypoints_from_heatmaps(heatmaps_tensor, min_confidence=0.0, min_peak_logit=None):
    """Extract keypoint coordinates from heatmaps and optionally mask low confidence."""
    B, N, H, W = heatmaps_tensor.shape
    heatmaps_flat = heatmaps_tensor.view(B, N, -1)
    max_indices = torch.argmax(heatmaps_flat, dim=-1)
    y = max_indices // W
    x = max_indices % W
    keypoints = torch.stack([x, y], dim=-1).float()  # (B, N, 2)

    # Convert max logit to probability-like confidence in [0, 1].
    peak_logits = heatmaps_flat.amax(dim=-1)  # (B, N)
    confidences = torch.sigmoid(peak_logits)  # (B, N)
    invalid = torch.zeros_like(confidences, dtype=torch.bool)
    if min_confidence > 0.0:
        invalid = invalid | (confidences < float(min_confidence))
    if min_peak_logit is not None:
        invalid = invalid | (peak_logits < float(min_peak_logit))
    if invalid.any():
        keypoints = keypoints.masked_fill(invalid.unsqueeze(-1), -999.0)

    return (
        keypoints[0].cpu().numpy(),
        confidences[0].cpu().numpy(),
        peak_logits[0].cpu().numpy(),
    )


def transform_robot_to_camera(
    robot_kpts,
    pred_2d,
    camera_K,
    confidences=None,
    topk=6,
    min_points=4,
    ransac_reproj_error=5.0,
    pnp_mode="epnp",
    reproj_outlier_thresh_px=12.0,
    reproj_refit=True,
    min_span_px=20.0,
    min_area_ratio=0.001,
):
    """
    Transform robot frame keypoints to camera frame using PnP.

    Args:
        robot_kpts: (N, 3) keypoints in robot frame
        pred_2d: (N, 2) predicted 2D keypoints in image
        camera_K: (3, 3) camera intrinsic matrix

    Returns:
        camera_kpts: (N, 3) keypoints in camera frame (or None if PnP fails)
        info: dict with selected/ignored index diagnostics
    """
    info = {
        "idx_valid": np.array([], dtype=np.int64),
        "idx_selected_initial": np.array([], dtype=np.int64),
        "idx_selected_final": np.array([], dtype=np.int64),
        "idx_removed_reproj": np.array([], dtype=np.int64),
        "reproj_errors_px": np.array([], dtype=np.float64),
        "inlier_count": 0,
        "k_used": 0,
        "refit_applied": False,
        "spread_ok": True,
        "spread_span_xy_px": [0.0, 0.0],
        "spread_area_ratio": 0.0,
        "proj_2d_all": None,
    }
    try:
        # PnP requires at least 4 points
        if len(robot_kpts) < min_points:
            print(f"Warning: PnP requires at least 4 points, got {len(robot_kpts)}")
            return None, info

        # Ensure correct data types
        robot_kpts = robot_kpts.astype(np.float64)
        pred_2d = pred_2d.astype(np.float64)
        camera_K = camera_K.astype(np.float64)
        if confidences is None:
            confidences = np.ones((len(robot_kpts),), dtype=np.float64)
        else:
            confidences = np.asarray(confidences, dtype=np.float64)

        valid = (
            np.isfinite(robot_kpts).all(axis=1)
            & np.isfinite(pred_2d).all(axis=1)
            & (pred_2d[:, 0] > -900.0)
            & (pred_2d[:, 1] > -900.0)
        )
        idx_valid = np.where(valid)[0]
        info["idx_valid"] = idx_valid
        if idx_valid.shape[0] < min_points:
            print("Warning: Not enough valid keypoints for PnP")
            return None, info

        # Baseline uses all valid points; robust modes can use top-k.
        if pnp_mode == "epnp":
            idx_sel = idx_valid
        else:
            idx_sorted = idx_valid[np.argsort(confidences[idx_valid])[::-1]]
            k = max(min_points, min(int(topk), idx_sorted.shape[0]))
            idx_sel = idx_sorted[:k]
        info["idx_selected_initial"] = idx_sel.copy()
        robot_sel = robot_kpts[idx_sel]
        pred2d_sel = pred_2d[idx_sel]

        # Spatial spread check: reject degenerate PnP sets concentrated in a tiny image region.
        x_span = float(np.max(pred2d_sel[:, 0]) - np.min(pred2d_sel[:, 0]))
        y_span = float(np.max(pred2d_sel[:, 1]) - np.min(pred2d_sel[:, 1]))
        bbox_area = max(0.0, x_span) * max(0.0, y_span)
        img_area = max(1.0, float(camera_K[0, 2] * 2.0) * float(camera_K[1, 2] * 2.0))
        area_ratio = bbox_area / img_area
        info["spread_span_xy_px"] = [x_span, y_span]
        info["spread_area_ratio"] = area_ratio
        if x_span < float(min_span_px) or y_span < float(min_span_px) or area_ratio < float(min_area_ratio):
            info["spread_ok"] = False
            print(
                f"Warning: PnP rejected by spatial spread check "
                f"(span=({x_span:.1f},{y_span:.1f})px, area_ratio={area_ratio:.5f})"
            )
            return None, info

        def solve_epnp(obj_pts, img_pts):
            return cv2.solvePnP(
                obj_pts, img_pts, camera_K, None, flags=cv2.SOLVEPNP_EPNP
            )

        def reproj_mse(rvec_in, tvec_in, obj_pts_all, img_pts_all):
            proj, _ = cv2.projectPoints(obj_pts_all, rvec_in, tvec_in, camera_K, None)
            proj = proj.reshape(-1, 2)
            err = np.linalg.norm(proj - img_pts_all, axis=1)
            return float(np.mean(err))

        if pnp_mode == "ransac":
            # First try RANSAC for robustness under occlusion/outliers.
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                robot_sel,
                pred2d_sel,
                camera_K,
                None,  # No distortion
                reprojectionError=float(ransac_reproj_error),
                flags=cv2.SOLVEPNP_EPNP,
            )
            # Fallback: plain EPnP on selected points.
            if not success:
                success, rvec, tvec = cv2.solvePnP(
                    robot_sel,
                    pred2d_sel,
                    camera_K,
                    None,
                    flags=cv2.SOLVEPNP_EPNP
                )
                inlier_count = 0
            else:
                inlier_count = 0 if inliers is None else len(inliers)
            k_used = len(idx_sel)
        elif pnp_mode == "loo_epnp":
            # Leave-one-out EPnP: fit multiple hypotheses and pick lowest reprojection error.
            if len(idx_sel) < min_points:
                print("Warning: Not enough selected keypoints for loo_epnp")
                return None, info
            best = None
            subsets = [idx_sel]
            if len(idx_sel) > min_points:
                for drop_i in range(len(idx_sel)):
                    sub = np.delete(idx_sel, drop_i)
                    if len(sub) >= min_points:
                        subsets.append(sub)
            for sub in subsets:
                ok, rv, tv = solve_epnp(robot_kpts[sub], pred_2d[sub])
                if not ok:
                    continue
                mse = reproj_mse(rv, tv, robot_kpts[idx_sel], pred_2d[idx_sel])
                if best is None or mse < best[0]:
                    best = (mse, rv, tv)
            if best is None:
                print("Warning: loo_epnp failed to find solution")
                return None, info
            success, rvec, tvec = True, best[1], best[2]
            inlier_count = 0
            k_used = len(idx_sel)
        else:  # epnp
            success, rvec, tvec = solve_epnp(robot_sel, pred2d_sel)
            inlier_count = 0
            k_used = len(idx_sel)

        if not success:
            print("Warning: PnP failed to find solution")
            return None, info

        # Reprojection outlier rejection + refit:
        # drop points with large reprojection residual, then solve again.
        idx_final = idx_sel.copy()
        if reproj_refit and idx_sel.shape[0] >= min_points:
            proj_sel, _ = cv2.projectPoints(robot_sel, rvec, tvec, camera_K, None)
            proj_sel = proj_sel.reshape(-1, 2)
            reproj_err = np.linalg.norm(proj_sel - pred2d_sel, axis=1)
            keep_mask = reproj_err <= float(reproj_outlier_thresh_px)
            removed = idx_sel[~keep_mask]
            if np.count_nonzero(keep_mask) >= min_points and removed.shape[0] > 0:
                idx_final = idx_sel[keep_mask]
                success2, rvec2, tvec2 = solve_epnp(robot_kpts[idx_final], pred_2d[idx_final])
                if success2:
                    rvec, tvec = rvec2, tvec2
                    info["refit_applied"] = True
                # If refit fails, keep original estimate but still report removed points.
            info["idx_removed_reproj"] = removed
            info["reproj_errors_px"] = reproj_err
        else:
            info["reproj_errors_px"] = np.array([], dtype=np.float64)

        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.flatten()

        # Transform: camera_frame = R @ robot_frame + t
        camera_kpts = (R @ robot_kpts.T).T + t.reshape(1, 3)
        proj_all, _ = cv2.projectPoints(robot_kpts, rvec, tvec, camera_K, None)
        info["proj_2d_all"] = proj_all.reshape(-1, 2)
        info["idx_selected_final"] = idx_final
        info["inlier_count"] = int(inlier_count)
        info["k_used"] = int(len(idx_final))
        print(f"# PnP mode={pnp_mode}, used {len(idx_final)} points, inliers={inlier_count}")
        if reproj_refit and info["idx_removed_reproj"].shape[0] > 0:
            print(
                f"# PnP reprojection outlier rejection: removed {info['idx_removed_reproj'].shape[0]} points "
                f"(thresh={reproj_outlier_thresh_px:.1f}px), final used={len(idx_final)}"
            )

        return camera_kpts, info

    except Exception as e:
        print(f"Warning: PnP failed with error: {e}")
        return None, info


def load_annotation(json_path, keypoint_names):
    """
    JSON 어노테이션에서 GT 정보를 로드.
    Returns:
        image_path: 이미지 절대경로
        gt_2d: (N, 2) projected keypoint 좌표
        gt_3d: (N, 3) 3D keypoint 좌표
        camera_K: (3, 3) camera intrinsic matrix (있으면)
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Extract image path
    image_path = None
    camera_K = None
    if 'meta' in data:
        if 'image_path' in data['meta']:
            raw_path = data['meta']['image_path']
            # Fix relative path: ../dataset/... -> resolve from JSON dir
            if raw_path.startswith('../dataset/'):
                raw_path = raw_path.replace('../dataset/', '../../../', 1)
            if not os.path.isabs(raw_path):
                image_path = os.path.normpath(os.path.join(os.path.dirname(json_path), raw_path))
            else:
                image_path = raw_path
        if 'K' in data['meta']:
            camera_K = np.array(data['meta']['K'], dtype=np.float64)

    # Extract keypoints
    gt_2d = np.zeros((len(keypoint_names), 2), dtype=np.float32)
    gt_3d = np.zeros((len(keypoint_names), 3), dtype=np.float32)
    found = [False] * len(keypoint_names)

    if 'objects' in data:
        for obj in data['objects']:
            if 'keypoints' in obj:
                for kp in obj['keypoints']:
                    if kp['name'] in keypoint_names:
                        idx = keypoint_names.index(kp['name'])
                        gt_2d[idx] = kp['projected_location']
                        if 'location' in kp:
                            gt_3d[idx] = kp['location']
                        found[idx] = True

    # SYN data 3D GT is in cm -> convert to meters
    is_synthetic = 'syn' in json_path.lower()
    if is_synthetic:
        gt_3d = gt_3d / 100.0

    # Extract GT joint angles from sim_state
    gt_angles = None
    if 'sim_state' in data and 'joints' in data['sim_state']:
        joints = data['sim_state']['joints']
        gt_angles = np.array([j['position'] for j in joints[:7]], dtype=np.float32)

    return image_path, gt_2d, gt_3d, camera_K, found, gt_angles


def compute_metrics(pred_2d, gt_2d, pred_3d, gt_3d, keypoint_names, found, orig_image_dim):
    """Compute per-keypoint and overall error metrics."""
    metrics = {}
    img_w, img_h = orig_image_dim

    # 2D pixel error (L2 distance)
    errors_2d = []
    for i, name in enumerate(keypoint_names):
        in_frame = (
            found[i] and
            (0.0 <= gt_2d[i][0] <= img_w) and
            (0.0 <= gt_2d[i][1] <= img_h)
        )
        if in_frame:
            pred_valid = np.isfinite(pred_2d[i]).all() and (pred_2d[i][0] > -900.0) and (pred_2d[i][1] > -900.0)
            if not pred_valid:
                metrics[f'{name}_2d_px'] = float('nan')
                continue
            err = np.linalg.norm(pred_2d[i] - gt_2d[i])
            errors_2d.append(err)
            metrics[f'{name}_2d_px'] = err
        else:
            metrics[f'{name}_2d_px'] = float('nan')

    if errors_2d:
        metrics['mean_2d_px'] = np.mean(errors_2d)
        metrics['max_2d_px'] = np.max(errors_2d)
        metrics['median_2d_px'] = np.median(errors_2d)

    # 3D Euclidean error (meters)
    errors_3d = []
    for i, name in enumerate(keypoint_names):
        in_frame = (
            found[i] and
            (0.0 <= gt_2d[i][0] <= img_w) and
            (0.0 <= gt_2d[i][1] <= img_h)
        )
        if in_frame and not np.allclose(gt_3d[i], 0):
            pred2d_valid = np.isfinite(pred_2d[i]).all() and (pred_2d[i][0] > -900.0) and (pred_2d[i][1] > -900.0)
            if not pred2d_valid:
                metrics[f'{name}_3d_m'] = float('nan')
                continue
            err = np.linalg.norm(pred_3d[i] - gt_3d[i])
            errors_3d.append(err)
            metrics[f'{name}_3d_m'] = err
        else:
            metrics[f'{name}_3d_m'] = float('nan')

    if errors_3d:
        metrics['mean_3d_m'] = np.mean(errors_3d)
        metrics['max_3d_m'] = np.max(errors_3d)
        metrics['median_3d_m'] = np.median(errors_3d)

    # Normalized 2D error (% of image diagonal)
    diag = np.sqrt(orig_image_dim[0]**2 + orig_image_dim[1]**2)
    if errors_2d and diag > 0:
        metrics['mean_2d_norm'] = np.mean(errors_2d) / diag * 100  # percentage

    return metrics


def network_inference(args):

    assert os.path.exists(args.json_path), \
        f'JSON path "{args.json_path}" does not exist.'
    assert args.json_path.lower().endswith('.json'), \
        f'--json-path must be a .json annotation file, got "{args.json_path}"'
    assert os.path.exists(args.model_path), \
        f'Model path "{args.model_path}" does not exist.'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"# Using device: {device}")

    # Default keypoint names
    keypoint_names = [
        'panda_link0', 'panda_link2', 'panda_link3',
        'panda_link4', 'panda_link6', 'panda_link7', 'panda_hand'
    ]

    # Load training config
    checkpoint_dir = Path(args.model_path).parent
    config_path = checkpoint_dir / 'config.yaml'

    use_joint_embedding = False
    use_iterative_refinement = False
    refinement_iterations = 3
    fix_joint7_zero = False
    model_name = args.model_name
    image_size = 512
    heatmap_size = 512
    urdf_file = args.robopepp_urdf

    if config_path.exists():
        with open(config_path, 'r') as f:
            train_config = yaml.safe_load(f)
        use_joint_embedding = train_config.get('use_joint_embedding', False)
        use_iterative_refinement = train_config.get('use_iterative_refinement', False)
        refinement_iterations = int(train_config.get('refinement_iterations', 3))
        fix_joint7_zero = bool(train_config.get('fix_joint7_zero', False))
        model_name = train_config.get('model_name', model_name)
        image_size = int(train_config.get('image_size', image_size))
        heatmap_size = int(train_config.get('heatmap_size', heatmap_size))
        if urdf_file is None:
            if isinstance(train_config.get('model'), dict):
                urdf_file = train_config['model'].get('urdf_file')
            if urdf_file is None:
                urdf_file = train_config.get('urdf_file')
            if urdf_file is not None and not os.path.isabs(urdf_file):
                urdf_file = os.path.abspath(checkpoint_dir / urdf_file)
        if 'keypoint_names' in train_config:
            keypoint_names = train_config['keypoint_names']
        print(
            f"# Config: use_joint_embedding={use_joint_embedding}, "
            f"iterative_refinement={use_iterative_refinement}, "
            f"fix_joint7_zero={fix_joint7_zero}"
        )

    # Load annotation JSON
    print(f"\n# Loading annotation: {args.json_path}")
    image_path, gt_2d, gt_3d, camera_K, found, gt_angles = load_annotation(args.json_path, keypoint_names)

    if image_path is None or not os.path.exists(image_path):
        print(f"# ERROR: Image not found at resolved path: {image_path}")
        print(f"# Check 'meta.image_path' in JSON and relative path resolution")
        return

    print(f"# Image path: {image_path}")
    print(f"# GT keypoints found: {sum(found)}/{len(found)}")
    if camera_K is not None:
        print(f"# Camera K:\n{camera_K}")

    # Create model
    model = DINOv3PoseEstimator(
        dino_model_name=model_name,
        heatmap_size=(heatmap_size, heatmap_size),
        unfreeze_blocks=0,
        use_joint_embedding=use_joint_embedding,
        use_iterative_refinement=use_iterative_refinement,
        refinement_iterations=refinement_iterations,
        fix_joint7_zero=fix_joint7_zero,
    ).to(device)

    # Load checkpoint
    print(f"# Loading weights: {args.model_path}")
    load_checkpoint_compat(
        model=model,
        checkpoint_path=args.model_path,
        device=device,
        is_main_process=True,
    )
    model.eval()

    # Load and preprocess image
    image_pil = PILImage.open(image_path).convert("RGB")
    orig_dim = image_pil.size  # (W, H)

    transform = TVTransforms.Compose([
        TVTransforms.Resize((image_size, image_size)),
        TVTransforms.ToTensor(),
        TVTransforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Run inference
    print("\n# Running inference...")
    image_tensor = transform(image_pil).unsqueeze(0).to(device)

    # Prepare camera_K tensor for depth_only mode
    camera_K_tensor = None
    if camera_K is not None:
        camera_K_tensor = torch.tensor(camera_K, dtype=torch.float32).unsqueeze(0).to(device)
    original_size_tensor = torch.tensor(
        [[orig_dim[0], orig_dim[1]]], dtype=torch.float32, device=device
    )

    with torch.no_grad():
        outputs = model(
            image_tensor, camera_K=camera_K_tensor, original_size=original_size_tensor,
            use_refinement=use_iterative_refinement
        )
        pred_heatmaps = outputs["heatmaps_2d"]
        if args.pred_3d_source == "fk":
            # In joint-angle mode, FK branch is the most stable robot-frame 3D source.
            pred_kpts_3d = outputs["keypoints_3d_fk"] if "keypoints_3d_fk" in outputs else outputs["keypoints_3d"]
        elif args.pred_3d_source == "fk_robopepp":
            if "joint_angles" not in outputs or outputs["joint_angles"] is None:
                print("# Warning: joint_angles missing, fallback to model FK output")
                pred_kpts_3d = outputs["keypoints_3d_fk"] if "keypoints_3d_fk" in outputs else outputs["keypoints_3d"]
            else:
                urdf_file = _resolve_robopepp_urdf(urdf_file)
                print(f"# FK source: RoboPEPP URDF FK ({urdf_file})")
                if args.robopepp_fix_joint7_zero:
                    print("# FK option: forcing joint7=0 to match RoboPEPP 6-DoF eval convention")
                pred_kpts_3d = _compute_robopepp_fk_keypoints(
                    outputs["joint_angles"], device, urdf_file,
                    fix_joint7_zero=args.robopepp_fix_joint7_zero
                )
        else:
            pred_kpts_3d = outputs["keypoints_3d"]

    # Extract 2D keypoints from heatmaps (in heatmap coordinate space)
    pred_2d_heatmap, pred_conf, pred_peak_logit = get_keypoints_from_heatmaps(
        pred_heatmaps,
        min_confidence=args.kp_min_confidence,
        min_peak_logit=args.kp_min_peak_logit,
    )

    # Scale predicted 2D to original image coordinates
    pred_2d_orig = pred_2d_heatmap.copy()
    pred_2d_orig[:, 0] *= orig_dim[0] / heatmap_size
    pred_2d_orig[:, 1] *= orig_dim[1] / heatmap_size

    # 3D predictions - handle different modes
    pred_3d_raw = pred_kpts_3d[0].cpu().numpy()
    print(f"# 3D source: {args.pred_3d_source}")
    n_invalid = int(np.count_nonzero(pred_conf < float(args.kp_min_confidence)))
    if args.kp_min_confidence > 0.0 and n_invalid > 0:
        print(f"# Low-confidence keypoints masked: {n_invalid}/{len(pred_conf)} (threshold={args.kp_min_confidence:.3f})")
    n_low_peak = int(np.count_nonzero(pred_peak_logit < float(args.kp_min_peak_logit)))
    if args.kp_min_peak_logit is not None and n_low_peak > 0:
        print(f"# Low-peak keypoints masked: {n_low_peak}/{len(pred_peak_logit)} (threshold={args.kp_min_peak_logit:.3f} logit)")

    low_conf_mask = pred_conf < float(args.kp_min_confidence)
    low_peak_mask = pred_peak_logit < float(args.kp_min_peak_logit)
    low_reliability_mask = low_conf_mask | low_peak_mask
    pnp_info = None
    # Transform from robot frame to camera frame using PnP
    if camera_K is not None:
        print("# Transforming robot frame → camera frame using PnP...")
        pred_3d, pnp_info = transform_robot_to_camera(
            pred_3d_raw,
            pred_2d_orig,
            camera_K,
            confidences=pred_conf,
            topk=args.pnp_topk,
            min_points=4,
            ransac_reproj_error=args.pnp_ransac_reproj_error,
            pnp_mode=args.pnp_mode,
            reproj_outlier_thresh_px=args.pnp_reproj_outlier_thresh,
            reproj_refit=(not args.disable_pnp_reproj_refit),
            min_span_px=args.pnp_min_span_px,
            min_area_ratio=args.pnp_min_area_ratio,
        )
        if pred_3d is None:
            print("# Warning: PnP failed, using robot frame coordinates (comparison with GT will be invalid)")
            pred_3d = pred_3d_raw
        else:
            print("# Successfully transformed to camera frame")
            if args.fill_invalid_2d_with_fk_reproj:
                proj_all = pnp_info.get("proj_2d_all", None) if pnp_info is not None else None
                if proj_all is not None:
                    replaced = 0
                    for i in range(len(pred_2d_orig)):
                        pred_valid = np.isfinite(pred_2d_orig[i]).all() and (pred_2d_orig[i][0] > -900.0) and (pred_2d_orig[i][1] > -900.0)
                        if (not pred_valid) or low_reliability_mask[i]:
                            pred_2d_orig[i] = proj_all[i]
                            replaced += 1
                    if replaced > 0:
                        print(f"# FK reprojection fill: replaced {replaced} low-reliability/invalid 2D keypoints")
    else:
        print("# Warning: No camera K available, using robot frame coordinates (comparison with GT will be invalid)")
        pred_3d = pred_3d_raw

    ignored_low_conf = [keypoint_names[i] for i in range(len(keypoint_names)) if low_conf_mask[i]]
    ignored_low_peak = [keypoint_names[i] for i in range(len(keypoint_names)) if low_peak_mask[i]]
    ignored_reproj = []
    ignored_pnp_select = []
    if pnp_info is not None:
        idx_removed_reproj = set([int(x) for x in pnp_info.get("idx_removed_reproj", [])])
        idx_used_final = set([int(x) for x in pnp_info.get("idx_selected_final", [])])
        ignored_reproj = [keypoint_names[i] for i in sorted(idx_removed_reproj)]
        ignored_pnp_select = [
            keypoint_names[i]
            for i in range(len(keypoint_names))
            if (i not in idx_used_final) and (i not in idx_removed_reproj) and (not low_conf_mask[i]) and (not low_peak_mask[i])
        ]

    if ignored_low_conf or ignored_low_peak or ignored_reproj or ignored_pnp_select:
        print("# Ignored keypoints summary:")
        if ignored_low_peak:
            print(f"#   low_peak_logit ({args.kp_min_peak_logit:.3f}): {', '.join(ignored_low_peak)}")
        if ignored_low_conf:
            print(f"#   low_conf ({args.kp_min_confidence:.3f}): {', '.join(ignored_low_conf)}")
        if ignored_reproj:
            print(f"#   reproj_outlier ({args.pnp_reproj_outlier_thresh:.1f}px): {', '.join(ignored_reproj)}")
        if ignored_pnp_select:
            print(f"#   pnp_select/topk: {', '.join(ignored_pnp_select)}")
    else:
        print("# Ignored keypoints summary: none")

    # === Compute Metrics ===
    metrics = compute_metrics(pred_2d_orig, gt_2d, pred_3d, gt_3d, keypoint_names, found, orig_dim)

    # Print results
    print("\n" + "=" * 80)
    print("  RESULTS: GT (green) vs Prediction (red)")
    print("=" * 80)

    idx_removed_reproj = set()
    idx_pnp_used = set()
    if pnp_info is not None:
        idx_removed_reproj = set([int(x) for x in pnp_info.get("idx_removed_reproj", [])])
        idx_pnp_used = set([int(x) for x in pnp_info.get("idx_selected_final", [])])

    print(
        f"\n{'Keypoint':<20} {'Status':<20} {'Peak(logit)':<12} "
        f"{'Peak(sigmoid)':<14} {'GT 2D (px)':<22} {'Pred 2D (px)':<22} {'2D Err (px)':<12}"
    )
    print("-" * 130)
    for i, name in enumerate(keypoint_names):
        in_frame = (
            found[i] and
            (0.0 <= gt_2d[i][0] <= orig_dim[0]) and
            (0.0 <= gt_2d[i][1] <= orig_dim[1])
        )
        gt_str = f"({gt_2d[i][0]:7.1f}, {gt_2d[i][1]:7.1f})" if found[i] else "  N/A"
        pred_valid = np.isfinite(pred_2d_orig[i]).all() and (pred_2d_orig[i][0] > -900.0) and (pred_2d_orig[i][1] > -900.0)
        pred_str = f"({pred_2d_orig[i][0]:7.1f}, {pred_2d_orig[i][1]:7.1f})" if pred_valid else "  IGNORED"
        if low_peak_mask[i]:
            status = "IGNORED(low_peak)"
        elif low_conf_mask[i]:
            status = "IGNORED(low_conf)"
        elif i in idx_removed_reproj:
            status = "IGNORED(reproj)"
        elif pnp_info is not None and i not in idx_pnp_used:
            status = "IGNORED(pnp_select)"
        else:
            status = "USED"

        if in_frame and pred_valid:
            err_str = f"{metrics.get(f'{name}_2d_px', float('nan')):8.2f}"
        elif in_frame:
            err_str = "  N/A"
        else:
            err_str = "  N/A"
        print(
            f"  {name:<18} {status:<20} {pred_peak_logit[i]:10.4f}   {pred_conf[i]:10.4f}   "
            f"{gt_str:<22} {pred_str:<22} {err_str}"
        )

    print(f"\n  Mean 2D error:   {metrics.get('mean_2d_px', float('nan')):.2f} px")
    print(f"  Median 2D error: {metrics.get('median_2d_px', float('nan')):.2f} px")
    print(f"  Max 2D error:    {metrics.get('max_2d_px', float('nan')):.2f} px")
    print(f"  Normalized error: {metrics.get('mean_2d_norm', float('nan')):.2f}% of image diagonal")

    print(f"\n{'Keypoint':<20} {'GT 3D (m)':<30} {'Pred 3D (m)':<30} {'3D Err (m)':<12}")
    print("-" * 92)
    for i, name in enumerate(keypoint_names):
        in_frame = (
            found[i] and
            (0.0 <= gt_2d[i][0] <= orig_dim[0]) and
            (0.0 <= gt_2d[i][1] <= orig_dim[1])
        )
        if in_frame and not np.allclose(gt_3d[i], 0):
            gt_str = f"({gt_3d[i][0]:8.4f}, {gt_3d[i][1]:8.4f}, {gt_3d[i][2]:8.4f})"
            pred_str = f"({pred_3d[i][0]:8.4f}, {pred_3d[i][1]:8.4f}, {pred_3d[i][2]:8.4f})"
            err_str = f"{metrics.get(f'{name}_3d_m', float('nan')):.4f}"
        else:
            gt_str = "  N/A"
            pred_str = f"({pred_3d[i][0]:8.4f}, {pred_3d[i][1]:8.4f}, {pred_3d[i][2]:8.4f})"
            err_str = "  N/A"
        print(f"  {name:<18} {gt_str:<30} {pred_str:<30} {err_str}")

    if 'mean_3d_m' in metrics:
        print(f"\n  Mean 3D error:   {metrics['mean_3d_m']:.4f} m ({metrics['mean_3d_m']*100:.2f} cm)")
        print(f"  Median 3D error: {metrics['median_3d_m']:.4f} m ({metrics['median_3d_m']*100:.2f} cm)")
        print(f"  Max 3D error:    {metrics['max_3d_m']:.4f} m ({metrics['max_3d_m']*100:.2f} cm)")

    # Joint angle output (if joint_angle mode)
    if "joint_angles" in outputs and outputs["joint_angles"] is not None:
        pred_angles = outputs["joint_angles"][0].cpu().numpy()
        joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7']
        has_gt = gt_angles is not None
        if has_gt:
            print(f"\n{'Joint':<12} {'GT (rad)':<12} {'GT (deg)':<12} {'Pred (rad)':<12} {'Pred (deg)':<12} {'Err (deg)':<10}")
            print("-" * 70)
            angle_errors = []
            for j in range(min(len(joint_names), len(pred_angles))):
                gt_r = gt_angles[j] if j < len(gt_angles) else 0.0
                err_deg = abs(np.degrees(pred_angles[j] - gt_r))
                angle_errors.append(err_deg)
                print(f"  {joint_names[j]:<10} {gt_r:9.4f}   {np.degrees(gt_r):9.2f}   {pred_angles[j]:9.4f}   {np.degrees(pred_angles[j]):9.2f}   {err_deg:8.2f}")
            print(f"\n  Mean joint angle error: {np.mean(angle_errors):.2f} deg")
            print(f"  Max joint angle error:  {np.max(angle_errors):.2f} deg")
        else:
            print(f"\n{'Joint':<12} {'Pred (rad)':<14} {'Pred (deg)':<14}")
            print("-" * 40)
            for j in range(min(len(joint_names), len(pred_angles))):
                print(f"  {joint_names[j]:<10} {pred_angles[j]:10.4f}   {np.degrees(pred_angles[j]):10.2f}")
            print("  (No GT joint angles available in this annotation)")

    # Iterative refinement per-iteration angle error
    if "all_refined_angles" in outputs and outputs["all_refined_angles"] is not None and gt_angles is not None:
        all_ref_angles = outputs["all_refined_angles"]
        print(f"\n  Iterative Refinement ({len(all_ref_angles)-1} iterations):")
        for step_i, step_angles in enumerate(all_ref_angles):
            step_a = step_angles[0].cpu().numpy()
            step_errors = np.abs(np.degrees(step_a - gt_angles))
            label = "initial" if step_i == 0 else f"iter {step_i}"
            print(f"    [{label}] Mean angle error: {np.mean(step_errors):.2f} deg, Max: {np.max(step_errors):.2f} deg")

    print("=" * 80)

    # === Visualizations ===
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"\n# Saving visualizations to: {args.output_dir}")

        input_dim = (image_size, image_size)
        image_resized = image_pil.resize(input_dim, resample=PILImage.BILINEAR)

        # Scale GT to network input coords
        gt_2d_input = gt_2d.copy()
        gt_2d_input[:, 0] *= input_dim[0] / orig_dim[0]
        gt_2d_input[:, 1] *= input_dim[1] / orig_dim[1]

        pred_2d_input = pred_2d_heatmap.copy()
        pred_2d_input[:, 0] *= input_dim[0] / heatmap_size
        pred_2d_input[:, 1] *= input_dim[1] / heatmap_size

        # Filter valid GTs
        gt_2d_input_list = [gt_2d_input[i].tolist() if found[i] else None for i in range(len(keypoint_names))]
        gt_2d_input_valid = [pt for pt in gt_2d_input_list if pt is not None]
        gt_names_valid = [name for i, name in enumerate(keypoint_names) if found[i]]
        pred_2d_input_valid = []
        pred_2d_input_valid_names = []
        pred_2d_orig_valid = []
        pred_2d_orig_valid_names = []
        for i, name in enumerate(keypoint_names):
            is_valid_in = np.isfinite(pred_2d_input[i]).all() and (pred_2d_input[i][0] > -900.0) and (pred_2d_input[i][1] > -900.0)
            is_valid_orig = np.isfinite(pred_2d_orig[i]).all() and (pred_2d_orig[i][0] > -900.0) and (pred_2d_orig[i][1] > -900.0)
            if is_valid_in:
                pred_2d_input_valid.append(pred_2d_input[i].tolist())
                pred_2d_input_valid_names.append(name)
            if is_valid_orig:
                pred_2d_orig_valid.append(pred_2d_orig[i].tolist())
                pred_2d_orig_valid_names.append(name)

        # 1. GT (green) + Pred (red) on image
        overlay = image_resized.copy()
        if gt_2d_input_valid:
            overlay = dream.image_proc.overlay_points_on_image(
                overlay, gt_2d_input_valid, gt_names_valid,
                annotation_color_dot="green", annotation_color_text="white",
            )
        if pred_2d_input_valid:
            overlay = dream.image_proc.overlay_points_on_image(
                overlay, pred_2d_input_valid, pred_2d_input_valid_names,
                annotation_color_dot="red", annotation_color_text="white",
            )
        out_path = os.path.join(args.output_dir, "01_gt_vs_pred_keypoints.png")
        overlay.save(out_path)
        print(f"  Saved: {out_path}")

        # 2. Belief map mosaic
        gt_2d_heatmap_list = None
        if any(found):
            gt_2d_heatmap = gt_2d.copy()
            gt_2d_heatmap[:, 0] *= heatmap_size / orig_dim[0]
            gt_2d_heatmap[:, 1] *= heatmap_size / orig_dim[1]
            gt_2d_heatmap_list = [gt_2d_heatmap[i].tolist() if found[i] else None
                                  for i in range(len(keypoint_names))]

        belief_map_images = dream.image_proc.images_from_belief_maps(
            pred_heatmaps[0], normalization_method=6
        )
        belief_map_images_kp = []
        for kp_idx in range(len(keypoint_names)):
            points = []
            colors = []
            if np.isfinite(pred_2d_heatmap[kp_idx]).all() and (pred_2d_heatmap[kp_idx][0] > -900.0) and (pred_2d_heatmap[kp_idx][1] > -900.0):
                points.append(pred_2d_heatmap[kp_idx])
                colors.append("red")
            if gt_2d_heatmap_list and gt_2d_heatmap_list[kp_idx] is not None:
                points.insert(0, gt_2d_heatmap_list[kp_idx])
                colors.insert(0, "green")
            if points:
                bm_kp = dream.image_proc.overlay_points_on_image(
                    belief_map_images[kp_idx], points,
                    annotation_color_dot=colors, annotation_color_text=colors,
                    point_diameter=4,
                )
            else:
                bm_kp = belief_map_images[kp_idx]
            belief_map_images_kp.append(bm_kp)

        n_cols = int(math.ceil(len(keypoint_names) / 2.0))
        mosaic = dream.image_proc.mosaic_images(
            belief_map_images_kp, rows=2, cols=n_cols,
            inner_padding_px=10, fill_color_rgb=(0, 0, 0),
        )
        out_path = os.path.join(args.output_dir, "02_belief_map_mosaic.png")
        mosaic.save(out_path)
        print(f"  Saved: {out_path}")

        # 3. Per-joint belief maps overlaid on image
        blended_array = []
        for n in range(len(keypoint_names)):
            bm = belief_map_images[n].resize(input_dim, resample=PILImage.BILINEAR)
            blended = PILImage.blend(image_resized, bm, alpha=0.5)
            if np.isfinite(pred_2d_input[n]).all() and (pred_2d_input[n][0] > -900.0) and (pred_2d_input[n][1] > -900.0):
                blended = dream.image_proc.overlay_points_on_image(
                    blended, [pred_2d_input[n]], [keypoint_names[n]],
                    annotation_color_dot="red", annotation_color_text="white",
                )
            if found[n]:
                blended = dream.image_proc.overlay_points_on_image(
                    blended, [gt_2d_input[n].tolist()], [keypoint_names[n]],
                    annotation_color_dot="green", annotation_color_text="white",
                )
            blended_array.append(blended)

        mosaic2 = dream.image_proc.mosaic_images(
            blended_array, rows=2, cols=n_cols, fill_color_rgb=(0, 0, 0)
        )
        out_path = os.path.join(args.output_dir, "03_belief_maps_overlay_mosaic.png")
        mosaic2.save(out_path)
        print(f"  Saved: {out_path}")

        # 4. Combined belief map on original image
        belief_combined = pred_heatmaps[0].sum(dim=0)
        belief_combined_img = dream.image_proc.image_from_belief_map(
            belief_combined, normalization_method=6
        )
        belief_orig = belief_combined_img.resize(orig_dim, resample=PILImage.BILINEAR)
        orig_overlay = PILImage.blend(image_pil, belief_orig, alpha=0.5)
        if any(found):
            gt_valid_orig = [gt_2d[i].tolist() for i in range(len(keypoint_names)) if found[i]]
            orig_overlay = dream.image_proc.overlay_points_on_image(
                orig_overlay, gt_valid_orig, gt_names_valid,
                annotation_color_dot="green", annotation_color_text="white",
            )
        if pred_2d_orig_valid:
            orig_overlay = dream.image_proc.overlay_points_on_image(
                orig_overlay, pred_2d_orig_valid, pred_2d_orig_valid_names,
                annotation_color_dot="red", annotation_color_text="white",
            )
        out_path = os.path.join(args.output_dir, "04_combined_on_original.png")
        orig_overlay.save(out_path)
        print(f"  Saved: {out_path}")

        # 5. Save metrics to JSON
        metrics_path = os.path.join(args.output_dir, "metrics.json")
        # Convert numpy types for JSON serialization
        metrics_json = {k: float(v) if not np.isnan(v) else None for k, v in metrics.items()}
        metrics_json['json_path'] = args.json_path
        metrics_json['image_path'] = image_path
        metrics_json['ignored_keypoints'] = {
            'low_peak_logit': ignored_low_peak,
            'low_conf': ignored_low_conf,
            'reproj_outlier': ignored_reproj,
            'pnp_select': ignored_pnp_select,
        }
        metrics_json['pnp'] = {
            'mode': args.pnp_mode,
            'reproj_outlier_thresh_px': float(args.pnp_reproj_outlier_thresh),
            'min_span_px': float(args.pnp_min_span_px),
            'min_area_ratio': float(args.pnp_min_area_ratio),
            'used_keypoints': sorted(list(idx_pnp_used)),
            'removed_reproj_keypoints': sorted(list(idx_removed_reproj)),
            'spread_ok': bool(pnp_info.get('spread_ok', True)) if pnp_info is not None else None,
            'spread_span_xy_px': (pnp_info.get('spread_span_xy_px') if pnp_info is not None else None),
            'spread_area_ratio': (float(pnp_info.get('spread_area_ratio')) if (pnp_info is not None and pnp_info.get('spread_area_ratio') is not None) else None),
        }
        metrics_json['heatmap_peak'] = {
            name: {
                'peak_logit': float(pred_peak_logit[i]),
                'peak_sigmoid': float(pred_conf[i]),
            }
            for i, name in enumerate(keypoint_names)
        }
        with open(metrics_path, 'w') as f:
            json.dump(metrics_json, f, indent=2)
        print(f"  Saved: {metrics_path}")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-j", "--json-path", required=True,
                        help="Path to annotation JSON (contains image path + GT keypoints)")
    parser.add_argument("-p", "--model-path", required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("-o", "--output-dir", default=None,
                        help="Directory to save visualizations and metrics")
    parser.add_argument("--model-name", default="facebook/dinov3-vitb16-pretrain-lvd1689m",
                        help="DINOv3 model name (overridden by config.yaml)")
    parser.add_argument("--pred-3d-source", type=str, default="fk", choices=["fk", "fk_robopepp", "fused"],
                        help="Robot-frame 3D source before PnP transform")
    parser.add_argument("--robopepp-urdf", type=str, default=None,
                        help="Override Panda URDF path used when --pred-3d-source=fk_robopepp")
    parser.add_argument("--robopepp-fix-joint7-zero", action="store_true",
                        help="When using fk_robopepp, force joint7=0 (RoboPEPP 6-joint convention)")
    parser.add_argument("--pnp-topk", type=int, default=6,
                        help="Use top-k confident keypoints for PnP")
    parser.add_argument("--pnp-ransac-reproj-error", type=float, default=5.0,
                        help="RANSAC reprojection threshold (px)")
    parser.add_argument("--pnp-mode", type=str, default="epnp", choices=["epnp", "loo_epnp", "ransac"],
                        help="PnP solver mode")
    parser.add_argument("--pnp-reproj-outlier-thresh", type=float, default=12.0,
                        help="After initial PnP, ignore keypoints with reprojection error above this threshold (px)")
    parser.add_argument("--disable-pnp-reproj-refit", action="store_true",
                        help="Disable reprojection outlier rejection + refit step")
    parser.add_argument("--pnp-min-span-px", type=float, default=20.0,
                        help="Minimum x/y span (px) of selected 2D keypoints required for PnP")
    parser.add_argument("--pnp-min-area-ratio", type=float, default=0.001,
                        help="Minimum 2D bbox area ratio of selected points for PnP")
    parser.add_argument("--kp-min-peak-logit", type=float, default=-1e9,
                        help="Mask predicted 2D keypoints when heatmap peak logit is below this threshold")
    parser.add_argument("--kp-min-confidence", type=float, default=0.0,
                        help="Mask predicted 2D keypoints when sigmoid(max_heatmap_logit) is below this threshold")
    parser.add_argument("--fill-invalid-2d-with-fk-reproj", action="store_true",
                        help="After successful PnP, fill invalid/low-reliability 2D keypoints using FK reprojection")
    args = parser.parse_args()

    network_inference(args)
