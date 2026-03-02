"""
Single Image Inference Script for DINOv3 Pose Estimation
Runs inference on a single image and visualizes/saves the results.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms

# Import model from TRAIN directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../Train')))
from model import DINOv3PoseEstimator


def load_from_json_annotation(json_path: str):
    """Load image path, optional camera K, and optional GT 2D keypoints from JSON."""
    with open(json_path, "r") as f:
        data = json.load(f)

    image_path = None
    camera_k = None
    gt_keypoints_2d = None
    json_dir = os.path.dirname(os.path.abspath(json_path))

    if "meta" in data:
        meta = data["meta"]
        if "image_path" in meta:
            raw_path = meta["image_path"]
            if not os.path.isabs(raw_path):
                image_path = os.path.normpath(os.path.join(json_dir, raw_path))
            else:
                image_path = raw_path

        if "camera" in meta and "intrinsic_matrix" in meta["camera"]:
            k_raw = meta["camera"]["intrinsic_matrix"]
            k = np.array(k_raw, dtype=np.float64)
            if k.shape == (3, 3):
                camera_k = k
        elif "K" in meta:
            k = np.array(meta["K"], dtype=np.float64)
            if k.shape == (3, 3):
                camera_k = k

    # Fallback path variants used in this repo
    if image_path is None and "image_paths" in data:
        ip = data.get("image_paths", {})
        for key in ["rgb", "color", "image"]:
            if key in ip:
                cand = ip[key]
                image_path = cand if os.path.isabs(cand) else os.path.normpath(os.path.join(json_dir, cand))
                break

    # If JSON path is stale, try robust fallbacks from current dataset layout.
    def _fallback_from_converted_layout():
        stem = Path(json_path).stem
        dataset_name = Path(json_path).parent.name  # e.g. panda-3cam_azure
        root = Path(json_path).resolve().parents[3]  # .../Dataset
        return root / "DREAM_real" / dataset_name / dataset_name / f"{stem}.rgb.jpg"

    candidates = []
    if image_path is not None:
        candidates.append(Path(image_path))
        # keep only basename and try near JSON
        candidates.append(Path(json_dir) / Path(image_path).name)
    candidates.append(_fallback_from_converted_layout())

    resolved_image = None
    for cand in candidates:
        if cand.exists():
            resolved_image = str(cand)
            break

    if resolved_image is None:
        raise ValueError(f"Failed to resolve image path from JSON: {json_path}")

    # Optional GT 2D keypoints from DREAM annotation format.
    if "objects" in data and len(data["objects"]) > 0:
        obj0 = data["objects"][0]
        if "keypoints" in obj0:
            gt_keypoints_2d = {}
            for kp in obj0["keypoints"]:
                name = kp.get("name")
                proj = kp.get("projected_location")
                if name is None or proj is None or len(proj) < 2:
                    continue
                gt_keypoints_2d[name] = [float(proj[0]), float(proj[1])]

    return resolved_image, camera_k, gt_keypoints_2d


def parse_camera_k(camera_k_str: str):
    """Parse camera intrinsics from 'fx,fy,cx,cy'."""
    if not camera_k_str:
        return None
    vals = [float(v.strip()) for v in camera_k_str.split(",")]
    if len(vals) != 4:
        raise ValueError("--camera-k must be formatted as 'fx,fy,cx,cy'")
    fx, fy, cx, cy = vals
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def transform_robot_to_camera(
    robot_kpts: np.ndarray,
    pred_2d: np.ndarray,
    camera_k: np.ndarray,
    min_span_px: float = 20.0,
    min_area_ratio: float = 0.001,
):
    """Estimate robot->camera pose with EPnP and transform all keypoints to camera frame."""
    valid = (
        np.isfinite(robot_kpts).all(axis=1)
        & np.isfinite(pred_2d).all(axis=1)
        & (pred_2d[:, 0] > -900.0)
        & (pred_2d[:, 1] > -900.0)
    )
    if np.count_nonzero(valid) < 4:
        return None, None
    pred_valid = pred_2d[valid]
    x_span = float(np.max(pred_valid[:, 0]) - np.min(pred_valid[:, 0]))
    y_span = float(np.max(pred_valid[:, 1]) - np.min(pred_valid[:, 1]))
    bbox_area = max(0.0, x_span) * max(0.0, y_span)
    img_area = max(1.0, float(camera_k[0, 2] * 2.0) * float(camera_k[1, 2] * 2.0))
    area_ratio = bbox_area / img_area
    if x_span < float(min_span_px) or y_span < float(min_span_px) or area_ratio < float(min_area_ratio):
        return None, None
    success, rvec, tvec = cv2.solvePnP(
        robot_kpts[valid].astype(np.float64),
        pred_2d[valid].astype(np.float64),
        camera_k.astype(np.float64),
        None,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not success:
        return None, None
    r, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)
    camera_pts = (r @ robot_kpts.T).T + t.reshape(1, 3)
    proj_all, _ = cv2.projectPoints(
        robot_kpts.astype(np.float64), rvec, tvec, camera_k.astype(np.float64), None
    )
    return camera_pts, proj_all.reshape(-1, 2)


def get_keypoints_from_heatmaps(
    heatmaps: torch.Tensor,
    min_confidence: float = 0.0,
    min_peak_logit: float = -1e9,
):
    """Extract argmax keypoints and optional validity masks from heatmaps."""
    b, n, h, w = heatmaps.shape
    heatmaps_flat = heatmaps.view(b, n, -1)
    max_indices = torch.argmax(heatmaps_flat, dim=-1)

    y = max_indices // w
    x = max_indices % w

    keypoints = torch.stack([x, y], dim=-1).float()
    peak_logits = heatmaps_flat.amax(dim=-1)
    confidences = torch.sigmoid(peak_logits)

    invalid = torch.zeros_like(confidences, dtype=torch.bool)
    if min_confidence > 0.0:
        invalid = invalid | (confidences < float(min_confidence))
    if min_peak_logit is not None:
        invalid = invalid | (peak_logits < float(min_peak_logit))
    if invalid.any():
        keypoints = keypoints.masked_fill(invalid.unsqueeze(-1), -999.0)

    return keypoints.cpu().numpy(), confidences.cpu().numpy(), peak_logits.cpu().numpy()


def visualize_keypoints(image: np.ndarray, keypoints: np.ndarray, keypoint_names: list, confidences=None, skeleton=None):
    """Draw predicted keypoints/skeleton."""
    vis_image = image.copy()
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 128, 0),
    ]

    def _valid(pt):
        return bool(np.isfinite(pt).all() and pt[0] > -900.0 and pt[1] > -900.0)

    if skeleton:
        for idx1, idx2 in skeleton:
            if idx1 < len(keypoints) and idx2 < len(keypoints):
                if _valid(keypoints[idx1]) and _valid(keypoints[idx2]):
                    pt1 = tuple(keypoints[idx1].astype(int))
                    pt2 = tuple(keypoints[idx2].astype(int))
                    cv2.line(vis_image, pt1, pt2, (128, 128, 128), 2)

    for i, (kp, name) in enumerate(zip(keypoints, keypoint_names)):
        if not _valid(kp):
            continue
        x, y = int(kp[0]), int(kp[1])
        color = colors[i % len(colors)]
        cv2.circle(vis_image, (x, y), 5, color, -1)
        cv2.circle(vis_image, (x, y), 7, (255, 255, 255), 2)

        label = name
        if confidences is not None:
            label += f" ({confidences[i]:.2f})"

        cv2.putText(vis_image, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
        cv2.putText(vis_image, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return vis_image


def visualize_heatmaps(heatmaps: torch.Tensor, keypoint_names: list, original_size: tuple):
    """Visualize heatmaps as a grid."""
    n, h, w = heatmaps.shape
    n_cols = min(4, n)
    n_rows = (n + n_cols - 1) // n_cols

    aspect_ratio = original_size[0] / original_size[1]
    if aspect_ratio > 1:
        cell_w = 200
        cell_h = int(200 / aspect_ratio)
    else:
        cell_h = 200
        cell_w = int(200 * aspect_ratio)

    grid = np.zeros((n_rows * cell_h, n_cols * cell_w, 3), dtype=np.uint8)

    for i in range(n):
        row = i // n_cols
        col = i % n_cols

        hm = heatmaps[i].cpu().numpy()
        hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
        hm = (hm * 255).astype(np.uint8)
        hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
        hm_resized = cv2.resize(hm_color, (cell_w, cell_h))

        cv2.putText(hm_resized, keypoint_names[i], (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
        cv2.putText(hm_resized, keypoint_names[i], (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        y0 = row * cell_h
        x0 = col * cell_w
        grid[y0:y0 + cell_h, x0:x0 + cell_w] = hm_resized

    return grid


def visualize_merged_heatmap(
    image: np.ndarray,
    heatmaps: torch.Tensor,
    keypoints: np.ndarray,
    keypoint_names: list,
    gt_keypoints: np.ndarray | None = None,
    mean_error_px: float | None = None,
):
    """Visualize all keypoint heatmaps merged into a single map."""
    merged = torch.max(heatmaps, dim=0).values.cpu().numpy()
    merged = (merged - merged.min()) / (merged.max() - merged.min() + 1e-8)
    merged_u8 = (merged * 255).astype(np.uint8)
    merged_color = cv2.applyColorMap(merged_u8, cv2.COLORMAP_JET)
    merged_color = cv2.resize(merged_color, (image.shape[1], image.shape[0]))

    overlay = cv2.addWeighted(image, 0.45, cv2.cvtColor(merged_color, cv2.COLOR_BGR2RGB), 0.55, 0.0)
    vis = overlay.copy()
    for i, (kp, name) in enumerate(zip(keypoints, keypoint_names)):
        if kp[0] <= -900.0 or kp[1] <= -900.0:
            continue
        x, y = int(kp[0]), int(kp[1])
        # Pred marker: small green.
        cv2.circle(vis, (x, y), 1, (0, 255, 0), -1)
        cv2.circle(vis, (x, y), 2, (0, 80, 0), 1)

        label = f"{i+1}:{name}"
        # Per-keypoint label offsets for readability.
        label_offsets = [
            (-10, 30),  # 1 (lower)
            (6, 28),    # 2
            (-10, 40),  # 3
            (-50, -30), # 4 (up + left)
            (-40, -18), # 5 (up + left)
            (38, 24),   # 6 (more right)
            (42, 40),   # 7 (more right)
        ]
        if i < len(label_offsets):
            ox, oy = label_offsets[i]
        else:
            ox, oy = (8, 18)
        tx, ty = x + ox, y + oy
        tx = int(np.clip(tx, 2, vis.shape[1] - 220))
        ty = int(np.clip(ty, 14, vis.shape[0] - 4))
        cv2.putText(vis, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    # Draw GT keypoints (if available) in red.
    if gt_keypoints is not None:
        for gt in gt_keypoints:
            if gt[0] <= -900.0 or gt[1] <= -900.0:
                continue
            gx, gy = int(gt[0]), int(gt[1])
            cv2.circle(vis, (gx, gy), 1, (255, 0, 0), -1)
            cv2.circle(vis, (gx, gy), 2, (120, 0, 0), 1)

    if mean_error_px is not None and np.isfinite(mean_error_px):
        text = f"Mean 2D err: {mean_error_px:.2f}px"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
        tx = max(6, vis.shape[1] - tw - 10)
        ty = max(th + 6, vis.shape[0] - 10)
        cv2.putText(vis, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def visualize_peak_values(keypoint_names: list, peak_logits: np.ndarray, peak_sigmoid: np.ndarray):
    """Draw per-keypoint peak values as a bar chart image."""
    n = len(keypoint_names)
    row_h = 34
    left = 210
    bar_w = 330
    width = left + bar_w + 260
    height = 30 + n * row_h + 20
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)

    min_v = float(np.min(peak_logits))
    max_v = float(np.max(peak_logits))
    if max_v - min_v < 1e-8:
        max_v = min_v + 1.0

    cv2.putText(canvas, "Peak logits per keypoint", (20, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)

    for i, name in enumerate(keypoint_names):
        y = 45 + i * row_h
        logit = float(peak_logits[i])
        sigm = float(peak_sigmoid[i])
        ratio = (logit - min_v) / (max_v - min_v)
        x2 = left + int(np.clip(ratio, 0.0, 1.0) * bar_w)

        cv2.putText(canvas, f"{name}", (20, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1)
        cv2.rectangle(canvas, (left, y - 10), (left + bar_w, y + 8), (220, 220, 220), 1)
        cv2.rectangle(canvas, (left, y - 10), (x2, y + 8), (70, 130, 255), -1)
        cv2.putText(
            canvas,
            f"logit={logit:.4f}  sigmoid={sigm:.4f}",
            (left + bar_w + 12, y + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (30, 30, 30),
            1,
        )

    return canvas


@torch.no_grad()
def inference_single_image(args):
    if not args.image_path and not args.json_path:
        raise ValueError("Provide either --image-path or --json-path")
    if args.image_path and args.json_path:
        raise ValueError("Use only one of --image-path or --json-path")

    image_path = args.image_path
    camera_k_from_json = None
    gt_keypoints_from_json = None
    if args.json_path:
        if not os.path.exists(args.json_path):
            raise FileNotFoundError(f"JSON not found: {args.json_path}")
        image_path, camera_k_from_json, gt_keypoints_from_json = load_from_json_annotation(args.json_path)

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    checkpoint_dir = Path(args.model_path).parent
    config_path = checkpoint_dir / 'config.yaml'

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
        print(f"Loaded training config from {config_path}")
    else:
        print(f"Warning: Config not found at {config_path}, using defaults")

    model_name = args.model_name or train_config.get('model_name', 'facebook/dinov3-vitb16-pretrain-lvd1689m')
    image_size = args.image_size or int(train_config.get('image_size', 512))
    heatmap_size = args.heatmap_size or int(train_config.get('heatmap_size', 512))
    use_joint_embedding = bool(train_config.get('use_joint_embedding', False))
    fix_joint7_zero = bool(train_config.get('fix_joint7_zero', False))

    print(f"\nLoading model from {args.model_path}")
    print(f"  model_name: {model_name}")
    print(f"  image_size: {image_size}, heatmap_size: {heatmap_size}")
    print(f"  use_joint_embedding: {use_joint_embedding}")
    print(f"  fix_joint7_zero: {fix_joint7_zero}")
    print(f"  keypoint_names ({len(keypoint_names)}): {keypoint_names}")

    skeleton = [(i, i + 1) for i in range(len(keypoint_names) - 1)]

    model = DINOv3PoseEstimator(
        dino_model_name=model_name,
        heatmap_size=(heatmap_size, heatmap_size),
        unfreeze_blocks=0,
        use_joint_embedding=use_joint_embedding,
        fix_joint7_zero=fix_joint7_zero,
    ).to(device)

    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    if 'model_state_dict' in checkpoint:
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    if 'backbone.model.embeddings.mask_token' in state_dict:
        mask_token_shape = state_dict['backbone.model.embeddings.mask_token'].shape
        if len(mask_token_shape) == 3 and mask_token_shape[1] == 1:
            print('Removing mask_token from state_dict due to shape mismatch')
            del state_dict['backbone.model.embeddings.mask_token']

    # Compatibility for old/new checkpoints.
    model_state = model.state_dict()
    filtered_state = {}
    dropped = []
    for k, v in state_dict.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered_state[k] = v
        elif k in model_state:
            dropped.append(k)
    if dropped:
        print(f"Dropping {len(dropped)} mismatched checkpoint keys (shape mismatch)")

    model.load_state_dict(filtered_state, strict=False)
    model.eval()

    print(f"\nLoading image: {image_path}")
    image_pil = Image.open(image_path).convert('RGB')
    original_size = image_pil.size
    print(f"Original image size: {original_size}")

    image_np = np.array(image_pil)

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = transform(image_pil).unsqueeze(0).to(device)

    print("\nRunning inference...")
    outputs = model(image_tensor, original_size=original_size)
    pred_heatmaps = outputs['heatmaps_2d']
    if args.pred_3d_source == 'fk':
        pred_kpts_3d = outputs['keypoints_3d_fk'] if 'keypoints_3d_fk' in outputs else outputs['keypoints_3d']
    else:
        pred_kpts_3d = outputs['keypoints_3d']

    pred_keypoints, pred_conf, pred_peak = get_keypoints_from_heatmaps(
        pred_heatmaps,
        min_confidence=args.kp_min_confidence,
        min_peak_logit=args.kp_min_peak_logit,
    )
    pred_keypoints = pred_keypoints[0]
    pred_conf = pred_conf[0]
    pred_peak = pred_peak[0]

    scale_x = original_size[0] / heatmap_size
    scale_y = original_size[1] / heatmap_size

    pred_keypoints_scaled = pred_keypoints.copy()
    valid_mask = (pred_keypoints_scaled[:, 0] > -900.0) & (pred_keypoints_scaled[:, 1] > -900.0)
    pred_keypoints_scaled[valid_mask, 0] *= scale_x
    pred_keypoints_scaled[valid_mask, 1] *= scale_y
    gt_keypoints_scaled = None
    mean_error_px = None
    if gt_keypoints_from_json is not None:
        gt_keypoints_scaled = np.full((len(keypoint_names), 2), -999.0, dtype=np.float32)
        for i, name in enumerate(keypoint_names):
            if name in gt_keypoints_from_json:
                gt_keypoints_scaled[i, 0] = gt_keypoints_from_json[name][0]
                gt_keypoints_scaled[i, 1] = gt_keypoints_from_json[name][1]
        valid_eval = (
            (pred_keypoints_scaled[:, 0] > -900.0)
            & (pred_keypoints_scaled[:, 1] > -900.0)
            & (gt_keypoints_scaled[:, 0] > -900.0)
            & (gt_keypoints_scaled[:, 1] > -900.0)
        )
        if np.any(valid_eval):
            diffs = pred_keypoints_scaled[valid_eval] - gt_keypoints_scaled[valid_eval]
            mean_error_px = float(np.mean(np.linalg.norm(diffs, axis=1)))

    pred_kpts_3d_np = pred_kpts_3d[0].cpu().numpy()
    pred_joint_angles = outputs['joint_angles'][0].cpu().numpy() if ('joint_angles' in outputs and outputs['joint_angles'] is not None) else None

    print('\n' + '=' * 80)
    print('INFERENCE RESULTS')
    print('=' * 80)

    print('\nPredicted Keypoints (original image coordinates):')
    for i, (name, kp) in enumerate(zip(keypoint_names, pred_keypoints_scaled)):
        if kp[0] > -900.0 and kp[1] > -900.0:
            print(
                f"  {i+1}. {name:20s}: ({kp[0]:7.2f}, {kp[1]:7.2f}) "
                f"peak={pred_peak[i]:.4f} conf={pred_conf[i]:.4f}"
            )
        else:
            print(
                f"  {i+1}. {name:20s}: IGNORED "
                f"(peak={pred_peak[i]:.4f}, conf={pred_conf[i]:.4f})"
            )

    if pred_joint_angles is not None:
        print('\nPredicted Joint Angles:')
        for i, ang in enumerate(pred_joint_angles):
            print(f"  joint{i+1}: {ang:8.4f} rad ({np.degrees(ang):7.2f} deg)")

    print('\nPredicted 3D Keypoints (FK, robot-base frame):')
    for i, (name, kp3d) in enumerate(zip(keypoint_names, pred_kpts_3d_np)):
        print(f"  {i+1}. {name:20s}: ({kp3d[0]:7.4f}, {kp3d[1]:7.4f}, {kp3d[2]:7.4f})")

    pred_kpts_3d_cam = None
    camera_k = parse_camera_k(args.camera_k)
    if camera_k is None:
        camera_k = camera_k_from_json
    if camera_k is not None:
        pred_kpts_3d_cam, proj_2d_all = transform_robot_to_camera(
            pred_kpts_3d_np,
            pred_keypoints_scaled,
            camera_k,
            min_span_px=args.pnp_min_span_px,
            min_area_ratio=args.pnp_min_area_ratio,
        )
        if pred_kpts_3d_cam is not None:
            if args.fill_invalid_2d_with_fk_reproj and proj_2d_all is not None:
                low_reliability_mask = (pred_conf < float(args.kp_min_confidence)) | (pred_peak < float(args.kp_min_peak_logit))
                replaced = 0
                for i in range(len(pred_keypoints_scaled)):
                    pred_valid = (
                        np.isfinite(pred_keypoints_scaled[i]).all()
                        and pred_keypoints_scaled[i][0] > -900.0
                        and pred_keypoints_scaled[i][1] > -900.0
                    )
                    if (not pred_valid) or low_reliability_mask[i]:
                        pred_keypoints_scaled[i] = proj_2d_all[i]
                        replaced += 1
                if replaced > 0:
                    print(f"\nFK reprojection fill: replaced {replaced} low-reliability/invalid 2D keypoints")
            print('\nPredicted 3D Keypoints (camera frame via PnP):')
            for i, (name, kp3d) in enumerate(zip(keypoint_names, pred_kpts_3d_cam)):
                print(f"  {i+1}. {name:20s}: ({kp3d[0]:7.4f}, {kp3d[1]:7.4f}, {kp3d[2]:7.4f})")
        else:
            print('\nPredicted 3D Keypoints (camera frame via PnP): unavailable (need >=4 spread valid 2D points)')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = Path(image_path).stem

    vis_keypoints = visualize_keypoints(
        image_np,
        pred_keypoints_scaled,
        keypoint_names,
        confidences=pred_conf,
        skeleton=skeleton,
    )
    keypoints_path = output_dir / f'{base_name}_keypoints.jpg'
    cv2.imwrite(str(keypoints_path), cv2.cvtColor(vis_keypoints, cv2.COLOR_RGB2BGR))
    print(f"\nSaved keypoint visualization: {keypoints_path}")

    heatmaps_vis = None
    merged_heatmap_vis = None
    peak_values_vis = None
    if args.save_heatmaps:
        heatmaps_vis = visualize_heatmaps(pred_heatmaps[0], keypoint_names, original_size)
        heatmaps_path = output_dir / f'{base_name}_heatmaps.jpg'
        cv2.imwrite(str(heatmaps_path), heatmaps_vis)
        print(f"Saved heatmap visualization: {heatmaps_path}")

        merged_heatmap_vis = visualize_merged_heatmap(
            image_np,
            pred_heatmaps[0],
            pred_keypoints_scaled,
            keypoint_names,
            gt_keypoints=gt_keypoints_scaled,
            mean_error_px=mean_error_px,
        )
        merged_heatmap_path = output_dir / f'{base_name}_heatmap_merged.jpg'
        cv2.imwrite(str(merged_heatmap_path), cv2.cvtColor(merged_heatmap_vis, cv2.COLOR_RGB2BGR))
        print(f"Saved merged heatmap visualization: {merged_heatmap_path}")

        peak_values_vis = visualize_peak_values(keypoint_names, pred_peak, pred_conf)
        peak_values_path = output_dir / f'{base_name}_peak_values.jpg'
        cv2.imwrite(str(peak_values_path), peak_values_vis)
        print(f"Saved peak-value visualization: {peak_values_path}")

    results = {
        'image_path': str(image_path),
        'json_path': str(args.json_path) if args.json_path else None,
        'image_size': original_size,
        'keypoints_2d': [
            {
                'name': name,
                'x': float(kp[0]),
                'y': float(kp[1]),
                'peak_logit': float(pred_peak[i]),
                'peak_sigmoid': float(pred_conf[i]),
            }
            for i, (name, kp) in enumerate(zip(keypoint_names, pred_keypoints_scaled))
        ],
        'keypoints_3d': [
            {'name': name, 'x': float(kp3d[0]), 'y': float(kp3d[1]), 'z': float(kp3d[2])}
            for name, kp3d in zip(keypoint_names, pred_kpts_3d_np)
        ],
        'joint_angles': [float(a) for a in pred_joint_angles] if pred_joint_angles is not None else None,
        'keypoints_3d_camera': (
            [
                {'name': name, 'x': float(kp3d[0]), 'y': float(kp3d[1]), 'z': float(kp3d[2])}
                for name, kp3d in zip(keypoint_names, pred_kpts_3d_cam)
            ]
            if pred_kpts_3d_cam is not None else None
        ),
        'model_path': str(args.model_path),
        'model_name': model_name,
        'pred_3d_source': args.pred_3d_source,
        'kp_min_confidence': float(args.kp_min_confidence),
        'kp_min_peak_logit': float(args.kp_min_peak_logit),
    }

    json_path = output_dir / f'{base_name}_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved JSON results: {json_path}")

    if args.save_combined:
        h_orig, w_orig = image_np.shape[:2]
        target_h = 480
        scale = target_h / h_orig
        target_w = int(w_orig * scale)

        img_resized = cv2.resize(image_np, (target_w, target_h))
        kp_resized = cv2.resize(vis_keypoints, (target_w, target_h))

        if heatmaps_vis is not None:
            hm_h, hm_w = heatmaps_vis.shape[:2]
            hm_scale = target_h / hm_h
            hm_w_new = int(hm_w * hm_scale)
            hm_resized = cv2.resize(heatmaps_vis, (hm_w_new, target_h))
            combined = np.hstack([img_resized, kp_resized, hm_resized])
        else:
            combined = np.hstack([img_resized, kp_resized])

        combined_path = output_dir / f'{base_name}_combined.jpg'
        cv2.imwrite(str(combined_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        print(f"Saved combined visualization: {combined_path}")

    print('\n' + '=' * 80)
    print(f'All results saved to: {output_dir}')
    print('=' * 80)

    return results


def main():
    parser = argparse.ArgumentParser(description='Single Image Inference for DINOv3 Pose Estimation')

    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--model-name', type=str, default=None, help='DINOv3 model name (auto-read from config.yaml if not specified)')
    parser.add_argument('--image-size', type=int, default=None, help='Input image size (auto-read from config.yaml if not specified)')
    parser.add_argument('--heatmap-size', type=int, default=None, help='Output heatmap size (auto-read from config.yaml if not specified)')
    parser.add_argument('--pred-3d-source', type=str, default='fk', choices=['fk', 'fused'], help='Robot-frame 3D source to print/save')

    parser.add_argument('--image-path', type=str, default='', help='Path to input image')
    parser.add_argument('--json-path', type=str, default='',
                        help='Path to annotation JSON (auto-load image_path and camera_K if present)')
    parser.add_argument('--output-dir', type=str, default='./inference_output', help='Output directory for results')
    parser.add_argument('--camera-k', type=str, default='',
                        help="Optional camera intrinsics as 'fx,fy,cx,cy' to also print camera-frame 3D via PnP")
    parser.add_argument('--pnp-min-span-px', type=float, default=20.0,
                        help='Minimum x/y span (px) of selected 2D keypoints required for PnP')
    parser.add_argument('--pnp-min-area-ratio', type=float, default=0.001,
                        help='Minimum 2D bbox area ratio of selected points for PnP')

    parser.add_argument('--kp-min-confidence', type=float, default=0.0, help='Mask predicted 2D keypoints when sigmoid(max_heatmap_logit) is below this threshold')
    parser.add_argument('--kp-min-peak-logit', type=float, default=-1e9, help='Mask predicted 2D keypoints when heatmap peak logit is below this threshold')
    parser.add_argument('--fill-invalid-2d-with-fk-reproj', action='store_true',
                        help='After successful PnP, fill invalid/low-reliability 2D keypoints using FK reprojection')

    parser.add_argument('--save-heatmaps', action='store_true', default=True, help='Save heatmap visualizations')
    parser.add_argument('--save-combined', action='store_true', default=True, help='Save combined visualization')
    parser.add_argument('--no-heatmaps', action='store_false', dest='save_heatmaps', help='Disable heatmap visualization')
    parser.add_argument('--no-combined', action='store_false', dest='save_combined', help='Disable combined visualization')

    args = parser.parse_args()
    inference_single_image(args)


if __name__ == '__main__':
    main()
