"""
Heatmap Visualization: GT (left) vs Predicted (right) overlay.
Per-joint heatmaps + combined all-joints view.
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image as PILImage

TRAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../TRAIN'))
EVAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Eval'))
sys.path.insert(0, TRAIN_DIR)
sys.path.insert(0, EVAL_DIR)

from model import DINOv3PoseEstimator, soft_argmax_2d
from checkpoint_compat import load_checkpoint_compat
from inference_with_real import load_annotation

KEYPOINT_NAMES = ['panda_link0', 'panda_link2', 'panda_link3',
                  'panda_link4', 'panda_link6', 'panda_link7', 'panda_hand']
SKELETON = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
HEATMAP_SIZE = 512


def generate_gt_heatmaps(keypoints_2d, heatmap_size, orig_w, orig_h, sigma=5.0):
    """Generate Gaussian heatmaps from GT 2D keypoints (same as dataset.py)."""
    H = W = heatmap_size
    num_kp = len(keypoints_2d)
    heatmaps = np.zeros((num_kp, H, W), dtype=np.float32)

    sx, sy = W / orig_w, H / orig_h
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))

    for i, (x_orig, y_orig) in enumerate(keypoints_2d):
        if x_orig < 0 or y_orig < 0:
            continue
        x = x_orig * sx
        y = y_orig * sy
        if x < 0 or y < 0 or x >= W or y >= H:
            continue
        d2 = (xx - x) ** 2 + (yy - y) ** 2
        hm = np.exp(-d2 / (2 * sigma ** 2))
        hm[hm < 0.01] = 0
        heatmaps[i] = hm

    return heatmaps


def heatmap_overlay(img_bgr, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """Overlay a single-channel heatmap on a BGR image."""
    hm_uint8 = np.clip(heatmap * 255, 0, 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_uint8, colormap)
    if hm_color.shape[:2] != img_bgr.shape[:2]:
        hm_color = cv2.resize(hm_color, (img_bgr.shape[1], img_bgr.shape[0]))
    return cv2.addWeighted(img_bgr, 1 - alpha, hm_color, alpha, 0)


def draw_label(img, text, pos, color=(255, 255, 255), scale=0.5, thickness=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_2d_overlay(image_rgb, gt_2d, pred_2d, found):
    """Draw predicted 2D keypoints with skeleton on the original image."""
    image_bgr = cv2.cvtColor(np.array(image_rgb), cv2.COLOR_RGB2BGR)
    pred_line_color = (55, 110, 255)
    pred_joint_color = (0, 90, 255)
    shadow_color = (20, 20, 20)

    # Add a subtle darkened layer so overlays read better on bright images.
    overlay = image_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (image_bgr.shape[1], image_bgr.shape[0]), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.10, image_bgr, 0.90, 0.0, dst=image_bgr)

    line_overlay = image_bgr.copy()
    for j0, j1 in SKELETON:
        p0 = tuple(np.round(pred_2d[j0]).astype(int))
        p1 = tuple(np.round(pred_2d[j1]).astype(int))
        cv2.line(line_overlay, p0, p1, shadow_color, 6, cv2.LINE_AA)
        cv2.line(line_overlay, p0, p1, pred_line_color, 3, cv2.LINE_AA)
    cv2.addWeighted(line_overlay, 0.48, image_bgr, 0.52, 0.0, dst=image_bgr)

    joint_overlay = image_bgr.copy()
    for idx, name in enumerate(KEYPOINT_NAMES):
        pred_pt = tuple(np.round(pred_2d[idx]).astype(int))
        cv2.circle(joint_overlay, pred_pt, 8, shadow_color, -1, cv2.LINE_AA)
        cv2.circle(joint_overlay, pred_pt, 6, pred_joint_color, -1, cv2.LINE_AA)
        cv2.circle(joint_overlay, pred_pt, 6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.addWeighted(joint_overlay, 0.58, image_bgr, 0.42, 0.0, dst=image_bgr)

    for idx, name in enumerate(KEYPOINT_NAMES):
        pred_pt = tuple(np.round(pred_2d[idx]).astype(int))
        if name == "panda_link7":
            label_offset_x = -92
        elif name == "panda_link4":
            label_offset_x = -70
        else:
            label_offset_x = 10
        draw_label(
            image_bgr,
            name.replace("panda_", ""),
            (pred_pt[0] + label_offset_x, pred_pt[1] - 10),
            color=(255, 255, 255),
            scale=0.9,
            thickness=3,
        )

    return image_bgr


def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load annotation
    image_path, gt_2d, gt_3d, camera_K, found, gt_angles = load_annotation(
        args.json_path, KEYPOINT_NAMES
    )
    if image_path is None or not os.path.exists(image_path):
        print(f"ERROR: Image not found: {image_path}")
        return

    # Load model
    model = DINOv3PoseEstimator(
        dino_model_name=args.model_name,
        heatmap_size=HEATMAP_SIZE,
        unfreeze_blocks=0,
        fix_joint7_zero=args.fix_joint7
    ).to(device)
    load_checkpoint_compat(model, args.model_path, device, is_main_process=True)
    model.eval()

    # Preprocess image
    image_pil = PILImage.open(image_path).convert("RGB")
    orig_w, orig_h = image_pil.size

    transform = T.Compose([
        T.Resize((HEATMAP_SIZE, HEATMAP_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image_pil).unsqueeze(0).to(device)

    # Scale camera K
    camera_K_scaled_t = None
    if camera_K is not None:
        sx, sy = HEATMAP_SIZE / orig_w, HEATMAP_SIZE / orig_h
        K_scaled = camera_K.copy()
        K_scaled[0, 0] *= sx; K_scaled[1, 1] *= sy
        K_scaled[0, 2] *= sx; K_scaled[1, 2] *= sy
        camera_K_scaled_t = torch.tensor(K_scaled, dtype=torch.float32).unsqueeze(0).to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(image_tensor, camera_K=camera_K_scaled_t)

    pred_heatmaps = outputs['heatmaps_2d'][0].cpu().numpy()  # (7, H, W)
    pred_2d_hm = soft_argmax_2d(outputs['heatmaps_2d'])[0].cpu().numpy()  # (7, 2)
    pred_2d_orig = pred_2d_hm.copy()
    pred_2d_orig[:, 0] *= orig_w / HEATMAP_SIZE
    pred_2d_orig[:, 1] *= orig_h / HEATMAP_SIZE

    # GT heatmaps (Gaussian from GT 2D keypoints, same as training)
    gt_heatmaps = generate_gt_heatmaps(gt_2d, HEATMAP_SIZE, orig_w, orig_h, sigma=args.sigma)

    # Base image at heatmap resolution for overlay
    img_resized = cv2.resize(
        cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR),
        (HEATMAP_SIZE, HEATMAP_SIZE)
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # Combined all-joints (GT left, Pred right) - no text
    gt_combined = gt_heatmaps.sum(axis=0)
    gt_combined = gt_combined / (gt_combined.max() + 1e-8)
    pred_combined = pred_heatmaps.sum(axis=0)
    pred_combined = pred_combined / (pred_combined.max() + 1e-8)

    gt_all_overlay = heatmap_overlay(img_resized, gt_combined, alpha=0.3)
    pred_all_overlay = heatmap_overlay(img_resized, pred_combined, alpha=0.3)

    combined_panel = np.hstack([gt_all_overlay, pred_all_overlay])
    cv2.imwrite(os.path.join(args.output_dir, "heatmap_combined.png"), combined_panel)
    print(f"Saved: {os.path.join(args.output_dir, 'heatmap_combined.png')}")

    overlay_image = draw_2d_overlay(image_pil, gt_2d, pred_2d_orig, found)
    overlay_path = os.path.join(args.output_dir, "pred_2d_skeleton_overlay.png")
    cv2.imwrite(overlay_path, overlay_image)
    print(f"Saved: {overlay_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Heatmap GT vs Pred Visualization")
    parser.add_argument("--json-path", required=True, help="Path to annotation JSON")
    parser.add_argument("--model-path", required=True, help="Path to model checkpoint")
    parser.add_argument("--output-dir", default="./heatmap_output", help="Output directory")
    parser.add_argument("--model-name", default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    parser.add_argument("--fix-joint7", action="store_true", help="Fix joint7 to zero")
    parser.add_argument("--sigma", type=float, default=5.0, help="GT heatmap Gaussian sigma")
    parser.add_argument("--thumb-size", type=int, default=256, help="Per-joint thumbnail size")
    args = parser.parse_args()
    run(args)
