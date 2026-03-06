"""
Real Image Inference Script for DINOv3 Pose Estimation
- JSON annotation -> image path / GT auto-load
- 3D pose via PnP (camera frame) evaluation
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from PIL import Image as PILImage

import numpy as np
import torch
import torchvision.transforms as TVTransforms
import cv2

# Import model from Train directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../TRAIN')))
from model import DINOv3PoseEstimator, panda_forward_kinematics, soft_argmax_2d, solve_pnp_batch
from checkpoint_compat import load_checkpoint_compat


def load_annotation(json_path, keypoint_names):
    """JSON에서 이미지 경로 및 GT 정보 로드"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    image_path = None
    camera_K = None
    if 'meta' in data:
        raw_path = data['meta'].get('image_path', "")
        json_dir = os.path.dirname(os.path.abspath(json_path))

        if not os.path.isabs(raw_path):
            image_path = os.path.normpath(os.path.join(json_dir, raw_path))
        else:
            image_path = raw_path

        if 'K' in data['meta']:
            camera_K = np.array(data['meta']['K'], dtype=np.float64)

    # Keypoints extraction
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

    # Synthetic data uses cm, real uses m
    if 'syn' in json_path.lower():
        gt_3d /= 100.0

    gt_angles = None
    if 'sim_state' in data and 'joints' in data['sim_state']:
        gt_angles = np.array([j['position'] for j in data['sim_state']['joints'][:7]], dtype=np.float32)

    return image_path, gt_2d, gt_3d, camera_K, found, gt_angles


def run_inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"# Using device: {device}")

    keypoint_names = ['panda_link0', 'panda_link2', 'panda_link3', 'panda_link4', 'panda_link6', 'panda_link7', 'panda_hand']
    heatmap_size = 512

    # 1. Load Annotation
    print(f"# Loading annotation: {args.json_path}")
    image_path, gt_2d, gt_3d, camera_K, found, gt_angles = load_annotation(args.json_path, keypoint_names)

    if image_path is None or not os.path.exists(image_path):
        print(f"# ERROR: Image not found at: {image_path}")
        return

    # 2. Model Setup
    model = DINOv3PoseEstimator(
        dino_model_name=args.model_name,
        heatmap_size=heatmap_size,
        unfreeze_blocks=0,
        fix_joint7_zero=args.fix_joint7
    ).to(device)

    print(f"# Loading weights: {args.model_path}")
    load_checkpoint_compat(model, args.model_path, device, is_main_process=True)
    model.eval()

    # 3. Preprocess Image
    image_pil = PILImage.open(image_path).convert("RGB")
    orig_w, orig_h = image_pil.size

    transform = TVTransforms.Compose([
        TVTransforms.Resize((heatmap_size, heatmap_size)),
        TVTransforms.ToTensor(),
        TVTransforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image_pil).unsqueeze(0).to(device)

    # 4. Scale camera_K from original resolution to heatmap resolution
    camera_K_scaled = None
    if camera_K is not None:
        scale_x = heatmap_size / orig_w
        scale_y = heatmap_size / orig_h
        camera_K_scaled = camera_K.copy()
        camera_K_scaled[0, 0] *= scale_x  # fx
        camera_K_scaled[1, 1] *= scale_y  # fy
        camera_K_scaled[0, 2] *= scale_x  # cx
        camera_K_scaled[1, 2] *= scale_y  # cy
        camera_K_scaled = torch.tensor(camera_K_scaled, dtype=torch.float32).unsqueeze(0).to(device)

    # 5. Forward Pass
    print("# Running model forward pass...")
    with torch.no_grad():
        outputs = model(image_tensor, camera_K=camera_K_scaled)

    # 6. Extract Results
    pred_heatmaps = outputs["heatmaps_2d"]
    pred_angles = outputs["joint_angles"][0].cpu().numpy()
    pred_3d_robot = outputs["keypoints_3d_robot"][0].cpu().numpy()

    # 2D keypoints via soft_argmax (not argmax)
    pred_2d_hm = soft_argmax_2d(pred_heatmaps)[0].cpu().numpy()  # (N, 2) in heatmap space

    # Scale 2D back to original resolution
    pred_2d_orig = pred_2d_hm.copy()
    pred_2d_orig[:, 0] *= (orig_w / heatmap_size)
    pred_2d_orig[:, 1] *= (orig_h / heatmap_size)

    # Camera-frame 3D (from PnP)
    pred_3d_cam = None
    pnp_valid = False
    reproj_err = None
    if 'keypoints_3d_cam' in outputs:
        pred_3d_cam = outputs['keypoints_3d_cam'][0].cpu().numpy()
        pnp_valid = outputs['pnp_valid'][0].item()
        if 'reproj_errors' in outputs:
            reproj_err = outputs['reproj_errors'][0].item()

    # RANSAC EPnP
    pred_3d_cam_ransac = None
    ransac_valid = False
    ransac_reproj = None
    ransac_n_inliers = 0
    if 'keypoints_3d_cam_ransac' in outputs:
        pred_3d_cam_ransac = outputs['keypoints_3d_cam_ransac'][0].cpu().numpy()
        ransac_valid = outputs['pnp_valid_ransac'][0].item()
        if 'reproj_errors_ransac' in outputs:
            ransac_reproj = outputs['reproj_errors_ransac'][0].item()
        if 'pnp_n_inliers_ransac' in outputs:
            ransac_n_inliers = outputs['pnp_n_inliers_ransac'][0].item()

    # 7. Quantitative Report
    print("\n" + "=" * 60)
    print("  INFERENCE RESULTS")
    print("=" * 60)

    if gt_angles is not None:
        angle_diff = pred_angles - gt_angles
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
        angle_errs = np.abs(np.degrees(angle_diff))
        print(f"\n  [Joint Angles]")
        print(f"  Mean Angle Error: {np.mean(angle_errs):.2f} deg")
        for i in range(7):
            print(f"    J{i}: Pred={np.degrees(pred_angles[i]):7.2f}, GT={np.degrees(gt_angles[i]):7.2f}, Err={angle_errs[i]:6.2f}")

    # 2D Error
    dist_2d = np.linalg.norm(pred_2d_orig - gt_2d, axis=1)
    print(f"\n  [2D Keypoints]")
    print(f"  Mean 2D Error: {np.mean(dist_2d):.2f} px")
    for i in range(len(keypoint_names)):
        status = "OK" if found[i] else "MISSING"
        print(f"    J{i}: err={dist_2d[i]:.1f}px  ({status})")

    # 3D Error (camera frame) - Iterative PnP
    if pred_3d_cam is not None and np.any(gt_3d):
        dist_3d = np.linalg.norm(pred_3d_cam - gt_3d, axis=1) * 1000  # m -> mm
        reproj_str = f", reproj={reproj_err:.2f}px" if reproj_err is not None else ""
        print(f"\n  [3D Iterative PnP] (valid: {pnp_valid}{reproj_str})")
        print(f"  Mean 3D Error: {np.mean(dist_3d):.2f} mm")
        for i in range(len(keypoint_names)):
            print(f"    J{i}: err={dist_3d[i]:.1f}mm")

    # 3D Error (camera frame) - RANSAC EPnP
    if pred_3d_cam_ransac is not None and np.any(gt_3d):
        dist_3d_r = np.linalg.norm(pred_3d_cam_ransac - gt_3d, axis=1) * 1000
        reproj_r_str = f", reproj={ransac_reproj:.2f}px" if ransac_reproj is not None else ""
        print(f"\n  [3D RANSAC EPnP] (valid: {ransac_valid}, inliers: {ransac_n_inliers}/7{reproj_r_str})")
        print(f"  Mean 3D Error: {np.mean(dist_3d_r):.2f} mm")
        for i in range(len(keypoint_names)):
            print(f"    J{i}: err={dist_3d_r[i]:.1f}mm")

    print("=" * 60)

    # 8. Visualization
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

        for i in range(len(keypoint_names)):
            if found[i]:  # GT: Green
                cv2.circle(img_cv, (int(gt_2d[i, 0]), int(gt_2d[i, 1])), 5, (0, 255, 0), -1)
            # Pred: Red
            cv2.circle(img_cv, (int(pred_2d_orig[i, 0]), int(pred_2d_orig[i, 1])), 4, (0, 0, 255), -1)
            cv2.putText(img_cv, keypoint_names[i].split('_')[-1],
                        (int(pred_2d_orig[i, 0]) + 5, int(pred_2d_orig[i, 1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        out_file = os.path.join(args.output_dir, "inference_overlay.png")
        cv2.imwrite(out_file, img_cv)
        print(f"\n# Visualization saved to: {out_file}")

    # 9. Save metrics JSON
    if args.output_dir:
        metrics = {
            "pred_angles_deg": np.degrees(pred_angles).tolist(),
            "pred_2d": pred_2d_orig.tolist(),
            "mean_2d_error_px": float(np.mean(dist_2d)),
        }
        if gt_angles is not None:
            metrics["gt_angles_deg"] = np.degrees(gt_angles).tolist()
            metrics["angle_errors_deg"] = angle_errs.tolist()
            metrics["mean_angle_error_deg"] = float(np.mean(angle_errs))
        if pred_3d_cam is not None and np.any(gt_3d):
            metrics["mean_3d_error_mm"] = float(np.mean(dist_3d))
            metrics["per_joint_3d_error_mm"] = dist_3d.tolist()
            metrics["pnp_valid"] = bool(pnp_valid)

        with open(os.path.join(args.output_dir, "metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--json-path", required=True)
    parser.add_argument("-p", "--model-path", required=True)
    parser.add_argument("-o", "--output-dir", default="./real_inference_output")
    parser.add_argument("--model-name", default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    parser.add_argument("--fix-joint7", action="store_true")
    args = parser.parse_args()
    run_inference(args)
