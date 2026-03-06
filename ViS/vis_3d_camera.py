"""
Visualize GT vs predicted 3D keypoints in camera coordinates.
Supports single JSON mode and one-batch folder mode.
"""

import argparse
import math
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image as PILImage
from torch.utils.data import DataLoader

TRAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../TRAIN"))
EVAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../Eval"))
sys.path.insert(0, TRAIN_DIR)
sys.path.insert(0, EVAL_DIR)

from checkpoint_compat import load_checkpoint_compat
from dataset import PoseEstimationDataset
from inference_with_real import load_annotation
from model import DINOv3PoseEstimator, panda_forward_kinematics, soft_argmax_2d, solve_pnp_batch


KEYPOINT_NAMES = [
    "panda_link0",
    "panda_link2",
    "panda_link3",
    "panda_link4",
    "panda_link6",
    "panda_link7",
    "panda_hand",
]
DATASET_KEYPOINT_NAMES = ["link0", "link2", "link3", "link4", "link6", "link7", "hand"]
SKELETON = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
HEATMAP_SIZE = 512


def scale_camera_k(camera_k, orig_w, orig_h, target_size):
    if torch.is_tensor(camera_k):
        camera_k = camera_k.detach().cpu().numpy()
    sx = target_size / float(orig_w)
    sy = target_size / float(orig_h)
    camera_k_scaled = camera_k.copy()
    camera_k_scaled[0, 0] *= sx
    camera_k_scaled[1, 1] *= sy
    camera_k_scaled[0, 2] *= sx
    camera_k_scaled[1, 2] *= sy
    return camera_k_scaled


def draw_2d_overlay(image_rgb, gt_2d, pred_2d, found):
    image_bgr = cv2.cvtColor(np.array(image_rgb), cv2.COLOR_RGB2BGR)
    for j0, j1 in SKELETON:
        if found[j0] and found[j1]:
            pt0 = tuple(np.round(gt_2d[j0]).astype(int))
            pt1 = tuple(np.round(gt_2d[j1]).astype(int))
            cv2.line(image_bgr, pt0, pt1, (0, 200, 0), 2, cv2.LINE_AA)

        p0 = tuple(np.round(pred_2d[j0]).astype(int))
        p1 = tuple(np.round(pred_2d[j1]).astype(int))
        cv2.line(image_bgr, p0, p1, (0, 0, 220), 2, cv2.LINE_AA)

    for idx, name in enumerate(KEYPOINT_NAMES):
        if found[idx]:
            gt_pt = tuple(np.round(gt_2d[idx]).astype(int))
            cv2.circle(image_bgr, gt_pt, 5, (0, 255, 0), -1, cv2.LINE_AA)
        pred_pt = tuple(np.round(pred_2d[idx]).astype(int))
        cv2.circle(image_bgr, pred_pt, 4, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(
            image_bgr,
            name.split("_")[-1],
            (pred_pt[0] + 6, pred_pt[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def set_equal_axes(ax, points):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max((maxs - mins).max() / 2.0, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def plot_skeleton_3d(ax, pts, color, linestyle, label):
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=color, s=28, label=label)
    for j0, j1 in SKELETON:
        seg = pts[[j0, j1]]
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=color, linestyle=linestyle, linewidth=2)


def format_metrics(pred_3d, gt_3d, found, pnp_valid, reproj_error, title="Camera-frame"):
    valid_mask = np.array(found, dtype=bool)
    if not valid_mask.any():
        valid_mask = np.ones(len(KEYPOINT_NAMES), dtype=bool)

    err_mm = np.linalg.norm(pred_3d - gt_3d, axis=1) * 1000.0
    valid_err = err_mm[valid_mask]
    lines = [
        f"{title}",
        "-" * len(title),
        f"PnP valid: {bool(pnp_valid)}",
        f"Reproj error: {reproj_error:.2f}px" if reproj_error is not None else "Reproj error: N/A",
        f"Mean 3D error: {valid_err.mean():.2f} mm",
        f"Median 3D error: {np.median(valid_err):.2f} mm",
        "",
    ]
    for i, name in enumerate(KEYPOINT_NAMES):
        status = "OK" if found[i] else "MISS"
        lines.append(f"{name.split('_')[-1]:>5}: {err_mm[i]:6.1f} mm [{status}]")
    return "\n".join(lines), err_mm


def format_fk_metrics(pred_fk, gt_fk, title="FK robot-frame"):
    err_mm = np.linalg.norm(pred_fk - gt_fk, axis=1) * 1000.0
    lines = [
        f"{title}",
        "-" * len(title),
        f"Mean 3D error: {err_mm.mean():.2f} mm",
        f"Median 3D error: {np.median(err_mm):.2f} mm",
        "",
    ]
    for i, name in enumerate(KEYPOINT_NAMES):
        lines.append(f"{name.split('_')[-1]:>5}: {err_mm[i]:6.1f} mm")
    return "\n".join(lines), err_mm


def format_angle_metrics(pred_angles, gt_angles):
    angle_diff = pred_angles - gt_angles
    angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
    err_deg = np.abs(np.degrees(angle_diff))
    lines = [
        "Joint-angle",
        "-----------",
        f"Mean angle error: {err_deg.mean():.2f} deg",
        f"Median angle error: {np.median(err_deg):.2f} deg",
        "",
    ]
    for i in range(len(err_deg)):
        lines.append(
            f"J{i}: pred={math.degrees(pred_angles[i]):7.2f} deg, "
            f"gt={math.degrees(gt_angles[i]):7.2f} deg, err={err_deg[i]:6.2f} deg"
        )
    return "\n".join(lines), err_deg


def format_oracle_pnp_metrics(pred_2d_hm, gt_fk, gt_3d_cam, camera_k_scaled, found):
    uv_2d = torch.from_numpy(pred_2d_hm).float().unsqueeze(0)
    gt_fk_t = torch.from_numpy(gt_fk).float().unsqueeze(0)
    camera_k_t = torch.from_numpy(camera_k_scaled).float().unsqueeze(0)

    kp_3d_cam_t, pnp_valid_t, reproj_errors_t = solve_pnp_batch(uv_2d, gt_fk_t, camera_k_t)
    kp_3d_cam = kp_3d_cam_t[0].cpu().numpy()
    pnp_valid = bool(pnp_valid_t[0].item())
    reproj_error = float(reproj_errors_t[0].item())

    valid_mask = np.array(found, dtype=bool)
    if not valid_mask.any():
        valid_mask = np.ones(len(KEYPOINT_NAMES), dtype=bool)

    err_mm = np.linalg.norm(kp_3d_cam - gt_3d_cam, axis=1) * 1000.0
    valid_err = err_mm[valid_mask]
    lines = [
        "Oracle PnP: Pred2D + GTFK3D",
        "---------------------------",
        f"PnP valid: {pnp_valid}",
        f"Reproj error: {reproj_error:.2f}px",
        f"Mean 3D error: {valid_err.mean():.2f} mm",
        f"Median 3D error: {np.median(valid_err):.2f} mm",
        "",
    ]
    for i, name in enumerate(KEYPOINT_NAMES):
        status = "OK" if found[i] else "MISS"
        lines.append(f"{name.split('_')[-1]:>5}: {err_mm[i]:6.1f} mm [{status}]")
    return "\n".join(lines), err_mm, pnp_valid, reproj_error


def save_figure(
    output_path,
    image_rgb,
    pred_2d_orig,
    gt_2d,
    found,
    gt_3d_cam,
    pred_3d_cam,
    camera_metrics_text,
    title_suffix,
    gt_3d_fk=None,
    pred_3d_fk=None,
    fk_metrics_text=None,
):
    fig = plt.figure(figsize=(18, 12), dpi=150)

    ax_img = fig.add_subplot(2, 3, 1)
    ax_img.imshow(draw_2d_overlay(image_rgb, gt_2d, pred_2d_orig, found))
    ax_img.set_title("2D overlay: GT(green) vs Pred(red)")
    ax_img.axis("off")

    ax_cam = fig.add_subplot(2, 3, 2, projection="3d")
    plot_skeleton_3d(ax_cam, gt_3d_cam, color="green", linestyle="-", label="GT")
    plot_skeleton_3d(ax_cam, pred_3d_cam, color="red", linestyle="--", label="Pred")
    set_equal_axes(ax_cam, np.concatenate([gt_3d_cam, pred_3d_cam], axis=0))
    ax_cam.set_title(f"Camera-frame 3D ({title_suffix})")
    ax_cam.set_xlabel("X (m)")
    ax_cam.set_ylabel("Y (m)")
    ax_cam.set_zlabel("Z (m)")
    ax_cam.legend()

    ax_cam_top = fig.add_subplot(2, 3, 3)
    ax_cam_top.plot(gt_3d_cam[:, 0], gt_3d_cam[:, 2], "go-", linewidth=2, markersize=5, label="GT")
    ax_cam_top.plot(pred_3d_cam[:, 0], pred_3d_cam[:, 2], "ro--", linewidth=2, markersize=5, label="Pred")
    ax_cam_top.set_title("Camera view (X-Z)")
    ax_cam_top.set_xlabel("X (m)")
    ax_cam_top.set_ylabel("Z (m)")
    ax_cam_top.grid(True, alpha=0.3)
    ax_cam_top.axis("equal")
    ax_cam_top.legend()

    ax_cam_txt = fig.add_subplot(2, 3, 4)
    ax_cam_txt.axis("off")
    ax_cam_txt.text(0.0, 1.0, camera_metrics_text, va="top", ha="left", family="monospace", fontsize=10)

    ax_fk = fig.add_subplot(2, 3, 5, projection="3d")
    if gt_3d_fk is not None and pred_3d_fk is not None:
        plot_skeleton_3d(ax_fk, gt_3d_fk, color="green", linestyle="-", label="GT")
        plot_skeleton_3d(ax_fk, pred_3d_fk, color="red", linestyle="--", label="Pred")
        set_equal_axes(ax_fk, np.concatenate([gt_3d_fk, pred_3d_fk], axis=0))
        ax_fk.legend()
    else:
        ax_fk.text2D(0.15, 0.5, "GT joint angles not available\nFK comparison skipped", transform=ax_fk.transAxes)
    ax_fk.set_title("FK robot-frame 3D")
    ax_fk.set_xlabel("X (m)")
    ax_fk.set_ylabel("Y (m)")
    ax_fk.set_zlabel("Z (m)")

    ax_fk_txt = fig.add_subplot(2, 3, 6)
    ax_fk_txt.axis("off")
    if fk_metrics_text is not None:
        ax_fk_txt.text(0.0, 1.0, fk_metrics_text, va="top", ha="left", family="monospace", fontsize=10)
    else:
        ax_fk_txt.text(0.0, 1.0, "FK robot-frame\n--------------\nGT joint angles not available", va="top", ha="left", family="monospace", fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_model(args, device):
    model = DINOv3PoseEstimator(
        dino_model_name=args.model_name,
        heatmap_size=HEATMAP_SIZE,
        unfreeze_blocks=0,
        fix_joint7_zero=args.fix_joint7,
    ).to(device)
    load_checkpoint_compat(model, args.model_path, device, is_main_process=True)
    model.eval()
    return model


def process_single_sample(
    args,
    image_path,
    image_pil,
    gt_2d,
    gt_3d,
    camera_k,
    found,
    gt_angles,
    outputs,
    output_name=None,
):
    orig_w, orig_h = image_pil.size
    pred_2d_hm = soft_argmax_2d(outputs["heatmaps_2d"])[0].cpu().numpy()
    pred_2d_orig = pred_2d_hm.copy()
    pred_2d_orig[:, 0] *= orig_w / HEATMAP_SIZE
    pred_2d_orig[:, 1] *= orig_h / HEATMAP_SIZE

    pred_3d = outputs[args.pred_key][0].cpu().numpy()
    pred_angles = outputs["joint_angles"][0].cpu().numpy()

    pnp_valid_key = {
        "keypoints_3d_cam": "pnp_valid",
        "keypoints_3d_cam_ransac": "pnp_valid_ransac",
        "keypoints_3d_cam_conf": "pnp_valid_conf",
    }[args.pred_key]
    reproj_key = {
        "keypoints_3d_cam": "reproj_errors",
        "keypoints_3d_cam_ransac": "reproj_errors_ransac",
        "keypoints_3d_cam_conf": "reproj_errors_conf",
    }[args.pred_key]

    pnp_valid = outputs[pnp_valid_key][0].item()
    reproj_error = outputs[reproj_key][0].item() if reproj_key in outputs else None

    camera_metrics_text, err_mm = format_metrics(
        pred_3d, gt_3d, found, pnp_valid, reproj_error, title="Camera-frame"
    )

    pred_fk = outputs["keypoints_3d_fk"][0].cpu().numpy()
    gt_fk = None
    fk_metrics_text = None
    fk_err_mm = None
    angle_metrics_text = None
    angle_err_deg = None
    oracle_pnp_text = None
    oracle_err_mm = None
    if gt_angles is not None:
        gt_angles_eval = gt_angles.copy()
        if args.fix_joint7 and gt_angles_eval.shape[0] >= 7:
            gt_angles_eval[6] = 0.0
        angle_metrics_text, angle_err_deg = format_angle_metrics(pred_angles, gt_angles_eval)

        gt_angles_t = torch.tensor(gt_angles, dtype=torch.float32).unsqueeze(0)
        if args.fix_joint7 and gt_angles_t.shape[1] >= 7:
            gt_angles_t = gt_angles_t.clone()
            gt_angles_t[:, 6] = 0.0
        gt_fk = panda_forward_kinematics(gt_angles_t)[0].cpu().numpy()
        fk_metrics_text, fk_err_mm = format_fk_metrics(pred_fk, gt_fk)
        camera_k_scaled = scale_camera_k(camera_k, orig_w, orig_h, HEATMAP_SIZE)
        oracle_pnp_text, oracle_err_mm, _, _ = format_oracle_pnp_metrics(
            pred_2d_hm, gt_fk, gt_3d, camera_k_scaled, found
        )

    os.makedirs(args.output_dir, exist_ok=True)
    final_name = output_name or args.output_name
    out_path = os.path.join(args.output_dir, f"{final_name}.png")
    save_figure(
        out_path,
        image_pil,
        pred_2d_orig,
        gt_2d,
        found,
        gt_3d,
        pred_3d,
        camera_metrics_text,
        args.pred_key.replace("keypoints_", ""),
        gt_3d_fk=gt_fk,
        pred_3d_fk=pred_fk,
        fk_metrics_text=fk_metrics_text,
    )

    print(f"Saved: {out_path}")
    print(f"Source image: {image_path}")
    print("")
    print(camera_metrics_text)
    print("")
    print("Per-joint 3D error [mm]:")
    for i, name in enumerate(KEYPOINT_NAMES):
        print(f"  {name}: {err_mm[i]:.2f}")
    if angle_metrics_text is not None and angle_err_deg is not None:
        print("")
        print(angle_metrics_text)
        print("")
        print("Per-joint angle error [deg]:")
        for i in range(len(angle_err_deg)):
            print(f"  joint_{i}: {angle_err_deg[i]:.2f}")
    if fk_metrics_text is not None and fk_err_mm is not None:
        print("")
        print(fk_metrics_text)
        print("")
        print("Per-joint FK 3D error [mm]:")
        for i, name in enumerate(KEYPOINT_NAMES):
            print(f"  {name}: {fk_err_mm[i]:.2f}")
    if oracle_pnp_text is not None and oracle_err_mm is not None:
        print("")
        print(oracle_pnp_text)
        print("")
        print("Per-joint Oracle-PnP 3D error [mm]:")
        for i, name in enumerate(KEYPOINT_NAMES):
            print(f"  {name}: {oracle_err_mm[i]:.2f}")


def run_json_mode(args, model, device):
    image_path, gt_2d, gt_3d, camera_k, found, gt_angles = load_annotation(args.json_path, KEYPOINT_NAMES)
    if image_path is None or not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if camera_k is None:
        raise ValueError("Camera intrinsic matrix K is required for camera-frame 3D visualization.")

    image_pil = PILImage.open(image_path).convert("RGB")
    orig_w, orig_h = image_pil.size
    transform = T.Compose([
        T.Resize((HEATMAP_SIZE, HEATMAP_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = transform(image_pil).unsqueeze(0).to(device)

    camera_k_scaled = scale_camera_k(camera_k, orig_w, orig_h, HEATMAP_SIZE)
    camera_k_scaled_t = torch.tensor(camera_k_scaled, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor, camera_K=camera_k_scaled_t)

    process_single_sample(args, image_path, image_pil, gt_2d, gt_3d, camera_k, found, gt_angles, outputs)


def run_batch_mode(args, model, device):
    dataset = PoseEstimationDataset(
        data_dir=args.data_dir,
        keypoint_names=DATASET_KEYPOINT_NAMES,
        image_size=(HEATMAP_SIZE, HEATMAP_SIZE),
        heatmap_size=(HEATMAP_SIZE, HEATMAP_SIZE),
        augment=False,
        include_angles=True,
    )
    if len(dataset) == 0:
        raise ValueError(f"No samples found in data dir: {args.data_dir}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    batch = next(iter(loader))

    imgs = batch["image"].to(device)
    camera_k = batch["camera_K"].to(device)
    original_size = batch["original_size"].to(device)
    gt_3d = batch["keypoints_3d"].to(device)
    valid_mask = batch["valid_mask"].to(device)

    scale_x = HEATMAP_SIZE / original_size[:, 0]
    scale_y = HEATMAP_SIZE / original_size[:, 1]
    camera_k_scaled = camera_k.clone()
    camera_k_scaled[:, 0, 0] *= scale_x
    camera_k_scaled[:, 1, 1] *= scale_y
    camera_k_scaled[:, 0, 2] *= scale_x
    camera_k_scaled[:, 1, 2] *= scale_y

    with torch.no_grad():
        outputs = model(imgs, camera_K=camera_k_scaled)

    pnp_valid_key = {
        "keypoints_3d_cam": "pnp_valid",
        "keypoints_3d_cam_ransac": "pnp_valid_ransac",
        "keypoints_3d_cam_conf": "pnp_valid_conf",
    }[args.pred_key]
    reproj_key = {
        "keypoints_3d_cam": "reproj_errors",
        "keypoints_3d_cam_ransac": "reproj_errors_ransac",
        "keypoints_3d_cam_conf": "reproj_errors_conf",
    }[args.pred_key]

    pnp_valid = outputs[pnp_valid_key]
    if pnp_valid.dim() > 1:
        pnp_valid = pnp_valid.all(dim=1)
    valid_samples = valid_mask.all(dim=1)
    combined_mask = valid_samples & pnp_valid
    n_total = imgs.shape[0]
    n_filtered = int(combined_mask.sum().item())
    n_valid_only = int(valid_samples.sum().item())
    n_pnp_only = int(pnp_valid.sum().item())

    reproj_errors = outputs[reproj_key]
    print("")
    print("=" * 60)
    print(f"3D POSE ERROR - PnP FILTERED ({n_filtered}/{n_total}, reproj<5px & depth OK)")
    print("=" * 60)
    print(f"  valid_mask(all joints present): {n_valid_only}/{n_total}")
    print(f"  pnp_valid: {n_pnp_only}/{n_total}")

    selection_reason = "filtered"
    if n_filtered > 0:
        candidate_mask = combined_mask
        valid_reproj = reproj_errors[combined_mask]
        print(f"  Reproj RMSE: mean={valid_reproj.mean().item():.2f}px, min={valid_reproj.min().item():.2f}px")
    else:
        print("  No filtered-valid sample found in the first batch.")
        finite_reproj = torch.isfinite(reproj_errors)
        if (valid_samples & finite_reproj).any():
            candidate_mask = valid_samples & finite_reproj
            selection_reason = "valid_only_fallback"
        elif (pnp_valid & finite_reproj).any():
            candidate_mask = pnp_valid & finite_reproj
            selection_reason = "pnp_only_fallback"
        elif finite_reproj.any():
            candidate_mask = finite_reproj
            selection_reason = "finite_reproj_fallback"
        else:
            candidate_mask = torch.ones_like(pnp_valid, dtype=torch.bool)
            selection_reason = "any_sample_fallback"
        print(f"  Fallback selection mode: {selection_reason}")
        print(f"  Batch reproj min={reproj_errors.min().item():.2f}px, mean={reproj_errors.mean().item():.2f}px")

    pred_cam = outputs[args.pred_key]
    kp_error_mm = torch.norm(pred_cam - gt_3d, dim=2) * 1000.0
    candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).squeeze(1)
    best_local = int(torch.argmin(reproj_errors[candidate_indices]).item())
    best_idx = int(candidate_indices[best_local].item())

    best_name = batch["name"][best_idx]
    best_ann_path = batch["annotation_path"][best_idx]
    best_reproj = reproj_errors[best_idx].item()
    best_mean_3d = kp_error_mm[best_idx].mean().item()
    print(f"  Selected sample: idx={best_idx}, name={best_name}")
    print(f"  Selection mode: {selection_reason}")
    print(f"  Annotation: {best_ann_path}")
    print(f"  Selected reproj={best_reproj:.2f}px, mean3d={best_mean_3d:.2f}mm")
    print("=" * 60)
    print("")

    image_path, gt_2d, gt_3d_np, _, found, gt_angles = load_annotation(best_ann_path, KEYPOINT_NAMES)
    if image_path is None or not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    image_pil = PILImage.open(image_path).convert("RGB")

    single_outputs = {}
    for key, value in outputs.items():
        if torch.is_tensor(value):
            single_outputs[key] = value[best_idx:best_idx + 1]
        else:
            single_outputs[key] = value

    process_single_sample(
        args,
        image_path,
        image_pil,
        gt_2d,
        gt_3d_np,
        camera_k,
        found,
        gt_angles,
        single_outputs,
        output_name=f"{args.output_name}_{best_name}",
    )


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args, device)
    if args.data_dir:
        run_batch_mode(args, model, device)
    else:
        run_json_mode(args, model, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize GT vs Pred 3D camera-frame keypoints")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--json-path", help="Path to annotation JSON")
    mode_group.add_argument("--data-dir", help="Dataset directory to load one batch from")
    parser.add_argument("--model-path", required=True, help="Path to model checkpoint")
    parser.add_argument("--output-dir", default="./camera_3d_output", help="Output directory")
    parser.add_argument("--output-name", default="camera_3d_comparison", help="Saved image filename without extension")
    parser.add_argument("--model-name", default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    parser.add_argument("--fix-joint7", action="store_true", help="Fix joint7 to zero")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for data-dir mode")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers for data-dir mode")
    parser.add_argument(
        "--pred-key",
        default="keypoints_3d_cam",
        choices=["keypoints_3d_cam", "keypoints_3d_cam_ransac", "keypoints_3d_cam_conf"],
        help="Which camera-frame prediction to visualize",
    )
    args = parser.parse_args()
    run(args)
