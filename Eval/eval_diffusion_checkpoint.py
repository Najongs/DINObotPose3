#!/usr/bin/env python3
"""Quantitative + qualitative evaluation for diffusion angle checkpoints."""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../TRAIN")))

from dataset import PoseEstimationDataset
from model import panda_forward_kinematics, soft_argmax_2d
from model_diffusion import DINOv3DiffusionPoseEstimator


PANDA_JOINT_MEAN = torch.tensor([-5.22e-02, 2.68e-01, 6.04e-03, -2.01e+00, 1.49e-02, 1.99e+00, 0.0])
PANDA_JOINT_STD = torch.tensor([1.025, 0.645, 0.511, 0.508, 0.769, 0.511, 1.0])
KEYPOINT_NAMES = ["link0", "link2", "link3", "link4", "link6", "link7", "hand"]


def denormalize_angles(angles_norm: torch.Tensor) -> torch.Tensor:
    mean = PANDA_JOINT_MEAN[: angles_norm.shape[-1]].to(angles_norm.device)
    std = PANDA_JOINT_STD[: angles_norm.shape[-1]].to(angles_norm.device)
    return angles_norm * std + mean


def load_diffusion_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> None:
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)


def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    mean = torch.tensor([0.485, 0.456, 0.406], device=image_tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=image_tensor.device).view(3, 1, 1)
    image = (image_tensor * std + mean).clamp(0.0, 1.0)
    image = (image.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(image)


def draw_points(draw: ImageDraw.ImageDraw, points: np.ndarray, color: tuple[int, int, int], radius: int = 4) -> None:
    for idx, (x, y) in enumerate(points):
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=2)
        draw.text((x + radius + 1, y - radius - 1), str(idx + 1), fill=color)


def scale_camera_k(camera_k: np.ndarray, original_size: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    orig_w, orig_h = float(original_size[0]), float(original_size[1])
    scale_x = target_w / max(orig_w, 1.0)
    scale_y = target_h / max(orig_h, 1.0)
    scaled = camera_k.copy().astype(np.float64)
    scaled[0, 0] *= scale_x
    scaled[1, 1] *= scale_y
    scaled[0, 2] *= scale_x
    scaled[1, 2] *= scale_y
    return scaled


def solve_pnp_pose(points_3d_robot: np.ndarray, points_2d: np.ndarray, valid_mask: np.ndarray,
                   camera_k: np.ndarray) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    valid_idx = np.where(valid_mask.astype(bool))[0]
    if len(valid_idx) < 4:
        return None, None

    pts3d = points_3d_robot[valid_idx].astype(np.float64)
    pts2d = points_2d[valid_idx].astype(np.float64)

    try:
        ok, rvec, tvec = cv2.solvePnP(
            pts3d,
            pts2d,
            camera_k,
            None,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except cv2.error:
        return None, None

    if not ok:
        return None, None
    return rvec, tvec


def project_robot_points(points_3d_robot: np.ndarray, rvec: np.ndarray, tvec: np.ndarray,
                         camera_k: np.ndarray) -> np.ndarray:
    proj, _ = cv2.projectPoints(
        points_3d_robot.astype(np.float64),
        rvec,
        tvec,
        camera_k.astype(np.float64),
        None,
    )
    return proj.reshape(-1, 2).astype(np.float32)


def build_vis_image(sample: dict, pred_uv: np.ndarray, pred_angles_deg: np.ndarray, gt_angles_deg: np.ndarray,
                    frame_angle_mae: float, frame_fk_mm: float, heatmap_size: tuple[int, int],
                    gt_reproj_uv: np.ndarray | None = None, pred_fk_reproj_uv: np.ndarray | None = None,
                    gt_fk_reproj_uv: np.ndarray | None = None) -> Image.Image:
    img = tensor_to_pil(sample["image"])
    draw = ImageDraw.Draw(img)

    img_w, img_h = img.size
    hm_h, hm_w = heatmap_size
    scale_x = img_w / float(hm_w)
    scale_y = img_h / float(hm_h)

    gt_uv = sample["keypoints"].cpu().numpy().copy()
    pred_uv_scaled = pred_uv.copy()
    gt_uv[:, 0] *= scale_x
    gt_uv[:, 1] *= scale_y
    pred_uv_scaled[:, 0] *= scale_x
    pred_uv_scaled[:, 1] *= scale_y

    draw_points(draw, gt_uv, (255, 64, 64))
    draw_points(draw, pred_uv_scaled, (64, 255, 64))
    if gt_reproj_uv is not None:
        gt_reproj_scaled = gt_reproj_uv.copy()
        gt_reproj_scaled[:, 0] *= scale_x
        gt_reproj_scaled[:, 1] *= scale_y
        draw_points(draw, gt_reproj_scaled, (255, 200, 0), radius=3)
    if gt_fk_reproj_uv is not None:
        gt_fk_scaled = gt_fk_reproj_uv.copy()
        gt_fk_scaled[:, 0] *= scale_x
        gt_fk_scaled[:, 1] *= scale_y
        draw_points(draw, gt_fk_scaled, (255, 128, 255), radius=3)
    if pred_fk_reproj_uv is not None:
        pred_fk_scaled = pred_fk_reproj_uv.copy()
        pred_fk_scaled[:, 0] *= scale_x
        pred_fk_scaled[:, 1] *= scale_y
        draw_points(draw, pred_fk_scaled, (64, 200, 255), radius=3)

    panel_h = 150
    canvas = Image.new("RGB", (img_w, img_h + panel_h), color=(18, 18, 18))
    canvas.paste(img, (0, 0))
    panel = ImageDraw.Draw(canvas)
    y = img_h + 10
    panel.text((10, y), f"name: {sample['name']}", fill=(240, 240, 240))
    y += 20
    panel.text((10, y), f"angle_mae_deg: {frame_angle_mae:.2f}", fill=(240, 240, 240))
    y += 20
    panel.text((10, y), f"fk_mean_mm: {frame_fk_mm:.2f}", fill=(240, 240, 240))
    y += 24
    panel.text((10, y), "GT angle deg:", fill=(255, 128, 128))
    panel.text((180, y), " ".join(f"{x:.1f}" for x in gt_angles_deg[:6]), fill=(255, 128, 128))
    y += 20
    panel.text((10, y), "PR angle deg:", fill=(128, 255, 128))
    panel.text((180, y), " ".join(f"{x:.1f}" for x in pred_angles_deg[:6]), fill=(128, 255, 128))
    y += 20
    panel.text((10, y), "Legend: GT2D=red Pred2D=green GT3Dproj=yellow GTFKproj=magenta PredFKproj=cyan", fill=(220, 220, 220))
    return canvas


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"# Using device: {device}")

    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "best_diffusion.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    qual_dir = output_dir / "qualitative"
    qual_dir.mkdir(parents=True, exist_ok=True)

    dataset = PoseEstimationDataset(
        args.data_dir,
        keypoint_names=KEYPOINT_NAMES,
        image_size=(args.image_size, args.image_size),
        heatmap_size=(args.heatmap_size, args.heatmap_size),
        augment=False,
        include_angles=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = DINOv3DiffusionPoseEstimator(
        dino_model_name=args.model_name,
        heatmap_size=(args.heatmap_size, args.heatmap_size),
        unfreeze_blocks=0,
        fix_joint7_zero=True,
        diffusion_steps=args.diffusion_steps,
        angle_dropout=args.angle_dropout,
    ).to(device)
    load_diffusion_checkpoint(model, str(checkpoint_path), device)
    model.eval()

    frame_records = []
    all_angle_errors_deg = []
    all_fk_errors_mm = []
    all_kp_errors_px = []
    all_reproj_errors_px = []

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
        images = batch["image"].to(device)
        gt_angles = batch["angles"].to(device)
        has_angles = batch["has_angles"].cpu().numpy().astype(bool)
        valid_mask = batch["valid_mask"].cpu().numpy()
        gt_kp_2d = batch["keypoints"].cpu().numpy()
        camera_k = batch["camera_K"].cpu().numpy()
        original_size = batch["original_size"].cpu().numpy()

        outputs = model(images, training=False)
        pred_angles_norm = outputs["joint_angles"][:, :6]
        pred_angles = denormalize_angles(pred_angles_norm)
        pred_heatmaps = outputs["heatmaps_2d"]
        pred_uv = soft_argmax_2d(pred_heatmaps, temperature=100.0).cpu().numpy()

        gt_angles_full = gt_angles.clone()
        gt_angles_full[:, 6] = 0.0
        pred_angles_full = torch.zeros_like(gt_angles)
        pred_angles_full[:, :6] = pred_angles

        gt_kp_3d = panda_forward_kinematics(gt_angles_full)
        pred_kp_3d = panda_forward_kinematics(pred_angles_full)

        angle_errors_deg = torch.rad2deg(torch.abs(pred_angles - gt_angles[:, :6])).cpu().numpy()
        fk_errors_mm = ((pred_kp_3d - gt_kp_3d).norm(dim=-1) * 1000.0).cpu().numpy()
        kp_errors_px = np.linalg.norm(pred_uv - gt_kp_2d, axis=-1)
        reproj_errors_px = []

        pred_fk_np = pred_kp_3d.cpu().numpy()
        gt_fk_np = gt_kp_3d.cpu().numpy()
        for i in range(images.shape[0]):
            if not has_angles[i]:
                reproj_errors_px.append(np.full(pred_uv.shape[1], np.nan, dtype=np.float32))
                continue
            scaled_k = scale_camera_k(camera_k[i], original_size[i], (args.heatmap_size, args.heatmap_size))
            rvec, tvec = solve_pnp_pose(gt_fk_np[i], gt_kp_2d[i], valid_mask[i], scaled_k)
            if rvec is None:
                reproj_errors_px.append(np.full(pred_uv.shape[1], np.nan, dtype=np.float32))
                continue
            pred_fk_reproj = project_robot_points(pred_fk_np[i], rvec, tvec, scaled_k)
            reproj_errors_px.append(np.linalg.norm(pred_fk_reproj - gt_kp_2d[i], axis=-1).astype(np.float32))
        reproj_errors_px = np.stack(reproj_errors_px, axis=0)

        if has_angles.any():
            all_angle_errors_deg.append(angle_errors_deg[has_angles])
            all_fk_errors_mm.append(fk_errors_mm[has_angles])
            valid_reproj = reproj_errors_px[has_angles]
            valid_reproj = valid_reproj[~np.isnan(valid_reproj).all(axis=1)]
            if len(valid_reproj) > 0:
                all_reproj_errors_px.append(valid_reproj)
        all_kp_errors_px.append(kp_errors_px)

        batch_size = images.shape[0]
        for i in range(batch_size):
            frame_idx = batch_idx * args.batch_size + i
            valid = valid_mask[i].astype(bool)
            frame_angle_mae = float(angle_errors_deg[i].mean())
            frame_fk_mm = float(fk_errors_mm[i].mean())
            frame_kp_px = float(kp_errors_px[i][valid].mean()) if valid.any() else 0.0
            valid_reproj_vals = reproj_errors_px[i][valid] if valid.any() else np.array([], dtype=np.float32)
            valid_reproj_vals = valid_reproj_vals[np.isfinite(valid_reproj_vals)]
            frame_records.append({
                "index": frame_idx,
                "name": batch["name"][i],
                "has_angles": bool(has_angles[i]),
                "valid_mask": [bool(x) for x in valid],
                "angle_mae_deg": frame_angle_mae,
                "per_joint_angle_deg": [float(x) for x in angle_errors_deg[i]],
                "fk_mean_mm": frame_fk_mm,
                "per_joint_fk_mm": [float(x) for x in fk_errors_mm[i]],
                "kp2d_mean_px": frame_kp_px,
                "per_joint_kp2d_px": [float(x) for x in kp_errors_px[i]],
                "pred_fk_reproj_mean_px": float(valid_reproj_vals.mean()) if len(valid_reproj_vals) > 0 else None,
                "per_joint_pred_fk_reproj_px": [None if np.isnan(x) else float(x) for x in reproj_errors_px[i]],
            })

    all_angle_errors_deg = np.concatenate(all_angle_errors_deg, axis=0) if all_angle_errors_deg else None
    all_fk_errors_mm = np.concatenate(all_fk_errors_mm, axis=0) if all_fk_errors_mm else None
    all_kp_errors_px = np.concatenate(all_kp_errors_px, axis=0)
    all_reproj_errors_px = np.concatenate(all_reproj_errors_px, axis=0) if all_reproj_errors_px else None

    kp_sum = np.zeros(all_kp_errors_px.shape[1], dtype=np.float64)
    kp_count = np.zeros(all_kp_errors_px.shape[1], dtype=np.float64)
    for rec in frame_records:
        valid = np.array(rec["valid_mask"], dtype=bool)
        kp_vals = np.array(rec["per_joint_kp2d_px"], dtype=np.float64)
        kp_sum[valid] += kp_vals[valid]
        kp_count[valid] += 1.0
    kp_mean = kp_sum / np.maximum(kp_count, 1.0)

    metrics = {
        "checkpoint": str(checkpoint_path),
        "data_dir": args.data_dir,
        "num_samples": len(frame_records),
        "num_samples_with_angles": int(sum(1 for rec in frame_records if rec["has_angles"])),
        "angle_mae_deg": float(all_angle_errors_deg.mean()) if all_angle_errors_deg is not None else None,
        "per_joint_angle_mae_deg": [float(x) for x in all_angle_errors_deg.mean(axis=0)] if all_angle_errors_deg is not None else None,
        "fk_mean_mm": float(all_fk_errors_mm.mean()) if all_fk_errors_mm is not None else None,
        "per_joint_fk_mean_mm": [float(x) for x in all_fk_errors_mm.mean(axis=0)] if all_fk_errors_mm is not None else None,
        "kp2d_mean_px": float(kp_mean.mean()),
        "per_joint_kp2d_mean_px": [float(x) for x in kp_mean],
        "pred_fk_reproj_mean_px": float(np.nanmean(all_reproj_errors_px)) if all_reproj_errors_px is not None else None,
        "per_joint_pred_fk_reproj_mean_px": [float(x) for x in np.nanmean(all_reproj_errors_px, axis=0)] if all_reproj_errors_px is not None else None,
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(output_dir / "per_frame_errors.json", "w") as f:
        json.dump(frame_records, f, indent=2)

    sortable_records = [rec for rec in frame_records if rec["has_angles"]]
    sorted_records = sorted(sortable_records, key=lambda x: x["angle_mae_deg"])
    num_each = min(args.num_qualitative, len(sorted_records))
    random.seed(args.seed)
    selected = []
    selected.extend([("best", rec["index"]) for rec in sorted_records[:num_each]])
    selected.extend([("worst", rec["index"]) for rec in sorted_records[-num_each:]])
    remaining = [rec["index"] for rec in sorted_records[num_each:-num_each]] if len(sorted_records) > 2 * num_each else []
    random.shuffle(remaining)
    selected.extend([("random", idx) for idx in remaining[:num_each]])

    seen = set()
    for rank_name, idx in selected:
        if idx in seen:
            continue
        seen.add(idx)
        sample = dataset[idx]
        image = sample["image"].unsqueeze(0).to(device)
        outputs = model(image, training=False)
        pred_angles = denormalize_angles(outputs["joint_angles"][0, :6]).cpu().numpy()
        gt_angles = sample["angles"].cpu().numpy()
        pred_uv = soft_argmax_2d(outputs["heatmaps_2d"], temperature=100.0)[0].cpu().numpy()

        gt_angles_full = sample["angles"].unsqueeze(0).to(device)
        gt_angles_full[:, 6] = 0.0
        pred_angles_full = torch.zeros_like(gt_angles_full)
        pred_angles_full[:, :6] = torch.tensor(pred_angles, dtype=gt_angles_full.dtype, device=device).unsqueeze(0)
        fk_mm = float((panda_forward_kinematics(pred_angles_full) - panda_forward_kinematics(gt_angles_full)).norm(dim=-1).mean().item() * 1000.0)
        angle_mae = float(np.degrees(np.abs(pred_angles - gt_angles[:6])).mean())

        gt_fk_robot = panda_forward_kinematics(gt_angles_full)[0].cpu().numpy()
        pred_fk_robot = panda_forward_kinematics(pred_angles_full)[0].cpu().numpy()
        scaled_k = scale_camera_k(
            sample["camera_K"].cpu().numpy(),
            sample["original_size"].cpu().numpy(),
            (args.heatmap_size, args.heatmap_size),
        )
        valid = sample["valid_mask"].cpu().numpy().astype(bool)
        gt_uv = sample["keypoints"].cpu().numpy()
        rvec, tvec = solve_pnp_pose(gt_fk_robot, gt_uv, valid, scaled_k)
        gt_reproj_uv = None
        gt_fk_reproj_uv = None
        pred_fk_reproj_uv = None
        if rvec is not None:
            gt_fk_reproj_uv = project_robot_points(gt_fk_robot, rvec, tvec, scaled_k)
            pred_fk_reproj_uv = project_robot_points(pred_fk_robot, rvec, tvec, scaled_k)
        gt_cam_3d = sample["keypoints_3d"].cpu().numpy().astype(np.float64)
        if valid.any():
            homog = (scaled_k @ gt_cam_3d.T).T
            z = np.clip(homog[:, 2:3], 1e-8, None)
            gt_reproj_uv = (homog[:, :2] / z).astype(np.float32)

        vis = build_vis_image(
            sample,
            pred_uv,
            np.degrees(pred_angles),
            np.degrees(gt_angles),
            angle_mae,
            fk_mm,
            (args.heatmap_size, args.heatmap_size),
            gt_reproj_uv=gt_reproj_uv,
            pred_fk_reproj_uv=pred_fk_reproj_uv,
            gt_fk_reproj_uv=gt_fk_reproj_uv,
        )
        vis.save(qual_dir / f"{rank_name}_{idx:05d}.png")

    print("# Evaluation complete")
    print(f"# Checkpoint: {checkpoint_path}")
    print(f"# Output dir: {output_dir}")
    if metrics["angle_mae_deg"] is not None:
        print(f"# Angle MAE: {metrics['angle_mae_deg']:.2f} deg")
        print(f"# Per-joint angle MAE: {metrics['per_joint_angle_mae_deg']}")
        print(f"# FK mean error: {metrics['fk_mean_mm']:.2f} mm")
    print(f"# 2D keypoint mean error: {metrics['kp2d_mean_px']:.2f} px")
    if metrics["pred_fk_reproj_mean_px"] is not None:
        print(f"# Pred-FK reprojection mean error: {metrics['pred_fk_reproj_mean_px']:.2f} px")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--heatmap-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--diffusion-steps", type=int, default=20)
    parser.add_argument("--angle-dropout", type=float, default=0.1)
    parser.add_argument("--num-qualitative", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
