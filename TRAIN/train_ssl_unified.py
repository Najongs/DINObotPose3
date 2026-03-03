"""
Unified Reprojection SSL fine-tuning for DINObotPose3.

Core idea:
1) Freeze backbone + 2D head, train only joint_angle_head.
2) Use per-sample camera intrinsics K (Dynamic K) in reprojection.
3) Use detached 2D pseudo targets from model's own 2D heatmaps.
"""

import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from dataset import PoseEstimationDataset
from model import DINOv3PoseEstimator, soft_argmax_2d


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def keypoint_names() -> List[str]:
    return [
        "panda_link0",
        "panda_link2",
        "panda_link3",
        "panda_link4",
        "panda_link6",
        "panda_link7",
        "panda_hand",
    ]


def build_dataset(
    data_dirs: List[str],
    image_size: int,
    heatmap_size: int,
    augment: bool,
    multi_robot: bool,
    robot_types: Optional[List[str]],
    json_allowlist_path: Optional[str],
) -> torch.utils.data.Dataset:
    datasets = []
    for d in data_dirs:
        datasets.append(
            PoseEstimationDataset(
                data_dir=d,
                keypoint_names=keypoint_names(),
                image_size=(image_size, image_size),
                heatmap_size=(heatmap_size, heatmap_size),
                augment=augment,
                multi_robot=multi_robot,
                robot_types=robot_types,
                include_angles=True,
                fda_real_dir=None,
                fda_prob=0.0,
                json_allowlist_path=json_allowlist_path,
            )
        )
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def load_checkpoint_model(model: nn.Module, checkpoint_path: str) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    state_dict = strip_module_prefix(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)}")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)}")
    print(f"Loaded checkpoint: {checkpoint_path}")


def freeze_for_ssl(model: DINOv3PoseEstimator) -> int:
    for p in model.parameters():
        p.requires_grad = False

    for p in model.joint_angle_head.parameters():
        p.requires_grad = True

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n_trainable


def solve_pnp_batch(
    kp3d_robot: torch.Tensor,
    target_2d_orig: torch.Tensor,
    camera_k: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Estimate extrinsics with detached PnP for each sample in batch.
    Returns:
        R: (B, 3, 3), t: (B, 3), success_mask: (B,)
    """
    bsz = kp3d_robot.shape[0]
    device = kp3d_robot.device
    dtype = kp3d_robot.dtype

    r_out = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(bsz, 1, 1)
    t_out = torch.zeros((bsz, 3), device=device, dtype=dtype)
    ok = torch.zeros((bsz,), device=device, dtype=torch.bool)

    kp3d_np = kp3d_robot.detach().cpu().numpy().astype(np.float64)
    uv_np = target_2d_orig.detach().cpu().numpy().astype(np.float64)
    k_np = camera_k.detach().cpu().numpy().astype(np.float64)

    for i in range(bsz):
        obj = kp3d_np[i]
        img = uv_np[i]
        K = k_np[i]
        try:
            ret, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_EPNP)
            if not ret:
                continue
            ret, rvec, tvec = cv2.solvePnP(
                obj,
                img,
                K,
                None,
                flags=cv2.SOLVEPNP_ITERATIVE,
                useExtrinsicGuess=True,
                rvec=rvec,
                tvec=tvec,
            )
            if not ret:
                continue
            rmat, _ = cv2.Rodrigues(rvec)
            r_out[i] = torch.from_numpy(rmat).to(device=device, dtype=dtype)
            t_out[i] = torch.from_numpy(tvec.reshape(3)).to(device=device, dtype=dtype)
            ok[i] = True
        except Exception:
            continue
    return r_out, t_out, ok


def project_with_dynamic_k(
    kp3d_robot: torch.Tensor,
    rmat: torch.Tensor,
    tvec: torch.Tensor,
    camera_k: torch.Tensor,
) -> torch.Tensor:
    """
    Differentiable projection with per-sample intrinsics K.
    Args:
        kp3d_robot: (B, 7, 3)
        rmat: (B, 3, 3)
        tvec: (B, 3)
        camera_k: (B, 3, 3)
    Returns:
        uv: (B, 7, 2), in original image coordinates
    """
    kp_cam = torch.bmm(kp3d_robot, rmat.transpose(1, 2)) + tvec.unsqueeze(1)
    z = kp_cam[..., 2:3].clamp(min=1e-6)
    kp_norm = kp_cam / z
    uv_h = torch.bmm(camera_k, kp_norm.transpose(1, 2)).transpose(1, 2)
    return uv_h[..., :2]


def train_one_epoch(
    model: DINOv3PoseEstimator,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    heatmap_size: int,
    grad_clip: float,
) -> Dict[str, float]:
    model.train()
    mse = nn.MSELoss(reduction="none")

    total_loss = 0.0
    total_valid = 0
    total_items = 0

    pbar = tqdm(loader, desc="train", dynamic_ncols=True)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        camera_k = batch["camera_K"].to(device, non_blocking=True)
        original_size = batch["original_size"].to(device, non_blocking=True)  # (B,2), W,H

        out = model(images)
        heatmaps = out["heatmaps_2d"]
        kp3d_fk = out["keypoints_3d_fk"]

        # 2D pseudo target from frozen 2D branch
        with torch.no_grad():
            target_2d_hm = soft_argmax_2d(heatmaps, temperature=10.0)
            sx = (original_size[:, 0] / float(heatmap_size)).view(-1, 1, 1)
            sy = (original_size[:, 1] / float(heatmap_size)).view(-1, 1, 1)
            target_2d_orig = torch.cat(
                [target_2d_hm[..., 0:1] * sx, target_2d_hm[..., 1:2] * sy], dim=-1
            ).detach()

        rmat, tvec, ok = solve_pnp_batch(kp3d_fk, target_2d_orig, camera_k)
        pred_2d_orig = project_with_dynamic_k(kp3d_fk, rmat, tvec, camera_k)

        per_sample = mse(pred_2d_orig, target_2d_orig).mean(dim=(1, 2))
        if ok.any():
            loss = per_sample[ok].mean()
            valid_count = int(ok.sum().item())
        else:
            # keep graph valid; practically this batch is skipped
            loss = per_sample.mean() * 0.0
            valid_count = 0

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.joint_angle_head.parameters(), grad_clip)
        optimizer.step()

        total_loss += float(loss.item()) * max(valid_count, 1)
        total_valid += valid_count
        total_items += images.shape[0]

        pbar.set_postfix(
            loss=f"{loss.item():.5f}",
            pnp_ok=f"{valid_count}/{images.shape[0]}",
        )

    avg = total_loss / max(total_valid, 1)
    return {"ssl_loss": avg, "pnp_valid": total_valid, "samples": total_items}


@torch.no_grad()
def validate_one_epoch(
    model: DINOv3PoseEstimator,
    loader: DataLoader,
    device: torch.device,
    heatmap_size: int,
) -> Dict[str, float]:
    model.eval()
    mse = nn.MSELoss(reduction="none")
    total_loss = 0.0
    total_valid = 0
    total_items = 0

    pbar = tqdm(loader, desc="val", dynamic_ncols=True)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        camera_k = batch["camera_K"].to(device, non_blocking=True)
        original_size = batch["original_size"].to(device, non_blocking=True)

        out = model(images)
        heatmaps = out["heatmaps_2d"]
        kp3d_fk = out["keypoints_3d_fk"]

        target_2d_hm = soft_argmax_2d(heatmaps, temperature=10.0)
        sx = (original_size[:, 0] / float(heatmap_size)).view(-1, 1, 1)
        sy = (original_size[:, 1] / float(heatmap_size)).view(-1, 1, 1)
        target_2d_orig = torch.cat(
            [target_2d_hm[..., 0:1] * sx, target_2d_hm[..., 1:2] * sy], dim=-1
        )

        rmat, tvec, ok = solve_pnp_batch(kp3d_fk, target_2d_orig, camera_k)
        pred_2d_orig = project_with_dynamic_k(kp3d_fk, rmat, tvec, camera_k)

        per_sample = mse(pred_2d_orig, target_2d_orig).mean(dim=(1, 2))
        if ok.any():
            loss = per_sample[ok].mean()
            valid_count = int(ok.sum().item())
        else:
            loss = per_sample.mean() * 0.0
            valid_count = 0

        total_loss += float(loss.item()) * max(valid_count, 1)
        total_valid += valid_count
        total_items += images.shape[0]
        pbar.set_postfix(loss=f"{loss.item():.5f}", pnp_ok=f"{valid_count}/{images.shape[0]}")

    avg = total_loss / max(total_valid, 1)
    return {"ssl_loss": avg, "pnp_valid": total_valid, "samples": total_items}


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_dataset = build_dataset(
        data_dirs=args.data_dir,
        image_size=args.image_size,
        heatmap_size=args.heatmap_size,
        augment=args.augment,
        multi_robot=args.multi_robot,
        robot_types=args.robot_types,
        json_allowlist_path=args.train_json_list,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = None
    if args.val_dir:
        val_dataset = build_dataset(
            data_dirs=[args.val_dir],
            image_size=args.image_size,
            heatmap_size=args.heatmap_size,
            augment=False,
            multi_robot=args.multi_robot,
            robot_types=args.robot_types,
            json_allowlist_path=args.val_json_list,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    model = DINOv3PoseEstimator(
        dino_model_name=args.model_name,
        heatmap_size=(args.heatmap_size, args.heatmap_size),
        unfreeze_blocks=0,
        use_joint_embedding=args.use_joint_embedding,
        fix_joint7_zero=args.fix_joint7_zero,
    ).to(device)

    load_checkpoint_model(model, args.checkpoint)
    trainable_params = freeze_for_ssl(model)
    print(f"Trainable params (joint_angle_head only): {trainable_params:,}")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            heatmap_size=args.heatmap_size,
            grad_clip=args.grad_clip,
        )
        print(
            f"[train] ssl_loss={train_stats['ssl_loss']:.6f}, "
            f"pnp_valid={train_stats['pnp_valid']}/{train_stats['samples']}"
        )

        save_obj = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_stats": train_stats,
            "args": vars(args),
        }
        torch.save(save_obj, out_dir / "last_ssl.pth")

        if val_loader is not None:
            val_stats = validate_one_epoch(
                model=model,
                loader=val_loader,
                device=device,
                heatmap_size=args.heatmap_size,
            )
            print(
                f"[val] ssl_loss={val_stats['ssl_loss']:.6f}, "
                f"pnp_valid={val_stats['pnp_valid']}/{val_stats['samples']}"
            )
            save_obj["val_stats"] = val_stats
            if val_stats["ssl_loss"] < best_val:
                best_val = val_stats["ssl_loss"]
                torch.save(save_obj, out_dir / "best_ssl.pth")
        else:
            # no val set: keep best=last
            torch.save(save_obj, out_dir / "best_ssl.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Reprojection SSL fine-tuning")
    parser.add_argument("--data-dir", type=str, nargs="+", required=True, help="One or more training roots")
    parser.add_argument("--val-dir", type=str, default=None, help="Validation root (optional)")
    parser.add_argument("--train-json-list", type=str, default=None, help="Train allowlist txt/json")
    parser.add_argument("--val-json-list", type=str, default=None, help="Val allowlist txt/json")
    parser.add_argument("--multi-robot", action="store_true", help="Enable multi-robot folder discovery")
    parser.add_argument("--robot-types", type=str, nargs="+", default=None, help="Optional robot type filter")

    parser.add_argument("--model-name", type=str, default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    parser.add_argument("--checkpoint", type=str, required=True, help="Supervised-trained checkpoint path")
    parser.add_argument("--output-dir", type=str, default="./outputs/ssl_unified")
    parser.add_argument("--use-joint-embedding", action="store_true")
    parser.add_argument("--fix-joint7-zero", action="store_true")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--heatmap-size", type=int, default=512)
    parser.add_argument("--augment", action="store_true")

    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    main(parser.parse_args())
