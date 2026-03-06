"""
DINOv3 Joint Angle Training v2
- Backbone unfreeze (last N blocks)
- Heatmap head unfreeze (joint regularization)
- Direct angle prediction (normalized, no sin/cos)
- Progressive heatmap loss weighting (RoboPEPP style)
"""

import argparse
import os
import math
import random
import io
from pathlib import Path
from datetime import timedelta

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
from PIL import Image

from model import DINOv3PoseEstimator, panda_forward_kinematics, soft_argmax_2d
from dataset import PoseEstimationDataset


# ─── Dataset statistics (precomputed from DREAM synthetic panda) ───
PANDA_JOINT_MEAN = torch.tensor([-5.22e-02, 2.68e-01, 6.04e-03, -2.01e+00, 1.49e-02, 1.99e+00, 0.0])
PANDA_JOINT_STD  = torch.tensor([1.025, 0.645, 0.511, 0.508, 0.769, 0.511, 1.0])


def compute_add_auc(kp_error_m, auc_threshold=0.1):
    frame_adds = kp_error_m.mean(dim=1).cpu().numpy()
    n_total = len(frame_adds)
    if n_total == 0:
        return 0.0, frame_adds
    delta = 0.00001
    thresholds = np.arange(0.0, auc_threshold, delta)
    counts = (frame_adds[None, :] <= thresholds[:, None]).sum(axis=1) / float(n_total)
    auc = float(np.trapz(counts, dx=delta) / auc_threshold)
    return auc, frame_adds


def compute_joint_stats(dataset, num_samples=5000):
    """Dataset에서 joint angle mean/std 계산"""
    angles_list = []
    n = min(num_samples, len(dataset))
    indices = random.sample(range(len(dataset)), n)
    for idx in indices:
        s = dataset[idx]
        if s.get('has_angles', torch.tensor(False)).item():
            angles_list.append(s['angles'])
    if len(angles_list) < 100:
        print(f"WARNING: Only {len(angles_list)} samples with angles. Using default stats.")
        return PANDA_JOINT_MEAN, PANDA_JOINT_STD
    angles = torch.stack(angles_list)
    mean = angles.mean(dim=0)
    std = angles.std(dim=0).clamp(min=0.1)
    print(f"Computed joint stats from {len(angles_list)} samples:")
    for j in range(len(mean)):
        print(f"  Joint {j}: mean={mean[j]:.4f} std={std[j]:.4f}")
    return mean, std


def get_alpha_heatmap(epoch):
    """RoboPEPP-style progressive heatmap loss weighting"""
    if epoch < 5:
        return 1e-4
    elif epoch < 10:
        return 1e-2
    elif epoch < 40:
        return 1e-1
    else:
        return 1.0


def generate_gt_heatmaps(keypoints_2d, valid_mask, heatmap_size, sigma=5.0):
    """GT 2D keypoints에서 Gaussian heatmap 생성"""
    B, N, _ = keypoints_2d.shape
    H, W = heatmap_size
    device = keypoints_2d.device

    x = torch.arange(W, device=device, dtype=torch.float32)
    y = torch.arange(H, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing='ij')  # (H, W)

    cx = keypoints_2d[:, :, 0].unsqueeze(-1).unsqueeze(-1)  # (B, N, 1, 1)
    cy = keypoints_2d[:, :, 1].unsqueeze(-1).unsqueeze(-1)

    heatmaps = torch.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))  # (B, N, H, W)

    # Zero out invalid keypoints
    if valid_mask is not None:
        if valid_mask.dim() == 2:  # (B, N)
            heatmaps = heatmaps * valid_mask.unsqueeze(-1).unsqueeze(-1).float()

    return heatmaps


def visualize_results(images, gt_kp_3d, pred_kp_3d, pred_heatmaps, num_samples=4):
    images_to_log = []
    B = images.shape[0]
    n = min(B, num_samples)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(images.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(images.device)

    for i in range(n):
        img_np = ((images[i] * std + mean).permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        fig = plt.figure(figsize=(12, 5), dpi=80)
        ax2d = fig.add_subplot(121)
        ax2d.imshow(img_np)
        ax2d.set_title("Image"); ax2d.axis('off')

        ax3d = fig.add_subplot(122, projection='3d')
        gt = gt_kp_3d[i].detach().cpu().numpy()
        pred = pred_kp_3d[i].detach().cpu().numpy()
        ax3d.plot(gt[:, 0], gt[:, 1], gt[:, 2], 'go-', label='GT', linewidth=2, markersize=4)
        ax3d.plot(pred[:, 0], pred[:, 1], pred[:, 2], 'ro--', label='Pred', linewidth=2, markersize=4)
        ax3d.legend(); ax3d.set_title("3D Pose (Robot Frame)")

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        images_to_log.append(wandb.Image(Image.open(buf), caption=f"sample_{i}"))
        plt.close(fig)
    return images_to_log


def main(args):
    # ─── DDP Init ───
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', timeout=timedelta(minutes=30))
        device = torch.device(f'cuda:{local_rank}')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rank = 0; world_size = 1

    is_main = rank == 0
    output_dir = Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    # ─── Dataset ───
    keypoint_names = ['link0', 'link2', 'link3', 'link4', 'link6', 'link7', 'hand']
    train_dataset = PoseEstimationDataset(
        data_dir=args.train_dir, keypoint_names=keypoint_names,
        image_size=(args.image_size, args.image_size),
        heatmap_size=(args.heatmap_size, args.heatmap_size),
        augment=not args.no_augment, include_angles=True,
        occlusion_prob=args.occlusion_prob,
        occlusion_max_size_frac=args.occlusion_size,
    )
    val_dataset = PoseEstimationDataset(
        data_dir=args.val_dir, keypoint_names=keypoint_names,
        image_size=(args.image_size, args.image_size),
        heatmap_size=(args.heatmap_size, args.heatmap_size),
        augment=False, include_angles=True,
    )

    train_sampler = DistributedSampler(train_dataset) if local_rank != -1 else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if local_rank != -1 else None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler,
                              shuffle=(train_sampler is None), num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler,
                            shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # ─── Joint angle statistics ───
    if is_main:
        print("Computing joint angle statistics from training data...")
    joint_mean, joint_std = compute_joint_stats(train_dataset)
    joint_mean = joint_mean.to(device)
    joint_std = joint_std.to(device)

    # ─── Model ───
    model = DINOv3PoseEstimator(
        dino_model_name=args.model_name,
        heatmap_size=(args.heatmap_size, args.heatmap_size),
        unfreeze_blocks=args.unfreeze_blocks,
        fix_joint7_zero=True,
    ).to(device)

    # Load 2D pretrained checkpoint
    if args.checkpoint and os.path.isfile(args.checkpoint):
        if is_main:
            print(f"Loading 2D checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
        model.load_state_dict(ckpt, strict=False)

    # ─── Freeze strategy ───
    # Phase 1 (epoch < warmup_frozen_epochs): backbone frozen, only angle head trains
    # Phase 2 (epoch >= warmup_frozen_epochs): unfreeze backbone last N blocks + heatmap head
    # Initially freeze everything, unfreeze angle head only
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.keypoint_head.parameters():
        param.requires_grad = False
    for param in model.joint_angle_head.parameters():
        param.requires_grad = True

    if local_rank != -1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    raw_model = model.module if hasattr(model, 'module') else model

    # ─── Optimizer (will be rebuilt when unfreezing) ───
    def build_optimizer(model_ref, lr):
        params = [p for p in model_ref.parameters() if p.requires_grad]
        return optim.AdamW(params, lr=lr, weight_decay=args.weight_decay)

    optimizer = build_optimizer(raw_model, args.lr)

    # ─── Loss ───
    heatmap_criterion = nn.MSELoss()
    joint_criterion = nn.MSELoss(reduction='none')
    joint_l1 = nn.L1Loss(reduction='none')

    # Per-joint weights
    joint_weights = torch.tensor([3.0, 1.0, 2.0, 1.0, 2.0, 1.0], device=device)  # 6 joints (joint7 fixed)
    joint_weights = joint_weights / joint_weights.mean()

    if is_main and args.use_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    best_val_auc = 0.0
    global_step = 0

    if is_main:
        n_total = sum(p.numel() for p in raw_model.parameters())
        n_train = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
        print(f"\nTotal params: {n_total:,}, Trainable: {n_train:,}")
        print(f"Joint mean: {joint_mean.cpu().tolist()}")
        print(f"Joint std:  {joint_std.cpu().tolist()}")
        print(f"Unfreeze backbone at epoch {args.warmup_frozen_epochs}\n")

    # ─── Training Loop ───
    for epoch in range(args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)

        # ─── Phase transition: unfreeze backbone + heatmap head ───
        if epoch == args.warmup_frozen_epochs:
            if is_main:
                print(f"\n{'='*60}")
                print(f"UNFREEZING backbone (last {args.unfreeze_blocks} blocks) + heatmap head")
                print(f"{'='*60}\n")

            # Unfreeze backbone last N blocks
            if hasattr(raw_model.backbone.model, 'encoder') and hasattr(raw_model.backbone.model.encoder, 'layers'):
                layers = raw_model.backbone.model.encoder.layers
            elif hasattr(raw_model.backbone.model, 'blocks'):
                layers = raw_model.backbone.model.blocks
            else:
                layers = []

            if args.unfreeze_blocks > 0 and len(layers) > 0:
                for i in range(len(layers) - args.unfreeze_blocks, len(layers)):
                    for param in layers[i].parameters():
                        param.requires_grad = True

            # Unfreeze heatmap head
            for param in raw_model.keypoint_head.parameters():
                param.requires_grad = True

            # Rebuild optimizer with all trainable params, lower LR for backbone
            param_groups = [
                {'params': [p for p in raw_model.joint_angle_head.parameters() if p.requires_grad], 'lr': args.lr},
                {'params': [p for p in raw_model.keypoint_head.parameters() if p.requires_grad], 'lr': args.lr * 0.1},
                {'params': [p for n, p in raw_model.backbone.named_parameters() if p.requires_grad], 'lr': args.lr * 0.01},
            ]
            optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay)

            if is_main:
                n_train = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
                print(f"Trainable params after unfreeze: {n_train:,}\n")

        alpha_hm = get_alpha_heatmap(epoch) if epoch >= args.warmup_frozen_epochs else 0.0

        # ─── Train ───
        model.train()
        train_loss_accum = 0.0
        train_joint_mae = np.zeros(6)
        train_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]") if is_main else train_loader

        for batch in pbar:
            # Warmup LR
            if global_step < args.warmup_steps:
                frac = global_step / args.warmup_steps
                for pg in optimizer.param_groups:
                    pg['lr'] = pg.get('lr', args.lr) * frac

            imgs = batch['image'].to(device)
            gt_angles = batch['angles'].to(device)  # (B, 7)
            gt_heatmaps = batch['heatmaps'].to(device) if 'heatmaps' in batch else None
            valid_mask = batch['valid_mask'].to(device)

            # Fix joint 7 to 0
            gt_angles_6 = gt_angles[:, :6]

            # Normalize GT angles
            gt_norm = (gt_angles_6 - joint_mean[:6]) / joint_std[:6]

            optimizer.zero_grad()
            preds = model(imgs)

            pred_angles = preds['joint_angles'][:, :6]  # (B, 6)

            # ─── Joint angle loss (normalized space, MSE) ───
            # Normalize predictions too
            pred_norm = (pred_angles - joint_mean[:6]) / joint_std[:6]
            joint_loss_per = joint_criterion(pred_norm, gt_norm)  # (B, 6)
            joint_loss = (joint_loss_per * joint_weights.unsqueeze(0)).mean()

            # ─── Heatmap loss (regularizer) ───
            hm_loss = torch.tensor(0.0, device=device)
            if alpha_hm > 0 and gt_heatmaps is not None:
                pred_hm = preds['heatmaps_2d']
                hm_loss = heatmap_criterion(pred_hm, gt_heatmaps)
            elif alpha_hm > 0 and 'keypoints' in batch:
                # Generate GT heatmaps from 2D keypoints on the fly
                kp_2d = batch['keypoints'].to(device)  # (B, N, 2)
                gt_hm = generate_gt_heatmaps(kp_2d, valid_mask,
                                              (args.heatmap_size, args.heatmap_size), sigma=5.0)
                pred_hm = preds['heatmaps_2d']
                hm_loss = heatmap_criterion(pred_hm, gt_hm)

            total_loss = joint_loss + alpha_hm * hm_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            train_loss_accum += total_loss.item()
            global_step += 1

            # Running joint MAE (degrees)
            with torch.no_grad():
                batch_mae = (pred_angles - gt_angles_6).abs().mean(dim=0).cpu().numpy() * (180 / math.pi)
                train_joint_mae = (train_joint_mae * train_count + batch_mae) / (train_count + 1)
                train_count += 1

            if is_main and hasattr(pbar, 'set_postfix_str'):
                jstr = ' '.join([f'J{j}:{v:.1f}' for j, v in enumerate(train_joint_mae)])
                pbar.set_postfix_str(
                    f"Lj={joint_loss.item():.4f} Lhm={hm_loss.item():.4f} α={alpha_hm:.0e} "
                    f"lr={optimizer.param_groups[0]['lr']:.1e} | {jstr}°"
                )

            if is_main and args.use_wandb and global_step % 50 == 0:
                wandb.log({
                    "train/joint_loss": joint_loss.item(),
                    "train/heatmap_loss": hm_loss.item(),
                    "train/total_loss": total_loss.item(),
                    "train/alpha_hm": alpha_hm,
                    "train/lr": optimizer.param_groups[0]['lr'],
                }, step=global_step)

        # ─── Validation ───
        model.eval()
        val_loss_accum = 0.0
        val_joint_mae = np.zeros(6)
        val_count = 0
        viz_data = None
        max_val_batches = max(1, int(len(val_loader) * args.val_ratio))

        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch} [Val]") if is_main else val_loader):
                if i >= max_val_batches:
                    break

                imgs = batch['image'].to(device)
                gt_angles = batch['angles'].to(device)
                valid_mask = batch['valid_mask'].to(device)

                gt_angles_6 = gt_angles[:, :6]
                gt_norm = (gt_angles_6 - joint_mean[:6]) / joint_std[:6]

                # Scale camera_K for PnP
                camera_K = batch['camera_K'].to(device)
                original_size = batch['original_size'].to(device)
                scale_x = args.heatmap_size / original_size[:, 0]
                scale_y = args.heatmap_size / original_size[:, 1]
                camera_K_scaled = camera_K.clone()
                camera_K_scaled[:, 0, 0] *= scale_x
                camera_K_scaled[:, 1, 1] *= scale_y
                camera_K_scaled[:, 0, 2] *= scale_x
                camera_K_scaled[:, 1, 2] *= scale_y

                preds = model(imgs, camera_K=camera_K_scaled)
                pred_angles = preds['joint_angles'][:, :6]

                pred_norm = (pred_angles - joint_mean[:6]) / joint_std[:6]
                joint_loss = (joint_criterion(pred_norm, gt_norm) * joint_weights.unsqueeze(0)).mean()
                val_loss_accum += joint_loss.item()

                # Per-joint MAE (degrees) - wrap-aware
                angle_diff = pred_angles - gt_angles_6
                angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
                batch_mae = angle_diff.abs().mean(dim=0).cpu().numpy() * (180 / math.pi)
                val_joint_mae = (val_joint_mae * val_count + batch_mae) / (val_count + 1)
                val_count += 1

                # Capture viz data
                if i == 0 and is_main:
                    gt_angles_full = gt_angles.clone()
                    gt_angles_full[:, 6] = 0.0
                    gt_kp_3d = panda_forward_kinematics(gt_angles_full)
                    viz_data = (imgs, gt_kp_3d, preds['keypoints_3d_fk'], preds['heatmaps_2d'])

        avg_val_loss = val_loss_accum / max_val_batches

        if is_main:
            # ─── Detailed logging ───
            print(f"\n{'='*60}")
            print(f"Epoch {epoch} | Val Loss: {avg_val_loss:.4f} | α_hm={alpha_hm:.0e}")
            print(f"{'='*60}")
            print(f"  {'Joint':<8} {'Train MAE':>12} {'Val MAE':>12}")
            print(f"  {'-'*8} {'-'*12} {'-'*12}")
            for j in range(6):
                marker = " ⚠️" if val_joint_mae[j] > 20 else ""
                print(f"  J{j:<7} {train_joint_mae[j]:>10.2f}° {val_joint_mae[j]:>10.2f}°{marker}")
            print(f"  {'MEAN':<8} {train_joint_mae.mean():>10.2f}° {val_joint_mae.mean():>10.2f}°")
            worst = np.argmax(val_joint_mae)
            print(f"  → Worst: J{worst} ({val_joint_mae[worst]:.2f}°)")
            print(f"{'='*60}\n")

            if args.use_wandb:
                log_dict = {
                    "val/loss": avg_val_loss,
                    "val/mean_joint_mae_deg": val_joint_mae.mean(),
                    "val/worst_joint_mae_deg": val_joint_mae.max(),
                    "epoch": epoch,
                }
                for j in range(6):
                    log_dict[f"val/J{j}_mae_deg"] = val_joint_mae[j]
                    log_dict[f"train/J{j}_mae_deg"] = train_joint_mae[j]
                if viz_data is not None:
                    log_dict["viz/pose"] = visualize_results(*viz_data, num_samples=4)
                wandb.log(log_dict, step=global_step)

            # Save best
            if val_joint_mae.mean() < (best_val_auc if best_val_auc > 0 else float('inf')):
                best_val_auc = val_joint_mae.mean()
                torch.save(raw_model.state_dict(), output_dir / 'best_joint_angle.pth')
                print(f"  >> NEW BEST: mean MAE = {best_val_auc:.2f}°")

            # Save last
            torch.save(raw_model.state_dict(), output_dir / 'last_joint_angle.pth')

    if is_main:
        print(f"\nTraining complete. Best mean MAE: {best_val_auc:.2f}°")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dir', type=str, required=True)
    parser.add_argument('--val-dir', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, help='2D heatmap pretrained weights')
    parser.add_argument('--output-dir', type=str, default='./outputs_3d_v2')
    parser.add_argument('--model-name', type=str, default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    parser.add_argument('--image-size', type=int, default=512)
    parser.add_argument('--heatmap-size', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--warmup-steps', type=int, default=500)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--unfreeze-blocks', type=int, default=2, help='Backbone blocks to unfreeze')
    parser.add_argument('--warmup-frozen-epochs', type=int, default=5,
                        help='Epochs to keep backbone frozen before unfreezing')
    parser.add_argument('--val-ratio', type=float, default=0.5)
    parser.add_argument('--occlusion-prob', type=float, default=0.25)
    parser.add_argument('--occlusion-size', type=float, default=0.2)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--no-augment', action='store_true')
    parser.add_argument('--use-wandb', action='store_true')
    parser.add_argument('--wandb-project', type=str, default='dinov3-joint-angle-v2')
    parser.add_argument('--wandb-run-name', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    main(parser.parse_args())
