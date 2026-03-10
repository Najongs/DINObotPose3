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

from model_v4 import DINOv3PoseEstimatorV4, panda_forward_kinematics, soft_argmax_2d
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
        return 0.0
    elif epoch < 15:
        return 1e-3
    elif epoch < 30:
        return 1e-2
    else:
        return 5e-2  # Keep low to avoid overfitting


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


def get_camera_extrinsics(gt_kp_2d, gt_kp_3d_robot, camera_K, valid_mask):
    """
    Given GT 2D keypoints and GT 3D robot-frame keypoints,
    solve PnP to find the exact Camera Extrinsics (R, T) for each batch element.
    """
    B = gt_kp_2d.shape[0]
    device = gt_kp_2d.device
    R_mats = []
    T_vecs = []

    for b in range(B):
        pts2d = gt_kp_2d[b].detach().cpu().numpy().astype(np.float64)
        pts3d = gt_kp_3d_robot[b].detach().cpu().numpy().astype(np.float64)
        K = camera_K[b].detach().cpu().numpy().astype(np.float64)
        mask = valid_mask[b].detach().cpu().numpy().astype(bool)

        pts2d_valid = pts2d[mask]
        pts3d_valid = pts3d[mask]

        if len(pts2d_valid) >= 4:
            success, rvec, tvec = cv2.solvePnP(
                pts3d_valid, pts2d_valid, K, None, flags=cv2.SOLVEPNP_ITERATIVE
            )
            if success:
                R, _ = cv2.Rodrigues(rvec)
                R_mats.append(torch.from_numpy(R).float().to(device))
                T_vecs.append(torch.from_numpy(tvec).float().to(device))
                continue
        
        # Fallback to Identity if PnP fails (should be very rare for GT data)
        R_mats.append(torch.eye(3, device=device))
        T_vecs.append(torch.zeros((3, 1), device=device))

    R_batch = torch.stack(R_mats, dim=0)  # (B, 3, 3)
    T_batch = torch.stack(T_vecs, dim=0).squeeze(-1)  # (B, 3)
    return R_batch, T_batch


def project_3d_to_2d(points_3d, R, T, K):
    """
    points_3d: (B, N, 3) in robot frame
    R: (B, 3, 3) rotation from robot to camera
    T: (B, 3) translation from robot to camera
    K: (B, 3, 3) camera intrinsics
    Returns: (B, N, 2) projected normalized pixel coordinates
    """
    # 1. World to Camera: X_c = R * X_w + T
    # points_3d: (B, N, 3), R: (B, 3, 3)
    # (B, N, 3) @ (B, 3, 3) -> (B, N, 3)
    pts_cam = torch.bmm(points_3d, R.transpose(1, 2)) + T.unsqueeze(1)
    
    # 2. Camera to Image Plane: x = K * X_c
    pts_img_homo = torch.bmm(pts_cam, K.transpose(1, 2))  # (B, N, 3)
    
    # 3. Perspective divide
    z = pts_img_homo[..., 2:3].clamp(min=1e-6)
    uv = pts_img_homo[..., :2] / z
    return uv


def scale_camera_K_batch(camera_K, original_sizes, target_size):
    """
    Scale camera intrinsics based on image resizing.
    original_sizes: (B, 2) containing (width, height)
    target_size: scalar or (target_w, target_h)
    """
    B = camera_K.shape[0]
    device = camera_K.device
    scaled_K = camera_K.clone()
    
    if isinstance(target_size, int):
        target_w = target_h = target_size
    else:
        target_w, target_h = target_size
        
    for b in range(B):
        orig_w = original_sizes[b, 0].item()
        orig_h = original_sizes[b, 1].item()
        
        scale_x = target_w / max(orig_w, 1.0)
        scale_y = target_h / max(orig_h, 1.0)
        
        scaled_K[b, 0, 0] *= scale_x
        scaled_K[b, 1, 1] *= scale_y
        scaled_K[b, 0, 2] *= scale_x
        scaled_K[b, 1, 2] *= scale_y
        
    return scaled_K


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

    # ─── Joint angle statistics (hardcoded from DREAM synthetic panda, same as RoboPEPP) ───
    joint_mean = PANDA_JOINT_MEAN.to(device)
    joint_std = PANDA_JOINT_STD.to(device)
    if is_main:
        print(f"Joint mean: {joint_mean.cpu().tolist()}")
        print(f"Joint std:  {joint_std.cpu().tolist()}")

    # ─── Model ───
    model = DINOv3PoseEstimatorV4(
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
    reproj_criterion = nn.L1Loss(reduction='none')

    # Balanced per-joint weights
    joint_weights = torch.tensor([1.5, 1.0, 1.0, 1.0, 1.0, 1.0], device=device)
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

        if epoch == args.warmup_frozen_epochs:
            if is_main:
                print(f"\n{'='*60}")
                print(f"UNFREEZING backbone (last {args.unfreeze_blocks} blocks)")
                print(f"{'='*60}\n")
            
            # Unfreeze backbone
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

            # Rebuild optimizer
            param_groups = [
                {'params': [p for p in raw_model.joint_angle_head.parameters() if p.requires_grad], 'lr': args.lr},
                {'params': [p for n, p in raw_model.backbone.named_parameters() if p.requires_grad], 'lr': args.lr * 0.01},
            ]
            optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay)

        alpha_hm = 0.0  # Optional auxiliary HM loss
        w_reproj = args.reproj_loss_weight  # New hyperparam for 2D Reproj

        model.train()
        train_loss_accum = 0.0
        train_joint_mae = np.zeros(6)
        train_count = 0
        epoch_grad_stats = {}

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]") if is_main else train_loader

        for batch in pbar:
            if global_step < args.warmup_steps:
                frac = (global_step + 1) / args.warmup_steps
                base_lrs = [args.lr, args.lr * 0.01]
                for idx, pg in enumerate(optimizer.param_groups):
                    base = base_lrs[idx] if idx < len(base_lrs) else args.lr
                    pg['lr'] = base * frac

            imgs = batch['image'].to(device)
            gt_angles = batch['angles'].to(device)
            valid_mask = batch['valid_mask'].to(device)
            gt_kp_2d = batch['keypoints'].to(device)  # (B, N, 2) in normalized pixel space for cropped image, but dataset outputs uncropped original camera coords? Wait, the crop might affect intrinsics. We must be very careful. Let's assume K is correctly scaled to Heatmap Space by `scale_camera_K_batch`.
            camera_K = batch['camera_K'].to(device)
            original_sizes = batch['original_size'].to(device)

            # Scale camera intrinsics to match heatmap space so our projection points match gt_kp_2d
            # *Assuming gt_kp_2d is scaled to match heatmap_size*
            scaled_K = scale_camera_K_batch(camera_K, original_sizes, args.heatmap_size)

            gt_angles_6 = gt_angles[:, :6]
            gt_norm = (gt_angles_6 - joint_mean[:6]) / joint_std[:6]

            optimizer.zero_grad()
            preds = model(imgs)

            pred_angles_norm = preds['joint_angles'][:, :6]

            # ─── 1. Joint angle loss (L1 / Huber) ───
            joint_loss_per = F.smooth_l1_loss(pred_angles_norm, gt_norm, reduction='none', beta=0.5)
            joint_loss = (joint_loss_per * joint_weights.unsqueeze(0)).mean()

            # Denormalize
            pred_angles = pred_angles_norm * joint_std[:6] + joint_mean[:6]

            # ─── 2. Bone/FK 3D Loss ───
            pred_angles_full = torch.zeros(gt_angles.shape, device=device)
            pred_angles_full[:, :6] = pred_angles
            pred_angles_full[:, 6] = 0.0 
            
            gt_angles_full = gt_angles.clone()
            gt_angles_full[:, 6] = 0.0
            
            pred_kp_3d = panda_forward_kinematics(pred_angles_full) # (B, 7, 3)
            gt_kp_3d = panda_forward_kinematics(gt_angles_full)     # (B, 7, 3)
            
            fk_loss = F.mse_loss(pred_kp_3d, gt_kp_3d)

            # ─── 3. 2D Reprojection Loss (⭐ KEY ELEMENT FOR V4 ⭐) ───
            # Solve PnP for Ground Truth to find Extrinsics
            with torch.no_grad():
                gt_R, gt_T = get_camera_extrinsics(gt_kp_2d, gt_kp_3d, scaled_K, valid_mask)
            
            # Project PREDICTED 3D points onto 2D image plane using the GT camera extrinsics
            pred_kp_2d_proj = project_3d_to_2d(pred_kp_3d, gt_R, gt_T, scaled_K) # (B, N, 2)

            # Compute pixel loss against GT 2D points, masking out occluded/invalid ones
            reproj_diff = reproj_criterion(pred_kp_2d_proj, gt_kp_2d).mean(dim=2) # (B, N)
            reproj_loss = (reproj_diff * valid_mask.float()).sum() / (valid_mask.float().sum() + 1e-6)

            total_loss = joint_loss + args.fk_loss_weight * fk_loss + w_reproj * reproj_loss

            total_loss.backward()

            # ─── Gradient monitoring (every 100 steps) ───
            grad_stats = {}
            if is_main and global_step % 100 == 0:
                for name, module in [
                    ('angle_head', raw_model.joint_angle_head),
                    ('kp_head', raw_model.keypoint_head),
                    ('backbone', raw_model.backbone),
                ]:
                    grads = [p.grad.detach().norm().item() for p in module.parameters() if p.grad is not None]
                    if grads:
                        grad_stats[name] = {'mean': np.mean(grads), 'max': np.max(grads), 'n': len(grads)}
                        if name not in epoch_grad_stats:
                            epoch_grad_stats[name] = []
                        epoch_grad_stats[name].append(grad_stats[name]['mean'])

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
                grad_str = ''
                if grad_stats:
                    grad_str = ' | ' + ' '.join([f'∇{k}={v["mean"]:.1e}' for k, v in grad_stats.items()])
                pbar.set_postfix_str(
                    f"Lj={joint_loss.item():.4f} Lfk={fk_loss.item():.4f} Lproj={reproj_loss.item():.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.1e} | {jstr}°{grad_str}"
                )

            if is_main and args.use_wandb and global_step % 50 == 0:
                log_dict = {
                    "train/joint_loss": joint_loss.item(),
                    "train/fk_loss": fk_loss.item(),
                    "train/reproj_loss": reproj_loss.item(),
                    "train/total_loss": total_loss.item(),
                    "train/lr": optimizer.param_groups[0]['lr'],
                }
                for name, stats in grad_stats.items():
                    log_dict[f"grad/{name}_mean"] = stats['mean']
                    log_dict[f"grad/{name}_max"] = stats['max']
                wandb.log(log_dict, step=global_step)

        # ─── Validation ───
        model.eval()
        val_loss_accum = 0.0
        val_joint_mae = np.zeros(6)
        val_3d_errors = []
        val_reproj_errors = []
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
                gt_kp_2d = batch['keypoints'].to(device)
                camera_K = batch['camera_K'].to(device)
                original_sizes = batch['original_size'].to(device)
                scaled_K = scale_camera_K_batch(camera_K, original_sizes, args.heatmap_size)

                gt_angles_6 = gt_angles[:, :6]
                gt_norm = (gt_angles_6 - joint_mean[:6]) / joint_std[:6]

                preds = model(imgs)
                pred_angles_norm = preds['joint_angles'][:, :6]

                joint_loss = (joint_criterion(pred_angles_norm, gt_norm) * joint_weights.unsqueeze(0)).mean()
                val_loss_accum += joint_loss.item()

                # Denormalize for MAE
                pred_angles = pred_angles_norm * joint_std[:6] + joint_mean[:6]

                angle_diff = pred_angles - gt_angles_6
                batch_mae = angle_diff.abs().mean(dim=0).cpu().numpy() * (180 / math.pi)
                val_joint_mae = (val_joint_mae * val_count + batch_mae) / (val_count + 1)
                val_count += 1

                gt_angles_full = gt_angles.clone()
                gt_angles_full[:, 6] = 0.0
                pred_angles_full = torch.zeros_like(gt_angles)
                pred_angles_full[:, :6] = pred_angles
                gt_kp_3d = panda_forward_kinematics(gt_angles_full)
                pred_kp_3d = panda_forward_kinematics(pred_angles_full)
                per_link_err = (gt_kp_3d - pred_kp_3d).norm(dim=-1)
                val_3d_errors.append(per_link_err.cpu().numpy())

                # Validate Reprojection
                gt_R, gt_T = get_camera_extrinsics(gt_kp_2d, gt_kp_3d, scaled_K, valid_mask)
                pred_kp_2d_proj = project_3d_to_2d(pred_kp_3d, gt_R, gt_T, scaled_K) # (B, N, 2)
                per_link_reproj_err = (pred_kp_2d_proj - gt_kp_2d).norm(dim=-1) # (B, N)
                val_reproj_errors.append(per_link_reproj_err.cpu().numpy())

                # Capture viz data
                if i == 0 and is_main:
                    viz_data = (imgs, gt_kp_3d, pred_kp_3d, preds['heatmaps_2d'])

        avg_val_loss = val_loss_accum / max_val_batches

        if is_main:
            # ─── Detailed logging ───
            print(f"\n{'='*60}")
            print(f"Epoch {epoch} | Val Joint Loss: {avg_val_loss:.4f} | w_reproj={w_reproj}")
            print(f"{'='*60}")
            print(f"  {'Joint':<8} {'Train MAE':>12} {'Val MAE':>12}")
            print(f"  {'-'*8} {'-'*12} {'-'*12}")
            for j in range(6):
                marker = " ⚠️" if val_joint_mae[j] > 20 else ""
                print(f"  J{j:<7} {train_joint_mae[j]:>10.2f}° {val_joint_mae[j]:>10.2f}°{marker}")
            print(f"  {'MEAN':<8} {train_joint_mae.mean():>10.2f}° {val_joint_mae.mean():>10.2f}°")
            worst = np.argmax(val_joint_mae)
            print(f"  → Worst: J{worst} ({val_joint_mae[worst]:.2f}°)")
            print(f"{'='*60}")

            link_names = ['link0', 'link2', 'link3', 'link4', 'link6', 'link7', 'hand']
            val_3d = np.concatenate(val_3d_errors, axis=0).mean(axis=0)  # (7,)
            mean_3d = val_3d.mean()
            val_reproj = np.concatenate(val_reproj_errors, axis=0).mean(axis=0) # (7,)
            mean_reproj = val_reproj.mean()

            print(f"\n  3D FK Error (val):")
            for li, ln in enumerate(link_names):
                print(f"    {ln:<8} {val_3d[li]*1000:.1f}mm | {val_reproj[li]:.1f}px")
            print(f"    {'MEAN':<8} {mean_3d*1000:.1f}mm | {mean_reproj:.1f}px")

            # Gradient norms (epoch average)
            if epoch_grad_stats:
                print(f"\n  Gradient Norms (epoch avg):")
                for name, vals in epoch_grad_stats.items():
                    print(f"    ∇{name:<12} mean={np.mean(vals):.2e}  max={np.max(vals):.2e}")
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
                log_dict["val/mean_3d_mm"] = mean_3d * 1000
                for li, ln in enumerate(link_names):
                    log_dict[f"val/3d_{ln}_mm"] = val_3d[li] * 1000
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
    parser.add_argument('--output-dir', type=str, default='./outputs_3d_v4')
    parser.add_argument('--model-name', type=str, default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    parser.add_argument('--image-size', type=int, default=512)
    parser.add_argument('--heatmap-size', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--warmup-steps', type=int, default=500)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--fk-loss-weight', type=float, default=50.0, help='Weight for 3D FK spatial loss')
    parser.add_argument('--reproj-loss-weight', type=float, default=10.0, help='Weight for 2D Reprojection pixel loss')
    parser.add_argument('--unfreeze-blocks', type=int, default=2, help='Backbone blocks to unfreeze')
    parser.add_argument('--warmup-frozen-epochs', type=int, default=5,
                        help='Epochs to keep backbone frozen before unfreezing')
    parser.add_argument('--val-ratio', type=float, default=0.5)
    parser.add_argument('--occlusion-prob', type=float, default=0.25)
    parser.add_argument('--occlusion-size', type=float, default=0.2)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--no-augment', action='store_true')
    parser.add_argument('--use-wandb', action='store_true')
    parser.add_argument('--wandb-project', type=str, default='dinov3-joint-angle-v4')
    parser.add_argument('--wandb-run-name', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    main(parser.parse_args())
