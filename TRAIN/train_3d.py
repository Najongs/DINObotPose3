"""
DINOv3 3D Pose Training Script (Stable FK-based)
Trains the Joint Angle Head using Robot-frame 3D Loss.
Features: Cosine-based Angle Loss, Kinematic Weights, 3D Skeleton Visualization.
"""

import argparse
import os
import time
import random
import io
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
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


def compute_add_auc(kp_error_m, auc_threshold=0.1):
    """
    RoboPEPP-style ADD AUC computation.

    Args:
        kp_error_m: (N_samples, N_joints) per-joint 3D error in meters
        auc_threshold: threshold in meters (default 0.1m = 100mm)

    Returns:
        auc: float, area under the ADD curve normalized by threshold
        frame_adds: (N_samples,) per-frame ADD values in meters
    """
    # Per-frame ADD = mean of per-joint distances
    frame_adds = kp_error_m.mean(dim=1).cpu().numpy()  # (N,)
    n_total = len(frame_adds)

    if n_total == 0:
        return 0.0, frame_adds

    # Integrate: fraction of frames with ADD <= t, for t in [0, threshold]
    delta = 0.00001
    thresholds = np.arange(0.0, auc_threshold, delta)
    # Vectorized: (N_thresholds, 1) vs (1, N_samples) → broadcast comparison
    counts = (frame_adds[None, :] <= thresholds[:, None]).sum(axis=1) / float(n_total)
    auc = float(np.trapz(counts, dx=delta) / auc_threshold)

    return auc, frame_adds

def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def visualize_3d_with_2d(images, gt_kp_3d, pred_kp_3d, pred_heatmaps, num_samples=4):
    """
    2D 예측 결과(이미지 오버레이)와 3D 스켈레톤을 나란히 시각화
    """
    images_to_log = []
    B = images.shape[0]
    num_to_viz = min(B, num_samples)
    
    # 이미지 역정규화 설정
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(images.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(images.device)
    
    for i in range(num_to_viz):
        # 1. 2D 이미지 준비
        img_tensor = images[i] * std + mean
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # 히트맵에서 2D 좌표 추출
        hm = pred_heatmaps[i]
        H_hm, W_hm = hm.shape[1], hm.shape[2]
        max_idx = hm.view(hm.shape[0], -1).argmax(dim=-1)
        px = (max_idx % W_hm).cpu().numpy()
        py = (max_idx // W_hm).cpu().numpy()
        
        scale_x, scale_y = img_bgr.shape[1] / W_hm, img_bgr.shape[0] / H_hm
        for k in range(len(px)):
            cv2.circle(img_bgr, (int(px[k]*scale_x), int(py[k]*scale_y)), 6, (0, 0, 255), -1)
            cv2.putText(img_bgr, str(k), (int(px[k]*scale_x)+5, int(py[k]*scale_y)-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 2. Matplotlib를 사용하여 2D와 3D 나란히 그리기
        # 🚀 고정된 DPI와 figsize를 사용
        fig = plt.figure(figsize=(16, 8), dpi=100)
        
        ax2d = fig.add_subplot(121)
        ax2d.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        ax2d.set_title(f"2D Keypoint Prediction (Sample {i})")
        ax2d.axis('off')
        
        ax3d = fig.add_subplot(122, projection='3d')
        gt = gt_kp_3d[i].detach().cpu().numpy()
        pred = pred_kp_3d[i].detach().cpu().numpy()
        
        ax3d.plot(gt[:, 0], gt[:, 1], gt[:, 2], 'go-', label='GT 3D', linewidth=2, markersize=5)
        ax3d.plot(pred[:, 0], pred[:, 1], pred[:, 2], 'ro--', label='Pred 3D', linewidth=2, markersize=5)
        ax3d.scatter(gt[0, 0], gt[0, 1], gt[0, 2], color='blue', s=100)
        
        ax3d.set_title("3D Pose (Robot Frame)")
        ax3d.set_xlabel('X (m)'); ax3d.set_ylabel('Y (m)'); ax3d.set_zlabel('Z (m)')
        ax3d.legend()
        
        all_p = np.concatenate([gt, pred])
        max_range = (all_p.max(axis=0) - all_p.min(axis=0)).max() / 2.0
        mid = (all_p.max(axis=0) + all_p.min(axis=0)) / 2.0
        ax3d.set_xlim(mid[0]-max_range, mid[0]+max_range)
        ax3d.set_ylim(mid[1]-max_range, mid[1]+max_range)
        ax3d.set_zlim(mid[2]-max_range, mid[2]+max_range)

        # 버퍼 저장
        buf = io.BytesIO()
        # 🚀 bbox_inches='tight'를 제거하여 고정 해상도 유지
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # 🚀 PIL Image로 연 후 강제로 고정 크기로 리사이즈 (WandB 경고 완벽 방어)
        final_img = Image.open(buf)
        final_img = final_img.resize((1200, 600), Image.Resampling.LANCZOS)
        
        images_to_log.append(wandb.Image(final_img, caption=f"Combined_Pose_{i}"))
        plt.close(fig)
        
    return images_to_log

class JointAnglePoseLoss(nn.Module):
    """Loss for Joint Angle prediction mode (sin/cos based)."""
    def __init__(self, angle_weight=1.0, fk_3d_weight=0.0, fix_joint7=False, compute_pnp_metric=False):
        super().__init__()
        self.angle_weight = angle_weight
        self.fk_3d_weight = fk_3d_weight
        self.fix_joint7 = fix_joint7
        self.compute_pnp_metric = compute_pnp_metric
        self.loss_fn = nn.SmoothL1Loss(beta=0.01, reduction='none')
        # Per-joint weights (user-specified kinematic importance)
        # Joint 0(base): highest, Joint 2,4,6: elevated, rest: baseline
        joint_w = torch.tensor([3.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
        self.register_buffer('joint_weights', joint_w)

    def forward(self, pred_dict, gt_dict):
        loss_dict = {}
        device = pred_dict['joint_angles'].device
        total_loss = torch.tensor(0.0, device=device)

        # 🚀 [개선] Sin/Cos 기반 손실 (각도 주기성 문제 해결)
        pred_sc = pred_dict.get('pred_sin_cos', None)
        gt_angles = gt_dict['angles']
        B, n_angle = gt_angles.shape

        if pred_sc is not None:
            # pred_sc: (B, num_angles*2) = [cos0, sin0, cos1, sin1, ...]
            pred_cos = pred_sc[:, 0::2]  # (B, n_angle)
            pred_sin = pred_sc[:, 1::2]

            # 🚀 [개선] Sin/Cos 정규화 (unit circle 강제)
            # Reshape to (B, n_angle, 2) for normalization
            pred_sc_norm = torch.sqrt(pred_cos**2 + pred_sin**2).clamp(min=1e-8)
            pred_cos_norm = pred_cos / pred_sc_norm
            pred_sin_norm = pred_sin / pred_sc_norm

            # GT sin/cos from angles
            gt_angles_sc = gt_angles.clone()
            if self.fix_joint7 and gt_angles_sc.shape[1] >= 7:
                # 🚀 Fix joint 7 (index 6) to 0 for consistency
                gt_angles_sc[:, 6] = 0.0

            gt_cos = torch.cos(gt_angles_sc)
            gt_sin = torch.sin(gt_angles_sc)

            # Weighted SmoothL1Loss on sin/cos per joint
            w = self.joint_weights[:pred_cos_norm.shape[1]].unsqueeze(0)  # (1, n_angle)
            cos_loss = (self.loss_fn(pred_cos_norm, gt_cos) * w).mean()
            sin_loss = (self.loss_fn(pred_sin_norm, gt_sin) * w).mean()
            sc_loss = cos_loss + sin_loss

            # 🚀 [신규] Norm penalty: ||[cos, sin]|| should be 1.0
            # Encourage numerical stability on unit circle
            norm_penalty = torch.mean((pred_sc_norm - 1.0)**2)

            combined_sc_loss = sc_loss + 0.1 * norm_penalty
            total_loss = total_loss + self.angle_weight * combined_sc_loss
            loss_dict['loss/sin_cos'] = sc_loss.item()
            loss_dict['loss/norm_penalty'] = norm_penalty.item()

        # FK loss is disabled during training (use only for validation metric)
        if self.fk_3d_weight > 0 and 'keypoints_3d_fk' in pred_dict:
            pred_kp_robot = pred_dict['keypoints_3d_fk']
            gt_angles_fk = gt_angles.clone()
            if self.fix_joint7 and gt_angles_fk.shape[1] >= 7:
                gt_angles_fk[:, 6] = 0.0

            gt_kp_robot = panda_forward_kinematics(gt_angles_fk)
            fk_loss = self.loss_fn(pred_kp_robot, gt_kp_robot).mean()

            total_loss = total_loss + self.fk_3d_weight * fk_loss
            loss_dict['metric/fk_3d_robot'] = fk_loss.item()

        # PnP metric (validation only, no gradient)
        if self.compute_pnp_metric and 'keypoints_3d_cam' in pred_dict and 'keypoints_3d' in gt_dict:
            pred_kp_cam = pred_dict['keypoints_3d_cam']
            gt_kp_cam = gt_dict['keypoints_3d']
            pnp_valid = pred_dict.get('pnp_valid', torch.ones(pred_kp_cam.shape[0], dtype=torch.bool, device=device))

            if pnp_valid.any():
                with torch.no_grad():
                    pnp_metric = self.loss_fn(pred_kp_cam[pnp_valid], gt_kp_cam[pnp_valid]).mean()
                    loss_dict['metric/pnp_3d'] = pnp_metric.item()
                    loss_dict['metric/pnp_valid_ratio'] = pnp_valid.float().mean().item()

        loss_dict['loss/total'] = total_loss.item()
        return total_loss, loss_dict


def main(args):
    # DDP Initialization
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', timeout=timedelta(minutes=30))
        device = torch.device(f'cuda:{local_rank}')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rank = 0
        world_size = 1

    is_main = rank == 0
    output_dir = Path(args.output_dir)
    if is_main: output_dir.mkdir(parents=True, exist_ok=True)
    
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    # 1. Dataset & Dataloader
    keypoint_names = ['link0', 'link2', 'link3', 'link4', 'link6', 'link7', 'hand']
    train_dataset = PoseEstimationDataset(
        data_dir=args.train_dir, keypoint_names=keypoint_names,
        image_size=(args.image_size, args.image_size), heatmap_size=(args.heatmap_size, args.heatmap_size),
        augment=not args.no_augment, include_angles=True
    )
    val_dataset = PoseEstimationDataset(
        data_dir=args.val_dir, keypoint_names=keypoint_names,
        image_size=(args.image_size, args.image_size), heatmap_size=(args.heatmap_size, args.heatmap_size),
        augment=False, include_angles=True
    )
    
    train_sampler = DistributedSampler(train_dataset) if local_rank != -1 else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if local_rank != -1 else None

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, shuffle=(train_sampler is None), num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # 2. Model Initialization
    model = DINOv3PoseEstimator(
        dino_model_name=args.model_name,
        heatmap_size=(args.heatmap_size, args.heatmap_size),
        unfreeze_blocks=0,
        fix_joint7_zero=args.fix_joint7
    ).to(device)

    # 3. Load weights
    if args.checkpoint_3d and os.path.isfile(args.checkpoint_3d):
        # Resume from a full 3D checkpoint (includes joint_angle_head)
        if is_main: print(f"==> Resuming from 3D checkpoint: {args.checkpoint_3d}")
        ckpt = torch.load(args.checkpoint_3d, map_location=device)
        ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        if is_main and missing:
            print(f"    Missing keys: {missing}")
        if is_main and unexpected:
            print(f"    Unexpected keys: {unexpected}")
    elif args.checkpoint and os.path.isfile(args.checkpoint):
        # Load 2D heatmap weights only (fresh 3D head)
        if is_main: print(f"==> Loading 2D weights from: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        model.load_state_dict(checkpoint, strict=False)

    # 4. Freeze 2D components and enable 3D head
    for param in model.backbone.parameters(): param.requires_grad = False
    for param in model.keypoint_head.parameters(): param.requires_grad = False

    for param in model.joint_angle_head.parameters(): param.requires_grad = True

    if local_rank != -1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    # 5. Optimizer & Loss & Scheduler
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    criterion = JointAnglePoseLoss(
        angle_weight=args.angle_weight,
        fk_3d_weight=args.fk_3d_weight,
        fix_joint7=args.fix_joint7,
        compute_pnp_metric=args.compute_pnp_metric
    ).to(device)
    
    if is_main and args.use_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=args)

    best_val_loss = float('inf')
    best_filt_auc = 0.0  # Best model saved by PnP filtered ADD AUC
    global_step = 0
    warmup_steps = args.warmup_steps

    # ==================== DIAGNOSTIC CHECKS (A-F) ====================
    if is_main:
        print("\n" + "="*70)
        print("DIAGNOSTIC CHECKS - Training Health Verification")
        print("="*70)

        # --- CHECK A: Optimizer parameter capture ---
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_optimizer = sum(p.numel() for g in optimizer.param_groups for p in g["params"])
        n_total = sum(p.numel() for p in model.parameters())
        print(f"\n[A] PARAMETER CHECK:")
        print(f"  Total params:     {n_total:,}")
        print(f"  Trainable params: {n_trainable:,}")
        print(f"  Optimizer params: {n_optimizer:,}")
        if n_trainable == 0:
            print(f"  *** CRITICAL: No trainable parameters! Training will NOT learn. ***")
        elif n_optimizer == 0:
            print(f"  *** CRITICAL: Optimizer has 0 params! Gradients computed but never applied. ***")
        elif n_trainable != n_optimizer:
            print(f"  *** WARNING: Trainable ({n_trainable}) != Optimizer ({n_optimizer}) mismatch ***")
        else:
            print(f"  OK: Optimizer captures all {n_trainable:,} trainable params")

        # --- CHECK B: Data/label sanity (first sample) ---
        print(f"\n[B] DATA/LABEL SANITY CHECK (first train sample):")
        sample0 = train_dataset[0]
        angles0 = sample0['angles']
        print(f"  angles shape: {angles0.shape}, dtype: {angles0.dtype}")
        print(f"  angles (rad): {angles0.tolist()}")
        angles_deg = angles0 * 180.0 / math.pi
        print(f"  angles (deg): {[f'{a:.1f}' for a in angles_deg.tolist()]}")
        # Check if angles are within Panda joint limits (rad)
        panda_limits = [(-2.8973, 2.8973), (-1.7628, 1.7628), (-2.8973, 2.8973),
                        (-3.0718, -0.0698), (-2.8973, 2.8973), (-0.0175, 3.7525), (-2.8973, 2.8973)]
        for j in range(min(7, len(angles0))):
            lo, hi = panda_limits[j]
            a = angles0[j].item()
            in_range = lo <= a <= hi
            print(f"    Joint {j}: {a:.4f} rad, limit=[{lo:.4f}, {hi:.4f}] {'OK' if in_range else '*** OUT OF RANGE ***'}")

        # Check cos^2 + sin^2 ≈ 1 for GT
        gt_cos = torch.cos(angles0)
        gt_sin = torch.sin(angles0)
        gt_norm = gt_cos**2 + gt_sin**2
        print(f"  GT cos²+sin² (should ≈ 1.0): {gt_norm.tolist()}")

        # Check has_angles flag
        has_angles = sample0.get('has_angles', None)
        print(f"  has_angles: {has_angles}")
        if has_angles is not None and not has_angles:
            print(f"  *** WARNING: has_angles=False → using dummy zero angles! No real GT! ***")

        # Check valid_mask
        vm = sample0['valid_mask']
        print(f"  valid_mask: {vm.tolist()} (sum={vm.sum().item()}/{vm.shape[0]})")
        if vm.sum() == 0:
            print(f"  *** CRITICAL: All keypoints invalid in sample 0 ***")

        # --- CHECK C: Image/Camera K/Resize pipeline ---
        print(f"\n[C] CAMERA K / COORDINATE PIPELINE CHECK:")
        kp_3d = sample0['keypoints_3d']
        camera_K = sample0['camera_K']
        orig_size = sample0['original_size']
        kp_2d = sample0['keypoints']
        print(f"  original_size (W, H): {orig_size.tolist()}")
        print(f"  camera_K:\n    fx={camera_K[0,0]:.2f}, fy={camera_K[1,1]:.2f}, cx={camera_K[0,2]:.2f}, cy={camera_K[1,2]:.2f}")
        print(f"  2D keypoints range: x=[{kp_2d[:,0].min():.1f}, {kp_2d[:,0].max():.1f}], y=[{kp_2d[:,1].min():.1f}, {kp_2d[:,1].max():.1f}]")
        print(f"  heatmap_size: {args.heatmap_size}x{args.heatmap_size}")
        print(f"  image_size: {args.image_size}x{args.image_size}")

        # Check if 2D points are in heatmap range
        x_max, y_max = kp_2d[:,0].max().item(), kp_2d[:,1].max().item()
        if x_max > args.heatmap_size or y_max > args.heatmap_size:
            print(f"  *** WARNING: 2D keypoints exceed heatmap size! ***")
        else:
            print(f"  2D keypoints within heatmap bounds: OK")

        # Scale K as done in val loop
        sx = args.heatmap_size / orig_size[0].item()
        sy = args.heatmap_size / orig_size[1].item()
        print(f"  K scale factors: sx={sx:.4f}, sy={sy:.4f}")
        print(f"  Scaled K: fx={camera_K[0,0]*sx:.2f}, fy={camera_K[1,1]*sy:.2f}, cx={camera_K[0,2]*sx:.2f}, cy={camera_K[1,2]*sy:.2f}")

        # Check 3D keypoints
        print(f"  3D keypoints (camera frame): shape={kp_3d.shape}")
        print(f"    x range: [{kp_3d[:,0].min():.4f}, {kp_3d[:,0].max():.4f}] m")
        print(f"    y range: [{kp_3d[:,1].min():.4f}, {kp_3d[:,1].max():.4f}] m")
        print(f"    z range: [{kp_3d[:,2].min():.4f}, {kp_3d[:,2].max():.4f}] m")
        if kp_3d[:,2].min() < 0:
            print(f"  *** WARNING: Negative Z (behind camera) in 3D keypoints ***")

        # --- CHECK D: LR/Schedule info ---
        print(f"\n[D] LR/SCHEDULE INFO:")
        print(f"  Initial LR: {args.lr}")
        print(f"  Min LR: {args.min_lr}")
        print(f"  Warmup steps: {args.warmup_steps}")
        print(f"  Scheduler: CosineAnnealing T_max={args.epochs}")
        print(f"  Backbone frozen: {not any(p.requires_grad for p in (model.module.backbone.parameters() if hasattr(model, 'module') else model.backbone.parameters()))}")

        # --- CHECK E: valid_mask in loss ---
        print(f"\n[E] VALID_MASK IN LOSS CHECK:")
        print(f"  Loss function uses valid_mask: NO (loss uses all samples)")
        print(f"  valid_mask is only used in validation metrics")
        # Scan a few samples for valid_mask coverage
        n_check = min(50, len(train_dataset))
        vm_sums = []
        has_angles_count = 0
        for idx in range(n_check):
            s = train_dataset[idx]
            vm_sums.append(s['valid_mask'].sum().item())
            if s.get('has_angles', torch.tensor(False)).item():
                has_angles_count += 1
        print(f"  Checked {n_check} samples: valid_mask avg={np.mean(vm_sums):.2f}/7, min={min(vm_sums)}, max={max(vm_sums)}")
        print(f"  Samples with real angles: {has_angles_count}/{n_check}")
        if has_angles_count == 0:
            print(f"  *** CRITICAL: NO samples have real angle GT! Training on dummy zeros! ***")

        # --- CHECK F: Data complexity overview ---
        print(f"\n[F] DATA OVERVIEW:")
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Val samples:   {len(val_dataset)}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Steps/epoch: ~{len(train_dataset) // (args.batch_size * world_size)}")
        print(f"  fix_joint7: {args.fix_joint7}")

        print("\n" + "="*70)
        print("END DIAGNOSTIC CHECKS")
        print("="*70 + "\n")

    # 6. Training Loop
    for epoch in range(args.epochs):
        if train_sampler: train_sampler.set_epoch(epoch)

        model.train()
        train_loss_accum = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]") if is_main else train_loader
        
        first_batch_checked = (global_step > 0)  # Skip if resuming

        for batch in pbar:
            if global_step < warmup_steps:
                curr_lr = args.min_lr + (args.lr - args.min_lr) * (global_step / warmup_steps)
                set_lr(optimizer, curr_lr)

            imgs = batch['image'].to(device)

            gt_dict = {
                'angles': batch['angles'].to(device),
                'valid_mask': batch['valid_mask'].to(device),
                'keypoints_3d': batch['keypoints_3d'].to(device)  # (B, N, 3) - camera frame from JSON
            }

            # [CHECK A.2] Clone weights before first step to verify update
            if not first_batch_checked and is_main:
                weight_snapshots = {}
                for name, p in model.named_parameters():
                    if p.requires_grad:
                        weight_snapshots[name] = p.data.clone()

            optimizer.zero_grad()
            preds = model(imgs)  # No camera_K during training (PnP is val-only metric)
            loss, loss_dict = criterion(preds, gt_dict)

            # [CHECK G] Feature & prediction analysis (first batch only)
            if not first_batch_checked and is_main:
                with torch.no_grad():
                    raw_model = model.module if hasattr(model, 'module') else model

                    # G.1: Backbone feature stats (frozen input to angle head)
                    dino_feat = raw_model.backbone(imgs)
                    print(f"\n[G.1] BACKBONE FEATURE (frozen, input to angle head):")
                    print(f"  shape: {dino_feat.shape}")  # (B, N_tokens, D)
                    print(f"  mean={dino_feat.mean():.4f}, std={dino_feat.std():.4f}")
                    print(f"  min={dino_feat.min():.4f}, max={dino_feat.max():.4f}")

                    # G.2: Global avg pooled feature (what angle head actually sees)
                    B_f = dino_feat.shape[0]
                    h_f = w_f = int(math.sqrt(dino_feat.shape[1]))
                    feat_2d = dino_feat.permute(0, 2, 1).reshape(B_f, dino_feat.shape[2], h_f, w_f)
                    global_pooled = F.adaptive_avg_pool2d(feat_2d, 1).flatten(1)  # (B, D)
                    print(f"\n[G.2] GLOBAL POOLED FEATURE:")
                    print(f"  shape: {global_pooled.shape}")
                    print(f"  mean={global_pooled.mean():.4f}, std={global_pooled.std():.4f}")

                    # Cross-sample similarity: are different images producing similar features?
                    if global_pooled.shape[0] >= 2:
                        cos_sims = []
                        for i in range(min(global_pooled.shape[0], 8)):
                            for j in range(i+1, min(global_pooled.shape[0], 8)):
                                sim = F.cosine_similarity(global_pooled[i:i+1], global_pooled[j:j+1]).item()
                                cos_sims.append(sim)
                        print(f"  Cross-sample cosine similarity: mean={np.mean(cos_sims):.4f}, min={min(cos_sims):.4f}, max={max(cos_sims):.4f}")
                        if np.mean(cos_sims) > 0.95:
                            print(f"  *** WARNING: Features nearly identical across samples! Backbone may not discriminate poses. ***")

                    # G.3: Predicted sin/cos analysis
                    pred_sc = preds['pred_sin_cos']
                    pred_angles = preds['joint_angles']
                    gt_angles = gt_dict['angles']
                    print(f"\n[G.3] PREDICTION ANALYSIS:")
                    print(f"  pred_sin_cos shape: {pred_sc.shape}")
                    pred_cos = pred_sc[:, 0::2]
                    pred_sin = pred_sc[:, 1::2]
                    for j in range(pred_angles.shape[1]):
                        print(f"  Joint {j}: pred_angle=[{pred_angles[:,j].min():.3f}, {pred_angles[:,j].max():.3f}] "
                              f"gt_angle=[{gt_angles[:,j].min():.3f}, {gt_angles[:,j].max():.3f}] "
                              f"pred_cos=[{pred_cos[:,j].min():.3f}, {pred_cos[:,j].max():.3f}] "
                              f"pred_sin=[{pred_sin[:,j].min():.3f}, {pred_sin[:,j].max():.3f}]")

                    # G.4: Are predictions collapsed? (all samples predicting same angle)
                    pred_std = pred_angles.std(dim=0)
                    gt_std = gt_angles.std(dim=0)
                    print(f"\n[G.4] PREDICTION DIVERSITY (std across batch):")
                    for j in range(pred_angles.shape[1]):
                        ratio = pred_std[j].item() / (gt_std[j].item() + 1e-8)
                        print(f"  Joint {j}: pred_std={pred_std[j]:.4f}, gt_std={gt_std[j]:.4f}, ratio={ratio:.4f}")
                    if pred_std.mean() < 0.01:
                        print(f"  *** WARNING: Predictions collapsed to near-constant! Model ignoring input. ***")

                    # G.5: Spatial feature (heatmap-derived) analysis
                    heatmaps = preds['heatmaps_2d']
                    uv = soft_argmax_2d(heatmaps, temperature=10.0)
                    hm_h, hm_w = heatmaps.shape[2:]
                    u_n = uv[:, :, 0] / hm_w
                    v_n = uv[:, :, 1] / hm_h
                    print(f"\n[G.5] SPATIAL FEATURE (heatmap UV, normalized):")
                    for j in range(uv.shape[1]):
                        print(f"  Joint {j}: u=[{u_n[:,j].min():.3f},{u_n[:,j].max():.3f}] v=[{v_n[:,j].min():.3f},{v_n[:,j].max():.3f}]")

            loss.backward()

            # [CHECK A.3] Gradient statistics before clipping
            if not first_batch_checked and is_main:
                grad_norms = {}
                for name, p in model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        grad_norms[name] = p.grad.data.norm().item()
                total_grad = sum(grad_norms.values())
                print(f"\n[A.3] GRADIENT CHECK (first batch):")
                print(f"  Total grad L2 sum: {total_grad:.6f}")
                if total_grad == 0:
                    print(f"  *** CRITICAL: All gradients are ZERO! Loss is not connected to parameters. ***")
                else:
                    sorted_grads = sorted(grad_norms.items(), key=lambda x: x[1], reverse=True)
                    print(f"  Top 5 grad norms:")
                    for name, gn in sorted_grads[:5]:
                        print(f"    {name}: {gn:.6f}")

                # [CHECK B.2] GT sin/cos sanity on actual batch
                gt_ang = gt_dict['angles']
                gt_c, gt_s = torch.cos(gt_ang), torch.sin(gt_ang)
                gt_norm_check = (gt_c**2 + gt_s**2).mean(dim=0)
                print(f"\n[B.2] GT cos²+sin² per joint (batch avg, should ≈ 1.0): {[f'{v:.6f}' for v in gt_norm_check.tolist()]}")

                # Check angle range in batch
                print(f"[B.3] GT angle range per joint (rad):")
                for j in range(gt_ang.shape[1]):
                    print(f"  Joint {j}: min={gt_ang[:,j].min():.4f}, max={gt_ang[:,j].max():.4f}, mean={gt_ang[:,j].mean():.4f}")

                # [CHECK E.2] valid_mask coverage in this batch
                vm_batch = gt_dict['valid_mask']
                if vm_batch.dim() > 1:
                    vm_sum = vm_batch.all(dim=1).sum().item()
                else:
                    vm_sum = vm_batch.sum().item()
                print(f"\n[E.2] valid_mask in batch: {vm_sum}/{vm_batch.shape[0]} samples fully valid")

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            # [CHECK A.2 continued] Verify weights actually changed
            if not first_batch_checked and is_main:
                print(f"\n[A.2] WEIGHT UPDATE CHECK (first step):")
                max_diffs = {}
                for name, p in model.named_parameters():
                    if p.requires_grad and name in weight_snapshots:
                        diff = (p.data - weight_snapshots[name]).abs().max().item()
                        max_diffs[name] = diff
                total_diff = sum(max_diffs.values())
                if total_diff == 0:
                    print(f"  *** CRITICAL: Weights did NOT change after optimizer.step()! ***")
                else:
                    sorted_diffs = sorted(max_diffs.items(), key=lambda x: x[1], reverse=True)
                    print(f"  Weights changed. Top 5 max abs diffs:")
                    for name, d in sorted_diffs[:5]:
                        print(f"    {name}: {d:.8f}")
                del weight_snapshots
                first_batch_checked = True

            train_loss_accum += loss.item()
            global_step += 1

            if is_main:
                postfix = {'total': f"{loss.item():.4f}", 'sin_cos': f"{loss_dict.get('loss/sin_cos', 0):.4f}", 'lr': f"{optimizer.param_groups[0]['lr']:.2e}"}
                pbar.set_postfix(postfix)
                if args.use_wandb:
                    wandb.log({f"train/{k}": v for k, v in loss_dict.items()})
                    wandb.log({"train/lr": optimizer.param_groups[0]['lr']})

        # Validation
        model.eval()
        val_loss_accum = 0.0
        viz_data = None
        epoch_filt_auc = 0.0  # Track filtered AUC for this epoch
        epoch_pnp_valid_ratio = 0.0
        max_val_batches = max(1, int(len(val_loader) * args.val_ratio))
        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch} [Val]") if is_main else val_loader):
                if i >= max_val_batches:
                    break
                imgs = batch['image'].to(device)

                # Scale camera_K from original image size to heatmap size
                camera_K = batch['camera_K'].to(device)  # (B, 3, 3) - original resolution
                original_size = batch['original_size'].to(device)  # (B, 2) [W, H]
                heatmap_size = torch.tensor([args.heatmap_size, args.heatmap_size], device=device, dtype=original_size.dtype)

                # Compute scale factors
                scale_x = heatmap_size[0] / original_size[:, 0]  # (B,)
                scale_y = heatmap_size[1] / original_size[:, 1]  # (B,)

                # Scale camera matrix K
                camera_K_scaled = camera_K.clone()
                camera_K_scaled[:, 0, 0] *= scale_x  # fx
                camera_K_scaled[:, 1, 1] *= scale_y  # fy
                camera_K_scaled[:, 0, 2] *= scale_x  # cx
                camera_K_scaled[:, 1, 2] *= scale_y  # cy

                gt_dict = {
                    'angles': batch['angles'].to(device),
                    'valid_mask': batch['valid_mask'].to(device),
                    'keypoints_3d': batch['keypoints_3d'].to(device)
                }
                preds = model(imgs, camera_K=camera_K_scaled)
                loss, loss_dict = criterion(preds, gt_dict)
                val_loss_accum += loss.item()

                # 첫 번째 배치의 데이터를 시각화용으로 캡처
                if i == 0 and is_main:
                    gt_angles = batch['angles'].to(device)
                    if args.fix_joint7:
                        gt_angles = gt_angles.clone()
                        gt_angles[:, 6] = 0.0
                    gt_kp_3d = panda_forward_kinematics(gt_angles)
                    # 🚀 (Images, GT_3D, Pred_3D, Pred_Heatmaps) 전달
                    pred_3d = preds['keypoints_3d_fk']
                    viz_data = (imgs, gt_kp_3d, pred_3d, preds['heatmaps_2d'])

        if local_rank != -1:
            val_loss_tensor = torch.tensor([val_loss_accum], device=device)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            avg_val_loss = val_loss_tensor.item() / (max_val_batches * world_size)
        else:
            avg_val_loss = val_loss_accum / max_val_batches

        if is_main:
            val_pct = int(args.val_ratio * 100)
            print(f"Epoch {epoch} | Val Loss: {avg_val_loss:.4f} (using {val_pct}% of val set, {max_val_batches}/{len(val_loader)} batches)")

            # 🚀 [DEBUG] 상세 메트릭 분석
            if 'pred_sin_cos' in preds:
                with torch.no_grad():
                    pred_angles = preds['joint_angles']  # (B, num_angles)
                    gt_angles = gt_dict['angles']  # (B, num_angles)

                    # 🚀 Apply fix_joint7 to GT for consistency with pred_angles
                    if args.fix_joint7:
                        gt_angles = gt_angles.clone()
                        gt_angles[:, 6] = 0.0

                    # Camera-frame 3D (PnP transformed) for evaluation
                    pred_kp_3d = preds['keypoints_3d_cam']  # (B, 7, 3) - camera frame
                    gt_kp_3d = gt_dict['keypoints_3d']  # (B, 7, 3) - camera frame (from JSON)

                    # ==================== 관절 각도 에러 ====================
                    # 방법 1: Angle space에서 계산 (wrap-aware)
                    angle_diff = pred_angles - gt_angles
                    angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
                    angle_error_deg = torch.abs(angle_diff) * 180.0 / math.pi

                    # Debug: Raw angle 범위 출력 (첫 epoch에만)
                    if is_main and epoch == 0 and i == 0:
                        print(f"\n[DEBUG] Joint 0 angle analysis:")
                        for j in [0, min(1, preds['joint_angles'].shape[1]-1)]:
                            print(f"  Joint {j}:")
                            print(f"    GT:   min={gt_angles[:, j].min():.4f}, max={gt_angles[:, j].max():.4f}, mean={gt_angles[:, j].mean():.4f}")
                            print(f"    Pred: min={pred_angles[:, j].min():.4f}, max={pred_angles[:, j].max():.4f}, mean={pred_angles[:, j].mean():.4f}")
                            print(f"    Error (deg): min={angle_error_deg[:, j].min():.2f}, max={angle_error_deg[:, j].max():.2f}")

                    # 🚀 [개선] valid_mask 적용
                    valid_mask = gt_dict.get('valid_mask', torch.ones(angle_error_deg.shape[0], dtype=torch.bool, device=device))
                    if valid_mask.dim() > 1:
                        valid_mask = valid_mask.all(dim=1)  # (B, 7) → (B,)

                    # Only compute metrics for valid samples
                    if valid_mask.any():
                        # Mask invalid samples using multiplication: (B,) → (B, 1)
                        valid_mask_float = valid_mask.float().unsqueeze(1)  # (B, 1)
                        angle_error_deg_masked = angle_error_deg * valid_mask_float  # Element-wise multiply

                        # Average only over valid samples
                        mae_per_joint = (angle_error_deg_masked.sum(dim=0) / valid_mask.float().sum()).clamp(min=0)
                        max_error_per_joint = angle_error_deg.max(dim=0)[0]
                    else:
                        mae_per_joint = torch.zeros_like(angle_error_deg[0])
                        max_error_per_joint = torch.zeros_like(angle_error_deg[0])

                    # ==================== 3D 복원 오차 (camera frame) ====================
                    kp_error_mm = torch.norm(pred_kp_3d - gt_kp_3d, dim=2) * 1000  # m → mm

                    # PnP quality mask: reprojection error + depth validated
                    pnp_valid = preds.get('pnp_valid', torch.ones(kp_error_mm.shape[0], dtype=torch.bool, device=device))
                    reproj_errors = preds.get('reproj_errors', torch.zeros(kp_error_mm.shape[0], device=device))
                    if pnp_valid.dim() > 1:
                        pnp_valid = pnp_valid.all(dim=1)

                    # Combined mask: valid keypoints AND quality-validated PnP
                    combined_mask = valid_mask & pnp_valid
                    n_total = valid_mask.shape[0]
                    n_pnp_valid = combined_mask.sum().item()

                    kp_error_m = torch.norm(pred_kp_3d - gt_kp_3d, dim=2)  # (B, 7) in meters
                    add_auc_thresh_m = 0.1  # 0.1m = 100mm (RoboPEPP standard)

                    # --- ALL samples (valid_mask only, no PnP filter) ---
                    if valid_mask.any():
                        all_kp_error = kp_error_mm[valid_mask]
                        all_mean_3d = all_kp_error.mean().item()
                        all_median_3d = torch.median(all_kp_error).item()
                        all_mae_per_joint = all_kp_error.mean(dim=0)
                        all_max_per_joint = all_kp_error.max(dim=0)[0]
                        all_add_auc, _ = compute_add_auc(kp_error_m[valid_mask], add_auc_thresh_m)
                        n_all = valid_mask.sum().item()
                    else:
                        all_mean_3d = all_median_3d = 0.0
                        all_mae_per_joint = torch.zeros(7, device=device)
                        all_max_per_joint = torch.zeros(7, device=device)
                        all_add_auc = 0.0
                        n_all = 0

                    # --- Filtered samples (Iterative PnP quality validated) ---
                    if combined_mask.any():
                        filt_kp_error = kp_error_mm[combined_mask]
                        filt_mean_3d = filt_kp_error.mean().item()
                        filt_median_3d = torch.median(filt_kp_error).item()
                        filt_mae_per_joint = filt_kp_error.mean(dim=0)
                        filt_max_per_joint = filt_kp_error.max(dim=0)[0]
                        filt_add_auc, _ = compute_add_auc(kp_error_m[combined_mask], add_auc_thresh_m)
                    else:
                        filt_mean_3d = filt_median_3d = 0.0
                        filt_mae_per_joint = torch.zeros(7, device=device)
                        filt_max_per_joint = torch.zeros(7, device=device)
                        filt_add_auc = 0.0

                    # --- RANSAC EPnP metrics ---
                    ransac_3d_cam = preds.get('keypoints_3d_cam_ransac')
                    ransac_valid = preds.get('pnp_valid_ransac', torch.zeros(n_total, dtype=torch.bool, device=device))
                    ransac_reproj = preds.get('reproj_errors_ransac', torch.zeros(n_total, device=device))
                    ransac_n_inliers = preds.get('pnp_n_inliers_ransac', torch.zeros(n_total, dtype=torch.int32, device=device))
                    if ransac_valid.dim() > 1:
                        ransac_valid = ransac_valid.all(dim=1)
                    ransac_mask = valid_mask & ransac_valid

                    if ransac_3d_cam is not None and ransac_mask.any():
                        ransac_kp_error_mm = torch.norm(ransac_3d_cam - gt_kp_3d, dim=2) * 1000
                        ransac_kp_error_m = torch.norm(ransac_3d_cam - gt_kp_3d, dim=2)
                        r_kp_error = ransac_kp_error_mm[ransac_mask]
                        r_mean_3d = r_kp_error.mean().item()
                        r_median_3d = torch.median(r_kp_error).item()
                        r_mae_per_joint = r_kp_error.mean(dim=0)
                        r_max_per_joint = r_kp_error.max(dim=0)[0]
                        r_add_auc, _ = compute_add_auc(ransac_kp_error_m[ransac_mask], add_auc_thresh_m)
                        n_ransac_valid = ransac_mask.sum().item()
                        r_reproj_valid = ransac_reproj[ransac_mask]
                        r_reproj_mean = r_reproj_valid.mean().item()
                        r_inliers_valid = ransac_n_inliers[ransac_mask].float()
                        r_inliers_mean = r_inliers_valid.mean().item()
                    else:
                        r_mean_3d = r_median_3d = 0.0
                        r_mae_per_joint = torch.zeros(7, device=device)
                        r_max_per_joint = torch.zeros(7, device=device)
                        r_add_auc = 0.0
                        n_ransac_valid = 0
                        r_reproj_mean = 0.0
                        r_inliers_mean = 0.0

                    # --- Confidence-filtered RANSAC metrics ---
                    conf_3d_cam = preds.get('keypoints_3d_cam_conf')
                    conf_valid = preds.get('pnp_valid_conf', torch.zeros(n_total, dtype=torch.bool, device=device))
                    conf_reproj = preds.get('reproj_errors_conf', torch.zeros(n_total, device=device))
                    conf_n_used = preds.get('pnp_n_used_conf', torch.zeros(n_total, dtype=torch.int32, device=device))
                    if conf_valid.dim() > 1:
                        conf_valid = conf_valid.all(dim=1)
                    conf_mask = valid_mask & conf_valid

                    if conf_3d_cam is not None and conf_mask.any():
                        conf_kp_error_mm = torch.norm(conf_3d_cam - gt_kp_3d, dim=2) * 1000
                        conf_kp_error_m = torch.norm(conf_3d_cam - gt_kp_3d, dim=2)
                        c_kp_error = conf_kp_error_mm[conf_mask]
                        c_mean_3d = c_kp_error.mean().item()
                        c_median_3d = torch.median(c_kp_error).item()
                        c_mae_per_joint = c_kp_error.mean(dim=0)
                        c_add_auc, _ = compute_add_auc(conf_kp_error_m[conf_mask], add_auc_thresh_m)
                        n_conf_valid = conf_mask.sum().item()
                        c_reproj_mean = conf_reproj[conf_mask].mean().item()
                        c_n_used_mean = conf_n_used[conf_mask].float().mean().item()
                    else:
                        c_mean_3d = c_median_3d = 0.0
                        c_mae_per_joint = torch.zeros(7, device=device)
                        c_add_auc = 0.0
                        n_conf_valid = 0
                        c_reproj_mean = 0.0
                        c_n_used_mean = 0.0

                    # ==================== 콘솔 로그 ====================
                    print("\n" + "="*60)
                    print(f"JOINT ANGLE ERROR (deg)")
                    print("="*60)
                    for j in range(len(mae_per_joint)):
                        print(f"  Joint {j}: MAE={mae_per_joint[j].item():.2f}°, Max={max_error_per_joint[j].item():.2f}°")

                    worst_joint = mae_per_joint.argmax()
                    print(f"  → Worst: Joint {worst_joint.item()} ({mae_per_joint[worst_joint].item():.2f}°)")

                    # 🔴 SANITY CHECK: Sin/Cos loss와 Angle error 간 일관성 검증
                    if 'pred_sin_cos' in preds and is_main:
                        pred_sc = preds['pred_sin_cos']
                        pred_cos_val = pred_sc[:, 0::2]
                        pred_sin_val = pred_sc[:, 1::2]
                        gt_cos_val = torch.cos(gt_angles)
                        gt_sin_val = torch.sin(gt_angles)

                        sc_diff = torch.sqrt((pred_cos_val - gt_cos_val)**2 + (pred_sin_val - gt_sin_val)**2)
                        sc_mae_per_joint = sc_diff.mean(dim=0)

                        print(f"\n{'='*60}")
                        print(f"[SANITY CHECK] Sin/Cos space vs Angle space MAE")
                        print("="*60)
                        for j in range(len(mae_per_joint)):
                            print(f"  Joint {j}: angle_mae={mae_per_joint[j].item():.2f}°, sc_mae={sc_mae_per_joint[j].item():.4f}")
                        print("="*60)

                    print(f"\n{'='*60}")
                    print(f"3D POSE ERROR - ALL SAMPLES ({n_all}/{n_total})")
                    print("="*60)
                    for j in range(len(all_mae_per_joint)):
                        print(f"  Joint {j}: MAE={all_mae_per_joint[j].item():.2f}mm, Max={all_max_per_joint[j].item():.2f}mm")
                    print(f"  → Mean: {all_mean_3d:.2f}mm, Median: {all_median_3d:.2f}mm")
                    print(f"  → ADD AUC@{add_auc_thresh_m*1000:.0f}mm (RoboPEPP): {all_add_auc:.4f}")

                    # Reprojection error stats
                    if combined_mask.any():
                        valid_reproj = reproj_errors[combined_mask]
                        reproj_mean = valid_reproj.mean().item()
                        reproj_max = valid_reproj.max().item()
                    else:
                        reproj_mean = reproj_max = 0.0

                    print(f"\n{'='*60}")
                    print(f"3D POSE ERROR - PnP FILTERED ({n_pnp_valid}/{n_total}, reproj<5px & depth OK)")
                    print("="*60)
                    for j in range(len(filt_mae_per_joint)):
                        print(f"  Joint {j}: MAE={filt_mae_per_joint[j].item():.2f}mm, Max={filt_max_per_joint[j].item():.2f}mm")
                    print(f"  → Mean: {filt_mean_3d:.2f}mm, Median: {filt_median_3d:.2f}mm")
                    print(f"  → Reproj RMSE: mean={reproj_mean:.2f}px, max={reproj_max:.2f}px")
                    print(f"  → ADD AUC@{add_auc_thresh_m*1000:.0f}mm (RoboPEPP): {filt_add_auc:.4f}")

                    print(f"\n{'='*60}")
                    print(f"3D POSE ERROR - RANSAC EPnP ({n_ransac_valid}/{n_total})")
                    print("="*60)
                    for j in range(len(r_mae_per_joint)):
                        print(f"  Joint {j}: MAE={r_mae_per_joint[j].item():.2f}mm, Max={r_max_per_joint[j].item():.2f}mm")
                    print(f"  → Mean: {r_mean_3d:.2f}mm, Median: {r_median_3d:.2f}mm")
                    print(f"  → Reproj RMSE: mean={r_reproj_mean:.2f}px, Inliers: mean={r_inliers_mean:.1f}/7")
                    print(f"  → ADD AUC@{add_auc_thresh_m*1000:.0f}mm (RoboPEPP): {r_add_auc:.4f}")

                    print(f"\n{'='*60}")
                    print(f"3D POSE ERROR - CONF-FILTERED RANSAC ({n_conf_valid}/{n_total})")
                    print("="*60)
                    for j in range(len(c_mae_per_joint)):
                        print(f"  Joint {j}: MAE={c_mae_per_joint[j].item():.2f}mm")
                    print(f"  → Mean: {c_mean_3d:.2f}mm, Median: {c_median_3d:.2f}mm")
                    print(f"  → Reproj RMSE: mean={c_reproj_mean:.2f}px, KPs used: mean={c_n_used_mean:.1f}/7")
                    print(f"  → ADD AUC@{add_auc_thresh_m*1000:.0f}mm (RoboPEPP): {c_add_auc:.4f}")
                    print("="*60 + "\n")

                    # Capture filtered AUC for best model selection
                    epoch_filt_auc = filt_add_auc
                    epoch_pnp_valid_ratio = n_pnp_valid / n_total if n_total > 0 else 0.0

                    # ==================== WandB 로깅 ====================
                    if args.use_wandb:
                        wandb_logs = {
                            "val/loss": avg_val_loss,
                            "epoch": epoch,
                        }

                        # 관절별 각도 에러
                        for j in range(len(mae_per_joint)):
                            wandb_logs[f"val/joint_{j}_angle_mae_deg"] = mae_per_joint[j].item()
                            wandb_logs[f"val/joint_{j}_angle_max_deg"] = max_error_per_joint[j].item()

                        # 관절별 3D 오차 (filtered)
                        for j in range(len(filt_mae_per_joint)):
                            wandb_logs[f"val/joint_{j}_3d_mae_mm"] = filt_mae_per_joint[j].item()
                            wandb_logs[f"val/joint_{j}_3d_max_mm"] = filt_max_per_joint[j].item()

                        # ALL samples metrics
                        wandb_logs["val/3d_mean_all_mm"] = all_mean_3d
                        wandb_logs["val/3d_median_all_mm"] = all_median_3d
                        wandb_logs["val/add_auc_all"] = all_add_auc

                        # PnP filtered metrics
                        wandb_logs["val/3d_mean_filtered_mm"] = filt_mean_3d
                        wandb_logs["val/3d_median_filtered_mm"] = filt_median_3d
                        wandb_logs["val/add_auc_filtered"] = filt_add_auc
                        wandb_logs["val/pnp_valid_ratio"] = n_pnp_valid / n_total if n_total > 0 else 0.0

                        # RANSAC EPnP metrics
                        wandb_logs["val/3d_mean_ransac_mm"] = r_mean_3d
                        wandb_logs["val/3d_median_ransac_mm"] = r_median_3d
                        wandb_logs["val/add_auc_ransac"] = r_add_auc
                        wandb_logs["val/ransac_valid_ratio"] = n_ransac_valid / n_total if n_total > 0 else 0.0
                        wandb_logs["val/ransac_inliers_mean"] = r_inliers_mean

                        # Confidence-filtered RANSAC metrics
                        wandb_logs["val/3d_mean_conf_mm"] = c_mean_3d
                        wandb_logs["val/3d_median_conf_mm"] = c_median_3d
                        wandb_logs["val/add_auc_conf"] = c_add_auc
                        wandb_logs["val/conf_valid_ratio"] = n_conf_valid / n_total if n_total > 0 else 0.0

                        wandb.log(wandb_logs)
            log_dict = {"val/loss": avg_val_loss, "epoch": epoch}
            if viz_data is not None and args.use_wandb:
                log_dict["visualizations/combined_pose"] = visualize_3d_with_2d(*viz_data, num_samples=4)
            wandb.log(log_dict)
            
            # Save best model by PnP filtered ADD AUC (higher = better)
            if epoch_filt_auc > best_filt_auc:
                best_filt_auc = epoch_filt_auc
                save_model = model.module if hasattr(model, 'module') else model
                torch.save(save_model.state_dict(), output_dir / 'best_3d_pose.pth')
                print(f"  >> NEW BEST: filtered ADD AUC = {best_filt_auc:.4f} (valid_ratio={epoch_pnp_valid_ratio:.2f})")

            # Also save best by val loss (secondary)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_model = model.module if hasattr(model, 'module') else model
                torch.save(save_model.state_dict(), output_dir / 'best_val_loss.pth')

            save_model = model.module if hasattr(model, 'module') else model
            torch.save(save_model.state_dict(), output_dir / 'last_3d_pose.pth')
        
        if global_step >= warmup_steps:
            scheduler.step()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dir', type=str, required=True)
    parser.add_argument('--val-dir', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, help='Path to 2D heatmap weights')
    parser.add_argument('--checkpoint-3d', type=str, help='Path to 3D pose checkpoint to resume from')
    parser.add_argument('--output-dir', type=str, default='./outputs_3d')
    parser.add_argument('--model-name', type=str, default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    parser.add_argument('--image-size', type=int, default=512)
    parser.add_argument('--heatmap-size', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--min-lr', type=float, default=1e-7)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--warmup-steps', type=int, default=100)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--mode', type=str, default='joint_angle', choices=['joint_angle', 'direct_3d'],
                        help='3D prediction mode: joint_angle (predict angles→FK) or direct_3d (predict 3D coords directly)')
    # Joint Angle mode hyperparameters
    # 🚀 [개선] Sin/Cos 기반 손실, FK loss 비활성화
    parser.add_argument('--angle-weight', type=float, default=1.0, help='Sin/Cos loss weight')
    parser.add_argument('--fk-3d-weight', type=float, default=0.0, help='FK 3D loss weight (disabled during training, metric only)')
    # Direct 3D mode hyperparameters
    parser.add_argument('--kp-weight', type=float, default=100.0, help='3D keypoint loss weight for direct_3d mode')
    parser.add_argument('--compute-pnp-metric', action='store_true', help='Compute PnP camera-frame metric (diagnostic only, no backprop)')
    parser.add_argument('--val-ratio', type=float, default=1.0, help='Fraction of validation set to use (0.0-1.0)')
    parser.add_argument('--fix-joint7', action='store_true')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--no-augment', action='store_true')
    parser.add_argument('--occlusion-prob', type=float, default=0.5, help='Probability of occlusion augmentation')
    parser.add_argument('--occlusion-size', type=float, default=0.2, help='Max size of occlusion patch relative to image')
    parser.add_argument('--use-wandb', action='store_true')
    parser.add_argument('--wandb-project', type=str, default='dinov3-3d-pose')
    parser.add_argument('--wandb-run-name', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    main(parser.parse_args())
