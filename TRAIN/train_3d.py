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

from model import DINOv3PoseEstimator, panda_forward_kinematics, MODE_JOINT_ANGLE, MODE_DIRECT_3D
from dataset import PoseEstimationDataset

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
        self.fk_3d_weight = fk_3d_weight  # Disable FK loss during training
        self.fix_joint7 = fix_joint7
        self.compute_pnp_metric = compute_pnp_metric
        self.loss_fn = nn.SmoothL1Loss(beta=0.01)
        # 🚀 Data-based weights will be computed in forward()
        self.register_buffer('angle_weights', None)

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

            # 🚀 SmoothL1Loss on sin/cos (정규화된 pred 사용)
            sc_loss = self.loss_fn(pred_cos_norm, gt_cos) + self.loss_fn(pred_sin_norm, gt_sin)

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
            fk_loss = self.loss_fn(pred_kp_robot, gt_kp_robot)

            total_loss = total_loss + self.fk_3d_weight * fk_loss
            loss_dict['metric/fk_3d_robot'] = fk_loss.item()  # Monitoring only

        # PnP metric (validation only, no gradient)
        if self.compute_pnp_metric and 'keypoints_3d_cam' in pred_dict and 'keypoints_3d' in gt_dict:
            pred_kp_cam = pred_dict['keypoints_3d_cam']
            gt_kp_cam = gt_dict['keypoints_3d']
            pnp_valid = pred_dict.get('pnp_valid', torch.ones(pred_kp_cam.shape[0], dtype=torch.bool, device=device))

            if pnp_valid.any():
                with torch.no_grad():
                    pnp_metric = self.loss_fn(pred_kp_cam[pnp_valid], gt_kp_cam[pnp_valid])
                    loss_dict['metric/pnp_3d'] = pnp_metric.item()
                    loss_dict['metric/pnp_valid_ratio'] = pnp_valid.float().mean().item()

        loss_dict['loss/total'] = total_loss.item()
        return total_loss, loss_dict


class Direct3DPoseLoss(nn.Module):
    """Loss for Direct 3D prediction mode."""
    def __init__(self, kp_weight=100.0, compute_pnp_metric=False):
        super().__init__()
        self.kp_weight = kp_weight
        self.compute_pnp_metric = compute_pnp_metric
        self.loss_fn = nn.SmoothL1Loss(beta=0.01)

    def forward(self, pred_dict, gt_dict):
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=pred_dict['keypoints_3d'].device)

        # 1. Robot-frame 3D Loss (Direct)
        pred_kp_robot = pred_dict['keypoints_3d_robot']
        gt_angles = gt_dict['angles']
        gt_kp_robot = panda_forward_kinematics(gt_angles)

        # Full 3D keypoint loss
        kp_3d_base_loss = self.loss_fn(pred_kp_robot, gt_kp_robot)

        # End-effector specific loss (5x weight)
        kp_3d_ee_loss = self.loss_fn(pred_kp_robot[:, -1, :], gt_kp_robot[:, -1, :])

        # Combined loss
        kp_3d_loss = kp_3d_base_loss + (kp_3d_ee_loss * 5.0)

        total_loss = total_loss + self.kp_weight * kp_3d_loss
        loss_dict['loss/kp_3d_robot'] = kp_3d_loss.item()
        loss_dict['loss/kp_3d_ee_only'] = kp_3d_ee_loss.item()

        # 2. Camera-frame 3D Metric (PnP supervision, no gradient)
        if self.compute_pnp_metric and 'keypoints_3d_cam' in pred_dict and 'keypoints_3d' in gt_dict:
            pred_kp_cam = pred_dict['keypoints_3d_cam']
            gt_kp_cam = gt_dict['keypoints_3d']
            pnp_valid = pred_dict.get('pnp_valid', torch.ones(pred_kp_cam.shape[0], dtype=torch.bool, device=pred_kp_cam.device))

            if pnp_valid.any():
                with torch.no_grad():
                    cam_metric = self.loss_fn(pred_kp_cam[pnp_valid], gt_kp_cam[pnp_valid])
                    cam_ee_metric = self.loss_fn(pred_kp_cam[pnp_valid, -1, :], gt_kp_cam[pnp_valid, -1, :])
                    fk_3d_cam_metric = cam_metric + cam_ee_metric * 5.0

                loss_dict['metric/pnp_3d_cam'] = fk_3d_cam_metric.item()
                loss_dict['metric/pnp_3d_cam_ee'] = cam_ee_metric.item()
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
        fix_joint7_zero=args.fix_joint7,
        mode=args.mode
    ).to(device)

    # 3. Load 2D Heatmap weights
    if args.checkpoint and os.path.isfile(args.checkpoint):
        if is_main: print(f"==> Loading 2D weights from: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        model.load_state_dict(checkpoint, strict=False)

    # 4. Freeze 2D components and enable 3D head
    for param in model.backbone.parameters(): param.requires_grad = False
    for param in model.keypoint_head.parameters(): param.requires_grad = False

    if args.mode == MODE_JOINT_ANGLE:
        for param in model.joint_angle_head.parameters(): param.requires_grad = True
    elif args.mode == MODE_DIRECT_3D:
        for param in model.direct_3d_head.parameters(): param.requires_grad = True

    if local_rank != -1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    # 5. Optimizer & Loss & Scheduler
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    if args.mode == MODE_JOINT_ANGLE:
        criterion = JointAnglePoseLoss(
            angle_weight=args.angle_weight,
            fk_3d_weight=args.fk_3d_weight,
            fix_joint7=args.fix_joint7,
            compute_pnp_metric=args.compute_pnp_metric
        ).to(device)
    elif args.mode == MODE_DIRECT_3D:
        criterion = Direct3DPoseLoss(
            kp_weight=args.kp_weight,
            compute_pnp_metric=args.compute_pnp_metric
        ).to(device)
    
    if is_main and args.use_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=args)

    best_val_loss = float('inf')
    global_step = 0
    warmup_steps = args.warmup_steps

    # 6. Training Loop
    for epoch in range(args.epochs):
        if train_sampler: train_sampler.set_epoch(epoch)

        model.train()
        train_loss_accum = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]") if is_main else train_loader
        
        for batch in pbar:
            if global_step < warmup_steps:
                curr_lr = args.min_lr + (args.lr - args.min_lr) * (global_step / warmup_steps)
                set_lr(optimizer, curr_lr)

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
                'keypoints_3d': batch['keypoints_3d'].to(device)  # (B, N, 3) - camera frame from JSON
            }

            optimizer.zero_grad()
            preds = model(imgs, camera_K=camera_K_scaled)
            loss, loss_dict = criterion(preds, gt_dict)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

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
        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch} [Val]") if is_main else val_loader):
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
                    if args.fix_joint7 and args.mode == MODE_JOINT_ANGLE:
                        gt_angles = gt_angles.clone()
                        gt_angles[:, 6] = 0.0
                    gt_kp_3d = panda_forward_kinematics(gt_angles)
                    # 🚀 (Images, GT_3D, Pred_3D, Pred_Heatmaps) 전달
                    pred_3d = preds['keypoints_3d_fk'] if args.mode == MODE_JOINT_ANGLE else preds['keypoints_3d']
                    viz_data = (imgs, gt_kp_3d, pred_3d, preds['heatmaps_2d'])

        if local_rank != -1:
            val_loss_tensor = torch.tensor([val_loss_accum], device=device)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            avg_val_loss = val_loss_tensor.item() / (len(val_loader) * world_size)
        else:
            avg_val_loss = val_loss_accum / len(val_loader)

        if is_main:
            print(f"Epoch {epoch} | Val Loss: {avg_val_loss:.4f}")

            # 🚀 [DEBUG] 상세 메트릭 분석
            if args.mode == MODE_JOINT_ANGLE and 'pred_sin_cos' in preds:
                with torch.no_grad():
                    pred_angles = preds['joint_angles']  # (B, num_angles)
                    gt_angles = gt_dict['angles']  # (B, num_angles)

                    # 🚀 Apply fix_joint7 to GT for consistency with pred_angles
                    if args.fix_joint7:
                        gt_angles = gt_angles.clone()
                        gt_angles[:, 6] = 0.0

                    # 🚀 Use camera-frame 3D (from PnP) for evaluation
                    if 'keypoints_3d_cam' in preds:
                        # Camera-frame 3D (PnP transformed)
                        pred_kp_3d = preds['keypoints_3d_cam']  # (B, 7, 3) - camera frame
                        gt_kp_3d = gt_dict['keypoints_3d']  # (B, 7, 3) - camera frame (from JSON)
                        is_camera_frame = True
                    else:
                        # Fallback: robot-frame 3D (old behavior)
                        pred_kp_3d = preds['keypoints_3d_robot']  # (B, 7, 3)
                        gt_kp_3d = panda_forward_kinematics(gt_angles)  # (B, 7, 3)
                        is_camera_frame = False

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

                    # Only compute metrics for valid joints
                    if valid_mask.any():
                        # Expand valid_mask for broadcasting: (B,) → (B, 1)
                        valid_mask_expanded = valid_mask.unsqueeze(1)  # (B, 1)
                        angle_error_deg_masked = angle_error_deg.clone()
                        angle_error_deg_masked[~valid_mask_expanded] = 0.0  # Mask invalid

                        mae_per_joint = (angle_error_deg_masked.sum(dim=0) / valid_mask.float().sum()).clamp(min=0)
                        max_error_per_joint = angle_error_deg.max(dim=0)[0]
                    else:
                        mae_per_joint = torch.zeros_like(angle_error_deg[0])
                        max_error_per_joint = torch.zeros_like(angle_error_deg[0])

                    # ==================== 3D 복원 오차 ====================
                    # Primary: camera-frame 또는 robot-frame (선택된 frame)
                    kp_error_mm = torch.norm(pred_kp_3d - gt_kp_3d, dim=2) * 1000  # m → mm

                    # 🚀 [개선] valid_mask 적용 (3D에도)
                    if valid_mask.any():
                        kp_error_mm_masked = kp_error_mm.clone()
                        valid_mask_expanded_2d = valid_mask.unsqueeze(1)  # (B, 1)
                        kp_error_mm_masked[~valid_mask_expanded_2d] = 0.0

                        mae_3d_per_joint = (kp_error_mm_masked.sum(dim=0) / valid_mask.float().sum()).clamp(min=0)
                        max_3d_per_joint = kp_error_mm.max(dim=0)[0]

                        valid_errors = kp_error_mm[valid_mask]
                        mean_3d_error = valid_errors.mean().item()
                        median_3d_error = torch.median(valid_errors).item()
                    else:
                        mae_3d_per_joint = torch.zeros(7, device=device)
                        max_3d_per_joint = torch.zeros(7, device=device)
                        mean_3d_error = 0.0
                        median_3d_error = 0.0

                    # Auxiliary: robot-frame 메트릭 (비교용)
                    if is_camera_frame:
                        kp_error_mm_robot = torch.norm(pred_kp_3d_robot - gt_kp_3d_robot, dim=2) * 1000

                        # 🚀 [개선] valid_mask 적용 (robot-frame도)
                        if valid_mask.any():
                            kp_error_mm_robot_masked = kp_error_mm_robot.clone()
                            kp_error_mm_robot_masked[~valid_mask_expanded_2d] = 0.0
                            mae_3d_per_joint_robot = (kp_error_mm_robot_masked.sum(dim=0) / valid_mask.float().sum()).clamp(min=0)
                            valid_errors_robot = kp_error_mm_robot[valid_mask]
                            mean_3d_error_robot = valid_errors_robot.mean().item()
                            median_3d_error_robot = torch.median(valid_errors_robot).item()
                            add_auc_robot = (valid_errors_robot < d_threshold).float().mean().item()
                        else:
                            mae_3d_per_joint_robot = torch.zeros(7, device=device)
                            mean_3d_error_robot = 0.0
                            median_3d_error_robot = 0.0
                            add_auc_robot = 0.0
                    else:
                        mae_3d_per_joint_robot = None
                        mean_3d_error_robot = None

                    # ==================== ADD AUC (6D Pose Metric) ====================
                    # ADD: Average Distance of model Dpoints
                    # ADD AUC: 오차가 0.1d (d=평균 객체 크기) 이내인 비율
                    # 로봇 팔 평균 크기 ~1m → d_threshold = 100mm
                    d_threshold = 100  # mm

                    # 🚀 [개선] valid_mask 적용
                    if valid_mask.any():
                        add_errors = kp_error_mm[valid_mask].flatten()
                        add_auc = (add_errors < d_threshold).float().mean().item()
                    else:
                        add_auc = 0.0

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
                    coord_frame_label = "3D POSE ERROR - CAMERA FRAME (mm)" if is_camera_frame else "3D POSE ERROR - ROBOT FRAME (mm)"
                    print(coord_frame_label)
                    print("="*60)
                    for j in range(len(mae_3d_per_joint)):
                        print(f"  Joint {j}: MAE={mae_3d_per_joint[j].item():.2f}mm, Max={max_3d_per_joint[j].item():.2f}mm")

                    worst_3d_joint = mae_3d_per_joint.argmax()
                    print(f"  → Worst: Joint {worst_3d_joint.item()} ({mae_3d_per_joint[worst_3d_joint].item():.2f}mm)")
                    print(f"  → Overall Mean: {mean_3d_error:.2f}mm, Median: {median_3d_error:.2f}mm")

                    # Robot-frame 메트릭도 출력 (비교용)
                    if is_camera_frame and mae_3d_per_joint_robot is not None:
                        print(f"\n{'='*60}")
                        print(f"3D POSE ERROR - ROBOT FRAME (mm) [FOR COMPARISON]")
                        print("="*60)
                        print(f"  Overall Mean: {mean_3d_error_robot:.2f}mm, Median: {median_3d_error_robot:.2f}mm")
                        print(f"  ADD AUC@{d_threshold}mm: {add_auc_robot*100:.2f}%")

                    print(f"\n{'='*60}")
                    print(f"ADD AUC (threshold={d_threshold}mm)")
                    print("="*60)
                    print(f"  ADD AUC@{d_threshold}mm: {add_auc*100:.2f}%")
                    print("="*60 + "\n")

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

                        # 관절별 3D 오차
                        for j in range(len(mae_3d_per_joint)):
                            wandb_logs[f"val/joint_{j}_3d_mae_mm"] = mae_3d_per_joint[j].item()
                            wandb_logs[f"val/joint_{j}_3d_max_mm"] = max_3d_per_joint[j].item()

                        # 전체 3D 메트릭
                        frame_suffix = "_cam" if is_camera_frame else "_robot"
                        wandb_logs[f"val/3d_mean_error_mm{frame_suffix}"] = mean_3d_error
                        wandb_logs[f"val/3d_median_error_mm{frame_suffix}"] = median_3d_error
                        wandb_logs[f"val/add_auc{frame_suffix}"] = add_auc

                        # Robot-frame 메트릭도 로깅 (비교용)
                        if is_camera_frame and mae_3d_per_joint_robot is not None:
                            wandb_logs["val/3d_mean_error_mm_robot"] = mean_3d_error_robot
                            wandb_logs["val/3d_median_error_mm_robot"] = median_3d_error_robot
                            wandb_logs["val/add_auc_robot"] = add_auc_robot

                        # 좌표계 정보 기록
                        wandb_logs["val/using_camera_frame_3d"] = float(is_camera_frame)

                        wandb.log(wandb_logs)
            log_dict = {"val/loss": avg_val_loss, "epoch": epoch}
            if viz_data is not None and args.use_wandb:
                log_dict["visualizations/combined_pose"] = visualize_3d_with_2d(*viz_data, num_samples=4)
            wandb.log(log_dict)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_model = model.module if hasattr(model, 'module') else model
                torch.save(save_model.state_dict(), output_dir / 'best_3d_pose.pth')
            save_model = model.module if hasattr(model, 'module') else model
            torch.save(save_model.state_dict(), output_dir / 'last_3d_pose.pth')
        
        if global_step >= warmup_steps:
            scheduler.step()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dir', type=str, required=True)
    parser.add_argument('--val-dir', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, help='Path to 2D heatmap weights')
    parser.add_argument('--output-dir', type=str, default='./outputs_3d')
    parser.add_argument('--model-name', type=str, default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    parser.add_argument('--image-size', type=int, default=512)
    parser.add_argument('--heatmap-size', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--min-lr', type=float, default=1e-7)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--warmup-steps', type=int, default=500)
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
