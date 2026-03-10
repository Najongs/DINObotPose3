"""
DINOv3 End-to-End Pose Training Script
Unfreezes all components (Backbone, 2D Head, 3D Head) for joint fine-tuning.
Losses: 2D Heatmap + Joint Angle + Camera-frame 3D Pose.
"""

import argparse
import os
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from tqdm import tqdm
import wandb
import cv2

from model import DINOv3PoseEstimator, panda_forward_kinematics
from dataset import PoseEstimationDataset

def get_keypoints_from_heatmaps(heatmaps):
    """Extract [x, y] coordinates from heatmaps using argmax."""
    B, N, H, W = heatmaps.shape
    heatmaps_flat = heatmaps.view(B, N, -1)
    max_indices = torch.argmax(heatmaps_flat, dim=-1)
    y = max_indices // W
    x = max_indices % W
    return torch.stack([x, y], dim=-1).float()

def solve_pnp_epnp(object_points, image_points, camera_matrix):
    """Solve PnP using EPnP algorithm."""
    try:
        if len(object_points) < 4:
            return False, None, None
        success, rvec, tvec = cv2.solvePnP(
            object_points.astype(np.float32),
            image_points.astype(np.float32),
            camera_matrix.astype(np.float32),
            None,
            flags=cv2.SOLVEPNP_EPNP
        )
        if not success:
            return False, None, None
        R, _ = cv2.Rodrigues(rvec)
        return True, R, tvec.flatten()
    except Exception:
        return False, None, None

class EndToEndPoseLoss(nn.Module):
    def __init__(self, heatmap_weight=1000.0, angle_weight=10.0, camera_3d_weight=100.0):
        super().__init__()
        self.heatmap_weight = heatmap_weight
        self.angle_weight = angle_weight
        self.camera_3d_weight = camera_3d_weight
        
        self.heatmap_loss_fn = nn.MSELoss()
        self.pose_loss_fn = nn.SmoothL1Loss(beta=0.01)
        self.pnp_failure_penalty = 0.1

    def forward(self, pred_dict, gt_dict):
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=pred_dict['joint_angles'].device)

        # 1. 2D Heatmap Loss
        if 'heatmaps_2d' in pred_dict and 'heatmaps' in gt_dict:
            h_loss = self.heatmap_loss_fn(pred_dict['heatmaps_2d'], gt_dict['heatmaps'])
            weighted_h_loss = self.heatmap_weight * h_loss
            total_loss = total_loss + weighted_h_loss
            loss_dict['loss/heatmap'] = h_loss.item()

        # 2. Joint Angle Loss (Radian)
        if 'joint_angles' in pred_dict and 'angles' in gt_dict:
            pred_angles = pred_dict['joint_angles']
            gt_angles = gt_dict['angles']
            n_angle = min(pred_angles.shape[1], gt_angles.shape[1])
            angle_loss = self.pose_loss_fn(pred_angles[:, :n_angle], gt_angles[:, :n_angle])
            total_loss = total_loss + self.angle_weight * angle_loss
            loss_dict['loss/angle'] = angle_loss.item()

        # 3. Camera-frame 3D Loss (Meters)
        if self.camera_3d_weight > 0:
            pred_kp_robot = pred_dict['keypoints_3d_fk']
            gt_kp_camera = gt_dict['keypoints_3d']
            camera_K = gt_dict['camera_K']
            original_size = gt_dict['original_size']
            pred_heatmaps = pred_dict['heatmaps_2d'].detach()
            
            pred_kp_2d_hm = get_keypoints_from_heatmaps(pred_heatmaps)
            H_hm, W_hm = pred_heatmaps.shape[2], pred_heatmaps.shape[3]
            scale_x = original_size[:, 0:1] / W_hm
            scale_y = original_size[:, 1:2] / H_hm
            pred_kp_2d_orig = torch.stack([pred_kp_2d_hm[:, :, 0] * scale_x, pred_kp_2d_hm[:, :, 1] * scale_y], dim=-1)
            
            pred_kp_conf = pred_heatmaps.flatten(2).amax(dim=-1)
            valid_mask = gt_dict.get('valid_mask', None)

            B = pred_kp_robot.shape[0]
            pred_kp_camera_list = []
            gt_kp_camera_list = []
            pnp_success_count = 0

            for b in range(B):
                conf_b = pred_kp_conf[b]
                v_b = valid_mask[b] if valid_mask is not None else torch.ones_like(conf_b, dtype=torch.bool)
                
                thresh = 0.25
                valid_idx = torch.where((conf_b > thresh) & v_b)[0]
                while len(valid_idx) < 4 and thresh > -0.5:
                    thresh -= 0.1
                    valid_idx = torch.where((conf_b > thresh) & v_b)[0]
                
                if len(valid_idx) >= 4:
                    success, R, t = solve_pnp_epnp(
                        pred_kp_robot[b][valid_idx].detach().cpu().numpy(),
                        pred_kp_2d_orig[b][valid_idx].detach().cpu().numpy(),
                        camera_K[b].detach().cpu().numpy()
                    )
                    if success:
                        R_t = torch.from_numpy(R).float().to(pred_kp_robot.device)
                        t_t = torch.from_numpy(t).float().to(pred_kp_robot.device)
                        pred_cam_b = (R_t @ pred_kp_robot[b].T).T + t_t
                        pred_kp_camera_list.append(pred_cam_b)
                        gt_kp_camera_list.append(gt_kp_camera[b])
                        pnp_success_count += 1

            if len(pred_kp_camera_list) > 0:
                pred_kp_camera = torch.stack(pred_kp_camera_list)
                gt_kp_camera_v = torch.stack(gt_kp_camera_list)
                cam_3d_loss = self.pose_loss_fn(pred_kp_camera, gt_kp_camera_v)
                total_loss = total_loss + self.camera_3d_weight * cam_3d_loss
                loss_dict['loss/camera_3d'] = cam_3d_loss.item()
            
            pnp_rate = pnp_success_count / B
            loss_dict['metrics/pnp_success_rate'] = pnp_rate
            total_loss = total_loss + (1.0 - pnp_rate) * self.pnp_failure_penalty

        loss_dict['loss/total'] = total_loss.item()
        return total_loss, loss_dict

def main(args):
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

    # 2. Model Initialization (Unfreeze all)
    model = DINOv3PoseEstimator(
        dino_model_name=args.model_name,
        heatmap_size=(args.heatmap_size, args.heatmap_size),
        unfreeze_blocks=args.unfreeze_blocks, # Set to high number or handle in loop
        fix_joint7_zero=args.fix_joint7
    ).to(device)

    # 3. Load 1st Stage Checkpoint (3D Head stable weights)
    if args.checkpoint and os.path.isfile(args.checkpoint):
        if is_main: print(f"==> Loading stage 1 weights from: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        model.load_state_dict(checkpoint, strict=True)
    else:
        if is_main: print("==> Warning: Starting E2E without pretrained weights!")

    # 4. Explicitly unfreeze ALL parameters
    for param in model.parameters():
        param.requires_grad = True

    if local_rank != -1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    # 5. Optimizer & Loss (Lower LR for fine-tuning)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = EndToEndPoseLoss(
        heatmap_weight=args.heatmap_weight,
        angle_weight=args.angle_weight,
        camera_3d_weight=args.camera_3d_weight
    )
    
    if is_main and args.use_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=args)

    best_val_loss = float('inf')

    # 6. Training Loop
    for epoch in range(args.epochs):
        if train_sampler: train_sampler.set_epoch(epoch)
        model.train()
        train_loss_accum = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]") if is_main else train_loader
        
        for batch in pbar:
            imgs = batch['image'].to(device)
            gt_dict = {
                'heatmaps': batch['heatmaps'].to(device),
                'angles': batch['angles'].to(device),
                'keypoints_3d': batch['keypoints_3d'].to(device),
                'camera_K': batch['camera_K'].to(device),
                'original_size': batch['original_size'].to(device),
                'valid_mask': batch['valid_mask'].to(device)
            }
            
            optimizer.zero_grad()
            preds = model(imgs)
            loss, loss_dict = criterion(preds, gt_dict)
            loss.backward()
            optimizer.step()
            
            train_loss_accum += loss.item()
            if is_main:
                postfix = {
                    'total': f"{loss.item():.4f}",
                    'hm': f"{loss_dict.get('loss/heatmap', 0):.6f}",
                    'angle': f"{loss_dict.get('loss/angle', 0):.4f}",
                    'cam3d': f"{loss_dict.get('loss/camera_3d', 0):.4f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
                }
                pbar.set_postfix(postfix)
                if args.use_wandb:
                    wandb.log({f"train/{k}": v for k, v in loss_dict.items()})
                    wandb.log({"train/lr": optimizer.param_groups[0]['lr']})

        # Validation
        model.eval()
        val_loss_accum = 0.0
        pnp_rates = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} [Val]") if is_main else val_loader:
                imgs = batch['image'].to(device)
                gt_dict = {
                    'heatmaps': batch['heatmaps'].to(device),
                    'angles': batch['angles'].to(device),
                    'keypoints_3d': batch['keypoints_3d'].to(device),
                    'camera_K': batch['camera_K'].to(device),
                    'original_size': batch['original_size'].to(device),
                    'valid_mask': batch['valid_mask'].to(device)
                }
                preds = model(imgs)
                loss, loss_dict = criterion(preds, gt_dict)
                val_loss_accum += loss.item()
                if 'metrics/pnp_success_rate' in loss_dict:
                    pnp_rates.append(loss_dict['metrics/pnp_success_rate'])

        if local_rank != -1:
            val_loss_tensor = torch.tensor([val_loss_accum], device=device)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            avg_val_loss = val_loss_tensor.item() / (len(val_loader) * world_size)
        else:
            avg_val_loss = val_loss_accum / len(val_loader)

        avg_pnp_rate = np.mean(pnp_rates) if pnp_rates else 0.0
        
        if is_main:
            print(f"Epoch {epoch} | Val Loss: {avg_val_loss:.4f} | PnP Rate: {avg_pnp_rate:.2f}")
            if args.use_wandb:
                wandb.log({"val/loss": avg_val_loss, "val/pnp_rate": avg_pnp_rate, "epoch": epoch})
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_model = model.module if hasattr(model, 'module') else model
                torch.save(save_model.state_dict(), output_dir / 'best_e2e_pose.pth')
            save_model = model.module if hasattr(model, 'module') else model
            torch.save(save_model.state_dict(), output_dir / 'last_e2e_pose.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dir', type=str, required=True)
    parser.add_argument('--val-dir', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, help='Path to 1st stage 3D weights')
    parser.add_argument('--output-dir', type=str, default='./outputs_e2e')
    parser.add_argument('--model-name', type=str, default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    parser.add_argument('--image-size', type=int, default=512)
    parser.add_argument('--heatmap-size', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-5) # Very low LR for fine-tuning
    parser.add_argument('--heatmap-weight', type=float, default=1000.0)
    parser.add_argument('--angle-weight', type=float, default=10.0)
    parser.add_argument('--camera-3d-weight', type=float, default=100.0)
    parser.add_argument('--unfreeze-blocks', type=int, default=12) # Unfreeze all ViT blocks
    parser.add_argument('--fix-joint7', action='store_true')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--no-augment', action='store_true')
    parser.add_argument('--use-wandb', action='store_true')
    parser.add_argument('--wandb-project', type=str, default='dinov3-e2e-pose')
    parser.add_argument('--wandb-run-name', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    main(parser.parse_args())
