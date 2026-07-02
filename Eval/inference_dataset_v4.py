import argparse
import os
import json
import math
import random
from pathlib import Path
from datetime import timedelta

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

# Try importing from current or parent directories
try:
    from model_v4 import DINOv3PoseEstimatorV4, panda_forward_kinematics
    from dataset import PoseEstimationDataset
    from checkpoint_compat import load_checkpoint_compat
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent.parent / 'TRAIN'))
    from model_v4 import DINOv3PoseEstimatorV4, panda_forward_kinematics
    from dataset import PoseEstimationDataset
    from checkpoint_compat import load_checkpoint_compat

# ─── Constants (Matched with train_3d_v4.py) ───
PANDA_JOINT_MEAN = torch.tensor([-5.22e-02, 2.68e-01, 6.04e-03, -2.01e+00, 1.49e-02, 1.99e+00, 0.0])
PANDA_JOINT_STD  = torch.tensor([1.025, 0.645, 0.511, 0.508, 0.769, 0.511, 1.0])
LINK_NAMES = ['link0', 'link2', 'link3', 'link4', 'link6', 'link7', 'hand']

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def compute_add_auc(errs_all, auc_threshold=0.1):
    """
    errs_all: (N, num_kp)
    Returns: ADD AUC score and mean per-link error list
    """
    frame_adds = errs_all.mean(axis=1) # (N,) Mean error per frame
    n_total = len(frame_adds)
    if n_total == 0:
        return 0.0, frame_adds
        
    delta = 0.0001
    thresholds = np.arange(0.0, auc_threshold, delta)
    counts = (frame_adds[None, :] <= thresholds[:, None]).sum(axis=1) / float(n_total)
    auc = float(np.trapz(counts, dx=delta) / auc_threshold)
    return auc, frame_adds

def run_inference(args):
    # Distributed setup
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    is_distributed = world_size > 1

    if is_distributed:
        dist.init_process_group(backend='nccl', timeout=timedelta(minutes=30))
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    is_main_process = (rank == 0)

    # 1. Dataset
    dataset = PoseEstimationDataset(
        data_dir=args.dataset_dir,
        keypoint_names=LINK_NAMES,
        image_size=(args.image_size, args.image_size),
        heatmap_size=(args.heatmap_size, args.heatmap_size),
        augment=False,
        include_angles=True
    )

    sampler = DistributedSampler(dataset, shuffle=False) if is_distributed else None
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 2. Model
    if is_main_process:
        print(f"\nBuilding V4 model (backbone: {args.model_name})")
    
    model = DINOv3PoseEstimatorV4(
        dino_model_name=args.model_name,
        heatmap_size=(args.heatmap_size, args.heatmap_size),
        unfreeze_blocks=0,
        fix_joint7_zero=True
    ).to(device)

    if is_main_process:
        print(f"Loading weights from {args.model_path}")
    
    load_checkpoint_compat(
        model=model,
        checkpoint_path=args.model_path,
        device=device,
        is_main_process=is_main_process,
        critical_keys=("joint_angle_head.decpose.weight", "joint_angle_head.decpose.bias")
    )
    model.eval()

    joint_mean = PANDA_JOINT_MEAN.to(device)
    joint_std = PANDA_JOINT_STD.to(device)

    # Containers
    all_angles_pred = []
    all_angles_gt = []
    all_kp_3d_pred = []
    all_kp_3d_gt = []
    all_json_names = []

    if is_main_process:
        print(f"Running inference on {len(dataset)} images...")

    with torch.no_grad():
        for batch in tqdm(dataloader, disable=not is_main_process):
            imgs = batch['image'].to(device)
            gt_angles = batch['angles'].to(device)
            
            output = model(imgs)
            pred_angles_norm = output['joint_angles'] # (B, 7)
            
            # Denormalize
            pred_angles = pred_angles_norm * joint_std + joint_mean
            
            # Compute FK
            pred_angles_fk = pred_angles.clone()
            pred_angles_fk[:, 6] = 0.0
            
            gt_angles_fk = gt_angles.clone()
            gt_angles_fk[:, 6] = 0.0
            
            kp_3d_pred = panda_forward_kinematics(pred_angles_fk) # (B, 7, 3)
            kp_3d_gt = panda_forward_kinematics(gt_angles_fk)     # (B, 7, 3)

            all_angles_pred.append(pred_angles[:, :6].cpu().numpy())
            all_angles_gt.append(gt_angles[:, :6].cpu().numpy())
            all_kp_3d_pred.append(kp_3d_pred.cpu().numpy())
            all_kp_3d_gt.append(kp_3d_gt.cpu().numpy())
            
            if 'name' in batch:
                all_json_names.extend(batch['name'])

    # Aggregate results across ranks
    all_angles_pred = np.concatenate(all_angles_pred, axis=0)
    all_angles_gt = np.concatenate(all_angles_gt, axis=0)
    all_kp_3d_pred = np.concatenate(all_kp_3d_pred, axis=0)
    all_kp_3d_gt = np.concatenate(all_kp_3d_gt, axis=0)

    if is_distributed:
        # 1. Gather counts to handle padding if DistributedSampler padded the dataset
        local_count = len(all_angles_pred)
        counts = [0] * world_size
        dist.all_gather_object(counts, local_count)
        
        # 2. Gather tensors
        def gather_tensor(local_tensor):
            gathered = [torch.zeros_like(torch.from_numpy(local_tensor)).to(device) for _ in range(world_size)]
            dist.all_gather(gathered, torch.from_numpy(local_tensor).to(device))
            # Concatenate and move back to CPU
            return torch.cat(gathered, dim=0).cpu().numpy()

        all_angles_pred = gather_tensor(all_angles_pred)
        all_angles_gt = gather_tensor(all_angles_gt)
        all_kp_3d_pred = gather_tensor(all_kp_3d_pred)
        all_kp_3d_gt = gather_tensor(all_kp_3d_gt)
        
        # 3. Gather names (strings need all_gather_object)
        all_names_list = [None] * world_size
        dist.all_gather_object(all_names_list, all_json_names)
        all_json_names = [name for sublist in all_names_list for name in sublist]

        # 4. Truncate to actual dataset size (handle padding)
        total_size = len(dataset)
        indices = list(range(total_size))
        # Reconstruct the interleaved indices used by DistributedSampler
        all_indices = []
        num_samples = math.ceil(total_size / world_size)
        for i in range(world_size):
            rank_indices = indices[i:total_size:world_size]
            if len(rank_indices) < num_samples:
                rank_indices += rank_indices[:(num_samples - len(rank_indices))]
            all_indices.extend(rank_indices)
        
        sort_idx = np.argsort(all_indices)
        
        all_angles_pred = all_angles_pred[sort_idx][:total_size]
        all_angles_gt = all_angles_gt[sort_idx][:total_size]
        all_kp_3d_pred = all_kp_3d_pred[sort_idx][:total_size]
        all_kp_3d_gt = all_kp_3d_gt[sort_idx][:total_size]
        all_json_names = [all_json_names[i] for i in sort_idx][:total_size]

    if is_main_process:
        # Calculate Metrics
        
        # 1. Joint Angle Error (Degrees)
        angle_err_rad = np.abs(all_angles_pred - all_angles_gt)
        angle_err_deg = angle_err_rad * (180.0 / np.pi)
        mean_angle_err_deg = angle_err_deg.mean(axis=0)
        overall_angle_err_deg = angle_err_deg.mean()
        
        # 2. 3D FK Error (Meters)
        dist_err_m = np.linalg.norm(all_kp_3d_pred - all_kp_3d_gt, axis=2) # (N, 7)
        mean_3d_err_m = dist_err_m.mean(axis=0)
        overall_3d_err_m = dist_err_m.mean()
        
        # 3. ADD AUC
        add_auc, frame_adds = compute_add_auc(dist_err_m, auc_threshold=0.1)

        print("\n" + "="*50)
        print(" V4 Inference Results (Joint Angle Model)")
        print("="*50)
        print(f"Overall Mean Joint Angle Error: {overall_angle_err_deg:.3f}°")
        for i in range(6):
            print(f"  Joint {i}: {mean_angle_err_deg[i]:.3f}°")
        print("-" * 30)
        print(f"Overall Mean 3D FK Error: {overall_3d_err_m * 1000:.2f} mm")
        for i, name in enumerate(LINK_NAMES):
            print(f"  {name:<10}: {mean_3d_err_m[i] * 1000:6.2f} mm")
        print("-" * 30)
        print(f"ADD AUC (threshold=10cm): {add_auc:.4f}")
        print("="*50 + "\n")

        # Outlier Report
        if args.save_outliers:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            frame_indices = np.argsort(-frame_adds) # Worst first
            
            outlier_report = []
            for idx in frame_indices[:args.outlier_topk]:
                report = {
                    'frame_index': int(idx),
                    'mean_3d_err_mm': float(frame_adds[idx] * 1000),
                    'joint_angle_errs_deg': angle_err_deg[idx].tolist(),
                    'per_link_3d_err_mm': (dist_err_m[idx] * 1000).tolist()
                }
                if all_json_names:
                    report['json_name'] = all_json_names[idx]
                outlier_report.append(report)
                
            with open(output_dir / 'outlier_v4_3d.json', 'w') as f:
                json.dump(outlier_report, f, indent=2)
            
            with open(output_dir / 'outlier_v4_3d_names.txt', 'w') as f:
                for r in outlier_report:
                    if 'json_name' in r:
                        f.write(f"{r['json_name']}\n")
            
            print(f"Outlier report saved to {output_dir}")

    cleanup_distributed()

def main():
    parser = argparse.ArgumentParser(description="V4 Joint Angle Model Inference")
    parser.add_argument('--model-path', type=str, required=True, help='Path to V4 .pth checkpoint')
    parser.add_argument('--dataset-dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--output-dir', type=str, default='./eval_v4_output')
    parser.add_argument('--model-name', type=str, default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    parser.add_argument('--image-size', type=int, default=512)
    parser.add_argument('--heatmap-size', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--save-outliers', action='store_true')
    parser.add_argument('--outlier-topk', type=int, default=20)
    args = parser.parse_args()
    run_inference(args)

if __name__ == '__main__':
    main()
