import argparse
import os
import sys
import math
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add TRAIN directory to Python path to import model and dataset classes
TRAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../TRAIN'))
sys.path.append(TRAIN_DIR)

from model_v3 import DINOv3PoseEstimator, panda_forward_kinematics
from dataset import PoseEstimationDataset

# Hardcoded joint statistics used during training
PANDA_JOINT_MEAN = torch.tensor([-5.22e-02, 2.68e-01, 6.04e-03, -2.01e+00, 1.49e-02, 1.99e+00, 0.0])
PANDA_JOINT_STD  = torch.tensor([1.025, 0.645, 0.511, 0.508, 0.769, 0.511, 1.0])

def optimize_ik_batch(pred_kp_3d, joint_mean, num_iters=150, lr=5e-2):
    """
    Given predicted 3D keypoints (B, 7, 3), solve IK using PyTorch autodiff.
    Returns optimized joint angles (B, 6).
    """
    B = pred_kp_3d.shape[0]
    device = pred_kp_3d.device
    
    # Initialize from joint mean
    angles = joint_mean[:6].unsqueeze(0).expand(B, 6).clone().to(device)
    angles.requires_grad = True
    
    optimizer = torch.optim.Adam([angles], lr=lr)
    
    for _ in range(num_iters):
        optimizer.zero_grad()
        
        angles_full = torch.zeros(B, 7, device=device)
        angles_full[:, :6] = angles
        
        current_kp_3d = panda_forward_kinematics(angles_full)
        loss = F.mse_loss(current_kp_3d, pred_kp_3d)
        loss.backward()
        optimizer.step()
        
    return angles.detach()


def run_evaluation(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    keypoint_names = ['link0', 'link2', 'link3', 'link4', 'link6', 'link7', 'hand']
    test_dataset = PoseEstimationDataset(
        data_dir=args.test_dir,
        keypoint_names=keypoint_names,
        image_size=(args.image_size, args.image_size),
        heatmap_size=(args.heatmap_size, args.heatmap_size),
        augment=False,
        include_angles=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Initialize model
    model = DINOv3PoseEstimator(
        dino_model_name=args.model_name,
        heatmap_size=(args.heatmap_size, args.heatmap_size),
        unfreeze_blocks=0,
        fix_joint7_zero=True,
    ).to(device)

    # Load weights
    print(f"Loading checkpoint from {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    # Remove 'module.' prefix if it was trained with DDP
    ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=True)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for evaluation")
        model = torch.nn.DataParallel(model)

    model.eval()

    joint_mean = PANDA_JOINT_MEAN.to(device)

    # Metrics accumulators
    val_loss_accum = 0.0
    val_joint_mae = np.zeros(6)
    val_3d_errors = []
    val_count = 0

    print("Starting evaluation...")
    for i, batch in enumerate(tqdm(test_loader, desc="Eval")):
        imgs = batch['image'].to(device)
        gt_angles = batch['angles'].to(device)
        
        gt_angles_6 = gt_angles[:, :6]
        gt_angles_full = gt_angles.clone()
        gt_angles_full[:, 6] = 0.0

        with torch.no_grad():
            preds = model(imgs)
            pred_kp_3d = preds['keypoints_3d']
            
            gt_kp_3d = panda_forward_kinematics(gt_angles_full)
            val_loss_accum += F.mse_loss(pred_kp_3d, gt_kp_3d).item()

            per_link_err = (gt_kp_3d - pred_kp_3d).norm(dim=-1)
            val_3d_errors.append(per_link_err.cpu().numpy())

        # IK optimization (requires gradients for angles, so detach pred_kp_3d)
        pred_angles_ik = optimize_ik_batch(pred_kp_3d.detach(), joint_mean, num_iters=150, lr=5e-2)

        with torch.no_grad():
            angle_diff = pred_angles_ik - gt_angles_6
            batch_mae = angle_diff.abs().mean(dim=0).cpu().numpy() * (180 / math.pi)
            val_joint_mae = (val_joint_mae * val_count + batch_mae) / (val_count + 1)
            val_count += 1

    avg_val_loss = val_loss_accum / len(test_loader)
    
    val_3d = np.concatenate(val_3d_errors, axis=0).mean(axis=0)  # (7,)
    mean_3d = val_3d.mean()
    
    # Print results
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  MSE Loss: {avg_val_loss:.4f}")
    print(f"  Dataset: {args.test_dir}")
    print(f"  Model: {args.checkpoint}")
    print(f"{'='*60}")
    print(f"  {'Joint':<8} {'IK Test MAE':>12}")
    print(f"  {'-'*8} {'-'*12}")
    for j in range(6):
        marker = " ⚠️" if val_joint_mae[j] > 20 else ""
        print(f"  J{j:<7} {val_joint_mae[j]:>10.2f}°{marker}")
    print(f"  {'MEAN':<8} {val_joint_mae.mean():>10.2f}°")
    worst = np.argmax(val_joint_mae)
    print(f"  → Worst: J{worst} ({val_joint_mae[worst]:.2f}°)")
    print(f"{'='*60}")

    print(f"\n  3D Keypoint Error (Test):")
    for li, ln in enumerate(keypoint_names):
        print(f"    {ln:<8} {val_3d[li]*1000:.1f}mm")
    print(f"    {'MEAN':<8} {mean_3d*1000:.1f}mm")
    print(f"{'='*60}\n")

    # Save results to output directory if specified
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            "dataset": str(args.test_dir),
            "model": str(args.checkpoint),
            "mse_loss": float(avg_val_loss),
            "joint_mae_deg": {f"J{j}": float(val_joint_mae[j]) for j in range(6)},
            "mean_joint_mae_deg": float(val_joint_mae.mean()),
            "keypoint_3d_error_mm": {ln: float(val_3d[li]*1000) for li, ln in enumerate(keypoint_names)},
            "mean_keypoint_3d_error_mm": float(mean_3d*1000),
        }
        
        results_file = out_dir / "eval_3d_v3_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {results_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-dir', type=str, required=True, help='Path to test dataset')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to 3D model weights (.pth)')
    parser.add_argument('--model-name', type=str, default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    parser.add_argument('--image-size', type=int, default=512)
    parser.add_argument('--heatmap-size', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--output-dir', type=str, default='./results_v3')
    
    args = parser.parse_args()
    run_evaluation(args)
