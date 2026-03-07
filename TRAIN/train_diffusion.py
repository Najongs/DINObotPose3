"""Training script for diffusion-based joint angle estimation"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from model_diffusion import DINOv3DiffusionPoseEstimator
from dataset import DREAMDataset

# Dataset stats
PANDA_JOINT_MEAN = torch.tensor([-5.22e-02, 2.68e-01, 6.04e-03, -2.01e+00, 1.49e-02, 1.99e+00, 0.0])
PANDA_JOINT_STD = torch.tensor([1.025, 0.645, 0.511, 0.508, 0.769, 0.511, 1.0])

def normalize_angles(angles):
    return (angles - PANDA_JOINT_MEAN.to(angles.device)) / PANDA_JOINT_STD.to(angles.device)

def denormalize_angles(angles_norm):
    return angles_norm * PANDA_JOINT_STD.to(angles_norm.device) + PANDA_JOINT_MEAN.to(angles_norm.device)

def train_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        images = batch['image'].to(device)
        gt_angles = batch['joint_angles'].to(device)
        gt_angles_norm = normalize_angles(gt_angles[:, :6])  # Only first 6
        
        optimizer.zero_grad()
        
        # Forward
        out = model(images, training=True)
        
        # Compute diffusion loss
        loss = model.joint_angle_head.compute_loss(
            model.backbone(images),
            out['heatmaps_2d'],
            gt_angles_norm,
            device
        )
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)

@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    errors = []
    
    for batch in tqdm(dataloader, desc="Validating"):
        images = batch['image'].to(device)
        gt_angles = batch['joint_angles'].to(device)
        
        # Inference
        out = model(images, training=False)
        pred_angles_norm = out['joint_angles']
        pred_angles = denormalize_angles(pred_angles_norm)
        
        # Compute MAE
        error = torch.abs(pred_angles - gt_angles).cpu().numpy()
        errors.append(error)
    
    errors = np.concatenate(errors, axis=0)
    mae_per_joint = np.rad2deg(errors.mean(axis=0))
    mean_mae = mae_per_joint.mean()
    
    return mean_mae, mae_per_joint

def main():
    # Config
    device = 'cuda'
    batch_size = 16
    num_epochs = 100
    lr = 5e-5
    
    output_dir = 'outputs_diffusion'
    os.makedirs(output_dir, exist_ok=True)
    
    # Data
    train_dataset = DREAMDataset(
        'panda_synth_train_dr',
        image_size=(512, 512),
        heatmap_size=(512, 512)
    )
    val_dataset = DREAMDataset(
        'panda-3cam_azure',
        image_size=(512, 512),
        heatmap_size=(512, 512)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Model
    model = DINOv3DiffusionPoseEstimator(
        dino_model_name='facebook/dinov3-vitb16-pretrain-lvd1689m',
        heatmap_size=(512, 512),
        unfreeze_blocks=0,
        fix_joint7_zero=True
    ).to(device)
    
    # Load heatmap checkpoint
    heatmap_ckpt = torch.load('outputs_heatmap/best_heatmap.pth', map_location='cpu')
    model.backbone.load_state_dict(heatmap_ckpt['backbone'], strict=True)
    model.keypoint_head.load_state_dict(heatmap_ckpt['keypoint_head'], strict=True)
    print("✅ Loaded heatmap checkpoint")
    
    # Freeze heatmap
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.keypoint_head.parameters():
        param.requires_grad = False
    
    # Optimizer (only angle head)
    optimizer = torch.optim.AdamW(model.joint_angle_head.parameters(), lr=lr, weight_decay=0.1)
    
    # Training loop
    best_mae = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        val_mae, mae_per_joint = validate(model, val_loader, device)
        
        print(f"\nEpoch {epoch}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val MAE: {val_mae:.2f}°")
        print(f"  Per-joint: {mae_per_joint}")
        
        # Save best
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_mae': val_mae,
            }, f'{output_dir}/best_diffusion.pth')
            print(f"  ✅ New best: {val_mae:.2f}°")
        
        # Save checkpoint
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_mae': val_mae,
            }, f'{output_dir}/epoch_{epoch:03d}.pth')

if __name__ == '__main__':
    main()
