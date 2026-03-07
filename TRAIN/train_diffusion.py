"""Training script for diffusion-based joint angle estimation"""
import os
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
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

def train_epoch(model, dataloader, optimizer, device, epoch, rank):
    model.train()
    total_loss = 0
    
    if rank == 0:
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    else:
        pbar = dataloader
    
    for batch in pbar:
        images = batch['image'].to(device)
        gt_angles = batch['joint_angles'].to(device)
        gt_angles_norm = normalize_angles(gt_angles[:, :6])
        
        optimizer.zero_grad()
        
        # Forward
        out = model(images, training=True)
        
        # Compute diffusion loss
        if isinstance(model, DDP):
            loss = model.module.joint_angle_head.compute_loss(
                model.module.backbone(images),
                out['heatmaps_2d'],
                gt_angles_norm,
                device
            )
        else:
            loss = model.joint_angle_head.compute_loss(
                model.backbone(images),
                out['heatmaps_2d'],
                gt_angles_norm,
                device
            )
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        if rank == 0:
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)

@torch.no_grad()
def validate(model, dataloader, device, rank):
    model.eval()
    errors = []
    
    if rank == 0:
        pbar = tqdm(dataloader, desc="Validating")
    else:
        pbar = dataloader
    
    for batch in pbar:
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dir', type=str, required=True)
    parser.add_argument('--val-dir', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--model-name', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--image-size', type=int, default=512)
    parser.add_argument('--heatmap-size', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight-decay', type=float, default=0.1)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--use-wandb', action='store_true')
    parser.add_argument('--wandb-project', type=str, default='dinov3-diffusion')
    parser.add_argument('--wandb-run-name', type=str, default='diffusion')
    args = parser.parse_args()
    
    # DDP setup
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
    
    device = torch.device(f'cuda:{local_rank}')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Data
    train_dataset = DREAMDataset(
        args.train_dir,
        image_size=(args.image_size, args.image_size),
        heatmap_size=(args.heatmap_size, args.heatmap_size)
    )
    val_dataset = DREAMDataset(
        args.val_dir,
        image_size=(args.image_size, args.image_size),
        heatmap_size=(args.heatmap_size, args.heatmap_size)
    )
    
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=args.num_workers)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    if rank == 0:
        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Model
    model = DINOv3DiffusionPoseEstimator(
        dino_model_name=args.model_name,
        heatmap_size=(args.heatmap_size, args.heatmap_size),
        unfreeze_blocks=0,
        fix_joint7_zero=True
    ).to(device)
    
    # Load heatmap checkpoint
    heatmap_ckpt = torch.load(args.checkpoint, map_location='cpu')
    model.backbone.load_state_dict(heatmap_ckpt['backbone'], strict=True)
    model.keypoint_head.load_state_dict(heatmap_ckpt['keypoint_head'], strict=True)
    if rank == 0:
        print("✅ Loaded heatmap checkpoint")
    
    # Freeze heatmap
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.keypoint_head.parameters():
        param.requires_grad = False
    
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    
    # Optimizer
    if isinstance(model, DDP):
        optimizer = torch.optim.AdamW(model.module.joint_angle_head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.joint_angle_head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Wandb
    if args.use_wandb and rank == 0:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
    
    # Training loop
    best_mae = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        if world_size > 1:
            train_sampler.set_epoch(epoch)
        
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch, rank)
        val_mae, mae_per_joint = validate(model, val_loader, device, rank)
        
        if rank == 0:
            print(f"\nEpoch {epoch}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val MAE: {val_mae:.2f}°")
            print(f"  Per-joint: {mae_per_joint}")
            
            if args.use_wandb:
                wandb.log({
                    'train_loss': train_loss,
                    'val_mae': val_mae,
                    'epoch': epoch
                })
            
            # Save best
            if val_mae < best_mae:
                best_mae = val_mae
                save_model = model.module if isinstance(model, DDP) else model
                torch.save({
                    'epoch': epoch,
                    'model': save_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'val_mae': val_mae,
                }, f'{args.output_dir}/best_diffusion.pth')
                print(f"  ✅ New best: {val_mae:.2f}°")
            
            # Save checkpoint
            if epoch % 10 == 0:
                save_model = model.module if isinstance(model, DDP) else model
                torch.save({
                    'epoch': epoch,
                    'model': save_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'val_mae': val_mae,
                }, f'{args.output_dir}/epoch_{epoch:03d}.pth')
    
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()
