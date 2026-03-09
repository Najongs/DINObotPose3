"""Training script for diffusion-based joint angle estimation"""
import os
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import numpy as np

from model_diffusion import DINOv3DiffusionPoseEstimator
from dataset import PoseEstimationDataset
from model import panda_forward_kinematics

# Dataset stats
PANDA_JOINT_MEAN = torch.tensor([-5.22e-02, 2.68e-01, 6.04e-03, -2.01e+00, 1.49e-02, 1.99e+00, 0.0])
PANDA_JOINT_STD = torch.tensor([1.025, 0.645, 0.511, 0.508, 0.769, 0.511, 1.0])
TRAIN_ANGLE_DIM = 6

def normalize_angles(angles):
    mean = PANDA_JOINT_MEAN[:angles.shape[-1]].to(angles.device)
    std = PANDA_JOINT_STD[:angles.shape[-1]].to(angles.device)
    return (angles - mean) / std

def denormalize_angles(angles_norm):
    mean = PANDA_JOINT_MEAN[:angles_norm.shape[-1]].to(angles_norm.device)
    std = PANDA_JOINT_STD[:angles_norm.shape[-1]].to(angles_norm.device)
    return angles_norm * std + mean

def reduce_mean_scalar(value, device):
    if not dist.is_available() or not dist.is_initialized():
        return value
    tensor = torch.tensor(value, device=device, dtype=torch.float32)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return tensor.item()

def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model

def maybe_unfreeze_backbone(raw_model, unfreeze_blocks):
    if unfreeze_blocks <= 0:
        return 0
    if hasattr(raw_model.backbone.model, 'encoder') and hasattr(raw_model.backbone.model.encoder, 'layers'):
        layers = raw_model.backbone.model.encoder.layers
    elif hasattr(raw_model.backbone.model, 'blocks'):
        layers = raw_model.backbone.model.blocks
    else:
        layers = []
    if len(layers) == 0:
        return 0
    start_idx = max(0, len(layers) - unfreeze_blocks)
    for i in range(start_idx, len(layers)):
        for param in layers[i].parameters():
            param.requires_grad = True
    return len(layers) - start_idx

def build_optimizer(raw_model, args, backbone_active=False):
    if backbone_active:
        param_groups = [
            {'params': [p for p in raw_model.joint_angle_head.parameters() if p.requires_grad], 'lr': args.lr},
            {'params': [p for p in raw_model.backbone.parameters() if p.requires_grad], 'lr': args.lr * args.backbone_lr_scale},
        ]
    else:
        param_groups = [
            {'params': [p for p in raw_model.joint_angle_head.parameters() if p.requires_grad], 'lr': args.lr},
        ]
    param_groups = [pg for pg in param_groups if len(pg['params']) > 0]
    return torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

def train_epoch(model, dataloader, optimizer, device, epoch, rank, args, global_step):
    model.train()
    loss_sums = {
        'loss': 0.0,
        'noise_loss': 0.0,
        'init_loss': 0.0,
        'recon_loss': 0.0,
        'fk_loss': 0.0,
    }
    
    if rank == 0:
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    else:
        pbar = dataloader
    
    for batch in pbar:
        images = batch['image'].to(device)
        gt_angles = batch['angles'].to(device)
        gt_angles_norm = normalize_angles(gt_angles[:, :TRAIN_ANGLE_DIM])

        if args.warmup_steps > 0 and global_step < args.warmup_steps:
            lr_scale = float(global_step + 1) / float(args.warmup_steps)
            base_lrs = [args.lr]
            if len(optimizer.param_groups) > 1:
                base_lrs.append(args.lr * args.backbone_lr_scale)
            for idx, pg in enumerate(optimizer.param_groups):
                base_lr = base_lrs[idx] if idx < len(base_lrs) else args.lr
                pg['lr'] = base_lr * lr_scale
        
        optimizer.zero_grad()
        
        # Forward
        out = model(images, training=True)
        
        # Compute diffusion loss
        if isinstance(model, DDP):
            loss_dict = model.module.joint_angle_head.compute_loss_from_condition(
                out['condition'],
                gt_angles_norm,
                device,
                init_loss_weight=args.init_loss_weight,
                recon_loss_weight=args.recon_loss_weight,
            )
        else:
            loss_dict = model.joint_angle_head.compute_loss_from_condition(
                out['condition'],
                gt_angles_norm,
                device,
                init_loss_weight=args.init_loss_weight,
                recon_loss_weight=args.recon_loss_weight,
            )
        loss = loss_dict['loss']

        fk_loss = torch.tensor(0.0, device=device)
        if args.fk_loss_weight > 0:
            gt_angles_full = gt_angles.clone()
            gt_angles_full[:, 6] = 0.0

            init_angles = denormalize_angles(loss_dict['init_pred'])
            recon_angles = denormalize_angles(loss_dict['pred_x0'])

            init_angles_full = torch.zeros_like(gt_angles)
            init_angles_full[:, :TRAIN_ANGLE_DIM] = init_angles
            recon_angles_full = torch.zeros_like(gt_angles)
            recon_angles_full[:, :TRAIN_ANGLE_DIM] = recon_angles

            gt_kp = panda_forward_kinematics(gt_angles_full)
            init_kp = panda_forward_kinematics(init_angles_full)
            recon_kp = panda_forward_kinematics(recon_angles_full)

            fk_init_loss = F.smooth_l1_loss(init_kp, gt_kp, beta=0.02)
            fk_recon_loss = F.smooth_l1_loss(recon_kp, gt_kp, beta=0.02)
            fk_loss = 0.5 * (fk_init_loss + fk_recon_loss)
            loss = loss + args.fk_loss_weight * fk_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        optimizer.step()
        global_step += 1
        
        for key in loss_sums:
            if key == 'fk_loss':
                tensor_val = fk_loss
            else:
                tensor_val = loss_dict[key]
            loss_sums[key] += float(tensor_val.detach().item() if torch.is_tensor(tensor_val) else tensor_val)
        if rank == 0:
            pbar.set_postfix({
                'loss': f'{loss.detach().item():.4f}',
                'noise': f'{loss_dict["noise_loss"].item():.4f}',
                'init': f'{loss_dict["init_loss"].item():.4f}',
                'recon': f'{loss_dict["recon_loss"].item():.4f}',
                'fk': f'{fk_loss.item():.4f}',
            })
    
    metrics = {
        key: reduce_mean_scalar(val / len(dataloader), device)
        for key, val in loss_sums.items()
    }
    return metrics, global_step

@torch.no_grad()
def validate(model, dataloader, device, rank):
    model.eval()
    abs_error_sum = None
    sample_count = 0
    
    if rank == 0:
        pbar = tqdm(dataloader, desc="Validating")
    else:
        pbar = dataloader
    
    for batch in pbar:
        images = batch['image'].to(device)
        gt_angles = batch['angles'].to(device)
        
        # Inference
        out = model(images, training=False)
        pred_angles_norm = out['joint_angles']
        pred_angles = denormalize_angles(pred_angles_norm[:, :TRAIN_ANGLE_DIM])
        gt_angles = gt_angles[:, :TRAIN_ANGLE_DIM]
        
        # Joint 7 is fixed to zero by design, so evaluate only the trained 6 joints.
        batch_error_sum = torch.abs(pred_angles - gt_angles).sum(dim=0)
        if abs_error_sum is None:
            abs_error_sum = batch_error_sum
        else:
            abs_error_sum += batch_error_sum
        sample_count += pred_angles.shape[0]

    if abs_error_sum is None:
        mae_per_joint = np.zeros(TRAIN_ANGLE_DIM, dtype=np.float32)
        return 0.0, mae_per_joint

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(abs_error_sum, op=dist.ReduceOp.SUM)
        count_tensor = torch.tensor(sample_count, device=device, dtype=torch.float32)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        sample_count = int(count_tensor.item())

    mae_per_joint = np.rad2deg((abs_error_sum / max(sample_count, 1)).cpu().numpy())
    mean_mae = float(mae_per_joint.mean())
    
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
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--unfreeze-blocks', type=int, default=0)
    parser.add_argument('--warmup-frozen-epochs', type=int, default=0)
    parser.add_argument('--backbone-lr-scale', type=float, default=0.05)
    parser.add_argument('--warmup-steps', type=int, default=500)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--init-loss-weight', type=float, default=0.5)
    parser.add_argument('--recon-loss-weight', type=float, default=0.5)
    parser.add_argument('--fk-loss-weight', type=float, default=0.1)
    parser.add_argument('--diffusion-steps', type=int, default=20)
    parser.add_argument('--angle-dropout', type=float, default=0.1)
    parser.add_argument('--no-augment', action='store_true')
    parser.add_argument('--fda-real-dir', type=str, default=None)
    parser.add_argument('--fda-prob', type=float, default=0.0)
    parser.add_argument('--fda-beta', type=float, default=0.01)
    parser.add_argument('--occlusion-prob', type=float, default=0.25)
    parser.add_argument('--occlusion-max-holes', type=int, default=4)
    parser.add_argument('--occlusion-max-size-frac', type=float, default=0.15)
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
    rank_seed = args.seed + rank
    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rank_seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Keypoint names
    keypoint_names = ['link0', 'link2', 'link3', 'link4', 'link6', 'link7', 'hand']
    
    # Data
    train_dataset = PoseEstimationDataset(
        args.train_dir,
        keypoint_names=keypoint_names,
        image_size=(args.image_size, args.image_size),
        heatmap_size=(args.heatmap_size, args.heatmap_size),
        augment=not args.no_augment,
        fda_real_dir=args.fda_real_dir,
        fda_prob=args.fda_prob,
        fda_beta=args.fda_beta,
        occlusion_prob=args.occlusion_prob,
        occlusion_max_holes=args.occlusion_max_holes,
        occlusion_max_size_frac=args.occlusion_max_size_frac,
    )
    val_dataset = PoseEstimationDataset(
        args.val_dir,
        keypoint_names=keypoint_names,
        image_size=(args.image_size, args.image_size),
        heatmap_size=(args.heatmap_size, args.heatmap_size),
        augment=False,
    )
    
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=args.num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    if rank == 0:
        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Model
    model = DINOv3DiffusionPoseEstimator(
        dino_model_name=args.model_name,
        heatmap_size=(args.heatmap_size, args.heatmap_size),
        unfreeze_blocks=0,
        fix_joint7_zero=True,
        diffusion_steps=args.diffusion_steps,
        angle_dropout=args.angle_dropout,
    ).to(device)
    
    # Load heatmap checkpoint
    heatmap_ckpt = torch.load(args.checkpoint, map_location='cpu')
    
    # Filter keys
    backbone_state = {k.replace('backbone.', ''): v for k, v in heatmap_ckpt.items() if k.startswith('backbone.')}
    keypoint_state = {k.replace('keypoint_head.', ''): v for k, v in heatmap_ckpt.items() if k.startswith('keypoint_head.')}
    
    model.backbone.load_state_dict(backbone_state, strict=True)
    model.keypoint_head.load_state_dict(keypoint_state, strict=True)
    if rank == 0:
        print("✅ Loaded heatmap checkpoint")
    
    # Freeze heatmap
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.keypoint_head.parameters():
        param.requires_grad = False
    
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    
    raw_model = unwrap_model(model)
    optimizer = build_optimizer(raw_model, args, backbone_active=False)
    
    # Wandb
    if args.use_wandb and rank == 0:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
    
    # Training loop
    best_mae = float('inf')
    global_step = 0
    
    for epoch in range(1, args.epochs + 1):
        if world_size > 1:
            train_sampler.set_epoch(epoch)

        if epoch == args.warmup_frozen_epochs + 1:
            unfrozen = maybe_unfreeze_backbone(raw_model, args.unfreeze_blocks)
            optimizer = build_optimizer(raw_model, args, backbone_active=(unfrozen > 0))
            if rank == 0:
                trainable = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
                print(f"\nUnfroze last {unfrozen} backbone blocks")
                print(f"Trainable params: {trainable:,}")
        
        train_metrics, global_step = train_epoch(model, train_loader, optimizer, device, epoch, rank, args, global_step)
        val_mae, mae_per_joint = validate(model, val_loader, device, rank)
        
        if rank == 0:
            print(f"\nEpoch {epoch}:")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"    Noise Loss: {train_metrics['noise_loss']:.4f}")
            print(f"    Init Loss: {train_metrics['init_loss']:.4f}")
            print(f"    Recon Loss: {train_metrics['recon_loss']:.4f}")
            print(f"    FK Loss: {train_metrics['fk_loss']:.4f}")
            print(f"  Val MAE: {val_mae:.2f}°")
            print(f"  Per-joint (J1-J6): {mae_per_joint}")
            
            if args.use_wandb:
                wandb.log({
                    'train_loss': train_metrics['loss'],
                    'train_noise_loss': train_metrics['noise_loss'],
                    'train_init_loss': train_metrics['init_loss'],
                    'train_recon_loss': train_metrics['recon_loss'],
                    'train_fk_loss': train_metrics['fk_loss'],
                    'val_mae': val_mae,
                    'epoch': epoch,
                    'global_step': global_step,
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
