"""Analyze per-joint heatmap prediction accuracy"""
import torch
import numpy as np
from model import DINOv3PoseEstimator, soft_argmax_2d
from dataset import PoseEstimationDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

device = 'cuda'
checkpoint = '/data/public/NAS/DINObotPose3/TRAIN/outputs_heatmap/best_heatmap.pth'

# Load model
model = DINOv3PoseEstimator(
    dino_model_name='facebook/dinov3-vitb16-pretrain-lvd1689m',
    heatmap_size=(512, 512),
    unfreeze_blocks=2,
    fix_joint7_zero=True
).to(device)

# Load heatmap checkpoint
ckpt = torch.load(checkpoint, map_location=device)
ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
model.load_state_dict(ckpt, strict=False)
model.eval()

# Load datasets
joint_names = ['link0', 'link2', 'link3', 'link4', 'link6', 'link7', 'hand']

print("Loading train dataset...")
train_dataset = PoseEstimationDataset(
    data_dir='/data/public/NAS/DINObotPose2/Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_train_dr',
    keypoint_names=joint_names,
    image_size=(512, 512),
    heatmap_size=(512, 512),
    augment=False,
    include_angles=True
)

print("Loading val dataset...")
val_dataset = PoseEstimationDataset(
    data_dir='/data/public/NAS/DINObotPose2/Dataset/Converted_dataset/DREAM_real/panda-3cam_azure',
    keypoint_names=joint_names,
    image_size=(512, 512),
    heatmap_size=(512, 512),
    augment=False,
    include_angles=True
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

def analyze_heatmap_accuracy(loader, name, max_batches=50):
    per_joint_errors = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc=f"Analyzing {name}")):
            if i >= max_batches:
                break
            
            imgs = batch['image'].to(device)
            gt_kp_2d = batch['keypoints'].to(device)  # (B, N, 2)
            valid_mask = batch['valid_mask'].to(device)
            
            # Predict heatmaps
            preds = model(imgs)
            pred_hm = preds['heatmaps_2d']  # (B, N, H, W)
            
            # Extract UV from heatmaps
            pred_uv = soft_argmax_2d(pred_hm, temperature=100.0)  # (B, N, 2)
            
            # Compute per-joint pixel error
            errors = (pred_uv - gt_kp_2d).norm(dim=-1)  # (B, N)
            errors = errors * valid_mask.float()  # Mask invalid
            
            per_joint_errors.append(errors.cpu().numpy())
    
    per_joint_errors = np.concatenate(per_joint_errors, axis=0)  # (N_samples, N_joints)
    return per_joint_errors

# Analyze
print("\n" + "="*60)
print("HEATMAP QUALITY ANALYSIS")
print("="*60)

train_errors = analyze_heatmap_accuracy(train_loader, "Train", max_batches=50)
val_errors = analyze_heatmap_accuracy(val_loader, "Val", max_batches=50)

print(f"\nPer-Joint 2D Prediction Error (pixels)")
print("="*80)
print(f"{'Joint':<10} {'Train Mean':>12} {'Train Med':>12} {'Val Mean':>12} {'Val Med':>12} {'Gap':>8}")
print("-"*80)

for j, jname in enumerate(joint_names):
    train_mean = train_errors[:, j].mean()
    train_med = np.median(train_errors[:, j])
    val_mean = val_errors[:, j].mean()
    val_med = np.median(val_errors[:, j])
    gap = val_mean - train_mean
    
    marker = " ⚠️" if val_mean > 10.0 or gap > 5.0 else ""
    print(f"{jname:<10} {train_mean:>10.2f}px {train_med:>10.2f}px {val_mean:>10.2f}px {val_med:>10.2f}px {gap:>6.2f}px{marker}")

print("-"*80)
print(f"{'MEAN':<10} {train_errors.mean():>10.2f}px {np.median(train_errors):>10.2f}px {val_errors.mean():>10.2f}px {np.median(val_errors):>10.2f}px {val_errors.mean()-train_errors.mean():>6.2f}px")

# Correlation with angle error
print(f"\n{'='*60}")
print("HYPOTHESIS: Bad heatmap joints → Bad angle prediction")
print("="*60)
print("If link0 (J0 base) has high 2D error, that explains J0 angle error!")
print("\nNext: Check if 2D error correlates with angle error")
