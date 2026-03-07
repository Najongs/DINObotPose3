"""Feature analysis: check if mean pooling loses critical info"""
import torch
import numpy as np
from model import DINOv3PoseEstimator

PANDA_JOINT_MEAN = torch.tensor([-5.22e-02, 2.68e-01, 6.04e-03, -2.01e+00, 1.49e-02, 1.99e+00, 0.0])
PANDA_JOINT_STD  = torch.tensor([1.025, 0.645, 0.511, 0.508, 0.769, 0.511, 1.0])
from dataset import PoseEstimationDataset
from torch.utils.data import DataLoader

device = 'cuda'
checkpoint = '/data/public/NAS/DINObotPose3/TRAIN/outputs_3d_v2/train_20260307_033000/last_joint_angle.pth'

# Load model
model = DINOv3PoseEstimator(
    dino_model_name='facebook/dinov3-vitb16-pretrain-lvd1689m',
    heatmap_size=(512, 512),
    unfreeze_blocks=2,
    fix_joint7_zero=True
).to(device)
model.load_state_dict(torch.load(checkpoint, map_location=device))
model.eval()

# Load val data
val_dataset = PoseEstimationDataset(
    data_dir='/data/public/NAS/DINObotPose2/Dataset/Converted_dataset/DREAM_real/panda-3cam_azure',
    keypoint_names=['link0', 'link2', 'link3', 'link4', 'link6', 'link7', 'hand'],
    image_size=(512, 512),
    heatmap_size=(512, 512),
    augment=False,
    include_angles=True
)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

joint_mean = PANDA_JOINT_MEAN.to(device)
joint_std = PANDA_JOINT_STD.to(device)

# Analyze
per_joint_errors = []
feature_stats = []

with torch.no_grad():
    for i, batch in enumerate(val_loader):
        if i >= 10:
            break
        imgs = batch['image'].to(device)
        gt_angles = batch['angles'][:, :6].to(device)
        
        # Get features
        dino_feat = model.backbone(imgs)  # (B, N, 768)
        
        # Feature variance per sample
        feat_var = dino_feat.var(dim=1).mean(dim=1)  # (B,)
        feature_stats.append(feat_var.cpu().numpy())
        
        # Predictions
        preds = model(imgs)
        pred_norm = preds['joint_angles'][:, :6]
        pred_angles = pred_norm * joint_std[:6] + joint_mean[:6]
        
        # Per-joint error
        errors = (pred_angles - gt_angles).abs() * (180 / np.pi)
        per_joint_errors.append(errors.cpu().numpy())

per_joint_errors = np.concatenate(per_joint_errors, axis=0)  # (N, 6)
feature_stats = np.concatenate(feature_stats)

print("="*60)
print("FEATURE ANALYSIS")
print("="*60)
print(f"Feature variance (mean): {feature_stats.mean():.4f}")
print(f"Feature variance (std):  {feature_stats.std():.4f}")
print(f"  → Low variance = features are similar (bad for angle prediction)")
print()

print("PER-JOINT ERROR DISTRIBUTION")
print("="*60)
for j in range(6):
    err = per_joint_errors[:, j]
    print(f"J{j}: mean={err.mean():.1f}° std={err.std():.1f}° "
          f"median={np.median(err):.1f}° max={err.max():.1f}°")
print()

# Check correlation between joints
print("JOINT ERROR CORRELATION (high = coupled errors)")
print("="*60)
corr = np.corrcoef(per_joint_errors.T)
for i in range(6):
    for j in range(i+1, 6):
        if abs(corr[i,j]) > 0.5:
            print(f"J{i} ↔ J{j}: {corr[i,j]:.2f}")
