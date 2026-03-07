"""Quick test: Diffusion vs Direct"""
import torch
from model_diffusion import DINOv3DiffusionPoseEstimator

device = 'cuda'

# Create model
model = DINOv3DiffusionPoseEstimator(
    dino_model_name='facebook/dinov3-vitb16-pretrain-lvd1689m',
    heatmap_size=(512, 512),
    unfreeze_blocks=0,
    fix_joint7_zero=True
).to(device)

# Test forward
dummy_img = torch.randn(2, 3, 512, 512).to(device)

print("Testing training mode...")
model.train()
out_train = model(dummy_img, training=True)
print(f"  Condition shape: {out_train['condition'].shape}")
print(f"  UV shape: {out_train['uv'].shape}")

print("\nTesting inference mode...")
model.eval()
with torch.no_grad():
    out_inf = model(dummy_img, training=False)
print(f"  Joint angles shape: {out_inf['joint_angles'].shape}")
print(f"  Angles: {out_inf['joint_angles'][0]}")

print("\n✅ Diffusion model works!")
