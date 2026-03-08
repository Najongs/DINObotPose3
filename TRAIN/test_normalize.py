import torch
from dataset import PoseEstimationDataset
from torch.utils.data import DataLoader

PANDA_JOINT_MEAN = torch.tensor([-5.22e-02, 2.68e-01, 6.04e-03, -2.01e+00, 1.49e-02, 1.99e+00, 0.0])
PANDA_JOINT_STD = torch.tensor([1.025, 0.645, 0.511, 0.508, 0.769, 0.511, 1.0])

def normalize_angles(angles):
    return (angles - PANDA_JOINT_MEAN.to(angles.device)) / PANDA_JOINT_STD.to(angles.device)

ds = PoseEstimationDataset(
    '/data/public/NAS/DINObotPose2/Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_train_dr',
    keypoint_names=['link0', 'link2', 'link3', 'link4', 'link6', 'link7', 'hand'],
    image_size=(512, 512),
    heatmap_size=(512, 512)
)

loader = DataLoader(ds, batch_size=4, shuffle=False)
batch = next(iter(loader))

print("Batch angles shape:", batch['angles'].shape)
print("Batch angles device:", batch['angles'].device)

angles = batch['angles'].cuda()
print("GPU angles shape:", angles.shape)

try:
    norm = normalize_angles(angles[:, :6])
    print("✅ Normalized shape:", norm.shape)
except Exception as e:
    print("❌ Error:", e)
    import traceback
    traceback.print_exc()
