import torch
from dataset import PoseEstimationDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

def check_z_depths():
    val_dir = "/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_test_dr"
    keypoint_names = ['link0', 'link2', 'link3', 'link4', 'link6', 'link7', 'hand']
    
    val_dataset = PoseEstimationDataset(
        data_dir=val_dir, keypoint_names=keypoint_names,
        image_size=(512, 512),
        heatmap_size=(512, 512),
        augment=False, include_angles=True,
    )
    
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True, num_workers=2)
    
    min_z = float('inf')
    max_z = float('-inf')
    avg_z = 0.0
    count = 0
    
    print("Checking true Camera Z Depths in real dataset (Sample size: 50 batches)...")
    
    for i, batch in enumerate(tqdm(val_loader)):
        if i >= 50:
            break
            
        gt_kp_3d = batch['keypoints_3d'] # (B, N, 3) relative to camera!
        z_vals = gt_kp_3d[..., 2] # (B, N)
        valid_mask = batch['valid_mask'] # (B, N)
        
        valid_z = z_vals[valid_mask.bool()]
        
        if len(valid_z) > 0:
            batch_min = valid_z.min().item()
            batch_max = valid_z.max().item()
            
            if batch_min < min_z: min_z = batch_min
            if batch_max > max_z: max_z = batch_max
            
            avg_z += valid_z.sum().item()
            count += len(valid_z)
            
    avg_z = avg_z / count if count > 0 else 0
    print(f"\n--- GT Camera Z Statistics ---")
    print(f"Dataset: {val_dir}")
    print(f"Min Z: {min_z:.4f} meters")
    print(f"Avg Z: {avg_z:.4f} meters")
    print(f"Max Z: {max_z:.4f} meters")
    
if __name__ == '__main__':
    check_z_depths()
