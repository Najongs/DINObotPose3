import argparse
import os
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm

# Try importing from current or parent directories
try:
    from model_v4 import DINOv3PoseEstimatorV4, panda_forward_kinematics
    from checkpoint_compat import load_checkpoint_compat
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent.parent / 'TRAIN'))
    from model_v4 import DINOv3PoseEstimatorV4, panda_forward_kinematics
    from checkpoint_compat import load_checkpoint_compat

# ─── Constants (Matched with train_3d_v4.py) ───
PANDA_JOINT_MEAN = torch.tensor([-5.22e-02, 2.68e-01, 6.04e-03, -2.01e+00, 1.49e-02, 1.99e+00, 0.0])
PANDA_JOINT_STD  = torch.tensor([1.025, 0.645, 0.511, 0.508, 0.769, 0.511, 1.0])
LINK_NAMES = ['link0', 'link2', 'link3', 'link4', 'link6', 'link7', 'hand']

def project_3d_to_2d(points_3d, R, T, K):
    pts_cam = (points_3d @ R.T) + T
    pts_img_homo = pts_cam @ K.T
    z = pts_img_homo[..., 2:3]
    uv = pts_img_homo[..., :2] / (z + 1e-6)
    return uv

def solve_pnp(pts2d, pts3d, K):
    success, rvec, tvec = cv2.solvePnP(
        pts3d.astype(np.float64), 
        pts2d.astype(np.float64), 
        K.astype(np.float64), 
        None, 
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if success:
        R, _ = cv2.Rodrigues(rvec)
        return R, tvec.flatten()
    return None, None

def visualize_and_save(image_bgr, kp_2d_pred, kp_3d_pred, kp_3d_gt, output_path, title="V4 Inference"):
    # Create side-by-side plot
    fig = plt.figure(figsize=(15, 7))
    
    # 2D Overlay
    ax1 = fig.add_subplot(121)
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    ax1.imshow(img_rgb)
    
    # Draw keypoints on overlay
    colors = plt.cm.rainbow(np.linspace(0, 1, 7))
    for i in range(7):
        u, v = kp_2d_pred[i]
        ax1.scatter(u, v, color=colors[i], s=50, edgecolors='white', label=LINK_NAMES[i])
        ax1.text(u+5, v+5, LINK_NAMES[i], color='white', fontsize=8, 
                 bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1))
    
    ax1.set_title("2D Projection Overlay")
    ax1.axis('off')
    
    # 3D Plot
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(kp_3d_pred[:, 0], kp_3d_pred[:, 1], kp_3d_pred[:, 2], 'ro-', label='Prediction', linewidth=2)
    if kp_3d_gt is not None:
        ax2.plot(kp_3d_gt[:, 0], kp_3d_gt[:, 1], kp_3d_gt[:, 2], 'go--', label='Ground Truth', alpha=0.6)
    
    # Plot links
    for i in range(7):
        ax2.scatter(kp_3d_pred[i, 0], kp_3d_pred[i, 1], kp_3d_pred[i, 2], color=colors[i], s=40)

    ax2.set_xlabel('X (m)'); ax2.set_ylabel('Y (m)'); ax2.set_zlabel('Z (m)')
    ax2.set_title("3D Pose (Robot Frame)")
    ax2.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def run_single_inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Model
    model = DINOv3PoseEstimatorV4(
        dino_model_name=args.model_name,
        heatmap_size=(args.heatmap_size, args.heatmap_size),
        unfreeze_blocks=0,
        fix_joint7_zero=True
    ).to(device)
    
    load_checkpoint_compat(
        model=model,
        checkpoint_path=args.model_path,
        device=device,
        is_main_process=True
    )
    model.eval()

    joint_mean = PANDA_JOINT_MEAN.to(device)
    joint_std = PANDA_JOINT_STD.to(device)

    # Resolve Inputs
    input_path = Path(args.input)
    if input_path.is_dir():
        json_files = sorted(list(input_path.glob("*.json")))
        if not json_files:
            # Maybe images?
            image_files = sorted(list(input_path.glob("*.jpg")) + list(input_path.glob("*.png")))
            work_items = [(img, None) for img in image_files]
        else:
            work_items = [(jf, jf) for jf in json_files]
    else:
        if input_path.suffix == '.json':
            work_items = [(input_path, input_path)]
        else:
            work_items = [(input_path, None)]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print(f"Processing {len(work_items)} items...")

    for i, (item_path, json_path) in enumerate(tqdm(work_items)):
        # Load image and metadata
        image_bgr = None
        camera_K = np.eye(3)
        gt_angles = None

        if json_path:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Metadata can be at top level or under 'meta'
            meta = data.get('meta', {})
            img_name = data.get('image_path') or meta.get('image_path') or item_path.with_suffix('.jpg').name
            
            # Robust path resolution
            if img_name.startswith("../dataset/"):
                img_name = img_name.replace("../dataset/", "../../../")
            
            full_img_path = item_path.parent / img_name
            if not full_img_path.exists():
                # Try relative to dataset root (DREAM style)
                full_img_path = item_path.parents[2] / img_name.replace("../../../", "")
            if not full_img_path.exists():
                # Fallback to sibling JPG
                full_img_path = item_path.with_suffix('.rgb.jpg')
            if not full_img_path.exists():
                full_img_path = item_path.with_suffix('.jpg')
            
            image_bgr = cv2.imread(str(full_img_path))
            
            k_data = data.get('camera_K') or meta.get('K') or np.eye(3)
            camera_K = np.array(k_data)
            
            # Joint angles from 'sim_state' or top level
            sim_state = data.get('sim_state', {})
            joints = sim_state.get('joints', [])
            if joints:
                gt_angles = [j['position'] for j in joints if 'joint' in j['name'] and 'finger' not in j['name']]
            else:
                gt_angles = data.get('joint_angles')
        else:
            image_bgr = cv2.imread(str(item_path))
            # Attempt to guess K for 640x480?
            if image_bgr is not None:
                h, w = image_bgr.shape[:2]
                camera_K = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]])

        if image_bgr is None:
            print(f"Warning: Could not load image for {item_path}")
            continue

        orig_h, orig_w = image_bgr.shape[:2]
        
        # Inference
        img_pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        img_tensor = transform(img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            pred_angles_norm = output['joint_angles'][0] # (7,)
            pred_angles = (pred_angles_norm * joint_std + joint_mean).cpu()
            
            pred_angles_fk = pred_angles.clone()
            pred_angles_fk[6] = 0.0
            kp_3d_pred = panda_forward_kinematics(pred_angles_fk.unsqueeze(0))[0].numpy()
            
            gt_kp_3d = None
            if gt_angles is not None:
                gt_angles_t = torch.tensor(gt_angles, dtype=torch.float32).unsqueeze(0)
                gt_angles_t[:, 6] = 0.0
                gt_kp_3d = panda_forward_kinematics(gt_angles_t)[0].numpy()

        # Visualization: We need PnP to project back to 2D for visualization
        # In real-world K might be different. Let's use the provided K if available.
        # Scale K to resizing
        scale_x = args.image_size / orig_w
        scale_y = args.image_size / orig_h
        scaled_K = camera_K.copy()
        scaled_K[0, 0] *= scale_x; scaled_K[1, 1] *= scale_y
        scaled_K[0, 2] *= scale_x; scaled_K[1, 2] *= scale_y

        # If we have GT angles, we can try to guess Extrinsics via PnP (DREAM style)
        # But for V4, let's just use the Predicted angles to get 3D.
        # How to project to 2D? 
        # We need Camera Extrinsics. If not in JSON, we can't project unless we assume something.
        # Usually datasets like DREAM have 'ROI' or 'camera_pose' (extrinsics).
        
        # Check for extrinsics in JSON
        R, T = None, None
        if json_path:
            with open(json_path, 'r') as f:
                d = json.load(f)
            if 'camera_pose' in d: # From DREAM
                pose = np.array(d['camera_pose'])
                R = pose[:3, :3]
                T = pose[:3, 3]
            elif 'keypoints_2d' in d and gt_kp_3d is not None:
                # Solve PnP from GT to find camera location
                kp2d = np.array(d['keypoints_2d'])
                # Scale kp2d to match resized image if they are in original pixels
                kp2d[:, 0] *= scale_x
                kp2d[:, 1] *= scale_y
                R, T = solve_pnp(kp2d, gt_kp_3d, scaled_K)

        if R is None:
            # Default or fail projection
            kp_2d_proj = np.zeros((7, 2))
            title_ext = " (No Extrinsics - 2D Projection Disabled)"
        else:
            kp_2d_proj = project_3d_to_2d(kp_3d_pred, R, T, scaled_K)
            title_ext = ""

        # Save Visual
        out_name = item_path.stem + "_v4_infer.png"
        image_resized = cv2.resize(image_bgr, (args.image_size, args.image_size))
        visualize_and_save(
            image_resized, 
            kp_2d_proj, 
            kp_3d_pred, 
            gt_kp_3d, 
            output_dir / out_name, 
            f"V4 Inference: {item_path.name}{title_ext}"
        )

def main():
    parser = argparse.ArgumentParser(description="V4 Single/Folder Inference Visualization")
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--input', type=str, required=True, help='Path to JSON, Image, or Folder')
    parser.add_argument('--output-dir', type=str, default='./inference_v4_viz')
    parser.add_argument('--model-name', type=str, default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    parser.add_argument('--image-size', type=int, default=512)
    parser.add_argument('--heatmap-size', type=int, default=512)
    args = parser.parse_args()
    
    run_single_inference(args)

if __name__ == '__main__':
    main()
