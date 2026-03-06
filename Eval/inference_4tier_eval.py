"""
4-Tier PnP Outlier Analysis for DINOv3 3D Pose Estimation

Evaluates a trained model on a dataset with 4 tiers of 3D metrics:
  1. ALL SAMPLES - raw PnP (iterative) on every frame
  2. PnP FILTERED - iterative PnP with reproj<5px & depth sanity
  3. RANSAC EPnP - RANSAC with iterative refinement
  4. CONF-FILTERED RANSAC - drop low-confidence keypoints, then RANSAC+refine

Outputs:
  - Console: 4-tier comparison table
  - metrics_4tier.json: all metrics
  - per_frame_errors.json: per-frame ADD for each tier
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../TRAIN')))
from model import (
    DINOv3PoseEstimator, panda_forward_kinematics, soft_argmax_2d,
    solve_pnp_batch, solve_pnp_ransac_batch, solve_pnp_conf_batch
)
from checkpoint_compat import load_checkpoint_compat


def setup_distributed(enable_distributed: bool):
    """Initialize distributed inference when launched with torchrun."""
    if not enable_distributed:
        return False, -1, 1, -1

    if not ('RANK' in os.environ and 'WORLD_SIZE' in os.environ and 'LOCAL_RANK' in os.environ):
        print("Warning: --distributed set but torchrun env not found. Falling back to single process.")
        return False, -1, 1, -1

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])

    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    dist.init_process_group(backend=backend)
    return True, rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Dataset (reuse pattern from inference_dataset.py)
# ---------------------------------------------------------------------------
class EvalDataset(Dataset):
    def __init__(self, data_dir, keypoint_names, image_size=(512, 512), verbose=True):
        self.data_dir = Path(data_dir)
        self.keypoint_names = keypoint_names
        self.image_size = image_size
        self.is_synthetic = 'syn' in str(self.data_dir).lower()

        self.json_files = sorted(list(self.data_dir.glob("*.json")))
        if not self.json_files:
            raise ValueError(f"No JSON files found in {data_dir}")

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        if verbose:
            print(f"Found {len(self.json_files)} frames in {data_dir}")

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        json_path = self.json_files[idx]
        with open(json_path, "r") as f:
            data = json.load(f)

        # Image path
        img_path_str = data.get("meta", {}).get("image_path", None)
        if img_path_str is None:
            raise KeyError(f"'meta.image_path' missing in {json_path}")
        if img_path_str.startswith('../dataset/'):
            img_path_str = img_path_str.replace('../dataset/', '../../../', 1)
        img_path = (json_path.parent / img_path_str).resolve()
        if not img_path.exists():
            img_path = (self.data_dir / img_path_str).resolve()
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.width, image.height

        # Keypoints
        kp_2d = np.zeros((len(self.keypoint_names), 2), dtype=np.float32)
        kp_3d = np.zeros((len(self.keypoint_names), 3), dtype=np.float32)
        found = np.zeros(len(self.keypoint_names), dtype=np.float32)

        if "objects" in data and len(data["objects"]) > 0:
            kp_dict = {}
            for obj in data["objects"]:
                if "keypoints" in obj:
                    for kp in obj["keypoints"]:
                        kp_dict[kp["name"]] = kp
            for i, name in enumerate(self.keypoint_names):
                if name in kp_dict:
                    kp_2d[i] = kp_dict[name]["projected_location"]
                    if "location" in kp_dict[name]:
                        kp_3d[i] = kp_dict[name]["location"]
                    found[i] = 1.0

        if self.is_synthetic:
            kp_3d /= 100.0

        # Camera K
        camera_K = np.zeros((3, 3), dtype=np.float32)
        if "meta" in data and "K" in data["meta"]:
            camera_K = np.array(data["meta"]["K"], dtype=np.float32)

        # Joint angles
        angles = np.zeros(7, dtype=np.float32)
        if "sim_state" in data and "joints" in data["sim_state"]:
            for i, j in enumerate(data["sim_state"]["joints"][:7]):
                if "position" in j:
                    angles[i] = j["position"]

        image_tensor = self.transform(image)

        return {
            "image": image_tensor,
            "gt_2d": kp_2d,
            "gt_3d": kp_3d,
            "found": found,
            "camera_K": camera_K,
            "original_size": np.array([orig_w, orig_h], dtype=np.float32),
            "gt_angles": angles,
            "name": json_path.stem,
        }


# ---------------------------------------------------------------------------
# RoboPEPP ADD AUC
# ---------------------------------------------------------------------------
def compute_add_auc(adds_m, threshold=0.1):
    """
    RoboPEPP-style ADD AUC.
    adds_m: 1D array of per-frame ADD values (meters).
    Returns AUC in [0, 1].
    """
    if len(adds_m) == 0:
        return 0.0
    delta = 0.00001
    thresholds = np.arange(0.0, threshold, delta)
    counts = (adds_m[None, :] <= thresholds[:, None]).sum(axis=1) / float(len(adds_m))
    return float(np.trapz(counts, dx=delta) / threshold)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def run_eval(args):
    is_distributed, rank, world_size, local_rank = setup_distributed(args.distributed)
    is_main_process = (not is_distributed) or (rank == 0)

    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}' if is_distributed else 'cuda')
    else:
        device = torch.device('cpu')

    if is_main_process:
        print(f"Device: {device}")

    keypoint_names = [
        'panda_link0', 'panda_link2', 'panda_link3',
        'panda_link4', 'panda_link6', 'panda_link7', 'panda_hand'
    ]
    heatmap_size = args.image_size

    # Model
    model = DINOv3PoseEstimator(
        dino_model_name=args.model_name,
        heatmap_size=heatmap_size,
        unfreeze_blocks=0,
        fix_joint7_zero=args.fix_joint7
    ).to(device)

    if is_main_process:
        print(f"Loading checkpoint: {args.model_path}")
    load_checkpoint_compat(model, args.model_path, device, is_main_process=is_main_process)
    model.eval()

    # Dataset
    dataset = EvalDataset(
        args.dataset_dir,
        keypoint_names,
        image_size=(heatmap_size, heatmap_size),
        verbose=is_main_process,
    )
    sampler = None
    if is_distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
    loader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=False, sampler=sampler, num_workers=args.num_workers, pin_memory=True
    )

    # Accumulators per tier
    # tier keys: 'all', 'filtered', 'ransac', 'conf'
    tier_names = ['all', 'filtered', 'ransac', 'conf']
    tier_labels = {
        'all': 'ALL SAMPLES',
        'filtered': 'PnP FILTERED',
        'ransac': 'RANSAC EPnP',
        'conf': 'CONF-FILTERED RANSAC',
    }

    # Per-frame results: list of dicts
    per_frame_results = []

    # Aggregated per-tier
    tier_adds = {t: [] for t in tier_names}  # per-frame ADD (m)
    tier_per_joint = {t: [] for t in tier_names}  # per-frame per-joint error (m), shape (N_valid, 7)
    tier_valid_count = {t: 0 for t in tier_names}
    tier_reproj = {t: [] for t in tier_names}
    tier_inliers = {'ransac': [], 'conf': []}

    total_frames = 0
    angle_errors_all = []

    if is_main_process:
        if is_distributed:
            print(f"\nRunning distributed inference on {len(dataset)} frames with world_size={world_size}...")
        else:
            print(f"\nRunning inference on {len(dataset)} frames...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", disable=not is_main_process):
            images = batch["image"].to(device)
            gt_3d = batch["gt_3d"].numpy()  # (B, 7, 3)
            gt_2d = batch["gt_2d"].numpy()
            gt_angles = batch["gt_angles"].numpy()
            found = batch["found"].numpy()
            camera_K = batch["camera_K"]
            orig_sizes = batch["original_size"].numpy()
            names = batch["name"]
            B = images.shape[0]

            # Scale camera_K to heatmap resolution
            camera_K_scaled = camera_K.clone().float()
            for b in range(B):
                ow, oh = orig_sizes[b]
                sx, sy = heatmap_size / ow, heatmap_size / oh
                camera_K_scaled[b, 0, 0] *= sx
                camera_K_scaled[b, 1, 1] *= sy
                camera_K_scaled[b, 0, 2] *= sx
                camera_K_scaled[b, 1, 2] *= sy
            camera_K_scaled = camera_K_scaled.to(device)

            outputs = model(images, camera_K=camera_K_scaled)

            pred_angles = outputs["joint_angles"].cpu().numpy()  # (B, 7)
            pred_2d_hm = soft_argmax_2d(outputs["heatmaps_2d"]).cpu().numpy()  # (B, 7, 2)

            # Angle errors
            for b in range(B):
                if np.any(gt_angles[b] != 0):
                    diff = pred_angles[b] - gt_angles[b]
                    diff = np.arctan2(np.sin(diff), np.cos(diff))
                    angle_errors_all.append(np.abs(np.degrees(diff)))

            # Extract 3D results for each tier
            for b in range(B):
                total_frames += 1
                frame_name = names[b]
                has_gt_3d = np.any(gt_3d[b] != 0)

                frame_result = {"name": frame_name}

                if not has_gt_3d:
                    per_frame_results.append(frame_result)
                    continue

                gt_3d_b = gt_3d[b]  # (7, 3) in meters

                # --- Tier 1: ALL (iterative PnP, no filtering) ---
                if 'keypoints_3d_cam' in outputs:
                    pred_cam = outputs['keypoints_3d_cam'][b].cpu().numpy()
                    per_joint_err = np.linalg.norm(pred_cam - gt_3d_b, axis=1)  # (7,) meters
                    add_m = float(per_joint_err.mean())
                    tier_adds['all'].append(add_m)
                    tier_per_joint['all'].append(per_joint_err)
                    tier_valid_count['all'] += 1
                    if 'reproj_errors' in outputs:
                        tier_reproj['all'].append(outputs['reproj_errors'][b].item())
                    frame_result['add_all_m'] = add_m

                # --- Tier 2: FILTERED (reproj < 5px & depth OK) ---
                if 'pnp_valid' in outputs and outputs['pnp_valid'][b].item():
                    pred_cam = outputs['keypoints_3d_cam'][b].cpu().numpy()
                    per_joint_err = np.linalg.norm(pred_cam - gt_3d_b, axis=1)
                    add_m = float(per_joint_err.mean())
                    tier_adds['filtered'].append(add_m)
                    tier_per_joint['filtered'].append(per_joint_err)
                    tier_valid_count['filtered'] += 1
                    if 'reproj_errors' in outputs:
                        tier_reproj['filtered'].append(outputs['reproj_errors'][b].item())
                    frame_result['add_filtered_m'] = add_m

                # --- Tier 3: RANSAC EPnP ---
                if 'pnp_valid_ransac' in outputs and outputs['pnp_valid_ransac'][b].item():
                    pred_cam = outputs['keypoints_3d_cam_ransac'][b].cpu().numpy()
                    per_joint_err = np.linalg.norm(pred_cam - gt_3d_b, axis=1)
                    add_m = float(per_joint_err.mean())
                    tier_adds['ransac'].append(add_m)
                    tier_per_joint['ransac'].append(per_joint_err)
                    tier_valid_count['ransac'] += 1
                    if 'reproj_errors_ransac' in outputs:
                        tier_reproj['ransac'].append(outputs['reproj_errors_ransac'][b].item())
                    if 'pnp_n_inliers_ransac' in outputs:
                        tier_inliers['ransac'].append(outputs['pnp_n_inliers_ransac'][b].item())
                    frame_result['add_ransac_m'] = add_m

                # --- Tier 4: CONF-FILTERED RANSAC ---
                if 'pnp_valid_conf' in outputs and outputs['pnp_valid_conf'][b].item():
                    pred_cam = outputs['keypoints_3d_cam_conf'][b].cpu().numpy()
                    per_joint_err = np.linalg.norm(pred_cam - gt_3d_b, axis=1)
                    add_m = float(per_joint_err.mean())
                    tier_adds['conf'].append(add_m)
                    tier_per_joint['conf'].append(per_joint_err)
                    tier_valid_count['conf'] += 1
                    if 'reproj_errors_conf' in outputs:
                        tier_reproj['conf'].append(outputs['reproj_errors_conf'][b].item())
                    if 'pnp_n_used_conf' in outputs:
                        tier_inliers['conf'].append(outputs['pnp_n_used_conf'][b].item())
                    frame_result['add_conf_m'] = add_m

                per_frame_results.append(frame_result)

    local_payload = {
        "per_frame_results": per_frame_results,
        "tier_adds": tier_adds,
        "tier_per_joint": tier_per_joint,
        "tier_valid_count": tier_valid_count,
        "tier_reproj": tier_reproj,
        "tier_inliers": tier_inliers,
        "total_frames": total_frames,
        "angle_errors_all": angle_errors_all,
    }

    if is_distributed:
        gathered_payloads = [None for _ in range(world_size)] if is_main_process else None
        dist.gather_object(local_payload, gathered_payloads, dst=0)
        dist.barrier()
        if not is_main_process:
            cleanup_distributed()
            return

        per_frame_results = []
        tier_adds = {t: [] for t in tier_names}
        tier_per_joint = {t: [] for t in tier_names}
        tier_valid_count = {t: 0 for t in tier_names}
        tier_reproj = {t: [] for t in tier_names}
        tier_inliers = {'ransac': [], 'conf': []}
        total_frames = 0
        angle_errors_all = []

        for payload in gathered_payloads:
            if payload is None:
                continue
            per_frame_results.extend(payload["per_frame_results"])
            total_frames += int(payload["total_frames"])
            angle_errors_all.extend(payload["angle_errors_all"])
            for tier in tier_names:
                tier_adds[tier].extend(payload["tier_adds"][tier])
                tier_per_joint[tier].extend(payload["tier_per_joint"][tier])
                tier_valid_count[tier] += int(payload["tier_valid_count"][tier])
                tier_reproj[tier].extend(payload["tier_reproj"][tier])
            for tier in tier_inliers:
                tier_inliers[tier].extend(payload["tier_inliers"][tier])

        per_frame_results.sort(key=lambda x: x.get("name", ""))

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  4-TIER PnP OUTLIER ANALYSIS")
    print("=" * 70)

    # Angle errors
    if angle_errors_all:
        ae = np.array(angle_errors_all)
        print(f"\n  [Joint Angles] ({len(ae)} frames with GT)")
        print(f"  Mean: {ae.mean():.2f} deg | Median: {np.median(ae.mean(axis=1)):.2f} deg")
        for j in range(7):
            print(f"    J{j}: mean={ae[:, j].mean():.2f}, max={ae[:, j].max():.2f}")

    # Per-tier 3D metrics
    metrics = {"total_frames": total_frames}

    for tier in tier_names:
        adds = np.array(tier_adds[tier]) if tier_adds[tier] else np.array([])
        n_valid = tier_valid_count[tier]
        reproj = np.array(tier_reproj[tier]) if tier_reproj[tier] else np.array([])

        print(f"\n  {'='*60}")
        print(f"  3D POSE ERROR - {tier_labels[tier]} ({n_valid}/{total_frames})")
        print(f"  {'='*60}")

        if len(adds) == 0:
            print(f"    No valid samples.")
            metrics[f"{tier}_valid"] = 0
            continue

        adds_mm = adds * 1000
        auc = compute_add_auc(adds, threshold=args.add_auc_threshold)

        print(f"    ADD AUC@{int(args.add_auc_threshold*1000)}mm (RoboPEPP): {auc:.4f}")
        print(f"    Mean: {adds_mm.mean():.2f} mm | Median: {np.median(adds_mm):.2f} mm")
        print(f"    Max:  {adds_mm.max():.2f} mm | Std: {adds_mm.std():.2f} mm")

        if len(reproj) > 0:
            print(f"    Reproj RMSE: mean={reproj.mean():.2f}px, max={reproj.max():.2f}px")

        if tier in tier_inliers and tier_inliers[tier]:
            inl = np.array(tier_inliers[tier])
            label = "Inliers" if tier == 'ransac' else "KPs used"
            print(f"    {label}: mean={inl.mean():.1f}/7")

        # Per-joint breakdown
        if tier_per_joint[tier]:
            pj = np.array(tier_per_joint[tier]) * 1000  # (N, 7) mm
            print(f"    Per-joint (mm):")
            for j in range(7):
                print(f"      J{j} ({keypoint_names[j]:>15s}): "
                      f"mean={pj[:, j].mean():6.1f}, median={np.median(pj[:, j]):6.1f}, max={pj[:, j].max():7.1f}")

        metrics[f"{tier}_valid"] = n_valid
        metrics[f"{tier}_add_auc"] = auc
        metrics[f"{tier}_mean_mm"] = float(adds_mm.mean())
        metrics[f"{tier}_median_mm"] = float(np.median(adds_mm))
        metrics[f"{tier}_max_mm"] = float(adds_mm.max())
        metrics[f"{tier}_std_mm"] = float(adds_mm.std())

    print("\n" + "=" * 70)

    # Save
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "metrics_4tier.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(args.output_dir, "per_frame_errors.json"), "w") as f:
        json.dump(per_frame_results, f, indent=2)

    print(f"\nSaved to: {args.output_dir}/")
    print(f"  metrics_4tier.json     - aggregated metrics")
    print(f"  per_frame_errors.json  - per-frame ADD for each tier")

    if is_distributed:
        cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="4-Tier PnP Outlier Analysis")
    parser.add_argument("-p", "--model-path", required=True, help="Model checkpoint path")
    parser.add_argument("-d", "--dataset-dir", required=True, help="Dataset directory with JSON+images")
    parser.add_argument("-o", "--output-dir", default="./eval_4tier_output", help="Output directory")
    parser.add_argument("--model-name", default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--distributed", action="store_true", default=False, help="Enable torchrun-based distributed inference")
    parser.add_argument("--fix-joint7", action="store_true")
    parser.add_argument("--add-auc-threshold", type=float, default=0.1, help="ADD AUC threshold in meters")
    args = parser.parse_args()
    run_eval(args)
