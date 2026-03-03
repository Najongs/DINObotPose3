"""
PCK-only dataset inference script for DINOv3 pose estimator.

This script evaluates only 2D keypoint quality:
- L2 pixel error stats (mean/median/std)
- PCK@{2.5, 5, 10}px
- AUC over keypoint L2 threshold
"""

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from inference_dataset import (
    DINOv3PoseEstimator,
    InferenceDataset,
    cleanup_distributed,
    collect_keypoint_l2_errors,
    compute_keypoint_metrics,
    get_keypoints_from_heatmaps,
    load_camera_from_first_frame,
    load_checkpoint_compat,
    setup_distributed,
)


def save_pck_plot(
    output_dir: Path,
    kp_errors: np.ndarray,
    kp_denom: int,
    kp_auc_threshold: float,
) -> Path:
    delta = 0.01
    xs = np.arange(0.0, kp_auc_threshold, delta)
    if kp_denom <= 0:
        ys = np.zeros_like(xs)
    else:
        ys = np.array([float(np.sum(kp_errors <= x)) / float(kp_denom) for x in xs], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys * 100.0, linewidth=2.5, color="#0072B2", label="PCK")
    ax.set_title("PCK Curve (2D)")
    ax.set_xlabel("Pixel threshold")
    ax.set_ylabel("Success rate (%)")
    ax.set_xlim(0.0, kp_auc_threshold)
    ax.set_ylim(0.0, 100.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()

    out_path = output_dir / "auc_curve_pck_2d.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


@torch.no_grad()
def run_inference(args):
    is_distributed, rank, world_size, local_rank = setup_distributed(args.distributed)
    is_main_process = (not is_distributed) or (rank == 0)

    dataset_dir = Path(args.dataset_dir)
    _, image_resolution = load_camera_from_first_frame(dataset_dir)

    checkpoint_dir = Path(args.model_path).parent
    config_path = checkpoint_dir / "config.yaml"
    train_config = {}

    keypoint_names: List[str] = [
        "panda_link0", "panda_link2", "panda_link3",
        "panda_link4", "panda_link6", "panda_link7", "panda_hand"
    ]

    if config_path.exists():
        import yaml
        with open(config_path, "r") as f:
            train_config = yaml.safe_load(f)
        if "keypoint_names" in train_config:
            keypoint_names = train_config["keypoint_names"]

    model_name = args.model_name or train_config.get("model_name", "facebook/dinov3-vitb16-pretrain-lvd1689m")
    image_size = args.image_size or int(train_config.get("image_size", 512))
    heatmap_size = args.heatmap_size or int(train_config.get("heatmap_size", 512))
    use_joint_embedding = bool(train_config.get("use_joint_embedding", False))
    fix_joint7_zero = args.fix_joint7_zero or bool(train_config.get("fix_joint7_zero", False))

    if is_main_process:
        print(f"Model path: {args.model_path}")
        print(f"Dataset dir: {args.dataset_dir}")
        print(f"model_name={model_name}, image_size={image_size}, heatmap_size={heatmap_size}")
        print(f"use_joint_embedding={use_joint_embedding}, fix_joint7_zero={fix_joint7_zero}")

    dataset = InferenceDataset(
        data_dir=args.dataset_dir,
        keypoint_names=keypoint_names,
        image_size=(image_size, image_size),
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

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}" if is_distributed else "cuda")
    else:
        device = torch.device("cpu")

    model = DINOv3PoseEstimator(
        dino_model_name=model_name,
        heatmap_size=(heatmap_size, heatmap_size),
        unfreeze_blocks=0,
        use_joint_embedding=use_joint_embedding,
        use_iterative_refinement=False,
        refinement_iterations=0,
        fix_joint7_zero=fix_joint7_zero,
    ).to(device)

    load_checkpoint_compat(
        model=model,
        checkpoint_path=args.model_path,
        device=device,
        is_main_process=is_main_process,
    )
    model.eval()

    all_kp_detected = []
    all_kp_gt = []
    all_n_inframe = []

    for batch in tqdm(dataloader, disable=not is_main_process):
        images = batch["image"].to(device)
        gt_keypoints = batch["keypoints"].numpy()
        batch_original_size = batch["original_size"].numpy()
        batch_camera_K = batch["camera_K"].to(device)
        batch_original_size_t = torch.from_numpy(batch_original_size).to(device)

        outputs = model(
            images,
            camera_K=batch_camera_K,
            original_size=batch_original_size_t,
            use_refinement=False,
        )
        pred_heatmaps = outputs["heatmaps_2d"]

        pred_keypoints, _ = get_keypoints_from_heatmaps(
            pred_heatmaps,
            min_confidence=args.kp_min_confidence,
            min_peak_logit=args.kp_min_peak_logit,
        )

        for i in range(len(pred_keypoints)):
            raw_w, raw_h = batch_original_size[i].astype(int)
            scale_x = raw_w / heatmap_size
            scale_y = raw_h / heatmap_size

            pred_kp_scaled = pred_keypoints[i].copy()
            pred_kp_scaled[:, 0] *= scale_x
            pred_kp_scaled[:, 1] *= scale_y

            all_kp_detected.append(pred_kp_scaled)
            all_kp_gt.append(gt_keypoints[i])

            n_inframe = 0
            for kp in gt_keypoints[i]:
                if 0 <= kp[0] <= raw_w and 0 <= kp[1] <= raw_h:
                    n_inframe += 1
            all_n_inframe.append(n_inframe)

    local_payload = {
        "kp_detected": all_kp_detected,
        "kp_gt": all_kp_gt,
        "n_inframe": all_n_inframe,
    }

    if is_distributed:
        gathered_payloads = [None for _ in range(world_size)] if is_main_process else None
        torch.distributed.gather_object(local_payload, gathered_payloads, dst=0)
        torch.distributed.barrier()
        if not is_main_process:
            cleanup_distributed()
            return

        all_kp_detected = []
        all_kp_gt = []
        all_n_inframe = []
        for payload in gathered_payloads:
            if payload is None:
                continue
            all_kp_detected.extend(payload["kp_detected"])
            all_kp_gt.extend(payload["kp_gt"])
            all_n_inframe.extend(payload["n_inframe"])

    all_kp_detected = np.array(all_kp_detected)
    all_kp_gt = np.array(all_kp_gt)
    n_samples = len(all_kp_detected)

    if n_samples == 0:
        raise RuntimeError("No samples processed.")

    kp_metrics = compute_keypoint_metrics(
        all_kp_detected.reshape(n_samples * len(keypoint_names), 2),
        all_kp_gt.reshape(n_samples * len(keypoint_names), 2),
        image_resolution,
        auc_threshold=args.kp_auc_threshold,
    )

    if is_main_process:
        print("\n" + "=" * 80)
        print("PCK-ONLY EVALUATION RESULTS")
        print("=" * 80)
        print(f"Dataset: {args.dataset_dir}")
        print(f"Model: {args.model_path}")
        print(f"Frames: {n_samples}")
        print(f"In-frame GT keypoints: {kp_metrics['num_gt_inframe']}")
        if kp_metrics["l2_error_auc"] is not None:
            print(f"AUC@{kp_metrics['l2_error_auc_thresh_px']:.1f}px: {kp_metrics['l2_error_auc']:.5f}")
            print(f"Mean L2(px): {kp_metrics['l2_error_mean_px']:.5f}")
            print(f"Median L2(px): {kp_metrics['l2_error_median_px']:.5f}")
            print(f"Std L2(px): {kp_metrics['l2_error_std_px']:.5f}")
            print(f"PCK@2.5px: {kp_metrics['pck@2.5px_percent']:.2f}%")
            print(f"PCK@5px: {kp_metrics['pck@5px_percent']:.2f}%")
            print(f"PCK@10px: {kp_metrics['pck@10px_percent']:.2f}%")
        else:
            print("No valid keypoints found.")
        print("=" * 80)

    if args.output_dir and is_main_process:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        kp_errors, kp_denom = collect_keypoint_l2_errors(
            all_kp_detected.reshape(n_samples * len(keypoint_names), 2),
            all_kp_gt.reshape(n_samples * len(keypoint_names), 2),
            image_resolution,
        )

        plot_path = None
        if args.save_metric_plots:
            plot_path = save_pck_plot(
                output_dir=output_dir,
                kp_errors=kp_errors,
                kp_denom=kp_denom,
                kp_auc_threshold=args.kp_auc_threshold,
            )

        results = {
            "dataset": str(args.dataset_dir),
            "model": str(args.model_path),
            "num_frames": int(n_samples),
            "keypoint_metrics": {k: float(v) if isinstance(v, (int, float, np.number)) else v for k, v in kp_metrics.items()},
            "plot_path": str(plot_path) if plot_path is not None else None,
        }

        results_path = output_dir / "eval_results_pck_only.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Saved: {results_path}")
        if plot_path is not None:
            print(f"Saved: {plot_path}")

    if is_distributed:
        cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description="PCK-only inference on dataset")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--model-name", type=str, default=None, help="Override model name")
    parser.add_argument("--image-size", type=int, default=None, help="Override input image size")
    parser.add_argument("--heatmap-size", type=int, default=None, help="Override output heatmap size")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Path to NDDS dataset directory")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--distributed", action="store_true", default=False, help="Enable torchrun-based distributed inference")
    parser.add_argument("--fix-joint7-zero", action="store_true", default=False, help="Force joint7=0 during model forward")
    parser.add_argument("--kp-auc-threshold", type=float, default=20.0, help="AUC threshold for 2D keypoint L2 (px)")
    parser.add_argument("--kp-min-confidence", type=float, default=0.0, help="Min sigmoid(max_heatmap_logit) for valid keypoint")
    parser.add_argument("--kp-min-peak-logit", type=float, default=-1e9, help="Min heatmap peak logit for valid keypoint")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save metrics/plots")
    parser.add_argument("--save-metric-plots", action="store_true", default=False, help="Save PCK curve plot")
    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
