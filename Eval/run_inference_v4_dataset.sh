#!/bin/bash

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="/home/najo/NAS/DIP/DINObotPose3/TRAIN/outputs_3d_v4/train_3d_v4_20260310_174640/best_joint_angle.pth"
DATASET_DIR="/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/Converted_dataset/DREAM_to_DREAM/panda-3cam_azure"
OUTPUT_DIR="${SCRIPT_DIR}/eval_v4_dataset_results"

# Inference parameters
BATCH_SIZE=32
IMAGE_SIZE=512
HEATMAP_SIZE=512

echo "=========================================="
echo "  V4 Dataset Inference (Joint Angle)"
echo "=========================================="
echo "  Model:  $MODEL_PATH"
echo "  Data:   $DATASET_DIR"
echo "  Output: $OUTPUT_DIR"
echo "=========================================="

# Run multi-GPU inference
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Using $NUM_GPUS GPUs via torchrun..."

/home/najo/.conda/envs/dino/bin/torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29510 \
    "${SCRIPT_DIR}/inference_dataset_v4.py" \
    --model-path "$MODEL_PATH" \
    --dataset-dir "$DATASET_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size $BATCH_SIZE \
    --image-size $IMAGE_SIZE \
    --heatmap-size $HEATMAP_SIZE \
    --save-outliers \
    --outlier-topk 20

echo "Done."
