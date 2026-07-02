#!/bin/bash

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="/home/najo/NAS/DIP/DINObotPose3/TRAIN/outputs_3d_v4/train_3d_v4_20260310_174640/best_joint_angle.pth"
INPUT_PATH="/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/Converted_dataset/DREAM_to_DREAM/panda-3cam_azure/000652.json"
OUTPUT_DIR="${SCRIPT_DIR}/inference_v4_viz"

# Inference parameters
IMAGE_SIZE=512
HEATMAP_SIZE=512

echo "=========================================="
echo "  V4 Single/Folder Inference (Viz)"
echo "=========================================="
echo "  Model:  $MODEL_PATH"
echo "  Input:  $INPUT_PATH"
echo "  Output: $OUTPUT_DIR"
echo "=========================================="

# Run inference
/home/najo/.conda/envs/dino/bin/python "${SCRIPT_DIR}/inference_single_v4.py" \
    --model-path "$MODEL_PATH" \
    --input "$INPUT_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --image-size $IMAGE_SIZE \
    --heatmap-size $HEATMAP_SIZE

echo "Done. Results in $OUTPUT_DIR"
