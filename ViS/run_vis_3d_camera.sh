#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

LATEST_3D_DIR=$(ls -dt ${PROJECT_DIR}/TRAIN/outputs_3d/train_3d_* 2>/dev/null | head -1)
if [ -n "$LATEST_3D_DIR" ] && [ -f "$LATEST_3D_DIR/best_3d_pose.pth" ]; then
    MODEL_PATH="/data/public/NAS/DINObotPose3/TRAIN/outputs_3d/train_3d_20260306_064622/last_3d_pose.pth"
elif [ -n "$LATEST_3D_DIR" ] && [ -f "$LATEST_3D_DIR/last_3d_pose.pth" ]; then
    MODEL_PATH="$LATEST_3D_DIR/last_3d_pose.pth"
else
    MODEL_PATH="${PROJECT_DIR}/TRAIN/outputs_3d/train_3d_20260306_044138/best_3d_pose.pth"
    echo "WARNING: No latest 3D checkpoint found, using fallback"
fi

INPUT_PATH="${1:-/data/public/NAS/DINObotPose2/Dataset/Converted_dataset/DREAM_real/panda-3cam_azure}"
OUTPUT_DIR="${2:-${SCRIPT_DIR}/camera_3d_output}"
PRED_KEY="${3:-keypoints_3d_cam}"
BATCH_SIZE="${4:-32}"

echo "=========================================="
echo "  Camera-frame 3D Visualization"
echo "=========================================="
echo "  Model:  $MODEL_PATH"
echo "  Input:  $INPUT_PATH"
echo "  Output: $OUTPUT_DIR"
echo "  Pred:   $PRED_KEY"
echo "  Batch:  $BATCH_SIZE"
echo "=========================================="

cd "$SCRIPT_DIR"

if [ -d "$INPUT_PATH" ]; then
    python vis_3d_camera.py \
        --data-dir "$INPUT_PATH" \
        --model-path "$MODEL_PATH" \
        --output-dir "$OUTPUT_DIR" \
        --model-name "facebook/dinov3-vitb16-pretrain-lvd1689m" \
        --fix-joint7 \
        --pred-key "$PRED_KEY" \
        --batch-size "$BATCH_SIZE"
else
    python vis_3d_camera.py \
        --json-path "$INPUT_PATH" \
        --model-path "$MODEL_PATH" \
        --output-dir "$OUTPUT_DIR" \
        --model-name "facebook/dinov3-vitb16-pretrain-lvd1689m" \
        --fix-joint7 \
        --pred-key "$PRED_KEY"
fi

echo ""
echo "Camera-frame outputs:"
echo "  camera_3d_comparison*.png - best filtered sample from one batch, or direct JSON result"
