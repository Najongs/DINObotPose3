#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Model checkpoint (use latest 3D training output)
# Find the most recent outputs_3d directory
LATEST_3D_DIR=$(ls -dt ${SCRIPT_DIR}/../TRAIN/outputs_3d/train_3d_* 2>/dev/null | head -1)

if [ -n "$LATEST_3D_DIR" ] && [ -f "$LATEST_3D_DIR/best_3d_pose.pth" ]; then
    MODEL_PATH="$LATEST_3D_DIR/best_3d_pose.pth"
elif [ -n "$LATEST_3D_DIR" ] && [ -f "$LATEST_3D_DIR/last_3d_pose.pth" ]; then
    MODEL_PATH="$LATEST_3D_DIR/last_3d_pose.pth"
else
    # Fallback to heatmap-only checkpoint
    MODEL_PATH="/data/public/NAS/DINObotPose3/TRAIN/outputs_3d/train_3d_20260306_062016/best_3d_pose.pth"
    echo "WARNING: No 3D checkpoint found, falling back to heatmap checkpoint"
fi

JSON_PATH="/data/public/NAS/DINObotPose2/Dataset/Converted_dataset/DREAM_real/panda-3cam_azure/001569.json"
OUTPUT_DIR="./real_inference_output"

echo "=========================================="
echo "  Real Image Inference (3D Pose)"
echo "=========================================="
echo "  Model:  $MODEL_PATH"
echo "  JSON:   $JSON_PATH"
echo "  Output: $OUTPUT_DIR"
echo "=========================================="

cd "$SCRIPT_DIR"

python inference_with_real.py \
    --json-path "$JSON_PATH" \
    --model-path "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --model-name "facebook/dinov3-vitb16-pretrain-lvd1689m" \
    --fix-joint7

echo ""
echo "Results saved to: $OUTPUT_DIR"
echo "  inference_overlay.png  - GT (Green) vs Prediction (Red)"
echo "  metrics.json           - Quantitative results"
