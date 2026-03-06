#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Find latest 3D checkpoint
LATEST_3D_DIR=$(ls -dt ${PROJECT_DIR}/TRAIN/outputs_3d/train_3d_* 2>/dev/null | head -1)
if [ -n "$LATEST_3D_DIR" ] && [ -f "$LATEST_3D_DIR/best_3d_pose.pth" ]; then
    MODEL_PATH="$LATEST_3D_DIR/best_3d_pose.pth"
elif [ -n "$LATEST_3D_DIR" ] && [ -f "$LATEST_3D_DIR/last_3d_pose.pth" ]; then
    MODEL_PATH="$LATEST_3D_DIR/last_3d_pose.pth"
else
    MODEL_PATH="${PROJECT_DIR}/TRAIN/outputs_heatmap/best_3d_pose.pth"
    echo "WARNING: No latest 3D checkpoint found, using fallback"
fi

# Default test image (can override with $1)
JSON_PATH="${1:-/data/public/NAS/DINObotPose2/Dataset/Converted_dataset/franka_research3_to_DREAM_modified/zed_49045152_left_1756282421.718.json}"
OUTPUT_DIR="${2:-${SCRIPT_DIR}/heatmap_output}"

# /data/public/NAS/DINObotPose2/Dataset/Converted_dataset/franka_research3_to_DREAM_modified/zed_49045152_left_1756282421.718.json 음 가려짐 굳 좋은데? 맛있다.
# /data/public/NAS/DINObotPose2/Dataset/Converted_dataset/franka_research3_to_DREAM_modified/zed_49429257_right_1756282411.169.json # 약간 가렸는데 잘 맞춤
# /data/public/NAS/DINObotPose2/Dataset/Converted_dataset/franka_research3_to_DREAM_modified/zed_49429257_left_1756279858.913.json 정자세 굳 이쁨 
# /data/public/NAS/DINObotPose2/Dataset/Converted_dataset/franka_research3_to_DREAM_modified/zed_49045152_left_1756280399.634.json 옆 자세 굳 

echo "=========================================="
echo "  Heatmap Visualization (GT vs Pred)"
echo "=========================================="
echo "  Model:  $MODEL_PATH"
echo "  JSON:   $JSON_PATH"
echo "  Output: $OUTPUT_DIR"
echo "=========================================="

cd "$SCRIPT_DIR"

python vis_heatmap.py \
    --json-path "$JSON_PATH" \
    --model-path "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --model-name "facebook/dinov3-vitb16-pretrain-lvd1689m" \
    --fix-joint7 \
    --sigma 5.0 \
    --thumb-size 256

echo ""
echo "Heatmap outputs:"
echo "  heatmap_per_joint.png  - Per-joint GT(left) vs Pred(right) grid"
echo "  heatmap_combined.png   - All joints combined GT(left) vs Pred(right)"
echo "  heatmap_strip.png      - Per-joint heatmap strip (no image overlay)"
echo "  pred_2d_skeleton_overlay.png - 2D GT(green) vs Pred(red) keypoints + skeleton"
