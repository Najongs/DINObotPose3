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
    MODEL_PATH="${PROJECT_DIR}/TRAIN/outputs_3d/train_3d_20260306_044138/best_3d_pose.pth"
    echo "WARNING: No 3D checkpoint found, using heatmap checkpoint"
fi

# Default test image (can override with $1)
JSON_PATH="${1:-/data/public/NAS/DINObotPose2/Dataset/Converted_dataset/DREAM_real/panda-3cam_azure/000000.json}"
OUTPUT_DIR="${2:-${SCRIPT_DIR}/vis_output}"

echo "=========================================="
echo "  Robot Mesh Overlay Visualization"
echo "=========================================="
echo "  Model:  $MODEL_PATH"
echo "  JSON:   $JSON_PATH"
echo "  Output: $OUTPUT_DIR"
echo "=========================================="

cd "$SCRIPT_DIR"

python render_overlay.py \
    --json-path "$JSON_PATH" \
    --model-path "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --model-name "facebook/dinov3-vitb16-pretrain-lvd1689m" \
    --fix-joint7

echo ""
echo "Visualization outputs:"
echo "  01_keypoints_skeleton.png     - 2D keypoints + skeleton"
echo "  02_mesh_overlay_pnp.png       - Robot mesh overlay (PnP-based)"
echo "  03_iterative_refinement.png   - Iterative refinement progression"
echo "  04_gt_vs_pred_comparison.png  - Side-by-side GT vs Pred"
echo "  05_metrics_summary.png        - Metrics panel"
