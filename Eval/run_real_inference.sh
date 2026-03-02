#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Model checkpoint
MODEL_PATH="/home/najo/NAS/DIP/DINObotPose2/Train/outputs/*dinov3_base_20260301_151520/*best_model_syn_2d.pth"

# Input annotation JSON (contains image path + GT keypoints + camera K)
JSON_PATH="/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/Converted_dataset/DREAM_to_DREAM/panda-3cam_azure/002138.json"

# Output directory
OUTPUT_DIR="${SCRIPT_DIR}/real_inference_output"

# Inference options
PRED_3D_SOURCE="fk_robopepp"
ROBOPEPP_FIX_JOINT7_ZERO=1
PNP_MODE="epnp"  # epnp (baseline) | loo_epnp (alternative) | ransac
PNP_TOPK=6
PNP_RANSAC_REPROJ_ERROR=5.0
KP_MIN_CONFIDENCE=0.1  # mask low-confidence 2D keypoints as invalid (-999)
KP_MIN_PEAK_LOGIT=0.1  # mask low-peak heatmap keypoints as invalid (-999)
PNP_REPROJ_OUTLIER_THRESH=12.0  # reject high-reprojection-error points and refit
PNP_MIN_SPAN_PX=20.0  # reject PnP if selected points are too concentrated
PNP_MIN_AREA_RATIO=0.001
FILL_INVALID_2D_WITH_FK_REPROJ=1  # fill low-reliability 2D with FK reprojection after successful PnP

echo "=========================================="
echo "  Real Image Inference (GT vs Prediction)"
echo "=========================================="
echo "  Model: ${MODEL_PATH}"
echo "  JSON:  ${JSON_PATH}"
echo "  Output: ${OUTPUT_DIR}"
echo "  PnP mode: ${PNP_MODE}"
echo "  Keypoint min confidence: ${KP_MIN_CONFIDENCE}"
echo "  Keypoint min peak logit: ${KP_MIN_PEAK_LOGIT}"
echo "  PnP reproj outlier threshold: ${PNP_REPROJ_OUTLIER_THRESH}px"
echo "  PnP min span: ${PNP_MIN_SPAN_PX}px, min area ratio: ${PNP_MIN_AREA_RATIO}"
echo "  RoboPEPP joint7=0: ${ROBOPEPP_FIX_JOINT7_ZERO}"
echo ""

if [[ "${JSON_PATH}" != *.json ]]; then
    echo "Error: JSON_PATH must point to a .json annotation file, got: ${JSON_PATH}"
    exit 1
fi

python "${SCRIPT_DIR}/inference_with_real.py" \
    -j "${JSON_PATH}" \
    -p "${MODEL_PATH}" \
    -o "${OUTPUT_DIR}" \
    --pred-3d-source "${PRED_3D_SOURCE}" \
    $( [[ "${ROBOPEPP_FIX_JOINT7_ZERO}" == "1" ]] && echo "--robopepp-fix-joint7-zero" ) \
    --pnp-mode "${PNP_MODE}" \
    --pnp-topk "${PNP_TOPK}" \
    --pnp-ransac-reproj-error "${PNP_RANSAC_REPROJ_ERROR}" \
    --pnp-reproj-outlier-thresh "${PNP_REPROJ_OUTLIER_THRESH}" \
    --pnp-min-span-px "${PNP_MIN_SPAN_PX}" \
    --pnp-min-area-ratio "${PNP_MIN_AREA_RATIO}" \
    --kp-min-peak-logit "${KP_MIN_PEAK_LOGIT}" \
    --kp-min-confidence "${KP_MIN_CONFIDENCE}" \
    $( [[ "${FILL_INVALID_2D_WITH_FK_REPROJ}" == "1" ]] && echo "--fill-invalid-2d-with-fk-reproj" )

echo ""
echo "Results saved to: ${OUTPUT_DIR}"
echo "  01_gt_vs_pred_keypoints.png  - Green=GT, Red=Prediction"
echo "  02_belief_map_mosaic.png     - Per-joint belief maps"
echo "  03_belief_maps_overlay_mosaic.png - Belief maps on image"
echo "  04_combined_on_original.png  - Combined heatmap on original"
echo "  metrics.json                 - Per-keypoint error metrics"
