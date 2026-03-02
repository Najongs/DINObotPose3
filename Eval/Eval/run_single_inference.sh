#!/bin/bash

# Single Image Inference Script for DINOv3 Pose Estimation
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Model configuration
MODEL_PATH="/data/public/NAS/DINObotPose2/Train/outputs/dinov3_base_20260301_045023/best_model.pth"

# Input (set one)
# IMAGE_PATH="/data/public/NAS/DINObotPose2/Eval/zed_41182735_left_1756275914.348.jpg"
JSON_PATH="/data/public/NAS/DINObotPose2/Dataset/Converted_dataset/DREAM_to_DREAM/panda-3cam_azure/002138.json"  # e.g. /data/public/NAS/DINObotPose2/Dataset/.../000123.json

# Output directory
OUTPUT_DIR="${SCRIPT_DIR}/inference_output"
KP_MIN_CONFIDENCE=0.01
KP_MIN_PEAK_LOGIT=0.10
PRED_3D_SOURCE="fk"
PNP_MIN_SPAN_PX=20.0
PNP_MIN_AREA_RATIO=0.001
FILL_INVALID_2D_WITH_FK_REPROJ=1

# Run inference (model-name, image-size, heatmap-size are read from config.yaml automatically)
if [[ -n "${JSON_PATH}" ]]; then
    python "${SCRIPT_DIR}/inference_single_image.py" \
        --model-path "$MODEL_PATH" \
        --json-path "$JSON_PATH" \
        --output-dir "$OUTPUT_DIR" \
        --pred-3d-source "${PRED_3D_SOURCE}" \
        --kp-min-confidence "${KP_MIN_CONFIDENCE}" \
        --kp-min-peak-logit "${KP_MIN_PEAK_LOGIT}" \
        --pnp-min-span-px "${PNP_MIN_SPAN_PX}" \
        --pnp-min-area-ratio "${PNP_MIN_AREA_RATIO}" \
        $( [[ "${FILL_INVALID_2D_WITH_FK_REPROJ}" == "1" ]] && echo "--fill-invalid-2d-with-fk-reproj" ) \
        --save-heatmaps \
        --save-combined
else
    python "${SCRIPT_DIR}/inference_single_image.py" \
        --model-path "$MODEL_PATH" \
        --image-path "$IMAGE_PATH" \
        --output-dir "$OUTPUT_DIR" \
        --pred-3d-source "${PRED_3D_SOURCE}" \
        --kp-min-confidence "${KP_MIN_CONFIDENCE}" \
        --kp-min-peak-logit "${KP_MIN_PEAK_LOGIT}" \
        --pnp-min-span-px "${PNP_MIN_SPAN_PX}" \
        --pnp-min-area-ratio "${PNP_MIN_AREA_RATIO}" \
        $( [[ "${FILL_INVALID_2D_WITH_FK_REPROJ}" == "1" ]] && echo "--fill-invalid-2d-with-fk-reproj" ) \
        --save-heatmaps \
        --save-combined
fi

echo ""
echo "Inference completed! Check results in: $OUTPUT_DIR"
