#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Model and dataset
MODEL_PATH="/home/najo/NAS/DIP/DINObotPose3/TRAIN/outputs/dinov3_base_20260303_163439_0.001/best_model.pth"
DATASET_DIR="/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/Converted_dataset/DREAM_to_DREAM/panda-3cam_azure"

# Output
OUTPUT_DIR="${SCRIPT_DIR}/eval_outputs_pck_only"

# Inference
BATCH_SIZE=64
NUM_WORKERS=4
FIX_JOINT7_ZERO=1
KP_MIN_CONFIDENCE=0.01
KP_MIN_PEAK_LOGIT=0.01

# Metrics
KP_AUC_THRESHOLD=20.0

# Execution mode
INFER_MODE="multi_gpu"
NUM_GPUS=5
GPU_IDS="0,1,2,3,4"

if [ "${INFER_MODE}" = "single_gpu" ]; then
    echo "Running single-GPU PCK-only inference..."
    export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
    python "${SCRIPT_DIR}/inference_dataset_pck_only.py" \
        --model-path "$MODEL_PATH" \
        --dataset-dir "$DATASET_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --batch-size $BATCH_SIZE \
        --num-workers $NUM_WORKERS \
        $( [[ "${FIX_JOINT7_ZERO}" == "1" ]] && echo "--fix-joint7-zero" ) \
        --kp-min-confidence "${KP_MIN_CONFIDENCE}" \
        --kp-min-peak-logit "${KP_MIN_PEAK_LOGIT}" \
        --kp-auc-threshold $KP_AUC_THRESHOLD \
        --save-metric-plots
elif [ "${INFER_MODE}" = "multi_gpu" ]; then
    echo "Running distributed PCK-only inference with ${NUM_GPUS} GPUs..."
    export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
    torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=${NUM_GPUS} \
        "${SCRIPT_DIR}/inference_dataset_pck_only.py" \
        --distributed \
        --model-path "$MODEL_PATH" \
        --dataset-dir "$DATASET_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --batch-size $BATCH_SIZE \
        --num-workers $NUM_WORKERS \
        $( [[ "${FIX_JOINT7_ZERO}" == "1" ]] && echo "--fix-joint7-zero" ) \
        --kp-min-confidence "${KP_MIN_CONFIDENCE}" \
        --kp-min-peak-logit "${KP_MIN_PEAK_LOGIT}" \
        --kp-auc-threshold $KP_AUC_THRESHOLD \
        --save-metric-plots
else
    echo "Error: Unknown INFER_MODE=${INFER_MODE}"
    exit 1
fi

echo "PCK-only inference completed."
echo "Check files in: ${OUTPUT_DIR}"
echo "  - eval_results_pck_only.json"
echo "  - auc_curve_pck_2d.png"

