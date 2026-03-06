#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =============================================================================
# Configuration
# =============================================================================
MODEL_PATH="/data/public/NAS/DINObotPose3/TRAIN/outputs_3d/train_3d_20260306_064622/last_3d_pose.pth"

# Dataset - override with:
#   DATASET_DIR=/path/to/dataset ./run_inference_4tier.sh
DATASET_DIR="${DATASET_DIR:-/data/public/NAS/DINObotPose2/Dataset/Converted_dataset/DREAM_real/panda-3cam_azure}"
# DATASET_DIR="/data/public/NAS/DINObotPose2/Dataset/Converted_dataset/DREAM_real/panda-3cam_kinect360"
# DATASET_DIR="/data/public/NAS/DINObotPose2/Dataset/Converted_dataset/DREAM_real/panda-3cam_realsense"
# DATASET_DIR="/data/public/NAS/DINObotPose2/Dataset/Converted_dataset/DREAM_real/panda-orb"
# DATASET_DIR="/data/public/NAS/DINObotPose2/Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_test_dr"
# DATASET_DIR="/data/public/NAS/DINObotPose2/Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_test_photo"

# Output
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/eval_4tier_${TIMESTAMP}}"

# Inference settings
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"
ADD_AUC_THRESHOLD="${ADD_AUC_THRESHOLD:-0.1}"
FIX_JOINT7="${FIX_JOINT7:-1}"

# Execution mode
INFER_MODE="${INFER_MODE:-multi_gpu}"      # single_gpu | multi_gpu
GPU_IDS="${GPU_IDS:-0,1,2}"
NUM_GPUS="${NUM_GPUS:-$(awk -F',' '{print NF}' <<< "${GPU_IDS}")}"

if [ ! -f "${MODEL_PATH}" ]; then
    echo "Error: model checkpoint not found: ${MODEL_PATH}"
    exit 1
fi

if [ ! -d "${DATASET_DIR}" ]; then
    echo "Error: dataset directory not found: ${DATASET_DIR}"
    exit 1
fi

# =============================================================================
# Execution
# =============================================================================

echo "======================================================================"
echo "  4-Tier PnP Outlier Analysis"
echo "======================================================================"
echo "  Model:       ${MODEL_PATH}"
echo "  Dataset:     ${DATASET_DIR}"
echo "  Output:      ${OUTPUT_DIR}"
echo "  Mode:        ${INFER_MODE}"
echo "  GPU_IDS:     ${GPU_IDS}"
echo "  NUM_GPUS:    ${NUM_GPUS}"
echo "  Batch size:  ${BATCH_SIZE}"
echo "======================================================================"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

cd "${SCRIPT_DIR}"

if [ "${INFER_MODE}" = "single_gpu" ]; then
    python "${SCRIPT_DIR}/inference_4tier_eval.py" \
        --model-path "${MODEL_PATH}" \
        --dataset-dir "${DATASET_DIR}" \
        --output-dir "${OUTPUT_DIR}" \
        --model-name "facebook/dinov3-vitb16-pretrain-lvd1689m" \
        --image-size "${IMAGE_SIZE}" \
        --batch-size "${BATCH_SIZE}" \
        --num-workers "${NUM_WORKERS}" \
        --add-auc-threshold "${ADD_AUC_THRESHOLD}" \
        $( [[ "${FIX_JOINT7}" == "1" ]] && echo "--fix-joint7" )
elif [ "${INFER_MODE}" = "multi_gpu" ]; then
    torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node="${NUM_GPUS}" \
        "${SCRIPT_DIR}/inference_4tier_eval.py" \
        --distributed \
        --model-path "${MODEL_PATH}" \
        --dataset-dir "${DATASET_DIR}" \
        --output-dir "${OUTPUT_DIR}" \
        --model-name "facebook/dinov3-vitb16-pretrain-lvd1689m" \
        --image-size "${IMAGE_SIZE}" \
        --batch-size "${BATCH_SIZE}" \
        --num-workers "${NUM_WORKERS}" \
        --add-auc-threshold "${ADD_AUC_THRESHOLD}" \
        $( [[ "${FIX_JOINT7}" == "1" ]] && echo "--fix-joint7" )
else
    echo "Error: unknown INFER_MODE=${INFER_MODE}"
    exit 1
fi

echo ""
echo "======================================================================"
echo "  Done! Results in: ${OUTPUT_DIR}"
echo "    metrics_4tier.json     - aggregated metrics"
echo "    per_frame_errors.json  - per-frame ADD for each tier"
echo "======================================================================"
