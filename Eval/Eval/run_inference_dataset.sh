#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Dataset Inference Script for DINOv3 Pose Estimation
# Evaluates trained model on panda-3cam_azure dataset

# Model configuration (update with your trained model path)
MODEL_PATH="/data/public/NAS/DINObotPose2/Train/outputs/dinov3_base_20260228_161218/best_model.pth"

# Dataset (choose one of the following datasets)
DATASET_DIR="/data/public/NAS/DINObotPose2/Dataset/Converted_dataset/DREAM_to_DREAM/panda-3cam_azure"
# /home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/Converted_dataset/DREAM_to_DREAM/panda-3cam_azure
# /home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/Converted_dataset/DREAM_to_DREAM/panda-3cam_kinect360
# /home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/Converted_dataset/DREAM_to_DREAM/panda-3cam_realsense
# /home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/Converted_dataset/DREAM_to_DREAM/panda-orb
# Output
OUTPUT_DIR="${SCRIPT_DIR}/eval_outputs"

# Inference parameters
BATCH_SIZE=64
NUM_WORKERS=4

# Metrics thresholds (matching DREAM defaults)
KP_AUC_THRESHOLD=20.0    # pixels
ADD_AUC_THRESHOLD=0.1    # meters

# Execution mode
# INFER_MODE="single_gpu"
INFER_MODE="multi_gpu"
NUM_GPUS=3
GPU_IDS="0,1,2"

if [ "${INFER_MODE}" = "single_gpu" ]; then
    echo "Running single-GPU inference..."
    export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
    python "${SCRIPT_DIR}/inference_dataset.py" \
        --model-path "$MODEL_PATH" \
        --dataset-dir "$DATASET_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --batch-size $BATCH_SIZE \
        --num-workers $NUM_WORKERS \
        --kp-auc-threshold $KP_AUC_THRESHOLD \
        --add-auc-threshold $ADD_AUC_THRESHOLD
elif [ "${INFER_MODE}" = "multi_gpu" ]; then
    echo "Running distributed inference with ${NUM_GPUS} GPUs..."
    export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
    torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=${NUM_GPUS} \
        "${SCRIPT_DIR}/inference_dataset.py" \
        --distributed \
        --model-path "$MODEL_PATH" \
        --dataset-dir "$DATASET_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --batch-size $BATCH_SIZE \
        --num-workers $NUM_WORKERS \
        --kp-auc-threshold $KP_AUC_THRESHOLD \
        --add-auc-threshold $ADD_AUC_THRESHOLD
else
    echo "Error: Unknown INFER_MODE=${INFER_MODE}"
    exit 1
fi
