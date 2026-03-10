#!/bin/bash

# DINOv3 End-to-End Pose Training Script
# 모든 파라미터를 해제(Unfreeze)하고 2D+3D 통합 파인튜닝을 진행하는 스크립트

# =============================================================================
# Global Configuration
# =============================================================================

# GPU Settings
GPU_IDS="0,1,2,3,4"
NUM_GPUS=5
export CUDA_VISIBLE_DEVICES=${GPU_IDS}

# Data paths
TRAIN_DIR="/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_train_dr"
VAL_DIR="/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/Converted_dataset/DREAM_to_DREAM/panda-3cam_azure"

# 1st Stage Checkpoint (필수)
# 1단계에서 3D Head가 안정화된 'best_3d_pose.pth' 경로를 입력하세요.
CHECKPOINT="/home/najo/NAS/DIP/DINObotPose3/TRAIN/outputs_3d/train_3d_XXXXXXXX_XXXXXX/best_3d_pose.pth"

# Model configuration
MODEL_NAME='facebook/dinov3-vitb16-pretrain-lvd1689m'
IMAGE_SIZE=512
HEATMAP_SIZE=512

# Training hyperparameters (E2E는 훨씬 낮은 LR 사용)
EPOCHS=20
BATCH_SIZE=8  # 메모리 부족 시 조절
NUM_WORKERS=4
LEARNING_RATE=1e-5  # 1st stage보다 10배 낮게 설정
HEATMAP_WEIGHT=1000.0
ANGLE_WEIGHT=10.0
CAMERA_3D_WEIGHT=100.0

# WANDB Settings
WANDB_PROJECT="dinov3-e2e-pose"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="train_e2e_${TIMESTAMP}"
OUTPUT_DIR="./outputs_e2e/train_e2e_${TIMESTAMP}"

# =============================================================================
# Execution
# =============================================================================

echo "============================================================================="
echo "==> STARTING END-TO-END POSE TRAINING (STAGE 2)"
echo "==> Stage 1 Checkpoint: ${CHECKPOINT}"
echo "==> Learning Rate: ${LEARNING_RATE}"
echo "==> Output: ${OUTPUT_DIR}"
echo "============================================================================="

torchrun --standalone --nnodes=1 --nproc_per_node=${NUM_GPUS} train_e2e.py \
    --train-dir "${TRAIN_DIR}" \
    --val-dir "${VAL_DIR}" \
    --checkpoint "${CHECKPOINT}" \
    --model-name "${MODEL_NAME}" \
    --output-dir "${OUTPUT_DIR}" \
    --image-size ${IMAGE_SIZE} \
    --heatmap-size ${HEATMAP_SIZE} \
    --batch-size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LEARNING_RATE} \
    --heatmap-weight ${HEATMAP_WEIGHT} \
    --angle-weight ${ANGLE_WEIGHT} \
    --camera-3d-weight ${CAMERA_3D_WEIGHT} \
    --num-workers ${NUM_WORKERS} \
    --use-wandb \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-run-name "${RUN_NAME}"

echo "==> E2E Training Completed!"
