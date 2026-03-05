#!/bin/bash

# DINOv3 Heatmap-only Training Script
# 2D Keypoint Heatmap 여러 모델 순차 학습 스크립트

# =============================================================================
# Global Configuration
# =============================================================================

# GPU Settings
GPU_IDS="0,1,2"
NUM_GPUS=3
export CUDA_VISIBLE_DEVICES=${GPU_IDS}

# Common Data paths
DATA_DIRS=("/data/public/NAS/DINObotPose2/Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_train_dr")
VAL_DIR="/data/public/NAS/DINObotPose2/Dataset/Converted_dataset/DREAM_to_DREAM/panda-3cam_azure"
FDA_REAL_DIR="/data/public/NAS/DINObotPose2/Dataset/DREAM_real"

# Model configuration
MODEL_NAME='facebook/dinov3-vitb16-pretrain-lvd1689m'
IMAGE_SIZE=512
HEATMAP_SIZE=512
UNFREEZE_BLOCKS=2

# Training hyperparameters (Base)
EPOCHS=100
BATCH_SIZE=32
NUM_WORKERS=4
LEARNING_RATE=1e-4
MIN_LR=1e-10
WEIGHT_DECAY=1e-5
WANDB_PROJECT="dinov3-heatmap-only"

# =============================================================================
# Training Function
# =============================================================================

run_finetune() {
    local CKPT=$1
    local FDA_BETA=$2
    local TAG=$3
    
    # 실행 시점의 시간을 반영하여 고유 경로 생성
    local TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    local CURRENT_OUT_DIR="./outputs_heatmap/finetune_${TAG}_beta${FDA_BETA}_${TIMESTAMP}"
    local CURRENT_RUN_NAME="finetune_${TAG}_beta${FDA_BETA}_${TIMESTAMP}"

    echo "============================================================================="
    echo "==> STARTING: ${TAG} (Beta: ${FDA_BETA})"
    echo "==> Checkpoint: ${CKPT}"
    echo "==> Output: ${CURRENT_OUT_DIR}"
    echo "============================================================================="

    torchrun --standalone --nnodes=1 --nproc_per_node=${NUM_GPUS} train_heatmap.py \
        --data-dir ${DATA_DIRS[*]} \
        --val-dir ${VAL_DIR} \
        --checkpoint "${CKPT}" \
        --model-name ${MODEL_NAME} \
        --output-dir "${CURRENT_OUT_DIR}" \
        --image-size ${IMAGE_SIZE} \
        --heatmap-size ${HEATMAP_SIZE} \
        --unfreeze-blocks ${UNFREEZE_BLOCKS} \
        --epochs ${EPOCHS} \
        --batch-size ${BATCH_SIZE} \
        --num-workers ${NUM_WORKERS} \
        --learning-rate ${LEARNING_RATE} \
        --min-lr ${MIN_LR} \
        --weight-decay ${WEIGHT_DECAY} \
        --fda-real-dir ${FDA_REAL_DIR} \
        --fda-prob 0.25 \
        --fda-beta ${FDA_BETA} \
        --wandb-project ${WANDB_PROJECT} \
        --wandb-run-name "${CURRENT_RUN_NAME}" \
        --no-augment

    echo "==> COMPLETED: ${TAG}"
    echo ""
}

# =============================================================================
# Execution Queue
# =============================================================================

# 1. FDA BETA 0.0 모델 추가 학습
# run_finetune "" "0.0" "no_fda"

# 2. FDA BETA 0.01 모델 추가 학습
# run_finetune "/home/najo/NAS/DIP/DINObotPose3/TRAIN/outputs_heatmap/finetune_beta_0.01_beta0.01_20260304_025505/best_heatmap.pth" "0.01" "beta_0.01"

# 3. FDA BETA 0.001 모델 추가 학습
run_finetune "/data/public/NAS/DINObotPose3/TRAIN/outputs_heatmap/finetune_beta_0.001_beta0.001_20260304_163045/best_heatmap.pth" "0.001" "beta_0.001"

# 4. FDA BETA 0.05 모델 추가 학습
# run_finetune "/home/najo/NAS/DIP/DINObotPose3/TRAIN/outputs_heatmap/finetune_beta_0.05_beta0.05_20260304_052019/best_heatmap.pth" "0.05" "beta_0.05"

echo "All scheduled training sessions have finished!"
