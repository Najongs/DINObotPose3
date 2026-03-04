#!/bin/bash

# DINOv3 Heatmap-only Training Script
# 2D Keypoint Heatmap 여러 모델 순차 학습 스크립트

# =============================================================================
# Global Configuration
# =============================================================================

# GPU Settings
GPU_IDS="0,1,2,3,4"
NUM_GPUS=5
export CUDA_VISIBLE_DEVICES=${GPU_IDS}

# Common Data paths
DATA_DIRS=("/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/DREAM_syn/panda_synth_test_dr")
VAL_DIR="/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/Converted_dataset/DREAM_to_DREAM/panda-3cam_azure"
FDA_REAL_DIR="/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/DREAM_real/panda-3cam_azure"

# Model configuration
MODEL_NAME='facebook/dinov3-vitb16-pretrain-lvd1689m'
IMAGE_SIZE=512
HEATMAP_SIZE=512
UNFREEZE_BLOCKS=2

# Training hyperparameters (Base)
EPOCHS=50
BATCH_SIZE=16
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
        --fda-prob 0.5 \
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
run_finetune "" "0.0" "no_fda"

# 2. FDA BETA 0.01 모델 추가 학습
# run_finetune "" "0.01" "beta_0.01"

# 3. FDA BETA 0.001 모델 추가 학습
run_finetune "" "0.001" "beta_0.001"

# 4. FDA BETA 0.05 모델 추가 학습
# run_finetune "/home/najo/NAS/DIP/DINObotPose3/TRAIN/outputs_heatmap/finetune_beta_0.05_beta0.05_20260304_052019/best_heatmap.pth" "0.05" "beta_0.05"

echo "All scheduled training sessions have finished!"
