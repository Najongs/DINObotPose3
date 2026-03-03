#!/bin/bash

# DINOv3 Heatmap-only Training Script
# 2D Keypoint Heatmap 학습 전용 실행 스크립트

# =============================================================================
# Configuration
# =============================================================================

# Data paths
# DATA_DIRS=("/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_train_dr")
DATA_DIRS=("/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_test_dr")
VAL_DIR="/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/Converted_dataset/DREAM_to_DREAM/panda-3cam_azure"

TRAIN_SPLIT=1.0
VAL_SPLIT=0.2

# Model configuration
MODEL_NAME='facebook/dinov3-vitb16-pretrain-lvd1689m'
IMAGE_SIZE=512
HEATMAP_SIZE=512
UNFREEZE_BLOCKS=2

# Training hyperparameters
EPOCHS=50
BATCH_SIZE=16
NUM_WORKERS=4
LEARNING_RATE=1e-4
MIN_LR=1e-10
WEIGHT_DECAY=1e-5

# FDA (Sim-to-Real)
FDA_REAL_DIR="/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/DREAM_real/panda-3cam_azure"
FDA_PROB=0.5
FDA_BETA=0.05

# GPU Settings
GPU_IDS="0,1,2,3,4"
NUM_GPUS=5

# Output and Logging
OUTPUT_DIR="./outputs_heatmap/heatmap_only_$(date +%Y%m%d_%H%M%S)"
WANDB_PROJECT="dinov3-heatmap-only"
WANDB_RUN_NAME="heatmap_only_$(date +%Y%m%d_%H%M%S)"

# =============================================================================
# Execute Training
# =============================================================================

export CUDA_VISIBLE_DEVICES=${GPU_IDS}

echo "Starting Heatmap-only training on GPUs: ${GPU_IDS}"

torchrun --standalone --nnodes=1 --nproc_per_node=${NUM_GPUS} train_heatmap.py \
    --data-dir ${DATA_DIRS[*]} \
    --val-dir ${VAL_DIR} \
    --val-split ${VAL_SPLIT} \
    --model-name ${MODEL_NAME} \
    --output-dir ${OUTPUT_DIR} \
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
    --fda-prob ${FDA_PROB} \
    --fda-beta ${FDA_BETA} \
    --wandb-project ${WANDB_PROJECT} \
    --wandb-run-name ${WANDB_RUN_NAME}

echo "Training completed! Results saved to: ${OUTPUT_DIR}"
