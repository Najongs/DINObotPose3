#!/bin/bash

# DINOv3 3D Pose Training Script (Direct 3D Former Mode)
# 🚀 Direct 3D Head: Features + Heatmaps → Robot-frame 3D Keypoints
# No joint angle prediction, purely regression-based 3D coordinates

# =============================================================================
# Global Configuration
# =============================================================================

# GPU Settings
GPU_IDS="0,1,2"
NUM_GPUS=3
export CUDA_VISIBLE_DEVICES=${GPU_IDS}

# Data paths
TRAIN_DIR="/data/public/NAS/DINObotPose2/Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_train_dr"
VAL_DIR="/data/public/NAS/DINObotPose2/Dataset/Converted_dataset/DREAM_real/panda-3cam_azure"

# 2D Pretrained Checkpoint (required)
CHECKPOINT="/data/public/NAS/DINObotPose3/TRAIN/outputs_heatmap/best_heatmap.pth"

# Model configuration
MODEL_NAME='facebook/dinov3-vitb16-pretrain-lvd1689m'
IMAGE_SIZE=512
HEATMAP_SIZE=512

# Training hyperparameters
EPOCHS=50
BATCH_SIZE=16
NUM_WORKERS=4
LEARNING_RATE=1e-4
MIN_LR=1e-7
WARMUP_STEPS=500
GRAD_CLIP=1.0

# Loss weights - 🚀 For Direct 3D mode
KP_WEIGHT=100.0         # 3D keypoint regression weight

# Sim-to-Real Augmentation (🚀 강화됨)
OCC_PROB=0.6            # 강력한 occlusion (60% 확률)
OCC_SIZE=0.4            # 더 큰 occlusion 패치

# WANDB Settings
WANDB_PROJECT="dinov3-3d-pose"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="train_3d_direct_${TIMESTAMP}"
OUTPUT_DIR="./outputs_3d/train_3d_direct_${TIMESTAMP}"

# =============================================================================
# Execution
# =============================================================================

echo "============================================================================="
echo "==> STARTING 3D POSE TRAINING (DIRECT 3D FORMER MODE)"
echo "==> Direct 3D Head: Features + Heatmaps → 3D Keypoints (no angles)"
echo "==> 2D Checkpoint: ${CHECKPOINT}"
echo "==> Params: LR=${LEARNING_RATE}, Warmup=${WARMUP_STEPS}, KP_Weight=${KP_WEIGHT}"
echo "==> Output: ${OUTPUT_DIR}"
echo "============================================================================="

torchrun --standalone --nnodes=1 --nproc_per_node=${NUM_GPUS} train_3d.py \
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
    --min-lr ${MIN_LR} \
    --warmup-steps ${WARMUP_STEPS} \
    --grad-clip ${GRAD_CLIP} \
    --kp-weight ${KP_WEIGHT} \
    --occlusion-prob ${OCC_PROB} \
    --occlusion-size ${OCC_SIZE} \
    --num-workers ${NUM_WORKERS} \
    --mode direct_3d \
    --use-wandb \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-run-name "${RUN_NAME}"

echo "==> 3D Training (Direct Mode) Completed!"
