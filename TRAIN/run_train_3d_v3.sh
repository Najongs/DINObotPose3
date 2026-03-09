#!/bin/bash

# DINOv3 3D Pose Training Script v3 (Pure 3D Keypoint Prediction)

# =============================================================================
# Global Configuration
# =============================================================================

# GPU Settings
GPU_IDS="0,1,2"
NUM_GPUS=3
export CUDA_VISIBLE_DEVICES=${GPU_IDS}

# Data paths
TRAIN_DIR="/data/public/NAS/DINObotPose3/Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_train_dr"
VAL_DIR="/data/public/NAS/DINObotPose3/Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_test_dr"

# 2D Pretrained Checkpoint (required for fresh start)
CHECKPOINT="/data/public/NAS/DINObotPose3/TRAIN/outputs_heatmap/best_heatmap.pth"

# 3D Checkpoint
CHECKPOINT_3D=""

# Model configuration
MODEL_NAME='facebook/dinov3-vitb16-pretrain-lvd1689m'
IMAGE_SIZE=512
HEATMAP_SIZE=512

# Training hyperparameters
EPOCHS=100       # 3D coordinate regression might take longer
BATCH_SIZE=32    
NUM_WORKERS=4
LEARNING_RATE=1e-4   # Slightly higher LR for 3D coordinates learning
MIN_LR=1e-7      
WARMUP_STEPS=500     
GRAD_CLIP=1.0

# Loss weights 
BONE_LOSS_WEIGHT=1.0    # Bone length prior

# Validation settings
VAL_RATIO=0.3       # Validate on 30% for faster iteration due to IK overhead

# Sim-to-Real Augmentation
OCC_PROB=0.25
OCC_SIZE=0.2

# WANDB Settings
WANDB_PROJECT="dinov3-3d-pose-v3"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="train_3d_v3_pure3d_${TIMESTAMP}"
OUTPUT_DIR="./outputs_3d/train_3d_v3_${TIMESTAMP}"

# =============================================================================
# Execution
# =============================================================================

# Build checkpoint args
CKPT_ARGS=""
if [ -n "${CHECKPOINT_3D}" ] && [ -f "${CHECKPOINT_3D}" ]; then
    CKPT_ARGS="--checkpoint-3d ${CHECKPOINT_3D}"
    echo "==> Resuming from 3D checkpoint: ${CHECKPOINT_3D}"
else
    CKPT_ARGS="--checkpoint ${CHECKPOINT}"
    echo "==> Fresh start from 2D checkpoint: ${CHECKPOINT}"
fi

echo "============================================================================="
echo "==> STARTING 3D POSE TRAINING (V3: PURE 3D KEYPOINT PREDICTION)"
echo "==> Params: LR=${LEARNING_RATE}, Warmup=${WARMUP_STEPS}, Occ=${OCC_PROB}"
echo "==> Output: ${OUTPUT_DIR}"
echo "============================================================================="

torchrun --standalone --nnodes=1 --nproc_per_node=${NUM_GPUS} train_3d_v3.py \
    --train-dir "${TRAIN_DIR}" \
    --val-dir "${VAL_DIR}" \
    ${CKPT_ARGS} \
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
    --bone-loss-weight ${BONE_LOSS_WEIGHT} \
    --val-ratio ${VAL_RATIO} \
    --occlusion-prob ${OCC_PROB} \
    --occlusion-size ${OCC_SIZE} \
    --num-workers ${NUM_WORKERS} \
    --use-wandb \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-run-name "${RUN_NAME}"

echo "==> V3 Training Completed!"
