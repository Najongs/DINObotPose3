#!/bin/bash

# DINOv3 Joint Angle Training v2
# Key changes from v1:
#   1. Backbone unfreeze (last 2 blocks) after warmup
#   2. Heatmap head unfreeze (regularizer)
#   3. Direct angle prediction (no sin/cos)
#   4. Progressive heatmap loss (RoboPEPP style)

GPU_IDS="0,1,2"
NUM_GPUS=3
export CUDA_VISIBLE_DEVICES=${GPU_IDS}

TRAIN_DIR="/data/public/NAS/DINObotPose2/Dataset/Converted_dataset/DREAM_real"
VAL_DIR="/data/public/NAS/DINObotPose2/Dataset/Converted_dataset/DREAM_real/panda-3cam_azure"
CHECKPOINT="/data/public/NAS/DINObotPose3/TRAIN/outputs_heatmap/best_heatmap.pth"

MODEL_NAME='facebook/dinov3-vitb16-pretrain-lvd1689m'
IMAGE_SIZE=512
HEATMAP_SIZE=512

EPOCHS=100
BATCH_SIZE=32
NUM_WORKERS=4
LEARNING_RATE=1e-4
WARMUP_STEPS=500
GRAD_CLIP=1.0

# Backbone unfreeze config
UNFREEZE_BLOCKS=2          # Last 2 transformer blocks
WARMUP_FROZEN_EPOCHS=5     # Keep frozen for first 5 epochs

VAL_RATIO=0.5
OCC_PROB=0.25
OCC_SIZE=0.2

WANDB_PROJECT="dinov3-joint-angle-v2"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="v2_unfreeze_${TIMESTAMP}"
OUTPUT_DIR="./outputs_3d_v2/train_${TIMESTAMP}"

echo "============================================"
echo "Joint Angle Training v2"
echo "  Backbone: unfreeze last ${UNFREEZE_BLOCKS} blocks at epoch ${WARMUP_FROZEN_EPOCHS}"
echo "  Heatmap: progressive loss (RoboPEPP style)"
echo "  Angle: direct prediction (normalized)"
echo "  Output: ${OUTPUT_DIR}"
echo "============================================"

torchrun --standalone --nnodes=1 --nproc_per_node=${NUM_GPUS} train_3d_v2.py \
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
    --warmup-steps ${WARMUP_STEPS} \
    --grad-clip ${GRAD_CLIP} \
    --unfreeze-blocks ${UNFREEZE_BLOCKS} \
    --warmup-frozen-epochs ${WARMUP_FROZEN_EPOCHS} \
    --val-ratio ${VAL_RATIO} \
    --occlusion-prob ${OCC_PROB} \
    --occlusion-size ${OCC_SIZE} \
    --num-workers ${NUM_WORKERS} \
    --use-wandb \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-run-name "${RUN_NAME}"

echo "Training Completed!"
