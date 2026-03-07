#!/bin/bash

# Diffusion-based Joint Angle Training
# Method: DDPM with UV + feature conditioning

GPU_IDS="0,1,2"
NUM_GPUS=3
export CUDA_VISIBLE_DEVICES=${GPU_IDS}

TRAIN_DIR="/data/public/NAS/DINObotPose2/Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_train_dr"
VAL_DIR="/data/public/NAS/DINObotPose2/Dataset/Converted_dataset/DREAM_real/panda-3cam_azure"
CHECKPOINT="/data/public/NAS/DINObotPose3/TRAIN/outputs_heatmap/best_heatmap.pth"

MODEL_NAME='facebook/dinov3-vitb16-pretrain-lvd1689m'
IMAGE_SIZE=512
HEATMAP_SIZE=512

EPOCHS=100
BATCH_SIZE=16
NUM_WORKERS=4
LEARNING_RATE=5e-5
WEIGHT_DECAY=0.1

WANDB_PROJECT="dinov3-diffusion-angle"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="diffusion_${TIMESTAMP}"
OUTPUT_DIR="./outputs_diffusion/train_${TIMESTAMP}"

echo "============================================"
echo "Diffusion Joint Angle Training"
echo "  Method: DDPM (50 steps)"
echo "  Condition: UV + global feature"
echo "  Output: ${OUTPUT_DIR}"
echo "============================================"

torchrun --standalone --nnodes=1 --nproc_per_node=${NUM_GPUS} train_diffusion.py \
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
    --weight-decay ${WEIGHT_DECAY} \
    --num-workers ${NUM_WORKERS} \
    --use-wandb \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-run-name "${RUN_NAME}"

echo "Training Completed!"
