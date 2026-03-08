#!/bin/bash

# Diffusion-based Joint Angle Training
# Strategy:
# - Freeze backbone + heatmap head
# - Train only angle head on top of pretrained 2D predictions
# - Use FDA + mild occlusion for sim-to-real robustness

GPU_IDS="0,1,2"
NUM_GPUS=3
export CUDA_VISIBLE_DEVICES=${GPU_IDS}

TRAIN_DIR="/data/public/NAS/DINObotPose2/Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_train_dr"
VAL_DIR="/data/public/NAS/DINObotPose2/Dataset/Converted_dataset/DREAM_real/panda-3cam_azure"
CHECKPOINT="/data/public/NAS/DINObotPose3/TRAIN/outputs_heatmap/best_heatmap.pth"
FDA_REAL_DIR="/data/public/NAS/DINObotPose2/Dataset/DREAM_real"

MODEL_NAME='facebook/dinov3-vitb16-pretrain-lvd1689m'
IMAGE_SIZE=512
HEATMAP_SIZE=512

EPOCHS=60
BATCH_SIZE=16
NUM_WORKERS=4
LEARNING_RATE=2e-4
WEIGHT_DECAY=1e-4
WARMUP_STEPS=1000
GRAD_CLIP=1.0

# Keep 2D front-end frozen by default
UNFREEZE_BLOCKS=0
WARMUP_FROZEN_EPOCHS=0
BACKBONE_LR_SCALE=0.05

DIFFUSION_STEPS=20
ANGLE_DROPOUT=0.1
INIT_LOSS_WEIGHT=1.0
RECON_LOSS_WEIGHT=0.5

# Sim-to-real augmentation
FDA_PROB=0.25
FDA_BETA=0.005
OCCLUSION_PROB=0.2
OCCLUSION_MAX_HOLES=4
OCCLUSION_MAX_SIZE_FRAC=0.15

WANDB_PROJECT="dinov3-diffusion-angle"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="diffusion_${TIMESTAMP}"
OUTPUT_DIR="./outputs_diffusion/train_${TIMESTAMP}"

echo "============================================"
echo "Diffusion Joint Angle Training"
echo "  Front-end: frozen"
echo "  FDA: prob=${FDA_PROB}, beta=${FDA_BETA}"
echo "  Occlusion: prob=${OCCLUSION_PROB}, holes=${OCCLUSION_MAX_HOLES}"
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
    --warmup-steps ${WARMUP_STEPS} \
    --grad-clip ${GRAD_CLIP} \
    --unfreeze-blocks ${UNFREEZE_BLOCKS} \
    --warmup-frozen-epochs ${WARMUP_FROZEN_EPOCHS} \
    --backbone-lr-scale ${BACKBONE_LR_SCALE} \
    --diffusion-steps ${DIFFUSION_STEPS} \
    --angle-dropout ${ANGLE_DROPOUT} \
    --init-loss-weight ${INIT_LOSS_WEIGHT} \
    --recon-loss-weight ${RECON_LOSS_WEIGHT} \
    --fda-real-dir "${FDA_REAL_DIR}" \
    --fda-prob ${FDA_PROB} \
    --fda-beta ${FDA_BETA} \
    --occlusion-prob ${OCCLUSION_PROB} \
    --occlusion-max-holes ${OCCLUSION_MAX_HOLES} \
    --occlusion-max-size-frac ${OCCLUSION_MAX_SIZE_FRAC} \
    --use-wandb \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-run-name "${RUN_NAME}"

echo "Training Completed!"
