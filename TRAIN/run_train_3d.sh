#!/bin/bash

# DINOv3 3D Pose Training Script

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

# 2D Pretrained Checkpoint (필수)
# 2D Heatmap 학습이 완료된 'best_heatmap.pth' 경로를 입력하세요.
CHECKPOINT="/data/public/NAS/DINObotPose3/TRAIN/outputs_heatmap/best_heatmap.pth"

# Model configuration
MODEL_NAME='facebook/dinov3-vitb16-pretrain-lvd1689m'
IMAGE_SIZE=512
HEATMAP_SIZE=512

# Training hyperparameters - 🚀 [최적화] 각도 학습 집중
EPOCHS=50        # 적당한 에폭 (각도만 맞으면 됨)
BATCH_SIZE=16    # 안정적 학습
NUM_WORKERS=4
LEARNING_RATE=1e-4   # 표준 LR (각도 직접 예측이므로 안정적)
MIN_LR=1e-7      # 표준 극저 LR
WARMUP_STEPS=500     # 표준 warmup
GRAD_CLIP=1.0

# Loss weights - 🚀 [개선] Sin/Cos 기반, FK loss 비활성화
ANGLE_WEIGHT=1.0    # Sin/Cos loss weight
FK_3D_WEIGHT=0.0    # FK disabled during training (metric only in validation)

# Sim-to-Real Augmentation (🚀 강화됨)
OCC_PROB=0.6        # 강력한 occlusion (60% 확률)
OCC_SIZE=0.4        # 더 큰 occlusion 패치

# WANDB Settings
WANDB_PROJECT="dinov3-3d-pose"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="train_3d_angle_focus_${TIMESTAMP}"
OUTPUT_DIR="./outputs_3d/train_3d_${TIMESTAMP}"

# =============================================================================
# Execution
# =============================================================================

echo "============================================================================="
echo "==> STARTING 3D POSE TRAINING (STABLE FK MODE)"
echo "==> 2D Checkpoint: ${CHECKPOINT}"
echo "==> Params: LR=${LEARNING_RATE}, Warmup=${WARMUP_STEPS}, Occ=${OCC_PROB}"
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
    --angle-weight ${ANGLE_WEIGHT} \
    --fk-3d-weight ${FK_3D_WEIGHT} \
    --occlusion-prob ${OCC_PROB} \
    --occlusion-size ${OCC_SIZE} \
    --num-workers ${NUM_WORKERS} \
    --mode joint_angle \
    --use-wandb \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-run-name "${RUN_NAME}"

echo "==> 3D Training Completed!"
