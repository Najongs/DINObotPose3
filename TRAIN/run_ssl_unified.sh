#!/bin/bash

set -euo pipefail

# Unified Reprojection SSL fine-tuning launcher
# 1) Freeze backbone + 2D head
# 2) Train only joint_angle_head with Dynamic K reprojection loss

# =============================================================================
# Configuration
# =============================================================================

# Training data roots (AK/XK/RS/ORB 등을 통합)
DATA_DIRS=(
  "/data/public/NAS/DINObotPose2/Dataset/Converted_dataset/DREAM_to_DREAM"
  # "/data/public/NAS/DINObotPose2/Dataset/Converted_dataset/DREAM_to_DREAM_syn"
)

# Optional validation root (leave empty to disable validation)
VAL_DIR="/data/public/NAS/DINObotPose2/Dataset/Converted_dataset/DREAM_to_DREAM/panda-3cam_azure"

# REQUIRED: supervised training checkpoint to start SSL from
CHECKPOINT="/data/public/NAS/DINObotPose3/TRAIN/outputs/dinov3_base_20260303_020716/epoch_56.pth"

# Optional allowlists
TRAIN_JSON_LIST=""
VAL_JSON_LIST=""

# Model
MODEL_NAME="facebook/dinov3-vitb16-pretrain-lvd1689m"
USE_JOINT_EMBEDDING=true
FIX_JOINT7_ZERO=true

MULTI_ROBOT=false
ROBOT_TYPES=""   # e.g. "franka_panda meca500"

# Training
EPOCHS=20
BATCH_SIZE=16
NUM_WORKERS=4
IMAGE_SIZE=512
HEATMAP_SIZE=512
AUGMENT=false
LEARNING_RATE=1e-5
WEIGHT_DECAY=1e-6
GRAD_CLIP=1.0
SEED=42

# Runtime
GPU_ID=0
OUTPUT_DIR="/data/public/NAS/DINObotPose3/TRAIN/outputs/ssl_unified_$(date +%Y%m%d_%H%M%S)"

# =============================================================================
# Validation
# =============================================================================

if [ ${#DATA_DIRS[@]} -eq 0 ]; then
  echo "Error: DATA_DIRS is empty."
  exit 1
fi

for d in "${DATA_DIRS[@]}"; do
  if [ ! -d "${d}" ]; then
    echo "Error: data dir not found: ${d}"
    exit 1
  fi
done

if [ ! -f "${CHECKPOINT}" ]; then
  echo "Error: checkpoint not found: ${CHECKPOINT}"
  exit 1
fi

if [ -n "${VAL_DIR}" ] && [ ! -d "${VAL_DIR}" ]; then
  echo "Error: VAL_DIR not found: ${VAL_DIR}"
  exit 1
fi

# =============================================================================
# Build flags
# =============================================================================

EXTRA_FLAGS=""
if [ "${USE_JOINT_EMBEDDING}" = true ]; then
  EXTRA_FLAGS="${EXTRA_FLAGS} --use-joint-embedding"
fi
if [ "${FIX_JOINT7_ZERO}" = true ]; then
  EXTRA_FLAGS="${EXTRA_FLAGS} --fix-joint7-zero"
fi
if [ "${MULTI_ROBOT}" = true ]; then
  EXTRA_FLAGS="${EXTRA_FLAGS} --multi-robot"
fi
if [ "${AUGMENT}" = true ]; then
  EXTRA_FLAGS="${EXTRA_FLAGS} --augment"
fi
if [ -n "${ROBOT_TYPES}" ]; then
  EXTRA_FLAGS="${EXTRA_FLAGS} --robot-types ${ROBOT_TYPES}"
fi
if [ -n "${TRAIN_JSON_LIST}" ]; then
  EXTRA_FLAGS="${EXTRA_FLAGS} --train-json-list ${TRAIN_JSON_LIST}"
fi
if [ -n "${VAL_JSON_LIST}" ]; then
  EXTRA_FLAGS="${EXTRA_FLAGS} --val-json-list ${VAL_JSON_LIST}"
fi
if [ -n "${VAL_DIR}" ]; then
  EXTRA_FLAGS="${EXTRA_FLAGS} --val-dir ${VAL_DIR}"
fi

mkdir -p "${OUTPUT_DIR}"

CMD="python /data/public/NAS/DINObotPose3/TRAIN/train_ssl_unified.py \
  --data-dir ${DATA_DIRS[*]} \
  --checkpoint ${CHECKPOINT} \
  --output-dir ${OUTPUT_DIR} \
  --model-name ${MODEL_NAME} \
  --epochs ${EPOCHS} \
  --batch-size ${BATCH_SIZE} \
  --num-workers ${NUM_WORKERS} \
  --image-size ${IMAGE_SIZE} \
  --heatmap-size ${HEATMAP_SIZE} \
  --learning-rate ${LEARNING_RATE} \
  --weight-decay ${WEIGHT_DECAY} \
  --grad-clip ${GRAD_CLIP} \
  --seed ${SEED} \
  ${EXTRA_FLAGS}"

echo "Running SSL unified fine-tuning..."
echo "Output: ${OUTPUT_DIR}"
echo "${CMD}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" bash -lc "${CMD}"

