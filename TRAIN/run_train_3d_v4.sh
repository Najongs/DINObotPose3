#!/bin/bash

# Configuration
export CUDA_VISIBLE_DEVICES="0,1,2,3,4"
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "Using $NUM_GPUS GPUs"

timestamp=$(date +%Y%m%d_%H%M%S)

# Output directories
BASE_OUTPUT_DIR="/home/najo/NAS/DIP/DINObotPose3/TRAIN/outputs_3d_v4"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/train_3d_v4_${timestamp}"
LOG_FILE="${OUTPUT_DIR}/train.log"

# Dataset
DATA_DIR="/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/Converted_dataset"
TRAIN_DIR="${DATA_DIR}/DREAM_to_DREAM_syn/panda_synth_train_dr"
VAL_DIR="${DATA_DIR}/DREAM_to_DREAM_syn/panda_synth_test_dr"

# Pretrained checkpoint (2D heatmap model)
PRETRAIN_CKPT="/home/najo/NAS/DIP/DINObotPose3/TRAIN/outputs_heatmap/*finetune_no_fda_with_occ_beta0.001_occ0.35_20260305_134104/best_heatmap.pth"

# Model configuration
MODEL_NAME="facebook/dinov3-vitb16-pretrain-lvd1689m"
IMAGE_SIZE=512
HEATMAP_SIZE=512

# Training hyperparameters
BATCH_SIZE=16            # per GPU
EPOCHS=100
LR=1e-4
WEIGHT_DECAY=1e-5
WARMUP_STEPS=1000
FK_LOSS_WEIGHT=50.0
REPROJ_LOSS_WEIGHT=100.0
GRAD_CLIP=1.0

# Freeze strategy
UNFREEZE_BLOCKS=2        # Same as RoboPEPP (last 2 layers of backbone)
WARMUP_FROZEN_EPOCHS=0   # Let's unfreeze from the start or keep 5

# Dataloader
NUM_WORKERS=32
OCCLUSION_PROB=0.5
OCCLUSION_SIZE=0.3
VAL_RATIO=0.2

# W&B
WANDB_PROJECT="dinov3-joint-angle-v4"
WANDB_RUN_NAME="v4_direct_reproj_${timestamp}"

# Create output dir
mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "Starting DINOv3 3D Pose Training v4 (E2E Reprojection)"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo "Using $NUM_GPUS GPUs: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "W&B Run Name: $WANDB_RUN_NAME"
echo "============================================================"

# Run Training via torchrun for DDP
OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$(shuf -i 10000-65535 -n 1) \
    train_3d_v4.py \
    --train-dir "$TRAIN_DIR" \
    --val-dir "$VAL_DIR" \
    --checkpoint "$PRETRAIN_CKPT" \
    --output-dir "$OUTPUT_DIR" \
    --model-name "$MODEL_NAME" \
    --image-size $IMAGE_SIZE \
    --heatmap-size $HEATMAP_SIZE \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --weight-decay $WEIGHT_DECAY \
    --warmup-steps $WARMUP_STEPS \
    --fk-loss-weight $FK_LOSS_WEIGHT \
    --reproj-loss-weight $REPROJ_LOSS_WEIGHT \
    --grad-clip $GRAD_CLIP \
    --unfreeze-blocks $UNFREEZE_BLOCKS \
    --warmup-frozen-epochs $WARMUP_FROZEN_EPOCHS \
    --val-ratio $VAL_RATIO \
    --occlusion-prob $OCCLUSION_PROB \
    --occlusion-size $OCCLUSION_SIZE \
    --num-workers $NUM_WORKERS \
    --use-wandb \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-run-name "$WANDB_RUN_NAME" \
    2>&1 | tee "$LOG_FILE"

echo "============================================================"
echo "Training completed."
echo "Results saved to: $OUTPUT_DIR"
echo "============================================================"
