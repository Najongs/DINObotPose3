#!/bin/bash
# Stage 1 — Strong keypoint detector (frozen backbone, strong aug, NO FDA)
# Goal: fewer gross-outlier keypoints -> better PnP / kinematic refine on real images.
# Warm-starts from the existing 2D detector and trains ONLY the keypoint head.

# GPU 2 by UUID — integer index "2" is broken by the faulty GPU 0's NVML enumeration,
# so we must select GPU 2 by its stable UUID (verified torch.cuda available=True).
export CUDA_VISIBLE_DEVICES=GPU-05f84104-40d4-c675-91bf-5427bc0fd5e9   # RTX 3090, physical GPU 2
export HF_HOME=/data/public/97_cache   # cached DINOv3 weights

cd /data/public/NAS/DINObotPose3/TRAIN

TS=$(date +%Y%m%d_%H%M%S)
OUT_DIR="./outputs_heatmap/stage1_strong_${TS}"
mkdir -p "$OUT_DIR"

TRAIN_DIR="../Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_train_dr"
VAL_DIR="../Dataset/Converted_dataset/DREAM_real/panda-3cam_azure"   # real -> tracks sim2real PCK
PRETRAIN="./outputs_heatmap/best_heatmap.pth"

python3 train_heatmap.py \
    --data-dir "$TRAIN_DIR" \
    --val-dir "$VAL_DIR" \
    --checkpoint "$PRETRAIN" \
    --model-name "facebook/dinov3-vitb16-pretrain-lvd1689m" \
    --output-dir "$OUT_DIR" \
    --image-size 512 --heatmap-size 512 \
    --unfreeze-blocks 0 \
    --aug-level strong \
    --occlusion-prob 0.0 \
    --fda-prob 0.0 \
    --epochs 40 \
    --batch-size 48 \
    --num-workers 12 \
    --learning-rate 5e-4 \
    --min-lr 1e-7 \
    --weight-decay 1e-5 \
    --wandb-project "dinov3-stage1-detector" \
    --wandb-run-name "stage1_strong_frozen_${TS}" \
    2>&1 | tee "$OUT_DIR/train.log"
