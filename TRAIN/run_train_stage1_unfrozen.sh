#!/bin/bash
# Stage 1 (variant B) — Strong detector with UNFROZEN backbone (last 4 blocks fine-tuned).
# Runs on GPU 1 in parallel with the frozen-backbone variant on GPU 2 -> clean A/B.
# Different method: fine-tune DINOv3 features for robot keypoints (low backbone LR)
# vs the frozen-backbone run. Keep whichever gives better real-azure PCK.

# GPU 1 by UUID (integer index "1"/"2" is unreliable due to faulty GPU 0 NVML enumeration)
export CUDA_VISIBLE_DEVICES=GPU-ab38c04c-0adf-17eb-fc9f-fab2e28559f5   # RTX 3090, physical GPU 1
export HF_HOME=/data/public/97_cache

cd /data/public/NAS/DINObotPose3/TRAIN

TS=$(date +%Y%m%d_%H%M%S)
OUT_DIR="./outputs_heatmap/stage1_unfrozen_${TS}"
mkdir -p "$OUT_DIR"

TRAIN_DIR="../Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_train_dr"
VAL_DIR="../Dataset/Converted_dataset/DREAM_real/panda-3cam_azure"
PRETRAIN="./outputs_heatmap/best_heatmap.pth"

python3 train_heatmap.py \
    --data-dir "$TRAIN_DIR" \
    --val-dir "$VAL_DIR" \
    --checkpoint "$PRETRAIN" \
    --model-name "facebook/dinov3-vitb16-pretrain-lvd1689m" \
    --output-dir "$OUT_DIR" \
    --image-size 512 --heatmap-size 512 \
    --unfreeze-blocks 4 \
    --aug-level strong \
    --occlusion-prob 0.0 \
    --fda-prob 0.0 \
    --epochs 40 \
    --batch-size 32 \
    --num-workers 12 \
    --learning-rate 2e-4 \
    --backbone-lr 2e-5 \
    --min-lr 1e-7 \
    --weight-decay 1e-5 \
    --wandb-project "dinov3-stage1-detector" \
    --wandb-run-name "stage1_unfrozen_b4_${TS}" \
    2>&1 | tee "$OUT_DIR/train.log"
