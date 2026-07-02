#!/bin/bash
# Detector FINE-TUNE — finish the LR schedule the stage1_unfrozen run never got.
# That run was stopped at epoch ~9/40, so cosine LR was still high (~1.8e-4) and the
# model never received the low-LR sharpening phase. real-azure PCK@5 was oscillating
# (0.79-0.83), consistent with too-high LR, not convergence. This warm-starts from its
# best checkpoint and runs a full cosine DECAY to min-lr to sharpen keypoint precision
# (the #1 bottleneck for the visible-frame ADD-AUC tail). Validates on real-azure and
# keeps best-by-AUC, so even if extra sim training overfits, we never regress.

export CUDA_VISIBLE_DEVICES=GPU-05f84104-40d4-c675-91bf-5427bc0fd5e9   # physical GPU 2
export HF_HOME=/data/public/97_cache
cd /data/public/NAS/DINObotPose3/TRAIN

TS=$(date +%Y%m%d_%H%M%S)
OUT_DIR="./outputs_heatmap/detector_ft_${TS}"
mkdir -p "$OUT_DIR"

TRAIN_DIR="../Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_train_dr"
VAL_DIR="../Dataset/Converted_dataset/DREAM_real/panda-3cam_azure"
# warm-start from the current best detector (epoch-9 weights)
WARM="./outputs_heatmap/stage1_unfrozen_20260602_145811/best_heatmap.pth"

python3 train_heatmap.py \
    --data-dir "$TRAIN_DIR" \
    --val-dir "$VAL_DIR" \
    --checkpoint "$WARM" \
    --model-name "facebook/dinov3-vitb16-pretrain-lvd1689m" \
    --output-dir "$OUT_DIR" \
    --image-size 512 --heatmap-size 512 \
    --unfreeze-blocks 4 \
    --aug-level strong \
    --occlusion-prob 0.0 \
    --fda-prob 0.0 \
    --epochs 25 \
    --batch-size 48 \
    --num-workers 12 \
    --learning-rate 1e-4 \
    --backbone-lr 1e-5 \
    --min-lr 1e-7 \
    --weight-decay 1e-5 \
    --wandb-project "dinov3-stage1-detector" \
    --wandb-run-name "detector_ft_${TS}" \
    2>&1 | tee "$OUT_DIR/train.log"
