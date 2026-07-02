#!/bin/bash
# Backbone experiment — SigLIP2 detector vs DINOv3 detector (apples-to-apples).
# siglip2-base-patch16-512 matches DINOv3-ViTB16's token grid (32x32x768), drop-in for the
# keypoint head. UNFROZEN (last 4 blocks) to match the winning DINOv3-unfrozen config.
# Trained from scratch (siglip features differ from dinov3, no warm-start). Uses the
# optimized windowed heatmap (low CPU) and SigLIP normalization (auto-detected).
#
# Run on the LOSER's GPU after stopping it. Default GPU 2 by UUID.
#   bash run_train_siglip.sh [gpu_uuid]

set -e
cd /data/public/NAS/DINObotPose3/TRAIN

GPU2_UUID=GPU-05f84104-40d4-c675-91bf-5427bc0fd5e9
export CUDA_VISIBLE_DEVICES="${1:-$GPU2_UUID}"
export HF_HOME=/data/public/97_cache

TS=$(date +%Y%m%d_%H%M%S)
OUT_DIR="./outputs_heatmap/siglip2_unfrozen_${TS}"
mkdir -p "$OUT_DIR"

TRAIN_DIR="../Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_train_dr"
VAL_DIR="../Dataset/Converted_dataset/DREAM_real/panda-3cam_azure"

# Warm-start the keypoint_head from the DINOv3-trained head to escape the sparse-heatmap
# cold-start (from-scratch siglip got stuck predicting blank heatmaps). The DINOv3 backbone
# keys in best_heatmap.pth do NOT match siglip's (vision_model.*), so ONLY the head loads;
# siglip's own pretrained backbone is preserved. This also makes the head init identical
# across both backbones -> fairer comparison.
PRETRAIN="./outputs_heatmap/best_heatmap.pth"

# Matches the DINOv3-unfrozen arm: unfreeze 4 blocks, batch 32, head-lr 2e-4, bb-lr 2e-5.
python3 train_heatmap.py \
    --data-dir "$TRAIN_DIR" \
    --val-dir "$VAL_DIR" \
    --checkpoint "$PRETRAIN" \
    --model-name "google/siglip2-base-patch16-512" \
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
    --wandb-run-name "siglip2_unfrozen_b4_${TS}" \
    2>&1 | tee "$OUT_DIR/train.log"
