#!/bin/bash
# Stage 1.5 — Learned angle predictor on top of a FROZEN Stage-1 detector.
# Usage:  bash run_train_angle.sh <detector_ckpt> [gpu_uuid]
#   <detector_ckpt> : Stage-1 best_heatmap.pth (backbone+keypoint_head). If omitted,
#                     auto-picks the latest stage1_unfrozen/best_heatmap.pth.
#   [gpu_uuid]      : default GPU 1 (RTX 3090). Use GPU 2 UUID to run there.
#
# NOTE: select GPU by UUID — integer index is broken by the faulty GPU 0 (see memory).

set -e
cd /data/public/NAS/DINObotPose3/TRAIN

GPU1_UUID=GPU-ab38c04c-0adf-17eb-fc9f-fab2e28559f5
GPU2_UUID=GPU-05f84104-40d4-c675-91bf-5427bc0fd5e9

DET_CKPT="${1:-}"
if [ -z "$DET_CKPT" ]; then
    DET_CKPT=$(ls -t ./outputs_heatmap/stage1_unfrozen_*/best_heatmap.pth 2>/dev/null | head -1)
fi
if [ -z "$DET_CKPT" ] || [ ! -f "$DET_CKPT" ]; then
    echo "ERROR: detector checkpoint not found. Pass it explicitly: bash run_train_angle.sh <ckpt>"
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${2:-$GPU1_UUID}"
export HF_HOME=/data/public/97_cache

TS=$(date +%Y%m%d_%H%M%S)
OUT_DIR="./outputs_angle/angle_${TS}"
mkdir -p "$OUT_DIR"

echo "==> detector: $DET_CKPT"
echo "==> GPU: $CUDA_VISIBLE_DEVICES"
echo "==> out: $OUT_DIR"

python3 train_angle.py \
    --detector-ckpt "$DET_CKPT" \
    --train-dir "../Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_train_dr" \
    --val-dir   "../Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_test_dr" \
    --output-dir "$OUT_DIR" \
    --model-name "facebook/dinov3-vitb16-pretrain-lvd1689m" \
    --image-size 512 \
    --batch-size 32 \
    --epochs 60 \
    --lr 1e-3 --min-lr 1e-6 --weight-decay 1e-4 \
    --fk-weight 10.0 \
    --num-workers 8 \
    --use-wandb --wandb-project "dinov3-angle-predictor" \
    --wandb-run-name "angle_${TS}" \
    2>&1 | tee "$OUT_DIR/train.log"
