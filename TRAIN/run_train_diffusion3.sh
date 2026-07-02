#!/bin/bash
# Diffusion angle head on the strong DINOv3-unfrozen detector (3rd head architecture).
# Frozen backbone + frozen keypoint head; trains only the DiffusionAngleHead.
# Conditions a DDIM denoiser on DINOv3 global + 2D skeleton geometry to GENERATE joint angles
# -> captures the multi-modal posterior of the (ambiguous) single-view angle problem.
# Fair vs MLP/Transformer: same detector ckpt, same strong aug, FDA off.
#   bash run_train_diffusion3.sh [gpu_uuid]

set -e
cd /data/public/NAS/DINObotPose3/TRAIN

GPU2_UUID=GPU-05f84104-40d4-c675-91bf-5427bc0fd5e9
export CUDA_VISIBLE_DEVICES="${1:-$GPU2_UUID}"
export HF_HOME=/data/public/97_cache

DET="outputs_heatmap/stage1_unfrozen_20260602_145811/best_heatmap.pth"
TRAIN_DIR="../Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_train_dr"
VAL_DIR="../Dataset/Converted_dataset/DREAM_real/panda-3cam_azure"

TS=$(date +%Y%m%d_%H%M%S)
OUT_DIR="./outputs_diffusion/diffusion3_${TS}"
mkdir -p "$OUT_DIR"

torchrun --standalone --nnodes=1 --nproc_per_node=1 train_diffusion.py \
    --train-dir "$TRAIN_DIR" \
    --val-dir "$VAL_DIR" \
    --checkpoint "$DET" \
    --model-name "facebook/dinov3-vitb16-pretrain-lvd1689m" \
    --output-dir "$OUT_DIR" \
    --image-size 512 --heatmap-size 512 \
    --batch-size 48 \
    --epochs 60 \
    --lr 2e-4 --weight-decay 1e-4 \
    --num-workers 12 \
    --warmup-steps 500 --grad-clip 1.0 \
    --unfreeze-blocks 0 \
    --warmup-frozen-epochs 0 \
    --backbone-lr-scale 0.05 \
    --diffusion-steps 20 \
    --angle-dropout 0.1 \
    --init-loss-weight 1.0 \
    --recon-loss-weight 0.5 \
    --fk-loss-weight 0.1 \
    --aug-level strong \
    --fda-prob 0.0 \
    --occlusion-prob 0.0 \
    --use-wandb --wandb-project "dinov3-angle-predictor" \
    --wandb-run-name "diffusion3_${TS}" \
    2>&1 | tee "$OUT_DIR/train.log"
