#!/bin/bash
# Angle head with end-effector APPEARANCE PATCH (head_type=mlp_patch) — the wrist-angle fix.
# Wrist angles stay ~12 deg even with ORACLE 2D (geometric under-determination: wrist rotations
# barely move keypoints). A single-point sampled feature misses link/gripper orientation; a 3x3
# token patch around each keypoint injects local orientation so the head can read wrist roll/pitch.
# Fair A/B vs the plain-mlp head (same detector, same data). Frozen backbone+detector, head-only.

export CUDA_VISIBLE_DEVICES=GPU-ab38c04c-0adf-17eb-fc9f-fab2e28559f5   # GPU 1
export HF_HOME=/data/public/97_cache
cd /data/public/NAS/DINObotPose3/TRAIN

TS=$(date +%Y%m%d_%H%M%S)
OUT_DIR="./outputs_angle/angle_patch_${TS}"
mkdir -p "$OUT_DIR"
DET=outputs_heatmap/stage1_unfrozen_20260602_145811/best_heatmap.pth

python3 train_angle.py \
    --detector-ckpt "$DET" \
    --train-dir ../Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_train_dr \
    --val-dir ../Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_test_dr \
    --output-dir "$OUT_DIR" \
    --model-name facebook/dinov3-vitb16-pretrain-lvd1689m \
    --head-type mlp_patch \
    --image-size 512 \
    --batch-size 32 \
    --epochs 60 \
    --lr 1e-3 --min-lr 1e-6 --weight-decay 1e-4 \
    --fk-weight 10.0 \
    --num-workers 8 \
    --use-wandb --wandb-project dinov3-angle-predictor --wandb-run-name "angle_patch_${TS}" \
    2>&1 | tee "$OUT_DIR/train.log"
