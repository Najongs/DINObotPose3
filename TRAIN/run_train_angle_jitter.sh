#!/bin/bash
# Angle head retrain with TRAIN-TIME keypoint-jitter augmentation. Diagnosis: the head amplifies
# detector 2D error on the gauge-sensitive base joint (realsense J0 = 44 deg vs 14 deg with oracle
# 2D). Jittering the detected 2D fed to geo+sampling should make J0 robust to the realsense 2D tail
# (lean on appearance, not exact base bearings). Fair A/B vs the plain-mlp head. GPU2.
export CUDA_VISIBLE_DEVICES=GPU-05f84104-40d4-c675-91bf-5427bc0fd5e9
export HF_HOME=/data/public/97_cache
cd /data/public/NAS/DINObotPose3/TRAIN

TS=$(date +%Y%m%d_%H%M%S)
OUT="./outputs_angle/angle_jitter_${TS}"; mkdir -p "$OUT"
DET=outputs_heatmap/stage1_unfrozen_20260602_145811/best_heatmap.pth

python3 train_angle.py \
    --detector-ckpt "$DET" \
    --train-dir ../Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_train_dr \
    --val-dir ../Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_test_dr \
    --output-dir "$OUT" \
    --model-name facebook/dinov3-vitb16-pretrain-lvd1689m \
    --head-type mlp --image-size 512 --batch-size 32 --epochs 50 \
    --lr 1e-3 --min-lr 1e-6 --weight-decay 1e-4 --fk-weight 10.0 \
    --kp-jitter 5.0 \
    --num-workers 8 \
    --use-wandb --wandb-project dinov3-angle-predictor --wandb-run-name "angle_jitter_${TS}" \
    2>&1 | tee "$OUT/train.log"
