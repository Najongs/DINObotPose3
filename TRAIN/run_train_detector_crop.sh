#!/bin/bash
# Detector retrain with ROBOT-CENTERED CROP (train+test). Diagnosis: realsense angle is dominated
# by J0=44deg which is 2D-limited (oracle-2D->14deg); failures are foreshortened poses where the
# small robot's base keypoints are imprecise. Cropping to the robot bbox (RoboPEPP-style) puts more
# pixels on the (foreshortened) arm -> better keypoints -> better J0. Warm-start from the current
# best detector so it only adapts to the crop scale distribution. GPU1.
export CUDA_VISIBLE_DEVICES=GPU-ab38c04c-0adf-17eb-fc9f-fab2e28559f5
export HF_HOME=/data/public/97_cache
cd /data/public/NAS/DINObotPose3/TRAIN

TS=$(date +%Y%m%d_%H%M%S)
OUT_DIR="./outputs_heatmap/crop_${TS}"; mkdir -p "$OUT_DIR"
TRAIN_DIR="../Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_train_dr"
VAL_DIR="../Dataset/Converted_dataset/DREAM_real/panda-3cam_realsense"   # watch the target camera
PRETRAIN="./outputs_heatmap/stage1_unfrozen_20260602_145811/best_heatmap.pth"

python3 train_heatmap.py \
    --data-dir "$TRAIN_DIR" \
    --val-dir "$VAL_DIR" \
    --checkpoint "$PRETRAIN" \
    --model-name "facebook/dinov3-vitb16-pretrain-lvd1689m" \
    --output-dir "$OUT_DIR" \
    --image-size 512 --heatmap-size 512 \
    --crop-to-robot --crop-margin 1.5 \
    --unfreeze-blocks 4 \
    --aug-level strong \
    --occlusion-prob 0.0 --fda-prob 0.0 \
    --epochs 25 \
    --batch-size 32 \
    --num-workers 12 \
    --learning-rate 2e-4 --backbone-lr 2e-5 --min-lr 1e-7 --weight-decay 1e-5 \
    --wandb-project "dinov3-stage1-detector" --wandb-run-name "crop_${TS}" \
    2>&1 | tee "$OUT_DIR/train.log"
