#!/bin/bash
# STEP 2 of the train+test bbox-crop pipeline. After the detector is retrained on robot-centered
# crops (run_train_detector_crop.sh), the angle head must ALSO see cropped inputs so its 2D bearings
# match the detector's crop-scale distribution. Diagnosis: realsense angle is dominated by J0=44deg
# (2D-limited); cropping puts more pixels on the foreshortened arm -> better base keypoints -> better J0.
# Pairs with --crop in Eval/refine_eval.py for evaluation. GPU2.
export CUDA_VISIBLE_DEVICES=GPU-05f84104-40d4-c675-91bf-5427bc0fd5e9
export HF_HOME=/data/public/97_cache
cd /data/public/NAS/DINObotPose3/TRAIN

TS=$(date +%Y%m%d_%H%M%S)
OUT="./outputs_angle/angle_crop_${TS}"; mkdir -p "$OUT"
# Pick the latest crop-trained detector checkpoint by default; override with $1.
DET="${1:-$(ls -t outputs_heatmap/crop_*/best_heatmap.pth 2>/dev/null | head -1)}"
echo "Using crop detector: $DET"

python3 train_angle.py \
    --detector-ckpt "$DET" \
    --train-dir ../Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_train_dr \
    --val-dir ../Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_test_dr \
    --output-dir "$OUT" \
    --model-name facebook/dinov3-vitb16-pretrain-lvd1689m \
    --head-type mlp --image-size 512 --batch-size 32 --epochs 50 \
    --lr 1e-3 --min-lr 1e-6 --weight-decay 1e-4 --fk-weight 10.0 \
    --crop-to-robot --crop-margin 1.5 \
    --num-workers 8 \
    --use-wandb --wandb-project dinov3-angle-predictor --wandb-run-name "angle_crop_${TS}" \
    2>&1 | tee "$OUT/train.log"
