#!/bin/bash
# Camera-rotation head: predict robot->camera 6D rotation from DINOv3 appearance to seed the
# kinematic solver's R_init and escape the far-camera rotation-basin ambiguity (oracle R-init =
# +0.11 realsense ADD-AUC). Frozen backbone+detector; only rot_head trains. GPU2.
export CUDA_VISIBLE_DEVICES=GPU-05f84104-40d4-c675-91bf-5427bc0fd5e9
export HF_HOME=/data/public/97_cache
cd /data/public/NAS/DINObotPose3/TRAIN

TS=$(date +%Y%m%d_%H%M%S)
OUT="./outputs_rotation/rot_${TS}"; mkdir -p "$OUT"
DET=outputs_heatmap/stage1_unfrozen_20260602_145811/best_heatmap.pth

python3 train_rotation.py \
    --detector-ckpt "$DET" \
    --train-dir ../Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_train_dr \
    --val-dir ../Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_test_dr \
    --output-dir "$OUT" \
    --model-name facebook/dinov3-vitb16-pretrain-lvd1689m \
    --image-size 512 --batch-size 32 --epochs 30 \
    --lr 1e-3 --min-lr 1e-6 --weight-decay 1e-4 --num-workers 8 \
    --t-weight 50.0 \
    --use-wandb --wandb-project dinov3-rotation --wandb-run-name "rot_${TS}" \
    2>&1 | tee "$OUT/train.log"
