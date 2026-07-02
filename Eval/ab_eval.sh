#!/bin/bash
# Quick same-frame 6-split ADD-AUC A/B for an angle head. Real cameras use the rot-head (locked
# pipeline); synth runs no-rot. 300 strided frames/split (representative, NOT the biased [:N] slice).
# Usage:  bash ab_eval.sh <angle_head.pth> <label> [--crop] [max_frames]
#   --crop : evaluate with robot-bbox crop (use a crop-trained detector+head). Optional.
# Env: pins GPU2 by default (override CUDA_VISIBLE_DEVICES before calling).
set -u
cd /data/public/NAS/DINObotPose3/Eval
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-GPU-05f84104-40d4-c675-91bf-5427bc0fd5e9}
export HF_HOME=/data/public/97_cache

HEAD="$1"; LABEL="$2"; CROP=""; MAXF=300
shift 2
for a in "$@"; do
  case "$a" in
    --crop) CROP="--crop" ;;
    [0-9]*) MAXF="$a" ;;
  esac
done

# crop needs a crop-trained detector; otherwise use the locked stage1 detector.
if [ -n "$CROP" ]; then
  DET=$(ls -t ../TRAIN/outputs_heatmap/crop_*/best_heatmap.pth 2>/dev/null | head -1)
else
  DET=../TRAIN/outputs_heatmap/stage1_unfrozen_20260602_145811/best_heatmap.pth
fi
ROT=${ROT_PATH:-$(ls -t ../TRAIN/outputs_rotation/rot_*/best_rot_head.pth 2>/dev/null | head -1)}
# NO_ROT=1 disables the rot-head on real cameras (use for crop A/B: rot-head was trained on
# full-frame features and is off-distribution on cropped inputs -> confounds pure-crop attribution).
RH="--rot-head $ROT"; [ "${NO_ROT:-0}" = "1" ] && RH=""
echo "label=$LABEL  head=$HEAD  det=$DET  crop=${CROP:-none}  rot=${RH:-none}  frames=$MAXF"

declare -A SPLITS=(
  [realsense]="../Dataset/Converted_dataset/DREAM_real/panda-3cam_realsense|$RH"
  [azure]="../Dataset/Converted_dataset/DREAM_real/panda-3cam_azure|"
  [kinect]="../Dataset/Converted_dataset/DREAM_real/panda-3cam_kinect360|$RH"
  [orb]="../Dataset/Converted_dataset/DREAM_real/panda-orb|$RH"
  [synth_dr]="../Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_test_dr|"
)
ORDER=(realsense azure kinect orb synth_dr)
SUM=0; N=0
for s in "${ORDER[@]}"; do
  V="${SPLITS[$s]%%|*}"; RF="${SPLITS[$s]##*|}"
  L=/tmp/ab_${LABEL}_${s}
  python3 refine_eval.py --detector "$DET" --mlp-head "$HEAD" $RF $CROP \
      --val-dir "$V" --max-frames "$MAXF" --batch-size 16 > "$L.log" 2>&1
  A=$(grep -aoE 'ADD-AUC@100mm: [0-9.]+' "$L.log" | head -1 | grep -oE '[0-9.]+$')
  printf "  %-10s ADD-AUC = %s\n" "$s" "${A:-FAIL}"
  if [ -n "$A" ]; then SUM=$(echo "$SUM + $A" | bc -l); N=$((N+1)); fi
done
[ "$N" -gt 0 ] && printf "  %-10s mean(%d)  = %.4f\n" "MEAN" "$N" "$(echo "$SUM / $N" | bc -l)"
