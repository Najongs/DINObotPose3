#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Model and dataset
MODEL_PATH="/home/najo/NAS/DIP/DINObotPose3/TRAIN/outputs/dinov3_base_20260303_163439_0.001/best_model.pth"
DATASET_DIR="/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/Converted_dataset/DREAM_to_DREAM/panda-3cam_azure"
# DATASET_DIR="/data/public/NAS/DINObotPose2/Dataset/Converted_dataset/DREAM_to_DREAM/panda-3cam_kinect360"
# DATASET_DIR="/data/public/NAS/DINObotPose2/Dataset/Converted_dataset/DREAM_to_DREAM/panda-3cam_realsense"
# DATASET_DIR="/data/public/NAS/DINObotPose2/Dataset/Converted_dataset/DREAM_to_DREAM/panda-orb"
# DATASET_DIR="/data/public/NAS/DINObotPose2/Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_test_dr"
# DATASET_DIR="/data/public/NAS/DINObotPose2/Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_test_photo"

# Output
OUTPUT_DIR="${SCRIPT_DIR}/eval_outputs_outlier"

# Inference
BATCH_SIZE=64
NUM_WORKERS=4
FIX_JOINT7_ZERO=1
KP_MIN_CONFIDENCE=0.25  # mask low-confidence 2D keypoints as invalid (-999)
KP_MIN_PEAK_LOGIT=0.25  # mask low-peak heatmap keypoints as invalid (-999)
PNP_MIN_SPAN_PX=10.0
PNP_MIN_AREA_RATIO=0.001
FILL_INVALID_2D_WITH_FK_REPROJ=0  # keep 0 for strict benchmark comparability
PNP_Z_SEARCH_MIN_M=-0.05
PNP_Z_SEARCH_MAX_M=0.05
PNP_Z_SEARCH_STEP_M=0.001

# Metrics thresholds
KP_AUC_THRESHOLD=20.0
ADD_AUC_THRESHOLD=0.1

# Outlier report
OUTLIER_TOPK=200

# Execution mode
INFER_MODE="multi_gpu"
NUM_GPUS=3
GPU_IDS="0,1,2"

if [ "${INFER_MODE}" = "single_gpu" ]; then
    echo "Running single-GPU outlier analysis..."
    export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
    python "${SCRIPT_DIR}/inference_dataset.py" \
        --model-path "$MODEL_PATH" \
        --dataset-dir "$DATASET_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --batch-size $BATCH_SIZE \
        --num-workers $NUM_WORKERS \
        --pred-3d-source fk \
        $( [[ "${FIX_JOINT7_ZERO}" == "1" ]] && echo "--fix-joint7-zero" ) \
        --kp-min-confidence "${KP_MIN_CONFIDENCE}" \
        --kp-min-peak-logit "${KP_MIN_PEAK_LOGIT}" \
        --pnp-min-span-px "${PNP_MIN_SPAN_PX}" \
        --pnp-min-area-ratio "${PNP_MIN_AREA_RATIO}" \
        --pnp-z-search-min-m "${PNP_Z_SEARCH_MIN_M}" \
        --pnp-z-search-max-m "${PNP_Z_SEARCH_MAX_M}" \
        --pnp-z-search-step-m "${PNP_Z_SEARCH_STEP_M}" \
        $( [[ "${FILL_INVALID_2D_WITH_FK_REPROJ}" == "1" ]] && echo "--fill-invalid-2d-with-fk-reproj" ) \
        --robopepp-pnp-init-thresh 0.25 \
        --robopepp-pnp-conf-step 0.025 \
        --kp-auc-threshold $KP_AUC_THRESHOLD \
        --add-auc-threshold $ADD_AUC_THRESHOLD \
        --save-metric-plots \
        --save-per-frame-errors \
        --outlier-topk $OUTLIER_TOPK
elif [ "${INFER_MODE}" = "multi_gpu" ]; then
    echo "Running distributed outlier analysis with ${NUM_GPUS} GPUs..."
    export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
    torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=${NUM_GPUS} \
        "${SCRIPT_DIR}/inference_dataset.py" \
        --distributed \
        --model-path "$MODEL_PATH" \
        --dataset-dir "$DATASET_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --batch-size $BATCH_SIZE \
        --num-workers $NUM_WORKERS \
        --pred-3d-source fk \
        $( [[ "${FIX_JOINT7_ZERO}" == "1" ]] && echo "--fix-joint7-zero" ) \
        --kp-min-confidence "${KP_MIN_CONFIDENCE}" \
        --kp-min-peak-logit "${KP_MIN_PEAK_LOGIT}" \
        --pnp-min-span-px "${PNP_MIN_SPAN_PX}" \
        --pnp-min-area-ratio "${PNP_MIN_AREA_RATIO}" \
        --pnp-z-search-min-m "${PNP_Z_SEARCH_MIN_M}" \
        --pnp-z-search-max-m "${PNP_Z_SEARCH_MAX_M}" \
        --pnp-z-search-step-m "${PNP_Z_SEARCH_STEP_M}" \
        $( [[ "${FILL_INVALID_2D_WITH_FK_REPROJ}" == "1" ]] && echo "--fill-invalid-2d-with-fk-reproj" ) \
        --robopepp-pnp-init-thresh 0.25 \
        --robopepp-pnp-conf-step 0.025 \
        --kp-auc-threshold $KP_AUC_THRESHOLD \
        --add-auc-threshold $ADD_AUC_THRESHOLD \
        --save-metric-plots \
        --save-per-frame-errors \
        --outlier-topk $OUTLIER_TOPK
else
    echo "Error: Unknown INFER_MODE=${INFER_MODE}"
    exit 1
fi

echo "Outlier analysis completed."
echo "Check files in: ${OUTPUT_DIR}"
echo "  - eval_results.json"
echo "  - auc_curve_pck_2d.png"
echo "  - auc_curve_add_camera_frame.png"
echo "  - per_frame_3d_errors.json"
echo "  - outlier_topk_3d_errors.json"
echo "  - outlier_topk_json_names.txt"
echo "  - outlier_topk_json_paths.txt"
echo "  - per_keypoint_3d_error_summary.json"
