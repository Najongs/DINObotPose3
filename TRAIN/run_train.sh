#!/bin/bash

# DINOv3 Pose Estimation Training Script
# This script provides various training configurations

# =============================================================================
# Configuration
# =============================================================================

# Data paths (REQUIRED - Update these paths!)
DATA_DIRS=("/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/Converted_dataset/DREAM_to_DREAM" "/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/Converted_dataset/DREAM_to_DREAM_syn")
VAL_DIR="/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/Converted_dataset/TOTAL/DREAM_to_DREAM/panda-3cam_azure"  

# DATA_DIR="/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_train_dr"  # Training data directory
# VAL_DIR="/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/Converted_dataset/DREAM_to_DREAM/panda-3cam_azure"  # Validation data directory (separate from training)

TRAIN_SPLIT=1.0  # Train split ratio (1.0 = use all training data when VAL_DIR is specified)
VAL_SPLIT=1  # Validation data usage ratio (0.1 = use 10% of validation data)
TRAIN_JSON_LIST=""  # e.g. /data/public/NAS/DINObotPose2/Eval/eval_outputs_outlier/outlier_topk_json_paths.txt
TRAIN_JSON_LIST_MODE="extra"  # extra: full train + json-list extra pass per epoch, filter: only json list
TRAIN_JSON_EXTRA_LOSS_SCALE=1.0
VAL_JSON_LIST=""    # Optional separate allowlist for val set

# Model configuration
MODEL_NAME='facebook/dinov3-vitb16-pretrain-lvd1689m'
IMAGE_SIZE=512
HEATMAP_SIZE=512
UNFREEZE_BLOCKS=2  # Number of backbone blocks to unfreeze for fine-tuning

USE_JOINT_EMBEDDING=True  # Enable joint identity embeddings in 3D head
FIX_JOINT7_ZERO=True      # RoboPEPP-style: train with joint7 fixed to zero

# Joint angle mode loss weights
# angle loss: 라디안 단위 (범위 ~0~6), 3D loss: 미터 단위 (범위 ~0.01~0.5)
# FK_3D는 robot frame 기준이므로 실제 성능 지표(camera frame ADD)와 좌표계가 다름
# → angle loss로 자세 추정 → FK_3D로 구조적 일관성 강제 순서로 학습
ANGLE_WEIGHT=10.0    # Joint angle MSE loss weight
FK_3D_WEIGHT=100.0   # FK 3D keypoint MSE loss weight (robot frame)
DIRECT_3D_WEIGHT=0.0  # FK-only: direct 3D branch supervision disabled
CONSISTENCY_WEIGHT=0.0  # FK-only: FK/direct consistency disabled
FUSION_DELTA_WEIGHT=0.0  # FK-only: fusion residual regularization disabled
KEYPOINT_3D_WEIGHTS="1.2,1.1,1.0,1.0,1.2,1.0,1.0"   # [link0,link2,link3,link4,link6,link7,hand]

# Iterative Refinement (joint_angle mode only)
USE_ITERATIVE_REFINEMENT=False  # Simplified mode: iterative refinement disabled
REFINEMENT_ITERATIONS=3        # Number of refinement iterations
REFINEMENT_WEIGHT=50.0         # Refinement loss weight

# Loss weights
HEATMAP_WEIGHT=1000.0
KP3D_WEIGHT=100.0
HEATMAP_ONLY_TRAIN=True  # True: train only 2D heatmap branch

# FDA (Fourier Domain Adaptation) for sim-to-real
FDA_REAL_DIR="/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/DREAM_real"  # Real images (no labels needed)
# FDA_REAL_DIR="/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/DREAM_real"
FDA_BETA=0.001   # Low-freq replacement ratio (0.01=subtle tone shift, 0.05=strong)
FDA_PROB=0.5   # Probability of applying FDA per sample (0.0 to disable)
OCCLUSION_PROB=0.2          # CoarseDropout probability for occlusion robustness
OCCLUSION_MAX_HOLES=3       # Maximum occlusion patches per image
OCCLUSION_MAX_SIZE_FRAC=0.2 # Max patch size ratio (image side fraction)

# Training hyperparameters
EPOCHS=50
BATCH_SIZE=16
NUM_WORKERS=4
OPTIMIZER="adam"  # Options: adam, adamw, sgd
LEARNING_RATE=1e-4
MIN_LR=1e-10
WEIGHT_DECAY=1e-5
SCHEDULER="cosine"  # Options: step, cosine, plateau, none
WARMUP_STEPS=200
WARMUP_START_LR=1e-10

# Loss configuration
LOSS_TYPE="smoothl1"  # Loss function type: mse, l1, smoothl1 (smoothl1 recommended for ADD AUC)
HARD_REPLAY=False     # Disable hard-batch replay when re-training selected json list
JSON_LIST_EXTRA_FINETUNE=False  # True: 1차 전체학습 후, json list만 2차 파인튜닝
JSON_FINETUNE_EPOCHS=1         # 2차(json list) 추가 학습 epoch
JSON_FINETUNE_LR=""             # Optional override LR for 2차 (empty=기본 LR 유지)
JSON_FINETUNE_HARD_REPLAY=False # 2차에서는 hard replay 보통 비권장
JSON_FINETUNE_OUTPUT_SUFFIX="_jsonlist_ft"

# Output and logging
OUTPUT_DIR="./outputs/dinov3_base_$(date +%Y%m%d_%H%M%S)"
WANDB_PROJECT="dinov3-pose-estimation"
WANDB_RUN_NAME="dinov3_base_$(date +%Y%m%d_%H%M%S)"

# Other settings
SEED=42
RESUME="/home/najo/NAS/DIP/DINObotPose2/Train/outputs/dinov3_base_20260302_180056/epoch_1.pth"  # Path to checkpoint for resuming (leave empty for new training)
RESUME_LR=""  # Learning rate to use when resuming (leave empty for automatic calculation from scheduler)
LOAD_2D_HEAD=""  # Path to checkpoint for loading pretrained 2D heatmap head (leave empty to train from scratch)
FREEZE_2D_HEAD_EPOCHS=50  # If LOAD_2D_HEAD is set, freeze loaded 2D head for first N epochs then unfreeze

# =============================================================================
# Training Modes
# =============================================================================

# Choose training mode by uncommenting one of the following:

# --- Single GPU Training ---
# TRAIN_MODE="single_gpu"

# --- Multi-GPU Training (Distributed Data Parallel) ---
TRAIN_MODE="multi_gpu"
NUM_GPUS=5  # 사용할 GPU 개수 (single GPU는 1로 설정)
GPU_IDS="0,1,2,3,4"  # 사용할 GPU ID (예: "0,1,2,3")

# =============================================================================
# Execute Training
# =============================================================================

# Build joint embedding flag
if [ "${USE_JOINT_EMBEDDING}" = "True" ] || [ "${USE_JOINT_EMBEDDING}" = "true" ]; then
    JOINT_EMBEDDING_FLAG="--use-joint-embedding"
else
    JOINT_EMBEDDING_FLAG=""
fi

# Build joint7-fix flag
if [ "${FIX_JOINT7_ZERO}" = "True" ] || [ "${FIX_JOINT7_ZERO}" = "true" ]; then
    FIX_JOINT7_FLAG="--fix-joint7-zero"
else
    FIX_JOINT7_FLAG=""
fi

# Build iterative refinement flag
if [ "${USE_ITERATIVE_REFINEMENT}" = "True" ] || [ "${USE_ITERATIVE_REFINEMENT}" = "true" ]; then
    REFINEMENT_FLAG="--use-iterative-refinement --refinement-iterations ${REFINEMENT_ITERATIONS} --refinement-weight ${REFINEMENT_WEIGHT}"
else
    REFINEMENT_FLAG=""
fi

# Build heatmap-only training flag
if [ "${HEATMAP_ONLY_TRAIN}" = "True" ] || [ "${HEATMAP_ONLY_TRAIN}" = "true" ]; then
    HEATMAP_ONLY_FLAG="--heatmap-only-train"
else
    HEATMAP_ONLY_FLAG=""
fi

# Build hard-replay flag
if [ "${HARD_REPLAY}" = "True" ] || [ "${HARD_REPLAY}" = "true" ]; then
    HARD_REPLAY_FLAG="--hard-replay"
else
    HARD_REPLAY_FLAG="--no-hard-replay"
fi

# Validate training directories
if [ ${#DATA_DIRS[@]} -eq 0 ]; then
    echo "Error: DATA_DIRS is empty. Set at least one training directory."
    exit 1
fi
for d in "${DATA_DIRS[@]}"; do
    if [ ! -d "${d}" ]; then
        echo "Error: Training directory does not exist: ${d}"
        exit 1
    fi
done

# Base args shared by all phases
COMMON_ARGS="\
    --data-dir ${DATA_DIRS[*]} \
    --train-split ${TRAIN_SPLIT} \
    --model-name ${MODEL_NAME} \
    --image-size ${IMAGE_SIZE} \
    --heatmap-size ${HEATMAP_SIZE} \
    --unfreeze-blocks ${UNFREEZE_BLOCKS} \
    ${JOINT_EMBEDDING_FLAG} \
    ${FIX_JOINT7_FLAG} \
    ${HEATMAP_ONLY_FLAG} \
    --angle-weight ${ANGLE_WEIGHT} \
    --fk-3d-weight ${FK_3D_WEIGHT} \
    --direct-3d-weight ${DIRECT_3D_WEIGHT} \
    --consistency-weight ${CONSISTENCY_WEIGHT} \
    --fusion-delta-weight ${FUSION_DELTA_WEIGHT} \
    $([ -n "${KEYPOINT_3D_WEIGHTS}" ] && echo "--keypoint-3d-weights ${KEYPOINT_3D_WEIGHTS}") \
    ${REFINEMENT_FLAG} \
    --batch-size ${BATCH_SIZE} \
    --num-workers ${NUM_WORKERS} \
    --optimizer ${OPTIMIZER} \
    --learning-rate ${LEARNING_RATE} \
    --min-lr ${MIN_LR} \
    --weight-decay ${WEIGHT_DECAY} \
    --scheduler ${SCHEDULER} \
    --warmup-steps ${WARMUP_STEPS} \
    --warmup-start-lr ${WARMUP_START_LR} \
    --occlusion-prob ${OCCLUSION_PROB} \
    --occlusion-max-holes ${OCCLUSION_MAX_HOLES} \
    --occlusion-max-size-frac ${OCCLUSION_MAX_SIZE_FRAC} \
    --freeze-2d-head-epochs ${FREEZE_2D_HEAD_EPOCHS} \
    --loss-type ${LOSS_TYPE} \
    --heatmap-weight ${HEATMAP_WEIGHT} \
    --kp3d-weight ${KP3D_WEIGHT} \
    --wandb-project ${WANDB_PROJECT} \
    --seed ${SEED}"

if [ -n "${VAL_DIR}" ]; then
    COMMON_ARGS="${COMMON_ARGS} --val-dir ${VAL_DIR} --val-split ${VAL_SPLIT}"
fi

if [ -n "${FDA_REAL_DIR}" ] && [ "${FDA_PROB}" != "0.0" ]; then
    COMMON_ARGS="${COMMON_ARGS} --fda-real-dir ${FDA_REAL_DIR} --fda-beta ${FDA_BETA} --fda-prob ${FDA_PROB}"
fi

run_training_phase() {
    local phase_name="$1"
    local phase_output_dir="$2"
    local phase_epochs="$3"
    local phase_resume="$4"
    local phase_resume_lr="$5"
    local phase_train_json_list="$6"
    local phase_val_json_list="$7"
    local phase_hard_replay="$8"
    local phase_wandb_name="$9"
    local phase_load_2d_head="${10}"
    local phase_lr_override="${11}"

    local phase_hard_replay_flag="--no-hard-replay"
    if [ "${phase_hard_replay}" = "True" ] || [ "${phase_hard_replay}" = "true" ]; then
        phase_hard_replay_flag="--hard-replay"
    fi

    local phase_args="${COMMON_ARGS} \
        --epochs ${phase_epochs} \
        --output-dir ${phase_output_dir} \
        ${phase_hard_replay_flag} \
        $([ -n "${phase_train_json_list}" ] && echo "--train-json-list ${phase_train_json_list}") \
        --train-json-list-mode ${TRAIN_JSON_LIST_MODE} \
        --train-json-extra-loss-scale ${TRAIN_JSON_EXTRA_LOSS_SCALE} \
        $([ -n "${phase_val_json_list}" ] && echo "--val-json-list ${phase_val_json_list}") \
        $([ -n "${phase_wandb_name}" ] && echo "--wandb-run-name ${phase_wandb_name}") \
        $([ -n "${phase_resume}" ] && echo "--resume ${phase_resume}") \
        $([ -n "${phase_resume_lr}" ] && echo "--resume-lr ${phase_resume_lr}") \
        $([ -n "${phase_load_2d_head}" ] && echo "--load-2d-head ${phase_load_2d_head}") \
        $([ -n "${phase_lr_override}" ] && echo "--learning-rate ${phase_lr_override}")"

    echo "============================================================"
    echo "Phase: ${phase_name}"
    echo "Output directory: ${phase_output_dir}"
    echo "Train data dirs:"
    for d in "${DATA_DIRS[@]}"; do
        echo "  - ${d}"
    done
    echo "Train JSON list: ${phase_train_json_list:-<none>}"
    echo "============================================================"

    if [ "${TRAIN_MODE}" = "single_gpu" ]; then
        eval "python train.py ${phase_args}"
    elif [ "${TRAIN_MODE}" = "multi_gpu" ]; then
        eval "torchrun --standalone --nnodes=1 --nproc_per_node=${NUM_GPUS} train.py ${phase_args}"
    else
        echo "Error: Unknown training mode: ${TRAIN_MODE}"
        exit 1
    fi
}

echo "Using GPU(s): ${GPU_IDS}"
export CUDA_VISIBLE_DEVICES=${GPU_IDS}

PHASE1_TRAIN_JSON_LIST="${TRAIN_JSON_LIST}"
PHASE1_VAL_JSON_LIST="${VAL_JSON_LIST}"
if [ "${JSON_LIST_EXTRA_FINETUNE}" = "True" ] || [ "${JSON_LIST_EXTRA_FINETUNE}" = "true" ]; then
    # 1차는 전체 데이터 학습으로 유지하고, 2차에서만 json list 집중 학습
    PHASE1_TRAIN_JSON_LIST=""
    PHASE1_VAL_JSON_LIST=""
fi

# Phase 1
run_training_phase \
    "base_training" \
    "${OUTPUT_DIR}" \
    "${EPOCHS}" \
    "${RESUME}" \
    "${RESUME_LR}" \
    "${PHASE1_TRAIN_JSON_LIST}" \
    "${PHASE1_VAL_JSON_LIST}" \
    "${HARD_REPLAY}" \
    "${WANDB_RUN_NAME}" \
    "${LOAD_2D_HEAD}" \
    ""

# Phase 2 (optional): fine-tune only selected json list
if [ "${JSON_LIST_EXTRA_FINETUNE}" = "True" ] || [ "${JSON_LIST_EXTRA_FINETUNE}" = "true" ]; then
    if [ -z "${TRAIN_JSON_LIST}" ]; then
        echo "JSON_LIST_EXTRA_FINETUNE=True but TRAIN_JSON_LIST is empty. Skipping phase 2."
    else
        PHASE2_RESUME="${OUTPUT_DIR}/best_model.pth"
        if [ ! -f "${PHASE2_RESUME}" ]; then
            echo "Warning: best_model.pth not found at ${PHASE2_RESUME}, trying last epoch checkpoint."
            PHASE2_RESUME="$(ls -1 ${OUTPUT_DIR}/epoch_*.pth 2>/dev/null | tail -n 1 || true)"
        fi
        if [ -z "${PHASE2_RESUME}" ] || [ ! -f "${PHASE2_RESUME}" ]; then
            echo "Error: cannot find phase1 checkpoint for phase2 resume."
            exit 1
        fi

        PHASE2_OUTPUT_DIR="${OUTPUT_DIR}${JSON_FINETUNE_OUTPUT_SUFFIX}"
        PHASE2_WANDB_NAME=""
        if [ -n "${WANDB_RUN_NAME}" ]; then
            PHASE2_WANDB_NAME="${WANDB_RUN_NAME}${JSON_FINETUNE_OUTPUT_SUFFIX}"
        fi

        run_training_phase \
            "json_list_finetune" \
            "${PHASE2_OUTPUT_DIR}" \
            "${JSON_FINETUNE_EPOCHS}" \
            "${PHASE2_RESUME}" \
            "" \
            "${TRAIN_JSON_LIST}" \
            "${VAL_JSON_LIST}" \
            "${JSON_FINETUNE_HARD_REPLAY}" \
            "${PHASE2_WANDB_NAME}" \
            "" \
            "${JSON_FINETUNE_LR}"

        echo "Phase 2 completed. Results saved to: ${PHASE2_OUTPUT_DIR}"
    fi
fi

echo "Training completed!"
echo "Phase 1 results: ${OUTPUT_DIR}"
