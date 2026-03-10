#!/bin/bash

# Configuration
TEST_DIR="/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/Converted_dataset/DREAM_to_DREAM/panda-3cam_azure"
CHECKPOINT="/home/najo/NAS/DIP/DINObotPose3/TRAIN/outputs_3d/train_3d_v3_20260309_154352/best_joint_angle.pth"
OUTPUT_DIR="./results_3d_v3"
BATCH_SIZE=160
NUM_WORKERS=20

export CUDA_VISIBLE_DEVICES="0,1,2,3,4"

# Run Evaluation
echo "============================================================================="
echo "==> STARTING 3D POSE (V3) EVALUATION"
echo "==> Dataset: ${TEST_DIR}"
echo "==> Checkpoint: ${CHECKPOINT}"
echo "============================================================================="

python eval_3d_v3.py \
    --test-dir "${TEST_DIR}" \
    --checkpoint "${CHECKPOINT}" \
    --output-dir "${OUTPUT_DIR}" \
    --batch-size ${BATCH_SIZE} \
    --num-workers ${NUM_WORKERS}

echo "==> Evaluation Completed!"


# export CUDA_VISIBLE_DEVICES="0,1,2,3,4"
# python eval_diffusion_checkpoint.py \
#     --data-dir /home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/Converted_dataset/DREAM_to_DREAM/panda-3cam_azure \
#     --checkpoint /home/najo/NAS/DIP/DINObotPose3/TRAIN/outputs_diffusion/train_20260308_212410/epoch_060.pth \
#     --output-dir results_diffusion_real \
#     --batch-size 32 \
#     --num-workers 4
