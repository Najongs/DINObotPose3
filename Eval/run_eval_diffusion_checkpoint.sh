python eval_diffusion_checkpoint.py \
--data-dir /home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/Converted_dataset/DREAM_to_DREAM/panda-3cam_azure \
--checkpoint /home/najo/NAS/DIP/DINObotPose3/TRAIN/outputs_diffusion/train_20260308_212410/best_diffusion.pth \
--output-dir /home/najo/NAS/DIP/DINObotPose3/Eval/eval_diffusion_best \
--batch-size 16 \
--num-workers 4