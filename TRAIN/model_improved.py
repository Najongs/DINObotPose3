import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import (
    DINOv3Backbone, ViTKeypointHead, soft_argmax_2d, 
    panda_forward_kinematics, _PANDA_JOINT_LIMITS,
    solve_pnp_batch, solve_pnp_ransac_batch, solve_pnp_conf_batch
)

class ImprovedJointAngleHead(nn.Module):
    """
    개선 사항:
    1. Joint 0 전용 branch 추가 (더 많은 context)
    2. Relative angle encoding (joint 간 관계 학습)
    3. Confidence-aware weighting
    """
    def __init__(self, input_dim=768, num_joints=7, num_angles=7):
        super().__init__()
        self.num_joints = num_joints
        self.num_angles = num_angles
        self.hidden_dim = 512
        self.n_iter = 4

        limits = torch.tensor(_PANDA_JOINT_LIMITS, dtype=torch.float32)[:self.num_angles]
        self.register_buffer('joint_lower', limits[:, 0])
        self.register_buffer('joint_upper', limits[:, 1])

        # 1. Feature extraction
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU()
        )

        # 2. Heatmap spatial encoding
        self.heatmap_encoder = nn.Sequential(
            nn.Linear(num_joints * 2, 256), nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, self.hidden_dim)
        )

        # 🚀 [NEW] 3. Joint 0 전용 branch (더 깊은 네트워크)
        self.joint0_branch = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, 2)  # cos, sin for joint 0
        )

        # 4. Initial prediction (joints 1-6)
        self.initial_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, (num_angles - 1) * 2),  # Exclude joint 0
        )

        # 5. Refinement (all joints)
        self.refine_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 2 + num_angles * 2, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
        )
        self.angle_delta = nn.Linear(1024, num_angles)
        self.dropout = nn.Dropout(p=0.1)

        nn.init.xavier_uniform_(self.angle_delta.weight, gain=0.01)
        if self.angle_delta.bias is not None:
            nn.init.zeros_(self.angle_delta.bias)

    def forward(self, dino_features, predicted_heatmaps, camera_K=None):
        B = dino_features.shape[0]
        device = dino_features.device

        # Global + spatial features
        global_feat = F.adaptive_avg_pool2d(
            dino_features.permute(0, 2, 1).reshape(B, dino_features.shape[2],
                                                    int(math.sqrt(dino_features.shape[1])),
                                                    int(math.sqrt(dino_features.shape[1]))), 1
        ).flatten(1)
        global_feat = self.feature_proj(global_feat)

        with torch.no_grad():
            uv_heatmap = soft_argmax_2d(predicted_heatmaps, temperature=100.0)
        hm_h, hm_w = predicted_heatmaps.shape[2:]
        u_norm = uv_heatmap[:, :, 0] / hm_w
        v_norm = uv_heatmap[:, :, 1] / hm_h
        spatial_info = torch.cat([u_norm, v_norm], dim=-1)
        spatial_feat = self.heatmap_encoder(spatial_info)

        feat_combined = torch.cat([global_feat, spatial_feat], dim=1)

        # 🚀 [NEW] Joint 0 전용 예측
        joint0_sc = self.joint0_branch(feat_combined)  # (B, 2)
        cos0 = joint0_sc[:, 0:1]
        sin0 = joint0_sc[:, 1:2]
        norm0 = torch.sqrt(cos0**2 + sin0**2).clamp(min=1e-8)
        cos0 = cos0 / norm0
        sin0 = sin0 / norm0

        # Joints 1-6 예측
        pred_sc_rest = self.initial_mlp(feat_combined)  # (B, 12)
        cos_rest = pred_sc_rest[:, 0::2]
        sin_rest = pred_sc_rest[:, 1::2]
        norm_rest = torch.sqrt(cos_rest**2 + sin_rest**2).clamp(min=1e-8)
        cos_rest = cos_rest / norm_rest
        sin_rest = sin_rest / norm_rest

        # Combine: [joint0, joints1-6]
        pred_cos = torch.cat([cos0, cos_rest], dim=1)  # (B, 7)
        pred_sin = torch.cat([sin0, sin_rest], dim=1)
        pred_sc = torch.stack([pred_cos, pred_sin], dim=2).reshape(B, self.num_angles * 2)

        # Refinement iterations
        for iter_step in range(self.n_iter - 1):
            combined = torch.cat([global_feat, spatial_feat, pred_sc], dim=1)
            x = self.refine_mlp(combined)
            x = self.dropout(x)
            delta = self.angle_delta(x)

            cos_prev = pred_sc[:, 0::2]
            sin_prev = pred_sc[:, 1::2]
            cos_new = cos_prev - sin_prev * delta
            sin_new = sin_prev + cos_prev * delta
            norm = torch.sqrt(cos_new**2 + sin_new**2).clamp(min=1e-8)
            cos_new = cos_new / norm
            sin_new = sin_new / norm
            pred_sc = torch.stack([cos_new, sin_new], dim=2).reshape(B, self.num_angles * 2)

        cos_final = pred_sc[:, 0::2]
        sin_final = pred_sc[:, 1::2]
        pred_angles = torch.atan2(sin_final, cos_final)

        return pred_angles, panda_forward_kinematics(
            pred_angles if self.num_angles == 7 else torch.cat([pred_angles, torch.zeros(B, 7-self.num_angles, device=device)], dim=1)
        ), pred_sc


class DINOv3PoseEstimatorImproved(nn.Module):
    def __init__(self, dino_model_name, heatmap_size, unfreeze_blocks=2, fix_joint7_zero=False):
        super().__init__()
        self.dino_model_name, self.heatmap_size, self.fix_joint7_zero = dino_model_name, heatmap_size, fix_joint7_zero
        self.backbone = DINOv3Backbone(dino_model_name, unfreeze_blocks=unfreeze_blocks)
        feat_dim = self.backbone.model.config.hidden_size if "conv" not in dino_model_name else self.backbone.model.config.hidden_sizes[-1]
        self.keypoint_head = ViTKeypointHead(input_dim=feat_dim, heatmap_size=heatmap_size)
        self.joint_angle_head = ImprovedJointAngleHead(input_dim=feat_dim, num_joints=7, num_angles=6 if fix_joint7_zero else 7)

    def forward(self, image_tensor_batch, camera_K=None, **kwargs):
        dino_features = self.backbone(image_tensor_batch)
        predicted_heatmaps = self.keypoint_head(dino_features)
        joint_angles, kpts_3d_robot, pred_sin_cos = self.joint_angle_head(dino_features, predicted_heatmaps, camera_K=camera_K)
        
        if self.fix_joint7_zero:
            zeros = torch.zeros(joint_angles.shape[0], 1, device=joint_angles.device)
            joint_angles = torch.cat([joint_angles, zeros], dim=1)
            kpts_3d_robot = panda_forward_kinematics(joint_angles)
            pad = torch.tensor([[1.0, 0.0]], device=pred_sin_cos.device).expand(pred_sin_cos.shape[0], -1)
            pred_sin_cos = torch.cat([pred_sin_cos, pad], dim=1)

        result = {
            'heatmaps_2d': predicted_heatmaps,
            'joint_angles': joint_angles,
            'keypoints_3d_fk': kpts_3d_robot,
            'keypoints_3d': kpts_3d_robot,
            'keypoints_3d_robot': kpts_3d_robot,
            'pred_sin_cos': pred_sin_cos
        }

        if camera_K is not None:
            uv_2d = soft_argmax_2d(predicted_heatmaps)
            B, N, H, W = predicted_heatmaps.shape
            hm_flat = predicted_heatmaps.reshape(B, N, -1)
            kp_confidence = hm_flat.max(dim=-1)[0]
            result['kp_confidence'] = kp_confidence

            kp_3d_cam, pnp_valid, reproj_errors = solve_pnp_batch(uv_2d, kpts_3d_robot, camera_K)
            result['keypoints_3d_cam'] = kp_3d_cam
            result['pnp_valid'] = pnp_valid
            result['reproj_errors'] = reproj_errors

            kp_3d_cam_r, pnp_valid_r, reproj_errors_r, n_inliers_r = solve_pnp_ransac_batch(uv_2d, kpts_3d_robot, camera_K)
            result['keypoints_3d_cam_ransac'] = kp_3d_cam_r
            result['pnp_valid_ransac'] = pnp_valid_r
            result['reproj_errors_ransac'] = reproj_errors_r
            result['pnp_n_inliers_ransac'] = n_inliers_r

            kp_3d_cam_cf, pnp_valid_cf, reproj_errors_cf, n_inliers_cf = solve_pnp_conf_batch(
                uv_2d, kpts_3d_robot, camera_K, kp_confidence, min_kp=4
            )
            result['keypoints_3d_cam_conf'] = kp_3d_cam_cf
            result['pnp_valid_conf'] = pnp_valid_cf
            result['reproj_errors_conf'] = reproj_errors_cf
            result['pnp_n_used_conf'] = n_inliers_cf

        return result
