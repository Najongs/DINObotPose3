import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from torchvision.ops import roi_align
import numpy as np
import cv2

FEATURE_DIM = 512
NUM_JOINTS = 7  # DO NOT CHANGE: This value is intentionally set to 7.


def solve_pnp_batch(kp_2d, kp_3d_robot, camera_K):
    """
    Solve PnP to find camera extrinsic and transform robot-frame 3D to camera-frame.

    Args:
        kp_2d: (B, N, 2) - soft-argmax 2D pixel coordinates
        kp_3d_robot: (B, N, 3) - FK robot-frame 3D keypoints
        camera_K: (B, 3, 3) - camera intrinsic matrix

    Returns:
        kp_3d_cam: (B, N, 3) - camera-frame 3D keypoints
        valid_mask: (B,) - PnP success flag per batch
    """
    B = kp_2d.shape[0]
    results = []
    valids = []

    for b in range(B):
        pts2d = kp_2d[b].detach().cpu().numpy().astype(np.float64)  # (N, 2)
        pts3d = kp_3d_robot[b].detach().cpu().numpy().astype(np.float64)  # (N, 3)
        K = camera_K[b].detach().cpu().numpy().astype(np.float64)  # (3, 3)

        try:
            success, rvec, tvec = cv2.solvePnP(
                pts3d, pts2d, K, None,
                flags=cv2.SOLVEPNP_ITERATIVE,
                useExtrinsicGuess=False,
                rvec=np.zeros(3),
                tvec=np.zeros(3)
            )

            if success:
                R, _ = cv2.Rodrigues(rvec)  # (3, 3)
                # Transform: p_cam = R @ p_robot + t
                kp_cam = (pts3d @ R.T) + tvec.T  # (N, 3)
                results.append(torch.from_numpy(kp_cam).float())
                valids.append(True)
            else:
                results.append(torch.zeros_like(kp_3d_robot[b]))
                valids.append(False)
        except Exception as e:
            results.append(torch.zeros_like(kp_3d_robot[b]))
            valids.append(False)

    kp_3d_cam = torch.stack(results, dim=0).to(kp_2d.device)
    valid_mask = torch.tensor(valids, device=kp_2d.device, dtype=torch.bool)
    return kp_3d_cam, valid_mask

# 3D prediction modes
MODE_JOINT_ANGLE = 'joint_angle'  # Predict joint angles → FK → robot-frame 3D keypoints
MODE_DIRECT_3D = 'direct_3d'      # Directly predict 3D keypoints from features


def soft_argmax_2d(heatmaps, temperature=10.0):
    """
    Differentiable soft-argmax to extract (u, v) from heatmaps.
    """
    B, N, H, W = heatmaps.shape
    device = heatmaps.device

    x_coords = torch.arange(W, device=device, dtype=torch.float32)
    y_coords = torch.arange(H, device=device, dtype=torch.float32)

    heatmaps_flat = heatmaps.reshape(B, N, -1)
    if isinstance(temperature, torch.Tensor):
        temperature = temperature.clamp(min=0.1, max=50.0)

    weights = F.softmax(heatmaps_flat * temperature, dim=-1)
    weights = weights.reshape(B, N, H, W)

    x = (weights.sum(dim=2) * x_coords).sum(dim=-1)  
    y = (weights.sum(dim=3) * y_coords).sum(dim=-1)  

    return torch.stack([x, y], dim=-1)  # (B, N, 2)

class DINOv3Backbone(nn.Module):
    def __init__(self, model_name, unfreeze_blocks=2):
        super().__init__()
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name)

        # Freeze backbone parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Unfreeze last N blocks for fine-tuning
        if unfreeze_blocks > 0:
            if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "layers"):
                layers = self.model.encoder.layers
                for i in range(len(layers) - unfreeze_blocks, len(layers)):
                    for param in layers[i].parameters():
                        param.requires_grad = True
            elif hasattr(self.model, "blocks"):
                layers = self.model.blocks
                for i in range(len(layers) - unfreeze_blocks, len(layers)):
                    for param in layers[i].parameters():
                        param.requires_grad = True

    def forward(self, image_tensor_batch):
        if "siglip" in self.model_name:
            outputs = self.model(pixel_values=image_tensor_batch, interpolate_pos_encoding=True)
            tokens = outputs.last_hidden_state
            patch_tokens = tokens[:, 1:, :]
        else: # DINOv3 계열
            outputs = self.model(pixel_values=image_tensor_batch)
            tokens = outputs.last_hidden_state
            num_reg = int(getattr(self.model.config, "num_register_tokens", 0))
            patch_tokens = tokens[:, 1 + num_reg :, :]
        return patch_tokens

class AdaptiveNorm2d(nn.Module):
    def __init__(self, num_channels, num_groups=32):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups, num_channels)
        self.ln = nn.LayerNorm(num_channels)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        gn_out = self.gn(x)
        ln_out = self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        alpha = torch.sigmoid(self.alpha)
        return alpha * gn_out + (1 - alpha) * ln_out


class TokenFuser(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.refine_blocks = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            AdaptiveNorm2d(out_channels, num_groups=32),
            nn.GELU(),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            AdaptiveNorm2d(out_channels, num_groups=32)
        )
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        projected = self.projection(x)
        refined = self.refine_blocks(projected)
        residual = self.residual_conv(x)
        return torch.nn.functional.gelu(refined + residual)


class SpatialGlobalModulation(nn.Module):
    def __init__(self, global_dim, feature_dim, dropout_p=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(global_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(feature_dim * 2, feature_dim * 2)
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x, global_context):
        gamma_beta = self.mlp(global_context)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1 + gamma) + beta

class ViTKeypointHead(nn.Module):
    def __init__(self, input_dim=768, num_joints=7, heatmap_size=(512, 512)):
        super().__init__()
        self.heatmap_size = heatmap_size
        self.token_fuser = TokenFuser(input_dim, 256)
        self.global_mod1 = SpatialGlobalModulation(global_dim=input_dim, feature_dim=256)
        self.global_mod2 = SpatialGlobalModulation(global_dim=input_dim, feature_dim=128)
        self.global_mod3 = SpatialGlobalModulation(global_dim=input_dim, feature_dim=64)
        self.spatial_dropout = nn.Dropout2d(p=0.1)

        self.decoder_block1 = nn.Sequential(
            nn.Conv2d(256, 128 * 4, kernel_size=3, padding=1, bias=False),  
            nn.PixelShuffle(upscale_factor=2),  
            AdaptiveNorm2d(128, num_groups=32),
            nn.GELU()
        )
        self.decoder_block2 = nn.Sequential(
            nn.Conv2d(128, 64 * 4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(upscale_factor=2),  
            AdaptiveNorm2d(64, num_groups=16),
            nn.GELU()
        )
        self.decoder_block3 = nn.Sequential(
            nn.Conv2d(64, 32 * 4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(upscale_factor=2),  
            AdaptiveNorm2d(32, num_groups=8),
            nn.GELU()
        )

        self.heatmap_predictor = nn.Conv2d(32, num_joints, kernel_size=3, padding=1)
        self.final_upsample = nn.Sequential(
            nn.Conv2d(num_joints, num_joints * 4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(upscale_factor=2),  
            nn.Conv2d(num_joints, num_joints * 4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(upscale_factor=2)   
        )

    def forward(self, dino_features):
        b, n, d = dino_features.shape
        h = w = int(math.sqrt(n))
        x = dino_features.permute(0, 2, 1).reshape(b, d, h, w)
        global_context = F.adaptive_avg_pool2d(x, 1).flatten(1)
        x = self.token_fuser(x)
        x = self.global_mod1(x, global_context)
        x = self.decoder_block1(x)
        x = self.global_mod2(x, global_context)
        x = self.decoder_block2(x)
        x = self.global_mod3(x, global_context)
        x = self.decoder_block3(x)
        x = self.spatial_dropout(x)
        heatmaps = self.heatmap_predictor(x)
        heatmaps = self.final_upsample(heatmaps)
        if heatmaps.shape[2:] != self.heatmap_size:
            heatmaps = F.interpolate(heatmaps, size=self.heatmap_size, mode='bilinear', align_corners=False)
        return heatmaps

# Forward Kinematics (Fixed for brevity)
def _rotation_matrix_z(theta):
    c, s = torch.cos(theta), torch.sin(theta)
    zero, one = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack([torch.stack([c, -s, zero], dim=-1), torch.stack([s, c, zero], dim=-1), torch.stack([zero, zero, one], dim=-1)], dim=-2)

def _make_transform(xyz, rpy):
    rx, ry, rz = rpy
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    R = [[cz*cy, cz*sy*sx - sz*cx, cz*sy*cx + sz*sx], [sz*cy, sz*sy*sx + cz*cx, sz*sy*cx - cz*sx], [-sy, cy*sx, cy*cx]]
    return [[R[0][0], R[0][1], R[0][2], xyz[0]], [R[1][0], R[1][1], R[1][2], xyz[1]], [R[2][0], R[2][1], R[2][2], xyz[2]], [0, 0, 0, 1]]

_PANDA_JOINTS = [{'xyz': (0, 0, 0.333), 'rpy': (0, 0, 0)}, {'xyz': (0, 0, 0), 'rpy': (-math.pi/2, 0, 0)}, {'xyz': (0, -0.316, 0), 'rpy': (math.pi/2, 0, 0)}, {'xyz': (0.0825, 0, 0), 'rpy': (math.pi/2, 0, 0)}, {'xyz': (-0.0825, 0.384, 0), 'rpy': (-math.pi/2, 0, 0)}, {'xyz': (0, 0, 0), 'rpy': (math.pi/2, 0, 0)}, {'xyz': (0.088, 0, 0), 'rpy': (math.pi/2, 0, 0)}]
_PANDA_FIXED_J8, _PANDA_FIXED_HAND = {'xyz': (0, 0, 0.107), 'rpy': (0, 0, 0)}, {'xyz': (0, 0, 0), 'rpy': (0, 0, -math.pi/4)}
_PANDA_JOINT_LIMITS = [(-2.8973, 2.8973), (-1.7628, 1.7628), (-2.8973, 2.8973), (-3.0718, -0.0698), (-2.8973, 2.8973), (-0.0175, 3.7525), (-2.8973, 2.8973)]

def panda_forward_kinematics(joint_angles):
    B = joint_angles.shape[0]; device, dtype = joint_angles.device, joint_angles.dtype
    fixed_transforms = [torch.tensor(_make_transform(j['xyz'], j['rpy']), device=device, dtype=dtype) for j in _PANDA_JOINTS]
    T_j8 = torch.tensor(_make_transform(_PANDA_FIXED_J8['xyz'], _PANDA_FIXED_J8['rpy']), device=device, dtype=dtype)
    T_hand = torch.tensor(_make_transform(_PANDA_FIXED_HAND['xyz'], _PANDA_FIXED_HAND['rpy']), device=device, dtype=dtype)
    cumul = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
    all_transforms = [cumul.clone()]
    for i in range(7):
        theta = joint_angles[:, i]
        R_joint = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1).clone()
        R_joint[:, :3, :3] = _rotation_matrix_z(theta)
        cumul = cumul @ fixed_transforms[i].unsqueeze(0) @ R_joint
        all_transforms.append(cumul.clone())
    cumul_j8 = cumul @ T_j8.unsqueeze(0); all_transforms.append(cumul_j8.clone())
    cumul_hand = cumul_j8 @ T_hand.unsqueeze(0); all_transforms.append(cumul_hand.clone())
    kp_indices = [0, 2, 3, 4, 6, 7, 9]; keypoints = [all_transforms[idx][:, :3, 3] for idx in kp_indices]
    return torch.stack(keypoints, dim=1)

class Direct3DHead(nn.Module):
    """
    Directly regresses 3D keypoint coordinates from visual features + heatmaps.
    [V2 IMPROVED]: Residual connections, LayerNorm, Confidence weighting for better generalization.
    """

    def __init__(self, input_dim=768, num_joints=7, heatmap_size=(512, 512)):
        super().__init__()
        self.num_joints = num_joints
        self.hidden_dim = 256
        self.heatmap_size = heatmap_size

        # Feature extraction with LayerNorm for stability
        self.visual_projector = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )
        self.global_feature_net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.GELU(), nn.LayerNorm(256),
            nn.Dropout(0.2),
            nn.Linear(256, self.hidden_dim), nn.LayerNorm(self.hidden_dim)
        )

        # Heatmap spatial encoding (2D pixel + confidence) with residual
        self.spatial_encoder = nn.Sequential(
            nn.Linear(3, 128), nn.GELU(), nn.LayerNorm(128),
            nn.Dropout(0.1),
            nn.Linear(128, self.hidden_dim)
        )

        # Transformer decoder for spatial reasoning (더 강화된 버전)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim, nhead=8, dim_feedforward=512,
            dropout=0.2, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)

        # Multi-head attention for confidence weighting
        self.confidence_attention = nn.MultiheadAttention(
            self.hidden_dim, num_heads=4, dropout=0.1, batch_first=True
        )

        # 3D coordinate regressors (per joint) with deeper networks
        self.kp_3d_regressors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, 256), nn.GELU(), nn.LayerNorm(256),
                nn.Dropout(0.2),
                nn.Linear(256, 128), nn.GELU(), nn.LayerNorm(128),
                nn.Dropout(0.1),
                nn.Linear(128, 3)
            )
            for _ in range(num_joints)
        ])

    def forward(self, dino_features, predicted_heatmaps):
        """
        Args:
            dino_features: (B, N_tokens, D) - image features from backbone
            predicted_heatmaps: (B, num_joints, H_hm, W_hm)
        Returns:
            kpts_3d_robot: (B, num_joints, 3) - 3D keypoints in robot frame
        """
        B, N_tokens, D = dino_features.shape
        device = dino_features.device

        # Global context with improved feature extraction
        feat_map = dino_features.permute(0, 2, 1).reshape(B, D, int(math.sqrt(N_tokens)), int(math.sqrt(N_tokens)))
        global_pool = F.adaptive_avg_pool2d(feat_map, 1).flatten(1)
        global_token = self.global_feature_net(global_pool).unsqueeze(1)  # (B, 1, hidden_dim)

        # Extract spatial info from heatmaps
        hm_h, hm_w = predicted_heatmaps.shape[2:]
        max_vals, max_indices = torch.max(predicted_heatmaps.view(B, self.num_joints, -1), dim=-1)

        u = (max_indices % hm_w).float()
        v = (max_indices // hm_w).float()
        conf = torch.sigmoid(max_vals)

        # Normalize coordinates to [0, 1]
        u_norm = u / hm_w
        v_norm = v / hm_h

        spatial_info = torch.stack([u_norm, v_norm, conf], dim=-1)  # (B, num_joints, 3)
        spatial_encoded = self.spatial_encoder(spatial_info)  # (B, num_joints, hidden_dim)

        # Extract visual features per joint (heatmap-weighted pooling at feature resolution)
        h_feat = w_feat = int(math.sqrt(N_tokens))
        hm_small = F.interpolate(predicted_heatmaps, size=(h_feat, w_feat), mode='bilinear', align_corners=False)
        hm_flat = hm_small.view(B, self.num_joints, -1)
        hm_weights = F.softmax(hm_flat * 10.0, dim=-1)  # (B, num_joints, HW_feat)

        joint_features = torch.bmm(hm_weights, dino_features)  # (B, num_joints, D)
        joint_visual = self.visual_projector(joint_features)  # (B, num_joints, hidden_dim)

        # 🚀 [개선] Residual connection + Confidence weighting
        joint_tokens = joint_visual + spatial_encoded  # (B, num_joints, hidden_dim)

        # Apply confidence weighting via attention
        joint_tokens_weighted, _ = self.confidence_attention(joint_tokens, joint_tokens, joint_tokens,
                                                              key_padding_mask=None)
        joint_tokens = joint_tokens + joint_tokens_weighted  # Residual connection

        # Transformer decoding with global context as memory
        memory = global_token  # (B, 1, hidden_dim)
        tgt = joint_tokens   # (B, num_joints, hidden_dim)
        decoded = self.transformer_decoder(tgt, memory)  # (B, num_joints, hidden_dim)

        # 🚀 [개선] Residual connection from input
        decoded = decoded + joint_tokens

        # Regress 3D coordinates per joint with confidence weighting
        kpts_3d_list = []
        for i, regressor in enumerate(self.kp_3d_regressors):
            kp = regressor(decoded[:, i, :])  # (B, 3)
            # Weight by confidence: high confidence → trust more
            kp_weighted = kp * conf[:, i:i+1]  # (B, 3) weighted by confidence
            kpts_3d_list.append(kp_weighted)

        kpts_3d = torch.stack(kpts_3d_list, dim=1)  # (B, num_joints, 3)

        return kpts_3d


class IterativeJointAngleHead(nn.Module):
    """
    Iterative angle refinement (inspired by RoboPEPP).
    Features → Initial angles → Refinement loop (4 iterations) with residual updates.
    """

    def __init__(self, input_dim=768, num_joints=7, num_angles=7):
        super().__init__()
        self.num_joints = num_joints
        self.num_angles = num_angles
        self.hidden_dim = 512
        self.n_iter = 4  # RoboPEPP: 4 iterations for refinement

        limits = torch.tensor(_PANDA_JOINT_LIMITS, dtype=torch.float32)[:self.num_angles]
        self.register_buffer('joint_lower', limits[:, 0])
        self.register_buffer('joint_upper', limits[:, 1])

        # 1. Feature extraction from DINOv3
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU()
        )

        # 2. Heatmap-based spatial encoding
        self.heatmap_encoder = nn.Sequential(
            nn.Linear(2, 64), nn.GELU(),  # (u/W, v/H)
            nn.LayerNorm(64),
            nn.Linear(64, self.hidden_dim)
        )

        # 3. Iterative refinement: [features + current_sin_cos] → delta_angles
        # combined = [global_feat(512) + spatial_feat(512) + pred_sc(num_angles*2)]
        self.fc_1 = nn.Linear(self.hidden_dim * 2 + num_angles * 2, 1024)
        self.fc_2 = nn.Linear(1024, 1024)
        self.angle_delta = nn.Linear(1024, num_angles)

        self.drop1 = nn.Dropout(p=0.3)
        self.drop2 = nn.Dropout(p=0.3)

        # Small init for angle delta (start with near-zero changes)
        nn.init.xavier_uniform_(self.angle_delta.weight, gain=0.001)
        if self.angle_delta.bias is not None:
            nn.init.zeros_(self.angle_delta.bias)

    def forward(self, dino_features, predicted_heatmaps, camera_K=None):
        """
        Iterative joint angle refinement with 4 iterations.
        Output: [cos(θ), sin(θ)] pairs for each joint (more stable than angles directly)
        """
        B = dino_features.shape[0]
        device = dino_features.device

        # 1. Global feature extraction
        global_feat = F.adaptive_avg_pool2d(
            dino_features.permute(0, 2, 1).reshape(B, dino_features.shape[2],
                                                    int(math.sqrt(dino_features.shape[1])),
                                                    int(math.sqrt(dino_features.shape[1]))), 1
        ).flatten(1)
        global_feat = self.feature_proj(global_feat)  # (B, hidden_dim)

        # 2. Spatial features from heatmaps (u, v normalized)
        uv_heatmap = soft_argmax_2d(predicted_heatmaps, temperature=10.0)
        hm_h, hm_w = predicted_heatmaps.shape[2:]
        u_norm = uv_heatmap[:, :, 0] / hm_w  # (B, num_joints)
        v_norm = uv_heatmap[:, :, 1] / hm_h

        # Pool spatial features (average across joints)
        spatial_info = torch.stack([u_norm, v_norm], dim=-1)  # (B, num_joints, 2)
        spatial_feat = self.heatmap_encoder(spatial_info.mean(dim=1))  # (B, hidden_dim)

        # 3. Iterative sin/cos refinement (more stable than angle)
        # Initialize with identity (angle=0): [cos(0), sin(0)] = [1, 0]
        pred_sc = torch.zeros(B, self.num_angles * 2, device=device)
        pred_sc[:, 0::2] = 1.0  # cos initialized to 1
        pred_sc[:, 1::2] = 0.0  # sin initialized to 0

        for iter_step in range(self.n_iter):
            # Concatenate: [global_feat, spatial_feat, current_sin_cos]
            combined = torch.cat([global_feat, spatial_feat, pred_sc], dim=1)

            # MLP refinement
            x = self.fc_1(combined)
            x = self.drop1(x)
            x = self.fc_2(x)
            x = self.drop2(x)

            # Sin/cos delta (residual)
            delta = self.angle_delta(x)  # (B, num_angles)

            cos_prev = pred_sc[:, 0::2]  # (B, num_angles)
            sin_prev = pred_sc[:, 1::2]  # (B, num_angles)

            # Update using small angle approximation
            cos_new = cos_prev - sin_prev * delta
            sin_new = sin_prev + cos_prev * delta

            # Renormalize to unit circle (avoid division by zero)
            norm = torch.sqrt(cos_new**2 + sin_new**2).clamp(min=1e-8)
            cos_new = cos_new / norm
            sin_new = sin_new / norm

            # Interleave back to [cos, sin, cos, sin, ...]
            pred_sc = torch.stack([cos_new, sin_new], dim=2).reshape(B, self.num_angles * 2)

        # Convert sin/cos back to angles for FK
        cos_final = pred_sc[:, 0::2]  # (B, num_angles)
        sin_final = pred_sc[:, 1::2]  # (B, num_angles)
        pred_angles = torch.atan2(sin_final, cos_final)  # (B, num_angles)

        return pred_angles, panda_forward_kinematics(
            pred_angles if self.num_angles == 7 else torch.cat([pred_angles, torch.zeros(B, 7-self.num_angles, device=device)], dim=1)
        ), pred_sc  # Return sin/cos for loss computation


class DINOv3PoseEstimator(nn.Module):
    def __init__(self, dino_model_name, heatmap_size, unfreeze_blocks=2, fix_joint7_zero=False, mode=MODE_JOINT_ANGLE):
        super().__init__()
        self.dino_model_name, self.heatmap_size, self.fix_joint7_zero = dino_model_name, heatmap_size, fix_joint7_zero
        self.mode = mode
        self.backbone = DINOv3Backbone(dino_model_name, unfreeze_blocks=unfreeze_blocks)
        feat_dim = self.backbone.model.config.hidden_size if "conv" not in dino_model_name else self.backbone.model.config.hidden_sizes[-1]
        self.keypoint_head = ViTKeypointHead(input_dim=feat_dim, heatmap_size=heatmap_size)

        # 🚀 Mode-specific heads
        if mode == MODE_JOINT_ANGLE:
            self.joint_angle_head = IterativeJointAngleHead(input_dim=feat_dim, num_joints=NUM_JOINTS, num_angles=6 if fix_joint7_zero else 7)
        elif mode == MODE_DIRECT_3D:
            self.direct_3d_head = Direct3DHead(input_dim=feat_dim, num_joints=NUM_JOINTS, heatmap_size=heatmap_size)
        else:
            raise ValueError(f"Unknown mode: {mode}. Choose from {MODE_JOINT_ANGLE}, {MODE_DIRECT_3D}")

    def forward(self, image_tensor_batch, camera_K=None, **kwargs):
        dino_features = self.backbone(image_tensor_batch)
        predicted_heatmaps = self.keypoint_head(dino_features)

        # 🚀 Mode-specific forward pass
        if self.mode == MODE_JOINT_ANGLE:
            # Joint Angle mode: angles → FK → robot-frame 3D
            joint_angles, kpts_3d_robot, pred_sin_cos = self.joint_angle_head(dino_features, predicted_heatmaps, camera_K=camera_K)
            if self.fix_joint7_zero:
                joint_angles = joint_angles.clone()
                joint_angles[:, 6] = 0.0
                kpts_3d_robot = panda_forward_kinematics(joint_angles)
                # 🚀 Also fix sin/cos representation for consistency with joint_angles
                pred_sin_cos = pred_sin_cos.clone()
                pred_sin_cos[:, 12] = 1.0  # cos(0) = 1.0 for joint 6
                pred_sin_cos[:, 13] = 0.0  # sin(0) = 0.0 for joint 6

            result = {
                'heatmaps_2d': predicted_heatmaps,
                'joint_angles': joint_angles,
                'keypoints_3d_fk': kpts_3d_robot,
                'keypoints_3d': kpts_3d_robot,
                'keypoints_3d_robot': kpts_3d_robot,
                'pred_sin_cos': pred_sin_cos  # 🚀 For sin/cos loss
            }

            # PnP transformation if camera_K provided
            if camera_K is not None:
                uv_2d = soft_argmax_2d(predicted_heatmaps)
                kp_3d_cam, pnp_valid = solve_pnp_batch(uv_2d, kpts_3d_robot, camera_K)
                result['keypoints_3d_cam'] = kp_3d_cam
                result['pnp_valid'] = pnp_valid

        elif self.mode == MODE_DIRECT_3D:
            # Direct 3D mode: features + heatmaps → robot-frame 3D directly
            kpts_3d_robot = self.direct_3d_head(dino_features, predicted_heatmaps)

            result = {
                'heatmaps_2d': predicted_heatmaps,
                'keypoints_3d': kpts_3d_robot,
                'keypoints_3d_robot': kpts_3d_robot
            }

            # PnP transformation if camera_K provided (convert to camera-frame)
            if camera_K is not None:
                uv_2d = soft_argmax_2d(predicted_heatmaps)
                kp_3d_cam, pnp_valid = solve_pnp_batch(uv_2d, kpts_3d_robot, camera_K)
                result['keypoints_3d_cam'] = kp_3d_cam
                result['pnp_valid'] = pnp_valid

        return result
