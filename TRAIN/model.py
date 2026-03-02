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

# 3D prediction mode
MODE_JOINT_ANGLE = 'joint_angle'  # Predict joint angles → FK → robot-frame 3D keypoints


def soft_argmax_2d(heatmaps, temperature=10.0):
    """
    Differentiable soft-argmax to extract (u, v) from heatmaps.
    Args:
        heatmaps: (B, N, H, W)
        temperature: scaling factor for softmax sharpness (float or Tensor)
    Returns:
        (B, N, 2) [x, y] coordinates in heatmap pixel space
    """
    B, N, H, W = heatmaps.shape
    device = heatmaps.device

    x_coords = torch.arange(W, device=device, dtype=torch.float32)
    y_coords = torch.arange(H, device=device, dtype=torch.float32)

    heatmaps_flat = heatmaps.reshape(B, N, -1)

    # Support both fixed temperature and learnable parameter
    if isinstance(temperature, torch.Tensor):
        temperature = temperature.clamp(min=0.1, max=50.0)  # Prevent extreme values

    weights = F.softmax(heatmaps_flat * temperature, dim=-1)
    weights = weights.reshape(B, N, H, W)

    x = (weights.sum(dim=2) * x_coords).sum(dim=-1)  # (B, N)
    y = (weights.sum(dim=3) * y_coords).sum(dim=-1)  # (B, N)

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
                # ViT / DINOv2 / SigLIP style
                layers = self.model.encoder.layers
                for i in range(len(layers) - unfreeze_blocks, len(layers)):
                    for param in layers[i].parameters():
                        param.requires_grad = True
            elif hasattr(self.model, "blocks"):
                # Alternative ViT style
                layers = self.model.blocks
                for i in range(len(layers) - unfreeze_blocks, len(layers)):
                    for param in layers[i].parameters():
                        param.requires_grad = True

    def forward(self, image_tensor_batch):
        # Removed torch.no_grad() to allow gradient flow for downstream heads
        if "siglip" in self.model_name:
            outputs = self.model(
                pixel_values=image_tensor_batch,
                interpolate_pos_encoding=True)
            tokens = outputs.last_hidden_state
            patch_tokens = tokens[:, 1:, :]
        else: # DINOv3 계열
            outputs = self.model(pixel_values=image_tensor_batch)
            tokens = outputs.last_hidden_state
            num_reg = int(getattr(self.model.config, "num_register_tokens", 0))
            patch_tokens = tokens[:, 1 + num_reg :, :]
        return patch_tokens

class AdaptiveNorm2d(nn.Module):
    """Adaptive normalization mixing GroupNorm and LayerNorm for sim-to-real robustness"""
    def __init__(self, num_channels, num_groups=32):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups, num_channels)
        self.ln = nn.LayerNorm(num_channels)
        # Learnable mixing coefficient
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        # x: (B, C, H, W)
        gn_out = self.gn(x)
        # LayerNorm over channels
        ln_out = self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # Mix with learnable alpha (clamped to [0, 1])
        alpha = torch.sigmoid(self.alpha)
        return alpha * gn_out + (1 - alpha) * ln_out


class TokenFuser(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # bias=True를 추가하여 PyTorch CUDA kernel의 gradient stride 문제 완화
        self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.refine_blocks = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            AdaptiveNorm2d(out_channels, num_groups=32),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            AdaptiveNorm2d(out_channels, num_groups=32)
        )
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        projected = self.projection(x)
        refined = self.refine_blocks(projected)
        residual = self.residual_conv(x)
        return torch.nn.functional.gelu(refined + residual)

class ViTKeypointHead(nn.Module):
    def __init__(self, input_dim=768, num_joints=NUM_JOINTS, heatmap_size=(512, 512)):
        super().__init__()
        self.heatmap_size = heatmap_size
        self.token_fuser = TokenFuser(input_dim, 256)

        # ViT-only decoder with Sub-Pixel Convolution (PixelShuffle)
        # PixelShuffle: 초해상도 표준 기법, edge 보존 우수, checkerboard artifact 없음
        self.decoder_block1 = nn.Sequential(
            nn.Conv2d(256, 128 * 4, kernel_size=3, padding=1, bias=False),  # 4 = 2^2 for 2x upsampling
            nn.PixelShuffle(upscale_factor=2),  # (B, 128*4, H, W) -> (B, 128, H*2, W*2)
            AdaptiveNorm2d(128, num_groups=32),
            nn.GELU()
        )
        self.decoder_block2 = nn.Sequential(
            nn.Conv2d(128, 64 * 4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(upscale_factor=2),  # 2x upsampling
            AdaptiveNorm2d(64, num_groups=16),
            nn.GELU()
        )
        self.decoder_block3 = nn.Sequential(
            nn.Conv2d(64, 32 * 4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(upscale_factor=2),  # 2x upsampling
            AdaptiveNorm2d(32, num_groups=8),
            nn.GELU()
        )

        self.heatmap_predictor = nn.Conv2d(32, num_joints, kernel_size=3, padding=1)

        # Final upsampling with Sub-Pixel Convolution (4x = 2x × 2x)
        # PixelShuffle은 ConvTranspose2d보다 checkerboard artifact가 적고 edge detail 보존 우수
        self.final_upsample = nn.Sequential(
            nn.Conv2d(num_joints, num_joints * 4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(upscale_factor=2),  # 2x upsampling
            nn.Conv2d(num_joints, num_joints * 4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(upscale_factor=2)   # 2x upsampling (total 4x)
        )

    def forward(self, dino_features):
        b, n, d = dino_features.shape
        h = w = int(math.sqrt(n))

        if h * w != n:
            n_new = h * w
            dino_features = dino_features[:, :n_new, :]
        x = dino_features.permute(0, 2, 1).reshape(b, d, h, w)

        x = self.token_fuser(x)
        x = self.decoder_block1(x)
        x = self.decoder_block2(x)
        x = self.decoder_block3(x)

        heatmaps = self.heatmap_predictor(x)

        # Use learned upsampling instead of bilinear interpolation
        heatmaps = self.final_upsample(heatmaps)

        # Final resize to exact target size if needed
        if heatmaps.shape[2:] != self.heatmap_size:
            heatmaps = F.interpolate(heatmaps, size=self.heatmap_size, mode='bilinear', align_corners=False)

        return heatmaps

# =============================================================================
# Forward Kinematics (Panda robot, URDF-based DH parameters)
# =============================================================================

def _rotation_matrix_z(theta):
    """Rotation matrix around Z axis. theta: (B,) or (B,1)"""
    c = torch.cos(theta)
    s = torch.sin(theta)
    zero = torch.zeros_like(c)
    one = torch.ones_like(c)
    # (B, 3, 3)
    return torch.stack([
        torch.stack([c, -s, zero], dim=-1),
        torch.stack([s,  c, zero], dim=-1),
        torch.stack([zero, zero, one], dim=-1),
    ], dim=-2)


def _rotation_matrix_x(angle):
    """Fixed rotation matrix around X axis. angle: scalar (float)"""
    c = math.cos(angle)
    s = math.sin(angle)
    return [[1, 0, 0], [0, c, -s], [0, s, c]]


def _rotation_matrix_z_fixed(angle):
    """Fixed rotation matrix around Z axis. angle: scalar (float)"""
    c = math.cos(angle)
    s = math.sin(angle)
    return [[c, -s, 0], [s, c, 0], [0, 0, 1]]


def _make_transform(xyz, rpy):
    """
    Create a 4x4 homogeneous transform from xyz translation and rpy rotation.
    Returns a list-of-lists (will be converted to tensor later).
    rpy = (roll, pitch, yaw) = rotations around (x, y, z) in that order.
    """
    # R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    # For Panda URDF, only roll (rx) rotations appear (pitch=yaw=0 mostly)
    rx, ry, rz = rpy
    # Build rotation: Rz @ Ry @ Rx
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    R = [
        [cz*cy, cz*sy*sx - sz*cx, cz*sy*cx + sz*sx],
        [sz*cy, sz*sy*sx + cz*cx, sz*sy*cx - cz*sx],
        [-sy,   cy*sx,            cy*cx            ],
    ]
    T = [
        [R[0][0], R[0][1], R[0][2], xyz[0]],
        [R[1][0], R[1][1], R[1][2], xyz[1]],
        [R[2][0], R[2][1], R[2][2], xyz[2]],
        [0,       0,       0,       1      ],
    ]
    return T


# Panda URDF joint parameters: (origin_xyz, origin_rpy, axis)
# Joint axis is always z for Panda revolute joints
_PANDA_JOINTS = [
    # J1: panda_joint1
    {'xyz': (0, 0, 0.333), 'rpy': (0, 0, 0)},
    # J2: panda_joint2
    {'xyz': (0, 0, 0), 'rpy': (-math.pi/2, 0, 0)},
    # J3: panda_joint3
    {'xyz': (0, -0.316, 0), 'rpy': (math.pi/2, 0, 0)},
    # J4: panda_joint4
    {'xyz': (0.0825, 0, 0), 'rpy': (math.pi/2, 0, 0)},
    # J5: panda_joint5
    {'xyz': (-0.0825, 0.384, 0), 'rpy': (-math.pi/2, 0, 0)},
    # J6: panda_joint6
    {'xyz': (0, 0, 0), 'rpy': (math.pi/2, 0, 0)},
    # J7: panda_joint7
    {'xyz': (0.088, 0, 0), 'rpy': (math.pi/2, 0, 0)},
]

# Fixed transforms after joint7
_PANDA_FIXED_J8 = {'xyz': (0, 0, 0.107), 'rpy': (0, 0, 0)}
_PANDA_FIXED_HAND = {'xyz': (0, 0, 0), 'rpy': (0, 0, -math.pi/4)}

# Panda joint limits (radians) from URDF
_PANDA_JOINT_LIMITS = [
    (-2.8973, 2.8973),   # J1
    (-1.7628, 1.7628),   # J2
    (-2.8973, 2.8973),   # J3
    (-3.0718, -0.0698),  # J4
    (-2.8973, 2.8973),   # J5
    (-0.0175, 3.7525),   # J6
    (-2.8973, 2.8973),   # J7
]

# Keypoint-to-joint mapping:
# link0 = base (before any joint)
# link2 = after J1, J2
# link3 = after J1, J2, J3
# link4 = after J1, J2, J3, J4
# link6 = after J1, ..., J6
# link7 = after J1, ..., J7
# hand  = after J1, ..., J7, J8_fixed, hand_fixed
_KEYPOINT_JOINT_INDICES = [0, 2, 3, 4, 6, 7, 8]  # 8 = hand (after all joints + fixed)


def panda_forward_kinematics(joint_angles):
    """
    Compute forward kinematics for Panda robot.

    Args:
        joint_angles: (B, 7) joint angles in radians

    Returns:
        keypoint_positions: (B, 7, 3) keypoint positions in robot base frame (meters)
    """
    B = joint_angles.shape[0]
    device = joint_angles.device
    dtype = joint_angles.dtype

    # Precompute fixed origin transforms as tensors
    fixed_transforms = []
    for j_info in _PANDA_JOINTS:
        T = _make_transform(j_info['xyz'], j_info['rpy'])
        fixed_transforms.append(torch.tensor(T, device=device, dtype=dtype))

    T_j8 = torch.tensor(_make_transform(_PANDA_FIXED_J8['xyz'], _PANDA_FIXED_J8['rpy']),
                         device=device, dtype=dtype)
    T_hand = torch.tensor(_make_transform(_PANDA_FIXED_HAND['xyz'], _PANDA_FIXED_HAND['rpy']),
                          device=device, dtype=dtype)

    # Identity for base
    eye4 = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)

    # Accumulate transforms: T_cumul[i] = T_0 @ ... @ T_i
    # where T_i = fixed_origin_i @ Rz(theta_i)
    cumul = eye4.clone()  # (B, 4, 4) - base frame

    # Store cumulative transforms after each joint
    # Index 0..6 = after J1..J7, index 7 = after J8_fixed, index 8 = after hand_fixed
    all_transforms = [cumul.clone()]  # [0] = base (before J1)

    for i in range(7):
        # Fixed origin transform (broadcast to batch)
        T_fixed = fixed_transforms[i].unsqueeze(0).expand(B, -1, -1)  # (B, 4, 4)

        # Joint rotation around z
        theta = joint_angles[:, i]  # (B,)
        R_joint = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1).clone()
        R_joint[:, :3, :3] = _rotation_matrix_z(theta)

        # Cumulative: cumul = cumul @ T_fixed @ R_joint
        cumul = cumul @ T_fixed @ R_joint

        all_transforms.append(cumul.clone())  # [i+1] = after J(i+1)

    # After J7, apply fixed J8 and hand transforms
    T_j8_batch = T_j8.unsqueeze(0).expand(B, -1, -1)
    T_hand_batch = T_hand.unsqueeze(0).expand(B, -1, -1)

    cumul_j8 = cumul @ T_j8_batch
    all_transforms.append(cumul_j8.clone())  # [8] = after J8 fixed

    cumul_hand = cumul_j8 @ T_hand_batch
    all_transforms.append(cumul_hand.clone())  # [9] = after hand fixed

    # Extract keypoint positions
    # link0=base=[0], link2=after J2=[2], link3=after J3=[3],
    # link4=after J4=[4], link6=after J6=[6], link7=after J7=[7], hand=[9]
    kp_indices = [0, 2, 3, 4, 6, 7, 9]
    keypoints = []
    for idx in kp_indices:
        pos = all_transforms[idx][:, :3, 3]  # (B, 3)
        keypoints.append(pos)

    return torch.stack(keypoints, dim=1)  # (B, 7, 3)


class JointAngleHead(nn.Module):
    """
    Predicts Panda joint angles (6 or 7 DoF) from visual features + heatmaps.
    Uses FK to produce 3D keypoints in robot base frame.
    """

    def __init__(self, input_dim=FEATURE_DIM, num_joints=NUM_JOINTS, num_angles=7, use_joint_embedding=False):
        super().__init__()
        self.num_joints = num_joints
        self.num_angles = num_angles
        self.hidden_dim = 256
        self.use_joint_embedding = use_joint_embedding

        # Joint limits as buffers (optionally use only first 6 for RoboPEPP-style mode)
        limits = torch.tensor(_PANDA_JOINT_LIMITS, dtype=torch.float32)  # (7, 2)
        limits = limits[:self.num_angles]
        self.register_buffer('joint_lower', limits[:, 0])
        self.register_buffer('joint_upper', limits[:, 1])
        self.register_buffer('joint_mid', (limits[:, 0] + limits[:, 1]) / 2)
        self.register_buffer('joint_range', (limits[:, 1] - limits[:, 0]) / 2)

        # Learnable temperature for soft-argmax
        self.temperature = nn.Parameter(torch.tensor(10.0))

        # 1. Per-joint feature refinement
        # Input: visual feature (input_dim) + normalized 2D coords (2)
        self.joint_feature_net = nn.Sequential(
            nn.Linear(input_dim + 2, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim)
        )

        # Joint identity embedding: "이 토큰은 1번 모터(Base)입니다" / "이 토큰은 7번 모터(Wrist)입니다"
        if use_joint_embedding:
            self.joint_embedding = nn.Embedding(num_joints, self.hidden_dim)

        # 2. Self-attention for kinematic constraint learning (reduced from 4 to 2 layers)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.joint_relation_net = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 3. Global angle decoding (stable baseline).
        self.global_angle_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, self.num_angles)
        )

        # 4. Per-joint residual correction after contextual interaction.
        self.per_joint_residual_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 1)
        )
        # Safety path when num_joints != num_angles (keeps API generic).
        self.residual_mixer = nn.Identity() if self.num_joints == self.num_angles else nn.Linear(self.num_joints, self.num_angles)
        # Small learnable gate keeps residual branch from destabilizing early training.
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, dino_features, predicted_heatmaps):
        """
        Args:
            dino_features: (B, N, D) backbone patch tokens
            predicted_heatmaps: (B, NUM_JOINTS, H, W) 2D belief maps
        Returns:
            joint_angles: (B, num_angles) predicted joint angles (radians, within limits)
            keypoints_3d_robot: (B, 7, 3) FK-computed 3D keypoints in robot base frame
        """
        b, n, d = dino_features.shape
        h = w = int(math.sqrt(n))

        feat_map = dino_features.permute(0, 2, 1).reshape(b, d, h, w)

        # Extract 2D keypoint locations from heatmaps with learnable temperature
        uv_heatmap = soft_argmax_2d(predicted_heatmaps, self.temperature)  # (B, NJ, 2)

        # Normalize 2D coords to [-1, 1] (in-place 회피)
        hm_h, hm_w = predicted_heatmaps.shape[2], predicted_heatmaps.shape[3]
        uv_norm_x = (uv_heatmap[:, :, 0] / hm_w) * 2.0 - 1.0
        uv_norm_y = (uv_heatmap[:, :, 1] / hm_h) * 2.0 - 1.0
        uv_normalized = torch.stack([uv_norm_x, uv_norm_y], dim=-1)  # (B, NJ, 2)

        # Scale keypoint coords from heatmap space to feature map space (ROIAlign용)
        scale_x = w / hm_w
        scale_y = h / hm_h
        uv_feat = torch.stack([uv_heatmap[:, :, 0] * scale_x,
                               uv_heatmap[:, :, 1] * scale_y], dim=-1)  # (B, NJ, 2)

        # Create RoI boxes — 벡터화 (roi_size=2: keypoint 겹침 방지)
        roi_size = 4
        batch_ids = torch.arange(b, device=feat_map.device, dtype=torch.float32).unsqueeze(1).expand(b, self.num_joints)
        cx = uv_feat[:, :, 0]
        cy = uv_feat[:, :, 1]
        rois = torch.stack([batch_ids, cx - roi_size/2, cy - roi_size/2,
                            cx + roi_size/2, cy + roi_size/2], dim=-1).reshape(-1, 5).detach()

        # ROIAlign + pooling
        roi_features = roi_align(feat_map, rois, output_size=(3, 3), spatial_scale=1.0)
        joint_features = roi_features.mean(dim=[2, 3])
        joint_features = joint_features.view(b, self.num_joints, d)

        # Concatenate 2D coordinates
        joint_features = torch.cat([joint_features, uv_normalized], dim=-1)  # (B, NJ, D+2)

        # Feature refinement
        refined = self.joint_feature_net(joint_features)  # (B, NJ, 256)

        # Add joint identity embedding: "이 토큰은 1번 모터(Base)입니다" / "이 토큰은 7번 모터(Wrist)입니다"
        # Without this, the transformer must infer joint identity from visual features alone!
        if self.use_joint_embedding:
            joint_ids = torch.arange(self.num_joints, device=dino_features.device).expand(b, self.num_joints)
            joint_embeds = self.joint_embedding(joint_ids)  # (B, NJ, 256)
            refined = refined + joint_embeds  # 시각+공간 특징에 이름표를 더함!

        # Self-attention for kinematic constraint learning
        related = self.joint_relation_net(refined)  # (B, NJ, 256)

        # Global decoding from integrated robot state.
        global_state = related.mean(dim=1)  # (B, 256)
        raw_global = self.global_angle_predictor(global_state)  # (B, num_angles)

        # Per-joint residual correction (context-aware due to self-attention).
        raw_residual = self.per_joint_residual_head(related).squeeze(-1)  # (B, NJ)
        raw_residual = self.residual_mixer(raw_residual)  # (B, num_angles)
        residual_scale = torch.sigmoid(self.residual_scale)  # (0, 1)
        raw_angles = raw_global + residual_scale * torch.tanh(raw_residual)

        # Apply joint limits via tanh scaling: mid + tanh(raw) * range
        joint_angles = self.joint_mid.unsqueeze(0) + torch.tanh(raw_angles) * self.joint_range.unsqueeze(0)

        # Forward kinematics always expects 7-DoF Panda vector.
        # In 6-DoF mode we append fixed joint7=0.
        if self.num_angles < 7:
            pad = torch.zeros((joint_angles.shape[0], 7 - self.num_angles), device=joint_angles.device, dtype=joint_angles.dtype)
            joint_angles_fk = torch.cat([joint_angles, pad], dim=1)
        else:
            joint_angles_fk = joint_angles
        keypoints_3d_robot = panda_forward_kinematics(joint_angles_fk)  # (B, 7, 3)

        return joint_angles, keypoints_3d_robot


class EnhancedKeypoint3DHead(nn.Module):
    """
    Predicts 3D coordinates by combining DINO features with explicit joint identity embeddings.
    """

    def __init__(self, input_dim=FEATURE_DIM, num_joints=NUM_JOINTS, use_joint_embedding=True):
        super().__init__()
        self.num_joints = num_joints
        self.hidden_dim = 512
        self.use_joint_embedding = use_joint_embedding

        # 1) Per-joint feature refinement
        self.joint_feature_net = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim)
        )

        # 2) Joint identity embedding ("name tag" for each joint token)
        if use_joint_embedding:
            self.joint_embedding = nn.Embedding(num_joints, self.hidden_dim)

        # 3) Self-attention over joint tokens to learn inter-joint constraints
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.joint_relation_net = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 4) Regress xyz per joint
        self.coord_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 3)
        )

    def forward(self, dino_features, predicted_heatmaps):
        b, n, d = dino_features.shape
        h = w = int(math.sqrt(n))
        feat_map = dino_features.permute(0, 2, 1).reshape(b, d, h, w)  # (B, D, h, w)

        # Spatial softmax pooling using heatmaps as attention over feature map
        weights = F.interpolate(predicted_heatmaps, size=(h, w), mode='bilinear', align_corners=False)
        weights = torch.clamp(weights, min=0.0)
        weights_flat = weights.reshape(b, self.num_joints, -1)
        weights_norm = F.softmax(weights_flat / 0.1, dim=-1).reshape(b, self.num_joints, h, w)

        # Weighted per-joint visual tokens: (B, NJ, D)
        joint_features = torch.einsum('bdhw,bjhw->bjd', feat_map, weights_norm)

        refined_features = self.joint_feature_net(joint_features)  # (B, NJ, 256)

        if self.use_joint_embedding:
            joint_ids = torch.arange(self.num_joints, device=dino_features.device).unsqueeze(0).expand(b, self.num_joints)
            joint_embeds = self.joint_embedding(joint_ids)  # (B, NJ, 256)
            fused_features = refined_features + joint_embeds
        else:
            fused_features = refined_features

        related_features = self.joint_relation_net(fused_features)  # (B, NJ, 256)
        pred_kpts_3d = self.coord_predictor(related_features)  # (B, NJ, 3)
        return pred_kpts_3d


class FusionCorrectionHead(nn.Module):
    """
    Fuse FK branch and direct 3D branch, then predict residual correction.
    """

    def __init__(self):
        super().__init__()
        # Input per joint: fk(3), direct(3), diff(3), confidence(1) = 10
        self.fusion_mlp = nn.Sequential(
            nn.Linear(10, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, 4),  # delta xyz (3) + alpha logit (1)
        )

    def forward(self, kp3d_fk, kp3d_direct, predicted_heatmaps):
        hm_max = predicted_heatmaps.flatten(2).amax(dim=-1, keepdim=True)  # (B, NJ, 1)
        confidence = torch.sigmoid(hm_max)

        diff = kp3d_direct - kp3d_fk
        fusion_input = torch.cat([kp3d_fk, kp3d_direct, diff, confidence], dim=-1)  # (B, NJ, 10)
        fusion_out = self.fusion_mlp(fusion_input)  # (B, NJ, 4)

        delta = fusion_out[:, :, :3]
        alpha = torch.sigmoid(fusion_out[:, :, 3:4])  # (B, NJ, 1)
        blended = alpha * kp3d_direct + (1.0 - alpha) * kp3d_fk
        kp3d_final = blended + delta

        return kp3d_final, alpha, delta


def compute_extrinsics_from_pnp(kp3d_robot, kp2d_image, camera_K):
    """
    Batch PnP solver: compute robot-to-camera extrinsics [R, t].

    Args:
        kp3d_robot: (B, 7, 3) 3D keypoints in robot frame (detached numpy or tensor)
        kp2d_image: (B, 7, 2) 2D keypoints in original image coords (detached numpy or tensor)
        camera_K: (B, 3, 3) camera intrinsic matrices

    Returns:
        R_batch: (B, 3, 3) rotation matrices (detached tensors on same device as input)
        t_batch: (B, 3) translation vectors (detached tensors)
        valid_mask: (B,) bool tensor indicating which samples had successful PnP
    """
    if isinstance(kp3d_robot, torch.Tensor):
        device = kp3d_robot.device
        kp3d_np = kp3d_robot.detach().cpu().numpy()
        kp2d_np = kp2d_image.detach().cpu().numpy()
        K_np = camera_K.detach().cpu().numpy()
    else:
        device = torch.device('cpu')
        kp3d_np = kp3d_robot
        kp2d_np = kp2d_image
        K_np = camera_K

    B = kp3d_np.shape[0]
    R_batch = torch.zeros(B, 3, 3, device=device)
    t_batch = torch.zeros(B, 3, device=device)
    valid_mask = torch.zeros(B, dtype=torch.bool, device=device)

    for b in range(B):
        try:
            obj_pts = kp3d_np[b].astype(np.float64)
            img_pts = kp2d_np[b].astype(np.float64)
            K = K_np[b].astype(np.float64)

            success, rvec, tvec = cv2.solvePnP(
                obj_pts, img_pts, K, None, flags=cv2.SOLVEPNP_EPNP
            )
            if success:
                R, _ = cv2.Rodrigues(rvec)
                R_batch[b] = torch.from_numpy(R).float().to(device)
                t_batch[b] = torch.from_numpy(tvec.flatten()).float().to(device)
                valid_mask[b] = True
        except Exception:
            pass

    return R_batch, t_batch, valid_mask


class IterativeRefinementModule(nn.Module):
    """
    Iterative refinement for joint angle prediction via render-and-compare.

    Pipeline per iteration:
        FK(θᵢ) → kp3d_robot → project(R,t,K) → pred_2d
        Δuv = target_2d - pred_2d
        CorrectionNet(Δuv, θᵢ, |Δuv|) → Δθ
        θᵢ₊₁ = clamp(θᵢ + step_scale * Δθ)

    R, t are obtained via PnP (detached, no gradient).
    """

    def __init__(self, num_iterations=3, num_angles=7):
        super().__init__()
        self.num_iterations = num_iterations
        self.num_angles = num_angles

        # Joint limits as buffers (same as JointAngleHead)
        limits = torch.tensor(_PANDA_JOINT_LIMITS, dtype=torch.float32)
        self.register_buffer('joint_lower', limits[:, 0])
        self.register_buffer('joint_upper', limits[:, 1])

        # Shared correction network (weight-tied across iterations)
        # Input: Δuv (7*2=14) + θ_current (7) + |Δuv| (7) = 28
        self.correction_net = nn.Sequential(
            nn.Linear(28, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, num_angles),
        )

        # Learnable step scale per iteration (sigmoid-gated, init ~0.1)
        # logit for sigmoid(x)=0.1 is ln(0.1/0.9) ≈ -2.2
        self.step_scale_logits = nn.Parameter(
            torch.full((num_iterations,), -2.2)
        )

    def _project_to_2d(self, kp3d_robot, R, t, K):
        """
        Differentiable projection: robot frame → camera frame → 2D image coords.

        Args:
            kp3d_robot: (B, 7, 3) robot-frame keypoints (differentiable)
            R: (B, 3, 3) rotation matrices (detached)
            t: (B, 3) translation vectors (detached)
            K: (B, 3, 3) camera intrinsics (detached)

        Returns:
            uv: (B, 7, 2) projected 2D coordinates in original image space
        """
        # Transform to camera frame: kp_cam = R @ kp_robot^T + t
        # (B, 3, 3) @ (B, 3, 7) → (B, 3, 7)
        kp_cam = torch.bmm(R, kp3d_robot.transpose(1, 2)) + t.unsqueeze(-1)
        kp_cam = kp_cam.transpose(1, 2)  # (B, 7, 3)

        # Project: uv = K @ kp_cam / z
        z = kp_cam[:, :, 2:3].clamp(min=1e-6)  # (B, 7, 1)
        kp_norm = kp_cam / z  # (B, 7, 3) - [x/z, y/z, 1]

        # Apply intrinsics: (B, 3, 3) @ (B, 3, 7) → (B, 3, 7)
        uv_h = torch.bmm(K, kp_norm.transpose(1, 2)).transpose(1, 2)  # (B, 7, 3)
        uv = uv_h[:, :, :2]  # (B, 7, 2)

        return uv

    def forward(self, initial_angles, target_2d, camera_K, original_size,
                R_ext, t_ext, valid_mask):
        """
        Run iterative refinement loop.

        Args:
            initial_angles: (B, 7) initial joint angle predictions from JointAngleHead
            target_2d: (B, 7, 2) target 2D keypoints in original image coords
                       (from heatmap hard-argmax during inference, or GT during training)
            camera_K: (B, 3, 3) camera intrinsic matrices
            original_size: (W, H) tuple of original image size
            R_ext: (B, 3, 3) rotation matrices from PnP (detached)
            t_ext: (B, 3) translation vectors from PnP (detached)
            valid_mask: (B,) bool mask for valid PnP samples

        Returns:
            dict with:
                'refined_angles': (B, 7) final refined angles
                'refined_kp3d_robot': (B, 7, 3) FK keypoints from refined angles
                'all_angles': list of (B, 7) angles at each iteration
                'all_kp3d_robot': list of (B, 7, 3) FK keypoints at each iteration
        """
        B = initial_angles.shape[0]
        theta = initial_angles  # (B, 7) - gradients flow through this

        all_angles = [theta]
        all_kp3d_robot = [panda_forward_kinematics(theta)]

        for i in range(self.num_iterations):
            # FK to get current 3D keypoints in robot frame
            kp3d_robot = panda_forward_kinematics(theta)  # (B, 7, 3)

            # Project to 2D (differentiable through FK and projection, R/t detached)
            pred_2d = self._project_to_2d(kp3d_robot, R_ext, t_ext, camera_K)  # (B, 7, 2)

            # Compute reprojection error
            delta_uv = target_2d - pred_2d  # (B, 7, 2)

            # For invalid PnP samples, zero out delta to avoid garbage gradients
            delta_uv = delta_uv * valid_mask.float().unsqueeze(-1).unsqueeze(-1)

            # Prepare correction net input
            delta_uv_flat = delta_uv.reshape(B, -1)  # (B, 14)
            delta_uv_mag = delta_uv.norm(dim=-1)  # (B, 7)
            correction_input = torch.cat([delta_uv_flat, theta, delta_uv_mag], dim=-1)  # (B, 28)

            # Predict angle correction
            delta_theta = self.correction_net(correction_input)  # (B, 7)

            # Apply step scale (sigmoid-gated)
            step_scale = torch.sigmoid(self.step_scale_logits[i])

            # Update angles with clamping to joint limits
            theta = torch.clamp(
                theta + step_scale * delta_theta,
                min=self.joint_lower.unsqueeze(0),
                max=self.joint_upper.unsqueeze(0)
            )

            all_angles.append(theta)
            all_kp3d_robot.append(panda_forward_kinematics(theta))

        return {
            'refined_angles': theta,
            'refined_kp3d_robot': all_kp3d_robot[-1],
            'all_angles': all_angles,
            'all_kp3d_robot': all_kp3d_robot,
        }


class DINOv3PoseEstimator(nn.Module):
    def __init__(self, dino_model_name, heatmap_size, unfreeze_blocks=2,
                 use_joint_embedding=False,
                 use_iterative_refinement=False, refinement_iterations=3,
                 fix_joint7_zero=False):
        super().__init__()
        self.dino_model_name = dino_model_name
        self.heatmap_size = heatmap_size  # (H, W) tuple
        self.fix_joint7_zero = fix_joint7_zero
        # Iterative refinement path is intentionally disabled to keep training/inference simple.
        self.use_iterative_refinement = False
        self.backbone = DINOv3Backbone(dino_model_name, unfreeze_blocks=unfreeze_blocks)

        if "siglip" in self.dino_model_name:
            config = self.backbone.model.config
            feature_dim = config.hidden_size
        else: # DINOv3 계열
            config = self.backbone.model.config
            feature_dim = config.hidden_sizes[-1] if "conv" in self.dino_model_name else config.hidden_size

        # 1. 2D Heatmap Predictor
        self.keypoint_head = ViTKeypointHead(
            input_dim=feature_dim,
            heatmap_size=heatmap_size
        )

        # 2. Joint Angle Head → FK → robot-frame 3D keypoints
        self.joint_angle_head = JointAngleHead(
            input_dim=feature_dim,
            num_joints=NUM_JOINTS,
            num_angles=6 if self.fix_joint7_zero else 7,
            use_joint_embedding=use_joint_embedding
        )

        # 3. Direct 3D regression branch
        self.direct_3d_head = EnhancedKeypoint3DHead(
            input_dim=feature_dim,
            num_joints=NUM_JOINTS,
            use_joint_embedding=use_joint_embedding,
        )

        # 4. Fusion + residual correction (FK branch + direct branch)
        self.fusion_head = FusionCorrectionHead()

        # Iterative refinement module is disabled in simplified mode.
        self.refinement_module = None

    def forward(self, image_tensor_batch, camera_K=None, original_size=None,
                gt_angles=None, gt_2d_image=None, use_refinement=None):
        """
        Args:
            image_tensor_batch: (B, 3, H, W) input images
            camera_K: (B, 3, 3) camera intrinsics (for iterative refinement PnP)
            original_size: (W, H) original image resolution (for iterative refinement)
            gt_angles: (B, 7) GT joint angles (for training refinement with stable PnP)
            gt_2d_image: (B, 7, 2) GT 2D keypoints in original image coords (for training refinement)
            use_refinement: bool override (None = use self.use_iterative_refinement)
        """
        # 1. Extract visual representations
        dino_features = self.backbone(image_tensor_batch)

        # 2. Estimate 2D pixel locations (Heatmaps)
        predicted_heatmaps = self.keypoint_head(dino_features)

        # 3. Predict joint angles → FK → robot-frame 3D keypoints
        joint_angles_pred, kpts_3d_fk = self.joint_angle_head(
            dino_features, predicted_heatmaps
        )
        current_angles = joint_angles_pred
        current_kp3d_fk = kpts_3d_fk
        if self.fix_joint7_zero and current_angles.shape[1] >= 7:
            # RoboPEPP-style convention: treat joint7 as fixed and only learn/use first 6 joints.
            current_angles = current_angles.clone()
            current_angles[:, 6] = 0.0
            current_kp3d_fk = panda_forward_kinematics(current_angles)

        # 4. Optional iterative refinement on angle/FK branch (disabled in simplified mode)
        result = {}

        # 5. Direct 3D branch + fusion correction
        kpts_3d_direct = self.direct_3d_head(dino_features, predicted_heatmaps)
        kpts_3d_final, fusion_alpha, fusion_delta = self.fusion_head(
            current_kp3d_fk, kpts_3d_direct, predicted_heatmaps
        )

        # Main outputs
        result.update({
            'heatmaps_2d': predicted_heatmaps,
            'joint_angles': current_angles,
            'keypoints_3d_fk': current_kp3d_fk,
            'keypoints_3d_direct': kpts_3d_direct,
            'fusion_alpha': fusion_alpha,
            'fusion_delta': fusion_delta,
            'keypoints_3d': kpts_3d_final,          # final 3D estimate (robot frame)
            'keypoints_3d_robot': kpts_3d_final,    # keep compatibility with existing losses/eval
        })

        return result
