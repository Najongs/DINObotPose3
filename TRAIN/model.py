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

class SpatialGlobalModulation(nn.Module):
    """ DINOv3의 전역 문맥(Global Context)을 CNN 피처맵에 강제로 주입하는 FiLM 모듈 """
    def __init__(self, global_dim, feature_dim):
        super().__init__()
        # 글로벌 벡터를 받아서 피처맵 채널 수의 2배(Gamma, Beta) 크기로 확장
        self.mlp = nn.Sequential(
            nn.Linear(global_dim, feature_dim * 2),
            nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim * 2)
        )

    def forward(self, x, global_context):
        # x: (B, feature_dim, H, W) - 현재 디코더의 로컬 피처맵
        # global_context: (B, global_dim) - DINOv3 전체 요약 벡터
        
        gamma_beta = self.mlp(global_context) # (B, feature_dim * 2)
        gamma, beta = gamma_beta.chunk(2, dim=1) # 각각 (B, feature_dim)
        
        # 공간 차원(H, W)에 브로드캐스팅하기 위해 차원 추가
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        
        # FiLM 연산: 피처맵에 글로벌 정보 곱하고 더하기
        return x * (1 + gamma) + beta

class ViTKeypointHead(nn.Module):
    def __init__(self, input_dim=768, num_joints=7, heatmap_size=(512, 512)):
        super().__init__()
        self.heatmap_size = heatmap_size
        self.token_fuser = TokenFuser(input_dim, 256)

        # --- 🚀 [NEW] 디코더 블록마다 글로벌 정보를 쏴줄 모듈 정의 ---
        # DINOv3 원본 피처(768)를 요약한 벡터를 글로벌 정보로 사용
        self.global_mod1 = SpatialGlobalModulation(global_dim=input_dim, feature_dim=256)
        self.global_mod2 = SpatialGlobalModulation(global_dim=input_dim, feature_dim=128)
        self.global_mod3 = SpatialGlobalModulation(global_dim=input_dim, feature_dim=64)

        # ViT-only decoder with Sub-Pixel Convolution (PixelShuffle)
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

        # Final upsampling with Sub-Pixel Convolution (4x = 2x × 2x)
        self.final_upsample = nn.Sequential(
            nn.Conv2d(num_joints, num_joints * 4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(upscale_factor=2),  
            nn.Conv2d(num_joints, num_joints * 4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(upscale_factor=2)   
        )

    def forward(self, dino_features):
        b, n, d = dino_features.shape
        h = w = int(math.sqrt(n))

        if h * w != n:
            n_new = h * w
            dino_features = dino_features[:, :n_new, :]
        
        # 1. 2D 공간 피처로 변환
        x = dino_features.permute(0, 2, 1).reshape(b, d, h, w)

        # --- 🚀 [NEW] 2. DINOv3 피처에서 글로벌 컨텍스트(요약본) 추출 ---
        # Global Average Pooling을 통해 전체 이미지의 문맥을 (B, d) 벡터로 압축
        global_context = F.adaptive_avg_pool2d(x, 1).flatten(1)

        # 3. Fuser 통과
        x = self.token_fuser(x)

        # 4. 디코더 통과 시마다 글로벌 문맥 지속 주입 (FiLM)
        x = self.global_mod1(x, global_context) # 글로벌 정보 묻히기
        x = self.decoder_block1(x)              # 업샘플링

        x = self.global_mod2(x, global_context) # 글로벌 정보 묻히기
        x = self.decoder_block2(x)              # 업샘플링

        x = self.global_mod3(x, global_context) # 글로벌 정보 묻히기
        x = self.decoder_block3(x)              # 업샘플링

        # 5. 최종 히트맵 예측 및 리사이즈
        heatmaps = self.heatmap_predictor(x)
        heatmaps = self.final_upsample(heatmaps)

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
    [IMPROVED]: Added Confidence Gating & Global Context for occlusion robustness.
    """

    def __init__(self, input_dim=768, num_joints=7, num_angles=7, use_joint_embedding=True):
        super().__init__()
        self.num_joints = num_joints
        self.num_angles = num_angles
        self.hidden_dim = 256
        self.use_joint_embedding = use_joint_embedding

        # (기존 Joint limits 설정 코드는 동일하게 유지)
        limits = torch.tensor(_PANDA_JOINT_LIMITS, dtype=torch.float32)  
        limits = limits[:self.num_angles]
        self.register_buffer('joint_lower', limits[:, 0])
        self.register_buffer('joint_upper', limits[:, 1])
        self.register_buffer('joint_mid', (limits[:, 0] + limits[:, 1]) / 2)
        self.register_buffer('joint_range', (limits[:, 1] - limits[:, 0]) / 2)

        self.temperature = nn.Parameter(torch.tensor(10.0))

        # 1. Per-joint feature refinement
        self.joint_feature_net = nn.Sequential(
            nn.Linear(input_dim + 2, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim)
        )

        # 🚀 [NEW] 전역 문맥(Global Context)을 위한 선형 변환기
        self.global_feature_net = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim)
        )

        if use_joint_embedding:
            # 관절 이름표 (7개) + 글로벌 토큰용 이름표 (1개) = 총 8개
            self.joint_embedding = nn.Embedding(num_joints + 1, self.hidden_dim)

        # 2. Self-attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.joint_relation_net = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 3. Global angle decoding
        self.global_angle_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, self.num_angles)
        )

        # 4. Per-joint residual correction
        self.per_joint_residual_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 1)
        )
        self.residual_mixer = nn.Identity() if self.num_joints == self.num_angles else nn.Linear(self.num_joints, self.num_angles)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, dino_features, predicted_heatmaps):
        b, n, d = dino_features.shape
        h = w = int(math.sqrt(n))

        feat_map = dino_features.permute(0, 2, 1).reshape(b, d, h, w)

        # 1. Soft-argmax로 2D 좌표 추출
        uv_heatmap = soft_argmax_2d(predicted_heatmaps, self.temperature)  # (B, NJ, 2)

        # 🚀 [NEW] 2. 히트맵에서 신뢰도(Confidence) 추출 (Softmax or Sigmoid)
        # 히트맵의 각 관절별 최대 활성화 값을 뽑아내어 [0, 1] 사이로 스케일링
        confidence = torch.amax(predicted_heatmaps, dim=(2, 3)) # (B, NJ)
        confidence = torch.sigmoid(confidence).unsqueeze(-1)    # (B, NJ, 1)

        # Normalize 2D coords
        hm_h, hm_w = predicted_heatmaps.shape[2], predicted_heatmaps.shape[3]
        uv_norm_x = (uv_heatmap[:, :, 0] / hm_w) * 2.0 - 1.0
        uv_norm_y = (uv_heatmap[:, :, 1] / hm_h) * 2.0 - 1.0
        uv_normalized = torch.stack([uv_norm_x, uv_norm_y], dim=-1)

        scale_x = w / hm_w
        scale_y = h / hm_h
        uv_feat = torch.stack([uv_heatmap[:, :, 0] * scale_x,
                               uv_heatmap[:, :, 1] * scale_y], dim=-1)

        # RoI extraction
        roi_size = 4
        batch_ids = torch.arange(b, device=feat_map.device, dtype=torch.float32).unsqueeze(1).expand(b, self.num_joints)
        cx = uv_feat[:, :, 0]
        cy = uv_feat[:, :, 1]
        rois = torch.stack([batch_ids, cx - roi_size/2, cy - roi_size/2,
                            cx + roi_size/2, cy + roi_size/2], dim=-1).reshape(-1, 5).detach()

        roi_features = roi_align(feat_map, rois, output_size=(3, 3), spatial_scale=1.0)
        joint_features = roi_features.mean(dim=[2, 3]).view(b, self.num_joints, d)

        # 🚀 [NEW] 3. 신뢰도 게이팅 (Confidence Gating) 적용!
        # 가려져서 확신이 없는(confidence가 낮은) 쓰레기 RoI 피처의 영향력을 지워버립니다.
        joint_features = joint_features * confidence
        uv_normalized = uv_normalized * confidence

        # Concatenate & Refine
        joint_features = torch.cat([joint_features, uv_normalized], dim=-1)
        refined_joints = self.joint_feature_net(joint_features)  # (B, NJ, 256)

        # 🚀 [NEW] 4. 글로벌 뷰 토큰 (Global Token) 생성
        # DINOv3 피처맵 전체를 평균 내어 전역 문맥 토큰을 만듭니다.
        global_pool = F.adaptive_avg_pool2d(feat_map, 1).flatten(1) # (B, D)
        global_token = self.global_feature_net(global_pool).unsqueeze(1) # (B, 1, 256)

        # 글로벌 토큰을 관절 토큰 배열의 맨 앞에 붙임 (B, NJ+1, 256)
        combined_tokens = torch.cat([global_token, refined_joints], dim=1)

        # 5. Joint Identity Embedding 추가
        if self.use_joint_embedding:
            # 0번은 Global Token용, 1~7번은 관절용 이름표
            token_ids = torch.arange(self.num_joints + 1, device=dino_features.device).expand(b, self.num_joints + 1)
            token_embeds = self.joint_embedding(token_ids)
            combined_tokens = combined_tokens + token_embeds

        # 6. Transformer Self-attention 통과
        related_tokens = self.joint_relation_net(combined_tokens)  # (B, NJ+1, 256)

        # 7. Decoding
        # 글로벌 예측은 맨 앞의 Global Token(0번째 인덱스) 결과를 사용
        global_state = related_tokens[:, 0, :]  # (B, 256)
        raw_global = self.global_angle_predictor(global_state)

        # 잔차(Residual) 보정은 뒤에 있는 관절 토큰(1~7번 인덱스) 결과를 사용
        joint_states = related_tokens[:, 1:, :] # (B, NJ, 256)
        raw_residual = self.per_joint_residual_head(joint_states).squeeze(-1)
        raw_residual = self.residual_mixer(raw_residual)

        residual_scale = torch.sigmoid(self.residual_scale)
        raw_angles = raw_global + residual_scale * torch.tanh(raw_residual)

        joint_angles = self.joint_mid.unsqueeze(0) + torch.tanh(raw_angles) * self.joint_range.unsqueeze(0)

        # Forward Kinematics 처리 (기존과 동일)
        if self.num_angles < 7:
            pad = torch.zeros((joint_angles.shape[0], 7 - self.num_angles), device=joint_angles.device, dtype=joint_angles.dtype)
            joint_angles_fk = torch.cat([joint_angles, pad], dim=1)
        else:
            joint_angles_fk = joint_angles
        keypoints_3d_robot = panda_forward_kinematics(joint_angles_fk)

        return joint_angles, keypoints_3d_robot


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

        # Direct 3D branch and Fusion head removed — using FK output directly.

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

        # Main outputs (FK is the final 3D estimate)
        result.update({
            'heatmaps_2d': predicted_heatmaps,
            'joint_angles': current_angles,
            'keypoints_3d_fk': current_kp3d_fk,
            'keypoints_3d': current_kp3d_fk,
            'keypoints_3d_robot': current_kp3d_fk,
        })

        return result
