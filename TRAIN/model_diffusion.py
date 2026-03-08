"""Diffusion-based joint angle estimation (RoboKeyGen style)"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import DINOv3Backbone, ViTKeypointHead, soft_argmax_2d, panda_forward_kinematics

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class DiffusionAngleHead(nn.Module):
    """
    Diffusion-based angle prediction.
    Condition: UV features from heatmap
    Target: Joint angles (6D)
    """
    def __init__(self, input_dim=768, num_joints=7, num_angles=6, num_steps=20, dropout=0.1):
        super().__init__()
        self.num_angles = num_angles
        
        # Time embedding
        time_dim = 128
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim)
        )
        
        # UV encoder (condition)
        self.uv_encoder = nn.Sequential(
            nn.Linear(num_joints * 2, 256),
            nn.GELU(),
            nn.Linear(256, 256)
        )

        self.conf_encoder = nn.Sequential(
            nn.Linear(num_joints, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Global feature encoder
        self.feat_encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256)
        )
        
        # Denoising network
        self.denoise_net = nn.Sequential(
            nn.Linear(num_angles + time_dim + 256 + 256, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_angles)
        )

        # Strong direct supervision for low-dimensional angle regression.
        self.init_head = nn.Sequential(
            nn.Linear(256 + 256, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_angles)
        )
        
        # Diffusion params
        self.num_steps = num_steps
        self.beta_start = 1e-4
        self.beta_end = 0.02

    def encode_condition(self, dino_features, predicted_heatmaps):
        """Build conditioning features from frozen backbone + heatmaps."""
        B = dino_features.shape[0]

        xf = dino_features.mean(dim=1)  # (B, 768)
        feat_cond = self.feat_encoder(xf)  # (B, 256)

        uv = soft_argmax_2d(predicted_heatmaps, temperature=100.0)
        uv_flat = uv.reshape(B, -1)
        uv_cond = self.uv_encoder(uv_flat)  # (B, 256)
        conf = predicted_heatmaps.flatten(2).amax(dim=2)
        uv_cond = uv_cond * self.conf_encoder(conf)

        condition = torch.cat([feat_cond, uv_cond], dim=1)  # (B, 512)
        return condition, uv
        
    def forward(self, dino_features, predicted_heatmaps, camera_K=None, training=True):
        device = dino_features.device
        condition, uv = self.encode_condition(dino_features, predicted_heatmaps)
        
        if training:
            # Training: predict noise
            # x_0 should come from GT (will be passed during training)
            return None, uv, condition
        else:
            # Inference: DDPM sampling
            init_angles = self.init_head(condition)
            angles = self.ddpm_sample(condition, device, start_angles=init_angles)
            return angles, uv, init_angles
    
    def ddpm_sample(self, condition, device, start_angles=None):
        """Deterministic DDIM-style sampling aligned with the noise objective."""
        B = condition.shape[0]
        
        # Start from a direct angle estimate when available, then refine it.
        if start_angles is None:
            x = torch.randn(B, self.num_angles, device=device)
        else:
            x = start_angles.clone()
        
        # Linear beta schedule
        betas = torch.linspace(self.beta_start, self.beta_end, self.num_steps, device=device)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        # Reverse diffusion
        for t in reversed(range(self.num_steps)):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            
            # Time embedding
            t_emb = self.time_mlp(t_batch.float())
            
            # Predict noise
            model_input = torch.cat([x, t_emb, condition], dim=1)
            noise_pred = self.denoise_net(model_input)
            
            alpha_bar_t = alphas_cumprod[t]
            alpha_bar_prev = alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0, device=device)
            
            # Estimate the clean sample x_0 from the current noisy state.
            pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
            pred_x0 = pred_x0.clamp(-4.0, 4.0)
            
            if t > 0:
                # Deterministic update avoids injecting fresh noise during validation.
                x = torch.sqrt(alpha_bar_prev) * pred_x0 + torch.sqrt(1 - alpha_bar_prev) * noise_pred
            else:
                x = pred_x0
        
        return x
    
    def compute_loss_from_condition(self, condition, gt_angles_norm, device,
                                    init_loss_weight=0.5, recon_loss_weight=0.5):
        """Hybrid training loss: noise prediction + direct angle supervision."""
        B = gt_angles_norm.shape[0]

        # Random timestep
        t = torch.randint(0, self.num_steps, (B,), device=device).long()
        
        # Noise schedule
        betas = torch.linspace(self.beta_start, self.beta_end, self.num_steps, device=device)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        # Add noise to GT
        noise = torch.randn_like(gt_angles_norm)
        alpha_t = alphas_cumprod[t].view(-1, 1)
        x_t = torch.sqrt(alpha_t) * gt_angles_norm + torch.sqrt(1 - alpha_t) * noise

        # Direct angle prediction gives the model a much stronger convergence signal.
        init_pred = self.init_head(condition)
        
        # Predict noise
        t_emb = self.time_mlp(t.float())
        model_input = torch.cat([x_t, t_emb, condition], dim=1)
        noise_pred = self.denoise_net(model_input)

        pred_x0 = (x_t - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)

        noise_loss = F.mse_loss(noise_pred, noise)
        init_loss = F.smooth_l1_loss(init_pred, gt_angles_norm, beta=0.5)
        recon_loss = F.smooth_l1_loss(pred_x0, gt_angles_norm, beta=0.5)
        total_loss = noise_loss + init_loss_weight * init_loss + recon_loss_weight * recon_loss

        return {
            'loss': total_loss,
            'noise_loss': noise_loss.detach(),
            'init_loss': init_loss.detach(),
            'recon_loss': recon_loss.detach(),
        }


class DINOv3DiffusionPoseEstimator(nn.Module):
    """DINOv3 + Heatmap + Diffusion angle head"""
    def __init__(self, dino_model_name, heatmap_size, unfreeze_blocks=0, fix_joint7_zero=True,
                 diffusion_steps=20, angle_dropout=0.1):
        super().__init__()
        self.fix_joint7_zero = fix_joint7_zero
        
        self.backbone = DINOv3Backbone(dino_model_name, unfreeze_blocks=unfreeze_blocks)
        feat_dim = self.backbone.model.config.hidden_size
        
        self.keypoint_head = ViTKeypointHead(input_dim=feat_dim, heatmap_size=heatmap_size)
        self.joint_angle_head = DiffusionAngleHead(
            input_dim=feat_dim,
            num_joints=7,
            num_angles=6,
            num_steps=diffusion_steps,
            dropout=angle_dropout,
        )
    
    def forward(self, image_tensor_batch, camera_K=None, training=True, **kwargs):
        dino_features = self.backbone(image_tensor_batch)
        predicted_heatmaps = self.keypoint_head(dino_features)
        
        if training:
            # Return conditions for loss computation
            _, uv, condition = self.joint_angle_head(dino_features, predicted_heatmaps, training=True)
            result = {
                'heatmaps_2d': predicted_heatmaps,
                'joint_angles': None,  # Will compute loss separately
                'condition': condition,
                'uv': uv
            }
        else:
            # Inference: sample from diffusion
            joint_angles_norm, uv, _ = self.joint_angle_head(dino_features, predicted_heatmaps, training=False)
            
            if self.fix_joint7_zero:
                zeros = torch.zeros(joint_angles_norm.shape[0], 1, device=joint_angles_norm.device)
                joint_angles_norm = torch.cat([joint_angles_norm, zeros], dim=1)
            
            result = {
                'heatmaps_2d': predicted_heatmaps,
                'joint_angles': joint_angles_norm,
            }
        
        return result
