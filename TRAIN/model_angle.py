"""
Learned joint-angle predictor (Stage 1.5).

Replaces the global-feature DirectJointAngleHead (which was ill-posed -> ~17-19 deg)
with a head that consumes the DETECTED 2D keypoint geometry + DINOv3 appearance:

  K-normalized bearings + all-pairs bearing differences   (probe: 7.36 deg from 2D ALONE)
  + per-joint heatmap confidence
  + DINOv3 global feature
  + per-keypoint sampled local features (grid_sample at the detected 2D locations)
      -> MLP -> sin/cos -> joint angles

The DINOv3 backbone and the (Stage-1) keypoint head are FROZEN; only AngleHead trains.
Training uses the model's OWN predicted 2D keypoints so the head is robust to real
detector noise. Output feeds the kinematic refiner (Eval/solve_pose_kinematic.py) as init.
"""
import itertools
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_v4 import DINOv3Backbone, ViTKeypointHead, soft_argmax_2d, panda_forward_kinematics

NUM_KP = 7
NUM_ANG = 6  # joint 7 fixed to 0


def to_bearings(kp2d, K):
    """kp2d (B,7,2) px, K (B,3,3) -> per-point bearings (B,7,2) = ((x-cx)/fx,(y-cy)/fy).
    Camera-intrinsics-invariant direction of each keypoint."""
    fx = K[:, 0, 0:1]; fy = K[:, 1, 1:2]
    cx = K[:, 0, 2:3]; cy = K[:, 1, 2:3]
    bx = (kp2d[:, :, 0] - cx) / fx          # (B,7)
    by = (kp2d[:, :, 1] - cy) / fy
    return torch.stack([bx, by], dim=-1)    # (B,7,2)


def keypoints_to_geo(kp2d, K):
    """
    Returns geometric feature (B, 14 + 42) = per-point bearings + all-pairs differences.
    Pairwise differences are translation-invariant relative geometry (encode link orientations).
    """
    bearings = to_bearings(kp2d, K)           # (B,7,2)
    diffs = []
    for i, j in itertools.combinations(range(NUM_KP), 2):
        diffs.append(bearings[:, i] - bearings[:, j])  # (B,2)
    diffs = torch.cat(diffs, dim=1)            # (B,42)
    return torch.cat([bearings.reshape(bearings.shape[0], -1), diffs], dim=1)  # (B,56)


def sample_kp_features(tokens, kp2d, heatmap_size):
    """
    tokens: (B, Npatch, C) DINOv3 patch tokens. kp2d: (B,7,2) px @ heatmap res.
    Returns per-keypoint features (B,7,C) via bilinear grid_sample on the token map.
    """
    B, Np, C = tokens.shape
    s = int(round(Np ** 0.5))
    fmap = tokens.transpose(1, 2).reshape(B, C, s, s)        # (B,C,s,s)
    H = W = heatmap_size
    grid = kp2d.clone()
    grid[..., 0] = grid[..., 0] / (W - 1) * 2 - 1            # -> [-1,1]
    grid[..., 1] = grid[..., 1] / (H - 1) * 2 - 1
    grid = grid.unsqueeze(1)                                  # (B,1,7,2)
    samp = F.grid_sample(fmap, grid, mode='bilinear', align_corners=True)  # (B,C,1,7)
    return samp.squeeze(2).transpose(1, 2)                    # (B,7,C)


def sample_kp_patch(tokens, kp2d, heatmap_size, k=3, step_px=16.0):
    """
    Per-keypoint k×k feature PATCH (vs the single-point sample above). A single bilinear
    sample misses the local ORIENTATION of a link/gripper — which is exactly what wrist-roll
    angles need (the keypoints barely move, so 2D geometry is degenerate; appearance resolves
    it). Returns (B, 7, k*k*C): the k×k grid of token features centered on each keypoint.
    step_px = spacing between grid taps in heatmap px (≈ one 16px patch token).
    """
    B, Np, C = tokens.shape
    s = int(round(Np ** 0.5))
    fmap = tokens.transpose(1, 2).reshape(B, C, s, s)        # (B,C,s,s)
    H = W = heatmap_size
    off = (torch.arange(k, device=kp2d.device, dtype=kp2d.dtype) - (k - 1) / 2) * step_px  # (k,)
    oy, ox = torch.meshgrid(off, off, indexing='ij')         # (k,k)
    offs = torch.stack([ox.reshape(-1), oy.reshape(-1)], dim=-1)  # (k*k,2) px
    # (B,7,1,2) + (1,1,kk,2) -> (B,7,kk,2)
    pts = kp2d.unsqueeze(2) + offs.view(1, 1, -1, 2)
    g = pts.clone()
    g[..., 0] = g[..., 0] / (W - 1) * 2 - 1
    g[..., 1] = g[..., 1] / (H - 1) * 2 - 1                   # (B,7,kk,2)
    samp = F.grid_sample(fmap, g, mode='bilinear', align_corners=True, padding_mode='border')  # (B,C,7,kk)
    samp = samp.permute(0, 2, 3, 1).reshape(B, NUM_KP, k * k * C)  # (B,7,kk*C)
    return samp


class AngleHead(nn.Module):
    def __init__(self, feat_dim=768, hidden=512, n_ang=NUM_ANG, dropout=0.1, kp_in=None):
        super().__init__()
        self.n_ang = n_ang
        kp_in = kp_in or feat_dim          # per-keypoint feature dim (feat_dim, or k*k*feat_dim for patch)
        self.geo_mlp = nn.Sequential(
            nn.Linear(14 + 42, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 256), nn.GELU(),
        )
        self.conf_proj = nn.Sequential(nn.Linear(NUM_KP, 64), nn.GELU())
        self.global_proj = nn.Sequential(nn.Linear(feat_dim, 256), nn.GELU())
        self.kp_proj = nn.Sequential(nn.Linear(kp_in, 128), nn.GELU())
        fuse_in = 256 + 64 + 256 + NUM_KP * 128
        self.fuse = nn.Sequential(
            nn.Linear(fuse_in, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, n_ang * 2),
        )

    def forward(self, geo, conf, gfeat, kpfeat):
        g = self.geo_mlp(geo)
        c = self.conf_proj(conf)
        gl = self.global_proj(gfeat)
        kp = self.kp_proj(kpfeat).flatten(1)          # (B, 7*128)
        x = torch.cat([g, c, gl, kp], dim=1)
        sc = self.fuse(x).view(-1, self.n_ang, 2)
        sc = F.normalize(sc, dim=-1)                  # unit sin/cos
        ang = torch.atan2(sc[..., 0], sc[..., 1])     # (B, n_ang)
        return ang, sc


class AngleHeadMCL(AngleHead):
    """Multiple-hypothesis (Multiple-Choice-Learning) angle head. Under occlusion the posterior
    p(angles|image) is MULTIMODAL (several joint configs explain the visible 2D, esp. base-yaw J0);
    a single regressor lands on the wrong MEAN of the modes. This head emits K hypotheses; an MCL
    'winner-take-all' loss (only the best hypothesis per frame is supervised) makes them SPECIALIZE
    into the modes, and at inference the kinematic solver/reprojection SELECTS the consistent one.
    Returns ang (B,K,n_ang), sc (B,K,n_ang,2)."""
    def __init__(self, feat_dim=768, hidden=512, n_ang=NUM_ANG, dropout=0.1, kp_in=None, n_hyp=4):
        super().__init__(feat_dim=feat_dim, hidden=hidden, n_ang=n_ang, dropout=dropout, kp_in=kp_in)
        self.n_hyp = n_hyp
        # replace the final projection to emit K hypotheses
        in_f = self.fuse[-1].in_features
        self.fuse[-1] = nn.Linear(in_f, n_hyp * n_ang * 2)

    def forward(self, geo, conf, gfeat, kpfeat):
        g = self.geo_mlp(geo)
        c = self.conf_proj(conf)
        gl = self.global_proj(gfeat)
        kp = self.kp_proj(kpfeat).flatten(1)
        x = torch.cat([g, c, gl, kp], dim=1)
        sc = self.fuse(x).view(-1, self.n_hyp, self.n_ang, 2)
        sc = F.normalize(sc, dim=-1)                  # (B,K,n_ang,2)
        ang = torch.atan2(sc[..., 0], sc[..., 1])     # (B,K,n_ang)
        return ang, sc


class RotationHead(nn.Module):
    """
    Predicts the robot->camera rotation as a 6D vector (Zhou et al.) from DINOv3 APPEARANCE +
    keypoint geometry. The far-camera failure (realsense) is the solver landing in a wrong
    rotation basin: 2D reprojection is degenerate at distance, but appearance (which way the
    robot faces) resolves it. Used as the kinematic solver's R_init to pick the right basin
    (oracle R-init = +0.11 realsense ADD-AUC). Heavy on the global/appearance feature.
    """
    def __init__(self, feat_dim=768, hidden=512, dropout=0.1, predict_t=False):
        super().__init__()
        self.predict_t = predict_t
        self.geo_mlp = nn.Sequential(nn.Linear(14 + 42, 256), nn.GELU(), nn.Dropout(dropout))
        self.conf_proj = nn.Sequential(nn.Linear(NUM_KP, 64), nn.GELU())
        self.global_proj = nn.Sequential(nn.Linear(feat_dim, 256), nn.GELU())
        self.kp_proj = nn.Sequential(nn.Linear(feat_dim, 128), nn.GELU())
        self.trunk = nn.Sequential(
            nn.Linear(256 + 64 + 256 + NUM_KP * 128, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.rot_out = nn.Linear(hidden, 6)
        # translation (robot-origin position in camera frame, meters). t_z (depth) is the lever;
        # apparent robot size in the bearings/features -> distance. Predicts a residual around a
        # base depth so the net starts near plausible scale.
        self.t_out = nn.Linear(hidden, 3) if predict_t else None
        self.t_base = torch.tensor([0.0, 0.0, 1.1])

    def forward(self, geo, conf, gfeat, kpfeat):
        x = torch.cat([self.geo_mlp(geo), self.conf_proj(conf), self.global_proj(gfeat),
                       self.kp_proj(kpfeat).flatten(1)], dim=1)
        h = self.trunk(x)
        d6 = self.rot_out(h)
        if self.predict_t:
            t = self.t_out(h) + self.t_base.to(h.device)
            return d6, t
        return d6


def rot6d_to_matrix(d6):
    """(B,6) 6D rotation -> (B,3,3). Zhou et al. 2019."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def kabsch_batch(P, Q, w):
    """Batched rigid (R,t) minimizing ||R P + t - Q||. P,Q (B,N,3), w (B,N). -> R (B,3,3), t (B,3)."""
    w = w / w.sum(1, keepdim=True).clamp(min=1e-6)
    mp = (w.unsqueeze(-1) * P).sum(1, keepdim=True); mq = (w.unsqueeze(-1) * Q).sum(1, keepdim=True)
    H = ((w.unsqueeze(-1) * (P - mp)).transpose(1, 2)) @ (Q - mq)   # (B,3,3)
    U, S, Vt = torch.linalg.svd(H)
    d = torch.sign(torch.linalg.det(Vt.transpose(1, 2) @ U.transpose(1, 2)))
    D = torch.eye(3, device=P.device).unsqueeze(0).repeat(P.shape[0], 1, 1)
    D[:, 2, 2] = d
    R = Vt.transpose(1, 2) @ D @ U.transpose(1, 2)
    t = mq.squeeze(1) - torch.bmm(R, mp.transpose(1, 2)).squeeze(2)   # (B,3)
    return R, t


class AngleHeadTransformer(nn.Module):
    """
    DETR-style alternative to the MLP head. Each detected keypoint becomes a token
    (sampled feature + bearing + joint-id + confidence); a Transformer ENCODER models
    inter-joint relations; N_ANG learnable angle queries CROSS-ATTEND (decoder) to the
    keypoint tokens and each emits sin/cos. Attention is a better inductive bias for the
    kinematic chain than a flat concat-MLP.
    """
    def __init__(self, feat_dim=768, d=256, n_kp=NUM_KP, n_ang=NUM_ANG,
                 enc_layers=3, dec_layers=3, heads=8, dropout=0.1):
        super().__init__()
        self.n_ang = n_ang
        self.feat_proj = nn.Linear(feat_dim, d)
        self.bearing_proj = nn.Linear(2, d)
        self.conf_proj = nn.Linear(1, d)
        self.kp_id = nn.Embedding(n_kp, d)
        enc = nn.TransformerEncoderLayer(d, heads, dim_feedforward=4 * d, dropout=dropout,
                                         activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, enc_layers)
        self.angle_queries = nn.Parameter(torch.randn(n_ang, d) * 0.02)
        dec = nn.TransformerDecoderLayer(d, heads, dim_feedforward=4 * d, dropout=dropout,
                                         activation='gelu', batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec, dec_layers)
        self.out = nn.Linear(d, 2)

    def forward(self, kpfeat, bearings, conf):
        # kpfeat (B,7,C), bearings (B,7,2), conf (B,7)
        B = kpfeat.shape[0]
        ids = torch.arange(NUM_KP, device=kpfeat.device)
        tok = (self.feat_proj(kpfeat) + self.bearing_proj(bearings)
               + self.conf_proj(conf.unsqueeze(-1)) + self.kp_id(ids).unsqueeze(0))
        mem = self.encoder(tok)                                 # (B,7,d)
        q = self.angle_queries.unsqueeze(0).expand(B, -1, -1)   # (B,n_ang,d)
        dec = self.decoder(q, mem)                              # (B,n_ang,d)
        sc = F.normalize(self.out(dec), dim=-1)                 # (B,n_ang,2)
        ang = torch.atan2(sc[..., 0], sc[..., 1])
        return ang, sc


class AnglePredictor(nn.Module):
    """Frozen backbone + frozen keypoint head + trainable angle head ('mlp' or 'transformer')."""
    def __init__(self, dino_model_name, heatmap_size, fix_joint7_zero=True, head_type='mlp',
                 with_rotation=False, with_translation=False, n_hyp=4):
        super().__init__()
        self.heatmap_size = heatmap_size if isinstance(heatmap_size, int) else heatmap_size[0]
        self.fix_joint7_zero = fix_joint7_zero
        self.head_type = head_type
        self.backbone = DINOv3Backbone(dino_model_name, unfreeze_blocks=0)
        feat_dim = self.backbone.model.config.hidden_size
        self.keypoint_head = ViTKeypointHead(
            input_dim=feat_dim, heatmap_size=(self.heatmap_size, self.heatmap_size))
        self.kp_patch = 3 if head_type == 'mlp_patch' else 0  # k×k end-effector appearance patch
        self.n_hyp = n_hyp if head_type == 'mlp_mcl' else 1
        if head_type == 'transformer':
            self.angle_head = AngleHeadTransformer(feat_dim=feat_dim)
        elif head_type == 'mlp_patch':
            self.angle_head = AngleHead(feat_dim=feat_dim, kp_in=feat_dim * self.kp_patch ** 2)
        elif head_type == 'mlp_mcl':
            self.angle_head = AngleHeadMCL(feat_dim=feat_dim, n_hyp=n_hyp)
        else:
            self.angle_head = AngleHead(feat_dim=feat_dim)
        self.with_rotation = with_rotation
        self.with_translation = with_translation
        self.rot_head = RotationHead(feat_dim=feat_dim, predict_t=with_translation) if with_rotation else None

    def freeze_detector(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.keypoint_head.parameters():
            p.requires_grad = False
        self.backbone.eval()
        self.keypoint_head.eval()

    def forward(self, image, camera_K, kp_jitter=0.0, kp_drop=0.0):
        with torch.no_grad():
            tokens = self.backbone(image)                 # (B,Np,C)
            heatmaps = self.keypoint_head(tokens)         # (B,7,H,W)
            _win = int(os.environ.get('DECODE_WINDOW', '0'))   # 0 = global soft-argmax (default/locked)
            if _win > 0:
                from model_v4 import windowed_soft_argmax_2d
                kp2d = windowed_soft_argmax_2d(heatmaps, win=_win)   # robust to distractor 2nd-modes
            else:
                kp2d = soft_argmax_2d(heatmaps)           # (B,7,2) @ heatmap res
            conf = heatmaps.flatten(2).max(dim=2)[0]      # (B,7)
        if kp_drop > 0.0:
            # OCCLUSION augmentation: randomly "occlude" keypoints (low conf + displaced position) to
            # induce the under-determination that makes p(angles|image) multimodal. Forces the MCL
            # hypotheses to diversify into the modes (synth has no real occlusion to do this).
            mask = torch.rand(conf.shape, device=conf.device) < kp_drop          # (B,7) occluded
            conf = conf * torch.where(mask, torch.full_like(conf, 0.05), torch.ones_like(conf))
            kp2d = kp2d + mask.unsqueeze(-1).float() * torch.randn_like(kp2d) * 25.0
        if kp_jitter > 0.0:
            # train-time keypoint-noise augmentation: the head AMPLIFIES detector 2D error on the
            # gauge-sensitive base joint (realsense J0 44 deg vs 14 deg oracle-2D). Jittering the
            # 2D fed to geo+sampling teaches robustness so J0 leans on appearance, not exact bearings.
            kp2d = kp2d + torch.randn_like(kp2d) * kp_jitter
        if self.head_type == 'transformer':
            kpfeat = sample_kp_features(tokens, kp2d, self.heatmap_size)
            bearings = to_bearings(kp2d, camera_K)            # (B,7,2)
            ang, sc = self.angle_head(kpfeat, bearings, conf)
        else:
            if self.kp_patch:
                kpfeat = sample_kp_patch(tokens, kp2d, self.heatmap_size, k=self.kp_patch)
            else:
                kpfeat = sample_kp_features(tokens, kp2d, self.heatmap_size)
            geo = keypoints_to_geo(kp2d, camera_K)
            gfeat = tokens.mean(dim=1)
            ang, sc = self.angle_head(geo, conf, gfeat, kpfeat)   # (B,6),(B,6,2)
        mcl = self.head_type == 'mlp_mcl'              # ang (B,K,6), sc (B,K,6,2)
        if self.fix_joint7_zero:
            zpad = torch.zeros(*ang.shape[:-1], 1, device=ang.device)
            ang_full = torch.cat([ang, zpad], dim=-1)  # (B,7) or (B,K,7)
        else:
            ang_full = ang
        out = {
            'joint_angles': ang_full,   # (B,7) or (B,K,7) radians
            'sin_cos': sc,              # (B,6,2) or (B,K,6,2)
            'keypoints_2d': kp2d,       # (B,7,2)
            'heatmaps_2d': heatmaps,
            'confidence': conf,
            'is_mcl': mcl,
            'global_feat': tokens.mean(dim=1),  # frozen DINOv3 appearance (situation-router input)
        }
        if self.with_rotation:
            # reuse the mlp-path geometry/appearance features for the rotation head
            geo_r = keypoints_to_geo(kp2d, camera_K)
            gfeat_r = tokens.mean(dim=1)
            kpfeat_r = sample_kp_features(tokens, kp2d, self.heatmap_size)
            r = self.rot_head(geo_r, conf, gfeat_r, kpfeat_r)
            d6, tpred = r if self.with_translation else (r, None)
            out['rot6d'] = d6
            out['rot_matrix'] = rot6d_to_matrix(d6)              # (B,3,3) robot->camera
            if tpred is not None:
                out['trans'] = tpred                             # (B,3) camera-frame meters
        return out
