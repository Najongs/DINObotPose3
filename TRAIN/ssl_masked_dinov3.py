"""
Stage 4 — SSL backbone adaptation (masked-feature prediction, data2vec/BYOL style) on UNLABELED
DREAM-real. The detector/angle/rot heads were all adapted to real via solver-distillation, but the
DINOv3 BACKBONE is still frozen synth-pretrained weights — the root sim2real gap. Adapt it on real
pixels with NO labels: mask ~50% of patches in the student input, predict the EMA-teacher's features
at the masked patch positions from the visible context. This is the robust HF-compatible cousin of
I-JEPA (no RoPE/token-drop surgery: both student and teacher use the standard full forward; masking is
done in pixel space). Output = an adapted backbone to drop into train_heatmap.py for a real-tuned
detector.

Honesty: pool the CONTIGUOUS first-70% adapt frames per camera (same split the head self-train used) so
the held-out last-30% eval stays unseen. SSL is label-free but we still keep eval frames out of pretrain.
"""
import argparse, os, sys, math, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)
from model import DINOv3Backbone                # HF DINOv3 wrapper (freeze + unfreeze last N)
from dataset import PoseEstimationDataset
from train_heatmap import HeatmapModel          # for the final merge convenience

KP = ['panda_link0', 'panda_link2', 'panda_link3', 'panda_link4', 'panda_link6', 'panda_link7', 'panda_hand']


class Predictor(nn.Module):
    """Small per-token MLP predictor (BYOL-style) mapping student patch features -> teacher feature space."""
    def __init__(self, dim=768, hidden=2048):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):                         # (B,N,D)
        return self.net(x)


def build_pixel_mask(B, gh, gw, patch, ratio, device, generator):
    """Random patch mask. Returns (mask_patches (B,N) bool=masked, pixel_keep (B,1,H,W) float=1 where VISIBLE)."""
    N = gh * gw
    nmask = int(round(ratio * N))
    mask = torch.zeros(B, N, dtype=torch.bool, device=device)
    for b in range(B):
        idx = torch.randperm(N, generator=generator, device=device)[:nmask]
        mask[b, idx] = True
    # expand to pixel keep-map (1 where visible, 0 where masked)
    pm = (~mask).float().view(B, 1, gh, gw)
    pixel_keep = F.interpolate(pm, scale_factor=patch, mode='nearest')   # (B,1,H,W)
    return mask, pixel_keep


@torch.no_grad()
def ema_update(student, teacher, m):
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.mul_(m).add_(ps.data, alpha=1 - m)
    for bs, bt in zip(student.buffers(), teacher.buffers()):
        bt.data.copy_(bs.data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--real-dirs', nargs='+', required=True, help='one or more DREAM-real camera dirs to pool')
    ap.add_argument('--model-name', default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    ap.add_argument('--init-detector', default=None, help='optional: load backbone weights from this detector ckpt to start from (else HF pretrained)')
    ap.add_argument('--image-size', type=int, default=224)
    ap.add_argument('--patch', type=int, default=16)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--adapt-frac', type=float, default=0.7)
    ap.add_argument('--adapt-cap', type=int, default=6000, help='per-camera cap on pooled adapt frames')
    ap.add_argument('--unfreeze-blocks', type=int, default=6, help='last N of 12 ViT blocks made trainable')
    ap.add_argument('--mask-ratio', type=float, default=0.5)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--lr', type=float, default=3e-5)
    ap.add_argument('--ema', type=float, default=0.996)
    ap.add_argument('--output-dir', default='./outputs_ssl')
    ap.add_argument('--save-merged-from', default=None, help='detector ckpt to merge the adapted backbone into (produces a ready-to-finetune detector ckpt)')
    args = ap.parse_args()

    device = torch.device('cuda'); assert torch.cuda.is_available()
    os.makedirs(args.output_dir, exist_ok=True)
    IS, P = args.image_size, args.patch
    gh = gw = IS // P
    gen = torch.Generator(device=device); gen.manual_seed(0)

    # ---- pooled real adapt frames (contiguous first 70% per camera, capped) ----
    subsets = []
    for rd in args.real_dirs:
        ds = PoseEstimationDataset(rd, keypoint_names=KP, image_size=(IS, IS), heatmap_size=(IS, IS),
                                   augment=True, aug_level='strong', include_angles=False)
        N = len(ds.samples); cut = int(args.adapt_frac * N)
        idx = list(range(cut))
        if args.adapt_cap > 0 and len(idx) > args.adapt_cap:
            st = max(1, len(idx) // args.adapt_cap); idx = idx[::st][:args.adapt_cap]
        subsets.append(Subset(ds, idx))
        print(f"  {os.path.basename(rd)}: N={N} adapt={len(idx)}", flush=True)
    pool = ConcatDataset(subsets)
    loader = DataLoader(pool, batch_size=args.batch_size, shuffle=True, num_workers=8,
                        pin_memory=True, drop_last=True)
    print(f"[SSL] pooled adapt frames: {len(pool)}  steps/epoch={len(loader)}", flush=True)

    # ---- student backbone (trainable last N) + EMA teacher (frozen copy) ----
    student = DINOv3Backbone(args.model_name, unfreeze_blocks=0).to(device)
    # DINOv3Backbone's unfreeze logic looks for .encoder.layers/.blocks, but HF DINOv3 exposes the
    # transformer blocks as model.layer (ModuleList) -> do the unfreeze explicitly here.
    blocks = None
    for attr in ['layer', 'blocks']:
        if hasattr(student.model, attr):
            blocks = getattr(student.model, attr); break
    if blocks is None and hasattr(student.model, 'encoder'):
        blocks = getattr(student.model.encoder, 'layer', getattr(student.model.encoder, 'layers', None))
    assert blocks is not None, "could not find transformer blocks to unfreeze"
    nblk = len(blocks); ntrain = 0
    for i in range(max(0, nblk - args.unfreeze_blocks), nblk):
        for p in blocks[i].parameters():
            p.requires_grad = True; ntrain += p.numel()
    # also unfreeze the final norm (cheap, helps adaptation)
    if hasattr(student.model, 'norm'):
        for p in student.model.norm.parameters():
            p.requires_grad = True; ntrain += p.numel()
    print(f"[SSL] unfroze last {args.unfreeze_blocks}/{nblk} blocks + final norm ({ntrain/1e6:.1f}M trainable params)", flush=True)
    if args.init_detector:
        sd = torch.load(args.init_detector, map_location=device)
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        bb = {k[len('backbone.'):]: v for k, v in sd.items() if k.startswith('backbone.')}
        miss = student.load_state_dict(bb, strict=False)
        print(f"[SSL] init backbone from {args.init_detector} (missing {len(miss.missing_keys)})", flush=True)
    teacher = copy.deepcopy(student).to(device)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()
    predictor = Predictor(student.model.config.hidden_size).to(device)

    params = [p for p in student.parameters() if p.requires_grad] + list(predictor.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.05)
    total_steps = args.epochs * len(loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.1)

    step = 0
    for epoch in range(args.epochs):
        student.train(); predictor.train()
        running = 0.0; nb = 0
        for batch in tqdm(loader, desc=f'Ep{epoch} ssl', leave=False):
            img = batch['image'].to(device, non_blocking=True)             # (B,3,IS,IS) normalized
            B = img.shape[0]
            mask, pixel_keep = build_pixel_mask(B, gh, gw, P, args.mask_ratio, device, gen)
            student_in = img * pixel_keep                                  # zero (==~mean) the masked patches

            with torch.no_grad():
                tgt = teacher(img)                                         # (B,N,D) clean features
                tgt = F.layer_norm(tgt, (tgt.shape[-1],))                  # normalize target (data2vec)
            feat = student(student_in)                                     # (B,N,D)
            pred = predictor(feat)
            # loss only on masked patch positions
            m = mask.unsqueeze(-1)                                         # (B,N,1)
            diff = F.smooth_l1_loss(pred, tgt, reduction='none')
            loss = (diff * m).sum() / (m.sum() * tgt.shape[-1] + 1e-6)

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step()
            mom = 1.0 - (1.0 - args.ema) * (math.cos(math.pi * step / total_steps) + 1) / 2  # ->1.0
            ema_update(student, teacher, mom)
            step += 1; running += float(loss); nb += 1
        print(f"Ep{epoch} | ssl masked-feat loss {running/max(nb,1):.4f} (mom {mom:.4f})", flush=True)
        # checkpoint each epoch (flush often — flaky GPU)
        torch.save(student.state_dict(), os.path.join(args.output_dir, 'ssl_backbone.pth'))

    # ---- save adapted backbone, and optionally a merged ready-to-finetune detector ckpt ----
    torch.save(student.state_dict(), os.path.join(args.output_dir, 'ssl_backbone.pth'))
    print(f"[SSL] saved adapted backbone -> {args.output_dir}/ssl_backbone.pth", flush=True)

    if args.save_merged_from:
        base = torch.load(args.save_merged_from, map_location='cpu')
        base = {k.replace('module.', ''): v for k, v in base.items()}
        adapted = {('backbone.' + k): v.cpu() for k, v in student.state_dict().items()}
        nrep = 0
        for k in list(base.keys()):
            if k in adapted:
                base[k] = adapted[k]; nrep += 1
        out = os.path.join(args.output_dir, 'merged_detector_sslbackbone.pth')
        torch.save(base, out)
        print(f"[SSL] merged adapted backbone into {args.save_merged_from} ({nrep} backbone keys) -> {out}", flush=True)


if __name__ == '__main__':
    main()
