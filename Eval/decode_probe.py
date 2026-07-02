"""
Is the 2D keypoint TAIL (esp. base link0, mean ~51px) caused by a DISTRACTOR-pulled global soft-argmax
(bimodal heatmap) — fixable by a windowed decode for FREE — or by a genuinely WRONG peak (decode can't help)?

For each real frame, decode the detector heatmaps 3 ways and compare to GT 2D @512:
  global-softargmax (current, temp=100) | hard-argmax (peak) | windowed-softargmax (±win around peak)
Report per-kp median + p90 (the tail) + the base. Also heatmap 'mass-outside-window' = bimodality proxy.
  windowed ≪ global on the tail  => global soft-argmax is distractor-pulled -> windowed decode = free win.
  all three ~equal               => the peak itself is wrong -> decode won't help (need crop/backbone).
"""
import argparse, glob, os, sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../TRAIN')))
sys.path.append(os.path.dirname(__file__))
from model_angle import AnglePredictor
from viz_hypothesis import load_frame, SHORT


def decode_all(hm, win=15, temp=100.0):
    """hm (B,7,H,W). Return dict of (B,7,2) px coords for 3 decoders + mass-outside-window (B,7)."""
    B, K, H, W = hm.shape
    flat = hm.flatten(2)                                   # B,7,HW
    # global soft-argmax (current)
    w = F.softmax(flat * temp, dim=-1)
    xs = torch.arange(W, device=hm.device).float(); ys = torch.arange(H, device=hm.device).float()
    gx = (w.view(B, K, H, W).sum(2) * xs).sum(-1); gy = (w.view(B, K, H, W).sum(3) * ys).sum(-1)
    glob = torch.stack([gx, gy], -1)
    # hard argmax
    idx = flat.argmax(-1); hy = (idx // W).float(); hx = (idx % W).float()
    hard = torch.stack([hx, hy], -1)
    # windowed soft-argmax around the hard peak
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    yy = yy.reshape(-1); xx = xx.reshape(-1)
    dist = (xx[None, None] - hx[..., None]).abs().clamp(max=999) + (yy[None, None] - hy[..., None]).abs().clamp(max=999)
    inwin = ((xx[None, None] - hx[..., None]).abs() <= win) & ((yy[None, None] - hy[..., None]).abs() <= win)
    masked = flat.masked_fill(~inwin, -1e4)
    ww = F.softmax(masked * temp, dim=-1)
    wx = (ww * xx[None, None]).sum(-1); wy = (ww * yy[None, None]).sum(-1)
    wind = torch.stack([wx, wy], -1)
    # bimodality proxy: softmax mass OUTSIDE the window
    mass_out = (F.softmax(flat * temp, dim=-1) * (~inwin).float()).sum(-1)
    return {'global': glob, 'hard': hard, 'windowed': wind}, mass_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--detector', required=True); ap.add_argument('--mlp-head', required=True)
    ap.add_argument('--rot-head', default=None)
    ap.add_argument('--val-dirs', nargs='+', required=True)
    ap.add_argument('--model-name', default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    ap.add_argument('--image-size', type=int, default=512); ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--max-frames', type=int, default=300); ap.add_argument('--win', type=int, default=15)
    args = ap.parse_args()
    device = torch.device('cuda'); S = args.image_size

    m = AnglePredictor(args.model_name, S, head_type='mlp', with_rotation=args.rot_head is not None,
                       with_translation=args.rot_head is not None).to(device).eval()
    sd = torch.load(args.detector, map_location=device); sd = {k.replace('module.', ''): v for k, v in sd.items()}
    m.load_state_dict({k: v for k, v in sd.items() if k in m.state_dict() and v.shape == m.state_dict()[k].shape}, strict=False)
    m.angle_head.load_state_dict(torch.load(args.mlp_head, map_location=device))

    err = {d: [] for d in ['global', 'hard', 'windowed']}; massO = []
    for vd in args.val_dirs:
        files = sorted(glob.glob(os.path.join(vd, '*.json')))
        if args.max_frames and args.max_frames < len(files):
            st = max(1, len(files) // args.max_frames); files = files[::st][:args.max_frames]
        buf = []
        def flush():
            if not buf: return
            imgs = torch.stack([b['img'] for b in buf]).to(device)
            with torch.no_grad():
                tok = m.backbone(imgs)
                hm = m.keypoint_head(tok)                  # (B,7,H,W) @ heatmap res (512)
                Hh = hm.shape[-1]
                dec, mo = decode_all(hm, win=args.win)
            for i, b in enumerate(buf):
                f = b['found']; sx, sy = Hh / b['W'], Hh / b['H']
                gt = b['gt2d'] * np.array([sx, sy])        # GT @ heatmap res
                for d in err:
                    p = dec[d][i].cpu().numpy()
                    e = np.full(7, np.nan)
                    for k in range(7):
                        if f[k]: e[k] = np.linalg.norm(p[k] - gt[k]) * (S / Hh)   # back to @512 px
                    err[d].append(e)
                massO.append(mo[i].cpu().numpy())
            buf.clear()
        for jf in files:
            fr = load_frame(jf, vd, S)
            if fr is None: continue
            buf.append(fr)
            if len(buf) >= args.batch_size: flush()
        flush()

    E = {d: np.array(err[d]) for d in err}; MO = np.array(massO)
    n = len(E['global'])
    print(f"\n===== DECODE PROBE (n={n} real frames, win=±{args.win}px @512, heatmap 512) =====")
    print(f"{'kp':<7}{'mass_out':>9} | {'global med/p90/mean':>22} | {'hard':>16} | {'windowed med/p90/mean':>22}")
    for k in range(7):
        def stats(d):
            v = E[d][:, k]; v = v[~np.isnan(v)]
            return np.median(v), np.percentile(v, 90), v.mean()
        gm, gp, gA = stats('global'); hm_, hp, hA = stats('hard'); wm, wp, wA = stats('windowed')
        print(f"{SHORT[k]:<7}{np.nanmedian(MO[:,k]):>9.2f} | {gm:>6.1f}/{gp:>5.1f}/{gA:>5.1f} | "
              f"{hm_:>5.1f}/{hp:>5.1f} | {wm:>6.1f}/{wp:>5.1f}/{wA:>5.1f}")
    def allstat(d):
        v = E[d][~np.isnan(E[d])]; return np.median(v), np.percentile(v, 90), v.mean()
    for d in err:
        md, p9, mn = allstat(d); print(f"  OVERALL {d:<9} median {md:.2f}  p90 {p9:.1f}  mean {mn:.1f} px")
    # base-specific verdict
    gb = E['global'][:, 0]; wb = E['windowed'][:, 0]; gb = gb[~np.isnan(gb)]; wb = wb[~np.isnan(wb)]
    print(f"\n  BASE(link0): global mean {gb.mean():.1f}px -> windowed mean {wb.mean():.1f}px "
          f"({100*(gb.mean()-wb.mean())/gb.mean():+.0f}%);  base mass_out median {np.nanmedian(MO[:,0]):.2f}")
    if wb.mean() < 0.85 * gb.mean():
        print("  READ: windowed decode cuts the base tail => global soft-argmax IS distractor-pulled. FREE win.")
    else:
        print("  READ: windowed ~ global => the PEAK itself is wrong; decode won't fix it (need crop/context).")


if __name__ == '__main__':
    main()
