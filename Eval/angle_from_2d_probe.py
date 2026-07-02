"""
Cheap CPU probe: can a small net map K-normalized 2D keypoint bearings -> joint angles
BETTER than (a) global-feature direct regression (~17-19 deg) and (b) the geometric
cold-start solver (~23 deg even with oracle 2D)?

Rationale (plan resilient-sparking-castle.md, EMPIRICAL FINDING): the 19.7 deg came from a
WEAK learned init, not the refiner. Single-view 2D->angle is ambiguous for pure geometry,
but a LEARNED prior can resolve it. We normalize each 2D keypoint by the camera intrinsics
into a bearing ((x-cx)/fx, (y-cy)/fy) so the mapping is camera-independent, then fit a tiny
MLP with a sin/cos head. This needs NO detector forward and NO training GPU — it runs on CPU
from cached GT 2D, so it won't compete with the detector runs.

If this lands in single digits from GT 2D (and stays low under 2D noise), it is a strong
learned-init candidate for the kinematic refiner -> directly attacks the 19.7 deg problem.
"""
import argparse, glob, json, math, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # force CPU, never touch the busy GPUs

NAMES = ['panda_link0', 'panda_link2', 'panda_link3', 'panda_link4',
         'panda_link6', 'panda_link7', 'panda_hand']


def extract(json_dir, limit):
    files = sorted(glob.glob(os.path.join(json_dir, "*.json")))[:limit]
    KP, K, ANG = [], [], []
    for f in files:
        d = json.load(open(f))
        kpd = {k['name']: k for o in d.get('objects', []) for k in o.get('keypoints', [])}
        if not all(n in kpd for n in NAMES):
            continue
        if 'meta' not in d or 'K' not in d['meta'] or 'sim_state' not in d:
            continue
        KP.append([kpd[n]['projected_location'] for n in NAMES])
        K.append(d['meta']['K'])
        a = np.zeros(7, np.float32)
        for i, j in enumerate(d['sim_state']['joints'][:7]):
            a[i] = j['position']
        ANG.append(a)
    return (np.array(KP, np.float32), np.array(K, np.float32), np.array(ANG, np.float32))


def to_bearings(kp, K):
    """kp (N,7,2) px, K (N,3,3) -> bearings (N,7,2) = ((x-cx)/fx,(y-cy)/fy)."""
    fx, fy = K[:, 0, 0:1], K[:, 1, 1:2]
    cx, cy = K[:, 0, 2:3], K[:, 1, 2:3]
    bx = (kp[:, :, 0] - cx) / fx
    by = (kp[:, :, 1] - cy) / fy
    return np.stack([bx, by], axis=-1)  # (N,7,2)


def to_features(kp, K):
    """Per-point bearings (14) + all-pairs bearing DIFFERENCES (21*2=42, translation-invariant).
    Relative geometry directly encodes link orientations -> resolves much of the ambiguity.
    Returns (N, 56)."""
    b = to_bearings(kp, K)            # (N,7,2)
    N = b.shape[0]
    diffs = []
    for i in range(7):
        for j in range(i + 1, 7):
            diffs.append(b[:, i] - b[:, j])   # (N,2)
    diffs = np.concatenate(diffs, axis=1)     # (N,42)
    return np.concatenate([b.reshape(N, -1), diffs], axis=1)  # (N,56)


class AngleNet(nn.Module):
    def __init__(self, n_in=14, hidden=512, n_ang=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, n_ang * 2),  # sin/cos
        )
        self.n_ang = n_ang

    def forward(self, x):
        sc = self.net(x).view(-1, self.n_ang, 2)
        sc = F.normalize(sc, dim=-1)             # unit sin/cos
        ang = torch.atan2(sc[..., 0], sc[..., 1])
        return ang, sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default="../Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_train_dr")
    ap.add_argument("--val-dir", default="../Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_test_dr")
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-val", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--noise-px", type=float, default=3.0, help="train-time 2D noise (px) to simulate detector error")
    ap.add_argument("--cache", default="Eval/_angle_probe_cache.npz")
    args = ap.parse_args()

    if os.path.exists(args.cache):
        z = np.load(args.cache)
        kp_tr, K_tr, a_tr, kp_va, K_va, a_va = (z['kp_tr'], z['K_tr'], z['a_tr'],
                                                z['kp_va'], z['K_va'], z['a_va'])
        print(f"loaded cache: train {len(a_tr)}  val {len(a_va)}")
    else:
        print("extracting train...")
        kp_tr, K_tr, a_tr = extract(args.train_dir, args.n_train)
        print("extracting val...")
        kp_va, K_va, a_va = extract(args.val_dir, args.n_val)
        os.makedirs(os.path.dirname(args.cache), exist_ok=True)
        np.savez(args.cache, kp_tr=kp_tr, K_tr=K_tr, a_tr=a_tr, kp_va=kp_va, K_va=K_va, a_va=a_va)
        print(f"cached: train {len(a_tr)}  val {len(a_va)}")

    # mean focal for converting px-noise -> bearing-noise
    f_mean = float(K_tr[:, 0, 0].mean())
    b_tr = torch.tensor(to_features(kp_tr, K_tr))
    b_va = torch.tensor(to_features(kp_va, K_va))
    y_tr = torch.tensor(a_tr[:, :6]); y_va = torch.tensor(a_va[:, :6])

    net = AngleNet(n_in=b_tr.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    bs = 512
    N = len(b_tr)

    def angle_mae(pred, gt):
        d = torch.atan2(torch.sin(pred - gt), torch.cos(pred - gt))
        return d.abs().mean(0) * 180 / math.pi  # (6,)

    g = torch.Generator().manual_seed(0)
    for ep in range(args.epochs):
        net.train()
        perm = torch.randperm(N, generator=g)
        for i in range(0, N, bs):
            idx = perm[i:i + bs]
            xb = b_tr[idx].clone()
            if args.noise_px > 0:  # simulate detector localization error
                xb = xb + torch.randn(xb.shape, generator=g) * (args.noise_px / f_mean)
            yb = y_tr[idx]
            ang, sc = net(xb)
            gt_sc = torch.stack([torch.sin(yb), torch.cos(yb)], -1)
            loss = F.smooth_l1_loss(sc, gt_sc)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if ep % 20 == 0 or ep == args.epochs - 1:
            net.eval()
            with torch.no_grad():
                pa, _ = net(b_va)
                mae = angle_mae(pa, y_va)
                # noised val (detector-like)
                pa_n, _ = net(b_va + torch.randn(b_va.shape, generator=g) * (args.noise_px / f_mean))
                mae_n = angle_mae(pa_n, y_va)
            print(f"ep {ep:3d}  val MAE(J0-5)={mae.mean():.2f}  noised={mae_n.mean():.2f}  "
                  f"per-joint=" + ",".join(f"{m:.1f}" for m in mae))

    print("\n=== SUMMARY ===")
    print(f"clean GT-2D val MAE(J0-5): {mae.mean():.2f} deg")
    print(f"noised(+{args.noise_px}px) val MAE(J0-5): {mae_n.mean():.2f} deg")
    print("compare: global-feature regression ~17-19 deg, geometric cold-start ~23 deg")


if __name__ == "__main__":
    main()
