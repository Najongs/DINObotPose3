"""
Detector PCK at ORIGINAL resolution (640x480) for a clean head-to-head vs RoboPEPP's PCK.

Our detector runs at 512; PCK@Npx at 512 is more lenient than at 640. This rescales predicted
AND GT keypoints back to original_size and computes PCK@2.5/5/10px at that scale, matching how
RoboPEPP/DREAM report PCK. Detector-only (no solve). If our PCK >= RoboPEPP's while our ADD-AUC
is lower, the bottleneck is the keypoint->pose lift, NOT the detector.
"""
import argparse, os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

TRAIN = os.path.abspath(os.path.join(os.path.dirname(__file__), '../TRAIN'))
sys.path.append(TRAIN); sys.path.append(os.path.dirname(__file__))
from model_angle import AnglePredictor
from dataset import PoseEstimationDataset
from refine_eval import scale_K  # noqa (unused but keeps import parity)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--detector', required=True); ap.add_argument('--mlp-head', required=True)
    ap.add_argument('--val-dir', required=True)
    ap.add_argument('--model-name', default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    ap.add_argument('--image-size', type=int, default=512); ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--max-frames', type=int, default=1000)
    args = ap.parse_args()

    device = torch.device('cuda'); assert torch.cuda.is_available(); S = args.image_size
    m = AnglePredictor(args.model_name, S, head_type='mlp').to(device).eval()
    sd = torch.load(args.detector, map_location=device); sd = {k.replace('module.', ''): v for k, v in sd.items()}
    m.load_state_dict({k: v for k, v in sd.items() if k in m.state_dict() and v.shape == m.state_dict()[k].shape}, strict=False)
    m.angle_head.load_state_dict(torch.load(args.mlp_head, map_location=device))

    ds = PoseEstimationDataset(args.val_dir, keypoint_names=['link0','link2','link3','link4','link6','link7','hand'],
                               image_size=(S, S), heatmap_size=(S, S), augment=False, include_angles=False, sigma=2.5)
    if args.max_frames and args.max_frames < len(ds): ds.samples = ds.samples[:args.max_frames]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    KPN = ['link0','link2','link3','link4','link6','link7','hand']
    errs = []                                  # all keypoints (orig px)
    per = [[] for _ in range(7)]               # per-keypoint
    for batch in tqdm(loader, desc="pck"):
        img = batch['image'].to(device)
        gtkp = batch['keypoints'].to(device)              # (B,7,2) px @ S (=512)
        osz = batch['original_size'].to(device)           # (B,2) = (W,H)
        Ks = scale_K(batch['camera_K'], batch['original_size'], S).to(device)
        with torch.no_grad():
            out = m(img, Ks)
        kp = out['keypoints_2d']                           # (B,7,2) px @ S
        for b in range(img.shape[0]):
            ow, oh = float(osz[b, 0]), float(osz[b, 1])
            sx, sy = ow / S, oh / S
            for j in range(7):
                gx, gy = float(gtkp[b, j, 0]), float(gtkp[b, j, 1])
                if not (0 <= gx < S and 0 <= gy < S):      # skip off-frame GT (PCK convention)
                    continue
                e = (((float(kp[b, j, 0]) - gx) * sx) ** 2 + ((float(kp[b, j, 1]) - gy) * sy) ** 2) ** 0.5
                errs.append(e); per[j].append(e)
    errs = np.array(errs)
    print(f"\nPCK @ orig-res  {os.path.basename(args.val_dir)}  (n={len(errs)} kp, {len(ds)} frames)")
    for thr in [2.5, 5, 10]:
        print(f"   PCK@{thr:>4}px: {(errs <= thr).mean():.3f}")
    print(f"   median L2: {np.median(errs):.2f}px  mean L2: {errs.mean():.2f}px")
    print(f"   per-keypoint [median / mean L2 px / PCK@5]:")
    for j, nm in enumerate(KPN):
        e = np.array(per[j]); tag = '  <- wrist' if nm in ('link6','link7','hand') else ''
        if len(e): print(f"     {nm:<6} {np.median(e):6.2f} / {e.mean():6.2f} / {(e<=5).mean():.2f}{tag}")


if __name__ == '__main__':
    main()
