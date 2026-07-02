"""
Why are wrist angles bad (ours J5/J6 ~13.7/11.9 vs RoboPEPP 4.9/5.4)?
Is it (a) the detector's wrist KEYPOINTS, or (b) the angle HEAD can't extract wrist angle
even from perfect 2D (geometric under-determination -> needs appearance)?

For synth val (has GT angles + GT 2D), feed the SAME angle head with:
  detected 2D  vs  oracle GT 2D (injected into geo + sampled features)
and report per-joint angle MAE for both, plus per-keypoint 2D error.
  GT-2D fixes wrist -> detector wrist keypoints are the problem.
  GT-2D still bad wrist -> head/appearance is the problem (2D geometry insufficient).
"""
import argparse, os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

TRAIN = os.path.abspath(os.path.join(os.path.dirname(__file__), '../TRAIN'))
sys.path.append(TRAIN); sys.path.append(os.path.dirname(__file__))
from model_angle import (AnglePredictor, keypoints_to_geo, sample_kp_features,
                         to_bearings, soft_argmax_2d)
from dataset import PoseEstimationDataset
from refine_eval import scale_K, wrapped_abs_deg

KPN = ['link0', 'link2', 'link3', 'link4', 'link6', 'link7', 'hand']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--detector', required=True); ap.add_argument('--mlp-head', required=True)
    ap.add_argument('--val-dir', default='../Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_test_dr')
    ap.add_argument('--model-name', default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    ap.add_argument('--image-size', type=int, default=512); ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--max-frames', type=int, default=1500)
    args = ap.parse_args()

    device = torch.device('cuda'); assert torch.cuda.is_available(); S = args.image_size
    m = AnglePredictor(args.model_name, S, head_type='mlp').to(device).eval()
    sd = torch.load(args.detector, map_location=device); sd = {k.replace('module.', ''): v for k, v in sd.items()}
    m.load_state_dict({k: v for k, v in sd.items() if k in m.state_dict() and v.shape == m.state_dict()[k].shape}, strict=False)
    m.angle_head.load_state_dict(torch.load(args.mlp_head, map_location=device))
    H = m.heatmap_size

    ds = PoseEstimationDataset(args.val_dir, keypoint_names=KPN, image_size=(S, S), heatmap_size=(S, S),
                               augment=False, include_angles=True, sigma=2.5)
    if args.max_frames and args.max_frames < len(ds): ds.samples = ds.samples[:args.max_frames]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    det_err = torch.zeros(6); gt_err = torch.zeros(6); n = 0
    kp_px = torch.zeros(7); kp_n = 0
    for batch in tqdm(loader, desc="wrist diag"):
        img = batch['image'].to(device)
        gta = batch['angles'][:, :6].to(device)
        gtkp = batch['keypoints'].to(device)             # (B,7,2) @ S
        K = scale_K(batch['camera_K'], batch['original_size'], S).to(device)
        with torch.no_grad():
            tokens = m.backbone(img)
            heatmaps = m.keypoint_head(tokens)
            kp2d = soft_argmax_2d(heatmaps)
            conf = heatmaps.flatten(2).max(dim=2)[0]
            gfeat = tokens.mean(dim=1)
            # detected-2D angles
            geo_d = keypoints_to_geo(kp2d, K); kpf_d = sample_kp_features(tokens, kp2d, H)
            ang_d, _ = m.angle_head(geo_d, conf, gfeat, kpf_d)
            # oracle GT-2D angles (inject GT keypoints into geo + sampled features)
            geo_g = keypoints_to_geo(gtkp, K); kpf_g = sample_kp_features(tokens, gtkp, H)
            ang_g, _ = m.angle_head(geo_g, conf, gfeat, kpf_g)
        det_err += wrapped_abs_deg(ang_d, gta).sum(0).cpu()
        gt_err += wrapped_abs_deg(ang_g, gta).sum(0).cpu()
        # per-keypoint 2D error (in-frame GT only)
        inb = (gtkp[..., 0] >= 0) & (gtkp[..., 0] < S) & (gtkp[..., 1] >= 0) & (gtkp[..., 1] < S)
        e = (kp2d - gtkp).norm(dim=-1)
        kp_px += (e * inb).sum(0).cpu();
        n += img.shape[0]; kp_n += img.shape[0]

    det = (det_err / n).numpy(); gt = (gt_err / n).numpy(); kp = (kp_px / kp_n).numpy()
    print(f"\n{'='*60}\nWRIST DIAG  {os.path.basename(args.val_dir)}  N={n}\n{'='*60}")
    print(f"  per-joint angle MAE (deg):   [our index J0..J5 = panda joint1..6]")
    print(f"  {'joint':<7}{'detected-2D':>12}{'oracle-2D':>11}{'delta':>8}")
    for j in range(6):
        print(f"  J{j:<6}{det[j]:>12.2f}{gt[j]:>11.2f}{gt[j]-det[j]:>+8.2f}")
    print(f"  {'MEAN':<7}{det.mean():>12.2f}{gt.mean():>11.2f}{gt.mean()-det.mean():>+8.2f}")
    print(f"\n  per-keypoint 2D error (px @ {S}):  [wrist = link6,link7,hand]")
    for i, nm in enumerate(KPN):
        tag = '  <-- wrist' if nm in ('link6', 'link7', 'hand') else ''
        print(f"    {nm:<7}{kp[i]:>7.2f}{tag}")
    print('='*60)


if __name__ == '__main__':
    main()
