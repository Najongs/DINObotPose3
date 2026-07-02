"""
Qualitative performance check: overlay the full pipeline's pose estimate on real images.

Per frame draws:
  GREEN  = GT 2D keypoints (+ kinematic skeleton)
  RED    = reprojected FK from the REFINED (theta, R, t) estimate  -> the model's actual pose
  YELLOW = raw detected 2D keypoints (detector argmax)             -> detector quality
Title: per-frame angle MAE (deg) and ADD (mm). conf-gate (occlusion handling) is ON.
"""
import argparse, glob, math, os, sys
import numpy as np
import torch
from PIL import Image, ImageDraw

TRAIN = os.path.abspath(os.path.join(os.path.dirname(__file__), '../TRAIN'))
sys.path.append(TRAIN); sys.path.append(os.path.dirname(__file__))
from model_angle import AnglePredictor
from dataset import PoseEstimationDataset
from solve_pose_kinematic import solve_batch
from refine_eval import scale_K, wrapped_abs_deg

CHAIN = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]  # link0-link2-link3-link4-link6-link7-hand
NAMES = ['link0', 'link2', 'link3', 'link4', 'link6', 'link7', 'hand']


def project(kp_cam, K):
    """kp_cam (N,3) m camera frame, K (3,3) -> (N,2) px."""
    z = kp_cam[:, 2:3].clamp(min=1e-4)
    uv = (K @ (kp_cam / z).T).T[:, :2]
    return uv


def draw(img_pil, gt2d, pred2d, det2d, S, scale_to):
    im = img_pil.convert('RGB').resize(scale_to)
    sx, sy = scale_to[0] / S, scale_to[1] / S
    dr = ImageDraw.Draw(im)
    def P(p): return (float(p[0]) * sx, float(p[1]) * sy)
    # skeleton (GT green, pred red)
    for a, b in CHAIN:
        dr.line([P(gt2d[a]), P(gt2d[b])], fill=(0, 220, 0), width=2)
        dr.line([P(pred2d[a]), P(pred2d[b])], fill=(240, 40, 40), width=2)
    for i in range(7):
        gx, gy = P(gt2d[i]); px, py = P(pred2d[i]); yx, yy = P(det2d[i])
        dr.ellipse([yx-4, yy-4, yx+4, yy+4], outline=(255, 210, 0), width=2)   # detected
        dr.ellipse([gx-4, gy-4, gx+4, gy+4], fill=(0, 220, 0))                 # GT
        dr.ellipse([px-3, py-3, px+3, py+3], fill=(240, 40, 40))               # pred
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--detector', required=True)
    ap.add_argument('--mlp-head', required=True)
    ap.add_argument('--val-dir', required=True)
    ap.add_argument('--model-name', default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    ap.add_argument('--image-size', type=int, default=512)
    ap.add_argument('--indices', default='0,400,1200,2798,2855,3500')
    ap.add_argument('--conf-gate', type=float, default=0.05)
    ap.add_argument('--cell', type=int, default=384, help='cell width px (height=3/4)')
    ap.add_argument('--cols', type=int, default=3)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'viz_outputs/viz_results.png'))
    args = ap.parse_args()
    CW, CH = args.cell, args.cell * 3 // 4

    device = torch.device('cuda'); assert torch.cuda.is_available()
    S = args.image_size
    m = AnglePredictor(args.model_name, S, head_type='mlp').to(device).eval()
    sd = torch.load(args.detector, map_location=device); sd = {k.replace('module.', ''): v for k, v in sd.items()}
    m.load_state_dict({k: v for k, v in sd.items() if k in m.state_dict()
                       and v.shape == m.state_dict()[k].shape}, strict=False)
    m.angle_head.load_state_dict(torch.load(args.mlp_head, map_location=device))

    ds = PoseEstimationDataset(args.val_dir, keypoint_names=NAMES,
                               image_size=(S, S), heatmap_size=(S, S),
                               augment=False, include_angles=True, sigma=2.5)
    idxs = [int(x) for x in args.indices.split(',') if int(x) < len(ds)]
    cells, titles = [], []
    for ix in idxs:
        b = ds[ix]
        img = b['image'].unsqueeze(0).to(device)
        gt = b['angles'][:6].unsqueeze(0).to(device)
        gt3d = b['keypoints_3d'].unsqueeze(0).to(device)
        gt2d = b['keypoints'].numpy()
        K = scale_K(b['camera_K'].unsqueeze(0), b['original_size'].unsqueeze(0), S).to(device)
        with torch.no_grad():
            out = m(img, K)
        init = out['joint_angles']; kp2d = out['keypoints_2d']; conf = out['confidence']
        refined, kp_cam, _ = solve_batch(kp2d, conf, K, fix_joint7=True, iters=200, lr=2e-2,
                                         img_size=S, device=device, prior_w=0.0,
                                         theta_init=init, conf_gate=args.conf_gate)
        pred2d = project(kp_cam[0], K[0]).cpu().numpy()
        det2d = kp2d[0].cpu().numpy()
        ang_mae = float(wrapped_abs_deg(refined[:, :6], gt).mean())
        valid = (gt3d[0].abs().sum(-1) > 0)
        add = float((kp_cam[0][valid] - gt3d[0][valid]).norm(dim=-1).mean() * 1000)
        noff = int(((gt2d[:, 0] < 0) | (gt2d[:, 0] >= S) | (gt2d[:, 1] < 0) | (gt2d[:, 1] >= S)).sum())
        # load original image for nicer overlay
        ann = b['annotation_path'] if 'annotation_path' in b else None
        import json
        meta = json.load(open(ann))['meta'] if ann else None
        ipath = os.path.join(os.path.dirname(ann), meta['image_path']) if meta else None
        src = Image.open(ipath) if ipath and os.path.exists(ipath) else Image.new('RGB', (S, S))
        cells.append(draw(src, gt2d, pred2d, det2d, S, (CW, CH)))
        titles.append(f"#{ix}  ang {ang_mae:.1f}deg  ADD {add:.0f}mm  off-frame:{noff}")
        print(titles[-1])

    cols = min(args.cols, len(cells)); rows = (len(cells) + cols - 1) // cols
    cw, ch = CW, CH + 20
    grid = Image.new('RGB', (cw * cols, ch * rows), (15, 15, 15))
    dr = ImageDraw.Draw(grid)
    for k, (im, t) in enumerate(zip(cells, titles)):
        x, y = (k % cols) * cw, (k // cols) * ch
        grid.paste(im, (x, y + 20))
        dr.text((x + 4, y + 5), t, fill=(255, 255, 255))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    grid.save(args.out)
    print("saved", args.out, "| legend: GREEN=GT  RED=pred(refined FK reproj)  YELLOW=detected")


if __name__ == '__main__':
    main()
