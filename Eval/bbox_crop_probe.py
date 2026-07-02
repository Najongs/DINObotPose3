"""
Ceiling probe for SCALE/APPEARANCE normalization: does a robot-centered SQUARE crop (+ correct K
adjustment) beat the standard full-image square-resize? Different cameras frame the robot at
different apparent scale + we distort 4:3->1:1 on resize. A robot-centered square crop normalizes
apparent scale (RoboPEPP-style) and removes aspect distortion. Uses an ORACLE bbox from GT-2D
keypoints to measure the ceiling (a real robot-detector bbox would approximate it).

Crop math: square box around GT-2D kp (margin m), crop original image, resize to S.
  K' : cx'=(cx-x0)*S/side, cy'=(cy-y0)*S/side, fx'=fx*S/side, fy'=fy*S/side  (side=square px).
gt_3d is metric camera-frame -> unchanged by crop/resize.
"""
import argparse, json, os, sys
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

TRAIN = os.path.abspath(os.path.join(os.path.dirname(__file__), '../TRAIN'))
sys.path.append(TRAIN); sys.path.append(os.path.dirname(__file__))
from model_angle import AnglePredictor
from inference_4tier_eval import EvalDataset, compute_add_auc
from solve_pose_kinematic import solve_batch

KPN = ['panda_link0', 'panda_link2', 'panda_link3', 'panda_link4', 'panda_link6', 'panda_link7', 'panda_hand']
NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
TT = transforms.ToTensor()


def preprocess(img, kp2d_found, S, margin, mode):
    """Return (tensor CHW, K-adjust fn inputs). mode='full' square-resize, 'crop' robot-centered."""
    W, H = img.width, img.height
    if mode == 'full':
        im = img.resize((S, S))
        # standard: independent x/y scale (matches EvalDataset/scale_K)
        return TT(im), (S / W, S / H, 0.0, 0.0)  # sx,sy,ox,oy
    # crop: square box around found GT-2D keypoints
    pts = kp2d_found
    x0, y0 = pts[:, 0].min(), pts[:, 1].min(); x1, y1 = pts[:, 0].max(), pts[:, 1].max()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0) * margin
    side = max(side, 16.0)
    bx0, by0 = cx - side / 2, cy - side / 2
    # clamp into image (shift, keep square)
    bx0 = min(max(bx0, 0), max(0, W - side)); by0 = min(max(by0, 0), max(0, H - side))
    side = min(side, W, H)
    im = img.crop((bx0, by0, bx0 + side, by0 + side)).resize((S, S))
    return TT(im), (S / side, S / side, bx0, by0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--detector', required=True); ap.add_argument('--mlp-head', required=True)
    ap.add_argument('--val-dir', required=True)
    ap.add_argument('--model-name', default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    ap.add_argument('--image-size', type=int, default=512); ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--max-frames', type=int, default=600); ap.add_argument('--iters', type=int, default=200)
    ap.add_argument('--margin', type=float, default=1.6)
    args = ap.parse_args()
    device = torch.device('cuda'); S = args.image_size
    m = AnglePredictor(args.model_name, S, head_type='mlp').to(device).eval()
    sd = torch.load(args.detector, map_location=device); sd = {k.replace('module.', ''): v for k, v in sd.items()}
    m.load_state_dict({k: v for k, v in sd.items() if k in m.state_dict() and v.shape == m.state_dict()[k].shape}, strict=False)
    m.angle_head.load_state_dict(torch.load(args.mlp_head, map_location=device))

    ds = EvalDataset(args.val_dir, KPN, image_size=(S, S), verbose=False)
    files = ds.json_files
    if args.max_frames and args.max_frames < len(files):
        stride = max(1, len(files) // args.max_frames); files = files[::stride][:args.max_frames]

    res = {'full': [], 'crop': []}
    for mode in ['full', 'crop']:
        imgs, Ks, gt3ds, founds = [], [], [], []
        meta = []
        def flush():
            if not imgs:
                return
            x = torch.stack(imgs).to(device); K = torch.stack(Ks).to(device)
            with torch.no_grad():
                out = m(x, K)
            theta, kp_cam, _ = solve_batch(out['keypoints_2d'], out['confidence'], K, fix_joint7=True,
                                           iters=args.iters, lr=2e-2, img_size=S, device=device,
                                           prior_w=0.0, theta_init=out['joint_angles'])
            kc = kp_cam.cpu().numpy()
            for i in range(len(imgs)):
                f = founds[i]
                res[mode].append(float(np.linalg.norm(kc[i] - gt3ds[i], axis=1)[f > 0].mean()))
            imgs.clear(); Ks.clear(); gt3ds.clear(); founds.clear()
        for jf in tqdm(files, desc=mode):
            d = json.load(open(jf))
            ip = d['meta']['image_path']
            if ip.startswith('../dataset/'): ip = ip.replace('../dataset/', '../../../', 1)
            p = (jf.parent / ip).resolve()
            if not p.exists(): p = (jf.parent / ip)
            img = Image.open(p).convert('RGB')
            kd = {k['name']: k for o in d.get('objects', []) for k in o.get('keypoints', [])}
            kp2d = np.zeros((7, 2)); kp3d = np.zeros((7, 3)); found = np.zeros(7)
            for i, nm in enumerate(KPN):
                if nm in kd:
                    kp2d[i] = kd[nm]['projected_location']; found[i] = 1
                    if 'location' in kd[nm]: kp3d[i] = kd[nm]['location']
            ga = np.zeros(7)
            for i, j in enumerate(d.get('sim_state', {}).get('joints', [])[:7]):
                ga[i] = j.get('position', 0.0)
            if found.sum() < 4 or not np.any(ga != 0):
                continue
            K0 = np.array(d['meta']['K'], dtype=np.float64)
            t, (sx, sy, ox, oy) = preprocess(img, kp2d[found > 0], S, args.margin, mode)
            Kn = K0.copy()
            Kn[0, 2] -= ox; Kn[1, 2] -= oy
            Kn[0, 0] *= sx; Kn[0, 2] *= sx; Kn[1, 1] *= sy; Kn[1, 2] *= sy
            imgs.append(NORM(t)); Ks.append(torch.from_numpy(Kn).float())
            gt3ds.append(kp3d.astype(np.float32)); founds.append(found)
            if len(imgs) >= args.batch_size:
                flush()
        flush()
    print(f"\n  {os.path.basename(args.val_dir)}  bbox-crop normalization probe (margin {args.margin})")
    print(f"  {'config':<8}{'ADD-AUC':>10}{'meanADD':>10}{'medADD':>9}")
    for k in ['full', 'crop']:
        a = np.array(res[k])
        print(f"  {k:<8}{compute_add_auc(a):>10.4f}{a.mean()*1000:>10.1f}{np.median(a)*1000:>9.1f}")


if __name__ == '__main__':
    main()
