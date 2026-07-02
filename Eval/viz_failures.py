"""
Find and SAVE the worst-case frames so we can SEE what the model fails on.

Runs the current best pipeline (detector -> mlp angle head -> optional rotation R_init -> kinematic
solve, pnp_drop=3) on strided frames, scores each by ADD (mm) and J0 (base-yaw) angle error,
then saves overlay images of the worst frames + a montage, annotated with diagnostics:
  GREEN = GT 2D keypoints (+skeleton), RED = predicted refined-FK reprojection, YELLOW = detected 2D.
Title per frame: ADD, angle MAE, J0 err, off-frame kp, foreshortening, min/mean detector conf,
detector 2D err on base keypoints (link0/link2 — the ones J0 depends on).
Output: Eval/viz_outputs/failures_<split>/{worst_add,worst_j0}/*.png + montage.png
"""
import argparse, json, math, os, sys
import numpy as np
import torch
from PIL import Image, ImageDraw

TRAIN = os.path.abspath(os.path.join(os.path.dirname(__file__), '../TRAIN'))
sys.path.append(TRAIN); sys.path.append(os.path.dirname(__file__))
from model_angle import AnglePredictor
from model_v4 import panda_forward_kinematics
from solve_pose_kinematic import solve_batch
from refine_eval import scale_K

KPN = ['panda_link0', 'panda_link2', 'panda_link3', 'panda_link4', 'panda_link6', 'panda_link7', 'panda_hand']
SHORT = ['link0', 'link2', 'link3', 'link4', 'link6', 'link7', 'hand']
CHAIN = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]


def project(kp_cam, K):
    z = kp_cam[:, 2:3].clamp(min=1e-4)
    return (K @ (kp_cam / z).T).T[:, :2]


def draw(src, gt2d, pred2d, det2d, found, W, H, out_wh):
    im = src.convert('RGB').resize(out_wh)
    sx, sy = out_wh[0] / W, out_wh[1] / H
    dr = ImageDraw.Draw(im)
    def P(p): return (float(p[0]) * sx, float(p[1]) * sy)
    for a, b in CHAIN:
        if found[a] and found[b]:
            dr.line([P(gt2d[a]), P(gt2d[b])], fill=(0, 220, 0), width=2)
        dr.line([P(pred2d[a]), P(pred2d[b])], fill=(240, 40, 40), width=2)
    for i in range(7):
        if found[i]:
            gx, gy = P(gt2d[i]); dr.ellipse([gx-4, gy-4, gx+4, gy+4], fill=(0, 220, 0))
        yx, yy = P(det2d[i]); dr.ellipse([yx-4, yy-4, yx+4, yy+4], outline=(255, 210, 0), width=2)
        px, py = P(pred2d[i]); dr.ellipse([px-3, py-3, px+3, py+3], fill=(240, 40, 40))
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--detector', required=True); ap.add_argument('--mlp-head', required=True)
    ap.add_argument('--rot-head', default=None)
    ap.add_argument('--val-dir', required=True)
    ap.add_argument('--model-name', default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    ap.add_argument('--image-size', type=int, default=512); ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--max-frames', type=int, default=600); ap.add_argument('--iters', type=int, default=200)
    ap.add_argument('--topk', type=int, default=12)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    device = torch.device('cuda'); S = args.image_size
    split = os.path.basename(os.path.normpath(args.val_dir))
    out = args.out or os.path.join(os.path.dirname(__file__), f'viz_outputs/failures_{split}')
    os.makedirs(out, exist_ok=True)

    m = AnglePredictor(args.model_name, S, head_type='mlp', with_rotation=args.rot_head is not None,
                       with_translation=args.rot_head is not None).to(device).eval()
    sd = torch.load(args.detector, map_location=device); sd = {k.replace('module.', ''): v for k, v in sd.items()}
    m.load_state_dict({k: v for k, v in sd.items() if k in m.state_dict() and v.shape == m.state_dict()[k].shape}, strict=False)
    m.angle_head.load_state_dict(torch.load(args.mlp_head, map_location=device))
    if args.rot_head:
        m.rot_head.load_state_dict(torch.load(args.rot_head, map_location=device))

    files = sorted(__import__('glob').glob(os.path.join(args.val_dir, '*.json')))
    if args.max_frames and args.max_frames < len(files):
        stride = max(1, len(files) // args.max_frames); files = files[::stride][:args.max_frames]

    recs = []
    buf = []
    def flush():
        if not buf: return
        imgs = torch.stack([b['img'] for b in buf]).to(device)
        K = torch.stack([b['K'] for b in buf]).to(device)
        with torch.no_grad():
            o = m(imgs, K)
        R_init = o.get('rot_matrix') if args.rot_head else None
        theta, kp_cam, _ = solve_batch(o['keypoints_2d'], o['confidence'], K, fix_joint7=True,
                                       iters=args.iters, lr=2e-2, img_size=S, device=device,
                                       prior_w=0.0, theta_init=o['joint_angles'], R_init=R_init)
        kp2d = o['keypoints_2d'].cpu().numpy(); conf = o['confidence'].cpu().numpy()
        theta = theta.cpu().numpy(); kc = kp_cam.cpu().numpy()
        for i, b in enumerate(buf):
            f = b['found']; ga = b['ang'].copy(); ga[6] = 0.0
            add = float(np.linalg.norm(kc[i] - b['kp3d'], axis=1)[f > 0].mean() * 1000)
            d = np.arctan2(np.sin(theta[i, :6] - ga[:6]), np.cos(theta[i, :6] - ga[:6]))
            ang = np.degrees(np.abs(d)); j0 = float(ang[0]); amae = float(ang.mean())
            # detected-2D error on base keypoints (J0 source) vs GT 2D (at heatmap res S)
            sx, sy = S / b['W'], S / b['H']
            gt2d_s = b['gt2d'] * np.array([sx, sy])
            base_err = float(np.mean([np.linalg.norm(kp2d[i, k] - gt2d_s[k]) for k in (0, 1) if f[k]]))
            recs.append(dict(jf=b['jf'], add=add, ang=amae, j0=j0, base_err=base_err,
                             noff=int((f == 0).sum()), conf_min=float(conf[i][f > 0].min()),
                             fore=b['fore'], kp2d=kp2d[i], theta=theta[i], kc=kc[i],
                             gt2d=b['gt2d'], found=f, K=b['K'].numpy(), W=b['W'], H=b['H'], src=b['src']))
        buf.clear()

    from torchvision import transforms
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]); tt = transforms.ToTensor()
    for jf in files:
        d = json.load(open(jf)); kd = {k['name']: k for o in d.get('objects', []) for k in o.get('keypoints', [])}
        kp2d = np.zeros((7, 2)); kp3d = np.zeros((7, 3)); found = np.zeros(7)
        for i, nm in enumerate(KPN):
            if nm in kd:
                kp2d[i] = kd[nm]['projected_location']; found[i] = 1
                if 'location' in kd[nm]: kp3d[i] = kd[nm]['location']
        ga = np.zeros(7)
        for i, j in enumerate(d.get('sim_state', {}).get('joints', [])[:7]): ga[i] = j.get('position', 0.0)
        if found.sum() < 4 or not np.any(ga != 0): continue
        ip = d['meta']['image_path']
        if ip.startswith('../dataset/'): ip = ip.replace('../dataset/', '../../../', 1)
        p = (os.path.dirname(jf) + '/' + ip)
        if not os.path.exists(p): p = os.path.join(args.val_dir, os.path.basename(ip))
        src = Image.open(p).convert('RGB'); W, H = src.width, src.height
        K0 = np.array(d['meta']['K'], dtype=np.float32)
        Kn = K0.copy(); Kn[0, 0] *= S/W; Kn[0, 2] *= S/W; Kn[1, 1] *= S/H; Kn[1, 2] *= S/H
        P = kp3d[found > 0]; Pc = P - P.mean(0)
        _, _, Vt = np.linalg.svd(Pc); axis = Vt[0]
        fore = math.degrees(math.acos(min(1.0, abs(axis[2]))))
        buf.append(dict(jf=jf, img=norm(tt(src.resize((S, S)))), K=torch.from_numpy(Kn).float(),
                        kp3d=kp3d.astype(np.float32), ang=ga, found=found, gt2d=kp2d, W=W, H=H, src=src, fore=fore))
        if len(buf) >= args.batch_size: flush()
    flush()

    add_arr = np.array([r['add'] for r in recs])
    print(f"{split}: n={len(recs)}  ADD mean {add_arr.mean():.1f} med {np.median(add_arr):.1f}mm  fail(>100)={100*(add_arr>100).mean():.0f}%")
    for key, label in [('add', 'worst_add'), ('j0', 'worst_j0')]:
        order = sorted(range(len(recs)), key=lambda i: -recs[i][key])[:args.topk]
        sub = os.path.join(out, label); os.makedirs(sub, exist_ok=True)
        cells, titles = [], []
        CW, CH = 384, 384 * 3 // 4
        for rank, i in enumerate(order):
            r = recs[i]
            # predicted 2D = reproject the solved camera-frame keypoints with K (@ heatmap res S)
            pred2d = project(torch.from_numpy(r['kc']).float(), torch.from_numpy(r['K']).float()).numpy()
            pred2d_orig = pred2d * np.array([r['W'] / S, r['H'] / S])
            det2d = r['kp2d'] * np.array([r['W'] / S, r['H'] / S])  # detected 2D at orig res
            im = draw(r['src'], r['gt2d'], pred2d_orig, det2d, r['found'], r['W'], r['H'], (CW, CH))
            t = f"ADD{r['add']:.0f} ang{r['ang']:.0f} J0:{r['j0']:.0f} base2d:{r['base_err']:.0f}px fore{r['fore']:.0f} conf{r['conf_min']:.2f} off{r['noff']}"
            im.save(os.path.join(sub, f"{rank:02d}_{os.path.basename(r['jf']).replace('.json','')}.png"))
            cells.append(im); titles.append(t)
        cols = 3; rows = (len(cells) + cols - 1) // cols; ch = CH + 18
        grid = Image.new('RGB', (CW * cols, ch * rows), (15, 15, 15)); dr = ImageDraw.Draw(grid)
        for k, (im, t) in enumerate(zip(cells, titles)):
            x, y = (k % cols) * CW, (k // cols) * ch
            grid.paste(im, (x, y + 18)); dr.text((x + 3, y + 4), t, fill=(255, 255, 255))
        grid.save(os.path.join(out, f'montage_{label}.png'))
        print(f"  saved {len(cells)} {label} -> {out}/montage_{label}.png")
    print("legend: GREEN=GT  RED=pred(refined FK reproj)  YELLOW=detected")


if __name__ == '__main__':
    main()
