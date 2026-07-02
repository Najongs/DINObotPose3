"""
Quantitatively test the user's 3-part hypothesis about WHERE the pipeline's error lives, on the
FINAL deployable pipeline (detector -> mlp angle head -> rot R_init -> kinematic solve):

  H1  Joint-ANGLE accuracy is LOW            -> per-joint angle MAE (deg)
  H2  2D KEYPOINT detection is GOOD          -> per-keypoint detector error (px) + PCK
  H3  OCCLUSION breaks the FK solve (outliers)-> ADD vs (#off-frame/occluded kp), correlation

Outputs to ViS/hypothesis/:
  panel_2d_vs_angle.png       bar: per-kp 2D px-err (good) | per-joint angle MAE (bad)
  panel_outliers.png          scatter: ADD vs 2D-err & ADD vs #occluded, w/ correlations
  montage_examples.png        3 columns: (good2D+good pose) | (good2D+BAD angle) | (occluded->outlier)
  summary printed to stdout

Legend on overlays: GREEN=GT 2D(+skeleton)  YELLOW=detected 2D  RED=solved-FK reproj.
Run on real cameras (where occlusion/foreshortening actually bite).
"""
import argparse, glob, json, math, os, sys
import numpy as np
import torch
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TRAIN = os.path.abspath(os.path.join(os.path.dirname(__file__), '../TRAIN'))
sys.path.append(TRAIN); sys.path.append(os.path.dirname(__file__))
from model_angle import AnglePredictor
from model_v4 import panda_forward_kinematics
from solve_pose_kinematic import solve_batch

KPN = ['panda_link0', 'panda_link2', 'panda_link3', 'panda_link4', 'panda_link6', 'panda_link7', 'panda_hand']
SHORT = ['link0', 'link2', 'link3', 'link4', 'link6', 'link7', 'hand']
JNAME = ['J0', 'J1', 'J2', 'J3', 'J4', 'J5']
CHAIN = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../ViS/hypothesis'))


def project(kp_cam, K):
    z = kp_cam[:, 2:3].clamp(min=1e-4)
    return (K @ (kp_cam / z).T).T[:, :2]


def load_frame(jf, val_dir, S):
    from torchvision import transforms
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]); tt = transforms.ToTensor()
    d = json.load(open(jf))
    kd = {k['name']: k for o in d.get('objects', []) for k in o.get('keypoints', [])}
    kp2d = np.zeros((7, 2)); kp3d = np.zeros((7, 3)); found = np.zeros(7)
    for i, nm in enumerate(KPN):
        if nm in kd:
            kp2d[i] = kd[nm]['projected_location']; found[i] = 1
            if 'location' in kd[nm]: kp3d[i] = kd[nm]['location']
    ga = np.zeros(7)
    for i, j in enumerate(d.get('sim_state', {}).get('joints', [])[:7]): ga[i] = j.get('position', 0.0)
    if found.sum() < 4 or not np.any(ga != 0): return None
    ip = d['meta']['image_path']
    if ip.startswith('../dataset/'): ip = ip.replace('../dataset/', '../../../', 1)
    p = (os.path.dirname(jf) + '/' + ip)
    if not os.path.exists(p): p = os.path.join(val_dir, os.path.basename(ip))
    src = Image.open(p).convert('RGB'); W, H = src.width, src.height
    K0 = np.array(d['meta']['K'], dtype=np.float32)
    Kn = K0.copy(); Kn[0, 0] *= S/W; Kn[0, 2] *= S/W; Kn[1, 1] *= S/H; Kn[1, 2] *= S/H
    P = kp3d[found > 0]; Pc = P - P.mean(0)
    _, _, Vt = np.linalg.svd(Pc); axis = Vt[0]
    fore = math.degrees(math.acos(min(1.0, abs(axis[2]))))
    # in-frame mask: GT projection lies inside the image (for "occluded/off-frame" accounting)
    inframe = ((kp2d[:, 0] >= 0) & (kp2d[:, 0] < W) & (kp2d[:, 1] >= 0) & (kp2d[:, 1] < H)).astype(float)
    return dict(jf=jf, img=norm(tt(src.resize((S, S)))), K=torch.from_numpy(Kn).float(),
                kp3d=kp3d.astype(np.float32), ang=ga, found=found, inframe=inframe,
                gt2d=kp2d, W=W, H=H, src=src, fore=fore)


def run(args, val_dir, device, S, m, recs):
    files = sorted(glob.glob(os.path.join(val_dir, '*.json')))
    if args.max_frames and args.max_frames < len(files):
        stride = max(1, len(files) // args.max_frames); files = files[::stride][:args.max_frames]
    cam = os.path.basename(os.path.normpath(val_dir)).replace('panda-3cam_', '').replace('panda-', '')
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
            dd = np.arctan2(np.sin(theta[i, :6] - ga[:6]), np.cos(theta[i, :6] - ga[:6]))
            ang = np.degrees(np.abs(dd))                                  # per-joint angle err (6,)
            sx, sy = S / b['W'], S / b['H']
            gt2d_s = b['gt2d'] * np.array([sx, sy])
            kperr = np.full(7, np.nan)                                     # per-kp detector px err @S
            for k in range(7):
                if f[k]: kperr[k] = np.linalg.norm(kp2d[i, k] - gt2d_s[k])
            n_off = int((b['inframe'] == 0).sum())                        # off-frame kp (occluded/out)
            recs.append(dict(cam=cam, jf=b['jf'], add=add, ang_per=ang, ang=float(ang.mean()),
                             j0=float(ang[0]), kperr=kperr, n_off=n_off, fore=b['fore'],
                             conf_min=float(conf[i][f > 0].min()), kp2d=kp2d[i], kc=kc[i],
                             gt2d=b['gt2d'], found=f, inframe=b['inframe'], K=b['K'].numpy(),
                             W=b['W'], H=b['H'], src=b['src']))
        buf.clear()
    for jf in files:
        fr = load_frame(jf, val_dir, S)
        if fr is None: continue
        buf.append(fr)
        if len(buf) >= args.batch_size: flush()
    flush()
    print(f"  loaded {cam}: total recs now {len(recs)}", flush=True)


def overlay(r, S, out_wh=(384, 288)):
    im = r['src'].convert('RGB').resize(out_wh)
    W, H = r['W'], r['H']; sx, sy = out_wh[0] / W, out_wh[1] / H
    dr = ImageDraw.Draw(im)
    def P(p): return (float(p[0]) * sx, float(p[1]) * sy)
    pred2d = project(torch.from_numpy(r['kc']).float(), torch.from_numpy(r['K']).float()).numpy()
    pred2d = pred2d * np.array([W / S, H / S])
    det2d = r['kp2d'] * np.array([W / S, H / S])
    f = r['found']
    for a, b in CHAIN:
        if f[a] and f[b]: dr.line([P(r['gt2d'][a]), P(r['gt2d'][b])], fill=(0, 220, 0), width=2)
        dr.line([P(pred2d[a]), P(pred2d[b])], fill=(240, 40, 40), width=2)
    for i in range(7):
        if f[i]:
            gx, gy = P(r['gt2d'][i]); dr.ellipse([gx-4, gy-4, gx+4, gy+4], fill=(0, 220, 0))
        yx, yy = P(det2d[i]); dr.ellipse([yx-4, yy-4, yx+4, yy+4], outline=(255, 210, 0), width=2)
        px, py = P(pred2d[i]); dr.ellipse([px-3, py-3, px+3, py+3], fill=(240, 40, 40))
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--detector', required=True); ap.add_argument('--mlp-head', required=True)
    ap.add_argument('--rot-head', default=None)
    ap.add_argument('--val-dirs', nargs='+', required=True)
    ap.add_argument('--model-name', default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    ap.add_argument('--image-size', type=int, default=512); ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--max-frames', type=int, default=400); ap.add_argument('--iters', type=int, default=200)
    args = ap.parse_args()
    device = torch.device('cuda'); S = args.image_size
    os.makedirs(OUT, exist_ok=True)

    m = AnglePredictor(args.model_name, S, head_type='mlp', with_rotation=args.rot_head is not None,
                       with_translation=args.rot_head is not None).to(device).eval()
    sd = torch.load(args.detector, map_location=device); sd = {k.replace('module.', ''): v for k, v in sd.items()}
    m.load_state_dict({k: v for k, v in sd.items() if k in m.state_dict() and v.shape == m.state_dict()[k].shape}, strict=False)
    m.angle_head.load_state_dict(torch.load(args.mlp_head, map_location=device))
    if args.rot_head: m.rot_head.load_state_dict(torch.load(args.rot_head, map_location=device))

    recs = []
    for vd in args.val_dirs: run(args, vd, device, S, m, recs)
    n = len(recs); assert n > 0
    add = np.array([r['add'] for r in recs])
    angP = np.array([r['ang_per'] for r in recs])                          # (n,6)
    kperr = np.array([r['kperr'] for r in recs])                           # (n,7) nan where unfound
    noff = np.array([r['n_off'] for r in recs])
    fore = np.array([r['fore'] for r in recs])
    from scipy.stats import spearmanr

    # ---- per-kp 2D error & PCK (H2) ----
    kp_med = np.nanmedian(kperr, axis=0)                                    # (7,) median px err @512
    pck5 = np.nanmean((kperr < 5).astype(float), axis=0)                    # frac < 5px
    pck10 = np.nanmean((kperr < 10).astype(float), axis=0)
    # ---- per-joint angle MAE (H1) ----
    ang_mae = angP.mean(0)                                                  # (6,)
    # ---- occlusion -> outlier (H3) ----
    mean2d = np.nanmean(kperr, axis=1)
    fail = add > 100
    rho_off, p_off = spearmanr(noff, add)
    rho_2d, p_2d = spearmanr(mean2d, add)
    rho_ang, p_ang = spearmanr(angP.mean(1), add)
    rho_fore, _ = spearmanr(fore, add)
    occ = noff > 0
    print("\n================ HYPOTHESIS TEST (n=%d frames, real cameras) ================" % n)
    print("ADD-AUC fail(>100mm) = %.0f%%   median ADD %.0f mm" % (100*fail.mean(), np.median(add)))
    print("\n[H2] 2D keypoint detection — GOOD?")
    for k in range(7):
        print("   %-6s  med %5.1f px   PCK@5 %.2f  PCK@10 %.2f" % (SHORT[k], kp_med[k], pck5[k], pck10[k]))
    print("   OVERALL median 2D err %.1f px @512 (%.1f px @224-equiv)" % (np.nanmedian(kperr), np.nanmedian(kperr)*224/512))
    print("\n[H1] joint ANGLE — LOW accuracy?")
    for k in range(6):
        print("   %-3s  MAE %5.1f deg" % (JNAME[k], ang_mae[k]))
    print("   OVERALL angle MAE %.1f deg" % ang_mae.mean())
    print("\n[H3] OCCLUSION drives OUTLIERS?")
    print("   Spearman ADD vs #off-frame kp : rho %+.2f (p=%.1e)" % (rho_off, p_off))
    print("   Spearman ADD vs 2D px-err     : rho %+.2f (p=%.1e)" % (rho_2d, p_2d))
    print("   Spearman ADD vs angle MAE     : rho %+.2f (p=%.1e)" % (rho_ang, p_ang))
    print("   Spearman ADD vs foreshorten   : rho %+.2f" % rho_fore)
    if occ.sum() and (~occ).sum():
        print("   median ADD: fully-visible %.0f mm  |  has-occluded-kp %.0f mm" % (
            np.median(add[~occ]), np.median(add[occ])))
        print("   fail-rate : fully-visible %.0f%%   |  has-occluded-kp %.0f%%" % (
            100*fail[~occ].mean(), 100*fail[occ].mean()))

    # ============ PANEL A: 2D good vs angle bad ============
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    x = np.arange(7)
    ax[0].bar(x, kp_med, color='#2c8', alpha=.85)
    ax[0].axhline(5, ls='--', c='gray', lw=1); ax[0].text(6.2, 5.3, 'PCK@5px', fontsize=8, c='gray')
    ax[0].set_xticks(x); ax[0].set_xticklabels(SHORT, rotation=30, fontsize=8)
    ax[0].set_ylabel('median 2D error (px @512)'); ax[0].set_title('H2: 2D keypoint detection — ACCURATE (low px err)')
    for i in range(7): ax[0].text(i, kp_med[i]+0.3, f'{kp_med[i]:.1f}', ha='center', fontsize=7)
    xa = np.arange(6)
    ax[1].bar(xa, ang_mae, color='#d55', alpha=.85)
    ax[1].set_xticks(xa); ax[1].set_xticklabels(JNAME, fontsize=9)
    ax[1].set_ylabel('angle MAE (deg)'); ax[1].set_title('H1: joint ANGLE — INACCURATE (high MAE)')
    for i in range(6): ax[1].text(i, ang_mae[i]+0.3, f'{ang_mae[i]:.0f}', ha='center', fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, 'panel_2d_vs_angle.png'), dpi=120); plt.close()

    # ============ PANEL B: occlusion -> outlier ============
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    sc = ax[0].scatter(mean2d, add, c=noff, cmap='inferno', s=14, vmin=0, vmax=max(1, noff.max()))
    ax[0].axhline(100, ls='--', c='r', lw=1); ax[0].set_xlabel('mean 2D detector err (px @512)')
    ax[0].set_ylabel('ADD (mm)'); ax[0].set_title('ADD vs 2D-err  (color=#occluded kp)\nspearman ADD~2D %+.2f' % rho_2d)
    plt.colorbar(sc, ax=ax[0], label='#off-frame/occluded kp')
    # box: ADD distribution by occlusion count
    groups = sorted(set(noff))
    data = [add[noff == g] for g in groups]
    ax[1].boxplot(data, tick_labels=[str(g) for g in groups], showfliers=True, sym='.')
    ax[1].axhline(100, ls='--', c='r', lw=1)
    ax[1].set_xlabel('# off-frame / occluded keypoints'); ax[1].set_ylabel('ADD (mm)')
    ax[1].set_title('H3: occlusion drives the ADD outliers\nspearman ADD~#occ %+.2f' % rho_off)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, 'panel_outliers.png'), dpi=120); plt.close()

    # ============ MONTAGE: 3 categories ============
    good2d = mean2d < np.nanpercentile(mean2d, 50)
    cat_good = [i for i in range(n) if good2d[i] and noff[i] == 0 and add[i] < 60]
    cat_badang = [i for i in range(n) if good2d[i] and noff[i] == 0 and add[i] > 120]  # good 2D, no occ, BAD pose
    cat_occ = [i for i in range(n) if noff[i] >= 1 and add[i] > 120]                    # occluded -> outlier
    cat_good.sort(key=lambda i: add[i]); cat_badang.sort(key=lambda i: -add[i]); cat_occ.sort(key=lambda i: -add[i])
    cols = [('GOOD 2D + good pose', cat_good[:3]),
            ('GOOD 2D + BAD angle (no occlusion)', cat_badang[:3]),
            ('OCCLUDED kp -> FK outlier', cat_occ[:3])]
    CW, CH = 384, 288; pad = 22
    grid = Image.new('RGB', (CW * 3, (CH + pad) * 3 + pad), (18, 18, 18)); dr = ImageDraw.Draw(grid)
    for c, (title, idxs) in enumerate(cols):
        dr.text((c * CW + 6, 4), title, fill=(255, 255, 120))
        for rrow, i in enumerate(idxs):
            r = recs[i]; im = overlay(r, S, (CW, CH))
            y = pad + rrow * (CH + pad)
            grid.paste(im, (c * CW, y))
            t = f"ADD{add[i]:.0f} ang{r['ang']:.0f} J0:{r['j0']:.0f} 2d{mean2d[i]:.1f}px off{noff[i]} fore{fore[i]:.0f}"
            dr.text((c * CW + 3, y - 13), t, fill=(220, 220, 220))
    grid.save(os.path.join(OUT, 'montage_examples.png'))
    print("\nlegend: GREEN=GT 2D(+skeleton)  YELLOW=detected 2D  RED=solved-FK reproj")
    print("saved -> %s/{panel_2d_vs_angle,panel_outliers,montage_examples}.png" % OUT)


if __name__ == '__main__':
    main()
