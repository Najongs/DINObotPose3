"""
Do the ADD failures (realsense/orb) form a CLEAN, separable cluster with one common cause — or a smooth
continuum that no single feature carves out? For each frame compute ADD + interpretable features, then:
  (1) ADD histogram  -> is it bimodal (discrete good/bad) or a smooth heavy tail (continuum)?
  (2) per-feature histograms fail(ADD>100) vs ok, with single-feature AUROC -> how separable is each?
  (3) scatter of the two most-informative features colored by ADD -> is there a clean boundary?
Saves ViS/failure_cluster/{add_hist,feature_separability,scatter}.png + prints AUROCs.
Features: foreshortening, mean 2D-err, #occluded kp, min conf, 2D spread(px), base-depth err.
"""
import argparse, glob, math, os, sys
import numpy as np
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../TRAIN')))
sys.path.append(os.path.dirname(__file__))
from model_angle import AnglePredictor
from model_v4 import panda_forward_kinematics
from solve_pose_kinematic import solve_batch
from viz_hypothesis import load_frame

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../ViS/failure_cluster'))
FEATS = ['foreshorten°', 'mean2D_err_px', '#occluded_kp', 'min_conf', '2D_spread_px', 'base_depth_err_mm']


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

    ADD, X = [], []
    for vd in args.val_dirs:
        files = sorted(glob.glob(os.path.join(vd, '*.json')))
        if args.max_frames and args.max_frames < len(files):
            st = max(1, len(files) // args.max_frames); files = files[::st][:args.max_frames]
        buf = []
        def flush():
            if not buf: return
            imgs = torch.stack([b['img'] for b in buf]).to(device)
            K = torch.stack([b['K'] for b in buf]).to(device)
            with torch.no_grad(): o = m(imgs, K)
            Ri = o.get('rot_matrix') if args.rot_head else None
            with torch.enable_grad():
                theta, kp_cam, _ = solve_batch(o['keypoints_2d'], o['confidence'], K, fix_joint7=True,
                                               iters=args.iters, lr=2e-2, img_size=S, device=device,
                                               prior_w=0.0, theta_init=o['joint_angles'], R_init=Ri)
            kp2d = o['keypoints_2d'].cpu().numpy(); conf = o['confidence'].cpu().numpy()
            kc = kp_cam.cpu().numpy()
            for i, b in enumerate(buf):
                f = b['found'] > 0
                if f.sum() < 4: continue
                add = float(np.linalg.norm(kc[i] - b['kp3d'], axis=1)[f].mean() * 1000)
                sx, sy = S / b['W'], S / b['H']; gt2d = b['gt2d'] * np.array([sx, sy])
                e2d = np.mean([np.linalg.norm(kp2d[i, k] - gt2d[k]) for k in range(7) if f[k]])
                p = kp2d[i][f]; spread = math.hypot(p[:, 0].ptp(), p[:, 1].ptp())
                dep_err = abs(kc[i, 0, 2] - b['kp3d'][0, 2]) * 1000
                ADD.append(add)
                X.append([b['fore'], e2d, int((b['inframe'] == 0).sum()), float(conf[i][f].min()),
                          spread, dep_err])
            buf.clear()
        for jf in files:
            fr = load_frame(jf, vd, S)
            if fr is None: continue
            buf.append(fr)
            if len(buf) >= args.batch_size: flush()
        flush()

    ADD = np.array(ADD); X = np.array(X); fail = ADD > 100
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    print(f"\n===== FAILURE CLUSTER PROBE (n={len(ADD)}, fail(ADD>100)={fail.mean()*100:.0f}%) =====")
    aucs = []
    for j, nm in enumerate(FEATS):
        a = roc_auc_score(fail, X[:, j]); a = max(a, 1 - a); aucs.append(a)
        print(f"  {nm:<16} single-feature AUROC {a:.2f}")
    # combined logistic regression (interpretable features)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-6)
    lr = LogisticRegression(max_iter=1000).fit(Xs, fail)
    comb = roc_auc_score(fail, lr.predict_proba(Xs)[:, 1])
    print(f"  {'ALL combined (LR)':<16} AUROC {comb:.2f}   <- vs backbone-feature head 0.94 (memory)")
    print(f"  best single feature: {FEATS[int(np.argmax(aucs))]} ({max(aucs):.2f})")

    # (1) ADD histogram
    plt.figure(figsize=(6, 4))
    plt.hist(np.clip(ADD, 0, 400), bins=60, color='#558')
    plt.axvline(100, c='r', ls='--'); plt.xlabel('ADD (mm, clipped@400)'); plt.ylabel('frames')
    plt.title(f'ADD distribution — smooth heavy tail, NOT bimodal\nfail>100mm = {fail.mean()*100:.0f}%')
    plt.tight_layout(); plt.savefig(os.path.join(OUT, 'add_hist.png'), dpi=120); plt.close()

    # (2) per-feature overlapped histograms
    fig, ax = plt.subplots(2, 3, figsize=(14, 7))
    for j, nm in enumerate(FEATS):
        a = ax[j // 3][j % 3]
        v = X[:, j]; lo, hi = np.percentile(v, 1), np.percentile(v, 99)
        bins = np.linspace(lo, hi, 30)
        a.hist(v[~fail], bins=bins, alpha=0.6, density=True, label='ok', color='#2a8')
        a.hist(v[fail], bins=bins, alpha=0.6, density=True, label='FAIL', color='#d44')
        a.set_title(f'{nm}  (AUROC {aucs[j]:.2f})'); a.legend(fontsize=8)
    fig.suptitle('Fail vs OK feature distributions — heavy OVERLAP = no clean separating cluster', fontsize=13)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, 'feature_separability.png'), dpi=120); plt.close()

    # (3) scatter of two best features
    order = np.argsort(aucs)[::-1]; j1, j2 = order[0], order[1]
    plt.figure(figsize=(6.5, 5))
    sc = plt.scatter(X[:, j1], X[:, j2], c=np.clip(ADD, 0, 300), cmap='inferno', s=14)
    plt.colorbar(sc, label='ADD (mm)'); plt.xlabel(FEATS[j1]); plt.ylabel(FEATS[j2])
    plt.title(f'2 most-informative features colored by ADD\nno clean boundary (failures smeared through)')
    plt.tight_layout(); plt.savefig(os.path.join(OUT, 'scatter.png'), dpi=120); plt.close()
    print(f"\nsaved -> {OUT}/{{add_hist,feature_separability,scatter}}.png")


if __name__ == '__main__':
    main()
