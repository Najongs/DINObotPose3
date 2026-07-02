"""
Should we build a RootNet/DepthNet (HPE-style)? Decide BEFORE training. HPE depth = gamma * k_value,
k_value = sqrt(fx*fy*A_real/A_image) (geometric prior), gamma = learned multiplicative correction.
We have a BETTER k_value than HPE (they use a fixed real_bbox=1m; we know theta -> A_real from FK).

Decisive question: does the GEOMETRIC depth (gamma=1) track GT root depth, or is it fooled by
foreshortening just like our solver? Compare three depth estimates vs GT root depth on real frames:
  solver_z   : our kinematic solver's base depth (what we already have)
  k_fix_z    : HPE geometric prior, fixed A_real=1m^2
  k_fk_z     : OUR geometric prior, A_real = FK(theta) 3D-extent^2  (better)
If k_fk_z (best geometric, after global scale) BEATS solver on the foreshortened tail -> a DepthNet
(gamma correction) has headroom -> build it. If it's no better / fooled by foreshortening -> same dead
end as the scalar depth head; render-compare (full silhouette) stays the only real depth lever.
"""
import argparse, glob, math, os, sys
import numpy as np
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../TRAIN')))
sys.path.append(os.path.dirname(__file__))
from model_angle import AnglePredictor
from model_v4 import panda_forward_kinematics
from solve_pose_kinematic import solve_batch
from viz_hypothesis import load_frame


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

    m = AnglePredictor(args.model_name, S, head_type='mlp', with_rotation=args.rot_head is not None,
                       with_translation=args.rot_head is not None).to(device).eval()
    sd = torch.load(args.detector, map_location=device); sd = {k.replace('module.', ''): v for k, v in sd.items()}
    m.load_state_dict({k: v for k, v in sd.items() if k in m.state_dict() and v.shape == m.state_dict()[k].shape}, strict=False)
    m.angle_head.load_state_dict(torch.load(args.mlp_head, map_location=device))
    if args.rot_head: m.rot_head.load_state_dict(torch.load(args.rot_head, map_location=device))

    gt_z, solv_z, kfix, kfk, fore = [], [], [], [], []
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
            kp2d = o['keypoints_2d']                                  # (B,7,2) @512
            fk = panda_forward_kinematics(theta)                     # (B,7,3) robot frame, metres
            for i, b in enumerate(buf):
                f = b['found'] > 0
                if f.sum() < 4: continue
                fx = float(b['K'][0, 0]); fy = float(b['K'][1, 1])
                p = kp2d[i].cpu().numpy()[f]
                w = p[:, 0].max() - p[:, 0].min(); h = p[:, 1].max() - p[:, 1].min()
                A_img = max(w, h) ** 2                                 # px^2 (HPE: max-side^2)
                if A_img < 1: continue
                fkp = fk[i].cpu().numpy()                              # 3D, m
                ext = np.linalg.norm(fkp.max(0) - fkp.min(0))         # 3D diagonal extent (m)
                A_real_fk = (ext * 1000.0) ** 2                       # mm^2
                A_real_fix = 1000.0 ** 2                              # HPE fixed 1m
                kfix.append(math.sqrt(fx * fy * A_real_fix / A_img) / 1000.0)   # -> m
                kfk.append(math.sqrt(fx * fy * A_real_fk / A_img) / 1000.0)
                gt_z.append(float(b['kp3d'][0, 2]))                   # base GT depth (m)
                solv_z.append(float(kp_cam[i, 0, 2].cpu()))           # solver base depth
                fore.append(b['fore'])
            buf.clear()
        for jf in files:
            fr = load_frame(jf, vd, S)
            if fr is None: continue
            buf.append(fr)
            if len(buf) >= args.batch_size: flush()
        flush()

    gt = np.array(gt_z); sv = np.array(solv_z); kfx = np.array(kfix); kf = np.array(kfk); fr = np.array(fore)
    from scipy.stats import spearmanr
    def scaled_err(est):
        s = np.median(gt / est)                                       # best global scale (gamma const)
        e = np.abs(est * s - gt)
        return s, np.median(e) * 1000, np.median(np.abs(est * s - gt) / gt) * 100   # mm, %
    print(f"\n===== ROOTNET DEPTH PROBE (n={len(gt)} real frames) =====")
    print(f"GT root depth: median {np.median(gt):.3f} m  range [{gt.min():.2f},{gt.max():.2f}]")
    print(f"\n{'estimate':<10}{'spearman~GT':>12}{'best gamma':>11}{'med|err|mm':>11}{'med err%':>9}")
    for nm, est in [('solver', sv), ('k_fix(HPE)', kfx), ('k_fk(ours)', kf)]:
        rho, _ = spearmanr(est, gt); s, em, ep = scaled_err(est)
        # solver needs no scale (already metric); report its raw error too
        raw = np.median(np.abs(est - gt)) * 1000 if nm == 'solver' else em
        print(f"{nm:<10}{rho:>+12.2f}{s:>11.3f}{raw:>11.0f}{ep:>8.0f}%")
    # foreshortened tail (where depth matters)
    tail = fr > np.percentile(fr, 70)
    print(f"\nForeshortened tail (top-30% fore, n={tail.sum()}):  median GT depth {np.median(gt[tail]):.3f}")
    for nm, est in [('solver', sv), ('k_fk(ours)', kf)]:
        s = 1.0 if nm == 'solver' else np.median(gt / est)
        em = np.median(np.abs(est[tail] * s - gt[tail])) * 1000
        rho, _ = spearmanr(est[tail], gt[tail])
        print(f"   {nm:<10} spearman~GT {rho:+.2f}   med|err| {em:.0f}mm")
    print("\nREAD: if k_fk spearman>~0.6 AND its tail err < solver's -> geometric depth has signal a DepthNet can")
    print("      refine (build it). If k_fk spearman low / tail err >= solver -> foreshortening fools it (skip).")


if __name__ == '__main__':
    main()
