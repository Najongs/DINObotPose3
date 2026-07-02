"""
Arbitrate the user's challenge: are the realsense/orb base/ADD failures caused by SELF-OCCLUSION (a front
link hiding a back link -> that keypoint's 2D is MISLOCALIZED) or by monocular DEPTH ambiguity (2D fine,
depth under-constrained)? Decisive per-frame test:

  ADD_det     : solve with the DETECTED 2D (deployed)
  ADD_oracle  : solve with GT 2D for all found kp (perfect 2D), same init
  If a FAILING frame (ADD_det>100) RECOVERS under oracle 2D (ADD_oracle<100) -> its cause was 2D
     localization error (self-occlusion / detection). If it does NOT recover -> depth/geometry (monocular).

Also per frame: self-occlusion OVERLAP = #pairs of keypoints >0.2m apart in 3D but <25px apart in 2D
(=one link projecting onto another, GT-based), and max single-kp 2D detector error (a single bad kp that
mean-2D-err hides). Reports how much of the failure tail each explains.
"""
import argparse, glob, os, sys
import numpy as np
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../TRAIN')))
sys.path.append(os.path.dirname(__file__))
from model_angle import AnglePredictor
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

    def solve(kp, cf, K, ti, Ri):
        with torch.enable_grad():
            th, kc, _ = solve_batch(kp, cf, K, fix_joint7=True, iters=args.iters, lr=2e-2,
                                    img_size=S, device=device, prior_w=0.0, theta_init=ti, R_init=Ri)
        return kc

    add_det, add_ora, overlap, max2d, mean2d = [], [], [], [], []
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
            Ri = o.get('rot_matrix') if args.rot_head else None; ti = o['joint_angles']
            kp2d = o['keypoints_2d']; conf = o['confidence']
            gt = torch.stack([torch.from_numpy(b['gt2d']).float() * torch.tensor([S/b['W'], S/b['H']]) for b in buf]).to(device)
            found = torch.stack([torch.from_numpy(b['found']).float() for b in buf]).to(device)
            hi = conf.max().item()
            kp_o = kp2d.clone(); cf_o = conf.clone()
            for k in range(7):
                msk = found[:, k] > 0; kp_o[msk, k] = gt[msk, k]; cf_o[msk, k] = hi
            kc_d = solve(kp2d, conf, K, ti, Ri)
            kc_o = solve(kp_o, cf_o, K, ti, Ri)
            kp2d_np = kp2d.cpu().numpy()
            for i, b in enumerate(buf):
                f = b['found'] > 0
                if f.sum() < 4: continue
                add_det.append(float(np.linalg.norm(kc_d[i].cpu().numpy() - b['kp3d'], axis=1)[f].mean() * 1000))
                add_ora.append(float(np.linalg.norm(kc_o[i].cpu().numpy() - b['kp3d'], axis=1)[f].mean() * 1000))
                # self-occlusion overlap (GT-based): far in 3D, close in 2D@512
                g2 = gt[i].cpu().numpy(); k3 = b['kp3d']; idx = np.where(f)[0]; ov = 0
                for a in range(len(idx)):
                    for c in range(a + 1, len(idx)):
                        ia, ic = idx[a], idx[c]
                        if np.linalg.norm(k3[ia] - k3[ic]) > 0.2 and np.linalg.norm(g2[ia] - g2[ic]) < 25:
                            ov += 1
                overlap.append(ov)
                e = [np.linalg.norm(kp2d_np[i, k] - g2[k]) for k in range(7) if f[k]]
                max2d.append(max(e)); mean2d.append(np.mean(e))
            buf.clear()
        for jf in files:
            fr = load_frame(jf, vd, S)
            if fr is None: continue
            buf.append(fr)
            if len(buf) >= args.batch_size: flush()
        flush()

    ad = np.array(add_det); ao = np.array(add_ora); ov = np.array(overlap)
    mx = np.array(max2d); mn = np.array(mean2d); fail = ad > 100
    from sklearn.metrics import roc_auc_score
    print(f"\n===== SELF-OCCLUSION vs DEPTH PROBE (n={len(ad)}, fail(ADD_det>100)={fail.mean()*100:.0f}%) =====")
    print(f"\n[Decisive] does the FAILURE tail recover under ORACLE 2D?")
    print(f"  failing frames (ADD_det>100): n={fail.sum()}")
    if fail.sum():
        rec = (ao[fail] < 100).mean() * 100
        print(f"    median ADD_det    {np.median(ad[fail]):.0f}mm")
        print(f"    median ADD_oracle {np.median(ao[fail]):.0f}mm   <- oracle 2D on the SAME frames")
        print(f"    RECOVERED (<100mm w/ oracle 2D): {rec:.0f}%")
        print(f"    -> {rec:.0f}% of failures were a 2D-LOCALIZATION problem (self-occ/detection);")
        print(f"       {100-rec:.0f}% stay failed even w/ perfect 2D = DEPTH/geometry (monocular).")
    print(f"\n[Separability of failure causes]  AUROC for predicting ADD_det>100:")
    for nm, v in [('self-occ overlap', ov.astype(float)), ('max single-kp 2D err', mx), ('mean 2D err', mn)]:
        a = roc_auc_score(fail, v); print(f"    {nm:<22} {max(a,1-a):.2f}")
    print(f"\n[Self-occlusion prevalence]  frames with overlap>=1: {(ov>=1).mean()*100:.0f}%  "
          f"(among failures: {(ov[fail]>=1).mean()*100:.0f}%)")
    print(f"  median max-2D-err: ok {np.median(mx[~fail]):.1f}px | FAIL {np.median(mx[fail]):.1f}px")


if __name__ == '__main__':
    main()
