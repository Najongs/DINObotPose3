"""
DECISIVE base-error decomposition: is the J0/base failure a 2D-DETECTION problem (sim2real, fixable by a
better base detector) or geometric UNDER-DETERMINATION (perfect 2D still can't pin base-yaw from one view)?

For each real frame, run the locked solver 3 ways and compare J0 MAE / ADD:
  baseline   : detected 2D (as deployed)
  base-oracle: replace ONLY link0+link2 (the base-defining kp) 2D with GT, conf high, re-solve
  all-oracle : replace ALL kp 2D with GT, conf high, re-solve   (under-determination floor)

theta_init / R_init kept from the model in all 3 (we isolate the SOLVER's dependence on base 2D).
Read: base-oracle≈all-oracle≪baseline  => base DETECTION is the lever (user's hypothesis right).
      base-oracle≈baseline             => J0 is under-determined; better 2D won't help (need rot/render).
"""
import argparse, glob, os, sys
import numpy as np
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../TRAIN')))
sys.path.append(os.path.dirname(__file__))
from model_angle import AnglePredictor
from solve_pose_kinematic import solve_batch
from viz_hypothesis import load_frame, KPN

BASE_KP = [0, 1]   # link0 (base), link2 (first arm) — the two that fix base position+yaw


def jerr(theta, ga):
    d = np.arctan2(np.sin(theta[:6] - ga[:6]), np.cos(theta[:6] - ga[:6]))
    return np.degrees(np.abs(d))


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

    def solve(kp2d, conf, K, ti, Ri):
        with torch.enable_grad():
            theta, kc, _ = solve_batch(kp2d, conf, K, fix_joint7=True, iters=args.iters, lr=2e-2,
                                       img_size=S, device=device, prior_w=0.0, theta_init=ti, R_init=Ri)
        return theta, kc

    rows = {'baseline': [], 'base_oracle': [], 'all_oracle': []}
    addr = {'baseline': [], 'base_oracle': [], 'all_oracle': []}
    base2d_err = []
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
            ti = o['joint_angles']
            kp2d = o['keypoints_2d']; conf = o['confidence']
            # GT 2D @ S
            gt = torch.stack([torch.from_numpy(b['gt2d']).float() * torch.tensor([S/b['W'], S/b['H']]) for b in buf]).to(device)
            found = torch.stack([torch.from_numpy(b['found']).float() for b in buf]).to(device)
            hi = conf.max().item()
            # base-oracle: replace base kp only
            kp_b = kp2d.clone(); cf_b = conf.clone()
            for k in BASE_KP:
                kp_b[:, k] = gt[:, k]; cf_b[:, k] = hi
            # all-oracle: replace all found kp
            kp_a = kp2d.clone(); cf_a = conf.clone()
            for k in range(7):
                msk = found[:, k] > 0
                kp_a[msk, k] = gt[msk, k]; cf_a[msk, k] = hi
            outs = {'baseline': solve(kp2d, conf, K, ti, Ri),
                    'base_oracle': solve(kp_b, cf_b, K, ti, Ri),
                    'all_oracle': solve(kp_a, cf_a, K, ti, Ri)}
            for i, b in enumerate(buf):
                ga = b['ang'].copy(); ga[6] = 0.0; f = b['found']
                for name, (th, kc) in outs.items():
                    rows[name].append(jerr(th[i].cpu().numpy(), ga))
                    addr[name].append(float(np.linalg.norm(kc[i].cpu().numpy() - b['kp3d'], axis=1)[f > 0].mean() * 1000))
                # base detector 2D err
                e = [np.linalg.norm(kp2d[i, k].cpu().numpy() - gt[i, k].cpu().numpy()) for k in BASE_KP if f[k]]
                base2d_err.append(np.mean(e) if e else np.nan)
            buf.clear()
        for jf in files:
            fr = load_frame(jf, vd, S)
            if fr is None: continue
            buf.append(fr)
            if len(buf) >= args.batch_size: flush()
        flush()

    n = len(rows['baseline'])
    print(f"\n========= BASE ERROR DECOMPOSITION (n={n} real frames) =========")
    print(f"base detector 2D err (link0/link2): median {np.nanmedian(base2d_err):.1f}px  mean {np.nanmean(base2d_err):.1f}px")
    print(f"\n{'variant':<12} {'J0 MAE':>8} {'J2 MAE':>8} {'J(0-5)MAE':>10} {'ADD med':>8} {'fail%':>6}")
    for name in ['baseline', 'base_oracle', 'all_oracle']:
        A = np.array(rows[name]); ad = np.array(addr[name])
        print(f"{name:<12} {A[:,0].mean():>7.1f}° {A[:,2].mean():>7.1f}° {A[:,:6].mean():>9.1f}° "
              f"{np.median(ad):>6.0f}mm {100*(ad>100).mean():>5.0f}%")
    b = np.array(rows['baseline']); bo = np.array(rows['base_oracle']); ao = np.array(rows['all_oracle'])
    j0b, j0bo, j0ao = b[:,0].mean(), bo[:,0].mean(), ao[:,0].mean()
    adb, adbo, adao = np.median(addr['baseline']), np.median(addr['base_oracle']), np.median(addr['all_oracle'])
    print(f"\n  J0 angle : baseline {j0b:.1f}° -> base-oracle {j0bo:.1f}° -> all-oracle {j0ao:.1f}°")
    print(f"  ADD(med) : baseline {adb:.0f}mm -> base-oracle {adbo:.0f}mm -> all-oracle {adao:.0f}mm")
    # Two SEPARATE questions:
    if j0b - j0ao < 3.0:
        print("  READ-ANGLE: even ALL-oracle 2D leaves J0 ~unchanged => J0 is GAUGE/UNDER-DETERMINED, "
              "not a detection problem. Better 2D will NOT fix the joint angle (need rot-head / multi-view / render).")
    else:
        print("  READ-ANGLE: oracle 2D reduces J0 => detection/sim2real is a real J0 lever.")
    if adb - adao > 0.5 * adb:
        print(f"  READ-ADD: oracle 2D collapses ADD ({adb:.0f}->{adao:.0f}mm) => the BENCHMARK metric is "
              f"bottlenecked by 2D PRECISION/outliers (base-only gets {adbo:.0f}mm; the tail needs ALL kp sharp).")


if __name__ == '__main__':
    main()
