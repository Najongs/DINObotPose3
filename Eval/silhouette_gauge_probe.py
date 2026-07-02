"""
Answer the objection: "render-compare needs a mesh, the mesh needs good angles, but our angles are bad
(J0 ~28deg) -> the rendered mesh is useless." Test whether the BAD-angle solved pose actually produces a
BAD silhouette, or whether the angle error is GAUGE (cancels with camera-R) and leaves the camera-frame
silhouette ~unchanged.

For each real frame: render the robot silhouette at (a) the deployed SOLVED pose (bad angles) and (b) the
GT pose (Kabsch of GT angles -> GT 3D kp). Both are camera-frame. Report IoU(a,b) vs the J0 angle error.
  High IoU even when J0 error is large  => the bad angle is gauge; the rendered mesh IS usable for render-compare.
  IoU degrades with J0 error            => angles really do corrupt the mesh; user's concern holds.
Also report the DEPTH/scale mismatch (median |z| ratio) — the thing render-compare is actually meant to fix.
"""
import argparse, glob, os, sys
import numpy as np
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../TRAIN')))
sys.path.append(os.path.dirname(__file__))
os.environ.setdefault('MESH_KIND', 'collision')
from model_angle import AnglePredictor
from model_v4 import panda_forward_kinematics
from solve_pose_kinematic import solve_batch
from silhouette_mesh_probe import load_obj_verts, robot_pointcloud, render_mesh, kabsch_batch, mesh_path, LINK_MESH
from viz_hypothesis import load_frame


def iou(a, b):
    A = a > 0.5; B = b > 0.5
    return ((A & B).sum((-1, -2)).float() / (A | B).sum((-1, -2)).clamp(min=1).float())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--detector', required=True); ap.add_argument('--mlp-head', required=True)
    ap.add_argument('--rot-head', default=None)
    ap.add_argument('--val-dirs', nargs='+', required=True)
    ap.add_argument('--model-name', default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    ap.add_argument('--image-size', type=int, default=512); ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--max-frames', type=int, default=200); ap.add_argument('--iters', type=int, default=200)
    ap.add_argument('--render-h', type=int, default=224)
    args = ap.parse_args()
    device = torch.device('cuda'); S = args.image_size; RH = args.render_h

    m = AnglePredictor(args.model_name, S, head_type='mlp', with_rotation=args.rot_head is not None,
                       with_translation=args.rot_head is not None).to(device).eval()
    sd = torch.load(args.detector, map_location=device); sd = {k.replace('module.', ''): v for k, v in sd.items()}
    m.load_state_dict({k: v for k, v in sd.items() if k in m.state_dict() and v.shape == m.state_dict()[k].shape}, strict=False)
    m.angle_head.load_state_dict(torch.load(args.mlp_head, map_location=device))
    if args.rot_head: m.rot_head.load_state_dict(torch.load(args.rot_head, map_location=device))
    mesh_verts = [((nm, fi), torch.from_numpy(load_obj_verts(mesh_path(nm))).to(device)) for nm, fi in LINK_MESH]

    ious, j0s, amaes, zratio = [], [], [], []
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
            ga = torch.stack([torch.from_numpy(b['ang']).float() for b in buf]).to(device).clone(); ga[:, 6] = 0
            gt3d = torch.stack([torch.from_numpy(b['kp3d']).float() for b in buf]).to(device)
            with torch.no_grad():
                # solved pose render
                Rs, ts = kabsch_batch(panda_forward_kinematics(theta), kp_cam)
                sol = render_mesh(robot_pointcloud(theta, mesh_verts), Rs, ts, K, RH, S)
                # GT pose render
                Rg, tg = kabsch_batch(panda_forward_kinematics(ga), gt3d)
                gtm = render_mesh(robot_pointcloud(ga, mesh_verts), Rg, tg, K, RH, S)
            io = iou(sol, gtm).cpu().numpy()
            th = theta.cpu().numpy()
            for i, b in enumerate(buf):
                gg = b['ang'].copy(); gg[6] = 0
                d = np.degrees(np.abs(np.arctan2(np.sin(th[i, :6] - gg[:6]), np.cos(th[i, :6] - gg[:6]))))
                ious.append(float(io[i])); j0s.append(float(d[0])); amaes.append(float(d.mean()))
                zratio.append(float(abs(ts[i, 2].item()) / max(1e-3, abs(tg[i, 2].item()))))
            buf.clear()
        for jf in files:
            fr = load_frame(jf, vd, S)
            if fr is None: continue
            buf.append(fr)
            if len(buf) >= args.batch_size: flush()
        flush()

    ious = np.array(ious); j0s = np.array(j0s); amaes = np.array(amaes); zr = np.array(zratio)
    from scipy.stats import spearmanr
    rho, p = spearmanr(j0s, ious)
    print(f"\n===== SILHOUETTE GAUGE PROBE (n={len(ious)} real frames, render {RH}px) =====")
    print(f"IoU(render@solved-pose , render@GT-pose):  median {np.median(ious):.3f}  mean {ious.mean():.3f}")
    print(f"J0 angle error:  median {np.median(j0s):.1f}°  mean {j0s.mean():.1f}°   (angle MAE median {np.median(amaes):.1f}°)")
    print(f"Spearman  IoU ~ J0-error:  rho {rho:+.2f} (p={p:.1e})   <- near 0 => angle error does NOT corrupt the silhouette")
    for lo, hi in [(0, 10), (10, 25), (25, 200)]:
        msk = (j0s >= lo) & (j0s < hi)
        if msk.sum(): print(f"   J0 err [{lo:>3},{hi:>3})°  n={msk.sum():>3}   median silhouette IoU {np.median(ious[msk]):.3f}")
    print(f"\nDEPTH/scale (what render-compare actually fixes):  median |z_solved/z_GT| = {np.median(zr):.3f}  "
          f"(IQR {np.percentile(zr,25):.3f}-{np.percentile(zr,75):.3f});  frames >10% off = {100*(np.abs(zr-1)>0.1).mean():.0f}%")


if __name__ == '__main__':
    main()
