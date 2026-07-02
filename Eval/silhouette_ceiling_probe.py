"""
Render-and-compare CEILING probe — pure PyTorch, NO pytorch3d/segmenter install.
The depth-head lever died (scalar depth too fragile, <5% needed). render-and-compare supplies the same
depth/scale constraint ROBUSTLY: silhouette AREA ~ 1/z^2, so matching a rendered robot silhouette to an
observed one pins depth via a DENSE pixel measurement (not a brittle scalar). This gates that idea:
- Render the robot as a soft silhouette: sample points ALONG the FK link polyline, project with (R,t,K),
  splat soft Gaussians whose 2D radius ~ f*linkwidth/z (so the silhouette SCALES with depth).
- Target = silhouette rendered from the GT pose (ORACLE mask — replaced by a real segmenter if this works).
- Refine (theta,R,t) by maximizing soft-IoU to the target (+ a light keypoint reprojection term).
- Measure ADD-AUC before/after. If it robustly recovers ~the depth ceiling (+0.116 realsense), render-and-
  compare is worth building for real; if not, it's dead too.
"""
import argparse, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

TRAIN = os.path.abspath(os.path.join(os.path.dirname(__file__), '../TRAIN'))
sys.path.append(TRAIN); sys.path.append(os.path.dirname(__file__))
from model_angle import AnglePredictor
from model_v4 import panda_forward_kinematics
from inference_4tier_eval import EvalDataset
from solve_pose_kinematic import solve_batch, rot6d_to_matrix, matrix_to_rot6d
from refine_eval import scale_K

KPN = ['panda_link0', 'panda_link2', 'panda_link3', 'panda_link4', 'panda_link6', 'panda_link7', 'panda_hand']


def add_auc(adds_m, thr=0.1):
    a = np.asarray(adds_m); d = 1e-5; ts = np.arange(0.0, thr, d)
    return float(np.trapz((a[None, :] <= ts[:, None]).sum(1) / len(a), dx=d) / thr)


def skeleton_points(theta, n_seg=6):
    """FK -> 7 cam-frame-able robot-frame keypoints; densify the polyline to ~n_seg per segment."""
    fk = panda_forward_kinematics(theta)                       # (B,7,3) robot frame
    segs = []
    for i in range(fk.shape[1] - 1):
        a, b = fk[:, i:i + 1], fk[:, i + 1:i + 2]
        ts = torch.linspace(0, 1, n_seg, device=fk.device).view(1, n_seg, 1)
        segs.append(a + (b - a) * ts)                          # (B,n_seg,3)
    return torch.cat(segs, dim=1)                              # (B, ~36, 3)


def render_silhouette(pts_robot, R, t, K, H, linkw=0.05):
    """Soft silhouette (B,H,H) from robot-frame skeleton points under pose (R,t) and intrinsics K.
    radius_2d ~ f*linkw/z  -> the silhouette EXTENT scales with depth (the constraint we want)."""
    B, P, _ = pts_robot.shape
    cam = torch.einsum('bij,bpj->bpi', R, pts_robot) + t.unsqueeze(1)     # (B,P,3) camera frame
    z = cam[..., 2].clamp(min=1e-3)
    fx = K[:, 0, 0].view(B, 1); fy = K[:, 1, 1].view(B, 1)
    u = cam[..., 0] / z * fx + K[:, 0, 2].view(B, 1)
    v = cam[..., 1] / z * fy + K[:, 1, 2].view(B, 1)
    # downsample pixel coords to HxH grid (val-image is image_size; we render at H)
    scale = H / float(args_image_size)
    u = u * scale; v = v * scale
    sigma_px = (fx.mean() * linkw) / z * scale                            # (B,P) depth-dependent radius
    ys = torch.arange(H, device=cam.device).view(1, 1, H, 1).float()
    xs = torch.arange(H, device=cam.device).view(1, 1, 1, H).float()
    du = xs - u.view(B, P, 1, 1); dv = ys - v.view(B, P, 1, 1)
    s2 = (sigma_px.view(B, P, 1, 1) ** 2).clamp(min=1.0)
    g = torch.exp(-(du * du + dv * dv) / (2 * s2))                        # (B,P,H,H)
    mask = 1.0 - torch.prod(1.0 - g.clamp(max=0.999), dim=1)              # soft union (B,H,H)
    return mask


args_image_size = 512


def soft_iou(a, b, eps=1e-6):
    inter = (a * b).sum((-1, -2)); union = (a + b - a * b).sum((-1, -2))
    return inter / (union + eps)


def main():
    global args_image_size
    ap = argparse.ArgumentParser()
    ap.add_argument('--detector', required=True); ap.add_argument('--mlp-head', required=True)
    ap.add_argument('--val-dir', required=True)
    ap.add_argument('--model-name', default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    ap.add_argument('--image-size', type=int, default=512); ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--max-frames', type=int, default=400)
    ap.add_argument('--iters', type=int, default=200); ap.add_argument('--render-h', type=int, default=96)
    ap.add_argument('--rc-iters', type=int, default=120); ap.add_argument('--rc-lr', type=float, default=5e-3)
    ap.add_argument('--repro-w', type=float, default=20.0)
    args = ap.parse_args()
    args_image_size = args.image_size

    device = torch.device('cuda'); S = args.image_size; H = args.render_h
    m = AnglePredictor(args.model_name, S, head_type='mlp').to(device).eval()
    sd = torch.load(args.detector, map_location=device); sd = {k.replace('module.', ''): v for k, v in sd.items()}
    m.load_state_dict({k: v for k, v in sd.items() if k in m.state_dict() and v.shape == m.state_dict()[k].shape}, strict=False)
    m.angle_head.load_state_dict(torch.load(args.mlp_head, map_location=device))

    ds = EvalDataset(args.val_dir, KPN, image_size=(S, S))
    if args.max_frames and args.max_frames < len(ds.json_files):
        stride = max(1, len(ds.json_files) // args.max_frames); ds.json_files = ds.json_files[::stride][:args.max_frames]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    adds_base, adds_rc = [], []
    for batch in tqdm(loader, desc='sil-ceil'):
        img = batch['image'].to(device)
        K = scale_K(batch['camera_K'], batch['original_size'], S).to(device)
        gt3d = batch['gt_3d'].to(device); found = batch['found'].to(device); gt_ang = batch['gt_angles'].to(device)
        kp2d_gt = batch['gt_2d'].to(device); osz = batch['original_size'].to(device)
        with torch.no_grad():
            out = m(img, K)
        kp2d = out['keypoints_2d']; conf = out['confidence']; init = out['joint_angles']
        theta, kp_cam, _ = solve_batch(kp2d, conf, K, fix_joint7=True, iters=args.iters, lr=2e-2,
                                       img_size=S, device=device, prior_w=0.0, theta_init=init)
        # recover R,t of the solved pose via the same param path: re-fit from kp_cam vs FK(theta)
        # (cheap kabsch) so we can refine them under the silhouette.
        fkp = panda_forward_kinematics(theta)
        R0, t0 = kabsch_batch(fkp, kp_cam)
        # GT pose for the ORACLE target silhouette
        ga = gt_ang.clone(); ga[:, 6] = 0.0
        fkg = panda_forward_kinematics(ga)
        Rg, tg = kabsch_batch(fkg, gt3d)
        with torch.no_grad():
            tgt = render_silhouette(skeleton_points(ga), Rg, tg, K, H)            # oracle target mask
        # ---- render-and-compare refine of (theta,R,t) to match the target silhouette ----
        d6 = matrix_to_rot6d(R0).clone().detach().requires_grad_(True)
        tt = t0.clone().detach().requires_grad_(True)
        p_th = theta.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([d6, tt, p_th], lr=args.rc_lr)
        for _ in range(args.rc_iters):
            opt.zero_grad()
            R = rot6d_to_matrix(d6)
            sk = skeleton_points(torch.cat([p_th[:, :6], torch.zeros(p_th.shape[0], 1, device=device)], 1))
            mask = render_silhouette(sk, R, tt, K, H)
            loss = (1 - soft_iou(mask, tgt)).mean()
            # light keypoint reprojection anchor (keep 2D aligned to detector)
            fk = panda_forward_kinematics(torch.cat([p_th[:, :6], torch.zeros(p_th.shape[0],1,device=device)],1))
            cam = torch.einsum('bij,bpj->bpi', R, fk) + tt.unsqueeze(1)
            z = cam[..., 2].clamp(min=1e-3)
            uu = cam[..., 0]/z*K[:,0,0:1] + K[:,0,2:3]; vv = cam[..., 1]/z*K[:,1,1:2] + K[:,1,2:3]
            uv = torch.stack([uu, vv], -1)
            loss = loss + args.repro_w * ((uv - kp2d)/S * conf.unsqueeze(-1)).pow(2).mean()
            loss.backward(); opt.step()
        with torch.no_grad():
            R = rot6d_to_matrix(d6)
            fk = panda_forward_kinematics(torch.cat([p_th[:, :6], torch.zeros(p_th.shape[0],1,device=device)],1))
            kp_rc = torch.einsum('bij,bpj->bpi', R, fk) + tt.unsqueeze(1)
        f = found.bool()
        for b in range(img.shape[0]):
            if f[b].sum() < 5 or not torch.any(gt_ang[b] != 0):
                continue
            fb = f[b]
            adds_base.append(float((kp_cam[b][fb] - gt3d[b][fb]).norm(dim=-1).mean()))
            adds_rc.append(float((kp_rc[b][fb] - gt3d[b][fb]).norm(dim=-1).mean()))

    cam = os.path.basename(args.val_dir)
    print(f"\n=== SILHOUETTE-RC CEILING  {cam}  (n={len(adds_base)}, oracle target mask) ===")
    print(f"  baseline solve          ADD-AUC@100mm {add_auc(adds_base):.4f}  mean {1000*np.mean(adds_base):.1f}mm")
    print(f"  + render-compare refine ADD-AUC@100mm {add_auc(adds_rc):.4f}  mean {1000*np.mean(adds_rc):.1f}mm")
    print(f"  Δ from silhouette refine: {add_auc(adds_rc)-add_auc(adds_base):+.4f}")


def kabsch_batch(A, B):
    """Batched Kabsch: R,t with R@A+t ~= B. A,B (b,n,3)."""
    ca = A.mean(1, keepdim=True); cb = B.mean(1, keepdim=True)
    Ac = A - ca; Bc = B - cb
    Hm = torch.einsum('bni,bnj->bij', Ac, Bc)
    U, _, Vt = torch.linalg.svd(Hm)
    d = torch.det(torch.einsum('bij,bjk->bik', Vt.transpose(1, 2), U.transpose(1, 2)))
    D = torch.eye(3, device=A.device).unsqueeze(0).repeat(A.shape[0], 1, 1)
    D[:, 2, 2] = d
    R = torch.einsum('bij,bjk,bkl->bil', Vt.transpose(1, 2), D, U.transpose(1, 2))
    t = (cb - torch.einsum('bij,bnj->bni', R, ca)).squeeze(1)
    return R, t


if __name__ == '__main__':
    main()
