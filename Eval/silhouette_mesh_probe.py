"""
Render-and-compare CEILING probe v2 — REAL collision-mesh silhouette, pure PyTorch (no pytorch3d).
The skeleton-splat v1 failed (a thin polyline has ~zero AREA, and AREA is what encodes depth ~1/z^2).
This renders the actual Panda collision meshes (link0-7 + hand) transformed by per-link FK, projected and
soft-splatted (bilinear scatter + gaussian blur) into a FILLED differentiable silhouette whose area scales
with depth. Refine (theta,R,t) to match an ORACLE target silhouette (GT pose); if ADD robustly recovers
toward the depth ceiling (+0.116 realsense), render-and-compare is worth building with a real segmenter.
"""
import argparse, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

TRAIN = os.path.abspath(os.path.join(os.path.dirname(__file__), '../TRAIN'))
sys.path.append(TRAIN); sys.path.append(os.path.dirname(__file__))
import model_v4 as MV
from model_v4 import panda_forward_kinematics
from model_angle import AnglePredictor
from inference_4tier_eval import EvalDataset
from solve_pose_kinematic import solve_batch, rot6d_to_matrix, matrix_to_rot6d
from refine_eval import scale_K

KPN = ['panda_link0', 'panda_link2', 'panda_link3', 'panda_link4', 'panda_link6', 'panda_link7', 'panda_hand']
_MESH_KIND = os.environ.get('MESH_KIND', 'collision')   # collision = SOLID -> cleaner area/depth cue (oracle +0.108 vs visual +0.055)
MESH_DIR = os.path.join(os.path.dirname(__file__), '../ViS/Panda/meshes', _MESH_KIND)
# link mesh -> index into FK all_transforms (base, j1..j7, j8, hand) = 0..9
LINK_MESH = [('link0', 0), ('link1', 1), ('link2', 2), ('link3', 3), ('link4', 4),
             ('link5', 5), ('link6', 6), ('link7', 7), ('hand', 9)]


def mesh_path(name):
    """visual meshes live in per-link subfolders (link0/link0.obj); collision are flat (link0.obj)."""
    sub = os.path.join(MESH_DIR, name, name + '.obj')
    return sub if os.path.exists(sub) else os.path.join(MESH_DIR, name + '.obj')


def add_auc(adds_m, thr=0.1):
    a = np.asarray(adds_m); d = 1e-5; ts = np.arange(0.0, thr, d)
    return float(np.trapz((a[None, :] <= ts[:, None]).sum(1) / len(a), dx=d) / thr)


def load_obj_verts(path, cap=600):
    vs = []
    with open(path) as fh:
        for ln in fh:
            if ln.startswith('v '):
                vs.append([float(x) for x in ln.split()[1:4]])
    vs = np.asarray(vs, dtype=np.float32)
    if len(vs) > cap:
        # UNIFORM random subsample (OBJ verts are listed in face order -> stride clusters spatially,
        # giving a patchy silhouette; random sampling covers the surface evenly for a solid depth cue).
        rng = np.random.RandomState(0)
        vs = vs[rng.choice(len(vs), cap, replace=False)]
    return vs


def all_link_transforms(joint_angles):
    """Replicate panda_forward_kinematics but return ALL cumulative link frames (B,10,4,4)."""
    B = joint_angles.shape[0]; device, dtype = joint_angles.device, joint_angles.dtype
    fixed = [torch.tensor(MV._make_transform(j['xyz'], j['rpy']), device=device, dtype=dtype) for j in MV._PANDA_JOINTS]
    T_j8 = torch.tensor(MV._make_transform(MV._PANDA_FIXED_J8['xyz'], MV._PANDA_FIXED_J8['rpy']), device=device, dtype=dtype)
    T_hand = torch.tensor(MV._make_transform(MV._PANDA_FIXED_HAND['xyz'], MV._PANDA_FIXED_HAND['rpy']), device=device, dtype=dtype)
    cumul = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1).contiguous()
    frames = [cumul.clone()]
    for i in range(7):
        Rj = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1).clone()
        Rj[:, :3, :3] = MV._rotation_matrix_z(joint_angles[:, i])
        cumul = cumul @ fixed[i].unsqueeze(0) @ Rj
        frames.append(cumul.clone())
    cj8 = cumul @ T_j8.unsqueeze(0); frames.append(cj8.clone())
    frames.append((cj8 @ T_hand.unsqueeze(0)).clone())
    return torch.stack(frames, dim=1)                                   # (B,10,4,4)


def robot_pointcloud(theta, mesh_verts):
    """Transform each link's mesh verts by its FK frame -> (B, Ptot, 3) in robot base frame."""
    frames = all_link_transforms(theta)                                 # (B,10,4,4)
    pts = []
    for (name, fidx), v in mesh_verts:
        T = frames[:, fidx]                                             # (B,4,4)
        vh = torch.cat([v, torch.ones(v.shape[0], 1, device=v.device)], 1)   # (Nv,4)
        pc = torch.einsum('bij,nj->bni', T, vh)[..., :3]               # (B,Nv,3)
        pts.append(pc)
    return torch.cat(pts, dim=1)


_GK = None
def gauss_kernel(device, k=5, sigma=1.2):
    global _GK
    if _GK is None:
        ax = torch.arange(k, device=device).float() - (k - 1) / 2
        g = torch.exp(-(ax ** 2) / (2 * sigma ** 2)); g = (g / g.sum())
        _GK = (g[:, None] * g[None, :]).view(1, 1, k, k)
    return _GK


def render_mesh(pts_robot, R, t, K, H, img_size):
    B, P, _ = pts_robot.shape
    cam = torch.einsum('bij,bpj->bpi', R, pts_robot) + t.unsqueeze(1)
    z = cam[..., 2].clamp(min=1e-3)
    sc = H / float(img_size)
    u = (cam[..., 0] / z * K[:, 0, 0:1] + K[:, 0, 2:3]) * sc
    v = (cam[..., 1] / z * K[:, 1, 1:2] + K[:, 1, 2:3]) * sc
    u0 = u.floor(); v0 = v.floor(); wu = u - u0; wv = v - v0
    u0 = u0.long(); v0 = v0.long()
    dens = pts_robot.new_zeros(B, H * H)
    for du, dv, w in [(0, 0, (1 - wu) * (1 - wv)), (1, 0, wu * (1 - wv)), (0, 1, (1 - wu) * wv), (1, 1, wu * wv)]:
        uu = (u0 + du).clamp(0, H - 1); vv = (v0 + dv).clamp(0, H - 1)
        dens = dens.scatter_add(1, vv * H + uu, w)
    dens = dens.view(B, 1, H, H)
    dens = F.conv2d(dens, gauss_kernel(pts_robot.device), padding=2)
    return 1.0 - torch.exp(-dens.squeeze(1))                            # (B,H,H) soft mask


def degrade_mask(mask, sev):
    """Simulate a real-segmenter mask from the oracle: random erode/dilate (boundary shift), boundary
    noise, and rectangular region dropout. sev scales severity. Reports nothing; caller can compute IoU."""
    B, H, W = mask.shape
    m = mask.unsqueeze(1)
    k = 3
    # random erode OR dilate per-sample (min/max pool) -> systematic boundary offset like a segmenter
    pooled_d = F.max_pool2d(m, k, 1, k // 2)
    pooled_e = -F.max_pool2d(-m, k, 1, k // 2)
    coin = (torch.rand(B, 1, 1, 1, device=mask.device) < 0.5).float()
    amt = torch.clamp(torch.tensor(sev * 0.6, device=mask.device), 0, 1)
    m = m + amt * (coin * (pooled_d - m) + (1 - coin) * (pooled_e - m))
    # boundary noise: gaussian on soft values then re-soften
    m = (m + sev * 0.25 * torch.randn_like(m)).clamp(0, 1)
    # region dropout: zero a random rectangle in ~half the samples (missing gripper/link)
    if sev >= 1.0:
        for b in range(B):
            if torch.rand(1).item() < 0.4:
                cy = int(torch.randint(0, H, (1,)).item()); cx = int(torch.randint(0, W, (1,)).item())
                r = int(H * 0.12 * sev)
                m[b, 0, max(0, cy - r):cy + r, max(0, cx - r):cx + r] = 0.0
    return m.squeeze(1).clamp(0, 1)


def soft_iou(a, b, eps=1e-6):
    inter = (a * b).sum((-1, -2)); union = (a + b - a * b).sum((-1, -2))
    return inter / (union + eps)


def kabsch_batch(A, B):
    ca = A.mean(1, keepdim=True); cb = B.mean(1, keepdim=True)
    H = torch.einsum('bni,bnj->bij', A - ca, B - cb)
    U, _, Vt = torch.linalg.svd(H)
    d = torch.det(torch.einsum('bij,bjk->bik', Vt.transpose(1, 2), U.transpose(1, 2)))
    D = torch.eye(3, device=A.device).unsqueeze(0).repeat(A.shape[0], 1, 1); D[:, 2, 2] = d
    R = torch.einsum('bij,bjk,bkl->bil', Vt.transpose(1, 2), D, U.transpose(1, 2))
    t = (cb - torch.einsum('bij,bnj->bni', R, ca)).squeeze(1)
    return R, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--detector', required=True); ap.add_argument('--mlp-head', required=True)
    ap.add_argument('--val-dir', required=True)
    ap.add_argument('--model-name', default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    ap.add_argument('--image-size', type=int, default=512); ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--max-frames', type=int, default=400)
    ap.add_argument('--iters', type=int, default=200); ap.add_argument('--render-h', type=int, default=96)
    ap.add_argument('--rc-iters', type=int, default=150); ap.add_argument('--rc-lr', type=float, default=3e-3)
    ap.add_argument('--repro-w', type=float, default=50.0)
    ap.add_argument('--mask-degrade', type=float, default=0.0, help='degrade the target mask to sim a real segmenter (0=oracle): morphological erode/dilate jitter + boundary noise + region dropout')
    ap.add_argument('--mask-head', default=None, help='trained DINOv3 mask head: PREDICT the target mask from the REAL image (deployable, non-oracle) instead of rendering the GT pose')
    ap.add_argument('--mask-res', type=int, default=256, help='out_res the mask head was trained at')
    ap.add_argument('--sam-checkpoint', default=None, help='SAM ViT-B checkpoint: prompt with detected keypoints to segment the REAL robot mask (correctly-placed) as the render-compare target')
    args = ap.parse_args()
    torch.manual_seed(0)

    device = torch.device('cuda'); S = args.image_size; H = args.render_h
    mesh_verts = [((nm, fi), torch.from_numpy(load_obj_verts(mesh_path(nm))).to(device))
                  for nm, fi in LINK_MESH]
    print('mesh pts total:', sum(v.shape[0] for _, v in mesh_verts), flush=True)

    m = AnglePredictor(args.model_name, S, head_type='mlp').to(device).eval()
    sd = torch.load(args.detector, map_location=device); sd = {k.replace('module.', ''): v for k, v in sd.items()}
    m.load_state_dict({k: v for k, v in sd.items() if k in m.state_dict() and v.shape == m.state_dict()[k].shape}, strict=False)
    m.angle_head.load_state_dict(torch.load(args.mlp_head, map_location=device))

    mask_head = None
    if args.mask_head:
        from model_v4 import ViTKeypointHead as VKH
        mask_head = VKH(input_dim=m.backbone.model.config.hidden_size, num_joints=1,
                        heatmap_size=(args.mask_res, args.mask_res)).to(device).eval()
        mask_head.load_state_dict(torch.load(args.mask_head, map_location=device))
        print(f"[mask-head] predicting REAL target masks from {args.mask_head}", flush=True)

    sam_pred = None
    if args.sam_checkpoint:
        from segment_anything import sam_model_registry, SamPredictor
        sam = sam_model_registry['vit_b'](checkpoint=args.sam_checkpoint).to(device); sam.eval()
        sam_pred = SamPredictor(sam)
        print(f"[SAM] segmenting REAL robot masks (prompted by detected keypoints) from {args.sam_checkpoint}", flush=True)
    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def sam_masks(img_norm, kp2d, conf, Hout):
        """Un-normalize the batch, prompt SAM per-image with confident detected keypoints, return (B,Hout,Hout)."""
        u8 = ((img_norm * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
        out = []
        for b in range(u8.shape[0]):
            sam_pred.set_image(u8[b])
            pts = kp2d[b].detach().cpu().numpy(); cf = conf[b].detach().cpu().numpy()
            sel = cf > 0.3
            if sel.sum() < 2:
                sel = cf >= np.sort(cf)[-2]
            p = pts[sel]
            # BOX prompt from confident keypoints (+margin) -> SAM segments the WHOLE robot; point prompts
            # alone gave partial/over-segmented masks. Keep the points as positive cues too.
            x0, y0 = p[:, 0].min(), p[:, 1].min(); x1, y1 = p[:, 0].max(), p[:, 1].max()
            mx = 0.15 * max(x1 - x0, y1 - y0)
            box = np.array([max(0, x0 - mx), max(0, y0 - mx), min(S, x1 + mx), min(S, y1 + mx)])
            m, sc, _ = sam_pred.predict(point_coords=p, point_labels=np.ones(len(p)), box=box, multimask_output=True)
            mask = torch.from_numpy(m[int(np.argmax(sc))].astype('float32')).to(device)   # (S,S) best
            out.append(F.interpolate(mask[None, None], size=(Hout, Hout), mode='bilinear', align_corners=False)[0, 0])
        return torch.stack(out)

    ds = EvalDataset(args.val_dir, KPN, image_size=(S, S))
    if args.max_frames and args.max_frames < len(ds.json_files):
        st = max(1, len(ds.json_files) // args.max_frames); ds.json_files = ds.json_files[::st][:args.max_frames]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    adds_base, adds_rc, mask_ious = [], [], []
    for batch in tqdm(loader, desc='mesh-rc'):
        img = batch['image'].to(device)
        K = scale_K(batch['camera_K'], batch['original_size'], S).to(device)
        gt3d = batch['gt_3d'].to(device); found = batch['found'].to(device); gt_ang = batch['gt_angles'].to(device)
        with torch.no_grad():
            out = m(img, K)
        kp2d = out['keypoints_2d']; conf = out['confidence']; init = out['joint_angles']
        theta, kp_cam, _ = solve_batch(kp2d, conf, K, fix_joint7=True, iters=args.iters, lr=2e-2,
                                       img_size=S, device=device, prior_w=0.0, theta_init=init)
        fkp = panda_forward_kinematics(theta); R0, t0 = kabsch_batch(fkp, kp_cam)
        ga = gt_ang.clone(); ga[:, 6] = 0.0
        fkg = panda_forward_kinematics(ga); Rg, tg = kabsch_batch(fkg, gt3d)
        with torch.no_grad():
            if sam_pred is not None:
                tgt = sam_masks(img, kp2d, conf, H)
                orc = (render_mesh(robot_pointcloud(ga, mesh_verts), Rg, tg, K, H, S) > 0.5).float()
                pb = (tgt > 0.5).float()
                mi = (pb * orc).sum((-1, -2)) / ((pb + orc) > 0).float().sum((-1, -2)).clamp(min=1)
                mask_ious.extend(mi.cpu().tolist())
            elif mask_head is not None:
                tok = m.backbone(img)
                pm = torch.sigmoid(mask_head(tok)).squeeze(1)               # (B,mask_res,mask_res) on REAL image
                tgt = F.interpolate(pm.unsqueeze(1), size=(H, H), mode='bilinear', align_corners=False).squeeze(1)
                # DIAGNOSTIC: IoU of the predicted real mask vs the ORACLE (GT-pose) mask -> is it well-placed?
                orc = (render_mesh(robot_pointcloud(ga, mesh_verts), Rg, tg, K, H, S) > 0.5).float()
                pb = (tgt > 0.5).float()
                mi = (pb * orc).sum((-1, -2)) / ((pb + orc) > 0).float().sum((-1, -2)).clamp(min=1)
                mask_ious.extend(mi.cpu().tolist())
            else:
                tgt = render_mesh(robot_pointcloud(ga, mesh_verts), Rg, tg, K, H, S)
                if args.mask_degrade > 0:
                    tgt = degrade_mask(tgt, args.mask_degrade)
        d6 = matrix_to_rot6d(R0).clone().detach().requires_grad_(True)
        tt = t0.clone().detach().requires_grad_(True)
        pth = theta[:, :6].clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([d6, tt, pth], lr=args.rc_lr)
        zc = torch.zeros(pth.shape[0], 1, device=device)
        for _ in range(args.rc_iters):
            opt.zero_grad()
            th = torch.cat([pth, zc], 1); R = rot6d_to_matrix(d6)
            mask = render_mesh(robot_pointcloud(th, mesh_verts), R, tt, K, H, S)
            loss = (1 - soft_iou(mask, tgt)).mean()
            fk = panda_forward_kinematics(th)
            cam = torch.einsum('bij,bpj->bpi', R, fk) + tt.unsqueeze(1); z = cam[..., 2].clamp(min=1e-3)
            uv = torch.stack([cam[..., 0] / z * K[:, 0, 0:1] + K[:, 0, 2:3],
                              cam[..., 1] / z * K[:, 1, 1:2] + K[:, 1, 2:3]], -1)
            loss = loss + args.repro_w * (((uv - kp2d) / S) * conf.unsqueeze(-1)).pow(2).mean()
            loss.backward(); opt.step()
        with torch.no_grad():
            th = torch.cat([pth, zc], 1); R = rot6d_to_matrix(d6)
            fk = panda_forward_kinematics(th); kp_rc = torch.einsum('bij,bpj->bpi', R, fk) + tt.unsqueeze(1)
        f = found.bool()
        for b in range(img.shape[0]):
            if f[b].sum() < 5 or not torch.any(gt_ang[b] != 0):
                continue
            fb = f[b]
            adds_base.append(float((kp_cam[b][fb] - gt3d[b][fb]).norm(dim=-1).mean()))
            adds_rc.append(float((kp_rc[b][fb] - gt3d[b][fb]).norm(dim=-1).mean()))

    cam = os.path.basename(args.val_dir)
    print(f"\n=== MESH-SILHOUETTE-RC CEILING  {cam}  (n={len(adds_base)}, oracle target) ===")
    print(f"  baseline solve          ADD-AUC@100mm {add_auc(adds_base):.4f}  mean {1000*np.mean(adds_base):.1f}mm")
    print(f"  + render-compare refine ADD-AUC@100mm {add_auc(adds_rc):.4f}  mean {1000*np.mean(adds_rc):.1f}mm")
    print(f"  Δ from mesh-silhouette refine: {add_auc(adds_rc)-add_auc(adds_base):+.4f}")
    if mask_ious:
        mi = np.array(mask_ious)
        print(f"  [diag] predicted-mask vs ORACLE-mask IoU: mean {mi.mean():.3f}  median {np.median(mi):.3f}  "
              f"frac<0.4: {(mi<0.4).mean():.2f}  (low => mis-placed mask => needs segmenter)")


if __name__ == '__main__':
    main()
