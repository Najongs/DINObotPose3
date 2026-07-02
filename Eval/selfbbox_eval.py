"""
Self-bbox (oracle-free) crop pipeline eval. Productionizes the train+test crop win without GT
keypoints: stage-1 (full-frame) detector runs a FIRST PASS on the full image -> detected 2D ->
square robot bbox -> roi_align crop+resize to image_size (K principal-point shift + focal scale) ->
crop pipeline (crop detector + crop angle head + crop rot-head) SECOND PASS -> kinematic solver.

Answers the honesty question: does the crop ADD gain survive a DETECTED (not GT) bbox? The crop
training used bbox jitter (margin 1.3-1.5, scale .9-1.1, center +-.1) so the head is robust to an
imperfect bbox. Mirrors dataset.py's GT-crop math but sourced from the first-pass detector.
"""
import argparse, glob, math, os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.ops import roi_align
from tqdm import tqdm

TRAIN = os.path.abspath(os.path.join(os.path.dirname(__file__), '../TRAIN'))
sys.path.append(TRAIN); sys.path.append(os.path.dirname(__file__))
from model_angle import AnglePredictor                       # noqa
from dataset import PoseEstimationDataset                    # noqa
from solve_pose_kinematic import solve_batch                 # noqa
from refine_eval import wrapped_abs_deg, add_auc, scale_K    # reuse helpers


def build_predictor(model_name, image_size, detector_ckpt, device,
                    angle_ckpt=None, rot_ckpt=None):
    with_rot = rot_ckpt is not None
    m = AnglePredictor(model_name, image_size, head_type='mlp',
                       with_rotation=with_rot, with_translation=with_rot).to(device).eval()
    sd = torch.load(detector_ckpt, map_location=device)
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    m.load_state_dict({k: v for k, v in sd.items()
                       if k in m.state_dict() and v.shape == m.state_dict()[k].shape}, strict=False)
    if angle_ckpt:
        m.angle_head.load_state_dict(torch.load(angle_ckpt, map_location=device))
    if rot_ckpt:
        m.rot_head.load_state_dict(torch.load(rot_ckpt, map_location=device))
    return m


def detected_bbox(kp2d, conf, img_size, margin, conf_thr):
    """Square robot bbox (B,4) x0y0x1y1 in img_size space from confident detected keypoints.
    Mirrors dataset.py: center=midpoint, side=max(dx,dy)*margin. Fallback to full frame."""
    B = kp2d.shape[0]
    boxes = kp2d.new_zeros(B, 4)
    for b in range(B):
        m = conf[b] > conf_thr
        if int(m.sum()) < 2:
            m = conf[b] > 0.0  # fallback: use whatever we have
        if int(m.sum()) < 2:
            boxes[b] = torch.tensor([0., 0., float(img_size), float(img_size)], device=kp2d.device)
            continue
        pts = kp2d[b][m]
        x0, x1 = pts[:, 0].min(), pts[:, 0].max()
        y0, y1 = pts[:, 1].min(), pts[:, 1].max()
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        side = torch.clamp(torch.max(x1 - x0, y1 - y0) * margin, min=16.0)
        boxes[b, 0] = cx - side / 2; boxes[b, 1] = cy - side / 2
        boxes[b, 2] = cx + side / 2; boxes[b, 3] = cy + side / 2
    return boxes


def project_points(kp_cam, K):
    """Project camera-frame 3D points (B,N,3) to pixel 2D (B,N,2) with intrinsics K (B,3,3)."""
    z = kp_cam[..., 2:3].clamp(min=1e-4)
    uv = kp_cam[..., :2] / z                                   # normalized image plane
    fx, fy = K[:, 0, 0].unsqueeze(1), K[:, 1, 1].unsqueeze(1)
    cx, cy = K[:, 0, 2].unsqueeze(1), K[:, 1, 2].unsqueeze(1)
    u = uv[..., 0] * fx + cx; v = uv[..., 1] * fy + cy
    return torch.stack([u, v], dim=-1)                         # (B,N,2)


def clamp_boxes(boxes, img_size, min_side=24.0):
    """Sanity-clamp boxes: replace NaN/inf, enforce a min side, keep the center on-frame. Prevents a
    degenerate bbox (side~0 or huge) from blowing up crop_K's scale -> solver divergence."""
    boxes = torch.nan_to_num(boxes, nan=0.0, posinf=float(img_size), neginf=0.0)
    cx = (boxes[:, 0] + boxes[:, 2]) / 2; cy = (boxes[:, 1] + boxes[:, 3]) / 2
    side = (boxes[:, 2] - boxes[:, 0]).clamp(min=min_side, max=float(img_size) * 2)
    cx = cx.clamp(0.0, float(img_size)); cy = cy.clamp(0.0, float(img_size))
    out = boxes.clone()
    out[:, 0] = cx - side / 2; out[:, 1] = cy - side / 2
    out[:, 2] = cx + side / 2; out[:, 3] = cy + side / 2
    return out


def crop_K(K, boxes, out_size):
    """Adjust K for crop [x0,y0]+side then resize side->out_size. scale=out_size/side."""
    Kc = K.clone()
    for b in range(K.shape[0]):
        x0, y0 = boxes[b, 0], boxes[b, 1]
        side = (boxes[b, 2] - boxes[b, 0]).clamp(min=1.0)
        s = out_size / side
        Kc[b, 0, 0] = K[b, 0, 0] * s; Kc[b, 1, 1] = K[b, 1, 1] * s
        Kc[b, 0, 2] = (K[b, 0, 2] - x0) * s; Kc[b, 1, 2] = (K[b, 1, 2] - y0) * s
    return Kc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage1-detector', required=True, help='full-frame detector for the bbox first pass')
    ap.add_argument('--stage1-angle', default=None, help='full-frame angle head (for --bbox-from-solved)')
    ap.add_argument('--stage1-rot', default=None, help='full-frame rot-head (for --bbox-from-solved)')
    ap.add_argument('--bbox-guard', action='store_true',
                    help='divergence guard: fall back to detected-kp bbox when the pass-1 solve diverged '
                         '(high reproj / off-frame projection), and clamp all bboxes. Use with --bbox-from-solved.')
    ap.add_argument('--bbox-reproj-thr', type=float, default=1e9, help='OPTIONAL extra gate: pass-1 reproj px above which the solved bbox is deemed diverged (OFF by default; geometric off-frame/NaN check is the primary guard)')
    ap.add_argument('--bbox-union', action='store_true',
                    help='UNION the solved-FK bbox with the confidently-detected-keypoint bbox. Solver '
                         'bias mildly mis-crops (the -0.04 vs GT-crop gap); detected keypoints are direct '
                         'observations, so unioning guarantees the crop never clips a confident keypoint '
                         'while still using the solved skeleton to fill the occluded base. Use with --bbox-from-solved.')
    ap.add_argument('--bbox-from-solved', action='store_true',
                    help='build the pass-1 bbox from the SOLVED full-frame pose: run the kinematic '
                         'solver on the full-frame detection, project ALL 7 FK keypoints (incl. the '
                         'occluded base that the raw detector misses) -> bbox. Training-free fix for '
                         'the keypoint-bbox failure (occluded base dropped -> mis-centered crop -> 0.55).')
    ap.add_argument('--crop-detector', required=True, help='crop-trained detector for the second pass')
    ap.add_argument('--crop-angle', required=True, help='crop-trained angle head')
    ap.add_argument('--rot-head', default=None, help='crop-native rot-head (R_init)')
    ap.add_argument('--val-dir', required=True)
    ap.add_argument('--model-name', default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    ap.add_argument('--image-size', type=int, default=512)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--max-frames', type=int, default=1000)
    ap.add_argument('--frac-range', nargs=2, type=float, default=None,
                    help='restrict to ds.samples[LO*N:HI*N] BEFORE striding (anti-leak held-out eval; '
                         'use 0.7 1.0 to match selftrain contiguous last-30%% split)')
    ap.add_argument('--iters', type=int, default=200)
    ap.add_argument('--conf-gate', type=float, default=0.05)
    ap.add_argument('--margin', type=float, default=1.5)
    ap.add_argument('--bbox-conf', type=float, default=0.1, help='conf threshold for bbox keypoints')
    ap.add_argument('--bbox-refine-iters', type=int, default=0,
                    help='iterative bbox refinement: after the stage1 full-frame bbox, re-detect with '
                         'the CROP detector on the zoomed crop (robot bigger -> sharper keypoints -> '
                         'tighter bbox) and re-crop, N times. RoboPEPP-style coarse-to-fine.')
    ap.add_argument('--oracle-bbox', action='store_true',
                    help='DEBUG: build the bbox from GT keypoints (not the detector) to isolate the '
                         'roi_align/K crop pipeline from detected-bbox error; should reproduce the '
                         'dataset.py GT-crop ADD (~0.768 realsense) if the crop math is correct')
    ap.add_argument('--dump-npz', default=None, help='save (fid, theta, kp_cam, gt3d, found) for cross-env render-compare')
    args = ap.parse_args()
    _DUMP = {'fid': [], 'theta': [], 'kp_cam': [], 'gt3d': [], 'found': [], 'feat': [], 'reproj': []} if args.dump_npz else None

    device = torch.device('cuda'); assert torch.cuda.is_available()
    IS = args.image_size
    # bbox pass: plain detector, OR a full predictor (detector+angle+rot) when solving for the bbox
    det1 = build_predictor(args.model_name, IS, args.stage1_detector, device,
                           angle_ckpt=args.stage1_angle if args.bbox_from_solved else None,
                           rot_ckpt=args.stage1_rot if args.bbox_from_solved else None)     # bbox pass
    cropm = build_predictor(args.model_name, IS, args.crop_detector, device,
                            angle_ckpt=args.crop_angle, rot_ckpt=args.rot_head)             # crop pass
    print(f"stage1: {args.stage1_detector}\ncrop det: {args.crop_detector}\ncrop angle: {args.crop_angle}\nrot: {args.rot_head}\nval: {args.val_dir}")

    # full-frame dataset (NO crop) -> we crop ourselves from detected bbox
    ds = PoseEstimationDataset(args.val_dir,
                               keypoint_names=['link0', 'link2', 'link3', 'link4', 'link6', 'link7', 'hand'],
                               image_size=(IS, IS), heatmap_size=(IS, IS),
                               augment=False, include_angles=True, sigma=2.5, crop_to_robot=False)
    if args.frac_range:
        lo, hi = args.frac_range; Nall = len(ds.samples)
        ds.samples = ds.samples[int(lo * Nall):int(hi * Nall)]
        print(f"frac-range {lo}-{hi}: {Nall} -> {len(ds.samples)} held-out frames")
    if args.max_frames and args.max_frames < len(ds):
        stride = max(1, len(ds.samples) // args.max_frames)
        ds.samples = ds.samples[::stride][:args.max_frames]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    raw_err = torch.zeros(6); ref_err = torch.zeros(6); n = 0
    adds = []
    for batch in tqdm(loader, desc="selfbbox eval"):
        img = batch['image'].to(device)
        gt = batch['angles'].to(device)[:, :6]
        gt3d = batch['keypoints_3d'].to(device)
        K = scale_K(batch['camera_K'], batch['original_size'], IS).to(device)
        bidx = torch.arange(img.shape[0], device=device).view(-1, 1).float()
        with torch.no_grad():
            # PASS 1: full-frame detector -> bbox  (or GT keypoints if --oracle-bbox)
            if args.oracle_bbox:
                gkp = batch['keypoints'].to(device).float()        # (B,7,2) in IS space
                gconf = batch['valid_mask'].to(device).float()     # 1 where keypoint valid
                boxes = detected_bbox(gkp, gconf, IS, args.margin, 0.5)
            elif args.bbox_from_solved:
                # PASS 1: full-frame detect + SOLVE -> project ALL 7 FK keypoints -> bbox.
                # The solved skeleton fills in the occluded base (link0) the raw detector drops,
                # since base position is fixed by the camera pose (R,t) even when base-yaw is ambiguous.
                o1 = det1(img, K)
                R1 = o1.get('rot_matrix') if args.stage1_rot else None
                # solve_batch optimizes via internal autograd (loss.backward) -> must run with grad
                # enabled even though we're inside the no_grad eval block.
                with torch.enable_grad():
                    _, kp_cam1, reproj1 = solve_batch(o1['keypoints_2d'], o1['confidence'], K, fix_joint7=True,
                                                      iters=args.iters, lr=2e-2, img_size=IS, device=device,
                                                      prior_w=0.0, theta_init=o1['joint_angles'],
                                                      conf_gate=args.conf_gate, R_init=R1)
                kp_cam1 = kp_cam1.detach(); reproj1 = reproj1.detach()
                uv = project_points(kp_cam1, K)                      # (B,7,2) all points, no conf drop
                if args.bbox_union:
                    # UNION: solved 7pts (always in, fills occluded base) + confidently-detected kp
                    # (direct observations, gated by conf). Counters solver-bias mis-cropping: a slightly
                    # biased solve no longer shrinks/shifts the crop off a keypoint the detector saw clearly.
                    upts = torch.cat([uv, o1['keypoints_2d']], dim=1)                       # (B,14,2)
                    uconf = torch.cat([torch.ones(uv.shape[:2], device=device),
                                       o1['confidence']], dim=1)                            # solved=1, det=conf
                    boxes = detected_bbox(upts, uconf, IS, args.margin, args.bbox_conf)
                else:
                    boxes = detected_bbox(uv, torch.ones(uv.shape[:2], device=device), IS, args.margin, 0.0)
                if args.bbox_guard:
                    # DIVERGENCE GUARD: a diverged pass-1 solve projects to a GARBAGE skeleton (points
                    # off-frame / NaN / absurd spread) -> garbage bbox -> blown-up crop. Detect that
                    # GEOMETRICALLY (NOT via reproj: realsense full-frame solves have naturally high
                    # reproj on foreshortened frames, so a reproj gate nukes good frames too) and fall
                    # back to the detected-kp bbox for those frames; clamp every bbox for safety.
                    det_boxes = detected_bbox(o1['keypoints_2d'], o1['confidence'], IS, args.margin, args.bbox_conf)
                    off = ((uv < -IS) | (uv > 2 * IS)).any(dim=2).any(dim=1)    # any point wildly off-frame
                    span = (uv.amax(dim=1) - uv.amin(dim=1)).amax(dim=1)        # bbox side in px
                    huge = span > 3.0 * IS                                      # absurd spread
                    bad = off | huge | torch.isnan(uv).any(dim=2).any(dim=1)
                    if args.bbox_reproj_thr < 1e6:                              # optional extra reproj gate (off by default)
                        bad = bad | (reproj1 > args.bbox_reproj_thr)
                    boxes = torch.where(bad.unsqueeze(1), det_boxes, boxes)
                    n_fb = int(bad.sum())
                    if n_fb:
                        globals().setdefault('_GUARD_FB', [0])[0] += n_fb
                boxes = clamp_boxes(boxes, IS)
            else:
                o1 = det1(img, K)
                boxes = detected_bbox(o1['keypoints_2d'], o1['confidence'], IS, args.margin, args.bbox_conf)
            # ITERATIVE coarse-to-fine: re-detect on the crop with the crop detector -> tighter bbox
            for _ in range(args.bbox_refine_iters):
                rois = torch.cat([bidx, boxes], dim=1)
                ci = roi_align(img, rois, output_size=(IS, IS), spatial_scale=1.0, aligned=True)
                oc = cropm(ci, crop_K(K, boxes, float(IS)))
                kpc = oc['keypoints_2d']                            # (B,7,2) in crop-IS space
                # map crop-space kp -> full-frame IS space, recompute bbox there
                side = (boxes[:, 2] - boxes[:, 0]).clamp(min=1.0).view(-1, 1)
                kpf = kpc * (side / IS).unsqueeze(-1) + boxes[:, :2].unsqueeze(1)
                boxes = detected_bbox(kpf, oc['confidence'], IS, args.margin, args.bbox_conf)
            rois = torch.cat([bidx, boxes], dim=1)
            crop_img = roi_align(img, rois, output_size=(IS, IS), spatial_scale=1.0, aligned=True)
            Kc = crop_K(K, boxes, float(IS))
            # FINAL PASS: crop pipeline on the refined crop
            o2 = cropm(crop_img, Kc)
        init_ang = o2['joint_angles']; kp2d = o2['keypoints_2d']; conf = o2['confidence']
        R_init = o2.get('rot_matrix') if args.rot_head else None
        refined, kp_cam, reproj2 = solve_batch(kp2d, conf, Kc, fix_joint7=True, iters=args.iters,
                                         lr=2e-2, img_size=IS, device=device, prior_w=0.0,
                                         theta_init=init_ang, conf_gate=args.conf_gate, R_init=R_init)
        raw_err += wrapped_abs_deg(init_ang[:, :6], gt).sum(0).cpu()
        ref_err += wrapped_abs_deg(refined[:, :6], gt).sum(0).cpu()
        valid = (gt3d.abs().sum(-1) > 0)
        per_j = (kp_cam - gt3d).norm(dim=-1)
        if _DUMP is not None:
            names = batch['name']; th = refined.detach().cpu().numpy(); kc = kp_cam.detach().cpu().numpy()
            g3 = gt3d.detach().cpu().numpy(); fv = valid.detach().cpu().numpy()
            ft = o2['global_feat'].detach().cpu().numpy(); rp = reproj2.detach().cpu().numpy()
            for b in range(img.shape[0]):
                _DUMP['fid'].append(names[b]); _DUMP['theta'].append(th[b]); _DUMP['kp_cam'].append(kc[b])
                _DUMP['gt3d'].append(g3[b]); _DUMP['found'].append(fv[b])
                _DUMP['feat'].append(ft[b]); _DUMP['reproj'].append(float(rp[b]))
        for b in range(img.shape[0]):
            if valid[b].any():
                adds.append(float(per_j[b][valid[b]].mean().item()))
        n += img.shape[0]

    if _DUMP is not None:
        import numpy as _np
        _np.savez(args.dump_npz, fid=_np.array(_DUMP['fid']), theta=_np.array(_DUMP['theta']),
                  kp_cam=_np.array(_DUMP['kp_cam']), gt3d=_np.array(_DUMP['gt3d']), found=_np.array(_DUMP['found']),
                  feat=_np.array(_DUMP['feat']), reproj=_np.array(_DUMP['reproj']))
        print(f"[dump] {len(_DUMP['fid'])} frames -> {args.dump_npz}", flush=True)
    raw = (raw_err / n).numpy(); ref = (ref_err / n).numpy(); adds = np.array(adds)
    print(f"\n{'='*54}\n  SELF-BBOX CROP  ({n} frames)  {os.path.basename(args.val_dir)}\n{'='*54}")
    print(f"  {'joint':<6}{'raw MLP':>10}{'refined':>10}{'delta':>9}")
    for j in range(6):
        print(f"  J{j:<5}{raw[j]:>10.2f}{ref[j]:>10.2f}{ref[j]-raw[j]:>+9.2f}")
    print('-'*54)
    if len(adds):
        print(f"  [Pose] ADD-AUC@100mm: {add_auc(adds):.4f} | mean ADD {adds.mean()*1000:.1f}mm | "
              f"median {np.median(adds)*1000:.1f}mm ({len(adds)} frames)")
    if args.bbox_guard:
        print(f"  [guard] bbox fell back to detected on {globals().get('_GUARD_FB', [0])[0]} frames")
    print('='*54)


if __name__ == '__main__':
    main()
