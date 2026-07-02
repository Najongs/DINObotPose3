"""Stage 1 (dino env): run OUR pipeline on DREAM realsense, dump per-frame estimate for cross-env
render-compare refinement (Stage 2 in ctrnetx). Saves theta, solved camera-frame keypoints, GT 3D kp,
found mask, and frame id. Keyed by frame number so Stage 2 can load the matching original DREAM image."""
import argparse, glob, os, sys
import numpy as np
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../TRAIN')))
sys.path.append(os.path.dirname(__file__))
from model_angle import AnglePredictor
from solve_pose_kinematic import solve_batch
from viz_hypothesis import load_frame

ap = argparse.ArgumentParser()
ap.add_argument('--detector', required=True); ap.add_argument('--mlp-head', required=True)
ap.add_argument('--rot-head', default=None)
ap.add_argument('--val-dir', required=True)
ap.add_argument('--model-name', default='facebook/dinov3-vitb16-pretrain-lvd1689m')
ap.add_argument('--image-size', type=int, default=512); ap.add_argument('--batch-size', type=int, default=16)
ap.add_argument('--max-frames', type=int, default=300); ap.add_argument('--iters', type=int, default=200)
ap.add_argument('--out', required=True)
a = ap.parse_args()
device = torch.device('cuda'); S = a.image_size

m = AnglePredictor(a.model_name, S, head_type='mlp', with_rotation=a.rot_head is not None,
                   with_translation=a.rot_head is not None).to(device).eval()
sd = torch.load(a.detector, map_location=device); sd = {k.replace('module.', ''): v for k, v in sd.items()}
m.load_state_dict({k: v for k, v in sd.items() if k in m.state_dict() and v.shape == m.state_dict()[k].shape}, strict=False)
m.angle_head.load_state_dict(torch.load(a.mlp_head, map_location=device))
if a.rot_head: m.rot_head.load_state_dict(torch.load(a.rot_head, map_location=device))

files = sorted(glob.glob(os.path.join(a.val_dir, '*.json')))
if a.max_frames and a.max_frames < len(files):
    st = max(1, len(files) // a.max_frames); files = files[::st][:a.max_frames]

FID, THETA, KPCAM, GT3D, FOUND = [], [], [], [], []
buf, ids = [], []
def flush():
    if not buf: return
    imgs = torch.stack([b['img'] for b in buf]).to(device)
    K = torch.stack([b['K'] for b in buf]).to(device)
    with torch.no_grad(): o = m(imgs, K)
    Ri = o.get('rot_matrix') if a.rot_head else None
    theta, kp_cam, _ = solve_batch(o['keypoints_2d'], o['confidence'], K, fix_joint7=True, iters=a.iters,
                                   lr=2e-2, img_size=S, device=device, prior_w=0.0,
                                   theta_init=o['joint_angles'], R_init=Ri)
    th = theta.cpu().numpy(); kc = kp_cam.cpu().numpy()
    for i, b in enumerate(buf):
        FID.append(ids[i]); THETA.append(th[i]); KPCAM.append(kc[i])
        GT3D.append(b['kp3d']); FOUND.append(b['found'])
    buf.clear(); ids.clear()
for jf in files:
    fr = load_frame(jf, a.val_dir, S)
    if fr is None: continue
    buf.append(fr); ids.append(os.path.basename(jf).replace('.json', ''))
    if len(buf) >= a.batch_size: flush()
flush()

np.savez(a.out, fid=np.array(FID), theta=np.array(THETA), kp_cam=np.array(KPCAM),
         gt3d=np.array(GT3D), found=np.array(FOUND))
# baseline ADD of our pose (sanity)
kc = np.array(KPCAM); g = np.array(GT3D); f = np.array(FOUND) > 0
err = np.array([np.linalg.norm(kc[i] - g[i], axis=1)[f[i]].mean() for i in range(len(kc))])
err.sort(); auc = np.mean([np.sum(err < i / 10000.0) / len(err) for i in range(1000)])
print(f"dumped {len(FID)} frames -> {a.out}", flush=True)
print(f"OUR baseline ADD-AUC@100mm = {auc:.4f}  mean {err.mean()*1000:.1f}mm", flush=True)
