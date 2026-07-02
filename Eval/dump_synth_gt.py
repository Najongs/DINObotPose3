"""Dump clean synth GT (fid, GT joint angles, gt3d camera-frame keypoints) from OUR converted dataset
for the Phase D refiner synth-training (the CtRNet NDDS 'location' is broken — 31m Kabsch residual).
ctrnetx then computes the GT pose via BPnP(project(gt3d), ctrnet_FK(angles)[SEL], K) — the same
self-consistent path the real in-domain proof used — and loads the synth image by fid from DREAM_syn."""
import argparse, os, sys
import numpy as np
from torch.utils.data import DataLoader

TRAIN = os.path.abspath(os.path.join(os.path.dirname(__file__), '../TRAIN'))
sys.path.append(TRAIN)
from dataset import PoseEstimationDataset

ap = argparse.ArgumentParser()
ap.add_argument('--val-dir', required=True)
ap.add_argument('--max-frames', type=int, default=12000)
ap.add_argument('--out', required=True)
a = ap.parse_args()

ds = PoseEstimationDataset(a.val_dir, keypoint_names=['link0', 'link2', 'link3', 'link4', 'link6', 'link7', 'hand'],
                           image_size=(512, 512), heatmap_size=(512, 512), augment=False,
                           include_angles=True, sigma=2.5, crop_to_robot=False)
if a.max_frames < len(ds):
    stride = max(1, len(ds.samples) // a.max_frames); ds.samples = ds.samples[::stride][:a.max_frames]
loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=8)
FID, ANG, GT3D = [], [], []
for batch in loader:
    names = batch['name']; ang = batch['angles'].numpy(); g3 = batch['keypoints_3d'].numpy()
    for b in range(len(names)):
        FID.append(names[b]); ANG.append(ang[b]); GT3D.append(g3[b])
np.savez(a.out, fid=np.array(FID), angles=np.array(ANG), gt3d=np.array(GT3D))
print(f"dumped {len(FID)} frames -> {a.out}  (angles {np.array(ANG).shape}, gt3d {np.array(GT3D).shape})")
