"""
Does the self-training +0.11 fix the OCCLUSION TAIL or just the bulk? Bin the realsense held-out
(last 30%) by an occlusion proxy = per-frame mean keypoint confidence (occluded keypoints -> low
heatmap peak -> low conf). Report ADD-AUC per conf-quartile for baseline vs self-trained head.
If the gain concentrates in the LOW-conf (occluded) bins, self-training addresses the user's
"weak under occlusion"; if it's uniform/bulk, the occlusion tail remains (-> MCL + appearance selector).
"""
import argparse, os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '../TRAIN')); sys.path.append(HERE)
from selftrain_pseudo import build_model, run_pipeline, scale_K, add_auc, KP  # reuse
from dataset import PoseEstimationDataset


class A:  # mimic argparse for build_model
    pass


def per_frame(m, loader, device, IS, rot):
    confs, adds = [], []
    for batch in tqdm(loader, desc='eval', leave=False):
        refined, kp_cam, conf = run_pipeline(m, batch, device, IS, rot)
        gt3d = batch['keypoints_3d'].to(device); valid = (gt3d.abs().sum(-1) > 0)
        per_j = (kp_cam - gt3d).norm(dim=-1)
        for b in range(kp_cam.shape[0]):
            if valid[b].any():
                adds.append(float(per_j[b][valid[b]].mean().item()))
                confs.append(float(conf[b].mean().item()))
    return np.array(confs), np.array(adds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--real-dir', default='../Dataset/Converted_dataset/DREAM_real/panda-3cam_realsense')
    ap.add_argument('--detector', required=True)
    ap.add_argument('--base-head', required=True)
    ap.add_argument('--self-head', required=True)
    ap.add_argument('--rot-head', required=True)
    ap.add_argument('--model-name', default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    ap.add_argument('--image-size', type=int, default=512); ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--adapt-frac', type=float, default=0.7); ap.add_argument('--eval-frames', type=int, default=800)
    args = ap.parse_args()
    device = torch.device('cuda'); IS = args.image_size

    ds = PoseEstimationDataset(args.real_dir, keypoint_names=KP, image_size=(IS, IS),
                               heatmap_size=(IS, IS), augment=False, include_angles=True, sigma=2.5)
    N = len(ds.samples); cut = int(args.adapt_frac * N); idx = list(range(cut, N))
    es = max(1, len(idx) // args.eval_frames); idx = idx[::es][:args.eval_frames]
    loader = DataLoader(Subset(ds, idx), batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    def mk(head):
        a = A(); a.model_name = args.model_name; a.image_size = IS; a.detector = args.detector
        a.angle_head = head; a.rot_head = args.rot_head
        return build_model(a, device).eval()

    base = mk(args.base_head); conf_b, add_b = per_frame(base, loader, device, IS, True)
    del base; torch.cuda.empty_cache()
    self_m = mk(args.self_head); conf_s, add_s = per_frame(self_m, loader, device, IS, True)

    # bins by baseline conf quartiles (occlusion proxy; low conf = occluded/hard)
    qs = np.quantile(conf_b, [0.25, 0.5, 0.75])
    print(f"\nconf quartiles: {qs.round(3)}  (low conf = more occluded)")
    print(f"{'bin':<18}{'n':>5}{'base AUC':>10}{'self AUC':>10}{'Δ':>8}{'base mADD':>11}{'self mADD':>11}")
    edges = [-1, qs[0], qs[1], qs[2], 2.0]
    names = ['Q1 most-occluded', 'Q2', 'Q3', 'Q4 least-occluded']
    for i, nm in enumerate(names):
        sel = (conf_b > edges[i]) & (conf_b <= edges[i + 1])
        if sel.sum() == 0:
            continue
        ab, as_ = add_b[sel], add_s[sel]
        print(f"{nm:<18}{int(sel.sum()):>5}{add_auc(ab):>10.4f}{add_auc(as_):>10.4f}"
              f"{add_auc(as_)-add_auc(ab):>+8.4f}{ab.mean()*1000:>10.1f}m{as_.mean()*1000:>10.1f}m")
    print(f"{'ALL':<18}{len(add_b):>5}{add_auc(add_b):>10.4f}{add_auc(add_s):>10.4f}"
          f"{add_auc(add_s)-add_auc(add_b):>+8.4f}", flush=True)


if __name__ == '__main__':
    main()
