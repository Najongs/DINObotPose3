"""
Eval the multi-hypothesis (MCL) angle head: the kinematic solver SELECTS among the K hypotheses.
For each frame, run solve_batch from each hypothesis's theta_init (with the rot R_init), then pick
the hypothesis whose refined pose has the LOWEST reprojection error. Reports:
  - SELECTED  : solver-selected (deployable) ADD-AUC
  - ORACLE    : pick the hypothesis with min ADD-to-GT (upper bound if selection were perfect)
  - 1-HYP     : hypothesis 0 only (no benefit) for reference
If SELECTED >> 1-HYP, the multimodal head + solver-selection resolves the occlusion ambiguity.
"""
import argparse, glob, os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

TRAIN = os.path.abspath(os.path.join(os.path.dirname(__file__), '../TRAIN'))
sys.path.append(TRAIN); sys.path.append(os.path.dirname(__file__))
from model_angle import AnglePredictor                       # noqa
from dataset import PoseEstimationDataset                    # noqa
from solve_pose_kinematic import solve_batch                 # noqa
from refine_eval import add_auc, scale_K                     # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--detector', required=True)
    ap.add_argument('--mcl-head', required=True)
    ap.add_argument('--rot-head', default=None)
    ap.add_argument('--val-dir', required=True)
    ap.add_argument('--model-name', default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    ap.add_argument('--image-size', type=int, default=512)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--max-frames', type=int, default=1000)
    ap.add_argument('--iters', type=int, default=200)
    ap.add_argument('--conf-gate', type=float, default=0.05)
    ap.add_argument('--n-hyp', type=int, default=4)
    ap.add_argument('--crop', action='store_true'); ap.add_argument('--crop-margin', type=float, default=1.5)
    args = ap.parse_args()

    device = torch.device('cuda'); assert torch.cuda.is_available()
    m = AnglePredictor(args.model_name, args.image_size, head_type='mlp_mcl', n_hyp=args.n_hyp,
                       with_rotation=args.rot_head is not None,
                       with_translation=args.rot_head is not None).to(device).eval()
    sd = torch.load(args.detector, map_location=device); sd = {k.replace('module.', ''): v for k, v in sd.items()}
    msd = m.state_dict()
    m.load_state_dict({k: v for k, v in sd.items() if k in msd and v.shape == msd[k].shape}, strict=False)
    m.angle_head.load_state_dict(torch.load(args.mcl_head, map_location=device))
    if args.rot_head:
        m.rot_head.load_state_dict(torch.load(args.rot_head, map_location=device))
    print(f"detector {args.detector}\nmcl {args.mcl_head}\nrot {args.rot_head}\nval {args.val_dir}")

    ds = PoseEstimationDataset(args.val_dir, keypoint_names=['link0', 'link2', 'link3', 'link4', 'link6', 'link7', 'hand'],
                               image_size=(args.image_size, args.image_size),
                               heatmap_size=(args.image_size, args.image_size),
                               augment=False, include_angles=True, sigma=2.5,
                               crop_to_robot=args.crop, crop_margin=args.crop_margin)
    if args.max_frames and args.max_frames < len(ds):
        stride = max(1, len(ds.samples) // args.max_frames)
        ds.samples = ds.samples[::stride][:args.max_frames]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    add_sel, add_oracle, add_1hyp = [], [], []
    for batch in tqdm(loader, desc="mcl eval"):
        img = batch['image'].to(device)
        gt3d = batch['keypoints_3d'].to(device)
        K = scale_K(batch['camera_K'], batch['original_size'], args.image_size).to(device)
        with torch.no_grad():
            out = m(img, K)
        ja = out['joint_angles']                              # (B,K,7)
        kp2d = out['keypoints_2d']; conf = out['confidence']
        R_init = out.get('rot_matrix') if args.rot_head else None
        B, Kn = ja.shape[0], ja.shape[1]
        kpcam_k = torch.zeros(B, Kn, 7, 3, device=device)
        reproj_k = torch.zeros(B, Kn, device=device)
        for k in range(Kn):
            refined, kp_cam, reproj = solve_batch(kp2d, conf, K, fix_joint7=True, iters=args.iters,
                                                  lr=2e-2, img_size=args.image_size, device=device,
                                                  prior_w=0.0, theta_init=ja[:, k], conf_gate=args.conf_gate,
                                                  R_init=R_init)
            kpcam_k[:, k] = kp_cam; reproj_k[:, k] = reproj
        valid = (gt3d.abs().sum(-1) > 0)
        perj = (kpcam_k - gt3d.unsqueeze(1)).norm(dim=-1)     # (B,K,7)
        frame_add = torch.stack([perj[b, :, valid[b]].mean(dim=1) if valid[b].any()
                                 else torch.zeros(Kn, device=device) for b in range(B)])  # (B,K)
        sel = reproj_k.argmin(dim=1)                          # solver-selected
        ora = frame_add.argmin(dim=1)                         # oracle (min ADD to GT)
        for b in range(B):
            if valid[b].any():
                add_sel.append(float(frame_add[b, sel[b]].item()))
                add_oracle.append(float(frame_add[b, ora[b]].item()))
                add_1hyp.append(float(frame_add[b, 0].item()))

    add_1hyp = np.array(add_1hyp); add_sel = np.array(add_sel); add_oracle = np.array(add_oracle)
    print(f"\n{'='*56}\n  MCL (K={args.n_hyp})  {os.path.basename(args.val_dir)}  ({len(add_sel)} frames)\n{'='*56}")
    print(f"  1-HYP    ADD-AUC={add_auc(add_1hyp):.4f} | mean ADD {np.mean(add_1hyp)*1000:.1f}mm")
    print(f"  SELECTED ADD-AUC={add_auc(add_sel):.4f} | mean ADD {np.mean(add_sel)*1000:.1f}mm  <- deployable")
    print(f"  ORACLE   ADD-AUC={add_auc(add_oracle):.4f} | mean ADD {np.mean(add_oracle)*1000:.1f}mm  <- ceiling")
    print('='*56, flush=True)


if __name__ == '__main__':
    main()
