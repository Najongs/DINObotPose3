"""
Decide whether occlusion-aware DETECTOR retraining is worth it.

Sim train data has 0% in-frame object occlusion (object visibility always 1.0); it only
has off-frame occlusion (22% of frames). So the detector has NEVER seen an in-frame
keypoint hidden by an object. Hypothesis: under in-frame occlusion it outputs a
CONFIDENT-WRONG peak (high conf), which the conf-gate cannot catch -> catastrophe.

Test (no training): take clean real frames, paste a gray box over ONE in-frame keypoint,
re-run the detector, and compare that keypoint's confidence + 2D error clean-vs-occluded.
  conf DROPS a lot  -> detector already generalizes; retraining adds little.
  conf STAYS high + 2D err jumps -> confident-wrong = the gap -> occlusion aug is worth it.
"""
import argparse, glob, os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

TRAIN = os.path.abspath(os.path.join(os.path.dirname(__file__), '../TRAIN'))
sys.path.append(TRAIN); sys.path.append(os.path.dirname(__file__))
from model_angle import AnglePredictor  # noqa
from dataset import PoseEstimationDataset
from refine_eval import scale_K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--detector', required=True)
    ap.add_argument('--mlp-head', required=True)
    ap.add_argument('--val-dir', required=True)
    ap.add_argument('--model-name', default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    ap.add_argument('--image-size', type=int, default=512)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--max-frames', type=int, default=300)
    ap.add_argument('--box', type=int, default=48, help='occluder half-size in px @ image-size res')
    ap.add_argument('--texture', default='gray', choices=['gray', 'noise'],
                    help='occluder fill: flat gray vs random-noise texture (proxy for real objects)')
    args = ap.parse_args()

    device = torch.device('cuda'); assert torch.cuda.is_available()
    m = AnglePredictor(args.model_name, args.image_size, head_type='mlp').to(device).eval()
    sd = torch.load(args.detector, map_location=device); sd = {k.replace('module.', ''): v for k, v in sd.items()}
    m.load_state_dict({k: v for k, v in sd.items() if k in m.state_dict()
                       and v.shape == m.state_dict()[k].shape}, strict=False)
    m.angle_head.load_state_dict(torch.load(args.mlp_head, map_location=device))

    S = args.image_size
    ds = PoseEstimationDataset(args.val_dir, keypoint_names=['link0','link2','link3','link4','link6','link7','hand'],
                               image_size=(S, S), heatmap_size=(S, S),
                               augment=False, include_angles=True, sigma=2.5)
    if args.max_frames and args.max_frames < len(ds):
        ds.samples = ds.samples[:args.max_frames]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # ImageNet de/normalize so we can paint a neutral gray box in pixel space
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    gray = ((0.5 - mean) / std)  # normalized value of mid-gray

    b = args.box
    clean_conf = []; occ_conf = []; occ_err = []; clean_err = []
    for batch in tqdm(loader, desc="inframe occ probe"):
        img = batch['image'].to(device)
        gtkp = batch['keypoints'].to(device)        # (B,7,2) px @ S
        K = scale_K(batch['camera_K'], batch['original_size'], S).to(device)
        B = img.shape[0]
        inb = ((gtkp[..., 0] >= b) & (gtkp[..., 0] < S - b) &
               (gtkp[..., 1] >= b) & (gtkp[..., 1] < S - b))   # keypoints we can fully box
        with torch.no_grad():
            out0 = m(img, K)
        conf0 = out0['confidence']; kp0 = out0['keypoints_2d']
        # occlude ONE in-frame keypoint per frame (the first boxable one), measure that kp only
        for bi in range(B):
            ks = torch.where(inb[bi])[0]
            if len(ks) == 0:
                continue
            j = int(ks[len(ks) // 2])               # a mid-chain visible keypoint
            u, v = int(gtkp[bi, j, 0]), int(gtkp[bi, j, 1])
            occ = img[bi:bi+1].clone()
            y0, y1, x0, x1 = max(0, v-b), v+b, max(0, u-b), u+b
            if args.texture == 'noise':
                patch = torch.randn(1, 3, y1-y0, x1-x0, device=device) * 1.0  # textured occluder
                occ[:, :, y0:y1, x0:x1] = patch
            else:
                occ[:, :, y0:y1, x0:x1] = gray
            with torch.no_grad():
                o = m(occ, K[bi:bi+1])
            clean_conf.append(float(conf0[bi, j])); occ_conf.append(float(o['confidence'][0, j]))
            clean_err.append(float((kp0[bi, j] - gtkp[bi, j]).norm()))
            occ_err.append(float((o['keypoints_2d'][0, j] - gtkp[bi, j]).norm()))

    cc = np.array(clean_conf); oc = np.array(occ_conf)
    ce = np.array(clean_err); oe = np.array(occ_err)
    print(f"\n{'='*60}\nIN-FRAME OCCLUSION PROBE  {os.path.basename(args.val_dir)}  n={len(cc)} kp\n{'='*60}")
    print(f"  box={2*args.box}px over one in-frame keypoint")
    print(f"  confidence  clean {cc.mean():.3f}  ->  occluded {oc.mean():.3f}   (drop {100*(1-oc.mean()/max(cc.mean(),1e-6)):.0f}%)")
    print(f"  2D px err   clean {ce.mean():5.1f}  ->  occluded {oe.mean():5.1f}")
    # confident-wrong = occluded conf still above gate AND error large
    cw = (oc >= 0.05) & (oe > 16)
    print(f"  CONFIDENT-WRONG (occ conf>=0.05 AND err>16px): {cw.mean()*100:.0f}% of occluded kp")
    print(f"  -> gate-catchable (occ conf<0.05): {(oc<0.05).mean()*100:.0f}%")
    print('='*60)


if __name__ == '__main__':
    main()
