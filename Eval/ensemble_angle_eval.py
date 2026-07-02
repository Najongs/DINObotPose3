"""
Ensemble angle eval: MLP angle head vs Diffusion angle head (same DINOv3 detector),
and a per-joint routing ensemble. Validates the EXPERIMENTS.md hypothesis that MLP wins the
body joints (J0-J3) and Diffusion wins the ambiguous wrist (J4/J5).

Ensemble = for each joint, route to whichever head has lower val MAE (routing decided on this
same val set — a mild upper bound; a held-out split would be stricter). Reports MLP, Diffusion,
and routed-ensemble per-joint + mean angle MAE (deg).
"""
import argparse, glob, math, os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader

TRAIN = os.path.abspath(os.path.join(os.path.dirname(__file__), '../TRAIN'))
sys.path.append(TRAIN)
from model_angle import AnglePredictor
from model_diffusion import DINOv3DiffusionPoseEstimator
from dataset import PoseEstimationDataset

PANDA_JOINT_MEAN = torch.tensor([-5.22e-02, 2.68e-01, 6.04e-03, -2.01e+00, 1.49e-02, 1.99e+00, 0.0])
PANDA_JOINT_STD = torch.tensor([1.025, 0.645, 0.511, 0.508, 0.769, 0.511, 1.0])


def wrapped_abs_deg(pred, gt):
    d = torch.atan2(torch.sin(pred - gt), torch.cos(pred - gt)).abs()
    return d * 180.0 / math.pi


def scale_K(camera_K, original_size, hm):
    K = camera_K.clone().float()
    for b in range(K.shape[0]):
        ow, oh = float(original_size[b][0]), float(original_size[b][1])
        sx, sy = hm / ow, hm / oh
        K[b, 0, 0] *= sx; K[b, 1, 1] *= sy
        K[b, 0, 2] *= sx; K[b, 1, 2] *= sy
    return K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--detector', default=None, help='detector best_heatmap.pth (auto: latest stage1_unfrozen)')
    ap.add_argument('--mlp-head', default=None, help='best_angle_head.pth (auto: latest outputs_angle/angle_*)')
    ap.add_argument('--diffusion', default=None, help='best_diffusion.pth (auto: latest diffusion3_*)')
    ap.add_argument('--val-dir', default='../Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_test_dr')
    ap.add_argument('--model-name', default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    ap.add_argument('--image-size', type=int, default=512)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--max-frames', type=int, default=2000)
    args = ap.parse_args()

    def latest(pat):
        f = sorted(glob.glob(pat), key=os.path.getmtime)
        return f[-1] if f else None
    det = args.detector or latest(f'{TRAIN}/outputs_heatmap/stage1_unfrozen_*/best_heatmap.pth')
    mlp_h = args.mlp_head or latest(f'{TRAIN}/outputs_angle/angle_*/best_angle_head.pth')
    diff = args.diffusion or latest(f'{TRAIN}/outputs_diffusion/diffusion3_*/best_diffusion.pth')
    print(f"detector : {det}\nmlp head : {mlp_h}\ndiffusion: {diff}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert torch.cuda.is_available(), "need GPU (select by UUID)"

    # --- MLP model ---
    mlp = AnglePredictor(args.model_name, args.image_size, head_type='mlp').to(device).eval()
    sd = torch.load(det, map_location=device); sd = {k.replace('module.', ''): v for k, v in sd.items()}
    mlp.load_state_dict({k: v for k, v in sd.items() if k in mlp.state_dict()
                         and v.shape == mlp.state_dict()[k].shape}, strict=False)
    mlp.angle_head.load_state_dict(torch.load(mlp_h, map_location=device))

    # --- Diffusion model ---
    dif = DINOv3DiffusionPoseEstimator(args.model_name, (args.image_size, args.image_size),
                                       unfreeze_blocks=0).to(device).eval()
    dck = torch.load(diff, map_location=device)['model']
    dck = {k.replace('module.', ''): v for k, v in dck.items()}
    dif.load_state_dict({k: v for k, v in dck.items() if k in dif.state_dict()
                         and v.shape == dif.state_dict()[k].shape}, strict=False)

    mean = PANDA_JOINT_MEAN.to(device); std = PANDA_JOINT_STD.to(device)

    ds = PoseEstimationDataset(args.val_dir, keypoint_names=['link0','link2','link3','link4','link6','link7','hand'],
                               image_size=(args.image_size, args.image_size),
                               heatmap_size=(args.image_size, args.image_size),
                               augment=False, include_angles=True, sigma=2.5)
    if args.max_frames and args.max_frames < len(ds):
        ds.samples = ds.samples[:args.max_frames]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    mlp_err = torch.zeros(6); dif_err = torch.zeros(6); n = 0
    from tqdm import tqdm
    with torch.no_grad():
        for batch in tqdm(loader, desc="ensemble eval"):
            img = batch['image'].to(device)
            gt = batch['angles'].to(device)[:, :6]
            K = scale_K(batch['camera_K'], batch['original_size'], args.image_size).to(device)
            a_mlp = mlp(img, K)['joint_angles'][:, :6]
            a_dif_norm = dif(img, training=False)['joint_angles'][:, :6]
            a_dif = a_dif_norm * std[:6] + mean[:6]
            mlp_err += wrapped_abs_deg(a_mlp, gt).sum(0).cpu()
            dif_err += wrapped_abs_deg(a_dif, gt).sum(0).cpu()
            n += img.shape[0]

    mlp_mae = (mlp_err / n).numpy(); dif_mae = (dif_err / n).numpy()
    route = np.where(mlp_mae <= dif_mae, 0, 1)  # 0=mlp, 1=diff per joint
    ens_mae = np.where(route == 0, mlp_mae, dif_mae)

    print(f"\n{'='*60}\n  ENSEMBLE ANGLE EVAL  ({n} frames)\n{'='*60}")
    print(f"  {'joint':<6}{'MLP':>8}{'Diff':>8}{'route':>8}")
    for j in range(6):
        print(f"  J{j:<5}{mlp_mae[j]:>8.2f}{dif_mae[j]:>8.2f}{'MLP' if route[j]==0 else 'DIFF':>8}")
    print(f"  {'-'*30}")
    print(f"  MEAN  {mlp_mae.mean():>8.2f}{dif_mae.mean():>8.2f}{ens_mae.mean():>8.2f}  <- ensemble")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
