"""
Train a camera-rotation head (robot->camera 6D) on a frozen Stage-1 detector.

WHY: the residual realsense ADD gap is the kinematic solver landing in a wrong ROTATION basin
(rot_err 47 deg even with good 2D); 2D reprojection is degenerate at distance. depth_diag +
rinit_probe showed an oracle ROTATION init recovers +0.11 realsense ADD-AUC. This head predicts
that rotation from DINOv3 APPEARANCE (which way the robot faces) + keypoint geometry, to seed the
solver's R_init. Only rot_head trains; backbone + keypoint head frozen. GT rotation =
Kabsch(FK(gt_angles) -> keypoints_3d) per frame. Loss = Frobenius matrix error; val = geodesic deg.
"""
import argparse, math, os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_angle import AnglePredictor, kabsch_batch
from model_v4 import panda_forward_kinematics
from dataset import PoseEstimationDataset

try:
    import wandb; _HAS_WANDB = True
except Exception:
    _HAS_WANDB = False


def scale_K(camera_K, original_size, hm):
    K = camera_K.clone().float()
    for b in range(K.shape[0]):
        ow, oh = float(original_size[b][0]), float(original_size[b][1])
        K[b, 0, 0] *= hm / ow; K[b, 1, 1] *= hm / oh
        K[b, 0, 2] *= hm / ow; K[b, 1, 2] *= hm / oh
    return K


def geodesic_deg(Rp, Rg):
    """(B,3,3),(B,3,3) -> (B,) geodesic angle in deg."""
    c = ((Rp.transpose(1, 2) @ Rg).diagonal(dim1=1, dim2=2).sum(-1) - 1) / 2
    return torch.acos(c.clamp(-1 + 1e-6, 1 - 1e-6)) * 180 / math.pi


def gt_pose(gt_angles, kp3d, valid):
    """GT robot->camera (R,t) via Kabsch. gt_angles (B,7), kp3d (B,7,3), valid (B,7)."""
    ga = gt_angles.clone(); ga[:, 6] = 0.0
    fk = panda_forward_kinematics(ga)               # (B,7,3) robot frame
    return kabsch_batch(fk, kp3d, valid.float())    # (B,3,3),(B,3)


def main(args):
    device = torch.device('cuda'); assert torch.cuda.is_available()
    kp_names = ['link0', 'link2', 'link3', 'link4', 'link6', 'link7', 'hand']
    mk = lambda d, aug: PoseEstimationDataset(
        data_dir=d, keypoint_names=kp_names, image_size=(args.image_size, args.image_size),
        heatmap_size=(args.image_size, args.image_size), augment=aug, aug_level='strong',
        include_angles=True, sigma=2.5,
        crop_to_robot=args.crop_to_robot, crop_margin=args.crop_margin)
    train_ds, val_ds = mk(args.train_dir, True), mk(args.val_dir, False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = AnglePredictor(args.model_name, args.image_size, fix_joint7_zero=True,
                           head_type='mlp', with_rotation=True, with_translation=True).to(device)
    ckpt = torch.load(args.detector_ckpt, map_location=device)
    ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    msd = model.state_dict()
    model.load_state_dict({k: v for k, v in ckpt.items() if k in msd and v.shape == msd[k].shape}, strict=False)
    model.freeze_detector()
    print("==> training rotation head only")

    opt = optim.AdamW(model.rot_head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.min_lr)
    if args.use_wandb and _HAS_WANDB:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    best = 1e9
    for epoch in range(args.epochs):
        model.rot_head.train()
        run = 0.0
        for batch in tqdm(train_loader, desc=f"Ep{epoch}"):
            imgs = batch['image'].to(device)
            gt_ang = batch['angles'].to(device)
            kp3d = batch['keypoints_3d'].to(device)
            K = scale_K(batch['camera_K'], batch['original_size'], args.image_size).to(device)
            valid = (kp3d.abs().sum(-1) > 1e-6) & (kp3d[..., 2] > 0)     # (B,7)
            keep = valid.sum(1) >= 4
            if keep.sum() == 0:
                continue
            Rg, tg = gt_pose(gt_ang, kp3d, valid)
            o = model(imgs, K)
            r_loss = ((o['rot_matrix'][keep] - Rg[keep]) ** 2).sum(dim=(1, 2)).mean()  # Frobenius^2
            t_loss = F.smooth_l1_loss(o['trans'][keep], tg[keep])                       # meters
            loss = r_loss + args.t_weight * t_loss
            opt.zero_grad(); loss.backward(); opt.step()
            run += loss.item()
        sched.step()

        model.rot_head.eval()
        gerrs, terrs = [], []
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch['image'].to(device)
                gt_ang = batch['angles'].to(device); kp3d = batch['keypoints_3d'].to(device)
                K = scale_K(batch['camera_K'], batch['original_size'], args.image_size).to(device)
                valid = (kp3d.abs().sum(-1) > 1e-6) & (kp3d[..., 2] > 0)
                keep = valid.sum(1) >= 4
                if keep.sum() == 0:
                    continue
                Rg, tg = gt_pose(gt_ang, kp3d, valid)
                o = model(imgs, K)
                gerrs.append(geodesic_deg(o['rot_matrix'][keep], Rg[keep]).cpu())
                terrs.append(((o['trans'][keep] - tg[keep]).norm(dim=-1) * 1000).cpu())  # mm
        ge = torch.cat(gerrs); te = torch.cat(terrs)
        med = ge.median().item(); mean = ge.mean().item()
        tz_med = te.median().item()
        print(f"Ep{epoch} | val geo med={med:.2f} mean={mean:.2f} deg | t-err med={tz_med:.1f}mm | "
              f"train_loss={run/max(1,len(train_loader)):.4f}")
        if args.use_wandb and _HAS_WANDB:
            wandb.log({'epoch': epoch, 'val_geo_median': med, 'val_geo_mean': mean,
                       'val_t_err_mm': tz_med, 'train_loss': run / max(1, len(train_loader)),
                       'lr': opt.param_groups[0]['lr']})
        torch.save(model.rot_head.state_dict(), out / 'last_rot_head.pth')
        # rank by combined pose error (geodesic deg + t-err in cm) so t actually matters
        score = med + tz_med / 10.0
        if score < best:
            best = score; torch.save(model.rot_head.state_dict(), out / 'best_rot_head.pth')
            print(f"  -> new best (geo {med:.2f}deg + t {tz_med:.1f}mm)")
    print(f"Done. Best pose score = {best:.2f}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--detector-ckpt', required=True)
    p.add_argument('--train-dir', required=True); p.add_argument('--val-dir', required=True)
    p.add_argument('--output-dir', default='./outputs_rotation')
    p.add_argument('--model-name', default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    p.add_argument('--image-size', type=int, default=512); p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--epochs', type=int, default=40); p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--min-lr', type=float, default=1e-6); p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--t-weight', type=float, default=1.0, help='weight on the translation SmoothL1 loss')
    p.add_argument('--crop-to-robot', action='store_true', help='robot-bbox crop (match a crop-trained detector)')
    p.add_argument('--crop-margin', type=float, default=1.5)
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--use-wandb', action='store_true')
    p.add_argument('--wandb-project', default='dinov3-rotation'); p.add_argument('--wandb-run-name', default=None)
    main(p.parse_args())
