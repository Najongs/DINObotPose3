"""
Train a MULTI-HYPOTHESIS (MCL) angle head on a frozen detector. Targets the occlusion tail:
p(angles|image) is multimodal under occlusion (esp. base-yaw J0), so a single regressor lands on
the wrong mean. The head emits K hypotheses; an MCL winner-take-all loss (only the best hypothesis
per frame is supervised) makes them specialize into the modes. At inference the kinematic solver
selects the hypothesis with the lowest reprojection (see Eval/mcl_eval.py).

Loss = min_k SmoothL1(sin_cos_k, gt) + eps*mean_k (keep all hypotheses alive) + fk on the winner.
Val = ORACLE min-over-K per-joint MAE (does SOME hypothesis cover the truth?) — the real test is the
solver-selected ADD in mcl_eval.py.
"""
import argparse, math, os
from pathlib import Path
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_angle import AnglePredictor
from model_v4 import panda_forward_kinematics
from dataset import PoseEstimationDataset
from train_angle import scale_K

try:
    import wandb; _HAS_WANDB = True
except Exception:
    _HAS_WANDB = False


def main(args):
    device = torch.device('cuda'); assert torch.cuda.is_available()
    kp = ['link0', 'link2', 'link3', 'link4', 'link6', 'link7', 'hand']
    mk = lambda d, aug: PoseEstimationDataset(d, keypoint_names=kp,
        image_size=(args.image_size, args.image_size), heatmap_size=(args.image_size, args.image_size),
        augment=aug, aug_level='strong', include_angles=True, sigma=2.5,
        crop_to_robot=args.crop_to_robot, crop_margin=args.crop_margin)
    train_ds, val_ds = mk(args.train_dir, True), mk(args.val_dir, False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = AnglePredictor(args.model_name, args.image_size, fix_joint7_zero=True,
                           head_type='mlp_mcl', n_hyp=args.n_hyp).to(device)
    ckpt = torch.load(args.detector_ckpt, map_location=device)
    ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    msd = model.state_dict()
    model.load_state_dict({k: v for k, v in ckpt.items() if k in msd and v.shape == msd[k].shape}, strict=False)
    # optionally warm-start the MCL head trunk from a trained single-hypothesis head (shared layers)
    if args.init_angle_head:
        sh = torch.load(args.init_angle_head, map_location=device)
        ah = model.angle_head.state_dict()
        model.angle_head.load_state_dict({k: v for k, v in sh.items()
                                          if k in ah and v.shape == ah[k].shape}, strict=False)
        print(f"warm-started MCL head from {args.init_angle_head} (shared trunk layers)")
    model.freeze_detector()
    print(f"==> MCL angle head, K={args.n_hyp}")

    opt = optim.AdamW(model.angle_head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.min_lr)
    if args.use_wandb and _HAS_WANDB:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    best = 1e9
    for epoch in range(args.epochs):
        model.angle_head.train()
        for batch in tqdm(train_loader, desc=f"Ep{epoch} [train]"):
            imgs = batch['image'].to(device)
            gt = batch['angles'].to(device).clone(); gt[:, 6] = 0.0
            has = batch['has_angles'].to(device).bool() if 'has_angles' in batch else torch.ones(len(imgs), dtype=torch.bool, device=device)
            K = scale_K(batch['camera_K'], batch['original_size'], args.image_size).to(device)
            od = model(imgs, K, kp_drop=args.kp_drop)
            sc = od['sin_cos']                                  # (B,K,6,2)
            gt6 = gt[:, :6]
            gt_sc = torch.stack([torch.sin(gt6), torch.cos(gt6)], dim=-1).unsqueeze(1)  # (B,1,6,2)
            per_hyp = F.smooth_l1_loss(sc, gt_sc.expand_as(sc), reduction='none').mean(dim=(2, 3))  # (B,K)
            win = per_hyp.min(dim=1)                            # winner per frame
            mcl = win.values                                   # (B,)
            # FK consistency on the winning hypothesis
            ja = od['joint_angles']                            # (B,K,7)
            win_ja = ja[torch.arange(ja.shape[0], device=device), win.indices]   # (B,7)
            fk = F.mse_loss(panda_forward_kinematics(win_ja)[has], panda_forward_kinematics(gt)[has])
            loss = mcl[has].mean() + args.eps_share * per_hyp[has].mean() + args.fk_weight * fk
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

        # val: ORACLE min-over-K per-joint MAE (does SOME hypothesis cover the truth?)
        model.angle_head.eval()
        errs = []
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch['image'].to(device)
                gt = batch['angles'].to(device).clone(); gt[:, 6] = 0.0
                has = batch['has_angles'].bool() if 'has_angles' in batch else torch.ones(len(imgs), dtype=torch.bool)
                K = scale_K(batch['camera_K'], batch['original_size'], args.image_size).to(device)
                ja = model(imgs, K)['joint_angles'][:, :, :6]   # (B,K,6)
                d = ja - gt[:, :6].unsqueeze(1)
                d = torch.atan2(torch.sin(d), torch.cos(d)).abs() * 180 / math.pi  # (B,K,6)
                # oracle: per-FRAME pick the hypothesis with min total error, then its per-joint errs
                tot = d.mean(dim=2)                              # (B,K)
                wi = tot.argmin(dim=1)
                best_d = d[torch.arange(d.shape[0]), wi]         # (B,6)
                errs.append(best_d[has.to(device)].cpu())
        errs = torch.cat(errs, 0); pj = errs.mean(0); mae = pj.mean().item()
        print(f"Ep{epoch} | val ORACLE-minK MAE(J0-5)={mae:.2f} deg | per-joint=" +
              ",".join(f"{v:.1f}" for v in pj), flush=True)
        if args.use_wandb and _HAS_WANDB:
            wandb.log({'epoch': epoch, 'val_oracle_mae': mae, **{f'J{j}': pj[j].item() for j in range(6)}})
        torch.save(model.angle_head.state_dict(), out / 'last_mcl_head.pth')
        if mae < best:
            best = mae; torch.save(model.angle_head.state_dict(), out / 'best_mcl_head.pth')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--detector-ckpt', required=True)
    p.add_argument('--init-angle-head', default=None, help='warm-start trunk from a single-hyp head')
    p.add_argument('--train-dir', required=True); p.add_argument('--val-dir', required=True)
    p.add_argument('--output-dir', default='./outputs_angle_mcl')
    p.add_argument('--model-name', default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    p.add_argument('--image-size', type=int, default=512); p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--epochs', type=int, default=40); p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--min-lr', type=float, default=1e-6); p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--fk-weight', type=float, default=10.0)
    p.add_argument('--n-hyp', type=int, default=4)
    p.add_argument('--eps-share', type=float, default=0.05, help='weight on mean-over-K (keeps all hypotheses alive)')
    p.add_argument('--kp-drop', type=float, default=0.0, help='per-keypoint occlusion-mask prob (induces multimodality)')
    p.add_argument('--crop-to-robot', action='store_true'); p.add_argument('--crop-margin', type=float, default=1.5)
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--use-wandb', action='store_true')
    p.add_argument('--wandb-project', default='dinov3-angle-predictor'); p.add_argument('--wandb-run-name', default=None)
    main(p.parse_args())
