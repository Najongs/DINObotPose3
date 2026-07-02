"""
Robot-silhouette MASK head for render-and-compare. Validated ([[render-compare-validated]]): a silhouette
IoU refine recovers ~the depth/scale ceiling (realsense +0.108 @224px) and is ROBUST to mask degradation
(+0.083 even with a sloppy mask) -> a lightweight DINOv3 mask head is enough; NO external segmenter.
Train: frozen DINOv3 backbone + 1-ch decoder -> sigmoid mask. Synth GT mask is rendered FREE on-the-fly by
the mesh renderer from GT angles + Kabsch(FK,kp3d) camera pose. Loss = BCE + Dice. (augment=False: 2D geo
aug would misalign the camera-frame-rendered mask; DR-synth already has appearance variation.)
"""
import argparse, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE); sys.path.append(os.path.join(HERE, '../Eval'))
from model_v4 import DINOv3Backbone, ViTKeypointHead, panda_forward_kinematics
from dataset import PoseEstimationDataset
from refine_eval import scale_K
from silhouette_mesh_probe import (load_obj_verts, robot_pointcloud, render_mesh, kabsch_batch, mesh_path,
                                   MESH_DIR, LINK_MESH)

KP = ['panda_link0', 'panda_link2', 'panda_link3', 'panda_link4', 'panda_link6', 'panda_link7', 'panda_hand']


class MaskModel(nn.Module):
    def __init__(self, model_name, out_res):
        super().__init__()
        self.backbone = DINOv3Backbone(model_name, unfreeze_blocks=0)
        self.head = ViTKeypointHead(input_dim=self.backbone.model.config.hidden_size,
                                    num_joints=1, heatmap_size=(out_res, out_res))
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def forward(self, img):
        with torch.no_grad():
            tok = self.backbone(img)
        return self.head(tok)                       # (B,1,out_res,out_res) logits


def dice_loss(p, t, eps=1.0):
    p = p.flatten(1); t = t.flatten(1)
    return (1 - (2 * (p * t).sum(1) + eps) / (p.sum(1) + t.sum(1) + eps)).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='../Dataset/Converted_dataset/DREAM_to_DREAM_syn/panda_synth_train_dr')
    ap.add_argument('--model-name', default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    ap.add_argument('--image-size', type=int, default=512)
    ap.add_argument('--out-res', type=int, default=256)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--max-train', type=int, default=40000)
    ap.add_argument('--output-dir', default='./outputs_mask/mask_r1')
    args = ap.parse_args()

    device = torch.device('cuda'); assert torch.cuda.is_available()
    os.makedirs(args.output_dir, exist_ok=True)
    S, R = args.image_size, args.out_res
    mesh_verts = [((nm, fi), torch.from_numpy(load_obj_verts(mesh_path(nm))).to(device))
                  for nm, fi in LINK_MESH]

    m = MaskModel(args.model_name, R).to(device)
    full = PoseEstimationDataset(args.data_dir, keypoint_names=KP, image_size=(S, S),
                                 heatmap_size=(S, S), augment=False, include_angles=True)
    N = len(full.samples)
    idx = list(range(N)); cut = int(0.97 * N)
    tr = idx[:cut]; va = idx[cut:][:600]
    if args.max_train > 0 and len(tr) > args.max_train:
        st = max(1, len(tr) // args.max_train); tr = tr[::st][:args.max_train]
    print(f"train {len(tr)} val {len(va)}  out_res {R}", flush=True)
    trl = DataLoader(Subset(full, tr), batch_size=args.batch_size, shuffle=True, num_workers=10, pin_memory=True, drop_last=True)
    val = DataLoader(Subset(full, va), batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    def gt_mask(batch):
        """Render the GT-pose robot silhouette (B,R,R) from GT angles + Kabsch camera pose."""
        ang = batch['angles'].to(device).float().clone(); ang[:, 6] = 0.0
        kp3d = batch['keypoints_3d'].to(device).float()
        K = scale_K(batch['camera_K'], batch['original_size'], S).to(device)
        fk = panda_forward_kinematics(ang)
        Rm, tm = kabsch_batch(fk, kp3d)
        with torch.no_grad():
            mask = render_mesh(robot_pointcloud(ang, mesh_verts), Rm, tm, K, R, S)
        return (mask > 0.4).float()                                  # binarize label

    opt = torch.optim.AdamW(m.head.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(trl), pct_start=0.1)

    def evaluate():
        m.head.eval(); ious = []
        with torch.no_grad():
            for b in val:
                img = b['image'].to(device); tgt = gt_mask(b)
                pr = torch.sigmoid(m(img)).squeeze(1)
                p = (pr > 0.5).float()
                inter = (p * tgt).sum((-1, -2)); union = ((p + tgt) > 0).float().sum((-1, -2))
                ious.append((inter / union.clamp(min=1)).cpu())
        return float(torch.cat(ious).mean())

    best = 0.0
    for ep in range(args.epochs):
        m.head.train()
        for b in tqdm(trl, desc=f'Ep{ep} mask', leave=False):
            img = b['image'].to(device); tgt = gt_mask(b)
            logit = m(img).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(logit, tgt) + dice_loss(torch.sigmoid(logit), tgt)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        iou = evaluate()
        flag = ''
        if iou > best:
            best = iou; torch.save(m.head.state_dict(), os.path.join(args.output_dir, 'best_mask_head.pth')); flag = ' *'
        print(f"Ep{ep} | synth val mask IoU {iou:.4f} (best {best:.4f}){flag}", flush=True)
    print(f"[RESULT] mask head best synth IoU {best:.4f} -> {args.output_dir}/best_mask_head.pth", flush=True)


if __name__ == '__main__':
    main()
