"""
Visualize render-compare mask alignment: for a few real frames, overlay the GT-pose mesh silhouette at
several mesh-SHRINK factors (shrink each link's verts toward its centroid -> thinner -> closer to the true
robot than the bulky collision mesh) vs the SAM mask. Saves comparison panels to ViS/rc_viz/ so the
mesh-shrink idea (option B-1) can be eyeballed and the best shrink picked.
Panel per frame: [ real | render s=1.0 | render s=0.8 | render s=0.65 | SAM ]  with SAM-IoU labels.
"""
import os, sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw

sys.path.append(os.path.dirname(__file__)); sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../TRAIN')))
os.environ.setdefault('MESH_KIND', 'collision')
from silhouette_mesh_probe import (load_obj_verts, all_link_transforms, render_mesh, kabsch_batch,
                                   mesh_path, LINK_MESH)
from model_v4 import panda_forward_kinematics
from inference_4tier_eval import EvalDataset
from refine_eval import scale_K

KPN = ['panda_link0', 'panda_link2', 'panda_link3', 'panda_link4', 'panda_link6', 'panda_link7', 'panda_hand']
DEV = 'cuda'; S = 512; H = 256
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../ViS/rc_viz'))
os.makedirs(OUT, exist_ok=True)
MEAN = np.array([0.485, 0.456, 0.406]); STD = np.array([0.229, 0.224, 0.225])
SHRINKS = [1.0, 0.8, 0.65]


def load_shrunk(s):
    """Load each link mesh and shrink its verts toward the mesh centroid by factor s (thinner robot)."""
    out = []
    for nm, fi in LINK_MESH:
        v = load_obj_verts(mesh_path(nm))
        c = v.mean(0, keepdims=True)
        v = c + s * (v - c)
        out.append(((nm, fi), torch.from_numpy(v).to(DEV)))
    return out


def robot_pc(theta, mv):
    frames = all_link_transforms(theta)
    pts = []
    for (nm, fi), v in mv:
        vh = torch.cat([v, torch.ones(v.shape[0], 1, device=DEV)], 1)
        pts.append(torch.einsum('bij,nj->bni', frames[:, fi], vh)[..., :3])
    return torch.cat(pts, 1)


def main():
    from segment_anything import sam_model_registry, SamPredictor
    sam = sam_model_registry['vit_b'](checkpoint='/data/public/97_cache/sam/sam_vit_b_01ec64.pth').to(DEV).eval()
    sam_pred = SamPredictor(sam)

    ds = EvalDataset('../Dataset/Converted_dataset/DREAM_real/panda-3cam_realsense', KPN, image_size=(S, S))
    ds.json_files = ds.json_files[::700][:6]
    mvs = {s: load_shrunk(s) for s in SHRINKS}

    for fi, batch in enumerate(DataLoader(ds, batch_size=1)):
        img = batch['image'].to(DEV); K = scale_K(batch['camera_K'], batch['original_size'], S).to(DEV)
        gt3d = batch['gt_3d'].to(DEV); ga = batch['gt_angles'].to(DEV).clone(); ga[:, 6] = 0
        gt2d = batch['gt_2d'][0].cpu().numpy(); fnd = batch['found'][0].cpu().numpy() > 0
        Rg, tg = kabsch_batch(panda_forward_kinematics(ga), gt3d)
        im = (img[0].cpu().numpy().transpose(1, 2, 0) * STD + MEAN).clip(0, 1)
        imS = np.array(Image.fromarray((im * 255).astype('uint8')).resize((H, H))) / 255.0

        # SAM mask (box prompt from GT visible keypoints)
        u8 = (im * 255).astype('uint8')
        sam_pred.set_image(u8)
        p = gt2d[fnd]
        x0, y0 = p[:, 0].min(), p[:, 1].min(); x1, y1 = p[:, 0].max(), p[:, 1].max()
        mg = 0.05 * max(x1 - x0, y1 - y0)                              # TIGHT box (was 0.15 -> over-segmented)
        box = np.array([max(0, x0 - mg), max(0, y0 - mg), min(S, x1 + mg), min(S, y1 + mg)])
        # positive: robot keypoints; NEGATIVE: image corners (definitely background) to kill over-segmentation
        neg = np.array([[8, 8], [S - 8, 8], [8, S - 8], [S - 8, S - 8], [S // 2, S - 8]])
        pc = np.concatenate([p, neg], 0)
        pl = np.concatenate([np.ones(len(p)), np.zeros(len(neg))])
        m, sc, _ = sam_pred.predict(point_coords=pc, point_labels=pl, box=box, multimask_output=True)
        sam_m = torch.from_numpy(m[int(np.argmax(sc))].astype('float32')).to(DEV)
        sam_s = F.interpolate(sam_m[None, None], size=(H, H), mode='bilinear')[0, 0].cpu().numpy()

        panels = []
        # real
        panels.append((imS.copy(), 'real'))
        # renders at each shrink
        for s in SHRINKS:
            rmask = render_mesh(robot_pc(ga, mvs[s]), Rg, tg, K, H, S)[0].detach().cpu().numpy()
            rb = (F.interpolate(torch.from_numpy(rmask)[None, None], size=(H, H))[0, 0].numpy() > 0.5)
            sb = sam_s > 0.5
            iou = (rb & sb).sum() / max(1, (rb | sb).sum())
            ov = imS.copy(); ov[..., 0] = np.clip(ov[..., 0] + 0.6 * rmask, 0, 1); ov[..., 1] *= (1 - 0.3 * rmask)
            panels.append((ov, f's={s} IoU={iou:.2f}'))
        # SAM
        ovs = imS.copy(); ovs[..., 2] = np.clip(ovs[..., 2] + 0.6 * sam_s, 0, 1); ovs[..., 1] *= (1 - 0.3 * sam_s)
        panels.append((ovs, 'SAM'))

        W = H * len(panels)
        canvas = Image.new('RGB', (W, H + 16), 'white')
        d = ImageDraw.Draw(canvas)
        for j, (pim, lab) in enumerate(panels):
            canvas.paste(Image.fromarray((pim * 255).astype('uint8')), (j * H, 16))
            d.text((j * H + 3, 2), lab, fill='black')
        path = os.path.join(OUT, f'frame_{fi}.png')
        canvas.save(path)
        print(f'saved {path}', flush=True)
    print(f'\n[DONE] {len(ds.json_files)} panels -> {OUT}', flush=True)


if __name__ == '__main__':
    main()
