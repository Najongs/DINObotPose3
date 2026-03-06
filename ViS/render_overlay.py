"""
Robot Mesh Overlay Visualization using direct OpenCV projection.
- OBJ mesh parsing + FK transforms + cv2.projectPoints for pixel-perfect overlay
- Shows PnP-based camera frame results
- Shows iterative refinement progression
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# Import model
TRAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../TRAIN'))
EVAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Eval'))
sys.path.insert(0, TRAIN_DIR)
sys.path.insert(0, EVAL_DIR)
from model import (DINOv3PoseEstimator, panda_forward_kinematics,
                    soft_argmax_2d, solve_pnp_batch,
                    _PANDA_JOINTS, _PANDA_FIXED_J8, _PANDA_FIXED_HAND,
                    _make_transform, _rotation_matrix_z)
from checkpoint_compat import load_checkpoint_compat

MESH_DIR = os.path.join(os.path.dirname(__file__), 'Panda', 'meshes', 'collision')


# ============================================================
# OBJ Parser + FK-based Mesh Renderer (no PyBullet needed)
# ============================================================

def load_obj(filepath):
    """Parse OBJ file, return vertices (N,3) and faces (M,3) as numpy arrays."""
    verts, faces = [], []
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'v':
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'f':
                # Handle face formats: "f 1 2 3", "f 1/2/3 4/5/6 7/8/9", etc.
                face_verts = []
                for p in parts[1:]:
                    face_verts.append(int(p.split('/')[0]) - 1)  # OBJ is 1-indexed
                # Triangulate if polygon has > 3 vertices
                for i in range(1, len(face_verts) - 1):
                    faces.append([face_verts[0], face_verts[i], face_verts[i+1]])
    return np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int32)


def panda_fk_all_transforms(joint_angles):
    """
    Compute all link transforms (4x4) for Panda, matching the URDF kinematic chain.
    Returns dict mapping link_name -> (4,4) transform in robot base frame.
    """
    angles = np.array(joint_angles, dtype=np.float64)
    # Build fixed transforms from URDF joint data
    fixed_transforms = []
    for j in _PANDA_JOINTS:
        T = np.array(_make_transform(j['xyz'], j['rpy']),
                      dtype=np.float64).reshape(4, 4)
        fixed_transforms.append(T)
    T_j8 = np.array(_make_transform(_PANDA_FIXED_J8['xyz'], _PANDA_FIXED_J8['rpy']),
                     dtype=np.float64).reshape(4, 4)
    T_hand = np.array(_make_transform(_PANDA_FIXED_HAND['xyz'], _PANDA_FIXED_HAND['rpy']),
                      dtype=np.float64).reshape(4, 4)

    def Rz(theta):
        c, s = math.cos(theta), math.sin(theta)
        R = np.eye(4, dtype=np.float64)
        R[0, 0] = c; R[0, 1] = -s
        R[1, 0] = s; R[1, 1] = c
        return R

    cumul = np.eye(4, dtype=np.float64)
    transforms = {}
    transforms['panda_link0'] = cumul.copy()

    link_names = ['panda_link1', 'panda_link2', 'panda_link3',
                  'panda_link4', 'panda_link5', 'panda_link6', 'panda_link7']
    for i in range(7):
        cumul = cumul @ fixed_transforms[i] @ Rz(angles[i])
        transforms[link_names[i]] = cumul.copy()

    cumul_j8 = cumul @ T_j8
    transforms['panda_link8'] = cumul_j8.copy()
    cumul_hand = cumul_j8 @ T_hand
    transforms['panda_hand'] = cumul_hand.copy()

    return transforms


# Link name -> OBJ filename mapping
LINK_MESH_MAP = {
    'panda_link0': 'link0.obj',
    'panda_link1': 'link1.obj',
    'panda_link2': 'link2.obj',
    'panda_link3': 'link3.obj',
    'panda_link4': 'link4.obj',
    'panda_link5': 'link5.obj',
    'panda_link6': 'link6.obj',
    'panda_link7': 'link7.obj',
    'panda_hand':  'hand.obj',
}


class MeshProjector:
    """Project robot mesh onto image using FK + OpenCV projectPoints."""

    def __init__(self, mesh_dir=MESH_DIR):
        self.meshes = {}
        for link_name, obj_file in LINK_MESH_MAP.items():
            path = os.path.join(mesh_dir, obj_file)
            if os.path.exists(path):
                verts, faces = load_obj(path)
                self.meshes[link_name] = (verts, faces)

    def render_wireframe(self, img, joint_angles, rvec, tvec, camera_K,
                          color=(0, 255, 0), thickness=1, alpha=0.4):
        """
        Render robot mesh as filled triangles with transparency.

        Args:
            img: (H, W, 3) BGR image (will be modified in place)
            joint_angles: (7,) radians
            rvec, tvec: PnP extrinsics (world=robot base -> camera)
            camera_K: (3, 3) intrinsic matrix
            color: BGR color tuple
            thickness: -1 for filled, >0 for wireframe line thickness
            alpha: transparency (0=invisible, 1=opaque)
        """
        R, _ = cv2.Rodrigues(rvec.astype(np.float64))
        t = tvec.astype(np.float64).reshape(3, 1)
        K = camera_K.astype(np.float64)

        transforms = panda_fk_all_transforms(joint_angles)

        overlay = img.copy()
        h, w = img.shape[:2]

        # Collect all triangles with depth for painter's algorithm
        all_tris = []
        for link_name, T_link in transforms.items():
            if link_name not in self.meshes:
                continue
            verts_local, faces = self.meshes[link_name]

            # Transform vertices: local -> robot base frame
            verts_world = (T_link[:3, :3] @ verts_local.T + T_link[:3, 3:4]).T  # (N, 3)

            # Transform to camera frame
            verts_cam = (R @ verts_world.T + t).T  # (N, 3)

            # Project to 2D
            verts_2d, _ = cv2.projectPoints(
                verts_world, rvec.astype(np.float64), tvec.astype(np.float64),
                K, None
            )
            verts_2d = verts_2d.reshape(-1, 2)

            for face in faces:
                pts_2d = verts_2d[face].astype(np.int32)
                # Skip faces behind camera
                depths = verts_cam[face, 2]
                if np.any(depths < 0.01):
                    continue
                # Skip faces entirely outside image
                if (np.all(pts_2d[:, 0] < 0) or np.all(pts_2d[:, 0] >= w) or
                    np.all(pts_2d[:, 1] < 0) or np.all(pts_2d[:, 1] >= h)):
                    continue
                mean_depth = np.mean(depths)
                all_tris.append((mean_depth, pts_2d))

        # Sort by depth (far to near = painter's algorithm)
        all_tris.sort(key=lambda x: -x[0])

        for _, pts_2d in all_tris:
            cv2.fillConvexPoly(overlay, pts_2d.reshape(-1, 1, 2), color, cv2.LINE_AA)

        # Blend
        result = cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0)
        np.copyto(img, result)
        return img


def solve_pnp_single(kp_2d, kp_3d_robot, camera_K):
    """Solve PnP for a single sample, return rvec, tvec."""
    pts2d = kp_2d.astype(np.float64)
    pts3d = kp_3d_robot.astype(np.float64)
    K = camera_K.astype(np.float64)

    success, rvec, tvec = cv2.solvePnP(
        pts3d, pts2d, K, None,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None, None, False

    proj, _ = cv2.projectPoints(pts3d, rvec, tvec, K, None)
    reproj_err = np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - pts2d) ** 2, axis=1)))
    return rvec.flatten(), tvec.flatten(), reproj_err < 10.0


def draw_keypoints_and_skeleton(img, kp_2d, color, label_prefix="", radius=5):
    """Draw 2D keypoints and skeleton links."""
    kp_names = ['L0', 'L2', 'L3', 'L4', 'L6', 'L7', 'Hand']
    skeleton = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]

    for (i, j) in skeleton:
        pt1 = (int(kp_2d[i, 0]), int(kp_2d[i, 1]))
        pt2 = (int(kp_2d[j, 0]), int(kp_2d[j, 1]))
        cv2.line(img, pt1, pt2, color, 2, cv2.LINE_AA)

    for i, (x, y) in enumerate(kp_2d):
        cv2.circle(img, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)
        cv2.circle(img, (int(x), int(y)), radius, (255, 255, 255), 1, cv2.LINE_AA)
        if label_prefix:
            cv2.putText(img, f"{label_prefix}{kp_names[i]}",
                        (int(x) + 8, int(y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    return img


def run_visualization(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    keypoint_names = ['panda_link0', 'panda_link2', 'panda_link3',
                      'panda_link4', 'panda_link6', 'panda_link7', 'panda_hand']
    heatmap_size = 512

    # --- Load annotation ---
    from inference_with_real import load_annotation
    image_path, gt_2d, gt_3d, camera_K, found, gt_angles = load_annotation(
        args.json_path, keypoint_names
    )
    if image_path is None or not os.path.exists(image_path):
        print(f"ERROR: Image not found: {image_path}")
        return

    # --- Load model ---
    model = DINOv3PoseEstimator(
        dino_model_name=args.model_name,
        heatmap_size=heatmap_size,
        unfreeze_blocks=0,
        fix_joint7_zero=args.fix_joint7
    ).to(device)
    load_checkpoint_compat(model, args.model_path, device, is_main_process=True)
    model.eval()

    # --- Preprocess image ---
    from PIL import Image as PILImage
    import torchvision.transforms as T

    image_pil = PILImage.open(image_path).convert("RGB")
    orig_w, orig_h = image_pil.size

    transform = T.Compose([
        T.Resize((heatmap_size, heatmap_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image_pil).unsqueeze(0).to(device)

    # Scale camera K
    camera_K_scaled_t = None
    if camera_K is not None:
        sx, sy = heatmap_size / orig_w, heatmap_size / orig_h
        camera_K_scaled = camera_K.copy()
        camera_K_scaled[0, 0] *= sx; camera_K_scaled[1, 1] *= sy
        camera_K_scaled[0, 2] *= sx; camera_K_scaled[1, 2] *= sy
        camera_K_scaled_t = torch.tensor(camera_K_scaled, dtype=torch.float32).unsqueeze(0).to(device)

    # --- Forward pass ---
    with torch.no_grad():
        outputs = model(image_tensor, camera_K=camera_K_scaled_t)

    pred_angles = outputs["joint_angles"][0].cpu().numpy()
    pred_2d_hm = soft_argmax_2d(outputs["heatmaps_2d"])[0].cpu().numpy()
    pred_3d_robot = outputs["keypoints_3d_robot"][0].cpu().numpy()

    # Scale 2D to original resolution
    pred_2d_orig = pred_2d_hm.copy()
    pred_2d_orig[:, 0] *= (orig_w / heatmap_size)
    pred_2d_orig[:, 1] *= (orig_h / heatmap_size)

    # --- PnP solve for camera extrinsics ---
    rvec_gt, tvec_gt, gt_pnp_ok = None, None, False
    gt_3d_robot = None
    if camera_K is not None and gt_angles is not None:
        gt_3d_robot = panda_forward_kinematics(
            torch.tensor(gt_angles, dtype=torch.float32).unsqueeze(0)
        )[0].numpy()
        rvec_gt, tvec_gt, gt_pnp_ok = solve_pnp_single(gt_2d, gt_3d_robot, camera_K)
        print(f"PnP (GT):   valid={gt_pnp_ok}")

    rvec_pred, tvec_pred, pnp_ok = None, None, False
    if camera_K is not None:
        rvec_pred, tvec_pred, pnp_ok = solve_pnp_single(
            pred_2d_orig, pred_3d_robot, camera_K
        )
        print(f"PnP (pred): valid={pnp_ok}")
        if not pnp_ok and gt_pnp_ok:
            rvec_pred, tvec_pred, pnp_ok = rvec_gt.copy(), tvec_gt.copy(), True
            print("  -> Using GT extrinsics as fallback for pred mesh")

    # --- Setup ---
    os.makedirs(args.output_dir, exist_ok=True)
    img_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    projector = MeshProjector()

    # =============================================
    # PANEL 1: 2D Keypoint + Skeleton overlay
    # =============================================
    panel_2d = img_bgr.copy()
    if gt_angles is not None:
        panel_2d = draw_keypoints_and_skeleton(panel_2d, gt_2d, (0, 255, 0), "GT:")
    panel_2d = draw_keypoints_and_skeleton(panel_2d, pred_2d_orig, (0, 0, 255), "P:")
    cv2.putText(panel_2d, "Green=GT  Red=Pred", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(os.path.join(args.output_dir, "01_keypoints_skeleton.png"), panel_2d)
    print("Saved: 01_keypoints_skeleton.png")

    # =============================================
    # PANEL 2: Mesh overlay using PnP extrinsics
    # =============================================
    if pnp_ok and camera_K is not None:
        panel_mesh = img_bgr.copy()
        # GT mesh (green) first (behind)
        if gt_pnp_ok and gt_angles is not None:
            projector.render_wireframe(
                panel_mesh, gt_angles, rvec_gt, tvec_gt, camera_K,
                color=(0, 200, 50), alpha=0.3
            )
        # Pred mesh (blue) on top
        projector.render_wireframe(
            panel_mesh, pred_angles, rvec_pred, tvec_pred, camera_K,
            color=(255, 150, 50), alpha=0.4
        )
        cv2.putText(panel_mesh, "Blue=Pred  Green=GT (PnP Mesh Overlay)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(args.output_dir, "02_mesh_overlay_pnp.png"), panel_mesh)
        print("Saved: 02_mesh_overlay_pnp.png")
    else:
        print("SKIP: 02_mesh_overlay_pnp.png (PnP failed)")

    # =============================================
    # PANEL 3: Iterative refinement visualization
    # =============================================
    print("\nGenerating iterative refinement visualization...")
    iter_angles_list = get_iterative_angles(model, image_tensor, camera_K_scaled_t, device)

    if iter_angles_list and (gt_pnp_ok or pnp_ok):
        rvec_iter = rvec_gt if gt_pnp_ok else rvec_pred
        tvec_iter = tvec_gt if gt_pnp_ok else tvec_pred
        n_iters = len(iter_angles_list)
        panels = []
        for step, angles_i in enumerate(iter_angles_list):
            panel_i = img_bgr.copy()
            t = step / max(n_iters - 1, 1)
            # Color gradient: orange -> cyan
            b_val = int(50 + 205 * t)
            r_val = int(50 + 205 * (1 - t))
            projector.render_wireframe(
                panel_i, angles_i, rvec_iter, tvec_iter, camera_K,
                color=(b_val, 150, r_val), alpha=0.45
            )
            angle_err_str = ""
            if gt_angles is not None:
                diff = np.arctan2(np.sin(angles_i - gt_angles), np.cos(angles_i - gt_angles))
                mean_err = np.mean(np.abs(np.degrees(diff)))
                angle_err_str = f"  MAE={mean_err:.1f}deg"
            cv2.putText(panel_i, f"Iter {step + 1}/{n_iters}{angle_err_str}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2, cv2.LINE_AA)
            panels.append(panel_i)

        tile = tile_images(panels, n_cols=min(n_iters, 4))
        cv2.imwrite(os.path.join(args.output_dir, "03_iterative_refinement.png"), tile)
        print("Saved: 03_iterative_refinement.png")

    # =============================================
    # PANEL 4: Side-by-side GT vs Pred comparison
    # =============================================
    if pnp_ok and gt_pnp_ok and camera_K is not None:
        left = img_bgr.copy()
        projector.render_wireframe(
            left, gt_angles, rvec_gt, tvec_gt, camera_K,
            color=(0, 220, 50), alpha=0.5
        )
        cv2.putText(left, "Ground Truth", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        right = img_bgr.copy()
        projector.render_wireframe(
            right, pred_angles, rvec_pred, tvec_pred, camera_K,
            color=(255, 150, 50), alpha=0.5
        )
        cv2.putText(right, "Prediction", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2, cv2.LINE_AA)

        comparison = np.hstack([left, right])
        cv2.imwrite(os.path.join(args.output_dir, "04_gt_vs_pred_comparison.png"), comparison)
        print("Saved: 04_gt_vs_pred_comparison.png")

    # =============================================
    # PANEL 5: Metrics summary image
    # =============================================
    metrics_img = create_metrics_panel(
        pred_angles, gt_angles, pred_2d_orig, gt_2d,
        pred_3d_robot, gt_3d, found, keypoint_names, orig_w, orig_h
    )
    cv2.imwrite(os.path.join(args.output_dir, "05_metrics_summary.png"), metrics_img)
    print("Saved: 05_metrics_summary.png")

    print(f"\nAll visualizations saved to: {args.output_dir}")


def get_iterative_angles(model, image_tensor, camera_K_scaled_t, device):
    """Extract intermediate joint angles from each iteration of the refinement loop."""
    intermediate = []
    jah = model.joint_angle_head
    original_forward = jah.forward

    def patched_forward(dino_features, predicted_heatmaps, camera_K=None):
        import torch.nn.functional as F
        B = dino_features.shape[0]
        dev = dino_features.device

        global_feat = F.adaptive_avg_pool2d(
            dino_features.permute(0, 2, 1).reshape(
                B, dino_features.shape[2],
                int(math.sqrt(dino_features.shape[1])),
                int(math.sqrt(dino_features.shape[1]))
            ), 1
        ).flatten(1)
        global_feat = jah.feature_proj(global_feat)

        uv_hm = soft_argmax_2d(predicted_heatmaps, temperature=10.0)
        hm_h, hm_w = predicted_heatmaps.shape[2:]
        u_n = uv_hm[:, :, 0] / hm_w
        v_n = uv_hm[:, :, 1] / hm_h
        spatial_info = torch.stack([u_n, v_n], dim=-1)
        spatial_feat = jah.heatmap_encoder(spatial_info.mean(dim=1))

        pred_sc = torch.zeros(B, jah.num_angles * 2, device=dev)
        pred_sc[:, 0::2] = 1.0

        for iter_step in range(jah.n_iter):
            combined = torch.cat([global_feat, spatial_feat, pred_sc], dim=1)
            x = jah.fc_1(combined)
            x = jah.drop1(x)
            x = jah.fc_2(x)
            x = jah.drop2(x)
            delta = jah.angle_delta(x)

            cos_prev = pred_sc[:, 0::2]
            sin_prev = pred_sc[:, 1::2]
            cos_new = cos_prev - sin_prev * delta
            sin_new = sin_prev + cos_prev * delta
            norm = torch.sqrt(cos_new**2 + sin_new**2).clamp(min=1e-8)
            cos_new = cos_new / norm
            sin_new = sin_new / norm
            pred_sc = torch.stack([cos_new, sin_new], dim=2).reshape(B, jah.num_angles * 2)

            angles_i = torch.atan2(sin_new, cos_new)
            angles_i = torch.clamp(angles_i, jah.joint_lower, jah.joint_upper)
            if jah.num_angles < 7:
                angles_i = torch.cat([angles_i, torch.zeros(B, 7 - jah.num_angles, device=dev)], dim=1)
            intermediate.append(angles_i[0].cpu().numpy())

        return original_forward(dino_features, predicted_heatmaps, camera_K)

    try:
        jah.forward = patched_forward
        model.eval()
        with torch.no_grad():
            model(image_tensor, camera_K=camera_K_scaled_t)
    finally:
        jah.forward = original_forward

    return intermediate


def tile_images(images, n_cols=4, border=3, bg_color=(40, 40, 40)):
    """Tile list of images into a grid."""
    n = len(images)
    n_rows = (n + n_cols - 1) // n_cols
    h0, w0 = images[0].shape[:2]
    for i in range(len(images)):
        if images[i].shape[:2] != (h0, w0):
            images[i] = cv2.resize(images[i], (w0, h0))
    total_h = n_rows * h0 + (n_rows + 1) * border
    total_w = n_cols * w0 + (n_cols + 1) * border
    canvas = np.full((total_h, total_w, 3), bg_color, dtype=np.uint8)
    for idx, img in enumerate(images):
        r, c = divmod(idx, n_cols)
        y = border + r * (h0 + border)
        x = border + c * (w0 + border)
        canvas[y:y+h0, x:x+w0] = img
    return canvas


def create_metrics_panel(pred_angles, gt_angles, pred_2d, gt_2d,
                          pred_3d_robot, gt_3d, found, kp_names, w, h):
    """Create a visual metrics summary panel."""
    panel = np.zeros((400, 600, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)
    y = 30
    cv2.putText(panel, "EVALUATION METRICS", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if gt_angles is not None:
        diff = np.arctan2(np.sin(pred_angles - gt_angles), np.cos(pred_angles - gt_angles))
        errs = np.abs(np.degrees(diff))
        y += 40
        cv2.putText(panel, f"Joint Angle MAE: {np.mean(errs):.2f} deg", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1)
        for i in range(7):
            y += 22
            color = (0, 255, 0) if errs[i] < 5 else (0, 200, 255) if errs[i] < 15 else (0, 0, 255)
            cv2.putText(panel, f"  J{i}: {errs[i]:6.2f} deg", (30, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    dist_2d = np.linalg.norm(pred_2d - gt_2d, axis=1)
    y += 35
    cv2.putText(panel, f"2D Keypoint MAE: {np.mean(dist_2d):.2f} px", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1)

    if np.any(gt_3d):
        dist_3d = np.linalg.norm(pred_3d_robot - gt_3d, axis=1) * 1000
        y += 30
        cv2.putText(panel, f"3D FK Error (robot frame): {np.mean(dist_3d):.1f} mm", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1)

    return panel


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot Mesh Overlay Visualization")
    parser.add_argument("-j", "--json-path", required=True, help="Annotation JSON")
    parser.add_argument("-p", "--model-path", required=True, help="Model checkpoint")
    parser.add_argument("-o", "--output-dir", default="./vis_output", help="Output directory")
    parser.add_argument("--model-name", default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    parser.add_argument("--fix-joint7", action="store_true")
    args = parser.parse_args()
    run_visualization(args)
