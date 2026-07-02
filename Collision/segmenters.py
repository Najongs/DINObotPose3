"""
segmenters.py — region extractors for the collision pipeline.

- HumanSegmenter: pretrained torchvision Mask R-CNN, keeps the 'person' class.
- Robot region: use collision.keypoints_to_region on OUR model's projected FK
  keypoints (preferred, "use the model"), or reuse the CtRNet DeepLabV3 mask.

Kept intentionally lazy/optional so collision.py stays importable and testable
without downloading model weights (the synthetic demo needs neither).
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class HumanSegmenter:
    """Pretrained Mask R-CNN person segmenter (torchvision)."""

    def __init__(self, device: str = "cuda", score_thr: float = 0.7, mask_thr: float = 0.5):
        import torch
        from torchvision.models.detection import (
            maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights,
        )
        self.torch = torch
        self.device = device
        self.score_thr = score_thr
        self.mask_thr = mask_thr
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = maskrcnn_resnet50_fpn(weights=weights).eval().to(device)
        self.preprocess = weights.transforms()
        # COCO: 'person' == label 1
        self.person_label = 1

    def segment(self, rgb: np.ndarray) -> np.ndarray:
        """
        rgb : (H, W, 3) uint8 image.
        Returns a boolean (H, W) mask = union of all confident person instances.
        """
        import torch
        from PIL import Image
        H, W = rgb.shape[:2]
        img = self.preprocess(Image.fromarray(rgb)).to(self.device)
        with torch.no_grad():
            out = self.model([img])[0]
        mask = np.zeros((H, W), dtype=bool)
        labels = out["labels"].cpu().numpy()
        scores = out["scores"].cpu().numpy()
        masks = out["masks"].cpu().numpy()  # (N,1,H,W)
        for lb, sc, m in zip(labels, scores, masks):
            if lb == self.person_label and sc >= self.score_thr:
                mask |= (m[0] >= self.mask_thr)
        return mask


def ctrnet_mask_to_region(mask: np.ndarray, mask_thr: float = 0.5) -> np.ndarray:
    """Threshold a CtRNet DeepLabV3 soft robot mask into a boolean region."""
    return np.asarray(mask) >= mask_thr
