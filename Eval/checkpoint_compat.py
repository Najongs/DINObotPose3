"""
Checkpoint compatibility loader for Eval scripts.

Handles:
- DDP "module." prefix removal
- known backbone mask_token shape differences across transformers versions
- shape-mismatched key dropping with concise diagnostics
- critical head presence check to fail fast on true model/code mismatch
"""

from __future__ import annotations

from typing import Dict, Tuple, Any

import torch


def load_checkpoint_compat(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: torch.device,
    *,
    is_main_process: bool = True,
    critical_keys: Tuple[str, ...] = (
        "keypoint_head.heatmap_predictor.weight",
        "keypoint_head.heatmap_predictor.bias",
    ),
) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model_state = model.state_dict()

    # Known transformers compatibility case: mask_token shape changed across versions.
    mt_key = "backbone.model.embeddings.mask_token"
    skipped_mask_token = False
    if mt_key in state_dict:
        if mt_key in model_state and state_dict[mt_key].shape != model_state[mt_key].shape:
            del state_dict[mt_key]
            skipped_mask_token = True

    filtered_state = {}
    dropped = []
    for k, v in state_dict.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered_state[k] = v
        elif k in model_state:
            dropped.append(k)

    missing_critical = [k for k in critical_keys if k in model_state and k not in filtered_state]
    if missing_critical:
        raise RuntimeError(
            "Critical model weights are missing/mismatched. "
            f"This likely indicates checkpoint/code incompatibility: {missing_critical}"
        )

    model.load_state_dict(filtered_state, strict=False)

    if is_main_process:
        epoch = checkpoint.get("epoch", None) if isinstance(checkpoint, dict) else None
        if epoch is not None:
            print(f"# Checkpoint epoch: {epoch}")
        if skipped_mask_token:
            print("# Compatibility: skipped backbone mask_token due to transformers shape difference")
        if dropped:
            print(f"# Compatibility: dropped {len(dropped)} mismatched key(s)")
        print(f"# Loaded {len(filtered_state)}/{len(model_state)} parameter tensors into current model")

    return {
        "checkpoint": checkpoint,
        "filtered_key_count": len(filtered_state),
        "model_key_count": len(model_state),
        "dropped_keys": dropped,
        "skipped_mask_token": skipped_mask_token,
    }

