"""Optional class-agnostic scene proposals; DINO chooses the reference target."""
from __future__ import annotations

import numpy as np


def make_mask_provider(backend="sam", *, points_per_batch=4):
    if isinstance(points_per_batch, bool) or points_per_batch not in (1, 2, 4, 8, 16, 32):
        # Transformers 4.44's end-of-grid marker assumes an exact batch divisor.
        raise ValueError("SAM points_per_batch must be one of 1, 2, 4, 8, 16, 32")
    if backend == "flat":
        return None  # explicit limited tabletop baseline in rgb_target_detection
    if backend != "sam":
        raise ValueError(f"unsupported mask backend: {backend}")
    import torch
    from transformers import pipeline

    generator = pipeline("mask-generation", model="facebook/sam-vit-base",
                         device=0 if torch.cuda.is_available() else -1)

    def masks(image):
        # 32 simultaneous SAM prompts exhausted RAM in the laptop container.
        # Keep the same proposal grid/thresholds, but decode fewer prompts at once.
        result = generator(image.convert("RGB"), points_per_batch=points_per_batch)
        proposals = [np.asarray(mask, dtype=bool) for mask in result["masks"]]
        if not proposals:
            raise RuntimeError("SAM produced no scene masks; cannot identify the target")
        return proposals

    return masks
