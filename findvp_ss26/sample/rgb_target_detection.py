"""Reference-guided RGB target selection without a known object image position.

Default proposals assume a flat, approximately uniform tabletop/background
visible at the image border, with separated foreground objects. This is a
deliberately limited segmentation baseline: textured tables, touching objects,
strong shadows, occlusion, and similar background/object colours can defeat it.
Supply full-resolution binary ``candidate_masks`` from an instance segmenter
(e.g. SAM) to replace that proposal stage. Neither stage returns metric depth
or a 6D object pose. DINO similarity thresholds require scene-specific tuning.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from PIL import Image

if __package__:
    from .dinov3_feature_matching import DenseFeatures
    from .target_guided_similarity import (
        _ObjectRegion, _dense_object_similarity, _extract_object_features,
    )
else:
    from dinov3_feature_matching import DenseFeatures
    from target_guided_similarity import (
        _ObjectRegion, _dense_object_similarity, _extract_object_features,
    )


class TargetDetectionError(RuntimeError):
    """The RGB observation does not identify one sufficiently confident target."""


def _flat_background_proposals(image: Image.Image) -> list[np.ndarray]:
    """Find all foreground components without an image-centre selection prior."""
    from scipy.ndimage import binary_fill_holes, label
    from skimage.color import rgb2lab
    from skimage.morphology import binary_closing, binary_opening, disk

    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.
    lab = rgb2lab(rgb)
    height, width = lab.shape[:2]
    border_width = max(1, round(min(height, width) * .02))
    border = np.concatenate((lab[:border_width].reshape(-1, 3),
                             lab[-border_width:].reshape(-1, 3),
                             lab[:, :border_width].reshape(-1, 3),
                             lab[:, -border_width:].reshape(-1, 3)))
    background = np.median(border, axis=0)
    border_distance = np.linalg.norm(border - background, axis=1)
    # A contaminated/varied border should yield no confident proposals rather
    # than silently choosing an image-centred blob or the whole scene.
    threshold = max(10., float(np.percentile(border_distance, 90)) + 4.)
    foreground = np.linalg.norm(lab - background, axis=2) > threshold
    foreground = binary_closing(foreground, disk(1))
    foreground = binary_opening(foreground, disk(1))
    components, count = label(foreground)
    return [binary_fill_holes(components == index) for index in range(1, count + 1)]


def _validated_mask(raw_mask, shape: tuple[int, int]) -> np.ndarray:
    mask = np.asarray(raw_mask)
    if mask.shape != shape:
        raise ValueError(f"candidate mask must have image shape {shape}; got {mask.shape}")
    if not np.issubdtype(mask.dtype, np.bool_):
        if not np.issubdtype(mask.dtype, np.number) or not np.all(np.isfinite(mask)):
            raise ValueError("candidate masks must contain finite binary values")
        if not np.all((mask == 0) | (mask == 1) | (mask == 255)):
            raise ValueError("candidate masks must be binary (0/1 or 0/255)")
    return mask.astype(bool, copy=True)


def _same_region(first: np.ndarray, second: np.ndarray) -> bool:
    intersection = np.count_nonzero(first & second)
    union = np.count_nonzero(first | second)
    return intersection / max(union, 1) >= .60


def detect_target_region(
    image: Image.Image,
    target_features: DenseFeatures,
    extractor,
    *,
    min_similarity: float = .55,
    ambiguity_margin: float = .03,
    candidate_masks: Iterable[np.ndarray] | None = None,
) -> dict:
    """Select a scene instance by symmetric object-only DINO cosine similarity.

    ``bbox_xyxy`` is the tight, exclusive-right/bottom pixel box. The DINO crop
    uses 15% padding internally, but only foreground patches are scored. Masks
    smaller than 16 pixels/0.1% of the image or larger than 70% are discarded.
    Similar-scoring spatially distinct candidates fail explicitly; overlapping
    duplicate masks are ignored for the ambiguity test. No candidate preference
    depends on distance to the image centre. This does not authenticate identity
    when only one semantically similar distractor is visible.
    """
    for name, value in (("min_similarity", min_similarity),
                        ("ambiguity_margin", ambiguity_margin)):
        if not math.isfinite(value) or not 0. <= value <= 1.:
            raise ValueError(f"{name} must be finite and between 0 and 1")
    if not isinstance(image, Image.Image) or min(image.size) < 2:
        raise ValueError("image must be a nonempty PIL image at least 2x2")
    if len(target_features.descriptors) == 0:
        raise ValueError("target_features must contain object descriptors")
    image = image.convert("RGB")
    source = "flat_background" if candidate_masks is None else "supplied_masks"
    proposals = (_flat_background_proposals(image)
                 if candidate_masks is None else candidate_masks)
    height, width = image.height, image.width
    min_pixels = max(16, round(height * width * .001))
    candidates = []
    rejected = []
    for index, raw_mask in enumerate(proposals):
        mask = _validated_mask(raw_mask, (height, width))
        area = int(mask.sum())
        if area < min_pixels or area > .70 * height * width:
            rejected.append({"index": index, "reason": "mask_area", "pixels": area})
            continue
        rows, columns = np.nonzero(mask)
        bbox = (int(columns.min()), int(rows.min()),
                int(columns.max()) + 1, int(rows.max()) + 1)
        left, top, right, bottom = bbox
        padding = round(.15 * max(right - left, bottom - top))
        padded_bbox = (max(0, left - padding), max(0, top - padding),
                       min(width, right + padding), min(height, bottom + padding))
        try:
            descriptors = _extract_object_features(
                extractor, image, _ObjectRegion(mask, padded_bbox))
        except ValueError as error:
            if "Mask contains no sufficiently covered" not in str(error):
                raise
            rejected.append({"index": index, "reason": "no_foreground_patches"})
            continue
        if len(descriptors.descriptors) == 0:
            rejected.append({"index": index, "reason": "no_foreground_patches"})
            continue
        similarity = _dense_object_similarity(target_features, descriptors)
        if not math.isfinite(similarity):
            raise TargetDetectionError("Nonfinite DINO similarity; cannot identify target")
        candidates.append({"index": index, "mask": mask, "bbox_xyxy": bbox,
                           "similarity": similarity})
    if not candidates:
        raise TargetDetectionError("No valid RGB target candidate; inspect segmentation or supply masks")
    candidates.sort(key=lambda candidate: candidate["similarity"], reverse=True)
    best = candidates[0]
    if best["similarity"] < min_similarity:
        raise TargetDetectionError(
            f"Best target similarity {best['similarity']:.3f} is below {min_similarity:.3f}")
    distinct = [other for other in candidates[1:]
                if not _same_region(best["mask"], other["mask"])]
    runner_up = distinct[0]["similarity"] if distinct else None
    if runner_up is not None and best["similarity"] - runner_up <= ambiguity_margin:
        raise TargetDetectionError(
            f"Target is ambiguous between distinct objects: {best['similarity']:.3f} vs {runner_up:.3f}")
    return {
        "mask": best["mask"], "bbox_xyxy": best["bbox_xyxy"],
        "similarity": best["similarity"],
        "diagnostics": {
            "proposal_source": source, "candidate_count": len(candidates),
            "selected_candidate_index": best["index"],
            "runner_up_similarity": runner_up, "rejected_candidates": rejected,
            "scores": [{key: candidate[key] for key in ("index", "bbox_xyxy", "similarity")}
                       for candidate in candidates],
        },
    }
