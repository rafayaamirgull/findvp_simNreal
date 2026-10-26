#!/usr/bin/env python3
"""Rank captured triangle views against one target image."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw

if __package__:
    from .dinov3_feature_matching import (
        DINOv3FeatureExtractor,
        DenseFeatures,
        features_in_mask,
        filter_matches_with_fundamental_ransac,
        load_image,
        match_features,
        offset_feature_keypoints,
    )
    from .dinov3_spatial_clustering import (
        DINOv3SpatialClusterer,
        SpatialClusterResult,
    )
else:
    from dinov3_feature_matching import (
        DINOv3FeatureExtractor,
        DenseFeatures,
        features_in_mask,
        filter_matches_with_fundamental_ransac,
        load_image,
        match_features,
        offset_feature_keypoints,
    )
    from dinov3_spatial_clustering import (
        DINOv3SpatialClusterer,
        SpatialClusterResult,
    )


@dataclass(frozen=True)
class SIFTScore:
    """Geometrically verified SIFT score and its diagnostics."""

    score: float
    target_keypoints: int
    candidate_keypoints: int
    good_matches: int
    inliers: int
    inlier_ratio: float
    target_inlier_xy: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float32),
        repr=False,
        compare=False,
    )
    candidate_inlier_xy: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float32),
        repr=False,
        compare=False,
    )


@dataclass(frozen=True)
class ViewScore:
    """One auditable target-to-candidate comparison."""

    triangle: int
    image_filename: str
    score: float
    dino_score: float
    dino_semantic_score: float
    dino_geometric_score: float
    dino_target_features: int
    dino_candidate_features: int
    dino_candidate_matches: int
    dino_ransac_inliers: int
    sift_score: float
    sift_good_matches: int
    sift_inliers: int
    sift_inlier_ratio: float


@dataclass(frozen=True)
class _ObjectRegion:
    """A full-resolution object mask and its padded crop."""

    mask: np.ndarray
    bbox_xyxy: tuple[int, int, int, int]


@dataclass(frozen=True)
class _SIFTFeatures:
    """Reusable masked SIFT keypoints and descriptors for one image."""

    points: np.ndarray
    descriptors: np.ndarray


@dataclass(frozen=True)
class PreparedTarget:
    """Target-side perception results reusable across refinement levels."""

    path: Path
    image: Image.Image
    region: _ObjectRegion
    object_features: DenseFeatures
    sift_features: _SIFTFeatures
    full_features: DenseFeatures | None = None
    clusters: SpatialClusterResult | None = None


def _central_object_region(image: Image.Image) -> _ObjectRegion:
    """Separate the centred object from flat backgrounds and border occluders.

    The eye-in-hand camera points at the workpiece.  We therefore model colours
    seen around the image border as background, remove regions connected to the
    border (for example the robot arm), and retain the central isolated region.
    """

    from scipy.ndimage import binary_fill_holes
    from sklearn.cluster import KMeans
    from skimage.color import rgb2lab
    from skimage.measure import label
    from skimage.morphology import (
        binary_closing,
        binary_opening,
        disk,
        remove_small_objects,
    )

    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    height, width = rgb.shape[:2]
    border_width = max(4, round(min(height, width) * 0.025))
    border_rgb = np.concatenate(
        (
            rgb[:border_width].reshape(-1, 3),
            rgb[-border_width:].reshape(-1, 3),
            rgb[border_width:-border_width, :border_width].reshape(-1, 3),
            rgb[border_width:-border_width, -border_width:].reshape(-1, 3),
        )
    )

    # Fit a compact colour model to the border.  Working in Lab makes the
    # threshold much less sensitive to brightness changes than raw RGB.
    quantized = np.round(border_rgb * 255.0).astype(np.uint8)
    unique_rgb, counts = np.unique(quantized, axis=0, return_counts=True)
    unique_lab = rgb2lab((unique_rgb.astype(np.float32) / 255.0)[None])[0]
    cluster_count = min(8, len(unique_lab))
    if cluster_count == 1:
        background_colours = unique_lab
    else:
        colour_model = KMeans(
            n_clusters=cluster_count,
            n_init=5,
            random_state=0,
            algorithm="lloyd",
        ).fit(unique_lab, sample_weight=counts)
        background_colours = colour_model.cluster_centers_

    image_lab = rgb2lab(rgb)
    distance = np.linalg.norm(
        image_lab[:, :, None, :] - background_colours[None, None, :, :], axis=3
    ).min(axis=2)
    border_lab = rgb2lab(border_rgb[None])[0]
    border_distance = np.linalg.norm(
        border_lab[:, None, :] - background_colours[None, :, :], axis=2
    ).min(axis=1)
    colour_threshold = max(8.0, float(np.percentile(border_distance, 99.5) + 3.0))

    foreground = distance > colour_threshold
    radius = max(2, round(min(height, width) * 0.008))
    foreground = binary_closing(foreground, disk(radius))
    foreground = binary_opening(foreground, disk(1))
    foreground = remove_small_objects(
        foreground, min_size=max(64, round(height * width * 0.001))
    )
    foreground = binary_fill_holes(foreground)

    components = label(foreground, connectivity=2)
    centre_y, centre_x = (height - 1) / 2.0, (width - 1) / 2.0
    max_centre_distance = max(float(np.hypot(centre_y, centre_x)), 1.0)
    best: tuple[float, np.ndarray] | None = None
    for component_id in range(1, int(components.max()) + 1):
        component = components == component_id
        area_fraction = float(component.mean())
        if not 0.003 <= area_fraction <= 0.45:
            continue
        rows, columns = np.nonzero(component)
        boundary_fraction = (
            int(component[0].sum())
            + int(component[-1].sum())
            + int(component[:, 0].sum())
            + int(component[:, -1].sum())
        ) / max(2 * height + 2 * width, 1)
        centre_score = 1.0 - float(
            np.hypot(rows.mean() - centre_y, columns.mean() - centre_x)
            / max_centre_distance
        )
        central_density = float(
            component[
                round(0.30 * height) : round(0.70 * height),
                round(0.30 * width) : round(0.70 * width),
            ].mean()
        )
        score = (
            0.50 * centre_score
            + 0.35 * min(1.0, central_density / 0.10)
            + 0.15 * min(1.0, area_fraction / 0.08)
            - 0.80 * boundary_fraction
        )
        if best is None or score > best[0]:
            best = (score, component)

    if best is None:
        # Uniform images are useful in unit tests and have no separable object.
        # For a real non-uniform frame, silently using the whole image would
        # bring back the background-match failure this function prevents.
        if float(rgb.std(axis=(0, 1)).max()) < 1e-4:
            object_mask = np.ones((height, width), dtype=bool)
        else:
            raise RuntimeError(
                "Could not isolate a central object from the image background."
            )
    else:
        object_mask = best[1]

    rows, columns = np.nonzero(object_mask)
    left, right = int(columns.min()), int(columns.max()) + 1
    top, bottom = int(rows.min()), int(rows.max()) + 1
    padding = round(0.15 * max(right - left, bottom - top))
    bbox = (
        max(0, left - padding),
        max(0, top - padding),
        min(width, right + padding),
        min(height, bottom + padding),
    )
    return _ObjectRegion(mask=object_mask, bbox_xyxy=bbox)


def _region_from_mask(mask, image: Image.Image, label: str) -> _ObjectRegion:
    """Use an explicitly supplied binary object mask and its 15%-padded box."""

    mask = np.asarray(mask)
    if mask.shape != (image.height, image.width):
        raise ValueError(f"{label} mask must match the full image dimensions")
    if not np.isin(mask, (0, 1, 255)).all() or not mask.any():
        raise ValueError(f"{label} mask must be nonempty and binary (0/1/255)")
    mask = mask.astype(bool)
    rows, columns = np.nonzero(mask)
    left, right = int(columns.min()), int(columns.max()) + 1
    top, bottom = int(rows.min()), int(rows.max()) + 1
    padding = round(.15 * max(right - left, bottom - top))
    return _ObjectRegion(mask, (
        max(0, left - padding), max(0, top - padding),
        min(image.width, right + padding), min(image.height, bottom + padding)))


def _extract_object_features(
    extractor: DINOv3FeatureExtractor,
    image: Image.Image,
    region: _ObjectRegion,
) -> DenseFeatures:
    """Extract a dense DINO grid from an enlarged object crop."""

    left, top, right, bottom = region.bbox_xyxy
    crop = image.crop((left, top, right, bottom))
    crop_mask = region.mask[top:bottom, left:right]
    crop_features = extractor.extract(crop, apply_background_filter=False)
    object_features = features_in_mask(
        crop_features, crop_mask, min_mask_fraction=0.15
    )
    return offset_feature_keypoints(
        object_features,
        (left, top),
        (image.height, image.width),
    )


def _dense_object_similarity(
    target_features: DenseFeatures,
    candidate_features: DenseFeatures,
) -> float:
    """Return symmetric best-patch cosine similarity for two object crops."""

    similarities = (
        target_features.descriptors.float()
        @ candidate_features.descriptors.float().T
    )
    target_to_candidate = float(similarities.max(dim=1).values.mean())
    candidate_to_target = float(similarities.max(dim=0).values.mean())
    return float(np.clip((target_to_candidate + candidate_to_target) / 2.0, 0.0, 1.0))


def _save_cluster_visualization(
    image: Image.Image,
    clusters: SpatialClusterResult,
    region: _ObjectRegion,
    output_path: Path,
) -> None:
    """Save input, DINO cluster labels, and selected object mask side by side."""

    image = image.convert("RGB")
    labels = clusters.labels.detach().cpu().numpy()
    mask = region.mask
    palette = np.asarray(
        (
            (0, 114, 178),
            (230, 159, 0),
            (0, 158, 115),
            (204, 121, 167),
            (86, 180, 233),
            (213, 94, 0),
        ),
        dtype=np.uint8,
    )
    label_image = Image.fromarray(palette[labels % len(palette)]).resize(
        image.size, Image.Resampling.NEAREST
    )
    cluster_overlay = Image.blend(image, label_image, 0.45)

    mask_image = Image.fromarray(mask.astype(np.uint8) * 180)
    red_overlay = Image.new("RGB", image.size, (255, 0, 70))
    object_overlay = Image.composite(red_overlay, image, mask_image)
    left, top, right, bottom = region.bbox_xyxy
    ImageDraw.Draw(object_overlay).rectangle(
        (left, top, right - 1, bottom - 1), outline=(0, 255, 0), width=2
    )

    header_height = 28
    canvas = Image.new(
        "RGB", (3 * image.width, image.height + header_height), "white"
    )
    canvas.paste(image, (0, header_height))
    canvas.paste(cluster_overlay, (image.width, header_height))
    canvas.paste(object_overlay, (2 * image.width, header_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), "Input", fill="black")
    draw.text((image.width + 8, 8), "DINO clusters", fill="black")
    draw.text((2 * image.width + 8, 8), "Selected object mask + crop", fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _resized_panel(image: Image.Image, max_size: tuple[int, int]):
    """Resize one image for a comparison panel and return its XY scale."""

    width, height = image.size
    scale = min(max_size[0] / width, max_size[1] / height, 1.0)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.convert("RGB").resize(size, Image.Resampling.LANCZOS), scale


def _save_correspondences(
    target: Image.Image,
    candidate: Image.Image,
    target_xy: np.ndarray,
    candidate_xy: np.ndarray,
    triangle_id: int,
    summary: str,
    output_path: Path,
) -> None:
    """Save target-to-triangle feature correspondences."""

    target_panel, target_scale = _resized_panel(target, (480, 360))
    candidate_panel, candidate_scale = _resized_panel(candidate, (480, 360))
    header_height = 42
    width = target_panel.width + candidate_panel.width
    height = header_height + max(target_panel.height, candidate_panel.height)
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(target_panel, (0, header_height))
    canvas.paste(candidate_panel, (target_panel.width, header_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), "Target image", fill="black")
    draw.text((target_panel.width + 8, 7), f"Triangle {triangle_id}", fill="black")
    draw.text((8, 23), summary, fill="black")

    colors = ((0, 114, 178), (230, 159, 0), (0, 158, 115), (204, 121, 167))
    for index, (first_xy, second_xy) in enumerate(
        zip(target_xy[:100], candidate_xy[:100])
    ):
        first = (
            float(first_xy[0] * target_scale),
            float(header_height + first_xy[1] * target_scale),
        )
        second = (
            float(target_panel.width + second_xy[0] * candidate_scale),
            float(header_height + second_xy[1] * candidate_scale),
        )
        color = colors[index % len(colors)]
        draw.line((first, second), fill=color, width=1)
        radius = 2
        draw.ellipse(
            (first[0] - radius, first[1] - radius, first[0] + radius, first[1] + radius),
            fill=color,
        )
        draw.ellipse(
            (
                second[0] - radius,
                second[1] - radius,
                second[0] + radius,
                second[1] + radius,
            ),
            fill=color,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _dino_correspondences(
    target_features: DenseFeatures,
    candidate_features: DenseFeatures,
) -> tuple[np.ndarray, np.ndarray, int, int, bool]:
    """Return unambiguous object-only DINO matches and geometry diagnostics."""

    ambiguity_options = {}
    if min(len(target_features.descriptors), len(candidate_features.descriptors)) >= 2:
        # DINO patch descriptors are deliberately smooth, so a SIFT-style
        # ratio test removes too many valid neighbouring object patches.  A
        # small bidirectional margin rejects ties; RANSAC handles geometry.
        ambiguity_options = {"min_similarity_margin": 0.002}
    matches = match_features(
        target_features,
        candidate_features,
        similarity_threshold=0.55,
        max_matches=300,
        **ambiguity_options,
    )
    candidate_count = len(matches)
    verified = False
    inlier_count = 0
    displayed = matches
    if candidate_count >= 8:
        try:
            displayed = filter_matches_with_fundamental_ransac(
                matches, reprojection_threshold_px=5.0
            )
            inlier_count = len(displayed)
            verified = True
        except RuntimeError:
            # Keep raw mutual matches visible, but label them as unverified.
            pass
    return (
        displayed.keypoints1.detach().cpu().numpy(),
        displayed.keypoints2.detach().cpu().numpy(),
        candidate_count,
        inlier_count,
        verified,
    )


def _save_best_comparison(
    target: Image.Image,
    candidate: Image.Image,
    best: ViewScore,
    output_path: Path,
) -> None:
    """Save a simple labeled target-versus-winning-triangle image."""

    target_panel, _ = _resized_panel(target, (480, 360))
    candidate_panel, _ = _resized_panel(candidate, (480, 360))
    header_height = 48
    canvas = Image.new(
        "RGB",
        (
            target_panel.width + candidate_panel.width,
            header_height + max(target_panel.height, candidate_panel.height),
        ),
        "white",
    )
    canvas.paste(target_panel, (0, header_height))
    canvas.paste(candidate_panel, (target_panel.width, header_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), "TARGET", fill="black")
    draw.text(
        (target_panel.width + 8, 7),
        f"BEST: triangle {best.triangle}",
        fill="black",
    )
    draw.text(
        (8, 25),
        f"Combined={best.score:.3f}  DINO={best.dino_score:.3f}  SIFT={best.sift_score:.3f}",
        fill="black",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _extract_sift_features(
    image: Image.Image,
    mask: np.ndarray | None,
) -> _SIFTFeatures:
    """Extract masked SIFT features once so target work can be cached."""

    try:
        from skimage.feature import SIFT
    except ImportError as exc:  # pragma: no cover - depends on the runtime image
        raise ImportError("SIFT ranking requires scikit-image.") from exc

    if mask is None:
        left, top, right, bottom = 0, 0, image.width, image.height
        crop_mask = None
    else:
        rows, columns = np.nonzero(mask)
        if not len(rows):
            return _SIFTFeatures(
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, 128), dtype=np.float32),
            )
        left, right = int(columns.min()), int(columns.max()) + 1
        top, bottom = int(rows.min()), int(rows.max()) + 1
        padding = round(0.15 * max(right - left, bottom - top))
        left, top = max(0, left - padding), max(0, top - padding)
        right = min(image.width, right + padding)
        bottom = min(image.height, bottom + padding)
        crop_mask = mask[top:bottom, left:right]

    crop = image.convert("L").crop((left, top, right, bottom))
    mask_coverage = 1.0 if crop_mask is None else float(crop_mask.mean())
    scale = (
        1.0
        if mask_coverage > 0.90
        else min(4.0, max(1.0, 720.0 / max(crop.size)))
    )
    resized_size = (
        max(1, round(crop.width * scale)),
        max(1, round(crop.height * scale)),
    )
    resized = np.asarray(
        crop.resize(resized_size, Image.Resampling.BICUBIC), dtype=np.float32
    ) / 255.0
    detector = SIFT(c_dog=0.005, c_edge=15)
    try:
        detector.detect_and_extract(resized)
    except RuntimeError:
        return _SIFTFeatures(
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 128), dtype=np.float32),
        )
    points = detector.keypoints.astype(np.float32)
    descriptors = detector.descriptors
    if crop_mask is not None:
        resized_mask = np.asarray(
            Image.fromarray(crop_mask.astype(np.uint8) * 255).resize(
                resized_size, Image.Resampling.NEAREST
            )
        )
        indices = np.rint(points).astype(int)
        indices[:, 0] = np.clip(indices[:, 0], 0, resized.shape[0] - 1)
        indices[:, 1] = np.clip(indices[:, 1], 0, resized.shape[1] - 1)
        keep = resized_mask[indices[:, 0], indices[:, 1]] > 0
        points, descriptors = points[keep], descriptors[keep]
    points[:, 0] = points[:, 0] / scale + top
    points[:, 1] = points[:, 1] / scale + left
    return _SIFTFeatures(points, descriptors)


def _compare_sift_features(
    target: _SIFTFeatures,
    candidate: _SIFTFeatures,
) -> SIFTScore:
    """Score two already-extracted SIFT feature sets."""

    try:
        from skimage.feature import match_descriptors
        from skimage.measure import ransac
        from skimage.transform import FundamentalMatrixTransform
    except ImportError as exc:  # pragma: no cover - depends on the runtime image
        raise ImportError("SIFT ranking requires scikit-image.") from exc

    target_points = target.points
    target_descriptors = target.descriptors
    candidate_points = candidate.points
    candidate_descriptors = candidate.descriptors
    target_count = len(target_points)
    candidate_count = len(candidate_points)
    if not target_count or not candidate_count:
        return SIFTScore(0.0, target_count, candidate_count, 0, 0, 0.0)

    matches = match_descriptors(
        target_descriptors,
        candidate_descriptors,
        metric="euclidean",
        max_ratio=0.80,
        cross_check=True,
    )
    good = list(matches)

    if len(good) < 8:
        return SIFTScore(0.0, target_count, candidate_count, len(good), 0, 0.0)

    # scikit-image keypoints use (row, column); the geometry is consistent as
    # long as both images use the same coordinate order.
    target_points = target_points[matches[:, 0]]
    candidate_points = candidate_points[matches[:, 1]]
    try:
        _, inlier_mask = ransac(
            (target_points, candidate_points),
            FundamentalMatrixTransform,
            min_samples=8,
            residual_threshold=2.0,
            max_trials=1000,
            rng=0,
        )
    except (RuntimeError, ValueError):
        inlier_mask = None
    inliers = 0 if inlier_mask is None else int(inlier_mask.sum())
    inlier_ratio = inliers / len(good)
    if inlier_mask is None:
        target_inlier_xy = np.empty((0, 2), dtype=np.float32)
        candidate_inlier_xy = np.empty((0, 2), dtype=np.float32)
    else:
        target_inlier_xy = target_points[inlier_mask][:, ::-1].astype(np.float32)
        candidate_inlier_xy = candidate_points[inlier_mask][:, ::-1].astype(
            np.float32
        )

    # The support term prevents four accidental matches from receiving a high score.
    support = min(1.0, inliers / 12.0)
    score = math.sqrt(inlier_ratio * support)
    return SIFTScore(
        score=float(np.clip(score, 0.0, 1.0)),
        target_keypoints=target_count,
        candidate_keypoints=candidate_count,
        good_matches=len(good),
        inliers=inliers,
        inlier_ratio=float(inlier_ratio),
        target_inlier_xy=target_inlier_xy,
        candidate_inlier_xy=candidate_inlier_xy,
    )


def sift_similarity(
    target: Image.Image,
    candidate: Image.Image,
    target_mask: np.ndarray | None = None,
    candidate_mask: np.ndarray | None = None,
) -> SIFTScore:
    """Compare object regions using SIFT, Lowe filtering, and RANSAC."""

    return _compare_sift_features(
        _extract_sift_features(target, target_mask),
        _extract_sift_features(candidate, candidate_mask),
    )


def combine_scores(
    dino_score: float,
    sift_score: float,
    *,
    dino_weight: float = 0.60,
    sift_weight: float = 0.40,
) -> float:
    """Return a weighted score in [0, 1]."""

    values = (dino_score, sift_score, dino_weight, sift_weight)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Similarity scores and weights must be finite.")
    if not 0.0 <= dino_score <= 1.0 or not 0.0 <= sift_score <= 1.0:
        raise ValueError("Similarity scores must be in [0, 1].")
    if dino_weight < 0.0 or sift_weight < 0.0 or dino_weight + sift_weight <= 0.0:
        raise ValueError("Similarity weights must be non-negative and not both zero.")
    return (dino_weight * dino_score + sift_weight * sift_score) / (
        dino_weight + sift_weight
    )


def prepare_target_image(
    target_path: Path,
    *,
    extractor: DINOv3FeatureExtractor,
    debug: bool = False,
    clusterer: DINOv3SpatialClusterer | None = None,
    target_mask: np.ndarray | None = None,
) -> PreparedTarget:
    """Compute invariant target features once for repeated ranking rounds."""

    resolved_path = target_path.expanduser().resolve()
    target = load_image(resolved_path)
    if target_mask is None:
        target_region = _central_object_region(target)
    else:
        # Real scenes need an explicitly selected object. Colour components can
        # otherwise turn a multi-colour drill into just its orange battery.
        target_region = _region_from_mask(target_mask, target, "target")
    target_features = _extract_object_features(extractor, target, target_region)
    sift_features = _extract_sift_features(target, target_region.mask)
    full_features: DenseFeatures | None = None
    target_clusters: SpatialClusterResult | None = None
    if debug:
        clusterer = clusterer or DINOv3SpatialClusterer()
        full_features = extractor.extract(target, apply_background_filter=False)
        target_clusters = clusterer.fit_predict(full_features, image=target)
    return PreparedTarget(
        path=resolved_path,
        image=target,
        region=target_region,
        object_features=target_features,
        sift_features=sift_features,
        full_features=full_features,
        clusters=target_clusters,
    )


def rank_triangle_views(
    target_path: Path,
    viewpoints_path: Path,
    *,
    output_path: Path | None = None,
    visualization_dir: Path | None = None,
    debug: bool = False,
    ambiguity_margin: float = 0.03,
    dino_weight: float = 0.75,
    sift_weight: float = 0.25,
    extractor: DINOv3FeatureExtractor | None = None,
    clusterer: DINOv3SpatialClusterer | None = None,
    prepared_target: PreparedTarget | None = None,
    target_guided_candidates: bool = False,
    candidate_mask_provider: Callable | None = None,
    sift_scorer: Callable[
        [Image.Image, Image.Image, np.ndarray | None, np.ndarray | None], SIFTScore
    ] = sift_similarity,
) -> dict[str, object]:
    """Rank all saved triangle images and write the winning pose to JSON.

    Set ``debug=True`` to run the additional full-frame DINO extraction and
    clustering passes and save clustering, correspondence, and comparison
    visualizations. These diagnostics are disabled by default because they are
    not required to select the winning viewpoint.
    """

    if not math.isfinite(ambiguity_margin) or not 0.0 <= ambiguity_margin <= 1.0:
        raise ValueError("ambiguity_margin must be in [0, 1].")
    target_path = target_path.expanduser().resolve()
    viewpoints_path = viewpoints_path.expanduser().resolve()
    if not viewpoints_path.is_file():
        raise FileNotFoundError(f"Viewpoint metadata does not exist: {viewpoints_path}")

    triangles = json.loads(viewpoints_path.read_text(encoding="utf-8"))
    if not isinstance(triangles, list) or not triangles:
        raise ValueError("Viewpoint metadata must be a non-empty JSON list.")

    extractor = extractor or DINOv3FeatureExtractor(
        max_image_size=640,
        upscale_small_images=True,
    )
    if prepared_target is not None and prepared_target.path != target_path:
        raise ValueError(
            "Prepared target path does not match target_path: "
            f"{prepared_target.path} != {target_path}."
        )
    if prepared_target is None or (
        debug and prepared_target.clusters is None
    ):
        prepared_target = prepare_target_image(
            target_path,
            extractor=extractor,
            debug=debug,
            clusterer=clusterer,
        )
    target = prepared_target.image
    target_region = prepared_target.region
    target_mask = target_region.mask
    target_features = prepared_target.object_features
    target_sift_features = prepared_target.sift_features

    target_clusters_path: Path | None = None
    if debug:
        clusterer = clusterer or DINOv3SpatialClusterer()
        visualization_dir = visualization_dir or viewpoints_path.with_name(
            f"{viewpoints_path.stem}_visualizations"
        )
        visualization_dir.mkdir(parents=True, exist_ok=True)

        target_clusters = prepared_target.clusters
        if target_clusters is None:  # pragma: no cover - guarded above
            raise RuntimeError("Debug target preparation did not produce clusters.")
        target_clusters_path = visualization_dir / "target_clustering.png"
        _save_cluster_visualization(
            target, target_clusters, target_region, target_clusters_path
        )

    ranking: list[ViewScore] = []
    failures: list[dict[str, object]] = []
    triangle_by_id: dict[int, dict[str, object]] = {}
    candidate_by_id: dict[int, Image.Image] = {}
    visualization_by_id: dict[int, dict[str, str]] = {}
    for triangle in triangles:
        if not isinstance(triangle, dict):
            raise ValueError("Every triangle record must be a JSON object.")
        triangle_id = int(triangle["triangle"])
        if triangle_id in triangle_by_id:
            raise ValueError(f"Duplicate triangle ID: {triangle_id}")
        image_filename = str(triangle.get("image_filename", ""))
        image_path = viewpoints_path.parent / image_filename
        triangle_by_id[triangle_id] = triangle
        try:
            candidate = load_image(image_path)
            if target_guided_candidates:
                # Real scenes must select the reference-matching instance, not
                # whichever foreground component happens to be central.
                if __package__:
                    from .rgb_target_detection import detect_target_region
                else:
                    from rgb_target_detection import detect_target_region
                masks = None
                if triangle.get("mask"):
                    with Image.open(viewpoints_path.parent / triangle["mask"]) as mask_image:
                        masks = [np.asarray(mask_image.convert("L")) > 0]
                elif candidate_mask_provider is not None:
                    masks = candidate_mask_provider(candidate)
                detected = detect_target_region(candidate, target_features, extractor,
                                                candidate_masks=masks, ambiguity_margin=ambiguity_margin)
                candidate_region = _ObjectRegion(detected["mask"], detected["bbox_xyxy"])
            elif triangle.get("mask"):
                # A known-geometry mask (e.g. the projected box on the real
                # robot) is authoritative; no colour-based guessing.
                with Image.open(viewpoints_path.parent / triangle["mask"]) as mask_image:
                    candidate_mask = np.asarray(mask_image.convert("L")) > 0
                candidate_region = _region_from_mask(candidate_mask, candidate, "candidate")
            else:
                candidate_region = _central_object_region(candidate)
            candidate_features = _extract_object_features(
                extractor, candidate, candidate_region
            )
            semantic_score = _dense_object_similarity(
                target_features, candidate_features
            )
            (
                dino_target_xy,
                dino_candidate_xy,
                dino_candidate_count,
                dino_inlier_count,
                dino_verified,
            ) = _dino_correspondences(
                target_features,
                candidate_features,
            )
            if dino_verified and dino_candidate_count:
                inlier_ratio = dino_inlier_count / dino_candidate_count
                support = min(1.0, dino_inlier_count / 30.0)
                geometric_score = math.sqrt(inlier_ratio * support)
            else:
                geometric_score = 0.0
            dino_score = 0.75 * semantic_score + 0.25 * geometric_score
            if sift_scorer is sift_similarity:
                sift = _compare_sift_features(
                    target_sift_features,
                    _extract_sift_features(candidate, candidate_region.mask),
                )
            else:
                sift = sift_scorer(
                    target,
                    candidate,
                    target_mask,
                    candidate_region.mask,
                )
            if debug:
                assert clusterer is not None
                assert visualization_dir is not None
                candidate_full_features = extractor.extract(
                    candidate, apply_background_filter=False
                )
                candidate_clusters = clusterer.fit_predict(
                    candidate_full_features, image=candidate
                )
                clustering_path = (
                    visualization_dir
                    / f"triangle_{triangle_id:02d}_clustering.png"
                )
                dino_matches_path = (
                    visualization_dir
                    / f"triangle_{triangle_id:02d}_dino_matches.png"
                )
                sift_matches_path = (
                    visualization_dir
                    / f"triangle_{triangle_id:02d}_sift_matches.png"
                )
                _save_cluster_visualization(
                    candidate,
                    candidate_clusters,
                    candidate_region,
                    clustering_path,
                )
                dino_summary = (
                    f"DINO: {len(target_features.descriptors)}/"
                    f"{len(candidate_features.descriptors)} object features, "
                    f"{dino_candidate_count} matches, "
                    f"{dino_inlier_count} fundamental-RANSAC inliers"
                    if dino_verified
                    else (
                        f"DINO: {len(target_features.descriptors)}/"
                        f"{len(candidate_features.descriptors)} object features, "
                        f"{dino_candidate_count} matches (geometry unverified)"
                    )
                )
                _save_correspondences(
                    target,
                    candidate,
                    dino_target_xy,
                    dino_candidate_xy,
                    triangle_id,
                    dino_summary,
                    dino_matches_path,
                )
                _save_correspondences(
                    target,
                    candidate,
                    sift.target_inlier_xy,
                    sift.candidate_inlier_xy,
                    triangle_id,
                    (
                        f"SIFT: {sift.target_keypoints}/"
                        f"{sift.candidate_keypoints} object keypoints, "
                        f"{sift.good_matches} good matches, "
                        f"{sift.inliers} fundamental-RANSAC inliers"
                    ),
                    sift_matches_path,
                )
                candidate_by_id[triangle_id] = candidate
                visualization_by_id[triangle_id] = {
                    "clustering": str(clustering_path.resolve()),
                    "feature_matches": str(dino_matches_path.resolve()),
                    "dino_feature_matches": str(dino_matches_path.resolve()),
                    "sift_feature_matches": str(sift_matches_path.resolve()),
                }
            final_score = combine_scores(
                dino_score,
                sift.score,
                dino_weight=dino_weight,
                sift_weight=sift_weight,
            )
            ranking.append(
                ViewScore(
                    triangle=triangle_id,
                    image_filename=image_filename,
                    score=final_score,
                    dino_score=dino_score,
                    dino_semantic_score=semantic_score,
                    dino_geometric_score=geometric_score,
                    dino_target_features=len(target_features.descriptors),
                    dino_candidate_features=len(candidate_features.descriptors),
                    dino_candidate_matches=dino_candidate_count,
                    dino_ransac_inliers=dino_inlier_count,
                    sift_score=sift.score,
                    sift_good_matches=sift.good_matches,
                    sift_inliers=sift.inliers,
                    sift_inlier_ratio=sift.inlier_ratio,
                )
            )
        except Exception as exc:
            failures.append(
                {
                    "triangle": triangle_id,
                    "image_filename": image_filename,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    ranking.sort(key=lambda item: (-item.score, item.triangle))
    if not ranking:
        raise RuntimeError(f"No triangle image could be ranked. Failures: {failures}")

    best = ranking[0]
    plausible = [
        item.triangle
        for item in ranking
        if best.score - item.score <= ambiguity_margin
    ]
    visualizations: dict[str, object] | None = None
    if debug:
        assert visualization_dir is not None
        assert target_clusters_path is not None
        best_comparison_path = (
            visualization_dir / f"best_triangle_{best.triangle:02d}.png"
        )
        _save_best_comparison(
            target,
            candidate_by_id[best.triangle],
            best,
            best_comparison_path,
        )
        visualizations = {
            "directory": str(visualization_dir.resolve()),
            "target_clustering": str(target_clusters_path.resolve()),
            "triangles": {
                str(triangle_id): paths
                for triangle_id, paths in sorted(visualization_by_id.items())
            },
            "best_comparison": str(best_comparison_path.resolve()),
        }
    report: dict[str, object] = {
        "schema_version": 1,
        "target_image": str(target_path),
        "viewpoints_file": str(viewpoints_path),
        "method": (
            ("target-guided instance selection" if target_guided_candidates else "central-object segmentation")
            + " + enlarged-crop DINOv3/ambiguity/RANSAC "
            "+ SIFT/Lowe/RANSAC"
        ),
        "weights": {"dino": dino_weight, "sift": sift_weight},
        "ambiguity_margin": ambiguity_margin,
        "best_triangle": best.triangle,
        "best_image": str(
            (viewpoints_path.parent / best.image_filename).resolve()
        ),
        "plausible_triangles": plausible,
        "best_viewpoint": triangle_by_id[best.triangle],
        "ranking": [asdict(item) for item in ranking],
        "failures": failures,
        "debug_visualizations": debug,
        "visualizations": visualizations,
    }
    output_path = output_path or viewpoints_path.with_name(
        f"{viewpoints_path.stem}_ranking.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(output_path)
    report["report_file"] = str(output_path.resolve())
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rank captured triangle views against a target RGB image."
    )
    parser.add_argument("--target-image", type=Path, required=True)
    parser.add_argument("--viewpoints-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--visualization-dir",
        type=Path,
        default=None,
        help="Directory for artifacts generated by --debug.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Generate expensive clustering, feature-match, and comparison "
            "visualizations."
        ),
    )
    parser.add_argument("--ambiguity-margin", type=float, default=0.03)
    parser.add_argument("--dino-weight", type=float, default=0.75)
    parser.add_argument("--sift-weight", type=float, default=0.25)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = rank_triangle_views(
        args.target_image,
        args.viewpoints_json,
        output_path=args.output,
        visualization_dir=args.visualization_dir,
        debug=args.debug,
        ambiguity_margin=args.ambiguity_margin,
        dino_weight=args.dino_weight,
        sift_weight=args.sift_weight,
    )
    print(f"Best triangle: {report['best_triangle']}")
    print(f"Target image: {report['target_image']}")
    print(f"Best triangle image: {report['best_image']}")
    print(f"Plausible triangles: {report['plausible_triangles']}")
    print(f"Saved ranking: {report['report_file']}")
    visualizations = report["visualizations"]
    if isinstance(visualizations, dict):
        print(f"Saved visual comparisons: {visualizations['directory']}")


if __name__ == "__main__":
    main()
