"""Zero-shot spatial object discovery and DINOv3 object similarity.

This module clusters a complete DINOv3 patch grid with PCA and K-Means,
selects object-like spatial clusters, and implements the directional score

    mean_i max_j cosine(F_current[i], F_target[j]).

The automatic mask is intentionally inspectable and can be overridden with
explicit cluster IDs. No learned segmentation model or task-specific training
is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

if __package__:
    from .dinov3_feature_matching import DenseFeatures
else:
    from dinov3_feature_matching import DenseFeatures


@dataclass(frozen=True)
class ClusterMetrics:
    """Diagnostics used to rank one spatial cluster as foreground/object-like."""

    cluster_id: int
    patch_count: int
    area_fraction: float
    border_fraction: float
    connected_fraction: float
    center_score: float
    foreground_fraction: Optional[float]
    selection_score: float


@dataclass(frozen=True)
class SpatialClusterResult:
    """K-Means labels, selected object mask, and clustering diagnostics."""

    labels: torch.Tensor
    object_mask: torch.Tensor
    object_indices: torch.Tensor
    selected_cluster_ids: Tuple[int, ...]
    metrics: Tuple[ClusterMetrics, ...]
    pca_components: int
    explained_variance_ratio: float
    inertia: float
    random_state: int


@dataclass(frozen=True)
class ObjectSimilarityScore:
    """Directional DINO score plus a reverse and symmetric diagnostic.

    ``score`` is the exact current-to-target score from the proposed method.
    ``best_target_indices`` contains flattened indices in the complete target
    patch grid, rather than indices local to the selected target mask.
    """

    score: float
    reverse_score: float
    symmetric_score: float
    source_patch_count: int
    target_patch_count: int
    per_source_scores: torch.Tensor
    best_target_indices: torch.Tensor


@dataclass(frozen=True)
class AutomaticMaskProposal:
    """One object selected from class-agnostic automatic segmentation masks."""

    mask: torch.Tensor
    bbox_xyxy: Tuple[int, int, int, int]
    model_score: float
    selection_score: float
    area_fraction: float
    center_score: float
    boundary_fraction: float


def select_automatic_object_mask(
    masks,
    scores,
    image_size: Tuple[int, int],
    *,
    min_area_fraction: float = 0.005,
    max_area_fraction: float = 0.70,
    max_boundary_fraction: float = 0.01,
    bbox_padding_fraction: float = 0.15,
) -> AutomaticMaskProposal:
    """Select one central, isolated object from segment-everything proposals.

    The selection is class- and colour-agnostic. Masks covering nearly the
    entire scene or connected to much of the image boundary are rejected;
    remaining masks are ranked by model quality and useful complete-object area.
    ``image_size`` uses ``(height, width)`` order.
    """

    height, width = (int(value) for value in image_size)
    if height <= 0 or width <= 0:
        raise ValueError("image_size dimensions must be positive")
    for value, name in (
        (min_area_fraction, "min_area_fraction"),
        (max_area_fraction, "max_area_fraction"),
        (max_boundary_fraction, "max_boundary_fraction"),
        (bbox_padding_fraction, "bbox_padding_fraction"),
    ):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if min_area_fraction <= 0.0 or max_area_fraction > 1.0:
        raise ValueError("Area fractions must satisfy 0 < min <= max <= 1")
    if min_area_fraction > max_area_fraction:
        raise ValueError("min_area_fraction must not exceed max_area_fraction")
    if max_boundary_fraction > 1.0:
        raise ValueError("max_boundary_fraction must not exceed 1")

    if torch.is_tensor(masks):
        mask_array = masks.detach().cpu().numpy()
    elif isinstance(masks, np.ndarray):
        mask_array = masks
    else:
        mask_array = np.stack(
            [
                mask.detach().cpu().numpy() if torch.is_tensor(mask) else np.asarray(mask)
                for mask in masks
            ]
        )
    if mask_array.ndim == 2:
        mask_array = mask_array[None]
    if mask_array.ndim != 3 or mask_array.shape[1:] != (height, width):
        raise ValueError(
            f"Masks must have shape [count, {height}, {width}], got {mask_array.shape}"
        )
    mask_array = mask_array.astype(bool)
    score_array = torch.as_tensor(scores).detach().float().cpu().numpy().reshape(-1)
    if len(score_array) != len(mask_array):
        raise ValueError("scores must contain one value per mask")
    if not np.isfinite(score_array).all():
        raise ValueError("scores must be finite")

    image_area = height * width
    image_centre_y, image_centre_x = (height - 1) / 2.0, (width - 1) / 2.0
    max_centre_distance = max(float(np.hypot(image_centre_y, image_centre_x)), 1.0)
    boundary_size = max(1, 2 * height + 2 * width - 4)
    ranked = []
    for mask, model_score in zip(mask_array, score_array):
        pixel_count = int(mask.sum())
        area_fraction = pixel_count / image_area
        if not min_area_fraction <= area_fraction <= max_area_fraction:
            continue
        boundary_pixels = (
            int(mask[0].sum())
            + int(mask[-1].sum())
            + int(mask[1:-1, 0].sum())
            + int(mask[1:-1, -1].sum())
        )
        boundary_fraction = boundary_pixels / boundary_size
        if boundary_fraction > max_boundary_fraction:
            continue
        rows, columns = np.nonzero(mask)
        centre_distance = float(
            np.hypot(rows.mean() - image_centre_y, columns.mean() - image_centre_x)
        )
        center_score = float(np.clip(1.0 - centre_distance / max_centre_distance, 0.0, 1.0))
        useful_area_score = min(1.0, area_fraction / 0.15)
        selection_score = (
            0.60 * float(np.clip(model_score, 0.0, 1.0))
            + 0.40 * useful_area_score
        )
        ranked.append(
            (
                selection_score,
                area_fraction,
                mask,
                float(model_score),
                center_score,
                boundary_fraction,
            )
        )

    if not ranked:
        raise RuntimeError(
            "Automatic segmentation produced no isolated object proposal. The "
            "target may fill the frame, touch its boundary, or be one of several "
            "equally plausible objects."
        )
    (
        selection_score,
        area_fraction,
        mask,
        model_score,
        center_score,
        boundary_fraction,
    ) = max(ranked, key=lambda item: (item[0], item[1]))
    rows, columns = np.nonzero(mask)
    left, right = int(columns.min()), int(columns.max()) + 1
    top, bottom = int(rows.min()), int(rows.max()) + 1
    padding = bbox_padding_fraction * max(right - left, bottom - top)
    bbox_xyxy = (
        max(0, int(np.floor(left - padding))),
        max(0, int(np.floor(top - padding))),
        min(width, int(np.ceil(right + padding))),
        min(height, int(np.ceil(bottom + padding))),
    )
    return AutomaticMaskProposal(
        mask=torch.from_numpy(mask),
        bbox_xyxy=bbox_xyxy,
        model_score=model_score,
        selection_score=float(selection_score),
        area_fraction=float(area_fraction),
        center_score=center_score,
        boundary_fraction=float(boundary_fraction),
    )


def plot_automatic_mask_proposal(
    image: Image.Image,
    proposal: AutomaticMaskProposal,
    *,
    figsize: Tuple[float, float] = (12, 5),
):
    """Show the selected class-agnostic mask and its automatic crop."""

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    image = image.convert("RGB")
    mask = proposal.mask.numpy()
    if mask.shape != (image.height, image.width):
        raise ValueError("Proposal mask size does not match image size")
    figure, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].imshow(image)
    axes[0].imshow(
        np.ma.masked_where(~mask, mask), cmap="spring", alpha=0.55, vmin=0, vmax=1
    )
    axes[0].set_title(
        f"Automatic object mask (SAM score {proposal.model_score:.3f})"
    )
    axes[1].imshow(image)
    left, top, right, bottom = proposal.bbox_xyxy
    axes[1].add_patch(
        Rectangle((left, top), right - left, bottom - top, fill=False, color="lime", linewidth=2)
    )
    axes[1].set_title(
        f"Automatic crop (selection score {proposal.selection_score:.3f})"
    )
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    return figure, axes


def _validate_complete_grid(features: DenseFeatures, name: str) -> Tuple[int, int]:
    if features.descriptors.ndim != 2:
        raise ValueError(f"{name}.descriptors must have shape [patches, channels]")
    grid_h, grid_w = map(int, features.grid_size)
    expected = grid_h * grid_w
    actual = int(features.descriptors.shape[0])
    if expected <= 0:
        raise ValueError(f"{name}.grid_size must contain positive dimensions")
    if actual != expected:
        raise ValueError(
            f"{name} contains {actual} descriptors for a {grid_h}x{grid_w} grid "
            f"({expected} required). Extract it with "
            "extract(..., apply_background_filter=False) before spatial clustering."
        )
    if int(features.descriptors.shape[1]) <= 0:
        raise ValueError(f"{name}.descriptors must have a non-empty channel dimension")
    return grid_h, grid_w


def _largest_connected_component(mask: np.ndarray) -> Tuple[np.ndarray, float]:
    """Return the largest 8-connected component and its fraction of ``mask``."""

    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("Connected-component mask must be two-dimensional")
    total = int(mask.sum())
    if total == 0:
        return np.zeros_like(mask), 0.0

    visited = np.zeros_like(mask)
    largest: list[Tuple[int, int]] = []
    height, width = mask.shape
    for start_y, start_x in zip(*np.nonzero(mask)):
        if visited[start_y, start_x]:
            continue
        component: list[Tuple[int, int]] = []
        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        while stack:
            y, x = stack.pop()
            component.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    next_y, next_x = y + dy, x + dx
                    if (
                        0 <= next_y < height
                        and 0 <= next_x < width
                        and mask[next_y, next_x]
                        and not visited[next_y, next_x]
                    ):
                        visited[next_y, next_x] = True
                        stack.append((next_y, next_x))

        if len(component) > len(largest):
            largest = component

    result = np.zeros_like(mask)
    for y, x in largest:
        result[y, x] = True
    return result, len(largest) / total


def _patch_foreground_fractions(
    image: Image.Image,
    features: DenseFeatures,
    white_background_threshold: float,
) -> np.ndarray:
    expected_h, expected_w = map(int, features.original_size)
    if image.size != (expected_w, expected_h):
        raise ValueError(
            f"image size {image.size} does not match feature input size "
            f"{(expected_w, expected_h)}"
        )

    grid_h, grid_w = map(int, features.grid_size)
    processed_h, processed_w = map(int, features.processed_size)
    if processed_h % grid_h or processed_w % grid_w:
        raise ValueError("Processed image size is not divisible by the DINO patch grid")
    patch_h, patch_w = processed_h // grid_h, processed_w // grid_w

    grayscale = image.convert("L").resize(
        (processed_w, processed_h), Image.Resampling.BILINEAR
    )
    intensities = np.asarray(grayscale, dtype=np.float32) / 255.0
    foreground = intensities < white_background_threshold
    return (
        foreground.reshape(grid_h, patch_h, grid_w, patch_w)
        .transpose(0, 2, 1, 3)
        .mean(axis=(2, 3))
    )


class DINOv3SpatialClusterer:
    """Discover an object mask by deterministic PCA + K-Means clustering.

    The automatic selector combines image-domain foreground coverage with
    spatial coherence, centrality, and border contact. On a white-background
    image, all eligible clusters with sufficient non-white coverage are joined;
    this is useful when K-Means splits one object into several semantic parts.
    Pass ``object_cluster_ids`` to :meth:`fit_predict` to override that choice.
    """

    def __init__(
        self,
        *,
        n_clusters: int = 4,
        pca_components: int = 64,
        random_state: int = 0,
        n_init: int = 20,
        white_background_threshold: float = 0.95,
        min_cluster_foreground_fraction: float = 0.35,
        min_cluster_fraction: float = 0.01,
        max_cluster_fraction: float = 0.85,
        keep_largest_component: bool = True,
    ) -> None:
        if n_clusters < 2:
            raise ValueError("n_clusters must be at least 2")
        if pca_components < 1:
            raise ValueError("pca_components must be positive")
        if n_init < 1:
            raise ValueError("n_init must be positive")
        for value, name in (
            (white_background_threshold, "white_background_threshold"),
            (min_cluster_foreground_fraction, "min_cluster_foreground_fraction"),
            (min_cluster_fraction, "min_cluster_fraction"),
            (max_cluster_fraction, "max_cluster_fraction"),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if min_cluster_fraction > max_cluster_fraction:
            raise ValueError("min_cluster_fraction must not exceed max_cluster_fraction")

        self.n_clusters = int(n_clusters)
        self.pca_components = int(pca_components)
        self.random_state = int(random_state)
        self.n_init = int(n_init)
        self.white_background_threshold = float(white_background_threshold)
        self.min_cluster_foreground_fraction = float(
            min_cluster_foreground_fraction
        )
        self.min_cluster_fraction = float(min_cluster_fraction)
        self.max_cluster_fraction = float(max_cluster_fraction)
        self.keep_largest_component = bool(keep_largest_component)

    def fit_predict(
        self,
        features: DenseFeatures,
        *,
        image: Optional[Image.Image] = None,
        object_cluster_ids: Optional[Sequence[int]] = None,
    ) -> SpatialClusterResult:
        """Cluster a complete patch grid and return an object-mask hypothesis."""

        grid_h, grid_w = _validate_complete_grid(features, "features")
        sample_count, channel_count = map(int, features.descriptors.shape)
        if sample_count < self.n_clusters:
            raise ValueError(
                f"n_clusters={self.n_clusters} exceeds the {sample_count} patch descriptors"
            )

        try:
            from sklearn.cluster import KMeans
            from sklearn.decomposition import PCA
        except ImportError as exc:
            raise ImportError(
                "Spatial clustering requires scikit-learn==1.3.2. Install "
                "requirements-dinov3.txt first."
            ) from exc

        descriptors = features.descriptors.detach().float().cpu().numpy()
        if not np.isfinite(descriptors).all():
            raise ValueError("Feature descriptors contain NaN or infinite values")

        component_count = min(self.pca_components, sample_count, channel_count)
        pca = PCA(n_components=component_count, svd_solver="full")
        reduced = pca.fit_transform(descriptors)
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            n_init=self.n_init,
            random_state=self.random_state,
            algorithm="lloyd",
        )
        labels_np = kmeans.fit_predict(reduced).reshape(grid_h, grid_w)

        patch_foreground = None
        if image is not None:
            if not isinstance(image, Image.Image):
                raise TypeError(f"image must be a PIL.Image.Image, got {type(image).__name__}")
            patch_foreground = _patch_foreground_fractions(
                image, features, self.white_background_threshold
            )

        max_center_distance = float(np.hypot(max((grid_h - 1) / 2, 0.5), max((grid_w - 1) / 2, 0.5)))
        center_y, center_x = (grid_h - 1) / 2, (grid_w - 1) / 2
        metrics = []
        for cluster_id in range(self.n_clusters):
            mask = labels_np == cluster_id
            patch_count = int(mask.sum())
            area_fraction = patch_count / sample_count
            rows, columns = np.nonzero(mask)
            on_border = (
                (rows == 0)
                | (rows == grid_h - 1)
                | (columns == 0)
                | (columns == grid_w - 1)
            )
            border_fraction = float(on_border.mean()) if patch_count else 1.0
            _, connected_fraction = _largest_connected_component(mask)
            distance = np.hypot(rows - center_y, columns - center_x)
            center_score = (
                float(np.clip(1.0 - distance.mean() / max_center_distance, 0.0, 1.0))
                if patch_count
                else 0.0
            )
            foreground_fraction = (
                float(patch_foreground[mask].mean())
                if patch_foreground is not None and patch_count
                else None
            )
            foreground_score = (
                foreground_fraction if foreground_fraction is not None else 0.5
            )
            selection_score = (
                0.45 * foreground_score
                + 0.20 * connected_fraction
                + 0.20 * center_score
                + 0.15 * (1.0 - border_fraction)
            )
            metrics.append(
                ClusterMetrics(
                    cluster_id=cluster_id,
                    patch_count=patch_count,
                    area_fraction=area_fraction,
                    border_fraction=border_fraction,
                    connected_fraction=connected_fraction,
                    center_score=center_score,
                    foreground_fraction=foreground_fraction,
                    selection_score=selection_score,
                )
            )

        if object_cluster_ids is not None:
            selected = tuple(sorted({int(value) for value in object_cluster_ids}))
            if not selected:
                raise ValueError("object_cluster_ids must not be empty")
            invalid = [value for value in selected if not 0 <= value < self.n_clusters]
            if invalid:
                raise ValueError(
                    f"object_cluster_ids contains invalid IDs {invalid}; expected "
                    f"values from 0 to {self.n_clusters - 1}"
                )
        else:
            eligible = [
                metric
                for metric in metrics
                if self.min_cluster_fraction
                <= metric.area_fraction
                <= self.max_cluster_fraction
            ]
            if not eligible:
                eligible = list(metrics)
            selected_metrics = []
            if patch_foreground is not None:
                selected_metrics = [
                    metric
                    for metric in eligible
                    if metric.foreground_fraction is not None
                    and metric.foreground_fraction
                    >= self.min_cluster_foreground_fraction
                ]
            if not selected_metrics:
                selected_metrics = [
                    max(eligible, key=lambda metric: metric.selection_score)
                ]
            selected = tuple(sorted(metric.cluster_id for metric in selected_metrics))

        object_mask_np = np.isin(labels_np, selected)
        if self.keep_largest_component:
            object_mask_np, _ = _largest_connected_component(object_mask_np)
        if not object_mask_np.any():
            raise RuntimeError("The selected clusters produced an empty object mask")

        object_indices = torch.from_numpy(
            np.flatnonzero(object_mask_np.reshape(-1)).astype(np.int64)
        )
        return SpatialClusterResult(
            labels=torch.from_numpy(labels_np.astype(np.int64)),
            object_mask=torch.from_numpy(object_mask_np),
            object_indices=object_indices,
            selected_cluster_ids=selected,
            metrics=tuple(metrics),
            pca_components=component_count,
            explained_variance_ratio=float(pca.explained_variance_ratio_.sum()),
            inertia=float(kmeans.inertia_),
            random_state=self.random_state,
        )


def format_cluster_report(result: SpatialClusterResult) -> str:
    """Create a compact, notebook-friendly table of mask-selection evidence."""

    selected = set(result.selected_cluster_ids)
    lines = [
        "    cluster  patches    area  foreground  border  connected  center   score",
        "    -------  -------  ------  ----------  ------  ---------  ------  ------",
    ]
    for metric in result.metrics:
        marker = "*" if metric.cluster_id in selected else " "
        foreground = (
            f"{metric.foreground_fraction:10.3f}"
            if metric.foreground_fraction is not None
            else "       n/a"
        )
        lines.append(
            f"{marker}   {metric.cluster_id:7d}  {metric.patch_count:7d}  "
            f"{metric.area_fraction:6.3f}  {foreground}  "
            f"{metric.border_fraction:6.3f}  {metric.connected_fraction:9.3f}  "
            f"{metric.center_score:6.3f}  {metric.selection_score:6.3f}"
        )
    lines.append(
        "* selected before connected-component cleanup; "
        f"final object mask: {len(result.object_indices)} patches"
    )
    return "\n".join(lines)


def plot_spatial_clusters(
    image: Image.Image,
    result: SpatialClusterResult,
    *,
    figsize: Tuple[float, float] = (15, 5),
):
    """Visualize the image, all cluster labels, and the final object mask."""

    if not isinstance(image, Image.Image):
        raise TypeError(f"image must be a PIL.Image.Image, got {type(image).__name__}")
    if result.labels.ndim != 2 or result.object_mask.shape != result.labels.shape:
        raise ValueError("labels and object_mask must be same-sized two-dimensional grids")

    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    width, height = image.size
    extent = (0, width, height, 0)
    labels = result.labels.cpu().numpy()
    mask = result.object_mask.cpu().numpy()
    figure, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].imshow(image.convert("RGB"))
    axes[0].set_title("Input")

    axes[1].imshow(image.convert("RGB"))
    axes[1].imshow(
        labels,
        cmap="tab10",
        alpha=0.55,
        interpolation="nearest",
        extent=extent,
    )
    axes[1].set_title("PCA + K-Means labels")

    axes[2].imshow(image.convert("RGB"))
    overlay = np.ma.masked_where(~mask, mask.astype(float))
    axes[2].imshow(
        overlay,
        cmap=ListedColormap(["#ff2d55"]),
        alpha=0.55,
        interpolation="nearest",
        extent=extent,
        vmin=0,
        vmax=1,
    )
    axes[2].set_title(
        f"Object mask: clusters {result.selected_cluster_ids}\n"
        f"{len(result.object_indices)} patches"
    )

    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    return figure, axes


def _directional_max_cosine(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    scores = []
    indices = []
    for start in range(0, len(source), chunk_size):
        similarities = source[start : start + chunk_size] @ target.T
        chunk_scores, chunk_indices = similarities.max(dim=1)
        scores.append(chunk_scores.cpu())
        indices.append(chunk_indices.cpu())
    return torch.cat(scores), torch.cat(indices)


def compute_object_similarity(
    current_features: DenseFeatures,
    current_clusters: SpatialClusterResult,
    target_features: DenseFeatures,
    target_clusters: Optional[SpatialClusterResult] = None,
    *,
    chunk_size: int = 4096,
) -> ObjectSimilarityScore:
    """Compute object-level maximum-cosine similarity between two patch grids.

    The primary ``score`` is directional: for each current-object patch, take
    its best target-object patch cosine similarity, then average. When
    ``target_clusters`` is omitted, all target patches are candidates. The
    reverse and mean-of-directions values are returned to expose asymmetric
    coverage (for example, a partial view matching a complete view).
    """

    _validate_complete_grid(current_features, "current_features")
    _validate_complete_grid(target_features, "target_features")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if current_features.descriptors.shape[1] != target_features.descriptors.shape[1]:
        raise ValueError("Current and target descriptor channel dimensions do not match")

    current_indices = current_clusters.object_indices.detach().long().cpu()
    if target_clusters is None:
        target_indices = torch.arange(len(target_features.descriptors), dtype=torch.long)
    else:
        target_indices = target_clusters.object_indices.detach().long().cpu()
    if len(current_indices) == 0 or len(target_indices) == 0:
        raise ValueError("Current and target object masks must contain at least one patch")
    if current_indices.min() < 0 or current_indices.max() >= len(current_features.descriptors):
        raise ValueError("current_clusters.object_indices is outside the current feature grid")
    if target_indices.min() < 0 or target_indices.max() >= len(target_features.descriptors):
        raise ValueError("target_clusters.object_indices is outside the target feature grid")

    current = F.normalize(
        current_features.descriptors[current_indices].detach().float().cpu(), dim=1
    )
    target = F.normalize(
        target_features.descriptors[target_indices].detach().float().cpu(), dim=1
    )
    forward_scores, forward_local_indices = _directional_max_cosine(
        current, target, chunk_size=chunk_size
    )
    reverse_scores, _ = _directional_max_cosine(
        target, current, chunk_size=chunk_size
    )
    forward = float(forward_scores.mean())
    reverse = float(reverse_scores.mean())
    return ObjectSimilarityScore(
        score=forward,
        reverse_score=reverse,
        symmetric_score=(forward + reverse) / 2.0,
        source_patch_count=len(current),
        target_patch_count=len(target),
        per_source_scores=forward_scores,
        best_target_indices=target_indices[forward_local_indices],
    )
