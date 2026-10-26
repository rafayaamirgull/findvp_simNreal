"""DINOv3 dense RGB feature extraction and geometrically verified matching.

The default model is Meta's small ViT/16 DINOv3 backbone exposed by timm. The
extractor keeps the image aspect ratio, aligns its size to the model patch size,
and returns one descriptor at the centre of every patch. Object ROIs, ambiguity
tests, and fundamental-matrix RANSAC remove common background false matches.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


DEFAULT_MODEL_ID = "vit_small_patch16_dinov3.lvd1689m"


@dataclass(frozen=True)
class DenseFeatures:
    """Dense DINOv3 features and their image-space locations.

    Sizes use ``(height, width)`` order. Keypoints use ``(x, y)`` order and
    refer to the original, unscaled input image.
    """

    descriptors: torch.Tensor
    keypoints: torch.Tensor
    global_descriptor: torch.Tensor
    grid_size: Tuple[int, int]
    original_size: Tuple[int, int]
    processed_size: Tuple[int, int]


@dataclass(frozen=True)
class FeatureMatches:
    """Corresponding DINOv3 patches in two images."""

    keypoints1: torch.Tensor
    keypoints2: torch.Tensor
    scores: torch.Tensor
    indices1: torch.Tensor
    indices2: torch.Tensor

    def __len__(self) -> int:
        return int(self.scores.shape[0])


@dataclass(frozen=True)
class ImageMatchScore:
    """Auditable fusion of DINO matches, geometry, and local RGB-image ZNCC."""

    score: float
    feature_score: float
    descriptor_score: float
    mean_descriptor_similarity: float
    geometric_score: float
    geometric_inlier_ratio: float
    median_zncc: float
    zncc_score: float
    candidate_count: int
    inlier_count: int
    zncc_patch_count: int
    per_match_zncc: np.ndarray


def load_image(path: Union[str, Path], *, grayscale: bool = False) -> Image.Image:
    """Read an image path as an independent three-channel PIL image.

    When ``grayscale`` is true, luminance is replicated across three channels
    because pretrained DINOv3 backbones still require three-channel input.
    """

    image_path = Path(path).expanduser()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image does not exist or is not a file: {image_path}")
    with Image.open(image_path) as image:
        if grayscale:
            return image.convert("L").convert("RGB").copy()
        return image.convert("RGB").copy()


def zero_mean_normalized_cross_correlation(
    first: np.ndarray,
    second: np.ndarray,
    *,
    minimum_standard_deviation: float = 1e-6,
) -> float:
    """Return ZNCC in ``[-1, 1]`` for equal-shaped numeric arrays.

    Constant or nearly constant arrays have no defined correlation and are
    rejected instead of being assigned a misleading perfect score.
    """

    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    if first_array.shape != second_array.shape or first_array.size == 0:
        raise ValueError("ZNCC inputs must have the same non-empty shape")
    if not np.isfinite(first_array).all() or not np.isfinite(second_array).all():
        raise ValueError("ZNCC inputs must contain only finite values")
    if (
        not np.isfinite(minimum_standard_deviation)
        or minimum_standard_deviation <= 0.0
    ):
        raise ValueError("minimum_standard_deviation must be finite and positive")

    first_centered = first_array.reshape(-1) - float(first_array.mean())
    second_centered = second_array.reshape(-1) - float(second_array.mean())
    first_std = float(np.sqrt(np.mean(first_centered * first_centered)))
    second_std = float(np.sqrt(np.mean(second_centered * second_centered)))
    if min(first_std, second_std) < minimum_standard_deviation:
        raise ValueError("ZNCC is undefined for a near-constant input")
    denominator = float(np.linalg.norm(first_centered) * np.linalg.norm(second_centered))
    correlation = float(np.dot(first_centered, second_centered) / denominator)
    return float(np.clip(correlation, -1.0, 1.0))


def features_in_roi(
    features: DenseFeatures,
    roi_xyxy: Tuple[float, float, float, float],
) -> DenseFeatures:
    """Keep patch descriptors whose centres fall inside an image-space ROI.

    The ROI is ``(left, top, right, bottom)`` in original-image pixels. Returned
    keypoints remain in the original full-image coordinate system, so match
    visualization needs no crop offset correction.
    """

    if features.descriptors.ndim != 2 or features.keypoints.ndim != 2:
        raise ValueError("Feature descriptors and keypoints must be two-dimensional")
    if features.keypoints.shape != (len(features.descriptors), 2):
        raise ValueError("Feature keypoints must have shape [patches, 2]")
    if len(roi_xyxy) != 4:
        raise ValueError("roi_xyxy must contain left, top, right, and bottom")
    left, top, right, bottom = (float(value) for value in roi_xyxy)
    image_h, image_w = features.original_size
    if not all(np.isfinite((left, top, right, bottom))):
        raise ValueError("ROI coordinates must be finite")
    if not (0.0 <= left < right <= image_w and 0.0 <= top < bottom <= image_h):
        raise ValueError(
            f"ROI {(left, top, right, bottom)} is outside the "
            f"{image_w}x{image_h} image"
        )

    x, y = features.keypoints[:, 0], features.keypoints[:, 1]
    keep = (x >= left) & (x < right) & (y >= top) & (y < bottom)
    if not bool(torch.any(keep)):
        raise ValueError("ROI contains no DINOv3 patch centres")
    return DenseFeatures(
        descriptors=features.descriptors[keep],
        keypoints=features.keypoints[keep],
        global_descriptor=features.global_descriptor,
        grid_size=features.grid_size,
        original_size=features.original_size,
        processed_size=features.processed_size,
    )


def features_in_mask(
    features: DenseFeatures,
    mask: Union[Image.Image, np.ndarray],
    *,
    min_mask_fraction: float = 0.25,
) -> DenseFeatures:
    """Keep complete-grid features covered by a binary foreground mask.

    The mask must have the feature input image's original size. It is resized
    exactly like the model input and averaged within every model patch.
    """

    if not 0.0 < min_mask_fraction <= 1.0:
        raise ValueError("min_mask_fraction must be in (0, 1]")
    grid_h, grid_w = features.grid_size
    if len(features.descriptors) != grid_h * grid_w:
        raise ValueError("Mask filtering requires a complete row-major feature grid")
    if features.keypoints.shape != (len(features.descriptors), 2):
        raise ValueError("Feature keypoints must have shape [patches, 2]")

    mask_array = np.asarray(mask.convert("L") if isinstance(mask, Image.Image) else mask)
    if mask_array.ndim != 2:
        raise ValueError("mask must be a two-dimensional image or array")
    original_h, original_w = features.original_size
    if mask_array.shape != (original_h, original_w):
        raise ValueError(
            f"Mask shape {mask_array.shape} does not match feature image size "
            f"{features.original_size}"
        )
    if not np.isfinite(mask_array).all():
        raise ValueError("mask values must be finite")
    mask_array = mask_array.astype(np.float32)
    if mask_array.max(initial=0.0) > 1.0:
        mask_array /= 255.0
    mask_array = np.clip(mask_array, 0.0, 1.0)

    processed_h, processed_w = features.processed_size
    if processed_h % grid_h or processed_w % grid_w:
        raise ValueError("Processed image dimensions do not align with the feature grid")
    resized_mask = Image.fromarray(mask_array, mode="F").resize(
        (processed_w, processed_h), Image.Resampling.BILINEAR
    )
    patch_h, patch_w = processed_h // grid_h, processed_w // grid_w
    fractions = np.asarray(resized_mask, dtype=np.float32).reshape(
        grid_h, patch_h, grid_w, patch_w
    ).transpose(0, 2, 1, 3).mean(axis=(2, 3))
    keep = torch.from_numpy(fractions.reshape(-1) >= min_mask_fraction)
    if not bool(torch.any(keep)):
        raise ValueError("Mask contains no sufficiently covered DINOv3 patches")
    return DenseFeatures(
        descriptors=features.descriptors[keep],
        keypoints=features.keypoints[keep],
        global_descriptor=features.global_descriptor,
        grid_size=features.grid_size,
        original_size=features.original_size,
        processed_size=features.processed_size,
    )


def offset_feature_keypoints(
    features: DenseFeatures,
    offset_xy: Tuple[float, float],
    full_image_size: Tuple[int, int],
) -> DenseFeatures:
    """Map crop-local keypoints into a full image without changing descriptors.

    ``full_image_size`` uses ``(height, width)`` order, while ``offset_xy`` is
    the crop's ``(left, top)`` location in full-image pixels.
    """

    offset_x, offset_y = (float(value) for value in offset_xy)
    full_h, full_w = (int(value) for value in full_image_size)
    crop_h, crop_w = features.original_size
    if not np.isfinite((offset_x, offset_y)).all():
        raise ValueError("Crop offset must be finite")
    if full_h <= 0 or full_w <= 0:
        raise ValueError("Full image dimensions must be positive")
    if not (
        0.0 <= offset_x
        and 0.0 <= offset_y
        and offset_x + crop_w <= full_w
        and offset_y + crop_h <= full_h
    ):
        raise ValueError("Crop dimensions and offset fall outside the full image")
    keypoints = features.keypoints
    if not torch.is_floating_point(keypoints):
        keypoints = keypoints.float()
    offset = torch.tensor(
        (offset_x, offset_y),
        dtype=keypoints.dtype,
        device=keypoints.device,
    )
    return DenseFeatures(
        descriptors=features.descriptors,
        keypoints=keypoints + offset,
        global_descriptor=features.global_descriptor,
        grid_size=features.grid_size,
        original_size=(full_h, full_w),
        processed_size=features.processed_size,
    )


def _pair(value: Any, name: str) -> Tuple[int, int]:
    if isinstance(value, int):
        pair = (value, value)
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        pair = (int(value[0]), int(value[1]))
    else:
        raise ValueError(f"{name} must be an int or a two-element sequence, got {value!r}")
    if pair[0] <= 0 or pair[1] <= 0:
        raise ValueError(f"{name} values must be positive, got {pair!r}")
    return pair


def _automatic_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DINOv3FeatureExtractor:
    """Load a frozen DINOv3 ViT and extract its dense patch features.

    Args:
        model_id: A timm DINOv3 model name.
        device: Torch device. ``None`` selects CUDA, then MPS, then CPU.
        max_image_size: Maximum processed image edge in pixels. By default,
            images smaller than this are not enlarged. Both edges are aligned
            to the patch size.
        upscale_small_images: Enlarge images whose longest edge is below
            ``max_image_size`` to produce a denser DINOv3 patch grid.
        white_background_threshold: If set, pixels with grayscale intensity at
            or above this normalized value are background. Patches without the
            required foreground coverage are removed from the returned features.
        min_foreground_fraction: Minimum non-background fraction required to
            retain a patch when ``white_background_threshold`` is set.
        model: Optional already-created model, primarily for offline/testing use.
        processor: Optional processor paired with an injected model. Production
            timm models use the preprocessing metadata bundled with the model.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        device: Optional[Union[str, torch.device]] = None,
        max_image_size: int = 448,
        upscale_small_images: bool = False,
        white_background_threshold: Optional[float] = None,
        min_foreground_fraction: float = 0.10,
        model: Optional[Any] = None,
        processor: Optional[Any] = None,
    ) -> None:
        if max_image_size <= 0:
            raise ValueError("max_image_size must be positive")
        if model is None and processor is not None:
            raise ValueError("processor can only be supplied with an injected model")
        if white_background_threshold is not None and not 0.0 <= white_background_threshold <= 1.0:
            raise ValueError("white_background_threshold must be between 0 and 1 or None")
        if not 0.0 <= min_foreground_fraction <= 1.0:
            raise ValueError("min_foreground_fraction must be between 0 and 1")

        self.device = torch.device(device) if device is not None else _automatic_device()
        self.max_image_size = int(max_image_size)
        self.upscale_small_images = bool(upscale_small_images)
        self.white_background_threshold = white_background_threshold
        self.min_foreground_fraction = float(min_foreground_fraction)

        if model is None:
            try:
                import timm
            except ImportError as exc:
                raise ImportError(
                    "DINOv3 extraction requires timm==1.0.20. Install "
                    "requirements-dinov3.txt first."
                ) from exc
            model = timm.create_model(
                model_id,
                pretrained=True,
                num_classes=0,
                dynamic_img_size=True,
            )

        self.processor = processor
        self.model = model.to(self.device).eval()
        config_patch_size = getattr(getattr(self.model, "config", None), "patch_size", None)
        patch_embed_size = getattr(getattr(self.model, "patch_embed", None), "patch_size", None)
        self.patch_size = _pair(
            config_patch_size if config_patch_size is not None else patch_embed_size,
            "patch_size",
        )

        pretrained_config = getattr(self.model, "pretrained_cfg", {})
        self.image_mean = tuple(pretrained_config.get("mean", (0.485, 0.456, 0.406)))
        self.image_std = tuple(pretrained_config.get("std", (0.229, 0.224, 0.225)))

    def _resize_for_model(self, image: Image.Image) -> Image.Image:
        patch_h, patch_w = self.patch_size
        width, height = image.size
        if width < patch_w or height < patch_h:
            raise ValueError(
                f"Image size {image.size} is smaller than the model patch size "
                f"(width={patch_w}, height={patch_h})"
            )

        requested_scale = self.max_image_size / max(width, height)
        scale = requested_scale if self.upscale_small_images else min(1.0, requested_scale)
        target_w = max(patch_w, int(width * scale) // patch_w * patch_w)
        target_h = max(patch_h, int(height * scale) // patch_h * patch_h)
        if (target_w, target_h) == image.size:
            return image
        return image.resize((target_w, target_h), Image.Resampling.BICUBIC)

    def extract(
        self,
        image: Image.Image,
        *,
        apply_background_filter: bool = True,
    ) -> DenseFeatures:
        """Extract L2-normalized global and spatial features from one image.

        Set ``apply_background_filter=False`` when a downstream method needs a
        complete row-major feature grid, such as spatial clustering.
        """

        if not isinstance(image, Image.Image):
            raise TypeError(f"image must be a PIL.Image.Image, got {type(image).__name__}")
        image = image.convert("RGB")
        original_w, original_h = image.size
        resized = self._resize_for_model(image)

        if self.processor is not None:
            # Keep support for explicitly injected processor/model pairs.
            inputs = self.processor(
                images=resized,
                return_tensors="pt",
                do_resize=False,
                do_center_crop=False,
            )
            if "pixel_values" not in inputs:
                raise RuntimeError("The image processor did not return pixel_values")
        else:
            image_array = np.asarray(resized, dtype=np.float32)
            pixel_values = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0) / 255.0
            mean = torch.tensor(self.image_mean, dtype=torch.float32).view(1, 3, 1, 1)
            std = torch.tensor(self.image_std, dtype=torch.float32).view(1, 3, 1, 1)
            inputs = {"pixel_values": (pixel_values - mean) / std}
        inputs = {
            name: value.to(self.device) if torch.is_tensor(value) else value
            for name, value in inputs.items()
        }
        pixel_values = inputs["pixel_values"]
        if pixel_values.ndim != 4:
            raise RuntimeError(
                f"Expected pixel_values with shape [B, C, H, W], got {tuple(pixel_values.shape)}"
            )
        processed_h, processed_w = map(int, pixel_values.shape[-2:])
        patch_h, patch_w = self.patch_size
        grid_h, grid_w = processed_h // patch_h, processed_w // patch_w
        patch_count = grid_h * grid_w

        with torch.inference_mode():
            if self.processor is None:
                tokens = self.model.forward_features(pixel_values)
            else:
                outputs = self.model(**inputs)
                tokens = getattr(outputs, "last_hidden_state", None)
        if tokens is None or tokens.ndim != 3 or tokens.shape[0] != 1:
            shape = None if tokens is None else tuple(tokens.shape)
            raise RuntimeError(
                "Expected a DINOv3 ViT last_hidden_state with shape [1, tokens, channels], "
                f"got {shape}. Use a DINOv3 ViT checkpoint, not a ConvNeXt checkpoint."
            )
        prefix_count = int(tokens.shape[1]) - patch_count
        if prefix_count < 1:
            raise RuntimeError(
                f"Model returned {tokens.shape[1]} tokens, but the {grid_h}x{grid_w} "
                f"patch grid requires {patch_count} plus at least one class token"
            )

        descriptors = F.normalize(tokens[0, prefix_count:].float(), dim=-1).cpu()
        global_descriptor = F.normalize(tokens[0, 0].float(), dim=-1).cpu()

        rows, columns = torch.meshgrid(
            torch.arange(grid_h, dtype=torch.float32),
            torch.arange(grid_w, dtype=torch.float32),
            indexing="ij",
        )
        x = (columns.reshape(-1) + 0.5) * patch_w * original_w / processed_w
        y = (rows.reshape(-1) + 0.5) * patch_h * original_h / processed_h
        keypoints = torch.stack((x, y), dim=1)

        if apply_background_filter and self.white_background_threshold is not None:
            grayscale = resized.convert("L")
            if grayscale.size != (processed_w, processed_h):
                grayscale = grayscale.resize(
                    (processed_w, processed_h), Image.Resampling.BILINEAR
                )
            intensities = np.asarray(grayscale, dtype=np.float32) / 255.0
            foreground = intensities < self.white_background_threshold
            foreground_fraction = foreground.reshape(
                grid_h, patch_h, grid_w, patch_w
            ).transpose(0, 2, 1, 3).mean(axis=(2, 3))
            keep = torch.from_numpy(
                foreground_fraction.reshape(-1) >= self.min_foreground_fraction
            )
            descriptors = descriptors[keep]
            keypoints = keypoints[keep]
            if len(descriptors) == 0:
                raise ValueError(
                    "No foreground patches remain; lower min_foreground_fraction "
                    "or raise white_background_threshold"
                )

        return DenseFeatures(
            descriptors=descriptors,
            keypoints=keypoints,
            global_descriptor=global_descriptor,
            grid_size=(grid_h, grid_w),
            original_size=(original_h, original_w),
            processed_size=(processed_h, processed_w),
        )


def match_features(
    features1: DenseFeatures,
    features2: DenseFeatures,
    *,
    similarity_threshold: float = 0.70,
    max_matches: Optional[int] = 200,
    min_similarity_margin: float = 0.0,
    max_distance_ratio: Optional[float] = None,
) -> FeatureMatches:
    """Match dense features with mutual-nearest and ambiguity filtering.

    ``min_similarity_margin`` requires the best cosine score to exceed the
    second-best score in both directions. ``max_distance_ratio`` applies the
    ratio test to cosine distances ``1 - similarity`` in both directions.
    """

    if not -1.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be between -1 and 1")
    if max_matches is not None and max_matches <= 0:
        raise ValueError("max_matches must be positive or None")
    if not 0.0 <= min_similarity_margin <= 2.0:
        raise ValueError("min_similarity_margin must be between 0 and 2")
    if max_distance_ratio is not None and not 0.0 < max_distance_ratio <= 1.0:
        raise ValueError("max_distance_ratio must be in (0, 1] or None")
    if features1.descriptors.ndim != 2 or features2.descriptors.ndim != 2:
        raise ValueError("Both descriptor tensors must have shape [patches, channels]")
    if features1.descriptors.shape[1] != features2.descriptors.shape[1]:
        raise ValueError("Descriptor channel dimensions do not match")
    if len(features1.descriptors) == 0 or len(features2.descriptors) == 0:
        raise ValueError("Descriptor tensors must not be empty")
    ambiguity_filtering = min_similarity_margin > 0.0 or max_distance_ratio is not None
    if ambiguity_filtering and min(
        len(features1.descriptors), len(features2.descriptors)
    ) < 2:
        raise ValueError("Ambiguity filtering requires at least two descriptors per image")

    descriptors1 = F.normalize(features1.descriptors.float(), dim=1)
    descriptors2 = F.normalize(features2.descriptors.float(), dim=1)
    similarities = descriptors1 @ descriptors2.T

    if ambiguity_filtering:
        forward_scores, forward_indices = similarities.topk(2, dim=1)
        reverse_scores, reverse_indices = similarities.topk(2, dim=0)
        scores1 = forward_scores[:, 0]
        nearest2 = forward_indices[:, 0]
        nearest1 = reverse_indices[0]
    else:
        scores1, nearest2 = similarities.max(dim=1)
        nearest1 = similarities.argmax(dim=0)
    indices1 = torch.arange(len(descriptors1), device=similarities.device)
    keep = (nearest1[nearest2] == indices1) & (scores1 >= similarity_threshold)
    if min_similarity_margin > 0.0:
        forward_margin = forward_scores[:, 0] - forward_scores[:, 1]
        reverse_margin = reverse_scores[0] - reverse_scores[1]
        keep &= forward_margin >= min_similarity_margin
        keep &= reverse_margin[nearest2] >= min_similarity_margin
    if max_distance_ratio is not None:
        forward_ratio = (1.0 - forward_scores[:, 0]).clamp_min(0.0) / (
            1.0 - forward_scores[:, 1]
        ).clamp_min(1e-12)
        reverse_ratio = (1.0 - reverse_scores[0]).clamp_min(0.0) / (
            1.0 - reverse_scores[1]
        ).clamp_min(1e-12)
        keep &= forward_ratio <= max_distance_ratio
        keep &= reverse_ratio[nearest2] <= max_distance_ratio
    indices1 = indices1[keep]
    indices2 = nearest2[keep]
    scores = scores1[keep]

    order = torch.argsort(scores, descending=True)
    if max_matches is not None:
        order = order[:max_matches]
    indices1 = indices1[order].cpu()
    indices2 = indices2[order].cpu()
    scores = scores[order].cpu()

    return FeatureMatches(
        keypoints1=features1.keypoints[indices1].cpu(),
        keypoints2=features2.keypoints[indices2].cpu(),
        scores=scores,
        indices1=indices1,
        indices2=indices2,
    )


def filter_matches_with_fundamental_ransac(
    matches: FeatureMatches,
    *,
    reprojection_threshold_px: float = 12.0,
    confidence: float = 0.999,
    max_iterations: int = 10000,
) -> FeatureMatches:
    """Keep matches consistent with one two-view epipolar geometry.

    This implementation uses NumPy instead of OpenCV so it does not interfere
    with robotics environments that already ship their own OpenCV binary.
    ``reprojection_threshold_px`` is applied to the Sampson approximation of
    geometric reprojection error.
    """

    if len(matches) < 8:
        raise ValueError("Fundamental-matrix RANSAC requires at least eight matches")
    if not np.isfinite(reprojection_threshold_px) or reprojection_threshold_px <= 0.0:
        raise ValueError("reprojection_threshold_px must be finite and positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    points1 = matches.keypoints1.detach().double().cpu().numpy()
    points2 = matches.keypoints2.detach().double().cpu().numpy()
    if points1.shape != points2.shape or points1.shape != (len(matches), 2):
        raise ValueError("Matched keypoints must both have shape [matches, 2]")
    if not np.isfinite(points1).all() or not np.isfinite(points2).all():
        raise ValueError("Matched keypoints must be finite")

    def normalize_points(points: np.ndarray):
        centre = points.mean(axis=0)
        centred = points - centre
        rms_distance = np.sqrt(np.mean(np.sum(centred * centred, axis=1)))
        if rms_distance <= np.finfo(np.float64).eps:
            raise np.linalg.LinAlgError("Coincident points cannot define a geometry")
        scale = np.sqrt(2.0) / rms_distance
        transform = np.asarray(
            (
                (scale, 0.0, -scale * centre[0]),
                (0.0, scale, -scale * centre[1]),
                (0.0, 0.0, 1.0),
            )
        )
        homogeneous = np.column_stack((points, np.ones(len(points))))
        return (transform @ homogeneous.T).T[:, :2], transform

    def estimate_fundamental(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        first_normalized, transform1 = normalize_points(first)
        second_normalized, transform2 = normalize_points(second)
        x1, y1 = first_normalized.T
        x2, y2 = second_normalized.T
        design = np.column_stack(
            (x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, np.ones(len(first)))
        )
        if np.linalg.matrix_rank(design) < 8:
            raise np.linalg.LinAlgError("Degenerate point sample")
        _, _, right_vectors = np.linalg.svd(design, full_matrices=True)
        fundamental = right_vectors[-1].reshape(3, 3)
        left, singular_values, right = np.linalg.svd(fundamental)
        singular_values[-1] = 0.0
        fundamental = left @ np.diag(singular_values) @ right
        fundamental = transform2.T @ fundamental @ transform1
        norm = np.linalg.norm(fundamental)
        if not np.isfinite(norm) or norm <= np.finfo(np.float64).eps:
            raise np.linalg.LinAlgError("Invalid fundamental matrix")
        return fundamental / norm

    homogeneous1 = np.column_stack((points1, np.ones(len(points1))))
    homogeneous2 = np.column_stack((points2, np.ones(len(points2))))

    def sampson_errors(fundamental: np.ndarray) -> np.ndarray:
        lines2 = (fundamental @ homogeneous1.T).T
        lines1 = (fundamental.T @ homogeneous2.T).T
        residual = np.sum(homogeneous2 * lines2, axis=1)
        denominator = (
            lines2[:, 0] ** 2
            + lines2[:, 1] ** 2
            + lines1[:, 0] ** 2
            + lines1[:, 1] ** 2
        )
        return residual**2 / np.maximum(denominator, np.finfo(np.float64).eps)

    threshold_squared = float(reprojection_threshold_px) ** 2
    generator = np.random.default_rng(0)
    best_keep = None
    best_count = 0
    best_error = np.inf
    iteration_limit = int(max_iterations)
    iteration = 0
    while iteration < iteration_limit:
        iteration += 1
        sample = generator.choice(len(matches), size=8, replace=False)
        try:
            fundamental = estimate_fundamental(points1[sample], points2[sample])
        except np.linalg.LinAlgError:
            continue
        errors = sampson_errors(fundamental)
        candidate_keep = errors <= threshold_squared
        candidate_count = int(candidate_keep.sum())
        candidate_error = float(np.median(errors[candidate_keep])) if candidate_count else np.inf
        if candidate_count > best_count or (
            candidate_count == best_count and candidate_error < best_error
        ):
            best_keep = candidate_keep
            best_count = candidate_count
            best_error = candidate_error
            inlier_probability = (candidate_count / len(matches)) ** 8
            if 0.0 < inlier_probability < 1.0:
                required = np.log1p(-confidence) / np.log1p(-inlier_probability)
                iteration_limit = min(iteration_limit, max(iteration, int(np.ceil(required))))
            elif inlier_probability >= 1.0:
                iteration_limit = iteration

    if best_keep is None or best_count < 8:
        raise RuntimeError("Fundamental-matrix RANSAC found fewer than eight inliers")
    try:
        refined = estimate_fundamental(points1[best_keep], points2[best_keep])
        refined_keep = sampson_errors(refined) <= threshold_squared
        if int(refined_keep.sum()) >= best_count:
            best_keep = refined_keep
    except np.linalg.LinAlgError:
        pass

    keep = torch.from_numpy(best_keep)
    return FeatureMatches(
        keypoints1=matches.keypoints1[keep],
        keypoints2=matches.keypoints2[keep],
        scores=matches.scores[keep],
        indices1=matches.indices1[keep],
        indices2=matches.indices2[keep],
    )


def score_image_match(
    image1: Image.Image,
    image2: Image.Image,
    candidate_matches: FeatureMatches,
    inlier_matches: FeatureMatches,
    *,
    feature_weight: float = 0.70,
    zncc_weight: float = 0.30,
    patch_radius_fraction: float = 0.025,
    normalized_patch_size: int = 21,
    full_support_inliers: int = 12,
) -> ImageMatchScore:
    """Score an image pair using DINO, epipolar geometry, and local ZNCC.

    ZNCC is evaluated only around geometrically verified correspondences. Each
    image uses a patch radius relative to its own dimensions and patches are
    resized to a common resolution, so different image sizes and translations
    do not invalidate the photometric comparison. The median correlation is
    used to resist occlusion and individual incorrect correspondences.

    The returned ``score`` is in ``[0, 1]`` and higher is better. The feature
    term is the geometric mean of normalized descriptor similarity and the
    RANSAC inlier ratio, with a small-support penalty below
    ``full_support_inliers``. ZNCC is mapped from ``[-1, 1]`` to ``[0, 1]``.
    """

    if not isinstance(image1, Image.Image) or not isinstance(image2, Image.Image):
        raise TypeError("image1 and image2 must be PIL images")
    if len(candidate_matches) == 0:
        raise ValueError("candidate_matches must not be empty")
    if len(inlier_matches) == 0:
        raise ValueError("inlier_matches must not be empty")
    if len(inlier_matches) > len(candidate_matches):
        raise ValueError("inlier_matches cannot outnumber candidate_matches")
    if not all(np.isfinite((feature_weight, zncc_weight))):
        raise ValueError("Score weights must be finite")
    if feature_weight < 0.0 or zncc_weight < 0.0 or feature_weight + zncc_weight <= 0.0:
        raise ValueError("Score weights must be non-negative with a positive sum")
    if not np.isfinite(patch_radius_fraction) or not 0.0 < patch_radius_fraction < 0.5:
        raise ValueError("patch_radius_fraction must be finite and in (0, 0.5)")
    if (
        not isinstance(normalized_patch_size, int)
        or isinstance(normalized_patch_size, bool)
        or normalized_patch_size < 3
    ):
        raise ValueError("normalized_patch_size must be an integer of at least 3")
    if (
        not isinstance(full_support_inliers, int)
        or isinstance(full_support_inliers, bool)
        or full_support_inliers <= 0
    ):
        raise ValueError("full_support_inliers must be a positive integer")

    points1 = inlier_matches.keypoints1.detach().float().cpu().numpy()
    points2 = inlier_matches.keypoints2.detach().float().cpu().numpy()
    if points1.shape != points2.shape or points1.shape != (len(inlier_matches), 2):
        raise ValueError("Inlier keypoints must both have shape [matches, 2]")
    if not np.isfinite(points1).all() or not np.isfinite(points2).all():
        raise ValueError("Inlier keypoints must contain only finite values")

    grayscale1 = image1.convert("L")
    grayscale2 = image2.convert("L")
    width1, height1 = grayscale1.size
    width2, height2 = grayscale2.size
    radius1 = max(1.5, patch_radius_fraction * min(width1, height1))
    radius2 = max(1.5, patch_radius_fraction * min(width2, height2))

    def patch_at(
        image: Image.Image,
        point: np.ndarray,
        radius: float,
    ) -> np.ndarray | None:
        width, height = image.size
        x, y = (float(value) for value in point)
        if x - radius < 0.0 or y - radius < 0.0:
            return None
        if x + radius > width or y + radius > height:
            return None
        patch = image.crop((x - radius, y - radius, x + radius, y + radius)).resize(
            (normalized_patch_size, normalized_patch_size),
            Image.Resampling.BILINEAR,
        )
        return np.asarray(patch, dtype=np.float32) / 255.0

    correlations = []
    for point1, point2 in zip(points1, points2):
        patch1 = patch_at(grayscale1, point1, radius1)
        patch2 = patch_at(grayscale2, point2, radius2)
        if patch1 is None or patch2 is None:
            continue
        try:
            correlation = zero_mean_normalized_cross_correlation(
                patch1,
                patch2,
                minimum_standard_deviation=1.0 / 255.0,
            )
        except ValueError:
            continue
        correlations.append(correlation)
    if not correlations:
        raise RuntimeError(
            "No geometrically verified match had two valid, textured ZNCC patches"
        )

    per_match_zncc = np.asarray(correlations, dtype=np.float64)
    median_zncc = float(np.median(per_match_zncc))
    zncc_score = float(np.clip((median_zncc + 1.0) * 0.5, 0.0, 1.0))
    similarities = inlier_matches.scores.detach().float().cpu().numpy()
    if similarities.shape != (len(inlier_matches),) or not np.isfinite(similarities).all():
        raise ValueError("Inlier descriptor scores must be one finite value per match")
    mean_similarity = float(np.mean(similarities))
    descriptor_score = float(np.clip((mean_similarity + 1.0) * 0.5, 0.0, 1.0))
    inlier_ratio = len(inlier_matches) / len(candidate_matches)
    support = min(1.0, len(inlier_matches) / full_support_inliers)
    geometric_score = float(inlier_ratio * support)
    feature_score = float(np.sqrt(descriptor_score * geometric_score))
    total_weight = feature_weight + zncc_weight
    combined_score = float(
        (feature_weight * feature_score + zncc_weight * zncc_score) / total_weight
    )
    return ImageMatchScore(
        score=combined_score,
        feature_score=feature_score,
        descriptor_score=descriptor_score,
        mean_descriptor_similarity=mean_similarity,
        geometric_score=geometric_score,
        geometric_inlier_ratio=float(inlier_ratio),
        median_zncc=median_zncc,
        zncc_score=zncc_score,
        candidate_count=len(candidate_matches),
        inlier_count=len(inlier_matches),
        zncc_patch_count=len(per_match_zncc),
        per_match_zncc=per_match_zncc,
    )


def plot_matches(
    image1: Image.Image,
    image2: Image.Image,
    matches: FeatureMatches,
    *,
    max_display: int = 100,
    figsize: Tuple[float, float] = (16, 8),
):
    """Plot the strongest matches and return ``(figure, axes)``."""

    if max_display <= 0:
        raise ValueError("max_display must be positive")
    import matplotlib.pyplot as plt

    image1 = image1.convert("RGB")
    image2 = image2.convert("RGB")
    width1, height1 = image1.size
    width2, height2 = image2.size
    canvas = Image.new("RGB", (width1 + width2, max(height1, height2)), "black")
    canvas.paste(image1, (0, 0))
    canvas.paste(image2, (width1, 0))

    figure, axes = plt.subplots(figsize=figsize)
    axes.imshow(canvas)
    displayed = min(len(matches), max_display)
    colors = plt.cm.turbo(torch.linspace(0, 1, max(displayed, 1)).numpy())
    for index in range(displayed):
        x1, y1 = matches.keypoints1[index].tolist()
        x2, y2 = matches.keypoints2[index].tolist()
        color = colors[index]
        axes.plot((x1, x2 + width1), (y1, y2), color=color, linewidth=1.0, alpha=0.8)
        axes.scatter((x1, x2 + width1), (y1, y2), color=color, s=12)
    axes.set_title(f"DINOv3 patch matches: showing {displayed} of {len(matches)}")
    axes.axis("off")
    figure.tight_layout()
    return figure, axes
