#!/usr/bin/env python3
"""Metric RGB-only aiming-point initialization from calibrated moving cameras.

Camera transforms map optical coordinates (+X right, +Y down, +Z forward)
into one named metric frame. The scene must remain static between captures.
No object geometry, depth sensor, or known target position is consumed.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import itertools
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

if __package__:
    from .rgb_localization_geometry import (
        _validate_intrinsics, _validate_transform, estimate_aim_point,
        triangulate_matches, world_from_camera,
    )
    from .rgb_target_detection import detect_target_region
else:
    from rgb_localization_geometry import (
        _validate_intrinsics, _validate_transform, estimate_aim_point,
        triangulate_matches, world_from_camera,
    )
    from rgb_target_detection import detect_target_region


@dataclass
class RGBObservation:
    image: Image.Image
    intrinsics: np.ndarray
    world_from_camera: np.ndarray
    label: str
    candidate_masks: list[np.ndarray] | None = None
    distortion_coefficients: np.ndarray | None = None

    def __post_init__(self):
        self.intrinsics = _validate_intrinsics(self.intrinsics, "intrinsics")
        self.world_from_camera = _validate_transform(self.world_from_camera, "world_from_camera")
        if self.distortion_coefficients is not None:
            self.distortion_coefficients = np.asarray(self.distortion_coefficients, dtype=float).reshape(-1)
            if self.distortion_coefficients.size not in (4, 5, 8, 12, 14) or not np.isfinite(self.distortion_coefficients).all():
                raise ValueError("distortion_coefficients must contain 4, 5, 8, 12 or 14 finite OpenCV coefficients")


def load_rgb_observations(path: Path) -> tuple[str, list[RGBObservation]]:
    path = Path(path).expanduser().resolve()
    document = json.loads(path.read_text())
    frame = document.get("coordinate_frame")
    if not isinstance(frame, str) or not frame.strip():
        raise ValueError("observations require a named coordinate_frame")
    records = document.get("observations", [])
    if not isinstance(records, list) or len(records) < 2:
        raise ValueError("at least two calibrated RGB observations are required")
    observations = []
    for index, record in enumerate(records):
        direct = "world_from_camera" in record
        composed = "world_from_tool" in record or "tool_from_camera" in record
        if direct == composed:
            raise ValueError("supply world_from_camera OR world_from_tool plus tool_from_camera")
        try:
            pose = record["world_from_camera"] if direct else world_from_camera(
                record["world_from_tool"], record["tool_from_camera"])
            with Image.open(path.parent / record["image"]) as image_file:
                image = image_file.convert("RGB")
            masks = None
            if "mask" in record:
                with Image.open(path.parent / record["mask"]) as mask_file:
                    mask = np.asarray(mask_file.convert("L")) > 0
                if mask.shape != (image.height, image.width):
                    raise ValueError("mask dimensions must match image")
                masks = [mask]
            observations.append(RGBObservation(
                image, record["intrinsics"], pose, str(record.get("label", index)),
                masks, record.get("distortion_coefficients"),
            ))
        except KeyError as error:
            raise ValueError(f"observation {index} missing calibration/image field {error}") from error
    return frame, observations


def extract_masked_sift(image: Image.Image, mask: np.ndarray):
    """Upscale the detected crop for matching; return original image pixels."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.empty((0, 2)), None
    x0, x1 = max(0, int(xs.min()) - 4), min(image.width, int(xs.max()) + 5)
    y0, y1 = max(0, int(ys.min()) - 4), min(image.height, int(ys.max()) + 5)
    crop = np.asarray(image.convert("L"))[y0:y1, x0:x1]
    scale = max(1., 720. / max(crop.shape))
    size = (round(crop.shape[1] * scale), round(crop.shape[0] * scale))
    gray = cv2.resize(crop, size, interpolation=cv2.INTER_CUBIC)
    binary = cv2.resize(mask[y0:y1, x0:x1].astype(np.uint8) * 255, size, interpolation=cv2.INTER_NEAREST)
    keypoints, descriptors = cv2.SIFT_create(nfeatures=3000, contrastThreshold=.02).detectAndCompute(gray, binary)
    if not keypoints:
        return np.empty((0, 2)), None
    points = np.asarray([key.pt for key in keypoints], dtype=float)
    points /= np.array([size[0] / crop.shape[1], size[1] / crop.shape[0]])
    points += [x0, y0]
    return points, descriptors


def _mutual_matches(first, second):
    if first is None or second is None or min(len(first), len(second)) < 2:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    def ratio(a, b):
        return {m.queryIdx: m.trainIdx for pair in matcher.knnMatch(a, b, k=2)
                if len(pair) == 2 for m, n in [pair] if m.distance < .75 * n.distance}
    forward, reverse = ratio(first, second), ratio(second, first)
    return [(a, b) for a, b in forward.items() if reverse.get(b) == a]


def _undistort(points, observation):
    if observation.distortion_coefficients is None:
        return points
    return cv2.undistortPoints(np.asarray(points, dtype=float).reshape(-1, 1, 2),
                              observation.intrinsics, observation.distortion_coefficients,
                              P=observation.intrinsics).reshape(-1, 2)


def localize_rgb_observations(observations, target_features, extractor, *,
                              coordinate_frame="world", min_similarity=.55,
                              ambiguity_margin=.03, candidate_mask_provider=None,
                              diagnostics_dir=None) -> dict:
    started = time.perf_counter()
    if len(observations) < 2:
        raise ValueError("at least two calibrated RGB observations are required")
    if not isinstance(coordinate_frame, str) or not coordinate_frame.strip():
        raise ValueError("coordinate_frame must be named")
    features, detections, pairs = [], [], []
    diagnostics_dir = Path(diagnostics_dir) if diagnostics_dir is not None else None
    def save_diagnostics():
        if diagnostics_dir is not None:
            diagnostics_dir.mkdir(parents=True, exist_ok=True)
            (diagnostics_dir / "matching.json").write_text(json.dumps(
                {"coordinate_frame": coordinate_frame, "views": detections, "pairs": pairs},
                indent=2) + "\n")
    for index, observation in enumerate(observations):
        if diagnostics_dir is not None:
            print(f"RGB localization: selecting target in view {index + 1}/{len(observations)}", flush=True)
        if observation.candidate_masks is None and candidate_mask_provider is not None:
            observation.candidate_masks = candidate_mask_provider(observation.image)
        detection = detect_target_region(
            observation.image, target_features, extractor,
            candidate_masks=observation.candidate_masks,
            min_similarity=min_similarity, ambiguity_margin=ambiguity_margin,
        )
        features.append(extract_masked_sift(observation.image, detection["mask"]))
        detections.append({"label": observation.label, "bbox_xyxy": list(detection["bbox_xyxy"]),
                           "similarity": float(detection["similarity"]),
                           "keypoints": len(features[-1][0]), "diagnostics": detection["diagnostics"]})
        save_diagnostics()
        if diagnostics_dir is not None:
            Image.fromarray(detection["mask"].astype(np.uint8) * 255).save(
                diagnostics_dir / f"view_{index:03d}_selected_mask.png")
            # Correspondence coordinates remain in the original RGB images.
            (diagnostics_dir / f"view_{index:03d}_keypoints.json").write_text(
                json.dumps(features[-1][0].tolist()))
    clouds = []
    for a, b in itertools.combinations(range(len(observations)), 2):
        matches = _mutual_matches(features[a][1], features[b][1])
        info = {"views": [a, b], "mutual_matches": len(matches)}
        if len(matches) < 8:
            info["rejected"] = "fewer than eight mutual feature matches"
            pairs.append(info)
            save_diagnostics()
            continue
        indices = np.asarray(matches)
        first, second = observations[a], observations[b]
        if diagnostics_dir is not None:
            (diagnostics_dir / f"pair_{a:03d}_{b:03d}_matches.json").write_text(json.dumps({
                "pixels_a": features[a][0][indices[:, 0]].tolist(),
                "pixels_b": features[b][0][indices[:, 1]].tolist(),
            }, indent=2) + "\n")
        try:
            result = triangulate_matches(
                _undistort(features[a][0][indices[:, 0]], first),
                _undistort(features[b][0][indices[:, 1]], second),
                first.intrinsics, second.intrinsics,
                first.world_from_camera, second.world_from_camera,
            )
        except ValueError as error:
            info["rejected"] = str(error)
            if hasattr(error, "diagnostics"):
                info["geometry"] = error.diagnostics
        else:
            clouds.append(result.points_world_m[result.inlier_mask])
            info["geometry"] = result.diagnostics
        pairs.append(info)
        save_diagnostics()
    if not clouds:
        raise ValueError(f"RGB triangulation failed; no reliable metric estimate: {pairs}")
    points = np.concatenate(clouds)
    center, diagnostics = estimate_aim_point(points)
    return {
        "method": "rgb_multiview", "coordinate_frame": coordinate_frame,
        "position_units": "metres", "estimate_kind": "visible_surface_aim_point",
        "center_world_m": center.tolist(), "orientation_world_xyzw": None,
        "metric_scale_source": "calibrated_camera_translation",
        "input_views": len(observations), "triangulated_points": len(points),
        "points_world_m": points.tolist(), "views": detections, "pairs": pairs,
        "aim_point_diagnostics": diagnostics, "localization_seconds": time.perf_counter() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--target-image", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--feature-max-image-size", type=int, default=448)
    parser.add_argument("--min-similarity", type=float, default=.55)
    parser.add_argument("--ambiguity-margin", type=float, default=.03)
    args = parser.parse_args()
    if __package__:
        from .dinov3_feature_matching import DINOv3FeatureExtractor
        from .target_guided_similarity import prepare_target_image
    else:
        from dinov3_feature_matching import DINOv3FeatureExtractor
        from target_guided_similarity import prepare_target_image
    started = time.perf_counter()
    frame, observations = load_rgb_observations(args.observations)
    extractor = DINOv3FeatureExtractor(max_image_size=args.feature_max_image_size, upscale_small_images=True)
    target = prepare_target_image(args.target_image, extractor=extractor)
    report = localize_rgb_observations(observations, target.object_features, extractor,
                                      coordinate_frame=frame, min_similarity=args.min_similarity,
                                      ambiguity_margin=args.ambiguity_margin)
    report["total_seconds"] = time.perf_counter() - started
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n")
    print(f"RGB aiming point ({frame}, metres): {report['center_world_m']}")


if __name__ == "__main__":
    main()
