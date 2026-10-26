"""CLI entry point for unknown-object RGB localization from real observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import math

from .runtime import file_digest, require_pose_close
from .target_selection import load_reference_mask


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Find the target visible-surface aim point from calibrated real RGB captures; "
            "no object mesh, class, pose, dimensions, or depth are required."
        )
    )
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--target-image", type=Path, required=True)
    parser.add_argument("--target-mask", type=Path, required=True,
                        help="Full-object reference mask from findvp_select_target")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--feature-max-image-size", type=int, default=448)
    parser.add_argument("--min-similarity", type=float, default=0.55)
    parser.add_argument("--ambiguity-margin", type=float, default=0.03)
    parser.add_argument("--mask-backend", choices=("sam", "flat"), default="sam")
    parser.add_argument("--sam-points-per-batch", type=int, choices=(1, 2, 4, 8, 16, 32), default=4)
    args = parser.parse_args(argv)
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_json}; choose a new --output-json")
    target_mask = load_reference_mask(args.target_mask, args.target_image)
    diagnostics_dir = args.output_json.with_suffix(".diagnostics")
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    from sample.dinov3_feature_matching import DINOv3FeatureExtractor
    from sample.rgb_object_localization import (
        load_rgb_observations,
        localize_rgb_observations,
    )
    from sample.target_guided_similarity import prepare_target_image

    started = time.perf_counter()
    manifest = json.loads(args.observations.read_text())
    records = manifest.get("observations", [])
    if len(records) < 2 or any("base_reference" not in item for item in records):
        raise ValueError("collect at least two new captures with base-pose provenance")
    base_reference = records[0]["base_reference"]
    for record in records:
        other = record["base_reference"]
        if other.get("mode", "odom") != base_reference.get("mode", "odom"):
            raise ValueError("base tracking mode changed during survey")
        if any(other[k] != base_reference[k] for k in ("frame", "child_frame")):
            raise ValueError("odometry frames changed during survey")
        require_pose_close(other["odom_from_base"], base_reference["odom_from_base"],
                           .005, math.radians(.5), "survey base pose")
    frame, observations = load_rgb_observations(args.observations)
    from .segmentation import make_mask_provider
    try:
        mask_provider = make_mask_provider(args.mask_backend, points_per_batch=args.sam_points_per_batch)
        extractor = DINOv3FeatureExtractor(
            max_image_size=args.feature_max_image_size, upscale_small_images=True
        )
        target = prepare_target_image(args.target_image, extractor=extractor, target_mask=target_mask)
        report = localize_rgb_observations(
            observations, target.object_features, extractor, coordinate_frame=frame,
            min_similarity=args.min_similarity, ambiguity_margin=args.ambiguity_margin,
            candidate_mask_provider=mask_provider, diagnostics_dir=diagnostics_dir,
        )
    except Exception as error:
        (diagnostics_dir / "failure.json").write_text(json.dumps(
            {"status": "failed", "error": f"{type(error).__name__}: {error}"}, indent=2) + "\n")
        raise SystemExit(f"Localization failed: {error}\nInspect masks and matches in {diagnostics_dir}") from error
    report.update(
        {
            "total_seconds": time.perf_counter() - started,
            "target_image": str(args.target_image.expanduser().resolve()),
            "observations_manifest": str(args.observations.expanduser().resolve()),
            "object_prior_used": False,
            "target_sha256": file_digest(args.target_image),
            "target_mask": str(args.target_mask.expanduser().resolve()),
            "target_mask_sha256": file_digest(args.target_mask),
            "base_reference": base_reference,
            "mask_backend": args.mask_backend,
        }
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"RGB visible-surface aim point ({frame}, metres): {report['center_world_m']}")
