"""Acquire an independent workspace survey for RGB-only localization."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image


def _survey(path):
    document = (json.loads(Path(path).read_text()) if path else {
        "coordinate_frame": "pybullet_world",
        "viewpoints": [{"position_xyz_m": [x, -.10, 1.30],
                        "orientation_xyzw": [1., 0., 0., 0.]} for x in (.36, .44, .52)],
    })
    if document.get("coordinate_frame") != "pybullet_world":
        raise ValueError("live survey must use pybullet_world coordinates")
    records = document.get("viewpoints", [])
    if len(records) < 2:
        raise ValueError("survey requires at least two camera views")
    positions = []
    for record in records:
        position = np.asarray(record.get("position_xyz_m"), dtype=float)
        quaternion = np.asarray(record.get("orientation_xyzw"), dtype=float)
        if (position.shape != (3,) or not np.isfinite(position).all()
                or quaternion.shape != (4,) or not np.isfinite(quaternion).all()
                or not np.isclose(np.linalg.norm(quaternion), 1., atol=1e-5)):
            raise ValueError("survey poses require finite xyz and unit xyzw quaternion")
        positions.append(position)
    if max(np.linalg.norm(a - b) for a in positions for b in positions) < .01:
        raise ValueError("survey needs at least a centimetre of translation baseline")
    return records


def bootstrap_rgb_localization(active, args, env, config, candidate_backend,
                               motion_planner, outward_paths, capture_pose_report,
                               capture_pose_json_path, capture_pose_ply_path,
                               evaluation_tracker):
    survey = _survey(args.localization_viewpoints)
    for name in ("localization_min_similarity", "localization_ambiguity_margin"):
        value = getattr(args, name)
        if not np.isfinite(value) or not 0. <= value <= 1.:
            raise ValueError(f"{name} must be finite and between zero and one")
    if args.target_image is None:
        raise ValueError("RGB localization requires --target-image")
    if __package__:
        from .rgb_object_localization import RGBObservation, localize_rgb_observations
        from .geometric_view_search import camera_intrinsics
        from .dinov3_feature_matching import DINOv3FeatureExtractor
        from .target_guided_similarity import prepare_target_image
    else:
        from rgb_object_localization import RGBObservation, localize_rgb_observations
        from geometric_view_search import camera_intrinsics
        from dinov3_feature_matching import DINOv3FeatureExtractor
        from target_guided_similarity import prepare_target_image
    started = time.perf_counter()
    config = replace(config, require_line_of_sight=False, roll_angles_deg=(0.0,))
    candidates = [active.ViewpointCandidate(
        index=i, position_xyz=tuple(pose["position_xyz_m"]),
        orientation_xyzw=tuple(pose["orientation_xyzw"]),
        radius_m=0., azimuth_deg=0., elevation_deg=90.,
    ) for i, pose in enumerate(survey)]
    # Center is unused with LOS disabled; survey orientations are explicit.
    reachability = env.filter_reachable_viewpoints(candidates, np.zeros(3), config,
                                                  candidate_backend=candidate_backend)
    records = [{"triangle": i + 1, "triangle_path": [-(i + 1)],
                "refinement_level": 0, "stage": "rgb_localization",
                "centroid_xyz_m": pose["position_xyz_m"]} for i, pose in enumerate(survey)]
    reachable = active._annotate_reachability(records, reachability)
    prefix = f"{args.name}_rgb_localization"
    before = len(capture_pose_report["captures"])
    active._capture_round(
        env, records, reachable, args.output_dir / f"{prefix}_viewpoints.json",
        prefix, args.motion_seconds, motion_planner, outward_paths,
        capture_pose_report, capture_pose_json_path, capture_pose_ply_path,
        evaluation_tracker, return_home_after_round=False,
    )
    captures = capture_pose_report["captures"][before:]
    if len(captures) < 2:
        raise RuntimeError("RGB localization requires at least two successful survey captures")
    observations, manifest_records = [], []
    for index, capture in enumerate(captures):
        with Image.open(capture["image_file"]) as file:
            image = file.convert("RGB")
        pose = capture["camera_pose_world"]
        transform = np.eye(4)
        transform[:3, :3] = active.quaternion_xyzw_to_matrix(pose["orientation_xyzw"])
        transform[:3, 3] = pose["position_xyz_m"]
        intrinsics = camera_intrinsics(image.size)
        observations.append(RGBObservation(image=image, intrinsics=intrinsics,
                                           world_from_camera=transform, label=str(index)))
        manifest_records.append({"image": str(Path(capture["image_file"]).resolve()),
                                 "intrinsics": intrinsics.tolist(),
                                 "world_from_camera": transform.tolist()})
    manifest_path = args.output_dir / f"{prefix}_observations.json"
    active.write_json(manifest_path, {"coordinate_frame": "pybullet_world", "observations": manifest_records})
    extractor = DINOv3FeatureExtractor(max_image_size=args.feature_max_image_size, upscale_small_images=True)
    prepared = prepare_target_image(args.target_image, extractor=extractor, debug=args.debug)
    report = localize_rgb_observations(
        observations, prepared.object_features, extractor, coordinate_frame="pybullet_world",
        min_similarity=args.localization_min_similarity,
        ambiguity_margin=args.localization_ambiguity_margin,
    )
    report_path = args.output_dir / f"{prefix}.json"
    report.update(captured_views=len(captures), bootstrap_seconds=time.perf_counter() - started,
                  observations_manifest=str(manifest_path.resolve()), report_file=str(report_path.resolve()))
    active.write_json(report_path, report)
    print(f"RGB surface aiming point: {report['center_world_m']}", flush=True)
    return report, extractor, prepared
