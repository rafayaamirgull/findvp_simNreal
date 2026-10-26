#!/usr/bin/env python3
"""Target-guided camera search using triangle refinement or Bayesian next best views."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

if __package__:
    from .geometric_view_search import GeometricViewSearch
    from .next_best_view import BayesianViewSearch
    from .capture_four_views import (
        CLEARANCE_M,
        move_home,
        plan_validated_path,
        play_path,
    )
    from .curobo_motion import CuRoboMotionPlanner
    from .pybullet_tabletop_ur5e_eye_in_hand import (
        DEFAULT_CUROBO_IK_SEED_COUNT,
        UR5EyeInHandPyBulletEnv,
        build_selected_candidate_backend,
        build_selected_motion_planner,
        resolve_ik_backend,
    )
    from .viewpoint_reachability import (
        ReachabilityConfig,
        ViewpointCandidate,
        look_at_rotation_matrix,
        matrix_to_quaternion_xyzw,
        quaternion_xyzw_to_matrix,
    )
    from .workcell_config import (
        ARM_JOINT_ORDER,
        DEFAULT_ROBOT_URDF,
        DEFAULT_WORKPIECE_NAME,
        DEFAULT_WORKPIECE_PLY,
        END_EFFECTOR_LINK_NAME,
        OUTPUT_DIR,
        resolve_workpiece_ply,
    )
else:
    from geometric_view_search import GeometricViewSearch
    from next_best_view import BayesianViewSearch
    from capture_four_views import (
        CLEARANCE_M,
        move_home,
        plan_validated_path,
        play_path,
    )
    from curobo_motion import CuRoboMotionPlanner
    from pybullet_tabletop_ur5e_eye_in_hand import (
        DEFAULT_CUROBO_IK_SEED_COUNT,
        UR5EyeInHandPyBulletEnv,
        build_selected_candidate_backend,
        build_selected_motion_planner,
        resolve_ik_backend,
    )
    from viewpoint_reachability import (
        ReachabilityConfig,
        ViewpointCandidate,
        look_at_rotation_matrix,
        matrix_to_quaternion_xyzw,
        quaternion_xyzw_to_matrix,
    )
    from workcell_config import (
        ARM_JOINT_ORDER,
        DEFAULT_ROBOT_URDF,
        DEFAULT_WORKPIECE_NAME,
        DEFAULT_WORKPIECE_PLY,
        END_EFFECTOR_LINK_NAME,
        OUTPUT_DIR,
        resolve_workpiece_ply,
    )


TRIANGLE_FALLBACK_BLEND_FRACTIONS = (0.20, 0.40)


def _apply_convergence_mode(args: argparse.Namespace) -> None:
    """Resolve search overrides before validation, also for direct run() callers."""

    if not args.until_convergence:
        return
    if args.target_image is None:
        raise ValueError("--until-convergence requires --target-image.")
    if (
        not math.isfinite(args.convergence_min_improvement)
        or not 0.0 < args.convergence_min_improvement <= 1.0
    ):
        raise ValueError("--convergence-min-improvement must be in (0, 1].")
    if args.convergence_patience < 1:
        raise ValueError("--convergence-patience must be positive.")
    args.views = 3
    args.similarity_threshold = None
    args.max_refinement_depth = None
    args.refinement_patience = args.convergence_patience
    args.min_score_improvement = args.convergence_min_improvement
    args.angular_lookahead_margin = None
    args.angular_lookahead_levels = None
    args.skip_radial_refinement = True


def _apply_search_mode(args: argparse.Namespace) -> None:
    if args.search_strategy in ("nbv", "geometric"):
        if args.until_convergence:
            raise ValueError(
                "--until-convergence is for triangle search; sequential search uses --nbv-patience "
                "and --nbv-max-views to allow exploratory views before stopping."
            )
        if args.target_image is None:
            raise ValueError(
                f"--search-strategy {args.search_strategy} requires --target-image."
            )
        # Validate before constructing the environment or loading any models.
        BayesianViewSearch(
            (0, 0, 0),
            args.radius,
            min_radius=args.radial_min_radius,
            max_radius=args.radial_max_radius,
            max_views=args.nbv_max_views,
            patience=args.nbv_patience,
            min_improvement=args.min_score_improvement,
            azimuth_offset=args.azimuth_offset or 0.0,
        )
        if not math.isfinite(args.nbv_time_budget) or args.nbv_time_budget < 0:
            raise ValueError("--nbv-time-budget must be finite and non-negative.")
        args.skip_radial_refinement = True
        if args.method_name == "Adaptive Spherical + Radial (DINOv3+SIFT)":
            args.method_name = (
                "Geometric NBV (DINOv3+SIFT)"
                if args.search_strategy == "geometric"
                else "Bayesian NBV (DINOv3+SIFT)"
            )
    _apply_convergence_mode(args)


def _nbv_viewpoint(proposal, center, level, strategy="nbv"):
    """Adapt a free-space proposal to the shared capture/ranking file schema.

    Legacy triangle ID/path fields are identifiers only; no triangle is created.
    """
    position = np.asarray(proposal["position_xyz_m"], dtype=float)
    ray = position - np.asarray(center, dtype=float)
    radius = float(np.linalg.norm(ray))
    rotation = look_at_rotation_matrix(position, center)
    candidate = ViewpointCandidate(
        index=0,
        position_xyz=tuple(position),
        orientation_xyzw=tuple(
            proposal.get("camera_orientation_xyzw", matrix_to_quaternion_xyzw(rotation))
        ),
        radius_m=radius,
        azimuth_deg=math.degrees(math.atan2(ray[1], ray[0])) % 360,
        elevation_deg=math.degrees(math.asin(float(ray[2] / radius))),
    )
    record = {
        "triangle": 1,
        "triangle_path": [proposal["proposal_id"]],
        "refinement_level": level,
        "stage": "next_best_view",
        "viewpoint_strategy": (
            "geometric_nbv" if strategy == "geometric" else "bayesian_nbv"
        ),
        "proposal": dict(proposal),
        "centroid_xyz_m": position.tolist(),
        "radius_m": radius,
    }
    return [record], [candidate]


def hemisphere_triangles(
    center_xyz: tuple[float, float, float],
    radius_m: float,
    count: int,
    azimuth_offset_deg: float = 0.0,
) -> tuple[list[dict[str, object]], list[ViewpointCandidate]]:
    """Split an upper hemisphere into congruent wedges and return their centroids."""

    center = np.asarray(center_xyz, dtype=float)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("center_xyz must contain three finite values.")
    if not math.isfinite(radius_m) or radius_m <= 0.0:
        raise ValueError("radius_m must be positive.")
    if not isinstance(count, int) or isinstance(count, bool) or count < 3:
        raise ValueError("count must be an integer of at least 3.")
    if not math.isfinite(azimuth_offset_deg):
        raise ValueError("azimuth_offset_deg must be finite.")

    triangles: list[dict[str, object]] = []
    viewpoints: list[ViewpointCandidate] = []
    top = center + (0.0, 0.0, radius_m)
    azimuth_offset = math.radians(azimuth_offset_deg)

    for index in range(count):
        angle_1 = azimuth_offset + 2.0 * math.pi * index / count
        angle_2 = azimuth_offset + 2.0 * math.pi * (index + 1) / count
        edge_1 = center + radius_m * np.asarray(
            (math.cos(angle_1), math.sin(angle_1), 0.0)
        )
        edge_2 = center + radius_m * np.asarray(
            (math.cos(angle_2), math.sin(angle_2), 0.0)
        )
        vertices = np.asarray((top, edge_1, edge_2))

        # Project the ordinary vertex centroid back onto the hemisphere shell.
        direction = np.mean(vertices, axis=0) - center
        centroid = center + radius_m * direction / np.linalg.norm(direction)
        rotation = look_at_rotation_matrix(centroid, center)
        azimuth_deg = math.degrees(math.atan2(direction[1], direction[0])) % 360.0
        elevation_deg = math.degrees(
            math.asin(direction[2] / np.linalg.norm(direction))
        )

        triangles.append(
            {
                "triangle": index + 1,
                "refinement_level": 0,
                "triangle_path": [index + 1],
                "parent_triangle_path": None,
                "vertices_xyz_m": vertices.tolist(),
                "centroid_xyz_m": centroid.tolist(),
                "surface_area_m2": 2.0 * math.pi * radius_m**2 / count,
            }
        )
        viewpoints.append(
            ViewpointCandidate(
                index=index,
                position_xyz=tuple(float(value) for value in centroid),
                orientation_xyzw=matrix_to_quaternion_xyzw(rotation),
                radius_m=radius_m,
                azimuth_deg=azimuth_deg,
                elevation_deg=elevation_deg,
            )
        )

    return triangles, viewpoints


def _spherical_centroid(
    vertices_xyz: np.ndarray,
    center_xyz: np.ndarray,
    radius_m: float,
) -> np.ndarray:
    """Project a triangle's ordinary centroid onto the viewpoint sphere."""

    direction = np.mean(vertices_xyz, axis=0) - center_xyz
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        raise ValueError("Triangle centroid direction is degenerate.")
    return center_xyz + radius_m * direction / norm


def _spherical_triangle_area(
    vertices_xyz: np.ndarray,
    center_xyz: np.ndarray,
    radius_m: float,
) -> float:
    """Return the minor spherical-triangle area using its solid angle."""

    directions = (vertices_xyz - center_xyz) / radius_m
    first, second, third = directions
    numerator = abs(float(np.dot(first, np.cross(second, third))))
    denominator = 1.0 + float(
        np.dot(first, second) + np.dot(second, third) + np.dot(third, first)
    )
    return 2.0 * math.atan2(numerator, denominator) * radius_m**2


def subdivide_spherical_triangle(
    parent: dict[str, object],
    center_xyz: tuple[float, float, float],
    radius_m: float,
) -> tuple[list[dict[str, object]], list[ViewpointCandidate]]:
    """Connect a spherical triangle's centroid to its edges, producing 3 children."""

    center = np.asarray(center_xyz, dtype=float)
    vertices = np.asarray(parent.get("vertices_xyz_m"), dtype=float)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("center_xyz must contain three finite values.")
    if not math.isfinite(radius_m) or radius_m <= 0.0:
        raise ValueError("radius_m must be positive.")
    if vertices.shape != (3, 3) or not np.isfinite(vertices).all():
        raise ValueError("Parent triangle must contain three finite XYZ vertices.")
    radii = np.linalg.norm(vertices - center, axis=1)
    if not np.allclose(radii, radius_m, atol=1e-7):
        raise ValueError("Parent triangle vertices must lie on the viewpoint sphere.")

    parent_path = [int(value) for value in parent.get("triangle_path", [])]
    if not parent_path:
        parent_path = [int(parent["triangle"])]
    level = int(parent.get("refinement_level", len(parent_path) - 1)) + 1
    split_point = _spherical_centroid(vertices, center, radius_m)
    child_vertices = (
        np.asarray((vertices[0], vertices[1], split_point)),
        np.asarray((vertices[1], vertices[2], split_point)),
        np.asarray((vertices[2], vertices[0], split_point)),
    )

    triangles: list[dict[str, object]] = []
    viewpoints: list[ViewpointCandidate] = []
    for index, child in enumerate(child_vertices):
        centroid = _spherical_centroid(child, center, radius_m)
        direction = centroid - center
        rotation = look_at_rotation_matrix(centroid, center)
        path = [*parent_path, index + 1]
        triangles.append(
            {
                "triangle": index + 1,
                "refinement_level": level,
                "triangle_path": path,
                "parent_triangle_path": parent_path,
                "vertices_xyz_m": child.tolist(),
                "centroid_xyz_m": centroid.tolist(),
                "surface_area_m2": _spherical_triangle_area(child, center, radius_m),
            }
        )
        viewpoints.append(
            ViewpointCandidate(
                index=index,
                position_xyz=tuple(float(value) for value in centroid),
                orientation_xyzw=matrix_to_quaternion_xyzw(rotation),
                radius_m=radius_m,
                azimuth_deg=(
                    math.degrees(math.atan2(direction[1], direction[0])) % 360.0
                ),
                elevation_deg=math.degrees(
                    math.asin(direction[2] / np.linalg.norm(direction))
                ),
            )
        )
    return triangles, viewpoints


def nearby_spherical_triangle_viewpoints(
    triangle: dict[str, object],
    center_xyz: tuple[float, float, float],
    radius_m: float,
    blend_fraction: float,
    *,
    start_index: int = 0,
) -> list[ViewpointCandidate]:
    """Sample three shell points near a triangle centroid, toward its vertices."""

    center = np.asarray(center_xyz, dtype=float)
    vertices = np.asarray(triangle.get("vertices_xyz_m"), dtype=float)
    centroid = np.asarray(triangle.get("centroid_xyz_m"), dtype=float)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("center_xyz must contain three finite values.")
    if vertices.shape != (3, 3) or not np.isfinite(vertices).all():
        raise ValueError("Triangle must contain three finite XYZ vertices.")
    if centroid.shape != (3,) or not np.isfinite(centroid).all():
        raise ValueError("Triangle must contain a finite centroid_xyz_m.")
    if not math.isfinite(radius_m) or radius_m <= 0.0:
        raise ValueError("radius_m must be positive.")
    if not math.isfinite(blend_fraction) or not 0.0 < blend_fraction < 1.0:
        raise ValueError("blend_fraction must be in (0, 1).")

    centroid_direction = centroid - center
    centroid_direction /= np.linalg.norm(centroid_direction)
    viewpoints: list[ViewpointCandidate] = []
    for vertex_index, vertex in enumerate(vertices):
        vertex_direction = vertex - center
        vertex_direction /= np.linalg.norm(vertex_direction)
        # A positive normalized blend follows the minor spherical segment from
        # the centroid toward one vertex and therefore stays inside this small,
        # convex spherical triangle.
        direction = (
            1.0 - blend_fraction
        ) * centroid_direction + blend_fraction * vertex_direction
        direction /= np.linalg.norm(direction)
        position = center + radius_m * direction
        rotation = look_at_rotation_matrix(position, center)
        viewpoints.append(
            ViewpointCandidate(
                index=start_index + vertex_index,
                position_xyz=tuple(float(value) for value in position),
                orientation_xyzw=matrix_to_quaternion_xyzw(rotation),
                radius_m=radius_m,
                azimuth_deg=(
                    math.degrees(math.atan2(direction[1], direction[0])) % 360.0
                ),
                elevation_deg=math.degrees(math.asin(direction[2])),
            )
        )
    return viewpoints


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_capture_poses_ply(path: Path, report: dict[str, object]) -> None:
    """Write one MeshLab vertex per image capture, with tool-Z normals."""

    palette = (
        (0, 114, 178),
        (230, 159, 0),
        (0, 158, 115),
        (204, 121, 167),
        (86, 180, 233),
        (213, 94, 0),
    )
    vertices: list[tuple[float, ...]] = []
    captures = report.get("captures")
    if not isinstance(captures, list):
        raise ValueError("Capture-pose report must contain a captures list.")

    for capture in captures:
        if not isinstance(capture, dict):
            raise ValueError("Each capture-pose entry must be an object.")
        pose = capture["end_effector_pose_world"]
        position = pose["position_xyz_m"]
        rotation = quaternion_xyzw_to_matrix(pose["orientation_xyzw"])
        normal = rotation[:, 2]
        level = int(capture["refinement_level"])
        color = palette[level % len(palette)]
        vertices.append(
            (
                float(position[0]),
                float(position[1]),
                float(position[2]),
                float(normal[0]),
                float(normal[1]),
                float(normal[2]),
                *color,
            )
        )

    lines = [
        "ply",
        "format ascii 1.0",
        "comment one tool0 pose per captured image",
        "comment PyBullet world coordinates in metres; normals are tool +Z",
        f"element vertex {len(vertices)}",
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    lines.extend(" ".join(str(value) for value in vertex) for vertex in vertices)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _append_capture_pose(
    report: dict[str, object],
    json_path: Path,
    ply_path: Path,
    capture: dict[str, object],
) -> int:
    """Append the robot state associated with one saved image."""

    captures = report.get("captures")
    if not isinstance(captures, list):
        raise ValueError("Capture-pose report must contain a captures list.")
    capture_index = len(captures)
    captures.append(
        {
            "capture_index": capture_index,
            "picture_number": capture_index + 1,
            **capture,
        }
    )
    write_json(json_path, report)
    write_capture_poses_ply(ply_path, report)
    return capture_index


def _new_evaluation_tracker(
    env: UR5EyeInHandPyBulletEnv,
) -> dict[str, object]:
    """Initialize executed-motion distance tracking at the current robot pose."""

    camera_position, _ = env.get_wrist_camera_pose()
    end_effector_position, _ = env.get_link_pose(END_EFFECTOR_LINK_NAME)
    return {
        "definition": (
            "Sum of Euclidean camera/end-effector displacement between every "
            "executed motion endpoint; final post-evaluation return home excluded."
        ),
        "camera_distance_traveled_m": 0.0,
        "end_effector_distance_traveled_m": 0.0,
        "last_camera_xyz_m": camera_position.astype(float).tolist(),
        "last_end_effector_xyz_m": [float(value) for value in end_effector_position],
        "motion_segments": [],
    }


def _record_motion_endpoint(
    env: UR5EyeInHandPyBulletEnv,
    tracker: dict[str, object] | None,
    label: str,
) -> None:
    """Accumulate straight-line displacement after one executed trajectory."""

    if tracker is None:
        return
    camera_position, _ = env.get_wrist_camera_pose()
    end_effector_position, _ = env.get_link_pose(END_EFFECTOR_LINK_NAME)
    previous_camera = np.asarray(tracker["last_camera_xyz_m"], dtype=float)
    previous_end_effector = np.asarray(tracker["last_end_effector_xyz_m"], dtype=float)
    camera_delta = float(np.linalg.norm(camera_position - previous_camera))
    end_effector_delta = float(
        np.linalg.norm(np.asarray(end_effector_position) - previous_end_effector)
    )
    tracker["camera_distance_traveled_m"] = (
        float(tracker["camera_distance_traveled_m"]) + camera_delta
    )
    tracker["end_effector_distance_traveled_m"] = (
        float(tracker["end_effector_distance_traveled_m"]) + end_effector_delta
    )
    tracker["last_camera_xyz_m"] = camera_position.astype(float).tolist()
    tracker["last_end_effector_xyz_m"] = [
        float(value) for value in end_effector_position
    ]
    segments = tracker.get("motion_segments")
    if isinstance(segments, list):
        segments.append(
            {
                "index": len(segments),
                "label": label,
                "camera_displacement_m": camera_delta,
                "end_effector_displacement_m": end_effector_delta,
                "camera_xyz_m": camera_position.astype(float).tolist(),
                "end_effector_xyz_m": [float(value) for value in end_effector_position],
            }
        )


def _camera_pose_from_mapping(
    value: dict[str, object],
) -> tuple[list[float], list[float] | None] | None:
    """Extract a camera position and optional quaternion from known schemas."""

    camera_pose = value.get("camera_pose_world")
    if isinstance(camera_pose, dict):
        position = camera_pose.get("position_xyz_m")
        orientation = camera_pose.get("orientation_xyzw")
        if isinstance(position, list) and len(position) == 3:
            return (
                [float(item) for item in position],
                (
                    [float(item) for item in orientation]
                    if isinstance(orientation, list) and len(orientation) == 4
                    else None
                ),
            )
    for position_key, orientation_key in (
        ("actual_camera_xyz_m", "actual_camera_orientation_xyzw"),
        ("capture_viewpoint_xyz_m", "camera_orientation_xyzw"),
        ("centroid_xyz_m", "camera_orientation_xyzw"),
    ):
        position = value.get(position_key)
        orientation = value.get(orientation_key)
        if isinstance(position, list) and len(position) == 3:
            return (
                [float(item) for item in position],
                (
                    [float(item) for item in orientation]
                    if isinstance(orientation, list) and len(orientation) == 4
                    else None
                ),
            )
    return None


def _iter_mappings(value: object):
    """Yield every nested mapping from a JSON-compatible value."""

    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _iter_mappings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_mappings(child)


def _mapping_matches_image(value: dict[str, object], image_path: Path) -> bool:
    image_file = value.get("image_file")
    if isinstance(image_file, str) and image_file:
        if Path(image_file).expanduser().resolve() == image_path:
            return True
    image_filename = value.get("image_filename")
    return isinstance(image_filename, str) and image_filename == image_path.name


def _resolve_target_camera_pose(
    args: argparse.Namespace,
) -> dict[str, object] | None:
    """Resolve target-view ground truth from CLI coordinates or a JSON sidecar."""

    if args.target_camera_position is not None:
        return {
            "position_xyz_m": [float(value) for value in args.target_camera_position],
            "orientation_xyzw": None,
            "source": "--target-camera-position",
        }
    if args.target_image is None:
        return None
    target_image = args.target_image.expanduser().resolve()
    json_paths = (
        [args.target_pose_json.expanduser().resolve()]
        if args.target_pose_json is not None
        else sorted(target_image.parent.glob("*.json"))
    )
    for json_path in json_paths:
        if not json_path.is_file():
            continue
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        for mapping in _iter_mappings(payload):
            if not _mapping_matches_image(mapping, target_image):
                continue
            pose = _camera_pose_from_mapping(mapping)
            if pose is not None:
                position, orientation = pose
                return {
                    "position_xyz_m": position,
                    "orientation_xyzw": orientation,
                    "source": str(json_path),
                }
    return None


def _find_capture_by_image(
    capture_pose_report: dict[str, object],
    image_file: str | Path,
) -> dict[str, object] | None:
    """Find the recorded camera pose associated with a selected result image."""

    image_path = Path(image_file).expanduser().resolve()
    captures = capture_pose_report.get("captures")
    if not isinstance(captures, list):
        return None
    for capture in reversed(captures):
        if isinstance(capture, dict) and _mapping_matches_image(capture, image_path):
            return capture
    return None


def _quaternion_error_deg(
    first_xyzw: list[float] | None,
    second_xyzw: list[float] | None,
) -> float | None:
    if first_xyzw is None or second_xyzw is None:
        return None
    first = np.asarray(first_xyzw, dtype=float)
    second = np.asarray(second_xyzw, dtype=float)
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm <= 1e-12 or second_norm <= 1e-12:
        return None
    cosine = abs(float(np.dot(first / first_norm, second / second_norm)))
    return math.degrees(2.0 * math.acos(float(np.clip(cosine, -1.0, 1.0))))


def _compute_image_quality(
    target_path: Path,
    observation_path: Path,
) -> dict[str, object]:
    """Compute full-reference PSNR and Gaussian-window SSIM on saved RGB images."""

    backend = "opencv"
    try:
        import cv2

        target = cv2.imread(str(target_path), cv2.IMREAD_COLOR)
        observation = cv2.imread(str(observation_path), cv2.IMREAD_COLOR)
        if target is None or observation is None:
            raise ValueError("Target or final observation image could not be decoded.")
        resized = observation.shape != target.shape
        if resized:
            observation = cv2.resize(
                observation,
                (target.shape[1], target.shape[0]),
                interpolation=cv2.INTER_AREA,
            )

        def gaussian_blur(image: np.ndarray) -> np.ndarray:
            return cv2.GaussianBlur(image, (11, 11), 1.5)

    except (ImportError, ValueError) as opencv_error:
        backend = "pillow_scipy"
        try:
            from PIL import Image
            from scipy.ndimage import gaussian_filter

            target_image = Image.open(target_path).convert("RGB")
            observation_image = Image.open(observation_path).convert("RGB")
            resized = observation_image.size != target_image.size
            if resized:
                observation_image = observation_image.resize(
                    target_image.size,
                    Image.Resampling.LANCZOS,
                )
            target = np.asarray(target_image)
            observation = np.asarray(observation_image)

            def gaussian_blur(image: np.ndarray) -> np.ndarray:
                return gaussian_filter(
                    image,
                    sigma=(1.5, 1.5, 0.0),
                    mode="reflect",
                    truncate=10.0 / 3.0,
                )

        except (ImportError, OSError, ValueError) as fallback_error:
            return {
                "psnr_db": None,
                "ssim": None,
                "error": (
                    f"OpenCV failed ({opencv_error}); image fallback failed "
                    f"({fallback_error})."
                ),
            }
    target_float = target.astype(np.float64)
    observation_float = observation.astype(np.float64)
    mean_squared_error = float(np.mean((target_float - observation_float) ** 2))
    psnr_db = (
        "inf"
        if mean_squared_error <= 1e-15
        else 10.0 * math.log10((255.0**2) / mean_squared_error)
    )

    constant_1 = (0.01 * 255.0) ** 2
    constant_2 = (0.03 * 255.0) ** 2
    mean_target = gaussian_blur(target_float)
    mean_observation = gaussian_blur(observation_float)
    variance_target = gaussian_blur(target_float**2) - (mean_target**2)
    variance_observation = gaussian_blur(observation_float**2) - mean_observation**2
    covariance = (
        gaussian_blur(target_float * observation_float) - mean_target * mean_observation
    )
    ssim_map = (
        (2.0 * mean_target * mean_observation + constant_1)
        * (2.0 * covariance + constant_2)
        / (
            (mean_target**2 + mean_observation**2 + constant_1)
            * (variance_target + variance_observation + constant_2)
        )
    )
    return {
        "psnr_db": psnr_db,
        "psnr_infinite": mean_squared_error <= 1e-15,
        "ssim": float(np.mean(ssim_map)),
        "mean_squared_error": mean_squared_error,
        "observation_resized_to_target": resized,
        "target_shape_hwc": list(target.shape),
        "implementation_backend": backend,
    }


def _evaluation_aggregate(
    csv_path: Path,
    method_name: str,
) -> dict[str, object]:
    """Return mean and sample standard deviation for one method's saved trials."""

    if not csv_path.is_file():
        return {"trial_count": 0, "metrics": {}}
    with csv_path.open("r", newline="", encoding="utf-8") as stream:
        rows = [
            row for row in csv.DictReader(stream) if row.get("method") == method_name
        ]
    field_names = (
        "n_views",
        "d_target_m",
        "d_traveled_cm",
        "t_exec_s",
        "psnr_db",
        "ssim",
    )
    metrics: dict[str, object] = {}
    for field_name in field_names:
        values: list[float] = []
        for row in rows:
            raw = row.get(field_name, "")
            if not raw:
                continue
            try:
                value = float(raw)
            except ValueError:
                continue
            if math.isfinite(value):
                values.append(value)
        metrics[field_name] = {
            "count": len(values),
            "mean": float(np.mean(values)) if values else None,
            "sample_std": (float(np.std(values, ddof=1)) if len(values) > 1 else 0.0),
        }
    return {"trial_count": len(rows), "metrics": metrics}


def _format_metric(value: object, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.{digits}f}"


def _write_evaluation_report(
    args: argparse.Namespace,
    capture_pose_report: dict[str, object],
    motion_tracker: dict[str, object],
    execution_time_s: float,
    final_observation: dict[str, object] | None,
    angular_termination_reason: str,
    radial_result: dict[str, object] | None,
) -> dict[str, object]:
    """Write per-run and aggregate metrics matching the comparison matrix."""

    timestamp = datetime.now(timezone.utc)
    run_id = timestamp.strftime("%Y%m%dT%H%M%S_%fZ")
    target_pose = _resolve_target_camera_pose(args)
    final_capture = None
    final_image = None
    if final_observation is not None:
        image_value = final_observation.get("best_image")
        if isinstance(image_value, str) and image_value:
            final_image = Path(image_value).expanduser().resolve()
            final_capture = _find_capture_by_image(
                capture_pose_report,
                final_image,
            )

    final_pose = (
        _camera_pose_from_mapping(final_capture)
        if isinstance(final_capture, dict)
        else None
    )
    distance_to_target_m = None
    orientation_error_deg = None
    if target_pose is not None and final_pose is not None:
        final_position, final_orientation = final_pose
        distance_to_target_m = float(
            np.linalg.norm(
                np.asarray(final_position, dtype=float)
                - np.asarray(target_pose["position_xyz_m"], dtype=float)
            )
        )
        orientation_error_deg = _quaternion_error_deg(
            final_orientation,
            target_pose.get("orientation_xyzw"),
        )

    image_quality = (
        _compute_image_quality(
            args.target_image.expanduser().resolve(),
            final_image,
        )
        if args.target_image is not None
        and final_image is not None
        and final_image.is_file()
        else {"psnr_db": None, "ssim": None}
    )
    captures = capture_pose_report.get("captures")
    view_count = len(captures) if isinstance(captures, list) else 0
    distance_traveled_m = float(motion_tracker["camera_distance_traveled_m"])
    radial_termination = (
        radial_result.get("termination_reason")
        if isinstance(radial_result, dict)
        else None
    )
    report: dict[str, object] = {
        "schema_version": 1,
        "run_id": run_id,
        "timestamp_utc": timestamp.isoformat(),
        "method": args.method_name,
        "target_image": (
            str(args.target_image.expanduser().resolve())
            if args.target_image is not None
            else None
        ),
        "final_observation_image": (
            str(final_image) if final_image is not None else None
        ),
        "final_similarity_score": (
            float(final_observation["best_score"])
            if final_observation is not None
            else None
        ),
        "metrics": {
            "n_views": view_count,
            "d_target_m": distance_to_target_m,
            "d_target_cm": (
                distance_to_target_m * 100.0
                if distance_to_target_m is not None
                else None
            ),
            "d_traveled_m": distance_traveled_m,
            "d_traveled_cm": distance_traveled_m * 100.0,
            "end_effector_distance_traveled_m": float(
                motion_tracker["end_effector_distance_traveled_m"]
            ),
            "t_exec_s": execution_time_s,
            "psnr_db": image_quality.get("psnr_db"),
            "ssim": image_quality.get("ssim"),
            "camera_orientation_error_deg": orientation_error_deg,
        },
        "target_pose_ground_truth": target_pose,
        "image_quality_details": image_quality,
        "motion_details": motion_tracker,
        "termination": {
            "angular": angular_termination_reason,
            "radial": radial_termination,
        },
        "definitions": {
            "n_views": "Number of wrist-camera images captured during this run.",
            "d_target_m": (
                "Euclidean distance between final-best and target camera centers."
            ),
            "d_traveled_m": (
                "Sum of camera-center displacement between executed motion "
                "endpoints, from initial pose through final-best revisit; final "
                "return home excluded."
            ),
            "t_exec_s": (
                "Wall-clock algorithm time from backend/search initialization "
                "through final-best selection, excluding final return home."
            ),
            "psnr_db": "Full-reference PSNR after matching image dimensions.",
            "ssim": (
                "Full-reference Gaussian-window SSIM (11x11, sigma 1.5) on color images."
            ),
        },
    }

    csv_path = args.output_dir / f"{args.name}_evaluation_runs.csv"
    field_names = (
        "run_id",
        "timestamp_utc",
        "method",
        "target_image",
        "n_views",
        "d_target_m",
        "d_target_cm",
        "d_traveled_m",
        "d_traveled_cm",
        "t_exec_s",
        "psnr_db",
        "ssim",
        "final_similarity_score",
        "angular_termination",
        "radial_termination",
    )
    metrics = report["metrics"]
    assert isinstance(metrics, dict)
    csv_row = {
        "run_id": run_id,
        "timestamp_utc": report["timestamp_utc"],
        "method": args.method_name,
        "target_image": report["target_image"],
        "n_views": metrics["n_views"],
        "d_target_m": metrics["d_target_m"],
        "d_target_cm": metrics["d_target_cm"],
        "d_traveled_m": metrics["d_traveled_m"],
        "d_traveled_cm": metrics["d_traveled_cm"],
        "t_exec_s": metrics["t_exec_s"],
        "psnr_db": metrics["psnr_db"],
        "ssim": metrics["ssim"],
        "final_similarity_score": report["final_similarity_score"],
        "angular_termination": angular_termination_reason,
        "radial_termination": radial_termination,
    }
    write_header = not csv_path.is_file() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=field_names)
        if write_header:
            writer.writeheader()
        writer.writerow(csv_row)

    aggregate = _evaluation_aggregate(csv_path, args.method_name)
    report["aggregate_for_method"] = aggregate
    report_paths = {
        "run_json": str(
            (args.output_dir / f"{args.name}_evaluation_{run_id}.json").resolve()
        ),
        "run_text": str(
            (args.output_dir / f"{args.name}_evaluation_{run_id}.txt").resolve()
        ),
        "latest_json": str(
            (args.output_dir / f"{args.name}_evaluation.json").resolve()
        ),
        "latest_text": str((args.output_dir / f"{args.name}_evaluation.txt").resolve()),
        "runs_csv": str(csv_path.resolve()),
    }
    report["report_files"] = report_paths

    text_lines = [
        "ACTIVE-PERCEPTION EVALUATION REPORT",
        f"Run ID: {run_id}",
        f"Method: {args.method_name}",
        f"Target: {report['target_image']}",
        "",
        "Matrix-ready metrics (single run)",
        f"n_views       = {_format_metric(metrics['n_views'], 0)}",
        f"d_target      = {_format_metric(metrics['d_target_m'])} m",
        f"d_traveled    = {_format_metric(metrics['d_traveled_cm'], 2)} cm",
        f"t_exec        = {_format_metric(metrics['t_exec_s'], 3)} s",
        f"PSNR          = {_format_metric(metrics['psnr_db'], 3)} dB",
        f"SSIM          = {_format_metric(metrics['ssim'], 5)}",
        "",
        f"Final similarity score = {_format_metric(report['final_similarity_score'], 5)}",
        f"Camera orientation error = {_format_metric(metrics['camera_orientation_error_deg'], 3)} deg",
        "",
        "Definitions",
    ]
    definitions = report["definitions"]
    assert isinstance(definitions, dict)
    text_lines.extend(f"- {key}: {value}" for key, value in definitions.items())
    if target_pose is None:
        text_lines.extend(
            (
                "",
                "WARNING: d_target is unavailable because target-camera ground "
                "truth was not found.",
                "Provide --target-camera-position X Y Z or --target-pose-json FILE.",
            )
        )
    text_lines.extend(("", "Aggregate for this method"))
    aggregate_metrics = aggregate.get("metrics")
    if isinstance(aggregate_metrics, dict):
        text_lines.append(f"Trials: {aggregate.get('trial_count', 0)}")
        for field_name, values in aggregate_metrics.items():
            if not isinstance(values, dict):
                continue
            text_lines.append(
                f"{field_name}: {_format_metric(values.get('mean'))} ± "
                f"{_format_metric(values.get('sample_std'))} "
                f"(n={values.get('count', 0)})"
            )
    text_value = "\n".join(text_lines) + "\n"
    run_json_path = Path(report_paths["run_json"])
    latest_json_path = Path(report_paths["latest_json"])
    run_text_path = Path(report_paths["run_text"])
    latest_text_path = Path(report_paths["latest_text"])
    write_json(run_json_path, report)
    write_json(latest_json_path, report)
    run_text_path.write_text(text_value, encoding="utf-8")
    latest_text_path.write_text(text_value, encoding="utf-8")
    print(f"Saved evaluation report: {latest_text_path}")
    print(f"Appended evaluation row: {csv_path}")
    return report


def _target_guided_perception_enabled(args: argparse.Namespace) -> bool:
    """Return whether target-guided ranking and refinement should run."""

    # Debugging controls diagnostics only. Requiring it for the actual search
    # made every production run pay for clustering and visualization output.
    return args.target_image is not None


def _save_final_best_comparison(
    best_observation: dict[str, object],
    output_dir: Path,
    name: str,
) -> Path | None:
    """Copy the overall winner's target/view panel to a stable final path."""

    source_value = best_observation.get("best_comparison")
    if not isinstance(source_value, str) or not source_value:
        return None
    source = Path(source_value).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Best comparison image not found: {source}")

    destination = output_dir / f"{name}_best_view_with_target.png"
    if source != destination.resolve():
        shutil.copy2(source, destination)
    return destination


def _finalize_refinement_run(
    args: argparse.Namespace,
    level_results: list[dict[str, object]],
    radial_result: dict[str, object] | None,
    termination_reason: str,
    refinement_summary_path: Path,
    capture_pose_json_path: Path,
    capture_pose_ply_path: Path,
    evaluation_report: dict[str, object] | None,
    nbv_summary: dict[str, object] | None = None,
    localization_report: dict[str, object] | None = None,
) -> None:
    """Write a complete summary, including graceful no-reachable termination."""

    final_result = level_results[-1] if level_results else None
    observation_candidates = list(level_results)
    if radial_result is not None:
        radial_best = radial_result.get("best_observation")
        if isinstance(radial_best, dict):
            observation_candidates.append(radial_best)
    best_observation = (
        max(observation_candidates, key=lambda item: float(item["best_score"]))
        if observation_candidates
        else None
    )
    final_comparison_path = (
        _save_final_best_comparison(
            best_observation,
            args.output_dir,
            args.name,
        )
        if best_observation is not None
        else None
    )
    summary = {
        "schema_version": 1,
        "search_strategy": args.search_strategy,
        "next_best_view": nbv_summary,
        "object_localization": {"source": args.object_localization, "report": localization_report},
        "search_mode": (
            ("geometric_nbv" if args.search_strategy == "geometric" else "bayesian_nbv")
            if nbv_summary is not None
            else "until_convergence" if args.until_convergence else "bounded"
        ),
        "views": 1 if nbv_summary is not None else args.views,
        "convergence": (
            {
                "min_improvement": args.convergence_min_improvement,
                "patience": args.convergence_patience,
                "reached": termination_reason == "similarity_converged",
            }
            if args.until_convergence
            else None
        ),
        "target_image": str(args.target_image.expanduser().resolve()),
        "similarity_threshold": args.similarity_threshold,
        "max_refinement_depth": (
            args.max_refinement_depth if nbv_summary is None else None
        ),
        "refinement_patience": (
            args.refinement_patience if nbv_summary is None else None
        ),
        "min_score_improvement": args.min_score_improvement,
        "angular_lookahead_margin": (
            args.angular_lookahead_margin if nbv_summary is None else None
        ),
        "angular_lookahead_levels": (
            args.angular_lookahead_levels if nbv_summary is None else None
        ),
        "return_home_each_level": args.return_home_each_level,
        "debug_diagnostics": args.debug,
        "feature_max_image_size": args.feature_max_image_size,
        "radial_refinement_enabled": not args.skip_radial_refinement,
        "radial_step_m": args.radial_step,
        "radial_probe_steps": args.radial_probe_steps,
        "radial_max_steps": args.radial_max_steps,
        "radial_patience": args.radial_patience,
        "radial_min_improvement": args.radial_min_improvement,
        "radial_min_radius_m": args.radial_min_radius,
        "radial_max_radius_m": args.radial_max_radius,
        "angular_termination_reason": termination_reason,
        "termination_reason": termination_reason,
        "levels_completed": len(level_results),
        "total_images_captured": int((localization_report or {}).get("captured_views", 0)) + sum(
            int(item["candidate_count"]) for item in level_results
        )
        + (
            int(radial_result.get("candidate_count", 0))
            if radial_result is not None
            else 0
        ),
        "levels": level_results,
        "final_refinement": final_result,
        "radial_refinement": radial_result,
        "evaluation": evaluation_report,
        "final_best": best_observation,
        "best_view_with_target": (
            str(final_comparison_path.resolve())
            if final_comparison_path is not None
            else None
        ),
        "end_effector_capture_poses": str(capture_pose_json_path.resolve()),
        "end_effector_capture_pose_mesh": str(capture_pose_ply_path.resolve()),
    }
    write_json(refinement_summary_path, summary)
    print(f"Search termination: {termination_reason}")
    if best_observation is not None:
        print(
            f"Final best observation: stage={best_observation.get('stage', 'angular_refinement')}, "
            f"score={float(best_observation['best_score']):.3f}."
        )
    print(f"Saved refinement summary: {refinement_summary_path}")
    if final_comparison_path is not None:
        print(f"Saved best view with target: {final_comparison_path}")


def _annotate_reachability(
    triangles: list[dict[str, object]],
    reachability_report,
    *,
    azimuth_offset_deg: float | None = None,
) -> dict[int, object]:
    """Attach reachability evidence and return accepted results by local index."""

    reachable = {item.viewpoint.index: item for item in reachability_report.reachable}
    rejected = {item.viewpoint.index: item for item in reachability_report.rejected}
    for index, triangle in enumerate(triangles):
        selected = reachable.get(index)
        if azimuth_offset_deg is not None:
            triangle["azimuth_offset_deg"] = azimuth_offset_deg
        triangle["reachable"] = selected is not None
        if selected is not None:
            triangle.setdefault("viewpoint_strategy", "triangle_centroid")
            triangle["capture_viewpoint_xyz_m"] = list(selected.viewpoint.position_xyz)
            triangle["camera_orientation_xyzw"] = list(
                selected.viewpoint.orientation_xyzw
            )
            triangle["manipulability"] = selected.manipulability
            triangle["quality_score"] = selected.quality_score
        else:
            rejection = rejected[index]
            triangle["rejection_reason"] = rejection.furthest_failure_stage
            triangle["failure_counts"] = rejection.failure_counts
    return reachable


def _recover_unreachable_triangle_viewpoints(
    env: UR5EyeInHandPyBulletEnv,
    triangles: list[dict[str, object]],
    reachable: dict[int, object],
    object_center: tuple[float, float, float],
    radius_m: float,
    config: ReachabilityConfig,
    candidate_backend: object | None,
) -> dict[int, object]:
    """Try nearby in-triangle viewpoints, then leave failures marked to skip."""

    unresolved = {index for index in range(len(triangles)) if index not in reachable}
    if not unresolved:
        return reachable

    for index in unresolved:
        triangles[index]["fallback_attempts"] = []

    next_candidate_index = 0
    for blend_fraction in TRIANGLE_FALLBACK_BLEND_FRACTIONS:
        if not unresolved:
            break
        candidates: list[ViewpointCandidate] = []
        owner_by_candidate: dict[int, tuple[int, int, float]] = {}
        for triangle_index in sorted(unresolved):
            alternatives = nearby_spherical_triangle_viewpoints(
                triangles[triangle_index],
                object_center,
                radius_m,
                blend_fraction,
                start_index=next_candidate_index,
            )
            next_candidate_index += len(alternatives)
            nominal = np.asarray(
                triangles[triangle_index]["centroid_xyz_m"], dtype=float
            )
            nominal_direction = nominal - np.asarray(object_center, dtype=float)
            nominal_direction /= np.linalg.norm(nominal_direction)
            for vertex_index, candidate in enumerate(alternatives):
                candidate_direction = (
                    np.asarray(candidate.position_xyz, dtype=float)
                    - np.asarray(object_center, dtype=float)
                ) / radius_m
                angular_offset_deg = math.degrees(
                    math.acos(
                        float(
                            np.clip(
                                np.dot(nominal_direction, candidate_direction),
                                -1.0,
                                1.0,
                            )
                        )
                    )
                )
                owner_by_candidate[candidate.index] = (
                    triangle_index,
                    vertex_index,
                    angular_offset_deg,
                )
                candidates.append(candidate)

        print(
            f"Trying {len(candidates)} nearby in-triangle fallback viewpoint(s) "
            f"for {len(unresolved)} unreachable triangle(s) "
            f"(blend={blend_fraction:.2f}) ...",
            flush=True,
        )
        report = env.filter_reachable_viewpoints(
            candidates,
            object_center,
            config,
            candidate_backend=candidate_backend,
        )
        accepted = {item.viewpoint.index: item for item in report.reachable}
        rejected = {item.viewpoint.index: item for item in report.rejected}
        accepted_by_triangle: dict[int, list[tuple[float, object]]] = {}
        for candidate in candidates:
            triangle_index, vertex_index, angular_offset_deg = owner_by_candidate[
                candidate.index
            ]
            selected = accepted.get(candidate.index)
            attempt: dict[str, object] = {
                "blend_fraction": blend_fraction,
                "toward_vertex": vertex_index + 1,
                "angular_offset_deg": angular_offset_deg,
                "position_xyz_m": list(candidate.position_xyz),
                "reachable": selected is not None,
            }
            if selected is None:
                rejection = rejected[candidate.index]
                attempt["rejection_reason"] = rejection.furthest_failure_stage
                attempt["failure_counts"] = rejection.failure_counts
            else:
                accepted_by_triangle.setdefault(triangle_index, []).append(
                    (angular_offset_deg, selected)
                )
            attempts = triangles[triangle_index]["fallback_attempts"]
            if not isinstance(attempts, list):  # pragma: no cover - initialized above
                raise RuntimeError("Triangle fallback_attempts must be a list.")
            attempts.append(attempt)

        for triangle_index, solutions in accepted_by_triangle.items():
            _, selected = min(
                solutions,
                key=lambda item: (
                    item[0],
                    -float(item[1].quality_score),
                ),
            )
            triangle = triangles[triangle_index]
            triangle["reachable"] = True
            triangle["viewpoint_strategy"] = "nearby_in_triangle_fallback"
            triangle["capture_viewpoint_xyz_m"] = list(selected.viewpoint.position_xyz)
            triangle["camera_orientation_xyzw"] = list(
                selected.viewpoint.orientation_xyzw
            )
            triangle["manipulability"] = selected.manipulability
            triangle["quality_score"] = selected.quality_score
            triangle["nominal_rejection_reason"] = triangle.pop(
                "rejection_reason", None
            )
            triangle["nominal_failure_counts"] = triangle.pop("failure_counts", None)
            triangle["fallback_angular_offset_deg"] = min(
                angular_offset
                for angular_offset, solution in solutions
                if solution is selected
            )
            triangle["fallback_blend_fraction"] = blend_fraction
            reachable[triangle_index] = selected
            unresolved.discard(triangle_index)
            path_label = ".".join(str(value) for value in triangle["triangle_path"])
            print(
                f"Recovered triangle path {path_label} with nearby viewpoint "
                f"{triangle['fallback_angular_offset_deg']:.2f} degrees from its centroid."
            )

    for triangle_index in sorted(unresolved):
        triangle = triangles[triangle_index]
        triangle["reachable"] = False
        triangle["skipped"] = True
        triangle["skip_reason"] = "unreachable_after_nearby_fallbacks"
        path_label = ".".join(str(value) for value in triangle["triangle_path"])
        print(
            f"Skipping triangle path {path_label}: no centroid or nearby "
            "in-triangle viewpoint passed reachability validation."
        )
    return reachable


def _capture_round(
    env: UR5EyeInHandPyBulletEnv,
    triangles: list[dict[str, object]],
    reachable: dict[int, object],
    coordinates_path: Path,
    image_prefix: str,
    motion_seconds: float,
    motion_planner: CuRoboMotionPlanner | None,
    outward_paths: list[np.ndarray],
    capture_pose_report: dict[str, object],
    capture_pose_json_path: Path,
    capture_pose_ply_path: Path,
    evaluation_tracker: dict[str, object] | None,
    *,
    return_home_after_round: bool,
) -> int:
    """Visit reachable viewpoints, skip rejected proposals, and count captures."""

    write_json(coordinates_path, triangles)
    missing = [
        ".".join(str(value) for value in triangle["triangle_path"])
        for index, triangle in enumerate(triangles)
        if index not in reachable
    ]
    if missing:
        print("Neglecting unreachable triangle path(s): " + ", ".join(missing) + ".")
    if not reachable:
        print(f"No reachable viewpoints remain in this round: {coordinates_path}")
        return 0

    captured_count = 0
    for index, triangle in enumerate(triangles):
        if index not in reachable:
            continue
        path_label = ".".join(str(value) for value in triangle["triangle_path"])
        view_label = (
            f"NBV proposal {path_label}"
            if triangle.get("stage") == "next_best_view"
            else f"triangle path {path_label}"
        )
        selected = reachable[index]
        path = None
        try:
            path, motion_metadata = plan_validated_path(
                env,
                selected.joint_positions_rad,
                motion_planner,
            )
        except RuntimeError as exc:
            motion_error = exc
            if outward_paths:
                print(f"Direct path failed ({exc}); returning home and replanning.")
                move_home(env, motion_seconds, outward_paths, motion_planner)
                _record_motion_endpoint(
                    env,
                    evaluation_tracker,
                    "motion_planning_recovery_home",
                )
                outward_paths.clear()
                try:
                    path, motion_metadata = plan_validated_path(
                        env,
                        selected.joint_positions_rad,
                        motion_planner,
                    )
                except RuntimeError as retry_exc:
                    motion_error = retry_exc
            if path is None:
                triangle["reachable"] = False
                triangle["skipped"] = True
                triangle["skip_reason"] = "motion_planning_failed"
                triangle["motion_planning_error"] = (
                    f"{type(motion_error).__name__}: {motion_error}"
                )
                reachable.pop(index, None)
                write_json(coordinates_path, triangles)
                print(
                    f"Skipping {view_label}: its validated IK "
                    f"endpoint has no executable path ({motion_error})."
                )
                continue
        play_path(env, path, motion_seconds)
        outward_paths.append(path)
        _record_motion_endpoint(
            env,
            evaluation_tracker,
            (
                "nbv_capture_target"
                if triangle.get("stage") == "next_best_view"
                else "angular_capture_target"
            ),
        )

        image_path = (
            coordinates_path.parent / f"{image_prefix}_view_{index + 1:02d}.png"
        )
        env.save_wrist_camera_frame(image_path)
        camera_position, camera_rotation = env.get_wrist_camera_pose()
        end_effector_position, end_effector_orientation = env.get_link_pose(
            END_EFFECTOR_LINK_NAME
        )
        joint_positions = env.get_joint_positions()
        camera_orientation = list(matrix_to_quaternion_xyzw(camera_rotation))
        end_effector_pose = {
            "position_xyz_m": [float(value) for value in end_effector_position],
            "orientation_xyzw": [float(value) for value in end_effector_orientation],
        }
        capture_pose_index = _append_capture_pose(
            capture_pose_report,
            capture_pose_json_path,
            capture_pose_ply_path,
            {
                "stage": triangle.get("stage", "angular_refinement"),
                "refinement_level": int(triangle["refinement_level"]),
                "triangle": int(triangle["triangle"]),
                "triangle_path": list(triangle["triangle_path"]),
                "image_filename": image_path.name,
                "image_file": str(image_path.resolve()),
                "viewpoints_file": str(coordinates_path.resolve()),
                "joint_positions_rad": [float(value) for value in joint_positions],
                "end_effector_pose_world": end_effector_pose,
                "camera_pose_world": {
                    "position_xyz_m": camera_position.astype(float).tolist(),
                    "orientation_xyzw": camera_orientation,
                },
            },
        )
        triangle["image_filename"] = image_path.name
        triangle["actual_camera_xyz_m"] = camera_position.tolist()
        triangle["actual_camera_orientation_xyzw"] = camera_orientation
        triangle["joint_positions_rad"] = joint_positions
        triangle["executed_motion"] = motion_metadata
        triangle["end_effector_link"] = END_EFFECTOR_LINK_NAME
        triangle["end_effector_pose_world"] = end_effector_pose
        triangle["capture_pose_file"] = str(capture_pose_json_path.resolve())
        triangle["capture_pose_index"] = capture_pose_index
        triangle["picture_number"] = capture_pose_index + 1
        captured_count += 1
        write_json(coordinates_path, triangles)
        print(f"Saved {view_label}: {image_path}")

    # A refinement level is local to its parent triangle, so returning home
    # here only adds a long reverse traversal before the next nearby capture.
    # Keep the current state by default; run() still performs one safe return
    # when the complete search finishes. The legacy behavior remains opt-in.
    if return_home_after_round:
        print("Returning robot to its home position ...")
        move_home(env, motion_seconds, outward_paths, motion_planner)
        _record_motion_endpoint(
            env,
            evaluation_tracker,
            "between_level_home",
        )
        outward_paths.clear()
    print(f"Saved round coordinates: {coordinates_path}")
    return captured_count


def _find_angular_best_capture(
    capture_pose_report: dict[str, object],
    triangle_path: list[int],
) -> dict[str, object]:
    """Return the recorded robot state for an angular-search winner."""

    captures = capture_pose_report.get("captures")
    if not isinstance(captures, list):
        raise ValueError("Capture-pose report must contain a captures list.")
    wanted = [int(value) for value in triangle_path]
    matches = [
        capture
        for capture in captures
        if isinstance(capture, dict)
        and capture.get("stage", "angular_refinement") == "angular_refinement"
        and [int(value) for value in capture.get("triangle_path", [])] == wanted
    ]
    if not matches:
        raise RuntimeError(
            "Could not recover the saved joint state for angular triangle path "
            + ".".join(str(value) for value in wanted)
            + "."
        )
    return matches[-1]


def _execute_joint_target(
    env: UR5EyeInHandPyBulletEnv,
    joint_positions_rad: list[float] | tuple[float, ...] | np.ndarray,
    motion_planner: CuRoboMotionPlanner | None,
    motion_seconds: float,
    outward_paths: list[np.ndarray],
    evaluation_tracker: dict[str, object] | None = None,
    motion_label: str = "radial_motion_target",
) -> dict[str, object]:
    """Plan and execute one target while retaining a reversible path history."""

    path, metadata = plan_validated_path(
        env,
        joint_positions_rad,
        motion_planner,
    )
    play_path(env, path, motion_seconds)
    outward_paths.append(path)
    _record_motion_endpoint(env, evaluation_tracker, motion_label)
    return metadata


def _make_radial_sample(
    sample_id: int,
    radial_step_index: int,
    center_xyz: np.ndarray,
    radial_direction: np.ndarray,
    base_radius_m: float,
    step_size_m: float,
    angular_level: int,
    angular_triangle_path: list[int],
) -> tuple[dict[str, object], ViewpointCandidate]:
    """Build one camera pose on the best angular ray."""

    radius_m = base_radius_m + radial_step_index * step_size_m
    if radius_m <= 0.0:
        raise ValueError("A radial candidate radius must remain positive.")
    position = center_xyz + radius_m * radial_direction
    rotation = look_at_rotation_matrix(position, center_xyz)
    orientation = matrix_to_quaternion_xyzw(rotation)
    direction_label = "inside" if radial_step_index < 0 else "outside"
    sample = {
        # rank_triangle_views consumes the same compact schema as an angular
        # round, while the radial fields preserve the one-dimensional search.
        "triangle": sample_id,
        "refinement_level": angular_level + 1,
        "triangle_path": list(angular_triangle_path),
        "stage": "radial_refinement",
        "radial_direction": direction_label,
        "radial_step_index": radial_step_index,
        "radial_offset_m": radial_step_index * step_size_m,
        "radius_m": radius_m,
        "centroid_xyz_m": position.astype(float).tolist(),
        "capture_viewpoint_xyz_m": position.astype(float).tolist(),
        "camera_orientation_xyzw": [float(value) for value in orientation],
        "reachable": False,
    }
    viewpoint = ViewpointCandidate(
        index=sample_id,
        position_xyz=tuple(float(value) for value in position),
        orientation_xyzw=orientation,
        radius_m=radius_m,
        azimuth_deg=(
            math.degrees(math.atan2(radial_direction[1], radial_direction[0])) % 360.0
        ),
        elevation_deg=math.degrees(math.asin(float(radial_direction[2]))),
    )
    return sample, viewpoint


def _capture_radial_sample(
    args: argparse.Namespace,
    env: UR5EyeInHandPyBulletEnv,
    sample: dict[str, object],
    selected: object,
    motion_planner: CuRoboMotionPlanner | None,
    outward_paths: list[np.ndarray],
    master_coordinates_path: Path,
    capture_pose_report: dict[str, object],
    capture_pose_json_path: Path,
    capture_pose_ply_path: Path,
    evaluation_tracker: dict[str, object] | None,
) -> bool:
    """Execute and record one validated radial viewpoint."""

    try:
        motion_metadata = _execute_joint_target(
            env,
            selected.joint_positions_rad,
            motion_planner,
            args.motion_seconds,
            outward_paths,
            evaluation_tracker,
            "radial_capture_target",
        )
    except RuntimeError as exc:
        sample["reachable"] = False
        sample["skipped"] = True
        sample["skip_reason"] = "motion_planning_failed"
        sample["motion_planning_error"] = f"{type(exc).__name__}: {exc}"
        return False

    step_index = int(sample["radial_step_index"])
    direction_label = str(sample["radial_direction"])
    image_path = (
        args.output_dir
        / f"{args.name}_radial_{direction_label}_{abs(step_index):02d}.png"
    )
    env.save_wrist_camera_frame(image_path)
    camera_position, camera_rotation = env.get_wrist_camera_pose()
    end_effector_position, end_effector_orientation = env.get_link_pose(
        END_EFFECTOR_LINK_NAME
    )
    joint_positions = env.get_joint_positions()
    camera_orientation = list(matrix_to_quaternion_xyzw(camera_rotation))
    end_effector_pose = {
        "position_xyz_m": [float(value) for value in end_effector_position],
        "orientation_xyzw": [float(value) for value in end_effector_orientation],
    }
    capture_pose_index = _append_capture_pose(
        capture_pose_report,
        capture_pose_json_path,
        capture_pose_ply_path,
        {
            "stage": "radial_refinement",
            "refinement_level": int(sample["refinement_level"]),
            "triangle": int(sample["triangle"]),
            "triangle_path": list(sample["triangle_path"]),
            "radial_direction": direction_label,
            "radial_step_index": step_index,
            "radial_offset_m": float(sample["radial_offset_m"]),
            "radius_m": float(sample["radius_m"]),
            "image_filename": image_path.name,
            "image_file": str(image_path.resolve()),
            "viewpoints_file": str(master_coordinates_path.resolve()),
            "joint_positions_rad": [float(value) for value in joint_positions],
            "end_effector_pose_world": end_effector_pose,
            "camera_pose_world": {
                "position_xyz_m": camera_position.astype(float).tolist(),
                "orientation_xyzw": camera_orientation,
            },
        },
    )
    sample.update(
        {
            "reachable": True,
            "image_filename": image_path.name,
            "image_file": str(image_path.resolve()),
            "actual_camera_xyz_m": camera_position.astype(float).tolist(),
            "actual_camera_orientation_xyzw": camera_orientation,
            "joint_positions_rad": [float(value) for value in joint_positions],
            "executed_motion": motion_metadata,
            "end_effector_link": END_EFFECTOR_LINK_NAME,
            "end_effector_pose_world": end_effector_pose,
            "capture_pose_file": str(capture_pose_json_path.resolve()),
            "capture_pose_index": capture_pose_index,
            "picture_number": capture_pose_index + 1,
        }
    )
    print(
        f"Saved radial {direction_label} step {abs(step_index)} "
        f"({float(sample['radial_offset_m']):+.3f} m): {image_path}"
    )
    return True


def _score_radial_sample(
    args: argparse.Namespace,
    sample: dict[str, object],
    rank_triangle_views: object,
    feature_extractor: object,
    spatial_clusterer: object | None,
    prepared_target: object,
) -> float:
    """Score only the new radial image so earlier DINO features are not repeated."""

    sample_id = int(sample["triangle"])
    evaluation_path = args.output_dir / f"{args.name}_radial_eval_{sample_id:02d}.json"
    write_json(evaluation_path, [sample])
    ranking_report = rank_triangle_views(
        args.target_image,
        evaluation_path,
        ambiguity_margin=args.ambiguity_margin,
        dino_weight=args.dino_weight,
        sift_weight=args.sift_weight,
        debug=args.debug,
        extractor=feature_extractor,
        clusterer=spatial_clusterer,
        prepared_target=prepared_target,
    )
    score = float(ranking_report["ranking"][0]["score"])
    visualizations = ranking_report.get("visualizations")
    sample.update(
        {
            "score": score,
            "ranking_file": ranking_report["report_file"],
            "best_image": ranking_report["best_image"],
            "best_viewpoint": ranking_report["best_viewpoint"],
            "best_comparison": (
                visualizations.get("best_comparison")
                if isinstance(visualizations, dict)
                else None
            ),
        }
    )
    write_json(evaluation_path, [sample])
    return score


def _run_radial_refinement(
    args: argparse.Namespace,
    env: UR5EyeInHandPyBulletEnv,
    object_center: tuple[float, float, float],
    config: ReachabilityConfig,
    candidate_backend: object | None,
    motion_planner: CuRoboMotionPlanner | None,
    outward_paths: list[np.ndarray],
    capture_pose_report: dict[str, object],
    capture_pose_json_path: Path,
    capture_pose_ply_path: Path,
    level_results: list[dict[str, object]],
    rank_triangle_views: object,
    feature_extractor: object,
    spatial_clusterer: object | None,
    prepared_target: object,
    evaluation_tracker: dict[str, object] | None,
) -> dict[str, object]:
    """Perform one-dimensional closer/farther search from the angular optimum."""

    angular_best = max(level_results, key=lambda item: float(item["best_score"]))
    angular_path = [int(value) for value in angular_best["best_triangle_path"]]
    base_capture = _find_angular_best_capture(capture_pose_report, angular_path)
    center = np.asarray(object_center, dtype=float)
    base_camera = np.asarray(
        base_capture["camera_pose_world"]["position_xyz_m"], dtype=float
    )
    ray = base_camera - center
    base_radius_m = float(np.linalg.norm(ray))
    if base_radius_m <= 1e-9:
        raise RuntimeError("The best camera pose coincides with the object center.")
    ray /= base_radius_m

    master_path = args.output_dir / f"{args.name}_radial_viewpoints.json"
    summary_path = args.output_dir / f"{args.name}_radial_summary.json"
    samples: list[dict[str, object]] = []
    captured_count = 0
    next_sample_id = 1
    selected_direction: str | None = None
    termination_reason = "initial_probes_complete"

    print("\n=== Radial fine alignment ===")
    print(
        f"Starting from angular best score={float(angular_best['best_score']):.3f}, "
        f"radius={base_radius_m:.3f} m."
    )
    try:
        _execute_joint_target(
            env,
            base_capture["joint_positions_rad"],
            motion_planner,
            args.motion_seconds,
            outward_paths,
            evaluation_tracker,
            "return_to_angular_best",
        )
    except RuntimeError as exc:
        result = {
            "enabled": True,
            "termination_reason": "could_not_return_to_angular_best",
            "error": f"{type(exc).__name__}: {exc}",
            "candidate_count": 0,
            "samples": [],
            "best_observation": angular_best,
            "summary_file": str(summary_path.resolve()),
        }
        write_json(summary_path, result)
        return result

    # Validate the symmetric initial probe set together so the CuRobo IK
    # backend can process it efficiently. Execute closer and farther sweeps
    # separately, returning to the angular optimum between them.
    initial_pairs: list[tuple[dict[str, object], ViewpointCandidate]] = []
    for signed_step in (
        *range(-1, -args.radial_probe_steps - 1, -1),
        *range(1, args.radial_probe_steps + 1),
    ):
        candidate_radius = base_radius_m + signed_step * args.radial_step
        if not args.radial_min_radius <= candidate_radius <= args.radial_max_radius:
            samples.append(
                {
                    "triangle": next_sample_id,
                    "stage": "radial_refinement",
                    "radial_direction": ("inside" if signed_step < 0 else "outside"),
                    "radial_step_index": signed_step,
                    "radial_offset_m": signed_step * args.radial_step,
                    "radius_m": candidate_radius,
                    "reachable": False,
                    "skipped": True,
                    "skip_reason": "outside_radial_radius_bounds",
                }
            )
            next_sample_id += 1
            continue
        sample, viewpoint = _make_radial_sample(
            next_sample_id,
            signed_step,
            center,
            ray,
            base_radius_m,
            args.radial_step,
            int(angular_best["level"]),
            angular_path,
        )
        next_sample_id += 1
        samples.append(sample)
        initial_pairs.append((sample, viewpoint))

    initial_report = (
        env.filter_reachable_viewpoints(
            [viewpoint for _, viewpoint in initial_pairs],
            object_center,
            config,
            candidate_backend=candidate_backend,
        )
        if initial_pairs
        else None
    )
    accepted = (
        {item.viewpoint.index: item for item in initial_report.reachable}
        if initial_report is not None
        else {}
    )
    rejected = (
        {item.viewpoint.index: item for item in initial_report.rejected}
        if initial_report is not None
        else {}
    )
    for sample, viewpoint in initial_pairs:
        selected = accepted.get(viewpoint.index)
        if selected is None:
            rejection = rejected[viewpoint.index]
            sample["skipped"] = True
            sample["skip_reason"] = rejection.furthest_failure_stage
            sample["failure_counts"] = rejection.failure_counts
            continue
        sample["manipulability"] = selected.manipulability
        sample["quality_score"] = selected.quality_score
        sample["reachable"] = True

    for direction_label in ("inside", "outside"):
        try:
            _execute_joint_target(
                env,
                base_capture["joint_positions_rad"],
                motion_planner,
                args.motion_seconds,
                outward_paths,
                evaluation_tracker,
                f"radial_{direction_label}_sweep_origin",
            )
        except RuntimeError as exc:
            termination_reason = "could_not_return_to_angular_best"
            print(f"Radial sweep interrupted: {exc}")
            break
        directional = [
            pair
            for pair in initial_pairs
            if pair[0]["radial_direction"] == direction_label
        ]
        for sample, viewpoint in directional:
            selected = accepted.get(viewpoint.index)
            if selected is None:
                continue
            if not _capture_radial_sample(
                args,
                env,
                sample,
                selected,
                motion_planner,
                outward_paths,
                master_path,
                capture_pose_report,
                capture_pose_json_path,
                capture_pose_ply_path,
                evaluation_tracker,
            ):
                continue
            captured_count += 1
            score = _score_radial_sample(
                args,
                sample,
                rank_triangle_views,
                feature_extractor,
                spatial_clusterer,
                prepared_target,
            )
            print(f"Radial {direction_label} score={score:.3f}")
            write_json(master_path, samples)

    scored_samples = [sample for sample in samples if "score" in sample]
    angular_score = float(angular_best["best_score"])
    best_sample = (
        max(scored_samples, key=lambda item: float(item["score"]))
        if scored_samples
        else None
    )
    improving_samples = [
        sample
        for sample in scored_samples
        if float(sample["score"]) >= angular_score + args.radial_min_improvement
    ]
    direction_seed = None
    if improving_samples:
        highest_initial_score = max(
            float(sample["score"]) for sample in improving_samples
        )
        direction_seed = min(
            (
                sample
                for sample in improving_samples
                if highest_initial_score - float(sample["score"])
                <= args.radial_min_improvement
            ),
            key=lambda item: abs(int(item["radial_step_index"])),
        )

    if direction_seed is None:
        termination_reason = "no_initial_radial_improvement"
    else:
        selected_direction = str(direction_seed["radial_direction"])
        direction_sign = -1 if selected_direction == "inside" else 1
        best_step = int(direction_seed["radial_step_index"])
        # If an already-tested following step did not win, the local maximum
        # lies inside the initial probe interval and no extra motion is needed.
        if abs(best_step) < args.radial_probe_steps:
            termination_reason = "initial_probe_local_maximum"
        else:
            significant_best_score = float(direction_seed["score"])
            plateau_count = 0
            next_step = best_step + direction_sign
            termination_reason = "maximum_radial_steps_reached"
            try:
                _execute_joint_target(
                    env,
                    direction_seed["joint_positions_rad"],
                    motion_planner,
                    args.motion_seconds,
                    outward_paths,
                    evaluation_tracker,
                    "return_to_best_radial_probe",
                )
            except RuntimeError as exc:
                termination_reason = "could_not_return_to_best_radial_probe"
                print(f"Radial continuation skipped: {exc}")

            while (
                termination_reason == "maximum_radial_steps_reached"
                and abs(next_step) <= args.radial_max_steps
            ):
                candidate_radius = base_radius_m + next_step * args.radial_step
                if (
                    not args.radial_min_radius
                    <= candidate_radius
                    <= args.radial_max_radius
                ):
                    termination_reason = "radial_radius_bound_reached"
                    break
                try:
                    sample, viewpoint = _make_radial_sample(
                        next_sample_id,
                        next_step,
                        center,
                        ray,
                        base_radius_m,
                        args.radial_step,
                        int(angular_best["level"]),
                        angular_path,
                    )
                except ValueError:
                    termination_reason = "minimum_radius_reached"
                    break
                next_sample_id += 1
                samples.append(sample)
                report = env.filter_reachable_viewpoints(
                    [viewpoint],
                    object_center,
                    config,
                    candidate_backend=candidate_backend,
                )
                if not report.reachable:
                    rejection = report.rejected[0]
                    sample["skipped"] = True
                    sample["skip_reason"] = rejection.furthest_failure_stage
                    sample["failure_counts"] = rejection.failure_counts
                    termination_reason = "radial_viewpoint_unreachable"
                    break
                selected = report.reachable[0]
                sample["manipulability"] = selected.manipulability
                sample["quality_score"] = selected.quality_score
                sample["reachable"] = True
                if not _capture_radial_sample(
                    args,
                    env,
                    sample,
                    selected,
                    motion_planner,
                    outward_paths,
                    master_path,
                    capture_pose_report,
                    capture_pose_json_path,
                    capture_pose_ply_path,
                    evaluation_tracker,
                ):
                    termination_reason = "radial_motion_planning_failed"
                    break
                captured_count += 1
                score = _score_radial_sample(
                    args,
                    sample,
                    rank_triangle_views,
                    feature_extractor,
                    spatial_clusterer,
                    prepared_target,
                )
                print(
                    f"Radial {selected_direction} continuation "
                    f"step {abs(next_step)} score={score:.3f}"
                )
                if score >= significant_best_score + args.radial_min_improvement:
                    significant_best_score = score
                    plateau_count = 0
                else:
                    plateau_count += 1
                if float(sample["score"]) > float(best_sample["score"]):
                    best_sample = sample
                write_json(master_path, samples)
                if plateau_count >= args.radial_patience:
                    termination_reason = "radial_score_plateau"
                    break
                next_step += direction_sign

    radial_improved = (
        best_sample is not None and float(best_sample["score"]) > angular_score
    )
    if radial_improved:
        best_observation = {
            "stage": "radial_refinement",
            "level": int(best_sample["refinement_level"]),
            "candidate_count": captured_count,
            "best_score": float(best_sample["score"]),
            "best_image": best_sample["best_image"],
            "best_comparison": best_sample.get("best_comparison"),
            "best_viewpoint": best_sample["best_viewpoint"],
            "best_radial_direction": best_sample["radial_direction"],
            "best_radial_step_index": best_sample["radial_step_index"],
            "best_radial_offset_m": best_sample["radial_offset_m"],
            "best_radius_m": best_sample["radius_m"],
        }
        final_joint_target = best_sample["joint_positions_rad"]
    else:
        best_observation = angular_best
        final_joint_target = base_capture["joint_positions_rad"]

    # Leave the camera at the best observation before the normal final-home
    # procedure runs, rather than at a worse plateau probe.
    try:
        _execute_joint_target(
            env,
            final_joint_target,
            motion_planner,
            args.motion_seconds,
            outward_paths,
            evaluation_tracker,
            "revisit_final_best_observation",
        )
    except RuntimeError as exc:
        print(f"Could not revisit the final best observation: {exc}")

    result = {
        "enabled": True,
        "base_angular_score": angular_score,
        "base_radius_m": base_radius_m,
        "step_size_m": args.radial_step,
        "minimum_radius_m": args.radial_min_radius,
        "maximum_radius_m": args.radial_max_radius,
        "initial_probe_steps_per_direction": args.radial_probe_steps,
        "selected_direction": selected_direction,
        "termination_reason": termination_reason,
        "candidate_count": captured_count,
        "improved_over_angular_best": radial_improved,
        "samples": samples,
        "best_observation": best_observation,
        "summary_file": str(summary_path.resolve()),
        "viewpoints_file": str(master_path.resolve()),
    }
    write_json(master_path, samples)
    write_json(summary_path, result)
    print(
        f"Radial termination: {termination_reason}; overall best "
        f"score={float(best_observation['best_score']):.3f}."
    )
    return result


def _bootstrap_rgb_localization(*args):
    import sys
    if __package__:
        from .rgb_localization_capture import bootstrap_rgb_localization
    else:
        from rgb_localization_capture import bootstrap_rgb_localization
    return bootstrap_rgb_localization(sys.modules[__name__], *args)


def run(args: argparse.Namespace) -> None:
    run_started = time.perf_counter()
    if args.object_localization == "rgb" and args.target_image is None:
        raise ValueError("RGB localization requires --target-image")
    _apply_search_mode(args)
    nbv = None
    if args.search_strategy in ("nbv", "geometric"):
        search_label = (
            "Geometric NBV" if args.search_strategy == "geometric" else "Bayesian NBV"
        )
        print(
            f"{search_label}: up to {args.nbv_max_views} scored views; "
            "one capture per decision, no triangle subdivision."
        )
    if args.until_convergence:
        print(
            "Convergence mode: 3 views per level; stop the entire search after "
            f"{args.convergence_patience} level(s) with best-score gain below "
            f"{args.convergence_min_improvement:.4f}. "
            "Similarity/depth/look-ahead limits are ignored; radial refinement is disabled."
        )
    backend_selection = resolve_ik_backend(
        args.ik_backend,
        args.curobo_robot_config,
    )
    print(
        f"IK backend: {backend_selection.effective_backend}; motion planner: "
        f"{'CuRobo' if backend_selection.uses_curobo else 'validated joint-linear'}"
    )
    workpiece = resolve_workpiece_ply(args.object, args.workpiece_ply)
    env = UR5EyeInHandPyBulletEnv(
        robot_urdf=args.robot_urdf.resolve(),
        workpiece_ply=workpiece.resolve(),
        workpiece_unit_scale=args.workpiece_unit_scale,
        workpiece_rotation_z_deg=args.workpiece_rotation_z,
        gui=not args.headless,
    )
    evaluation_tracker = _new_evaluation_tracker(env)
    evaluation_started = time.perf_counter()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outward_paths: list[np.ndarray] = []
    motion_planner: CuRoboMotionPlanner | None = None
    refinement_summary_path = args.output_dir / f"{args.name}_refinement_summary.json"
    level_results: list[dict[str, object]] = []
    radial_result: dict[str, object] | None = None
    evaluation_report: dict[str, object] | None = None
    capture_pose_json_path = (
        args.output_dir / f"{args.name}_end_effector_capture_poses.json"
    )
    capture_pose_ply_path = (
        args.output_dir / f"{args.name}_end_effector_capture_poses.ply"
    )
    capture_pose_report: dict[str, object] = {
        "schema_version": 1,
        "status": "in_progress",
        "coordinate_frame": "pybullet_world",
        "position_units": "metres",
        "orientation_convention": "quaternion_xyzw",
        "end_effector_link": END_EFFECTOR_LINK_NAME,
        "joint_order": list(ARM_JOINT_ORDER),
        "robot_urdf": str(args.robot_urdf.expanduser().resolve()),
        "workpiece_ply": str(workpiece.expanduser().resolve()),
        "workpiece_unit_scale": args.workpiece_unit_scale,
        "workpiece_rotation_z_deg": args.workpiece_rotation_z,
        "mesh_file": str(capture_pose_ply_path.resolve()),
        "captures": [],
    }
    write_json(capture_pose_json_path, capture_pose_report)
    write_capture_poses_ply(capture_pose_ply_path, capture_pose_report)
    run_completed = False

    try:
        candidate_backend = build_selected_candidate_backend(
            env,
            backend_selection,
            curobo_seeds=args.curobo_seeds,
        )
        motion_planner = (
            build_selected_motion_planner(env, backend_selection, CLEARANCE_M)
            if backend_selection.uses_curobo
            else None
        )
        # The filter checks IK, collisions and line of sight, then prefers high
        # manipulability and joint-limit clearance when several IK poses exist.
        config = ReachabilityConfig(
            roll_angles_deg=(0.0, 90.0, -90.0, 180.0),
            seed_count=(
                args.curobo_seeds if backend_selection.uses_curobo else args.ik_seeds
            ),
            refinement_steps=(0 if backend_selection.uses_curobo else 2),
            clearance_m=CLEARANCE_M,
            require_line_of_sight=args.object_localization != "rgb",
        )
        feature_extractor = None
        prepared_target = None
        localization_report = None
        if args.object_localization == "rgb":
            localization_report, feature_extractor, prepared_target = _bootstrap_rgb_localization(
                args, env, config, candidate_backend, motion_planner, outward_paths,
                capture_pose_report, capture_pose_json_path, capture_pose_ply_path,
                evaluation_tracker,
            )
            object_center = np.asarray(localization_report["center_world_m"], dtype=float)
        else:
            object_center = env.get_workpiece_vertex_centroid()

        if args.search_strategy in ("nbv", "geometric"):
            search_class = (
                GeometricViewSearch
                if args.search_strategy == "geometric"
                else BayesianViewSearch
            )
            extra_settings = (
                {"target_image": args.target_image}
                if args.search_strategy == "geometric"
                else {}
            )
            nbv = search_class(
                object_center,
                args.radius,
                min_radius=args.radial_min_radius,
                max_radius=args.radial_max_radius,
                max_views=args.nbv_max_views,
                patience=args.nbv_patience,
                min_improvement=args.min_score_improvement,
                azimuth_offset=args.azimuth_offset or 0.0,
                **extra_settings,
            )
        else:
            # Rotating all wedges preserves equal area. In automatic mode, keep the
            # first rotation for which every centroid is in the robot's capability.
            offsets = (
                [args.azimuth_offset]
                if args.azimuth_offset is not None
                else [index * 360.0 / (12 * args.views) for index in range(12)]
            )
            best = None
            for offset in offsets:
                triangles, candidates = hemisphere_triangles(
                    object_center, args.radius, args.views, offset
                )
                report = env.filter_reachable_viewpoints(
                    candidates,
                    object_center,
                    config,
                    candidate_backend=candidate_backend,
                )
                reachable = {item.viewpoint.index: item for item in report.reachable}
                if best is None or len(reachable) > len(best[3]):
                    best = (triangles, report, offset, reachable)
                if len(reachable) == args.views:
                    break

            if best is None:
                raise RuntimeError("No hemisphere partition was evaluated.")
            triangles, reachability_report, chosen_offset, _ = best
            print(f"Using hemisphere azimuth offset {chosen_offset:.1f} degrees.")
            reachable = _annotate_reachability(
                triangles,
                reachability_report,
                azimuth_offset_deg=chosen_offset,
            )
            reachable = _recover_unreachable_triangle_viewpoints(
                env,
                triangles,
                reachable,
                object_center,
                args.radius,
                config,
                candidate_backend,
            )

        spatial_clusterer = None
        level = 0
        best_score_seen = -math.inf
        non_improving_levels = 0
        lookahead_levels_used = 0
        termination_reason = "capture_only"
        while True:
            if nbv is not None:
                reason = nbv.stop_reason()
                if (
                    args.nbv_time_budget
                    and time.perf_counter() - run_started >= args.nbv_time_budget
                ):
                    reason = "nbv_time_budget_reached"
                if reason:
                    termination_reason = reason
                    break
                camera_position, _ = env.get_wrist_camera_pose()
                proposal = nbv.propose(camera_position)
                if proposal is None:
                    termination_reason = (
                        nbv.exhaustion_reason
                        if isinstance(nbv, GeometricViewSearch)
                        else "nbv_candidates_exhausted"
                    )
                    break
                level = len(nbv.attempts) - 1
                triangles, candidates = _nbv_viewpoint(
                    proposal, object_center, level, args.search_strategy
                )
                proposal_config = (
                    replace(config, roll_angles_deg=(0.0,))
                    if "camera_orientation_xyzw" in proposal
                    else config
                )
                reachability_report = env.filter_reachable_viewpoints(
                    candidates,
                    object_center,
                    proposal_config,
                    candidate_backend=candidate_backend,
                )
                reachable = _annotate_reachability(triangles, reachability_report)
                if (
                    args.nbv_time_budget
                    and time.perf_counter() - run_started >= args.nbv_time_budget
                ):
                    nbv.reject("time_budget_before_capture")
                    termination_reason = "nbv_time_budget_reached"
                    break
            coordinates_path = (
                args.output_dir / f"{args.name}_viewpoints.json"
                if level == 0
                else args.output_dir / f"{args.name}_level_{level:02d}_viewpoints.json"
            )
            image_prefix = args.name if level == 0 else f"{args.name}_level_{level:02d}"
            label = (
                f"NBV attempt {level + 1}"
                if nbv is not None
                else f"Active-perception level {level}"
            )
            print(f"\n=== {label} ===")
            capture_started = time.perf_counter()
            captured_count = _capture_round(
                env,
                triangles,
                reachable,
                coordinates_path,
                image_prefix,
                args.motion_seconds,
                motion_planner,
                outward_paths,
                capture_pose_report,
                capture_pose_json_path,
                capture_pose_ply_path,
                evaluation_tracker,
                return_home_after_round=args.return_home_each_level,
            )
            capture_seconds = time.perf_counter() - capture_started

            if args.target_image is None:
                break
            if captured_count == 0:
                if nbv is not None:
                    nbv.reject(triangles[0].get("skip_reason", "unreachable_viewpoint"))
                    continue
                termination_reason = "no_reachable_viewpoints"
                print(
                    "No reachable triangle remains at this refinement level; "
                    "finishing with the best earlier observation."
                )
                break

            # Keep this import outside the extractor branch: RGB localization
            # may already have prepared the extractor before the first level.
            if __package__:
                from .target_guided_similarity import rank_triangle_views
            else:
                from target_guided_similarity import rank_triangle_views

            # Import and construct the perception models only after the first
            # capture. Reuse them for every refinement level.
            if feature_extractor is None:
                if __package__:
                    from .dinov3_feature_matching import DINOv3FeatureExtractor
                    from .target_guided_similarity import prepare_target_image
                else:
                    from dinov3_feature_matching import DINOv3FeatureExtractor
                    from target_guided_similarity import prepare_target_image

                feature_extractor = DINOv3FeatureExtractor(
                    max_image_size=args.feature_max_image_size,
                    upscale_small_images=True,
                )
                if args.debug:
                    if __package__:
                        from .dinov3_spatial_clustering import DINOv3SpatialClusterer
                    else:
                        from dinov3_spatial_clustering import DINOv3SpatialClusterer

                    spatial_clusterer = DINOv3SpatialClusterer()
                print("Preparing reusable target-image features ...", flush=True)
                prepared_target = prepare_target_image(
                    args.target_image,
                    extractor=feature_extractor,
                    debug=args.debug,
                    clusterer=spatial_clusterer,
                )

            ranking_started = time.perf_counter()
            ranking_report = rank_triangle_views(
                args.target_image,
                coordinates_path,
                ambiguity_margin=args.ambiguity_margin,
                dino_weight=args.dino_weight,
                sift_weight=args.sift_weight,
                debug=args.debug,
                extractor=feature_extractor,
                clusterer=spatial_clusterer,
                prepared_target=prepared_target,
            )
            ranking_seconds = time.perf_counter() - ranking_started
            best_triangle_id = int(ranking_report["best_triangle"])
            best_triangle = next(
                triangle
                for triangle in triangles
                if int(triangle["triangle"]) == best_triangle_id
            )
            best_score = float(ranking_report["ranking"][0]["score"])
            if nbv is not None:
                nbv.observe(
                    best_triangle.get(
                        "actual_camera_xyz_m", proposal["position_xyz_m"]
                    ),
                    best_score,
                )
                if (
                    isinstance(nbv, GeometricViewSearch)
                    and best_score < args.similarity_threshold
                ):
                    geometry_started = time.perf_counter()
                    nbv.observe_image(
                        Path(ranking_report["best_image"]),
                        quaternion_xyzw_to_matrix(
                            best_triangle["actual_camera_orientation_xyzw"]
                        ),
                    )
                    nbv.attempts[-1]["geometry_seconds"] = (
                        time.perf_counter() - geometry_started
                    )
                    evidence = nbv.attempts[-1]["geometry"]
                    print(
                        f"Relative-pose evidence: {evidence.get('inliers', 0)} inliers; "
                        f"{evidence['reason']}."
                    )
            previous_best_score = best_score_seen
            score_improvement = (
                None
                if not math.isfinite(previous_best_score)
                else best_score - previous_best_score
            )
            meaningful_improvement = (
                not math.isfinite(previous_best_score)
                or best_score >= previous_best_score + args.min_score_improvement
            )
            accepted_as_global_best = (
                not math.isfinite(previous_best_score)
                or best_score > previous_best_score
            )
            best_score_seen = max(best_score_seen, best_score)
            if meaningful_improvement:
                non_improving_levels = 0
                lookahead_levels_used = 0
            else:
                non_improving_levels += 1
            score_gap_to_incumbent = max(0.0, best_score_seen - best_score)
            visualizations = ranking_report.get("visualizations")
            best_comparison = (
                visualizations.get("best_comparison")
                if isinstance(visualizations, dict)
                else None
            )
            result = {
                "stage": "next_best_view" if nbv is not None else "angular_refinement",
                "level": level,
                "candidate_count": captured_count,
                "triangle_count": len(triangles),
                "skipped_triangle_paths": [
                    triangle["triangle_path"]
                    for triangle in triangles
                    if not bool(triangle.get("reachable"))
                ],
                "viewpoints_file": str(coordinates_path.resolve()),
                "ranking_file": ranking_report["report_file"],
                "best_triangle": best_triangle_id,
                "best_triangle_path": best_triangle["triangle_path"],
                "best_image": ranking_report["best_image"],
                "best_comparison": best_comparison,
                "best_score": best_score,
                "score_improvement_over_previous_global_best": score_improvement,
                "accepted_as_global_best": accepted_as_global_best,
                "meaningful_improvement": meaningful_improvement,
                "global_best_score_after_level": best_score_seen,
                "score_gap_to_incumbent": score_gap_to_incumbent,
                "non_improving_levels": non_improving_levels,
                "best_viewpoint": ranking_report["best_viewpoint"],
                "timing_seconds": {
                    "capture_and_motion": capture_seconds,
                    "ranking": ranking_seconds,
                    "total": capture_seconds + ranking_seconds,
                },
            }
            level_results.append(result)
            path_label = ".".join(
                str(value) for value in best_triangle["triangle_path"]
            )
            winner_label = (
                f"NBV observation {len(nbv.scores)} (proposal {path_label})"
                if nbv is not None
                else f"Level {level} winner: triangle path {path_label}"
            )
            print(
                f"{winner_label}, "
                f"candidate={best_score:.3f}, global={best_score_seen:.3f}, "
                f"accepted={'yes' if accepted_as_global_best else 'no'} "
                f"(capture={capture_seconds:.1f}s, ranking={ranking_seconds:.1f}s)"
            )

            should_refine = True
            if nbv is not None:
                if best_score_seen >= args.similarity_threshold:
                    termination_reason = "similarity_threshold_reached"
                    should_refine = False
            elif args.until_convergence:
                if non_improving_levels >= args.convergence_patience:
                    termination_reason = "similarity_converged"
                    should_refine = False
                    print(
                        "Stopping entire search: best similarity has plateaued "
                        f"for {non_improving_levels} level(s); "
                        f"keeping score={best_score_seen:.3f}."
                    )
            elif best_score_seen >= args.similarity_threshold:
                termination_reason = "similarity_threshold_reached"
                should_refine = False
            elif level >= args.max_refinement_depth:
                termination_reason = "maximum_refinement_depth_reached"
                should_refine = False
            elif (
                args.refinement_patience > 0
                and non_improving_levels >= args.refinement_patience
            ):
                termination_reason = "refinement_stalled"
                should_refine = False
                print(
                    "Stopping refinement: no global-best improvement of at least "
                    f"{args.min_score_improvement:.4f} for "
                    f"{non_improving_levels} consecutive level(s)."
                )
            elif not meaningful_improvement:
                if score_gap_to_incumbent > args.angular_lookahead_margin:
                    termination_reason = "angular_candidate_not_promising"
                    should_refine = False
                    print(
                        "Stopping angular refinement: the current candidate is "
                        f"{score_gap_to_incumbent:.3f} below the incumbent, beyond "
                        f"the {args.angular_lookahead_margin:.3f} look-ahead margin."
                    )
                elif lookahead_levels_used >= args.angular_lookahead_levels:
                    termination_reason = "angular_lookahead_exhausted"
                    should_refine = False
                    print(
                        "Stopping angular refinement: the ambiguous-branch "
                        "look-ahead budget is exhausted."
                    )
                else:
                    lookahead_levels_used += 1
                    print(
                        "Candidate did not improve the incumbent, but remains "
                        f"within {args.angular_lookahead_margin:.3f}; allowing "
                        f"look-ahead {lookahead_levels_used}/"
                        f"{args.angular_lookahead_levels}."
                    )

            if should_refine:
                if nbv is not None:
                    continue
                triangles, candidates = subdivide_spherical_triangle(
                    best_triangle,
                    object_center,
                    args.radius,
                )
                level += 1
                reachability_report = env.filter_reachable_viewpoints(
                    candidates,
                    object_center,
                    config,
                    candidate_backend=candidate_backend,
                )
                reachable = _annotate_reachability(triangles, reachability_report)
                reachable = _recover_unreachable_triangle_viewpoints(
                    env,
                    triangles,
                    reachable,
                    object_center,
                    args.radius,
                    config,
                    candidate_backend,
                )
                continue
            break
        if (
            _target_guided_perception_enabled(args)
            and termination_reason != "capture_only"
            and level_results
            and not args.skip_radial_refinement
        ):
            radial_result = _run_radial_refinement(
                args,
                env,
                object_center,
                config,
                candidate_backend,
                motion_planner,
                outward_paths,
                capture_pose_report,
                capture_pose_json_path,
                capture_pose_ply_path,
                level_results,
                rank_triangle_views,
                feature_extractor,
                spatial_clusterer,
                prepared_target,
                evaluation_tracker,
            )
        final_observation_candidates = list(level_results)
        if radial_result is not None:
            radial_best = radial_result.get("best_observation")
            if isinstance(radial_best, dict):
                final_observation_candidates.append(radial_best)
        final_observation = (
            max(
                final_observation_candidates,
                key=lambda item: float(item["best_score"]),
            )
            if final_observation_candidates
            else None
        )
        evaluation_report = _write_evaluation_report(
            args,
            capture_pose_report,
            evaluation_tracker,
            time.perf_counter() - evaluation_started,
            final_observation,
            termination_reason,
            radial_result,
        )
        if (
            _target_guided_perception_enabled(args)
            and termination_reason != "capture_only"
        ):
            _finalize_refinement_run(
                args,
                level_results,
                radial_result,
                termination_reason,
                refinement_summary_path,
                capture_pose_json_path,
                capture_pose_ply_path,
                evaluation_report,
                (
                    {
                        **nbv.summary(),
                        "time_budget_seconds": args.nbv_time_budget,
                        "elapsed_before_final_report_seconds": time.perf_counter()
                        - run_started,
                    }
                    if nbv is not None
                    else None
                ),
                localization_report,
            )
        run_completed = True
    finally:
        try:
            move_home(
                env,
                args.motion_seconds,
                outward_paths,
                motion_planner,
            )
        except Exception as exc:
            print(f"Safe return failed ({exc}); resetting the simulation to home.")
            env.reset_to_home_pose()
        capture_pose_report["status"] = "complete" if run_completed else "incomplete"
        write_json(capture_pose_json_path, capture_pose_report)
        write_capture_poses_ply(capture_pose_ply_path, capture_pose_report)
        if run_completed:
            print(f"Saved end-effector capture poses: {capture_pose_json_path}")
            print(f"Saved MeshLab capture poses: {capture_pose_ply_path}")
        env.disconnect()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Target-guided triangle refinement or Bayesian next-best-view camera search."
        )
    )
    parser.add_argument("--object-localization", choices=("simulator", "rgb"), default="simulator",
                        help="Initialize from simulator center or calibrated RGB survey triangulation.")
    parser.add_argument("--localization-viewpoints", type=Path,
                        help="JSON workspace survey camera poses in pybullet_world coordinates.")
    parser.add_argument("--localization-min-similarity", type=float, default=.55)
    parser.add_argument("--localization-ambiguity-margin", type=float, default=.03)
    parser.add_argument("--views", type=int, choices=(3, 4, 5), default=3)
    parser.add_argument(
        "--search-strategy",
        choices=("triangles", "nbv", "geometric"),
        default="triangles",
        help="Triangle subdivision (default), Bayesian NBV, or RGB relative-pose guided search.",
    )
    parser.add_argument(
        "--nbv-max-views",
        "--search-max-views",
        type=int,
        default=8,
        help="Bayesian/geometric scored-image budget including coverage views (default: 8).",
    )
    parser.add_argument(
        "--nbv-patience",
        "--search-patience",
        type=int,
        default=3,
        help="NBV stale-view patience, applied after at least 6 scored views (default: 3).",
    )
    parser.add_argument(
        "--nbv-time-budget",
        "--search-time-budget",
        type=float,
        default=0.0,
        help="Soft NBV wall-time budget in seconds; checked between operations (0: unlimited).",
    )
    parser.add_argument(
        "--radius", type=float, default=0.40, help="Camera radius in metres."
    )
    parser.add_argument(
        "--azimuth-offset",
        type=float,
        default=None,
        help="Force a world-Z partition rotation; by default a reachable one is searched.",
    )
    parser.add_argument("--name", default="object", help="Output filename prefix.")
    parser.add_argument(
        "--method-name",
        default="Adaptive Spherical + Radial (DINOv3+SIFT)",
        help="Method label written to evaluation reports and CSV rows.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR / "triangle_views"
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--motion-seconds", type=float, default=1.5)
    parser.add_argument(
        "--return-home-each-level",
        action="store_true",
        help=(
            "Return to the home pose after every refinement level. By default "
            "the robot stays at its current pose and returns home only once at "
            "the end, which is substantially faster in simulation."
        ),
    )
    parser.add_argument("--ik-seeds", type=int, default=8)
    parser.add_argument(
        "--ik-backend",
        choices=("auto", "pybullet", "curobo"),
        default="auto",
        help="Use CuRobo when available (default), or force one IK backend.",
    )
    parser.add_argument("--curobo-robot-config", type=Path, default=None)
    parser.add_argument(
        "--curobo-seeds", type=int, default=DEFAULT_CUROBO_IK_SEED_COUNT
    )
    parser.add_argument(
        "--target-image",
        type=Path,
        default=None,
        help="Target image for angular ranking and radial fine alignment.",
    )
    target_pose_source = parser.add_mutually_exclusive_group()
    target_pose_source.add_argument(
        "--target-pose-json",
        type=Path,
        default=None,
        help=(
            "JSON containing a target image_file/image_filename record and camera "
            "pose. Without this option, JSON files beside the target are searched."
        ),
    )
    target_pose_source.add_argument(
        "--target-camera-position",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Ground-truth target camera center in PyBullet world metres.",
    )
    parser.add_argument(
        "--until-convergence",
        action="store_true",
        help=(
            "Use 3 views and stop the entire search when best similarity plateaus. "
            "Overrides --views, similarity/depth/patience/improvement/look-ahead "
            "search controls and disables radial refinement. Requires --target-image."
        ),
    )
    parser.add_argument(
        "--convergence-min-improvement",
        type=float,
        default=0.002,
        help="Meaningful best-score gain in convergence mode (default: 0.002; must be > 0).",
    )
    parser.add_argument(
        "--convergence-patience",
        type=int,
        default=1,
        help="Consecutive levels below the convergence gain before stopping (default: 1).",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.90,
        help="Stop refinement when the best combined score reaches this value.",
    )
    parser.add_argument(
        "--max-refinement-depth",
        type=int,
        default=3,
        help="Maximum subdivisions after the initial hemisphere (default: 3).",
    )
    parser.add_argument(
        "--refinement-patience",
        type=int,
        default=2,
        help=(
            "Stop after this many consecutive levels fail to improve the global "
            "best score (default: 2; use 0 to disable)."
        ),
    )
    parser.add_argument(
        "--min-score-improvement",
        type=float,
        default=0.002,
        help="Minimum score gain that resets refinement patience (default: 0.002).",
    )
    parser.add_argument(
        "--angular-lookahead-margin",
        type=float,
        default=0.03,
        help=(
            "Explore one more level only when a worse child remains this close "
            "to the global incumbent (default: 0.03)."
        ),
    )
    parser.add_argument(
        "--angular-lookahead-levels",
        type=int,
        default=1,
        help="Maximum ambiguous worse-child look-ahead levels (default: 1).",
    )
    parser.add_argument(
        "--skip-radial-refinement",
        action="store_true",
        help="Skip the final closer/farther fine-alignment search.",
    )
    parser.add_argument(
        "--radial-step",
        type=float,
        default=0.02,
        help="Camera-radius increment in metres for fine alignment (default: 0.02).",
    )
    parser.add_argument(
        "--radial-probe-steps",
        type=int,
        default=2,
        help="Initial steps evaluated both closer and farther (default: 2).",
    )
    parser.add_argument(
        "--radial-max-steps",
        type=int,
        default=10,
        help="Maximum absolute radial step in the selected direction (default: 10).",
    )
    parser.add_argument(
        "--radial-patience",
        type=int,
        default=2,
        help="Stop after this many radial steps without a meaningful gain (default: 2).",
    )
    parser.add_argument(
        "--radial-min-improvement",
        type=float,
        default=0.002,
        help="Minimum radial score gain treated as an improvement (default: 0.002).",
    )
    parser.add_argument(
        "--radial-min-radius",
        type=float,
        default=0.20,
        help="Minimum permitted object-to-camera radius in metres (default: 0.20).",
    )
    parser.add_argument(
        "--radial-max-radius",
        type=float,
        default=0.60,
        help="Maximum permitted object-to-camera radius in metres (default: 0.60).",
    )
    parser.add_argument("--ambiguity-margin", type=float, default=0.03)
    parser.add_argument("--dino-weight", type=float, default=0.75)
    parser.add_argument("--sift-weight", type=float, default=0.25)
    parser.add_argument(
        "--feature-max-image-size",
        type=int,
        default=448,
        help=(
            "Maximum DINOv3 image side in pixels (default: 448; use 640 to "
            "reproduce the previous, slower setting)."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Enable spatial clustering and diagnostic visualizations. Target-"
            "guided ranking/refinement only requires --target-image."
        ),
    )
    parser.add_argument("--robot-urdf", type=Path, default=DEFAULT_ROBOT_URDF)
    objects = parser.add_mutually_exclusive_group()
    objects.add_argument(
        "--object",
        default=None,
        help=f"Bundled object name (default: {DEFAULT_WORKPIECE_NAME}).",
    )
    objects.add_argument("--workpiece-ply", type=Path, default=DEFAULT_WORKPIECE_PLY)
    parser.add_argument("--workpiece-unit-scale", type=float, default=0.001)
    parser.add_argument("--workpiece-rotation-z", type=float, default=0.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _apply_search_mode(args)
    if not args.method_name.strip():
        raise ValueError("--method-name must not be empty.")
    if args.ik_seeds <= 0:
        raise ValueError("--ik-seeds must be positive.")
    if args.curobo_seeds <= 0:
        raise ValueError("--curobo-seeds must be positive.")
    if not math.isfinite(args.motion_seconds) or args.motion_seconds < 0.0:
        raise ValueError("--motion-seconds must be finite and non-negative.")
    if (
        not math.isfinite(args.ambiguity_margin)
        or not 0.0 <= args.ambiguity_margin <= 1.0
    ):
        raise ValueError("--ambiguity-margin must be in [0, 1].")
    if (
        not all(math.isfinite(value) for value in (args.dino_weight, args.sift_weight))
        or args.dino_weight < 0.0
        or args.sift_weight < 0.0
        or args.dino_weight + args.sift_weight <= 0.0
    ):
        raise ValueError("Similarity weights must be non-negative and not both zero.")
    if not args.until_convergence:
        if (
            not math.isfinite(args.similarity_threshold)
            or not 0.0 <= args.similarity_threshold <= 1.0
        ):
            raise ValueError("--similarity-threshold must be in [0, 1].")
        if (
            not isinstance(args.max_refinement_depth, int)
            or isinstance(args.max_refinement_depth, bool)
            or args.max_refinement_depth < 0
        ):
            raise ValueError("--max-refinement-depth must be a non-negative integer.")
        if (
            not isinstance(args.refinement_patience, int)
            or isinstance(args.refinement_patience, bool)
            or args.refinement_patience < 0
        ):
            raise ValueError("--refinement-patience must be a non-negative integer.")
        if (
            not math.isfinite(args.min_score_improvement)
            or args.min_score_improvement < 0.0
            or args.min_score_improvement > 1.0
        ):
            raise ValueError("--min-score-improvement must be in [0, 1].")
        if (
            not math.isfinite(args.angular_lookahead_margin)
            or args.angular_lookahead_margin < 0.0
            or args.angular_lookahead_margin > 1.0
        ):
            raise ValueError("--angular-lookahead-margin must be in [0, 1].")
        if args.angular_lookahead_levels < 0:
            raise ValueError("--angular-lookahead-levels must be non-negative.")
        if not math.isfinite(args.radial_step) or args.radial_step <= 0.0:
            raise ValueError("--radial-step must be finite and positive.")
        if args.radial_probe_steps <= 0:
            raise ValueError("--radial-probe-steps must be positive.")
        if args.radial_max_steps < args.radial_probe_steps:
            raise ValueError(
                "--radial-max-steps must be at least --radial-probe-steps."
            )
        if args.radial_patience <= 0:
            raise ValueError("--radial-patience must be positive.")
        if (
            not math.isfinite(args.radial_min_improvement)
            or args.radial_min_improvement < 0.0
            or args.radial_min_improvement > 1.0
        ):
            raise ValueError("--radial-min-improvement must be in [0, 1].")
        if (
            not math.isfinite(args.radial_min_radius)
            or not math.isfinite(args.radial_max_radius)
            or args.radial_min_radius <= 0.0
            or args.radial_max_radius <= args.radial_min_radius
        ):
            raise ValueError(
                "Radial radius bounds must be finite, positive, and ordered."
            )
    if args.feature_max_image_size <= 0:
        raise ValueError("--feature-max-image-size must be positive.")
    if (
        args.target_pose_json is not None
        and not args.target_pose_json.expanduser().is_file()
    ):
        raise FileNotFoundError(f"Target pose JSON not found: {args.target_pose_json}")
    if args.target_camera_position is not None and not all(
        math.isfinite(value) for value in args.target_camera_position
    ):
        raise ValueError("--target-camera-position must contain finite values.")
    if args.target_image is not None and not args.target_image.expanduser().is_file():
        raise FileNotFoundError(f"Target image not found: {args.target_image}")
    run(args)


if __name__ == "__main__":
    main()
