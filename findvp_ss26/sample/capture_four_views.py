#!/usr/bin/env python3
"""Capture four hemisphere views with timestamps and reproducible robot poses."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

if __package__:
    from .curobo_motion import CuRoboMotionPlanner
    from .pybullet_tabletop_ur5e_eye_in_hand import (
        DEFAULT_CUROBO_IK_SEED_COUNT,
        ReachableViewpoint,
        UR5EyeInHandPyBulletEnv,
        build_selected_candidate_backend,
        build_selected_motion_planner,
        resolve_ik_backend,
    )
    from .viewpoint_reachability import (
        ReachabilityConfig,
        ViewpointCandidate,
        generate_four_equal_area_viewpoints,
        look_at_rotation_matrix,
        matrix_to_quaternion_xyzw,
    )
    from .workcell_config import (
        ARM_JOINT_ORDER,
        CAMERA_HEIGHT,
        CAMERA_WIDTH,
        DEFAULT_ROBOT_URDF,
        DEFAULT_WORKPIECE_NAME,
        DEFAULT_WORKPIECE_PLY,
        HOME_JOINT_POSITIONS_RAD,
        OUTPUT_DIR,
        compute_camera_vertical_fov_deg,
        resolve_workpiece_ply,
    )
else:
    from curobo_motion import CuRoboMotionPlanner
    from pybullet_tabletop_ur5e_eye_in_hand import (
        DEFAULT_CUROBO_IK_SEED_COUNT,
        ReachableViewpoint,
        UR5EyeInHandPyBulletEnv,
        build_selected_candidate_backend,
        build_selected_motion_planner,
        resolve_ik_backend,
    )
    from viewpoint_reachability import (
        ReachabilityConfig,
        ViewpointCandidate,
        generate_four_equal_area_viewpoints,
        look_at_rotation_matrix,
        matrix_to_quaternion_xyzw,
    )
    from workcell_config import (
        ARM_JOINT_ORDER,
        CAMERA_HEIGHT,
        CAMERA_WIDTH,
        DEFAULT_ROBOT_URDF,
        DEFAULT_WORKPIECE_NAME,
        DEFAULT_WORKPIECE_PLY,
        HOME_JOINT_POSITIONS_RAD,
        OUTPUT_DIR,
        compute_camera_vertical_fov_deg,
        resolve_workpiece_ply,
    )


REGION_ORDER = ("top", "side_1", "side_2", "side_3")
NOMINAL_RADIUS_M = 0.40
MAX_RADIUS_M = 0.50
FRAMING_MARGIN_FRACTION = 0.10
CLEARANCE_M = 0.005
MAX_JOINT_STEP_RAD = 0.02
PRIMARY_CAMERA_ROLLS_DEG = (0.0,)
FALLBACK_CAMERA_ROLLS_DEG = (90.0, -90.0, 180.0)
FALLBACK_VIEWPOINT_INDEX_BASE = 1000


@dataclass(frozen=True)
class RegionViewpointAlternative:
    """One deterministic fallback position inside a capture region."""

    region_index: int
    rank: int
    viewpoint: ViewpointCandidate


def _spherical_viewpoint(
    index: int,
    object_center_xyz: tuple[float, float, float],
    radius_m: float,
    azimuth_deg: float,
    elevation_deg: float,
) -> ViewpointCandidate:
    center = np.asarray(object_center_xyz, dtype=float)
    elevation = math.radians(elevation_deg)
    azimuth = math.radians(azimuth_deg)
    horizontal = radius_m * math.cos(elevation)
    position = center + np.asarray(
        (
            horizontal * math.cos(azimuth),
            horizontal * math.sin(azimuth),
            radius_m * math.sin(elevation),
        ),
        dtype=float,
    )
    rotation = look_at_rotation_matrix(position, center)
    return ViewpointCandidate(
        index=index,
        position_xyz=tuple(float(value) for value in position),
        orientation_xyzw=matrix_to_quaternion_xyzw(rotation),
        radius_m=float(radius_m),
        azimuth_deg=float(azimuth_deg % 360.0),
        elevation_deg=float(elevation_deg),
    )


def generate_region_viewpoint_alternatives(
    object_center_xyz: tuple[float, float, float],
    radius_m: float,
    region_indices: Iterable[int],
) -> list[RegionViewpointAlternative]:
    """Generate nearby deterministic positions that stay inside each region."""

    center = np.asarray(object_center_xyz, dtype=float)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("object_center_xyz must contain exactly three finite values.")
    if not math.isfinite(radius_m) or radius_m <= 0.0:
        raise ValueError("radius_m must be finite and positive.")
    normalized_center = tuple(float(value) for value in center)
    requested_regions = sorted(set(region_indices))
    if any(index not in range(len(REGION_ORDER)) for index in requested_regions):
        raise ValueError("Capture region indices must be in the range 0 through 3.")

    alternatives: list[RegionViewpointAlternative] = []
    for region_index in requested_regions:
        if region_index == 0:
            # The top cap begins at asin(3/4) ~= 48.6 degrees. These rings
            # remain comfortably inside it and are ordered by angular distance
            # from the nominal zenith representative.
            spherical_coordinates = (
                (0.0, 75.0),
                (90.0, 75.0),
                (180.0, 75.0),
                (270.0, 75.0),
                (45.0, 60.0),
                (135.0, 60.0),
                (225.0, 60.0),
                (315.0, 60.0),
            )
        else:
            center_azimuth = 60.0 + 120.0 * (region_index - 1)
            representative_elevation = math.degrees(math.asin(3.0 / 8.0))
            low_elevation = math.degrees(math.asin(1.0 / 4.0))
            high_elevation = math.degrees(math.asin(1.0 / 2.0))
            spherical_coordinates = (
                (center_azimuth, low_elevation),
                (center_azimuth, high_elevation),
                (center_azimuth - 15.0, representative_elevation),
                (center_azimuth + 15.0, representative_elevation),
                (center_azimuth - 30.0, representative_elevation),
                (center_azimuth + 30.0, representative_elevation),
                (center_azimuth - 30.0, low_elevation),
                (center_azimuth + 30.0, low_elevation),
                (center_azimuth - 30.0, high_elevation),
                (center_azimuth + 30.0, high_elevation),
                (center_azimuth - 45.0, representative_elevation),
                (center_azimuth + 45.0, representative_elevation),
            )

        for rank, (azimuth_deg, elevation_deg) in enumerate(
            spherical_coordinates, start=1
        ):
            viewpoint_index = (
                FALLBACK_VIEWPOINT_INDEX_BASE + region_index * 100 + rank
            )
            alternatives.append(
                RegionViewpointAlternative(
                    region_index=region_index,
                    rank=rank,
                    viewpoint=_spherical_viewpoint(
                        viewpoint_index,
                        normalized_center,
                        radius_m,
                        azimuth_deg,
                        elevation_deg,
                    ),
                )
            )
    return alternatives


def find_reachable_region_alternatives(
    env: UR5EyeInHandPyBulletEnv,
    object_center: tuple[float, float, float],
    view_radius: float,
    missing_region_indices: list[int],
    reachability_kwargs: dict[str, object],
    candidate_backend: object | None,
) -> tuple[dict[int, ReachableViewpoint], dict[int, dict[str, object]]]:
    """Find one validated fallback per missing region, preferring roll zero."""

    alternatives = generate_region_viewpoint_alternatives(
        object_center,
        view_radius,
        missing_region_indices,
    )
    alternative_by_index = {
        alternative.viewpoint.index: alternative for alternative in alternatives
    }
    remaining = set(missing_region_indices)
    selected: dict[int, ReachableViewpoint] = {}
    selection_metadata: dict[int, dict[str, object]] = {}

    for roll_angles in (PRIMARY_CAMERA_ROLLS_DEG, FALLBACK_CAMERA_ROLLS_DEG):
        candidates = [
            alternative.viewpoint
            for alternative in alternatives
            if alternative.region_index in remaining
        ]
        if not candidates:
            break
        report = env.filter_reachable_viewpoints(
            candidates,
            object_center,
            ReachabilityConfig(
                roll_angles_deg=roll_angles,
                **reachability_kwargs,
            ),
            candidate_backend=candidate_backend,
        )
        reachable_by_region: dict[int, tuple[int, ReachableViewpoint]] = {}
        for reachable in report.reachable:
            alternative = alternative_by_index[reachable.viewpoint.index]
            current = reachable_by_region.get(alternative.region_index)
            if current is None or alternative.rank < current[0]:
                reachable_by_region[alternative.region_index] = (
                    alternative.rank,
                    reachable,
                )

        for region_index, (rank, reachable) in reachable_by_region.items():
            selected[region_index] = reachable
            selection_metadata[region_index] = {
                "strategy": "within_region_fallback",
                "fallback_rank": rank,
                "nominal_position_changed": True,
            }
            remaining.discard(region_index)
        if not remaining:
            break

    return selected, selection_metadata


def utc_timestamp() -> tuple[str, str]:
    """Return an ISO timestamp and a filesystem-safe version of the same time."""

    now = datetime.now(timezone.utc)
    return now.isoformat(timespec="microseconds"), now.strftime("%Y%m%dT%H%M%S_%fZ")


def write_json(path: Path, payload: dict[str, object]) -> None:
    """Atomically write human-readable capture metadata."""

    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_path.replace(path)


def capture_metadata(
    env: UR5EyeInHandPyBulletEnv,
    selected: ReachableViewpoint,
    region_name: str,
    viewpoint_selection: dict[str, object],
    image_path: Path,
    captured_at_utc: str,
    motion_metadata: dict[str, object],
) -> dict[str, object]:
    """Describe the exact robot and camera state associated with one image."""

    joint_positions_rad = env.get_joint_positions()
    camera_position, camera_rotation = env.get_wrist_camera_pose()
    viewpoint = selected.viewpoint
    return {
        "region": region_name,
        "viewpoint_selection": viewpoint_selection,
        "image_filename": image_path.name,
        "captured_at_utc": captured_at_utc,
        "robot_joint_order": list(ARM_JOINT_ORDER),
        "robot_joint_positions_rad": {
            name: float(value)
            for name, value in zip(ARM_JOINT_ORDER, joint_positions_rad)
        },
        "robot_joint_positions_deg": {
            name: float(math.degrees(value))
            for name, value in zip(ARM_JOINT_ORDER, joint_positions_rad)
        },
        "actual_camera_pose_world": {
            "position_xyz_m": camera_position.astype(float).tolist(),
            "orientation_xyzw": list(matrix_to_quaternion_xyzw(camera_rotation)),
            "rotation_matrix": camera_rotation.astype(float).tolist(),
        },
        "requested_viewpoint": {
            "position_xyz_m": list(viewpoint.position_xyz),
            "orientation_xyzw": list(viewpoint.orientation_xyzw),
            "radius_m": viewpoint.radius_m,
            "azimuth_deg": viewpoint.azimuth_deg,
            "elevation_deg": viewpoint.elevation_deg,
            "camera_roll_deg": viewpoint.roll_deg,
        },
        "validated_ik": {
            "position_error_m": selected.position_error_m,
            "orientation_error_deg": selected.orientation_error_deg,
            "clearance_m": selected.clearance_m,
            "seed_index": selected.seed_index,
            "solver_provenance": selected.solver_provenance,
        },
        "executed_motion": motion_metadata,
    }


def choose_view_radius(object_bounding_radius_m: float) -> float:
    """Choose one shell radius that frames the complete workpiece."""

    if not math.isfinite(object_bounding_radius_m) or object_bounding_radius_m <= 0.0:
        raise ValueError("object_bounding_radius_m must be finite and positive.")

    vertical_fov = math.radians(compute_camera_vertical_fov_deg())
    horizontal_fov = 2.0 * math.atan(
        math.tan(vertical_fov * 0.5) * (CAMERA_WIDTH / CAMERA_HEIGHT)
    )
    usable_half_fov = (
        0.5 * min(vertical_fov, horizontal_fov) * (1.0 - FRAMING_MARGIN_FRACTION)
    )
    framing_radius = object_bounding_radius_m / math.sin(usable_half_fov)
    radius = max(NOMINAL_RADIUS_M, framing_radius)
    if radius > MAX_RADIUS_M:
        raise RuntimeError(
            f"The workpiece requires a {radius:.3f} m camera radius to fit in frame, "
            f"outside the configured {MAX_RADIUS_M:.3f} m robot workspace limit."
        )
    return radius


def linear_joint_path(
    start: list[float], goal: list[float], max_step_rad: float = MAX_JOINT_STEP_RAD
) -> np.ndarray:
    """Create a dense joint-linear path including its start and goal."""

    start_array = np.asarray(start, dtype=float)
    goal_array = np.asarray(goal, dtype=float)
    largest_move = float(np.max(np.abs(goal_array - start_array)))
    step_count = max(1, int(math.ceil(largest_move / max_step_rad)))
    return np.linspace(start_array, goal_array, step_count + 1)


def play_path(
    env: UR5EyeInHandPyBulletEnv,
    path: np.ndarray,
    duration_s: float,
) -> None:
    """Play a validated path in PyBullet."""

    delay_s = duration_s / max(1, len(path) - 1) if env.gui else 0.0
    for joints in path[1:]:
        env.command_joint_positions(joints.tolist(), blend=1.0)
        env.physics.stepSimulation()
        if delay_s > 0.0:
            time.sleep(delay_s)


def plan_validated_path(
    env: UR5EyeInHandPyBulletEnv,
    goal_joint_positions_rad: list[float],
    motion_planner: CuRoboMotionPlanner | None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Plan a path and require PyBullet to validate every dense waypoint."""

    start = env.get_joint_positions()
    if motion_planner is None:
        path = linear_joint_path(start, goal_joint_positions_rad)
        planner_metadata: dict[str, object] = {
            "planner": "validated_joint_linear_interpolation",
            "planning_time_s": 0.0,
            "used_graph": None,
            "status": "generated",
        }
    else:
        plan = motion_planner.plan(start, goal_joint_positions_rad)
        initialization_time_s = float(
            getattr(plan, "initialization_time_s", 0.0)
        )
        path = plan.joint_positions_rad
        endpoint_error = float(
            np.max(
                np.abs(
                    path[-1] - np.asarray(goal_joint_positions_rad, dtype=float)
                )
            )
        )
        if endpoint_error > 0.025:
            raise RuntimeError(
                "CuRobo path ended too far from the validated IK endpoint: "
                f"{endpoint_error:.6f} rad."
            )
        planner_metadata = {
            "planner": "curobo_collision_aware_motion",
            "planning_time_s": plan.planning_time_s,
            "planner_initialization_time_s": initialization_time_s,
            "total_planning_request_time_s": (
                initialization_time_s + plan.planning_time_s
            ),
            "planned_duration_s": plan.duration_s,
            "used_graph": plan.used_graph,
            "status": plan.status,
        }

    validation = env.validate_joint_trajectory(path, CLEARANCE_M)
    if not validation.valid:
        raise RuntimeError(
            "PyBullet rejected the planned trajectory: "
            f"{validation.failure_reason or 'unknown collision failure'}."
        )
    planner_metadata.update(
        {
            "waypoint_count": int(len(path)),
            "pybullet_samples_checked": validation.samples_checked,
            "pybullet_minimum_clearance_m": validation.minimum_clearance_m,
        }
    )
    return path, planner_metadata


def move_home(
    env: UR5EyeInHandPyBulletEnv,
    duration_s: float,
    outward_paths: list[np.ndarray],
    motion_planner: CuRoboMotionPlanner | None,
) -> None:
    """Plan home directly, or retrace already validated paths if necessary."""

    home = [HOME_JOINT_POSITIONS_RAD[name] for name in ARM_JOINT_ORDER]
    if np.allclose(env.get_joint_positions(), home, atol=1e-6):
        return

    try:
        direct_path, _ = plan_validated_path(env, home, motion_planner)
    except RuntimeError as exc:
        print(f"Direct planned return home failed ({exc}); retracing validated paths.")
    else:
        play_path(env, direct_path, duration_s)
        return

    for path in reversed(outward_paths):
        reverse_path = path[::-1].copy()
        if not env.validate_joint_trajectory(reverse_path, CLEARANCE_M).valid:
            raise RuntimeError("Could not find a collision-free return path to home.")
        play_path(env, reverse_path, duration_s)


def run(args: argparse.Namespace) -> None:
    backend_selection = resolve_ik_backend(
        args.ik_backend,
        args.curobo_robot_config,
    )
    print(
        f"Reachability backend: {backend_selection.effective_backend}; "
        f"motion planner: "
        f"{'CuRobo MotionGen' if backend_selection.uses_curobo else 'validated joint-linear'}"
    )
    workpiece_ply = resolve_workpiece_ply(
        getattr(args, "object", None),
        getattr(args, "workpiece_ply", DEFAULT_WORKPIECE_PLY),
    )
    if not workpiece_ply.is_file():
        raise FileNotFoundError(f"Workpiece PLY not found: {workpiece_ply}")
    env = UR5EyeInHandPyBulletEnv(
        robot_urdf=args.robot_urdf.resolve(),
        workpiece_ply=workpiece_ply.resolve(),
        workpiece_unit_scale=args.workpiece_unit_scale,
        workpiece_rotation_z_deg=args.workpiece_rotation_z,
        gui=not args.headless,
    )
    outward_paths: list[np.ndarray] = []
    motion_planner: CuRoboMotionPlanner | None = None

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
        object_center = env.get_workpiece_vertex_centroid()
        view_radius = choose_view_radius(env.workpiece_bounding_radius_m)
        viewpoints = generate_four_equal_area_viewpoints(
            object_center_xyz=object_center, radius_m=view_radius
        )
        print(
            f"Checking four equal-solid-angle views around object center "
            f"{object_center} at radius {view_radius:.3f} m ..."
        )
        reachability_kwargs = dict(
            seed_count=(args.curobo_seeds if backend_selection.uses_curobo else 6),
            refinement_steps=(0 if backend_selection.uses_curobo else 2),
            clearance_m=CLEARANCE_M,
            require_line_of_sight=True,
        )
        report = env.filter_reachable_viewpoints(
            viewpoints,
            object_center,
            ReachabilityConfig(
                roll_angles_deg=PRIMARY_CAMERA_ROLLS_DEG,
                **reachability_kwargs,
            ),
            candidate_backend=candidate_backend,
        )

        reachable_by_index = {
            reachable.viewpoint.index: reachable for reachable in report.reachable
        }
        selection_metadata_by_index: dict[int, dict[str, object]] = {
            index: {
                "strategy": "nominal_region_representative",
                "fallback_rank": None,
                "nominal_position_changed": False,
            }
            for index in reachable_by_index
        }
        roll_fallback_viewpoints = [
            viewpoint
            for viewpoint in viewpoints
            if viewpoint.index not in reachable_by_index
        ]
        if roll_fallback_viewpoints:
            print(
                "Retrying unreachable positions with optical-axis roll redundancy; "
                "camera positions remain unchanged ..."
            )
            fallback_report = env.filter_reachable_viewpoints(
                roll_fallback_viewpoints,
                object_center,
                ReachabilityConfig(
                    roll_angles_deg=FALLBACK_CAMERA_ROLLS_DEG,
                    **reachability_kwargs,
                ),
                candidate_backend=candidate_backend,
            )
            reachable_by_index.update(
                {
                    reachable.viewpoint.index: reachable
                    for reachable in fallback_report.reachable
                }
            )
            for reachable in fallback_report.reachable:
                selection_metadata_by_index[reachable.viewpoint.index] = {
                    "strategy": "nominal_region_representative",
                    "fallback_rank": None,
                    "nominal_position_changed": False,
                }

        missing_indices = [
            index for index in range(len(REGION_ORDER)) if index not in reachable_by_index
        ]
        if missing_indices and getattr(args, "strict_region_representatives", False):
            missing_names = [REGION_ORDER[index] for index in missing_indices]
            raise RuntimeError(
                "The nominal partition representatives are not reachable for: "
                f"{', '.join(missing_names)}. Within-region fallback was disabled."
            )
        if missing_indices:
            print(
                "Searching deterministic alternative positions inside the "
                "unreachable capture regions ..."
            )
            alternative_reachable, alternative_metadata = (
                find_reachable_region_alternatives(
                    env,
                    object_center,
                    view_radius,
                    missing_indices,
                    reachability_kwargs,
                    candidate_backend,
                )
            )
            reachable_by_index.update(alternative_reachable)
            selection_metadata_by_index.update(alternative_metadata)
            for region_index, reachable in sorted(alternative_reachable.items()):
                viewpoint = reachable.viewpoint
                print(
                    f"Selected within-region fallback for {REGION_ORDER[region_index]}: "
                    f"elevation={viewpoint.elevation_deg:.1f} deg, "
                    f"azimuth={viewpoint.azimuth_deg:.1f} deg, "
                    f"roll={viewpoint.roll_deg:+.0f} deg"
                )

        missing = [
            region_name
            for index, region_name in enumerate(REGION_ORDER)
            if index not in reachable_by_index
        ]
        if missing:
            raise RuntimeError(
                "No validated camera position was found inside the capture regions: "
                f"{', '.join(missing)}."
            )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        run_started_at_utc, run_tag = utc_timestamp()
        manifest_path = args.output_dir / f"capture_manifest_{run_tag}.json"
        manifest: dict[str, object] = {
            "schema_version": 1,
            "status": "in_progress",
            "run_started_at_utc": run_started_at_utc,
            "object_vertex_centroid_world_xyz_m": list(object_center),
            "camera_shell_radius_m": view_radius,
            "ik_backend": backend_selection.report_metadata(),
            "captures": [],
        }
        write_json(manifest_path, manifest)

        for index, region_name in enumerate(REGION_ORDER):
            selected = reachable_by_index[index]
            viewpoint = selected.viewpoint
            print(
                f"Moving to {region_name}: elevation={viewpoint.elevation_deg:.1f} deg, "
                f"azimuth={viewpoint.azimuth_deg:.1f} deg, radius={viewpoint.radius_m:.3f} m, "
                f"roll={viewpoint.roll_deg:+.0f} deg"
            )
            try:
                path, motion_metadata = plan_validated_path(
                    env, selected.joint_positions_rad, motion_planner
                )
            except RuntimeError as exc:
                if not outward_paths:
                    raise
                print(
                    f"Direct path to {region_name} failed ({exc}); returning home "
                    "and replanning from the validated home configuration."
                )
                move_home(
                    env, args.motion_seconds, outward_paths, motion_planner
                )
                outward_paths.clear()
                path, motion_metadata = plan_validated_path(
                    env, selected.joint_positions_rad, motion_planner
                )
            play_path(env, path, args.motion_seconds)
            outward_paths.append(path)

            captured_at_utc, capture_tag = utc_timestamp()
            image_path = args.output_dir / f"{region_name}_{capture_tag}.png"
            metadata = capture_metadata(
                env,
                selected,
                region_name,
                selection_metadata_by_index[index],
                image_path,
                captured_at_utc,
                motion_metadata,
            )
            image_path = env.save_wrist_camera_frame(
                image_path, png_metadata={"capture_metadata_json": json.dumps(metadata)}
            )
            metadata_path = image_path.with_suffix(".json")
            write_json(metadata_path, metadata)
            captures = manifest["captures"]
            if not isinstance(captures, list):
                raise RuntimeError("Internal capture manifest has an invalid captures list.")
            captures.append(metadata)
            write_json(manifest_path, manifest)
            print(f"Saved {image_path} and pose metadata {metadata_path}")

        print("Returning robot to home position ...")
        move_home(env, args.motion_seconds, outward_paths, motion_planner)
        completed_at_utc, _ = utc_timestamp()
        manifest["status"] = "complete"
        manifest["completed_at_utc"] = completed_at_utc
        write_json(manifest_path, manifest)
        print(f"Saved run manifest {manifest_path}")
        print(f"Done. Images are in {args.output_dir.resolve()}")
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
        env.disconnect()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture top and three side images with the simulated wrist camera."
    )
    parser.add_argument("--headless", action="store_true", help="Run without the PyBullet GUI.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR / "four_views",
        help="Directory for timestamped images and robot-pose JSON metadata.",
    )
    parser.add_argument(
        "--motion-seconds",
        type=float,
        default=1.5,
        help="GUI playback time for each move (ignored in headless mode).",
    )
    parser.add_argument("--robot-urdf", type=Path, default=DEFAULT_ROBOT_URDF)
    workpiece_group = parser.add_mutually_exclusive_group()
    workpiece_group.add_argument(
        "--object",
        default=None,
        metavar="NAME",
        help=(
            "Bundled PLY model name from tasks/3d_models, with or without the "
            f".ply suffix (default: {DEFAULT_WORKPIECE_NAME})."
        ),
    )
    workpiece_group.add_argument(
        "--workpiece-ply",
        type=Path,
        default=DEFAULT_WORKPIECE_PLY,
        help="Explicit PLY path; use instead of --object for an external model.",
    )
    parser.add_argument("--workpiece-unit-scale", type=float, default=0.001)
    parser.add_argument("--workpiece-rotation-z", type=float, default=0.0)
    parser.add_argument(
        "--strict-region-representatives",
        action="store_true",
        help=(
            "Stop if a nominal top/side representative is unreachable instead of "
            "searching deterministic positions inside that same region."
        ),
    )
    parser.add_argument(
        "--ik-backend",
        choices=("auto", "pybullet", "curobo"),
        default="auto",
        help="Use CuRobo when available (default), or force one IK backend.",
    )
    parser.add_argument(
        "--curobo-robot-config",
        type=Path,
        default=None,
        help="Camera-aware CuRobo robot YAML; required for strict CuRobo mode.",
    )
    parser.add_argument(
        "--curobo-seeds",
        type=int,
        default=DEFAULT_CUROBO_IK_SEED_COUNT,
        help=f"CuRobo seeds per viewpoint (default: {DEFAULT_CUROBO_IK_SEED_COUNT}).",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    if not math.isfinite(args.motion_seconds) or args.motion_seconds < 0.0:
        raise ValueError("--motion-seconds must be finite and non-negative.")
    if args.curobo_seeds <= 0:
        raise ValueError("--curobo-seeds must be positive.")
    try:
        args.workpiece_ply = resolve_workpiece_ply(args.object, args.workpiece_ply)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    args.object = None
    run(args)


if __name__ == "__main__":
    main()
