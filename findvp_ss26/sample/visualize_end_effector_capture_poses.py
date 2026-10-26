#!/usr/bin/env python3
"""Visualize captured views as image frustums and an obstacle-aware 3D path."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

if __package__:
    from .pybullet_tabletop_ur5e_eye_in_hand import UR5EyeInHandPyBulletEnv
    from .viewpoint_reachability import quaternion_xyzw_to_matrix
    from .workcell_config import (
        CAMERA_HEIGHT,
        CAMERA_WIDTH,
        compute_camera_vertical_fov_deg,
    )
else:
    from pybullet_tabletop_ur5e_eye_in_hand import UR5EyeInHandPyBulletEnv
    from viewpoint_reachability import quaternion_xyzw_to_matrix
    from workcell_config import (
        CAMERA_HEIGHT,
        CAMERA_WIDTH,
        compute_camera_vertical_fov_deg,
    )


VIEW_COLORS = (
    (0.90, 0.10, 0.10),
    (0.0, 0.45, 0.70),
    (0.90, 0.62, 0.0),
    (0.0, 0.62, 0.45),
    (0.80, 0.47, 0.65),
    (0.34, 0.71, 0.91),
    (0.84, 0.37, 0.0),
    (0.45, 0.25, 0.75),
    (0.95, 0.80, 0.15),
)
BEST_VIEW_COLOR = (0.20, 1.0, 0.05)
PATH_COLOR = (0.50, 0.0, 0.50)
PYBULLET_WINDOW_TITLE = "Bullet Physics ExampleBrowser"
WINDOW_GEOMETRY_PATTERN = re.compile(
    r"(?P<width>\d+)x(?P<height>\d+)[+-]\d+[+-]\d+\s+"
    r"(?P<x>[+-]\d+)(?P<y>[+-]\d+)\s*$"
)


@dataclass(frozen=True)
class VisualizationPath:
    """Polyline and length for one obstacle-aware camera transition."""

    points_xyz_m: np.ndarray
    length_m: float
    mode: str
    used_long_arc: bool = False


@dataclass(frozen=True)
class BestView:
    """A capture matched to the global-best record from the search report."""

    capture_index: int
    picture_number: int
    image_filename: str
    score: float | None
    source: str


def parse_xwininfo_window_bounds(
    tree_output: str,
    *,
    title: str = PYBULLET_WINDOW_TITLE,
) -> tuple[int, int, int, int]:
    """Return the largest matching X11 window as a Pillow capture box."""

    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for line in tree_output.splitlines():
        if title not in line:
            continue
        match = WINDOW_GEOMETRY_PATTERN.search(line)
        if match is None:
            continue
        width = int(match.group("width"))
        height = int(match.group("height"))
        x = int(match.group("x"))
        y = int(match.group("y"))
        if width > 1 and height > 1:
            candidates.append((width * height, (x, y, x + width, y + height)))
    if not candidates:
        raise RuntimeError(
            f"Could not locate an X11 window containing {title!r}."
        )
    return max(candidates, key=lambda candidate: candidate[0])[1]


def find_pybullet_window_bounds() -> tuple[int, int, int, int]:
    """Locate the interactive PyBullet window on an X11 desktop."""

    try:
        result = subprocess.run(
            ["xwininfo", "-root", "-tree"],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "GIF export requires the X11 utility 'xwininfo'."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Could not inspect X11 windows: {exc.stderr}") from exc
    return parse_xwininfo_window_bounds(result.stdout)


def save_trajectory_orbit_gif(
    env: UR5EyeInHandPyBulletEnv,
    output_path: Path,
    *,
    object_center_xyz_m: tuple[float, float, float],
    viewer_distance_m: float,
    viewer_yaw_deg: float,
    viewer_pitch_deg: float,
    orbit_degrees: float,
    frame_count: int,
    frames_per_second: float,
    output_width_px: int,
) -> Path:
    """Capture a deterministic orbit of the actual PyBullet GUI as a GIF."""

    if not env.gui:
        raise ValueError("Trajectory GIF export requires the PyBullet GUI.")
    try:
        from PIL import ImageGrab
    except ImportError as exc:
        raise RuntimeError("Trajectory GIF export requires Pillow ImageGrab.") from exc

    capture_box = find_pybullet_window_bounds()
    frames: list[Image.Image] = []
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    for frame_index in range(frame_count):
        fraction = float(frame_index) / float(frame_count)
        env.physics.resetDebugVisualizerCamera(
            cameraDistance=viewer_distance_m,
            cameraYaw=viewer_yaw_deg + orbit_degrees * fraction,
            cameraPitch=viewer_pitch_deg,
            cameraTargetPosition=list(object_center_xyz_m),
        )
        env.physics.stepSimulation()
        time.sleep(1.0 / 60.0)
        frame = ImageGrab.grab(bbox=capture_box).convert("RGB")
        if frame.width != output_width_px:
            output_height_px = max(
                1,
                round(frame.height * output_width_px / frame.width),
            )
            frame = frame.resize((output_width_px, output_height_px), resampling)
        frames.append(frame)

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_duration_ms = max(1, round(1000.0 / frames_per_second))
    frames[0].save(
        output_path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )
    return output_path


def load_capture_poses(path: Path) -> dict[str, object]:
    """Load and validate the fields needed to show capture poses."""

    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Capture-pose JSON not found: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("Capture-pose JSON root must be an object.")
    captures = report.get("captures")
    if not isinstance(captures, list) or not captures:
        raise ValueError("Capture-pose JSON must contain at least one capture.")

    for capture_index, capture in enumerate(captures):
        try:
            image_filename = capture["image_filename"]
            joints = np.asarray(capture["joint_positions_rad"], dtype=float)
            end_effector_pose = capture["end_effector_pose_world"]
            end_effector_position = np.asarray(
                end_effector_pose["position_xyz_m"], dtype=float
            )
            end_effector_orientation = np.asarray(
                end_effector_pose["orientation_xyzw"], dtype=float
            )
            camera_pose = capture["camera_pose_world"]
            camera_position = np.asarray(
                camera_pose["position_xyz_m"], dtype=float
            )
            camera_orientation = np.asarray(
                camera_pose["orientation_xyzw"], dtype=float
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid capture-pose entry {capture_index}.") from exc
        if not isinstance(image_filename, str) or not image_filename:
            raise ValueError(f"Capture {capture_index} has no image filename.")
        if joints.shape != (6,) or not np.isfinite(joints).all():
            raise ValueError(f"Capture {capture_index} must have six finite joints.")
        for label, position in (
            ("end-effector", end_effector_position),
            ("camera", camera_position),
        ):
            if position.shape != (3,) or not np.isfinite(position).all():
                raise ValueError(
                    f"Capture {capture_index} has an invalid {label} XYZ position."
                )
        for label, orientation in (
            ("end-effector", end_effector_orientation),
            ("camera", camera_orientation),
        ):
            if orientation.shape != (4,) or not np.isfinite(orientation).all():
                raise ValueError(
                    f"Capture {capture_index} has an invalid {label} quaternion."
                )
            quaternion_xyzw_to_matrix(orientation)
    return report


def _capture_matches_image(capture: dict[str, object], image_value: object) -> bool:
    if not isinstance(image_value, str) or not image_value:
        return False
    wanted = Path(image_value).expanduser()
    image_filename = capture.get("image_filename")
    if isinstance(image_filename, str) and image_filename == wanted.name:
        return True
    image_file = capture.get("image_file")
    return (
        isinstance(image_file, str)
        and image_file
        and Path(image_file).expanduser().resolve() == wanted.resolve()
    )


def _best_candidate_from_report(payload: dict[str, object]) -> tuple[object, object]:
    final_best = payload.get("final_best")
    if isinstance(final_best, dict):
        return final_best.get("best_image"), final_best.get("best_score")
    final_image = payload.get("final_observation_image")
    if final_image is not None:
        return final_image, payload.get("final_similarity_score")
    best_observation = payload.get("best_observation")
    if isinstance(best_observation, dict):
        return best_observation.get("best_image"), best_observation.get("best_score")
    return None, None


def resolve_best_view(
    report: dict[str, object],
    capture_report_path: Path,
    *,
    refinement_summary_path: Path | None = None,
    best_picture: int | None = None,
) -> BestView | None:
    """Resolve the highlighted capture from an override or sibling reports."""

    captures = report["captures"]
    if best_picture is not None:
        for capture_index, capture in enumerate(captures):
            picture_number = int(capture.get("picture_number", capture_index + 1))
            if picture_number == best_picture:
                return BestView(
                    capture_index=capture_index,
                    picture_number=picture_number,
                    image_filename=str(capture["image_filename"]),
                    score=None,
                    source="--best-picture",
                )
        raise ValueError(f"--best-picture {best_picture} is not present in the report.")

    capture_report_path = capture_report_path.expanduser().resolve()
    suffix = "_end_effector_capture_poses.json"
    prefix = (
        capture_report_path.name[: -len(suffix)]
        if capture_report_path.name.endswith(suffix)
        else capture_report_path.stem
    )
    candidate_paths = (
        [refinement_summary_path.expanduser().resolve()]
        if refinement_summary_path is not None
        else [
            capture_report_path.parent / f"{prefix}_refinement_summary.json",
            capture_report_path.parent / f"{prefix}_evaluation.json",
            capture_report_path.parent / f"{prefix}_radial_summary.json",
        ]
    )
    for candidate_path in candidate_paths:
        if not candidate_path.is_file():
            if refinement_summary_path is not None:
                raise FileNotFoundError(
                    f"Refinement summary not found: {candidate_path}"
                )
            continue
        try:
            payload = json.loads(candidate_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            if refinement_summary_path is not None:
                raise ValueError(
                    f"Could not read refinement summary {candidate_path}: {exc}"
                ) from exc
            print(f"Skipping unreadable best-view report {candidate_path}: {exc}")
            continue
        if not isinstance(payload, dict):
            continue
        image_value, score_value = _best_candidate_from_report(payload)
        for capture_index, capture in enumerate(captures):
            if not _capture_matches_image(capture, image_value):
                continue
            picture_number = int(capture.get("picture_number", capture_index + 1))
            try:
                parsed_score = float(score_value)
            except (TypeError, ValueError):
                parsed_score = math.nan
            score = parsed_score if math.isfinite(parsed_score) else None
            return BestView(
                capture_index=capture_index,
                picture_number=picture_number,
                image_filename=str(capture["image_filename"]),
                score=score,
                source=str(candidate_path),
            )
    return None


def _view_color(capture_index: int, best_view: BestView | None) -> tuple[float, ...]:
    if best_view is not None and capture_index == best_view.capture_index:
        return BEST_VIEW_COLOR
    return VIEW_COLORS[capture_index % len(VIEW_COLORS)]


def _capture_label(
    capture: dict[str, object],
    capture_index: int,
    best_view: BestView | None,
) -> str:
    picture_number = int(capture.get("picture_number", capture_index + 1))
    if best_view is not None and capture_index == best_view.capture_index:
        score = f" | score={best_view.score:.4f}" if best_view.score is not None else ""
        return f"BEST VIEW | Picture {picture_number}{score}"
    stage_value = capture.get("stage")
    if not isinstance(stage_value, str) or not stage_value:
        return f"Picture {picture_number}"
    stage = stage_value.replace("_refinement", "")
    level = capture.get("refinement_level")
    if stage == "radial":
        direction = capture.get("radial_direction", "")
        step = capture.get("radial_step_index", "")
        detail = f"{direction} {step}".strip()
    else:
        path = capture.get("triangle_path")
        detail = (
            ".".join(str(value) for value in path)
            if isinstance(path, list)
            else ""
        )
    level_label = f" L{level}" if level is not None else ""
    detail_label = f" | {detail}" if detail else ""
    return f"Picture {picture_number} | {stage}{level_label}{detail_label}"


def _sample_line(
    start_xyz: np.ndarray,
    end_xyz: np.ndarray,
    maximum_step_m: float,
) -> np.ndarray:
    distance = float(np.linalg.norm(end_xyz - start_xyz))
    if distance <= 1e-12:
        return start_xyz.reshape(1, 3).copy()
    segment_count = max(1, int(math.ceil(distance / maximum_step_m)))
    return np.linspace(start_xyz, end_xyz, segment_count + 1)


def _rotate_about_axis(
    vector: np.ndarray,
    axis: np.ndarray,
    angles_rad: np.ndarray,
) -> np.ndarray:
    """Vectorized Rodrigues rotation for one vector and many angles."""

    cosine = np.cos(angles_rad)[:, None]
    sine = np.sin(angles_rad)[:, None]
    return (
        vector[None, :] * cosine
        + np.cross(axis, vector)[None, :] * sine
        + axis[None, :] * float(np.dot(axis, vector)) * (1.0 - cosine)
    )


def sphere_avoiding_trajectory(
    start_xyz_m: tuple[float, float, float] | list[float] | np.ndarray,
    end_xyz_m: tuple[float, float, float] | list[float] | np.ndarray,
    sphere_center_xyz_m: tuple[float, float, float] | list[float] | np.ndarray,
    sphere_radius_m: float,
    maximum_step_m: float,
    minimum_height_m: float = 0.05,
) -> VisualizationPath:
    """Sample a line, or line/arc/line path around a spherical obstacle.

    This is the geometric technique used by the supplied C++ visualizer. When
    the direct segment crosses the sphere, the entry and exit intersections are
    connected along a great-circle arc. The higher of the short and long arcs
    is selected if the short arc would pass below ``minimum_height_m``.
    """

    start = np.asarray(start_xyz_m, dtype=float)
    end = np.asarray(end_xyz_m, dtype=float)
    center = np.asarray(sphere_center_xyz_m, dtype=float)
    if start.shape != (3,) or end.shape != (3,) or center.shape != (3,):
        raise ValueError("Trajectory endpoints and sphere center must be XYZ values.")
    if not np.isfinite(np.concatenate((start, end, center))).all():
        raise ValueError("Trajectory geometry must contain only finite values.")
    if not math.isfinite(sphere_radius_m) or sphere_radius_m <= 0.0:
        raise ValueError("Sphere radius must be finite and positive.")
    if not math.isfinite(maximum_step_m) or maximum_step_m <= 0.0:
        raise ValueError("Maximum path step must be finite and positive.")
    if not math.isfinite(minimum_height_m):
        raise ValueError("Minimum path height must be finite.")

    start_radius = float(np.linalg.norm(start - center))
    end_radius = float(np.linalg.norm(end - center))
    endpoint_tolerance = max(1e-9, sphere_radius_m * 1e-9)
    if (
        start_radius < sphere_radius_m - endpoint_tolerance
        or end_radius < sphere_radius_m - endpoint_tolerance
    ):
        raise ValueError("A trajectory endpoint lies inside the obstacle sphere.")

    direction = end - start
    direct_distance = float(np.linalg.norm(direction))
    if direct_distance <= 1e-12:
        return VisualizationPath(start.reshape(1, 3).copy(), 0.0, "line")
    offset = start - center
    quadratic_a = float(np.dot(direction, direction))
    quadratic_b = 2.0 * float(np.dot(direction, offset))
    quadratic_c = float(np.dot(offset, offset) - sphere_radius_m**2)
    discriminant = quadratic_b**2 - 4.0 * quadratic_a * quadratic_c
    discriminant_tolerance = 1e-12 * max(
        1.0,
        quadratic_b**2,
        abs(4.0 * quadratic_a * quadratic_c),
    )
    if discriminant <= discriminant_tolerance:
        return VisualizationPath(
            _sample_line(start, end, maximum_step_m), direct_distance, "line"
        )

    square_root = math.sqrt(max(0.0, discriminant))
    roots = sorted(
        (
            (-quadratic_b - square_root) / (2.0 * quadratic_a),
            (-quadratic_b + square_root) / (2.0 * quadratic_a),
        )
    )
    root_tolerance = 1e-9
    if roots[0] < -root_tolerance or roots[1] > 1.0 + root_tolerance:
        return VisualizationPath(
            _sample_line(start, end, maximum_step_m), direct_distance, "line"
        )

    entry = start + float(np.clip(roots[0], 0.0, 1.0)) * direction
    exit_point = start + float(np.clip(roots[1], 0.0, 1.0)) * direction
    entry_unit = (entry - center) / sphere_radius_m
    exit_unit = (exit_point - center) / sphere_radius_m
    cosine = float(np.clip(np.dot(entry_unit, exit_unit), -1.0, 1.0))
    short_angle = math.acos(cosine)
    axis = np.cross(entry_unit, exit_unit)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-10:
        basis = np.eye(3)[int(np.argmin(np.abs(entry_unit)))]
        axis = np.cross(entry_unit, basis)
        axis_norm = float(np.linalg.norm(axis))
    axis /= axis_norm

    short_segments = max(
        1, int(math.ceil(sphere_radius_m * short_angle / maximum_step_m))
    )
    short_angles = np.linspace(0.0, short_angle, short_segments + 1)
    short_arc = center + sphere_radius_m * _rotate_about_axis(
        entry_unit, axis, short_angles
    )
    long_angle = 2.0 * math.pi - short_angle
    long_segments = max(
        1, int(math.ceil(sphere_radius_m * long_angle / maximum_step_m))
    )
    long_angles = np.linspace(0.0, -long_angle, long_segments + 1)
    long_arc = center + sphere_radius_m * _rotate_about_axis(
        entry_unit, axis, long_angles
    )
    use_long_arc = (
        float(np.min(short_arc[:, 2])) < minimum_height_m
        and float(np.min(long_arc[:, 2])) > float(np.min(short_arc[:, 2]))
    )
    arc = long_arc if use_long_arc else short_arc
    selected_angle = long_angle if use_long_arc else short_angle

    first_line = _sample_line(start, entry, maximum_step_m)
    last_line = _sample_line(exit_point, end, maximum_step_m)
    points = np.concatenate((first_line, arc[1:], last_line[1:]), axis=0)
    length = (
        float(np.linalg.norm(start - entry))
        + sphere_radius_m * selected_angle
        + float(np.linalg.norm(exit_point - end))
    )
    return VisualizationPath(points, length, "sphere_arc", use_long_arc)


def camera_frustum_geometry(
    position_xyz_m: list[float] | tuple[float, ...] | np.ndarray,
    orientation_xyzw: list[float] | tuple[float, ...] | np.ndarray,
    depth_m: float,
    image_width: int = CAMERA_WIDTH,
    image_height: int = CAMERA_HEIGHT,
) -> tuple[np.ndarray, np.ndarray]:
    """Return optical center and four world-space image-plane corners."""

    if not math.isfinite(depth_m) or depth_m <= 0.0:
        raise ValueError("Frustum depth must be finite and positive.")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Frustum image dimensions must be positive.")
    origin = np.asarray(position_xyz_m, dtype=float)
    rotation = quaternion_xyzw_to_matrix(orientation_xyzw)
    vertical_half_angle = math.radians(compute_camera_vertical_fov_deg()) * 0.5
    half_height = depth_m * math.tan(vertical_half_angle)
    half_width = half_height * float(image_width) / float(image_height)
    local_corners = np.asarray(
        (
            (-half_width, -half_height, depth_m),
            (half_width, -half_height, depth_m),
            (half_width, half_height, depth_m),
            (-half_width, half_height, depth_m),
        )
    )
    return origin, origin + local_corners @ rotation.T


def draw_camera_frustums(
    env: UR5EyeInHandPyBulletEnv,
    report: dict[str, object],
    *,
    depth_m: float,
    line_width: float,
    show_labels: bool,
    best_view: BestView | None = None,
    label_size: float = 0.75,
    best_label_size: float = 1.05,
    best_view_scale: float = 1.15,
) -> int:
    """Draw uniquely colored wireframes with an emphasized global best."""

    line_count = 0
    for capture_index, capture in enumerate(report["captures"]):
        camera_pose = capture["camera_pose_world"]
        is_best = best_view is not None and capture_index == best_view.capture_index
        origin, corners = camera_frustum_geometry(
            camera_pose["position_xyz_m"],
            camera_pose["orientation_xyzw"],
            depth_m * (best_view_scale if is_best else 1.0),
        )
        color = _view_color(capture_index, best_view)
        current_line_width = line_width * (2.5 if is_best else 1.0)
        edges = [
            *((origin, corner) for corner in corners),
            *((corners[index], corners[(index + 1) % 4]) for index in range(4)),
        ]
        for start, end in edges:
            env.physics.addUserDebugLine(
                start.tolist(),
                end.tolist(),
                lineColorRGB=color,
                lineWidth=current_line_width,
                lifeTime=0.0,
            )
            line_count += 1
        if show_labels:
            label_position = origin + np.asarray(
                (0.0, 0.0, 0.025 if is_best else 0.015)
            )
            env.physics.addUserDebugText(
                _capture_label(capture, capture_index, best_view),
                label_position.tolist(),
                textColorRGB=color,
                textSize=best_label_size if is_best else label_size,
                lifeTime=0.0,
            )
    return line_count


def draw_best_view_marker(
    env: UR5EyeInHandPyBulletEnv,
    report: dict[str, object],
    best_view: BestView,
    *,
    object_center_xyz_m: tuple[float, float, float],
    marker_radius_m: float,
    line_width: float,
) -> None:
    """Add a bright marker and object-pointing ray at the best camera center."""

    capture = report["captures"][best_view.capture_index]
    camera_position = capture["camera_pose_world"]["position_xyz_m"]
    visual_shape = env.physics.createVisualShape(
        shapeType=env.physics.GEOM_SPHERE,
        radius=marker_radius_m,
        rgbaColor=[*BEST_VIEW_COLOR, 1.0],
    )
    env.physics.createMultiBody(
        baseMass=0.0,
        baseVisualShapeIndex=visual_shape,
        basePosition=camera_position,
    )
    env.physics.addUserDebugLine(
        camera_position,
        list(object_center_xyz_m),
        lineColorRGB=BEST_VIEW_COLOR,
        lineWidth=line_width * 2.5,
        lifeTime=0.0,
    )


def _resolve_capture_image(
    capture: dict[str, object],
    capture_report_path: Path,
) -> Path | None:
    candidates = []
    image_file = capture.get("image_file")
    if isinstance(image_file, str) and image_file:
        candidates.append(Path(image_file).expanduser())
    image_filename = capture.get("image_filename")
    if isinstance(image_filename, str) and image_filename:
        candidates.append(capture_report_path.parent / image_filename)
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    return None


def draw_projected_images(
    env: UR5EyeInHandPyBulletEnv,
    report: dict[str, object],
    capture_report_path: Path,
    *,
    depth_m: float,
    sample_step_px: int,
    point_size: float,
    flip_images: bool,
) -> tuple[int, int]:
    """Project sampled RGB pixels onto each saved optical image plane."""

    displayed = 0
    skipped = 0
    vertical_half_angle = math.radians(compute_camera_vertical_fov_deg()) * 0.5
    for capture in report["captures"]:
        image_path = _resolve_capture_image(capture, capture_report_path)
        if image_path is None:
            print(f"Skipping missing capture image: {capture.get('image_filename')}")
            skipped += 1
            continue
        try:
            with Image.open(image_path) as source:
                image = source.convert("RGB")
                if flip_images:
                    image = image.transpose(Image.Transpose.ROTATE_180)
                pixels = np.asarray(image)
        except (OSError, ValueError) as exc:
            print(f"Skipping unreadable capture image {image_path}: {exc}")
            skipped += 1
            continue
        height, width = pixels.shape[:2]
        rows = np.arange(0, height, sample_step_px, dtype=float)
        columns = np.arange(0, width, sample_step_px, dtype=float)
        grid_x, grid_y = np.meshgrid(columns, rows)
        focal_length_px = height / (2.0 * math.tan(vertical_half_angle))
        local_points = np.stack(
            (
                (grid_x.ravel() + 0.5 - width * 0.5)
                * depth_m
                / focal_length_px,
                (grid_y.ravel() + 0.5 - height * 0.5)
                * depth_m
                / focal_length_px,
                np.full(grid_x.size, depth_m),
            ),
            axis=1,
        )
        camera_pose = capture["camera_pose_world"]
        origin = np.asarray(camera_pose["position_xyz_m"], dtype=float)
        rotation = quaternion_xyzw_to_matrix(camera_pose["orientation_xyzw"])
        world_points = origin + local_points @ rotation.T
        colors = pixels[
            ::sample_step_px,
            ::sample_step_px,
            :,
        ].reshape(-1, 3) / 255.0
        for batch_start in range(0, len(world_points), 10000):
            batch_end = batch_start + 10000
            env.physics.addUserDebugPoints(
                world_points[batch_start:batch_end].tolist(),
                colors[batch_start:batch_end].tolist(),
                pointSize=point_size,
                lifeTime=0.0,
            )
        displayed += 1
    return displayed, skipped


def draw_camera_trajectory(
    env: UR5EyeInHandPyBulletEnv,
    report: dict[str, object],
    *,
    object_center_xyz_m: tuple[float, float, float],
    obstacle_radius_m: float,
    maximum_step_m: float,
    minimum_height_m: float,
    line_width: float,
) -> list[VisualizationPath]:
    """Connect sequential camera views with purple sphere-avoiding polylines."""

    captures = report["captures"]
    paths: list[VisualizationPath] = []
    for transition_index, (previous, current) in enumerate(
        zip(captures, captures[1:]), start=1
    ):
        try:
            path = sphere_avoiding_trajectory(
                previous["camera_pose_world"]["position_xyz_m"],
                current["camera_pose_world"]["position_xyz_m"],
                object_center_xyz_m,
                obstacle_radius_m,
                maximum_step_m,
                minimum_height_m,
            )
        except ValueError as exc:
            print(f"Skipping view transition {transition_index}: {exc}")
            continue
        paths.append(path)
        for start, end in zip(path.points_xyz_m[:-1], path.points_xyz_m[1:]):
            env.physics.addUserDebugLine(
                start.tolist(),
                end.tolist(),
                lineColorRGB=PATH_COLOR,
                lineWidth=line_width,
                lifeTime=0.0,
            )
    return paths


def _metadata_path(
    report: dict[str, object],
    key: str,
    override: Path | None,
) -> Path:
    value = override if override is not None else report.get(key)
    if not isinstance(value, (str, Path)):
        raise ValueError(f"Capture-pose JSON does not define {key!r}.")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{key} not found: {path}")
    return path


def draw_capture_poses(
    env: UR5EyeInHandPyBulletEnv,
    report: dict[str, object],
    *,
    rectangle_width: float,
    rectangle_height: float,
    rectangle_thickness: float,
    line_width: float = 3.0,
    best_view: BestView | None = None,
    label_size: float = 0.75,
    best_label_size: float = 1.05,
    best_view_scale: float = 1.15,
) -> None:
    """Draw one oriented, numbered rectangle per captured image."""

    for capture_index, capture in enumerate(report["captures"]):
        pose = capture["end_effector_pose_world"]
        origin = np.asarray(pose["position_xyz_m"], dtype=float)
        orientation = [float(value) for value in pose["orientation_xyzw"]]
        rotation = quaternion_xyzw_to_matrix(orientation)
        is_best = best_view is not None and capture_index == best_view.capture_index
        color = _view_color(capture_index, best_view)
        size_scale = best_view_scale if is_best else 1.0
        current_width = rectangle_width * size_scale
        current_height = rectangle_height * size_scale
        visual_shape = env.physics.createVisualShape(
            shapeType=env.physics.GEOM_BOX,
            halfExtents=[
                current_width * 0.5,
                current_height * 0.5,
                rectangle_thickness * 0.5,
            ],
            rgbaColor=[*color, 0.85 if is_best else 0.40],
        )
        env.physics.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=visual_shape,
            basePosition=origin.tolist(),
            baseOrientation=orientation,
        )

        half_x = rotation[:, 0] * current_width * 0.5
        half_y = rotation[:, 1] * current_height * 0.5
        corners = (
            origin - half_x - half_y,
            origin + half_x - half_y,
            origin + half_x + half_y,
            origin - half_x + half_y,
        )
        for start, end in zip(corners, (*corners[1:], corners[0])):
            env.physics.addUserDebugLine(
                start.tolist(),
                end.tolist(),
                lineColorRGB=color,
                lineWidth=line_width * (2.5 if is_best else 1.0),
                lifeTime=0.0,
            )

        label_position = origin + rotation[:, 1] * (
            current_height * 0.5 + (0.020 if is_best else 0.012)
        )
        env.physics.addUserDebugText(
            _capture_label(capture, capture_index, best_view),
            label_position.tolist(),
            textColorRGB=color,
            textSize=best_label_size if is_best else label_size,
            lifeTime=0.0,
        )


def draw_supervisor_visualization(
    env: UR5EyeInHandPyBulletEnv,
    report: dict[str, object],
    capture_report_path: Path,
    *,
    frustum_depth_m: float,
    image_depth_m: float,
    image_sample_step_px: int,
    image_point_size: float,
    path_step_m: float,
    path_minimum_height_m: float,
    obstacle_radius_m: float,
    best_marker_radius_m: float,
    line_width: float,
    label_size: float,
    best_label_size: float,
    best_view_scale: float,
    show_frustums: bool,
    show_path: bool,
    show_images: bool,
    show_labels: bool,
    flip_images: bool,
    show_obstacle_sphere: bool,
    best_view: BestView | None,
) -> dict[str, object]:
    """Draw the supplied C++ visualizer's frustum/image/path composition."""

    object_center = env.get_workpiece_vertex_centroid()
    if show_obstacle_sphere:
        visual_shape = env.physics.createVisualShape(
            shapeType=env.physics.GEOM_SPHERE,
            radius=obstacle_radius_m,
            rgbaColor=[*PATH_COLOR, 0.10],
        )
        env.physics.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=visual_shape,
            basePosition=list(object_center),
        )

    frustum_line_count = (
        draw_camera_frustums(
            env,
            report,
            depth_m=frustum_depth_m,
            line_width=line_width,
            show_labels=show_labels,
            best_view=best_view,
            label_size=label_size,
            best_label_size=best_label_size,
            best_view_scale=best_view_scale,
        )
        if show_frustums
        else 0
    )
    image_count, skipped_image_count = (
        draw_projected_images(
            env,
            report,
            capture_report_path,
            depth_m=image_depth_m,
            sample_step_px=image_sample_step_px,
            point_size=image_point_size,
            flip_images=flip_images,
        )
        if show_images
        else (0, 0)
    )
    paths = (
        draw_camera_trajectory(
            env,
            report,
            object_center_xyz_m=object_center,
            obstacle_radius_m=obstacle_radius_m,
            maximum_step_m=path_step_m,
            minimum_height_m=path_minimum_height_m,
            line_width=line_width,
        )
        if show_path
        else []
    )
    if best_view is not None:
        draw_best_view_marker(
            env,
            report,
            best_view,
            object_center_xyz_m=object_center,
            marker_radius_m=best_marker_radius_m,
            line_width=line_width,
        )
    return {
        "view_count": len(report["captures"]),
        "frustum_line_count": frustum_line_count,
        "projected_image_count": image_count,
        "skipped_image_count": skipped_image_count,
        "path_segment_count": len(paths),
        "sphere_arc_segment_count": sum(path.mode == "sphere_arc" for path in paths),
        "path_length_m": float(sum(path.length_m for path in paths)),
        "object_center_xyz_m": list(object_center),
        "obstacle_radius_m": obstacle_radius_m,
        "best_picture_number": (
            best_view.picture_number if best_view is not None else None
        ),
        "best_score": best_view.score if best_view is not None else None,
        "best_view_source": best_view.source if best_view is not None else None,
    }


def show_capture_states(
    env: UR5EyeInHandPyBulletEnv,
    report: dict[str, object],
    pose_delay_s: float,
) -> None:
    """Step the robot through the discrete joint states used for the images."""

    captures = report["captures"]
    for capture_index, capture in enumerate(captures):
        env.command_joint_positions(
            [float(value) for value in capture["joint_positions_rad"]],
            blend=1.0,
        )
        env.physics.stepSimulation()
        position = capture["end_effector_pose_world"]["position_xyz_m"]
        picture_number = int(capture.get("picture_number", capture_index + 1))
        print(
            f"[Picture {picture_number}/{len(captures)}] "
            f"{capture['image_filename']}: tool0 xyz={position}"
        )
        if env.gui and pose_delay_s > 0.0:
            time.sleep(pose_delay_s)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Show saved RGB views on camera frustums and connect them with an "
            "object-avoiding 3D trajectory."
        )
    )
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--robot-urdf", type=Path, default=None)
    parser.add_argument("--workpiece-ply", type=Path, default=None)
    parser.add_argument(
        "--refinement-summary",
        type=Path,
        default=None,
        help="Override the automatically discovered best-view summary JSON.",
    )
    parser.add_argument(
        "--best-picture",
        type=int,
        default=None,
        help="Highlight this one-based picture number instead of report metadata.",
    )
    parser.add_argument("--headless", action="store_true")
    cycle = parser.add_mutually_exclusive_group()
    cycle.add_argument(
        "--cycle-robot",
        action="store_true",
        help="Also step the UR5e through the discrete saved capture states.",
    )
    cycle.add_argument(
        "--no-cycle",
        dest="cycle_robot",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(cycle_robot=False)
    parser.add_argument("--pose-delay", type=float, default=0.75)
    parser.add_argument("--frustum-depth", type=float, default=0.08)
    parser.add_argument("--image-depth", type=float, default=0.079)
    parser.add_argument("--image-sample-step", type=int, default=4)
    parser.add_argument("--image-point-size", type=float, default=2.0)
    parser.add_argument("--path-step", type=float, default=0.01)
    parser.add_argument("--path-min-height", type=float, default=0.05)
    parser.add_argument(
        "--obstacle-padding",
        type=float,
        default=0.02,
        help="Clearance added to the workpiece bounding-sphere radius.",
    )
    parser.add_argument(
        "--obstacle-radius",
        type=float,
        default=None,
        help="Override the object avoidance sphere radius in metres.",
    )
    parser.add_argument("--line-width", type=float, default=3.0)
    parser.add_argument("--best-marker-radius", type=float, default=0.012)
    parser.add_argument(
        "--label-size",
        type=float,
        default=0.75,
        help="PyBullet text size for ordinary view labels.",
    )
    parser.add_argument(
        "--best-label-size",
        type=float,
        default=1.05,
        help="PyBullet text size for the highlighted best-view label.",
    )
    parser.add_argument(
        "--best-view-scale",
        type=float,
        default=1.15,
        help="Size multiplier for the best frustum and tool rectangle.",
    )
    parser.add_argument("--viewer-distance", type=float, default=1.4)
    parser.add_argument("--viewer-yaw", type=float, default=0.0)
    parser.add_argument("--viewer-pitch", type=float, default=-35.0)
    parser.add_argument(
        "--gif-output",
        type=Path,
        default=None,
        help=(
            "Save a looping orbit of the actual PyBullet viewport as a GIF; "
            "requires an unobscured X11 GUI window."
        ),
    )
    parser.add_argument("--gif-frames", type=int, default=48)
    parser.add_argument("--gif-fps", type=float, default=12.0)
    parser.add_argument("--gif-orbit-degrees", type=float, default=360.0)
    parser.add_argument("--gif-width", type=int, default=800)
    parser.add_argument("--no-frustums", action="store_true")
    parser.add_argument("--no-path", action="store_true")
    parser.add_argument("--no-images", action="store_true")
    parser.add_argument("--no-labels", action="store_true")
    parser.add_argument(
        "--flip-images",
        action="store_true",
        help="Rotate projected images by 180 degrees as in the supplied C++ example.",
    )
    parser.add_argument("--show-obstacle-sphere", action="store_true")
    parser.add_argument(
        "--show-tool-rectangles",
        action="store_true",
        help="Overlay the previous numbered tool0 rectangles.",
    )
    parser.add_argument("--rectangle-width", type=float, default=0.045)
    parser.add_argument("--rectangle-height", type=float, default=0.034)
    parser.add_argument("--rectangle-thickness", type=float, default=0.001)
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=0.0,
        help="GUI display time; zero keeps it open until interrupted.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for value, name in (
        (args.pose_delay, "--pose-delay"),
        (args.hold_seconds, "--hold-seconds"),
        (args.obstacle_padding, "--obstacle-padding"),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative.")
    for value, name in (
        (args.frustum_depth, "--frustum-depth"),
        (args.image_depth, "--image-depth"),
        (args.image_point_size, "--image-point-size"),
        (args.path_step, "--path-step"),
        (args.line_width, "--line-width"),
        (args.best_marker_radius, "--best-marker-radius"),
        (args.label_size, "--label-size"),
        (args.best_label_size, "--best-label-size"),
        (args.best_view_scale, "--best-view-scale"),
        (args.gif_fps, "--gif-fps"),
        (args.viewer_distance, "--viewer-distance"),
        (args.rectangle_width, "--rectangle-width"),
        (args.rectangle_height, "--rectangle-height"),
        (args.rectangle_thickness, "--rectangle-thickness"),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")
    if args.best_view_scale < 1.0:
        raise ValueError("--best-view-scale must be at least 1.0.")
    if args.image_sample_step <= 0:
        raise ValueError("--image-sample-step must be positive.")
    if args.gif_frames <= 0:
        raise ValueError("--gif-frames must be positive.")
    if args.gif_width <= 0:
        raise ValueError("--gif-width must be positive.")
    if not math.isfinite(args.gif_orbit_degrees):
        raise ValueError("--gif-orbit-degrees must be finite.")
    if args.gif_output is not None and args.headless:
        raise ValueError("--gif-output cannot be combined with --headless.")
    if args.best_picture is not None and args.best_picture <= 0:
        raise ValueError("--best-picture must be positive.")
    if not math.isfinite(args.path_min_height):
        raise ValueError("--path-min-height must be finite.")
    if args.obstacle_radius is not None and (
        not math.isfinite(args.obstacle_radius) or args.obstacle_radius <= 0.0
    ):
        raise ValueError("--obstacle-radius must be finite and positive.")
    for value, name in (
        (args.viewer_yaw, "--viewer-yaw"),
        (args.viewer_pitch, "--viewer-pitch"),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite.")

    report = load_capture_poses(args.captures)
    best_view = resolve_best_view(
        report,
        args.captures,
        refinement_summary_path=args.refinement_summary,
        best_picture=args.best_picture,
    )
    env = UR5EyeInHandPyBulletEnv(
        robot_urdf=_metadata_path(report, "robot_urdf", args.robot_urdf),
        workpiece_ply=_metadata_path(report, "workpiece_ply", args.workpiece_ply),
        workpiece_unit_scale=float(report.get("workpiece_unit_scale", 0.001)),
        workpiece_rotation_z_deg=float(
            report.get("workpiece_rotation_z_deg", 0.0)
        ),
        gui=not args.headless,
    )
    try:
        object_center = env.get_workpiece_vertex_centroid()
        obstacle_radius = (
            args.obstacle_radius
            if args.obstacle_radius is not None
            else float(env.workpiece_bounding_radius_m) + args.obstacle_padding
        )
        if env.gui:
            if args.gif_output is not None:
                env.physics.configureDebugVisualizer(
                    env.physics.COV_ENABLE_GUI,
                    0,
                )
            env.physics.resetDebugVisualizerCamera(
                cameraDistance=args.viewer_distance,
                cameraYaw=args.viewer_yaw,
                cameraPitch=args.viewer_pitch,
                cameraTargetPosition=list(object_center),
            )
        summary = draw_supervisor_visualization(
            env,
            report,
            args.captures.expanduser().resolve(),
            frustum_depth_m=args.frustum_depth,
            image_depth_m=args.image_depth,
            image_sample_step_px=args.image_sample_step,
            image_point_size=args.image_point_size,
            path_step_m=args.path_step,
            path_minimum_height_m=args.path_min_height,
            obstacle_radius_m=obstacle_radius,
            best_marker_radius_m=args.best_marker_radius,
            line_width=args.line_width,
            label_size=args.label_size,
            best_label_size=args.best_label_size,
            best_view_scale=args.best_view_scale,
            show_frustums=not args.no_frustums,
            show_path=not args.no_path,
            show_images=not args.no_images,
            show_labels=not args.no_labels,
            flip_images=args.flip_images,
            show_obstacle_sphere=args.show_obstacle_sphere,
            best_view=best_view,
        )
        print(
            "Supervisor-style visualization: "
            f"{summary['view_count']} views, "
            f"{summary['projected_image_count']} projected images, "
            f"{summary['path_segment_count']} transitions, "
            f"{summary['path_length_m']:.3f} m visualized path "
            f"({summary['sphere_arc_segment_count']} sphere-avoiding arc(s))."
        )
        print(
            f"Object avoidance sphere: center={summary['object_center_xyz_m']}, "
            f"radius={summary['obstacle_radius_m']:.3f} m."
        )
        if best_view is not None:
            score_label = (
                f", score={best_view.score:.4f}" if best_view.score is not None else ""
            )
            print(
                f"Highlighted BEST VIEW: Picture {best_view.picture_number} "
                f"({best_view.image_filename}{score_label}); source={best_view.source}."
            )
        else:
            print(
                "No global-best metadata was found; use --refinement-summary "
                "or --best-picture to select one explicitly."
            )
        if args.show_tool_rectangles:
            draw_capture_poses(
                env,
                report,
                rectangle_width=args.rectangle_width,
                rectangle_height=args.rectangle_height,
                rectangle_thickness=args.rectangle_thickness,
                line_width=args.line_width,
                best_view=best_view,
                label_size=args.label_size,
                best_label_size=args.best_label_size,
                best_view_scale=args.best_view_scale,
            )
        if args.gif_output is not None:
            saved_gif = save_trajectory_orbit_gif(
                env,
                args.gif_output,
                object_center_xyz_m=object_center,
                viewer_distance_m=args.viewer_distance,
                viewer_yaw_deg=args.viewer_yaw,
                viewer_pitch_deg=args.viewer_pitch,
                orbit_degrees=args.gif_orbit_degrees,
                frame_count=args.gif_frames,
                frames_per_second=args.gif_fps,
                output_width_px=args.gif_width,
            )
            print(f"Saved trajectory GIF: {saved_gif}")
        if args.cycle_robot:
            show_capture_states(env, report, args.pose_delay)

        if env.gui and (args.gif_output is None or args.hold_seconds > 0.0):
            print("View trajectory displayed. Close PyBullet or press Ctrl+C to exit.")
            deadline = (
                time.monotonic() + args.hold_seconds
                if args.hold_seconds > 0.0
                else None
            )
            while env.physics.isConnected() and (
                deadline is None or time.monotonic() < deadline
            ):
                env.physics.stepSimulation()
                time.sleep(1.0 / 60.0)
    except KeyboardInterrupt:
        pass
    finally:
        env.disconnect()


if __name__ == "__main__":
    main()
