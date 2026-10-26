"""Real-camera observation records compatible with RGB-only localization."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from .geometry import validate_transform


def camera_matrix_from_info(message: Any) -> np.ndarray:
    matrix = np.asarray(message.k, dtype=float).reshape(3, 3)
    if not np.all(np.isfinite(matrix)) or matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
        raise ValueError("CameraInfo K must contain finite, positive focal lengths")
    if not np.allclose(matrix[2], (0.0, 0.0, 1.0), atol=1e-10):
        raise ValueError("CameraInfo K has an invalid homogeneous row")
    return matrix


def _ordered_joint_field(
    message: Any, joint_names: tuple[str, ...], field_name: str
) -> list[float]:
    field = getattr(message, field_name)
    if len(message.name) != len(field):
        raise ValueError(f"JointState name and {field_name} lengths differ")
    values = {name: float(value) for name, value in zip(message.name, field)}
    missing = [name for name in joint_names if name not in values]
    if missing:
        raise ValueError(f"JointState is missing required joints: {missing}")
    result = [values[name] for name in joint_names]
    if not np.all(np.isfinite(result)):
        raise ValueError("JointState contains non-finite positions")
    return result


def ordered_joint_positions(message: Any, joint_names: tuple[str, ...]) -> list[float]:
    return _ordered_joint_field(message, joint_names, "position")


def ordered_joint_velocities(message: Any, joint_names: tuple[str, ...]) -> list[float]:
    return _ordered_joint_field(message, joint_names, "velocity")


def make_observation_record(
    *,
    image: str,
    label: str,
    intrinsics: np.ndarray,
    distortion_coefficients: list[float],
    planning_from_camera: np.ndarray,
    stamp_ns: int,
    camera_frame: str,
    image_header_frame: str,
    joint_names: tuple[str, ...],
    joint_positions_rad: list[float],
) -> dict[str, Any]:
    if stamp_ns < 0:
        raise ValueError("stamp_ns must be non-negative")
    transform = validate_transform(planning_from_camera, "planning_from_camera")
    matrix = np.asarray(intrinsics, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("intrinsics must be a finite 3x3 matrix")
    distortion = np.asarray(distortion_coefficients, dtype=float).reshape(-1)
    if distortion.size == 0:
        distortion = np.zeros(5)
    if distortion.size not in (0, 4, 5, 8, 12, 14) or not np.all(np.isfinite(distortion)):
        raise ValueError("distortion coefficients have an unsupported shape")
    if len(joint_names) != len(joint_positions_rad):
        raise ValueError("joint name and position counts differ")
    return {
        "label": str(label),
        "image": str(image),
        "intrinsics": matrix.tolist(),
        "distortion_coefficients": distortion.tolist(),
        "world_from_camera": transform.tolist(),
        "stamp_ns": int(stamp_ns),
        "camera_frame": str(camera_frame),
        "image_header_frame": str(image_header_frame),
        "joint_names": list(joint_names),
        "joint_positions_rad": [float(value) for value in joint_positions_rad],
    }


def append_observation(path: Path, coordinate_frame: str, record: dict[str, Any]) -> int:
    """Atomically append one record and return its zero-based index."""

    manifest = Path(path).expanduser().resolve()
    if not coordinate_frame.strip():
        raise ValueError("coordinate_frame must be nonempty")
    if manifest.exists():
        document = json.loads(manifest.read_text(encoding="utf-8"))
        if document.get("coordinate_frame") != coordinate_frame:
            raise ValueError(
                "manifest coordinate frame does not match the capture configuration"
            )
        observations = document.get("observations")
        if not isinstance(observations, list):
            raise ValueError("manifest observations must be a list")
    else:
        document = {
            "schema_version": 1,
            "coordinate_frame": coordinate_frame,
            "position_units": "metres",
            "orientation_convention": "quaternion_xyzw",
            "estimate_input": "calibrated_rgb_only",
            "observations": [],
        }
        observations = document["observations"]
    observations.append(record)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{manifest.name}.", suffix=".tmp", dir=manifest.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, manifest)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return len(observations) - 1
