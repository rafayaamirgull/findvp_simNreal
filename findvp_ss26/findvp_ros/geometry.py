"""Small, ROS-independent SE(3) helpers used by the hardware adapter."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


def quaternion_xyzw_to_matrix(value: Iterable[float]) -> np.ndarray:
    quaternion = np.asarray(tuple(value), dtype=float)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("quaternion must contain four finite xyzw values")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("quaternion norm must be positive")
    x, y, z, w = quaternion / norm
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=float,
    )


def matrix_to_quaternion_xyzw(rotation: np.ndarray) -> tuple[float, float, float, float]:
    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("rotation must be a finite 3x3 matrix")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = (
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
            0.25 * scale,
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = (0.25 * scale, (matrix[0, 1] + matrix[1, 0]) / scale,
                          (matrix[0, 2] + matrix[2, 0]) / scale,
                          (matrix[2, 1] - matrix[1, 2]) / scale)
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = ((matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale,
                          (matrix[1, 2] + matrix[2, 1]) / scale,
                          (matrix[0, 2] - matrix[2, 0]) / scale)
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = ((matrix[0, 2] + matrix[2, 0]) / scale,
                          (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale,
                          (matrix[1, 0] - matrix[0, 1]) / scale)
    result = np.asarray(quaternion, dtype=float)
    result /= np.linalg.norm(result)
    if result[3] < 0.0:
        result *= -1.0
    return tuple(float(item) for item in result)


def transform_matrix(translation: Iterable[float], quaternion_xyzw: Iterable[float]) -> np.ndarray:
    xyz = np.asarray(tuple(translation), dtype=float)
    if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
        raise ValueError("translation must contain three finite values")
    result = np.eye(4, dtype=float)
    result[:3, :3] = quaternion_xyzw_to_matrix(quaternion_xyzw)
    result[:3, 3] = xyz
    return result


def camera_pose_to_tool_pose(
    base_from_camera: np.ndarray, tool_from_camera: np.ndarray
) -> np.ndarray:
    """Convert an optical-camera target into the corresponding tool0 target."""

    base_from_camera = validate_transform(base_from_camera, "base_from_camera")
    tool_from_camera = validate_transform(tool_from_camera, "tool_from_camera")
    return base_from_camera @ np.linalg.inv(tool_from_camera)


def fixed_joint_transform_from_urdf(model, parent_frame: str, child_frame: str) -> np.ndarray:
    """Read a direct fixed joint's parent-to-child transform from a URDF."""
    joints = [joint for joint in model.findall("joint")
              if joint.find("child") is not None
              and joint.find("child").get("link") == child_frame]
    if len(joints) != 1 or joints[0].get("type") != "fixed":
        raise ValueError(f"URDF needs one fixed joint to {child_frame}")
    joint = joints[0]
    parent = joint.find("parent")
    if parent is None or parent.get("link") != parent_frame:
        raise ValueError(f"URDF joint to {child_frame} is not attached to {parent_frame}")
    origin = joint.find("origin")
    if origin is None:
        raise ValueError(f"URDF joint to {child_frame} has no origin")
    try:
        xyz = [float(value) for value in origin.get("xyz", "").split()]
        roll, pitch, yaw = [float(value) for value in origin.get("rpy", "").split()]
    except (TypeError, ValueError) as error:
        raise ValueError(f"URDF joint to {child_frame} has invalid XYZ/RPY") from error
    if len(xyz) != 3 or not np.isfinite([*xyz, roll, pitch, yaw]).all():
        raise ValueError(f"URDF joint to {child_frame} has invalid XYZ/RPY")
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    result = np.eye(4)
    result[:3, :3] = np.asarray((
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    ))
    result[:3, 3] = xyz
    return result


def camera_mount_from_urdf(model, tool_frame: str, camera_frame: str) -> np.ndarray:
    """Read a direct fixed wrist-camera joint and reject a mismatched URDF."""
    joints = [joint for joint in model.findall("joint")
              if joint.find("child") is not None
              and joint.find("child").get("link") == camera_frame]
    if len(joints) != 1 or joints[0].get("type") != "fixed":
        raise ValueError("MoveIt URDF needs one fixed calibrated camera joint")
    joint = joints[0]
    parent = joint.find("parent")
    if parent is None or parent.get("link") != tool_frame:
        raise ValueError("MoveIt camera joint is not attached directly to the configured tool")
    origin = joint.find("origin")
    if origin is None:
        raise ValueError("MoveIt camera joint has no origin")
    try:
        xyz = [float(value) for value in origin.get("xyz", "").split()]
        roll, pitch, yaw = [float(value) for value in origin.get("rpy", "").split()]
    except (TypeError, ValueError) as error:
        raise ValueError("MoveIt camera joint has invalid XYZ/RPY") from error
    if len(xyz) != 3 or not np.isfinite([*xyz, roll, pitch, yaw]).all():
        raise ValueError("MoveIt camera joint has invalid XYZ/RPY")
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    result = np.eye(4)
    result[:3, :3] = np.asarray((
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    ))
    result[:3, 3] = xyz
    return result


def validate_transform(value: np.ndarray, name: str = "transform") -> np.ndarray:
    transform = np.asarray(value, dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8):
        raise ValueError(f"{name} must have homogeneous last row [0, 0, 0, 1]")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-6
    ):
        raise ValueError(f"{name} must contain a proper rotation")
    return transform.copy()
