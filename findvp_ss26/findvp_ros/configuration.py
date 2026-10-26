"""Validated deployment configuration for Steve and other ROS 2 robots."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .geometry import transform_matrix, validate_transform


@dataclass(frozen=True)
class RobotConfiguration:
    planning_frame: str
    tool_frame: str
    camera_frame: str
    move_group: str
    joint_names: tuple[str, ...]
    image_topic: str
    camera_info_topic: str
    joint_states_topic: str
    move_action: str
    controller_name: str
    tool_from_camera: np.ndarray
    odom_topic: str = "/odom"
    image_frame: str = "d415_calib_color_optical_frame"
    base_tracking_mode: str = "odom"


@dataclass(frozen=True)
class NamedJointPose:
    name: str
    joint_positions: tuple[float, ...]
    source: Path


def _required_string(document: dict[str, Any], key: str) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"configuration field {key!r} must be a nonempty string")
    return value.strip()


def load_robot_configuration(path: Path) -> RobotConfiguration:
    source = Path(path).expanduser().resolve()
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("robot configuration must be a YAML mapping")
    joints = document.get("joint_names")
    if not isinstance(joints, list) or len(joints) != 6 or any(
        not isinstance(item, str) or not item.strip() for item in joints
    ):
        raise ValueError("joint_names must contain exactly six nonempty names")
    if len(set(joints)) != len(joints):
        raise ValueError("joint_names must be unique")

    handeye = document.get("tool_from_camera")
    if not isinstance(handeye, dict):
        raise ValueError("tool_from_camera must be a mapping")
    tool_from_camera = transform_matrix(
        handeye.get("translation_m", ()), handeye.get("quaternion_xyzw", ())
    )
    base_tracking_mode = document.get("base_tracking_mode", "odom")
    if base_tracking_mode not in ("odom", "robot_mounted"):
        raise ValueError("base_tracking_mode must be 'odom' or 'robot_mounted'")
    return RobotConfiguration(
        planning_frame=_required_string(document, "planning_frame"),
        tool_frame=_required_string(document, "tool_frame"),
        camera_frame=_required_string(document, "camera_frame"),
        move_group=_required_string(document, "move_group"),
        joint_names=tuple(item.strip() for item in joints),
        image_topic=_required_string(document, "image_topic"),
        camera_info_topic=_required_string(document, "camera_info_topic"),
        joint_states_topic=_required_string(document, "joint_states_topic"),
        move_action=_required_string(document, "move_action"),
        controller_name=_required_string(document, "controller_name"),
        tool_from_camera=validate_transform(tool_from_camera, "tool_from_camera"),
        odom_topic=document.get("odom_topic", "/odom"),
        image_frame=_required_string(document, "image_frame"),
        base_tracking_mode=base_tracking_mode,
    )


def load_handeye_json(path: Path) -> np.ndarray:
    document = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    return transform_matrix(document["translation_m"], document["quaternion_xyzw"])


def load_named_joint_pose(
    path: Path, expected_joint_names: tuple[str, ...]
) -> NamedJointPose:
    """Load an operator-recorded pose and reject ambiguous joint ordering."""
    source = Path(path).expanduser().resolve()
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("start pose must be a YAML mapping")
    name = document.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("start pose name must be a nonempty string")
    positions = document.get("joint_positions_rad")
    if not isinstance(positions, dict):
        raise ValueError("start pose joint_positions_rad must be a joint-name mapping")
    expected = set(expected_joint_names)
    actual = set(positions)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"start pose joint names differ; missing={missing}, extra={extra}")
    ordered = np.asarray([positions[joint] for joint in expected_joint_names], dtype=float)
    if ordered.shape != (len(expected_joint_names),) or not np.isfinite(ordered).all():
        raise ValueError("start pose joint positions must be finite numbers")
    return NamedJointPose(name.strip(), tuple(float(value) for value in ordered), source)
