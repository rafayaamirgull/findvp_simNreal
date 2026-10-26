"""Base provenance for stationary external targets or targets on Steve's top deck."""

from __future__ import annotations

import numpy as np


def robot_mounted_reference(planning_frame: str) -> dict:
    """The target is rigidly fixed relative to the arm base throughout the run."""
    return {
        "mode": "robot_mounted",
        "frame": planning_frame,
        "child_frame": planning_frame,
        "odom_from_base": np.eye(4).tolist(),
    }


def require_reference_mode(reference: dict, mode: str, planning_frame: str) -> None:
    if not isinstance(reference, dict):
        raise ValueError("localization lacks base reference")
    actual = reference.get("mode", "odom")
    if actual != mode:
        raise ValueError(f"localization base mode {actual!r} differs from configured {mode!r}")
    if mode == "robot_mounted" and reference != robot_mounted_reference(planning_frame):
        raise ValueError("invalid robot-mounted base reference")
