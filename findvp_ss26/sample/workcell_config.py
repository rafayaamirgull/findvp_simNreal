"""Side-effect-free constants and transforms for the tabletop UR5e workcell."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
STEVE_DESCRIPTION_ROOT = REPO_ROOT / "assets" / "steve_description"

DEFAULT_ROBOT_URDF = STEVE_DESCRIPTION_ROOT / "urdf" / "steve_ur5e.urdf"
WORKPIECE_MODEL_DIR = REPO_ROOT / "tasks" / "3d_models"
DEFAULT_WORKPIECE_NAME = "obj_000020"
DEFAULT_WORKPIECE_PLY = WORKPIECE_MODEL_DIR / f"{DEFAULT_WORKPIECE_NAME}.ply"
GENERATED_ASSET_DIR = REPO_ROOT / "artifacts" / "pybullet"
OUTPUT_DIR = REPO_ROOT / "output" / "pybullet"
DEFAULT_CAPABILITY_MAP = REPO_ROOT / "output" / "ur5e_capability_map.npz"

DEFAULT_WORKPIECE_XY = (0.45, -0.10)
TABLE_TOP_Z = 0.75
# Cover the complete visible work area so every wrist-camera view has the same
# tabletop background instead of seeing the ground plane past a table edge.
TABLE_SIZE_XYZ = (100.0, 100.0, 0.05)
TABLE_CENTER_XYZ = (0.25, 0.0, TABLE_TOP_Z - TABLE_SIZE_XYZ[2] * 0.5)
ROBOT_BASE_XYZ = (0.0, 0.0, TABLE_TOP_Z)
WORKPIECE_BASE_XYZ = (DEFAULT_WORKPIECE_XY[0], DEFAULT_WORKPIECE_XY[1], TABLE_TOP_Z)
TABLE_UNDERSIDE_Z = TABLE_TOP_Z - TABLE_SIZE_XYZ[2]
TABLE_LEG_HALF_EXTENTS = (0.035, 0.035, TABLE_UNDERSIDE_Z * 0.5)
TABLE_LEG_POSITIONS = (
    (-0.0175, -0.13475, TABLE_LEG_HALF_EXTENTS[2]),
    (-0.0175, 0.13475, TABLE_LEG_HALF_EXTENTS[2]),
    (0.5175, -0.13475, TABLE_LEG_HALF_EXTENTS[2]),
    (0.5175, 0.13475, TABLE_LEG_HALF_EXTENTS[2]),
)

HOME_JOINT_POSITIONS_RAD = {
    "ur5e_shoulder_pan_joint": 0.0,
    "ur5e_shoulder_lift_joint": -1.57,
    "ur5e_elbow_joint": 1.57,
    "ur5e_wrist_1_joint": -1.57,
    "ur5e_wrist_2_joint": -1.57,
    "ur5e_wrist_3_joint": 0.0,
}

ARM_JOINT_ORDER = [
    "ur5e_shoulder_pan_joint",
    "ur5e_shoulder_lift_joint",
    "ur5e_elbow_joint",
    "ur5e_wrist_1_joint",
    "ur5e_wrist_2_joint",
    "ur5e_wrist_3_joint",
]

KEPT_LINK_NAMES = {
    "world",
    "ur5e_base_link",
    "ur5e_base_link_inertia",
    "ur5e_shoulder_link",
    "ur5e_upper_arm_link",
    "ur5e_forearm_link",
    "ur5e_wrist_1_link",
    "ur5e_wrist_2_link",
    "ur5e_wrist_3_link",
    "ur5e_ft_frame",
    "ur5e_base",
    "ur5e_flange",
    "ur5e_tool0",
}

KEPT_JOINT_NAMES = {
    "ur5e_base_joint",
    "ur5e_base_link-base_link_inertia",
    "ur5e_shoulder_pan_joint",
    "ur5e_shoulder_lift_joint",
    "ur5e_elbow_joint",
    "ur5e_wrist_1_joint",
    "ur5e_wrist_2_joint",
    "ur5e_wrist_3_joint",
    "ur5e_wrist_3_link-ft_frame",
    "ur5e_base_link-base_fixed_joint",
    "ur5e_wrist_3-flange",
    "ur5e_flange-tool0",
}

CAMERA_PARENT_LINK_NAME = "ur5e_wrist_3_link"
END_EFFECTOR_LINK_NAME = "ur5e_tool0"

CAMERA_BODY_HALF_EXTENTS = (0.021, 0.021, 0.0115)
CAMERA_RGBA = (0.15, 0.15, 0.17, 1.0)

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FOCAL_LENGTH_MM = 2.0
CAMERA_VERTICAL_APERTURE_MM = 2.453
CAMERA_NEAR = 0.02
CAMERA_FAR = 3.0
COMMAND_BLEND = 0.18

ROBOT_TABLE_ALLOWED_CONTACT_LINKS = {
    "ur5e_base_link",
    "ur5e_base_link_inertia",
    "ur5e_base",
}
CAMERA_ROBOT_ALLOWED_CONTACT_LINKS = {CAMERA_PARENT_LINK_NAME}


def compose_transform(
    translation_xyz: tuple[float, float, float],
    rotation_deg_xyz: tuple[float, float, float],
) -> np.ndarray:
    """Build ``T_parent_child`` from XYZ translation and fixed-axis XYZ Euler angles."""

    roll, pitch, yaw = (math.radians(value) for value in rotation_deg_xyz)
    cos_x, sin_x = math.cos(roll), math.sin(roll)
    cos_y, sin_y = math.cos(pitch), math.sin(pitch)
    cos_z, sin_z = math.cos(yaw), math.sin(yaw)
    rotation_x = np.asarray(
        ((1.0, 0.0, 0.0), (0.0, cos_x, -sin_x), (0.0, sin_x, cos_x))
    )
    rotation_y = np.asarray(
        ((cos_y, 0.0, sin_y), (0.0, 1.0, 0.0), (-sin_y, 0.0, cos_y))
    )
    rotation_z = np.asarray(
        ((cos_z, -sin_z, 0.0), (sin_z, cos_z, 0.0), (0.0, 0.0, 1.0))
    )
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation_z @ rotation_y @ rotation_x
    transform[:3, 3] = np.asarray(translation_xyz, dtype=float)
    return transform


def compute_camera_vertical_fov_deg() -> float:
    return math.degrees(
        2.0 * math.atan(CAMERA_VERTICAL_APERTURE_MM / (2.0 * CAMERA_FOCAL_LENGTH_MM))
    )


CAMERA_OPTICAL_FROM_PARENT = (
    compose_transform((0.0, 0.0, -0.03), (0.0, 0.0, 90.0))
    @ compose_transform((0.08, 0.0, 0.05), (180.0, -101.46, 0.0))
    @ compose_transform((0.01085, 0.009, 0.021), (-90.0, 0.0, -90.0))
)

# The visible housing and collision proxy share one frame behind the optical
# center. Keeping both public names preserves the report/calibration contract.
CAMERA_BODY_FROM_PARENT = CAMERA_OPTICAL_FROM_PARENT @ compose_transform(
    (0.0, 0.0, -CAMERA_BODY_HALF_EXTENTS[2]), (0.0, 0.0, 0.0)
)
CAMERA_COLLISION_FROM_PARENT = CAMERA_BODY_FROM_PARENT.copy()
CAMERA_COLLISION_FROM_BODY = np.eye(4, dtype=float)


def resolve_workpiece_ply(
    object_name: str | None = None,
    explicit_path: Path | None = None,
) -> Path:
    """Resolve a bundled model name, or preserve an explicit PLY path override."""

    if object_name is None:
        return (
            explicit_path.expanduser()
            if explicit_path is not None
            else DEFAULT_WORKPIECE_PLY
        )

    cleaned_name = object_name.strip()
    if not cleaned_name:
        raise ValueError("--object must not be empty.")
    if "/" in cleaned_name or "\\" in cleaned_name or cleaned_name in (".", ".."):
        raise ValueError(
            "--object accepts a model name from tasks/3d_models, not a path. "
            "Use --workpiece-ply for an arbitrary path."
        )

    filename = (
        f"{cleaned_name[:-4]}.ply"
        if cleaned_name.lower().endswith(".ply")
        else f"{cleaned_name}.ply"
    )
    selected_path = WORKPIECE_MODEL_DIR / filename
    if selected_path.is_file():
        return selected_path

    available_names = sorted(path.stem for path in WORKPIECE_MODEL_DIR.glob("*.ply"))
    available = ", ".join(available_names) or "(none)"
    raise FileNotFoundError(
        f"Unknown workpiece object {cleaned_name!r}. Available objects: {available}"
    )
