#!/usr/bin/env python3
"""
PyBullet tabletop UR5e environment for eye-in-hand view planning.

What this builds:
- UR5e arm loaded from the local `steve_ur5e.urdf`
- fixed-base robot mounted on a tabletop
- sample workpiece placed on the same table
- monocular RGB wrist camera rendered from the robot end effector

What it does not try to solve yet:
- grasping
- ROS 2 integration

Control model:
- the arm is driven kinematically for stable viewport/camera control
- this avoids the unstable dynamic behavior of the simplified URDF in PyBullet
- the workpiece and table still exist in the physics world
- gravity is disabled because this scene is intended for viewpoint simulation, not contact dynamics

Typical usage:
  python3 sample/pybullet_tabletop_ur5e_eye_in_hand.py

Keyboard shortcuts in GUI mode:
- `c`: save the latest wrist-camera RGB frame
- `h`: return the robot to its home pose
- `r`: reset the scene camera
- `q`: quit

Scene camera controls in GUI mode:
- left-drag: orbit
- right-drag or middle-drag: pan
- mouse wheel: zoom

When viewpoint planning is enabled, click a displayed viewpoint and use the
`Move to selected viewpoint` button in the Params panel. Green points have a
validated IK endpoint; red points are reported as unreachable and cannot be
commanded. CuRobo plans the motion and PyBullet revalidates the dense path
before it is played in the simulation.

The PyBullet Params panel also contains a `Save wrist-camera image...` button
that opens a file picker for choosing the output directory and filename.
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image, PngImagePlugin

if __package__:
    from .curobo_motion import (
        DEFAULT_MOTION_INTERPOLATION_DT_S,
        DEFAULT_MOTION_MAX_ATTEMPTS,
        CuRoboMotionInitializationError,
        CuRoboNoMotionPlanError,
        CuRoboMotionPlan,
        CuRoboMotionPlanner,
        CuRoboMotionPlanningError,
    )
    from .curobo_reachability import (
        CUROBO_API_V1,
        CuRoboRuntimeProbe,
        CuRoboIKCandidateBackend,
        CuRoboUnavailableError,
        build_curobo_scene_model,
        detect_curobo_robot_config_api,
        require_curobo_runtime,
    )
    from .viewpoint_reachability import (
        DEFAULT_COLLISION_CLEARANCE_M,
        DEFAULT_IK_ORIENTATION_TOLERANCE_DEG,
        DEFAULT_IK_POSITION_TOLERANCE_M,
        DEFAULT_IK_REFINEMENT_STEPS,
        DEFAULT_IK_ROLLS_DEG,
        DEFAULT_IK_SEED_COUNT,
        IKCandidate,
        IKCandidateBackend,
        IKGoal,
        IKProblem,
        IKProblemResult,
        ReachabilityConfig,
        ViewpointCandidate,
        matrix_to_quaternion_xyzw as matrix_to_quat_xyzw,
        quaternion_angular_distance_deg,
        quaternion_xyzw_to_matrix as quat_xyzw_to_matrix,
        sample_fibonacci_viewpoints,
    )
    from .workcell_config import (
        ARM_JOINT_ORDER,
        CAMERA_BODY_FROM_PARENT,
        CAMERA_BODY_HALF_EXTENTS,
        CAMERA_COLLISION_FROM_BODY,
        CAMERA_COLLISION_FROM_PARENT,
        CAMERA_FAR,
        CAMERA_HEIGHT,
        CAMERA_NEAR,
        CAMERA_OPTICAL_FROM_PARENT,
        CAMERA_PARENT_LINK_NAME,
        CAMERA_RGBA,
        CAMERA_ROBOT_ALLOWED_CONTACT_LINKS,
        CAMERA_WIDTH,
        COMMAND_BLEND,
        DEFAULT_CAPABILITY_MAP,
        DEFAULT_ROBOT_URDF,
        DEFAULT_WORKPIECE_NAME,
        DEFAULT_WORKPIECE_PLY,
        DEFAULT_WORKPIECE_XY,
        END_EFFECTOR_LINK_NAME,
        GENERATED_ASSET_DIR,
        HOME_JOINT_POSITIONS_RAD,
        KEPT_JOINT_NAMES,
        KEPT_LINK_NAMES,
        OUTPUT_DIR,
        ROBOT_BASE_XYZ,
        ROBOT_TABLE_ALLOWED_CONTACT_LINKS,
        STEVE_DESCRIPTION_ROOT,
        TABLE_CENTER_XYZ,
        TABLE_LEG_HALF_EXTENTS,
        TABLE_LEG_POSITIONS,
        TABLE_SIZE_XYZ,
        TABLE_TOP_Z,
        WORKPIECE_BASE_XYZ,
        compute_camera_vertical_fov_deg,
        resolve_workpiece_ply,
    )
else:  # Direct execution: python3 sample/<script>.py
    from curobo_motion import (
        DEFAULT_MOTION_INTERPOLATION_DT_S,
        DEFAULT_MOTION_MAX_ATTEMPTS,
        CuRoboMotionInitializationError,
        CuRoboNoMotionPlanError,
        CuRoboMotionPlan,
        CuRoboMotionPlanner,
        CuRoboMotionPlanningError,
    )
    from curobo_reachability import (
        CUROBO_API_V1,
        CuRoboRuntimeProbe,
        CuRoboIKCandidateBackend,
        CuRoboUnavailableError,
        build_curobo_scene_model,
        detect_curobo_robot_config_api,
        require_curobo_runtime,
    )
    from viewpoint_reachability import (
        DEFAULT_COLLISION_CLEARANCE_M,
        DEFAULT_IK_ORIENTATION_TOLERANCE_DEG,
        DEFAULT_IK_POSITION_TOLERANCE_M,
        DEFAULT_IK_REFINEMENT_STEPS,
        DEFAULT_IK_ROLLS_DEG,
        DEFAULT_IK_SEED_COUNT,
        IKCandidate,
        IKCandidateBackend,
        IKGoal,
        IKProblem,
        IKProblemResult,
        ReachabilityConfig,
        ViewpointCandidate,
        matrix_to_quaternion_xyzw as matrix_to_quat_xyzw,
        quaternion_angular_distance_deg,
        quaternion_xyzw_to_matrix as quat_xyzw_to_matrix,
        sample_fibonacci_viewpoints,
    )
    from workcell_config import (
        ARM_JOINT_ORDER,
        CAMERA_BODY_FROM_PARENT,
        CAMERA_BODY_HALF_EXTENTS,
        CAMERA_COLLISION_FROM_BODY,
        CAMERA_COLLISION_FROM_PARENT,
        CAMERA_FAR,
        CAMERA_HEIGHT,
        CAMERA_NEAR,
        CAMERA_OPTICAL_FROM_PARENT,
        CAMERA_PARENT_LINK_NAME,
        CAMERA_RGBA,
        CAMERA_ROBOT_ALLOWED_CONTACT_LINKS,
        CAMERA_WIDTH,
        COMMAND_BLEND,
        DEFAULT_CAPABILITY_MAP,
        DEFAULT_ROBOT_URDF,
        DEFAULT_WORKPIECE_NAME,
        DEFAULT_WORKPIECE_PLY,
        DEFAULT_WORKPIECE_XY,
        END_EFFECTOR_LINK_NAME,
        GENERATED_ASSET_DIR,
        HOME_JOINT_POSITIONS_RAD,
        KEPT_JOINT_NAMES,
        KEPT_LINK_NAMES,
        OUTPUT_DIR,
        ROBOT_BASE_XYZ,
        ROBOT_TABLE_ALLOWED_CONTACT_LINKS,
        STEVE_DESCRIPTION_ROOT,
        TABLE_CENTER_XYZ,
        TABLE_LEG_HALF_EXTENTS,
        TABLE_LEG_POSITIONS,
        TABLE_SIZE_XYZ,
        TABLE_TOP_Z,
        WORKPIECE_BASE_XYZ,
        compute_camera_vertical_fov_deg,
        resolve_workpiece_ply,
    )

try:
    import pybullet as p
    import pybullet_data
    from pybullet_utils.bullet_client import BulletClient
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit(
        "PyBullet is not installed.\n"
        "Install it with `python3 -m pip install pybullet` or rebuild the Docker image "
        "after adding the PyBullet dependency."
    ) from exc


DEFAULT_VIEWPOINT_COUNT = 100
DEFAULT_VIEW_RADIUS_MIN_M = 0.30
DEFAULT_VIEW_RADIUS_MAX_M = 0.50
DEFAULT_VIEW_ELEVATION_MIN_DEG = 12.0
DEFAULT_VIEW_ELEVATION_MAX_DEG = 82.0
DEFAULT_FRAMING_MARGIN_FRACTION = 0.10
DEFAULT_CUROBO_IK_SEED_COUNT = 32
DEFAULT_SCENE_CAMERA_DISTANCE_M = 1.8
DEFAULT_SCENE_CAMERA_YAW_DEG = 45.0
DEFAULT_SCENE_CAMERA_PITCH_DEG = -35.0
DEFAULT_SCENE_CAMERA_TARGET = (0.25, -0.05, 0.80)
DEFAULT_CUROBO_V1_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "curobo"
    / "ur5e_camera_curobo_v1.yml"
)
DEFAULT_VIEWPOINT_OUTPUT = OUTPUT_DIR / "reachable_viewpoints.json"
DEFAULT_CAPABILITY_SAMPLE_COUNT = 100_000
DEFAULT_CAPABILITY_VOXEL_M = 0.05
DEFAULT_CAPABILITY_TCP_Z_RANGE_M = (0.02, 0.80)


@dataclass(frozen=True)
class IKBackendSelection:
    """Resolved candidate generator and auditable automatic-fallback details."""

    requested_backend: str
    effective_backend: str
    policy: str
    curobo_config_path: Path | None = None
    fallback_reason: str | None = None
    runtime_probe: CuRoboRuntimeProbe | None = None

    @property
    def uses_curobo(self) -> bool:
        return self.effective_backend == "curobo"

    def report_metadata(self) -> dict[str, object]:
        metadata: dict[str, object] = {
            "requested_backend": self.requested_backend,
            "effective_backend": self.effective_backend,
            "policy": self.policy,
            "automatic_fallback_used": self.fallback_reason is not None,
        }
        if self.curobo_config_path is not None:
            metadata["curobo_config_path"] = self.curobo_config_path.as_posix()
        if self.fallback_reason is not None:
            metadata["fallback_reason"] = self.fallback_reason
        if self.runtime_probe is not None:
            metadata["curobo_runtime"] = {
                "api_generation": self.runtime_probe.api_generation,
                "curobo_version": self.runtime_probe.curobo_version,
                "python_version": self.runtime_probe.python_version,
                "nvidia_driver_version": self.runtime_probe.nvidia_driver_version,
            }
        return metadata


def resolve_ik_backend(
    requested_backend: str,
    curobo_robot_config: Path | None,
    *,
    message_sink=print,
) -> IKBackendSelection:
    """Resolve the CLI policy before creating or moving the PyBullet robot.

    ``auto`` is deliberately limited to the pinned V1 adapter.  Only runtime
    capability failures trigger its CPU fallback. Configuration and solver
    failures after a successful preflight remain visible to the caller.
    """

    if requested_backend == "pybullet":
        return IKBackendSelection(
            requested_backend=requested_backend,
            effective_backend="pybullet",
            policy="forced_pybullet",
        )
    if requested_backend not in ("auto", "curobo"):
        raise ValueError(f"Unknown IK backend policy: {requested_backend!r}")

    is_automatic = requested_backend == "auto"
    if not is_automatic and curobo_robot_config is None:
        raise ValueError(
            "--curobo-robot-config is required when --ik-backend=curobo. "
            "Generate it with sample/prepare_curobo_robot.py first."
        )
    config_path = Path(
        curobo_robot_config
        if curobo_robot_config is not None
        else DEFAULT_CUROBO_V1_CONFIG
    ).resolve()
    if is_automatic:
        try:
            runtime_probe = require_curobo_runtime(expected_api=CUROBO_API_V1)
        except CuRoboUnavailableError as exc:
            # Fall back only for an unavailable package/driver/Torch/CUDA/GPU.
            # Configuration and solver defects remain hard errors once V1 is usable.
            fallback_reason = str(exc)
            message_sink(
                "CuRobo V1 is unavailable; falling back to PyBullet multi-start IK.\n"
                f"Reason: {fallback_reason}"
            )
            return IKBackendSelection(
                requested_backend=requested_backend,
                effective_backend="pybullet",
                policy="prefer_curobo_v1_then_pybullet",
                curobo_config_path=config_path,
                fallback_reason=fallback_reason,
            )
    if not config_path.exists():
        if is_automatic:
            raise FileNotFoundError(
                f"CuRobo V1 robot config not found: {config_path}. "
                "Generate it with `python3 sample/prepare_curobo_robot.py "
                "--api-version v1`."
            )
        raise FileNotFoundError(f"CuRobo robot config not found: {config_path}")
    config_api = detect_curobo_robot_config_api(config_path)
    if is_automatic and config_api != CUROBO_API_V1:
        raise ValueError(
            "The automatic backend policy only prefers CuRobo V1, but "
            f"{config_path} uses the {config_api.upper()} schema."
        )
    if not is_automatic:
        runtime_probe = require_curobo_runtime(expected_api=config_api)

    if is_automatic:
        message_sink(
            "CuRobo V1 runtime and CUDA device are available; using GPU IK "
            "with PyBullet endpoint validation."
        )
    return IKBackendSelection(
        requested_backend=requested_backend,
        effective_backend="curobo",
        policy=("prefer_curobo_v1_then_pybullet" if is_automatic else "forced_curobo"),
        curobo_config_path=config_path,
        runtime_probe=runtime_probe,
    )


def make_transform(
    translation_xyz: tuple[float, float, float],
    rotation_quat_xyzw: tuple[float, float, float, float],
) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = quat_xyzw_to_matrix(rotation_quat_xyzw)
    transform[:3, 3] = np.asarray(translation_xyz, dtype=float)
    return transform


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < 1e-9:
        return vector.copy()
    return vector / norm


PLY_SCALAR_DTYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


@dataclass
class _PlyElement:
    name: str
    count: int
    properties: list[tuple[str, ...]]


@dataclass(frozen=True)
class _PlyTextureMapping:
    """Artec-style texture coordinates and their geometry-face assignments."""

    texture_vertex_ids: np.ndarray
    coordinates_uv: np.ndarray
    face_texture_ids: np.ndarray
    face_numbers: np.ndarray
    face_coordinate_indices: np.ndarray


def _triangulate_polygon(
    polygon_indices: list[int], triangles: list[list[int]]
) -> None:
    if len(polygon_indices) < 3:
        return
    for corner_index in range(1, len(polygon_indices) - 1):
        triangles.append(
            [
                polygon_indices[0],
                polygon_indices[corner_index],
                polygon_indices[corner_index + 1],
            ]
        )


def _parse_ply_header(
    handle: BinaryIO, ply_path: Path
) -> tuple[str, list[_PlyElement]]:
    if handle.readline().strip() != b"ply":
        raise ValueError(f"Unsupported PLY header in {ply_path}")

    format_name: str | None = None
    elements: list[_PlyElement] = []
    current_element: _PlyElement | None = None
    while True:
        raw_line = handle.readline()
        if not raw_line:
            raise ValueError(f"PLY header ended unexpectedly in {ply_path}")
        try:
            stripped = raw_line.decode("ascii").strip()
        except UnicodeDecodeError as exc:
            raise ValueError(f"PLY header is not ASCII in {ply_path}") from exc
        fields = stripped.split()
        if not fields or fields[0] in ("comment", "obj_info"):
            continue
        if fields[0] == "format" and len(fields) >= 2:
            format_name = fields[1]
        elif fields[0] == "element" and len(fields) == 3:
            current_element = _PlyElement(fields[1], int(fields[2]), [])
            elements.append(current_element)
        elif fields[0] == "property":
            if current_element is None:
                raise ValueError(f"PLY property precedes its element in {ply_path}")
            properties = current_element.properties
            if len(fields) == 5 and fields[1] == "list":
                properties.append(("list", fields[2], fields[3], fields[4]))
            elif len(fields) == 3:
                properties.append(("scalar", fields[1], fields[2]))
            else:
                raise ValueError(
                    f"Unsupported PLY property declaration in {ply_path}: {stripped}"
                )
        elif fields[0] == "end_header":
            break

    if format_name not in ("ascii", "binary_little_endian", "binary_big_endian"):
        raise ValueError(f"Unsupported PLY format {format_name!r} in {ply_path}")
    return format_name, elements


def _ply_numpy_dtype(type_name: str, byte_order: str, ply_path: Path) -> np.dtype:
    try:
        return np.dtype(byte_order + PLY_SCALAR_DTYPES[type_name])
    except KeyError as exc:
        raise ValueError(
            f"Unsupported PLY scalar type {type_name!r} in {ply_path}"
        ) from exc


def _read_binary_value(
    handle: BinaryIO, type_name: str, byte_order: str, ply_path: Path
) -> int | float:
    dtype = _ply_numpy_dtype(type_name, byte_order, ply_path)
    raw_value = handle.read(dtype.itemsize)
    if len(raw_value) != dtype.itemsize:
        raise ValueError(f"PLY data ended unexpectedly in {ply_path}")
    return np.frombuffer(raw_value, dtype=dtype, count=1)[0].item()


def _read_binary_faces(
    handle: BinaryIO,
    face_count: int,
    face_properties: list[tuple[str, ...]],
    byte_order: str,
    ply_path: Path,
) -> np.ndarray:
    vertex_property_names = {"vertex_indices", "vertex_index"}

    # Scanner exports in this repository contain a single uchar/int triangle
    # list per face. Reading those fixed records in one pass avoids millions of
    # tiny Python file reads for the larger meshes. Fall back to the general
    # polygon reader if a record is not triangular.
    if (
        len(face_properties) == 1
        and face_properties[0][0] == "list"
        and face_properties[0][3] in vertex_property_names
    ):
        _, count_type, index_type, _ = face_properties[0]
        count_dtype = _ply_numpy_dtype(count_type, byte_order, ply_path)
        index_dtype = _ply_numpy_dtype(index_type, byte_order, ply_path)
        if count_dtype.itemsize == 1:
            face_start = handle.tell()
            triangle_dtype = np.dtype(
                [("count", count_dtype), ("indices", index_dtype, (3,))]
            )
            raw_faces = handle.read(face_count * triangle_dtype.itemsize)
            if len(raw_faces) == face_count * triangle_dtype.itemsize:
                packed_faces = np.frombuffer(raw_faces, dtype=triangle_dtype)
                if np.all(packed_faces["count"] == 3):
                    return packed_faces["indices"].astype(np.int32, copy=True)
            handle.seek(face_start)

    triangles: list[list[int]] = []
    for _ in range(face_count):
        polygon_indices: list[int] | None = None
        for prop in face_properties:
            if prop[0] == "scalar":
                _read_binary_value(handle, prop[1], byte_order, ply_path)
                continue
            _, count_type, item_type, property_name = prop
            item_count = int(
                _read_binary_value(handle, count_type, byte_order, ply_path)
            )
            values = [
                int(_read_binary_value(handle, item_type, byte_order, ply_path))
                for _ in range(item_count)
            ]
            if property_name in vertex_property_names:
                polygon_indices = values
        if polygon_indices is None:
            raise ValueError(f"PLY face has no vertex index list in {ply_path}")
        _triangulate_polygon(polygon_indices, triangles)
    return np.asarray(triangles, dtype=np.int32).reshape(-1, 3)


def _read_ply_texture_mapping(
    handle: BinaryIO,
    format_name: str,
    extra_elements: list[_PlyElement],
    triangle_count: int,
    ply_path: Path,
) -> _PlyTextureMapping | None:
    """Read Artec ``multi_texture_*`` elements following PLY geometry."""

    if not extra_elements:
        return None
    element_names = [element.name for element in extra_elements]
    texture_names = {"multi_texture_vertex", "multi_texture_face"}
    present_texture_names = texture_names.intersection(element_names)
    if not present_texture_names:
        return None
    if present_texture_names != texture_names or element_names[:2] != [
        "multi_texture_vertex",
        "multi_texture_face",
    ]:
        raise ValueError(
            "PLY texture data must contain consecutive multi_texture_vertex "
            f"and multi_texture_face elements after geometry in {ply_path}"
        )

    texture_vertex_element, texture_face_element = extra_elements[:2]
    texture_vertex_properties = texture_vertex_element.properties
    texture_face_properties = texture_face_element.properties
    if any(prop[0] != "scalar" for prop in texture_vertex_properties):
        raise ValueError(f"PLY texture vertices must be scalar in {ply_path}")
    texture_vertex_names = [prop[2] for prop in texture_vertex_properties]
    try:
        texture_id_index = texture_vertex_names.index("tx")
        u_index = texture_vertex_names.index("u")
        v_index = texture_vertex_names.index("v")
    except ValueError as exc:
        raise ValueError(
            f"PLY texture vertices need tx, u, and v properties in {ply_path}"
        ) from exc

    texture_face_names = [prop[-1] for prop in texture_face_properties]
    try:
        face_texture_id_index = texture_face_names.index("tx")
        face_number_index = texture_face_names.index("tn")
        coordinate_list_index = texture_face_names.index("texture_vertex_indices")
    except ValueError as exc:
        raise ValueError(
            "PLY texture faces need tx, tn, and texture_vertex_indices "
            f"properties in {ply_path}"
        ) from exc
    if texture_face_properties[coordinate_list_index][0] != "list":
        raise ValueError(
            f"PLY texture_vertex_indices must be a list in {ply_path}"
        )

    texture_vertex_count = texture_vertex_element.count
    texture_face_count = texture_face_element.count
    if format_name == "ascii":
        texture_vertex_ids = np.empty(texture_vertex_count, dtype=np.int32)
        coordinates_uv = np.empty((texture_vertex_count, 2), dtype=np.float32)
        for row_index in range(texture_vertex_count):
            values = handle.readline().split()
            if len(values) < len(texture_vertex_properties):
                raise ValueError(
                    f"Texture vertex {row_index} in {ply_path} is incomplete"
                )
            texture_vertex_ids[row_index] = int(values[texture_id_index])
            coordinates_uv[row_index] = (
                float(values[u_index]),
                float(values[v_index]),
            )

        face_texture_ids = np.empty(texture_face_count, dtype=np.int32)
        face_numbers = np.empty(texture_face_count, dtype=np.int64)
        face_coordinate_indices = np.empty(
            (texture_face_count, 3), dtype=np.int32
        )
        for row_index in range(texture_face_count):
            values = handle.readline().split()
            cursor = 0
            scalar_values: dict[int, int] = {}
            coordinate_indices = None
            for property_index, prop in enumerate(texture_face_properties):
                if prop[0] == "scalar":
                    if cursor >= len(values):
                        raise ValueError(
                            f"Texture face {row_index} in {ply_path} is incomplete"
                        )
                    scalar_values[property_index] = int(values[cursor])
                    cursor += 1
                    continue
                if cursor >= len(values):
                    raise ValueError(
                        f"Texture face {row_index} in {ply_path} is incomplete"
                    )
                item_count = int(values[cursor])
                cursor += 1
                items = [
                    int(value) for value in values[cursor : cursor + item_count]
                ]
                cursor += item_count
                if property_index == coordinate_list_index:
                    coordinate_indices = items
            if coordinate_indices is None or len(coordinate_indices) != 3:
                raise ValueError(
                    f"Texture face {row_index} in {ply_path} is not triangular"
                )
            face_texture_ids[row_index] = scalar_values[face_texture_id_index]
            face_numbers[row_index] = scalar_values[face_number_index]
            face_coordinate_indices[row_index] = coordinate_indices
    else:
        byte_order = "<" if format_name == "binary_little_endian" else ">"
        texture_vertex_dtype = np.dtype(
            [
                (
                    f"property_{index}",
                    _ply_numpy_dtype(prop[1], byte_order, ply_path),
                )
                for index, prop in enumerate(texture_vertex_properties)
            ]
        )
        packed_texture_vertices = np.fromfile(
            handle,
            dtype=texture_vertex_dtype,
            count=texture_vertex_count,
        )
        if len(packed_texture_vertices) != texture_vertex_count:
            raise ValueError(f"PLY texture vertices ended unexpectedly in {ply_path}")
        texture_vertex_ids = packed_texture_vertices[
            f"property_{texture_id_index}"
        ].astype(np.int32)
        coordinates_uv = np.column_stack(
            (
                packed_texture_vertices[f"property_{u_index}"],
                packed_texture_vertices[f"property_{v_index}"],
            )
        ).astype(np.float32, copy=False)

        packed_fields: list[tuple] = []
        for property_index, prop in enumerate(texture_face_properties):
            if prop[0] == "scalar":
                packed_fields.append(
                    (
                        f"property_{property_index}",
                        _ply_numpy_dtype(prop[1], byte_order, ply_path),
                    )
                )
            else:
                packed_fields.extend(
                    (
                        (
                            f"property_{property_index}_count",
                            _ply_numpy_dtype(prop[1], byte_order, ply_path),
                        ),
                        (
                            f"property_{property_index}_values",
                            _ply_numpy_dtype(prop[2], byte_order, ply_path),
                            (3,),
                        ),
                    )
                )
        packed_texture_faces = np.fromfile(
            handle,
            dtype=np.dtype(packed_fields),
            count=texture_face_count,
        )
        if len(packed_texture_faces) != texture_face_count:
            raise ValueError(f"PLY texture faces ended unexpectedly in {ply_path}")
        list_counts = packed_texture_faces[
            f"property_{coordinate_list_index}_count"
        ]
        if not np.all(list_counts == 3):
            raise ValueError(f"PLY contains non-triangular texture faces: {ply_path}")
        face_texture_ids = packed_texture_faces[
            f"property_{face_texture_id_index}"
        ].astype(np.int32)
        face_numbers = packed_texture_faces[
            f"property_{face_number_index}"
        ].astype(np.int64)
        face_coordinate_indices = packed_texture_faces[
            f"property_{coordinate_list_index}_values"
        ].astype(np.int32)

    if texture_face_count != triangle_count:
        raise ValueError(
            f"PLY has {texture_face_count} texture faces for {triangle_count} "
            f"geometry triangles in {ply_path}"
        )
    if (
        face_numbers.size == 0
        or int(face_numbers.min()) < 0
        or int(face_numbers.max()) >= triangle_count
        or len(np.unique(face_numbers)) != triangle_count
    ):
        raise ValueError(f"PLY texture faces do not map every triangle in {ply_path}")
    if (
        face_coordinate_indices.size == 0
        or int(face_coordinate_indices.min()) < 0
        or int(face_coordinate_indices.max()) >= texture_vertex_count
    ):
        raise ValueError(f"PLY texture-coordinate index is out of range in {ply_path}")
    if not np.all(
        texture_vertex_ids[face_coordinate_indices]
        == face_texture_ids[:, None]
    ):
        raise ValueError(f"PLY texture IDs disagree across face UVs in {ply_path}")
    if not np.isfinite(coordinates_uv).all():
        raise ValueError(f"PLY contains non-finite texture coordinates in {ply_path}")
    return _PlyTextureMapping(
        texture_vertex_ids=texture_vertex_ids,
        coordinates_uv=coordinates_uv,
        face_texture_ids=face_texture_ids,
        face_numbers=face_numbers,
        face_coordinate_indices=face_coordinate_indices,
    )


def _load_ply_mesh_data(
    ply_path: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    _PlyTextureMapping | None,
]:
    """Load PLY geometry plus optional vertex colours or Artec texture UVs."""

    with ply_path.open("rb") as handle:
        format_name, elements = _parse_ply_header(handle, ply_path)
        if len(elements) < 2 or [element.name for element in elements[:2]] != [
            "vertex",
            "face",
        ]:
            raise ValueError(
                f"PLY must declare vertex then face elements first in {ply_path}"
            )

        vertex_element, face_element = elements[:2]
        vertex_count = vertex_element.count
        face_count = face_element.count
        vertex_properties = vertex_element.properties
        face_properties = face_element.properties
        if vertex_count <= 0:
            raise ValueError(f"PLY contains no vertices: {ply_path}")
        if any(prop[0] != "scalar" for prop in vertex_properties):
            raise ValueError(
                f"PLY vertex list properties are unsupported in {ply_path}"
            )

        property_names = [prop[2] for prop in vertex_properties]
        try:
            xyz_indices = tuple(
                property_names.index(axis) for axis in ("x", "y", "z")
            )
        except ValueError as exc:
            raise ValueError(
                f"PLY vertices need x, y, and z properties in {ply_path}"
            ) from exc
        try:
            rgb_indices: tuple[int, int, int] | None = tuple(
                property_names.index(channel) for channel in ("red", "green", "blue")
            )
        except ValueError:
            rgb_indices = None

        def normalize_rgb(values: np.ndarray) -> np.ndarray:
            colours = np.asarray(values, dtype=np.float32)
            if colours.size and float(colours.max()) <= 1.0:
                colours *= 255.0
            return np.clip(np.rint(colours), 0.0, 255.0).astype(np.uint8)

        if format_name == "ascii":
            vertices = np.empty((vertex_count, 3), dtype=np.float32)
            vertex_colours = (
                np.empty((vertex_count, 3), dtype=np.uint8)
                if rgb_indices is not None
                else None
            )
            for vertex_index in range(vertex_count):
                values = handle.readline().split()
                if len(values) < len(vertex_properties):
                    raise ValueError(
                        f"Vertex {vertex_index} in {ply_path} is incomplete"
                    )
                vertices[vertex_index] = tuple(
                    float(values[index]) for index in xyz_indices
                )
                if vertex_colours is not None and rgb_indices is not None:
                    vertex_colours[vertex_index] = normalize_rgb(
                        np.asarray([float(values[index]) for index in rgb_indices])
                    )

            triangles: list[list[int]] = []
            vertex_property_names = {"vertex_indices", "vertex_index"}
            for face_index in range(face_count):
                values = handle.readline().split()
                cursor = 0
                polygon_indices: list[int] | None = None
                for prop in face_properties:
                    if prop[0] == "scalar":
                        if cursor >= len(values):
                            raise ValueError(
                                f"Face {face_index} in {ply_path} is incomplete"
                            )
                        cursor += 1
                        continue
                    if cursor >= len(values):
                        raise ValueError(
                            f"Face {face_index} in {ply_path} is incomplete"
                        )
                    item_count = int(values[cursor])
                    cursor += 1
                    if item_count < 0 or cursor + item_count > len(values):
                        raise ValueError(
                            f"Face {face_index} in {ply_path} has an invalid list length"
                        )
                    items = [
                        int(value)
                        for value in values[cursor : cursor + item_count]
                    ]
                    cursor += item_count
                    if prop[3] in vertex_property_names:
                        polygon_indices = items
                if polygon_indices is None:
                    raise ValueError(
                        f"Face {face_index} has no vertex index list in {ply_path}"
                    )
                _triangulate_polygon(polygon_indices, triangles)
            triangle_array = np.asarray(triangles, dtype=np.int32).reshape(-1, 3)
            return (
                vertices,
                triangle_array,
                vertex_colours,
                _read_ply_texture_mapping(
                    handle,
                    format_name,
                    elements[2:],
                    len(triangle_array),
                    ply_path,
                ),
            )

        byte_order = "<" if format_name == "binary_little_endian" else ">"
        structured_dtype = np.dtype(
            [
                (f"property_{index}", _ply_numpy_dtype(prop[1], byte_order, ply_path))
                for index, prop in enumerate(vertex_properties)
            ]
        )
        packed_vertices = np.fromfile(handle, dtype=structured_dtype, count=vertex_count)
        if len(packed_vertices) != vertex_count:
            raise ValueError(f"PLY vertex data ended unexpectedly in {ply_path}")
        vertices = np.column_stack(
            [packed_vertices[f"property_{index}"] for index in xyz_indices]
        ).astype(np.float32, copy=False)
        vertex_colours = (
            normalize_rgb(
                np.column_stack(
                    [
                        packed_vertices[f"property_{index}"]
                        for index in rgb_indices
                    ]
                )
            )
            if rgb_indices is not None
            else None
        )
        triangles = _read_binary_faces(
            handle,
            face_count,
            face_properties,
            byte_order,
            ply_path,
        )
        return (
            vertices,
            triangles,
            vertex_colours,
            _read_ply_texture_mapping(
                handle,
                format_name,
                elements[2:],
                len(triangles),
                ply_path,
            ),
        )


def load_ply_mesh(
    ply_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Load PLY vertices, triangles, and optional per-vertex RGB colours."""

    vertices, triangles, vertex_colours, _ = _load_ply_mesh_data(ply_path)
    return vertices, triangles, vertex_colours


def load_ply_geometry(ply_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load geometry from an ASCII or endian-declared binary PLY mesh."""

    vertices, triangles, _ = load_ply_mesh(ply_path)
    return vertices, triangles


def load_ascii_ply_geometry(ply_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible alias for the PLY geometry loader."""

    return load_ply_geometry(ply_path)


PLY_TEXTURE_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")


def _find_ply_texture_images(
    ply_path: Path,
    texture_ids: np.ndarray,
) -> dict[int, Path]:
    """Resolve ``<ply-stem>_<texture-id>`` images next to a textured PLY."""

    images_by_stem: dict[str, list[Path]] = defaultdict(list)
    for candidate in ply_path.parent.iterdir():
        if (
            candidate.is_file()
            and candidate.suffix.lower() in PLY_TEXTURE_IMAGE_SUFFIXES
        ):
            images_by_stem[candidate.stem.casefold()].append(candidate)

    resolved: dict[int, Path] = {}
    missing: list[int] = []
    ply_stem = ply_path.stem.casefold()
    for texture_id in sorted(int(value) for value in np.unique(texture_ids)):
        expected_stems = [f"{ply_stem}_{texture_id}"]
        if texture_id == 0:
            expected_stems.append(ply_stem)
        matches = [
            path
            for expected_stem in expected_stems
            for path in images_by_stem.get(expected_stem, [])
        ]
        if not matches:
            missing.append(texture_id)
            continue
        resolved[texture_id] = sorted(
            matches,
            key=lambda path: (
                PLY_TEXTURE_IMAGE_SUFFIXES.index(path.suffix.lower()),
                path.name.casefold(),
            ),
        )[0]

    if missing:
        examples = ", ".join(
            f"{ply_path.stem}_{texture_id}.png" for texture_id in missing
        )
        raise FileNotFoundError(
            f"{ply_path} contains UV texture coordinates but its companion "
            f"texture image(s) are missing. Expected next to the PLY: {examples} "
            "(PNG, JPG, JPEG, or BMP). Artec PLY exports store the bitmap "
            "separately from the mesh."
        )
    return resolved


def _prepare_ply_texture_atlas(
    ply_path: Path,
    mapping: _PlyTextureMapping,
    output_dir: Path,
) -> tuple[Path, np.ndarray]:
    """Return one PyBullet texture and UVs, packing multiple bitmaps if needed."""

    texture_paths = _find_ply_texture_images(
        ply_path,
        mapping.texture_vertex_ids,
    )
    if len(texture_paths) == 1:
        # Artec stores image-space V from the top edge; OBJ uses the bottom.
        coordinates = mapping.coordinates_uv.astype(np.float64, copy=True)
        coordinates[:, 1] = 1.0 - coordinates[:, 1]
        return next(iter(texture_paths.values())), coordinates

    texture_images: dict[int, Image.Image] = {}
    for texture_id, texture_path in texture_paths.items():
        with Image.open(texture_path) as image:
            texture_images[texture_id] = image.convert("RGB").copy()
    texture_count = len(texture_images)
    columns = max(1, math.ceil(math.sqrt(texture_count)))
    rows = math.ceil(texture_count / columns)
    cell_width = max(image.width for image in texture_images.values())
    cell_height = max(image.height for image in texture_images.values())
    atlas_width = columns * cell_width
    atlas_height = rows * cell_height
    atlas = Image.new("RGB", (atlas_width, atlas_height), "white")
    atlas_coordinates = mapping.coordinates_uv.astype(np.float64, copy=True)
    for slot, texture_id in enumerate(sorted(texture_images)):
        image = texture_images[texture_id]
        row, column = divmod(slot, columns)
        left = column * cell_width
        top = row * cell_height
        atlas.paste(image, (left, top))
        selected = mapping.texture_vertex_ids == texture_id
        original_u = mapping.coordinates_uv[selected, 0]
        original_v = mapping.coordinates_uv[selected, 1]
        atlas_coordinates[selected, 0] = (
            left + original_u * image.width
        ) / atlas_width
        atlas_coordinates[selected, 1] = 1.0 - (
            top + original_v * image.height
        ) / atlas_height

    atlas_path = output_dir / f"{ply_path.stem}_pybullet_texture_atlas.png"
    atlas.save(atlas_path)
    return atlas_path, atlas_coordinates


def convert_ply_to_obj(
    ply_path: Path,
    output_dir: Path,
    unit_scale: float,
    rotation_z_deg: float,
) -> tuple[Path, Path | None, np.ndarray, np.ndarray, float]:
    """Create a PyBullet OBJ and optional texture preserving PLY colours."""

    (
        vertices,
        triangles,
        vertex_colours,
        texture_mapping,
    ) = _load_ply_mesh_data(ply_path)
    vertices_m = vertices.astype(np.float64) * float(unit_scale)

    min_bound = vertices_m.min(axis=0)
    max_bound = vertices_m.max(axis=0)
    extent = max_bound - min_bound

    centered_vertices = vertices_m.copy()
    centered_vertices[:, 0] -= 0.5 * (min_bound[0] + max_bound[0])
    centered_vertices[:, 1] -= 0.5 * (min_bound[1] + max_bound[1])
    centered_vertices[:, 2] -= min_bound[2]

    if abs(rotation_z_deg) > 1e-6:
        angle_rad = math.radians(rotation_z_deg)
        rotation = np.asarray(
            [
                [math.cos(angle_rad), -math.sin(angle_rad), 0.0],
                [math.sin(angle_rad), math.cos(angle_rad), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        centered_vertices = centered_vertices @ rotation.T
        rotated_min = centered_vertices.min(axis=0)
        centered_vertices[:, 2] -= rotated_min[2]
        extent = centered_vertices.max(axis=0) - centered_vertices.min(axis=0)

    vertex_centroid = centered_vertices.mean(axis=0)
    bounding_radius = float(
        np.linalg.norm(centered_vertices - vertex_centroid, axis=1).max()
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_obj = output_dir / f"{ply_path.stem}_pybullet.obj"
    texture_path: Path | None = None
    palette = None
    face_palette_indices = None
    texture_columns = 0
    texture_rows = 0
    texture_coordinates = None
    if texture_mapping is not None:
        texture_path, texture_coordinates = _prepare_ply_texture_atlas(
            ply_path,
            texture_mapping,
            output_dir,
        )
    elif vertex_colours is not None:
        # PyBullet ignores PLY and OBJ vertex colours, but reliably supports
        # OBJ UVs with an explicitly loaded PNG texture.  The source meshes are
        # dense, so one averaged colour per triangle remains visually smooth.
        face_colours = vertex_colours[triangles].astype(np.float32).mean(axis=1)
        quantized = np.clip(
            np.rint(face_colours / 16.0) * 16.0, 0.0, 255.0
        ).astype(np.uint8)
        palette, face_palette_indices = np.unique(
            quantized, axis=0, return_inverse=True
        )
        texture_columns = max(1, math.ceil(math.sqrt(len(palette))))
        texture_rows = math.ceil(len(palette) / texture_columns)
        cell_size = 8
        texture = np.full(
            (texture_rows * cell_size, texture_columns * cell_size, 3),
            255,
            dtype=np.uint8,
        )
        for palette_index, colour in enumerate(palette):
            row, column = divmod(palette_index, texture_columns)
            texture[
                row * cell_size : (row + 1) * cell_size,
                column * cell_size : (column + 1) * cell_size,
            ] = colour
        texture_path = output_dir / f"{ply_path.stem}_pybullet_palette.png"
        Image.fromarray(texture, mode="RGB").save(texture_path)

    with output_obj.open("w", encoding="utf-8") as handle:
        handle.write("# Generated from PLY for PyBullet; collision geometry and visual UVs\n")
        for vertex in centered_vertices:
            handle.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        if texture_mapping is not None and texture_coordinates is not None:
            for u, v in texture_coordinates:
                handle.write(f"vt {u:.8f} {v:.8f}\n")
            for face_number, coordinate_indices in zip(
                texture_mapping.face_numbers,
                texture_mapping.face_coordinate_indices,
            ):
                triangle = triangles[int(face_number)]
                handle.write(
                    f"f {triangle[0] + 1}/{coordinate_indices[0] + 1} "
                    f"{triangle[1] + 1}/{coordinate_indices[1] + 1} "
                    f"{triangle[2] + 1}/{coordinate_indices[2] + 1}\n"
                )
        elif palette is not None and face_palette_indices is not None:
            for palette_index in range(len(palette)):
                row, column = divmod(palette_index, texture_columns)
                u = (column + 0.5) / texture_columns
                v = 1.0 - (row + 0.5) / texture_rows
                handle.write(f"vt {u:.8f} {v:.8f}\n")
            for triangle, palette_index in zip(
                triangles, face_palette_indices
            ):
                texture_index = int(palette_index) + 1
                handle.write(
                    f"f {triangle[0] + 1}/{texture_index} "
                    f"{triangle[1] + 1}/{texture_index} "
                    f"{triangle[2] + 1}/{texture_index}\n"
                )
        else:
            for triangle in triangles:
                handle.write(
                    f"f {triangle[0] + 1} {triangle[1] + 1} {triangle[2] + 1}\n"
                )

    return output_obj, texture_path, extent, vertex_centroid, bounding_radius


def resolve_generated_asset_dir() -> Path:
    preferred_dir = GENERATED_ASSET_DIR
    try:
        preferred_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=preferred_dir, prefix=".write_test_"
        ):
            pass
        return preferred_dir
    except OSError:
        fallback_dir = Path(tempfile.gettempdir()) / "findvp_pybullet_assets"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        return fallback_dir


def rewrite_mesh_path(mesh_path: str, source_urdf_dir: Path) -> str:
    if mesh_path.startswith("package://steve_description/"):
        local_path = STEVE_DESCRIPTION_ROOT / mesh_path[len("package://steve_description/") :]
    else:
        local_path = Path(mesh_path)
        if not local_path.is_absolute():
            local_path = source_urdf_dir / local_path

    return local_path.resolve().as_posix()


def generate_minimal_ur5e_urdf(source_urdf: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_urdf = output_dir / "ur5e_pybullet_minimal.urdf"

    tree = ET.parse(source_urdf)
    root = tree.getroot()

    for child in list(root):
        name = child.attrib.get("name", "")
        if child.tag == "link":
            if name not in KEPT_LINK_NAMES:
                root.remove(child)
                continue
            for mesh in child.iter("mesh"):
                filename = mesh.attrib.get("filename")
                if filename:
                    mesh.attrib["filename"] = rewrite_mesh_path(filename, source_urdf.parent)
        elif child.tag == "joint":
            if name not in KEPT_JOINT_NAMES:
                root.remove(child)
        else:
            root.remove(child)

    tree.write(output_urdf, encoding="utf-8", xml_declaration=True)
    return output_urdf


@dataclass
class CameraCapture:
    """RGB-only wrist-camera capture."""

    rgb: np.ndarray


@dataclass(frozen=True)
class CapabilityMap:
    """Validated capability-map arrays loaded from the generator output."""

    workspace_xyz_m: np.ndarray
    reachability: np.ndarray
    manipulability: np.ndarray
    inverse_reachability: np.ndarray
    inverse_manipulability: np.ndarray


def load_capability_map(path: Path) -> CapabilityMap:
    """Load the five numeric arrays without allowing pickled data."""

    names = (
        "workspace_xyz_m",
        "reachability",
        "manipulability",
        "inverse_reachability",
        "inverse_manipulability",
    )
    with np.load(path, allow_pickle=False) as archive:
        missing = [name for name in names if name not in archive]
        if missing:
            raise ValueError(f"Capability map is missing arrays: {missing}")
        arrays = {name: np.asarray(archive[name], dtype=float) for name in names}

    workspace = arrays["workspace_xyz_m"]
    reachability = arrays["reachability"]
    manipulability = arrays["manipulability"]
    inverse = arrays["inverse_reachability"]
    inverse_manipulability = arrays["inverse_manipulability"]
    valid_shapes = (
        len(workspace) > 0
        and len(inverse) > 0
        and workspace.ndim == 2
        and workspace.shape[1] == 3
        and reachability.shape == (len(workspace),)
        and manipulability.shape == (len(workspace),)
        and inverse.ndim == 3
        and inverse.shape[1:] == (4, 4)
        and inverse_manipulability.shape == (len(inverse),)
    )
    if not valid_shapes or not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Capability-map arrays have invalid shapes or non-finite values.")
    return CapabilityMap(workspace, reachability, manipulability, inverse, inverse_manipulability)


def generate_capability_map(
    env: "UR5EyeInHandPyBulletEnv",
    samples: int = DEFAULT_CAPABILITY_SAMPLE_COUNT,
    voxel_m: float = DEFAULT_CAPABILITY_VOXEL_M,
    seed: int = 0,
) -> CapabilityMap:
    """Monte-Carlo sample the six arm joints using PyBullet FK and Jacobians."""

    if samples <= 0 or voxel_m <= 0.0:
        raise ValueError("Capability samples and voxel size must be positive.")
    rng = np.random.default_rng(seed)
    lower, upper = np.asarray(env.arm_joint_lower_limits), np.asarray(
        env.arm_joint_upper_limits
    )
    tcp_index = env.link_name_to_index[END_EFFECTOR_LINK_NAME]
    base_position, base_orientation = env.physics.getBasePositionAndOrientation(
        env.robot_id
    )
    base_from_world = np.linalg.inv(make_transform(base_position, base_orientation))
    entry_joints = env.get_joint_positions()
    voxels = defaultdict(lambda: [0, 0.0])
    tcp_poses, pose_manipulability = [], []

    try:
        for _ in range(samples):
            joints = rng.uniform(lower, upper)
            for joint_index, value in zip(env.arm_joint_indices, joints):
                env.physics.resetJointState(env.robot_id, joint_index, float(value))

            pose = base_from_world @ env.get_link_transform(END_EFFECTOR_LINK_NAME)
            if not (
                DEFAULT_CAPABILITY_TCP_Z_RANGE_M[0]
                <= pose[2, 3]
                <= DEFAULT_CAPABILITY_TCP_Z_RANGE_M[1]
            ):
                continue

            linear, angular = env.physics.calculateJacobian(
                env.robot_id,
                tcp_index,
                (0.0, 0.0, 0.0),
                joints.tolist(),
                [0.0] * len(joints),
                [0.0] * len(joints),
            )
            jacobian = np.vstack((linear, angular))[:, : len(ARM_JOINT_ORDER)]
            score = float(np.prod(np.linalg.svd(jacobian, compute_uv=False)))
            voxel = tuple(np.floor(pose[:3, 3] / voxel_m).astype(int))
            voxels[voxel][0] += 1
            voxels[voxel][1] += score
            tcp_poses.append(pose)
            pose_manipulability.append(score)
    finally:
        env._apply_joint_positions(entry_joints)

    if not voxels:
        raise RuntimeError("No TCP samples lie inside the tabletop height range.")
    keys, values = zip(*voxels.items())
    counts = np.asarray([value[0] for value in values])
    return CapabilityMap(
        workspace_xyz_m=(np.asarray(keys) + 0.5) * voxel_m,
        reachability=counts / counts.max(),
        manipulability=np.asarray([value[1] for value in values]) / counts,
        inverse_reachability=np.linalg.inv(np.asarray(tcp_poses)),
        inverse_manipulability=np.asarray(pose_manipulability),
    )


def save_capability_map(path: Path, capability_map: CapabilityMap) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        workspace_xyz_m=capability_map.workspace_xyz_m,
        reachability=capability_map.reachability,
        manipulability=capability_map.manipulability,
        inverse_reachability=capability_map.inverse_reachability,
        inverse_manipulability=capability_map.inverse_manipulability,
    )
    return path


@dataclass
class ReachableViewpoint:
    viewpoint: ViewpointCandidate
    joint_positions_rad: list[float]
    position_error_m: float
    orientation_error_deg: float
    clearance_m: float
    joint_limit_margin: float
    manipulability: float
    quality_score: float
    seed_index: int | None
    solver_provenance: dict[str, object] | None = None
    successful_solution_count: int = 1
    attempted_solutions: int = 1
    linear_path_collision_free: bool | None = None
    failure_counts: dict[str, int] | None = None


@dataclass
class RejectedViewpoint:
    viewpoint: ViewpointCandidate
    furthest_failure_stage: str
    attempted_solutions: int
    failure_counts: dict[str, int]


@dataclass
class ReachabilityReport:
    sampled: list[ViewpointCandidate]
    reachable: list[ReachableViewpoint]
    rejected: list[RejectedViewpoint]
    object_center_xyz: tuple[float, float, float]
    start_joint_positions_rad: list[float]
    elapsed_sec: float
    sampling_metadata: dict[str, object] | None = None
    solver_backend: str = "pybullet_multistart_ik"
    validation_backend: str = "pybullet_endpoint_validation"
    backend_metadata: dict[str, object] | None = None


@dataclass(frozen=True)
class TrajectoryValidationResult:
    """Authoritative PyBullet result for a dense CuRobo trajectory."""

    valid: bool
    samples_checked: int
    minimum_clearance_m: float
    failure_reason: str | None = None
    failure_segment_index: int | None = None


@dataclass(frozen=True)
class MotionPlanningWorkerResult:
    """Thread-to-GUI handoff for one CuRobo planning request."""

    viewpoint_index: int
    plan: CuRoboMotionPlan | None = None
    error: str | None = None
    failure_kind: str | None = None


@dataclass
class TrajectoryPlayer:
    """Wall-clock player for an already validated dense joint trajectory."""

    plan: CuRoboMotionPlan
    start_time: float

    def sample(self, now: float) -> tuple[list[float], bool]:
        positions = self.plan.joint_positions_rad
        if positions.ndim != 2 or positions.shape[0] < 2:
            raise ValueError("A playable trajectory requires at least two waypoints.")
        elapsed = max(0.0, float(now) - self.start_time)
        waypoint = elapsed / self.plan.interpolation_dt_s
        lower_index = min(int(math.floor(waypoint)), positions.shape[0] - 1)
        if lower_index >= positions.shape[0] - 1:
            return positions[-1].astype(float).tolist(), True
        fraction = waypoint - lower_index
        interpolated = (
            positions[lower_index]
            + fraction * (positions[lower_index + 1] - positions[lower_index])
        )
        return interpolated.astype(float).tolist(), False


class CameraPreview:
    def __init__(self, width: int, height: int) -> None:
        import matplotlib.pyplot as plt

        self._plt = plt
        self._plt.ion()
        self._figure, self._axes = self._plt.subplots(figsize=(width / 120.0, height / 120.0))
        self._artist = self._axes.imshow(np.zeros((height, width, 3), dtype=np.uint8))
        self._axes.set_title("UR5e Wrist Camera")
        self._axes.axis("off")
        self._figure.tight_layout()

    def update(self, rgb: np.ndarray) -> None:
        self._artist.set_data(rgb)
        self._figure.canvas.draw_idle()
        self._figure.canvas.flush_events()
        self._plt.pause(0.001)


class UR5EyeInHandPyBulletEnv:
    def __init__(
        self,
        robot_urdf: Path,
        workpiece_ply: Path,
        workpiece_unit_scale: float,
        workpiece_rotation_z_deg: float,
        gui: bool,
    ) -> None:
        self.robot_urdf = robot_urdf
        self.workpiece_ply = workpiece_ply
        self.workpiece_unit_scale = workpiece_unit_scale
        self.workpiece_rotation_z_deg = workpiece_rotation_z_deg
        self.gui = gui

        self.physics = BulletClient(connection_mode=p.GUI if gui else p.DIRECT)
        self.client_id = self.physics._client
        self.physics.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.physics.setGravity(0.0, 0.0, 0.0)
        self.physics.setTimeStep(1.0 / 240.0)
        self.physics.configureDebugVisualizer(self.physics.COV_ENABLE_GUI, 1 if gui else 0)
        if gui:
            self.physics.configureDebugVisualizer(
                self.physics.COV_ENABLE_RGB_BUFFER_PREVIEW,
                1,
            )
            self.physics.configureDebugVisualizer(
                self.physics.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
                0,
            )
            self.physics.configureDebugVisualizer(
                self.physics.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,
                0,
            )
            # The interactive OpenGL viewport has lighting state independent
            # of getCameraImage(). Its shadow shader also adds a strong point-
            # light specular highlight, which appears as a white circle on the
            # tabletop. Use the non-shadow shader and a straight-down light so
            # the horizontal work surface is illuminated uniformly.
            self.physics.configureDebugVisualizer(
                self.physics.COV_ENABLE_SHADOWS,
                0,
            )
            self.physics.configureDebugVisualizer(lightPosition=(0.0, 0.0, 100.0))

        self.robot_id = -1
        self.table_body_ids: list[int] = []
        self.workpiece_body_id = -1
        self.workpiece_texture_id = -1
        self.camera_marker_body_id = -1

        self.arm_joint_indices: list[int] = []
        self.arm_joint_name_to_index: dict[str, int] = {}
        self.link_name_to_index: dict[str, int] = {}
        self.link_index_to_name: dict[int, str] = {}
        self.link_parent_index: dict[int, int] = {}
        self.arm_joint_lower_limits: list[float] = []
        self.arm_joint_upper_limits: list[float] = []
        self.current_joint_command_rad = [
            HOME_JOINT_POSITIONS_RAD[name] for name in ARM_JOINT_ORDER
        ]

        self.generated_asset_dir = resolve_generated_asset_dir()
        self.generated_robot_urdf = generate_minimal_ur5e_urdf(
            self.robot_urdf,
            self.generated_asset_dir,
        )
        (
            self.generated_workpiece_obj,
            self.generated_workpiece_texture,
            self.workpiece_extent_m,
            self.workpiece_vertex_centroid_local_m,
            self.workpiece_bounding_radius_m,
        ) = convert_ply_to_obj(
            self.workpiece_ply,
            self.generated_asset_dir,
            workpiece_unit_scale,
            workpiece_rotation_z_deg,
        )

        self._build_world()

    def disconnect(self) -> None:
        if self.physics.isConnected():
            self.physics.disconnect()

    def reset_scene_camera(self) -> None:
        """Restore the interactive viewport to its initial useful framing."""

        if not self.gui:
            return
        self.physics.resetDebugVisualizerCamera(
            cameraDistance=DEFAULT_SCENE_CAMERA_DISTANCE_M,
            cameraYaw=DEFAULT_SCENE_CAMERA_YAW_DEG,
            cameraPitch=DEFAULT_SCENE_CAMERA_PITCH_DEG,
            cameraTargetPosition=DEFAULT_SCENE_CAMERA_TARGET,
        )

    def _build_world(self) -> None:
        self.physics.loadURDF("plane.urdf")
        self._spawn_table()
        self._spawn_robot()
        self._spawn_workpiece()
        self._spawn_camera_marker()
        self.reset_to_home_pose()
        self.reset_scene_camera()

    def _spawn_table(self) -> None:
        tabletop_half_extents = [0.5 * value for value in TABLE_SIZE_XYZ]
        tabletop_collision = self.physics.createCollisionShape(
            self.physics.GEOM_BOX,
            halfExtents=tabletop_half_extents,
        )
        tabletop_visual = self.physics.createVisualShape(
            self.physics.GEOM_BOX,
            halfExtents=tabletop_half_extents,
            rgbaColor=[0.80, 0.72, 0.60, 1.0],
            specularColor=[0.0, 0.0, 0.0],
        )
        tabletop_id = self.physics.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=tabletop_collision,
            baseVisualShapeIndex=tabletop_visual,
            basePosition=TABLE_CENTER_XYZ,
        )
        self.table_body_ids.append(tabletop_id)

        leg_collision = self.physics.createCollisionShape(
            self.physics.GEOM_BOX,
            halfExtents=TABLE_LEG_HALF_EXTENTS,
        )
        leg_visual = self.physics.createVisualShape(
            self.physics.GEOM_BOX,
            halfExtents=TABLE_LEG_HALF_EXTENTS,
            rgbaColor=[0.55, 0.48, 0.40, 1.0],
            specularColor=[0.0, 0.0, 0.0],
        )
        for leg_position in TABLE_LEG_POSITIONS:
            leg_id = self.physics.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=leg_collision,
                baseVisualShapeIndex=leg_visual,
                basePosition=leg_position,
            )
            self.table_body_ids.append(leg_id)

    def _spawn_robot(self) -> None:
        self.robot_id = self.physics.loadURDF(
            self.generated_robot_urdf.as_posix(),
            basePosition=ROBOT_BASE_XYZ,
            baseOrientation=(0.0, 0.0, 0.0, 1.0),
            useFixedBase=True,
        )

        for joint_index in range(self.physics.getNumJoints(self.robot_id)):
            joint_info = self.physics.getJointInfo(self.robot_id, joint_index)
            joint_name = joint_info[1].decode("utf-8")
            link_name = joint_info[12].decode("utf-8")
            self.link_name_to_index[link_name] = joint_index
            self.link_index_to_name[joint_index] = link_name
            self.link_parent_index[joint_index] = int(joint_info[16])

            if joint_info[2] == self.physics.JOINT_REVOLUTE and joint_name in ARM_JOINT_ORDER:
                self.arm_joint_indices.append(joint_index)
                self.arm_joint_name_to_index[joint_name] = joint_index

        missing_joints = [
            name
            for name in ARM_JOINT_ORDER
            if name not in self.arm_joint_name_to_index
        ]
        if missing_joints:
            raise RuntimeError(f"Missing expected UR5e joints in PyBullet URDF: {missing_joints}")

        if CAMERA_PARENT_LINK_NAME not in self.link_name_to_index:
            raise RuntimeError(
                f"Missing camera parent link '{CAMERA_PARENT_LINK_NAME}' in robot model."
            )
        if END_EFFECTOR_LINK_NAME not in self.link_name_to_index:
            raise RuntimeError(
                f"Missing end-effector link '{END_EFFECTOR_LINK_NAME}' in robot model."
            )

        self.arm_joint_indices = [self.arm_joint_name_to_index[name] for name in ARM_JOINT_ORDER]
        for joint_index in self.arm_joint_indices:
            joint_info = self.physics.getJointInfo(self.robot_id, joint_index)
            lower_limit = float(joint_info[8])
            upper_limit = float(joint_info[9])
            if upper_limit <= lower_limit:
                lower_limit, upper_limit = -2.0 * math.pi, 2.0 * math.pi
            self.arm_joint_lower_limits.append(lower_limit)
            self.arm_joint_upper_limits.append(upper_limit)

    def _spawn_workpiece(self) -> None:
        visual_shape = self.physics.createVisualShape(
            self.physics.GEOM_MESH,
            fileName=self.generated_workpiece_obj.as_posix(),
            rgbaColor=(
                [1.0, 1.0, 1.0, 1.0]
                if self.generated_workpiece_texture is not None
                else [0.62, 0.70, 0.78, 1.0]
            ),
            specularColor=[0.15, 0.15, 0.15],
        )
        collision_shape = self.physics.createCollisionShape(
            self.physics.GEOM_MESH,
            fileName=self.generated_workpiece_obj.as_posix(),
        )

        self.workpiece_body_id = self.physics.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=WORKPIECE_BASE_XYZ,
            baseOrientation=(0.0, 0.0, 0.0, 1.0),
        )
        if self.generated_workpiece_texture is not None:
            self.workpiece_texture_id = self.physics.loadTexture(
                self.generated_workpiece_texture.as_posix()
            )
            self.physics.changeVisualShape(
                self.workpiece_body_id,
                -1,
                textureUniqueId=self.workpiece_texture_id,
                rgbaColor=[1.0, 1.0, 1.0, 1.0],
            )

    def _spawn_camera_marker(self) -> None:
        camera_collision = self.physics.createCollisionShape(
            self.physics.GEOM_BOX,
            halfExtents=CAMERA_BODY_HALF_EXTENTS,
            collisionFramePosition=CAMERA_COLLISION_FROM_BODY[:3, 3].tolist(),
            collisionFrameOrientation=matrix_to_quat_xyzw(
                CAMERA_COLLISION_FROM_BODY[:3, :3]
            ),
        )
        camera_visual = self.physics.createVisualShape(
            self.physics.GEOM_BOX,
            halfExtents=CAMERA_BODY_HALF_EXTENTS,
            rgbaColor=CAMERA_RGBA,
        )
        self.camera_marker_body_id = self.physics.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=camera_visual,
            baseCollisionShapeIndex=camera_collision,
            basePosition=(0.0, 0.0, 0.0),
            baseOrientation=(0.0, 0.0, 0.0, 1.0),
        )

        camera_parent_index = self.link_name_to_index[CAMERA_PARENT_LINK_NAME]
        camera_body_quat = matrix_to_quat_xyzw(CAMERA_BODY_FROM_PARENT[:3, :3])
        self.physics.createConstraint(
            self.robot_id,
            camera_parent_index,
            self.camera_marker_body_id,
            -1,
            self.physics.JOINT_FIXED,
            jointAxis=(0.0, 0.0, 0.0),
            parentFramePosition=CAMERA_BODY_FROM_PARENT[:3, 3].tolist(),
            childFramePosition=(0.0, 0.0, 0.0),
            parentFrameOrientation=camera_body_quat,
            childFrameOrientation=(0.0, 0.0, 0.0, 1.0),
        )

    def reset_to_home_pose(self) -> None:
        self.current_joint_command_rad = [
            HOME_JOINT_POSITIONS_RAD[name] for name in ARM_JOINT_ORDER
        ]
        self._apply_joint_positions(self.current_joint_command_rad)

    @staticmethod
    def _validated_joint_positions(joint_positions_rad: list[float]) -> list[float]:
        values = [float(value) for value in joint_positions_rad]
        if len(values) != len(ARM_JOINT_ORDER):
            raise ValueError(f"Expected exactly {len(ARM_JOINT_ORDER)} joint positions.")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Joint positions must contain only finite values.")
        return values

    def command_joint_positions(
        self,
        joint_positions_rad: list[float],
        blend: float = COMMAND_BLEND,
    ) -> None:
        targets = self._validated_joint_positions(joint_positions_rad)
        if not math.isfinite(blend) or not 0.0 <= blend <= 1.0:
            raise ValueError("Joint-command blend must be finite and in [0, 1].")
        if blend >= 1.0:
            self.current_joint_command_rad = targets
        else:
            self.current_joint_command_rad = [
                float(current + blend * (target - current))
                for current, target in zip(self.current_joint_command_rad, targets)
            ]

        self._apply_joint_positions(self.current_joint_command_rad)

    def get_joint_positions(self) -> list[float]:
        positions = []
        for joint_name in ARM_JOINT_ORDER:
            joint_index = self.arm_joint_name_to_index[joint_name]
            positions.append(float(self.physics.getJointState(self.robot_id, joint_index)[0]))
        return positions

    def hold_current_pose(self) -> None:
        self._apply_joint_positions(self.current_joint_command_rad)

    def _apply_joint_positions(self, joint_positions_rad: list[float]) -> None:
        targets = self._validated_joint_positions(joint_positions_rad)
        for joint_name, target_value in zip(ARM_JOINT_ORDER, targets):
            joint_index = self.arm_joint_name_to_index[joint_name]
            self.physics.resetJointState(
                self.robot_id,
                joint_index,
                targetValue=float(target_value),
                targetVelocity=0.0,
            )
        self._sync_camera_marker_pose()

    def _sync_camera_marker_pose(self) -> None:
        if self.camera_marker_body_id < 0:
            return
        parent_position, parent_quat = self.get_link_pose(CAMERA_PARENT_LINK_NAME)
        camera_body_transform = (
            make_transform(parent_position, parent_quat) @ CAMERA_BODY_FROM_PARENT
        )
        self.physics.resetBasePositionAndOrientation(
            self.camera_marker_body_id,
            camera_body_transform[:3, 3].tolist(),
            matrix_to_quat_xyzw(camera_body_transform[:3, :3]),
        )

    def solve_tool_ik(
        self,
        target_position_xyz: tuple[float, float, float],
        target_orientation_xyzw: tuple[float, float, float, float] | None = None,
        seed_joint_positions_rad: list[float] | None = None,
    ) -> list[float]:
        end_effector_index = self.link_name_to_index[END_EFFECTOR_LINK_NAME]
        if target_orientation_xyzw is None:
            _, current_orientation = self.get_link_pose(END_EFFECTOR_LINK_NAME)
            target_orientation_xyzw = current_orientation

        if seed_joint_positions_rad is None:
            seed_joint_positions_rad = self.get_joint_positions()
        if len(seed_joint_positions_rad) != len(ARM_JOINT_ORDER):
            raise ValueError(f"Expected {len(ARM_JOINT_ORDER)} IK seed joints.")

        self._apply_joint_positions(seed_joint_positions_rad)
        joint_ranges = [
            upper - lower
            for lower, upper in zip(self.arm_joint_lower_limits, self.arm_joint_upper_limits)
        ]

        solution = self.physics.calculateInverseKinematics(
            self.robot_id,
            end_effector_index,
            targetPosition=target_position_xyz,
            targetOrientation=target_orientation_xyzw,
            lowerLimits=self.arm_joint_lower_limits,
            upperLimits=self.arm_joint_upper_limits,
            jointRanges=joint_ranges,
            restPoses=seed_joint_positions_rad,
            jointDamping=[0.01] * len(ARM_JOINT_ORDER),
            solver=self.physics.IK_DLS,
            maxNumIterations=200,
            residualThreshold=1e-5,
        )
        return [float(solution[index]) for index in range(len(ARM_JOINT_ORDER))]

    def get_link_pose(
        self,
        link_name: str,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        link_index = self.link_name_to_index[link_name]
        link_state = self.physics.getLinkState(
            self.robot_id,
            link_index,
            computeForwardKinematics=True,
        )
        return (
            tuple(float(value) for value in link_state[4]),
            tuple(float(value) for value in link_state[5]),
        )

    def get_link_transform(self, link_name: str) -> np.ndarray:
        position, quaternion = self.get_link_pose(link_name)
        return make_transform(position, quaternion)

    def get_wrist_camera_pose(self) -> tuple[np.ndarray, np.ndarray]:
        parent_position, parent_quat = self.get_link_pose(CAMERA_PARENT_LINK_NAME)
        parent_transform = make_transform(parent_position, parent_quat)
        optical_transform = parent_transform @ CAMERA_OPTICAL_FROM_PARENT
        return optical_transform[:3, 3], optical_transform[:3, :3]

    def get_workpiece_center(self) -> tuple[float, float, float]:
        lower, upper = self.physics.getAABB(self.workpiece_body_id)
        center = 0.5 * (np.asarray(lower, dtype=float) + np.asarray(upper, dtype=float))
        return tuple(float(value) for value in center)

    def get_workpiece_vertex_centroid(self) -> tuple[float, float, float]:
        """Return the transformed mean mesh-vertex position in world metres."""

        position, orientation = self.physics.getBasePositionAndOrientation(
            self.workpiece_body_id
        )
        world_from_workpiece = make_transform(position, orientation)
        centroid_homogeneous = np.append(self.workpiece_vertex_centroid_local_m, 1.0)
        centroid_world = world_from_workpiece @ centroid_homogeneous
        return tuple(float(value) for value in centroid_world[:3])

    def get_tool_from_camera_transform(self) -> np.ndarray:
        """Return the fixed optical-camera transform expressed from tool0."""

        world_from_parent = self.get_link_transform(CAMERA_PARENT_LINK_NAME)
        world_from_tool = self.get_link_transform(END_EFFECTOR_LINK_NAME)
        parent_from_tool = np.linalg.inv(world_from_parent) @ world_from_tool
        return np.linalg.inv(parent_from_tool) @ CAMERA_OPTICAL_FROM_PARENT

    def camera_viewpoint_to_tool_pose(
        self, viewpoint: ViewpointCandidate, tool_from_camera: np.ndarray | None = None
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        if tool_from_camera is None:
            tool_from_camera = self.get_tool_from_camera_transform()
        world_from_camera = make_transform(viewpoint.position_xyz, viewpoint.orientation_xyzw)
        world_from_tool = world_from_camera @ np.linalg.inv(tool_from_camera)
        return (
            tuple(float(value) for value in world_from_tool[:3, 3]),
            matrix_to_quat_xyzw(world_from_tool[:3, :3]),
        )

    @staticmethod
    def _halton(index: int, base: int) -> float:
        value = 0.0
        fraction = 1.0 / base
        while index > 0:
            value += fraction * (index % base)
            index //= base
            fraction /= base
        return value

    def _generate_ik_seeds(
        self, count: int, reference_joint_positions_rad: list[float] | None = None
    ) -> list[list[float]]:
        if count <= 0:
            raise ValueError("IK seed count must be positive.")

        reference = np.asarray(
            reference_joint_positions_rad
            if reference_joint_positions_rad is not None
            else [HOME_JOINT_POSITIONS_RAD[name] for name in ARM_JOINT_ORDER],
            dtype=float,
        )
        seeds = [reference.tolist()]
        primes = (2, 3, 5, 7, 11, 13)
        lower = np.asarray(self.arm_joint_lower_limits, dtype=float)
        upper = np.asarray(self.arm_joint_upper_limits, dtype=float)
        span = upper - lower

        for seed_index in range(1, count):
            # Keep low-discrepancy seeds away from exact hard stops.
            unit = np.asarray(
                [self._halton(seed_index, prime) for prime in primes], dtype=float
            )
            seed = lower + (0.08 + 0.84 * unit) * span
            seeds.append(seed.tolist())
        return seeds

    def _joint_limits_contain(
        self,
        joint_positions_rad: list[float],
        epsilon: float = 1e-5,
    ) -> bool:
        return all(
            lower - epsilon <= value <= upper + epsilon
            for value, lower, upper in zip(
                joint_positions_rad,
                self.arm_joint_lower_limits,
                self.arm_joint_upper_limits,
            )
        )

    def _canonicalize_joint_solution(
        self, joint_positions_rad: list[float], seed_joint_positions_rad: list[float]
    ) -> list[float]:
        """Select an equivalent 2-pi revolution inside limits and nearest the seed."""

        canonical: list[float] = []
        for value, seed, lower, upper in zip(
            joint_positions_rad,
            seed_joint_positions_rad,
            self.arm_joint_lower_limits,
            self.arm_joint_upper_limits,
        ):
            equivalents = [
                value + turns * 2.0 * math.pi
                for turns in range(-3, 4)
                if lower - 1e-8 <= value + turns * 2.0 * math.pi <= upper + 1e-8
            ]
            canonical.append(
                min(equivalents, key=lambda candidate: abs(candidate - seed))
                if equivalents
                else value
            )
        return canonical

    def _self_links_are_adjacent(self, first_link: int, second_link: int) -> bool:
        return (
            first_link == second_link
            or self.link_parent_index.get(first_link) == second_link
            or self.link_parent_index.get(second_link) == first_link
        )

    def _configuration_clearance(
        self, required_clearance_m: float
    ) -> tuple[bool, float, str | None]:
        """Check robot, camera, environment, and non-adjacent self collision."""

        self.physics.performCollisionDetection()
        query_distance = max(0.10, required_clearance_m)
        minimum_clearance = query_distance

        external_bodies = [(body_id, True) for body_id in self.table_body_ids]
        external_bodies.append((self.workpiece_body_id, False))
        for body_id, is_table in external_bodies:
            for point in self.physics.getClosestPoints(self.robot_id, body_id, query_distance):
                robot_link_name = self.link_index_to_name.get(int(point[3]), "")
                if is_table and robot_link_name in ROBOT_TABLE_ALLOWED_CONTACT_LINKS:
                    continue
                distance = float(point[8])
                minimum_clearance = min(minimum_clearance, distance)
                if distance < required_clearance_m:
                    return False, minimum_clearance, "environment_collision"

            for point in self.physics.getClosestPoints(
                self.camera_marker_body_id,
                body_id,
                query_distance,
            ):
                distance = float(point[8])
                minimum_clearance = min(minimum_clearance, distance)
                if distance < required_clearance_m:
                    return False, minimum_clearance, "camera_collision"

        for point in self.physics.getClosestPoints(
            self.camera_marker_body_id, self.robot_id, query_distance
        ):
            robot_link_name = self.link_index_to_name.get(int(point[4]), "")
            if robot_link_name in CAMERA_ROBOT_ALLOWED_CONTACT_LINKS:
                continue
            distance = float(point[8])
            minimum_clearance = min(minimum_clearance, distance)
            if distance < required_clearance_m:
                return False, minimum_clearance, "camera_robot_collision"

        checked_pairs: set[tuple[int, int]] = set()
        for point in self.physics.getClosestPoints(self.robot_id, self.robot_id, query_distance):
            first_link, second_link = int(point[3]), int(point[4])
            pair = tuple(sorted((first_link, second_link)))
            if pair in checked_pairs or self._self_links_are_adjacent(first_link, second_link):
                continue
            checked_pairs.add(pair)
            distance = float(point[8])
            minimum_clearance = min(minimum_clearance, distance)
            if distance < required_clearance_m:
                return False, minimum_clearance, "self_collision"

        return True, minimum_clearance, None

    def _has_object_line_of_sight(self, object_center_xyz: tuple[float, float, float]) -> bool:
        camera_position, camera_rotation = self.get_wrist_camera_pose()
        forward = normalize(camera_rotation[:, 2])
        ray_start = camera_position + forward * 0.05
        hit = self.physics.rayTest(ray_start.tolist(), list(object_center_xyz))[0]
        hit_body_id = int(hit[0])
        return hit_body_id in (-1, self.workpiece_body_id)

    def _manipulability(self, joint_positions_rad: list[float]) -> float:
        try:
            translational, rotational = self.physics.calculateJacobian(
                self.robot_id,
                self.link_name_to_index[CAMERA_PARENT_LINK_NAME],
                localPosition=CAMERA_OPTICAL_FROM_PARENT[:3, 3].tolist(),
                objPositions=joint_positions_rad,
                objVelocities=[0.0] * len(ARM_JOINT_ORDER),
                objAccelerations=[0.0] * len(ARM_JOINT_ORDER),
            )
            # This is the Jacobian at the optical center, not at tool0. Scale
            # angular velocity by a characteristic arm length so the SVD combines
            # translational and rotational conditioning sensibly.
            jacobian = np.vstack(
                (np.asarray(translational, dtype=float), 0.20 * np.asarray(rotational, dtype=float))
            )
            return float(np.linalg.svd(jacobian, compute_uv=False)[-1])
        except Exception:
            return 0.0

    def _joint_limit_margin(self, joint_positions_rad: list[float]) -> float:
        margins = []
        for value, lower, upper in zip(
            joint_positions_rad, self.arm_joint_lower_limits, self.arm_joint_upper_limits
        ):
            span = max(upper - lower, 1e-9)
            margins.append(2.0 * min(value - lower, upper - value) / span)
        return float(np.clip(min(margins), 0.0, 1.0))

    def _quality_score(
        self,
        joint_positions_rad: list[float],
        clearance_m: float,
        manipulability: float,
        joint_limit_margin: float,
        reference_joint_positions_rad: list[float],
    ) -> float:
        reference = np.asarray(reference_joint_positions_rad, dtype=float)
        joints = np.asarray(joint_positions_rad, dtype=float)
        spans = np.asarray(self.arm_joint_upper_limits) - np.asarray(self.arm_joint_lower_limits)
        travel = float(
            np.linalg.norm((joints - reference) / np.maximum(spans, 1e-9))
            / math.sqrt(len(joints))
        )
        travel_score = float(np.clip(1.0 - 2.0 * travel, 0.0, 1.0))
        clearance_score = float(np.clip(clearance_m / 0.05, 0.0, 1.0))
        manipulability_score = manipulability / (manipulability + 0.04)
        return float(
            0.30 * joint_limit_margin
            + 0.30 * manipulability_score
            + 0.20 * clearance_score
            + 0.20 * travel_score
        )

    def _joint_path_is_clear(
        self,
        start_joint_positions_rad: list[float],
        goal_joint_positions_rad: list[float],
        steps: int,
        clearance_m: float,
    ) -> bool:
        if steps <= 0:
            return True
        start = np.asarray(start_joint_positions_rad, dtype=float)
        goal = np.asarray(goal_joint_positions_rad, dtype=float)
        for step_index in range(1, steps + 1):
            fraction = step_index / steps
            self._apply_joint_positions((start + fraction * (goal - start)).tolist())
            collision_free, _, _ = self._configuration_clearance(clearance_m)
            if not collision_free:
                return False
        return True

    def validate_joint_trajectory(
        self,
        joint_positions_rad: np.ndarray,
        clearance_m: float,
        *,
        max_joint_step_rad: float = 0.02,
    ) -> TrajectoryValidationResult:
        """Revalidate a CuRobo path with PyBullet before simulated execution.

        CuRobo's dense waypoints are checked directly. Every segment is also
        subdivided so no checked joint-space increment exceeds
        ``max_joint_step_rad``. The entry pose is restored even on failure.
        """

        positions = np.asarray(joint_positions_rad, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != len(ARM_JOINT_ORDER):
            raise ValueError(
                "Trajectory positions must have shape (waypoints, "
                f"{len(ARM_JOINT_ORDER)})."
            )
        if positions.shape[0] < 2 or not np.all(np.isfinite(positions)):
            raise ValueError("Trajectory must contain at least two finite waypoints.")
        if not math.isfinite(clearance_m) or clearance_m < 0.0:
            raise ValueError("Trajectory clearance must be finite and non-negative.")
        if not math.isfinite(max_joint_step_rad) or max_joint_step_rad <= 0.0:
            raise ValueError("Maximum trajectory joint step must be finite and positive.")

        original_positions = self.get_joint_positions()
        original_command = list(self.current_joint_command_rad)
        start_error = float(
            np.max(np.abs(positions[0] - np.asarray(original_positions, dtype=float)))
        )
        if start_error > max(0.05, 2.0 * max_joint_step_rad):
            return TrajectoryValidationResult(
                valid=False,
                samples_checked=0,
                minimum_clearance_m=0.0,
                failure_reason="start_state_mismatch",
                failure_segment_index=0,
            )

        samples_checked = 0
        minimum_clearance = float("inf")
        previous = np.asarray(original_positions, dtype=float)
        try:
            for waypoint_index, waypoint in enumerate(positions):
                delta = waypoint - previous
                subdivision_count = max(
                    1, int(math.ceil(float(np.max(np.abs(delta))) / max_joint_step_rad))
                )
                for subdivision_index in range(1, subdivision_count + 1):
                    fraction = subdivision_index / subdivision_count
                    sample = previous + fraction * delta
                    samples_checked += 1
                    if not self._joint_limits_contain(sample.tolist()):
                        return TrajectoryValidationResult(
                            valid=False,
                            samples_checked=samples_checked,
                            minimum_clearance_m=(
                                minimum_clearance
                                if math.isfinite(minimum_clearance)
                                else 0.0
                            ),
                            failure_reason="joint_limits",
                            failure_segment_index=max(0, waypoint_index - 1),
                        )
                    self._apply_joint_positions(sample.tolist())
                    collision_free, measured_clearance, collision_reason = (
                        self._configuration_clearance(clearance_m)
                    )
                    minimum_clearance = min(minimum_clearance, measured_clearance)
                    if not collision_free:
                        return TrajectoryValidationResult(
                            valid=False,
                            samples_checked=samples_checked,
                            minimum_clearance_m=minimum_clearance,
                            failure_reason=collision_reason or "collision",
                            failure_segment_index=max(0, waypoint_index - 1),
                        )
                previous = waypoint
        finally:
            self.current_joint_command_rad = original_command
            self._apply_joint_positions(original_positions)

        return TrajectoryValidationResult(
            valid=True,
            samples_checked=samples_checked,
            minimum_clearance_m=(
                minimum_clearance if math.isfinite(minimum_clearance) else 0.0
            ),
        )

    def _build_ik_problems(
        self,
        viewpoints: list[ViewpointCandidate],
        roll_angles_deg: tuple[float, ...],
    ) -> list[IKProblem]:
        """Precompute grouped roll goals once for every candidate backend."""

        tool_from_camera = self.get_tool_from_camera_transform()
        problems: list[IKProblem] = []
        for candidate in viewpoints:
            goals: list[IKGoal] = []
            for roll_deg in roll_angles_deg:
                rolled_candidate = candidate.with_roll(roll_deg)
                tool_position, tool_orientation = self.camera_viewpoint_to_tool_pose(
                    rolled_candidate, tool_from_camera
                )
                goals.append(
                    IKGoal(
                        camera_viewpoint=rolled_candidate,
                        tool_position_xyz=tool_position,
                        tool_orientation_xyzw=tool_orientation,
                    )
                )
            problems.append(IKProblem(sampled_viewpoint=candidate, goals=tuple(goals)))
        return problems

    def _solve_pybullet_ik_problems(
        self,
        problems: list[IKProblem],
        original_joints: list[float],
        config: ReachabilityConfig,
    ) -> list[IKProblemResult]:
        """Generate deterministic multi-start candidates without accepting them."""

        seeds = self._generate_ik_seeds(config.seed_count, original_joints)
        results: list[IKProblemResult] = []
        for problem in problems:
            candidates: list[IKCandidate] = []
            for goal_index, goal in enumerate(problem.goals):
                for seed_index, seed in enumerate(seeds):
                    joint_solution = self.solve_tool_ik(
                        goal.tool_position_xyz,
                        goal.tool_orientation_xyzw,
                        seed_joint_positions_rad=seed,
                    )
                    joint_solution = self._canonicalize_joint_solution(
                        joint_solution, original_joints
                    )
                    for _ in range(config.refinement_steps):
                        if not self._joint_limits_contain(joint_solution):
                            break
                        joint_solution = self.solve_tool_ik(
                            goal.tool_position_xyz,
                            goal.tool_orientation_xyzw,
                            seed_joint_positions_rad=joint_solution,
                        )
                        joint_solution = self._canonicalize_joint_solution(
                            joint_solution, original_joints
                        )
                    candidates.append(
                        IKCandidate(
                            goal_index=goal_index,
                            joint_positions_rad=tuple(joint_solution),
                            seed_index=seed_index,
                            solver_provenance={
                                "roll_goal_index": goal_index,
                                "seed_index": seed_index,
                            },
                        )
                    )
            results.append(
                IKProblemResult(
                    viewpoint_index=problem.sampled_viewpoint.index,
                    candidates=tuple(candidates),
                    attempted_solutions=len(problem.goals) * len(seeds),
                )
            )
        return results

    def _validate_ik_candidate(
        self,
        candidate: IKCandidate,
        goal: IKGoal,
        object_center_xyz: tuple[float, float, float],
        original_joints: list[float],
        config: ReachabilityConfig,
    ) -> tuple[ReachableViewpoint | None, str | None]:
        """Apply the backend-independent endpoint acceptance gates."""

        if len(candidate.joint_positions_rad) != len(ARM_JOINT_ORDER):
            raise RuntimeError(
                f"IK backend returned {len(candidate.joint_positions_rad)} joints; "
                f"expected {len(ARM_JOINT_ORDER)}."
            )
        joint_solution = self._canonicalize_joint_solution(
            list(candidate.joint_positions_rad), original_joints
        )
        if not all(math.isfinite(value) for value in joint_solution):
            raise RuntimeError("IK backend returned non-finite joint positions.")
        if not self._joint_limits_contain(joint_solution):
            return None, "joint_limits"

        self._apply_joint_positions(joint_solution)
        actual_camera_position, actual_camera_rotation = self.get_wrist_camera_pose()
        actual_camera_quaternion = matrix_to_quat_xyzw(actual_camera_rotation)
        position_error = float(
            np.linalg.norm(
                actual_camera_position
                - np.asarray(goal.camera_viewpoint.position_xyz, dtype=float)
            )
        )
        orientation_error = quaternion_angular_distance_deg(
            actual_camera_quaternion, goal.camera_viewpoint.orientation_xyzw
        )
        if (
            position_error > config.position_tolerance_m
            or orientation_error > config.orientation_tolerance_deg
        ):
            return None, "ik_residual"

        collision_free, clearance, collision_reason = self._configuration_clearance(
            config.clearance_m
        )
        if not collision_free:
            return None, collision_reason or "collision"

        if config.require_line_of_sight and not self._has_object_line_of_sight(
            object_center_xyz
        ):
            return None, "occluded"

        linear_path_collision_free = None
        if config.path_validation_steps > 0:
            linear_path_collision_free = self._joint_path_is_clear(
                original_joints,
                joint_solution,
                config.path_validation_steps,
                config.clearance_m,
            )

        self._apply_joint_positions(joint_solution)
        manipulability = self._manipulability(joint_solution)
        joint_limit_margin = self._joint_limit_margin(joint_solution)
        return (
            ReachableViewpoint(
                viewpoint=goal.camera_viewpoint,
                joint_positions_rad=joint_solution,
                position_error_m=position_error,
                orientation_error_deg=orientation_error,
                clearance_m=clearance,
                joint_limit_margin=joint_limit_margin,
                manipulability=manipulability,
                quality_score=self._quality_score(
                    joint_solution,
                    clearance,
                    manipulability,
                    joint_limit_margin,
                    original_joints,
                ),
                seed_index=candidate.seed_index,
                solver_provenance=candidate.solver_provenance,
                linear_path_collision_free=linear_path_collision_free,
            ),
            None,
        )

    def filter_reachable_viewpoints(
        self,
        viewpoints: list[ViewpointCandidate],
        object_center_xyz: tuple[float, float, float],
        config: ReachabilityConfig,
        candidate_backend: IKCandidateBackend | None = None,
    ) -> ReachabilityReport:
        """Generate IK candidates, then retain only PyBullet-validated endpoints."""

        if not viewpoints:
            raise ValueError("At least one viewpoint is required.")
        config.validate()

        start_time = time.monotonic()
        original_joints = self.get_joint_positions()
        problems = self._build_ik_problems(viewpoints, config.roll_angles_deg)
        reachable: list[ReachableViewpoint] = []
        rejected: list[RejectedViewpoint] = []
        solver_backend = "pybullet_multistart_ik"
        backend_metadata: dict[str, object] = {
            "roll_goal_count": len(config.roll_angles_deg),
            "seed_count_per_goal": config.seed_count,
            "refinement_steps": config.refinement_steps,
        }

        try:
            if candidate_backend is None:
                problem_results = self._solve_pybullet_ik_problems(
                    problems, original_joints, config
                )
            else:
                solver_backend = str(
                    getattr(candidate_backend, "backend_name", type(candidate_backend).__name__)
                )
                solve_method = getattr(candidate_backend, "solve", None)
                if not callable(solve_method):
                    raise TypeError("IK candidate backend must provide a callable solve method.")
                problem_results = solve_method(problems, original_joints, config)
                backend_metadata = dict(getattr(candidate_backend, "metadata", {}) or {})

            result_by_index: dict[int, IKProblemResult] = {}
            for result in problem_results:
                if result.viewpoint_index in result_by_index:
                    raise RuntimeError(
                        f"IK backend returned duplicate viewpoint index {result.viewpoint_index}."
                    )
                result_by_index[result.viewpoint_index] = result
            expected_indices = {problem.sampled_viewpoint.index for problem in problems}
            if set(result_by_index) != expected_indices:
                raise RuntimeError(
                    "IK backend result indices do not match the submitted viewpoints."
                )

            self._apply_joint_positions(original_joints)
            for problem in problems:
                result = result_by_index[problem.sampled_viewpoint.index]
                if result.attempted_solutions < 0:
                    raise RuntimeError("IK backend returned a negative attempt count.")
                viable_solutions: list[ReachableViewpoint] = []
                failure_counts = dict(result.solver_failure_counts or {})
                for candidate in result.candidates:
                    if not 0 <= candidate.goal_index < len(problem.goals):
                        raise RuntimeError(
                            f"IK backend returned invalid goal index {candidate.goal_index}."
                        )
                    solution, failure_reason = self._validate_ik_candidate(
                        candidate,
                        problem.goals[candidate.goal_index],
                        object_center_xyz,
                        original_joints,
                        config,
                    )
                    if solution is None:
                        reason = failure_reason or "validation_failure"
                        failure_counts[reason] = failure_counts.get(reason, 0) + 1
                    else:
                        viable_solutions.append(solution)

                if viable_solutions:
                    best_solution = max(
                        viable_solutions,
                        key=lambda solution: (
                            solution.linear_path_collision_free is not False,
                            solution.quality_score,
                        ),
                    )
                    best_solution.successful_solution_count = len(viable_solutions)
                    best_solution.attempted_solutions = result.attempted_solutions
                    best_solution.failure_counts = failure_counts
                    reachable.append(best_solution)
                else:
                    rejection_priority = (
                        "occluded",
                        "camera_robot_collision",
                        "camera_collision",
                        "environment_collision",
                        "self_collision",
                        "ik_residual",
                        "joint_limits",
                        "ik_solver_no_solution",
                    )
                    furthest_failure_stage = next(
                        (failure for failure in rejection_priority if failure in failure_counts),
                        "no_ik_solution",
                    )
                    rejected.append(
                        RejectedViewpoint(
                            viewpoint=problem.sampled_viewpoint,
                            furthest_failure_stage=furthest_failure_stage,
                            attempted_solutions=result.attempted_solutions,
                            failure_counts=failure_counts,
                        )
                    )
        finally:
            self._apply_joint_positions(original_joints)

        reachable.sort(key=lambda solution: solution.quality_score, reverse=True)
        return ReachabilityReport(
            sampled=viewpoints,
            reachable=reachable,
            rejected=rejected,
            object_center_xyz=object_center_xyz,
            start_joint_positions_rad=original_joints,
            elapsed_sec=time.monotonic() - start_time,
            solver_backend=solver_backend,
            backend_metadata=backend_metadata,
        )

    def capture_wrist_camera(
        self,
        width: int = CAMERA_WIDTH,
        height: int = CAMERA_HEIGHT,
    ) -> CameraCapture:
        camera_position, camera_rotation = self.get_wrist_camera_pose()
        forward = normalize(camera_rotation[:, 2])
        up = normalize(-camera_rotation[:, 1])
        target = camera_position + forward * 1.0

        view_matrix = self.physics.computeViewMatrix(
            cameraEyePosition=camera_position.tolist(),
            cameraTargetPosition=target.tolist(),
            cameraUpVector=up.tolist(),
        )
        projection_matrix = self.physics.computeProjectionMatrixFOV(
            fov=compute_camera_vertical_fov_deg(),
            aspect=float(width) / float(height),
            nearVal=CAMERA_NEAR,
            farVal=CAMERA_FAR,
        )

        renderer = (
            self.physics.ER_BULLET_HARDWARE_OPENGL
            if self.gui
            else self.physics.ER_TINY_RENDERER
        )
        _, _, rgba, _, _ = self.physics.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=renderer,
            lightDirection=(0.0, 0.0, -1.0),
            lightColor=(1.0, 1.0, 1.0),
            lightDistance=10.0,
            lightAmbientCoeff=0.8,
            lightDiffuseCoeff=0.2,
            lightSpecularCoeff=0.0,
            shadow=0,
        )

        rgb = np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)[..., :3]
        return CameraCapture(rgb=rgb)

    def save_wrist_camera_frame(
        self,
        output_path: Path,
        png_metadata: dict[str, str] | None = None,
    ) -> Path:
        capture = self.capture_wrist_camera()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        png_info = None
        if png_metadata:
            png_info = PngImagePlugin.PngInfo()
            for key, value in png_metadata.items():
                png_info.add_text(str(key), str(value))
        Image.fromarray(capture.rgb).save(output_path, pnginfo=png_info)
        return output_path

    def print_scene_summary(self) -> None:
        print(f"PyBullet client id: {self.client_id}")
        print(f"Robot URDF source: {self.robot_urdf}")
        print(f"Generated PyBullet URDF: {self.generated_robot_urdf}")
        print(f"Robot base position (m): {ROBOT_BASE_XYZ}")
        print(f"Camera parent link: {CAMERA_PARENT_LINK_NAME}")
        print(f"IK end-effector link: {END_EFFECTOR_LINK_NAME}")
        print(f"Workpiece source: {self.workpiece_ply}")
        print(f"Workpiece extent (m): {np.round(self.workpiece_extent_m, 4).tolist()}")
        print(f"Table top z (m): {TABLE_TOP_Z}")
        print(f"Default workpiece xy (m): {DEFAULT_WORKPIECE_XY}")


def _viewpoint_json(viewpoint: ViewpointCandidate) -> dict[str, object]:
    return {
        "index": viewpoint.index,
        "position_xyz_m": list(viewpoint.position_xyz),
        "orientation_xyzw": list(viewpoint.orientation_xyzw),
        "radius_m": viewpoint.radius_m,
        "azimuth_deg": viewpoint.azimuth_deg,
        "elevation_deg": viewpoint.elevation_deg,
        "roll_deg": viewpoint.roll_deg,
    }


def save_reachability_report(
    report: ReachabilityReport,
    config: ReachabilityConfig,
    env: UR5EyeInHandPyBulletEnv,
    output_path: Path,
) -> Path:
    reachable_by_index = {solution.viewpoint.index: solution for solution in report.reachable}
    rejected_by_index = {rejection.viewpoint.index: rejection for rejection in report.rejected}
    workpiece_aabb_lower, workpiece_aabb_upper = env.physics.getAABB(
        env.workpiece_body_id
    )
    records: list[dict[str, object]] = []

    for sampled_viewpoint in report.sampled:
        if sampled_viewpoint.index in reachable_by_index:
            solution = reachable_by_index[sampled_viewpoint.index]
            records.append(
                {
                    **_viewpoint_json(solution.viewpoint),
                    "reachable": True,
                    "reachability_status": "validated_endpoint_solution",
                    "endpoint_reachable": True,
                    "linear_path_checked": config.path_validation_steps > 0,
                    "linear_path_collision_free": solution.linear_path_collision_free,
                    "joint_positions_rad": solution.joint_positions_rad,
                    "position_error_m": solution.position_error_m,
                    "orientation_error_deg": solution.orientation_error_deg,
                    "clearance_m": solution.clearance_m,
                    "joint_limit_margin": solution.joint_limit_margin,
                    "manipulability": solution.manipulability,
                    "quality_score": solution.quality_score,
                    "seed_index": solution.seed_index,
                    "solver_provenance": solution.solver_provenance or {},
                    "successful_solution_count": solution.successful_solution_count,
                    "attempted_solutions": solution.attempted_solutions,
                    "failure_counts": solution.failure_counts or {},
                }
            )
        else:
            rejection = rejected_by_index[sampled_viewpoint.index]
            records.append(
                {
                    **_viewpoint_json(rejection.viewpoint),
                    "reachable": False,
                    "reachability_status": "no_valid_solution_found",
                    "endpoint_reachable": None,
                    "linear_path_checked": False,
                    "linear_path_collision_free": None,
                    "rejection_reason": "no_valid_solution_found",
                    "furthest_failure_stage_observed": rejection.furthest_failure_stage,
                    "attempted_solutions": rejection.attempted_solutions,
                    "failure_counts": rejection.failure_counts,
                }
            )

    payload = {
        "schema_version": 1,
        "backend": report.solver_backend,
        "validation_backend": report.validation_backend,
        "backend_metadata": report.backend_metadata or {},
        "reachability_scope": "collision_free_ik_endpoint",
        "rejection_semantics": (
            "No valid solution was found in the finite deterministic IK search; "
            "this is not a mathematical proof of unreachability."
        ),
        "collision_model_notes": [
            (
                "Checks cover the simplified UR5e arm meshes, optical-frame camera "
                "housing proxy, table, legs, and workpiece."
            ),
            (
                "The generated runtime URDF intentionally strips the source "
                "gripper/end-effector assembly."
            ),
            (
                "The ground plane is not part of the reachability clearance query; "
                "the elevated table is included."
            ),
        ],
        "pose_convention": (
            "T_world_camera optical frame (+Z forward, +X right, -Y up), "
            "quaternion XYZW"
        ),
        "object_center_xyz_m": list(report.object_center_xyz),
        "robot": {
            "urdf": env.robot_urdf.as_posix(),
            "base_position_xyz_m": list(ROBOT_BASE_XYZ),
            "end_effector_link": END_EFFECTOR_LINK_NAME,
            "joint_names": ARM_JOINT_ORDER,
            "joint_lower_limits_rad": env.arm_joint_lower_limits,
            "joint_upper_limits_rad": env.arm_joint_upper_limits,
            "start_joint_positions_rad": report.start_joint_positions_rad,
        },
        "camera": {
            "parent_link": CAMERA_PARENT_LINK_NAME,
            "tool_from_optical_camera": env.get_tool_from_camera_transform().tolist(),
            "parent_from_optical_camera": CAMERA_OPTICAL_FROM_PARENT.tolist(),
            "parent_from_collision_proxy": CAMERA_COLLISION_FROM_PARENT.tolist(),
            "collision_half_extents_m": list(CAMERA_BODY_HALF_EXTENTS),
        },
        "world": {
            "workpiece_ply": env.workpiece_ply.as_posix(),
            "workpiece_unit_scale": env.workpiece_unit_scale,
            "workpiece_rotation_z_deg_baked_into_mesh": env.workpiece_rotation_z_deg,
            "workpiece_base_position_xyz_m": list(WORKPIECE_BASE_XYZ),
            "workpiece_base_orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            "workpiece_extent_m": env.workpiece_extent_m.tolist(),
            "workpiece_aabb_lower_xyz_m": list(workpiece_aabb_lower),
            "workpiece_aabb_upper_xyz_m": list(workpiece_aabb_upper),
            "table_top_z_m": TABLE_TOP_Z,
            "table_size_xyz_m": list(TABLE_SIZE_XYZ),
            "table_center_xyz_m": list(TABLE_CENTER_XYZ),
            "table_leg_half_extents_m": list(TABLE_LEG_HALF_EXTENTS),
            "table_leg_positions_xyz_m": [list(position) for position in TABLE_LEG_POSITIONS],
        },
        "sampled_count": len(report.sampled),
        "reachable_count": len(report.reachable),
        "rejected_count": len(report.rejected),
        "elapsed_sec": report.elapsed_sec,
        "sampling": report.sampling_metadata
        or {
            "algorithm": "equal-area_fibonacci_shell_with_radical_inverse_radius",
            "count": len(report.sampled),
        },
        "reachability_config": {
            "roll_angles_deg": list(config.roll_angles_deg),
            "seed_count": config.seed_count,
            "refinement_steps": config.refinement_steps,
            "position_tolerance_m": config.position_tolerance_m,
            "orientation_tolerance_deg": config.orientation_tolerance_deg,
            "clearance_m": config.clearance_m,
            "require_line_of_sight": config.require_line_of_sight,
            "path_validation_steps": config.path_validation_steps,
        },
        "viewpoints": records,
        "reachable_indices_ranked_by_quality": [
            solution.viewpoint.index for solution in report.reachable
        ],
        "linear_path_collision_free_indices": [
            solution.viewpoint.index
            for solution in report.reachable
            if solution.linear_path_collision_free is True
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return output_path


def visualize_reachability_report(
    env: UR5EyeInHandPyBulletEnv, report: ReachabilityReport
) -> None:
    reachable_indices = {solution.viewpoint.index for solution in report.reachable}
    points = [list(viewpoint.position_xyz) for viewpoint in report.sampled]
    colors = [
        [0.15, 0.85, 0.25] if viewpoint.index in reachable_indices else [0.90, 0.15, 0.15]
        for viewpoint in report.sampled
    ]
    env.physics.addUserDebugPoints(points, colors, pointSize=7.0, lifeTime=0.0)
    env.physics.addUserDebugPoints(
        [list(report.object_center_xyz)],
        [[0.1, 0.3, 1.0]],
        pointSize=12.0,
    )


def _unit_scores(values: np.ndarray) -> np.ndarray:
    """Robustly scale scores for display while preserving their ordering."""

    low, high = np.quantile(values, (0.05, 0.95))
    if high <= low:
        return np.ones_like(values)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def _draw_scored_points(
    env: UR5EyeInHandPyBulletEnv,
    points: np.ndarray,
    colors: np.ndarray,
    density: np.ndarray,
) -> None:
    """Draw low-density samples small and high-density samples large."""

    bins = np.digitize(np.clip(density, 0.0, 1.0), (1.0 / 3.0, 2.0 / 3.0))
    for bin_index, point_size in enumerate((2.0, 4.0, 7.0)):
        selected = bins == bin_index
        if np.any(selected):
            env.physics.addUserDebugPoints(
                points[selected].tolist(),
                colors[selected].tolist(),
                pointSize=point_size,
                lifeTime=0.0,
            )


def visualize_capability_map(
    env: UR5EyeInHandPyBulletEnv,
    capability_map: CapabilityMap,
    placeability_voxel_m: float = 0.05,
) -> tuple[int, int]:
    """Show TCP capability and planar base placeability around the workpiece.

    Workspace point size is reachability and blue-to-yellow color is
    manipulability. Placeability uses the same encoding in magenta on the table.
    """

    workspace_points = capability_map.workspace_xyz_m + np.asarray(ROBOT_BASE_XYZ)
    quality = _unit_scores(capability_map.manipulability)
    workspace_colors = np.column_stack((quality, 0.25 + 0.75 * quality, 1.0 - quality))
    _draw_scored_points(
        env, workspace_points, workspace_colors, capability_map.reachability
    )

    # Place the target TCP at the object while preserving the home TCP orientation.
    world_from_tcp = env.get_link_transform(END_EFFECTOR_LINK_NAME).copy()
    world_from_tcp[:3, 3] = env.get_workpiece_center()
    world_from_bases = world_from_tcp @ capability_map.inverse_reachability
    base_xyz = world_from_bases[:, :3, 3]
    upright = world_from_bases[:, 2, 2] > math.cos(math.radians(10.0))
    planar = upright & (np.abs(base_xyz[:, 2] - TABLE_TOP_Z) <= 0.05)

    if not np.any(planar):
        return len(workspace_points), 0

    cells = np.floor(base_xyz[planar, :2] / placeability_voxel_m).astype(int)
    unique_cells, cell_ids, counts = np.unique(
        cells, axis=0, return_inverse=True, return_counts=True
    )
    cell_quality = np.zeros(len(unique_cells))
    np.maximum.at(
        cell_quality, cell_ids, capability_map.inverse_manipulability[planar]
    )
    placeability_points = np.column_stack(
        (
            (unique_cells + 0.5) * placeability_voxel_m,
            np.full(len(unique_cells), TABLE_TOP_Z + 0.01),
        )
    )
    quality = _unit_scores(cell_quality)
    placeability_colors = np.column_stack(
        (np.ones_like(quality), 0.15 + 0.70 * quality, 0.90 * np.ones_like(quality))
    )
    _draw_scored_points(
        env, placeability_points, placeability_colors, counts / counts.max()
    )
    return len(workspace_points), len(placeability_points)


def build_workcell_curobo_scene(
    env: UR5EyeInHandPyBulletEnv,
) -> dict[str, object]:
    """Build the shared CuRobo collision scene used by IK and motion planning."""

    return build_curobo_scene_model(
        workpiece_mesh_path=env.generated_workpiece_obj,
        robot_base_xyz=ROBOT_BASE_XYZ,
        table_size_xyz=TABLE_SIZE_XYZ,
        table_center_xyz=TABLE_CENTER_XYZ,
        table_leg_half_extents=TABLE_LEG_HALF_EXTENTS,
        table_leg_positions_xyz=TABLE_LEG_POSITIONS,
        workpiece_base_xyz=WORKPIECE_BASE_XYZ,
    )


def build_selected_candidate_backend(
    env: UR5EyeInHandPyBulletEnv,
    selection: IKBackendSelection,
    curobo_seeds: int = DEFAULT_CUROBO_IK_SEED_COUNT,
) -> IKCandidateBackend | None:
    """Build one reusable CuRobo backend, or ``None`` for built-in PyBullet IK."""

    if not selection.uses_curobo:
        return None
    if selection.curobo_config_path is None:
        raise ValueError("Resolved CuRobo selection is missing its robot config path.")
    scene_model = build_workcell_curobo_scene(env)
    return CuRoboIKCandidateBackend(
        robot_config_path=selection.curobo_config_path,
        scene_model=scene_model,
        world_from_base=make_transform(ROBOT_BASE_XYZ, (0.0, 0.0, 0.0, 1.0)),
        expected_joint_names=ARM_JOINT_ORDER,
        num_seeds=curobo_seeds,
    )


def build_selected_motion_planner(
    env: UR5EyeInHandPyBulletEnv,
    selection: IKBackendSelection,
    clearance_m: float,
    *,
    interpolation_dt_s: float = DEFAULT_MOTION_INTERPOLATION_DT_S,
    max_attempts: int = DEFAULT_MOTION_MAX_ATTEMPTS,
) -> CuRoboMotionPlanner:
    """Build the lazy motion planner for a collision-aware CuRobo selection."""

    if not selection.uses_curobo:
        raise CuRoboMotionPlanningError(
            "Interactive viewpoint motion requires the CuRobo backend."
        )
    if selection.curobo_config_path is None:
        raise CuRoboMotionPlanningError(
            "Resolved CuRobo selection is missing its robot config path."
        )
    return CuRoboMotionPlanner(
        robot_config_path=selection.curobo_config_path,
        scene_model=build_workcell_curobo_scene(env),
        expected_joint_names=ARM_JOINT_ORDER,
        clearance_m=clearance_m,
        interpolation_dt_s=interpolation_dt_s,
        max_attempts=max_attempts,
    )


def run_viewpoint_reachability(
    env: UR5EyeInHandPyBulletEnv,
    count: int,
    radius_min_m: float,
    radius_max_m: float,
    elevation_min_deg: float,
    elevation_max_deg: float,
    config: ReachabilityConfig,
    output_path: Path,
    visualize: bool,
    candidate_backend: IKCandidateBackend | None = None,
    backend_selection_metadata: dict[str, object] | None = None,
) -> ReachabilityReport:
    object_center = env.get_workpiece_center()
    vertical_fov = math.radians(compute_camera_vertical_fov_deg())
    horizontal_fov = 2.0 * math.atan(
        math.tan(vertical_fov * 0.5) * (CAMERA_WIDTH / CAMERA_HEIGHT)
    )
    usable_half_fov = 0.5 * min(vertical_fov, horizontal_fov) * (
        1.0 - DEFAULT_FRAMING_MARGIN_FRACTION
    )
    object_bounding_radius = 0.5 * float(np.linalg.norm(env.workpiece_extent_m))
    framing_radius_min = object_bounding_radius / max(math.sin(usable_half_fov), 1e-9)
    effective_radius_min = max(radius_min_m, framing_radius_min)
    if effective_radius_min > radius_max_m:
        raise ValueError(
            f"--view-radius-max={radius_max_m:.3f} m cannot frame the object's "
            f"{object_bounding_radius:.3f} m bounding radius; need at least "
            f"{framing_radius_min:.3f} m with the {DEFAULT_FRAMING_MARGIN_FRACTION:.0%} margin."
        )
    if effective_radius_min > radius_min_m + 1e-9:
        print(
            f"Raised minimum view radius from {radius_min_m:.3f} to "
            f"{effective_radius_min:.3f} m to keep the object inside the camera frustum."
        )
    viewpoints = sample_fibonacci_viewpoints(
        object_center_xyz=object_center,
        count=count,
        radius_min_m=effective_radius_min,
        radius_max_m=radius_max_m,
        elevation_min_deg=elevation_min_deg,
        elevation_max_deg=elevation_max_deg,
    )
    backend_label = (
        "PyBullet multi-start IK"
        if candidate_backend is None
        else str(getattr(candidate_backend, "backend_name", type(candidate_backend).__name__))
    )
    print(
        f"Filtering {len(viewpoints)} Fibonacci-shell viewpoints with "
        f"{len(config.roll_angles_deg)} roll goals using {backend_label} ..."
    )
    report = env.filter_reachable_viewpoints(
        viewpoints, object_center, config, candidate_backend=candidate_backend
    )
    if backend_selection_metadata is not None:
        report.backend_metadata = dict(report.backend_metadata or {})
        report.backend_metadata["selection"] = dict(backend_selection_metadata)
    report.sampling_metadata = {
        "algorithm": "equal-area_fibonacci_shell_with_radical_inverse_radius",
        "count": count,
        "requested_radius_min_m": radius_min_m,
        "requested_radius_max_m": radius_max_m,
        "effective_radius_min_m": effective_radius_min,
        "effective_radius_max_m": radius_max_m,
        "requested_elevation_min_deg": elevation_min_deg,
        "requested_elevation_max_deg": elevation_max_deg,
        "actual_elevation_min_deg": min(viewpoint.elevation_deg for viewpoint in viewpoints),
        "actual_elevation_max_deg": max(viewpoint.elevation_deg for viewpoint in viewpoints),
        "camera_vertical_fov_deg": math.degrees(vertical_fov),
        "camera_horizontal_fov_deg": math.degrees(horizontal_fov),
        "framing_margin_fraction": DEFAULT_FRAMING_MARGIN_FRACTION,
        "object_bounding_radius_m": object_bounding_radius,
        "framing_required_radius_min_m": framing_radius_min,
    }
    saved_path = save_reachability_report(report, config, env, output_path)
    print(
        f"Reachability result: {len(report.reachable)}/{len(report.sampled)} accepted, "
        f"{len(report.rejected)} rejected in {report.elapsed_sec:.3f} s"
    )
    if config.path_validation_steps > 0:
        path_clear_count = sum(
            solution.linear_path_collision_free is True for solution in report.reachable
        )
        print(
            f"Straight-path diagnostic: {path_clear_count}/{len(report.reachable)} accepted "
            f"endpoints passed {config.path_validation_steps} collision samples."
        )
    print(f"Saved viewpoint report to: {saved_path}")
    if report.reachable:
        best = report.reachable[0]
        print(
            f"Best reachable viewpoint: index={best.viewpoint.index}, "
            f"roll={best.viewpoint.roll_deg:.1f} deg, quality={best.quality_score:.3f}"
        )
    if visualize and env.gui:
        visualize_reachability_report(env, report)
    return report


def parse_roll_angles(value: str) -> tuple[float, ...]:
    try:
        angles = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("IK rolls must be comma-separated numbers.") from exc
    if not angles:
        raise argparse.ArgumentTypeError("At least one IK roll angle is required.")
    return angles


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PyBullet tabletop UR5e scene with an eye-in-hand RGB wrist camera."
    )
    parser.add_argument(
        "--robot-urdf",
        type=Path,
        default=DEFAULT_ROBOT_URDF,
        help="Path to the source UR5e URDF that will be simplified for PyBullet.",
    )
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
    parser.add_argument(
        "--workpiece-unit-scale",
        type=float,
        default=0.001,
        help="Scale factor for the PLY model. Default assumes millimeters.",
    )
    parser.add_argument(
        "--workpiece-rotation-z",
        type=float,
        default=0.0,
        help="Extra Z rotation for the workpiece in degrees.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without the PyBullet GUI and save a single wrist-camera frame before exiting.",
    )
    camera_feed_group = parser.add_mutually_exclusive_group()
    camera_feed_group.add_argument(
        "--show-camera-feed",
        dest="show_camera_feed",
        action="store_true",
        default=True,
        help="Open a live matplotlib wrist-camera preview (default).",
    )
    camera_feed_group.add_argument(
        "--no-camera-feed",
        dest="show_camera_feed",
        action="store_false",
        help="Disable the live wrist-camera preview.",
    )
    capability_group = parser.add_mutually_exclusive_group()
    capability_group.add_argument(
        "--show-capability-map",
        dest="show_capability_map",
        action="store_true",
        default=True,
        help="Show the saved capability map in the GUI (default).",
    )
    capability_group.add_argument(
        "--no-capability-map",
        dest="show_capability_map",
        action="store_false",
        help="Hide the capability-map overlay.",
    )
    parser.add_argument(
        "--capability-map",
        type=Path,
        default=DEFAULT_CAPABILITY_MAP,
        help="NPZ capability map produced by sample/generate_capability_map.py.",
    )
    parser.add_argument(
        "--capability-samples",
        type=int,
        default=DEFAULT_CAPABILITY_SAMPLE_COUNT,
        help="Joint samples used only when the default-on capability map is missing.",
    )
    parser.add_argument(
        "--save-camera-path",
        type=Path,
        default=OUTPUT_DIR / "wrist_camera_latest.png",
        help="Where to save captured wrist-camera frames.",
    )
    parser.add_argument(
        "--camera-update-hz",
        type=float,
        default=5.0,
        help="Live camera preview update rate in GUI mode.",
    )
    parser.add_argument(
        "--plan-viewpoints",
        action="store_true",
        help="Sample 3D viewpoints and remove poses that fail validated IK reachability.",
    )
    parser.add_argument(
        "--ik-backend",
        choices=("auto", "pybullet", "curobo"),
        default="auto",
        help=(
            "IK candidate generator policy: auto prefers CuRobo V1 and falls back "
            "to PyBullet when its GPU runtime is unavailable; explicit curobo is strict."
        ),
    )
    parser.add_argument(
        "--curobo-robot-config",
        type=Path,
        default=None,
        help=(
            "Camera-aware CuRobo YAML: V1 v0.7.8 or V2 v0.8.x schema "
            "(auto-detected; Level 7 also requires collision spheres)."
        ),
    )
    parser.add_argument(
        "--curobo-seeds",
        type=int,
        default=DEFAULT_CUROBO_IK_SEED_COUNT,
        help=(
            "GPU seeds per viewpoint shared across roll goals "
            f"(default: {DEFAULT_CUROBO_IK_SEED_COUNT})."
        ),
    )
    parser.add_argument(
        "--viewpoint-count",
        type=int,
        default=DEFAULT_VIEWPOINT_COUNT,
        help=f"Number of deterministic shell samples (default: {DEFAULT_VIEWPOINT_COUNT}).",
    )
    parser.add_argument(
        "--view-radius-min",
        type=float,
        default=DEFAULT_VIEW_RADIUS_MIN_M,
        help="Minimum camera-to-object-center radius in metres.",
    )
    parser.add_argument(
        "--view-radius-max",
        type=float,
        default=DEFAULT_VIEW_RADIUS_MAX_M,
        help="Maximum camera-to-object-center radius in metres.",
    )
    parser.add_argument(
        "--view-elevation-min",
        type=float,
        default=DEFAULT_VIEW_ELEVATION_MIN_DEG,
        help="Minimum tabletop-relative viewpoint elevation in degrees.",
    )
    parser.add_argument(
        "--view-elevation-max",
        type=float,
        default=DEFAULT_VIEW_ELEVATION_MAX_DEG,
        help="Maximum tabletop-relative viewpoint elevation in degrees.",
    )
    parser.add_argument(
        "--ik-rolls",
        type=parse_roll_angles,
        default=DEFAULT_IK_ROLLS_DEG,
        help=(
            "Comma-separated absolute optical-axis rolls. The default 0 keeps "
            "camera images world-up; alternatives such as 0,90,-90,180 can "
            "increase reachability but permit rotated images."
        ),
    )
    parser.add_argument(
        "--ik-seeds",
        type=int,
        default=DEFAULT_IK_SEED_COUNT,
        help="Number of deterministic joint-space seeds per pose/roll for PyBullet IK.",
    )
    parser.add_argument(
        "--ik-refinement-steps",
        type=int,
        default=DEFAULT_IK_REFINEMENT_STEPS,
        help="Warm-started IK polishing passes for the PyBullet IK backend.",
    )
    parser.add_argument(
        "--ik-position-tolerance",
        type=float,
        default=DEFAULT_IK_POSITION_TOLERANCE_M,
        help="Maximum FK camera-position residual in metres.",
    )
    parser.add_argument(
        "--ik-orientation-tolerance",
        type=float,
        default=DEFAULT_IK_ORIENTATION_TOLERANCE_DEG,
        help="Maximum FK camera-orientation residual in degrees.",
    )
    parser.add_argument(
        "--collision-clearance",
        type=float,
        default=DEFAULT_COLLISION_CLEARANCE_M,
        help="Required robot/camera collision clearance in metres.",
    )
    parser.add_argument(
        "--skip-line-of-sight-check",
        action="store_true",
        help="Do not reject poses whose camera-to-object-center ray is occluded.",
    )
    parser.add_argument(
        "--path-validation-steps",
        type=int,
        default=0,
        help=(
            "Annotate this many collision samples on a straight joint path; "
            "endpoint status stays separate (0 disables)."
        ),
    )
    parser.add_argument(
        "--motion-interpolation-dt",
        type=float,
        default=DEFAULT_MOTION_INTERPOLATION_DT_S,
        help=(
            "Dense CuRobo motion-plan timestep in seconds "
            f"(default: {DEFAULT_MOTION_INTERPOLATION_DT_S})."
        ),
    )
    parser.add_argument(
        "--motion-plan-attempts",
        type=int,
        default=DEFAULT_MOTION_MAX_ATTEMPTS,
        help=(
            "Maximum CuRobo attempts for an interactive selected viewpoint "
            f"(default: {DEFAULT_MOTION_MAX_ATTEMPTS})."
        ),
    )
    parser.add_argument(
        "--viewpoint-output",
        type=Path,
        default=DEFAULT_VIEWPOINT_OUTPUT,
        help="JSON output containing all accepted/rejected viewpoints and diagnostics.",
    )
    parser.add_argument(
        "--visualize-viewpoints",
        action="store_true",
        help="Show reachable points in green and rejected points in red in the PyBullet GUI.",
    )
    return parser


def reachability_config_from_args(
    args: argparse.Namespace,
    effective_ik_backend: str | None = None,
) -> ReachabilityConfig:
    """Create the single validated configuration used by planning and execution."""

    selected_backend = effective_ik_backend or args.ik_backend
    if selected_backend == "auto":
        raise ValueError(
            "The effective IK backend must be resolved before constructing the "
            "reachability configuration for --ik-backend=auto."
        )
    if selected_backend not in ("pybullet", "curobo"):
        raise ValueError(f"Unknown effective IK backend: {selected_backend!r}")
    using_curobo = selected_backend == "curobo"
    return ReachabilityConfig(
        roll_angles_deg=args.ik_rolls,
        seed_count=args.curobo_seeds if using_curobo else args.ik_seeds,
        refinement_steps=0 if using_curobo else args.ik_refinement_steps,
        position_tolerance_m=args.ik_position_tolerance,
        orientation_tolerance_deg=args.ik_orientation_tolerance,
        clearance_m=args.collision_clearance,
        require_line_of_sight=not args.skip_line_of_sight_check,
        path_validation_steps=args.path_validation_steps,
    )


def validate_cli_arguments(args: argparse.Namespace) -> None:
    if not math.isfinite(args.workpiece_unit_scale) or args.workpiece_unit_scale <= 0.0:
        raise ValueError("--workpiece-unit-scale must be finite and positive.")
    if not math.isfinite(args.workpiece_rotation_z):
        raise ValueError("--workpiece-rotation-z must be finite.")
    if not math.isfinite(args.camera_update_hz) or args.camera_update_hz <= 0.0:
        raise ValueError("--camera-update-hz must be finite and positive.")
    if args.capability_samples <= 0:
        raise ValueError("--capability-samples must be positive.")

    if not args.plan_viewpoints:
        return

    if args.viewpoint_count <= 0:
        raise ValueError("--viewpoint-count must be positive.")
    sampling_values = (
        args.view_radius_min,
        args.view_radius_max,
        args.view_elevation_min,
        args.view_elevation_max,
    )
    if not all(math.isfinite(value) for value in sampling_values):
        raise ValueError("View radii and elevations must be finite.")
    if args.view_radius_min <= 0.0 or args.view_radius_max < args.view_radius_min:
        raise ValueError("View radii must satisfy 0 < minimum <= maximum.")
    if not (-90.0 < args.view_elevation_min <= args.view_elevation_max < 90.0):
        raise ValueError("View elevations must satisfy -90 < minimum <= maximum < 90 degrees.")

    if args.ik_backend in ("auto", "pybullet"):
        if args.ik_seeds <= 0:
            raise ValueError("--ik-seeds must be positive.")
        if args.ik_refinement_steps < 0:
            raise ValueError("--ik-refinement-steps must be non-negative.")
    if args.ik_backend in ("auto", "curobo") and args.curobo_seeds <= 0:
        raise ValueError("--curobo-seeds must be positive.")
    if (
        not math.isfinite(args.motion_interpolation_dt)
        or args.motion_interpolation_dt <= 0.0
    ):
        raise ValueError("--motion-interpolation-dt must be finite and positive.")
    if args.motion_plan_attempts <= 0:
        raise ValueError("--motion-plan-attempts must be positive.")

    if args.ik_backend == "curobo":
        if args.curobo_robot_config is None:
            raise ValueError(
                "--curobo-robot-config is required when --ik-backend=curobo. "
                "Generate it with sample/prepare_curobo_robot.py first."
            )
        if not args.curobo_robot_config.exists():
            raise ValueError(f"CuRobo robot config not found: {args.curobo_robot_config}")

    validation_backend = "pybullet" if args.ik_backend == "auto" else args.ik_backend
    reachability_config_from_args(
        args, effective_ik_backend=validation_backend
    ).validate()


def choose_camera_image_save_path(suggested_path: Path) -> Path | None:
    """Open a native save dialog, returning ``None`` when it is cancelled."""

    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:  # pragma: no cover - depends on host GUI packages
        print(
            "Could not open the camera-image save dialog because Tk is not installed. "
            "Install python3-tk or use the 'c' shortcut instead."
        )
        print(f"Tk import error: {exc}")
        return None

    suggested_path = suggested_path.expanduser()
    initial_directory = suggested_path.parent.resolve()
    if not initial_directory.is_dir():
        initial_directory = Path.cwd()

    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        root.update_idletasks()
        selected_path = filedialog.asksaveasfilename(
            parent=root,
            title="Save wrist-camera image",
            initialdir=initial_directory.as_posix(),
            initialfile=suggested_path.name,
            defaultextension=".png",
            filetypes=(
                ("PNG image", "*.png"),
                ("JPEG image", "*.jpg *.jpeg"),
                ("All files", "*.*"),
            ),
        )
    except Exception as exc:  # pragma: no cover - display/window-manager dependent
        print(f"Could not open the camera-image save dialog: {exc}")
        return None
    finally:
        if root is not None:
            root.destroy()

    if not selected_path:
        return None
    return Path(selected_path).expanduser()


def save_wrist_camera_frame_with_dialog(
    env: UR5EyeInHandPyBulletEnv,
    suggested_path: Path,
) -> Path | None:
    """Ask for an output path and save the current wrist-camera RGB frame."""

    selected_path = choose_camera_image_save_path(suggested_path)
    if selected_path is None:
        print("Camera-image save cancelled.")
        return None
    try:
        return env.save_wrist_camera_frame(selected_path)
    except (OSError, ValueError) as exc:
        print(f"Could not save wrist-camera RGB frame to '{selected_path}': {exc}")
        return None


def project_world_point_to_screen(
    point_xyz: tuple[float, float, float],
    view_matrix: tuple[float, ...] | list[float],
    projection_matrix: tuple[float, ...] | list[float],
    viewport_width: int,
    viewport_height: int,
) -> tuple[float, float] | None:
    """Project one world point using PyBullet's column-major OpenGL matrices."""

    if viewport_width <= 0 or viewport_height <= 0:
        return None
    view = np.asarray(view_matrix, dtype=float).reshape(4, 4, order="F")
    projection = np.asarray(projection_matrix, dtype=float).reshape(4, 4, order="F")
    homogeneous = np.asarray((*point_xyz, 1.0), dtype=float)
    clip = projection @ view @ homogeneous
    if not np.all(np.isfinite(clip)) or clip[3] <= 1e-9:
        return None
    normalized = clip[:3] / clip[3]
    if (
        normalized[0] < -1.0
        or normalized[0] > 1.0
        or normalized[1] < -1.0
        or normalized[1] > 1.0
        or normalized[2] < -1.0
        or normalized[2] > 1.0
    ):
        return None
    screen_x = 0.5 * (normalized[0] + 1.0) * viewport_width
    screen_y = 0.5 * (1.0 - normalized[1]) * viewport_height
    return float(screen_x), float(screen_y)


def select_nearest_viewpoint_from_screen(
    report: ReachabilityReport,
    mouse_x: float,
    mouse_y: float,
    camera_info: tuple[object, ...],
    *,
    selection_radius_pixels: float = 18.0,
) -> ViewpointCandidate | None:
    """Return the visible report point nearest a viewport click."""

    if len(camera_info) < 4:
        raise ValueError("PyBullet camera information is incomplete.")
    width, height = int(camera_info[0]), int(camera_info[1])
    view_matrix, projection_matrix = camera_info[2], camera_info[3]
    best: ViewpointCandidate | None = None
    best_distance = float(selection_radius_pixels)
    for viewpoint in report.sampled:
        projected = project_world_point_to_screen(
            viewpoint.position_xyz,
            view_matrix,  # type: ignore[arg-type]
            projection_matrix,  # type: ignore[arg-type]
            width,
            height,
        )
        if projected is None:
            continue
        distance = math.hypot(projected[0] - mouse_x, projected[1] - mouse_y)
        if distance <= best_distance:
            best = viewpoint
            best_distance = distance
    return best


class InteractiveSceneCamera:
    """Modifier-free mouse navigation for PyBullet's debug visualizer.

    PyBullet's built-in camera gestures require Alt or Ctrl on several
    platforms.  Handling the raw mouse events here gives the simulator the
    conventional drag controls advertised by this application and also lets
    us distinguish a viewpoint click from an orbit drag.
    """

    MOUSE_MOVE_EVENT = 1
    MOUSE_BUTTON_EVENT = 2
    ORBIT_BUTTON = 0
    PAN_BUTTONS = frozenset((1, 2))

    def __init__(
        self,
        physics,
        *,
        orbit_degrees_per_pixel: float = 0.3,
        pan_distance_per_pixel: float = 0.0015,
        drag_threshold_pixels: float = 3.0,
    ) -> None:
        self.physics = physics
        self.orbit_degrees_per_pixel = float(orbit_degrees_per_pixel)
        self.pan_distance_per_pixel = float(pan_distance_per_pixel)
        self.drag_threshold_pixels = float(drag_threshold_pixels)
        self._key_is_down = int(getattr(physics, "KEY_IS_DOWN", 1))
        self._key_was_triggered = int(getattr(physics, "KEY_WAS_TRIGGERED", 2))
        self._key_was_released = int(getattr(physics, "KEY_WAS_RELEASED", 4))
        self._pressed_buttons: set[int] = set()
        self._press_positions: dict[int, tuple[float, float]] = {}
        self._dragged_buttons: set[int] = set()
        self._last_pointer: tuple[float, float] | None = None

    def process(
        self,
        mouse_events: Iterable[tuple[object, ...]],
    ) -> list[tuple[float, float]]:
        """Apply camera drags and return left-button clicks in screen pixels."""

        clicks: list[tuple[float, float]] = []
        for mouse_event in mouse_events:
            if len(mouse_event) < 5:
                continue
            event_type, raw_x, raw_y, raw_button, raw_state = mouse_event[:5]
            try:
                event_type = int(event_type)
                mouse_x = float(raw_x)
                mouse_y = float(raw_y)
                button = int(raw_button)
                button_state = int(raw_state)
            except (TypeError, ValueError):
                continue

            if event_type == self.MOUSE_MOVE_EVENT:
                self._handle_pointer_move(mouse_x, mouse_y)
                continue
            if event_type != self.MOUSE_BUTTON_EVENT:
                continue
            if button != self.ORBIT_BUTTON and button not in self.PAN_BUTTONS:
                continue

            position = (mouse_x, mouse_y)
            if button_state & (self._key_was_triggered | self._key_is_down):
                if button not in self._pressed_buttons:
                    self._pressed_buttons.add(button)
                    self._press_positions[button] = position
                    self._dragged_buttons.discard(button)
                self._last_pointer = position

            if button_state & self._key_was_released:
                origin = self._press_positions.get(button, position)
                travelled = math.hypot(mouse_x - origin[0], mouse_y - origin[1])
                was_drag = (
                    button in self._dragged_buttons
                    or travelled > self.drag_threshold_pixels
                )
                if button == self.ORBIT_BUTTON and not was_drag:
                    clicks.append(position)
                self._pressed_buttons.discard(button)
                self._press_positions.pop(button, None)
                self._dragged_buttons.discard(button)
                self._last_pointer = position
        return clicks

    def _handle_pointer_move(self, mouse_x: float, mouse_y: float) -> None:
        current = (mouse_x, mouse_y)
        previous = self._last_pointer
        self._last_pointer = current
        if previous is None or not self._pressed_buttons:
            return

        delta_x = mouse_x - previous[0]
        delta_y = mouse_y - previous[1]
        for button in self._pressed_buttons:
            origin = self._press_positions.get(button, current)
            if (
                math.hypot(mouse_x - origin[0], mouse_y - origin[1])
                > self.drag_threshold_pixels
            ):
                self._dragged_buttons.add(button)

        if (
            self.ORBIT_BUTTON in self._pressed_buttons
            and self.ORBIT_BUTTON in self._dragged_buttons
        ):
            self._orbit(delta_x, delta_y)
        elif any(
            button in self._dragged_buttons
            for button in self._pressed_buttons & self.PAN_BUTTONS
        ):
            self._pan(delta_x, delta_y)

    def _camera_state(
        self,
    ) -> tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray] | None:
        camera_info = tuple(self.physics.getDebugVisualizerCamera())
        if len(camera_info) < 12:
            return None
        try:
            horizontal = np.asarray(camera_info[6], dtype=float)
            vertical = np.asarray(camera_info[7], dtype=float)
            yaw = float(camera_info[8])
            pitch = float(camera_info[9])
            distance = float(camera_info[10])
            target = np.asarray(camera_info[11], dtype=float)
        except (TypeError, ValueError):
            return None
        if (
            horizontal.shape != (3,)
            or vertical.shape != (3,)
            or target.shape != (3,)
            or not np.all(np.isfinite(horizontal))
            or not np.all(np.isfinite(vertical))
            or not np.all(np.isfinite(target))
            or not all(math.isfinite(value) for value in (yaw, pitch, distance))
        ):
            return None
        return yaw, pitch, max(distance, 1e-3), target, horizontal, vertical

    def _orbit(self, delta_x: float, delta_y: float) -> None:
        state = self._camera_state()
        if state is None:
            return
        yaw, pitch, distance, target, _, _ = state
        yaw -= delta_x * self.orbit_degrees_per_pixel
        pitch = float(
            np.clip(
                pitch - delta_y * self.orbit_degrees_per_pixel,
                -89.0,
                89.0,
            )
        )
        self._reset_camera(distance, yaw, pitch, target)

    def _pan(self, delta_x: float, delta_y: float) -> None:
        state = self._camera_state()
        if state is None:
            return
        yaw, pitch, distance, target, horizontal, vertical = state
        right = normalize(horizontal)
        up = normalize(vertical)
        if np.linalg.norm(right) < 1e-9 or np.linalg.norm(up) < 1e-9:
            return
        scale = distance * self.pan_distance_per_pixel
        target = target - delta_x * scale * right + delta_y * scale * up
        self._reset_camera(distance, yaw, pitch, target)

    def _reset_camera(
        self,
        distance: float,
        yaw: float,
        pitch: float,
        target: np.ndarray,
    ) -> None:
        self.physics.resetDebugVisualizerCamera(
            cameraDistance=distance,
            cameraYaw=yaw,
            cameraPitch=pitch,
            cameraTargetPosition=target.astype(float).tolist(),
        )


class PyBulletStatusDisplay:
    """World-space status text mirrored to stdout for interactive planning."""

    COLORS = {
        "info": (0.20, 0.65, 1.00),
        "success": (0.15, 0.90, 0.25),
        "warning": (1.00, 0.75, 0.10),
        "error": (1.00, 0.20, 0.15),
    }

    def __init__(self, env: UR5EyeInHandPyBulletEnv) -> None:
        self.env = env
        self.debug_item_id: int | None = None

    def set(self, message: str, level: str = "info") -> None:
        if self.debug_item_id is not None:
            self.env.physics.removeUserDebugItem(self.debug_item_id)
        color = self.COLORS.get(level, self.COLORS["info"])
        self.debug_item_id = self.env.physics.addUserDebugText(
            text=message,
            textPosition=(TABLE_CENTER_XYZ[0] - 0.55, TABLE_CENTER_XYZ[1] - 0.55, 1.35),
            textColorRGB=color,
            textSize=1.25,
            lifeTime=0.0,
        )
        print(f"Viewpoint motion: {message}")


def _plan_motion_in_worker(
    planner: CuRoboMotionPlanner,
    viewpoint_index: int,
    start_joint_positions_rad: list[float],
    goal_joint_positions_rad: list[float],
    output_queue: "queue.Queue[MotionPlanningWorkerResult]",
) -> None:
    try:
        plan = planner.plan(start_joint_positions_rad, goal_joint_positions_rad)
    except (CuRoboMotionInitializationError, CuRoboUnavailableError) as exc:
        output_queue.put(
            MotionPlanningWorkerResult(
                viewpoint_index=viewpoint_index,
                error=f"{type(exc).__name__}: {exc}",
                failure_kind="planner_unavailable",
            )
        )
    except CuRoboNoMotionPlanError as exc:
        output_queue.put(
            MotionPlanningWorkerResult(
                viewpoint_index=viewpoint_index,
                error=f"{type(exc).__name__}: {exc}",
                failure_kind="no_path",
            )
        )
    except Exception as exc:
        output_queue.put(
            MotionPlanningWorkerResult(
                viewpoint_index=viewpoint_index,
                error=f"{type(exc).__name__}: {exc}",
                failure_kind="planning_error",
            )
        )
    else:
        output_queue.put(
            MotionPlanningWorkerResult(viewpoint_index=viewpoint_index, plan=plan)
        )


def run_interactive_demo(
    env: UR5EyeInHandPyBulletEnv,
    show_camera_feed: bool,
    save_camera_path: Path,
    camera_update_hz: float,
    reachability_report: ReachabilityReport | None = None,
    motion_planner: CuRoboMotionPlanner | None = None,
    motion_planner_unavailable_reason: str | None = None,
    collision_clearance_m: float = DEFAULT_COLLISION_CLEARANCE_M,
) -> None:
    physics = env.physics
    scene_camera = InteractiveSceneCamera(physics)
    preview = None
    if show_camera_feed:
        try:
            preview = CameraPreview(CAMERA_WIDTH, CAMERA_HEIGHT)
        except Exception as exc:  # pragma: no cover - optional GUI dependency path
            print(f"Could not start live camera preview window: {exc}")

    sliders: dict[str, int] = {}
    for joint_name in ARM_JOINT_ORDER:
        joint_index = env.arm_joint_name_to_index[joint_name]
        joint_info = physics.getJointInfo(env.robot_id, joint_index)
        lower_limit = float(joint_info[8])
        upper_limit = float(joint_info[9])
        default_value = HOME_JOINT_POSITIONS_RAD[joint_name]
        sliders[joint_name] = physics.addUserDebugParameter(
            paramName=joint_name,
            rangeMin=lower_limit,
            rangeMax=upper_limit,
            startValue=default_value,
        )

    save_button = physics.addUserDebugParameter(
        paramName="Save wrist-camera image...",
        rangeMin=1.0,
        rangeMax=0.0,
        startValue=0.0,
    )
    save_button_click_count = float(physics.readUserDebugParameter(save_button))

    reachable_by_index = {
        solution.viewpoint.index: solution
        for solution in (reachability_report.reachable if reachability_report else [])
    }
    rejected_by_index = {
        rejection.viewpoint.index: rejection
        for rejection in (reachability_report.rejected if reachability_report else [])
    }
    move_button: int | None = None
    move_button_click_count = 0.0
    status_display: PyBulletStatusDisplay | None = None
    selected_viewpoint_index: int | None = None
    selected_marker_id: int | None = None
    plan_results: "queue.Queue[MotionPlanningWorkerResult]" = queue.Queue()
    planning_in_progress = False
    planning_start_joint_positions: list[float] | None = None
    trajectory_player: TrajectoryPlayer | None = None
    trajectory_latched = False
    slider_snapshot_at_trajectory_end: list[float] = []

    if reachability_report is not None:
        move_button = physics.addUserDebugParameter(
            paramName="Move to selected viewpoint (CuRobo)",
            rangeMin=1.0,
            rangeMax=0.0,
            startValue=0.0,
        )
        move_button_click_count = float(physics.readUserDebugParameter(move_button))
        status_display = PyBulletStatusDisplay(env)
        status_display.set("Click a viewpoint, then press Move.")
        if motion_planner_unavailable_reason:
            print(
                "Interactive CuRobo motion is currently unavailable: "
                f"{motion_planner_unavailable_reason}"
            )

    print("PyBullet GUI controls:")
    print("- left-drag in the scene to orbit the view")
    print("- right-drag or middle-drag to pan; use the mouse wheel to zoom")
    print("- use the six sliders to move the UR5e joints")
    if reachability_report is not None:
        print("- click a displayed viewpoint to select it")
        print("- click 'Move to selected viewpoint (CuRobo)' to plan and move")
    print("- click 'Save wrist-camera image...' to choose where to save a frame")
    print("- press 'c' to save a wrist-camera frame")
    print("- press 'h' to return to the home pose")
    print("- press 'r' to reset the scene camera")
    print("- press 'q' to quit")

    last_preview_time = 0.0
    update_period = 1.0 / max(0.5, camera_update_hz)
    home_latched = False
    slider_snapshot_at_home = [
        float(physics.readUserDebugParameter(sliders[name]))
        for name in ARM_JOINT_ORDER
    ]

    while physics.isConnected():
        current_time = time.monotonic()
        scene_clicks = scene_camera.process(physics.getMouseEvents())
        slider_values = [
            float(physics.readUserDebugParameter(sliders[name]))
            for name in ARM_JOINT_ORDER
        ]

        if (
            reachability_report is not None
            and not planning_in_progress
            and trajectory_player is None
        ):
            for mouse_x, mouse_y in scene_clicks:
                selected = select_nearest_viewpoint_from_screen(
                    reachability_report,
                    mouse_x,
                    mouse_y,
                    tuple(physics.getDebugVisualizerCamera()),
                )
                if selected is None:
                    continue
                selected_viewpoint_index = selected.index
                if selected_marker_id is not None:
                    physics.removeUserDebugItem(selected_marker_id)
                selected_marker_id = physics.addUserDebugPoints(
                    [list(selected.position_xyz)],
                    [[1.0, 0.85, 0.05]],
                    pointSize=14.0,
                    lifeTime=0.0,
                )
                if status_display is not None:
                    reachable_solution = reachable_by_index.get(selected.index)
                    if reachable_solution is not None:
                        status_display.set(
                            f"Selected reachable point {selected.index} "
                            f"(camera roll {reachable_solution.viewpoint.roll_deg:+.0f} deg). "
                            "Press Move."
                        )
                    else:
                        rejection = rejected_by_index.get(selected.index)
                        reason = (
                            rejection.furthest_failure_stage
                            if rejection is not None
                            else "no valid IK endpoint"
                        )
                        status_display.set(
                            f"Point {selected.index} is not reachable ({reason}). "
                            "Select a green point.",
                            "error",
                        )

        if planning_in_progress:
            try:
                worker_result = plan_results.get_nowait()
            except queue.Empty:
                worker_result = None
            if worker_result is not None:
                planning_in_progress = False
                planning_start_joint_positions = None
                target_solution = reachable_by_index.get(worker_result.viewpoint_index)
                if worker_result.error is not None:
                    if status_display is not None:
                        if worker_result.failure_kind == "no_path":
                            status_display.set(
                                "No collision-free path to point "
                                f"{worker_result.viewpoint_index}. Select a different "
                                "green point.",
                                "error",
                            )
                        elif worker_result.failure_kind == "planner_unavailable":
                            status_display.set(
                                "CuRobo could not start. Fix the planner environment; "
                                "see the console.",
                                "error",
                            )
                        else:
                            status_display.set(
                                "CuRobo motion planning failed; see the console.", "error"
                            )
                    print(f"CuRobo motion planning error: {worker_result.error}")
                elif worker_result.plan is None or target_solution is None:
                    if status_display is not None:
                        status_display.set(
                            "Motion result was incomplete. Select a different point.", "error"
                        )
                else:
                    endpoint_error = float(
                        np.max(
                            np.abs(
                                worker_result.plan.joint_positions_rad[-1]
                                - np.asarray(target_solution.joint_positions_rad, dtype=float)
                            )
                        )
                    )
                    if endpoint_error > 0.025:
                        if status_display is not None:
                            status_display.set(
                                "CuRobo path ended too far from the validated endpoint. "
                                "Select a different point.",
                                "error",
                            )
                    else:
                        if status_display is not None:
                            status_display.set(
                                f"Revalidating path to point {worker_result.viewpoint_index}..."
                            )
                        validation = env.validate_joint_trajectory(
                            worker_result.plan.joint_positions_rad,
                            collision_clearance_m,
                        )
                        if not validation.valid:
                            if status_display is not None:
                                status_display.set(
                                    "PyBullet rejected the CuRobo path: "
                                    f"{validation.failure_reason}. Select a different point.",
                                    "error",
                                )
                        else:
                            trajectory_player = TrajectoryPlayer(
                                plan=worker_result.plan,
                                start_time=time.monotonic(),
                            )
                            trajectory_latched = False
                            if status_display is not None:
                                status_display.set(
                                    f"Moving to point {worker_result.viewpoint_index}: "
                                    f"{worker_result.plan.waypoint_count} waypoints, "
                                    f"{worker_result.plan.duration_s:.2f} s.",
                                    "success",
                                )

        if trajectory_player is not None:
            commanded_positions, complete = trajectory_player.sample(current_time)
            env.command_joint_positions(commanded_positions, blend=1.0)
            if complete:
                completed_plan = trajectory_player.plan
                trajectory_player = None
                trajectory_latched = True
                slider_snapshot_at_trajectory_end = list(slider_values)
                if status_display is not None:
                    status_display.set(
                        f"Reached point {selected_viewpoint_index}. "
                        "Select another point or move a joint slider.",
                        "success",
                    )
                print(
                    "Completed validated CuRobo trajectory: "
                    f"{completed_plan.waypoint_count} waypoints, "
                    f"minimum playback duration {completed_plan.duration_s:.3f} s."
                )
        elif planning_in_progress and planning_start_joint_positions is not None:
            env.command_joint_positions(planning_start_joint_positions, blend=1.0)
        elif trajectory_latched:
            slider_delta = max(
                abs(current - remembered)
                for current, remembered in zip(
                    slider_values, slider_snapshot_at_trajectory_end
                )
            )
            if slider_delta > 1e-3:
                trajectory_latched = False
                env.command_joint_positions(slider_values)
        else:
            if home_latched:
                slider_delta = max(
                    abs(current - remembered)
                    for current, remembered in zip(slider_values, slider_snapshot_at_home)
                )
                if slider_delta > 1e-3:
                    home_latched = False
                else:
                    slider_values = [
                        HOME_JOINT_POSITIONS_RAD[name] for name in ARM_JOINT_ORDER
                    ]
            env.command_joint_positions(slider_values)

        current_save_button_click_count = float(
            physics.readUserDebugParameter(save_button)
        )
        if current_save_button_click_count != save_button_click_count:
            save_button_click_count = current_save_button_click_count
            timestamp = int(time.time())
            suggested_path = save_camera_path.with_name(
                f"{save_camera_path.stem}_{timestamp}{save_camera_path.suffix or '.png'}"
            )
            saved_path = save_wrist_camera_frame_with_dialog(env, suggested_path)
            if saved_path is not None:
                print(f"Saved wrist-camera RGB frame to: {saved_path}")

        if move_button is not None:
            current_move_button_click_count = float(
                physics.readUserDebugParameter(move_button)
            )
            if current_move_button_click_count != move_button_click_count:
                move_button_click_count = current_move_button_click_count
                if planning_in_progress:
                    if status_display is not None:
                        status_display.set("CuRobo is already planning; please wait.", "warning")
                elif trajectory_player is not None:
                    if status_display is not None:
                        status_display.set("The robot is already moving; please wait.", "warning")
                elif selected_viewpoint_index is None:
                    if status_display is not None:
                        status_display.set(
                            "No point selected. Click a displayed point first.", "warning"
                        )
                elif selected_viewpoint_index not in reachable_by_index:
                    if status_display is not None:
                        status_display.set(
                            f"Point {selected_viewpoint_index} is not reachable. "
                            "Select a green point.",
                            "error",
                        )
                elif motion_planner is None:
                    if status_display is not None:
                        status_display.set(
                            "CuRobo motion planning is unavailable; see the console.",
                            "error",
                        )
                    if motion_planner_unavailable_reason:
                        print(motion_planner_unavailable_reason)
                else:
                    target = reachable_by_index[selected_viewpoint_index]
                    planning_start_joint_positions = env.get_joint_positions()
                    planning_in_progress = True
                    trajectory_latched = False
                    if status_display is not None:
                        status_display.set(
                            f"CuRobo is planning to point {selected_viewpoint_index}..."
                        )
                    threading.Thread(
                        target=_plan_motion_in_worker,
                        args=(
                            motion_planner,
                            selected_viewpoint_index,
                            list(planning_start_joint_positions),
                            list(target.joint_positions_rad),
                            plan_results,
                        ),
                        name="curobo-viewpoint-motion",
                        daemon=True,
                    ).start()

        keyboard_events = physics.getKeyboardEvents()
        if (
            ord("q") in keyboard_events
            and keyboard_events[ord("q")] & physics.KEY_WAS_TRIGGERED
        ):
            break
        if (
            ord("h") in keyboard_events
            and keyboard_events[ord("h")] & physics.KEY_WAS_TRIGGERED
        ):
            if planning_in_progress:
                if status_display is not None:
                    status_display.set(
                        "Wait for the active CuRobo planning request to finish.", "warning"
                    )
            else:
                trajectory_player = None
                trajectory_latched = False
                env.reset_to_home_pose()
                home_latched = True
                slider_snapshot_at_home = [
                    float(physics.readUserDebugParameter(sliders[name]))
                    for name in ARM_JOINT_ORDER
                ]
                print("Robot reset to the home pose. Move any slider to leave home mode.")
        if (
            ord("r") in keyboard_events
            and keyboard_events[ord("r")] & physics.KEY_WAS_TRIGGERED
        ):
            env.reset_scene_camera()
        if (
            ord("c") in keyboard_events
            and keyboard_events[ord("c")] & physics.KEY_WAS_TRIGGERED
        ):
            timestamp = int(time.time())
            output_path = save_camera_path.with_name(
                f"{save_camera_path.stem}_{timestamp}{save_camera_path.suffix}"
            )
            saved_path = env.save_wrist_camera_frame(output_path)
            print(f"Saved wrist-camera RGB frame to: {saved_path}")

        physics.stepSimulation()
        env.hold_current_pose()

        if preview is not None and (current_time - last_preview_time) >= update_period:
            preview.update(env.capture_wrist_camera().rgb)
            last_preview_time = current_time

        time.sleep(1.0 / 240.0)


def run_headless_snapshot(env: UR5EyeInHandPyBulletEnv, save_camera_path: Path) -> None:
    env.reset_to_home_pose()
    for _ in range(60):
        env.physics.stepSimulation()
        env.hold_current_pose()
    saved_path = env.save_wrist_camera_frame(save_camera_path)
    print(f"Saved headless wrist-camera RGB frame to: {saved_path}")


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    try:
        validate_cli_arguments(args)
    except ValueError as exc:
        parser.error(str(exc))

    if not args.robot_urdf.exists():
        raise FileNotFoundError(f"Robot URDF not found: {args.robot_urdf}")
    try:
        workpiece_ply = resolve_workpiece_ply(args.object, args.workpiece_ply)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    if not workpiece_ply.is_file():
        raise FileNotFoundError(f"Workpiece PLY not found: {workpiece_ply}")
    backend_selection: IKBackendSelection | None = None
    if args.plan_viewpoints:
        try:
            backend_selection = resolve_ik_backend(
                args.ik_backend,
                args.curobo_robot_config,
            )
        except (CuRoboUnavailableError, FileNotFoundError, ValueError) as exc:
            parser.error(str(exc))

    env = UR5EyeInHandPyBulletEnv(
        robot_urdf=args.robot_urdf.resolve(),
        workpiece_ply=workpiece_ply.resolve(),
        workpiece_unit_scale=args.workpiece_unit_scale,
        workpiece_rotation_z_deg=args.workpiece_rotation_z,
        gui=not args.headless,
    )
    env.print_scene_summary()

    reachability_report: ReachabilityReport | None = None
    reachability_config: ReachabilityConfig | None = None
    motion_planner: CuRoboMotionPlanner | None = None
    motion_planner_unavailable_reason: str | None = None
    try:
        if args.show_capability_map and not args.headless:
            capability_path = args.capability_map.expanduser().resolve()
            if capability_path.is_file():
                capability_map = load_capability_map(capability_path)
            else:
                print(
                    f"Capability map not found; generating {args.capability_samples} "
                    "joint samples in PyBullet..."
                )
                capability_map = generate_capability_map(
                    env, samples=args.capability_samples
                )
                save_capability_map(capability_path, capability_map)
                print(f"Saved capability map to: {capability_path}")
            workspace_count, placeability_count = visualize_capability_map(
                env, capability_map
            )
            print(
                "Capability map: "
                f"{workspace_count} workspace voxels, "
                f"{placeability_count} planar base-placement voxels."
            )

        if args.plan_viewpoints:
            if backend_selection is None:  # pragma: no cover - defensive narrowing
                raise RuntimeError("IK backend selection was not resolved.")
            candidate_backend = build_selected_candidate_backend(
                env,
                backend_selection,
                curobo_seeds=args.curobo_seeds,
            )

            reachability_config = reachability_config_from_args(
                args,
                effective_ik_backend=backend_selection.effective_backend,
            )
            reachability_report = run_viewpoint_reachability(
                env=env,
                count=args.viewpoint_count,
                radius_min_m=args.view_radius_min,
                radius_max_m=args.view_radius_max,
                elevation_min_deg=args.view_elevation_min,
                elevation_max_deg=args.view_elevation_max,
                config=reachability_config,
                output_path=args.viewpoint_output.resolve(),
                visualize=args.visualize_viewpoints or not args.headless,
                candidate_backend=candidate_backend,
                backend_selection_metadata=backend_selection.report_metadata(),
            )
            if not args.headless:
                try:
                    motion_planner = build_selected_motion_planner(
                        env,
                        backend_selection,
                        reachability_config.clearance_m,
                        interpolation_dt_s=args.motion_interpolation_dt,
                        max_attempts=args.motion_plan_attempts,
                    )
                except (CuRoboMotionPlanningError, FileNotFoundError, ValueError) as exc:
                    motion_planner_unavailable_reason = str(exc)

        if args.headless:
            if not args.plan_viewpoints:
                run_headless_snapshot(env, args.save_camera_path.resolve())
        else:
            run_interactive_demo(
                env=env,
                show_camera_feed=args.show_camera_feed,
                save_camera_path=args.save_camera_path.resolve(),
                camera_update_hz=args.camera_update_hz,
                reachability_report=reachability_report,
                motion_planner=motion_planner,
                motion_planner_unavailable_reason=motion_planner_unavailable_reason,
                collision_clearance_m=(
                    reachability_config.clearance_m
                    if reachability_config is not None
                    else DEFAULT_COLLISION_CLEARANCE_M
                ),
            )
    finally:
        env.disconnect()


if __name__ == "__main__":
    main()
