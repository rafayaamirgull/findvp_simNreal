"""Optional CuRobo candidate generation for viewpoint reachability.

This module is deliberately import-safe on machines without CuRobo or CUDA. It
contains pure pose/scene/URDF helpers, while all torch and CuRobo imports happen
only when automatic or explicit backend selection preflights CuRobo. Two
intentionally separate adapters are supported:

* CuRobo V1 ``v0.7.8`` for CUDA 12.2 / NVIDIA R535 systems.  It proposes IK
  branches only; the shared PyBullet validator remains responsible for every
  collision, clearance, FK, and line-of-sight acceptance decision.
* CuRoboV2 ``v0.8.x`` for current R580+ systems.  It additionally uses CuRobo's
  collision-aware optimizer before the same PyBullet revalidation.
"""

from __future__ import annotations

import copy
import importlib
import importlib.metadata
import math
import platform
import re
import subprocess
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

if __package__:
    from .viewpoint_reachability import (
        IKCandidate,
        IKProblem,
        IKProblemResult,
        ReachabilityConfig,
        matrix_to_quaternion_xyzw,
        quaternion_xyzw_to_matrix,
    )
else:  # Support direct execution/import from sample/.
    from viewpoint_reachability import (
        IKCandidate,
        IKProblem,
        IKProblemResult,
        ReachabilityConfig,
        matrix_to_quaternion_xyzw,
        quaternion_xyzw_to_matrix,
    )


CUROBO_DISTRIBUTION = "nvidia-curobo"
CUROBO_API_V1 = "v1"
CUROBO_API_V2 = "v2"
SUPPORTED_CUROBO_V1_VERSION = (0, 7, 8)
SUPPORTED_CUROBO_V2_MAJOR_MINOR = (0, 8)
MINIMUM_PYTHON_V1 = (3, 8)
MINIMUM_PYTHON_V2 = (3, 10)
MINIMUM_NVIDIA_DRIVER_V1 = (535, 54, 3)
MINIMUM_NVIDIA_DRIVER_V2 = (580, 65, 6)
SUPPORTED_V1_TORCH_CUDA_MAJOR_MINOR = (12, 2)
MINIMUM_COMPUTE_CAPABILITY_V1 = (8, 6)
MINIMUM_COMPUTE_CAPABILITY_V2 = (8, 0)
DEFAULT_CUROBO_SEEDS = 32
DEFAULT_CUROBO_RANDOM_SEED = 123
DEFAULT_CUROBO_TOOL_FRAME = "ur5e_camera_optical_frame"

CUROBO_V1_UR5E_LINK_MAP = {
    "base_link": "ur5e_base_link",
    "base_link_inertia": "ur5e_base_link_inertia",
    "shoulder_link": "ur5e_shoulder_link",
    "upper_arm_link": "ur5e_upper_arm_link",
    "forearm_link": "ur5e_forearm_link",
    "wrist_1_link": "ur5e_wrist_1_link",
    "wrist_2_link": "ur5e_wrist_2_link",
    "wrist_3_link": "ur5e_wrist_3_link",
    "flange": "ur5e_flange",
    "tool0": "ur5e_tool0",
}
CUROBO_V1_UR5E_JOINT_MAP = {
    "shoulder_pan_joint": "ur5e_shoulder_pan_joint",
    "shoulder_lift_joint": "ur5e_shoulder_lift_joint",
    "elbow_joint": "ur5e_elbow_joint",
    "wrist_1_joint": "ur5e_wrist_1_joint",
    "wrist_2_joint": "ur5e_wrist_2_joint",
    "wrist_3_joint": "ur5e_wrist_3_joint",
    "wrist_3-flange": "ur5e_wrist_3-flange",
    "flange-tool0": "ur5e_flange-tool0",
}


class CuRoboUnavailableError(RuntimeError):
    """Raised when the requested CuRobo runtime cannot be loaded or use CUDA."""


@dataclass(frozen=True)
class CuRoboRuntimeProbe:
    compatible: bool
    api_generation: str | None
    python_version: str
    nvidia_driver_version: str | None
    curobo_version: str | None
    issues: tuple[str, ...]

    def format_error(self) -> str:
        details = "\n".join(f"- {issue}" for issue in self.issues)
        target = (
            "CuRobo v0.7.8 / CUDA 12.2 / driver >=535.54.03"
            if self.api_generation in (None, CUROBO_API_V1)
            else "CuRobo v0.8.x / driver >=580.65.06"
        )
        return (
            "CuRobo backend is unavailable in this environment:\n"
            f"{details}\n"
            f"Required stack for this robot config: {target}. "
            "The PyBullet backend remains available."
        )


@dataclass(frozen=True)
class CuRoboGoalBatch:
    """Viewpoint-major, roll-minor optical targets in the robot-base frame."""

    positions_base_m: np.ndarray
    quaternions_wxyz: np.ndarray

    @property
    def batch_size(self) -> int:
        return int(self.positions_base_m.shape[0])

    @property
    def goalset_size(self) -> int:
        return int(self.positions_base_m.shape[1])


def _parse_version_triplet(value: str) -> tuple[int, int, int] | None:
    match = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", value)
    if match is None:
        return None
    return tuple(int(component or 0) for component in match.groups())


def curobo_v1_rotation_metric_from_degrees(angle_deg: float) -> float:
    """Convert a full quaternion angle to CuRobo V1's ``sin(angle / 2)`` metric."""

    angle = float(angle_deg)
    if not math.isfinite(angle) or not 0.0 <= angle <= 180.0:
        raise ValueError("CuRobo V1 orientation angle must be finite and in [0, 180] degrees.")
    return math.sin(0.5 * math.radians(angle))


def curobo_v1_rotation_metric_to_degrees(metric: float) -> float:
    """Convert CuRobo V1's non-negative quaternion residual to a full angle."""

    value = float(metric)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("CuRobo V1 orientation residual must be finite and non-negative.")
    return math.degrees(2.0 * math.asin(min(1.0, value)))


def _read_nvidia_driver_version() -> str | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    versions = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return versions[0] if versions else None


def _api_for_curobo_version(version: str | None) -> str | None:
    """Map only explicitly supported CuRobo releases to their API generation."""

    parsed = _parse_version_triplet(version or "")
    if parsed == SUPPORTED_CUROBO_V1_VERSION:
        return CUROBO_API_V1
    if parsed is not None and parsed[:2] == SUPPORTED_CUROBO_V2_MAJOR_MINOR:
        return CUROBO_API_V2
    return None


def detect_curobo_robot_config_api(config_path: Path) -> str:
    """Identify the mutually incompatible V1/V2 YAML schemas without importing CuRobo."""

    import yaml

    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"CuRobo robot config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"CuRobo robot config is not a YAML mapping: {path}")
    if isinstance(data.get("robot_cfg"), dict):
        return CUROBO_API_V1
    if isinstance(data.get("kinematics"), dict):
        return CUROBO_API_V2
    raise ValueError(
        "Cannot identify CuRobo robot config API: expected V1 'robot_cfg' or "
        f"V2 'kinematics' in {path}."
    )


def probe_curobo_runtime(expected_api: str | None = None) -> CuRoboRuntimeProbe:
    """Check version-specific host requirements without importing CuRobo or torch."""

    if expected_api not in (None, CUROBO_API_V1, CUROBO_API_V2):
        raise ValueError(f"Unknown CuRobo API generation: {expected_api!r}")

    issues: list[str] = []
    try:
        curobo_version = importlib.metadata.version(CUROBO_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        curobo_version = None
        installed_api = None
        issues.append("Python package 'nvidia-curobo' is not installed.")
    else:
        installed_api = _api_for_curobo_version(curobo_version)
        if installed_api is None:
            issues.append(
                f"CuRobo {curobo_version} is unsupported; install exactly v0.7.8 "
                "for R535/CUDA 12.2 or a v0.8.x release for R580+."
            )
        elif expected_api is not None and installed_api != expected_api:
            issues.append(
                f"CuRobo {curobo_version} uses {installed_api.upper()} APIs, but the "
                f"robot config uses {expected_api.upper()} APIs."
            )

    target_api = expected_api or installed_api or CUROBO_API_V1
    python_minimum = (
        MINIMUM_PYTHON_V1 if target_api == CUROBO_API_V1 else MINIMUM_PYTHON_V2
    )
    driver_minimum = (
        MINIMUM_NVIDIA_DRIVER_V1
        if target_api == CUROBO_API_V1
        else MINIMUM_NVIDIA_DRIVER_V2
    )

    python_version = platform.python_version()
    python_triplet = _parse_version_triplet(python_version)
    if python_triplet is None or python_triplet[:2] < python_minimum:
        issues.append(
            f"Python {python_version} is too old for CuRobo {target_api.upper()} "
            f"(need >={python_minimum[0]}.{python_minimum[1]})."
        )

    driver_version = _read_nvidia_driver_version()
    driver_triplet = (
        _parse_version_triplet(driver_version) if driver_version is not None else None
    )
    if driver_triplet is None:
        issues.append("NVIDIA driver version could not be read with nvidia-smi.")
    elif driver_triplet < driver_minimum:
        required = ".".join(str(value) for value in driver_minimum)
        issues.append(
            f"NVIDIA driver {driver_version} is too old for CuRobo {target_api.upper()} "
            f"(need >={required})."
        )

    return CuRoboRuntimeProbe(
        compatible=not issues,
        api_generation=target_api,
        python_version=python_version,
        nvidia_driver_version=driver_version,
        curobo_version=curobo_version,
        issues=tuple(issues),
    )


def require_curobo_runtime(expected_api: str | None = None) -> CuRoboRuntimeProbe:
    """Fail explicitly for host, package, CUDA, or generation-specific API mismatch."""

    probe = probe_curobo_runtime(expected_api=expected_api)
    if not probe.compatible:
        raise CuRoboUnavailableError(probe.format_error())

    try:
        import torch

        if probe.api_generation == CUROBO_API_V1:
            required_symbols = {
                "curobo.types.base": ("TensorDeviceType",),
                "curobo.types.math": ("Pose",),
                "curobo.wrap.reacher.ik_solver": ("IKSolver", "IKSolverConfig"),
            }
        else:
            required_symbols = {
                "curobo.inverse_kinematics": (
                    "InverseKinematics",
                    "InverseKinematicsCfg",
                ),
                "curobo.types": ("DeviceCfg", "GoalToolPose", "Pose"),
            }
        for module_name, symbol_names in required_symbols.items():
            module = importlib.import_module(module_name)
            for symbol_name in symbol_names:
                getattr(module, symbol_name)
    except ModuleNotFoundError as exc:
        raise CuRoboUnavailableError(
            f"CuRobo {probe.api_generation.upper()} is installed, but required module "
            f"{exc.name!r} is missing. Reinstall the pinned environment."
        ) from exc
    except AttributeError as exc:
        raise CuRoboUnavailableError(
            f"CuRobo {probe.api_generation.upper()} is missing a required public API: {exc}."
        ) from exc
    except Exception as exc:
        raise CuRoboUnavailableError(
            f"CuRobo {probe.api_generation.upper()} import failed (likely a "
            f"CUDA/PyTorch/ABI mismatch): {type(exc).__name__}: {exc}"
        ) from exc

    if probe.api_generation == CUROBO_API_V1:
        torch_cuda = _parse_version_triplet(str(torch.version.cuda or ""))
        if (
            torch_cuda is None
            or torch_cuda[:2] != SUPPORTED_V1_TORCH_CUDA_MAJOR_MINOR
        ):
            raise CuRoboUnavailableError(
                "CuRobo v0.7.8 compatibility mode requires a PyTorch CUDA 12.2 "
                f"build; found torch CUDA {torch.version.cuda!r}. Use the provided "
                "NVIDIA PyTorch 23.08 based image."
            )

    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:
        raise CuRoboUnavailableError(
            "PyTorch CUDA availability check failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not cuda_available:
        raise CuRoboUnavailableError(
            "PyTorch cannot access CUDA in the selected environment. Check container "
            "GPU passthrough and the CUDA-matched torch build."
        )
    try:
        device_properties = torch.cuda.get_device_properties(0)
    except Exception as exc:
        raise CuRoboUnavailableError(
            f"CUDA device 0 properties could not be read: {type(exc).__name__}: {exc}"
        ) from exc
    if int(device_properties.total_memory) < 4 * 1024**3:
        raise CuRoboUnavailableError(
            "CUDA device 0 has less than the required 4 GB VRAM."
        )
    device_compute = (
        int(device_properties.major),
        int(device_properties.minor),
    )
    minimum_compute = (
        MINIMUM_COMPUTE_CAPABILITY_V1
        if probe.api_generation == CUROBO_API_V1
        else MINIMUM_COMPUTE_CAPABILITY_V2
    )
    if device_compute < minimum_compute:
        raise CuRoboUnavailableError(
            "CUDA device 0 has unsupported compute capability: need at least "
            f"{minimum_compute[0]}.{minimum_compute[1]} for CuRobo "
            f"{probe.api_generation.upper()}, found "
            f"{device_compute[0]}.{device_compute[1]}."
        )
    try:
        # ``is_available`` and device properties can succeed even when the first
        # real CUDA kernel cannot run because of container/ABI/architecture issues.
        cuda_smoke = torch.zeros(1, device="cuda:0")
        cuda_smoke.square_()
        torch.cuda.synchronize(device=0)
    except Exception as exc:
        raise CuRoboUnavailableError(
            "CUDA device 0 was detected but failed a tensor execution smoke test: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    return probe


def pack_curobo_goalset(
    problems: Sequence[IKProblem], world_from_base: np.ndarray
) -> CuRoboGoalBatch:
    """Transform optical goals to base coordinates and convert XYZW to WXYZ."""

    if not problems:
        raise ValueError("At least one IK problem is required.")
    goalset_size = len(problems[0].goals)
    if goalset_size <= 0 or any(len(problem.goals) != goalset_size for problem in problems):
        raise ValueError("All IK problems must contain the same non-empty goal set.")

    transform = np.asarray(world_from_base, dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("world_from_base must be a finite 4x4 transform.")
    if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9):
        raise ValueError("world_from_base must be a homogeneous rigid transform.")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7) or not math.isclose(
        float(np.linalg.det(rotation)), 1.0, abs_tol=1e-7
    ):
        raise ValueError("world_from_base rotation must be orthonormal and right-handed.")
    base_from_world = np.linalg.inv(transform)

    positions = np.empty((len(problems), goalset_size, 3), dtype=np.float32)
    quaternions = np.empty((len(problems), goalset_size, 4), dtype=np.float32)
    for batch_index, problem in enumerate(problems):
        for goal_index, goal in enumerate(problem.goals):
            viewpoint = goal.camera_viewpoint
            world_from_camera = np.eye(4, dtype=float)
            world_from_camera[:3, :3] = quaternion_xyzw_to_matrix(
                viewpoint.orientation_xyzw
            )
            world_from_camera[:3, 3] = np.asarray(viewpoint.position_xyz, dtype=float)
            base_from_camera = base_from_world @ world_from_camera
            quaternion_xyzw = matrix_to_quaternion_xyzw(base_from_camera[:3, :3])
            positions[batch_index, goal_index] = base_from_camera[:3, 3]
            quaternions[batch_index, goal_index] = (
                quaternion_xyzw[3],
                quaternion_xyzw[0],
                quaternion_xyzw[1],
                quaternion_xyzw[2],
            )

    return CuRoboGoalBatch(positions_base_m=positions, quaternions_wxyz=quaternions)


def build_curobo_scene_model(
    *,
    workpiece_mesh_path: Path,
    robot_base_xyz: Iterable[float],
    table_size_xyz: Iterable[float],
    table_center_xyz: Iterable[float],
    table_leg_half_extents: Iterable[float],
    table_leg_positions_xyz: Sequence[Iterable[float]],
    workpiece_base_xyz: Iterable[float],
) -> dict[str, object]:
    """Build the exact PyBullet workcell scene in the URDF base frame.

    The fixed robot base geometry is intentionally omitted from the fitted
    CuRobo collision links because it is mounted in contact with the tabletop.
    Returned arm/camera candidates are still rechecked against the full scene in
    PyBullet.
    """

    base = np.asarray(tuple(robot_base_xyz), dtype=float)
    table_size = np.asarray(tuple(table_size_xyz), dtype=float)
    table_center = np.asarray(tuple(table_center_xyz), dtype=float) - base
    leg_half_extents = np.asarray(tuple(table_leg_half_extents), dtype=float)
    workpiece_position = np.asarray(tuple(workpiece_base_xyz), dtype=float) - base
    values = np.concatenate((base, table_size, table_center, leg_half_extents, workpiece_position))
    if not np.all(np.isfinite(values)) or np.any(table_size <= 0.0) or np.any(
        leg_half_extents <= 0.0
    ):
        raise ValueError("CuRobo scene dimensions and poses must be finite and positive.")

    cuboids: dict[str, dict[str, object]] = {
        "tabletop": {
            "dims": table_size.tolist(),
            "pose": [*table_center.tolist(), 1.0, 0.0, 0.0, 0.0],
        }
    }
    for index, position in enumerate(table_leg_positions_xyz):
        center = np.asarray(tuple(position), dtype=float) - base
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            raise ValueError("Every table leg position must contain three finite values.")
        cuboids[f"table_leg_{index}"] = {
            "dims": (2.0 * leg_half_extents).tolist(),
            "pose": [*center.tolist(), 1.0, 0.0, 0.0, 0.0],
        }

    mesh_path = Path(workpiece_mesh_path).resolve()
    if not mesh_path.exists():
        raise FileNotFoundError(f"Generated workpiece mesh not found: {mesh_path}")
    return {
        "cuboid": cuboids,
        "mesh": {
            "workpiece": {
                "file_path": mesh_path.as_posix(),
                "scale": [1.0, 1.0, 1.0],
                "pose": [*workpiece_position.tolist(), 1.0, 0.0, 0.0, 0.0],
            }
        },
    }


def _rotation_matrix_to_rpy(rotation: np.ndarray) -> tuple[float, float, float]:
    """Return URDF fixed-axis roll, pitch, yaw for ``Rz(yaw) Ry(pitch) Rx(roll)``."""

    matrix = np.asarray(rotation, dtype=float)
    pitch = math.asin(float(np.clip(-matrix[2, 0], -1.0, 1.0)))
    if abs(math.cos(pitch)) > 1e-9:
        roll = math.atan2(float(matrix[2, 1]), float(matrix[2, 2]))
        yaw = math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))
    else:
        roll = math.atan2(float(-matrix[1, 2]), float(matrix[1, 1]))
        yaw = 0.0
    return roll, pitch, yaw


def _format_vector(values: Iterable[float]) -> str:
    return " ".join(f"{float(value):.17g}" for value in values)


def generate_curobo_camera_urdf(
    *,
    source_urdf: Path,
    output_urdf: Path,
    kept_link_names: set[str],
    kept_joint_names: set[str],
    package_description_root: Path,
    tool_from_optical_camera: np.ndarray,
    camera_body_half_extents: Iterable[float],
    tool_link_name: str = "ur5e_tool0",
    base_link_name: str = "ur5e_base_link",
    optical_link_name: str = DEFAULT_CUROBO_TOOL_FRAME,
    camera_body_link_name: str = "ur5e_camera_body",
) -> Path:
    """Write a base-rooted URDF whose fitting geometry includes the camera box."""

    source_urdf = Path(source_urdf).resolve()
    output_urdf = Path(output_urdf).resolve()
    package_description_root = Path(package_description_root).resolve()
    if source_urdf == output_urdf:
        raise ValueError("CuRobo output URDF must not overwrite the source robot URDF.")
    transform = np.asarray(tool_from_optical_camera, dtype=float)
    half_extents = np.asarray(tuple(camera_body_half_extents), dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("tool_from_optical_camera must be a finite 4x4 transform.")
    if half_extents.shape != (3,) or not np.all(np.isfinite(half_extents)) or np.any(
        half_extents <= 0.0
    ):
        raise ValueError("Camera half extents must contain three positive finite values.")

    tree = ET.parse(source_urdf)
    root = tree.getroot()
    root.attrib["name"] = f"{root.attrib.get('name', 'robot')}_curobo_camera"
    links_to_keep = set(kept_link_names) - {"world"}
    joints_to_keep = set(kept_joint_names) - {"ur5e_base_joint"}

    for child in list(root):
        name = child.attrib.get("name", "")
        if child.tag == "link":
            if name not in links_to_keep:
                root.remove(child)
                continue
            for mesh in child.iter("mesh"):
                filename = mesh.attrib.get("filename")
                if not filename:
                    continue
                if filename.startswith("package://steve_description/"):
                    resolved = package_description_root / filename.split(
                        "package://steve_description/", 1
                    )[1]
                else:
                    resolved = Path(filename)
                    if not resolved.is_absolute():
                        resolved = source_urdf.parent / resolved
                resolved = resolved.resolve()
                if not resolved.exists():
                    raise FileNotFoundError(f"URDF mesh not found: {resolved}")
                mesh.attrib["filename"] = resolved.as_posix()

            # MorphIt currently fits URDF visual geometry.  Copy the reviewed
            # collision proxies to visual geometry so sphere fitting uses STLs,
            # not the more decorative DAE meshes.
            for visual in list(child.findall("visual")):
                child.remove(visual)
            if name == "ur5e_base_link_inertia":
                # The fixed base is intentionally in contact with the table.
                for collision in list(child.findall("collision")):
                    child.remove(collision)
            else:
                collisions = list(child.findall("collision"))
                for collision in collisions:
                    visual = copy.deepcopy(collision)
                    visual.tag = "visual"
                    child.insert(0, visual)
        elif child.tag == "joint":
            if name not in joints_to_keep:
                root.remove(child)
        else:
            root.remove(child)

    existing_links = {link.attrib.get("name") for link in root.findall("link")}
    if base_link_name not in existing_links or tool_link_name not in existing_links:
        raise ValueError("The pruned URDF is missing its expected base or tool link.")

    ET.SubElement(root, "link", {"name": optical_link_name})
    optical_joint = ET.SubElement(
        root,
        "joint",
        {"name": f"{tool_link_name}-{optical_link_name}", "type": "fixed"},
    )
    ET.SubElement(optical_joint, "parent", {"link": tool_link_name})
    ET.SubElement(optical_joint, "child", {"link": optical_link_name})
    roll_pitch_yaw = _rotation_matrix_to_rpy(transform[:3, :3])
    ET.SubElement(
        optical_joint,
        "origin",
        {
            "xyz": _format_vector(transform[:3, 3]),
            "rpy": _format_vector(roll_pitch_yaw),
        },
    )

    camera_body_link = ET.SubElement(root, "link", {"name": camera_body_link_name})
    box_size = _format_vector(2.0 * half_extents)
    for geometry_tag in ("visual", "collision"):
        element = ET.SubElement(camera_body_link, geometry_tag)
        geometry = ET.SubElement(element, "geometry")
        ET.SubElement(geometry, "box", {"size": box_size})

    camera_body_joint = ET.SubElement(
        root,
        "joint",
        {
            "name": f"{optical_link_name}-{camera_body_link_name}",
            "type": "fixed",
        },
    )
    ET.SubElement(camera_body_joint, "parent", {"link": optical_link_name})
    ET.SubElement(camera_body_joint, "child", {"link": camera_body_link_name})
    ET.SubElement(
        camera_body_joint,
        "origin",
        {
            "xyz": _format_vector((0.0, 0.0, -half_extents[2])),
            "rpy": "0 0 0",
        },
    )

    # Enforce a single-root model before writing it.
    child_links = {
        joint.find("child").attrib["link"]
        for joint in root.findall("joint")
        if joint.find("child") is not None
    }
    roots = existing_links.union({optical_link_name, camera_body_link_name}) - child_links
    if roots != {base_link_name}:
        raise ValueError(f"Expected exactly root {base_link_name!r}, found {sorted(roots)}.")

    output_urdf.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_urdf, encoding="utf-8", xml_declaration=True)
    return output_urdf


def write_curobo_v1_ik_robot_config(
    *,
    config_path: Path,
    camera_urdf_path: Path,
    joint_names: Sequence[str],
    retract_config: Sequence[float],
    base_link: str = "ur5e_base_link",
    ee_link: str = DEFAULT_CUROBO_TOOL_FRAME,
) -> Path:
    """Write the sphere-free base schema for legacy CuRobo v0.7.8.

    Level 6 can use this schema directly because PyBullet remains its
    authoritative endpoint validator.  The normal V1 preparation workflow
    immediately augments it with a geometry-verified copy of the pinned UR5e
    sphere model and a conservative camera sphere for Level 7.  Keeping this
    base writer separate also supports the explicit ``--v1-ik-only`` mode.
    """

    import yaml

    output = Path(config_path).resolve()
    urdf = Path(camera_urdf_path).resolve()
    if not urdf.exists():
        raise FileNotFoundError(f"CuRobo camera URDF not found: {urdf}")
    names = [str(name) for name in joint_names]
    retract = [float(value) for value in retract_config]
    if not names or len(names) != len(set(names)):
        raise ValueError("CuRobo V1 joint names must be non-empty and unique.")
    if len(retract) != len(names) or not all(math.isfinite(value) for value in retract):
        raise ValueError("CuRobo V1 retract_config must contain one finite value per joint.")

    data = {
        "robot_cfg": {
            "kinematics": {
                "urdf_path": urdf.as_posix(),
                "asset_root_path": urdf.parent.as_posix(),
                "base_link": base_link,
                "ee_link": ee_link,
                "link_names": [ee_link],
                "lock_joints": None,
                "extra_links": None,
                "collision_link_names": [],
                "collision_spheres": None,
                "collision_sphere_buffer": 0.0,
                "extra_collision_spheres": {},
                "self_collision_ignore": {},
                "self_collision_buffer": {},
                "use_global_cumul": True,
                "mesh_link_names": [],
                "load_meshes": False,
                "cspace": {
                    "joint_names": names,
                    "retract_config": retract,
                    "null_space_weight": [1.0] * len(names),
                    "cspace_distance_weight": [1.0] * len(names),
                    "max_acceleration": 12.0,
                    "max_jerk": 500.0,
                    "velocity_scale": 1.0,
                    "acceleration_scale": 1.0,
                    "jerk_scale": 1.0,
                    "position_limit_clip": 0.0,
                },
            }
        },
        "findvp_adapter": {
            "schema": "curobo_v1_ik_only",
            "required_curobo_version": "0.7.8",
            "required_torch_cuda": "12.2",
            "curobo_collision_check": False,
            "authoritative_endpoint_validator": "pybullet",
            "model_limitations": [
                "legacy CuRobo proposes IK branches without collision checking",
                "reduced robot model omits the Robotiq gripper",
            ],
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return output


def _urdf_origin_vector(element: ET.Element) -> np.ndarray:
    origin = element.find("origin")
    if origin is None:
        return np.zeros(6, dtype=float)
    xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
    rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")
    if xyz.shape != (3,) or rpy.shape != (3,):
        raise RuntimeError("URDF origin must contain three xyz and three rpy values.")
    return np.concatenate((xyz, rpy))


def add_curobo_v1_reference_collision_model(
    *,
    config_path: Path,
    reference_config_path: Path,
    reference_urdf_path: Path,
    camera_body_half_extents: Sequence[float],
) -> Path:
    """Map the pinned CuRobo v0.7.8 UR5e spheres after geometry verification.

    The local robot uses prefixed link and joint names. Before copying any
    reference spheres, this function proves that the six arm collision-mesh
    basenames, collision origins, moving-joint kinematics, and fixed
    wrist-to-tool chain match CuRobo's bundled UR5e model. A conservative
    bounding sphere is then added for the camera box.
    """

    import yaml

    output = Path(config_path).resolve()
    reference_config = Path(reference_config_path).resolve()
    reference_urdf = Path(reference_urdf_path).resolve()
    for path, label in (
        (output, "generated V1 config"),
        (reference_config, "CuRobo reference config"),
        (reference_urdf, "CuRobo reference URDF"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")

    data = yaml.safe_load(output.read_text(encoding="utf-8"))
    reference_data = yaml.safe_load(reference_config.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("robot_cfg"), dict):
        raise RuntimeError("Generated V1 config is missing robot_cfg.")
    if not isinstance(reference_data, dict) or not isinstance(
        reference_data.get("robot_cfg"), dict
    ):
        raise RuntimeError("CuRobo reference config is missing robot_cfg.")
    kinematics = data["robot_cfg"].get("kinematics")
    reference_kinematics = reference_data["robot_cfg"].get("kinematics")
    if not isinstance(kinematics, dict) or not isinstance(reference_kinematics, dict):
        raise RuntimeError("V1 robot configurations are missing kinematics mappings.")
    target_urdf = Path(str(kinematics.get("urdf_path", ""))).resolve()
    if not target_urdf.exists():
        raise FileNotFoundError(f"Generated camera URDF not found: {target_urdf}")

    target_root = ET.parse(target_urdf).getroot()
    reference_root = ET.parse(reference_urdf).getroot()
    target_links = {link.attrib.get("name"): link for link in target_root.findall("link")}
    reference_links = {
        link.attrib.get("name"): link for link in reference_root.findall("link")
    }
    for reference_name, target_name in CUROBO_V1_UR5E_LINK_MAP.items():
        if reference_name in ("base_link", "base_link_inertia", "flange", "tool0"):
            continue
        reference_link = reference_links.get(reference_name)
        target_link = target_links.get(target_name)
        if reference_link is None or target_link is None:
            raise RuntimeError(
                f"Cannot map CuRobo collision link {reference_name!r} to {target_name!r}."
            )
        reference_collision = reference_link.find("collision")
        target_collision = target_link.find("collision")
        if reference_collision is None or target_collision is None:
            raise RuntimeError(
                f"Collision geometry is missing for mapped link {reference_name!r}."
            )
        reference_mesh = reference_collision.find("geometry/mesh")
        target_mesh = target_collision.find("geometry/mesh")
        if reference_mesh is None or target_mesh is None:
            raise RuntimeError(
                f"Mapped link {reference_name!r} does not use a collision mesh."
            )
        reference_basename = Path(reference_mesh.attrib.get("filename", "")).name
        target_basename = Path(target_mesh.attrib.get("filename", "")).name
        if reference_basename != target_basename or not np.allclose(
            _urdf_origin_vector(reference_collision),
            _urdf_origin_vector(target_collision),
            atol=1e-8,
        ):
            raise RuntimeError(
                "Local UR5e collision geometry differs from CuRobo's reviewed reference "
                f"at link {reference_name!r}."
            )

    target_joints = {
        joint.attrib.get("name"): joint for joint in target_root.findall("joint")
    }
    reference_joints = {
        joint.attrib.get("name"): joint for joint in reference_root.findall("joint")
    }
    for reference_name, target_name in CUROBO_V1_UR5E_JOINT_MAP.items():
        reference_joint = reference_joints.get(reference_name)
        target_joint = target_joints.get(target_name)
        if reference_joint is None or target_joint is None:
            raise RuntimeError(
                f"Cannot map CuRobo joint {reference_name!r} to {target_name!r}."
            )
        reference_parent = reference_joint.find("parent")
        reference_child = reference_joint.find("child")
        target_parent = target_joint.find("parent")
        target_child = target_joint.find("child")
        if any(
            element is None
            for element in (
                reference_parent,
                reference_child,
                target_parent,
                target_child,
            )
        ):
            raise RuntimeError(f"Mapped joint {reference_name!r} has incomplete links.")
        expected_parent = CUROBO_V1_UR5E_LINK_MAP[
            reference_parent.attrib["link"]  # type: ignore[union-attr]
        ]
        expected_child = CUROBO_V1_UR5E_LINK_MAP[
            reference_child.attrib["link"]  # type: ignore[union-attr]
        ]
        reference_axis = reference_joint.find("axis")
        target_axis = target_joint.find("axis")
        reference_axis_xyz = np.fromstring(
            reference_axis.attrib.get("xyz", "1 0 0")  # type: ignore[union-attr]
            if reference_axis is not None
            else "1 0 0",
            sep=" ",
        )
        target_axis_xyz = np.fromstring(
            target_axis.attrib.get("xyz", "1 0 0")  # type: ignore[union-attr]
            if target_axis is not None
            else "1 0 0",
            sep=" ",
        )
        if (
            target_parent.attrib["link"] != expected_parent  # type: ignore[union-attr]
            or target_child.attrib["link"] != expected_child  # type: ignore[union-attr]
            or not np.allclose(
                _urdf_origin_vector(reference_joint),
                _urdf_origin_vector(target_joint),
                atol=1e-8,
            )
            or not np.allclose(reference_axis_xyz, target_axis_xyz, atol=1e-12)
        ):
            raise RuntimeError(
                "Local UR5e kinematics differ from CuRobo's reviewed reference at "
                f"joint {reference_name!r}."
            )

    reference_spheres = reference_kinematics.get("collision_spheres")
    reference_collision_links = reference_kinematics.get("collision_link_names")
    if not isinstance(reference_spheres, dict) or not isinstance(
        reference_collision_links, list
    ):
        raise RuntimeError("CuRobo reference config has no collision sphere model.")
    missing_reference_spheres = [
        name for name in reference_collision_links if name not in reference_spheres
    ]
    if missing_reference_spheres:
        raise RuntimeError(
            f"CuRobo reference spheres are missing links: {missing_reference_spheres}."
        )

    mapped_spheres = {
        CUROBO_V1_UR5E_LINK_MAP[name]: copy.deepcopy(reference_spheres[name])
        for name in reference_collision_links
    }
    camera_extents = np.asarray(tuple(camera_body_half_extents), dtype=float)
    if camera_extents.shape != (3,) or not np.all(np.isfinite(camera_extents)) or np.any(
        camera_extents <= 0.0
    ):
        raise ValueError("Camera half extents must contain three positive finite values.")
    mapped_spheres["ur5e_camera_body"] = [
        {"center": [0.0, 0.0, 0.0], "radius": float(np.linalg.norm(camera_extents))}
    ]
    kinematics["collision_link_names"] = [
        CUROBO_V1_UR5E_LINK_MAP[name] for name in reference_collision_links
    ] + ["ur5e_camera_body"]
    kinematics["collision_spheres"] = mapped_spheres
    kinematics["collision_sphere_buffer"] = 0.005
    reference_ignores = reference_kinematics.get("self_collision_ignore") or {}
    if not isinstance(reference_ignores, dict):
        raise RuntimeError("CuRobo reference self_collision_ignore is malformed.")
    mapped_ignores = {
        CUROBO_V1_UR5E_LINK_MAP[name]: [
            CUROBO_V1_UR5E_LINK_MAP[ignored] for ignored in ignored_names
        ]
        for name, ignored_names in reference_ignores.items()
    }
    mapped_ignores["ur5e_camera_body"] = ["ur5e_wrist_3_link"]
    kinematics["self_collision_ignore"] = mapped_ignores
    reference_buffer = reference_kinematics.get("self_collision_buffer") or {}
    if not isinstance(reference_buffer, dict):
        raise RuntimeError("CuRobo reference self_collision_buffer is malformed.")
    kinematics["self_collision_buffer"] = {
        CUROBO_V1_UR5E_LINK_MAP[name]: float(value)
        for name, value in reference_buffer.items()
    }
    adapter = data.setdefault("findvp_adapter", {})
    if not isinstance(adapter, dict):
        raise RuntimeError("findvp_adapter must be a mapping.")
    adapter.update(
        {
            "schema": "curobo_v1_collision_aware",
            "curobo_collision_check": True,
            "collision_sphere_source": reference_config.as_posix(),
            "collision_geometry_verified_against": reference_urdf.as_posix(),
            "camera_collision_sphere": "conservative bounding sphere",
            "authoritative_endpoint_and_trajectory_validator": "pybullet",
        }
    )
    adapter["model_limitations"] = [
        "mapped from the pinned CuRobo v0.7.8 UR5e sphere model after geometry checks",
        "camera box is conservatively enclosed by one sphere",
        "reduced robot model omits the Robotiq gripper",
    ]
    output.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return output


class CuRoboIKCandidateBackend:
    """Goal-set IK proposer for either legacy V1 or current V2 CuRobo schemas."""

    def __init__(
        self,
        *,
        robot_config_path: Path,
        scene_model: dict[str, object],
        world_from_base: np.ndarray,
        expected_joint_names: Sequence[str],
        tool_frame: str = DEFAULT_CUROBO_TOOL_FRAME,
        num_seeds: int = DEFAULT_CUROBO_SEEDS,
        random_seed: int = DEFAULT_CUROBO_RANDOM_SEED,
    ) -> None:
        self.robot_config_path = Path(robot_config_path).resolve()
        self.scene_model = scene_model
        self.world_from_base = np.asarray(world_from_base, dtype=float)
        self.expected_joint_names = tuple(expected_joint_names)
        self.tool_frame = tool_frame
        self.num_seeds = int(num_seeds)
        # Returning every optimized seed avoids discarding a lower-ranked
        # branch that later passes the higher-fidelity PyBullet gates.
        self.return_seeds = self.num_seeds
        self.random_seed = int(random_seed)
        self.metadata: dict[str, object] = {}
        if not self.robot_config_path.exists():
            raise FileNotFoundError(f"CuRobo robot config not found: {self.robot_config_path}")
        if self.num_seeds <= 0:
            raise ValueError("CuRobo seed count must be positive.")
        if not self.expected_joint_names or len(set(self.expected_joint_names)) != len(
            self.expected_joint_names
        ):
            raise ValueError("Expected CuRobo joint names must be non-empty and unique.")

        self.api_generation = detect_curobo_robot_config_api(self.robot_config_path)
        self.backend_name = (
            "curobo_v1_legacy_goalset_ik"
            if self.api_generation == CUROBO_API_V1
            else "curobo_v2_batched_goalset_ik"
        )
        # V1 CUDA graphs are shape-specific.  Cache the solver so production
        # streams with B=1 pay initialization/capture only on the first request.
        self._v1_solver: object | None = None
        self._v1_solver_signature: tuple[object, ...] | None = None
        self._v1_solver_lock = threading.Lock()
        self._runtime_probe: CuRoboRuntimeProbe | None = None
        self._runtime_probe_lock = threading.Lock()

    @staticmethod
    def _to_numpy(tensor: object, field_name: str) -> np.ndarray:
        try:
            return np.asarray(tensor.detach().cpu().numpy())  # type: ignore[attr-defined]
        except Exception as exc:
            raise RuntimeError(f"Malformed CuRobo result field {field_name!r}.") from exc

    def _require_runtime_once(self) -> CuRoboRuntimeProbe:
        """Run and cache full package/driver/CUDA validation for this backend."""

        if self._runtime_probe is None:
            with self._runtime_probe_lock:
                if self._runtime_probe is None:
                    self._runtime_probe = require_curobo_runtime(
                        expected_api=self.api_generation
                    )
        probe = self._runtime_probe
        if probe is None:  # pragma: no cover - defensive type narrowing
            raise RuntimeError("CuRobo runtime validation did not produce a probe.")
        return probe

    def _problem_results_from_arrays(
        self,
        *,
        problems: Sequence[IKProblem],
        success: np.ndarray,
        joint_positions: np.ndarray,
        position_error: np.ndarray,
        rotation_error: np.ndarray,
        goalset_index: np.ndarray,
        result_joint_names: Sequence[str],
        rotation_error_units: str,
    ) -> list[IKProblemResult]:
        batch_size = len(problems)
        goalset_size = len(problems[0].goals)
        expected_prefix = (batch_size, self.return_seeds)
        if goalset_index.shape == (*expected_prefix, 1):
            goalset_index = goalset_index[..., 0]
        if (
            success.shape != expected_prefix
            or joint_positions.ndim != 3
            or joint_positions.shape[:2] != expected_prefix
            or position_error.shape != expected_prefix
            or rotation_error.shape != expected_prefix
            or goalset_index.shape != expected_prefix
        ):
            raise RuntimeError(
                "Unexpected CuRobo IK result shapes: "
                f"success={success.shape}, q={joint_positions.shape}, "
                f"position_error={position_error.shape}, rotation_error={rotation_error.shape}, "
                f"goalset_index={goalset_index.shape}."
            )

        names = tuple(result_joint_names)
        if len(names) != joint_positions.shape[2] or len(set(names)) != len(names):
            raise RuntimeError(
                "CuRobo result joint names do not uniquely match the full joint axis: "
                f"names={list(names)}, q_shape={joint_positions.shape}."
            )
        missing_joints = [name for name in self.expected_joint_names if name not in names]
        if missing_joints:
            raise RuntimeError(
                "CuRobo result is missing expected active joints: "
                f"{missing_joints}; result order={list(names)}."
            )
        reorder = [names.index(name) for name in self.expected_joint_names]

        problem_results: list[IKProblemResult] = []
        for batch_index, problem in enumerate(problems):
            candidates: list[IKCandidate] = []
            for branch_index in range(self.return_seeds):
                if not success[batch_index, branch_index]:
                    continue
                selected_goal = int(goalset_index[batch_index, branch_index])
                if not 0 <= selected_goal < goalset_size:
                    raise RuntimeError(
                        f"CuRobo returned invalid goalset index {selected_goal} "
                        f"for batch {batch_index}."
                    )
                ordered_q = tuple(
                    float(joint_positions[batch_index, branch_index, index])
                    for index in reorder
                )
                solver_position_error = float(position_error[batch_index, branch_index])
                solver_rotation_error = float(rotation_error[batch_index, branch_index])
                if not all(math.isfinite(value) for value in ordered_q) or not all(
                    math.isfinite(value)
                    for value in (solver_position_error, solver_rotation_error)
                ):
                    raise RuntimeError("CuRobo returned a non-finite successful solution.")
                if rotation_error_units == "sin_half_angle":
                    solver_rotation_error_deg = curobo_v1_rotation_metric_to_degrees(
                        solver_rotation_error
                    )
                elif rotation_error_units == "radians":
                    solver_rotation_error_deg = math.degrees(solver_rotation_error)
                else:
                    raise RuntimeError(
                        f"Unknown CuRobo rotation-error units: {rotation_error_units!r}."
                    )
                candidates.append(
                    IKCandidate(
                        goal_index=selected_goal,
                        joint_positions_rad=ordered_q,
                        seed_index=None,
                        solver_provenance={
                            "api_generation": self.api_generation,
                            "goalset_index": selected_goal,
                            "returned_branch_index": branch_index,
                            "solver_position_error_m": solver_position_error,
                            "solver_rotation_error": solver_rotation_error,
                            "solver_rotation_error_units": rotation_error_units,
                            "solver_rotation_error_deg": solver_rotation_error_deg,
                        },
                    )
                )
            failed_seed_count = int(np.count_nonzero(~success[batch_index]))
            failures = (
                {"ik_solver_no_solution": failed_seed_count}
                if failed_seed_count > 0
                else None
            )
            problem_results.append(
                IKProblemResult(
                    viewpoint_index=problem.sampled_viewpoint.index,
                    candidates=tuple(candidates),
                    attempted_solutions=self.num_seeds,
                    solver_failure_counts=failures,
                )
            )
        return problem_results

    def solve(
        self,
        problems: Sequence[IKProblem],
        start_joint_positions_rad: Sequence[float],
        reachability_config: ReachabilityConfig,
    ) -> list[IKProblemResult]:
        if not problems:
            raise ValueError("At least one CuRobo IK problem is required.")
        runtime = self._require_runtime_once()
        if self.api_generation == CUROBO_API_V1:
            # CuRobo solvers mutate cached CUDA-graph and seed buffers.  One
            # backend instance is therefore serialized for safe stream use.
            with self._v1_solver_lock:
                return self._solve_v1(
                    problems, start_joint_positions_rad, reachability_config, runtime
                )
        return self._solve_v2(problems, reachability_config, runtime)

    def _solve_v1(
        self,
        problems: Sequence[IKProblem],
        start_joint_positions_rad: Sequence[float],
        reachability_config: ReachabilityConfig,
        runtime: CuRoboRuntimeProbe,
    ) -> list[IKProblemResult]:
        """Use the v0.7.8 V1 IK API; collisions remain a PyBullet responsibility."""

        import torch
        import yaml
        from curobo.types.base import TensorDeviceType
        from curobo.types.math import Pose
        from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

        goal_batch = pack_curobo_goalset(problems, self.world_from_base)
        goalset_size = goal_batch.goalset_size
        position_tolerance = reachability_config.position_tolerance_m
        orientation_tolerance_deg = reachability_config.orientation_tolerance_deg
        # V1's pose kernel reports the norm of the relative quaternion vector,
        # i.e. sin(full_orientation_angle / 2), rather than radians.
        v1_rotation_threshold = curobo_v1_rotation_metric_from_degrees(
            orientation_tolerance_deg
        )
        signature = (
            goalset_size,
            self.num_seeds,
            position_tolerance,
            v1_rotation_threshold,
        )
        solver_reused = self._v1_solver is not None and self._v1_solver_signature == signature
        tensor_args = TensorDeviceType()
        if not solver_reused:
            config_data = yaml.safe_load(
                self.robot_config_path.read_text(encoding="utf-8")
            )
            robot_cfg = config_data["robot_cfg"]
            solver_cfg = IKSolverConfig.load_from_robot_config(
                robot_cfg,
                None,
                tensor_args=tensor_args,
                num_seeds=self.num_seeds,
                position_threshold=position_tolerance,
                rotation_threshold=v1_rotation_threshold,
                self_collision_check=False,
                self_collision_opt=False,
                collision_checker_type=None,
                use_cuda_graph=True,
                use_fixed_samples=True,
                regularization=True,
                seed=self.random_seed,
            )
            self._v1_solver = IKSolver(solver_cfg)
            self._v1_solver_signature = signature
        solver = self._v1_solver
        if solver is None:  # pragma: no cover - defensive type narrowing
            raise RuntimeError("CuRobo V1 solver initialization failed.")

        solver_joint_names = tuple(solver.joint_names)  # type: ignore[attr-defined]
        missing = [name for name in self.expected_joint_names if name not in solver_joint_names]
        unexpected = [name for name in solver_joint_names if name not in self.expected_joint_names]
        if missing or unexpected:
            raise RuntimeError(
                "CuRobo V1 solver joint mismatch: "
                f"missing={missing}, unexpected={unexpected}."
            )
        start = tuple(float(value) for value in start_joint_positions_rad)
        if len(start) != len(self.expected_joint_names):
            raise ValueError("Current joint state does not match the expected CuRobo joints.")
        start_by_name = dict(zip(self.expected_joint_names, start))
        retract = torch.as_tensor(
            [start_by_name[name] for name in solver_joint_names],
            device=tensor_args.device,
            dtype=tensor_args.dtype,
        ).view(1, -1)

        torch.cuda.synchronize()
        solve_started = time.monotonic()
        problem_results: list[IKProblemResult] = []
        for batch_index, problem in enumerate(problems):
            # Always use B=1 + goal-set solve.  The Level 6 100-view harness is
            # intentionally run through the exact same shape-stable path as the
            # production stream, where viewpoints arrive one at a time.
            goal_pose = Pose(
                position=torch.as_tensor(
                    goal_batch.positions_base_m[batch_index : batch_index + 1],
                    device=tensor_args.device,
                    dtype=tensor_args.dtype,
                ).contiguous(),
                quaternion=torch.as_tensor(
                    goal_batch.quaternions_wxyz[batch_index : batch_index + 1],
                    device=tensor_args.device,
                    dtype=tensor_args.dtype,
                ).contiguous(),
                normalize_rotation=True,
            )
            # The cached solver's Halton generator advances after each request.
            # Reset it so equal sequential inputs remain exactly reproducible.
            solver.reset_seed()  # type: ignore[attr-defined]
            result = solver.solve_goalset(  # type: ignore[attr-defined]
                goal_pose,
                retract_config=retract,
                return_seeds=self.return_seeds,
                num_seeds=self.num_seeds,
                use_nn_seed=False,
            )

            success = self._to_numpy(result.success, "success").astype(bool, copy=False)
            joint_positions = self._to_numpy(result.solution, "solution")
            position_error = self._to_numpy(result.position_error, "position_error")
            rotation_error = self._to_numpy(result.rotation_error, "rotation_error")
            if result.goalset_index is None:
                if goalset_size != 1:
                    raise RuntimeError("CuRobo V1 omitted goalset_index for a multi-goal solve.")
                goalset_index = np.zeros_like(success, dtype=np.int64)
            else:
                goalset_index = self._to_numpy(result.goalset_index, "goalset_index")
            result_joint_names = tuple(result.js_solution.joint_names or ())
            problem_results.extend(
                self._problem_results_from_arrays(
                    problems=(problem,),
                    success=success,
                    joint_positions=joint_positions,
                    position_error=position_error,
                    rotation_error=rotation_error,
                    goalset_index=goalset_index,
                    result_joint_names=result_joint_names,
                    rotation_error_units="sin_half_angle",
                )
            )
        torch.cuda.synchronize()
        solve_elapsed = time.monotonic() - solve_started
        self.metadata = {
            "api_generation": CUROBO_API_V1,
            "curobo_version": runtime.curobo_version,
            "python_version": runtime.python_version,
            "nvidia_driver_version": runtime.nvidia_driver_version,
            "torch_cuda_version": str(torch.version.cuda),
            "robot_config": self.robot_config_path.as_posix(),
            "tool_frame": self.tool_frame,
            "submitted_problem_count": goal_batch.batch_size,
            "solver_batch_size": 1,
            "sequential_goalset_solves": goal_batch.batch_size,
            "goalset_size": goalset_size,
            "num_seeds_per_viewpoint_shared_across_goalset": self.num_seeds,
            "return_seeds": self.return_seeds,
            "random_seed": self.random_seed,
            "rotation_threshold_sin_half_angle": v1_rotation_threshold,
            "orientation_tolerance_deg": orientation_tolerance_deg,
            "cuda_graph": True,
            "cuda_graph_shape_b_g_k": [1, goalset_size, self.num_seeds],
            "seed_generator_reset_each_solve": True,
            "persistent_solver_reused_from_previous_call": solver_reused,
            "persistent_solver_reused_within_submission": len(problems) > 1,
            "curobo_self_collision_check": False,
            "curobo_world_collision_check": False,
            "solver_elapsed_sec": solve_elapsed,
            "validation_note": (
                "Legacy CuRobo proposes IK only; PyBullet authoritatively validates "
                "joint limits, FK, robot/camera/world collisions, clearance, and LOS."
            ),
        }
        return problem_results

    def _solve_v2(
        self,
        problems: Sequence[IKProblem],
        reachability_config: ReachabilityConfig,
        runtime: CuRoboRuntimeProbe,
    ) -> list[IKProblemResult]:
        """Use the collision-aware CuRoboV2 v0.8.x goal-set API."""

        import torch
        from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
        from curobo.types import DeviceCfg, GoalToolPose, Pose

        goal_batch = pack_curobo_goalset(problems, self.world_from_base)
        batch_size = goal_batch.batch_size
        goalset_size = goal_batch.goalset_size
        position_tolerance = reachability_config.position_tolerance_m
        orientation_tolerance_rad = math.radians(
            reachability_config.orientation_tolerance_deg
        )
        clearance_m = reachability_config.clearance_m

        device_cfg = DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)
        solver_cfg = InverseKinematicsCfg.create(
            robot=self.robot_config_path.as_posix(),
            scene_model=self.scene_model,
            collision_cache={"cuboid": 8, "mesh": 2},
            device_cfg=device_cfg,
            load_collision_spheres=True,
            self_collision_check=True,
            success_requires_convergence=True,
            max_batch_size=batch_size,
            max_goalset=goalset_size,
            multi_env=False,
            num_seeds=self.num_seeds,
            seed_solver_num_seeds=self.num_seeds,
            position_tolerance=position_tolerance,
            orientation_tolerance=orientation_tolerance_rad,
            optimizer_collision_activation_distance=max(0.01, clearance_m),
            random_seed=self.random_seed,
            use_cuda_graph=True,
        )
        solver_cfg.exit_early = False
        solver = InverseKinematics(solver_cfg)
        if list(solver.tool_frames) != [self.tool_frame]:
            raise RuntimeError(
                "CuRobo robot config tool_frames mismatch: expected exactly "
                f"[{self.tool_frame!r}], found {list(solver.tool_frames)!r}."
            )

        positions = torch.as_tensor(
            goal_batch.positions_base_m,
            device=device_cfg.device,
            dtype=device_cfg.dtype,
        )
        quaternions = torch.as_tensor(
            goal_batch.quaternions_wxyz,
            device=device_cfg.device,
            dtype=device_cfg.dtype,
        )
        flat_pose = Pose(
            position=positions.reshape(batch_size * goalset_size, 3).contiguous(),
            quaternion=quaternions.reshape(batch_size * goalset_size, 4).contiguous(),
            normalize_rotation=True,
        )
        goals = GoalToolPose.from_poses(
            {self.tool_frame: flat_pose},
            ordered_tool_frames=solver.tool_frames,
            num_goalset=goalset_size,
        )
        if tuple(goals.position.shape) != (batch_size, 1, 1, goalset_size, 3):
            raise RuntimeError(
                "Unexpected CuRobo GoalToolPose shape: "
                f"{tuple(goals.position.shape)}. Expected "
                f"{(batch_size, 1, 1, goalset_size, 3)}."
            )

        torch.cuda.synchronize()
        solve_started = time.monotonic()
        result = solver.solve_pose(goal_tool_poses=goals, return_seeds=self.return_seeds)
        torch.cuda.synchronize()
        solve_elapsed = time.monotonic() - solve_started

        problem_results = self._problem_results_from_arrays(
            problems=problems,
            success=self._to_numpy(result.success, "success").astype(bool, copy=False),
            joint_positions=self._to_numpy(
                result.js_solution.position, "js_solution.position"
            ),
            position_error=self._to_numpy(result.position_error, "position_error"),
            rotation_error=self._to_numpy(result.rotation_error, "rotation_error"),
            goalset_index=self._to_numpy(result.goalset_index, "goalset_index"),
            result_joint_names=tuple(result.js_solution.joint_names or ()),
            rotation_error_units="radians",
        )
        self.metadata = {
            "api_generation": CUROBO_API_V2,
            "curobo_version": runtime.curobo_version,
            "python_version": runtime.python_version,
            "nvidia_driver_version": runtime.nvidia_driver_version,
            "robot_config": self.robot_config_path.as_posix(),
            "tool_frame": self.tool_frame,
            "batch_size": batch_size,
            "goalset_size": goalset_size,
            "num_seeds_per_viewpoint_shared_across_goalset": self.num_seeds,
            "return_seeds": self.return_seeds,
            "random_seed": self.random_seed,
            "cuda_graph": True,
            "self_collision_check": True,
            "scene_collision_types": ["cuboid", "mesh"],
            "optimizer_collision_activation_distance_m": max(0.01, clearance_m),
            "solver_elapsed_sec": solve_elapsed,
            "clearance_note": (
                "CuRobo's optimizer activation distance is not measured clearance; "
                "exported clearance comes from PyBullet endpoint revalidation."
            ),
        }
        return problem_results
