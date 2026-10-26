"""Deterministic 3D viewpoint sampling primitives.

The robot-specific IK and collision checks live in the PyBullet environment.  This
module deliberately stays independent of PyBullet so the sampling, grouped IK
problems, and pose maths can be tested cheaply and shared with the CuRobo backend.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Iterable, Protocol, Sequence

import numpy as np


# Keep the image's vertical axis aligned with world up by default. Additional
# optical-axis rolls are useful IK redundancy, but selecting one physically
# rotates the camera image by the same amount.
DEFAULT_IK_ROLLS_DEG = (0.0,)
DEFAULT_IK_SEED_COUNT = 3
DEFAULT_IK_REFINEMENT_STEPS = 1
DEFAULT_IK_POSITION_TOLERANCE_M = 0.001
DEFAULT_IK_ORIENTATION_TOLERANCE_DEG = 1.0
DEFAULT_COLLISION_CLEARANCE_M = 0.005


@dataclass(frozen=True)
class ReachabilityConfig:
    """Solver effort and endpoint-safety thresholds for viewpoint filtering."""

    roll_angles_deg: tuple[float, ...] = DEFAULT_IK_ROLLS_DEG
    seed_count: int = DEFAULT_IK_SEED_COUNT
    refinement_steps: int = DEFAULT_IK_REFINEMENT_STEPS
    position_tolerance_m: float = DEFAULT_IK_POSITION_TOLERANCE_M
    orientation_tolerance_deg: float = DEFAULT_IK_ORIENTATION_TOLERANCE_DEG
    clearance_m: float = DEFAULT_COLLISION_CLEARANCE_M
    require_line_of_sight: bool = True
    path_validation_steps: int = 0

    def validate(self) -> None:
        if not self.roll_angles_deg or not all(
            math.isfinite(angle) for angle in self.roll_angles_deg
        ):
            raise ValueError("IK roll angles must be a non-empty set of finite values.")
        if (
            not isinstance(self.seed_count, int)
            or isinstance(self.seed_count, bool)
            or self.seed_count <= 0
        ):
            raise ValueError("IK seed count must be positive.")
        if (
            not isinstance(self.refinement_steps, int)
            or isinstance(self.refinement_steps, bool)
            or self.refinement_steps < 0
        ):
            raise ValueError("IK refinement steps cannot be negative.")
        if not math.isfinite(self.position_tolerance_m) or self.position_tolerance_m <= 0.0:
            raise ValueError("IK position tolerance must be finite and positive.")
        if (
            not math.isfinite(self.orientation_tolerance_deg)
            or self.orientation_tolerance_deg <= 0.0
            or self.orientation_tolerance_deg > 180.0
        ):
            raise ValueError(
                "IK orientation tolerance must be finite and in (0, 180] degrees."
            )
        if not math.isfinite(self.clearance_m) or self.clearance_m < 0.0:
            raise ValueError("Collision clearance must be finite and non-negative.")
        if (
            not isinstance(self.path_validation_steps, int)
            or isinstance(self.path_validation_steps, bool)
            or self.path_validation_steps < 0
        ):
            raise ValueError("Path validation steps cannot be negative.")


@dataclass(frozen=True)
class ViewpointCandidate:
    """One optical-camera pose sampled around an object."""

    index: int
    position_xyz: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]
    radius_m: float
    azimuth_deg: float
    elevation_deg: float
    roll_deg: float = 0.0

    def with_roll(self, roll_deg: float) -> "ViewpointCandidate":
        rotation = look_at_rotation_matrix(
            np.asarray(self.position_xyz, dtype=float),
            np.asarray(self.position_xyz, dtype=float)
            + quaternion_xyzw_to_matrix(self.orientation_xyzw)[:, 2],
            roll_deg=roll_deg,
        )
        return replace(
            self,
            orientation_xyzw=matrix_to_quaternion_xyzw(rotation),
            roll_deg=float(roll_deg),
        )


@dataclass(frozen=True)
class IKGoal:
    """One rolled camera goal and its equivalent robot-tool pose."""

    camera_viewpoint: ViewpointCandidate
    tool_position_xyz: tuple[float, float, float]
    tool_orientation_xyzw: tuple[float, float, float, float]


@dataclass(frozen=True)
class IKProblem:
    """All roll alternatives that belong to one sampled 3D position."""

    sampled_viewpoint: ViewpointCandidate
    goals: tuple[IKGoal, ...]


@dataclass(frozen=True)
class IKCandidate:
    """A backend-proposed joint branch for one goal in an :class:`IKProblem`."""

    goal_index: int
    joint_positions_rad: tuple[float, ...]
    seed_index: int | None = None
    solver_provenance: dict[str, object] | None = None


@dataclass(frozen=True)
class IKProblemResult:
    """Finite solver output for one sampled viewpoint."""

    viewpoint_index: int
    candidates: tuple[IKCandidate, ...]
    attempted_solutions: int
    solver_failure_counts: dict[str, int] | None = None


class IKCandidateBackend(Protocol):
    """Structural interface for an IK candidate generator."""

    backend_name: str
    metadata: dict[str, object]

    def solve(
        self,
        problems: Sequence[IKProblem],
        start_joint_positions_rad: Sequence[float],
        reachability_config: ReachabilityConfig,
    ) -> list[IKProblemResult]: ...


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.all(np.isfinite(vector)) or not math.isfinite(norm) or norm < 1e-12:
        raise ValueError("Cannot normalize a non-finite or near-zero vector.")
    return vector / norm


def matrix_to_quaternion_xyzw(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a proper 3x3 rotation matrix to a normalized XYZW quaternion."""

    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 rotation matrix, got {matrix.shape}.")

    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (matrix[2, 1] - matrix[1, 2]) / scale
        qy = (matrix[0, 2] - matrix[2, 0]) / scale
        qz = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        qw = (matrix[2, 1] - matrix[1, 2]) / scale
        qx = 0.25 * scale
        qy = (matrix[0, 1] + matrix[1, 0]) / scale
        qz = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        qw = (matrix[0, 2] - matrix[2, 0]) / scale
        qx = (matrix[0, 1] + matrix[1, 0]) / scale
        qy = 0.25 * scale
        qz = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        qw = (matrix[1, 0] - matrix[0, 1]) / scale
        qx = (matrix[0, 2] + matrix[2, 0]) / scale
        qy = (matrix[1, 2] + matrix[2, 1]) / scale
        qz = 0.25 * scale

    quaternion = _normalize(np.asarray([qx, qy, qz, qw], dtype=float))
    return tuple(float(value) for value in quaternion)


def quaternion_xyzw_to_matrix(quaternion_xyzw: Iterable[float]) -> np.ndarray:
    quaternion = _normalize(np.asarray(tuple(quaternion_xyzw), dtype=float))
    if quaternion.shape != (4,):
        raise ValueError(f"Expected four quaternion values, got {quaternion.shape}.")
    x, y, z, w = quaternion
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def look_at_rotation_matrix(
    camera_position: np.ndarray,
    target_position: np.ndarray,
    roll_deg: float = 0.0,
    world_up: np.ndarray | None = None,
) -> np.ndarray:
    """Return ``R_world_camera`` for +Z-forward, -Y-up optical coordinates."""

    eye = np.asarray(camera_position, dtype=float)
    target = np.asarray(target_position, dtype=float)
    if eye.shape != (3,) or target.shape != (3,):
        raise ValueError("Camera and target positions must each contain three values.")

    forward = _normalize(target - eye)
    up = np.asarray((0.0, 0.0, 1.0) if world_up is None else world_up, dtype=float)
    up = _normalize(up)
    if abs(float(np.dot(forward, up))) > 0.98:
        up = np.asarray((0.0, 1.0, 0.0), dtype=float)

    right = _normalize(np.cross(forward, up))
    down = _normalize(np.cross(forward, right))

    roll = math.radians(float(roll_deg))
    rolled_right = math.cos(roll) * right + math.sin(roll) * down
    rolled_down = -math.sin(roll) * right + math.cos(roll) * down
    return np.column_stack((_normalize(rolled_right), _normalize(rolled_down), forward))


def _radical_inverse_base_two(index: int) -> float:
    value = 0.0
    factor = 0.5
    while index:
        value += factor * (index & 1)
        index >>= 1
        factor *= 0.5
    return value


def sample_fibonacci_viewpoints(
    object_center_xyz: Iterable[float],
    count: int = 100,
    radius_min_m: float = 0.30,
    radius_max_m: float = 0.50,
    elevation_min_deg: float = 12.0,
    elevation_max_deg: float = 82.0,
) -> list[ViewpointCandidate]:
    """Sample an equal-area, low-discrepancy shell over the upper hemisphere.

    Elevation is sampled uniformly in ``sin(elevation)`` (surface area), azimuth
    uses the golden angle, and radius uses a base-two radical inverse.  The three
    dimensions therefore avoid the aligned rings and poles produced by a regular
    latitude/longitude grid while remaining completely reproducible.
    """

    center = np.asarray(tuple(object_center_xyz), dtype=float)
    if center.shape != (3,):
        raise ValueError("object_center_xyz must contain exactly three values.")
    if not np.all(np.isfinite(center)):
        raise ValueError("object_center_xyz must contain only finite values.")
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError("count must be positive.")
    if not all(
        math.isfinite(value)
        for value in (
            radius_min_m,
            radius_max_m,
            elevation_min_deg,
            elevation_max_deg,
        )
    ):
        raise ValueError("Radii and elevations must contain only finite values.")
    if radius_min_m <= 0.0 or radius_max_m < radius_min_m:
        raise ValueError("Radii must satisfy 0 < radius_min_m <= radius_max_m.")
    if not (-90.0 < elevation_min_deg <= elevation_max_deg < 90.0):
        raise ValueError("Elevations must satisfy -90 < min <= max < 90 degrees.")

    sin_min = math.sin(math.radians(elevation_min_deg))
    sin_max = math.sin(math.radians(elevation_max_deg))
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))

    viewpoints: list[ViewpointCandidate] = []
    for index in range(count):
        area_fraction = (index + 0.5) / count
        sin_elevation = sin_min + area_fraction * (sin_max - sin_min)
        elevation = math.asin(sin_elevation)
        azimuth = (index * golden_angle) % (2.0 * math.pi)

        radius_fraction = (
            0.5
            if radius_min_m == radius_max_m
            else _radical_inverse_base_two(index + 1)
        )
        radius = radius_min_m + radius_fraction * (radius_max_m - radius_min_m)
        horizontal = radius * math.cos(elevation)
        position = center + np.asarray(
            (
                horizontal * math.cos(azimuth),
                horizontal * math.sin(azimuth),
                radius * math.sin(elevation),
            ),
            dtype=float,
        )
        rotation = look_at_rotation_matrix(position, center)

        viewpoints.append(
            ViewpointCandidate(
                index=index,
                position_xyz=tuple(float(value) for value in position),
                orientation_xyzw=matrix_to_quaternion_xyzw(rotation),
                radius_m=float(radius),
                azimuth_deg=float(math.degrees(azimuth)),
                elevation_deg=float(math.degrees(elevation)),
            )
        )

    return viewpoints


def generate_four_equal_area_viewpoints(
    object_center_xyz: Iterable[float],
    radius_m: float,
    *,
    azimuth_offset_deg: float = 0.0,
) -> list[ViewpointCandidate]:
    """Return one representative for each of four equal-solid-angle regions.

    The upper hemisphere is partitioned analytically into a top cap with
    ``sin(elevation) >= 3/4`` and a lower band split into three 120-degree
    azimuth sectors. Every region therefore subtends exactly ``pi/2`` sr.
    The top-cap representative is its symmetry axis (the zenith). Side
    representatives use the midpoint in equal-area coordinate
    ``sin(elevation)`` and the centre azimuth of each sector.

    This is a direct O(1) construction, not a sampling or search algorithm.
    Orientations use the repository's tested +Z-forward look-at convention.
    """

    center = np.asarray(tuple(object_center_xyz), dtype=float)
    if center.shape != (3,):
        raise ValueError("object_center_xyz must contain exactly three values.")
    if not np.all(np.isfinite(center)):
        raise ValueError("object_center_xyz must contain only finite values.")
    if not math.isfinite(radius_m) or radius_m <= 0.0:
        raise ValueError("radius_m must be finite and positive.")
    if not math.isfinite(azimuth_offset_deg):
        raise ValueError("azimuth_offset_deg must be finite.")

    side_elevation_deg = math.degrees(math.asin(3.0 / 8.0))
    spherical_coordinates = (
        (0, 90.0, azimuth_offset_deg),
        (1, side_elevation_deg, azimuth_offset_deg + 60.0),
        (2, side_elevation_deg, azimuth_offset_deg + 180.0),
        (3, side_elevation_deg, azimuth_offset_deg + 300.0),
    )

    viewpoints: list[ViewpointCandidate] = []
    for index, elevation_deg, azimuth_deg in spherical_coordinates:
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
        viewpoints.append(
            ViewpointCandidate(
                index=index,
                position_xyz=tuple(float(value) for value in position),
                orientation_xyzw=matrix_to_quaternion_xyzw(rotation),
                radius_m=float(radius_m),
                azimuth_deg=float(azimuth_deg % 360.0),
                elevation_deg=float(elevation_deg),
            )
        )

    return viewpoints


def quaternion_angular_distance_deg(
    first_xyzw: Iterable[float], second_xyzw: Iterable[float]
) -> float:
    first = _normalize(np.asarray(tuple(first_xyzw), dtype=float))
    second = _normalize(np.asarray(tuple(second_xyzw), dtype=float))
    cosine = float(np.clip(abs(np.dot(first, second)), 0.0, 1.0))
    return math.degrees(2.0 * math.acos(cosine))
