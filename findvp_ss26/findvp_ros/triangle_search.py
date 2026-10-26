"""Hardware-independent spherical triangle candidates for the ROS runner."""

from __future__ import annotations

import math

import numpy as np

from sample.viewpoint_reachability import (
    ViewpointCandidate,
    look_at_rotation_matrix,
    matrix_to_quaternion_xyzw,
)


def _centroid(vertices: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    direction = np.mean(vertices, axis=0) - center
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        raise ValueError("spherical triangle has a degenerate centroid")
    return center + radius * direction / norm


def _area(vertices: np.ndarray, center: np.ndarray, radius: float) -> float:
    first, second, third = (vertices - center) / radius
    numerator = abs(float(np.dot(first, np.cross(second, third))))
    denominator = 1.0 + float(
        np.dot(first, second) + np.dot(second, third) + np.dot(third, first)
    )
    return 2.0 * math.atan2(numerator, denominator) * radius**2


def _viewpoint(index: int, position: np.ndarray, center: np.ndarray, radius: float):
    direction = position - center
    rotation = look_at_rotation_matrix(position, center)
    return ViewpointCandidate(
        index=index,
        position_xyz=tuple(float(value) for value in position),
        orientation_xyzw=matrix_to_quaternion_xyzw(rotation),
        radius_m=radius,
        azimuth_deg=math.degrees(math.atan2(direction[1], direction[0])) % 360.0,
        elevation_deg=math.degrees(math.asin(direction[2] / np.linalg.norm(direction))),
    )


def hemisphere_triangles(center_xyz, radius_m: float, count: int, azimuth_offset_deg=0.0):
    center = np.asarray(center_xyz, dtype=float)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("center_xyz must contain three finite values")
    if not math.isfinite(radius_m) or radius_m <= 0.0:
        raise ValueError("radius_m must be finite and positive")
    if isinstance(count, bool) or not isinstance(count, int) or count < 3:
        raise ValueError("count must be an integer of at least three")
    offset = math.radians(float(azimuth_offset_deg))
    top = center + (0.0, 0.0, radius_m)
    triangles, viewpoints = [], []
    for index in range(count):
        angles = offset + 2.0 * math.pi * np.asarray((index, index + 1)) / count
        vertices = np.asarray(
            (
                top,
                center + radius_m * np.asarray((math.cos(angles[0]), math.sin(angles[0]), 0.0)),
                center + radius_m * np.asarray((math.cos(angles[1]), math.sin(angles[1]), 0.0)),
            )
        )
        point = _centroid(vertices, center, radius_m)
        triangles.append(
            {
                "triangle": index + 1,
                "refinement_level": 0,
                "triangle_path": [index + 1],
                "parent_triangle_path": None,
                "vertices_xyz_m": vertices.tolist(),
                "centroid_xyz_m": point.tolist(),
                "surface_area_m2": 2.0 * math.pi * radius_m**2 / count,
            }
        )
        viewpoints.append(_viewpoint(index, point, center, radius_m))
    return triangles, viewpoints


def subdivide_spherical_triangle(parent, center_xyz, radius_m: float):
    center = np.asarray(center_xyz, dtype=float)
    vertices = np.asarray(parent.get("vertices_xyz_m"), dtype=float)
    if center.shape != (3,) or vertices.shape != (3, 3):
        raise ValueError("center and parent triangle shapes are invalid")
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(vertices)):
        raise ValueError("center and parent triangle must be finite")
    radii = np.linalg.norm(vertices - center, axis=1)
    if not np.allclose(radii, radius_m, atol=1e-7):
        raise ValueError("parent vertices must lie on the requested sphere")
    split = _centroid(vertices, center, radius_m)
    children = (
        np.asarray((vertices[0], vertices[1], split)),
        np.asarray((vertices[1], vertices[2], split)),
        np.asarray((vertices[2], vertices[0], split)),
    )
    parent_path = [int(value) for value in parent["triangle_path"]]
    level = int(parent["refinement_level"]) + 1
    triangles, viewpoints = [], []
    for index, child in enumerate(children):
        point = _centroid(child, center, radius_m)
        path = [*parent_path, index + 1]
        triangles.append(
            {
                "triangle": index + 1,
                "refinement_level": level,
                "triangle_path": path,
                "parent_triangle_path": parent_path,
                "vertices_xyz_m": child.tolist(),
                "centroid_xyz_m": point.tolist(),
                "surface_area_m2": _area(child, center, radius_m),
            }
        )
        viewpoints.append(_viewpoint(index, point, center, radius_m))
    return triangles, viewpoints

