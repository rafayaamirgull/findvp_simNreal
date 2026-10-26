"""Metric localization from RGB matches and calibrated robot camera motion.

Pixels must be undistorted, with matching pinhole intrinsics. Camera transforms
use optical axes: x right, y down, z forward. Translations are in metres. A
measured robot baseline provides scale; no depth image, mesh, simulator object
pose, or arbitrary monocular scale is used. A recovered visible-surface aiming
point is not the complete object's geometric center or its 6D pose.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TriangulationResult:
    """Arrays retain input match order; use ``inlier_mask`` before consuming points."""

    points_world_m: np.ndarray
    inlier_mask: np.ndarray
    reprojection_errors_px: np.ndarray
    parallax_deg: np.ndarray
    diagnostics: dict


class TriangulationRejected(ValueError):
    """A failed geometric estimate with diagnostics, never an accepted position."""
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def _validate_transform(value, name):
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f'{name} must be a finite 4x4 SE3 transform.')
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8, rtol=0):
        raise ValueError(f'{name} must have homogeneous last row [0, 0, 0, 1].')
    rotation = transform[:3, :3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6, rtol=0)):
        raise ValueError(f'{name} must contain a proper orthonormal rotation.')
    return transform


def _validate_intrinsics(value, name):
    intrinsics = np.asarray(value, dtype=np.float64)
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
        raise ValueError(f'{name} must be a finite 3x3 pinhole calibration.')
    if (intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0
            or not np.allclose(intrinsics[2], [0, 0, 1], atol=1e-10, rtol=0)
            or not np.isclose(intrinsics[1, 0], 0, atol=1e-10, rtol=0)):
        raise ValueError(f'{name} requires positive focal lengths and a pinhole calibration matrix.')
    return intrinsics


def world_from_camera(world_from_tool, tool_from_camera):
    """Compose measured forward kinematics with calibrated tool-to-optical transform."""
    return (_validate_transform(world_from_tool, 'world_from_tool')
            @ _validate_transform(tool_from_camera, 'tool_from_camera'))


def triangulate_matches(
    points_a_xy, points_b_xy, intrinsics_a, intrinsics_b,
    world_from_camera_a, world_from_camera_b, *,
    max_reprojection_error_px=2., min_parallax_deg=1.,
    min_baseline_m=.01, min_points=8,
):
    """Triangulate matched RGB features, rejecting unsupported metric estimates.

    Output arrays have one row per input match. Reprojection error is the larger
    of the two pixel errors. Invalid pixels/points have NaN coordinates or infinite
    errors and a false inlier flag. At least ``min_points`` must pass finite-value,
    positive-depth, reprojection, and parallax checks, otherwise ValueError is
    raised. Correspondences must describe a static object in a static world.
    """
    ka = _validate_intrinsics(intrinsics_a, 'intrinsics_a')
    kb = _validate_intrinsics(intrinsics_b, 'intrinsics_b')
    ta = _validate_transform(world_from_camera_a, 'world_from_camera_a')
    tb = _validate_transform(world_from_camera_b, 'world_from_camera_b')
    for name, value in [('max_reprojection_error_px', max_reprojection_error_px),
                        ('min_parallax_deg', min_parallax_deg),
                        ('min_baseline_m', min_baseline_m)]:
        if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive.')
    if min_parallax_deg >= 180:
        raise ValueError('min_parallax_deg must be below 180 degrees.')
    if isinstance(min_points, (bool, np.bool_)) or not isinstance(min_points, (int, np.integer)) or min_points < 1:
        raise ValueError('min_points must be a positive integer.')
    a, b = np.asarray(points_a_xy, dtype=np.float64), np.asarray(points_b_xy, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 2 or b.shape != a.shape:
        raise ValueError('Matched points must have equal Nx2 shapes.')
    if len(a) < min_points:
        raise ValueError(f'At least {min_points} matched points are required.')
    baseline = float(np.linalg.norm(ta[:3, 3] - tb[:3, 3]))
    if baseline < min_baseline_m:
        raise ValueError(f'Camera baseline {baseline:.6g} m is below {min_baseline_m:.6g} m.')

    finite_pixels = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    indices = np.flatnonzero(finite_pixels)
    if len(indices) < min_points:
        raise ValueError(f'Insufficient valid matched points: {len(indices)} < {min_points}.')
    # Normalize each camera separately, including skew and unequal image sizes.
    na = np.column_stack((a[indices], np.ones(len(indices)))) @ np.linalg.inv(ka).T
    nb = np.column_stack((b[indices], np.ones(len(indices)))) @ np.linalg.inv(kb).T
    pa, pb = np.linalg.inv(ta)[:3], np.linalg.inv(tb)[:3]
    system = np.stack((na[:, 0, None] * pa[2] - pa[0],
                       na[:, 1, None] * pa[2] - pa[1],
                       nb[:, 0, None] * pb[2] - pb[0],
                       nb[:, 1, None] * pb[2] - pb[1]), axis=1)
    _, _, vh = np.linalg.svd(system)
    homogeneous = vh[:, -1]
    points = np.full((len(a), 3), np.nan)
    finite_w = np.abs(homogeneous[:, 3]) > 1e-12
    points[indices[finite_w]] = homogeneous[finite_w, :3] / homogeneous[finite_w, 3, None]
    local_a = (points - ta[:3, 3]) @ ta[:3, :3]
    local_b = (points - tb[:3, 3]) @ tb[:3, :3]
    positive_depth = (local_a[:, 2] > 1e-9) & (local_b[:, 2] > 1e-9)
    errors = np.full(len(a), np.inf)
    valid = np.isfinite(points).all(axis=1) & positive_depth
    if valid.any():
        qa, qb = local_a[valid] @ ka.T, local_b[valid] @ kb.T
        errors[valid] = np.maximum(np.linalg.norm(qa[:, :2] / qa[:, 2, None] - a[valid], axis=1),
                                   np.linalg.norm(qb[:, :2] / qb[:, 2, None] - b[valid], axis=1))
    parallax = np.zeros(len(a))
    rays_a, rays_b = points[valid] - ta[:3, 3], points[valid] - tb[:3, 3]
    if len(rays_a):
        cosines = np.sum(rays_a * rays_b, axis=1) / (np.linalg.norm(rays_a, axis=1) * np.linalg.norm(rays_b, axis=1))
        parallax[valid] = np.rad2deg(np.arccos(np.clip(cosines, -1, 1)))
    inliers = valid & (errors <= max_reprojection_error_px) & (parallax >= min_parallax_deg)
    count = int(inliers.sum())
    finite_errors = errors[np.isfinite(errors)]
    checks = {
        'matches': len(a), 'finite_matches': int(finite_pixels.sum()),
        'positive_depth_points': int(valid.sum()), 'inliers': count,
        'baseline_m': baseline,
        'reprojection_pass': int((valid & (errors <= max_reprojection_error_px)).sum()),
        'parallax_pass': int((valid & (parallax >= min_parallax_deg)).sum()),
        'median_finite_reprojection_error_px': float(np.median(finite_errors)) if len(finite_errors) else None,
        'required_inliers': int(min_points),
        'max_reprojection_error_px': float(max_reprojection_error_px),
        'min_parallax_deg': float(min_parallax_deg),
    }
    if count < min_points:
        raise TriangulationRejected(
            f'Insufficient valid triangulation inliers: {count} < {min_points}; '
            'check matching, baseline, parallax, camera calibration, and hand-eye transform.', checks)
    diagnostics = {
        **checks,
        'matches': len(a), 'finite_matches': int(finite_pixels.sum()), 'inliers': count,
        'baseline_m': baseline, 'median_reprojection_error_px': float(np.median(errors[inliers])),
        'median_parallax_deg': float(np.median(parallax[inliers])),
        'metric_scale_source': 'measured_camera_baseline',
    }
    return TriangulationResult(points, inliers, errors, parallax, diagnostics)


def estimate_aim_point(points_world_m):
    """Return a robust visible-surface aiming point and spread diagnostics.

    Radial median-absolute-deviation trimming removes gross spatial outliers.
    Partial visibility and uneven feature coverage can bias this estimate; it is
    not a complete-object center, orientation, or formal uncertainty bound.
    """
    points = np.asarray(points_world_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError('Aim-point input must be a nonempty finite Nx3 point array.')
    median = np.median(points, axis=0)
    distances = np.linalg.norm(points - median, axis=1)
    median_distance = float(np.median(distances))
    mad = float(np.median(np.abs(distances - median_distance)))
    cutoff = median_distance + max(3.5 * 1.4826 * mad, 1e-9)
    keep = distances <= cutoff
    retained = points[keep]
    center = np.median(retained, axis=0)
    return center, {
        'estimate_type': 'visible_surface_aim_point',
        'input_points': len(points), 'retained_points': int(keep.sum()),
        'rejected_points': int((~keep).sum()),
        'median_surface_radius_m': float(np.median(np.linalg.norm(retained - center, axis=1))),
        'surface_extent_xyz_m': np.ptp(retained, axis=0).tolist(),
    }
