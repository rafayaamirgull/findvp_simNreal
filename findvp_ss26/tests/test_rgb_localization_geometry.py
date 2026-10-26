"""Synthetic RGB-only geometry tests: metric scale comes from camera motion."""
import unittest

import numpy as np

from sample.rgb_localization_geometry import (
    estimate_aim_point,
    triangulate_matches,
    world_from_camera,
)


def pose(position, angle=0.0):
    c, s = np.cos(angle), np.sin(angle)
    result = np.eye(4)
    result[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    result[:3, 3] = position
    return result


def project(points, transform, intrinsics):
    local = (points - transform[:3, 3]) @ transform[:3, :3]
    homogeneous = local @ intrinsics.T
    return homogeneous[:, :2] / homogeneous[:, 2:]


class RGBLocalizationGeometryTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(812)
        self.points = rng.uniform([-0.08, -0.06, -0.05], [0.08, 0.06, 0.05], (60, 3)) + [1.2, -0.3, 1.7]
        self.a = pose([0.95, -0.3, 0.7], 0.13)
        self.b = pose([1.40, -0.3, 0.72], -0.12)
        self.ka = np.array([[620., 0, 320], [0, 625, 240], [0, 0, 1]])
        self.kb = np.array([[820., 0, 420], [0, 810, 300], [0, 0, 1]])
        self.xy_a = project(self.points, self.a, self.ka)
        self.xy_b = project(self.points, self.b, self.kb)

    def triangulate(self, **kwargs):
        values = dict(points_a_xy=self.xy_a, points_b_xy=self.xy_b,
                      intrinsics_a=self.ka, intrinsics_b=self.kb,
                      world_from_camera_a=self.a, world_from_camera_b=self.b)
        values.update(kwargs)
        return triangulate_matches(**values)

    def test_metric_recovery_off_origin_with_two_rotated_cameras(self):
        result = self.triangulate()
        self.assertTrue(result.inlier_mask.all())
        np.testing.assert_allclose(result.points_world_m, self.points, atol=1e-9)
        self.assertLess(result.reprojection_errors_px.max(), 1e-8)
        self.assertGreater(result.parallax_deg.min(), 10)
        self.assertAlmostEqual(result.diagnostics['baseline_m'], np.linalg.norm(self.a[:3, 3] - self.b[:3, 3]))

    def test_corrupt_matches_and_nonfinite_pixels_are_not_inliers(self):
        corrupted = self.xy_b.copy()
        corrupted[:10, 1] += 75
        corrupted[10] = [np.nan, 3]
        result = self.triangulate(points_b_xy=corrupted)
        self.assertFalse(result.inlier_mask[:11].any())
        self.assertTrue(result.inlier_mask[11:].all())
        np.testing.assert_allclose(result.points_world_m[11:], self.points[11:], atol=1e-9)

    def test_rejects_zero_baseline_and_pure_rotation(self):
        for b in (self.a.copy(), pose(self.a[:3, 3], -0.2)):
            with self.subTest(b=b), self.assertRaisesRegex(ValueError, 'baseline'):
                self.triangulate(world_from_camera_b=b)

    def test_rejects_insufficient_parallax(self):
        far = self.points + [0, 0, 1000]
        with self.assertRaisesRegex(ValueError, 'valid|inlier'):
            self.triangulate(points_a_xy=project(far, self.a, self.ka),
                             points_b_xy=project(far, self.b, self.kb))

    def test_rejects_negative_depth(self):
        behind = self.points - [0, 0, 3]
        with self.assertRaisesRegex(ValueError, 'valid|inlier'):
            self.triangulate(points_a_xy=project(behind, self.a, self.ka),
                             points_b_xy=project(behind, self.b, self.kb))

    def test_rejects_too_few_correspondences(self):
        with self.assertRaisesRegex(ValueError, 'points|matches|correspondences'):
            self.triangulate(points_a_xy=self.xy_a[:4], points_b_xy=self.xy_b[:4])

    def test_rejects_mismatched_or_invalid_pixel_shapes(self):
        for points in (self.xy_a[:-1], np.zeros((60, 3)), np.zeros(60)):
            with self.subTest(shape=points.shape), self.assertRaises(ValueError):
                self.triangulate(points_a_xy=points)

    def test_rejects_invalid_calibration(self):
        for bad in (np.zeros((3, 3)), np.eye(4), self.ka * np.nan,
                    np.diag([-1., 1, 1]), np.diag([1., 1, 2])):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.triangulate(intrinsics_a=bad)

    def test_rejects_invalid_se3(self):
        bad_rotation = self.a.copy()
        bad_rotation[0, 0] = 2
        reflection = np.diag([-1., 1, 1, 1])
        bad_row = self.a.copy()
        bad_row[3, 0] = 1
        for bad in (bad_rotation, reflection, bad_row, self.a * np.nan, np.eye(3)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.triangulate(world_from_camera_a=bad)

    def test_validates_quality_thresholds(self):
        for kwargs in ({'max_reprojection_error_px': -1}, {'min_parallax_deg': 0},
                       {'min_baseline_m': 0}, {'min_points': 0},
                       {'min_points': 2.5}, {'max_reprojection_error_px': np.nan}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.triangulate(**kwargs)

    def test_handeye_composition_transforms_tool_offset(self):
        base_tool = pose([1, 2, 3], np.pi / 2)
        tool_camera = pose([0.1, 0, 0.2], -0.1)
        actual = world_from_camera(base_tool, tool_camera)
        np.testing.assert_allclose(actual, base_tool @ tool_camera)
        np.testing.assert_allclose(actual[:3, 3], [1.2, 2, 2.9])
        with self.assertRaises(ValueError):
            world_from_camera(base_tool, np.zeros((4, 4)))

    def test_aim_point_is_robust_visible_surface_estimate(self):
        points = np.vstack([self.points, np.full((8, 3), 100.)])
        center, diagnostics = estimate_aim_point(points)
        np.testing.assert_allclose(center, np.median(self.points, axis=0), atol=0.015)
        self.assertEqual(diagnostics['estimate_type'], 'visible_surface_aim_point')
        self.assertGreaterEqual(diagnostics['rejected_points'], 8)

    def test_aim_point_handles_flat_and_identical_surfaces(self):
        center, _ = estimate_aim_point(np.tile([1., 2, 3], (12, 1)))
        np.testing.assert_allclose(center, [1, 2, 3])
        flat = self.points.copy()
        flat[:, 2] = 1
        center, _ = estimate_aim_point(flat)
        self.assertEqual(center[2], 1)

    def test_aim_point_rejects_empty_invalid_and_nonfinite_data(self):
        for points in (np.empty((0, 3)), np.zeros((8, 2)), np.full((8, 3), np.nan)):
            with self.subTest(shape=points.shape), self.assertRaises(ValueError):
                estimate_aim_point(points)


if __name__ == '__main__':
    unittest.main()
