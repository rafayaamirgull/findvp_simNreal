import math
import unittest

import numpy as np

from sample.viewpoint_reachability import (
    DEFAULT_IK_ROLLS_DEG,
    ReachabilityConfig,
    generate_four_equal_area_viewpoints,
    quaternion_angular_distance_deg,
    quaternion_xyzw_to_matrix,
    sample_fibonacci_viewpoints,
)


class EqualAreaFourViewpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.center = np.asarray((0.45, -0.10, 0.86), dtype=float)

    def test_returns_exact_analytic_region_representatives(self) -> None:
        viewpoints = generate_four_equal_area_viewpoints(self.center, 0.4)

        self.assertEqual([0, 1, 2, 3], [view.index for view in viewpoints])
        self.assertEqual(
            [90.0, 60.0, 180.0, 300.0],
            [
                view.elevation_deg if view.index == 0 else view.azimuth_deg
                for view in viewpoints
            ],
        )
        expected_side_elevation = math.degrees(math.asin(3.0 / 8.0))
        for viewpoint in viewpoints[1:]:
            self.assertAlmostEqual(expected_side_elevation, viewpoint.elevation_deg)

    def test_all_points_lie_on_shell_and_look_at_object(self) -> None:
        for viewpoint in generate_four_equal_area_viewpoints(self.center, 0.4):
            offset = np.asarray(viewpoint.position_xyz) - self.center
            self.assertAlmostEqual(0.4, float(np.linalg.norm(offset)))
            rotation = quaternion_xyzw_to_matrix(viewpoint.orientation_xyzw)
            np.testing.assert_allclose(
                rotation[:, 2], -offset / np.linalg.norm(offset), atol=1e-10
            )
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-10)
            self.assertAlmostEqual(1.0, float(np.linalg.det(rotation)), places=10)

    def test_offset_rotates_side_sectors_without_changing_partition(self) -> None:
        viewpoints = generate_four_equal_area_viewpoints(
            self.center, 0.4, azimuth_offset_deg=15.0
        )
        self.assertEqual(
            [15.0, 75.0, 195.0, 315.0],
            [viewpoint.azimuth_deg for viewpoint in viewpoints],
        )

    def test_rejects_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            generate_four_equal_area_viewpoints((0.0, 0.0), 0.4)
        with self.assertRaises(ValueError):
            generate_four_equal_area_viewpoints((0.0, 0.0, float("nan")), 0.4)
        with self.assertRaises(ValueError):
            generate_four_equal_area_viewpoints(self.center, 0.0)
        with self.assertRaises(ValueError):
            generate_four_equal_area_viewpoints(
                self.center, 0.4, azimuth_offset_deg=float("inf")
            )


class FibonacciViewpointSamplingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.center = np.asarray((0.45, -0.10, 0.86), dtype=float)

    def test_default_sampler_returns_exactly_100_unique_deterministic_points(self) -> None:
        first = sample_fibonacci_viewpoints(self.center)
        second = sample_fibonacci_viewpoints(self.center)

        self.assertEqual(100, len(first))
        self.assertEqual(first, second)
        rounded_positions = {
            tuple(round(coordinate, 10) for coordinate in viewpoint.position_xyz)
            for viewpoint in first
        }
        self.assertEqual(100, len(rounded_positions))

    def test_default_reachability_keeps_camera_world_up(self) -> None:
        self.assertEqual((0.0,), DEFAULT_IK_ROLLS_DEG)
        self.assertEqual((0.0,), ReachabilityConfig().roll_angles_deg)

    def test_samples_stay_inside_requested_shell_and_elevation_band(self) -> None:
        viewpoints = sample_fibonacci_viewpoints(
            self.center,
            count=100,
            radius_min_m=0.30,
            radius_max_m=0.50,
            elevation_min_deg=12.0,
            elevation_max_deg=82.0,
        )

        for viewpoint in viewpoints:
            offset = np.asarray(viewpoint.position_xyz) - self.center
            radius = float(np.linalg.norm(offset))
            elevation = math.degrees(math.asin(offset[2] / radius))
            self.assertGreaterEqual(radius, 0.30 - 1e-12)
            self.assertLessEqual(radius, 0.50 + 1e-12)
            self.assertGreaterEqual(elevation, 12.0)
            self.assertLessEqual(elevation, 82.0)

    def test_every_camera_optical_axis_points_at_object_center(self) -> None:
        for viewpoint in sample_fibonacci_viewpoints(self.center):
            rotation = quaternion_xyzw_to_matrix(viewpoint.orientation_xyzw)
            expected_forward = self.center - np.asarray(viewpoint.position_xyz)
            expected_forward /= np.linalg.norm(expected_forward)
            np.testing.assert_allclose(rotation[:, 2], expected_forward, atol=1e-10)
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-10)
            self.assertAlmostEqual(1.0, float(np.linalg.det(rotation)), places=10)

    def test_roll_is_pose_redundancy_and_preserves_position_and_forward_axis(self) -> None:
        original = sample_fibonacci_viewpoints(self.center, count=1)[0]
        rolled = original.with_roll(90.0)
        original_rotation = quaternion_xyzw_to_matrix(original.orientation_xyzw)
        rolled_rotation = quaternion_xyzw_to_matrix(rolled.orientation_xyzw)

        self.assertEqual(original.position_xyz, rolled.position_xyz)
        np.testing.assert_allclose(original_rotation[:, 2], rolled_rotation[:, 2], atol=1e-10)
        self.assertAlmostEqual(
            90.0,
            quaternion_angular_distance_deg(
                original.orientation_xyzw, rolled.orientation_xyzw
            ),
            places=8,
        )

    def test_invalid_sampling_bounds_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            sample_fibonacci_viewpoints(self.center, count=0)
        with self.assertRaises(ValueError):
            sample_fibonacci_viewpoints(self.center, radius_min_m=0.5, radius_max_m=0.4)
        with self.assertRaises(ValueError):
            sample_fibonacci_viewpoints(
                self.center, elevation_min_deg=80.0, elevation_max_deg=10.0
            )
        with self.assertRaises(ValueError):
            sample_fibonacci_viewpoints(self.center, radius_min_m=float("nan"))
        with self.assertRaises(ValueError):
            sample_fibonacci_viewpoints((float("inf"), 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
