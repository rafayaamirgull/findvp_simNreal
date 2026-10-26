from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import numpy as np

from sample.capture_four_views import (
    FALLBACK_CAMERA_ROLLS_DEG,
    PRIMARY_CAMERA_ROLLS_DEG,
    find_reachable_region_alternatives,
    generate_region_viewpoint_alternatives,
    plan_validated_path,
)
from sample.viewpoint_reachability import quaternion_xyzw_to_matrix


class _FakeEnvironment:
    def __init__(self, validation) -> None:
        self.validation = validation

    def get_joint_positions(self) -> list[float]:
        return [0.0] * 6

    def validate_joint_trajectory(self, path, clearance):
        self.validated_path = np.asarray(path)
        self.validated_clearance = clearance
        return self.validation


class _FakeMotionPlanner:
    def __init__(self, path: np.ndarray) -> None:
        self.path = path

    def plan(self, start, goal):
        self.start = list(start)
        self.goal = list(goal)
        return SimpleNamespace(
            joint_positions_rad=self.path,
            planning_time_s=0.25,
            initialization_time_s=0.75,
            duration_s=0.5,
            used_graph=True,
            status="success",
        )


class _FakeReachabilityEnvironment:
    def __init__(self, reachable_by_roll) -> None:
        self.reachable_by_roll = reachable_by_roll
        self.calls = []

    def filter_reachable_viewpoints(
        self, viewpoints, object_center, config, candidate_backend=None
    ):
        self.calls.append((viewpoints, object_center, config, candidate_backend))
        reachable_indices = self.reachable_by_roll.get(config.roll_angles_deg, set())
        reachable = [
            SimpleNamespace(viewpoint=viewpoint, quality_score=1.0)
            for viewpoint in reversed(viewpoints)
            if viewpoint.index in reachable_indices
        ]
        return SimpleNamespace(reachable=reachable)


class CaptureMotionTests(unittest.TestCase):
    def test_curobo_path_is_revalidated_and_reported(self) -> None:
        goal = [0.2] * 6
        path = np.asarray(([0.0] * 6, [0.1] * 6, goal), dtype=float)
        planner = _FakeMotionPlanner(path)
        environment = _FakeEnvironment(
            SimpleNamespace(
                valid=True,
                samples_checked=3,
                minimum_clearance_m=0.012,
                failure_reason=None,
            )
        )

        result, metadata = plan_validated_path(environment, goal, planner)

        np.testing.assert_allclose(path, result)
        np.testing.assert_allclose(path, environment.validated_path)
        self.assertEqual("curobo_collision_aware_motion", metadata["planner"])
        self.assertAlmostEqual(0.75, metadata["planner_initialization_time_s"])
        self.assertAlmostEqual(1.0, metadata["total_planning_request_time_s"])
        self.assertEqual(3, metadata["pybullet_samples_checked"])
        self.assertAlmostEqual(0.012, metadata["pybullet_minimum_clearance_m"])

    def test_pybullet_rejection_stops_motion(self) -> None:
        goal = [0.2] * 6
        path = np.asarray(([0.0] * 6, goal), dtype=float)
        planner = _FakeMotionPlanner(path)
        environment = _FakeEnvironment(
            SimpleNamespace(
                valid=False,
                samples_checked=2,
                minimum_clearance_m=-0.01,
                failure_reason="environment_collision",
            )
        )

        with self.assertRaisesRegex(RuntimeError, "environment_collision"):
            plan_validated_path(environment, goal, planner)


class RegionFallbackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.center = (0.45, -0.10, 0.95)
        self.radius = 0.49
        self.reachability_kwargs = {
            "seed_count": 1,
            "refinement_steps": 0,
            "clearance_m": 0.005,
            "require_line_of_sight": True,
        }

    def test_alternatives_are_unique_and_stay_inside_their_regions(self) -> None:
        alternatives = generate_region_viewpoint_alternatives(
            self.center, self.radius, range(4)
        )

        indices = [alternative.viewpoint.index for alternative in alternatives]
        self.assertEqual(len(indices), len(set(indices)))
        for alternative in alternatives:
            viewpoint = alternative.viewpoint
            offset = np.asarray(viewpoint.position_xyz) - np.asarray(self.center)
            self.assertAlmostEqual(self.radius, float(np.linalg.norm(offset)))
            rotation = quaternion_xyzw_to_matrix(viewpoint.orientation_xyzw)
            np.testing.assert_allclose(
                rotation[:, 2], -offset / np.linalg.norm(offset), atol=1e-10
            )
            if alternative.region_index == 0:
                self.assertGreater(math.sin(math.radians(viewpoint.elevation_deg)), 0.75)
            else:
                center_azimuth = 60.0 + 120.0 * (alternative.region_index - 1)
                azimuth_delta = (
                    (viewpoint.azimuth_deg - center_azimuth + 180.0) % 360.0
                ) - 180.0
                self.assertLess(abs(azimuth_delta), 60.0)
                self.assertGreaterEqual(viewpoint.elevation_deg, 0.0)
                self.assertLess(
                    math.sin(math.radians(viewpoint.elevation_deg)), 0.75
                )

    def test_search_prefers_roll_zero_and_lowest_fallback_rank(self) -> None:
        alternatives = generate_region_viewpoint_alternatives(
            self.center, self.radius, [3]
        )
        environment = _FakeReachabilityEnvironment(
            {
                PRIMARY_CAMERA_ROLLS_DEG: {
                    alternatives[0].viewpoint.index,
                    alternatives[2].viewpoint.index,
                }
            }
        )

        selected, metadata = find_reachable_region_alternatives(
            environment,
            self.center,
            self.radius,
            [3],
            self.reachability_kwargs,
            candidate_backend="backend",
        )

        self.assertEqual(alternatives[0].viewpoint.index, selected[3].viewpoint.index)
        self.assertEqual(1, metadata[3]["fallback_rank"])
        self.assertEqual(1, len(environment.calls))
        self.assertEqual(PRIMARY_CAMERA_ROLLS_DEG, environment.calls[0][2].roll_angles_deg)
        self.assertEqual("backend", environment.calls[0][3])

    def test_search_uses_roll_redundancy_when_roll_zero_fails(self) -> None:
        alternatives = generate_region_viewpoint_alternatives(
            self.center, self.radius, [3]
        )
        environment = _FakeReachabilityEnvironment(
            {FALLBACK_CAMERA_ROLLS_DEG: {alternatives[1].viewpoint.index}}
        )

        selected, metadata = find_reachable_region_alternatives(
            environment,
            self.center,
            self.radius,
            [3],
            self.reachability_kwargs,
            candidate_backend=None,
        )

        self.assertEqual(alternatives[1].viewpoint.index, selected[3].viewpoint.index)
        self.assertEqual(2, metadata[3]["fallback_rank"])
        self.assertEqual(
            [PRIMARY_CAMERA_ROLLS_DEG, FALLBACK_CAMERA_ROLLS_DEG],
            [call[2].roll_angles_deg for call in environment.calls],
        )


if __name__ == "__main__":
    unittest.main()
