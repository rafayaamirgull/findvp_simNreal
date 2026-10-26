import math
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from sample.active_perception_triangles import _nbv_viewpoint, _apply_search_mode, build_parser
from sample.geometric_view_search import (
    GeometricViewSearch, camera_intrinsics, estimate_target_rotation,
)
from sample.viewpoint_reachability import look_at_rotation_matrix, quaternion_xyzw_to_matrix


class GeometricViewTests(unittest.TestCase):
    def _scene(self, target_below=False):
        rng = np.random.default_rng(2)
        points = rng.uniform(-0.04, 0.04, (100, 3))
        source = np.array([0.30, 0, 0.26])
        target = np.array([0.12, -0.30, -0.237 if target_below else 0.237])
        source_rotation = look_at_rotation_matrix(source, np.zeros(3))
        target_rotation = look_at_rotation_matrix(target, np.zeros(3))
        roll = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        target_rotation = target_rotation @ roll
        source_k = camera_intrinsics((640, 480))
        target_k = camera_intrinsics((1280, 960))

        def project(position, rotation, intrinsics):
            camera = (points - position) @ rotation
            pixels = camera @ intrinsics.T
            return pixels[:, :2] / pixels[:, 2:]

        source_xy = project(source, source_rotation, source_k)
        target_xy = project(target, target_rotation, target_k)
        target_xy[:10] = rng.uniform([0, 0], [1280, 960], (10, 2))
        return source_xy, target_xy, source_k, target_k, source_rotation, target_rotation

    def test_estimates_world_rotation_with_outliers_and_different_image_sizes(self):
        source, target, source_k, target_k, source_rotation, expected = self._scene()
        actual, evidence = estimate_target_rotation(source, target, source_k, target_k, source_rotation)
        self.assertTrue(evidence["accepted"], evidence)
        self.assertGreaterEqual(evidence["inliers"], 80)
        angle = math.degrees(math.acos(np.clip((np.trace(actual.T @ expected) - 1) / 2, -1, 1)))
        self.assertLess(angle, 1)

    def test_rejects_below_hemisphere_pose(self):
        source, target, source_k, target_k, source_rotation, _ = self._scene(target_below=True)
        actual, evidence = estimate_target_rotation(source, target, source_k, target_k, source_rotation)
        self.assertIsNone(actual)
        self.assertEqual("outside_upper_hemisphere", evidence["reason"])

    def test_insufficient_correspondences_do_not_produce_a_pose(self):
        rotation, evidence = estimate_target_rotation(np.zeros((4, 2)), np.zeros((4, 2)), np.eye(3), np.eye(3), np.eye(3))
        self.assertIsNone(rotation)
        self.assertEqual("insufficient_matches", evidence["reason"])

    def test_proposals_preserve_estimated_camera_roll(self):
        search = GeometricViewSearch((0, 0, 0), target_image=Path("target.png"))
        seed = search.propose([0.4, 0, 0])
        search.observe(seed["position_xyz_m"], 0.6)
        search.estimates.append({
            "position_xyz_m": [0.2, -0.2, math.sqrt(0.08)],
            "camera_orientation_xyzw": [0, 0, 1, 0],
            "source_proposal_id": 1, "inliers": 20, "used": False,
        })
        with patch("sample.next_best_view.BayesianViewSearch.predict", side_effect=AssertionError("No GP")):
            proposal = search.propose(seed["position_xyz_m"])
        self.assertEqual("essential_matrix_pose", proposal["selection"])
        records, candidates = _nbv_viewpoint(proposal, (0, 0, 0), 1, "geometric")
        self.assertEqual((0, 0, 1, 0), candidates[0].orientation_xyzw)
        self.assertEqual("geometric_nbv", records[0]["viewpoint_strategy"])
        search.reject("unreachable")
        next_seed = search.propose(seed["position_xyz_m"])
        self.assertEqual("initial_coverage", next_seed["selection"])
        self.assertEqual([0.6], search.scores)

    def test_search_ends_after_seeds_without_supported_estimates(self):
        search = GeometricViewSearch((0, 0, 0), target_image=Path("target.png"))
        for _ in range(3):
            proposal = search.propose([0.4, 0, 0])
            search.observe(proposal["position_xyz_m"], 0.5)
        self.assertIsNone(search.propose([0.4, 0, 0]))
        self.assertEqual(3, search.summary()["scored_views"])

    def test_nearby_pose_converges_but_different_roll_is_still_tried(self):
        for orientation, expected in (([1, 0, 0, 0], None), ([0, 1, 0, 0], "essential_matrix_pose")):
            with self.subTest(orientation=orientation):
                search = GeometricViewSearch((0, 0, 0), target_image=Path("target.png"))
                seed = search.propose([0.4, 0, 0])
                search.observe(seed["position_xyz_m"], 0.8)
                search.attempts[-1]["actual_orientation_xyzw"] = [1, 0, 0, 0]
                search.seed_index = 3
                search.estimates.append({
                    "position_xyz_m": seed["position_xyz_m"],
                    "camera_orientation_xyzw": orientation,
                    "source_proposal_id": 1, "inliers": 30, "used": False,
                })
                proposal = search.propose(seed["position_xyz_m"])
                if expected is None:
                    self.assertIsNone(proposal)
                    self.assertEqual("geometric_pose_converged", search.exhaustion_reason)
                else:
                    self.assertEqual(expected, proposal["selection"])

    def test_missing_descriptors_are_reported_without_an_estimate(self):
        search = GeometricViewSearch((0, 0, 0), target_image=Path("target.png"))
        proposal = search.propose([0.4, 0, 0])
        search.observe(proposal["position_xyz_m"], 0.5)
        with patch("sample.geometric_view_search.extract_geometry_features", return_value=(np.empty((0, 2)), None, np.eye(3))):
            search.observe_image(Path("source.png"), np.eye(3))
        self.assertEqual("no_descriptors", search.attempts[-1]["geometry"]["reason"])
        self.assertEqual([], search.estimates)

    def test_geometric_configuration_uses_shared_budget_aliases(self):
        args = build_parser().parse_args([
            "--search-strategy", "geometric", "--target-image", "target.png",
            "--search-max-views", "6", "--search-time-budget", "15",
        ])
        _apply_search_mode(args)
        self.assertEqual(6, args.nbv_max_views)
        self.assertEqual(15, args.nbv_time_budget)
        self.assertTrue(args.skip_radial_refinement)
        self.assertEqual("Geometric NBV (DINOv3+SIFT)", args.method_name)
        args.until_convergence = True
        with self.assertRaises(ValueError):
            _apply_search_mode(args)


if __name__ == "__main__":
    unittest.main()
