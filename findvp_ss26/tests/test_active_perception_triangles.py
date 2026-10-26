import json
import math
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from sample.active_perception_triangles import (
    _apply_convergence_mode,
    _apply_search_mode,
    _capture_round,
    _compute_image_quality,
    _new_evaluation_tracker,
    _make_radial_sample,
    _record_motion_endpoint,
    _recover_unreachable_triangle_viewpoints,
    _resolve_target_camera_pose,
    _run_radial_refinement,
    _save_final_best_comparison,
    _target_guided_perception_enabled,
    _write_evaluation_report,
    build_parser,
    hemisphere_triangles,
    nearby_spherical_triangle_viewpoints,
    run,
    subdivide_spherical_triangle,
    write_capture_poses_ply,
)
from sample.viewpoint_reachability import quaternion_xyzw_to_matrix
from sample.workcell_config import ARM_JOINT_ORDER, HOME_JOINT_POSITIONS_RAD


class HemisphereTriangleTests(unittest.TestCase):
    def _run_score_sequence(self, scores, flags=(), capture_counts=None):
        """Exercise real search decisions/summary writing without GPU or robot motion."""
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            args = build_parser().parse_args(
                ["--target-image", "target.png", "--output-dir", directory, *flags]
            )
            env = MagicMock()
            env.get_workpiece_vertex_centroid.return_value = (0.0, 0.0, 0.0)
            env.get_wrist_camera_pose.return_value = (np.array([0.4, 0, 0.3]), np.eye(3))

            def reachable(candidates, *unused_args, **unused_kwargs):
                return SimpleNamespace(
                    reachable=[
                        SimpleNamespace(
                            viewpoint=candidate, manipulability=0.2, quality_score=0.8
                        )
                        for candidate in candidates
                    ],
                    rejected=[],
                )

            env.filter_reachable_viewpoints.side_effect = reachable
            module = "sample.active_perception_triangles."
            replacements = {
                "UR5EyeInHandPyBulletEnv": env,
                "resolve_ik_backend": SimpleNamespace(
                    effective_backend="pybullet", uses_curobo=False
                ),
                "resolve_workpiece_ply": Path("object.ply"),
                "build_selected_candidate_backend": None,
                "_new_evaluation_tracker": {},
                "_write_evaluation_report": None,
                "move_home": None,
            }
            for name, value in replacements.items():
                stack.enter_context(patch(module + name, return_value=value))
            counts = iter(capture_counts) if capture_counts is not None else None

            def fake_capture(env, triangles, *unused_args, **unused_kwargs):
                for triangle in triangles:
                    triangle["actual_camera_orientation_xyzw"] = triangle["camera_orientation_xyzw"]
                if counts is not None:
                    return next(counts)
                return 3 if args.search_strategy == "triangles" else 1

            capture = stack.enter_context(
                patch(
                    module + "_capture_round",
                    side_effect=fake_capture,
                )
            )
            radial = stack.enter_context(patch(module + "_run_radial_refinement"))
            stack.enter_context(
                patch("sample.dinov3_feature_matching.DINOv3FeatureExtractor")
            )
            stack.enter_context(patch("sample.target_guided_similarity.prepare_target_image"))
            ranking = stack.enter_context(
                patch(
                    "sample.target_guided_similarity.rank_triangle_views",
                    side_effect=[
                        {
                            "best_triangle": 1,
                            "ranking": [{"score": score}],
                            "report_file": "ranking.json",
                            "best_image": f"level_{index}.png",
                            "best_viewpoint": {},
                        }
                        for index, score in enumerate(scores)
                    ],
                )
            )
            run(args)
            summary = json.loads(
                (Path(directory) / "object_refinement_summary.json").read_text()
            )
            radial.assert_not_called()
            env.disconnect.assert_called_once()
            return summary, capture.call_count, ranking.call_count

    def test_geometric_no_supported_pose_preserves_best_and_finishes(self):
        def no_geometry(search, *unused_args):
            search.attempts[-1]["geometry"] = {
                "accepted": False, "reason": "insufficient_matches", "inliers": 0,
            }

        with patch(
            "sample.geometric_view_search.GeometricViewSearch.observe_image",
            autospec=True, side_effect=no_geometry,
        ):
            summary, captures, rankings = self._run_score_sequence(
                [0.7, 0.6, 0.65], ["--search-strategy", "geometric"],
            )
        self.assertEqual(3, captures)
        self.assertEqual(3, rankings)
        self.assertEqual("geometric_no_supported_pose", summary["termination_reason"])
        self.assertEqual(0.7, summary["final_best"]["best_score"])

    def test_geometric_uses_pose_estimate_before_remaining_seeds(self):
        def supported_geometry(search, *unused_args):
            search.attempts[-1]["geometry"] = {
                "accepted": True, "reason": "supported_relative_pose", "inliers": 30,
            }
            search.estimates.append({
                "position_xyz_m": [0, 0, 0.4],
                "camera_orientation_xyzw": [1, 0, 0, 0],
                "source_proposal_id": 1, "inliers": 30, "used": False,
            })

        with patch(
            "sample.geometric_view_search.GeometricViewSearch.observe_image",
            autospec=True, side_effect=supported_geometry,
        ):
            summary, captures, rankings = self._run_score_sequence(
                [0.7, 0.98], ["--search-strategy", "geometric", "--similarity-threshold", "0.97"],
            )
        self.assertEqual(2, captures)
        self.assertEqual(2, rankings)
        self.assertEqual("geometric_nbv", summary["search_mode"])
        self.assertEqual("similarity_threshold_reached", summary["termination_reason"])
        self.assertEqual("essential_matrix_pose", summary["next_best_view"]["attempts"][1]["selection"])

    def test_nbv_uses_one_image_per_decision_without_triangles_or_radial(self):
        with patch(
            "sample.active_perception_triangles.hemisphere_triangles",
            side_effect=AssertionError("NBV must not partition the hemisphere"),
        ), patch(
            "sample.active_perception_triangles.subdivide_spherical_triangle",
            side_effect=AssertionError("NBV must not subdivide triangles"),
        ):
            summary, captures, rankings = self._run_score_sequence(
                [0.4, 0.5, 0.6, 0.7, 0.71, 0.72, 0.73, 0.72],
                ["--search-strategy", "nbv", "--similarity-threshold", "0.99",
                 "--max-refinement-depth", "0"],
            )
        self.assertEqual(8, captures)
        self.assertEqual(8, rankings)
        self.assertEqual(8, summary["total_images_captured"])
        self.assertEqual("nbv_view_budget_reached", summary["angular_termination_reason"])
        self.assertEqual("next_best_view", summary["final_best"]["stage"])
        self.assertEqual(0.73, summary["final_best"]["best_score"])
        self.assertEqual("expected_improvement", summary["next_best_view"]["attempts"][3]["selection"])

    def test_nbv_recovers_from_failed_motion_without_training_on_failure(self):
        summary, captures, rankings = self._run_score_sequence(
            [0.4, 0.5, 0.6],
            ["--search-strategy", "nbv", "--nbv-max-views", "3"],
            capture_counts=[0, 1, 1, 1],
        )
        self.assertEqual(4, captures)
        self.assertEqual(3, rankings)
        attempts = summary["next_best_view"]["attempts"]
        self.assertEqual("rejected", attempts[0]["status"])
        self.assertNotIn("score", attempts[0])
        self.assertEqual("coverage_recovery", attempts[-1]["selection"])

    def test_nbv_failure_budget_preserves_empty_summary(self):
        summary, captures, rankings = self._run_score_sequence(
            [], ["--search-strategy", "nbv", "--nbv-max-views", "3"],
            capture_counts=[0] * 12,
        )
        self.assertEqual(12, captures)
        self.assertEqual(0, rankings)
        self.assertIsNone(summary["final_best"])
        self.assertEqual("nbv_attempt_budget_reached", summary["angular_termination_reason"])

    def test_nbv_plateau_waits_for_exploration_and_keeps_best(self):
        summary, _, rankings = self._run_score_sequence(
            [0.8, 0.7, 0.6, 0.81, 0.7, 0.6, 0.5],
            ["--search-strategy", "nbv"],
        )
        self.assertEqual(7, rankings)
        self.assertEqual("nbv_similarity_plateau", summary["angular_termination_reason"])
        self.assertEqual(0.81, summary["final_best"]["best_score"])

    def test_nbv_threshold_and_time_budget(self):
        summary, _, rankings = self._run_score_sequence(
            [0.95], ["--search-strategy", "nbv"],
        )
        self.assertEqual(1, rankings)
        self.assertEqual("similarity_threshold_reached", summary["angular_termination_reason"])
        summary, captures, rankings = self._run_score_sequence(
            [], ["--search-strategy", "nbv", "--nbv-time-budget", "0.000000000001"],
        )
        self.assertEqual(0, captures)
        self.assertEqual(0, rankings)
        self.assertEqual("nbv_time_budget_reached", summary["angular_termination_reason"])

    def test_nbv_validates_configuration_before_starting_simulation(self):
        for flags in (
            ["--nbv-max-views", "2"], ["--nbv-patience", "0"],
            ["--nbv-time-budget", "nan"], ["--nbv-time-budget", "-1"],
            ["--min-score-improvement", "0"], ["--until-convergence"],
            ["--radius", "0.1"],
        ):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                _apply_search_mode(build_parser().parse_args(
                    ["--search-strategy", "nbv", "--target-image", "target.png", *flags]
                ))
        with self.assertRaises(ValueError):
            _apply_search_mode(build_parser().parse_args(["--search-strategy", "nbv"]))

    def test_convergence_overrides_limits_and_stops_before_radial_search(self):
        summary, captures, rankings = self._run_score_sequence(
            [0.80, 0.90, 0.899],
            [
                "--until-convergence", "--views", "5",
                "--similarity-threshold", "0.1", "--max-refinement-depth", "0",
                "--refinement-patience", "100", "--min-score-improvement", "0.9",
                "--angular-lookahead-margin", "0", "--angular-lookahead-levels", "0",
            ],
        )
        self.assertEqual(3, captures)
        self.assertEqual(3, rankings)
        self.assertEqual(9, summary["total_images_captured"])
        self.assertEqual(3, summary["views"])
        self.assertIsNone(summary["similarity_threshold"])
        self.assertIsNone(summary["max_refinement_depth"])
        self.assertFalse(summary["radial_refinement_enabled"])
        self.assertEqual("similarity_converged", summary["angular_termination_reason"])
        self.assertTrue(summary["convergence"]["reached"])
        self.assertEqual(0.90, summary["final_best"]["best_score"])
        self.assertEqual("level_1.png", summary["final_best"]["best_image"])

    def test_convergence_keeps_small_gain_as_best_but_stops(self):
        summary, _, rankings = self._run_score_sequence(
            [0.8, 0.801], ["--until-convergence"]
        )
        self.assertEqual(2, rankings)
        self.assertEqual(0.801, summary["final_best"]["best_score"])
        self.assertEqual("similarity_converged", summary["angular_termination_reason"])

    def test_convergence_patience_resets_after_recovery(self):
        summary, _, rankings = self._run_score_sequence(
            [0.8, 0.7, 0.85, 0.85, 0.85],
            ["--until-convergence", "--convergence-patience", "2"],
        )
        self.assertEqual(5, rankings)
        self.assertEqual(0.85, summary["final_best"]["best_score"])
        self.assertEqual([0, 1, 0, 1, 2], [
            level["non_improving_levels"] for level in summary["levels"]
        ])

    def test_convergence_stops_when_no_more_views_can_be_captured(self):
        summary, captures, rankings = self._run_score_sequence(
            [0.8], ["--until-convergence"], capture_counts=[3, 0]
        )
        self.assertEqual(2, captures)
        self.assertEqual(1, rankings)
        self.assertEqual("no_reachable_viewpoints", summary["angular_termination_reason"])
        self.assertFalse(summary["convergence"]["reached"])
        self.assertEqual(0.8, summary["final_best"]["best_score"])

    def test_default_search_still_honors_threshold_and_depth(self):
        for flags, reason in (
            (["--similarity-threshold", "0.5"], "similarity_threshold_reached"),
            (["--max-refinement-depth", "0"], "maximum_refinement_depth_reached"),
        ):
            with self.subTest(reason=reason):
                summary, captures, _ = self._run_score_sequence(
                    [0.8], ["--skip-radial-refinement", *flags]
                )
                self.assertEqual(1, captures)
                self.assertEqual("bounded", summary["search_mode"])
                self.assertEqual(reason, summary["angular_termination_reason"])

    def test_convergence_requires_target_and_positive_finite_tolerance(self):
        for flags in (
            [],
            ["--target-image", "target.png", "--convergence-min-improvement", "0"],
            ["--target-image", "target.png", "--convergence-min-improvement", "nan"],
            ["--target-image", "target.png", "--convergence-min-improvement", "inf"],
            ["--target-image", "target.png", "--convergence-min-improvement", "1.1"],
            ["--target-image", "target.png", "--convergence-patience", "0"],
        ):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                _apply_convergence_mode(
                    build_parser().parse_args(["--until-convergence", *flags])
                )

    def test_radial_sample_preserves_direction_and_looks_at_object(self) -> None:
        center = np.asarray((0.45, -0.10, 0.80))
        direction = np.asarray((2.0, -1.0, 2.0))
        direction /= np.linalg.norm(direction)

        sample, viewpoint = _make_radial_sample(
            sample_id=7,
            radial_step_index=-3,
            center_xyz=center,
            radial_direction=direction,
            base_radius_m=0.40,
            step_size_m=0.02,
            angular_level=2,
            angular_triangle_path=[3, 2, 1],
        )

        offset = np.asarray(viewpoint.position_xyz) - center
        self.assertAlmostEqual(0.34, float(np.linalg.norm(offset)))
        np.testing.assert_allclose(offset / 0.34, direction, atol=1e-10)
        camera_rotation = quaternion_xyzw_to_matrix(viewpoint.orientation_xyzw)
        np.testing.assert_allclose(camera_rotation[:, 2], -direction, atol=1e-10)
        self.assertEqual("radial_refinement", sample["stage"])
        self.assertEqual("inside", sample["radial_direction"])
        self.assertEqual(-3, sample["radial_step_index"])
        self.assertAlmostEqual(-0.06, sample["radial_offset_m"])
        self.assertEqual([3, 2, 1], sample["triangle_path"])

    def test_nearby_fallbacks_stay_on_shell_and_close_to_centroid(self) -> None:
        center = np.asarray((0.45, -0.10, 0.80))
        radius = 0.40
        triangles, _ = hemisphere_triangles(tuple(center), radius, 3)
        triangle = triangles[0]
        viewpoints = nearby_spherical_triangle_viewpoints(
            triangle,
            tuple(center),
            radius,
            0.20,
        )

        self.assertEqual(3, len(viewpoints))
        centroid_direction = (
            np.asarray(triangle["centroid_xyz_m"]) - center
        ) / radius
        for viewpoint, vertex in zip(viewpoints, triangle["vertices_xyz_m"]):
            candidate_direction = (
                np.asarray(viewpoint.position_xyz) - center
            ) / radius
            vertex_direction = (np.asarray(vertex) - center) / radius
            self.assertAlmostEqual(1.0, float(np.linalg.norm(candidate_direction)))
            candidate_offset = math.acos(
                float(np.clip(np.dot(centroid_direction, candidate_direction), -1, 1))
            )
            vertex_offset = math.acos(
                float(np.clip(np.dot(centroid_direction, vertex_direction), -1, 1))
            )
            self.assertGreater(candidate_offset, 0.0)
            self.assertLess(candidate_offset, vertex_offset)

    def test_unreachable_triangles_use_fallbacks_then_are_individually_skipped(
        self,
    ) -> None:
        center = (0.45, -0.10, 0.80)
        radius = 0.40
        triangles, _ = hemisphere_triangles(center, radius, 3)
        for triangle in triangles[:2]:
            triangle["reachable"] = False
            triangle["rejection_reason"] = "ik_solver_no_solution"
            triangle["failure_counts"] = {"ik_solver_no_solution": 32}

        existing = SimpleNamespace(viewpoint=SimpleNamespace(index=2))

        class FakeEnvironment:
            def __init__(self):
                self.calls = 0

            def filter_reachable_viewpoints(
                self, candidates, object_center, config, candidate_backend=None
            ):
                self.calls += 1
                accepted = candidates[:1] if self.calls == 1 else []
                accepted_ids = {candidate.index for candidate in accepted}
                return SimpleNamespace(
                    reachable=[
                        SimpleNamespace(
                            viewpoint=candidate,
                            manipulability=0.2,
                            quality_score=0.8,
                        )
                        for candidate in accepted
                    ],
                    rejected=[
                        SimpleNamespace(
                            viewpoint=candidate,
                            furthest_failure_stage="ik_solver_no_solution",
                            failure_counts={"ik_solver_no_solution": 32},
                        )
                        for candidate in candidates
                        if candidate.index not in accepted_ids
                    ],
                )

        environment = FakeEnvironment()
        reachable = _recover_unreachable_triangle_viewpoints(
            environment,
            triangles,
            {2: existing},
            center,
            radius,
            SimpleNamespace(),
            None,
        )

        self.assertEqual({0, 2}, set(reachable))
        self.assertEqual(2, environment.calls)
        self.assertEqual("nearby_in_triangle_fallback", triangles[0]["viewpoint_strategy"])
        self.assertTrue(triangles[0]["reachable"])
        self.assertEqual(3, len(triangles[0]["fallback_attempts"]))
        self.assertFalse(triangles[1]["reachable"])
        self.assertTrue(triangles[1]["skipped"])
        self.assertEqual(
            "unreachable_after_nearby_fallbacks", triangles[1]["skip_reason"]
        )
        self.assertEqual(6, len(triangles[1]["fallback_attempts"]))

    def test_capture_round_cleanly_returns_when_every_triangle_is_skipped(self) -> None:
        triangles, _ = hemisphere_triangles((0.45, -0.10, 0.80), 0.40, 3)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            captured = _capture_round(
                None,
                triangles,
                {},
                root / "viewpoints.json",
                "object",
                0.0,
                None,
                [],
                {"captures": []},
                root / "poses.json",
                root / "poses.ply",
                None,
                return_home_after_round=False,
            )

            self.assertEqual(0, captured)
            self.assertTrue((root / "viewpoints.json").is_file())

    def test_capture_round_skips_an_individual_motion_planning_failure(self) -> None:
        triangles, _ = hemisphere_triangles((0.45, -0.10, 0.80), 0.40, 3)
        selected = SimpleNamespace(joint_positions_rad=[0.2] * 6)

        class HomeEnvironment:
            def get_joint_positions(self):
                return [HOME_JOINT_POSITIONS_RAD[name] for name in ARM_JOINT_ORDER]

        class FailingPlanner:
            def plan(self, start, goal):
                raise RuntimeError("no collision-free path")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            captured = _capture_round(
                HomeEnvironment(),
                triangles,
                {0: selected},
                root / "viewpoints.json",
                "object",
                0.0,
                FailingPlanner(),
                [],
                {"captures": []},
                root / "poses.json",
                root / "poses.ply",
                None,
                return_home_after_round=False,
            )

            self.assertEqual(0, captured)
            self.assertFalse(triangles[0]["reachable"])
            self.assertTrue(triangles[0]["skipped"])
            self.assertEqual("motion_planning_failed", triangles[0]["skip_reason"])
            self.assertIn("no collision-free path", triangles[0]["motion_planning_error"])

    def test_three_four_and_five_equal_triangles(self) -> None:
        center = np.asarray((0.45, -0.10, 0.80))
        radius = 0.40

        for count in (3, 4, 5):
            triangles, viewpoints = hemisphere_triangles(center, radius, count)

            self.assertEqual(count, len(triangles))
            self.assertEqual(count, len(viewpoints))
            for triangle, viewpoint in zip(triangles, viewpoints):
                self.assertAlmostEqual(
                    2.0 * math.pi * radius**2 / count,
                    triangle["surface_area_m2"],
                )
                offset = np.asarray(viewpoint.position_xyz) - center
                self.assertAlmostEqual(radius, float(np.linalg.norm(offset)))
                camera_rotation = quaternion_xyzw_to_matrix(
                    viewpoint.orientation_xyzw
                )
                np.testing.assert_allclose(
                    camera_rotation[:, 2], -offset / radius, atol=1e-10
                )

    def test_rejects_invalid_input(self) -> None:
        with self.assertRaises(ValueError):
            hemisphere_triangles((0.0, 0.0), 0.4, 3)
        with self.assertRaises(ValueError):
            hemisphere_triangles((0.0, 0.0, 0.0), 0.0, 3)
        with self.assertRaises(ValueError):
            hemisphere_triangles((0.0, 0.0, 0.0), 0.4, 2)
        with self.assertRaises(ValueError):
            hemisphere_triangles(
                (0.0, 0.0, 0.0), 0.4, 3, azimuth_offset_deg=float("nan")
            )

    def test_winning_triangle_is_recursively_split_into_three_children(self) -> None:
        center = (0.45, -0.10, 0.80)
        radius = 0.40
        parents, _ = hemisphere_triangles(center, radius, 3)

        children, viewpoints = subdivide_spherical_triangle(
            parents[1], center, radius
        )

        self.assertEqual(3, len(children))
        self.assertEqual(3, len(viewpoints))
        self.assertEqual([[2, 1], [2, 2], [2, 3]], [c["triangle_path"] for c in children])
        self.assertTrue(all(c["refinement_level"] == 1 for c in children))
        self.assertAlmostEqual(
            parents[1]["surface_area_m2"],
            sum(float(child["surface_area_m2"]) for child in children),
        )
        for child, viewpoint in zip(children, viewpoints):
            vertices = np.asarray(child["vertices_xyz_m"])
            np.testing.assert_allclose(
                np.linalg.norm(vertices - np.asarray(center), axis=1), radius
            )
            self.assertAlmostEqual(
                radius,
                float(np.linalg.norm(np.asarray(viewpoint.position_xyz) - center)),
            )

        grandchildren, _ = subdivide_spherical_triangle(
            children[2], center, radius
        )
        self.assertEqual([2, 3, 1], grandchildren[0]["triangle_path"])
        self.assertEqual(2, grandchildren[0]["refinement_level"])

    def test_refinement_cli_defaults_are_bounded_and_debug_is_opt_in(self) -> None:
        args = build_parser().parse_args([])
        self.assertEqual(
            "Adaptive Spherical + Radial (DINOv3+SIFT)", args.method_name
        )
        self.assertIsNone(args.target_pose_json)
        self.assertIsNone(args.target_camera_position)
        self.assertEqual(0.90, args.similarity_threshold)
        self.assertEqual(3, args.max_refinement_depth)
        self.assertEqual(2, args.refinement_patience)
        self.assertEqual(0.002, args.min_score_improvement)
        self.assertEqual(0.03, args.angular_lookahead_margin)
        self.assertEqual(1, args.angular_lookahead_levels)
        self.assertFalse(args.return_home_each_level)
        self.assertFalse(args.skip_radial_refinement)
        self.assertEqual(0.02, args.radial_step)
        self.assertEqual(2, args.radial_probe_steps)
        self.assertEqual(10, args.radial_max_steps)
        self.assertEqual(2, args.radial_patience)
        self.assertEqual(0.002, args.radial_min_improvement)
        self.assertEqual(0.20, args.radial_min_radius)
        self.assertEqual(0.60, args.radial_max_radius)
        self.assertEqual(448, args.feature_max_image_size)
        self.assertFalse(args.debug)
        self.assertTrue(build_parser().parse_args(["--debug"]).debug)

    def test_radial_search_selects_a_direction_and_stops_on_a_plateau(self) -> None:
        scores = {
            -1: 0.81,
            -2: 0.84,
            -3: 0.86,
            -4: 0.86,
            1: 0.81,
            2: 0.8105,
        }
        evaluated_steps: list[int] = []

        class FakeEnvironment:
            def filter_reachable_viewpoints(
                self, candidates, object_center, config, candidate_backend=None
            ):
                return SimpleNamespace(
                    reachable=[
                        SimpleNamespace(
                            viewpoint=candidate,
                            joint_positions_rad=[float(candidate.index)] * 6,
                            manipulability=0.2,
                            quality_score=0.8,
                        )
                        for candidate in candidates
                    ],
                    rejected=[],
                )

        def fake_capture(args, env, sample, selected, *unused_args):
            sample["joint_positions_rad"] = list(selected.joint_positions_rad)
            return True

        def fake_score(args, sample, *unused_args):
            step = int(sample["radial_step_index"])
            evaluated_steps.append(step)
            score = scores[step]
            sample.update(
                {
                    "score": score,
                    "best_image": f"radial_{step}.png",
                    "best_comparison": None,
                    "best_viewpoint": {"radial_step_index": step},
                }
            )
            return score

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(
                output_dir=root,
                name="object",
                motion_seconds=0.0,
                radial_step=0.02,
                radial_probe_steps=2,
                radial_max_steps=10,
                radial_patience=1,
                radial_min_improvement=0.002,
                radial_min_radius=0.20,
                radial_max_radius=0.60,
            )
            levels = [
                {
                    "stage": "angular_refinement",
                    "level": 1,
                    "best_triangle": 3,
                    "best_triangle_path": [3, 2],
                    "best_score": 0.80,
                    "best_image": "angular.png",
                    "best_comparison": None,
                    "best_viewpoint": {"actual_camera_xyz_m": [0.40, 0.0, 0.0]},
                }
            ]
            captures = {
                "captures": [
                    {
                        "stage": "angular_refinement",
                        "triangle_path": [3, 2],
                        "camera_pose_world": {
                            "position_xyz_m": [0.40, 0.0, 0.0]
                        },
                        "joint_positions_rad": [0.0] * 6,
                    }
                ]
            }
            with patch(
                "sample.active_perception_triangles._execute_joint_target",
                return_value={},
            ):
                with patch(
                    "sample.active_perception_triangles._capture_radial_sample",
                    side_effect=fake_capture,
                ):
                    with patch(
                        "sample.active_perception_triangles._score_radial_sample",
                        side_effect=fake_score,
                    ):
                        result = _run_radial_refinement(
                            args=args,
                            env=FakeEnvironment(),
                            object_center=(0.0, 0.0, 0.0),
                            config=SimpleNamespace(),
                            candidate_backend=None,
                            motion_planner=None,
                            outward_paths=[],
                            capture_pose_report=captures,
                            capture_pose_json_path=root / "poses.json",
                            capture_pose_ply_path=root / "poses.ply",
                            level_results=levels,
                            rank_triangle_views=None,
                            feature_extractor=None,
                            spatial_clusterer=None,
                            prepared_target=None,
                            evaluation_tracker=None,
                        )

        self.assertEqual([-1, -2, 1, 2, -3, -4], evaluated_steps)
        self.assertTrue(result["improved_over_angular_best"])
        self.assertEqual("inside", result["selected_direction"])
        self.assertEqual("radial_score_plateau", result["termination_reason"])
        self.assertEqual(-3, result["best_observation"]["best_radial_step_index"])

    def test_target_image_enables_search_without_debug(self) -> None:
        parser = build_parser()
        capture_only_args = parser.parse_args([])
        default_args = parser.parse_args(["--target-image", "target.png"])
        debug_args = parser.parse_args(
            ["--target-image", "target.png", "--debug"]
        )

        self.assertFalse(_target_guided_perception_enabled(capture_only_args))
        self.assertTrue(_target_guided_perception_enabled(default_args))
        self.assertTrue(_target_guided_perception_enabled(debug_args))

    def test_evaluation_tracker_records_camera_and_end_effector_motion(self) -> None:
        class FakeEnvironment:
            camera_position = np.asarray((0.0, 0.0, 0.0))
            end_effector_position = (0.0, 0.0, 0.0)

            def get_wrist_camera_pose(self):
                return self.camera_position, np.eye(3)

            def get_link_pose(self, link_name):
                self.asserted_link_name = link_name
                return self.end_effector_position, (0.0, 0.0, 0.0, 1.0)

        environment = FakeEnvironment()
        tracker = _new_evaluation_tracker(environment)
        environment.camera_position = np.asarray((0.03, 0.04, 0.0))
        environment.end_effector_position = (0.0, 0.0, 0.12)
        _record_motion_endpoint(environment, tracker, "angular_capture_target")

        self.assertAlmostEqual(0.05, tracker["camera_distance_traveled_m"])
        self.assertAlmostEqual(0.12, tracker["end_effector_distance_traveled_m"])
        self.assertEqual(
            "angular_capture_target", tracker["motion_segments"][0]["label"]
        )

    def test_target_pose_sidecar_is_matched_to_the_target_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target_path = root / "target.png"
            target_path.write_bytes(b"image-placeholder")
            sidecar_path = root / "target_pose.json"
            sidecar_path.write_text(
                """{
  "image_filename": "target.png",
  "camera_pose_world": {
    "position_xyz_m": [0.1, 0.2, 0.3],
    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]
  }
}\n""",
                encoding="utf-8",
            )
            args = SimpleNamespace(
                target_camera_position=None,
                target_image=target_path,
                target_pose_json=sidecar_path,
            )

            pose = _resolve_target_camera_pose(args)

            self.assertEqual([0.1, 0.2, 0.3], pose["position_xyz_m"])
            self.assertEqual([0.0, 0.0, 0.0, 1.0], pose["orientation_xyzw"])
            self.assertEqual(str(sidecar_path.resolve()), pose["source"])

    def test_evaluation_report_matches_the_competition_matrix_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target_path = root / "target.png"
            final_path = root / "final.png"
            pixels = np.full((8, 8, 3), 127, dtype=np.uint8)
            Image.fromarray(pixels).save(target_path)
            Image.fromarray(pixels).save(final_path)
            image_metrics = _compute_image_quality(
                target_path,
                final_path,
            )
            self.assertTrue(image_metrics["psnr_infinite"])
            self.assertEqual("inf", image_metrics["psnr_db"])
            self.assertAlmostEqual(1.0, image_metrics["ssim"])

            args = SimpleNamespace(
                method_name="Incumbent Angular + Radial",
                target_camera_position=[1.0, 0.0, 0.0],
                target_pose_json=None,
                target_image=target_path,
                output_dir=root,
                name="trial",
            )
            final_observation = {
                "stage": "angular_refinement",
                "level": 0,
                "best_score": 0.9,
                "best_image": str(final_path),
                "best_viewpoint": {"actual_camera_xyz_m": [0.0, 0.0, 0.0]},
            }
            captures = {
                "captures": [
                    {
                        "stage": "angular_refinement",
                        "image_file": str(final_path),
                        "camera_pose_world": {"position_xyz_m": [0.0, 0.0, 0.0]},
                    }
                ]
            }
            motion_tracker = {
                "definition": "test",
                "camera_distance_traveled_m": 0.05,
                "end_effector_distance_traveled_m": 0.06,
                "last_camera_xyz_m": [0.0, 0.0, 0.0],
                "last_end_effector_xyz_m": [0.0, 0.0, 0.0],
                "motion_segments": [],
            }
            report = _write_evaluation_report(
                args,
                captures,
                motion_tracker,
                execution_time_s=12.5,
                final_observation=final_observation,
                angular_termination_reason="similarity_threshold_reached",
                radial_result=None,
            )
            metrics = report["metrics"]
            self.assertEqual("Incumbent Angular + Radial", report["method"])
            self.assertEqual(1, metrics["n_views"])
            self.assertEqual(1.0, metrics["d_target_m"])
            self.assertEqual(0.05, metrics["d_traveled_m"])
            self.assertEqual(5.0, metrics["d_traveled_cm"])
            self.assertEqual(12.5, metrics["t_exec_s"])
            self.assertEqual("inf", metrics["psnr_db"])
            self.assertAlmostEqual(1.0, metrics["ssim"])
            self.assertEqual(1, report["aggregate_for_method"]["trial_count"])

            paths = report["report_files"]
            self.assertTrue(Path(paths["run_json"]).is_file())
            self.assertTrue(Path(paths["run_text"]).is_file())
            self.assertTrue(Path(paths["latest_json"]).is_file())
            self.assertTrue(Path(paths["latest_text"]).is_file())
            self.assertTrue(Path(paths["runs_csv"]).is_file())
            text_report = Path(paths["latest_text"]).read_text(encoding="utf-8")
            self.assertIn("Matrix-ready metrics (single run)", text_report)
            self.assertIn("d_target      = 1.0000 m", text_report)
            self.assertIn("d_traveled    = 5.00 cm", text_report)
            self.assertIn("PSNR          = inf dB", text_report)

            motion_tracker["camera_distance_traveled_m"] = 0.07
            second_report = _write_evaluation_report(
                args,
                captures,
                motion_tracker,
                execution_time_s=15.0,
                final_observation=final_observation,
                angular_termination_reason="maximum_refinement_depth_reached",
                radial_result=None,
            )
            aggregate = second_report["aggregate_for_method"]
            self.assertEqual(2, aggregate["trial_count"])
            self.assertAlmostEqual(
                6.0, aggregate["metrics"]["d_traveled_cm"]["mean"]
            )
            self.assertAlmostEqual(
                math.sqrt(2.0),
                aggregate["metrics"]["d_traveled_cm"]["sample_std"],
            )

    def test_final_best_comparison_gets_a_stable_output_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            source = output_dir / "level_02_best.png"
            source.write_bytes(b"target-and-best-view")

            saved = _save_final_best_comparison(
                {"best_comparison": str(source)},
                output_dir,
                "object",
            )

            self.assertEqual(output_dir / "object_best_view_with_target.png", saved)
            self.assertEqual(source.read_bytes(), saved.read_bytes())

    def test_meshlab_capture_poses_contain_one_oriented_vertex_per_image(self) -> None:
        report = {
            "captures": [
                {
                    "refinement_level": 0,
                    "image_filename": "object_view_01.png",
                    "end_effector_pose_world": {
                        "position_xyz_m": [0.0, 0.0, 0.75],
                        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                },
                {
                    "refinement_level": 1,
                    "image_filename": "object_level_01_view_01.png",
                    "end_effector_pose_world": {
                        "position_xyz_m": [0.1, 0.2, 0.8],
                        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                },
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture_poses.ply"

            write_capture_poses_ply(path, report)

            contents = path.read_text(encoding="utf-8")
            self.assertIn("element vertex 2", contents)
            self.assertIn("property float nx", contents)
            self.assertNotIn("element edge", contents)
            self.assertIn("0.0 0.0 0.75 0.0 0.0 1.0", contents)


if __name__ == "__main__":
    unittest.main()
