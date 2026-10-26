"""RGB bootstrap must replace the search's simulator target-location oracle."""

import json
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from sample import active_perception_triangles as active


class RGBLocalizationIntegrationTests(unittest.TestCase):
    @contextmanager
    def mocked_run(self, strategy="triangles", failure=None):
        """Keep real search geometry, decisions, and reporting; isolate hardware."""
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            args = active.build_parser().parse_args([
                "--target-image", "target.png", "--output-dir", directory,
                "--search-strategy", strategy, "--skip-radial-refinement",
                "--similarity-threshold", "0.97",
            ])
            args.object_localization = "rgb"
            args.localization_viewpoints = None
            center = np.array([0.32, -0.17, 0.86])
            report = {
                "center_world_m": center.tolist(),
                "captured_views": 2,
                "method": "rgb_multiview",
            }
            extractor, prepared_target = object(), object()
            env = MagicMock()
            for method in (
                "get_workpiece_vertex_centroid", "get_workpiece_center",
                "_has_object_line_of_sight",
            ):
                getattr(env, method).side_effect = AssertionError(
                    f"RGB localization must not consult {method}"
                )
            env.get_wrist_camera_pose.return_value = (
                np.array([0.4, -0.1, 1.3]), np.eye(3)
            )

            def reachable(candidates, object_center, config, **kwargs):
                np.testing.assert_allclose(object_center, center)
                self.assertFalse(config.require_line_of_sight)
                for candidate in candidates:
                    offset = np.asarray(candidate.position_xyz) - center
                    self.assertAlmostEqual(np.linalg.norm(offset), args.radius)
                    rotation = active.quaternion_xyzw_to_matrix(
                        candidate.orientation_xyzw
                    )
                    np.testing.assert_allclose(
                        rotation[:, 2], -offset / np.linalg.norm(offset), atol=1e-7
                    )
                return SimpleNamespace(
                    reachable=[SimpleNamespace(
                        viewpoint=candidate, manipulability=0.2, quality_score=0.8
                    ) for candidate in candidates],
                    rejected=[],
                )

            env.filter_reachable_viewpoints.side_effect = reachable
            replacements = {
                "UR5EyeInHandPyBulletEnv": env,
                "resolve_ik_backend": SimpleNamespace(
                    effective_backend="pybullet", uses_curobo=False
                ),
                "resolve_workpiece_ply": Path("object.ply"),
                "build_selected_candidate_backend": None,
                "_new_evaluation_tracker": {},
                "_write_evaluation_report": None,
            }
            for name, value in replacements.items():
                stack.enter_context(patch.object(active, name, return_value=value))
            home = stack.enter_context(patch.object(active, "move_home"))
            bootstrap = stack.enter_context(patch.object(
                active, "_bootstrap_rgb_localization", create=True,
                return_value=(report, extractor, prepared_target), side_effect=failure,
            ))
            constructor = stack.enter_context(patch(
                "sample.dinov3_feature_matching.DINOv3FeatureExtractor"
            ))
            preparation = stack.enter_context(patch(
                "sample.target_guided_similarity.prepare_target_image"
            ))
            capture = stack.enter_context(patch.object(
                active, "_capture_round",
                side_effect=lambda env, triangles, *args, **kwargs: len(triangles),
            ))
            ranking = stack.enter_context(patch(
                "sample.target_guided_similarity.rank_triangle_views",
                return_value={
                    "best_triangle": 1,
                    "ranking": [{"score": 0.99}],
                    "report_file": "ranking.json",
                    "best_image": "winner.png",
                    "best_viewpoint": {},
                },
            ))
            yield SimpleNamespace(
                args=args, env=env, bootstrap=bootstrap, report=report,
                extractor=extractor, prepared_target=prepared_target,
                constructor=constructor, preparation=preparation,
                capture=capture, ranking=ranking, home=home,
                directory=Path(directory),
            )

    def test_parser_preserves_simulator_default_and_accepts_rgb_viewpoints(self):
        self.assertEqual("simulator", active.build_parser().parse_args([]).object_localization)
        args = active.build_parser().parse_args([
            "--object-localization", "rgb", "--localization-viewpoints", "survey.json"
        ])
        self.assertEqual("rgb", args.object_localization)
        self.assertEqual(Path("survey.json"), args.localization_viewpoints)

    def test_measured_center_drives_every_strategy_and_reuses_bootstrap_models(self):
        for strategy in ("triangles", "nbv", "geometric"):
            with self.subTest(strategy=strategy), self.mocked_run(strategy) as case:
                active.run(case.args)
                case.bootstrap.assert_called_once()
                case.env.get_workpiece_vertex_centroid.assert_not_called()
                case.env.get_workpiece_center.assert_not_called()
                case.env._has_object_line_of_sight.assert_not_called()
                self.assertGreater(case.env.filter_reachable_viewpoints.call_count, 0)
                case.constructor.assert_not_called()
                case.preparation.assert_not_called()
                case.ranking.assert_called_once()
                self.assertIs(case.extractor, case.ranking.call_args.kwargs["extractor"])
                self.assertIs(
                    case.prepared_target, case.ranking.call_args.kwargs["prepared_target"]
                )
                case.env.disconnect.assert_called_once()

    def test_summary_includes_localization_provenance_and_bootstrap_capture_cost(self):
        with self.mocked_run("nbv") as case:
            active.run(case.args)
            summary = json.loads(
                (case.directory / "object_refinement_summary.json").read_text()
            )
            self.assertEqual("rgb", summary["object_localization"]["source"])
            self.assertEqual(case.report, summary["object_localization"]["report"])
            self.assertEqual(3, summary["total_images_captured"])
            self.assertEqual("winner.png", summary["final_best"]["best_image"])

    def test_bootstrap_failure_stops_search_and_still_cleans_up(self):
        with self.mocked_run(failure=RuntimeError("insufficient parallax")) as case:
            with self.assertRaisesRegex(RuntimeError, "insufficient parallax"):
                active.run(case.args)
            case.env.filter_reachable_viewpoints.assert_not_called()
            case.capture.assert_not_called()
            case.ranking.assert_not_called()
            case.constructor.assert_not_called()
            case.home.assert_called_once()
            case.env.disconnect.assert_called_once()
            ledger = json.loads(
                (case.directory / "object_end_effector_capture_poses.json").read_text()
            )
            self.assertEqual("incomplete", ledger["status"])


if __name__ == "__main__":
    unittest.main()
