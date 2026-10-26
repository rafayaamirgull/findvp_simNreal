"""Survey acquisition uses RGB and measured camera calibration, never target pose."""

import json
import sys
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from sample import active_perception_triangles as active
from sample.rgb_localization_capture import bootstrap_rgb_localization
from sample.viewpoint_reachability import ReachabilityConfig


class RGBCaptureTests(unittest.TestCase):
    @contextmanager
    def fixture(self, survey=None, reachable=True):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            args = SimpleNamespace(
                output_dir=root, name="test", target_image=Path("target.png"),
                localization_viewpoints=None, feature_max_image_size=448,
                motion_seconds=0.0, debug=False, localization_min_similarity=.55,
                localization_ambiguity_margin=.03,
            )
            if survey is not None:
                args.localization_viewpoints = root / "survey.json"
                args.localization_viewpoints.write_text(json.dumps(survey))
            env = MagicMock()
            for name in ("get_workpiece_vertex_centroid", "get_workpiece_center"):
                getattr(env, name).side_effect = AssertionError("object pose oracle")

            def filter_views(candidates, center, config, **kwargs):
                self.assertFalse(config.require_line_of_sight)
                self.assertEqual((0.0,), config.roll_angles_deg)
                return SimpleNamespace(
                    reachable=[SimpleNamespace(
                        viewpoint=c, manipulability=.2, quality_score=.8
                    ) for c in candidates] if reachable else [],
                    rejected=[] if reachable else [SimpleNamespace(
                        viewpoint=c, furthest_failure_stage="unreachable", failure_counts={}
                    ) for c in candidates],
                )

            env.filter_reachable_viewpoints.side_effect = filter_views
            ledger = {"captures": []}
            actual_positions = []

            def capture(env, records, selected, coordinates_path, image_prefix,
                        motion_seconds, motion_planner, outward_paths, capture_report,
                        json_path, ply_path, tracker, **kwargs):
                count = 0
                for index, record in enumerate(records):
                    if index not in selected:
                        continue
                    position = np.asarray(selected[index].viewpoint.position_xyz) + [.001, .002, -.003]
                    actual_positions.append(position)
                    image_path = root / f"capture_{len(actual_positions)}.png"
                    Image.new("RGB", (64, 48), (100, 140, 200)).save(image_path)
                    capture_report["captures"].append({
                        "stage": "rgb_localization", "image_file": str(image_path),
                        "camera_pose_world": {
                            "position_xyz_m": position.tolist(),
                            "orientation_xyzw": [1., 0., 0., 0.],
                        },
                    })
                    count += 1
                return count

            capture_mock = stack.enter_context(patch.object(active, "_capture_round", side_effect=capture))
            extractor = object()
            target_features = object()
            prepared = SimpleNamespace(object_features=target_features)
            constructor = stack.enter_context(patch(
                "sample.dinov3_feature_matching.DINOv3FeatureExtractor", return_value=extractor
            ))
            stack.enter_context(patch(
                "sample.target_guided_similarity.prepare_target_image", return_value=prepared
            ))
            pipeline = MagicMock(return_value={
                "center_world_m": [.41, -.08, .87], "method": "rgb_multiview"
            })
            stub = SimpleNamespace(
                RGBObservation=lambda **kwargs: SimpleNamespace(**kwargs),
                localize_rgb_observations=pipeline,
            )
            stack.enter_context(patch.dict(sys.modules, {"sample.rgb_object_localization": stub}))
            invocation = (
                active, args, env, ReachabilityConfig(), None, None, [], ledger,
                root / "poses.json", root / "poses.ply", {},
            )
            yield SimpleNamespace(
                args=args, env=env, ledger=ledger, invocation=invocation,
                pipeline=pipeline, extractor=extractor, prepared=prepared,
                target_features=target_features, constructor=constructor,
                capture=capture_mock, actual_positions=actual_positions, root=root,
            )

    def test_default_survey_uses_measured_camera_poses_and_saves_replay_manifest(self):
        with self.fixture() as case:
            report, extractor, prepared = bootstrap_rgb_localization(*case.invocation)
            self.assertIs(case.extractor, extractor)
            self.assertIs(case.prepared, prepared)
            self.assertEqual(3, report["captured_views"])
            case.pipeline.assert_called_once()
            observations, target_features, used_extractor = case.pipeline.call_args.args
            self.assertIs(case.target_features, target_features)
            self.assertIs(case.extractor, used_extractor)
            self.assertEqual(3, len(observations))
            for observation, position in zip(observations, case.actual_positions):
                np.testing.assert_allclose(observation.world_from_camera[:3, 3], position)
                np.testing.assert_allclose(observation.world_from_camera[:3, :3], np.diag([1., -1., -1.]))
                self.assertEqual((64, 48), observation.image.size)
                self.assertEqual((3, 3), observation.intrinsics.shape)
            manifest = json.loads(Path(report["observations_manifest"]).read_text())
            self.assertEqual("pybullet_world", manifest["coordinate_frame"])
            self.assertEqual(3, len(manifest["observations"]))
            saved = json.loads(Path(report["report_file"]).read_text())
            self.assertEqual(report, saved)
            case.env.get_workpiece_vertex_centroid.assert_not_called()
            case.env.get_workpiece_center.assert_not_called()

    def test_custom_survey_poses_are_used_without_target_center(self):
        survey = {
            "coordinate_frame": "pybullet_world",
            "viewpoints": [
                {"position_xyz_m": [x, -.2, 1.4], "orientation_xyzw": [1, 0, 0, 0]}
                for x in (.3, .4)
            ],
        }
        with self.fixture(survey) as case:
            report, _, _ = bootstrap_rgb_localization(*case.invocation)
            self.assertEqual(2, report["captured_views"])
            np.testing.assert_allclose(case.actual_positions[0], [.301, -.198, 1.397])

    def test_invalid_entire_survey_rejected_before_any_motion_or_model_load(self):
        valid = {"position_xyz_m": [.3, -.1, 1.3], "orientation_xyzw": [1, 0, 0, 0]}
        surveys = [
            {"coordinate_frame": "robot_base", "viewpoints": [valid, valid]},
            {"coordinate_frame": "pybullet_world", "viewpoints": [valid]},
            {"coordinate_frame": "pybullet_world", "viewpoints": [valid, valid]},
            {"coordinate_frame": "pybullet_world", "viewpoints": [valid, {
                "position_xyz_m": [.4, -.1, 1.3], "orientation_xyzw": [0, 0, 0, 0]
            }]},
            {"coordinate_frame": "pybullet_world", "viewpoints": [valid, {
                "position_xyz_m": [float("nan"), -.1, 1.3], "orientation_xyzw": [1, 0, 0, 0]
            }]},
        ]
        for survey in surveys:
            with self.subTest(survey=survey), self.fixture(survey) as case:
                with self.assertRaises(ValueError):
                    bootstrap_rgb_localization(*case.invocation)
                case.capture.assert_not_called()
                case.constructor.assert_not_called()
                case.env.filter_reachable_viewpoints.assert_not_called()

    def test_invalid_threshold_fails_before_motion(self):
        with self.fixture() as case:
            case.args.localization_min_similarity = float("nan")
            with self.assertRaises(ValueError):
                bootstrap_rgb_localization(*case.invocation)
            case.capture.assert_not_called()
            case.constructor.assert_not_called()

    def test_insufficient_reachable_observations_fails_without_localizing(self):
        with self.fixture(reachable=False) as case:
            with self.assertRaisesRegex(RuntimeError, "observations|views|captures"):
                bootstrap_rgb_localization(*case.invocation)
            case.pipeline.assert_not_called()


if __name__ == "__main__":
    unittest.main()
