import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from sample.rgb_object_localization import (
    RGBObservation, load_rgb_observations, localize_rgb_observations,
)


class RGBObjectLocalizationTests(unittest.TestCase):
    def observations(self):
        rng = np.random.default_rng(41)
        points = rng.uniform([-0.08, -0.025, 0.58], [-0.04, 0.025, 0.62], (40, 3))
        k = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1.0]])
        poses = [np.eye(4), np.eye(4)]
        poses[1][0, 3] = 0.08
        pixels = []
        for pose in poses:
            camera = points - pose[:3, 3]
            homogeneous = camera @ k.T
            pixels.append(homogeneous[:, :2] / homogeneous[:, 2:])
        observations = [RGBObservation(Image.new("RGB", (640, 480)), k, t, str(i)) for i, t in enumerate(poses)]
        return points, pixels, observations

    def test_pipeline_recovers_metric_aim_point_without_object_geometry(self):
        points, pixels, observations = self.observations()
        descriptors = np.eye(40, 128, dtype=np.float32) * 255
        detection = {"mask": np.ones((480, 640), dtype=bool), "bbox_xyxy": [0, 0, 640, 480], "similarity": 0.9, "diagnostics": {}}
        with patch("sample.rgb_object_localization.detect_target_region", return_value=detection), patch(
            "sample.rgb_object_localization.extract_masked_sift",
            side_effect=[(xy, descriptors) for xy in pixels],
        ):
            report = localize_rgb_observations(observations, object(), object())
        np.testing.assert_allclose(report["center_world_m"], np.median(points, axis=0), atol=0.005)
        self.assertEqual("visible_surface_aim_point", report["estimate_kind"])
        self.assertIsNone(report["orientation_world_xyzw"])
        self.assertEqual(2, report["input_views"])
        self.assertGreaterEqual(report["triangulated_points"], 30)

    def test_distorted_measurements_are_undistorted_before_triangulation(self):
        import cv2
        points, _, observations = self.observations()
        distortion = np.array([.3, -.08, .001, -.002, .01])
        pixels = []
        for observation in observations:
            observation.distortion_coefficients = distortion
            camera_points = points - observation.world_from_camera[:3, 3]
            xy, _ = cv2.projectPoints(camera_points, np.zeros(3), np.zeros(3),
                                      observation.intrinsics, distortion)
            pixels.append(xy.reshape(-1, 2))
        descriptors = np.eye(40, 128, dtype=np.float32) * 255
        detection = {"mask": np.ones((480, 640), dtype=bool), "bbox_xyxy": [0, 0, 640, 480], "similarity": .9, "diagnostics": {}}
        with patch("sample.rgb_object_localization.detect_target_region", return_value=detection), patch(
            "sample.rgb_object_localization.extract_masked_sift", side_effect=[(xy, descriptors) for xy in pixels],
        ):
            report = localize_rgb_observations(observations, object(), object())
        np.testing.assert_allclose(report["center_world_m"], np.median(points, axis=0), atol=.005)

    def test_single_view_fails_without_inventing_metric_depth(self):
        _, _, observations = self.observations()
        with self.assertRaisesRegex(ValueError, "two"):
            localize_rgb_observations(observations[:1], object(), object())

    def test_no_object_features_fails_without_fallback_center(self):
        _, _, observations = self.observations()
        detection = {"mask": np.ones((480, 640), dtype=bool), "bbox_xyxy": [0, 0, 640, 480], "similarity": 0.9, "diagnostics": {}}
        with patch("sample.rgb_object_localization.detect_target_region", return_value=detection), patch(
            "sample.rgb_object_localization.extract_masked_sift", return_value=(np.empty((0, 2)), None),
        ), self.assertRaisesRegex(ValueError, "triangulation"):
            localize_rgb_observations(observations, object(), object())

    def test_loader_composes_handeye_and_reads_mask_relative_to_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (32, 24)).save(root / "view.png")
            Image.new("L", (32, 24), 255).save(root / "mask.png")
            tool = np.eye(4); tool[:3, 3] = [0.3, 0.1, 0.8]
            handeye = np.eye(4); handeye[0, 3] = 0.06
            record = {
                "image": "view.png", "mask": "mask.png",
                "intrinsics": [[100, 0, 16], [0, 100, 12], [0, 0, 1]],
                "world_from_tool": tool.tolist(), "tool_from_camera": handeye.tolist(),
            }
            path = root / "observations.json"
            path.write_text(json.dumps({"coordinate_frame": "robot_base", "observations": [record, record]}))
            frame, observations = load_rgb_observations(path)
            self.assertEqual("robot_base", frame)
            np.testing.assert_allclose(observations[0].world_from_camera, tool @ handeye)
            self.assertEqual((24, 32), observations[0].candidate_masks[0].shape)

    def test_loader_requires_calibration_and_named_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "bad.json"
            path.write_text(json.dumps({"observations": []}))
            with self.assertRaises(ValueError):
                load_rgb_observations(path)


if __name__ == "__main__":
    unittest.main()
