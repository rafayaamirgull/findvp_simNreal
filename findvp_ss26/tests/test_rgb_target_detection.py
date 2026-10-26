"""Scene selection tests use deterministic descriptors, without model downloads."""

import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image, ImageDraw

from sample.dinov3_feature_matching import DenseFeatures
from sample.rgb_target_detection import TargetDetectionError, detect_target_region


def features(vector=(1.0, 0.0)):
    descriptor = torch.tensor([vector], dtype=torch.float32)
    return DenseFeatures(descriptor, torch.tensor([[16., 16.]]), descriptor[0],
                         (1, 1), (32, 32), (32, 32))


def rectangle(x, y, width=24, height=24):
    mask = np.zeros((100, 160), dtype=bool)
    mask[y:y + height, x:x + width] = True
    return mask


class RGBTargetDetectionTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (160, 100), (180, 180, 180))
        self.target = features()
        self.extractor = object()

    def detect(self, **kwargs):
        return detect_target_region(self.image, self.target, self.extractor, **kwargs)

    @patch("sample.rgb_target_detection._extract_object_features")
    def test_off_center_identity_wins_over_central_distractor(self, extract):
        off_center, central = rectangle(8, 12), rectangle(68, 40)
        extract.side_effect = [features(), features((0., 1.))]
        result = self.detect(candidate_masks=[off_center, central])
        np.testing.assert_array_equal(result["mask"], off_center)
        self.assertEqual(result["bbox_xyxy"], (8, 12, 32, 36))
        self.assertAlmostEqual(result["similarity"], 1.)
        self.assertEqual(result["diagnostics"]["candidate_count"], 2)

    @patch("sample.rgb_target_detection._extract_object_features", return_value=features())
    def test_default_proposals_follow_shifted_target(self, extract):
        for x in (10, 120):
            self.image = Image.new("RGB", (160, 100), (180, 180, 180))
            ImageDraw.Draw(self.image).rectangle((x, 15, x + 23, 38), fill=(240, 30, 20))
            result = self.detect()
            self.assertTrue(result["mask"][25, x + 10])
            self.assertFalse(result["mask"][50, 80])
            self.assertEqual(result["mask"].dtype, np.bool_)
            self.assertEqual(result["diagnostics"]["proposal_source"], "flat_background")

    @patch("sample.rgb_target_detection._extract_object_features", return_value=features())
    def test_distinct_similar_objects_are_ambiguous(self, extract):
        with self.assertRaisesRegex(TargetDetectionError, "ambiguous"):
            self.detect(candidate_masks=[rectangle(8, 12), rectangle(90, 50)])

    @patch("sample.rgb_target_detection._extract_object_features", return_value=features())
    def test_overlapping_duplicate_proposals_are_not_distinct_objects(self, extract):
        result = self.detect(candidate_masks=[rectangle(8, 12), rectangle(9, 13)])
        self.assertAlmostEqual(result["similarity"], 1.)

    @patch("sample.rgb_target_detection._extract_object_features", return_value=features((0., 1.)))
    def test_low_similarity_is_rejected(self, extract):
        with self.assertRaisesRegex(TargetDetectionError, "similarity"):
            self.detect(candidate_masks=[rectangle(8, 12)])

    def test_uniform_scene_does_not_silently_select_whole_image(self):
        with self.assertRaisesRegex(TargetDetectionError, "candidate"):
            self.detect()

    def test_explicit_empty_candidates_do_not_fall_back(self):
        with self.assertRaisesRegex(TargetDetectionError, "candidate"):
            self.detect(candidate_masks=[])

    @patch("sample.rgb_target_detection._extract_object_features", return_value=features())
    def test_tiny_and_full_frame_masks_are_filtered(self, extract):
        valid = rectangle(8, 12)
        result = self.detect(candidate_masks=[rectangle(3, 4, 1, 1),
                                             np.ones((100, 160), bool), valid])
        np.testing.assert_array_equal(result["mask"], valid)
        self.assertEqual(extract.call_count, 1)

    def test_incorrect_mask_shape_and_nonbinary_values_are_rejected(self):
        for mask in (np.ones((20, 20), bool), np.full((100, 160), np.nan),
                     np.full((100, 160), .5)):
            with self.assertRaises(ValueError):
                self.detect(candidate_masks=[mask])

    def test_invalid_thresholds_are_rejected(self):
        for kwargs in ({"min_similarity": float("nan")}, {"min_similarity": -1},
                       {"ambiguity_margin": -1}, {"ambiguity_margin": float("inf")}):
            with self.assertRaises(ValueError):
                self.detect(candidate_masks=[rectangle(8, 12)], **kwargs)

    def test_real_crop_mask_and_similarity_helpers_select_shifted_reference(self):
        class ColourDescriptorExtractor:
            def extract(self, image, *, apply_background_filter):
                self.assert_unfiltered = not apply_background_filter
                rgb = np.asarray(image)
                descriptor = torch.tensor([[1., 0.]]) if np.any(
                    rgb[:, :, 0] > 220) else torch.tensor([[0., 1.]])
                return DenseFeatures(
                    descriptor, torch.tensor([[image.width / 2, image.height / 2]]),
                    descriptor[0], (1, 1), (image.height, image.width),
                    (image.height, image.width))

        drawing = ImageDraw.Draw(self.image)
        drawing.rectangle((8, 12, 31, 35), fill=(240, 30, 20))
        drawing.rectangle((68, 40, 91, 63), fill=(20, 30, 240))
        self.extractor = ColourDescriptorExtractor()
        result = self.detect()
        self.assertTrue(result["mask"][20, 15])
        self.assertFalse(result["mask"][50, 80])
        self.assertTrue(self.extractor.assert_unfiltered)

    @patch("sample.rgb_target_detection._extract_object_features")
    def test_model_failure_is_not_disguised_as_detection_failure(self, extract):
        extract.side_effect = ValueError("checkpoint incompatible")
        with self.assertRaisesRegex(ValueError, "checkpoint incompatible"):
            self.detect(candidate_masks=[rectangle(8, 12)])


if __name__ == "__main__":
    unittest.main()
