import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from sample.dinov3_feature_matching import (
    DINOv3FeatureExtractor,
    DenseFeatures,
    FeatureMatches,
    features_in_mask,
    features_in_roi,
    filter_matches_with_fundamental_ransac,
    load_image,
    match_features,
    offset_feature_keypoints,
    score_image_match,
    zero_mean_normalized_cross_correlation,
)


class _FakeProcessor:
    def __init__(self) -> None:
        self.last_options = None

    def __call__(self, *, images, return_tensors, **options):
        self.last_options = options
        width, height = images.size
        return {"pixel_values": torch.zeros(1, 3, height, width)}


class _FakeModel:
    config = SimpleNamespace(patch_size=16)

    def __init__(self) -> None:
        self.device = None
        self.training = True

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.training = False
        return self

    def __call__(self, *, pixel_values):
        grid_h = pixel_values.shape[-2] // 16
        grid_w = pixel_values.shape[-1] // 16
        # CLS + four register tokens + patch tokens.
        token_count = 5 + grid_h * grid_w
        tokens = torch.arange(token_count * 4, dtype=torch.float32).reshape(1, token_count, 4)
        return SimpleNamespace(last_hidden_state=tokens)


class _FakeTimmModel:
    patch_embed = SimpleNamespace(patch_size=(16, 16))
    pretrained_cfg = {
        "mean": (0.0, 0.0, 0.0),
        "std": (1.0, 1.0, 1.0),
    }

    def __init__(self) -> None:
        self.pixel_values = None

    def to(self, device):
        return self

    def eval(self):
        return self

    def forward_features(self, pixel_values):
        self.pixel_values = pixel_values.detach().cpu()
        grid_h = pixel_values.shape[-2] // 16
        grid_w = pixel_values.shape[-1] // 16
        return torch.ones(1, 5 + grid_h * grid_w, 4, device=pixel_values.device)


def _features(descriptors):
    descriptors = torch.tensor(descriptors, dtype=torch.float32)
    count = descriptors.shape[0]
    return DenseFeatures(
        descriptors=descriptors,
        keypoints=torch.arange(count * 2, dtype=torch.float32).reshape(count, 2),
        global_descriptor=torch.ones(descriptors.shape[1]),
        grid_size=(1, count),
        original_size=(10, 10),
        processed_size=(16, 16 * count),
    )


def _matches(points1, points2, scores):
    count = len(scores)
    return FeatureMatches(
        keypoints1=torch.tensor(points1, dtype=torch.float32),
        keypoints2=torch.tensor(points2, dtype=torch.float32),
        scores=torch.tensor(scores, dtype=torch.float32),
        indices1=torch.arange(count),
        indices2=torch.arange(count),
    )


class DINOv3FeatureMatchingTests(unittest.TestCase):
    def test_zncc_is_invariant_to_affine_brightness(self):
        first = np.arange(25, dtype=np.float64).reshape(5, 5)

        self.assertAlmostEqual(
            1.0,
            zero_mean_normalized_cross_correlation(first, 0.4 * first + 17.0),
        )
        self.assertAlmostEqual(
            -1.0,
            zero_mean_normalized_cross_correlation(first, -first + 100.0),
        )
        with self.assertRaisesRegex(ValueError, "near-constant"):
            zero_mean_normalized_cross_correlation(first, np.ones_like(first))

    def test_combined_score_uses_local_zncc_at_verified_matches(self):
        generator = np.random.default_rng(8)
        first_array = generator.integers(0, 256, size=(100, 100), dtype=np.uint8)
        second_array = np.zeros_like(first_array)
        second_array[:, 5:] = first_array[:, :-5]
        first = Image.fromarray(first_array, mode="L").convert("RGB")
        second = Image.fromarray(second_array, mode="L").convert("RGB")
        points1 = ((30, 30), (60, 30), (30, 60))
        points2 = ((35, 30), (65, 30), (35, 60))
        inliers = _matches(points1, points2, (0.9, 0.85, 0.8))
        candidates = _matches(
            (*points1, (60, 60)),
            (*points2, (10, 10)),
            (0.9, 0.85, 0.8, 0.7),
        )

        result = score_image_match(
            first,
            second,
            candidates,
            inliers,
            patch_radius_fraction=0.05,
            full_support_inliers=3,
        )

        self.assertGreater(result.median_zncc, 0.999)
        self.assertEqual(3, result.zncc_patch_count)
        self.assertAlmostEqual(0.75, result.geometric_inlier_ratio)
        self.assertGreater(result.score, 0.85)
        self.assertLessEqual(result.score, 1.0)

    def test_local_zncc_supports_different_image_sizes(self):
        coordinates = np.indices((80, 80)).sum(axis=0)
        pattern = ((coordinates % 16) * 16).astype(np.uint8)
        small = Image.fromarray(pattern, mode="L").convert("RGB")
        large = small.resize((160, 160), Image.Resampling.BILINEAR)
        small_points = ((20, 20), (40, 20), (20, 40), (40, 40))
        large_points = tuple((2 * x, 2 * y) for x, y in small_points)
        matches = _matches(small_points, large_points, (0.9,) * 4)

        result = score_image_match(
            small,
            large,
            matches,
            matches,
            patch_radius_fraction=0.05,
            full_support_inliers=4,
        )

        self.assertGreater(result.median_zncc, 0.95)

    def test_load_image_converts_to_rgb_and_detaches_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.png"
            Image.new("L", (8, 6), 127).save(path)
            image = load_image(path)
            path.unlink()
            self.assertEqual("RGB", image.mode)
            self.assertEqual((8, 6), image.size)
            self.assertEqual((127, 127, 127), image.getpixel((0, 0)))

    def test_load_image_can_convert_color_input_to_three_channel_grayscale(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.png"
            Image.new("RGB", (4, 3), (255, 0, 0)).save(path)

            image = load_image(path, grayscale=True)

            red, green, blue = image.getpixel((0, 0))
            self.assertEqual(red, green)
            self.assertEqual(green, blue)

    def test_extractor_returns_patch_centres_in_original_coordinates(self):
        model = _FakeModel()
        processor = _FakeProcessor()
        extractor = DINOv3FeatureExtractor(
            model=model,
            processor=processor,
            device="cpu",
            max_image_size=48,
        )

        features = extractor.extract(Image.new("RGB", (50, 34)))

        self.assertEqual((2, 3), features.grid_size)
        self.assertEqual((34, 50), features.original_size)
        self.assertEqual((32, 48), features.processed_size)
        self.assertEqual((6, 4), tuple(features.descriptors.shape))
        torch.testing.assert_close(
            features.keypoints[0], torch.tensor((8.0 * 50.0 / 48.0, 8.0 * 34.0 / 32.0))
        )
        torch.testing.assert_close(
            features.keypoints[-1], torch.tensor((40.0 * 50.0 / 48.0, 24.0 * 34.0 / 32.0))
        )
        self.assertEqual(
            {"do_resize": False, "do_center_crop": False}, processor.last_options
        )
        self.assertFalse(model.training)

    def test_timm_path_normalizes_pixels_without_an_external_processor(self):
        model = _FakeTimmModel()
        extractor = DINOv3FeatureExtractor(
            model=model,
            device="cpu",
            max_image_size=32,
        )

        features = extractor.extract(Image.new("RGB", (32, 32), (255, 127, 0)))

        self.assertEqual((2, 2), features.grid_size)
        self.assertEqual((4, 4), tuple(features.descriptors.shape))
        self.assertEqual((1, 3, 32, 32), tuple(model.pixel_values.shape))
        torch.testing.assert_close(
            model.pixel_values[0, :, 0, 0],
            torch.tensor((1.0, 127 / 255, 0.0)),
        )

    def test_small_images_can_be_upscaled_for_a_denser_patch_grid(self):
        model = _FakeTimmModel()
        extractor = DINOv3FeatureExtractor(
            model=model,
            device="cpu",
            max_image_size=64,
            upscale_small_images=True,
        )

        features = extractor.extract(Image.new("RGB", (32, 32)))

        self.assertEqual((4, 4), features.grid_size)
        self.assertEqual((16, 4), tuple(features.descriptors.shape))
        self.assertEqual((64, 64), features.processed_size)

    def test_white_background_filter_removes_background_only_patches(self):
        model = _FakeTimmModel()
        extractor = DINOv3FeatureExtractor(
            model=model,
            device="cpu",
            max_image_size=32,
            white_background_threshold=0.95,
            min_foreground_fraction=0.10,
        )
        image = Image.new("RGB", (32, 32), "white")
        for y in range(16):
            for x in range(16):
                image.putpixel((x, y), (0, 0, 0))

        features = extractor.extract(image)

        self.assertEqual((2, 2), features.grid_size)
        self.assertEqual((1, 4), tuple(features.descriptors.shape))
        torch.testing.assert_close(features.keypoints, torch.tensor(((8.0, 8.0),)))

        full_grid = extractor.extract(image, apply_background_filter=False)
        self.assertEqual((4, 4), tuple(full_grid.descriptors.shape))
        self.assertEqual((4, 2), tuple(full_grid.keypoints.shape))

    def test_roi_filter_preserves_full_image_coordinates(self):
        features = DenseFeatures(
            descriptors=torch.arange(12, dtype=torch.float32).reshape(4, 3),
            keypoints=torch.tensor(((8, 8), (24, 8), (8, 24), (24, 24))),
            global_descriptor=torch.ones(3),
            grid_size=(2, 2),
            original_size=(32, 32),
            processed_size=(32, 32),
        )

        selected = features_in_roi(features, (16, 0, 32, 32))

        torch.testing.assert_close(
            selected.keypoints, torch.tensor(((24, 8), (24, 24)))
        )
        torch.testing.assert_close(selected.descriptors, features.descriptors[[1, 3]])
        with self.assertRaisesRegex(ValueError, "outside"):
            features_in_roi(features, (-1, 0, 32, 32))

    def test_mask_filter_and_crop_offset_preserve_full_image_coordinates(self):
        features = DenseFeatures(
            descriptors=torch.arange(12, dtype=torch.float32).reshape(4, 3),
            keypoints=torch.tensor(((8, 8), (24, 8), (8, 24), (24, 24))),
            global_descriptor=torch.ones(3),
            grid_size=(2, 2),
            original_size=(32, 32),
            processed_size=(32, 32),
        )
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[:, 16:] = 255

        selected = features_in_mask(features, mask, min_mask_fraction=0.5)
        mapped = offset_feature_keypoints(selected, (10, 20), (80, 100))

        torch.testing.assert_close(
            mapped.keypoints, torch.tensor(((34, 28), (34, 44)), dtype=torch.float32)
        )
        self.assertEqual((80, 100), mapped.original_size)
        with self.assertRaisesRegex(ValueError, "complete row-major"):
            features_in_mask(selected, mask)

    def test_mutual_nearest_matching_filters_and_sorts(self):
        features1 = _features(((1, 0, 0), (0, 1, 0), (0, 0, 1)))
        features2 = _features(((0.99, 0.01, 0), (0, 0, 1), (-1, 0, 0)))

        matches = match_features(features1, features2, similarity_threshold=0.9)

        self.assertEqual(2, len(matches))
        self.assertEqual([2, 0], matches.indices1.tolist())
        self.assertEqual([1, 0], matches.indices2.tolist())
        self.assertGreaterEqual(matches.scores[0], matches.scores[1])

    def test_max_matches_limits_output(self):
        features = _features(((1, 0), (0, 1)))
        matches = match_features(features, features, similarity_threshold=0.0, max_matches=1)
        self.assertEqual(1, len(matches))

    def test_ambiguity_filters_remove_near_tied_matches(self):
        features1 = _features(((1, 0), (0, 1)))
        features2 = _features(((1, 0), (0.1, 0.995), (-0.1, 0.995)))

        matches = match_features(
            features1,
            features2,
            similarity_threshold=0.7,
            min_similarity_margin=0.01,
            max_distance_ratio=0.9,
            max_matches=None,
        )

        self.assertEqual(1, len(matches))
        self.assertEqual([0], matches.indices1.tolist())
        self.assertEqual([0], matches.indices2.tolist())

    def test_fundamental_ransac_rejects_geometric_outliers(self):
        generator = np.random.default_rng(4)
        inlier_count = 30
        outlier_count = 8
        points1 = generator.uniform((20, 20), (620, 460), size=(inlier_count, 2))
        disparities = generator.uniform(15, 80, size=inlier_count)
        points2 = points1.copy()
        points2[:, 0] -= disparities
        points2[:, 1] += generator.normal(0, 0.05, size=inlier_count)
        points1 = np.vstack(
            (points1, generator.uniform((20, 20), (620, 460), size=(outlier_count, 2)))
        )
        points2 = np.vstack(
            (points2, generator.uniform((20, 20), (620, 460), size=(outlier_count, 2)))
        )
        count = inlier_count + outlier_count
        matches = FeatureMatches(
            keypoints1=torch.tensor(points1, dtype=torch.float32),
            keypoints2=torch.tensor(points2, dtype=torch.float32),
            scores=torch.linspace(1.0, 0.8, count),
            indices1=torch.arange(count),
            indices2=torch.arange(count),
        )

        filtered = filter_matches_with_fundamental_ransac(
            matches,
            reprojection_threshold_px=0.5,
            max_iterations=2000,
        )

        self.assertGreaterEqual(len(filtered), 28)
        self.assertLessEqual(sum(index >= inlier_count for index in filtered.indices1), 1)

    def test_invalid_matching_arguments_are_rejected(self):
        features = _features(((1, 0),))
        with self.assertRaises(ValueError):
            match_features(features, features, similarity_threshold=1.1)
        with self.assertRaises(ValueError):
            match_features(features, features, max_matches=0)
        with self.assertRaises(ValueError):
            match_features(features, features, min_similarity_margin=-0.1)
        with self.assertRaises(ValueError):
            match_features(features, features, max_distance_ratio=1.1)
        with self.assertRaisesRegex(ValueError, "at least two"):
            match_features(features, features, min_similarity_margin=0.1)

    def test_invalid_background_filter_arguments_are_rejected(self):
        model = _FakeTimmModel()
        with self.assertRaises(ValueError):
            DINOv3FeatureExtractor(model=model, white_background_threshold=1.1)
        with self.assertRaises(ValueError):
            DINOv3FeatureExtractor(model=model, min_foreground_fraction=-0.1)


if __name__ == "__main__":
    unittest.main()
