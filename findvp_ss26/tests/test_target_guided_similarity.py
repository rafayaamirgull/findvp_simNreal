from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from sample.dinov3_feature_matching import DenseFeatures
from sample.dinov3_spatial_clustering import SpatialClusterResult
from sample.target_guided_similarity import (
    SIFTScore,
    _central_object_region,
    combine_scores,
    prepare_target_image,
    rank_triangle_views,
    sift_similarity,
)


class _FakeExtractor:
    def __init__(self):
        self.calls = 0

    def extract(self, image, *, apply_background_filter):
        self.calls += 1
        self.last_background_filter = apply_background_filter
        if image.getpixel((0, 0))[0] > 128:
            descriptor = torch.tensor([[1.0, 0.0]])
        else:
            descriptor = torch.tensor([[0.0, 1.0]])
        return DenseFeatures(
            descriptors=descriptor,
            keypoints=torch.tensor([[0.5, 0.5]]),
            global_descriptor=descriptor[0],
            grid_size=(1, 1),
            original_size=(image.height, image.width),
            processed_size=(image.height, image.width),
        )


class _FakeClusterer:
    def __init__(self):
        self.calls = 0

    def fit_predict(self, features, *, image):
        self.calls += 1
        return SpatialClusterResult(
            labels=torch.zeros((1, 1), dtype=torch.long),
            object_mask=torch.ones((1, 1), dtype=torch.bool),
            object_indices=torch.tensor([0]),
            selected_cluster_ids=(0,),
            metrics=(),
            pca_components=1,
            explained_variance_ratio=1.0,
            inertia=0.0,
            random_state=0,
        )


def _fake_sift(target, candidate, target_mask, candidate_mask):
    score = 0.8 if candidate.getpixel((0, 0))[0] > 128 else 0.2
    return SIFTScore(score, 10, 10, 8, 6, 0.75)


class TargetGuidedSimilarityTests(unittest.TestCase):
    def test_real_scene_ranking_uses_target_masks_not_central_blob(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.png"
            candidate = root / "candidate.png"
            Image.new("RGB", (32, 32), (255, 0, 0)).save(target)
            Image.new("RGB", (32, 32), (255, 20, 20)).save(candidate)
            views = root / "views.json"
            views.write_text(json.dumps([{"triangle": 1, "image_filename": candidate.name,
                                          "observation": {"stamp_ns": 123}}]))
            extractor = _FakeExtractor()
            prepared = prepare_target_image(target, extractor=extractor)
            proposals = [np.ones((32, 32), dtype=bool)]
            with patch("sample.target_guided_similarity._central_object_region",
                       side_effect=AssertionError("must not select central blob")), patch(
                "sample.rgb_target_detection.detect_target_region",
                return_value={"mask": proposals[0], "bbox_xyxy": (0, 0, 32, 32)},
            ) as detector:
                result = rank_triangle_views(
                    target, views, extractor=extractor, prepared_target=prepared,
                    target_guided_candidates=True, candidate_mask_provider=lambda _: proposals,
                    sift_scorer=_fake_sift)
            self.assertIs(detector.call_args.kwargs["candidate_masks"], proposals)
            self.assertEqual(result["best_viewpoint"]["observation"]["stamp_ns"], 123)
            self.assertIn("target-guided", result["method"])

    def test_prepared_target_is_reused_across_ranking_rounds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.png"
            candidate = root / "candidate.png"
            Image.new("RGB", (32, 32), (255, 0, 0)).save(target)
            Image.new("RGB", (32, 32), (255, 20, 20)).save(candidate)
            viewpoints = root / "views.json"
            viewpoints.write_text(
                json.dumps(
                    [
                        {
                            "triangle": 1,
                            "image_filename": candidate.name,
                            "actual_camera_xyz_m": [1, 2, 3],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            extractor = _FakeExtractor()
            clusterer = _FakeClusterer()
            prepared = prepare_target_image(
                target,
                extractor=extractor,
                debug=True,
                clusterer=clusterer,
            )

            self.assertEqual(2, extractor.calls)
            self.assertEqual(1, clusterer.calls)
            for index in range(2):
                rank_triangle_views(
                    target,
                    viewpoints,
                    output_path=root / f"ranking_{index}.json",
                    visualization_dir=root / f"visualizations_{index}",
                    debug=True,
                    extractor=extractor,
                    clusterer=clusterer,
                    prepared_target=prepared,
                    sift_scorer=_fake_sift,
                )

            # Each round performs object-crop and full-frame extraction only
            # for its candidate; the target's two DINO passes and clustering
            # happened once in prepare_target_image.
            self.assertEqual(6, extractor.calls)
            self.assertEqual(3, clusterer.calls)

    def test_central_object_mask_rejects_background_and_border_occluder(self):
        image = Image.new("RGB", (160, 120), (201, 184, 153))
        draw = ImageDraw.Draw(image)
        draw.ellipse((55, 30, 105, 90), fill=(120, 145, 170))
        draw.rectangle((130, 45, 159, 75), fill=(245, 245, 245))

        region = _central_object_region(image)

        self.assertTrue(region.mask[60, 80])
        self.assertFalse(region.mask[5, 5])
        self.assertFalse(region.mask[60, 150])
        self.assertLess(region.bbox_xyxy[2], 130)

    def test_weighted_score_is_normalized_and_validated(self):
        self.assertAlmostEqual(0.68, combine_scores(0.8, 0.5))
        self.assertAlmostEqual(
            0.5, combine_scores(1.0, 0.0, dino_weight=1.0, sift_weight=1.0)
        )
        with self.assertRaises(ValueError):
            combine_scores(1.1, 0.5)
        with self.assertRaises(ValueError):
            combine_scores(0.5, 0.5, dino_weight=0.0, sift_weight=0.0)

    def test_sift_scores_an_identical_textured_image_highly(self):
        generator = np.random.default_rng(7)
        pixels = generator.integers(0, 256, size=(160, 160), dtype=np.uint8)
        image = Image.fromarray(pixels, mode="L").convert("RGB")
        mask = np.full((160, 160), 255, dtype=np.uint8)

        result = sift_similarity(image, image, mask, mask)

        self.assertGreater(result.good_matches, 12)
        self.assertGreater(result.inliers, 12)
        self.assertGreater(result.score, 0.95)
        self.assertEqual((result.inliers, 2), result.target_inlier_xy.shape)
        self.assertEqual((result.inliers, 2), result.candidate_inlier_xy.shape)

    def test_ranking_skips_debug_artifacts_by_default_and_can_enable_them(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.png"
            good = root / "good.png"
            bad = root / "bad.png"
            Image.new("RGB", (32, 32), (255, 0, 0)).save(target)
            Image.new("RGB", (32, 32), (255, 20, 20)).save(good)
            Image.new("RGB", (32, 32), (0, 0, 255)).save(bad)
            viewpoints = root / "views.json"
            viewpoints.write_text(
                json.dumps(
                    [
                        {
                            "triangle": 1,
                            "image_filename": bad.name,
                            "actual_camera_xyz_m": [1, 2, 3],
                        },
                        {
                            "triangle": 2,
                            "image_filename": good.name,
                            "actual_camera_xyz_m": [4, 5, 6],
                        },
                    ]
                ),
                encoding="utf-8",
            )

            default_extractor = _FakeExtractor()
            default_clusterer = _FakeClusterer()
            default_report = rank_triangle_views(
                target,
                viewpoints,
                ambiguity_margin=0.05,
                extractor=default_extractor,
                clusterer=default_clusterer,
                sift_scorer=_fake_sift,
            )

            self.assertEqual(2, default_report["best_triangle"])
            self.assertFalse(default_report["debug_visualizations"])
            self.assertIsNone(default_report["visualizations"])
            self.assertFalse((root / "views_visualizations").exists())
            self.assertEqual(3, default_extractor.calls)
            self.assertEqual(0, default_clusterer.calls)
            default_saved = json.loads(
                Path(default_report["report_file"]).read_text(encoding="utf-8")
            )
            self.assertFalse(default_saved["debug_visualizations"])
            self.assertIsNone(default_saved["visualizations"])

            debug_extractor = _FakeExtractor()
            debug_clusterer = _FakeClusterer()
            report = rank_triangle_views(
                target,
                viewpoints,
                ambiguity_margin=0.05,
                debug=True,
                extractor=debug_extractor,
                clusterer=debug_clusterer,
                sift_scorer=_fake_sift,
            )

            self.assertEqual(2, report["best_triangle"])
            self.assertEqual([2], report["plausible_triangles"])
            self.assertEqual([4, 5, 6], report["best_viewpoint"]["actual_camera_xyz_m"])
            self.assertEqual(str(good.resolve()), report["best_image"])
            self.assertTrue(report["debug_visualizations"])
            self.assertEqual(6, debug_extractor.calls)
            self.assertEqual(3, debug_clusterer.calls)
            visualizations = report["visualizations"]
            self.assertTrue(Path(visualizations["target_clustering"]).is_file())
            self.assertTrue(Path(visualizations["best_comparison"]).is_file())
            for triangle_id in ("1", "2"):
                self.assertTrue(
                    Path(
                        visualizations["triangles"][triangle_id]["clustering"]
                    ).is_file()
                )
                self.assertTrue(
                    Path(
                        visualizations["triangles"][triangle_id][
                            "dino_feature_matches"
                        ]
                    ).is_file()
                )
                self.assertTrue(
                    Path(
                        visualizations["triangles"][triangle_id][
                            "sift_feature_matches"
                        ]
                    ).is_file()
                )
            saved = json.loads(Path(report["report_file"]).read_text(encoding="utf-8"))
            self.assertEqual(2, saved["ranking"][0]["triangle"])


if __name__ == "__main__":
    unittest.main()
