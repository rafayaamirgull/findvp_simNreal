import unittest

import torch
from PIL import Image

from sample.dinov3_feature_matching import DenseFeatures
from sample.dinov3_spatial_clustering import (
    DINOv3SpatialClusterer,
    SpatialClusterResult,
    compute_object_similarity,
    format_cluster_report,
    select_automatic_object_mask,
)


def _grid_features(descriptors, grid_size):
    descriptors = torch.as_tensor(descriptors, dtype=torch.float32)
    grid_h, grid_w = grid_size
    count = grid_h * grid_w
    return DenseFeatures(
        descriptors=descriptors,
        keypoints=torch.zeros(count, 2),
        global_descriptor=torch.ones(descriptors.shape[1]),
        grid_size=grid_size,
        original_size=grid_size,
        processed_size=grid_size,
    )


def _synthetic_object_case():
    grid_size = (6, 6)
    descriptors = torch.tensor([[1.0, 0.0, 0.0]]).repeat(36, 1)
    expected_mask = torch.zeros(grid_size, dtype=torch.bool)
    expected_mask[2:4, 2:4] = True
    descriptors[expected_mask.reshape(-1)] = torch.tensor([0.0, 1.0, 0.0])

    image = Image.new("RGB", (6, 6), "white")
    for y in range(2, 4):
        for x in range(2, 4):
            image.putpixel((x, y), (0, 0, 0))
    return _grid_features(descriptors, grid_size), image, expected_mask


def _cluster_result(mask):
    mask = torch.as_tensor(mask, dtype=torch.bool)
    return SpatialClusterResult(
        labels=torch.zeros_like(mask, dtype=torch.long),
        object_mask=mask,
        object_indices=torch.nonzero(mask.reshape(-1), as_tuple=False).reshape(-1),
        selected_cluster_ids=(0,),
        metrics=(),
        pca_components=1,
        explained_variance_ratio=1.0,
        inertia=0.0,
        random_state=0,
    )


class DINOv3SpatialClusteringTests(unittest.TestCase):
    def test_automatic_masks_select_complete_isolated_object(self):
        height, width = 100, 140
        scene = torch.ones((height, width), dtype=torch.bool)
        whole_object = torch.zeros_like(scene)
        whole_object[30:75, 45:100] = True
        object_part = torch.zeros_like(scene)
        object_part[40:62, 58:83] = True

        proposal = select_automatic_object_mask(
            torch.stack((scene, object_part, whole_object)),
            torch.tensor((0.99, 0.96, 0.92)),
            (height, width),
            bbox_padding_fraction=0.0,
        )

        torch.testing.assert_close(proposal.mask, whole_object)
        self.assertEqual((45, 30, 100, 75), proposal.bbox_xyxy)
        self.assertGreater(proposal.selection_score, 0.8)

    def test_automatic_masks_reject_boundary_only_proposals(self):
        mask = torch.zeros((1, 20, 30), dtype=torch.bool)
        mask[0, :, :10] = True
        with self.assertRaisesRegex(RuntimeError, "no isolated object"):
            select_automatic_object_mask(mask, (0.99,), (20, 30))

    def test_white_background_object_is_discovered(self):
        features, image, expected_mask = _synthetic_object_case()
        clusterer = DINOv3SpatialClusterer(
            n_clusters=2,
            pca_components=3,
            random_state=7,
            min_cluster_foreground_fraction=0.5,
        )

        result = clusterer.fit_predict(features, image=image)

        torch.testing.assert_close(result.object_mask, expected_mask)
        self.assertEqual(4, len(result.object_indices))
        self.assertEqual(3, result.pca_components)
        self.assertEqual(7, result.random_state)
        selected_metric = next(
            metric
            for metric in result.metrics
            if metric.cluster_id in result.selected_cluster_ids
        )
        self.assertEqual(1.0, selected_metric.foreground_fraction)
        self.assertIn("final object mask: 4 patches", format_cluster_report(result))

    def test_clustering_is_reproducible(self):
        features, image, _ = _synthetic_object_case()
        clusterer = DINOv3SpatialClusterer(n_clusters=2, random_state=13)

        first = clusterer.fit_predict(features, image=image)
        second = clusterer.fit_predict(features, image=image)

        torch.testing.assert_close(first.labels, second.labels)
        torch.testing.assert_close(first.object_mask, second.object_mask)
        self.assertEqual(first.selected_cluster_ids, second.selected_cluster_ids)

    def test_cluster_selection_can_be_overridden(self):
        features, image, expected_mask = _synthetic_object_case()
        clusterer = DINOv3SpatialClusterer(
            n_clusters=2,
            keep_largest_component=False,
        )
        automatic = clusterer.fit_predict(features, image=image)
        background_id = next(
            cluster_id
            for cluster_id in range(2)
            if cluster_id not in automatic.selected_cluster_ids
        )

        overridden = clusterer.fit_predict(
            features, image=image, object_cluster_ids=(background_id,)
        )

        torch.testing.assert_close(overridden.object_mask, ~expected_mask)
        self.assertEqual((background_id,), overridden.selected_cluster_ids)
        with self.assertRaises(ValueError):
            clusterer.fit_predict(features, object_cluster_ids=(2,))

    def test_filtered_feature_grid_is_rejected(self):
        features, _, _ = _synthetic_object_case()
        filtered = DenseFeatures(
            descriptors=features.descriptors[:-1],
            keypoints=features.keypoints[:-1],
            global_descriptor=features.global_descriptor,
            grid_size=features.grid_size,
            original_size=features.original_size,
            processed_size=features.processed_size,
        )

        with self.assertRaisesRegex(ValueError, "apply_background_filter=False"):
            DINOv3SpatialClusterer(n_clusters=2).fit_predict(filtered)

    def test_image_size_must_match_feature_input(self):
        features, _, _ = _synthetic_object_case()
        with self.assertRaisesRegex(ValueError, "does not match"):
            DINOv3SpatialClusterer(n_clusters=2).fit_predict(
                features, image=Image.new("RGB", (5, 6))
            )

    def test_similarity_implements_directional_maximum_cosine_score(self):
        current = _grid_features([[1.0, 0.0], [0.0, 1.0]], (1, 2))
        target = _grid_features([[1.0, 0.0], [-1.0, 0.0]], (1, 2))
        current_mask = _cluster_result([[True, True]])
        target_mask = _cluster_result([[True, False]])

        result = compute_object_similarity(
            current, current_mask, target, target_mask, chunk_size=1
        )

        self.assertAlmostEqual(0.5, result.score)
        self.assertAlmostEqual(1.0, result.reverse_score)
        self.assertAlmostEqual(0.75, result.symmetric_score)
        torch.testing.assert_close(result.per_source_scores, torch.tensor([1.0, 0.0]))
        self.assertEqual([0, 0], result.best_target_indices.tolist())
        self.assertEqual(2, result.source_patch_count)
        self.assertEqual(1, result.target_patch_count)

    def test_similarity_uses_all_target_patches_when_mask_is_omitted(self):
        current = _grid_features([[0.0, 1.0]], (1, 1))
        target = _grid_features([[1.0, 0.0], [0.0, 1.0]], (1, 2))

        result = compute_object_similarity(
            current, _cluster_result([[True]]), target
        )

        self.assertAlmostEqual(1.0, result.score)
        self.assertEqual([1], result.best_target_indices.tolist())
        self.assertEqual(2, result.target_patch_count)

    def test_similarity_validates_arguments(self):
        current = _grid_features([[1.0, 0.0]], (1, 1))
        target = _grid_features([[1.0, 0.0]], (1, 1))
        mask = _cluster_result([[True]])
        with self.assertRaises(ValueError):
            compute_object_similarity(current, mask, target, chunk_size=0)

        wrong_channels = _grid_features([[1.0, 0.0, 0.0]], (1, 1))
        with self.assertRaisesRegex(ValueError, "channel dimensions"):
            compute_object_similarity(current, mask, wrong_channels)


if __name__ == "__main__":
    unittest.main()
