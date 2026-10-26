import math
import unittest

import numpy as np

from sample.next_best_view import BayesianViewSearch
from sample.active_perception_triangles import _nbv_viewpoint
from sample.viewpoint_reachability import quaternion_xyzw_to_matrix


class NextBestViewTests(unittest.TestCase):
    def test_candidates_look_at_object_and_stay_inside_radius_bounds(self):
        center = np.array([0.4, -0.2, 0.8])
        search = BayesianViewSearch(center, min_radius=0.38, max_radius=0.41)
        current = center + [0.4, 0, 0.3]
        for index in range(8):
            proposal = search.propose(current)
            current = np.asarray(proposal["position_xyz_m"])
            records, candidates = _nbv_viewpoint(proposal, center, index)
            candidate = candidates[0]
            radius = np.linalg.norm(current - center)
            self.assertGreaterEqual(radius, 0.38 - 1e-12)
            self.assertLessEqual(radius, 0.41 + 1e-12)
            self.assertGreater(current[2], center[2])
            rotation = quaternion_xyzw_to_matrix(candidate.orientation_xyzw)
            np.testing.assert_allclose(rotation[:, 2], (center - current) / radius, atol=1e-12)
            self.assertNotIn("vertices_xyz_m", records[0])
            self.assertEqual("next_best_view", records[0]["stage"])
            search.observe(current, 0.5 + index * 0.01)
        self.assertEqual("nbv_view_budget_reached", search.stop_reason())

    def test_posterior_is_continuous_across_azimuth_seam_and_has_uncertainty(self):
        search = BayesianViewSearch([0, 0, 0])
        for score in [0.4, 0.5, 0.8]:
            proposal = search.propose([0.4, 0, 0.3])
            search.observe(proposal["position_xyz_m"], score)
        observed = search.positions[-1]
        far = np.array([0, 0, 0.4])
        mean, std = search.predict([observed, far])
        self.assertAlmostEqual(0.8, mean[0], delta=0.005)
        self.assertLess(std[0], std[1])
        a = np.array([math.cos(1e-8), math.sin(1e-8), 0.5])
        b = np.array([math.cos(-1e-8), math.sin(-1e-8), 0.5])
        means, stds = search.predict([a / np.linalg.norm(a) * 0.4, b / np.linalg.norm(b) * 0.4])
        self.assertAlmostEqual(means[0], means[1], places=7)
        self.assertAlmostEqual(stds[0], stds[1], places=7)

    def test_rejections_are_not_scores_and_proposals_are_never_repeated(self):
        search = BayesianViewSearch([0, 0, 0], max_views=3)
        points = []
        while search.stop_reason() is None:
            proposal = search.propose([0.4, 0, 0.3])
            point = np.asarray(proposal["position_xyz_m"])
            for earlier in points:
                self.assertGreater(np.linalg.norm(point - earlier), 0.001)
            points.append(point)
            search.reject("collision")
        self.assertEqual([], search.scores)
        self.assertEqual(12, len(points))
        self.assertEqual("nbv_attempt_budget_reached", search.stop_reason())
        self.assertIsNone(search.propose([0.4, 0, 0.3]))

    def test_scores_change_the_next_selected_view(self):
        next_views = []
        for scores in ([0.85, 0.4, 0.4], [0.4, 0.4, 0.85]):
            search = BayesianViewSearch([0, 0, 0], motion_weight=0)
            for score in scores:
                proposal = search.propose([0, 0, 0.4])
                search.observe(proposal["position_xyz_m"], score)
            proposal = search.propose([0, 0, 0.4])
            self.assertEqual("expected_improvement", proposal["selection"])
            next_views.append(np.asarray(proposal["position_xyz_m"]))
        self.assertGreater(np.linalg.norm(next_views[0] - next_views[1]), 0.1)

    def test_adaptive_search_improves_over_seeds_on_smooth_objective(self):
        search = BayesianViewSearch([0, 0, 0])
        direction = np.array([0.2, -0.1, 0.3])
        direction /= np.linalg.norm(direction)
        current = np.array([0.4, 0, 0.4])
        while search.stop_reason() is None:
            proposal = search.propose(current)
            current = np.asarray(proposal["position_xyz_m"])
            ray = current / np.linalg.norm(current)
            score = 0.3 + 0.65 * math.exp(-np.linalg.norm(ray - direction)**2 / 0.25)
            search.observe(current, score)
        self.assertGreater(max(search.scores), max(search.scores[:3]) + 0.2)

    def test_repeated_actual_positions_are_numerically_stable(self):
        search = BayesianViewSearch([0, 0, 0])
        for score in [0.5, 0.51, 0.49]:
            search.propose([0.4, 0, 0.3])
            search.observe([0.4, 0, 0], score)
        mean, std = search.predict([[0.4, 0, 0]])
        self.assertTrue(np.isfinite(mean).all())
        self.assertTrue(np.isfinite(std).all())

    def test_invalid_scores_are_not_added(self):
        for score in (math.nan, math.inf, -0.1, 1.1):
            with self.subTest(score=score):
                search = BayesianViewSearch([0, 0, 0])
                proposal = search.propose([0.4, 0, 0.3])
                with self.assertRaises(ValueError):
                    search.observe(proposal["position_xyz_m"], score)
                self.assertEqual([], search.scores)

    def test_a_proposal_can_only_be_resolved_once(self):
        search = BayesianViewSearch([0, 0, 0])
        with self.assertRaises(RuntimeError):
            search.reject("no proposal")
        proposal = search.propose([0.4, 0, 0.3])
        with self.assertRaises(RuntimeError):
            search.propose([0.4, 0, 0.3])
        search.observe(proposal["position_xyz_m"], 0.6)
        with self.assertRaises(RuntimeError):
            search.observe(proposal["position_xyz_m"], 0.6)
        self.assertEqual([0.6], search.scores)


if __name__ == "__main__":
    unittest.main()
