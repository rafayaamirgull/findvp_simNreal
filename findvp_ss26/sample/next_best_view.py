"""Small, deterministic Bayesian search over camera positions around an object.

The objective is target-image similarity, not reconstruction coverage. No images,
robot state, or target ground-truth poses are used to predict unobserved scores.
"""

from __future__ import annotations

import math

import numpy as np


class BayesianViewSearch:
    """Select one pose using expected improvement divided by a travel-cost proxy.

    A Matérn-5/2 GP with fixed length scales avoids costly hyperparameter fitting. Its
    features are unit viewing direction and radius, so azimuth has no seam.
    Rejected proposals are excluded, never interpreted as low similarity.
    """

    def __init__(
        self,
        center,
        radius=0.4,
        *,
        min_radius=0.2,
        max_radius=0.6,
        max_views=8,
        patience=3,
        min_improvement=0.002,
        azimuth_offset=0.0,
        motion_weight=0.25,
    ):
        self.center = np.asarray(center, dtype=float)
        if self.center.shape != (3,) or not np.isfinite(self.center).all():
            raise ValueError("NBV center must contain three finite coordinates.")
        if not all(math.isfinite(x) for x in (radius, min_radius, max_radius)):
            raise ValueError("NBV radii must be finite.")
        if not 0 < min_radius <= radius <= max_radius:
            raise ValueError("NBV radius must lie inside positive radial bounds.")
        if max_views < 3 or patience < 1:
            raise ValueError("NBV requires at least 3 views and positive patience.")
        if not math.isfinite(min_improvement) or not 0 < min_improvement <= 1:
            raise ValueError("NBV minimum improvement must be in (0, 1].")
        if not math.isfinite(motion_weight) or motion_weight < 0:
            raise ValueError("NBV motion weight must be finite and non-negative.")
        if not math.isfinite(azimuth_offset):
            raise ValueError("NBV azimuth offset must be finite.")
        self.radius = radius
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.max_views = max_views
        self.patience = patience
        self.min_improvement = min_improvement
        self.motion_weight = motion_weight
        self.positions: list[np.ndarray] = []
        self.scores: list[float] = []
        self.attempts: list[dict] = []
        self.stale_views = 0
        self.max_attempts = 4 * max_views
        offset = math.radians(azimuth_offset)
        angles = offset + np.radians([60, 180, 300])
        # Match the baseline's three-view coverage for a fair algorithm comparison.
        elevation = math.radians(45)
        self.seeds = self.center + radius * np.column_stack(
            (np.cos(angles) * math.cos(elevation),
             np.sin(angles) * math.cos(elevation),
             np.full(3, math.sin(elevation)))
        )
        # Equal-area Fibonacci samples between elevations 10 and 80 degrees.
        z = np.linspace(math.sin(math.radians(10)), math.sin(math.radians(80)), 128)
        phi = offset + np.arange(len(z)) * math.pi * (3 - math.sqrt(5))
        self.directions = np.column_stack(
            (np.sqrt(1 - z * z) * np.cos(phi),
             np.sqrt(1 - z * z) * np.sin(phi), z)
        )

    def _features(self, positions):
        rays = np.asarray(positions, dtype=float) - self.center
        radii = np.linalg.norm(rays, axis=-1, keepdims=True)
        return np.concatenate((rays / radii / 0.65, radii / 0.08), axis=-1)

    @staticmethod
    def _kernel(a, b):
        distance = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
        scaled = math.sqrt(5) * distance
        return (1 + scaled + scaled**2 / 3) * np.exp(-scaled)

    def predict(self, positions):
        """Return posterior mean and standard deviation of latent similarity."""
        query = self._features(positions)
        if not self.scores:
            return np.full(len(query), 0.5), np.full(len(query), 0.25)
        observed = self._features(self.positions)
        # Scale uncertainty to the observed objective; a large fixed score
        # variance otherwise spends a small view budget on distant exploration.
        signal_variance = max(0.05, float(np.std(self.scores))) ** 2
        covariance = signal_variance * self._kernel(observed, observed)
        covariance += np.eye(len(observed)) * (0.01**2 + 1e-9)
        factor = np.linalg.cholesky(covariance)
        cross = signal_variance * self._kernel(observed, query)
        baseline = float(np.mean(self.scores))
        alpha = np.linalg.solve(factor.T, np.linalg.solve(factor, np.asarray(self.scores) - baseline))
        mean = baseline + cross.T @ alpha
        projected = np.linalg.solve(factor, cross)
        variance = signal_variance - np.sum(projected**2, axis=0)
        return mean, np.sqrt(np.maximum(variance, 1e-12))

    def _pool(self):
        radii = np.unique(np.clip(
            [self.radius - 0.04, self.radius, self.radius + 0.04],
            self.min_radius, self.max_radius,
        ))
        pools = [self.center + radius * self.directions for radius in radii]
        if self.scores:
            best = self.positions[int(np.argmax(self.scores))] - self.center
            radius = float(np.linalg.norm(best))
            ray = best / radius
            tangent = np.cross(ray, [0, 0, 1])
            if np.linalg.norm(tangent) < 1e-6:
                tangent = np.cross(ray, [1, 0, 0])
            tangent /= np.linalg.norm(tangent)
            other = np.cross(ray, tangent)
            local = [ray]
            for angle in np.radians([2, 5, 10, 20]):
                for phi in np.linspace(0, 2 * math.pi, 8, endpoint=False):
                    local.append(math.cos(angle) * ray + math.sin(angle) * (
                        math.cos(phi) * tangent + math.sin(phi) * other
                    ))
            local = np.asarray(local)
            local = local[local[:, 2] >= math.sin(math.radians(10))]
            for r in np.unique(np.clip(
                [radius - 0.02, radius, radius + 0.02], self.min_radius, self.max_radius
            )):
                pools.append(self.center + r * local)
        pool = np.vstack(pools)
        if self.attempts:
            tried = np.asarray([item["position_xyz_m"] for item in self.attempts])
            separation = np.linalg.norm(pool[:, None, :] - tried[None, :, :], axis=-1)
            pool = pool[np.min(separation, axis=1) > 0.001]
        return pool

    def stop_reason(self):
        if len(self.scores) >= self.max_views:
            return "nbv_view_budget_reached"
        if len(self.scores) >= min(6, self.max_views) and self.stale_views >= self.patience:
            return "nbv_similarity_plateau"
        if len(self.attempts) >= self.max_attempts:
            return "nbv_attempt_budget_reached"
        return None

    def propose(self, current_position):
        """Reserve a new proposal. Call observe() or reject() before proposing again."""
        if self.attempts and self.attempts[-1]["status"] == "proposed":
            raise RuntimeError("Resolve the current NBV proposal before requesting another.")
        if self.stop_reason():
            return None
        current = np.asarray(current_position, dtype=float)
        if current.shape != (3,) or not np.isfinite(current).all():
            raise ValueError("Current camera position must contain three finite coordinates.")
        if len(self.attempts) < 3:
            position = self.seeds[len(self.attempts)]
            diagnostics = {"selection": "initial_coverage"}
        else:
            pool = self._pool()
            if not len(pool):
                return None
            travel = np.linalg.norm(pool - current, axis=1) / self.radius
            if len(self.scores) < 3:
                # Replace failed seeds with well-separated proposals.
                tried = np.asarray([a["position_xyz_m"] for a in self.attempts])
                separation = np.linalg.norm(
                    self._features(pool)[:, None, :] - self._features(tried)[None, :, :], axis=-1
                ).min(axis=1)
                index = int(np.argmax(separation / (1 + self.motion_weight * travel)))
                diagnostics = {"selection": "coverage_recovery"}
            else:
                mean, std = self.predict(pool)
                gain = mean - max(self.scores) - self.min_improvement
                z = gain / std
                cdf = np.asarray([0.5 * (1 + math.erf(float(v) / math.sqrt(2))) for v in z])
                ei = np.maximum(0, gain * cdf + std * np.exp(-0.5 * z**2) / math.sqrt(2 * math.pi))
                utility = ei / (1 + self.motion_weight * travel)
                index = int(np.argmax(utility))
                diagnostics = {
                    "selection": "expected_improvement",
                    "predicted_score": float(mean[index]),
                    "predicted_std": float(std[index]),
                    "expected_improvement": float(ei[index]),
                    "acquisition_value": float(utility[index]),
                }
            position = pool[index]
        proposal = {
            "proposal_id": len(self.attempts) + 1,
            "position_xyz_m": position.tolist(),
            "status": "proposed",
            **diagnostics,
        }
        self.attempts.append(proposal)
        return proposal

    def reject(self, reason):
        if not self.attempts or self.attempts[-1]["status"] != "proposed":
            raise RuntimeError("There is no pending NBV proposal to reject.")
        self.attempts[-1].update(status="rejected", rejection_reason=reason)

    def observe(self, position, score):
        if not self.attempts or self.attempts[-1]["status"] != "proposed":
            raise RuntimeError("There is no pending NBV proposal to score.")
        position = np.asarray(position, dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("Observed camera position must contain three finite coordinates.")
        if np.linalg.norm(position - self.center) <= 1e-9:
            raise ValueError("Observed camera position cannot coincide with the object center.")
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Observed similarity must be in [0, 1].")
        improved = not self.scores or score >= max(self.scores) + self.min_improvement
        self.stale_views = 0 if improved else self.stale_views + 1
        self.positions.append(position.copy())
        self.scores.append(float(score))
        self.attempts[-1].update(status="scored", actual_position_xyz_m=position.tolist(), score=float(score))

    def summary(self):
        return {
            "kernel": "Matern-5/2",
            "direction_length_scale": 0.65,
            "radius_length_scale_m": 0.08,
            "score_noise_std": 0.01,
            "signal_std": max(0.05, float(np.std(self.scores))) if self.scores else 0.25,
            "acquisition": "expected_improvement / (1 + motion_weight * camera_distance / radius)",
            "motion_weight": self.motion_weight,
            "max_views": self.max_views,
            "max_attempts": self.max_attempts,
            "patience": self.patience,
            "min_improvement": self.min_improvement,
            "scored_views": len(self.scores),
            "stale_views": self.stale_views,
            "attempts": self.attempts,
        }
