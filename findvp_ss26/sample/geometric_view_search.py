"""RGB correspondence-guided camera search on a known object-centered sphere.

Essential-matrix geometry estimates target orientation, not metric translation.
The configured viewing radius and known object center supply the missing scale.
Target camera intrinsics are assumed to match the simulator's field of view.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

if __package__:
    from .next_best_view import BayesianViewSearch
    from .viewpoint_reachability import matrix_to_quaternion_xyzw
    from .workcell_config import compute_camera_vertical_fov_deg
else:
    from next_best_view import BayesianViewSearch
    from viewpoint_reachability import matrix_to_quaternion_xyzw
    from workcell_config import compute_camera_vertical_fov_deg


def camera_intrinsics(size: tuple[int, int]) -> np.ndarray:
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive.")
    focal = height / (2 * math.tan(math.radians(compute_camera_vertical_fov_deg()) / 2))
    return np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1.0]])


def extract_geometry_features(path: Path):
    """OpenCV SIFT on a masked object crop; coordinates remain in the full image."""
    import cv2

    if __package__:
        from .target_guided_similarity import _central_object_region
    else:
        from target_guided_similarity import _central_object_region

    with Image.open(path) as opened:
        image = opened.convert("RGB")
    region = _central_object_region(image)
    left, top, right, bottom = region.bbox_xyxy
    scale = 720 / max(right - left, bottom - top)
    crop = cv2.resize(
        np.asarray(image.convert("L"))[top:bottom, left:right],
        None, fx=scale, fy=scale,
    )
    mask = cv2.resize(
        region.mask[top:bottom, left:right].astype(np.uint8) * 255,
        (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST,
    )
    points, descriptors = cv2.SIFT_create(
        contrastThreshold=0.005, edgeThreshold=15,
    ).detectAndCompute(crop, mask)
    xy = np.asarray([np.asarray(point.pt) / scale + [left, top] for point in points])
    return xy.reshape(-1, 2), descriptors, camera_intrinsics(image.size)


def estimate_target_rotation(
    source_xy, target_xy, source_intrinsics, target_intrinsics, source_rotation,
) -> tuple[np.ndarray | None, dict]:
    """Recover a supported world-from-target-camera rotation from RGB matches."""
    import cv2

    diagnostics = {"matches": len(source_xy), "inliers": 0, "accepted": False}
    if len(source_xy) < 15:
        return None, {**diagnostics, "reason": "insufficient_matches"}
    a = cv2.undistortPoints(
        np.asarray(source_xy, dtype=np.float64).reshape(-1, 1, 2), source_intrinsics, None,
    ).reshape(-1, 2)
    b = cv2.undistortPoints(
        np.asarray(target_xy, dtype=np.float64).reshape(-1, 1, 2), target_intrinsics, None,
    ).reshape(-1, 2)
    cv2.setRNGSeed(0)
    essential, mask = cv2.findEssentialMat(
        a, b, np.eye(3), method=cv2.RANSAC, prob=0.999,
        threshold=0.7 / max(source_intrinsics[0, 0], target_intrinsics[0, 0]),
    )
    if essential is None or mask is None:
        return None, {**diagnostics, "reason": "essential_estimation_failed"}
    best = None
    for matrix in np.asarray(essential).reshape(-1, 3, 3):
        count, relative_rotation, _, _ = cv2.recoverPose(
            matrix, a, b, np.eye(3), mask=mask.copy(),
        )
        # OpenCV returns target_from_source; convert to world_from_target.
        rotation = np.asarray(source_rotation) @ relative_rotation.T
        elevation = math.degrees(math.asin(float(np.clip(-rotation[2, 2], -1, 1))))
        evidence = {
            **diagnostics, "inliers": int(count), "inlier_ratio": count / len(a),
            "predicted_elevation_deg": elevation,
        }
        if count < 15 or count / len(a) < 0.25:
            evidence["reason"] = "insufficient_geometric_support"
        elif elevation < 10:
            evidence["reason"] = "outside_upper_hemisphere"
        else:
            evidence.update(accepted=True, reason="supported_relative_pose")
        if best is None or (evidence["accepted"], count) > (best[1]["accepted"], best[1]["inliers"]):
            best = (rotation, evidence)
    return (best[0] if best[1]["accepted"] else None), best[1]


class GeometricViewSearch:
    """Try supported relative-pose estimates, otherwise acquire the next seed.

    The Bayesian search object supplies only the shared seed geometry, budgets,
    and observation ledger. Its model/acquisition/propose methods are never used.
    """

    def __init__(self, center, radius=0.4, *, target_image: Path, **settings):
        self.state = BayesianViewSearch(center, radius, **settings)
        self.attempts = self.state.attempts
        self.scores = self.state.scores
        self.target_image = target_image
        self.target_features = None
        self.estimates: list[dict] = []
        self.seed_index = 0
        self.exhaustion_reason = "geometric_no_supported_pose"

    def stop_reason(self):
        reason = self.state.stop_reason()
        return reason.replace("nbv_", "geometric_") if reason else None

    def propose(self, current_position):
        if self.attempts and self.attempts[-1]["status"] == "proposed":
            raise RuntimeError("Resolve the current geometric proposal first.")
        if self.stop_reason():
            return None
        proposal = None
        # Prefer the strongest unused estimate; never silently replace its roll.
        for estimate in sorted(self.estimates, key=lambda item: item["inliers"], reverse=True):
            if estimate["used"]:
                continue
            estimate["used"] = True
            position = np.asarray(estimate["position_xyz_m"])
            orientation = np.asarray(estimate["camera_orientation_xyzw"])
            duplicate = False
            for previous in self.attempts:
                previous_position = previous.get("actual_position_xyz_m", previous["position_xyz_m"])
                previous_orientation = previous.get(
                    "actual_orientation_xyzw", previous.get("camera_orientation_xyzw")
                )
                if np.linalg.norm(position - previous_position) >= 0.005:
                    continue
                if previous_orientation is None:
                    # A rejected seed should not cause repeated IK proposals.
                    duplicate = previous["status"] == "rejected"
                else:
                    cosine = abs(float(orientation @ np.asarray(previous_orientation)))
                    duplicate = math.degrees(2 * math.acos(np.clip(cosine, 0, 1))) < 2
                if duplicate:
                    break
            if duplicate:
                estimate["skip_reason"] = "near_previous_proposal"
                if previous["status"] == "scored":
                    self.exhaustion_reason = "geometric_pose_converged"
                continue
            proposal = {
                "position_xyz_m": position.tolist(),
                "camera_orientation_xyzw": estimate["camera_orientation_xyzw"],
                "selection": "essential_matrix_pose",
                "source_proposal_id": estimate["source_proposal_id"],
                "inliers": estimate["inliers"],
            }
            break
        if proposal is None:
            if self.seed_index >= len(self.state.seeds):
                return None
            proposal = {
                "position_xyz_m": self.state.seeds[self.seed_index].tolist(),
                "selection": "initial_coverage",
            }
            self.seed_index += 1
        proposal.update(proposal_id=len(self.attempts) + 1, status="proposed")
        self.attempts.append(proposal)
        return proposal

    def observe(self, position, score):
        self.state.observe(position, score)

    def reject(self, reason):
        self.state.reject(reason)

    def observe_image(self, path: Path, source_rotation: np.ndarray):
        import cv2

        self.attempts[-1]["actual_orientation_xyzw"] = list(matrix_to_quaternion_xyzw(source_rotation))
        if self.target_features is None:
            self.target_features = extract_geometry_features(self.target_image)
        target_xy, target_descriptors, target_intrinsics = self.target_features
        source_xy, source_descriptors, source_intrinsics = extract_geometry_features(path)
        if source_descriptors is None or target_descriptors is None:
            self.attempts[-1]["geometry"] = {"accepted": False, "reason": "no_descriptors"}
            return
        matches = cv2.BFMatcher().knnMatch(source_descriptors, target_descriptors, k=2)
        good = [pair[0] for pair in matches if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
        rotation, evidence = estimate_target_rotation(
            [source_xy[m.queryIdx] for m in good],
            [target_xy[m.trainIdx] for m in good],
            source_intrinsics, target_intrinsics, source_rotation,
        )
        self.attempts[-1]["geometry"] = evidence
        if rotation is not None:
            self.estimates.append({
                "source_proposal_id": self.attempts[-1]["proposal_id"],
                "position_xyz_m": (self.state.center - self.state.radius * rotation[:, 2]).tolist(),
                "camera_orientation_xyzw": list(matrix_to_quaternion_xyzw(rotation)),
                "inliers": evidence["inliers"], "used": False,
            })

    def summary(self):
        return {
            "selection": "essential_matrix_relative_pose",
            "radius_m": self.state.radius,
            "assumptions": "Known object center; target looks at center at configured radius and camera FOV.",
            "max_views": self.state.max_views,
            "max_attempts": self.state.max_attempts,
            "patience": self.state.patience,
            "scored_views": len(self.scores),
            "attempts": self.attempts,
            "pose_estimates": self.estimates,
        }
