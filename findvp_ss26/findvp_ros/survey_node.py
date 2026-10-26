"""Supervised RGB survey using the same guarded motion session as triangle search."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import yaml

from .geometry import transform_matrix, validate_transform, matrix_to_quaternion_xyzw
from .runtime import DeploymentFault


def load_survey_anchor(path, config):
    document = yaml.safe_load(Path(path).read_text())
    pose = document.get("recorded_camera_pose", {})
    if (pose.get("planning_frame") != config.planning_frame
            or pose.get("camera_frame") != config.camera_frame):
        raise ValueError("survey start pose must contain recorded camera pose in the configured frames")
    return transform_matrix(pose.get("translation_m", ()), pose.get("quaternion_xyzw", ()))


def survey_viewpoints(anchor, baseline):
    """Three lateral optical-camera poses; no depth or guessed object center."""
    from sample.viewpoint_reachability import ViewpointCandidate
    anchor = validate_transform(anchor)
    if not math.isfinite(baseline) or not .02 <= baseline <= .10:
        raise ValueError("survey baseline must be between 0.02 and 0.10 metres")
    orientation = matrix_to_quaternion_xyzw(anchor[:3, :3])
    return [ViewpointCandidate(
        index=i, position_xyz=tuple(anchor[:3, 3] + offset * anchor[:3, 0]),
        orientation_xyzw=orientation, radius_m=0., azimuth_deg=0., elevation_deg=0.,
    ) for i, offset in enumerate((0., baseline, -baseline))]


def run_survey(node, parsed, anchor, report, persist):
    """Capture the reached start pose, then two independently approved translations.

    Failures are terminal: no skipping a failed move/capture to try another pose.
    The preview plans all offsets from the current joints; execution replans each
    segment after measuring its actual start and displays it for new approval.
    """
    report["survey"] = []
    for index, viewpoint in enumerate(survey_viewpoints(anchor, parsed.survey_baseline)):
        record = {"index": index, "requested_camera_pose": {
            "position_xyz_m": list(viewpoint.position_xyz),
            "orientation_xyzw": list(viewpoint.orientation_xyzw)}}
        report["survey"].append(record)
        if index:
            trajectory, record["plan"], scene = node.plan(viewpoint)
            persist()
            if parsed.execute:
                record["execution"] = node.execute(trajectory, scene)
        if parsed.execute:
            record["observation"] = node.capture(viewpoint)
        persist()
    if parsed.execute:
        poses = [np.asarray(v["observation"]["world_from_camera"])[:3, 3]
                 for v in report["survey"]]
        if min(np.linalg.norm(poses[i] - poses[j]) for i in range(3) for j in range(i)) < .01:
            raise DeploymentFault("survey did not achieve distinct translated camera positions")
    report["status"] = "survey_complete" if parsed.execute else "plan_only_complete"
    persist()


def main(argv=None):
    from .search_node import main as run
    run(argv, survey_only=True)


if __name__ == "__main__":
    main()
