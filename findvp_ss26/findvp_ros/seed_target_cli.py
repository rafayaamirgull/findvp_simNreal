"""Seed FindVP with a measured object centre for Steve's known carton box.

Use only after confirming the box's current placement. This bypasses RGB
triangulation; image scoring and triangle subdivision remain unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .base_tracking import robot_mounted_reference
from .configuration import load_robot_configuration
from .runtime import file_digest
from .target_selection import load_reference_mask


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--target-image", type=Path, required=True)
    parser.add_argument("--target-mask", type=Path, required=True)
    parser.add_argument("--target-center", nargs=3, type=float, required=True,
                        metavar=("X", "Y", "Z"))
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_robot_configuration(args.config)
    if config.base_tracking_mode != "robot_mounted":
        parser.error("known-position seed requires the robot-mounted target configuration")
    if not all(math.isfinite(v) for v in args.target_center):
        parser.error("target centre coordinates must be finite")
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_json}")
    load_reference_mask(args.target_mask, args.target_image)
    report = {
        "schema_version": 1,
        "method": "operator_measured_known_box",
        "estimate_kind": "known_object_center",
        "coordinate_frame": config.planning_frame,
        "center_world_m": args.target_center,
        "position_units": "metres",
        "object_prior_used": True,
        "target_image": str(args.target_image.resolve()),
        "target_sha256": file_digest(args.target_image),
        "target_mask": str(args.target_mask.resolve()),
        "target_mask_sha256": file_digest(args.target_mask),
        "base_reference": robot_mounted_reference(config.planning_frame),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Known-box centre recorded in {args.output_json}; verify its physical placement before search")


if __name__ == "__main__":
    main()
