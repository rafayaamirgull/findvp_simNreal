#!/usr/bin/env python3
"""Prepare camera-aware robot files for either CuRobo API generation."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

if __package__:
    from .curobo_reachability import (
        CUROBO_API_V1,
        CUROBO_API_V2,
        DEFAULT_CUROBO_TOOL_FRAME,
        add_curobo_v1_reference_collision_model,
        generate_curobo_camera_urdf,
        probe_curobo_runtime,
        require_curobo_runtime,
        write_curobo_v1_ik_robot_config,
    )
    from .viewpoint_reachability import DEFAULT_COLLISION_CLEARANCE_M
    from .workcell_config import (
        ARM_JOINT_ORDER,
        CAMERA_BODY_HALF_EXTENTS,
        CAMERA_OPTICAL_FROM_PARENT,
        DEFAULT_ROBOT_URDF,
        HOME_JOINT_POSITIONS_RAD,
        KEPT_JOINT_NAMES,
        KEPT_LINK_NAMES,
        REPO_ROOT,
        STEVE_DESCRIPTION_ROOT,
    )
else:  # Direct execution: python3 sample/prepare_curobo_robot.py
    from curobo_reachability import (
        CUROBO_API_V1,
        CUROBO_API_V2,
        DEFAULT_CUROBO_TOOL_FRAME,
        add_curobo_v1_reference_collision_model,
        generate_curobo_camera_urdf,
        probe_curobo_runtime,
        require_curobo_runtime,
        write_curobo_v1_ik_robot_config,
    )
    from viewpoint_reachability import DEFAULT_COLLISION_CLEARANCE_M
    from workcell_config import (
        ARM_JOINT_ORDER,
        CAMERA_BODY_HALF_EXTENTS,
        CAMERA_OPTICAL_FROM_PARENT,
        DEFAULT_ROBOT_URDF,
        HOME_JOINT_POSITIONS_RAD,
        KEPT_JOINT_NAMES,
        KEPT_LINK_NAMES,
        REPO_ROOT,
        STEVE_DESCRIPTION_ROOT,
    )


DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "curobo"
DEFAULT_OUTPUT_URDF = DEFAULT_OUTPUT_DIR / "ur5e_camera_curobo.urdf"
DEFAULT_V1_OUTPUT_CONFIG = DEFAULT_OUTPUT_DIR / "ur5e_camera_curobo_v1.yml"
DEFAULT_V2_OUTPUT_CONFIG = DEFAULT_OUTPUT_DIR / "ur5e_camera_curobo_v2.yml"
EXPECTED_COLLISION_LINKS = {
    "ur5e_shoulder_link",
    "ur5e_upper_arm_link",
    "ur5e_forearm_link",
    "ur5e_wrist_1_link",
    "ur5e_wrist_2_link",
    "ur5e_wrist_3_link",
    "ur5e_camera_body",
}


def finalize_curobo_robot_config(config_path: Path) -> None:
    """Enforce this workcell's V2 frame, c-space, and clearance invariants."""

    import yaml

    config_path = Path(config_path).resolve()
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("kinematics"), dict):
        raise RuntimeError("RobotBuilder output is missing the V2 'kinematics' mapping.")
    kinematics = data["kinematics"]
    if kinematics.get("base_link") != "ur5e_base_link":
        raise RuntimeError(f"Unexpected CuRobo base link: {kinematics.get('base_link')!r}")
    if kinematics.get("tool_frames") != [DEFAULT_CUROBO_TOOL_FRAME]:
        raise RuntimeError(f"Unexpected CuRobo tool frames: {kinematics.get('tool_frames')!r}")

    collision_links = list(kinematics.get("collision_link_names") or [])
    missing_collision_links = sorted(EXPECTED_COLLISION_LINKS - set(collision_links))
    if missing_collision_links:
        raise RuntimeError(
            "RobotBuilder did not fit spheres for required collision links: "
            f"{missing_collision_links}"
        )
    if "ur5e_base_link_inertia" in collision_links:
        raise RuntimeError(
            "The fixed table-mounted base unexpectedly received collision spheres."
        )

    self_collision_ignore = kinematics.get("self_collision_ignore")
    if not isinstance(self_collision_ignore, dict):
        raise RuntimeError("RobotBuilder output is missing self_collision_ignore.")
    camera_ignored_links = set(self_collision_ignore.get("ur5e_camera_body") or [])
    for link_name, ignored_links in self_collision_ignore.items():
        if "ur5e_camera_body" in (ignored_links or []):
            camera_ignored_links.add(link_name)
    unsafe_camera_ignores = sorted(
        (camera_ignored_links & EXPECTED_COLLISION_LINKS)
        - {"ur5e_camera_body", "ur5e_wrist_3_link"}
    )
    if unsafe_camera_ignores:
        raise RuntimeError(
            "Unsafe camera self-collision ignores require manual review: "
            f"{unsafe_camera_ignores}"
        )
    kinematics["collision_sphere_buffer"] = DEFAULT_COLLISION_CLEARANCE_M

    cspace = kinematics.get("cspace")
    if not isinstance(cspace, dict):
        raise RuntimeError("RobotBuilder output is missing its V2 cspace configuration.")
    joint_names = list(cspace.get("joint_names") or [])
    missing_joints = [name for name in ARM_JOINT_ORDER if name not in joint_names]
    unexpected_joints = [name for name in joint_names if name not in ARM_JOINT_ORDER]
    if missing_joints or unexpected_joints:
        raise RuntimeError(
            "RobotBuilder cspace joint mismatch: "
            f"missing={missing_joints}, unexpected={unexpected_joints}"
        )
    cspace["default_joint_position"] = [
        float(HOME_JOINT_POSITIONS_RAD[name]) for name in joint_names
    ]

    config_path.write_text(
        yaml.safe_dump(data, sort_keys=False), encoding="utf-8"
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a base-rooted UR5e + optical-camera URDF and a generation-specific "
            "CuRobo robot YAML. V1 maps the pinned package's verified UR5e spheres "
            "for R535; V2 can fit spheres in a newer R580+ environment."
        )
    )
    parser.add_argument(
        "--api-version",
        choices=(CUROBO_API_V1, CUROBO_API_V2),
        default=CUROBO_API_V1,
        help="Robot-config/API generation (default: v1 for CuRobo 0.7.8 on R535).",
    )
    parser.add_argument(
        "--output-urdf",
        type=Path,
        default=DEFAULT_OUTPUT_URDF,
    )
    parser.add_argument(
        "--output-config",
        type=Path,
        default=None,
        help="Output YAML (defaults to a distinct _v1.yml or _v2.yml path).",
    )
    parser.add_argument(
        "--run-builder",
        action="store_true",
        help="V2 only: run the official MorphIt RobotBuilder after writing the URDF.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="V2 only: open the official collision-sphere visualization after fitting.",
    )
    parser.add_argument(
        "--v1-ik-only",
        action="store_true",
        help=(
            "V1 only: keep the legacy sphere-free IK config instead of mapping the "
            "pinned CuRobo UR5e sphere model. This disables Level 7 motion planning."
        ),
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    output_config = (
        args.output_config.resolve()
        if args.output_config is not None
        else (
            DEFAULT_V1_OUTPUT_CONFIG
            if args.api_version == CUROBO_API_V1
            else DEFAULT_V2_OUTPUT_CONFIG
        ).resolve()
    )
    if args.api_version == CUROBO_API_V1 and (args.run_builder or args.visualize):
        parser.error(
            "--run-builder/--visualize require --api-version v2; CuRobo v0.7.8 "
            "does not provide MorphIt RobotBuilder."
        )
    if args.api_version == CUROBO_API_V2 and args.v1_ik_only:
        parser.error("--v1-ik-only can only be used with --api-version v1.")
    if args.api_version == CUROBO_API_V1 and not args.v1_ik_only:
        # The exact pinned package supplies the reviewed reference spheres and
        # URDF used by the guarded name/geometry mapping below. Sphere mapping
        # is CPU-only, so a live CUDA tensor is not required until planning.
        probe = probe_curobo_runtime(expected_api=CUROBO_API_V1)
        if not probe.compatible:
            raise RuntimeError(probe.format_error())
    if args.api_version == CUROBO_API_V2 and args.run_builder:
        # Fail before writing artifacts when the selected package/API/CUDA stack
        # cannot execute the fitter.
        require_curobo_runtime(expected_api=CUROBO_API_V2)
    if not DEFAULT_ROBOT_URDF.exists():
        raise FileNotFoundError(f"Source robot URDF not found: {DEFAULT_ROBOT_URDF}")

    # The current fixed wrist_3 -> flange -> tool0 chain composes to identity,
    # so the simulator's parent-from-optical matrix is also tool0-from-optical.
    output_urdf = generate_curobo_camera_urdf(
        source_urdf=DEFAULT_ROBOT_URDF,
        output_urdf=args.output_urdf,
        kept_link_names=KEPT_LINK_NAMES,
        kept_joint_names=KEPT_JOINT_NAMES,
        package_description_root=STEVE_DESCRIPTION_ROOT,
        tool_from_optical_camera=CAMERA_OPTICAL_FROM_PARENT,
        camera_body_half_extents=CAMERA_BODY_HALF_EXTENTS,
    )
    print(f"Generated CuRobo camera URDF: {output_urdf}")

    if args.api_version == CUROBO_API_V1:
        output_config = write_curobo_v1_ik_robot_config(
            config_path=output_config,
            camera_urdf_path=output_urdf,
            joint_names=ARM_JOINT_ORDER,
            retract_config=[HOME_JOINT_POSITIONS_RAD[name] for name in ARM_JOINT_ORDER],
        )
        if args.v1_ik_only:
            print(f"Generated CuRobo v0.7.8 IK-only config: {output_config}")
            print(
                "CuRobo-side collision checking is disabled; Level 7 motion planning "
                "will reject this config."
            )
            return

        import yaml
        from curobo.util_file import get_assets_path, get_robot_configs_path, join_path

        reference_config = Path(join_path(get_robot_configs_path(), "ur5e.yml")).resolve()
        reference_data = yaml.safe_load(reference_config.read_text(encoding="utf-8"))
        reference_kinematics = reference_data["robot_cfg"]["kinematics"]
        reference_urdf = (
            Path(get_assets_path()) / str(reference_kinematics["urdf_path"])
        ).resolve()
        add_curobo_v1_reference_collision_model(
            config_path=output_config,
            reference_config_path=reference_config,
            reference_urdf_path=reference_urdf,
            camera_body_half_extents=CAMERA_BODY_HALF_EXTENTS,
        )
        print(f"Generated collision-aware CuRobo v0.7.8 config: {output_config}")
        print(
            "Verified local UR5e mesh/origin/kinematic equivalence, mapped the pinned "
            "reference spheres, and added a conservative camera bounding sphere. "
            "PyBullet remains authoritative for endpoints and returned trajectories."
        )
        return

    command = [
        sys.executable,
        "-m",
        "curobo.examples.getting_started.build_robot_model",
        "--urdf",
        output_urdf.as_posix(),
        "--asset-path",
        STEVE_DESCRIPTION_ROOT.resolve().as_posix(),
        "--output",
        output_config.as_posix(),
        "--tool-frames",
        DEFAULT_CUROBO_TOOL_FRAME,
        "--sphere-density",
        "1.5",
        "--num-collision-samples",
        "4000",
        "--compute-metrics",
        "--seed",
        "123",
    ]
    if args.visualize:
        command.append("--visualize")

    if not args.run_builder:
        wrapper_command = [
            sys.executable,
            Path(__file__).resolve().as_posix(),
            "--api-version",
            CUROBO_API_V2,
            "--output-urdf",
            output_urdf.as_posix(),
            "--output-config",
            output_config.as_posix(),
            "--run-builder",
        ]
        if args.visualize:
            wrapper_command.append("--visualize")
        print("Run the reviewed V2 sphere fitter in the compatible R580+ environment:")
        print("  " + shlex.join(wrapper_command))
        print(
            "Visually audit the arm/camera spheres and self-collision exclusions before "
            "using the YAML on hardware; this reduced model intentionally omits the gripper."
        )
        return

    output_config.parent.mkdir(parents=True, exist_ok=True)
    if args.visualize:
        print(
            "Close the RobotBuilder viewer with Ctrl+C after inspection so the saved "
            "YAML can be finalized and validated."
        )
    subprocess.run(command, check=True)
    finalize_curobo_robot_config(output_config)
    print(f"Generated CuRobo V2 robot config: {output_config}")
    print(
        f"Pinned collision_sphere_buffer to {DEFAULT_COLLISION_CLEARANCE_M:.3f} m "
        "and the default joint state to the simulator home pose."
    )


if __name__ == "__main__":
    main()
