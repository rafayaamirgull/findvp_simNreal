"""Record the current measured UR5e joints as a reviewed FindVP start pose."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import yaml

from .configuration import load_robot_configuration
from .manifest import ordered_joint_positions, ordered_joint_velocities
from .runtime import SettlingWindow, stamp_ns


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", default="table_view_start")
    parser.add_argument("--state-max-age", type=float, default=2.0)
    return parser


def main(argv=None) -> None:
    args, ros_arguments = _parser().parse_known_args(argv)
    if not args.name.strip():
        raise ValueError("--name must not be empty")
    if args.state_max_age <= 0.0:
        raise ValueError("--state-max-age must be positive")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite start pose: {output}")
    config = load_robot_configuration(args.config)

    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import JointState
        from tf2_ros import Buffer, TransformListener
        from rclpy.time import Time
    except ImportError as error:  # pragma: no cover - requires ROS installation
        raise SystemExit(
            f"findvp_record_pose requires the sourced Steve ROS 2 workspace: {error}"
        ) from error

    class PoseRecorder(Node):
        def __init__(self):
            super().__init__("findvp_record_pose")
            self.joints = None
            self.joints_received_at = None
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)
            self.create_subscription(
                JointState, config.joint_states_topic, self._receive, qos_profile_sensor_data
            )

        def _receive(self, message):
            try:
                ordered_joint_positions(message, config.joint_names)
            except ValueError:
                return
            self.joints = message
            self.joints_received_at = time.monotonic()

    rclpy.init(args=ros_arguments)
    node = PoseRecorder()
    try:
        deadline = time.monotonic() + 10.0
        settled = SettlingWindow(0.5)
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            if node.joints is not None and settled.update(
                stamp_ns(node.joints),
                ordered_joint_velocities(node.joints, config.joint_names),
            ):
                break
        else:
            raise RuntimeError("timed out waiting for a stationary UR5e joint-state window")
        # JointState is produced by Steve's onboard PC. Its ROS header can be
        # offset from the laptop clock even though messages are arriving live.
        # Receipt freshness plus the distinct-stamp settling window above is
        # the correct guard for this read-only recording operation.
        if (node.joints_received_at is None or
                time.monotonic() - node.joints_received_at > args.state_max_age):
            raise RuntimeError("joint state stream stopped before pose recording")
        deadline = time.monotonic() + 8.0
        while not node.tf_buffer.can_transform(
            config.planning_frame, config.camera_frame, Time()
        ) and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        if not node.tf_buffer.can_transform(config.planning_frame, config.camera_frame, Time()):
            raise RuntimeError(
                f"timed out waiting for TF {config.planning_frame} -> {config.camera_frame}"
            )
        transform = node.tf_buffer.lookup_transform(
            config.planning_frame, config.camera_frame, Time()
        )
        t, q = transform.transform.translation, transform.transform.rotation
        document = {
            "schema_version": 1,
            "name": args.name.strip(),
            "joint_positions_rad": dict(zip(
                config.joint_names,
                ordered_joint_positions(node.joints, config.joint_names),
            )),
            "recorded_camera_pose": {
                "planning_frame": config.planning_frame,
                "camera_frame": config.camera_frame,
                "translation_m": [float(t.x), float(t.y), float(t.z)],
                "quaternion_xyzw": [float(q.x), float(q.y), float(q.z), float(q.w)],
            },
            "joint_state_stamp_ns": stamp_ns(node.joints),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        node.get_logger().info(f"Recorded start pose without moving the robot: {output}")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
