"""Capture calibrated wrist-camera observations through a ROS Trigger service."""

from __future__ import annotations

from pathlib import Path
import json
import math
import threading
import time

import numpy as np

from .configuration import load_robot_configuration
from .base_tracking import robot_mounted_reference, require_reference_mode
from .geometry import transform_matrix
from .image_conversion import ros_image_to_pil
from .manifest import (
    append_observation,
    camera_matrix_from_info,
    make_observation_record,
    ordered_joint_positions,
    ordered_joint_velocities,
)
from .runtime import (pose_matrix, require_pose_close, require_recent, stamp_ns,
                      validate_capture_times)


def create_capture_node():
    """Construct the node in an initialized ROS context (also used by isolated tests)."""
    try:
        from ament_index_python.packages import get_package_share_directory
        from rclpy.duration import Duration
        from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from rclpy.time import Time
        from sensor_msgs.msg import CameraInfo, Image, JointState
        from nav_msgs.msg import Odometry
        from std_srvs.srv import Trigger
        from tf2_ros import Buffer, TransformListener
    except ImportError as error:  # pragma: no cover - requires ROS installation
        raise SystemExit(
            "findvp_capture requires a sourced ROS 2 installation and built findvp_ros package"
        ) from error

    class CaptureNode(Node):
        def __init__(self) -> None:
            super().__init__("findvp_capture")
            default_config = str(
                Path(get_package_share_directory("findvp_ros")) / "config" / "steve_real.yaml"
            )
            self.declare_parameter("config_file", default_config)
            self.declare_parameter("output_dir", "output/real_robot/survey")
            self.declare_parameter("manifest_name", "observations.json")
            self.declare_parameter("service_name", "/findvp/capture")
            self.declare_parameter("max_message_age_sec", 1.0)
            self.declare_parameter("tf_timeout_sec", 1.0)
            self.declare_parameter("capture_timeout_sec", 3.0)
            self.declare_parameter("max_sync_skew_sec", 0.1)
            self.config = load_robot_configuration(
                Path(self.get_parameter("config_file").value)
            )
            self.output_dir = Path(self.get_parameter("output_dir").value).expanduser().resolve()
            self.manifest_path = self.output_dir / str(
                self.get_parameter("manifest_name").value
            )
            self.maximum_age = float(self.get_parameter("max_message_age_sec").value)
            self.tf_timeout = float(self.get_parameter("tf_timeout_sec").value)
            self.capture_timeout = float(self.get_parameter("capture_timeout_sec").value)
            self.max_skew = float(self.get_parameter("max_sync_skew_sec").value)
            if any(not math.isfinite(v) or v <= 0 for v in (
                self.maximum_age, self.tf_timeout, self.capture_timeout, self.max_skew
            )):
                raise ValueError("capture timing parameters must be finite and positive")
            self._lock = threading.Condition()
            self._image = None
            self._camera_info = None
            self._joint_state = None
            self._joint_received_at = None
            self._image_received_at = None
            self._odom = None
            self._last_capture_stamp = 0
            self._base_reference = None
            old = []
            if self.manifest_path.exists():
                old = json.loads(self.manifest_path.read_text())["observations"]
                if old:
                    self._last_capture_stamp = int(old[-1]["stamp_ns"])
                    self._base_reference = old[0]["base_reference"]
                    require_reference_mode(self._base_reference, self.config.base_tracking_mode,
                                           self.config.planning_frame)
            if self.config.base_tracking_mode == "robot_mounted":
                self._base_reference = robot_mounted_reference(self.config.planning_frame)
            self._capture_sequence = len(old)
            self._buffer = Buffer()
            self._listener = TransformListener(self._buffer, self)
            self.create_subscription(
                Image, self.config.image_topic, self._on_image, qos_profile_sensor_data
            )
            self.create_subscription(
                CameraInfo,
                self.config.camera_info_topic,
                self._on_camera_info,
                qos_profile_sensor_data,
            )
            self.create_subscription(
                JointState,
                self.config.joint_states_topic,
                self._on_joint_state,
                qos_profile_sensor_data,
            )
            if self.config.base_tracking_mode == "odom":
                self.create_subscription(Odometry, self.config.odom_topic, self._on_odom,
                                         qos_profile_sensor_data)
            service_name = str(self.get_parameter("service_name").value)
            # The waiting service must not block image/JointState subscription callbacks.
            self._service_group = MutuallyExclusiveCallbackGroup()
            self.create_service(Trigger, service_name, self._capture,
                                callback_group=self._service_group)
            self.get_logger().info(
                f"Ready: call {service_name}; writing {self.manifest_path}"
            )

        def _on_image(self, message) -> None:
            with self._lock:
                self._image = message
                self._image_received_at = time.monotonic()
                self._lock.notify_all()

        def _on_camera_info(self, message) -> None:
            with self._lock:
                self._camera_info = message
                self._lock.notify_all()

        def _on_joint_state(self, message) -> None:
            # /joint_states has more than one publisher on Steve. Ignore base-only
            # drive messages and retain the newest complete UR5e state.
            try:
                ordered_joint_positions(message, self.config.joint_names)
            except ValueError:
                return
            with self._lock:
                self._joint_state = message
                self._joint_received_at = time.monotonic()
                self._lock.notify_all()

        def _on_odom(self, message):
            with self._lock:
                self._odom = message
                self._lock.notify_all()

        def _snapshot_after(self, after):
            deadline = time.monotonic() + self.capture_timeout
            reason = "waiting for image, calibration, joints and odometry"
            with self._lock:
                while time.monotonic() < deadline:
                    messages = (self._image, self._camera_info, self._joint_state, self._odom)
                    needed = messages if self.config.base_tracking_mode == "odom" else messages[:3]
                    if all(value is not None for value in needed):
                        image, info, joints, odom = messages
                        try:
                            now = self.get_clock().now().nanoseconds
                            validate_capture_times(
                                image=stamp_ns(image), info=stamp_ns(info), joints=stamp_ns(joints),
                                now=now, after=after, previous=self._last_capture_stamp,
                                max_age=self.maximum_age, max_skew=self.max_skew,
                                require_joint_sync=self.config.base_tracking_mode == "odom")
                            if self.config.base_tracking_mode == "robot_mounted":
                                # The arm and camera are on different computers.
                                # Require a newly received arm sample at/after the
                                # new image instead of comparing unsynchronized
                                # ROS header clocks.
                                receipt_age = (time.monotonic() - self._joint_received_at
                                               if self._joint_received_at is not None else math.inf)
                                if (receipt_age > self.maximum_age or
                                        self._image_received_at is None or
                                        self._joint_received_at < self._image_received_at):
                                    raise RuntimeError("waiting for live post-exposure joint state")
                            if odom is not None and self.config.base_tracking_mode == "odom":
                                require_recent(stamp_ns(odom), now, self.maximum_age, "odometry")
                                if abs(stamp_ns(odom) - stamp_ns(image)) > self.max_skew * 1e9:
                                    raise RuntimeError("odometry is not synchronized with the image")
                            # Do not accept a pre-exposure 'stationary' sample while
                            # newer motion/base state is still queued in DDS.
                            if self.config.base_tracking_mode == "odom" and (
                                stamp_ns(joints) < stamp_ns(image) or stamp_ns(odom) < stamp_ns(image)
                            ):
                                raise RuntimeError("waiting for post-exposure joint and base state")
                            if max(abs(v) for v in ordered_joint_velocities(joints, self.config.joint_names)) >= .01:
                                raise RuntimeError("arm is not stationary")
                            if self.config.base_tracking_mode == "odom":
                                twist = odom.twist.twist
                                speeds = [getattr(vector, axis) for vector in (twist.linear, twist.angular)
                                          for axis in ("x", "y", "z")]
                                if not np.isfinite(speeds).all() or max(abs(v) for v in speeds) >= .005:
                                    raise RuntimeError("mobile base is not stationary")
                            return messages
                        except (ValueError, RuntimeError) as error:
                            reason = str(error)
                    else:
                        names = (self.config.image_topic, self.config.camera_info_topic,
                                 self.config.joint_states_topic + " (complete arm state)",
                                 self.config.odom_topic)
                        reason = "no messages received on: " + ", ".join(
                            name for name, value in zip(names, needed) if value is None
                        )
                    self._lock.wait(timeout=min(.05, max(0., deadline - time.monotonic())))
            raise RuntimeError(f"fresh capture timed out: {reason}")

        def _capture(self, _request, response):
            try:
                after = self.get_clock().now().nanoseconds
                image_message, camera_info, joint_state, odom = self._snapshot_after(after)
                image_stamp_ns = stamp_ns(image_message)
                if (image_message.header.frame_id != self.config.image_frame or
                        camera_info.header.frame_id != self.config.image_frame):
                    raise RuntimeError("image/CameraInfo frame does not match configured optical frame")
                if camera_info.distortion_model not in ("plumb_bob", "rational_polynomial", ""):
                    raise RuntimeError("unsupported distortion model for OpenCV localization")
                if not camera_info.distortion_model and any(camera_info.d):
                    raise RuntimeError("nonzero distortion requires a named distortion model")
                base_reference = (robot_mounted_reference(self.config.planning_frame)
                    if self.config.base_tracking_mode == "robot_mounted" else
                    {"mode": "odom", "frame": odom.header.frame_id,
                     "child_frame": odom.child_frame_id,
                     "odom_from_base": pose_matrix(odom.pose.pose).tolist()})
                if self._base_reference is not None and self.config.base_tracking_mode == "odom":
                    if any(base_reference[k] != self._base_reference[k] for k in ("frame", "child_frame")):
                        raise RuntimeError("odometry frames changed; start a new survey")
                    require_pose_close(base_reference["odom_from_base"],
                                       self._base_reference["odom_from_base"], .005, math.radians(.5),
                                       "survey base pose; start a new survey if the base moved")
                if (
                    int(camera_info.width) != int(image_message.width)
                    or int(camera_info.height) != int(image_message.height)
                ):
                    raise RuntimeError("CameraInfo dimensions do not match the RGB image")
                # With a robot-mounted target the arm is verified stationary,
                # so latest TF avoids requiring clock synchronization between
                # Steve's onboard PC and the laptop camera. External-table mode
                # keeps exposure-time TF for metric multi-view localization.
                transform_time = (Time() if self.config.base_tracking_mode == "robot_mounted"
                                  else Time.from_msg(image_message.header.stamp))
                transform = self._buffer.lookup_transform(
                    self.config.planning_frame,
                    self.config.camera_frame,
                    transform_time,
                    timeout=Duration(seconds=self.tf_timeout),
                )
                translation = transform.transform.translation
                rotation = transform.transform.rotation
                planning_from_camera = transform_matrix(
                    (translation.x, translation.y, translation.z),
                    (rotation.x, rotation.y, rotation.z, rotation.w),
                )
                positions = ordered_joint_positions(joint_state, self.config.joint_names)
                pil_image = ros_image_to_pil(image_message).convert("RGB")
                self.output_dir.mkdir(parents=True, exist_ok=True)
                # Wall time only makes filenames unique; sensor time remains authoritative.
                filename = f"observation_{image_stamp_ns}_{time.time_ns()}.png"
                image_path = self.output_dir / filename
                temporary = image_path.with_suffix(".png.tmp")
                pil_image.save(temporary, format="PNG")
                temporary.replace(image_path)
                label = f"capture_{self._capture_sequence:03d}"
                record = make_observation_record(
                    image=filename,
                    label=label,
                    intrinsics=camera_matrix_from_info(camera_info),
                    distortion_coefficients=list(camera_info.d),
                    planning_from_camera=planning_from_camera,
                    stamp_ns=image_stamp_ns,
                    camera_frame=self.config.camera_frame,
                    image_header_frame=image_message.header.frame_id,
                    joint_names=self.config.joint_names,
                    joint_positions_rad=positions,
                )
                record["base_reference"] = base_reference
                record["capture_after_ns"] = after
                record["joint_stamp_ns"] = stamp_ns(joint_state)
                if odom is not None and self.config.base_tracking_mode == "odom":
                    record["odom_stamp_ns"] = stamp_ns(odom)
                index = append_observation(
                    self.manifest_path, self.config.planning_frame, record
                )
                self._capture_sequence += 1
                self._last_capture_stamp = image_stamp_ns
                self._base_reference = self._base_reference or base_reference
                response.success = True
                response.message = json.dumps({"index": index, "stamp_ns": image_stamp_ns,
                                               "manifest": str(self.manifest_path), "image": filename})
                self.get_logger().info(response.message)
            except Exception as error:  # service must report failures, not kill the node
                response.success = False
                response.message = f"capture failed: {type(error).__name__}: {error}"
                self.get_logger().error(response.message)
            return response

    return CaptureNode()


def main(args=None) -> None:
    import rclpy
    from rclpy.executors import MultiThreadedExecutor

    rclpy.init(args=args)
    node = create_capture_node()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
