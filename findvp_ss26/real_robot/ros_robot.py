"""The only ROS node of the three-view run: joints, safety state, camera, TF, motion.

Runs with the system ROS Humble Python (plus /home/ws/.venv-findvp).  The motion
part is a port of the friend's verified ``ros_send_trajectory.py``:

* joints are merged from ``/joint_states`` **by name** (three partial publishers);
* ``robot_mode``/``safety_mode`` are latched -> Transient Local + Reliable QoS;
* a plan is refused if its first waypoint is >0.05 rad from the live arm;
* the whole plan is ONE multi-waypoint FollowJointTrajectory goal and we wait
  for the real result (never stack goals: that once crashed controller_manager);
* if the result message is lost, the actual pose decides success.
"""

from __future__ import annotations

import math
import time

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Point
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, JointState
from tf2_ros import Buffer, ConnectivityException, ExtrapolationException, LookupException, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker, MarkerArray

from geometry import BASE_FRAME, CANONICAL_JOINTS, PREFIX, box_corners, pose_to_matrix

COLOR_TOPIC = "/d415_calib/d415_calib/color/image_raw"
INFO_TOPIC = "/d415_calib/d415_calib/color/camera_info"
LIVE_CONTROLLER = "/scaled_joint_trajectory_controller"
MARKER_TOPIC = "/findvp/three_view/markers"
OVERLAY_TOPIC = "/findvp/three_view/overlay"
ROBOT_MODE_RUNNING = 7
SAFETY_MODE_NORMAL = 1


def _now() -> float:
    return time.monotonic()


class SteveRobot(Node):
    def __init__(self, controller: str = LIVE_CONTROLLER, check_safety: bool = True):
        super().__init__("findvp_three_view")
        self.controller = controller
        self.check_safety = check_safety
        self.joint_names = [PREFIX + name for name in CANONICAL_JOINTS]
        self._joints: dict[str, tuple[float, float]] = {}
        self._image = None
        self._image_time = 0.0
        self._info = None
        self.robot_mode = None
        self.safety_mode = None

        # Best effort subscribers accept both reliable and best-effort publishers.
        sensor = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(JointState, "/joint_states", self._on_joints, QoSProfile(
            depth=50, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.create_subscription(Image, COLOR_TOPIC, self._on_image, sensor)
        self.create_subscription(CameraInfo, INFO_TOPIC, self._on_info, sensor)
        if check_safety:
            from ur_dashboard_msgs.msg import RobotMode, SafetyMode

            latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                 reliability=ReliabilityPolicy.RELIABLE)
            self.create_subscription(RobotMode, "/io_and_status_controller/robot_mode",
                                     lambda m: setattr(self, "robot_mode", int(m.mode)), latched)
            self.create_subscription(SafetyMode, "/io_and_status_controller/safety_mode",
                                     lambda m: setattr(self, "safety_mode", int(m.mode)), latched)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)
        self.action = ActionClient(self, FollowJointTrajectory, f"{controller}/follow_joint_trajectory")
        marker_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                reliability=ReliabilityPolicy.RELIABLE)
        self.marker_pub = self.create_publisher(MarkerArray, MARKER_TOPIC, marker_qos)
        self.overlay_pub = self.create_publisher(Image, OVERLAY_TOPIC, 1)

    # ------------------------------------------------------------------ callbacks

    def _on_joints(self, msg):
        t = _now()
        for name, position in zip(msg.name, msg.position):
            self._joints[name] = (float(position), t)

    def _on_image(self, msg):
        self._image, self._image_time = msg, _now()

    def _on_info(self, msg):
        self._info = msg

    # ------------------------------------------------------------------ spinning

    def spin_for(self, seconds: float) -> None:
        end = _now() + seconds
        while _now() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def spin_until(self, predicate, timeout_s: float) -> bool:
        end = _now() + timeout_s
        while _now() < end:
            if predicate():
                return True
            rclpy.spin_once(self, timeout_sec=0.05)
        return bool(predicate())

    # ------------------------------------------------------------------ state

    def _fresh_joints(self, max_age_s: float):
        t = _now()
        values = []
        for name in self.joint_names:
            entry = self._joints.get(name)
            if entry is None or t - entry[1] > max_age_s:
                return None
            values.append(entry[0])
        return np.asarray(values)

    def joints(self, max_age_s: float = 0.5, timeout_s: float = 5.0) -> np.ndarray:
        """Six live arm joints (canonical order), received within ``max_age_s``."""

        if not self.spin_until(lambda: self._fresh_joints(max_age_s) is not None, timeout_s):
            missing = [n for n in self.joint_names if n not in self._joints]
            raise RuntimeError(
                f"no fresh /joint_states for the UR5e (missing: {missing or 'none, but stale'}); "
                "is steve_env.sh sourced and the robot bringup running?")
        return self._fresh_joints(max_age_s)

    def wait_still(self, hold_s: float = 1.0, tol_rad: float = 1e-3, timeout_s: float = 20.0) -> np.ndarray:
        """Joints once they have not changed by more than ``tol_rad`` for ``hold_s``."""

        end = _now() + timeout_s
        reference, since = None, 0.0
        while _now() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
            q = self._fresh_joints(0.5)
            if q is None:
                continue
            if reference is None or np.max(np.abs(q - reference)) > tol_rad:
                reference, since = q, _now()
            elif _now() - since >= hold_s:
                return q
        raise RuntimeError("the arm did not become stationary")

    def modes(self, timeout_s: float = 5.0):
        if self.check_safety:
            self.spin_until(lambda: self.robot_mode is not None and self.safety_mode is not None, timeout_s)
        return self.robot_mode, self.safety_mode

    def safety_ok(self) -> tuple[bool, str]:
        if not self.check_safety:
            return True, "safety check disabled"
        mode, safety = self.modes()
        if mode != ROBOT_MODE_RUNNING:
            return False, f"robot_mode={mode} (need {ROBOT_MODE_RUNNING}=RUNNING)"
        if safety != SAFETY_MODE_NORMAL:
            return False, f"safety_mode={safety} (need {SAFETY_MODE_NORMAL}=NORMAL)"
        return True, f"robot_mode={mode} RUNNING, safety_mode={safety} NORMAL"

    def lookup(self, frame: str, timeout_s: float = 3.0) -> np.ndarray:
        """Latest ``T_base_frame`` from TF."""

        result = {}

        def ready():
            try:
                tf = self.tf_buffer.lookup_transform(BASE_FRAME, frame, Time())
            except (LookupException, ConnectivityException, ExtrapolationException):
                return False
            t, q = tf.transform.translation, tf.transform.rotation
            result["T"] = pose_to_matrix([t.x, t.y, t.z], [q.x, q.y, q.z, q.w])
            return True

        if not self.spin_until(ready, timeout_s):
            raise RuntimeError(f"no TF {BASE_FRAME} -> {frame} (is start_camera.sh running?)")
        return result["T"]

    def intrinsics(self, timeout_s: float = 5.0):
        if not self.spin_until(lambda: self._info is not None, timeout_s):
            raise RuntimeError(f"no {INFO_TOPIC} (is start_camera.sh running?)")
        return np.asarray(self._info.k, dtype=float).reshape(3, 3), int(self._info.width), int(self._info.height)

    def latest_image(self, newer_than: float, timeout_s: float = 5.0):
        from findvp_ros.image_conversion import ros_image_to_pil

        if not self.spin_until(lambda: self._image is not None and self._image_time > newer_than, timeout_s):
            raise RuntimeError(f"no fresh image on {COLOR_TOPIC} (is start_camera.sh running?)")
        return ros_image_to_pil(self._image).convert("RGB")

    def capture(self, camera_frame: str, settle_s: float = 0.3) -> dict:
        """RGB image + camera pose taken while the arm is verifiably stationary."""

        q_before = self.wait_still()
        image = self.latest_image(newer_than=_now() + settle_s, timeout_s=5.0 + settle_s)
        T_base_camera = self.lookup(camera_frame)
        q_after = self.joints()
        if np.max(np.abs(q_after - q_before)) > 2e-3:
            raise RuntimeError("the arm moved during the capture; image discarded")
        K, width, height = self.intrinsics()
        if image.size != (width, height):
            raise RuntimeError(f"image size {image.size} != camera_info {(width, height)}")
        return {"image": image, "T_base_camera": T_base_camera, "joints": q_after, "K": K}

    # ------------------------------------------------------------------ motion

    def controller_available(self, timeout_s: float = 5.0) -> bool:
        return self.action.wait_for_server(timeout_sec=timeout_s)

    def controller_state(self, timeout_s: float = 5.0):
        """State of our trajectory controller from controller_manager ('active', ...), or None if unknown.

        An inactive controller still has its action server, but rejects every goal.
        """

        from controller_manager_msgs.srv import ListControllers

        client = self.create_client(ListControllers, "/controller_manager/list_controllers")
        try:
            if not client.wait_for_service(timeout_sec=timeout_s):
                return None
            future = client.call_async(ListControllers.Request())
            rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_s)
            if future.result() is None:
                return None
            name = self.controller.strip("/")
            for controller in future.result().controller:
                if controller.name == name:
                    return controller.state
            return "not loaded"
        finally:
            self.destroy_client(client)

    def motion_ready(self) -> tuple[bool, str]:
        """Safety modes + controller active (the same preflight for every command that moves)."""

        ok, why = self.safety_ok()
        if not ok:
            return False, why
        if not self.controller_available():
            return False, f"no action server {self.controller}/follow_joint_trajectory"
        state = self.controller_state()
        if state is None:
            print("  (controller_manager did not answer; relying on goal acceptance)")
        elif state != "active":
            return False, (f"{self.controller.strip('/')} is '{state}', not active: stop and play the "
                           "External Control program on the pendant")
        return True, why

    def execute(self, traj: dict, slow_factor: float, max_start_error: float = 0.05,
                result_timeout_s: float = 180.0) -> tuple[bool, str]:
        if slow_factor < 1.0:
            raise ValueError("slow_factor below 1.0 would speed up the planned motion")
        positions = np.asarray(traj["positions"], dtype=float)
        times = [float(t) * slow_factor for t in traj["time_from_start"]]
        if list(traj["joint_names"]) != CANONICAL_JOINTS:
            return False, "trajectory joint order is not canonical"

        live = self.joints()
        error = np.abs(live - positions[0])
        if error.max() > max_start_error:
            return False, (f"arm is NOT at the planned start (max joint error {error.max():.3f} rad > "
                           f"{max_start_error}); re-plan from the current pose")
        ok, why = self.safety_ok()
        if not ok:
            return False, why
        if not self.controller_available():
            return False, f"action server {self.controller}/follow_joint_trajectory not available"

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.joint_names = self.joint_names
        for row, t in zip(positions, times):
            point = JointTrajectoryPoint()
            point.positions = [float(v) for v in row]
            point.time_from_start = Duration(sec=int(t), nanosec=int((t % 1.0) * 1e9))
            goal.trajectory.points.append(point)
        print(f"  sending {len(positions)} waypoints to {self.controller}, duration {times[-1]:.1f} s ...")

        future = self.action.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        handle = future.result()
        if handle is None or not handle.accepted:
            return False, ("goal rejected -- check that scaled_joint_trajectory_controller is active "
                           "(timeout 10 ros2 control list_controllers --no-daemon) and External Control is Playing")
        result_future = handle.get_result_async()
        started = _now()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=result_timeout_s)
        wrapper = result_future.result()
        if wrapper is not None:
            code = wrapper.result.error_code
            if code != FollowJointTrajectory.Result.SUCCESSFUL:
                return False, f"controller reported failure: error_code={code} ({wrapper.result.error_string})"
        else:
            print("  ! no result message; checking the arm's actual pose instead ...")
        try:
            q = self.wait_still(hold_s=1.0, timeout_s=20.0)
        except RuntimeError as exc:
            return False, str(exc)
        final_error = float(np.max(np.abs(q - positions[-1])))
        if final_error > 0.01:
            return False, f"arm stopped {final_error:.3f} rad away from the planned goal"
        return True, f"reached goal in {_now() - started:.1f} s (max joint error {final_error:.4f} rad)"

    # ------------------------------------------------------------------ visualization

    def publish_overlay(self, image) -> None:
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "d415_calib_color_optical_frame"
        msg.height, msg.width = image.height, image.width
        msg.encoding, msg.step = "rgb8", 3 * image.width
        msg.data = np.asarray(image.convert("RGB"), dtype=np.uint8).tobytes()
        self.overlay_pub.publish(msg)

    def publish_markers(self, scene: dict, views: list[dict], target=None, path=None) -> None:
        """Object, obstacles, the three camera views, the target view and a planned tool0 path."""

        markers = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        def marker(ns, mid, kind, rgba):
            m = Marker()
            m.header.frame_id, m.header.stamp = BASE_FRAME, stamp
            m.ns, m.id, m.type, m.action = ns, mid, kind, Marker.ADD
            m.color.r, m.color.g, m.color.b, m.color.a = rgba
            m.pose.orientation.w = 1.0
            return m

        clear = Marker()
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)

        obj = scene["object"]
        m = marker("object", 0, Marker.CUBE, (0.1, 0.8, 0.2, 0.8))
        m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, obj["center_xyz_m"])
        yaw = math.radians(float(obj.get("yaw_deg", 0.0)))
        m.pose.orientation.z, m.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        m.scale.x, m.scale.y, m.scale.z = map(float, obj["dims_xyz_m"])
        markers.markers.append(m)
        for i, obstacle in enumerate(scene.get("obstacles", [])):
            m = marker("obstacles", i, Marker.CUBE, (1.0, 0.55, 0.1, 0.25))
            m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, obstacle["center_xyz_m"])
            qw, qx, qy, qz = map(float, obstacle.get("quat_wxyz", [1, 0, 0, 0]))
            m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w = qx, qy, qz, qw
            pad = 2 * float(obstacle.get("margin_m", 0.0))
            m.scale.x, m.scale.y, m.scale.z = (float(d) + pad for d in obstacle["dims_xyz_m"])
            markers.markers.append(m)

        colors = {"planned": (0.2, 0.5, 1.0, 1.0), "captured": (0.1, 0.9, 0.2, 1.0),
                  "best": (1.0, 0.8, 0.0, 1.0), "failed": (0.9, 0.1, 0.1, 1.0), "target": (0.9, 0.2, 0.9, 1.0)}
        entries = [(v["label"], v["T_base_camera"], v.get("state", "planned")) for v in views]
        if target is not None:
            entries.append(("target", target, "target"))
        for i, (label, T, state) in enumerate(entries):
            origin, forward = np.asarray(T)[:3, 3], np.asarray(T)[:3, 2]
            m = marker("views", i, Marker.ARROW, colors[state])
            m.points = [Point(x=float(a), y=float(b), z=float(c)) for a, b, c in (origin, origin + 0.12 * forward)]
            m.scale.x, m.scale.y, m.scale.z = 0.012, 0.025, 0.03
            markers.markers.append(m)
            m = marker("labels", i, Marker.TEXT_VIEW_FACING, (1.0, 1.0, 1.0, 1.0))
            m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, origin + [0, 0, 0.05])
            m.scale.z = 0.035
            m.text = label
            markers.markers.append(m)
        if path is not None and len(path) >= 2:
            m = marker("tool0_path", 0, Marker.LINE_STRIP, (0.2, 0.9, 1.0, 1.0))
            m.points = [Point(x=float(a), y=float(b), z=float(c)) for a, b, c in path]
            m.scale.x = 0.006
            markers.markers.append(m)
        self.marker_pub.publish(markers)


def draw_outline(image, obj, T_base_camera, K, text=None, color=(0, 255, 0)):
    """Copy of ``image`` with the projected object box (and optional caption) drawn on it."""

    from PIL import ImageDraw

    from geometry import box_outline, project_points

    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    polygon = box_outline(obj, T_base_camera, K, margin_m=0.0)
    if polygon is not None and len(polygon) >= 3:
        draw.line([tuple(p) for p in polygon.tolist()] + [tuple(polygon[0].tolist())], fill=color, width=3)
        uv, depth = project_points(np.asarray([obj["center_xyz_m"]]), T_base_camera, K)
        if depth[0] > 0:
            u, v = uv[0]
            draw.line([(u - 8, v), (u + 8, v)], fill=color, width=2)
            draw.line([(u, v - 8), (u, v + 8)], fill=color, width=2)
    if text:
        draw.rectangle([0, 0, canvas.width, 24], fill=(0, 0, 0))
        draw.text((6, 5), text, fill=(255, 255, 255))
    return canvas


__all__ = ["SteveRobot", "draw_outline", "box_corners"]
