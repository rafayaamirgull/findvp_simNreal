"""Closed-loop real-robot triangle viewpoint search through MoveIt 2.

The target center comes from RGB-only multi-view localization. No mesh, class,
dimensions, simulator state, or known object pose is consumed. Hardware motion
is opt-in and guarded by live robot/controller checks and explicit model review.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time
import queue
import threading
import signal
import select
import sys
import xml.etree.ElementTree as ET

import numpy as np

from .configuration import load_named_joint_pose, load_robot_configuration
from .base_tracking import robot_mounted_reference, require_reference_mode
from .geometry import (
    camera_mount_from_urdf,
    camera_pose_to_tool_pose,
    fixed_joint_transform_from_urdf,
    matrix_to_quaternion_xyzw,
    transform_matrix,
)
from .manifest import ordered_joint_positions, ordered_joint_velocities
from .steve_workcell import known_carton_keepout_radius
from .target_selection import load_reference_mask, validate_target_provenance
from .runtime import (ActionSession, DeploymentFault, PlanningRejected, SettlingWindow,
                      StateWatchdog, collision_primitive_center, pose_matrix, require_pose_close,
                      require_recent, require_remote_control, stamp_ns, validate_trajectory_start)


def _load_localization(path: Path, expected_frame: str) -> np.ndarray:
    document = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if document.get("coordinate_frame") != expected_frame:
        raise ValueError(
            f"localization frame {document.get('coordinate_frame')!r} does not match "
            f"planning frame {expected_frame!r}"
        )
    if document.get("estimate_kind") not in ("visible_surface_aim_point", "known_object_center"):
        raise ValueError("localization must contain a visible-surface aim point or a known object centre")
    center = np.asarray(document.get("center_world_m"), dtype=float)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("localization center_world_m must contain three finite values")
    return center


def _argument_parser(survey_only=False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(
        "Supervised three-view RGB survey: recorded start, optical-right and optical-left. "
        "Default is plan-only; every execution requires operator approval."
        if survey_only else __doc__))
    parser.set_defaults(survey_only=survey_only)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--localization", type=Path, required=not survey_only)
    parser.add_argument("--target-image", type=Path, required=not survey_only)
    parser.add_argument("--target-mask", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument(
        "--start-pose",
        type=Path,
        required=survey_only,
        help="Operator-recorded joint pose used as the supervised table-view start state.",
    )
    parser.add_argument("--start-look-at-tolerance-deg", type=float, default=30.0)
    parser.add_argument("--name", default="real_object")
    parser.add_argument("--views", type=int, choices=(3, 4, 5), default=3)
    parser.add_argument("--radius", type=float, default=0.40)
    parser.add_argument("--azimuth-offset", type=float, default=0.0)
    parser.add_argument("--max-refinement-depth", type=int, default=2)
    parser.add_argument("--three-view-only", action="store_true",
                        help="Capture and rank exactly the first three triangle views, then return to the best captured joint pose")
    parser.add_argument("--similarity-threshold", type=float, default=0.90)
    parser.add_argument("--position-tolerance", type=float, default=0.005)
    parser.add_argument("--orientation-tolerance-deg", type=float, default=2.0)
    parser.add_argument("--planning-time", type=float, default=5.0)
    parser.add_argument("--velocity-scale", type=float, default=0.05)
    parser.add_argument("--acceleration-scale", type=float, default=0.05)
    parser.add_argument("--settle-seconds", type=float, default=0.5)
    parser.add_argument("--feature-max-image-size", type=int, default=448)
    parser.add_argument("--state-max-age", type=float, default=2.0)
    parser.add_argument("--survey-baseline", type=float, default=0.04,
                        help="Survey translation to each side of the start, in metres (0.02–0.10)")
    parser.add_argument("--sam-points-per-batch", type=int, choices=(1, 2, 4, 8, 16, 32), default=4)
    parser.add_argument("--mask-backend", choices=("sam", "flat"), default="sam")
    parser.add_argument("--required-scene-object", action="append", default=[],
                        help="Reviewed collision-object ID; repeat for support, target keepout, etc.")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--acknowledge-model-reviewed",
        action="store_true",
        help=(
            "Required with --execute. Confirms that the current MoveIt scene, wrist-camera "
            "collision body, table/obstacles, and generated plans were reviewed."
        ),
    )
    return parser


def _validate_arguments(args) -> None:
    if args.execute and not args.acknowledge_model_reviewed:
        raise ValueError("--execute requires --acknowledge-model-reviewed")
    if args.execute and not args.required_scene_object:
        raise ValueError("--execute requires --required-scene-object IDs for the reviewed environment")
    if args.execute and args.start_pose is None:
        raise ValueError("--execute requires an operator-recorded --start-pose")
    if args.execute and not args.survey_only and args.target_mask is None:
        raise ValueError("--execute requires --target-mask from findvp_select_target")
    if not math.isfinite(args.survey_baseline) or not .02 <= args.survey_baseline <= .10:
        raise ValueError("--survey-baseline must be between 0.02 and 0.10 metres")
    if args.sam_points_per_batch not in (1, 2, 4, 8, 16, 32):
        raise ValueError("--sam-points-per-batch must divide the SAM prompt grid")
    if not math.isfinite(args.azimuth_offset):
        raise ValueError("--azimuth-offset must be finite")
    if Path(args.name).name != args.name or args.name in ("", ".", ".."):
        raise ValueError("--name must be a nonempty filename without directories")
    for name in (
        "radius",
        "position_tolerance",
        "orientation_tolerance_deg",
        "planning_time",
        "settle_seconds",
        "state_max_age",
        "start_look_at_tolerance_deg",
    ):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")
    for name in ("velocity_scale", "acceleration_scale", "similarity_threshold"):
        if not math.isfinite(getattr(args, name)) or not 0.0 < getattr(args, name) <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in (0, 1]")
    if args.max_refinement_depth < 0:
        raise ValueError("--max-refinement-depth must be non-negative")
    if args.three_view_only and args.views != 3:
        raise ValueError("--three-view-only requires --views 3")


def main(argv=None, *, survey_only=False) -> None:
    parsed, ros_arguments = _argument_parser(survey_only).parse_known_args(argv)
    _validate_arguments(parsed)
    try:
        import rclpy
        from action_msgs.msg import GoalStatus
        from controller_manager_msgs.srv import ListControllers
        from geometry_msgs.msg import Pose
        from moveit_msgs.action import MoveGroup, ExecuteTrajectory
        from moveit_msgs.srv import GetPlanningScene
        from moveit_msgs.msg import (
            Constraints,
            DisplayTrajectory,
            PlanningSceneComponents,
            MoveItErrorCodes,
            JointConstraint,
            OrientationConstraint,
            PositionConstraint,
        )
        from neo_msgs2.msg import EmergencyStopState
        from nav_msgs.msg import Odometry
        from rcl_interfaces.srv import GetParameters
        from rosidl_runtime_py.convert import message_to_ordereddict
        from rclpy.action import ActionClient
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy
        from sensor_msgs.msg import JointState
        from shape_msgs.msg import SolidPrimitive
        from std_msgs.msg import Bool
        from std_srvs.srv import Trigger
        from ur_dashboard_msgs.msg import RobotMode, SafetyMode
        from ur_dashboard_msgs.srv import (GetRobotMode, GetSafetyMode,
                                           IsProgramRunning, IsInRemoteControl,
                                           RawRequest)
        from tf2_ros import Buffer, TransformListener
        from rclpy.time import Time
        from rclpy.signals import SignalHandlerOptions
    except ImportError as error:  # pragma: no cover - requires ROS installation
        raise SystemExit(
            f"findvp_search requires the sourced Steve ROS 2 workspace: {error}"
        ) from error

    try:
        from neo_srvs2.srv import LockPlatform
    except ImportError as error:  # plan-only remains useful on partial student builds
        if parsed.execute:
            raise SystemExit(
                "--execute requires neo_srvs2 Python interfaces; build and source neo_srvs2"
            ) from error
        LockPlatform = None

    config = load_robot_configuration(parsed.config)
    start_pose = (
        load_named_joint_pose(parsed.start_pose, config.joint_names)
        if parsed.start_pose is not None
        else None
    )
    center = None if survey_only else _load_localization(parsed.localization, config.planning_frame)
    target_image = None if survey_only else parsed.target_image.expanduser().resolve()
    if target_image is not None and not target_image.is_file():
        raise FileNotFoundError(f"target image not found: {target_image}")
    parsed.output_dir = parsed.output_dir.expanduser().resolve()
    parsed.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = parsed.output_dir / f"{parsed.name}_{'survey' if survey_only else 'real_search'}.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite {report_path}; choose a new --output-dir or --name")
    parsed.capture_manifest = parsed.capture_manifest.expanduser().resolve()
    base_reference = None
    survey_anchor = None
    if survey_only:
        from .survey_node import load_survey_anchor
        survey_anchor = load_survey_anchor(parsed.start_pose, config)
        if parsed.execute and parsed.capture_manifest.exists():
            if json.loads(parsed.capture_manifest.read_text()).get("observations"):
                raise ValueError("automatic survey requires an empty capture manifest; use a new run directory")
    else:
        localization = json.loads(parsed.localization.expanduser().read_text())
        validate_target_provenance(localization, target_image, parsed.target_mask)
        base_reference = localization.get("base_reference")
        if not base_reference:
            raise ValueError("localization lacks base-pose provenance; collect a new survey")
        require_reference_mode(base_reference, config.base_tracking_mode, config.planning_frame)

    # Resolve model weights, imports and reference segmentation BEFORE any motion.
    extractor = prepared_target = mask_provider = None
    if parsed.execute and not survey_only:
        from sample.dinov3_feature_matching import DINOv3FeatureExtractor
        from sample.target_guided_similarity import prepare_target_image, rank_triangle_views
        extractor = DINOv3FeatureExtractor(
            max_image_size=parsed.feature_max_image_size, upscale_small_images=True)
        prepared_target = prepare_target_image(target_image, extractor=extractor,
            target_mask=load_reference_mask(parsed.target_mask, target_image))
        from .segmentation import make_mask_provider
        mask_provider = make_mask_provider(parsed.mask_backend, points_per_batch=parsed.sam_points_per_batch)
        if mask_provider is not None:
            mask_provider(prepared_target.image)  # validate inference before motion

    from findvp_ros.triangle_search import (
        hemisphere_triangles,
        subdivide_spherical_triangle,
    )

    class SearchNode(Node):
        def __init__(self) -> None:
            super().__init__("findvp_survey" if survey_only else "findvp_search")
            self.emergency = None
            self.safety_mode = None
            self.robot_mode = None
            self.program_running = None
            self.remote_control = None
            self.joint_state = None
            self.odom = None
            self.watchdog = StateWatchdog(parsed.state_max_age)
            self.actions = ActionSession(lambda: rclpy.spin_once(self, timeout_sec=.05))
            self.dashboard_pending = {}
            self.dashboard_sent = {}
            self.controller_state = None
            self.armed = False
            self.abort_reason = None
            latched_status = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.create_subscription(
                EmergencyStopState,
                "/emergency_stop_state",
                lambda value: self._receive("emergency", value),
                qos_profile_sensor_data,
            )
            self.create_subscription(
                SafetyMode,
                "/io_and_status_controller/safety_mode",
                lambda value: self._receive("safety_mode", value, heartbeat=False),
                latched_status,
            )
            self.create_subscription(
                RobotMode,
                "/io_and_status_controller/robot_mode",
                lambda value: self._receive("robot_mode", value, heartbeat=False),
                latched_status,
            )
            self.create_subscription(
                Bool,
                "/io_and_status_controller/robot_program_running",
                lambda value: self._receive("external_program", bool(value.data), heartbeat=False),
                latched_status,
            )
            self.create_subscription(
                JointState, config.joint_states_topic, self._on_joint_state,
                qos_profile_sensor_data,
            )
            self.move_group = ActionClient(self, MoveGroup, config.move_action)
            self.execute_action = ActionClient(self, ExecuteTrajectory, "/execute_trajectory")
            self.display = self.create_publisher(DisplayTrajectory, "/display_planned_path", 1)
            if config.base_tracking_mode == "odom":
                self.create_subscription(Odometry, config.odom_topic,
                                         lambda value: self._receive("odom", value), qos_profile_sensor_data)
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)
            self.scene_client = self.create_client(GetPlanningScene, "/get_planning_scene")
            self.description_client = self.create_client(GetParameters, "/move_group/get_parameters")
            self.controller_client = self.create_client(
                ListControllers, "/controller_manager/list_controllers"
            )
            self.capture_client = self.create_client(Trigger, "/findvp/capture")
            self.capture_parameters = self.create_client(GetParameters, "/findvp_capture/get_parameters")
            self.lock_client = (
                self.create_client(LockPlatform, "/lock_platform")
                if LockPlatform is not None
                else None
            )
            self.dashboard = {
                "safety_mode": (self.create_client(GetSafetyMode, "/dashboard_client/get_safety_mode"), GetSafetyMode),
                "robot_mode": (self.create_client(GetRobotMode, "/dashboard_client/get_robot_mode"), GetRobotMode),
                "program_running": (self.create_client(IsProgramRunning, "/dashboard_client/program_running"), IsProgramRunning),
                "remote_control": (self.create_client(IsInRemoteControl, "/dashboard_client/is_in_remote_control"), IsInRemoteControl),
            }
            # Some Steve deployments use an older dashboard node which exposes
            # RawRequest but omits the typed IsInRemoteControl service.
            self.remote_control_raw = self.create_client(
                RawRequest, "/dashboard_client/raw_request"
            )
            self.external_program = None
            self.create_timer(.5, self._poll_status)

        def _receive(self, name, value, heartbeat=True):
            setattr(self, name, value)
            if heartbeat:
                self.watchdog.update(name)
            if self.armed:
                unsafe = (
                    (name == "emergency" and value.emergency_state != EmergencyStopState.EMFREE)
                    or (name == "safety_mode" and value.mode != 1)
                    or (name == "robot_mode" and value.mode != 7)
                    or (name in ("program_running", "external_program", "remote_control") and value is not True)
                    or (name == "controller_state" and value != "active")
                )
                if unsafe:
                    self.abort_reason = self.abort_reason or f"unsafe {name} reported during session"

        def _poll_status(self):
            # UR mode topics can be change-only. Poll read-only dashboard services
            # for liveness instead of pretending cached mode messages are heartbeats.
            queries = dict(self.dashboard)
            queries["controller_state"] = (self.controller_client, ListControllers)
            for name, (client, service) in queries.items():
                pending = self.dashboard_pending.get(name)
                if pending is not None:
                    if not pending.done():
                        continue  # watchdog expires a hung request
                    self.dashboard_pending.pop(name)
                    try:
                        response = pending.result()
                        if time.monotonic() - self.dashboard_sent[name] > parsed.state_max_age:
                            continue  # an old service response is not a current status sample
                        if name == "controller_state":
                            state = {c.name: c.state for c in response.controller}.get(config.controller_name)
                            self._receive(name, state)
                        elif name == "remote_control" and hasattr(response, "answer"):
                            answer = str(response.answer).strip().lower()
                            if answer in ("true", "false"):
                                self._receive(name, answer == "true")
                        elif response.success:
                            self._receive(name, getattr(response, name))
                    except Exception as error:
                        self.get_logger().error(f"status query failed: {error}")
                selected_client, request = client, service.Request()
                if (name == "remote_control" and not client.service_is_ready()
                        and self.remote_control_raw.service_is_ready()):
                    selected_client = self.remote_control_raw
                    request = RawRequest.Request()
                    request.query = "is in remote control"
                if selected_client.service_is_ready():
                    self.dashboard_pending[name] = selected_client.call_async(request)
                    self.dashboard_sent[name] = time.monotonic()

        def _on_joint_state(self, value) -> None:
            try:
                ordered_joint_positions(value, config.joint_names)
            except ValueError:
                return
            self._receive("joint_state", value)

        def spin_until(self, predicate, timeout: float, description: str) -> None:
            deadline = time.monotonic() + timeout
            while not predicate() and time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=0.05)
            if not predicate():
                raise RuntimeError(f"timed out waiting for {description}")

        def motion_is_safe(self) -> tuple[bool, str]:
            try:
                streams = ["emergency", "joint_state", "safety_mode", "robot_mode",
                           "program_running", "controller_state", "remote_control"]
                if config.base_tracking_mode == "odom":
                    streams.append("odom")
                self.watchdog.require(*streams)
                require_remote_control(self.remote_control)
                now = self.get_clock().now().nanoseconds
                # In robot-mounted mode Steve's onboard JointState clock need
                # not be synchronized to the camera laptop. The watchdog is
                # based on local receipt time and still fails closed if the
                # live stream stops. External-table mode retains strict stamp
                # checks because it combines robot state with laptop imagery
                # and odometry in a shared time domain.
                if config.base_tracking_mode == "odom":
                    require_recent(stamp_ns(self.joint_state), now,
                                   parsed.state_max_age, "joint state")
                self.require_base_unchanged()
            except (ValueError, RuntimeError) as error:
                return False, str(error)
            if self.emergency is None or self.safety_mode is None or self.robot_mode is None:
                return False, "robot safety/status topics are not ready"
            if int(self.emergency.emergency_state) != int(EmergencyStopState.EMFREE):
                return False, "mobile platform emergency stop is not clear"
            if int(self.safety_mode.mode) != 1:
                return False, f"UR safety mode is not NORMAL (mode={self.safety_mode.mode})"
            if int(self.robot_mode.mode) != 7:
                return False, f"UR robot mode is not RUNNING (mode={self.robot_mode.mode})"
            if self.program_running is not True:
                return False, "UR external-control program is not reported running"
            # Older Steve driver builds do not publish this optional status topic.
            # A received false is fatal; when absent, the continuously polled
            # dashboard program_running state and active controller are authoritative.
            if self.external_program is False:
                return False, "UR driver reports the external-control program is not running"
            if self.controller_state != "active":
                return False, "required trajectory controller is not active"
            return True, "ready"

        def require_safe(self):
            if self.abort_reason is not None:
                raise DeploymentFault(self.abort_reason)
            if self.actions.faulted:
                raise DeploymentFault("motion session is aborted")
            safe, reason = self.motion_is_safe()
            if not safe:
                raise DeploymentFault(reason)

        def require_base_unchanged(self):
            if config.base_tracking_mode == "robot_mounted":
                require_reference_mode(base_reference, "robot_mounted", config.planning_frame)
                return
            self.watchdog.require("odom")
            require_recent(stamp_ns(self.odom), self.get_clock().now().nanoseconds,
                           parsed.state_max_age, "odometry")
            if (self.odom.header.frame_id != base_reference["frame"] or
                    self.odom.child_frame_id != base_reference["child_frame"]):
                raise DeploymentFault("odometry frames differ from survey")
            require_pose_close(pose_matrix(self.odom.pose.pose), base_reference["odom_from_base"],
                               .005, math.radians(.5), "base pose since survey")
            twist = self.odom.twist.twist
            speeds = [getattr(v, a) for v in (twist.linear, twist.angular) for a in ("x", "y", "z")]
            if not np.isfinite(speeds).all() or max(abs(v) for v in speeds) >= .005:
                raise DeploymentFault("mobile base is moving")

        def service(self, client, request, label):
            if not client.wait_for_service(timeout_sec=3.):
                raise DeploymentFault(f"{label} service unavailable")
            future = client.call_async(request)
            self.spin_until(future.done, 5., label)
            return future.result()

        def scene_snapshot(self):
            request = GetPlanningScene.Request()
            request.components.components = (PlanningSceneComponents.WORLD_OBJECT_GEOMETRY |
                                             PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS |
                                             PlanningSceneComponents.OCTOMAP |
                                             PlanningSceneComponents.ALLOWED_COLLISION_MATRIX |
                                             PlanningSceneComponents.TRANSFORMS)
            scene = self.service(self.scene_client, request, "planning scene").scene
            objects = {obj.id: obj for obj in scene.world.collision_objects}
            for name in parsed.required_scene_object:
                obj = objects.get(name)
                if obj is None or not (obj.primitives or obj.meshes or obj.planes):
                    raise DeploymentFault(f"required collision geometry missing: {name}")
            if (not survey_only and parsed.required_scene_object and
                    localization.get("estimate_kind") == "known_object_center"):
                keepout = objects.get("findvp_target_keepout")
                if keepout is None or not keepout.primitives or not keepout.primitive_poses:
                    raise DeploymentFault("known target needs a target keepout in MoveIt")
                sphere = keepout.primitives[0]
                if keepout.header.frame_id == config.planning_frame:
                    planning_from_header = np.eye(4)
                else:
                    source_frame = keepout.header.frame_id
                    if not source_frame:
                        raise DeploymentFault("target keepout has no coordinate frame")
                    self.spin_until(
                        lambda: self.tf_buffer.can_transform(
                            config.planning_frame, source_frame, Time()
                        ),
                        3.0,
                        "target keepout frame transform",
                    )
                    planning_from_header = pose_matrix(self.tf_buffer.lookup_transform(
                        config.planning_frame, source_frame, Time()
                    ).transform)
                keepout_center = collision_primitive_center(keepout, planning_from_header)
                # The registered sphere must contain the carton itself.  Use
                # zero placement uncertainty here because any extra margin is
                # already represented by the sphere radius.
                carton_radius = known_carton_keepout_radius(0.0)
                if (sphere.type != SolidPrimitive.SPHERE or not sphere.dimensions or
                        np.linalg.norm(keepout_center - center) + carton_radius > sphere.dimensions[0]):
                    raise DeploymentFault(
                        "target keepout does not enclose the known carton at its seeded centre: "
                        f"seeded={center.tolist()}, scene={keepout_center.tolist()}, "
                        f"scene_radius={list(sphere.dimensions)}; verify the physical box placement"
                    )
            # Ignore transport timestamps: compare geometry, transforms and collision policy.
            def without_stamps(value):
                if isinstance(value, dict):
                    return {k: without_stamps(v) for k, v in value.items() if k != "stamp"}
                if isinstance(value, list):
                    return [without_stamps(v) for v in value]
                return value
            return without_stamps({"world": message_to_ordereddict(scene.world),
                                   "attached": [message_to_ordereddict(v) for v in scene.robot_state.attached_collision_objects],
                                   "transforms": [message_to_ordereddict(v) for v in scene.fixed_frame_transforms],
                                   "acm": message_to_ordereddict(scene.allowed_collision_matrix)})

        def preflight(self, execute: bool) -> dict:
            nonlocal base_reference
            self.spin_until(
                lambda: self.joint_state is not None
                and self.emergency is not None
                and self.safety_mode is not None
                and self.robot_mode is not None
                and (config.base_tracking_mode == "robot_mounted" or self.odom is not None),
                8.0,
                "joint and safety state",
            )
            if not self.move_group.wait_for_server(timeout_sec=5.0):
                raise RuntimeError(f"MoveGroup action unavailable: {config.move_action}")
            action_status_publishers = self.get_publishers_info_by_topic(
                config.move_action.rstrip("/") + "/_action/status")
            if len(action_status_publishers) != 1:
                raise DeploymentFault(
                    f"expected exactly one MoveIt action server for {config.move_action}; "
                    f"found {len(action_status_publishers)}. Stop duplicate MoveIt launches before planning")
            if not self.controller_client.wait_for_service(timeout_sec=5.0):
                raise RuntimeError("controller manager list service is unavailable")
            future = self.controller_client.call_async(ListControllers.Request())
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
            if not future.done() or future.exception() is not None:
                raise RuntimeError("could not inspect controller states")
            controllers = {item.name: item.state for item in future.result().controller}
            controller_state = controllers.get(config.controller_name, "missing")
            if base_reference is None:
                base_reference = (robot_mounted_reference(config.planning_frame)
                    if config.base_tracking_mode == "robot_mounted" else
                    {"mode": "odom", "frame": self.odom.header.frame_id,
                     "child_frame": self.odom.child_frame_id,
                     "odom_from_base": pose_matrix(self.odom.pose.pose).tolist()})
            self.require_base_unchanged()
            if parsed.required_scene_object:
                # Preview must use the same populated scene as execution.
                # A plan generated against an empty scene is not a useful
                # clearance preview for this known-box workcell.
                self.scene_snapshot()
            if execute:
                try:
                    self.spin_until(lambda: self.motion_is_safe()[0], 8., "fresh, normal robot/controller status")
                except RuntimeError as error:
                    raise DeploymentFault(f"execution preflight failed: {self.motion_is_safe()[1]}") from error
                safe, reason = self.motion_is_safe()
                if not safe:
                    raise RuntimeError(reason)
                self.armed = True  # subsequent unsafe events stay latched even if they clear
                if controller_state != "active":
                    raise RuntimeError(
                        f"required controller {config.controller_name!r} is {controller_state!r}"
                    )
                if not self.capture_client.wait_for_service(timeout_sec=5.0):
                    raise RuntimeError("findvp capture service is unavailable")
                request = GetParameters.Request()
                request.names = ["output_dir", "manifest_name"]
                capture_config = self.service(self.capture_parameters, request, "capture configuration")
                destination = (Path(capture_config.values[0].string_value).expanduser()
                               / capture_config.values[1].string_value).resolve()
                if destination != parsed.capture_manifest:
                    raise DeploymentFault("capture recorder writes a different manifest; restart it for this run")
                if self.lock_client is None or not self.lock_client.wait_for_service(timeout_sec=5.0):
                    raise RuntimeError("platform lock service is unavailable")
                lock_future = self.lock_client.call_async(LockPlatform.Request())
                rclpy.spin_until_future_complete(self, lock_future, timeout_sec=5.0)
                if not lock_future.done() or not lock_future.result().success:
                    raise RuntimeError("mobile platform could not be locked")
                if not self.execute_action.wait_for_server(timeout_sec=5.):
                    raise DeploymentFault("ExecuteTrajectory action unavailable")
                if len(self.get_publishers_info_by_topic("/execute_trajectory/_action/status")) != 1:
                    raise DeploymentFault("expected exactly one ExecuteTrajectory action server")
                self.scene_snapshot()
                description_request = GetParameters.Request()
                description_request.names = ["robot_description"]
                response = self.service(self.description_client, description_request, "MoveIt robot model")
                model = ET.fromstring(response.values[0].string_value)
                model_base_from_arm = (
                    fixed_joint_transform_from_urdf(model, "base_link", "cabinet_link")
                    @ fixed_joint_transform_from_urdf(model, "cabinet_link", config.planning_frame)
                )
                live_base_from_arm = pose_matrix(self.tf_buffer.lookup_transform(
                    "base_link", config.planning_frame, Time()
                ).transform)
                require_pose_close(live_base_from_arm, model_base_from_arm,
                                   .005, math.radians(.5),
                                   "MoveIt arm mount versus live robot TF")
                camera_link = next((link for link in model.findall("link")
                                    if link.get("name") == config.camera_frame), None)
                if camera_link is None or camera_link.find("collision/geometry") is None:
                    raise DeploymentFault("calibrated camera link has no collision geometry in MoveIt's URDF")
                require_pose_close(camera_mount_from_urdf(model, config.tool_frame, config.camera_frame),
                                   config.tool_from_camera, .001, math.radians(.2),
                                   "MoveIt camera mount versus hand-eye calibration")
                calibrated = self.tf_buffer.lookup_transform(config.tool_frame, config.camera_frame, Time())
                require_pose_close(pose_matrix(calibrated.transform), config.tool_from_camera,
                                   .001, math.radians(.2), "hand-eye calibration versus TF")
            return {"controller_state": controller_state, "execute": execute}

        def _goal(self, camera_viewpoint, execute: bool):
            base_from_camera = transform_matrix(
                camera_viewpoint.position_xyz, camera_viewpoint.orientation_xyzw
            )
            base_from_tool = camera_pose_to_tool_pose(
                base_from_camera, config.tool_from_camera
            )
            tool_quaternion = matrix_to_quaternion_xyzw(base_from_tool[:3, :3])

            primitive = SolidPrimitive()
            primitive.type = SolidPrimitive.SPHERE
            primitive.dimensions = [float(parsed.position_tolerance)]
            region_pose = Pose()
            region_pose.position.x, region_pose.position.y, region_pose.position.z = (
                float(value) for value in base_from_tool[:3, 3]
            )
            region_pose.orientation.w = 1.0
            position = PositionConstraint()
            position.header.frame_id = config.planning_frame
            position.link_name = config.tool_frame
            position.constraint_region.primitives = [primitive]
            position.constraint_region.primitive_poses = [region_pose]
            position.weight = 1.0

            orientation = OrientationConstraint()
            orientation.header.frame_id = config.planning_frame
            orientation.link_name = config.tool_frame
            (
                orientation.orientation.x,
                orientation.orientation.y,
                orientation.orientation.z,
                orientation.orientation.w,
            ) = tool_quaternion
            tolerance = math.radians(parsed.orientation_tolerance_deg)
            orientation.absolute_x_axis_tolerance = tolerance
            orientation.absolute_y_axis_tolerance = tolerance
            orientation.absolute_z_axis_tolerance = tolerance
            orientation.weight = 1.0

            constraints = Constraints()
            constraints.name = "findvp_camera_pose"
            constraints.position_constraints = [position]
            constraints.orientation_constraints = [orientation]
            goal = MoveGroup.Goal()
            goal.request.group_name = config.move_group
            goal.request.num_planning_attempts = 3
            goal.request.allowed_planning_time = float(parsed.planning_time)
            goal.request.max_velocity_scaling_factor = float(parsed.velocity_scale)
            goal.request.max_acceleration_scaling_factor = float(parsed.acceleration_scale)
            goal.request.goal_constraints = [constraints]
            goal.planning_options.plan_only = not execute
            goal.planning_options.look_around = False
            goal.planning_options.replan = False
            return goal

        def _start_goal(self, joint_positions):
            constraints = Constraints()
            constraints.name = "findvp_reviewed_start_pose"
            for name, value in zip(config.joint_names, joint_positions):
                joint = JointConstraint()
                joint.joint_name = name
                joint.position = float(value)
                joint.tolerance_above = 0.01
                joint.tolerance_below = 0.01
                joint.weight = 1.0
                constraints.joint_constraints.append(joint)
            goal = MoveGroup.Goal()
            goal.request.group_name = config.move_group
            goal.request.num_planning_attempts = 3
            goal.request.allowed_planning_time = float(parsed.planning_time)
            goal.request.max_velocity_scaling_factor = float(parsed.velocity_scale)
            goal.request.max_acceleration_scaling_factor = float(parsed.acceleration_scale)
            goal.request.goal_constraints = [constraints]
            goal.planning_options.plan_only = True
            goal.planning_options.look_around = False
            goal.planning_options.replan = False
            return goal

        def _checked_plan_result(self, wrapped, scene, label):
            result = wrapped.result
            code = int(result.error_code.val)
            if code in (MoveItErrorCodes.PLANNING_FAILED, MoveItErrorCodes.NO_IK_SOLUTION,
                        MoveItErrorCodes.GOAL_IN_COLLISION, MoveItErrorCodes.GOAL_CONSTRAINTS_VIOLATED):
                raise PlanningRejected(f"no valid {label} plan: MoveIt code {code}")
            if wrapped.status != GoalStatus.STATUS_SUCCEEDED or code != MoveItErrorCodes.SUCCESS:
                raise DeploymentFault(
                    f"{label} planning failed: action status {wrapped.status}, MoveIt code {code}"
                )
            trajectory = result.planned_trajectory.joint_trajectory
            if not trajectory.points:
                raise PlanningRejected(f"planner returned an empty {label} trajectory")
            display = DisplayTrajectory()
            display.trajectory_start = result.trajectory_start
            display.trajectory = [result.planned_trajectory]
            self.display.publish(display)
            metadata = {
                "planning_time_s": float(result.planning_time),
                "trajectory_points": len(trajectory.points),
                "joint_names": list(trajectory.joint_names),
                "trajectory": message_to_ordereddict(result.planned_trajectory),
            }
            return result.planned_trajectory, metadata, scene

        def plan_start(self, joint_positions):
            self.require_base_unchanged()
            scene = self.scene_snapshot() if parsed.execute else None
            wrapped = self.actions.run(
                self.move_group,
                self._start_goal(joint_positions),
                parsed.planning_time + 10.0,
                self.require_safe if parsed.execute else None,
            )
            return self._checked_plan_result(wrapped, scene, "start-pose")

        def plan(self, viewpoint):
            self.require_base_unchanged()
            scene = self.scene_snapshot() if parsed.execute else None
            wrapped = self.actions.run(self.move_group, self._goal(viewpoint, False),
                                       parsed.planning_time + 10.,
                                       self.require_safe if parsed.execute else None)
            return self._checked_plan_result(wrapped, scene, "candidate")

        def execute(self, trajectory, scene):
            # Remain responsive to status updates while the operator inspects RViz.
            if not sys.stdin.isatty():
                raise DeploymentFault("execution requires an interactive operator terminal")
            print("Inspect /display_planned_path in RViz. Type EXECUTE for this trajectory: ",
                  end="", flush=True)
            deadline = time.monotonic() + 120.
            answer = None
            while answer is None and time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=.05)
                self.require_safe()
                if select.select([sys.stdin], [], [], 0)[0]:
                    answer = sys.stdin.readline().strip()
            if answer != "EXECUTE":
                raise DeploymentFault("operator did not approve this trajectory")
            self.require_safe()
            if self.scene_snapshot() != scene:
                raise DeploymentFault("planning scene changed after planning; refusing trajectory")
            self.require_safe()
            validate_trajectory_start(trajectory, config.joint_names,
                                      ordered_joint_positions(self.joint_state, config.joint_names))
            goal = ExecuteTrajectory.Goal()
            goal.trajectory = trajectory  # exact displayed plan; never plan-and-execute again
            wrapped = self.actions.run(self.execute_action, goal, 120., self.require_safe)
            if (wrapped.status != GoalStatus.STATUS_SUCCEEDED or
                    wrapped.result.error_code.val != MoveItErrorCodes.SUCCESS):
                raise DeploymentFault(f"trajectory execution failed: {wrapped.result.error_code.val}")
            return {"status": "succeeded", "exact_planned_trajectory": True}

        def wait_stationary(self) -> None:
            window = SettlingWindow(parsed.settle_seconds)
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=0.05)
                self.require_safe()
                if window.update(stamp_ns(self.joint_state),
                                 ordered_joint_velocities(self.joint_state, config.joint_names)):
                    return
            raise RuntimeError("robot did not become stationary before capture")

        def verify_start_view(self, expected_joint_positions) -> dict:
            """Require the optical +Z axis to point near the localized aim point."""
            self.wait_stationary()
            actual_joints = np.asarray(
                ordered_joint_positions(self.joint_state, config.joint_names), dtype=float
            )
            joint_error = float(np.max(np.abs(
                actual_joints - np.asarray(expected_joint_positions, dtype=float)
            )))
            if joint_error > 0.02:
                raise DeploymentFault(
                    f"robot did not reach the recorded start pose: max error {joint_error:.4f} rad"
                )
            stamped = self.tf_buffer.lookup_transform(
                config.planning_frame, config.camera_frame, Time()
            )
            planning_from_camera = pose_matrix(stamped.transform)
            if config.base_tracking_mode == "odom":
                require_recent(stamp_ns(stamped), self.get_clock().now().nanoseconds,
                               parsed.state_max_age, "camera TF")
            if survey_only:
                require_pose_close(planning_from_camera, survey_anchor,
                                   parsed.position_tolerance * 2.,
                                   math.radians(parsed.orientation_tolerance_deg * 2.),
                                   "recorded survey camera pose")
                return {"max_joint_error_rad": joint_error,
                        "world_from_camera": planning_from_camera.tolist()}
            direction = center - planning_from_camera[:3, 3]
            distance = float(np.linalg.norm(direction))
            if not math.isfinite(distance) or distance <= 0.05:
                raise DeploymentFault("start pose camera is too close to the localized aim point")
            cosine = float(np.dot(planning_from_camera[:3, 2], direction / distance))
            error_deg = math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))
            if error_deg > parsed.start_look_at_tolerance_deg:
                raise DeploymentFault(
                    f"start pose camera does not face the localized object: {error_deg:.1f} deg"
                )
            return {
                "max_joint_error_rad": joint_error,
                "aim_error_deg": error_deg,
                "aim_distance_m": distance,
            }

        def capture(self, viewpoint) -> dict:
            self.wait_stationary()
            after = self.get_clock().now().nanoseconds
            future = self.capture_client.call_async(Trigger.Request())
            deadline = time.monotonic() + 8.
            while not future.done() and time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=.05)
                self.require_safe()
            if not future.done() or future.exception() is not None or not future.result().success:
                detail = (future.result().message if future.done() and future.exception() is None
                          else "service response missing")
                raise DeploymentFault(f"camera capture failed or timed out: {detail}")
            self.require_safe()
            receipt = json.loads(future.result().message)
            if Path(receipt["manifest"]).resolve() != parsed.capture_manifest:
                raise DeploymentFault("capture service writes a different manifest")
            document = json.loads(parsed.capture_manifest.read_text(encoding="utf-8"))
            record = dict(document["observations"][int(receipt["index"])])
            if (document["coordinate_frame"] != config.planning_frame or
                    record["stamp_ns"] != receipt["stamp_ns"] or record["image"] != receipt["image"] or
                    record["stamp_ns"] <= after or record["camera_frame"] != config.camera_frame):
                raise DeploymentFault("capture receipt/pose/timestamp mismatch")
            recorded_base = record.get("base_reference", {})
            require_reference_mode(recorded_base, config.base_tracking_mode,
                                   config.planning_frame)
            if any(recorded_base.get(k) != base_reference[k] for k in ("frame", "child_frame")):
                raise DeploymentFault("captured base frames differ from the motion session")
            require_pose_close(recorded_base.get("odom_from_base"), base_reference["odom_from_base"],
                               .005, math.radians(.5), "captured base pose")
            requested = transform_matrix(viewpoint.position_xyz, viewpoint.orientation_xyzw)
            require_pose_close(record["world_from_camera"], requested, parsed.position_tolerance * 2.,
                               math.radians(parsed.orientation_tolerance_deg * 2.), "camera endpoint")
            record["image_file"] = str((parsed.capture_manifest.parent / record["image"]).resolve())
            return record

        def rank_responsively(self, operation):
            """Keep status subscriptions alive during potentially slow GPU inference."""
            result = queue.Queue()
            def worker():
                try:
                    result.put((True, operation()))
                except BaseException as error:
                    result.put((False, error))
            threading.Thread(target=worker, daemon=True).start()
            while result.empty():
                rclpy.spin_once(self, timeout_sec=.05)
                self.require_safe()
            success, value = result.get()
            if not success:
                raise value
            return value

    # Keep the ROS context alive for bounded cancellation on Ctrl+C/SIGTERM.
    rclpy.init(args=ros_arguments, signal_handler_options=SignalHandlerOptions.NO)
    def interrupt(_signum, _frame):
        raise KeyboardInterrupt()
    old_term_handler = signal.signal(signal.SIGTERM, interrupt)
    node = SearchNode()
    report = {
        "schema_version": 1,
        "status": "in_progress",
        "mode": "execute" if parsed.execute else "plan_only",
        "coordinate_frame": config.planning_frame,
        "target_image": str(target_image) if target_image is not None else None,
        "localization": str(parsed.localization.expanduser().resolve()) if parsed.localization else None,
        "object_prior_used": bool(localization.get("object_prior_used", False)) if not survey_only else False,
        "aim_point_xyz_m": center.tolist() if center is not None else None,
        "levels": [],
        "best_observation": None,
        "base_reference": base_reference,
        "start_pose": None,
    }
    try:
        report["preflight"] = node.preflight(parsed.execute)
        report["base_reference"] = base_reference
        if start_pose is not None:
            trajectory, metadata, scene = node.plan_start(start_pose.joint_positions)
            report["start_pose"] = {
                "name": start_pose.name,
                "file": str(start_pose.source),
                "plan": metadata,
                "executed": False,
            }
            report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            if parsed.execute:
                report["start_pose"]["execution"] = node.execute(trajectory, scene)
                report["start_pose"]["executed"] = True
                report["start_pose"]["view_check"] = node.verify_start_view(
                    start_pose.joint_positions
                )
                if survey_only:
                    survey_anchor = np.asarray(report["start_pose"]["view_check"]["world_from_camera"])
                report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if survey_only:
            from .survey_node import run_survey
            run_survey(node, parsed, survey_anchor, report,
                       lambda: report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8"))
            node.get_logger().info(f"Survey report: {report_path}")
            return
        triangles, viewpoints = hemisphere_triangles(
            tuple(center), parsed.radius, parsed.views, parsed.azimuth_offset
        )
        best_score = -math.inf
        for level in range(parsed.max_refinement_depth + 1):
            level_records = []
            level_report = {"level": level, "candidates": level_records}
            report["levels"].append(level_report)
            for triangle, viewpoint in zip(triangles, viewpoints):
                record = dict(triangle)
                record["requested_camera_pose"] = {
                    "position_xyz_m": list(viewpoint.position_xyz),
                    "orientation_xyzw": list(viewpoint.orientation_xyzw),
                }
                level_records.append(record)
                try:
                    trajectory, record["plan"], scene = node.plan(viewpoint)
                    record["planned"] = True
                    # Persist the full plan before approval or hardware execution.
                    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                    if parsed.execute:
                        record["execution"] = node.execute(trajectory, scene)
                        record["observation"] = node.capture(viewpoint)
                        record["image_file"] = record["observation"]["image_file"]
                        record["captured"] = True
                except PlanningRejected as error:
                    if parsed.three_view_only:
                        raise DeploymentFault(
                            f"three-view run requires every candidate to be reachable: {error}"
                        ) from error
                    record["planned"] = False
                    record["captured"] = False
                    record["failure"] = f"{type(error).__name__}: {error}"
                    node.get_logger().error(record["failure"])

            viewpoints_path = parsed.output_dir / f"{parsed.name}_level_{level:02d}_viewpoints.json"
            for record in level_records:
                if record.get("captured"):
                    record["image_filename"] = os.path.relpath(
                        record["image_file"], viewpoints_path.parent
                    )
            scored_records = [item for item in level_records if item.get("captured")]
            viewpoints_path.write_text(json.dumps(scored_records if parsed.execute else level_records,
                                                  indent=2) + "\n", encoding="utf-8")
            level_report["viewpoints_file"] = str(viewpoints_path)
            report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            if not parsed.execute:
                if not any(item.get("planned") for item in level_records):
                    raise DeploymentFault("no candidate has a valid plan")
                report["status"] = "plan_only_complete"
                break
            captured = [item for item in level_records if item.get("captured")]
            if parsed.three_view_only and len(captured) != 3:
                raise DeploymentFault("three-view run requires exactly three captured images")
            if not captured:
                raise RuntimeError("no candidate at this level was captured")
            ranking = node.rank_responsively(lambda: rank_triangle_views(
                target_image,
                viewpoints_path,
                extractor=extractor,
                prepared_target=prepared_target,
                target_guided_candidates=True,
                candidate_mask_provider=mask_provider,
            ))
            level_report["ranking_file"] = ranking["report_file"]
            level_report["scores"] = ranking["ranking"]
            if parsed.three_view_only:
                for item in ranking["ranking"]:
                    node.get_logger().info(
                        f"Triangle {item['triangle']}: combined={item['score']:.4f}, "
                        f"DINOv3={item['dino_score']:.4f}, SIFT={item['sift_score']:.4f}, "
                        f"image={item['image_filename']}"
                    )
            if parsed.three_view_only and (
                len(ranking["ranking"]) != 3 or
                {int(item["triangle"]) for item in ranking["ranking"]} !=
                {int(item["triangle"]) for item in captured}
            ):
                raise DeploymentFault("three-view run requires valid DINOv3/SIFT scores for all three images")
            level_report["best_score"] = float(ranking["ranking"][0]["score"])
            winner_id = int(ranking["best_triangle"])
            winner = next(item for item in triangles if int(item["triangle"]) == winner_id)
            improved = level_report["best_score"] > best_score
            if improved:
                best_score = level_report["best_score"]
                best_record = next(item for item in captured if int(item["triangle"]) == winner_id)
                report["best_observation"] = {"score": best_score, "level": level,
                                              "triangle": winner_id, **best_record["observation"]}
            report["best_score"] = best_score
            if parsed.three_view_only:
                best_record = next(item for item in captured if int(item["triangle"]) == winner_id)
                goal_joints = best_record["observation"]["joint_positions_rad"]
                if best_record["observation"]["joint_names"] != list(config.joint_names):
                    raise DeploymentFault("winning observation joint order differs from robot configuration")
                report["winner"] = {
                    "triangle": winner_id,
                    "score": best_score,
                    "image_file": best_record["image_file"],
                    "joint_positions_rad": goal_joints,
                }
                report["return_to_winner"] = {"status": "pending"}
                report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                node.wait_stationary()
                current_joints = ordered_joint_positions(node.joint_state, config.joint_names)
                if np.max(np.abs(np.asarray(current_joints) - np.asarray(goal_joints))) > 0.02:
                    trajectory, plan_metadata, scene = node.plan_start(goal_joints)
                    report["return_to_winner"]["plan"] = plan_metadata
                    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                    report["return_to_winner"]["execution"] = node.execute(trajectory, scene)
                    node.wait_stationary()
                reached_joints = ordered_joint_positions(node.joint_state, config.joint_names)
                joint_error = float(np.max(np.abs(np.asarray(reached_joints) - np.asarray(goal_joints))))
                if joint_error > 0.02:
                    raise DeploymentFault(f"robot did not return to winning view: joint error {joint_error:.4f} rad")
                reached_camera = pose_matrix(node.tf_buffer.lookup_transform(
                    config.planning_frame, config.camera_frame, Time()
                ).transform)
                require_pose_close(
                    reached_camera,
                    best_record["observation"]["world_from_camera"],
                    parsed.position_tolerance * 2.,
                    math.radians(parsed.orientation_tolerance_deg * 2.),
                    "returned winning camera pose",
                )
                report["return_to_winner"].update({"status": "reached", "max_joint_error_rad": joint_error})
                report["status"] = "complete"
                report["stop_reason"] = "three_views_ranked_and_winner_reached"
                node.get_logger().info(
                    f"Best triangle: {winner_id}, score={best_score:.4f}, "
                    f"image={best_record['image_file']}; returned to its captured joint pose"
                )
                break
            if best_score >= parsed.similarity_threshold or level >= parsed.max_refinement_depth:
                report["status"] = "complete"
                report["stop_reason"] = ("similarity_threshold" if best_score >= parsed.similarity_threshold
                                         else "refinement_budget_exhausted")
                break
            if not improved:
                report["status"] = "complete"
                report["stop_reason"] = "no_improvement"
                break
            triangles, viewpoints = subdivide_spherical_triangle(
                winner, tuple(center), parsed.radius
            )
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        node.get_logger().info(f"Search report: {report_path}")
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        try:
            node.actions.cancel()
            report["action_cleanup"] = "no_pending_action"
        except BaseException as cleanup_error:
            report["action_cleanup"] = str(cleanup_error)
            node.get_logger().fatal(str(cleanup_error))
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        raise
    finally:
        if parsed.execute:
            node.get_logger().warning("Base remains locked. No automatic return motion; use the reviewed stow procedure.")
        node.destroy_node()
        rclpy.shutdown()
        signal.signal(signal.SIGTERM, old_term_handler)


if __name__ == "__main__":
    main()
