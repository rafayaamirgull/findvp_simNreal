"""Execute the real survey runner against fake controllers on localhost 197 only."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import numpy as np
import pytest
import yaml

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("neo_msgs2.msg")
from controller_manager_msgs.msg import ControllerState
from controller_manager_msgs.srv import ListControllers
from geometry_msgs.msg import TransformStamped, Pose
from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from moveit_msgs.msg import CollisionObject
from moveit_msgs.srv import GetPlanningScene
from nav_msgs.msg import Odometry
from neo_msgs2.msg import EmergencyStopState
from neo_srvs2.srv import LockPlatform
from rcl_interfaces.msg import ParameterValue, ParameterType
from rcl_interfaces.srv import GetParameters
from rclpy.action import ActionServer
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from sensor_msgs.msg import JointState, CameraInfo, Image
from shape_msgs.msg import SolidPrimitive
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
from trajectory_msgs.msg import JointTrajectoryPoint
from ur_dashboard_msgs.msg import SafetyMode, RobotMode
from ur_dashboard_msgs.srv import GetSafetyMode, GetRobotMode, IsProgramRunning, RawRequest

from findvp_ros.capture_node import create_capture_node
from findvp_ros.configuration import load_robot_configuration
from findvp_ros.geometry import matrix_to_quaternion_xyzw
from findvp_ros.runtime import pose_matrix


@pytest.mark.parametrize("execution_failure", [False, True])
@pytest.mark.parametrize("base_mode", ["odom", "robot_mounted"])
def test_survey_exact_execution_capture_and_abort(tmp_path, monkeypatch, execution_failure, base_mode):
    root = Path(__file__).resolve().parents[1]
    config_path = tmp_path / "robot.yaml"
    config_document = yaml.safe_load((root / "config/steve_real.yaml").read_text())
    config_document["base_tracking_mode"] = base_mode
    config_path.write_text(yaml.safe_dump(config_document))
    config = load_robot_configuration(config_path)
    monkeypatch.setenv("ROS_LOCALHOST_ONLY", "1")
    monkeypatch.setenv("ROS_DOMAIN_ID", "197")
    rclpy.init(domain_id=197, args=["--ros-args", "-p", f"config_file:={config_path}",
                                  "-p", f"output_dir:={tmp_path}"])
    fake, recorder = Node("findvp_fake_survey_hardware"), create_capture_node()
    anchor = np.eye(4)
    anchor[:3, 3] = [.3, 0., .4]
    state = {"camera": anchor.copy(), "joints": [0.] * 6, "planned": [], "executed": []}
    latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    kinds = [(JointState, config.joint_states_topic, 10), (Odometry, config.odom_topic, 10),
             (EmergencyStopState, "/emergency_stop_state", 10),
             (SafetyMode, "/io_and_status_controller/safety_mode", latched),
             (RobotMode, "/io_and_status_controller/robot_mode", latched),
             (Image, config.image_topic, 10), (CameraInfo, config.camera_info_topic, 10)]
    if base_mode == "robot_mounted":
        kinds = [item for item in kinds if item[0] is not Odometry]
    pubs = [fake.create_publisher(kind, topic, qos) for kind, topic, qos in kinds]
    dynamic, static = TransformBroadcaster(fake), StaticTransformBroadcaster(fake)
    def transform(matrix, parent, child, stamp):
        message = TransformStamped()
        message.header.stamp, message.header.frame_id, message.child_frame_id = stamp, parent, child
        message.transform.translation.x, message.transform.translation.y, message.transform.translation.z = map(float, matrix[:3, 3])
        q = matrix_to_quaternion_xyzw(matrix[:3, :3])
        message.transform.rotation.x, message.transform.rotation.y, message.transform.rotation.z, message.transform.rotation.w = map(float, q)
        return message
    static.sendTransform(transform(config.tool_from_camera, config.tool_frame, config.camera_frame, fake.get_clock().now().to_msg()))
    def publish():
        stamp = fake.get_clock().now().to_msg()
        joints = JointState()
        joints.header.stamp, joints.name = stamp, list(config.joint_names)
        joints.position, joints.velocity = state["joints"], [0.] * 6
        odom = Odometry()
        odom.header.stamp, odom.header.frame_id, odom.child_frame_id = stamp, "odom", "base_link"
        odom.pose.pose.orientation.w = 1.
        safety, mode = SafetyMode(), RobotMode()
        safety.mode, mode.mode = 1, 7
        emergency = EmergencyStopState()
        emergency.emergency_state = emergency.EMFREE
        rgb = Image()
        rgb.header.stamp, rgb.header.frame_id = stamp, config.image_frame
        rgb.height, rgb.width, rgb.step, rgb.encoding, rgb.data = 2, 2, 6, "rgb8", bytes([100] * 12)
        info = CameraInfo()
        info.header, info.height, info.width = rgb.header, 2, 2
        info.k = [100., 0., 1., 0., 100., 1., 0., 0., 1.]
        info.distortion_model = "plumb_bob"
        messages = (joints, emergency, safety, mode, rgb, info) if base_mode == "robot_mounted" else (
            joints, odom, emergency, safety, mode, rgb, info)
        for publisher, message in zip(pubs, messages):
            publisher.publish(message)
        dynamic.sendTransform(transform(state["camera"] @ np.linalg.inv(config.tool_from_camera),
                                        config.planning_frame, config.tool_frame, stamp))
    fake.create_timer(.03, publish)
    def controllers(request, response):
        item = ControllerState()
        item.name, item.state = config.controller_name, "active"
        response.controller = [item]
        return response
    fake.create_service(ListControllers, "/controller_manager/list_controllers", controllers)
    def dashboard(request, response):
        if hasattr(response, "success"): response.success = True
        if hasattr(response, "safety_mode"): response.safety_mode.mode = 1
        if hasattr(response, "robot_mode"): response.robot_mode.mode = 7
        if hasattr(response, "program_running"): response.program_running = True
        if isinstance(request, RawRequest.Request): response.answer = "true"
        return response
    for service, name in [(GetSafetyMode, "get_safety_mode"), (GetRobotMode, "get_robot_mode"),
                          (IsProgramRunning, "program_running"), (RawRequest, "raw_request")]:
        fake.create_service(service, f"/dashboard_client/{name}", dashboard)
    fake.create_service(LockPlatform, "/lock_platform", dashboard)
    def description(request, response):
        value = ParameterValue()
        value.type = ParameterType.PARAMETER_STRING
        value.string_value = (
            f'<robot><link name="{config.camera_frame}"><collision><geometry>'
            '<box size=".1 .1 .1"/></geometry></collision></link>'
            f'<joint name="camera_mount" type="fixed"><parent link="{config.tool_frame}"/>'
            f'<child link="{config.camera_frame}"/>'
            '<origin xyz="0.034826821173538705 -0.003987870576912486 0.06553079550290158" '
            'rpy="0.01532675 0.00383024 3.07538020"/></joint></robot>'
        )
        response.values = [value]
        return response
    fake.create_service(GetParameters, "/move_group/get_parameters", description)
    def scene(request, response):
        obj = CollisionObject()
        obj.id, obj.header.frame_id = "table", config.planning_frame
        shape, pose = SolidPrimitive(), Pose()
        shape.type, shape.dimensions, pose.orientation.w = shape.BOX, [1., 1., .1], 1.
        obj.primitives, obj.primitive_poses = [shape], [pose]
        response.scene.world.collision_objects = [obj]
        return response
    fake.create_service(GetPlanningScene, "/get_planning_scene", scene)
    def plan(handle):
        assert handle.request.planning_options.plan_only
        goal = handle.request.request.goal_constraints[0]
        result = MoveGroup.Result()
        result.error_code.val = 1
        result.planned_trajectory.joint_trajectory.joint_names = list(config.joint_names)
        camera = anchor.copy()
        if goal.joint_constraints:
            end = [constraint.position for constraint in goal.joint_constraints]
        else:
            pose = goal.position_constraints[0].constraint_region.primitive_poses[0]
            pose.orientation = goal.orientation_constraints[0].orientation
            camera = pose_matrix(pose) @ config.tool_from_camera
            end = [float(len(state["planned"])) * .1] * 6
        first, last = JointTrajectoryPoint(), JointTrajectoryPoint()
        first.positions, last.positions, last.time_from_start.sec = state["joints"], end, 1
        result.planned_trajectory.joint_trajectory.points = [first, last]
        state["planned"].append((result.planned_trajectory, camera))
        handle.succeed()
        return result
    def execute(handle):
        state["executed"].append(handle.request.trajectory)
        trajectory, camera = state["planned"][-1]
        assert handle.request.trajectory == trajectory  # exact reviewed plan
        result = ExecuteTrajectory.Result()
        if execution_failure and len(state["executed"]) == 2:
            result.error_code.val = -4
            handle.abort()
        else:
            state["joints"] = list(trajectory.joint_trajectory.points[-1].positions)
            state["camera"] = camera
            result.error_code.val = 1
            handle.succeed()
        return result
    servers = [ActionServer(fake, MoveGroup, config.move_action, plan),
               ActionServer(fake, ExecuteTrajectory, "/execute_trajectory", execute)]
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(fake)
    executor.add_node(recorder)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    start = tmp_path / "start.yaml"
    start.write_text(yaml.safe_dump({"name": "start", "joint_positions_rad": dict.fromkeys(config.joint_names, 0.),
        "recorded_camera_pose": {"planning_frame": config.planning_frame, "camera_frame": config.camera_frame,
                                 "translation_m": [.3,0.,.4], "quaternion_xyzw": [0.,0.,0.,1.]}}))
    # Simulated approvals exist only in this isolated subprocess; production still requires a TTY.
    child = ('import io,sys; import findvp_ros.search_node as s; '
             'sys.stdin=io.StringIO("EXECUTE\\n"*3); sys.stdin.isatty=lambda:True; '
             's.select.select=lambda r,w,e,t:(r,[],[]); s.main(sys.argv[1:],survey_only=True)')
    try:
        result = subprocess.run([sys.executable, "-c", child, "--config", str(config_path),
            "--start-pose", str(start), "--capture-manifest", str(tmp_path / "observations.json"),
            "--output-dir", str(tmp_path / "survey"), "--execute", "--acknowledge-model-reviewed",
            "--required-scene-object", "table"], cwd=root, env=os.environ.copy(),
            capture_output=True, text=True, timeout=45.)
        assert result.returncode == (1 if execution_failure else 0), result.stdout + result.stderr
        report = json.loads((tmp_path / "survey/real_object_survey.json").read_text())
        assert report["status"] == ("failed" if execution_failure else "survey_complete")
        assert len(state["executed"]) == (2 if execution_failure else 3)
        observations = json.loads((tmp_path / "observations.json").read_text())["observations"]
        assert len(observations) == (1 if execution_failure else 3)
        if observations:
            assert observations[0]["base_reference"].get("mode", "odom") == base_mode
        if not execution_failure:
            xs = [v["world_from_camera"][0][3] for v in observations]
            np.testing.assert_allclose(xs, [.3,.34,.26], atol=1e-6)
    finally:
        executor.shutdown()
        thread.join(timeout=3.)
        for server in servers: server.destroy()
        recorder.destroy_node()
        fake.destroy_node()
        rclpy.shutdown()
