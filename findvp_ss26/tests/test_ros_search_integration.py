"""Run the actual plan-only CLI against fake ROS endpoints on localhost 197."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import yaml

import pytest

rclpy = pytest.importorskip("rclpy")
neo_messages = pytest.importorskip("neo_msgs2.msg")
from controller_manager_msgs.msg import ControllerState
from controller_manager_msgs.srv import ListControllers
from moveit_msgs.action import MoveGroup
from nav_msgs.msg import Odometry
from rclpy.action import ActionServer
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint
from ur_dashboard_msgs.msg import RobotMode, SafetyMode

from findvp_ros.configuration import load_robot_configuration
from findvp_ros.runtime import file_digest


@pytest.mark.parametrize("planning_success", [True, False])
@pytest.mark.parametrize("survey_only", [False, True])
def test_plan_only_cli_never_executes_and_rejects_zero_valid_plans(tmp_path, monkeypatch, planning_success, survey_only):
    root = Path(__file__).resolve().parents[1]
    config_path = root / "config/steve_real.yaml"
    config = load_robot_configuration(config_path)
    monkeypatch.setenv("ROS_LOCALHOST_ONLY", "1")
    monkeypatch.setenv("ROS_DOMAIN_ID", "197")
    rclpy.init(domain_id=197)
    fake = Node("findvp_fake_robot_for_planning")
    latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    publishers = [fake.create_publisher(kind, topic, qos) for kind, topic, qos in (
        (JointState, config.joint_states_topic, 10),
        (Odometry, config.odom_topic, 10),
        (neo_messages.EmergencyStopState, "/emergency_stop_state", 10),
        (SafetyMode, "/io_and_status_controller/safety_mode", latched),
        (RobotMode, "/io_and_status_controller/robot_mode", latched),
    )]

    def publish():
        stamp = fake.get_clock().now().to_msg()
        joints = JointState()
        joints.header.stamp = stamp
        joints.name = list(config.joint_names)
        joints.position = [0.] * 6
        joints.velocity = [0.] * 6
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.pose.pose.orientation.w = 1.
        emergency = neo_messages.EmergencyStopState()
        emergency.emergency_state = emergency.EMFREE
        safety = SafetyMode()
        safety.mode = 1
        robot = RobotMode()
        robot.mode = 7
        for publisher, message in zip(publishers, (joints, odom, emergency, safety, robot)):
            publisher.publish(message)

    fake.create_timer(.03, publish)

    def controllers(_request, response):
        controller = ControllerState()
        controller.name = config.controller_name
        controller.state = "active"
        response.controller = [controller]
        return response
    fake.create_service(ListControllers, "/controller_manager/list_controllers", controllers)
    goals = []

    def plan(handle):
        goals.append(handle.request)
        result = MoveGroup.Result()
        if planning_success:
            result.error_code.val = 1
            result.planned_trajectory.joint_trajectory.joint_names = list(config.joint_names)
            point = JointTrajectoryPoint()
            point.positions = [0.] * 6
            result.planned_trajectory.joint_trajectory.points = [point]
            handle.succeed()
        else:
            result.error_code.val = -1  # PLANNING_FAILED, safely skippable
            handle.abort()
        return result
    server = ActionServer(fake, MoveGroup, config.move_action, plan)
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(fake)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    target = tmp_path / "target.png"
    target.write_bytes(b"not loaded in plan-only mode")
    localization = tmp_path / "localization.json"
    localization.write_text(json.dumps({
        "coordinate_frame": config.planning_frame,
        "estimate_kind": "visible_surface_aim_point",
        "center_world_m": [.4, 0., .4],
        "target_sha256": file_digest(target),
        "base_reference": {"frame": "odom", "child_frame": "base_link",
                           "odom_from_base": [[1, 0, 0, 0], [0, 1, 0, 0],
                                              [0, 0, 1, 0], [0, 0, 0, 1]]},
    }))
    try:
        start = tmp_path / "start.yaml"
        start.write_text(yaml.safe_dump({"name": "start", "joint_positions_rad": dict.fromkeys(config.joint_names, 0.),
            "recorded_camera_pose": {"planning_frame": config.planning_frame, "camera_frame": config.camera_frame,
                                     "translation_m": [.3,0.,.4], "quaternion_xyzw": [0.,0.,0.,1.]}}))
        mode_args = (["--start-pose", str(start)] if survey_only else
                     ["--localization", str(localization), "--target-image", str(target)])
        result = subprocess.run([
            sys.executable, "-m", "findvp_ros.survey_node" if survey_only else "findvp_ros.search_node",
            "--config", str(config_path), *mode_args,
            "--capture-manifest", str(tmp_path / "observations.json"),
            "--output-dir", str(tmp_path / "search"),
        ], cwd=root, env=os.environ.copy(), capture_output=True, text=True, timeout=20.)
        assert result.returncode == (0 if planning_success else 1), result.stdout + result.stderr
        report = json.loads((tmp_path / f"search/real_object_{'survey' if survey_only else 'real_search'}.json").read_text())
        assert report["status"] == ("plan_only_complete" if planning_success else "failed")
        assert len(goals) == (1 if survey_only and not planning_success else 3)
        assert all(goal.planning_options.plan_only for goal in goals)
        assert all(goal.request.goal_constraints[0].position_constraints[0].link_name == config.tool_frame
                   for goal in (goals[1:] if survey_only else goals))
        assert not (tmp_path / "observations.json").exists()
    finally:
        executor.shutdown()
        thread.join(timeout=3.)
        server.destroy()
        fake.destroy_node()
        rclpy.shutdown()
