"""Real rclpy/pub-sub/service test on localhost domain 197; never Steve's domain.

Skipped on non-ROS hosts. No camera, MoveIt, driver, or robot is started.
"""
import json
from pathlib import Path
import threading
import time

import pytest

rclpy = pytest.importorskip("rclpy")
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_srvs.srv import Trigger
from tf2_ros import StaticTransformBroadcaster

from findvp_ros.capture_node import create_capture_node
from findvp_ros.configuration import load_robot_configuration


@pytest.fixture
def capture_graph(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("ROS_LOCALHOST_ONLY", "1")
    monkeypatch.setenv("ROS_DOMAIN_ID", "197")
    monkeypatch.setattr("ament_index_python.packages.get_package_share_directory", lambda _: str(root))
    rclpy.init(args=["--ros-args", "-p", f"config_file:={root / 'config/steve_real.yaml'}",
                    "-p", f"output_dir:={tmp_path}", "-p", "capture_timeout_sec:=0.8"], domain_id=197)
    recorder = create_capture_node()
    source = Node("findvp_fake_camera")
    config = load_robot_configuration(root / "config/steve_real.yaml")
    publishers = [source.create_publisher(kind, topic, 10) for kind, topic in (
        (Image, config.image_topic), (CameraInfo, config.camera_info_topic),
        (JointState, config.joint_states_topic), (Odometry, config.odom_topic))]
    transform = TransformStamped()
    transform.header.frame_id = config.planning_frame
    transform.child_frame_id = config.camera_frame
    transform.transform.rotation.w = 1.
    broadcaster = StaticTransformBroadcaster(source)
    broadcaster.sendTransform(transform)
    stream = {"enabled": True, "base_x": 0.}

    def publish():
        if not stream["enabled"]:
            return
        stamp = source.get_clock().now().to_msg()
        image = Image()
        image.header.stamp = stamp
        image.header.frame_id = config.image_frame
        image.height, image.width, image.step = 2, 2, 6
        image.encoding = "rgb8"
        image.data = bytes([100] * 12)
        info = CameraInfo()
        info.header = image.header
        info.height, info.width = 2, 2
        info.k = [100., 0., 1., 0., 100., 1., 0., 0., 1.]
        info.distortion_model = "plumb_bob"
        info.d = []  # valid uncalibrated-distortion/zero-distortion representation
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
        odom.pose.pose.position.x = stream["base_x"]
        for publisher, message in zip(publishers, (image, info, joints, odom)):
            publisher.publish(message)

    source.create_timer(.03, publish)
    client = source.create_client(Trigger, "/findvp/capture")
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(source)
    executor.add_node(recorder)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    assert client.wait_for_service(timeout_sec=3.)

    def trigger():
        future = client.call_async(Trigger.Request())
        completed = threading.Event()
        future.add_done_callback(lambda _: completed.set())
        assert completed.wait(4.), "capture service did not respond"
        return future.result()

    try:
        yield trigger, stream, tmp_path
    finally:
        executor.shutdown()
        thread.join(timeout=3.)
        recorder.destroy_node()
        source.destroy_node()
        rclpy.shutdown()


def test_capture_waits_for_new_exposure_and_records_measured_pose(capture_graph):
    trigger, stream, directory = capture_graph
    first = trigger()
    assert first.success, first.message
    second = trigger()
    assert second.success, second.message
    observations = json.loads((directory / "observations.json").read_text())["observations"]
    assert len(observations) == 2
    assert observations[1]["stamp_ns"] > observations[0]["stamp_ns"]
    assert all(record["stamp_ns"] > record["capture_after_ns"] for record in observations)
    assert observations[0]["distortion_coefficients"] == [0.] * 5
    assert observations[0]["world_from_camera"][3] == [0., 0., 0., 1.]
    assert all((directory / record["image"]).is_file() for record in observations)


def test_stopped_stream_is_not_recaptured(capture_graph):
    trigger, stream, directory = capture_graph
    assert trigger().success
    stream["enabled"] = False
    time.sleep(.15)  # drain messages already queued before the trigger barrier
    result = trigger()
    assert not result.success
    assert "timed out" in result.message
    assert len(json.loads((directory / "observations.json").read_text())["observations"]) == 1


def test_base_movement_requires_new_survey(capture_graph):
    trigger, stream, directory = capture_graph
    assert trigger().success
    stream["base_x"] = .02
    result = trigger()
    assert not result.success
    assert "base pose" in result.message
