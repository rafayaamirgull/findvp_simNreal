"""Verify that the scene CLI sends the intended collision objects to MoveIt."""
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest

rclpy = pytest.importorskip("rclpy")
from moveit_msgs.srv import ApplyPlanningScene
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node


def test_known_box_scene_registers_three_objects(tmp_path, monkeypatch):
    monkeypatch.setenv("ROS_DOMAIN_ID", "197")
    monkeypatch.setenv("ROS_LOCALHOST_ONLY", "1")
    rclpy.init(domain_id=197)
    fake = Node("fake_moveit_scene")
    scenes = []

    def apply(request, response):
        scenes.append(request.scene)
        response.success = True
        return response

    fake.create_service(ApplyPlanningScene, "/apply_planning_scene", apply)
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(fake)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    try:
        root = Path(__file__).resolve().parents[1]
        completed = subprocess.run([
            sys.executable, "-m", "findvp_ros.scene_cli", "--target-center", ".18", ".30", ".0635"
        ], cwd=root, env=os.environ.copy(), capture_output=True, text=True, timeout=15)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert len(scenes) == 1
        scene = scenes[0]
        assert scene.is_diff
        assert {obj.id for obj in scene.world.collision_objects} == {
            "findvp_support", "findvp_front_box", "findvp_target_keepout"}
        assert all(obj.header.frame_id == "ur5ebase_link" for obj in scene.world.collision_objects)
    finally:
        executor.shutdown()
        thread.join(timeout=3)
        fake.destroy_node()
        rclpy.shutdown()
