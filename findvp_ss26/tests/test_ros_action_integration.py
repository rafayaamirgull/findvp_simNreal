"""Exercise the action lifecycle against a fake MoveIt server, not a robot."""
import threading
import time

import pytest

rclpy = pytest.importorskip("rclpy")
from moveit_msgs.action import ExecuteTrajectory
from rclpy.action import ActionClient, ActionServer, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectoryPoint

from findvp_ros.runtime import ActionSession, DeploymentFault


@pytest.fixture
def action_graph(monkeypatch):
    monkeypatch.setenv("ROS_LOCALHOST_ONLY", "1")
    monkeypatch.setenv("ROS_DOMAIN_ID", "197")
    rclpy.init(domain_id=197)
    server_node = Node("fake_moveit")
    client_node = Node("findvp_test_client")
    received = []
    cancelled = threading.Event()

    def execute(handle):
        received.append(handle.request)
        deadline = time.monotonic() + .4
        while time.monotonic() < deadline:
            if handle.is_cancel_requested:
                handle.canceled()
                cancelled.set()
                result = ExecuteTrajectory.Result()
                result.error_code.val = -7
                return result
            time.sleep(.01)
        handle.succeed()
        result = ExecuteTrajectory.Result()
        result.error_code.val = 1
        return result

    server = ActionServer(server_node, ExecuteTrajectory, "/findvp_test_execute", execute,
                          cancel_callback=lambda _: CancelResponse.ACCEPT,
                          callback_group=ReentrantCallbackGroup())
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(server_node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    client = ActionClient(client_node, ExecuteTrajectory, "/findvp_test_execute")
    assert client.wait_for_server(timeout_sec=3.)
    session = ActionSession(lambda: rclpy.spin_once(client_node, timeout_sec=.02), cancel_timeout=2.)
    goal = ExecuteTrajectory.Goal()
    goal.trajectory.joint_trajectory.joint_names = ["test_joint"]
    point = JointTrajectoryPoint()
    point.positions = [.123]
    goal.trajectory.joint_trajectory.points = [point]
    try:
        yield session, client, goal, received, cancelled
    finally:
        executor.shutdown()
        thread.join(timeout=3.)
        server.destroy()
        client.destroy()
        client_node.destroy_node()
        server_node.destroy_node()
        rclpy.shutdown()


def test_exact_trajectory_reaches_fake_execution_server(action_graph):
    session, client, goal, received, _ = action_graph
    result = session.run(client, goal, 2.)
    assert result.status == 4
    assert received[0].trajectory == goal.trajectory
    assert session.handle is None


def test_fault_cancels_real_ros_action_and_blocks_next_goal(action_graph):
    session, client, goal, received, cancelled = action_graph

    def fault_after_acceptance():
        if received:
            raise DeploymentFault("injected safety loss")

    with pytest.raises(DeploymentFault, match="safety loss"):
        session.run(client, goal, 2., fault_after_acceptance)
    assert cancelled.wait(1.)
    with pytest.raises(DeploymentFault, match="aborted"):
        session.run(client, goal, 2.)
    assert len(received) == 1
