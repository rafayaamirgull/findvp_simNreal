"""Fault-injection tests. No ROS graph, model downloads, or hardware are used."""
from concurrent.futures import Future
from types import SimpleNamespace as NS

import numpy as np
import pytest

from findvp_ros.runtime import (
    ActionSession, DeploymentFault, SettlingWindow, StateWatchdog,
    require_pose_close, require_recent, require_remote_control,
    validate_capture_times, validate_trajectory_start,
)


class Clock:
    def __init__(self):
        self.now = 10.
        self.steps = []

    def __call__(self):
        return self.now

    def spin(self):
        self.now += .05
        for operation in list(self.steps):
            operation()


@pytest.mark.parametrize("value", [None, False])
def test_local_or_unknown_control_mode_is_not_execution_ready(value):
    with pytest.raises(DeploymentFault, match="AND select Remote Control"):
        require_remote_control(value)


def test_remote_control_confirmation_is_accepted():
    require_remote_control(True)


def done(value):
    future = Future()
    future.set_result(value)
    return future


class Handle:
    accepted = True

    def __init__(self, finish_on_cancel=True):
        self.result = Future()
        self.cancelled = 0
        self.finish_on_cancel = finish_on_cancel

    def get_result_async(self):
        return self.result

    def cancel_goal_async(self):
        self.cancelled += 1
        if self.finish_on_cancel and not self.result.done():
            self.result.set_result(NS(status=5))
        return done(NS(goals_canceling=[1]))


class Client:
    def __init__(self, handle=None, future=None):
        self.handle = handle or Handle()
        self.future = future or done(self.handle)
        self.goals = []

    def send_goal_async(self, goal):
        self.goals.append(goal)
        return self.future


def test_watchdog_expires_cached_normal_state():
    clock = Clock()
    guard = StateWatchdog(1., clock)
    guard.update("safety")
    guard.require("safety")
    clock.now += 1.01
    with pytest.raises(DeploymentFault, match="stale"):
        guard.require("safety")


def test_watchdog_requires_every_stream():
    guard = StateWatchdog()
    guard.update("joints")
    with pytest.raises(DeploymentFault, match="odom"):
        guard.require("joints", "odom")


def test_old_zero_velocity_message_cannot_advance_settling():
    window = SettlingWindow(.5)
    assert not any(window.update(1_000_000_000, [0.] * 6) for _ in range(100))


def test_settling_requires_contiguous_new_samples():
    window = SettlingWindow(.5)
    for i in range(5):
        assert not window.update(1_000_000_000 + i * 100_000_000, [0.] * 6)
    assert window.update(1_500_000_000, [0.] * 6)
    assert not window.update(1_600_000_000, [.02] * 6)
    assert not window.update(2_000_000_000, [0.] * 6)
    assert not window.update(2_600_000_000, [0.] * 6)  # gap resets the interval


@pytest.mark.parametrize("stamp,now", [(0, 10), (11, 10), (1, 2_000_000_000)])
def test_invalid_acquisition_stamps_fail(stamp, now):
    with pytest.raises(DeploymentFault):
        require_recent(stamp, now, 1., "image")


def capture_times(**changes):
    values = dict(image=10_100_000_000, info=10_100_000_000, joints=10_110_000_000,
                  now=10_200_000_000, after=10_000_000_000, previous=9_000_000_000,
                  max_age=1., max_skew=.1)
    values.update(changes)
    validate_capture_times(**values)


def test_capture_accepts_new_synchronized_exposure():
    capture_times()


def test_robot_mounted_capture_allows_a_different_live_joint_clock():
    capture_times(joints=3_000_000_000, require_joint_sync=False)


@pytest.mark.parametrize("changes", [
    {"now": 12_000_000_000},
    {"after": 10_100_000_000},
    {"previous": 10_100_000_000},
    {"info": 9_000_000_000},
    {"joints": 10_300_000_000},
    {"image": 0},
])
def test_capture_rejects_stale_reused_pretrigger_or_unsynchronized_frames(changes):
    with pytest.raises(DeploymentFault):
        capture_times(**changes)


def test_base_movement_and_camera_endpoint_mismatch_are_rejected():
    moved = np.eye(4)
    moved[0, 3] = .02
    with pytest.raises(DeploymentFault):
        require_pose_close(moved, np.eye(4), .005, .01, "base")
    require_pose_close(np.eye(4), np.eye(4), .005, .01, "camera")


def test_trajectory_start_checks_names_and_positions():
    trajectory = NS(joint_trajectory=NS(joint_names=["b", "a"], points=[NS(positions=[2., 1.])]))
    validate_trajectory_start(trajectory, ("a", "b"), [1., 2.])
    with pytest.raises(DeploymentFault, match="moved"):
        validate_trajectory_start(trajectory, ("a", "b"), [1.1, 2.])
    with pytest.raises(DeploymentFault, match="exactly"):
        validate_trajectory_start(trajectory, ("a", "c"), [1., 2.])


def test_action_success_returns_terminal_result_and_exact_goal():
    clock = Clock()
    client = Client()
    result = NS(status=4, result=NS(error_code=1))
    client.handle.result.set_result(result)
    session = ActionSession(clock.spin, clock)
    goal = object()
    assert session.run(client, goal, 1.) is result
    assert client.goals == [goal]
    assert session.handle is None
    assert not session.faulted


def test_safety_fault_confirms_cancellation_and_latches_next_goal():
    clock = Clock()
    client = Client()
    session = ActionSession(clock.spin, clock)

    def safety():
        if clock.now > 10.1:
            raise DeploymentFault("protective stop")

    with pytest.raises(DeploymentFault, match="protective stop"):
        session.run(client, object(), 1., safety)
    assert client.handle.cancelled == 1
    assert session.handle is None
    with pytest.raises(DeploymentFault, match="aborted"):
        session.run(client, object(), 1.)
    assert len(client.goals) == 1


def test_timeout_does_not_treat_cancel_ack_as_terminal_stop():
    clock = Clock()
    client = Client(Handle(finish_on_cancel=False))
    session = ActionSession(clock.spin, clock, cancel_timeout=.1)
    with pytest.raises(DeploymentFault, match="TERMINATION UNCONFIRMED"):
        session.run(client, object(), .1)
    assert session.faulted
    assert session.handle is client.handle


def test_late_goal_acceptance_is_cancelled_during_cleanup():
    clock = Clock()
    pending = Future()
    client = Client(future=pending)
    session = ActionSession(clock.spin, clock, cancel_timeout=1.)

    def respond_late():
        if clock.now > 18.2 and not pending.done():
            pending.set_result(client.handle)
    clock.steps.append(respond_late)
    with pytest.raises(DeploymentFault, match="timed out"):
        session.run(client, object(), .1)
    assert client.handle.cancelled == 1
    assert session.handle is None


def test_keyboard_interrupt_cancels_before_propagating():
    clock = Clock()
    client = Client()
    session = ActionSession(clock.spin, clock)

    def interrupt():
        if clock.now > 10.1:
            raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        session.run(client, object(), 1., interrupt)
    assert client.handle.cancelled == 1
    assert session.faulted


def test_sam_adapter_uses_class_agnostic_masks_without_model_download(monkeypatch):
    import sys
    from PIL import Image
    from findvp_ros.segmentation import make_mask_provider
    calls = []

    def pipeline(task, model, device):
        calls.append((task, model, device))
        def generate(image, points_per_batch):
            assert points_per_batch == 4
            return {"masks": [np.ones((image.height, image.width), dtype=bool)]}
        return generate
    monkeypatch.setitem(sys.modules, "torch", NS(cuda=NS(is_available=lambda: False)))
    monkeypatch.setitem(sys.modules, "transformers", NS(pipeline=pipeline))
    provider = make_mask_provider("sam")
    assert calls == [("mask-generation", "facebook/sam-vit-base", -1)]
    assert provider(Image.new("RGB", (5, 4)))[0].shape == (4, 5)
    assert make_mask_provider("flat") is None
    with pytest.raises(ValueError):
        make_mask_provider("unknown")
