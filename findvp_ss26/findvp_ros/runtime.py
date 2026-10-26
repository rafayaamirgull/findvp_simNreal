"""ROS-independent, fail-closed guards and action lifecycle used by the adapter.

These checks supplement, and never replace, the robot's safety controller.
"""
from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path

import numpy as np

from .geometry import transform_matrix, validate_transform


class DeploymentFault(RuntimeError):
    """Terminal fault: do not issue another motion goal in this session."""


class PlanningRejected(RuntimeError):
    """A completed planning request found no usable path; candidate may be skipped."""


def require_remote_control(value):
    if value is not True:
        raise DeploymentFault(
            "PolyScope Remote Control is not confirmed. Enable it in Settings > System > "
            "Remote Control AND select Remote Control in the header; then follow Steve's "
            "operator-reviewed dashboard reconnect/play procedure. FindVP never starts the program."
        )


def stamp_ns(message):
    return int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)


def require_recent(stamp, now, max_age, label):
    if stamp <= 0 or stamp > now or now - stamp > int(max_age * 1e9):
        raise DeploymentFault(f"{label} timestamp is missing, future-dated, or stale")


def pose_matrix(pose):
    if hasattr(pose, "position"):
        p, q = pose.position, pose.orientation
    else:
        p, q = pose.translation, pose.rotation
    return transform_matrix((p.x, p.y, p.z), (q.x, q.y, q.z, q.w))


def collision_primitive_center(collision_object, frame_from_header):
    """Return a primitive centre in the requested frame.

    MoveIt may move a CollisionObject into another header frame and place its
    geometry in ``CollisionObject.pose``. ``primitive_poses`` are local to that
    pose, so reading a primitive pose alone can incorrectly report (0, 0, 0).
    """
    if not collision_object.primitive_poses:
        raise ValueError("collision object has no primitive pose")
    return (
        validate_transform(frame_from_header, "frame_from_header")
        @ pose_matrix(collision_object.pose)
        @ pose_matrix(collision_object.primitive_poses[0])
    )[:3, 3]


def require_pose_close(actual, expected, position_tolerance, angle_tolerance, label):
    actual = validate_transform(actual)
    expected = validate_transform(expected)
    translation = float(np.linalg.norm(actual[:3, 3] - expected[:3, 3]))
    cosine = (np.trace(actual[:3, :3].T @ expected[:3, :3]) - 1.) / 2.
    angle = math.acos(float(np.clip(cosine, -1., 1.)))
    if translation > position_tolerance or angle > angle_tolerance:
        raise DeploymentFault(f"{label} changed: {translation:.4f} m, {math.degrees(angle):.2f} deg")


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class StateWatchdog:
    def __init__(self, max_age=1.0, clock=time.monotonic):
        self.max_age = max_age
        self.clock = clock
        self.received = {}

    def update(self, name):
        self.received[name] = self.clock()

    def require(self, *names):
        now = self.clock()
        for name in names:
            received = self.received.get(name)
            if received is None or not 0 <= now - received <= self.max_age:
                raise DeploymentFault(f"{name} stream is missing or stale")


class SettlingWindow:
    """Only distinct, increasing acquisition stamps advance settling."""
    def __init__(self, duration, max_gap=0.25):
        self.duration_ns = int(duration * 1e9)
        self.max_gap_ns = int(max_gap * 1e9)
        self.first = None
        self.last = None

    def update(self, stamp, velocities):
        values = np.asarray(velocities, dtype=float)
        if self.last is not None and stamp <= self.last:
            if stamp < self.last:
                self.first = None
            return False
        stable = values.size > 0 and np.isfinite(values).all() and np.max(np.abs(values)) < .01
        if not stable:
            self.first = None
        elif self.first is None or (self.last is not None and stamp - self.last > self.max_gap_ns):
            self.first = stamp
        self.last = stamp
        return self.first is not None and stamp - self.first >= self.duration_ns


def validate_capture_times(*, image, info, joints, now, after, previous, max_age, max_skew,
                           require_joint_sync=True):
    require_recent(image, now, max_age, "image")
    if image <= max(after, previous):
        raise DeploymentFault("capture requires a new image exposed after the request")
    synchronized = [("CameraInfo", info)]
    if require_joint_sync:
        require_recent(joints, now, max_age, "joints")
        synchronized.append(("JointState", joints))
    for label, value in synchronized:
        if value <= 0 or abs(value - image) > int(max_skew * 1e9):
            raise DeploymentFault(f"{label} is not synchronized with the image")


def validate_trajectory_start(trajectory, joint_names, positions, tolerance=.01):
    names = list(trajectory.joint_trajectory.joint_names)
    points = trajectory.joint_trajectory.points
    if len(names) != len(set(names)) or set(names) != set(joint_names) or not points:
        raise DeploymentFault("planned trajectory must contain exactly the configured arm joints")
    expected = np.asarray(points[0].positions, dtype=float)
    current = dict(zip(joint_names, positions))
    actual = np.asarray([current[name] for name in names], dtype=float)
    if expected.shape != actual.shape or not np.isfinite(expected).all():
        raise DeploymentFault("planned start positions are invalid")
    if not np.isfinite(actual).all() or np.max(np.abs(actual - expected)) > tolerance:
        raise DeploymentFault("robot moved since planning; refusing this trajectory")


class ActionSession:
    """Own one action until terminal result; any interruption latches the session.

    The spin callback pumps the owning ROS executor. A delayed goal response is
    retained and cancelled if it arrives during cleanup. If the server cannot
    confirm termination, the caller MUST report motion as unconfirmed, not stopped.
    """
    def __init__(self, spin, clock=time.monotonic, cancel_timeout=5.0):
        self.spin = spin
        self.clock = clock
        self.cancel_timeout = cancel_timeout
        self.faulted = False
        self.pending = self.handle = self.result = None

    def _wait(self, future, timeout, check=None):
        deadline = self.clock() + timeout
        while not future.done():
            if check is not None:
                check()
            if self.clock() >= deadline:
                raise DeploymentFault("action response timed out")
            self.spin()
        if check is not None:
            check()
        return future.result()

    def cancel(self):
        self.faulted = True
        if self.pending is None and self.handle is None:
            return
        try:
            if self.handle is None:
                self.handle = self._wait(self.pending, self.cancel_timeout)
            if not self.handle.accepted:
                self._clear()
                return
            if self.result is None:
                self.result = self.handle.get_result_async()
            if not self.result.done():
                cancellation = self.handle.cancel_goal_async()
                self._wait(cancellation, self.cancel_timeout)
                self._wait(self.result, self.cancel_timeout)
            terminal = self.result.result()
            if terminal.status not in (4, 5, 6):  # SUCCEEDED, CANCELED, ABORTED
                raise DeploymentFault("action did not reach a terminal state")
            self._clear()
        except Exception as error:
            raise DeploymentFault(
                "MOTION TERMINATION UNCONFIRMED: use the robot's physical stop procedure; "
                "do not restart the search"
            ) from error

    def _clear(self):
        self.pending = self.handle = self.result = None

    def run(self, client, goal, timeout, check=None):
        if self.faulted:
            raise DeploymentFault("action session is aborted; no further goals permitted")
        try:
            if check is not None:
                check()
            self.pending = client.send_goal_async(goal)
            self.handle = self._wait(self.pending, 8., check)
            if not self.handle.accepted:
                self._clear()
                raise DeploymentFault("action goal rejected")
            self.result = self.handle.get_result_async()
            wrapped = self._wait(self.result, timeout, check)
            self._clear()
            return wrapped
        except BaseException:
            self.cancel()
            raise
