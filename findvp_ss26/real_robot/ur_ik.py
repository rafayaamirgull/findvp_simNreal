"""Closed-form UR5e inverse kinematics (all 8 branches), NumPy only.

Standard UR DH chain (A_i = Rz(q_i) Tz(d_i) Tx(a_i) Rx(alpha_i)) with the
nominal UR5e parameters. The ROS ``base_link``/``tool0`` frames differ from
the DH frames by constant transforms; the planner server calibrates those
against CuRobo's own forward kinematics at startup and refuses to use this
solver unless the round trip matches (see ``curobo_server.Planner``).
"""

from __future__ import annotations

import math

import numpy as np

UR5E_DH = {"d1": 0.1625, "a2": -0.425, "a3": -0.3922, "d4": 0.1333, "d5": 0.0997, "d6": 0.0996}
_ALPHA = (math.pi / 2, 0.0, 0.0, math.pi / 2, -math.pi / 2, 0.0)


def _dh(theta, d, a, alpha):
    ct, st, ca, sa = math.cos(theta), math.sin(theta), math.cos(alpha), math.sin(alpha)
    return np.array([[ct, -st * ca, st * sa, a * ct],
                     [st, ct * ca, -ct * sa, a * st],
                     [0.0, sa, ca, d],
                     [0.0, 0.0, 0.0, 1.0]])


def fk_dh(q, p=UR5E_DH) -> np.ndarray:
    d = (p["d1"], 0.0, 0.0, p["d4"], p["d5"], p["d6"])
    a = (0.0, p["a2"], p["a3"], 0.0, 0.0, 0.0)
    T = np.eye(4)
    for i in range(6):
        T = T @ _dh(float(q[i]), d[i], a[i], _ALPHA[i])
    return T


def ik_dh(T, p=UR5E_DH) -> list[np.ndarray]:
    """All real solutions q (6,) with ``fk_dh(q) == T`` (angles in (-pi, pi])."""

    T = np.asarray(T, dtype=float)
    d1, a2, a3, d4, d6 = p["d1"], p["a2"], p["a3"], p["d4"], p["d6"]
    p05 = T[:3, 3] - d6 * T[:3, 2]
    radius = math.hypot(p05[0], p05[1])
    if radius < abs(d4) + 1e-9:
        return []
    psi = math.atan2(p05[1], p05[0])
    offset = math.asin(d4 / radius)
    solutions = []
    for q1 in (psi + offset, psi + math.pi - offset):
        z1 = np.array([math.sin(q1), -math.cos(q1), 0.0])  # joint-2 axis in the base
        c5 = float(np.clip(T[:3, 2] @ z1, -1.0, 1.0))
        for q5 in (math.acos(c5), -math.acos(c5)):
            s5 = math.sin(q5)
            if abs(s5) < 1e-6:
                continue  # wrist singularity: q4/q6 not separable; skip this branch
            T16 = np.linalg.inv(_dh(q1, d1, 0.0, _ALPHA[0])) @ T
            M = T16[:3, :3]
            sin_b = -s5  # R_16 = Rz(q2+q3+q4) Ry(-q5) Rz(q6)
            q234 = math.atan2(M[1, 2] / sin_b, M[0, 2] / sin_b)
            q6 = math.atan2(M[2, 1] / sin_b, -M[2, 0] / sin_b)
            R14 = _dh(q234, 0.0, 0.0, math.pi / 2)[:3, :3]
            p46 = np.array([-d6 * math.sin(q5), d6 * math.cos(q5), p["d5"]])
            p14 = T16[:3, 3] - R14 @ p46
            x, y = p14[0], p14[1]
            c3 = (x * x + y * y - a2 * a2 - a3 * a3) / (2.0 * a2 * a3)
            if abs(c3) > 1.0:
                continue
            for q3 in (math.acos(c3), -math.acos(c3)):
                q2 = math.atan2(y, x) - math.atan2(a3 * math.sin(q3), a2 + a3 * math.cos(q3))
                q4 = q234 - q2 - q3
                q = np.array([q1, q2, q3, q4, q5, q6])
                solutions.append(np.arctan2(np.sin(q), np.cos(q)))
    return solutions


def unwrap_near(q, reference, lower, upper):
    """Shift each joint by 2*pi*k to be nearest ``reference`` within [lower, upper]; None if impossible."""

    out = np.empty(6)
    for i in range(6):
        best = None
        for k in range(-2, 3):
            v = q[i] + 2.0 * math.pi * k
            if lower[i] - 1e-9 <= v <= upper[i] + 1e-9 and (best is None or abs(v - reference[i]) < abs(best - reference[i])):
                best = v
        if best is None:
            return None
        out[i] = best
    return out
