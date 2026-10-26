#!/usr/bin/env python3
"""Collision-aware CuRobo planner for Steve's UR5e, served over localhost HTTP.

Runs inside the ``findvp-curobo:0.7.8`` image (CuRobo + CUDA); the ROS side
(``steve_three_view.py``, dev container) never imports CuRobo.  Both share
``--net=host``, so the runner reaches this server at ``http://127.0.0.1:8431``.
This mirrors the friend's verified split (CuRobo process <-> JSON <-> rclpy
process) but keeps MotionGen warm between the four moves of a run.

Planning model = the friend's verified choice: CuRobo's stock gripper-less
``ur5e.yml`` rooted at ``ur5ebase_link`` with end effector ``tool0``,
PRIMITIVE collision checker, 0.02 s interpolation.  In addition, the wrist
D415 and its bracket are modelled as spheres on ``tool0`` (from scene.json).

Endpoints (JSON in / JSON out):
  GET  /health
  POST /fk    {"positions": [[6 joints], ...]}
  POST /plan  {"start": [6], "goal": {"joints": [6]} | {"position": [3], "quaternion_xyzw": [4]},
               "world": [cuboids from geometry.world_cuboids], "max_attempts": 10}

It never talks to the robot.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geometry import CANONICAL_JOINTS, DEFAULT_SCENE, load_scene, pose_to_matrix, world_cuboids  # noqa: E402
from ur_ik import fk_dh, ik_dh, unwrap_near  # noqa: E402

XYZW_TO_WXYZ = [3, 0, 1, 2]
# Moving the big joints sweeps the whole arm, so they count double when choosing an IK branch.
IK_DISTANCE_WEIGHTS = [2.0, 2.0, 2.0, 1.0, 1.0, 1.0]


class Planner:
    def __init__(self, scene: dict, camera_spheres: bool = True):
        import torch
        from curobo.geom.sdf.world import CollisionCheckerType
        from curobo.types.base import TensorDeviceType
        from curobo.util_file import get_robot_configs_path, join_path, load_yaml
        from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

        self.torch = torch
        self.tensor_args = TensorDeviceType()
        robot_cfg = load_yaml(join_path(get_robot_configs_path(), "ur5e.yml"))["robot_cfg"]
        self.camera_spheres = []
        if camera_spheres:
            self.camera_spheres = copy.deepcopy(scene.get("camera_collision_spheres_tool0", []))
            if self.camera_spheres:
                # tool0 is already a collision link in ur5e.yml (with a disabled
                # placeholder sphere); give it the camera + bracket volume.
                robot_cfg["kinematics"]["collision_spheres"]["tool0"] = [
                    {"center": [float(c) for c in s["center"]], "radius": float(s["radius"])}
                    for s in self.camera_spheres
                ]
        self.world_key = None
        world = self._world_config(world_cuboids(scene))
        config = MotionGenConfig.load_from_robot_config(
            robot_cfg,
            world,
            self.tensor_args,
            collision_checker_type=CollisionCheckerType.PRIMITIVE,  # MESH crashes (warp interop), friend's note
            collision_cache={"obb": 32},
            interpolation_dt=0.02,
        )
        self.motion_gen = MotionGen(config)
        print("warming up CuRobo MotionGen (one-time, ~1 min on first start) ...", flush=True)
        self.motion_gen.warmup(enable_graph=True)
        self.world_key = json.dumps(world_cuboids(scene), sort_keys=True)
        limits = self.motion_gen.kinematics.get_joint_limits().position.cpu().numpy()
        self.lower, self.upper = limits[0], limits[1]
        self._calibrate_analytic_ik()
        print("planner ready", flush=True)

    def _calibrate_analytic_ik(self):
        """Find the constant DH<->URDF frame offsets and prove them against CuRobo FK."""

        self.analytic = None
        rng = np.random.default_rng(0)
        samples = rng.uniform(-np.pi, np.pi, (20, 6))
        curobo = [pose_to_matrix(f["position"], f["quaternion_xyzw"]) for f in self.fk(samples.tolist())]
        for name, C0 in (("Rz(pi)", np.diag([-1.0, -1.0, 1.0, 1.0])), ("identity", np.eye(4))):
            C6 = np.linalg.inv(C0 @ fk_dh(samples[0])) @ curobo[0]
            error = max(np.abs(C0 @ fk_dh(q) @ C6 - T).max() for q, T in zip(samples, curobo))
            if error < 1e-4:
                self.analytic = (C0, C6)
                print(f"analytic UR5e IK verified against CuRobo FK (base offset {name}, max error {error:.1e})",
                      flush=True)
                return
        print("! analytic IK does not match CuRobo's model; falling back to CuRobo's numerical IK", flush=True)

    def _world_config(self, cuboids):
        from curobo.geom.types import Cuboid, WorldConfig

        return WorldConfig(cuboid=[Cuboid(name=c["name"], pose=c["pose"], dims=c["dims"]) for c in cuboids])

    def _set_world(self, cuboids):
        key = json.dumps(cuboids, sort_keys=True)
        if key != self.world_key:
            self.motion_gen.update_world(self._world_config(cuboids))
            self.world_key = key
            print("world updated:", [c["name"] for c in cuboids], flush=True)

    def _joint_state(self, positions):
        from curobo.types.robot import JointState

        return JointState.from_position(
            self.tensor_args.to_device([[float(v) for v in positions]]), joint_names=CANONICAL_JOINTS
        )

    def fk(self, positions):
        q = self.tensor_args.to_device([[float(v) for v in row] for row in positions])
        state = self.motion_gen.kinematics.get_state(q)
        p = state.ee_position.cpu().numpy()
        q_wxyz = state.ee_quaternion.cpu().numpy()
        return [
            {"position": p[i].tolist(), "quaternion_xyzw": [*q_wxyz[i, 1:].tolist(), float(q_wxyz[i, 0])]}
            for i in range(len(positions))
        ]

    def check_state(self, positions):
        """Collision/limit validity of one configuration in the current world."""

        valid, status = self.motion_gen.check_start_state(self._joint_state(positions))
        return bool(valid), str(status)

    def plan(self, request):
        from curobo.types.math import Pose
        from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig

        start = request["start"]
        goal = request["goal"]
        if len(start) != 6:
            raise ValueError("start must contain six joint positions")
        if "world" in request:
            self._set_world(request["world"])
        valid, status = self.check_state(start)
        if not valid:
            return {"success": False, "status": f"start state invalid: {status}"}
        plan_config = MotionGenPlanConfig(max_attempts=int(request.get("max_attempts", 10)), timeout=10.0)
        q0 = self._joint_state(start)
        t0 = time.time()
        ik_info = {}
        if "joints" in goal:
            goal_joints = [float(v) for v in goal["joints"]]
        else:
            # Do NOT use plan_single(pose): it accepts any of the UR5e's IK branches and on
            # Steve produced elbow flips and 350 deg wrist_3 turns. Instead take the
            # collision-free IK solution nearest to the live joints (analytic, all 8
            # branches, deterministic) and plan in joint space to it.
            q = [float(v) for v in goal["quaternion_xyzw"]]
            pose = Pose(
                self.tensor_args.to_device([[float(v) for v in goal["position"]]]),
                self.tensor_args.to_device([[q[i] for i in XYZW_TO_WXYZ]]),
            )
            goal_joints, ik_info = self.nearest_ik(pose, start, pose_to_matrix(goal["position"], q))
            if goal_joints is None:
                return {"success": False, "status": "IK failed: view unreachable or in collision",
                        "planning_time_s": time.time() - t0}
        result = self.motion_gen.plan_single_js(q0, self._joint_state(goal_joints), plan_config)
        elapsed = time.time() - t0
        success = result.success is not None and bool(result.success.view(-1)[0].item())
        if not success:
            return {"success": False, "status": str(result.status), "planning_time_s": elapsed, **ik_info}
        positions = result.get_interpolated_plan().position.cpu().numpy()
        dt = float(result.interpolation_dt)
        stride = max(1, len(positions) // 40)
        path_rows = positions[::stride].tolist() + [positions[-1].tolist()]
        return {
            "success": True,
            "status": str(result.status),
            "planning_time_s": elapsed,
            "joint_names": CANONICAL_JOINTS,
            "positions": positions.tolist(),
            "time_from_start": [k * dt for k in range(len(positions))],
            "tool0_path": [row["position"] for row in self.fk(path_rows)],
            "goal_tool0": self.fk([positions[-1].tolist()])[0],
            **ik_info,
        }

    def nearest_ik(self, pose, start, T_goal=None):
        """Collision-free IK solution closest to ``start`` (weighted joint distance).

        Analytic first: every UR5e branch, each joint unwrapped by 2*pi towards the
        live pose, deterministic. CuRobo's seeded numerical IK only as a fallback.
        """

        if self.analytic is not None and T_goal is not None:
            C0, C6 = self.analytic
            reference = np.asarray(start, dtype=float)
            candidates = []
            for q in ik_dh(np.linalg.inv(C0) @ T_goal @ np.linalg.inv(C6)):
                q = unwrap_near(q, reference, self.lower, self.upper)
                if q is not None:
                    candidates.append((float((np.abs(q - reference) * IK_DISTANCE_WEIGHTS).sum()), q))
            candidates.sort(key=lambda item: item[0])
            for distance, q in candidates:
                if self.check_state(q.tolist())[0]:
                    return q.tolist(), {"ik": "analytic", "ik_solutions": len(candidates),
                                        "ik_weighted_distance_rad": distance}
            return None, {"ik": "analytic", "ik_solutions": len(candidates)}

        solver = self.motion_gen.ik_solver
        retract = self.tensor_args.to_device([[float(v) for v in start]])
        result = solver.solve_single(pose, retract_config=retract, return_seeds=solver.num_seeds)
        solutions = result.solution.view(-1, 6).cpu().numpy()
        success = result.success.view(-1).cpu().numpy().astype(bool)
        if not success.any():
            return None, {}
        distance = (np.abs(solutions - np.asarray(start, dtype=float)) * IK_DISTANCE_WEIGHTS).sum(axis=1)
        distance[~success] = float("inf")
        best = int(distance.argmin())
        return solutions[best].tolist(), {"ik": "curobo", "ik_solutions": int(success.sum()),
                                          "ik_weighted_distance_rad": float(distance[best])}


def make_handler(planner: Planner, scene_path: str):
    class Handler(BaseHTTPRequestHandler):
        def _reply(self, code, document):
            body = json.dumps(document).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path == "/health":
                self._reply(
                    200,
                    {
                        "status": "ok",
                        "robot": "ur5e.yml (CuRobo stock, gripper-less)",
                        "scene_file": scene_path,
                        "camera_collision_spheres_tool0": planner.camera_spheres,
                    },
                )
            else:
                self._reply(404, {"error": "unknown path"})

        def do_POST(self):  # noqa: N802
            try:
                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length) or b"{}")
                if self.path == "/plan":
                    self._reply(200, planner.plan(request))
                elif self.path == "/fk":
                    self._reply(200, {"poses": planner.fk(request["positions"])})
                elif self.path == "/check":
                    valid, status = planner.check_state(request["positions"])
                    self._reply(200, {"valid": valid, "status": status})
                else:
                    self._reply(404, {"error": "unknown path"})
            except Exception as exc:  # report, keep serving
                traceback.print_exc()
                self._reply(500, {"success": False, "status": f"{type(exc).__name__}: {exc}"})

        def log_message(self, fmt, *args):
            print(f"[{time.strftime('%H:%M:%S')}] {self.command} {self.path}", flush=True)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", default=str(DEFAULT_SCENE))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8431)
    parser.add_argument(
        "--no-camera-spheres",
        action="store_true",
        help="plan with the bare arm like the friend's pipeline (camera body not modelled)",
    )
    args = parser.parse_args()
    scene = load_scene(args.scene)
    planner = Planner(scene, camera_spheres=not args.no_camera_spheres)
    server = HTTPServer((args.host, args.port), make_handler(planner, args.scene))
    print(f"CuRobo planner listening on http://{args.host}:{args.port}  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
