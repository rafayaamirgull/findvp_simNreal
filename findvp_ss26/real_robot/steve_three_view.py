#!/usr/bin/env python3
"""FindVP on the real Steve robot: the three level-0 triangle views, SIFT+DINOv3, return to the best.

    check    read-only: joints, safety, camera, TF, planner-vs-robot agreement, box overlay
    overlay  read-only: live image with the projected carton outline on /findvp/three_view/overlay
    locate   read-only: click the carton's two bottom front corners -> its centre/yaw into scene.json
    goto     move the camera to (azimuth, elevation, radius) around the carton  [moves, asks 'go']
    target   save the current camera image as the target (no motion)
    run      visit triangles 1-3, capture, score, return to the best             [moves, asks 'go']
    score    re-score a finished run offline (no ROS, no robot)

Motion path (friend's verified chain, no MoveIt):
    live /joint_states -> CuRobo server (findvp-curobo image) -> one FollowJointTrajectory goal
    -> scaled_joint_trajectory_controller, slowed x8, each move approved by typing 'go'.

Run with the ROS Humble system Python inside the dev container, after
``source /home/ws/steve_env.sh`` and ``source /home/ws/.venv-findvp/bin/activate``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

REAL_ROBOT_DIR = Path(__file__).resolve().parent
FINDVP_ROOT = REAL_ROBOT_DIR.parent
sys.path.insert(0, str(FINDVP_ROOT))
sys.path.insert(0, str(REAL_ROBOT_DIR))

from geometry import (  # noqa: E402
    CAMERA_FRAME,
    CANONICAL_JOINTS,
    DEFAULT_HANDEYE,
    DEFAULT_SCENE,
    TOOL0_FRAME,
    box_mask,
    camera_to_tool0,
    direction_angle_deg,
    load_handeye,
    load_json,
    load_scene,
    locate_box_from_bottom_edge,
    matrix_to_pose,
    pixel_to_plane,
    pose_to_matrix,
    rotation_angle_deg,
    scale_view,
    spherical_view,
    triangle_views,
    view_angles,
    world_cuboids,
    write_json,
)

DEFAULT_PLANNER = "http://127.0.0.1:8431"


# =========================================================================== planner client


class PlannerClient:
    def __init__(self, url: str):
        self.url = url.rstrip("/")

    def _call(self, path, payload=None, timeout=120.0):
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(self.url + path, data, {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            return json.loads(exc.read() or b"{}") or {"success": False, "status": str(exc)}
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            raise RuntimeError(
                f"CuRobo planner not reachable at {self.url} ({exc}). Start it on the HOST with:\n"
                "  bash findvp_ss26/real_robot/start_planner.sh") from exc

    def health(self):
        return self._call("/health", timeout=5.0)

    def fk_tool0(self, joints) -> np.ndarray:
        pose = self._call("/fk", {"positions": [list(map(float, joints))]})["poses"][0]
        return pose_to_matrix(pose["position"], pose["quaternion_xyzw"])

    def check(self, joints):
        return self._call("/check", {"positions": list(map(float, joints))})

    def plan_pose(self, start, T_base_tool0, world):
        t, q = matrix_to_pose(T_base_tool0)
        return self._call("/plan", {"start": list(map(float, start)),
                                    "goal": {"position": t, "quaternion_xyzw": q}, "world": world})

    def plan_joints(self, start, goal, world):
        return self._call("/plan", {"start": list(map(float, start)),
                                    "goal": {"joints": list(map(float, goal))}, "world": world})


# =========================================================================== helpers


class Context:
    """Everything a sub-command needs: scene, hand-eye, planner, and (lazily) ROS."""

    def __init__(self, args):
        self.args = args
        center = getattr(args, "center", None)
        self.scene = load_scene(args.scene, center=center, yaw_deg=getattr(args, "yaw", None))
        self.obj = self.scene["object"]
        self.center = np.asarray(self.obj["center_xyz_m"], dtype=float)
        self.T_tool0_camera = load_handeye(args.handeye)
        self.world = world_cuboids(self.scene)
        self.planner = PlannerClient(args.planner_url)
        self.robot = None

    def connect(self, check_safety=True):
        import rclpy

        from ros_robot import SteveRobot

        rclpy.init()
        self.robot = SteveRobot(controller=self.args.controller, check_safety=check_safety)
        self.robot.spin_for(1.0)  # discovery + latched topics
        return self.robot

    def close(self):
        if self.robot is not None:
            import rclpy

            self.robot.destroy_node()
            rclpy.try_shutdown()


def banner(text):
    print("\n" + "=" * 78 + f"\n{text}\n" + "=" * 78)


def fmt(v, digits=3):
    return "[" + ", ".join(f"{float(x):+.{digits}f}" for x in v) + "]"


def ask_go(prompt: str) -> bool:
    """Operator interlock: a human must type 'go' on a real terminal."""

    if not sys.stdin.isatty():
        print("! refusing: 'go' must be typed by the operator on a terminal (stdin is not a TTY)")
        return False
    try:
        return input(prompt).strip() == "go"
    except EOFError:
        return False


def plan_camera_view(ctx: Context, start, T_base_camera, radii):
    """Plan to the view at the first reachable radius. Returns (plan, T_base_camera) or (last failure, None)."""

    last = None
    for radius in radii:
        T = scale_view(T_base_camera, ctx.center, radius)
        plan = ctx.planner.plan_pose(start, camera_to_tool0(T, ctx.T_tool0_camera), ctx.world)
        if plan.get("success"):
            plan["radius_m"] = radius
            return plan, T
        print(f"    radius {radius:.2f} m: planning failed ({plan.get('status')})")
        last = plan
    return last, None


def review(plan, start, slow_factor, label):
    """Print what the operator is about to approve."""

    positions = np.asarray(plan["positions"])
    delta = np.degrees(positions[-1] - np.asarray(start))
    excursion = np.degrees(np.ptp(positions, axis=0))
    duration = plan["time_from_start"][-1]
    print(f"  {label}")
    print(f"  {len(positions)} waypoints, planned {duration:.2f} s -> executed {duration * slow_factor:.1f} s "
          f"(slow factor {slow_factor:g})")
    names = ["pan", "lift", "elbow", "wrist1", "wrist2", "wrist3"]
    print("  joint change (deg): " + "  ".join(f"{n} {d:+.0f}" for n, d in zip(names, delta)))
    print("  path range   (deg): " + "  ".join(f"{n} {d:.0f}" for n, d in zip(names, excursion)))
    if float(np.max(excursion[:3])) > 120.0 or float(np.max(excursion)) > 180.0:
        print("  ! LARGE joint motion (see path range) - watch the whole arm sweep and the camera cable;"
              " type anything but 'go' to abort and move the arm closer by hand")


def move(ctx: Context, plan, start, label) -> bool:
    review(plan, start, ctx.args.slow_factor, label)
    ctx.robot.publish_markers(ctx.scene, ctx.marker_views, target=ctx.marker_target, path=plan.get("tool0_path"))
    if not ask_go("  Watch RViz + the arm. Type 'go' to execute on the REAL arm (anything else aborts): "):
        print("  aborted by operator; the arm was not moved.")
        return False
    ok, why = ctx.robot.execute(plan, ctx.args.slow_factor)
    print(("  done: " if ok else "  ! FAILED: ") + why)
    return ok


def save_capture(ctx: Context, capture, directory: Path, stem: str, caption: str) -> dict:
    from PIL import Image

    from ros_robot import draw_outline

    directory.mkdir(parents=True, exist_ok=True)
    image, T, K = capture["image"], capture["T_base_camera"], capture["K"]
    mask = box_mask(ctx.obj, T, K, image.width, image.height)
    image.save(directory / f"{stem}.png")
    Image.fromarray(mask.astype(np.uint8) * 255).save(directory / f"{stem}_mask.png")
    draw_outline(image, ctx.obj, T, K, caption).save(directory / f"{stem}_overlay.png")
    t, q = matrix_to_pose(T)
    record = {
        "image_filename": f"{stem}.png",
        "mask": f"{stem}_mask.png",
        "overlay": f"{stem}_overlay.png",
        "mask_fraction": float(mask.mean()),
        "camera_position_m": t,
        "camera_quaternion_xyzw": q,
        "joint_names": CANONICAL_JOINTS,
        "joints_rad": [float(v) for v in capture["joints"]],
        "K": np.asarray(K).tolist(),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **view_angles(t, ctx.center),
    }
    if mask.mean() < 0.002:
        print(f"  ! the carton is (almost) outside image {stem}.png - check the object centre in scene.json")
    return record


def containing_triangle(azimuth_deg, offset_deg):
    return int(((azimuth_deg - offset_deg) % 360.0) // 120.0) + 1


# =========================================================================== commands


def cmd_check(ctx: Context) -> int:
    banner("CHECK (read-only: nothing moves)")
    ok_all = True
    try:
        health = ctx.planner.health()
        print(f"planner      OK  {ctx.planner.url}  camera spheres: {len(health['camera_collision_spheres_tool0'])}")
    except RuntimeError as exc:
        print(f"planner      !!  {exc}")
        health, ok_all = None, False

    robot = ctx.connect()
    q = robot.joints()
    print("joints (rad) " + fmt(q, 4))
    print("joints (deg) " + fmt(np.degrees(q), 1) + "   <- compare with the pendant's Move tab")
    mode_ok, why = robot.safety_ok()
    print(f"robot state  {'OK' if mode_ok else '!!'}  {why}")
    ctl = robot.controller_available(3.0)
    state = robot.controller_state() if ctl else None
    ctl_ok = ctl and state in ("active", None)
    detail = "no action server" if not ctl else (
        "state unknown (controller_manager did not answer)" if state is None else f"state '{state}'")
    print(f"controller   {'OK' if ctl_ok else '!!'}  {robot.controller.strip('/')}: {detail}")
    if ctl and state not in ("active", None):
        print("             !! not active: stop + play External Control on the pendant")
    ok_all &= mode_ok and ctl_ok

    T_tool0 = robot.lookup(TOOL0_FRAME)
    T_cam = robot.lookup(CAMERA_FRAME)
    expected = T_tool0 @ ctx.T_tool0_camera
    err_mm = 1000 * np.linalg.norm(expected[:3, 3] - T_cam[:3, 3])
    print(f"hand-eye TF  {'OK' if err_mm < 2 else '!!'}  camera TF vs tool0*handeye_result.json: {err_mm:.1f} mm")
    ok_all &= err_mm < 2
    if health is not None:
        T_fk = ctx.planner.fk_tool0(q)
        pos_mm = 1000 * np.linalg.norm(T_fk[:3, 3] - T_tool0[:3, 3])
        rot = rotation_angle_deg(T_fk[:3, :3], T_tool0[:3, :3])
        agree = pos_mm < 5 and rot < 1
        print(f"planner FK   {'OK' if agree else '!!'}  CuRobo tool0 vs robot TF tool0: {pos_mm:.1f} mm, {rot:.2f} deg")
        if not agree:
            print("             !! the planner model does NOT match the real arm: do not move until this is resolved")
        check = ctx.planner.check(q)
        print(f"start state  {'OK' if check['valid'] else '!!'}  "
              + ("collision-free in the planning world" if check["valid"] else f"IN COLLISION: {check['status']}"))
        ok_all &= agree and check["valid"]

    capture = robot.capture(CAMERA_FRAME)
    out = Path(ctx.args.out_dir or f"/home/ws/findvp_runs/check_{time.strftime('%Y%m%d_%H%M%S')}")
    record = save_capture(ctx, capture, out, "check", "green = carton from scene.json projected with the live TF")
    print(f"camera       OK  image {capture['image'].size}, carton covers {100 * record['mask_fraction']:.1f} % of it")
    print(f"             overlay: {out / 'check_overlay.png'}  (the green outline must sit on the real carton)")
    angles = view_angles(record["camera_position_m"], ctx.center)
    print(f"camera now   az {angles['azimuth_deg']:.0f} deg  el {angles['elevation_deg']:.0f} deg  "
          f"r {angles['radius_m']:.3f} m from the carton centre")

    print("\ntriangle views (ur5ebase_link):")
    for view in triangle_views(ctx.scene, ctx.args.radius, ctx.args.azimuth_offset):
        print(f"  T{view['triangle']}: camera {fmt(view['T_base_camera'][:3, 3])}  az {view['azimuth_deg']:.0f}  "
              f"el {view['elevation_deg']:.0f}  r {view['radius_m']:.2f}")
    print("\nRESULT: " + ("all checks passed" if ok_all else "!! fix the items marked !! before moving the arm"))
    return 0 if ok_all else 1


def cmd_overlay(ctx: Context) -> int:
    from ros_robot import draw_outline

    robot = ctx.connect(check_safety=False)
    ctx.marker_views = []
    views = triangle_views(ctx.scene, ctx.args.radius, ctx.args.azimuth_offset)
    robot.publish_markers(ctx.scene, [{"label": f"T{v['triangle']}", "T_base_camera": v["T_base_camera"]}
                                      for v in views])
    print("publishing /findvp/three_view/overlay at ~2 Hz (view it with rqt_image_view); Ctrl+C to stop")
    try:
        while True:
            image = robot.latest_image(newer_than=time.monotonic() - 0.2)
            T = robot.lookup(CAMERA_FRAME)
            K, _, _ = robot.intrinsics()
            a = view_angles(T[:3, 3], ctx.center)
            text = (f"carton outline from scene.json | camera az {a['azimuth_deg']:.0f} el {a['elevation_deg']:.0f} "
                    f"r {a['radius_m']:.2f} m")
            robot.publish_overlay(draw_outline(image, ctx.obj, T, K, text))
            robot.spin_for(0.5)
    except KeyboardInterrupt:
        return 0


def cmd_goto(ctx: Context) -> int:
    args = ctx.args
    T_goal = spherical_view(ctx.center, args.radius_goto, args.azimuth, args.elevation)
    banner(f"GOTO camera az {args.azimuth:.0f} deg, el {args.elevation:.0f} deg, r {args.radius_goto:.2f} m")
    robot = ctx.connect()
    ok, why = robot.motion_ready()
    if not ok:
        print(f"! {why}")
        return 1
    ctx.marker_views = [{"label": "goto", "T_base_camera": T_goal}]
    ctx.marker_target = None
    start = robot.joints()
    plan, T = plan_camera_view(ctx, start, T_goal, [args.radius_goto])
    if T is None:
        print("! no collision-free plan to that view; pick another azimuth/elevation/radius")
        return 3
    if not move(ctx, plan, start, "move camera to the requested view"):
        return 1
    T_now = robot.lookup(CAMERA_FRAME)
    print(f"camera error vs request: {1000 * np.linalg.norm(T_now[:3, 3] - T[:3, 3]):.1f} mm, "
          f"{rotation_angle_deg(T_now[:3, :3], T[:3, :3]):.2f} deg")
    return 0


def cmd_target(ctx: Context) -> int:
    run_dir = Path(ctx.args.run_dir)
    if (run_dir / "target.png").exists():
        print(f"! {run_dir / 'target.png'} already exists; use a new --run-dir for a new target")
        return 1
    banner("TARGET capture (no motion)")
    robot = ctx.connect(check_safety=False)
    capture = robot.capture(CAMERA_FRAME)
    record = save_capture(ctx, capture, run_dir, "target", "TARGET (green = carton mask used for matching)")
    record["scene"] = ctx.scene
    write_json(run_dir / "target.json", record)
    triangle = containing_triangle(record["azimuth_deg"], ctx.args.azimuth_offset
                                   if ctx.args.azimuth_offset is not None
                                   else ctx.scene["search"].get("azimuth_offset_deg", 0.0))
    print(f"saved {run_dir / 'target.png'}  (+ target_mask.png, target_overlay.png, target.json)")
    print(f"target camera: az {record['azimuth_deg']:.0f} deg, el {record['elevation_deg']:.0f} deg, "
          f"r {record['radius_m']:.3f} m  -> lies in triangle T{triangle}")
    print("Open target_overlay.png: the green outline must cover the carton. Then move the arm away (optional) "
          "and start 'run'.")
    return 0


def load_extractor():
    print("loading DINOv3 (CPU) ...", flush=True)
    from sample.dinov3_feature_matching import DINOv3FeatureExtractor

    return DINOv3FeatureExtractor(max_image_size=640, upscale_small_images=True)


def prepare_target(run_dir: Path, extractor, target_image=None, target_mask=None):
    from PIL import Image

    from sample.target_guided_similarity import prepare_target_image

    image_path = Path(target_image or run_dir / "target.png").resolve()
    mask_path = Path(target_mask or run_dir / "target_mask.png")
    if not image_path.is_file() or not mask_path.is_file():
        raise FileNotFoundError(f"need {image_path} and {mask_path}; run the 'target' command first")
    with Image.open(mask_path) as m:
        mask = (np.asarray(m.convert("L")) > 0).astype(np.uint8) * 255
    return image_path, prepare_target_image(image_path, extractor=extractor, target_mask=mask)


def score_views(run_dir: Path, records: list[dict], target_path: Path, prepared, extractor, debug=False) -> dict:
    from sample.target_guided_similarity import rank_triangle_views

    views_dir = run_dir / "views"
    viewpoints_json = views_dir / "viewpoints.json"
    write_json(viewpoints_json, records)
    report = rank_triangle_views(
        target_path, viewpoints_json, output_path=run_dir / "ranking.json",
        prepared_target=prepared, extractor=extractor, debug=debug,
    )
    return report


def print_ranking(report, records, target_record, offset_deg):
    by_triangle = {r["triangle"]: r for r in records}
    print(f"\n{'rank':>4} {'view':>5} {'score':>7} {'DINOv3':>7} {'SIFT':>6} {'inliers':>7}   camera az/el/r"
          + ("   angle to target" if target_record else ""))
    for rank, item in enumerate(report["ranking"], 1):
        r = by_triangle[item["triangle"]]
        line = (f"{rank:>4} {'T' + str(item['triangle']):>5} {item['score']:7.3f} {item['dino_score']:7.3f} "
                f"{item['sift_score']:6.3f} {item['sift_inliers']:7d}   "
                f"{r['azimuth_deg']:4.0f} / {r['elevation_deg']:3.0f} / {r['radius_m']:.2f}")
        if target_record:
            angle = direction_angle_deg(r["camera_position_m"], target_record["camera_position_m"],
                                        target_record["scene"]["object"]["center_xyz_m"])
            line += f"   {angle:5.1f} deg"
        print(line)
    for failure in report["failures"]:
        print(f"   T{failure['triangle']}: could not be scored ({failure['error']})")
    print(f"\nHIGHEST SIMILARITY: T{report['best_triangle']}   (image {report['best_image']})")
    if len(report["plausible_triangles"]) > 1:
        print(f"  note: scores within the ambiguity margin: {report['plausible_triangles']}")
    if target_record:
        expected = containing_triangle(target_record["azimuth_deg"], offset_deg)
        verdict = "matches" if expected == report["best_triangle"] else "DIFFERS from"
        print(f"  ground truth: the target camera lies in triangle T{expected} -> the winner {verdict} it")


def save_summary(run_dir: Path, target_path: Path, records, report):
    from PIL import Image, ImageDraw

    scores = {item["triangle"]: item for item in report["ranking"]}
    tiles = [("TARGET", Image.open(run_dir / "target_overlay.png") if (run_dir / "target_overlay.png").exists()
              else Image.open(target_path), (255, 0, 255))]
    for r in records:
        item = scores.get(r["triangle"])
        best = item is not None and r["triangle"] == report["best_triangle"]
        text = f"T{r['triangle']}  score {item['score']:.3f}" if item else f"T{r['triangle']}  not scored"
        tiles.append((text + ("  BEST" if best else ""), Image.open(run_dir / "views" / r["overlay"]),
                      (255, 200, 0) if best else (80, 80, 80)))
    w = 480
    h = int(tiles[0][1].height * w / tiles[0][1].width)
    canvas = Image.new("RGB", (w * len(tiles), h + 30), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    for i, (text, image, color) in enumerate(tiles):
        canvas.paste(image.convert("RGB").resize((w, h)), (i * w, 30))
        draw.rectangle([i * w, 30, i * w + w - 1, 30 + h - 1], outline=color, width=4)
        draw.text((i * w + 8, 8), text, fill=(255, 255, 255))
    canvas.save(run_dir / "summary.png")
    return run_dir / "summary.png"


def cmd_run(ctx: Context) -> int:
    args = ctx.args
    run_dir = Path(args.run_dir)
    offset = args.azimuth_offset if args.azimuth_offset is not None else ctx.scene["search"].get("azimuth_offset_deg", 0.0)
    radius = args.radius if args.radius is not None else ctx.scene["search"].get("radius_m", 0.40)
    radii = [radius] + [r for r in ctx.scene["search"].get("fallback_radii_m", []) if r > radius]
    views = triangle_views(ctx.scene, radius, offset)
    target_record = load_json(run_dir / "target.json") if (run_dir / "target.json").exists() else None

    banner(f"THREE-VIEW RUN {'(PLAN ONLY - nothing moves)' if args.plan_only else '(REAL ARM)'}  ->  {run_dir}")
    if not args.plan_only:
        if (run_dir / "views").exists():
            print(f"! {run_dir / 'views'} already exists; use a new --run-dir (or 'score' to re-score it)")
            return 1
        extractor = load_extractor()
        target_path, prepared = prepare_target(run_dir, extractor, args.target_image, args.target_mask)
        print(f"target ready: {target_path} ({len(prepared.object_features.descriptors)} DINOv3 object patches)")

    health = ctx.planner.health()
    print(f"planner: {ctx.planner.url} ({health['robot']})")
    robot = ctx.connect()
    if not args.plan_only:
        ok, why = robot.motion_ready()
        print(f"robot: {why}")
        if not ok:
            print("! robot not ready for motion (see 'check'); nothing was moved")
            return 1
    start = robot.joints()
    check = ctx.planner.check(start)
    if not check["valid"]:
        print(f"! the CURRENT arm pose is in collision in the planning world ({check['status']}).\n"
              "  Move the arm to a clearer pose (or fix scene.json) and retry.")
        return 3

    ctx.marker_views = [{"label": f"T{v['triangle']}", "T_base_camera": v["T_base_camera"], "state": "planned"}
                        for v in views]
    ctx.marker_target = (pose_to_matrix(target_record["camera_position_m"], target_record["camera_quaternion_xyzw"])
                         if target_record else None)
    robot.publish_markers(ctx.scene, ctx.marker_views, target=ctx.marker_target)
    for v in views:
        print(f"T{v['triangle']}: camera {fmt(v['T_base_camera'][:3, 3])}  az {v['azimuth_deg']:.0f}  "
              f"el {v['elevation_deg']:.0f}  r {v['radius_m']:.2f}")

    if args.plan_only:
        q = start
        for i, view in enumerate(views):
            print(f"\nplan {i + 1}/3 -> T{view['triangle']} (from {'the live pose' if i == 0 else 'the previous view'})")
            plan, T = plan_camera_view(ctx, q, view["T_base_camera"], radii)
            if T is None:
                print(f"  ! T{view['triangle']} unreachable at radii {radii}")
                ctx.marker_views[i]["state"] = "failed"
                continue
            review(plan, q, args.slow_factor, f"T{view['triangle']} at radius {plan['radius_m']:.2f} m")
            ctx.marker_views[i]["T_base_camera"] = T
            q = np.asarray(plan["positions"][-1])
            robot.publish_markers(ctx.scene, ctx.marker_views, target=ctx.marker_target, path=plan["tool0_path"])
            robot.spin_for(2.0)
        print("\nplan-only finished: nothing moved. Markers are on /findvp/three_view/markers.")
        return 0

    # ------------------------------------------------------------- visit and capture
    records = []
    for i, view in enumerate(views):
        banner(f"MOVE {i + 1}/4 -> triangle T{view['triangle']}  (az {view['azimuth_deg']:.0f}, "
               f"el {view['elevation_deg']:.0f})")
        q = robot.joints()
        plan, T = plan_camera_view(ctx, q, view["T_base_camera"], radii)
        if T is None:
            print(f"  ! T{view['triangle']} is unreachable at radii {radii}; skipping this view")
            ctx.marker_views[i]["state"] = "failed"
            continue
        ctx.marker_views[i]["T_base_camera"] = T
        if not move(ctx, plan, q, f"T{view['triangle']} at radius {plan['radius_m']:.2f} m"):
            print("run stopped; the arm stays where it is. Captured so far:", [r["triangle"] for r in records])
            return 1
        capture = robot.capture(CAMERA_FRAME)
        T_err = 1000 * np.linalg.norm(capture["T_base_camera"][:3, 3] - T[:3, 3])
        record = save_capture(ctx, capture, run_dir / "views", f"triangle_{view['triangle']}",
                              f"T{view['triangle']}  az {view['azimuth_deg']:.0f} el {view['elevation_deg']:.0f}")
        record.update({"triangle": view["triangle"], "triangle_record": view["triangle_record"],
                       "planned_radius_m": plan["radius_m"], "camera_position_error_mm": T_err})
        records.append(record)
        ctx.marker_views[i]["state"] = "captured"
        print(f"  captured {run_dir / 'views' / record['image_filename']}  (camera {T_err:.1f} mm from the plan)")

    if not records:
        print("! no view could be captured")
        return 3

    # ------------------------------------------------------------- score
    banner("SCORING with SIFT + DINOv3 (FindVP target-guided similarity)")
    report = score_views(run_dir, records, target_path, prepared, extractor, debug=args.debug)
    print_ranking(report, records, target_record, offset)
    summary = save_summary(run_dir, target_path, records, report)
    print(f"side-by-side: {summary}")
    best = next(r for r in records if r["triangle"] == report["best_triangle"])
    for mv in ctx.marker_views:
        if mv["label"] == f"T{best['triangle']}":
            mv["state"] = "best"

    # ------------------------------------------------------------- return to the best view
    banner(f"MOVE 4/4 -> back to the best view T{best['triangle']}")
    q = robot.joints()
    result = {"status": "not_moved"}
    if np.max(np.abs(q - np.asarray(best["joints_rad"]))) < 0.01:
        print("  already at the best view")
        result = {"status": "reached", "note": "already there"}
    else:
        plan = ctx.planner.plan_joints(q, best["joints_rad"], ctx.world)
        if not plan.get("success"):
            print(f"  ! could not plan back to T{best['triangle']}: {plan.get('status')}")
            result = {"status": "plan_failed", "planner_status": plan.get("status")}
        elif move(ctx, plan, q, f"return to the joints recorded at T{best['triangle']}"):
            result = {"status": "reached"}
        else:
            result = {"status": "aborted_or_failed"}
    if result["status"] == "reached":
        capture = robot.capture(CAMERA_FRAME)
        final = save_capture(ctx, capture, run_dir, "final", f"FINAL at best view T{best['triangle']}")
        result["camera_position_error_mm"] = float(1000 * np.linalg.norm(
            np.asarray(final["camera_position_m"]) - np.asarray(best["camera_position_m"])))
        result["final_image"] = str(run_dir / "final.png")
        print(f"  final image {run_dir / 'final.png'} (camera {result['camera_position_error_mm']:.1f} mm "
              "from where the best image was taken)")
    robot.publish_markers(ctx.scene, ctx.marker_views, target=ctx.marker_target)

    write_json(run_dir / "run.json", {
        "status": "complete" if result["status"] == "reached" else "incomplete",
        "best_triangle": report["best_triangle"],
        "ranking": report["ranking"],
        "views": records,
        "return_to_best": result,
        "target": target_record,
        "scene": ctx.scene,
        "slow_factor": args.slow_factor,
    })
    print(f"\nreport: {run_dir / 'run.json'}")
    return 0 if result["status"] == "reached" else 1


def click_two_points(image, title):
    """Tk window: the operator clicks two pixels. Returns [(u, v), (u, v)] or None if cancelled."""

    import tkinter as tk

    from PIL import ImageTk

    root = tk.Tk()
    root.title(title)
    clicks = []
    tk.Label(root, justify="left", font=("TkDefaultFont", 11), text=(
        "Click the two BOTTOM corners of the carton's broad front face, where the carton touches the deck\n"
        "(left corner, then right corner). Enter = accept    r = reset    Esc = cancel")).pack()
    canvas = tk.Canvas(root, width=image.width, height=image.height, cursor="crosshair")
    canvas.pack()
    photo = ImageTk.PhotoImage(image)
    canvas.create_image(0, 0, anchor="nw", image=photo)
    result = {"value": None}

    def on_click(event):
        if len(clicks) < 2:
            clicks.append((float(event.x), float(event.y)))
            r = 5
            canvas.create_oval(event.x - r, event.y - r, event.x + r, event.y + r, outline="red", width=2, tags="pt")
            if len(clicks) == 2:
                canvas.create_line(*clicks[0], *clicks[1], fill="red", width=2, tags="pt")

    def reset(_event=None):
        clicks.clear()
        canvas.delete("pt")

    def accept(_event=None):
        if len(clicks) == 2:
            result["value"] = list(clicks)
            root.destroy()

    canvas.bind("<Button-1>", on_click)
    root.bind("<Return>", accept)
    root.bind("r", reset)
    root.bind("<Escape>", lambda _e: root.destroy())
    root.mainloop()
    return result["value"]


def cmd_locate(ctx: Context) -> int:
    """Measure the carton's centre and yaw from two clicked bottom corners (no motion)."""

    from ros_robot import draw_outline

    banner("LOCATE the carton (no motion)")
    robot = ctx.connect(check_safety=False)
    capture = robot.capture(CAMERA_FRAME)
    image, T, K = capture["image"], capture["T_base_camera"], capture["K"]
    if ctx.args.clicks:
        c = ctx.args.clicks
        clicks = [(c[0], c[1]), (c[2], c[3])]
    else:
        clicks = click_two_points(image, "FindVP: click the carton's two bottom front corners")
        if clicks is None:
            print("cancelled; scene.json unchanged")
            return 1
    dims = np.asarray(ctx.obj["dims_xyz_m"], dtype=float)
    deck_z = float(ctx.obj["center_xyz_m"][2]) - dims[2] / 2.0
    points = [pixel_to_plane(uv, T, K, deck_z) for uv in clicks]
    if any(p is None for p in points):
        print("! a clicked pixel does not look down onto the deck; click the corners where the carton meets the deck")
        return 1
    found = locate_box_from_bottom_edge(points[0], points[1], ctx.obj, T[:3, 3])
    ratio = found["edge_length_m"] / found["expected_edge_m"]
    print(f"clicked edge: {100 * found['edge_length_m']:.1f} cm -> matched to the {found['face']} face "
          f"({100 * found['expected_edge_m']:.1f} cm)")
    if not 0.75 <= ratio <= 1.25:
        print("! the clicked edge length does not match the carton; click exactly the two bottom corners "
              "of one face and try again (scene.json unchanged)")
        return 1

    new_obj = dict(ctx.obj)
    new_obj["center_xyz_m"] = [*found["center_xy_m"], float(ctx.obj["center_xyz_m"][2])]
    new_obj["yaw_deg"] = round(found["yaw_deg"], 2)
    old_c = np.asarray(ctx.obj["center_xyz_m"], dtype=float)
    shift_mm = 1000 * np.linalg.norm(np.asarray(new_obj["center_xyz_m"]) - old_c)
    print(f"old: centre {fmt(old_c, 4)}  yaw {ctx.obj.get('yaw_deg', 0.0):+.1f} deg")
    print(f"new: centre {fmt(new_obj['center_xyz_m'], 4)}  yaw {new_obj['yaw_deg']:+.1f} deg   "
          f"(moved {shift_mm:.0f} mm)")

    out = Path(ctx.args.out_dir or f"/home/ws/findvp_runs/locate_{time.strftime('%Y%m%d_%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    overlay = draw_outline(image, ctx.obj, T, K, "red = old scene.json   green = located carton", color=(255, 0, 0))
    overlay = draw_outline(overlay, new_obj, T, K)
    overlay.save(out / "locate_overlay.png")
    print(f"check {out / 'locate_overlay.png'}: the GREEN outline must now sit on the carton")

    flags = (f"--center {new_obj['center_xyz_m'][0]:.4f} {new_obj['center_xyz_m'][1]:.4f} "
             f"{new_obj['center_xyz_m'][2]:.4f} --yaw {new_obj['yaw_deg']:.2f}")
    if not sys.stdin.isatty():
        print(f"not a terminal: scene.json unchanged. Use these flags instead: {flags}")
        return 0
    if input(f"Write this carton pose into {ctx.args.scene}? [y/N] ").strip().lower() != "y":
        print(f"scene.json unchanged. To use it without saving, add: {flags}")
        return 0
    scene_path = Path(ctx.args.scene)
    raw = load_json(scene_path)
    backup = scene_path.with_name(f"{scene_path.stem}.bak-{time.strftime('%Y%m%d_%H%M%S')}.json")
    backup.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    raw["object"]["center_xyz_m"] = [round(v, 4) for v in new_obj["center_xyz_m"]]
    raw["object"]["yaw_deg"] = new_obj["yaw_deg"]
    write_json(scene_path, raw)
    print(f"saved {scene_path} (backup {backup.name}). Re-run 'check' and look at check_overlay.png.")
    return 0


def cmd_score(ctx: Context) -> int:
    run_dir = Path(ctx.args.run_dir)
    records = load_json(run_dir / "views" / "viewpoints.json")
    target_record = load_json(run_dir / "target.json") if (run_dir / "target.json").exists() else None
    offset = ctx.args.azimuth_offset if ctx.args.azimuth_offset is not None else \
        ctx.scene["search"].get("azimuth_offset_deg", 0.0)
    extractor = load_extractor()
    target_path, prepared = prepare_target(run_dir, extractor, ctx.args.target_image, ctx.args.target_mask)
    report = score_views(run_dir, records, target_path, prepared, extractor, debug=ctx.args.debug)
    print_ranking(report, records, target_record, offset)
    print(f"side-by-side: {save_summary(run_dir, target_path, records, report)}")
    return 0


# =========================================================================== CLI


def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--scene", default=str(DEFAULT_SCENE), help="scene JSON (carton pose, obstacles)")
    common.add_argument("--handeye", default=str(DEFAULT_HANDEYE))
    common.add_argument("--planner-url", default=DEFAULT_PLANNER)
    common.add_argument("--controller", default="/scaled_joint_trajectory_controller")
    common.add_argument("--center", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        help="override the carton centre (ur5ebase_link, m)")
    common.add_argument("--yaw", type=float, help="override the carton yaw (deg)")
    common.add_argument("--radius", type=float, help="triangle view distance (default scene.json, 0.40 m)")
    common.add_argument("--azimuth-offset", type=float, help="rotate the three triangles (deg)")
    common.add_argument("--slow-factor", type=float, default=8.0,
                        help="time-stretch every plan (default 8, like the friend's sender; never below 1)")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("check", parents=[common], help="read-only system check + box overlay")
    p.add_argument("--out-dir")
    sub.add_parser("overlay", parents=[common], help="live overlay topic for placing the carton")
    p = sub.add_parser("locate", parents=[common], help="measure the carton pose by clicking two corners")
    p.add_argument("--out-dir")
    p.add_argument("--clicks", type=float, nargs=4, metavar=("U1", "V1", "U2", "V2"),
                   help="pixel coordinates of the two bottom corners instead of the click window")
    p = sub.add_parser("goto", parents=[common], help="move the camera to a view around the carton")
    p.add_argument("--azimuth", type=float, required=True, help="deg, in ur5ebase_link (0 = +x = Steve's right)")
    p.add_argument("--elevation", type=float, required=True, help="deg above the deck")
    p.add_argument("--radius-goto", type=float, default=0.45, help="camera distance to the carton centre (m)")
    for name, helptext in (("target", "save the current image as the target"),
                           ("run", "three views -> score -> return to best"),
                           ("score", "re-score a finished run offline")):
        p = sub.add_parser(name, parents=[common], help=helptext)
        p.add_argument("--run-dir", required=True)
        if name in ("run", "score"):
            p.add_argument("--target-image", help="default <run-dir>/target.png")
            p.add_argument("--target-mask", help="default <run-dir>/target_mask.png")
            p.add_argument("--debug", action="store_true", help="also save FindVP's match visualizations")
        if name == "run":
            p.add_argument("--plan-only", action="store_true", help="plan and show the three moves, move nothing")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.slow_factor < 1.0:
        raise SystemExit("--slow-factor must be >= 1 (it may only slow the motion down)")
    ctx = Context(args)
    ctx.marker_views, ctx.marker_target = [], None
    commands = {"check": cmd_check, "overlay": cmd_overlay, "locate": cmd_locate, "goto": cmd_goto,
                "target": cmd_target, "run": cmd_run, "score": cmd_score}
    try:
        return commands[args.command](ctx)
    except RuntimeError as exc:
        print(f"\n! {exc}")
        return 2
    finally:
        ctx.close()


if __name__ == "__main__":
    sys.exit(main())
