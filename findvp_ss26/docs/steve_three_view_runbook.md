# Steve: FindVP three-view real-robot run

This is the current real-robot route. It replaces the MoveIt-based
[known-box runbook](steve_known_box_runbook.md) for the three-view experiment.

**Goal:** capture a reference image of the carton, visit FindVP's three initial
triangle viewpoints, compare the new images with SIFT + DINOv3, and return to
the highest-scoring captured view. With the default scene, the nominal camera
views are at azimuths 60°/180°/300°, elevation 45°, radius 0.40 m from the
carton centre. An unreachable view may be retried at 0.45 or 0.50 m. Check the
printed actual radius for each capture.

**Motion gate:** `check` must pass, its saved box outline must visibly fit the
real carton, the physical arm must agree with RViz's *live* model, and each
trajectory must be inspected before you type `go`. A successful `check` alone
cannot verify physical clearance, cable routing, or the correctness of a box
outline. Keep the pendant/E-stop reachable and use the facility's supervised
robot procedure. Do not run another arm-motion client at the same time.

**How it moves the arm:** the same chain as the friend's verified `full_pipeline`,
with no MoveIt and no local URDF:

```
live /joint_states ─► CuRobo server (findvp-curobo image, stock ur5e.yml in ur5ebase_link)
   ─► one FollowJointTrajectory goal ─► /scaled_joint_trajectory_controller  (×8 slower, you type 'go')
```

Differences from the friend's pipeline (validated in the local tests; inspect
on the real setup before motion):

| Change | Why |
|---|---|
| IK is solved in closed form: all 8 UR5e branches, each joint unwrapped towards the live pose, then collision-checked. The planner chooses the nearest valid joint solution. | Reduces the elbow flips and long wrist turns seen with random IK branches. The planner checks analytic IK against CuRobo FK at startup. |
| The wrist D415 and its bracket are three collision spheres on `tool0` (`scene.json`). | The friend's model had no camera body. |
| Object masks come from **projecting the known carton box** through the live TF and hand-eye, not from SAM. | Deterministic, but only as accurate as the measured carton pose, camera calibration and TF. Visually inspect the overlay before motion. |

Code: [`real_robot/`](../real_robot). Configuration: [`real_robot/scene.json`](../real_robot/scene.json).

---

## 0. Before the session (once, with internet)

Complete the [one-time CuRobo image and ROS 2 environment setup](../../README.md)
first. The planner script expects the local `findvp-curobo:0.7.8` tag; the
setup commands pull and tag the image when available, or build it from source.
The Docker Hub pull name is `rafayaamirgull/findvp-curobo:0.7.8`.
The dev container must contain `/home/ws/.venv-findvp` before running these
steps. In the dev container, confirm DINOv3 loads (it caches the weights):

```bash
source /home/ws/.venv-findvp/bin/activate
cd /home/ws/findvp_ss26
python3 -c 'from sample.dinov3_feature_matching import DINOv3FeatureExtractor; DINOv3FeatureExtractor(max_image_size=640); print("DINOv3 ready")'
```

Nothing needs to be built with colcon: the scripts run directly.

## 1. Terminals

| # | Where | What | Leave running |
|---|---|---|---|
| H | **host** terminal (not the dev container) | CuRobo planner | yes |
| A | dev container | camera + hand-eye TF | yes |
| B | dev container | RViz | yes |
| C | dev container | the run commands | – |

Start every dev-container terminal (A, B, C) with:

```bash
cd /home/ws
source /opt/ros/humble/setup.bash
source /home/ws/steve_env.sh
source /home/ws/.venv-findvp/bin/activate
```

`steve_env.sh` must report `192.168.60.90` reachable and `DOMAIN_ID=74`.
In terminal **C**, choose a fresh output directory for this attempt:

```bash
export FV=/home/ws/findvp_ss26/real_robot/steve_three_view.py
export RUN=/home/ws/findvp_runs/three_$(date +%Y%m%d_%H%M%S)
echo "$RUN"
```

**Terminal H (host)**: the dev container has no `docker`, so this is the only command
that runs outside it. In the VS Code dev-container window press Ctrl+Shift+P →
**Terminal: Create New Integrated Terminal (Local)** (or open an Ubuntu terminal with
Ctrl+Alt+T). From the workspace root on the host, run:

```bash
bash findvp_ss26/real_robot/start_planner.sh
```

Wait for `analytic UR5e IK verified against CuRobo FK` and `CuRobo planner listening on http://127.0.0.1:8431`.
The first start may compile CUDA kernels; restarts reuse the cache.
If the analytic-IK line says `! ... falling back`, stop: the planner model does
not match.

**Terminal A:**

```bash
bash /home/ws/start_camera.sh
```

Wait for `RealSense Node Is Up!`.

**Terminal B:**

```bash
bash /home/ws/findvp_ss26/real_robot/start_rviz.sh
```

RViz draws Steve from the **onboard** `/robot_description` + `/tf`, which is exactly
what the real robot reports. Fixed frame is `base_link` (red X axis = Steve's
front). The initial view looks from behind Steve. You will see the green carton, orange
obstacles, blue arrows for T1–T3, magenta for the target and a cyan line for the pending
path.

## 2. Pendant: verify Remote Control + External Control

On the pendant: **enable and select** Remote Control, power on and initialize,
load `external_control.urp`, reduce the speed slider, and keep the E-stop in
reach. With the arm stationary, inspect the state in terminal C:

```bash
timeout 10 ros2 service call /dashboard_client/raw_request ur_dashboard_msgs/srv/RawRequest "{query: 'is in remote control'}"
timeout 10 ros2 service call /dashboard_client/program_running ur_dashboard_msgs/srv/IsProgramRunning '{}'
```

You need remote control `true` and `program_running=True`. If the correct
program is loaded but not running, start it while the arm is stationary:

```bash
timeout 10 ros2 service call /dashboard_client/play std_srvs/srv/Trigger '{}'
```

Do not `stop`, `quit` or reconnect a running External Control session merely
because you opened a new terminal. If `play` fails, inspect the pendant for a
popup/error and confirm Remote Control is **selected** and the intended program
is loaded. Section 3 reports the controller state. If it remains inactive
despite a running program, use the site's approved controller-recovery
procedure with the arm stationary; do not retry motion until `check` passes.

## 3. Check before motion (no robot motion)

```bash
python3 "$FV" check --out-dir "$RUN/check"
```

The command saves `check.png` and `check_overlay.png` but does not move the arm.
Require `RESULT: all checks passed` **and** inspect the overlay yourself:

* **joints (deg)**: compare these six numbers with the pendant's Move tab. A
  mismatch is a stop condition even if the other checks pass.
* **hand-eye TF**: the static camera TF's *translation* agrees with
  `handeye_result.json` within the check's tolerance. This check does not
  numerically verify the calibration rotation; visually inspect the projected
  outline as well.
* **planner FK**: CuRobo's `tool0`, computed from the live joints, equals the robot's
  reported TF `tool0` (mm / deg). This catches a model/TF discrepancy; it does
  **not** substitute for checking the physical arm and surrounding obstacles.
* **start state**: the current pose is collision-free in the planning world.
* **camera** + `check_overlay.png`: open the PNG. **The green outline must sit on
  the real carton**, including its edges; the printed `camera OK` only confirms
  an image arrived. If the outline misses, correct the scene (next step).

Also compare RViz's live arm pose with the stationary physical robot. The
viewer uses the onboard robot description; an apparently rotated or displaced
arm still requires investigation before motion.

## 4. Match `scene.json` to the physical carton

[`scene.json`](../real_robot/scene.json) contains the **current configured**
carton centre, yaw, dimensions, deck, other obstacles and camera-body collision
proxy. Do not assume its values still match a new setup. Coordinates are in
`ur5ebase_link` (+y = Steve's forward, +x = Steve's right, +z = up).
`box_in_front` models the extra stand/box from the friend's setup: keep it only
if that obstacle is really there, and check the dimensions before motion.

**Easiest: measure it with two clicks** (nothing moves). With the whole carton in view:

```bash
python3 "$FV" locate
```

A window shows the live image. Click the two **bottom** corners of the carton's
broad front face, where it touches the deck: left, then right. Press Enter.
The tool intersects both rays with the configured deck plane, derives a centre
and yaw, checks the clicked edge against the carton dimensions, and saves
`locate_overlay.png` (red = old, green = proposed). It then asks whether to
write the pose into `scene.json`; answering `y` makes a timestamped backup.
Inspect the overlay and measured edge before accepting the edit.
If the window does not open, read the two pixel coordinates from `check/check.png` in any
image viewer and pass them as `--clicks U1 V1 U2 V2`. Then run `check` again.

Alternative live alignment aid (terminal C, Ctrl+C to stop):

```bash
python3 "$FV" overlay
```

View it in another container terminal:

```bash
ros2 run rqt_image_view rqt_image_view /findvp/three_view/overlay
```

With the arm stopped and only when it is safe to reach into the workcell,
reposition the carton to match the outline; otherwise measure and update
`object.center_xyz_m` / `yaw_deg`. For temporary trials, pass the *same*
`--center X Y Z --yaw DEG` overrides to **every** command. Re-run `check` and
inspect the new overlay. Wrong carton coordinates mean wrong collision geometry
and wrong image masks; do not proceed until they agree.

## 5. Take the target image

Show the whole carton from a safe view that differs from the three search
views. `goto` is an **arm-motion command**: it plans, displays the path and asks
for `go` before executing. Use it only after sections 2–4 pass:

```bash
python3 "$FV" goto --azimuth 100 --elevation 30 --radius-goto 0.45
```

Azimuth is measured in `ur5ebase_link` around the carton: 0° = Steve's right (+x),
90° = Steve's forward (+y), 180° = Steve's left. If a view is unreachable, try another.
Alternatively, use the site's supervised hand-guiding/freedrive procedure,
then restore External Control and re-run `check`. Once the arm is still, save
the target (this command itself does **not** move the arm):

```bash
python3 "$FV" target --run-dir "$RUN"
```

This writes `target.png`, `target_mask.png`, `target_overlay.png` and `target.json` (camera pose).
Open `target_overlay.png`. The green outline must cover the actual carton and
exclude as much background as possible. A poor outline biases all three
similarity scores. The reported triangle containing the target camera is a
**geometric reference**, not a guarantee that the image scorer must choose it.
`target` refuses to overwrite an existing `target.png`; choose a new `RUN` for
a different reference image.
Keep the carton, camera mount and `scene.json` unchanged between target capture
and the three search captures. Otherwise use a new target and re-run `check`.

## 6. Preview the three moves (nothing moves)

```bash
python3 "$FV" run --run-dir "$RUN" --plan-only
```

For each move it prints the planned duration, joint changes and joint path
ranges. RViz shows the view arrows and the current planned `tool0` path. Check
**T1, T2 and T3 each have a successful plan**; the current preview may finish
with exit code 0 even when it prints an unreachable view. A `! LARGE joint
motion` warning means one of the first three joints spans over 120° or any
joint spans over 180°. Inspect the *whole* swept path, not just the endpoint,
including wrist-camera bracket and cables. If any move or clearance is
unacceptable, stop and adjust the scene, radius or azimuth offset before a
new preview.

## 7. Run it

```bash
python3 "$FV" run --run-dir "$RUN"
```

Before each actual move (T1, T2, T3 and, if needed, back to the winner), it
prints the plan and asks you to type `go`. Anything else aborts **that move**;
the arm remains at its current position. Piped input is refused. The default
trajectory time-stretch is 8×, and the command waits for the controller result.
After each triangle move it waits for stationary joints, captures a fresh RGB
image and checks that the joints did not move during capture.

**Current limitation:** if planning fails for a triangle, the script prints
`skipping this view` and can rank/return using fewer than three images. The
scorer can also exclude an image it cannot score. Do not call such a result a
three-view comparison: check the counts in section 8 and repeat with a new
run directory after resolving the cause. If a motion or capture fails, do not
blindly rerun in the same directory; inspect the saved images and robot state.

At the end it prints, for example:

```
rank  view   score DINOv3   SIFT inliers   camera az/el/r   angle to target
   1    T2   0.812   0.790  0.878      64    180 /  45 / 0.40    38.1 deg
HIGHEST SIMILARITY: T2
  ground truth: the target camera lies in triangle T2 -> the winner matches it
```

The "ground truth" line is only the target camera's geometric triangle; use
the actual scores and images to judge the match. The command then plans back
to the joints recorded for the best captured image (unless it is already
there), asks for approval, and saves `final.png` after arrival.

## 8. Results

Everything is in `$RUN`:

| File | Content |
|---|---|
| `summary.png` | target + T1/T2/T3 with outlines and scores (best in gold) |
| `views/triangle_N.png`, `_mask.png`, `_overlay.png` | each captured image, the mask used, the outline check |
| `ranking.json` | FindVP's full ranking (DINOv3 semantic/geometric, SIFT inliers, …) |
| `run.json` | captured views, poses, joints, scores and return-to-best result |
| `final.png` | image after returning to the best view |

Verify that this was genuinely a three-image result. `complete` alone says
the return-to-best step succeeded; it does not verify the capture count:

```bash
python3 -c 'import json,os; r=json.load(open(os.path.join(os.environ["RUN"],"run.json"))); print("status:",r["status"],"captured:",len(r["views"]),"scored:",len(r["ranking"]),"winner: T"+str(r["best_triangle"]),"return:",r["return_to_best"]["status"]); assert r["status"]=="complete" and len(r["views"])==3 and len(r["ranking"])==3 and r["return_to_best"]["status"]=="reached"'
```

Open `summary.png`, all three `triangle_N_overlay.png` files and `final.png`.
Check that the mask actually outlines the carton in every view, and that the
final camera image resembles the winning capture. The reported score is a
relative comparison among these captures, not a calibrated probability.

Re-score without the robot, for example with `--debug` to get FindVP's match visualisations:

```bash
python3 "$FV" score --run-dir "$RUN" --debug
```

If execution was interrupted after some images were saved, `run` refuses to
reuse that `views/` directory. Preserve it for diagnosis. To start a new
attempt with the **same still-valid target**, choose a fresh `RUN` and pass
`--target-image /path/to/old/target.png --target-mask
/path/to/old/target_mask.png` to `run`; do not reuse that reference if the
carton or camera mounting changed.

## Options

* `--radius 0.45` changes the nominal camera distance. The configured fallback
  radii are 0.45 and 0.50 m, tried only when larger than the requested radius.
  Confirm that the carton is visible and in focus at the chosen distance.
* `--azimuth-offset 30` rotates the three triangle directions. Use the same
  setting for `check`, `target`, preview and execution if changed.
* `--slow-factor 8` is the default trajectory time-stretch. Values below 1
  are rejected. Slowing a trajectory does not make incorrect geometry safe.
* `start_planner.sh --no-camera-spheres` disables the camera-body collision
  approximation. It is for **diagnosis only**; do not execute arm motion with
  the camera body omitted from the planning world.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `CuRobo planner not reachable` | start terminal H; the dev container reaches it via `--net=host` |
| `no fresh /joint_states` | Confirm `steve_env.sh` was sourced, the onboard system is reachable, and the ROS domain/interface are correct. Do not move on stale state. |
| `robot_mode=3` / `safety_mode=5` | safeguard stop (lidar / E-stop). Hardware fix only, no software bypass |
| `goal rejected` | Check controller state, safety mode and External Control. Resolve the cause with the arm stationary; do not repeatedly send goals. |
| `planner FK !!` | Planner model and reported robot TF disagree. **Do not move.** Record the position/rotation errors. |
| `start state !! IN COLLISION` | Inspect the modelled carton, deck, camera and obstacles against the real workcell. Reposition only through the supervised robot procedure; do not disable collision bodies to pass the check. |
| `IK failed: view unreachable or in collision` | The view may be out of reach or blocked. Check the scene and path; try a reviewed radius or azimuth offset, then preview again. |
| outline not on the carton | carton moved, or wrong `scene.json`; see section 4 |
| RViz arm looks rotated | Confirm Fixed Frame `base_link`, compare live joints with the pendant, and compare RViz's onboard-description RobotModel with the physical arm. `planner FK OK` is supporting evidence, not permission to ignore a physical mismatch. |
| Preview says `unreachable` but exits 0 | This is current preview behaviour. **Do not execute** until T1, T2 and T3 each show a valid plan. |
| `views/` already exists on retry | Preserve that partial attempt. Use a fresh `RUN` and, if still valid, pass the old target image and mask paths explicitly. |
