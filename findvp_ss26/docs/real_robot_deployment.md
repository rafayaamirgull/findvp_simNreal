# FindVP on Steve: supervised real-robot deployment

For the known carton box fixed on Steve's own top deck, use the
[known-box RGB command sequence](steve_known_box_runbook.md). The route below
is for unknown objects or targets on a separate table.

This ROS 2 adapter uses Steve's UR5e, wrist D415, existing camera/viewer and
MoveIt. It does not launch robot drivers, replace the viewer, drive the mobile
base, unlock the platform, or return the arm home. It can move first to an
operator-recorded table-view start pose. Motion requires `--execute`
and approval of EACH trajectory. This is a research adapter tested with fake
ROS endpoints, **not a hardware-validated safety system**.

For the complete current command sequence, use the
[laptop USB RGB + Ethernet Steve runbook](steve_rgb_runbook.md). It includes
`findvp_select_target`, the explicit full-object reference mask, and
`findvp_survey` for supervised automatic survey motion before localization.

This guide follows the current `steve_ros2_ws_student/README.md`, its
`steve_env.sh`, `start_camera.sh`, and `neo_ur_moveit.launch.py`. In this setup,
the onboard computer owns the robot driver, controllers and robot TF; the
development container runs the D415, MoveIt/RViz and FindVP. Do not launch a
second UR driver, controller manager or robot_state_publisher from FindVP.

## Unknown objects: capabilities and limits

Localization/search needs no object mesh, category, dimensions, known center,
6D pose, simulator state or depth. It DOES need a reference photo of the desired
object/view. Without a reference, FindVP cannot know which view to match.
Use a clearly visible object in the reference. Outline the full object once with
`findvp_select_target` and pass its `--target-mask` to BOTH localization and search.
This avoids the simulator's central-colour heuristic selecting only one part of
a multi-colour real object. Your existing camera viewer can provide the photo.

SAM generates class-agnostic scene masks; DINO selects the reference-matching
instance. SIFT correspondences across calibrated, translated survey views
recover a metric `visible_surface_aim_point`, not a complete object centroid.
The runner samples camera poses around that point and refines spherical
triangles with DINO/SIFT ranking. This adapter does not yet reproduce every
simulator look-ahead, radial-search, or NBV option.

Arbitrary class does NOT mean guaranteed success for every object: transparent,
reflective, textureless, repetitive, occluded or visually indistinguishable
objects can defeat segmentation, matching or triangulation. Ambiguous/weak
localization stops before search. Ranking reports rejected images and aborts if
none are usable. A single similar-looking distractor is not guaranteed to be
identifiable as wrong. `--mask-backend flat` is an explicit uniform-background
fallback, not a general clutter solution.

## 0. Physical prerequisites

The operator must verify calibration, payload, reduced speeds, clear workcell,
robot safety configuration and emergency-stop access. Synchronize robot and
camera-computer clocks; timestamps are compared in ROS time. Do not use
`use_sim_time` on real hardware. No Python guard replaces the robot's safety
controller or physical stop procedure.

### Camera collision model

`urdf/steve_d415_mount.urdf.xacro` contains an UNVERIFIED placeholder housing.
Measure and adjust its box/origin to enclose the camera AND bracket. The local
Steve description now includes it when `findvp_wrist_camera:=true`:

```xml
<xacro:include filename="$(find findvp_ros)/urdf/steve_d415_mount.urdf.xacro"/>
<xacro:findvp_steve_d415 parent="ur5etool0"/>
```

The local source is
`/home/ws/src/neo_mpo_700-2/robot_model/mpo_700.urdf.xacro`; the provided
FindVP MoveIt launch enables that flag. Rebuild `neo_mpo_700-2` and
`neo_ur_moveit_config`, source the overlay, then restart local MoveIt while
the arm is stationary. The collision dimensions remain unmeasured and must be
reviewed against the physical mount.

Keep `bash start_camera.sh` unchanged for this local-model approach: it publishes
the calibrated `ur5etool0 -> d415_calib_color_optical_frame_handeye` transform.
Adding a fixed joint to MoveIt's internal model does NOT itself publish TF.
Steve's local MoveIt launch starts only MoveGroup and RViz, not
robot_state_publisher. Its `use_d435` model option describes a different,
base-mounted camera; it is not the wrist D415.

Only if you separately update the onboard robot_state_publisher to publish the
same calibrated joint, disable the duplicate hand-eye publisher. The script supports
skipping it with a deliberately nonexistent calibration file:

```bash
HANDEYE_FILE=/nonexistent/findvp-handeye-already-in-urdf.json bash /home/ws/start_camera.sh
```

Use this ONLY after verifying robot_state_publisher actually publishes that
calibrated joint, not just that it exists in MoveIt's model. Otherwise retain
the original camera command for both survey and search. Never
publish the same calibrated child frame from two TF authorities.

Also compare the local arm model with the onboard driver's calibrated model.
The inspected arm Xacro defaults to `ur_description`'s `default_kinematics.yaml`;
that is not evidence of this particular UR5e's factory calibration. Resolve
model/TF discrepancies through the Steve setup procedure before physical use.

### Scene geometry

In RViz's MoveIt Planning Scene editor, add measured conservative geometry for
the support/table, floor, nearby obstacles and a keepout enclosing the observed
target. Publish the scene to MoveIt. Example IDs used below: `findvp_support`
and `findvp_target_keepout`. Do not permit collisions with these objects in the
allowed-collision matrix.

The target geometry need not be known beforehand, but its occupied region must
be measured/observed conservatively before physical planning. This collision
protection is separate from FindVP's object-model-free scorer. An aiming point
is not a collision volume. Automatic obstacle reconstruction is NOT implemented.
A 0.40 m radius or valid IK solution does not establish clearance.

Execution checks camera collision geometry in MoveIt's URDF, required scene IDs,
TF/config hand-eye agreement, and scene stability between planning/execution.
Those checks cannot prove that your geometry matches the actual environment.

## 1. Install inside Steve's ROS Humble container

`findvp_ss26` now lives inside `steve_ros2_ws_student`, so the existing workspace
mount exposes it directly as `/home/ws/findvp_ss26`. Do not add the obsolete
`/home/findvp` sibling bind mount and do not create a second copy of the project.

All remaining commands run INSIDE that Humble container, not the host or the
Python 3.8 CuRobo environment. Create a separate perception environment:

```bash
source /opt/ros/humble/setup.bash
source /home/ws/install/setup.bash
cd /home/ws
# Only when interfaces have not already been built successfully:
/usr/bin/python3 -m colcon build --base-paths src --packages-select neo_msgs2 neo_srvs2
# Rebuild updated Steve launch/model packages when changed:
/usr/bin/python3 -m colcon build --base-paths src --packages-select \
  mp_components neo_mpo_700-2 neo_ur_moveit_config
source /home/ws/install/setup.bash
python3 -m venv --system-site-packages /home/ws/.venv-findvp
touch /home/ws/.venv-findvp/COLCON_IGNORE
source /home/ws/.venv-findvp/bin/activate
python3 -m pip install --upgrade pip "packaging>=22"
```

Install Ubuntu's `python3-venv` package first if needed. Install matching torch
and torchvision wheels in this venv. This CPU baseline needs no CUDA choice:

```bash
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install -r /home/ws/findvp_ss26/requirements-ros.txt
python3 -m pip check
```

Do not switch existing interface builds between normal and symlink installation:
their cached CMake mode can retain the old setting. Your already working
`neo_msgs2` and `neo_srvs2` builds can be reused. Requirements pin setuptools below
80 for colcon and pytest below 9 for Humble's test plugin.

The explicit `packaging` upgrade is required with `--system-site-packages`.
Without it, ROS Humble's older system copy can be combined with a newer
`setuptools`, causing `canonicalize_version() got an unexpected keyword
argument 'strip_trailing_zero'` during `colcon build`. If the venv already
exists and shows that error, activate it and run the same `packaging` upgrade;
there is no need to recreate the venv.

CPU inference can be slow. For GPU use, substitute a matching CUDA wheel pair
compatible with the container's driver; do not overwrite working CuRobo torch.
DINO/SAM download pretrained weights on first use: initialize them before the
hardware session while internet is available. Full model downloads/inference
were NOT performed in the isolated ROS tests.

Build with the activated interpreter so entry points use the perception venv:

```bash
cd /home/ws
python3 -m colcon build --symlink-install \
  --base-paths src findvp_ss26 --packages-select findvp_ros
source /home/ws/install/setup.bash
ros2 pkg executables findvp_ros
ros2 run findvp_ros findvp_search --help
```

The explicit `--base-paths` is important because FindVP is intentionally kept
at the workspace top level rather than copied or symlinked into `src/`.

### Why the first hardware version uses MoveIt, not CuRobo

CuRobo remains useful in the simulator and may later generate or rank candidate
paths. The checked-in CuRobo model is a reduced simulation model with a separate
CUDA/Python runtime and manually constructed collision world. It is not kept in
lockstep with Steve's live `robot_description`, controller state, MoveIt planning
scene, or attached wrist hardware. Therefore the first real deployment uses
MoveIt for IK, collision checking, trajectory generation and execution. Do not
send a CuRobo trajectory to the hardware until a separate adapter validates its
robot model, joint ordering, limits, time parameterization, live start state and
collision world against MoveIt immediately before execution.

Verify the generated interfaces after sourcing the rebuilt workspace:

```bash
python3 -c 'from neo_msgs2.msg import EmergencyStopState; from neo_srvs2.srv import LockPlatform'
```

## 2. Start Steve using its README's sequence

Use native ROS terminals for robot startup, camera and MoveIt. In each:

```bash
cd /home/ws
source steve_env.sh
source /home/ws/install/setup.bash
```

The expected status is `192.168.60.90 reachable`, domain `74`,
`ROS_LOCALHOST_ONLY=0` and `rmw_fastrtps_cpp`. The updated helper auto-detects the
network interface rather than hardcoding `enp3s0`. If it chooses the wrong NIC,
set `STEVE_IFACE` to the actual Ethernet interface before sourcing; do not
copy a guessed interface name. `source steve_env.sh --check` checks without
reconfiguring the network. Normal sourcing changes local networking and stops
the local ROS daemon; it does not start the onboard driver.

In terminals running FindVP, additionally activate the perception environment
AFTER the ROS setup above:

```bash
source /home/ws/.venv-findvp/bin/activate
```

### A. Camera and viewer (keep both running)

Camera terminal, if not already running:

```bash
bash /home/ws/start_camera.sh
```

Wait for `RealSense Node Is Up!`. Use the `HANDEYE_FILE` variant only when the
onboard robot_state_publisher supplies the alias, as explained in section 0.
The updated script handles RealSense device nodes that appeared after container
startup; use its diagnostics for USB/video-group issues instead of launching a
second camera driver. In another terminal (or retain your current viewer):

```bash
ros2 run rqt_image_view rqt_image_view /d415_calib/d415_calib/color/image_raw
```

The updated camera script starts RGB only; no runtime depth-disable commands are
needed. It still publishes the calibrated optical alias.

### B. Enable Remote Control and start External Control (operator action)

With the robot stationary, workspace clear, emergency stop accessible and the
teach-pendant speed slider reduced:

1. On PolyScope, open **Menu > Settings > System > Remote Control**, then **Enable**.
2. Select **Remote Control** from the header's mode/profile selector. Enabling
   the feature and selecting the mode are separate steps.
3. Power on/initialize using the normal robot procedure. Verify that the intended
   `external_control.urp` is loaded; do not play an unknown program.

Then follow Steve's dashboard reconnect/start sequence in a sourced terminal:

```bash
ros2 service call /dashboard_client/quit std_srvs/srv/Trigger '{}'
ros2 service call /dashboard_client/connect std_srvs/srv/Trigger '{}'
ros2 service call /dashboard_client/raw_request ur_dashboard_msgs/srv/RawRequest "{query: 'is in remote control'}"
ros2 service call /dashboard_client/play std_srvs/srv/Trigger '{}'
ros2 service call /dashboard_client/program_running ur_dashboard_msgs/srv/IsProgramRunning '{}'
```

Confirm service success, remote-control `answer: 'true'` and `program_running: true`.
The raw query supports Steve dashboard versions without the typed remote-control
service; the motion runner uses the same fallback.
Also check the driver's External Control status, not just that any UR program runs:

```bash
ros2 topic echo /io_and_status_controller/robot_program_running \
  --qos-reliability reliable --qos-durability transient_local --once
```

Expect `data: true` when this optional topic is published by the driver. Older
Steve versions omit it; the runner uses fresh dashboard status and an active
trajectory controller in that case. A received false still prevents execution.
A safety-related `play` rejection can mean Local Control is
still selected; inspect PolyScope and its diagnostic rather than bypassing it.
These commands can enable robot motion. Do not reconnect/play during an active
trajectory or a FindVP run. FindVP itself only queries dashboard status: it
never calls `quit`, `connect`, `play`, power-on or safety-reset services.

### C. Start local MoveIt and RViz once

In another sourced native-ROS terminal, use the exact Steve README command:

```bash
ros2 launch neo_ur_moveit_config neo_ur_moveit.launch.py \
  robot_type:=mpo_700 \
  arm_type:=ur5e \
  prefix:=ur5e \
  launch_rviz:=True
```

Or, after building FindVP, use its wrapper with these choices pinned and
simulation/mock hardware explicitly off:

```bash
ros2 launch findvp_ros steve_moveit.launch.py
```

Choose ONE command, not both, and do not start another MoveGroup if one is
already active. The wrapper only includes Steve's launch; it does not add a
driver, controller manager, TF publisher, camera or program-start call. It keeps
Steve's default AC arm mounting (`use_ur_dc=False`) and no gripper; if the actual
hardware differs, review/change the robot configuration rather than using it.

Wait for `You can start planning now!`. The `prefix` argument is essential:
URDF derives `ur5e` from `arm_type`, while SRDF/controller YAML separately use
`prefix`; omitting it gives mismatched joint names. Keep the spelling `ur5e`,
not `ur5e_`.

In RViz select **Planning Group: `ur_manipulator`**. The checked-in RViz file
still saves `arm`, which does not match Steve's SRDF. Its fixed frame `base_link`
and planned-path topic `/display_planned_path` are appropriate; FindVP's pose
coordinates remain in `ur5ebase_link` and are transformed by MoveIt/TF. Keep
RViz scaling low (Steve suggests 0.10; FindVP defaults to 0.05). Only perform a
small reviewed Plan/Execute commissioning check once the model and scene are
correct. During FindVP execution, use RViz to inspect plans, NOT to issue
competing Execute commands.

### D. Verify readiness

Read-only checks before proceeding:

```bash
ros2 action list -t
ros2 service list -t
ros2 topic echo /joint_states --once
ros2 topic echo /odom --once
ros2 topic echo /emergency_stop_state --once
ros2 topic echo /d415_calib/d415_calib/color/camera_info --once
ros2 service call /dashboard_client/get_safety_mode ur_dashboard_msgs/srv/GetSafetyMode '{}'
ros2 service call /dashboard_client/get_robot_mode ur_dashboard_msgs/srv/GetRobotMode '{}'
ros2 service call /dashboard_client/program_running ur_dashboard_msgs/srv/IsProgramRunning '{}'
ros2 service call /dashboard_client/raw_request ur_dashboard_msgs/srv/RawRequest "{query: 'is in remote control'}"
```

Required endpoints include `/move_action`, `/execute_trajectory`,
`/get_planning_scene`, `/move_group/get_parameters`,
`/controller_manager/list_controllers` and `/lock_platform`. The scaled
trajectory controller and external-control program must be active, and Remote
Control must remain selected. FindVP monitors Remote Control with the read-only
dashboard service; unknown, false or stale status prevents execution. Plan-only
preview does not require enabling Remote Control or playing a program.

UR mode/program subscriptions are reliable/transient-local; read-only dashboard
polling supplies liveness because these topics can publish only on changes.
Missing/stale dashboard, controller, emergency, joints or odometry aborts motion.
The 2-second watchdog is not a certified stop-response time. Diagnose missing
streams instead of increasing the timeout to bypass failures.

## 3. Record the reviewed table-view start pose

Use RViz/MoveIt under direct supervision to place the arm in a collision-free
rest/start configuration where the D415 optical view contains the work surface.
This is installation-specific: the photos do not provide safe joint angles, so
none are hard-coded. With the robot stationary and the live viewer confirming
the view, record the measured joints without moving the robot:

```bash
ros2 run findvp_ros findvp_record_pose \
  --config /home/ws/findvp_ss26/config/steve_real.yaml \
  --output /home/ws/findvp_config/table_view_start.yaml \
  --name table_view_start
```

Review the YAML and keep it only while the arm mount, camera mount and workcell
layout remain unchanged. The search plans to this joint pose, displays the exact
trajectory, and requires the same typed `EXECUTE` approval as every viewpoint.
After reaching it, the measured camera optical axis must point within 30 degrees
of the localized target or the run aborts.

## 4. Lock the base and collect a new survey

The recommended workflow is now **automatic supervised survey**. First lock the
base and start the recorder. Do not manually call the capture service during
automatic survey or search.

Keep the base, object and surrounding scene fixed through survey AND search.
Lock the base before the FIRST survey image:

```bash
ros2 service call /lock_platform neo_srvs2/srv/LockPlatform '{}'
```

Confirm `success: true`. The implementation refuses the lock if the latest base
velocity command is too large; stop navigation/teleop commands first and verify
the base is physically stationary. A successful lock response alone does not
prove that the base has stopped. Start the recorder in a separate FindVP/venv terminal:

```bash
ros2 launch findvp_ros real_robot_capture.launch.py \
  config_file:=/home/ws/findvp_ss26/config/steve_real.yaml \
  output_dir:=/home/ws/findvp_runs/object_01
```

With that recorder writing an EMPTY run directory and the start pose reviewed,
run in another sourced FindVP terminal:

```bash
SURVEY_ARGS=(
  --config /home/ws/findvp_ss26/config/steve_real.yaml
  --start-pose /home/ws/findvp_config/table_view_start.yaml
  --capture-manifest /home/ws/findvp_runs/object_01/observations.json
  --survey-baseline 0.04
)
ros2 run findvp_ros findvp_survey "${SURVEY_ARGS[@]}" \
  --output-dir /home/ws/findvp_runs/object_01/survey_preview
# After reviewing actual model, scene and plans:
ros2 run findvp_ros findvp_survey "${SURVEY_ARGS[@]}" \
  --output-dir /home/ws/findvp_runs/object_01/survey \
  --required-scene-object findvp_support \
  --required-scene-object findvp_target_keepout \
  --execute --acknowledge-model-reviewed
```

The runner executes the approved start-pose plan, captures there, then plans and
captures 4 cm right and 4 cm left in camera coordinates, preserving orientation.
Every trajectory still requires `EXECUTE`. No initial object position or depth
is needed. Keep the target visible; the offsets are sampling parameters, not
guarantees of visibility or clearance. Execution/capture/planning failures abort;
restart with a new run directory after a partial survey. All motion reuses the
search runner's exact-trajectory, freshness, cancellation and scene guards.

It subscribes to the existing camera, without opening/reconfiguring it. For an
optional MANUAL survey only, move the arm with reviewed RViz/MoveIt goals to at
least two, preferably three, views.
Include several centimetres of translation, overlapping target visibility and
useful parallax; pure wrist rotation cannot recover metric position. Wait until
stationary, then capture each view:

```bash
ros2 service call /findvp/capture std_srvs/srv/Trigger '{}'
```

The recorder waits for a NEW exposure after the request, synchronized
calibration, post-exposure joint/base samples, and camera TF at exposure time.
It saves PNGs plus `observations.json` with measured camera transforms, named
joints, timestamps and base-odometry provenance. Do not run multiple recorders
against one manifest or manually trigger captures during automated search.

Moving the base requires a new survey/output directory. Older manifests without
base provenance are rejected. Odometry cannot guarantee detection of slip or
resets that reproduce the same pose: repeat the survey after any base/odometry
restart. Object motion is not reliably detected; repeat if anyone moves it.

Save the desired-view reference photo using your viewer, for example at
`/home/ws/targets/target.png`. No object category is entered.

## 5. Localize using the real observations

```bash
ros2 run findvp_ros findvp_select_target \
  --image /home/ws/targets/target.png --output /home/ws/targets/target_mask.png
# Outline the whole object, close the polygon, then press Enter. Reuse an
# existing reviewed mask instead of rerunning this command over its output.
ros2 run findvp_ros findvp_localize \
  --observations /home/ws/findvp_runs/object_01/observations.json \
  --target-image /home/ws/targets/target.png \
  --target-mask /home/ws/targets/target_mask.png \
  --output-json /home/ws/findvp_runs/object_01/localization.json \
  --mask-backend sam
```

Both localization and search validate the reference image AND mask digests.
Localize refuses an existing output file. Failed estimates leave selected masks,
feature coordinates and pair diagnostics in `localization.diagnostics/` rather
than writing a usable localization. SAM defaults to four prompts per batch to
reduce peak memory; `--sam-points-per-batch 1` is available for tight memory.

Inspect detections, baseline/parallax, reprojection error, triangulated-point
spread and the aiming point. Improve lighting, baseline or visibility if this
fails; do not invent a center to bypass failed localization. An observation can
contain a `mask` path to a same-resolution binary PNG to override survey masks.
SAM handles subsequent search images; an external segmentation-service adapter
is not implemented yet.

## 6. Plan-only preview

In the search terminal, define this shell array (recreate it in new terminals):

```bash
FINDVP_ARGS=(
  --config /home/ws/findvp_ss26/config/steve_real.yaml
  --start-pose /home/ws/findvp_config/table_view_start.yaml
  --localization /home/ws/findvp_runs/object_01/localization.json
  --target-image /home/ws/targets/target.png
  --target-mask /home/ws/targets/target_mask.png
  --capture-manifest /home/ws/findvp_runs/object_01/observations.json
  --views 3 --radius 0.40 --max-refinement-depth 2
  --mask-backend sam
)
ros2 run findvp_ros findvp_search "${FINDVP_ARGS[@]}" \
  --output-dir /home/ws/findvp_runs/object_01/preview
```

This first previews the start-pose path and then the initial candidate paths,
all from the CURRENT arm state, with no execution or capture. Because plan-only
mode does not change the planning start state, candidate previews are not chained
from the start pose. Set RViz MotionPlanning's planned-path topic to
`/display_planned_path`.
Full trajectories are saved in the JSON report. Review clearance, reachability,
optical direction and radius. The default radius is not an inferred safe size.
No valid plans is a failure. Adaptive candidates only exist after real ranking;
this preview is not approval of every future trajectory.

## 7. Supervised execution, after model/scene review

Keep the recorder/viewer running, base locked, workcell clear and operator at
the physical stop. Use IDs actually assigned to the reviewed collision objects:

```bash
ros2 run findvp_ros findvp_search "${FINDVP_ARGS[@]}" \
  --output-dir /home/ws/findvp_runs/object_01/search \
  --required-scene-object findvp_support \
  --required-scene-object findvp_target_keepout \
  --execute --acknowledge-model-reviewed
```

The runner first plans the recorded table-view start pose and requires approval.
It then confirms that the camera faces the localized aim point. For EACH
candidate it plans, displays/saves, requires you to type `EXECUTE` within 120
seconds, checks scene/start state again, and executes that EXACT trajectory,
waits for distinct stationary joint samples, captures a fresh image and verifies
the measured camera endpoint. Images from a round are ranked together before
refinement. Velocity/acceleration scaling defaults to 5%; hardware limits and
safety settings remain the operator's responsibility.

Only completed no-path/IK planning failures can skip candidates. Execution,
capture, freshness, safety or cancellation faults abort the whole run.
Ctrl+C/SIGTERM requests bounded cancellation while ROS remains alive. If you
see `MOTION TERMINATION UNCONFIRMED`, use the physical stop procedure and do
NOT restart FindVP. A killed process, lost network or computer failure cannot
be made safe by Python cleanup. Action termination is not proof of safe physical
stopping; confirm the robot is stopped before recovery.

The base stays locked. Use the reviewed Steve stow procedure before the operator
unlocks it. There is no automatic return home or move to the best pose.
Execution requires an interactive terminal; approval cannot be piped in.

## Outputs, configuration and testing

`search/real_object_real_search.json` includes attempts, full trajectories,
stop reason and `best_observation`: score, image path, exposure time, measured
camera transform and joint positions. Earlier best results survive a worse
refinement round. `complete` means the search finished: inspect `stop_reason`
for threshold reached, no improvement or budget exhausted. The arm remains at
the last executed candidate, not necessarily the best one.
Existing search reports are not overwritten: choose a new `--output-dir` or
`--name` for another run. Keep the same live capture manifest only while the
survey's base/object/calibration remain valid.

`config/steve_real.yaml` uses Steve's actual joint names, e.g.
`ur5eshoulder_pan_joint`, and distinguishes the driver's optical image frame
from the calibrated hand-eye alias. If the mount changes, recalibrate, update
both configuration AND actual URDF, restart relevant nodes and repeat the
survey. An unchanged frame name does not make an old calibration valid.

Host tests (ROS modules skip if rclpy is unavailable):

```bash
python3 -m pytest -q
```

Real ROS pub/sub/action tests, with fake endpoints and no robot, inside Humble:

```bash
cd /home/ws/findvp_ss26
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=197 python3 -m pytest -q \
  tests/test_ros_runtime.py \
  tests/test_ros_capture_integration.py \
  tests/test_ros_action_integration.py \
  tests/test_ros_search_integration.py \
  tests/test_ros_survey_integration.py \
  tests/test_real_rgb_workflow.py \
  tests/test_steve_launch.py
```

Fixtures force localhost domain 197, not Steve's domain 74. Coverage includes
stale/reused frames, callback scheduling, moved base, exact trajectory transport,
confirmed cancellation, delayed acceptance, interruption, and the real plan-only
CLI (including zero feasible candidates). The CLI tests need `neo_msgs2` from
the Steve overlay. They do not
validate neural accuracy, live MoveIt/controller setup, real calibration,
collision envelopes or physical stopping behavior.

Interface references: [MoveIt ExecuteTrajectory](https://docs.ros.org/en/iron/p/moveit_msgs/interfaces/action/ExecuteTrajectory.html)
and [UR Humble GPIO controller source](https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver/blob/humble/ur_controllers/src/gpio_controller.cpp).
