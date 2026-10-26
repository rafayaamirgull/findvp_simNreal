# FindVP RGB on Steve: external-table/unknown-object route

For the carton box fixed to Steve's own top deck, use the
[known-box runbook](steve_known_box_runbook.md). This older route needs live
odometry and RGB multiview localization for a separate/unknown target.

Run these commands in the **laptop's ROS Humble development container**, where
this workspace is `/home/ws`. The camera connects to the laptop by USB. Steve's
onboard computer (`192.168.60.90`) runs the existing drivers/controllers and
publishes robot state over Ethernet. Do not start another UR driver on the laptop.

The workflow is **select reference → supervised automatic RGB survey → localize
→ preview → supervised triangle search**. No depth sensor, object class, CAD
model or known object dimensions are consumed. The selected reference mask tells
FindVP which pixels depict the desired object. The start pose and conservative
collision geometry are installation-specific and still require operator review.
No hardware motion was used to test this adapter; fake ROS tests are not hardware
commissioning.

The saved `rgb_run_02` observations were checked offline with a whole-object
reference mask. They still failed triangulation: the first pair had 25 matches
but only one passing the unchanged checks, with most reprojection errors around
3–4 pixels (limit 2 pixels). Review hand-eye calibration/current mounting and
image-to-pose consistency before physical commissioning. This is not proof that
calibration alone is the cause. Do not reuse that failed survey as a valid
localization or widen the thresholds to bypass it.

## 1. Enter the container and prepare terminals

Open `steve_ros2_ws_student` in VS Code and choose **Dev Containers: Reopen in
Container**. Keep using the existing container if it is already open.

Run this block in **every new container terminal**. Keep the same run name in
all terminals; use a previously unused name for each new session:

```bash
cd /home/ws
export STEVE_IFACE=enx503eaadfddca
source /opt/ros/humble/setup.bash
source /home/ws/steve_env.sh
source /home/ws/install/setup.bash
source /home/ws/.venv-findvp/bin/activate

export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
export FINDVP_RUN=rgb_run_03
export FINDVP_DIR="/home/ws/findvp_runs/$FINDVP_RUN"
export FINDVP_CONFIG=/home/ws/findvp_ss26/config/steve_real.yaml
export FINDVP_START=/home/ws/findvp_config/table_view_start_v2.yaml
export FINDVP_TARGET=/home/ws/targets/target.png
export FINDVP_MASK=/home/ws/targets/target_mask.png
```

`enx503eaadfddca` is the Ethernet interface from your working session. If it has
changed, check `ip -br addr` and substitute its actual name. Expect onboard
reachable, domain 74, localhost-only 0 and `rmw_fastrtps_cpp`.

You already have the virtual environment. If missing after a clean installation,
install `python3.10-venv` inside the container, create it with
`python3 -m venv --system-site-packages /home/ws/.venv-findvp`, then activate it.
Do not continue to pip installation if activation failed.

## 2. Install the corrections and build — terminal A

No need to rebuild the previously working `neo_msgs2` / `neo_srvs2` packages or
switch their installation mode again.

```bash
python3 -c 'import sys; assert sys.prefix != sys.base_prefix, "Activate .venv-findvp first"'
touch /home/ws/.venv-findvp/COLCON_IGNORE
python3 -m pip install -r /home/ws/findvp_ss26/requirements-ros.txt
python3 -m pip check

python3 -m colcon build --symlink-install \
  --base-paths src findvp_ss26 --packages-select findvp_ros
source /home/ws/install/setup.bash

ros2 pkg executables findvp_ros
python3 -c 'from neo_msgs2.msg import EmergencyStopState; from neo_srvs2.srv import LockPlatform; print("Steve interfaces OK")'
ros2 run findvp_ros findvp_survey --help
```

Expect six executables: capture, localize, record_pose, search, select_target,
survey. Printed package/executable pairs are output, not shell commands.
The requirements keep setuptools below 80 for colcon and pytest below 9 for
Humble. Your existing torch/torchvision installation is retained. For a clean
installation without them, install a matching CPU pair with
`python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`.
The selector needs PyQt5 from the ROS desktop environment; if absent install
`sudo apt-get install python3-pyqt5` inside the container.

## 3. Check robot feedback — terminal A

```bash
ros2 node list --no-daemon --spin-time 3

timeout 10 ros2 topic echo /joint_states sensor_msgs/msg/JointState \
  --qos-reliability best_effort --qos-durability volatile \
  --filter "'ur5eshoulder_pan_joint' in m.name" --once

timeout 10 ros2 topic echo /odom nav_msgs/msg/Odometry \
  --qos-reliability best_effort --qos-durability volatile --once

timeout 10 ros2 topic echo /emergency_stop_state neo_msgs2/msg/EmergencyStopState \
  --qos-reliability best_effort --qos-durability volatile --once
```

All three streams must deliver current messages. The base must be stationary and
stop flags clear. **Stop here if odometry is absent.** A topic publisher appearing
in discovery does not establish that it is publishing data. Capture requires
odometry to detect base movement; this is unrelated to camera depth.

For missing odometry, use `hostname` and `ip -details -statistics link show can0`
on **Steve's onboard computer**. `can0` is not expected on the laptop. Do not
fabricate odometry or blindly restart the base driver; auto-homing can move its
steering. Keep the two computers' clocks synchronized for exposure-time TF.

## 4. Start RGB and hand-eye TF — terminal B

Prepare terminal B using step 1. Stop an old camera script with Ctrl+C in its own
terminal, then start the updated script once:

```bash
bash /home/ws/start_camera.sh
```

Leave it running. It now disables depth/infrared/alignment/point clouds at startup
and retains the calibrated optical-frame TF publisher. Wait for the RealSense
node to start. Do not launch a second camera driver.

## 5. Viewer — terminal C; camera checks — terminal A

Prepare terminal C, then leave this running:

```bash
ros2 run rqt_image_view rqt_image_view /d415_calib/d415_calib/color/image_raw
```

In terminal A:

```bash
ros2 param get /d415_calib/d415_calib enable_depth

timeout 10 ros2 topic echo /d415_calib/d415_calib/color/image_raw sensor_msgs/msg/Image \
  --qos-reliability best_effort --qos-durability volatile --field header --once

timeout 10 ros2 topic echo /d415_calib/d415_calib/color/camera_info sensor_msgs/msg/CameraInfo \
  --qos-reliability best_effort --qos-durability volatile --once

timeout 10 ros2 run tf2_ros tf2_echo ur5ebase_link d415_calib_color_optical_frame_handeye
```

Depth should be false, both camera topics should deliver data, and TF should
print transforms. An initial TF discovery warning followed by transforms is
normal. CameraInfo supplies RGB intrinsics, not a depth image.

## 6. Prepare the robot and start External Control — terminal A

On the teach pendant, enable **and select Remote Control**, initialize the robot,
and verify the intended `external_control.urp` is loaded. Keep the workspace clear
and the speed slider reduced. Only while stationary, run:

```bash
timeout 10 ros2 service call /dashboard_client/quit std_srvs/srv/Trigger '{}'
timeout 10 ros2 service call /dashboard_client/connect std_srvs/srv/Trigger '{}'

timeout 10 ros2 service call /dashboard_client/raw_request \
  ur_dashboard_msgs/srv/RawRequest "{query: 'is in remote control'}"
```

Require `answer='true'`. Your dashboard lacks the typed remote-control service;
the search/survey runner also supports this read-only fallback.

```bash
timeout 10 ros2 service call /dashboard_client/play std_srvs/srv/Trigger '{}'
timeout 10 ros2 service call /dashboard_client/program_running \
  ur_dashboard_msgs/srv/IsProgramRunning '{}'
timeout 10 ros2 service call /controller_manager/list_controllers \
  controller_manager_msgs/srv/ListControllers '{}'
```

Require program-running true and `scaled_joint_trajectory_controller` active.
If play fails, inspect the pendant's actual diagnostic; do not keep retrying.
Do not reconnect or play during an active trajectory. The controller service
works even when the optional `ros2 control` CLI plugin is absent.

## 7. MoveIt and measured scene — terminal D

Prepare terminal D and run one MoveIt instance:

```bash
ros2 launch findvp_ros steve_moveit.launch.py
```

Wait for planning readiness. In RViz select `ur_manipulator` and planned-path
topic `/display_planned_path`. The saved RViz default `arm` is incorrect.

**Execution prerequisite:** check the now-integrated wrist-camera/bracket
collision geometry and publish the table, nearby obstacles and target keepout to MoveIt.
Follow [the model/scene instructions](real_robot_deployment.md#0-physical-prerequisites).
The supplied camera dimensions are an unverified placeholder; no command can
infer the real bracket envelope from the photos. The flags below do not create
collision objects. This is required for automatic survey as well as search.

Use actual scene IDs; the examples below assume `findvp_support` and
`findvp_target_keepout`. Adding the camera only to the simulator model has no
effect on real MoveIt planning. Keep the camera script's hand-eye TF publisher.

## 8. Review the start pose — terminal A

```bash
sed -n '1,160p' "$FINDVP_START"
```

Your latest file is `table_view_start_v2.yaml`. It must still be appropriate for
the current mount and workcell, with the whole target visible. The automated
survey plans back to these measured joints before capturing.

If a new pose is needed, position the arm once using reviewed RViz planning,
confirm the camera view, wait until stationary, then record a NEW filename:

```bash
ros2 run findvp_ros findvp_record_pose \
  --config "$FINDVP_CONFIG" \
  --output /home/ws/findvp_config/table_view_start_v3.yaml \
  --name table_view_start_v3
export FINDVP_START=/home/ws/findvp_config/table_view_start_v3.yaml
```

Recording never moves the robot. Do not run this example if reusing v2.

## 9. Select the complete reference object once — terminal A

```bash
ls -lh "$FINDVP_TARGET"
ros2 run findvp_ros findvp_select_target \
  --image "$FINDVP_TARGET" --output "$FINDVP_MASK"
```

In the window, outline the **whole object** with polygon clicks. Close the polygon
by clicking its first point, then press Enter to save; Escape cancels. For the
drill include housing, chuck, handle and battery, excluding the table/marker.
This selects reference pixels only; it does not require object dimensions or a
class. Inspect the resulting black/white mask in an image viewer.

Reuse a reviewed mask only with the unchanged reference. The command refuses
overwrites. To redo selection, choose another mask filename and update
`FINDVP_MASK`. Both localization and search must use the same mask. An external
full-resolution binary mask can also be supplied directly.

## 10. Lock the base; start the recorder — terminals A and E

Stop navigation/teleoperation and confirm the base is stationary. Terminal A:

```bash
timeout 10 ros2 service call /lock_platform neo_srvs2/srv/LockPlatform '{}'
```

Require success. Prepare terminal E using step 1 and run:

```bash
mkdir -p "$FINDVP_DIR"
ros2 launch findvp_ros real_robot_capture.launch.py \
  config_file:="$FINDVP_CONFIG" output_dir:="$FINDVP_DIR"
```

Leave the recorder running through survey and search. Stop old recorder
instances first. Changing an environment variable in another terminal does
not redirect a running recorder. The runner now checks its output path before
any movement. Automatic survey requires an empty manifest; don't manually
trigger a test capture into the survey run beforehand.

## 11. Preview the automatic survey — terminal A

```bash
SURVEY_ARGS=(
  --config "$FINDVP_CONFIG"
  --start-pose "$FINDVP_START"
  --capture-manifest "$FINDVP_DIR/observations.json"
  --name rgb_survey
  --survey-baseline 0.04
  --velocity-scale 0.05 --acceleration-scale 0.05
)

ros2 run findvp_ros findvp_survey "${SURVEY_ARGS[@]}" \
  --output-dir "$FINDVP_DIR/survey_preview"
```

No motion or capture occurs. Inspect plans in RViz. The survey uses the recorded
camera orientation and positions 4 cm to either side along optical X. The value
is a sampling distance, not a guarantee of collision clearance or object
visibility. The preview plans from current joints; execution plans each segment
from its measured start and asks for fresh approval.

## 12. Execute the survey — terminal A

Only after model/scene review:

```bash
ros2 run findvp_ros findvp_survey "${SURVEY_ARGS[@]}" \
  --output-dir "$FINDVP_DIR/survey" \
  --required-scene-object findvp_support \
  --required-scene-object findvp_target_keepout \
  --execute --acknowledge-model-reviewed
```

Inspect and type `EXECUTE` for each trajectory. The arm moves to the recorded
start, captures, moves right, captures, moves left, and captures. You do not move
it manually between pictures and do not call `/findvp/capture` yourself.
Keep the object visible, base fixed and scene unchanged. Any failed planning,
execution, capture or status check aborts the survey. A partial survey requires
a new run directory/recorder for a retry.

```bash
python3 -m json.tool "$FINDVP_DIR/survey/rgb_survey_survey.json"
```

Require `status: survey_complete`. The arm stays at the last survey view.

## 13. RGB localization — terminal A

```bash
ros2 run findvp_ros findvp_localize \
  --observations "$FINDVP_DIR/observations.json" \
  --target-image "$FINDVP_TARGET" \
  --target-mask "$FINDVP_MASK" \
  --output-json "$FINDVP_DIR/localization.json" \
  --mask-backend sam --sam-points-per-batch 4

python3 -m json.tool "$FINDVP_DIR/localization.json"
```

First use may download DINO/SAM weights; CPU inference can take minutes. SAM now
decodes four prompts at once to reduce peak memory without changing its grid or
confidence checks. Use batch 1 if memory is tight. It still requires sufficient
texture and shared visible surface between images.

On failure, inspect `$FINDVP_DIR/localization.diagnostics/`: `matching.json`,
selected-mask PNGs, feature coordinates and `failure.json`. No valid localization
file is written on failure. Do not proceed to search or lower geometric rejection
thresholds to get past a failed estimate. A reference mask fixes the earlier
partial-target selection; it cannot guarantee success for every scene.

## 14. Preview triangle search — terminal A

```bash
FINDVP_ARGS=(
  --config "$FINDVP_CONFIG"
  --start-pose "$FINDVP_START"
  --localization "$FINDVP_DIR/localization.json"
  --target-image "$FINDVP_TARGET"
  --target-mask "$FINDVP_MASK"
  --capture-manifest "$FINDVP_DIR/observations.json"
  --name rgb_triangles
  --views 3 --radius 0.40 --max-refinement-depth 2
  --similarity-threshold 0.97
  --velocity-scale 0.05 --acceleration-scale 0.05
  --mask-backend sam --sam-points-per-batch 4
)
ros2 run findvp_ros findvp_search "${FINDVP_ARGS[@]}" \
  --output-dir "$FINDVP_DIR/preview"
```

Inspect initial paths in RViz. The 40 cm radius is an example requiring workspace
review. Full adaptive refinement needs captured images and only occurs during
execution. Simulation CuRobo is not used to command Steve in this adapter.

## 15. Execute triangle search — terminal A

```bash
ros2 run findvp_ros findvp_search "${FINDVP_ARGS[@]}" \
  --output-dir "$FINDVP_DIR/search" \
  --required-scene-object findvp_support \
  --required-scene-object findvp_target_keepout \
  --execute --acknowledge-model-reviewed
```

For each displayed trajectory, review and type `EXECUTE`. The runner performs
move → settle → fresh RGB capture → score → triangle subdivision. Keep camera,
MoveIt, viewer and recorder running. Don't issue competing RViz motions or manual
captures. Execution failures stop the run. If motion termination is unconfirmed,
use the physical stop procedure and don't restart FindVP.

## 16. Read the best observation

```bash
python3 -m json.tool "$FINDVP_DIR/search/rgb_triangles_real_search.json"
```

Inspect `status`, `stop_reason`, `best_score`, `best_observation`. The best record
links score, image, exposure timestamp, measured camera pose and joints. The arm
remains at the last visited pose, not necessarily the best; the base stays locked.
Use the reviewed operator stow procedure afterward. Moving the base/object,
restarting odometry, or changing camera calibration requires a new survey.

## Offline software verification (never launches hardware)

```bash
cd /home/ws/findvp_ss26
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=197 \
  python3 -m pytest -q \
    tests/test_real_robot_deployment.py tests/test_real_rgb_workflow.py \
    tests/test_ros_runtime.py tests/test_ros_capture_integration.py \
    tests/test_ros_action_integration.py tests/test_ros_search_integration.py \
    tests/test_ros_survey_integration.py tests/test_steve_launch.py
```

The plugin-autoload setting prevents Humble's launch-testing plugin from importing
unrelated simulator modules during collection. Fake execution tests use localhost
domain 197 and do not contact Steve's domain 74.

Verification on 2026-09-23: 250 host tests passed (5 ROS-dependent modules skipped),
60 deployment tests passed in Humble, the ROS package built and installed all six
executables, and `pip check` found no broken requirements. Fake survey tests cover
exact-plan execution, three fresh translated captures, and abort after an injected
execution failure. The saved real-image test described above still failed metric
localization; these software results are not a claim of physical readiness.
