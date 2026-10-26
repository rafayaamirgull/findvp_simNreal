# Steve real robot: FindVP RGB triangle search on the known carton box

> **Superseded.** Use the [three-view runbook](steve_three_view_runbook.md): it
> plans with CuRobo in `ur5ebase_link` (the friend's verified chain) instead of the
> local MoveIt model whose RViz display disagreed with the real arm.

Use this route only when the carton box is on **Steve's own top deck**, fixed
relative to `ur5ebase_link`. The friend's `full_pipeline` supplies the tested
hand-eye calibration, joint/frame names and measured tabletop/box dimensions.
FindVP supplies RGB snapshots, target matching and adaptive triangle search.
This route uses no depth images, FoundationPose, SAM3 server, or `/odom`.

This is the authoritative route for the requested same-object transfer.  The
older `steve_real.yaml` survey route failed in the saved `rgb_run_03` report
because Steve's `/odom` sample was stale.  Do **not** work around that by
increasing a timestamp tolerance: a top-deck object is rigidly attached to the
arm base, so this runbook deliberately uses `steve_robot_table.yaml` and removes
odometry from capture and motion preflight.

Steve's joint states are stamped by the onboard computer while RGB images are
stamped by the laptop. In `robot_mounted` mode FindVP therefore verifies joint
messages by fresh local receipt, requires a post-image stationary joint sample,
and uses the latest stationary camera TF. It does not compare the two machines'
absolute clocks. The external-table workflow retains strict exposure-time
synchronization because its metric multi-view localization requires it.

The friend's archived box centre `[0.180309, 0.299400, 0.0635]` m describes
its placement on 2026-09-21. Use it only if you have physically reproduced
that placement; otherwise substitute your current measured centre in all
commands. For an object on a separate room table, use
[the external-table RGB runbook](steve_rgb_runbook.md), which requires odometry
and a survey. The carton box prior here is for locating the search hemisphere
and collision avoidance. The viewpoint choice still comes from actual RGB
image similarity after every arm move.

## A. Build once in the laptop's ROS Humble dev container

```bash
cd /home/ws
source /opt/ros/humble/setup.bash
unset STEVE_IFACE
source /home/ws/steve_env.sh
source /home/ws/install/setup.bash
source /home/ws/.venv-findvp/bin/activate
python3 -m colcon build --symlink-install --base-paths src findvp_ss26 \
  --packages-select neo_mpo_700-2 neo_ur_moveit_config findvp_ros
source /home/ws/install/setup.bash
ros2 pkg executables findvp_ros
cd /home/ws/findvp_ss26
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=/home/ws/findvp_ss26 \
  python3 -m pytest -q tests/test_real_robot_deployment.py
```

The helper auto-detects one connected wired interface. If it reports multiple
candidates, run `ip -br link`, identify the interface whose carrier appears
when Steve's Ethernet cable is connected, then run
`export STEVE_IFACE=<that-interface>` and source `steve_env.sh` again. Never
copy an `enx...` name from another laptop or container instance.
The executable list should contain `findvp_snapshot`, `findvp_select_target`,
`findvp_seed_target`, `findvp_register_scene`, `findvp_capture`, and
`findvp_search`.

Before the hardware session, while internet access is available, download and
instantiate the two perception models in separate processes.  Keeping these as
separate commands avoids retaining both models in one process:

```bash
cd /home/ws/findvp_ss26
python3 -c 'from sample.dinov3_feature_matching import DINOv3FeatureExtractor; DINOv3FeatureExtractor(max_image_size=448); print("DINOv3 ready")'
python3 -c 'from findvp_ros.segmentation import make_mask_provider; make_mask_provider("sam", points_per_batch=4); print("SAM ready")'
```

## B. Run in every new terminal

Use the same run name and box centre in all terminals. Choose an unused run
name for a retry. The example centre below is the friend's archived location;
replace it if the carton was placed elsewhere.

```bash
cd /home/ws
source /opt/ros/humble/setup.bash
source /home/ws/steve_env.sh
source /home/ws/install/setup.bash
source /home/ws/.venv-findvp/bin/activate
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export FINDVP_RUN=box_real_01
export FINDVP_DIR="/home/ws/findvp_runs/$FINDVP_RUN"
export FINDVP_CONFIG=/home/ws/findvp_ss26/config/steve_robot_table.yaml
export FINDVP_TARGET="$FINDVP_DIR/target.png"
export FINDVP_MASK="$FINDVP_DIR/target_mask.png"
export FINDVP_START=/home/ws/findvp_config/box_table_start.yaml
export FINDVP_CENTER_X=0.180309
export FINDVP_CENTER_Y=0.299400
export FINDVP_CENTER_Z=0.0635
mkdir -p "$FINDVP_DIR" /home/ws/findvp_config
```

`FINDVP_RUN` must name a new directory for every attempt.  The existing
`/home/ws/targets/target.png` and `target_mask.png` depict the drill, not the
carton; never mix those files with this run.

If you already captured the carton in an earlier run, you can reuse that
reference image and its mask while writing new observations under the new
`FINDVP_DIR`. Set **both** paths together and verify the pair before seeding:

```bash
export FINDVP_TARGET=/home/ws/findvp_runs/box_01/target.png
export FINDVP_MASK=/home/ws/findvp_runs/box_01/target_mask.png
python3 -c 'import os; from findvp_ros.target_selection import load_reference_mask; load_reference_mask(os.environ["FINDVP_MASK"], os.environ["FINDVP_TARGET"]); print("Carton image and mask match")'
```

The mask selector records the reference image's SHA-256 beside the mask.
Changing either the image or mask independently is rejected. If the old mask
does not accurately cover the whole carton, create a new mask under a new
filename with `findvp_select_target --image "$FINDVP_TARGET" --output
"$FINDVP_DIR/target_mask_v2.png"`, then update `FINDVP_MASK` to that file.

## C. Verify feedback and start RGB camera

Terminal A:

```bash
timeout 10 ros2 topic echo /joint_states sensor_msgs/msg/JointState \
  --qos-reliability best_effort --once \
  --filter "'ur5eshoulder_pan_joint' in m.name"
timeout 10 ros2 topic echo /emergency_stop_state neo_msgs2/msg/EmergencyStopState \
  --qos-reliability best_effort --once
timeout 10 ros2 topic echo /io_and_status_controller/robot_mode \
  ur_dashboard_msgs/msg/RobotMode --qos-reliability reliable \
  --qos-durability transient_local --once
timeout 10 ros2 topic echo /io_and_status_controller/safety_mode \
  ur_dashboard_msgs/msg/SafetyMode --qos-reliability reliable \
  --qos-durability transient_local --once
```

All streams must return current messages. Robot mode must be `RUNNING` (7) and
safety mode `NORMAL` (1) before execution. The onboard PC owns the arm driver;
do not start another driver on the laptop. Terminal B, leave running:

```bash
bash /home/ws/start_camera.sh
```

This starts D415 RGB and publishes the calibrated wrist camera TF. Optional
viewer in terminal C:

```bash
ros2 run rqt_image_view rqt_image_view /d415_calib/d415_calib/color/image_raw
```

Terminal A checks:

```bash
timeout 10 ros2 topic echo /d415_calib/d415_calib/color/image_raw \
  sensor_msgs/msg/Image --qos-reliability best_effort --field header --once
timeout 10 ros2 run tf2_ros tf2_echo ur5ebase_link \
  d415_calib_color_optical_frame_handeye
```

## D. External Control and one MoveIt instance

On the pendant enable **and select** Remote Control, initialize the UR5e,
check `external_control.urp`, keep the pendant stop within reach, and reduce
the speed slider. While stationary, terminal A:

```bash
timeout 10 ros2 service call /dashboard_client/quit std_srvs/srv/Trigger '{}'
timeout 10 ros2 service call /dashboard_client/connect std_srvs/srv/Trigger '{}'
timeout 10 ros2 service call /dashboard_client/raw_request \
  ur_dashboard_msgs/srv/RawRequest "{query: 'is in remote control'}"
timeout 10 ros2 service call /dashboard_client/play std_srvs/srv/Trigger '{}'
timeout 10 ros2 service call /dashboard_client/program_running \
  ur_dashboard_msgs/srv/IsProgramRunning '{}'
timeout 10 ros2 service call /controller_manager/list_controllers \
  controller_manager_msgs/srv/ListControllers '{}'
```

Require remote control true, program running true and
`scaled_joint_trajectory_controller` active. Do not reconnect or play during
a trajectory. If `play` returns `success=False`, stop here. Read-only checks:

```bash
ros2 service call /dashboard_client/get_robot_mode ur_dashboard_msgs/srv/GetRobotMode '{}'
ros2 service call /dashboard_client/get_safety_mode ur_dashboard_msgs/srv/GetSafetyMode '{}'
ros2 service call /dashboard_client/get_loaded_program ur_dashboard_msgs/srv/GetLoadedProgram '{}'
ros2 service call /dashboard_client/program_state ur_dashboard_msgs/srv/GetProgramState '{}'
```

Inspect the teach pendant's exact error or popup before another attempt; the
dashboard `play` response may not report the reason. Terminal D, leave running:

```bash
ros2 launch findvp_ros steve_moveit.launch.py
```

In RViz select `ur_manipulator`. The FindVP RViz preset now uses
`ur5ebase_link` as its Fixed Frame and overlays the onboard `/robot_description`
as a translucent **Live robot** display. Compare that display with MoveIt's
current robot, not its orange planned path. If the two model poses disagree,
stop; do not approve a trajectory. Terminal A:

```bash
ros2 action info /move_action
ros2 action info /execute_trajectory
```

Each must show exactly **one** server. Stop duplicate local MoveIt launches
in their own terminals while the arm is idle. FindVP rejects duplicate
`/move_action` servers.

Before any FindVP execution, compare the **current-state RobotModel** in RViz
with the stationary physical arm, including which side of Steve the elbow and
wrist occupy. A 180-degree disagreement is a stop condition, even if MoveIt
finds a collision-free plan. Check RViz's Fixed Frame and distinguish the
current robot from the orange planned/interactive robot. Then read the live
`/joint_states` by name and compare the six UR joint angles with the pendant.
Do not compensate by negating coordinates or adding 180 degrees to a FindVP
parameter; diagnose the model, joint-state source, or TF first.

## E. Register the collision scene

Terminal A:

```bash
ros2 run findvp_ros findvp_register_scene \
  --target-center "$FINDVP_CENTER_X" "$FINDVP_CENTER_Y" "$FINDVP_CENTER_Z"
ros2 run findvp_ros findvp_register_scene --check
```

This adds Steve's top deck, the fixed front box in the friend's scene, and
an inflated keepout for the carton. Inspect their alignment in RViz. The
front box entry applies only if that obstacle is still present at the same
place. The included wrist camera collision body is a placeholder size;
inspect and adjust the Xacro to enclose the actual camera, bracket and cables
before approving motion. Software cannot verify physical dimensions.
Registration is stored in the currently running MoveIt process. If MoveIt is
restarted, repeat both commands above. `--check` must report an empty
`missing_required` list before preview or execution.

`--check` proves only that the three objects exist. FindVP additionally checks
that the target keepout contains the centre recorded in `known_box.json`.
If it reports `target keepout does not enclose the known carton at its seeded
centre`, inspect the two centres; they must describe the same physical box in
`ur5ebase_link`. Do not remove the keepout or weaken this check.
When `known_box.json` already exists and still describes the physically
measured box location, use it as the **single source** for registration and
checking, avoiding a stale shell centre:

```bash
ros2 run findvp_ros findvp_register_scene \
  --localization "$FINDVP_DIR/known_box.json"
ros2 run findvp_ros findvp_register_scene \
  --check --localization "$FINDVP_DIR/known_box.json"
```

Do not make a stale seed and the collision scene agree merely in software:
first confirm the physical carton has not moved.

The carton keepout radius is the box's **half-diagonal** plus 3 cm placement
uncertainty.  Earlier code accidentally used the full diagonal, producing an
approximately 19 cm radius and rejecting otherwise valid viewpoints; the
registered radius is now approximately 11 cm.

## F. Save a target RGB photo and mask

Using normal supervised robot controls, show the whole carton in the camera
from the view FindVP should reproduce. Wait for the arm to stop. Terminal A:

```bash
ros2 run findvp_ros findvp_snapshot \
  --config "$FINDVP_CONFIG" --output "$FINDVP_TARGET"
ros2 run findvp_ros findvp_select_target \
  --image "$FINDVP_TARGET" --output "$FINDVP_MASK"
```

In the mask window outline the **whole box**, click the first point to close,
then press Enter. Inspect the resulting mask. Use new filenames if repeating:
these tools refuse overwrite. The previous drill reference/mask cannot be
used for this carton.

## G. Record one table-view rest pose

Position the arm once, with reviewed controls, so the camera sees the box and
has space to search. When still, terminal A:

```bash
ros2 run findvp_ros findvp_record_pose \
  --config "$FINDVP_CONFIG" --output "$FINDVP_START" \
  --name box_table_start
sed -n '1,120p' "$FINDVP_START"
```

Recording does not move the arm. In future executions FindVP plans to this
recorded rest pose first. The friend's `scan_home.json` is a visual reference,
not an automatically safe current command. A second recording to the same
filename raises `FileExistsError` by design; inspect the existing file and
reuse it if it is still the intended start pose.

## H. Seed known placement and start capture

Terminal A:

```bash
python3 -c 'import os; from findvp_ros.target_selection import load_reference_mask; load_reference_mask(os.environ["FINDVP_MASK"], os.environ["FINDVP_TARGET"]); print("Reference pair verified")'
ros2 run findvp_ros findvp_seed_target \
  --config "$FINDVP_CONFIG" \
  --target-image "$FINDVP_TARGET" --target-mask "$FINDVP_MASK" \
  --target-center "$FINDVP_CENTER_X" "$FINDVP_CENTER_Y" "$FINDVP_CENTER_Z" \
  --output-json "$FINDVP_DIR/known_box.json"
ros2 run findvp_ros findvp_register_scene \
  --check --localization "$FINDVP_DIR/known_box.json"
```

Only after `findvp_seed_target` succeeds, lock the platform and confirm the
service reports success:

```bash
timeout 10 ros2 service call /lock_platform neo_srvs2/srv/LockPlatform '{}'
```

Require lock success. Terminal E, leave running:

```bash
ros2 launch findvp_ros real_robot_capture.launch.py \
  config_file:="$FINDVP_CONFIG" output_dir:="$FINDVP_DIR"
```

Only one recorder should run. In this robot-mounted mode it stores fresh RGB
images, exposure-time camera poses, and measured arm joints without depth or
odometry. Leave this run's `observations.json` empty until search executes.

## I. Preview and execute the three-view comparison

This run visits the three initial triangle viewpoints exactly once. It saves
one RGB image at each point, ranks all three against the target using the
masked DINOv3/SIFT scorer, prints their scores, and plans back to the joint
pose recorded with the highest-scoring image. It does not subdivide triangles.

The earlier `box_real_01` execution report stopped before motion because the
registered target keepout did not contain the centre in `known_box.json`.
Re-register the scene from that localization and check it before previewing:

```bash
ros2 run findvp_ros findvp_register_scene \
  --localization "$FINDVP_DIR/known_box.json"
ros2 run findvp_ros findvp_register_scene \
  --check --localization "$FINDVP_DIR/known_box.json"
```

The RViz current robot must also match the stationary physical arm. If it
appears rotated by 180 degrees, stop here and resolve the model, joint-state
or TF discrepancy before approving any trajectory.

Terminal A:

```bash
FINDVP_ARGS=(
  --config "$FINDVP_CONFIG"
  --start-pose "$FINDVP_START"
  --localization "$FINDVP_DIR/known_box.json"
  --target-image "$FINDVP_TARGET" --target-mask "$FINDVP_MASK"
  --capture-manifest "$FINDVP_DIR/observations.json"
  --name box_triangles
  --views 3 --radius 0.40 --three-view-only
  --velocity-scale 0.05 --acceleration-scale 0.05
  --mask-backend sam --sam-points-per-batch 4
  --required-scene-object findvp_support
  --required-scene-object findvp_front_box
  --required-scene-object findvp_target_keepout
)
ros2 run findvp_ros findvp_search "${FINDVP_ARGS[@]}" \
  --output-dir "$FINDVP_DIR/preview_three_01"
```

This previews the three candidate plans without moving. Inspect them in RViz.
The friend's box viewpoint range includes 0.40 m, but actual paths must fit
the current scene. Once the robot model, camera body, obstacles, and displayed
paths have been checked, terminal A:

```bash
ros2 run findvp_ros findvp_search "${FINDVP_ARGS[@]}" \
  --output-dir "$FINDVP_DIR/execute_three_01" \
  --execute --acknowledge-model-reviewed
```

The arm first plans to the recorded rest pose. Inspect each displayed path
and type `EXECUTE` to approve it. Each reached view triggers a fresh RGB
capture. After the third capture, the program scores all three images and
plans a separate move to the highest-scoring view. Inspect and approve that
final move too. Any motion, capture, or scoring failure stops the run. The
arm remains at its last view and the base remains locked.

Never pipe or script the word `EXECUTE`; it is the per-trajectory operator
interlock.  Keep the pendant and E-stop reachable throughout the run.

Use a new output directory on each retry. Never delete a failed report just
to reuse its path; it records why that attempt stopped.

## J. Inspect result

```bash
python3 -m json.tool "$FINDVP_DIR/execute_three_01/box_triangles_real_search.json" | less
python3 -m json.tool "$FINDVP_DIR/observations.json" | less
```

Check `levels[0].scores` for all three combined, DINOv3, and SIFT scores,
`winner` for the selected image, and `return_to_winner.status` for `reached`.
The final `status` must be `complete`. Use the reviewed operator stow procedure
afterward. Start a new run if the box or mounting changes or a previous run
failed.
