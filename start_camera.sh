#!/usr/bin/env bash
# start_camera.sh -- the wrist D415 + its calibrated hand-eye extrinsic, in ONE terminal.
# Leave running. Ctrl+C stops both.
#   bash start_camera.sh
#
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# Keep this script self-contained. The old version sourced config.sh, which is not
# part of this workspace and caused an immediate "command not found" failure.
if [[ -f /opt/ros/humble/setup.bash ]]; then
  source /opt/ros/humble/setup.bash
fi
if [[ -f "$SCRIPT_DIR/install/setup.bash" ]]; then
  source "$SCRIPT_DIR/install/setup.bash"
fi
# Put the camera on the same DDS domain as Steve when the workspace helper is
# available. This replaces the missing `setup_ros_env live` call.
if [[ -f "$SCRIPT_DIR/steve_env.sh" ]]; then
  source "$SCRIPT_DIR/steve_env.sh"
fi

set -u

if ! command -v ros2 >/dev/null 2>&1; then
  echo "error: ROS 2 is not installed or its environment could not be sourced" >&2
  exit 1
fi

# The RealSense backend opens V4L2 nodes as well as the USB device. A privileged
# container does not bypass their Unix mode bits for a non-root process.
mapfile -t REALSENSE_VIDEO_NODES < <(
  for sys_node in /sys/class/video4linux/video*; do
    [[ -e "$sys_node" ]] || continue
    if grep -q 'RealSense' "$sys_node/name" 2>/dev/null; then
      printf '/dev/%s\n' "${sys_node##*/}"
    fi
  done
)

if ((${#REALSENSE_VIDEO_NODES[@]} == 0)); then
  echo "error: no Intel RealSense video devices are visible in the container" >&2
  echo "Reconnect the D415, then rebuild/reopen the dev container if necessary." >&2
  exit 1
fi

for video_node in "${REALSENSE_VIDEO_NODES[@]}"; do
  # Privileged containers do not automatically receive device nodes created
  # after container startup. Recreate only RealSense nodes confirmed via sysfs.
  if [[ ! -e "$video_node" ]]; then
    node_name=${video_node##*/}
    device_number=$(<"/sys/class/video4linux/$node_name/dev")
    if [[ "$node_name" =~ ^video[0-9]+$ && "$device_number" =~ ^([0-9]+):([0-9]+)$ ]]; then
      sudo mknod "$video_node" c "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"
      sudo chown root:video "$video_node"
      sudo chmod 660 "$video_node"
    fi
  fi

  if [[ ! -e "$video_node" ]]; then
    echo "error: $video_node is visible in sysfs but missing from the container's /dev" >&2
    exit 1
  fi
  if [[ ! -r "$video_node" || ! -w "$video_node" ]]; then
    echo "error: cannot access $video_node (the container user needs the video group)" >&2
    echo "Rebuild the dev container after the Dockerfile change, or run: newgrp video" >&2
    exit 1
  fi
done

# Publishing the calibrated hand-eye transform is optional. Set HANDEYE_FILE to
# override its location. Accept both layouts used by this workspace.
if [[ -z "${HANDEYE_FILE:-}" ]]; then
  if [[ -f "$SCRIPT_DIR/data/handeye_result.json" ]]; then
    HANDEYE_FILE="$SCRIPT_DIR/data/handeye_result.json"
  else
    HANDEYE_FILE="$SCRIPT_DIR/handeye_result.json"
  fi
fi
TF_PID=""
if [[ -f "$HANDEYE_FILE" ]]; then
  if ! read -r X Y Z QX QY QZ QW < <(
    python3 - "$HANDEYE_FILE" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    result = json.load(stream)
print(*result["translation_m"], *result["quaternion_xyzw"])
PY
  ); then
    echo "error: failed to read hand-eye calibration: $HANDEYE_FILE" >&2
    exit 1
  fi

  # Use a distinct child frame from the frame published by the camera driver.
  echo "hand-eye static TF: ur5etool0 -> d415_calib_color_optical_frame_handeye"
  ros2 run tf2_ros static_transform_publisher \
    --x "$X" --y "$Y" --z "$Z" --qx "$QX" --qy "$QY" --qz "$QZ" --qw "$QW" \
    --frame-id ur5etool0 --child-frame-id d415_calib_color_optical_frame_handeye \
    >/dev/null 2>&1 &
  TF_PID=$!
else
  echo "warning: no hand-eye calibration at $HANDEYE_FILE; starting camera without its static TF" >&2
fi

cleanup() {
  if [[ -n "$TF_PID" ]]; then
    kill "$TF_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# NOTE: output goes to the terminal only -- an unbounded log file here once grew to 33GB.
echo "starting D415 (RGB only, namespace d415_calib) -- Ctrl+C to stop"
ros2 launch realsense2_camera rs_launch.py camera_name:=d415_calib camera_namespace:=d415_calib \
  enable_color:=true enable_depth:=false \
  enable_infra:=false enable_infra1:=false enable_infra2:=false \
  align_depth.enable:=false pointcloud.enable:=false
