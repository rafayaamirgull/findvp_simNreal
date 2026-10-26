#!/usr/bin/env bash
# start_planner.sh -- CuRobo planning server for the Steve three-view run. Leave running.
#
# Run on the HOST (not inside the ROS dev container: it has no docker and no CuRobo):
#   bash findvp_ss26/real_robot/start_planner.sh            # camera body modelled (default)
#   bash findvp_ss26/real_robot/start_planner.sh --no-camera-spheres
#
# Uses the findvp-curobo:0.7.8 image (same CuRobo version as the friend's pipeline),
# --net=host so the dev container reaches it at http://127.0.0.1:8431. It never
# talks to the robot; it only answers planning requests. Ctrl+C stops it.
set -euo pipefail
if ! command -v docker >/dev/null 2>&1; then
  cat >&2 <<'EOF'
error: 'docker' is not available here -- you are most likely inside the ROS dev container.
The CuRobo planner runs in the findvp-curobo Docker image and must be started from the HOST:

  1. In VS Code: Ctrl+Shift+P -> "Terminal: Create New Integrated Terminal (Local)"
     (or open a normal Ubuntu terminal with Ctrl+Alt+T)
  2. cd to the workspace folder on the host (the folder you opened in VS Code)
  3. bash findvp_ss26/real_robot/start_planner.sh

Both the host and the dev container use --net=host, so everything else keeps running
inside the dev container and reaches the planner at http://127.0.0.1:8431.
EOF
  exit 1
fi
REAL_ROBOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WS=$(cd -- "$REAL_ROBOT_DIR/../.." && pwd)
IMAGE=${FINDVP_CUROBO_IMAGE:-findvp-curobo:0.7.8}
NAME=findvp-planner

if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "error: a container named $NAME already exists (another planner?). Stop it first:" >&2
  echo "  docker rm -f $NAME" >&2
  exit 1
fi

# The warp/CuRobo kernel cache lives in a named volume so restarts skip recompilation.
exec docker run --rm -it --name "$NAME" --gpus all --net=host -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$WS":/home/ws -v findvp_planner_cache:/root/.cache \
  -w /home/ws/findvp_ss26/real_robot \
  --entrypoint python3 "$IMAGE" \
  /home/ws/findvp_ss26/real_robot/curobo_server.py "$@"
