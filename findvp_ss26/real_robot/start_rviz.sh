#!/usr/bin/env bash
# start_rviz.sh -- viewer for the three-view run (dev container). Leave running.
#   bash /home/ws/findvp_ss26/real_robot/start_rviz.sh
#
# Pure viewer: the robot is drawn from Steve's ONBOARD /robot_description and /tf,
# i.e. exactly what the real robot reports (no local URDF, no MoveIt). Fixed frame
# is base_link and the initial view looks from BEHIND Steve towards its front
# (+X, red axis), so the screen matches what you see standing behind the robot.
# Nothing here can command the robot.
REAL_ROBOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source /opt/ros/humble/setup.bash
source /home/ws/steve_env.sh
exec rviz2 -d "$REAL_ROBOT_DIR/three_view.rviz"
