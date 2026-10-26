"""Use Steve's existing MoveIt launch with explicit real UR5e conventions.

Does not start drivers, camera, robot_state_publisher, or a dashboard program.
Do not start this if another MoveGroup for Steve is already running.
"""
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

from findvp_ros.steve import STEVE_MOVEIT_ARGUMENTS


def generate_launch_description():
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution([
                FindPackageShare("neo_ur_moveit_config"), "launch", "neo_ur_moveit.launch.py",
            ])),
            launch_arguments=STEVE_MOVEIT_ARGUMENTS.items(),
        ),
    ])
