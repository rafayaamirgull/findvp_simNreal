from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config = LaunchConfiguration("config_file")
    output = LaunchConfiguration("output_dir")
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("findvp_ros"), "config", "steve_real.yaml"]
                ),
            ),
            DeclareLaunchArgument("output_dir", default_value="output/real_robot/survey"),
            Node(
                package="findvp_ros",
                executable="findvp_capture",
                name="findvp_capture",
                output="screen",
                parameters=[{"config_file": config, "output_dir": output}],
            ),
        ]
    )

