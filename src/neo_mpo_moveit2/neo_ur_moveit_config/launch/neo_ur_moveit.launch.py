# Copyright (c) 2021 PickNik, Inc.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#    * Neither the name of the {copyright_holder} nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# Contributor: Pradheep Padmanabhan, Adarsh Karan K P

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
from launch_ros.descriptions import ParameterFile
from moveit_configs_utils import MoveItConfigsBuilder
from pathlib import Path
from neo_ur_moveit_config.launch_common import load_yaml

def launch_setup(context, *args, **kwargs):

    # Initialize Arguments
    robot_typ = str(context.perform_substitution(LaunchConfiguration("robot_type")))
    arm_type = LaunchConfiguration("arm_type")
    ur_dc = LaunchConfiguration("use_ur_dc")
    gripper_type = LaunchConfiguration("gripper_type")

    # General arguments
    moveit_config_package = LaunchConfiguration("moveit_config_package")
    prefix = LaunchConfiguration("prefix")
    use_sim_time = LaunchConfiguration("use_sim_time")
    launch_rviz = LaunchConfiguration("launch_rviz")
    use_gz = LaunchConfiguration("use_gz")
    use_mock = LaunchConfiguration("use_mock_hardware")
    findvp_wrist_camera = LaunchConfiguration("findvp_wrist_camera")
    simulation_enabled = context.perform_substitution(LaunchConfiguration("use_gz")).lower()

    if simulation_enabled == "true":
        use_sim_time = True

    robot_description_pkg = get_package_share_directory('neo_mpo_700-2')
    moveit_config_pkg = get_package_share_directory(moveit_config_package.perform(context))
    # Mapping of robot types to their respective package names
    robot_type_packages = {
        'mpo_700': 'neo_mpo_700-2',
        'mpo_500': 'neo_mpo_500-2',
    }

    # Checking if the user has selected a robot that is valid
    if robot_typ not in robot_type_packages:
        # Incase of an invalid selection
        print("Invalid option, setting mpo_700 by default")
        robot_typ = "mpo_700"

    # Get the robot-specific package if it exists
    robot_description_pkg = get_package_share_directory(robot_type_packages[robot_typ])
    
    # Getting the robot description xacro
    urdf = os.path.join(robot_description_pkg, 'robot_model', f'{robot_typ}.urdf.xacro')
    
    # MoveIt Configuration
    srdf = os.path.join(
        moveit_config_pkg,
        "srdf",
        f"{robot_typ}.srdf.xacro"
    )

    # Controllers Configuration
    controllers_yaml = os.path.join(
        moveit_config_pkg,
        "config",
        "moveit_controllers.yaml",
    )
    controllers_yaml_with_substitutions = ParameterFile(controllers_yaml, allow_substs=True)
    # Evaluate the parameter file to apply dynamic substitutions
    controllers_yaml_with_substitutions.evaluate(context)
    
    # Load the controllers YAML
    controllers_yaml_dict = load_yaml(
        "neo_ur_moveit_config",
        str(controllers_yaml_with_substitutions.param_file)
    )

    # The scaled_joint_trajectory_controller does not work on mock hardware
    use_mock_hardware = context.perform_substitution(LaunchConfiguration("use_mock_hardware")).lower()
    if use_mock_hardware == "true" or simulation_enabled == "true":
        controllers_yaml_dict["moveit_simple_controller_manager"]["scaled_joint_trajectory_controller"]["default"] = False
        controllers_yaml_dict["moveit_simple_controller_manager"]["joint_trajectory_controller"]["default"] = True

    # Joint Limits Configuration
    joint_limits_yaml = os.path.join(
        moveit_config_pkg,
        "config",
        "joint_limits.yaml",
    )
    joint_limits_yaml_with_substitutions = ParameterFile(joint_limits_yaml, allow_substs=True)
    # Evaluate the parameter file to apply dynamic substitutions
    joint_limits_yaml_with_substitutions.evaluate(context)
    joint_limits_yaml = os.path.join(
        moveit_config_pkg,
        "config",
        str(joint_limits_yaml_with_substitutions.param_file)
    )

    moveit_config = (
        MoveItConfigsBuilder(robot_name=f"{robot_typ}", package_name="neo_ur_moveit_config")
        .robot_description_semantic(file_path=srdf, mappings={
            "prefix": prefix,
            "gripper_type": gripper_type,
            })
        .robot_description(file_path=urdf, mappings={
            "use_gz": use_gz,
            "arm_type": arm_type,
            "use_ur_dc": ur_dc,
            "gripper_type": gripper_type,
            "force_abs_paths": use_gz,
            "use_mock_hardware": use_mock,
            "mock_sensor_commands": use_mock,
            "findvp_wrist_camera": findvp_wrist_camera,
            })
        .planning_pipelines(pipelines=["ompl"], default_planning_pipeline="ompl")
        .joint_limits(file_path=joint_limits_yaml)
        .to_moveit_configs()
    )

    moveit_config.trajectory_execution = controllers_yaml_dict

    # Start the actual move_group node/action server
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {"use_sim_time": use_sim_time},
        ],
    )

    # rviz with moveit configuration
    rviz_config_file = PathJoinSubstitution(
        [FindPackageShare(moveit_config_package), "rviz", "view_robot.rviz"]
    )
    rviz_node = Node(
        package="rviz2",
        condition=IfCondition(launch_rviz),
        executable="rviz2",
        name="rviz2_moveit",
        output="log",
        arguments=["-d", rviz_config_file],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
            {
                "use_sim_time": use_sim_time,
            },
        ],
    )

    # Steve's onboard driver already publishes robot_description. Older versions
    # of this launch file used ur_robot_driver/wait_for_robot_description here,
    # but that executable is not part of the ROS 2 Humble driver package.
    nodes_to_start = [move_group_node, rviz_node]

    return nodes_to_start

def generate_launch_description():

    declared_arguments = []

    declared_arguments.append(
            DeclareLaunchArgument(
            'robot_type',
            default_value='mpo_700',
            choices=['', 'mpo_700', 'mpo_500'],
            description='Robot Types\n\t'
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            'arm_type', default_value='',
            choices=['', 'ur5', 'ur10', 'ur5e', 'ur10e', 'ec66', 'cs66'],
            description='Arm Types - Supported Robots [mpo-700, mpo-500]\n\t'        
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            'use_ur_dc', default_value='False',
            description='Set this argument to True if you have an UR arm with DC variant'
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            'gripper_type', default_value='',
            choices=['', '2f_140', '2f_85', 'epick'],
            description='Gripper Types - Supported Robots [mpo-700, mpo-500]\n\t'
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            "moveit_config_package",
            default_value="neo_ur_moveit_config",
            description='MoveIt config package with robot SRDF/XACRO files. Usually the argument\n'
            '\t is not set, it enables use of a custom moveit config.',
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="False",
            description='Make MoveIt to use simulation time.\n'
              '\t This is needed for the trajectory planing in simulation.',
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            "prefix",
            default_value='',
            description='Prefix of the joint names in controllers configuration.\n'
            '\t (same as "arm_type" argument)',
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            "use_gz",
            default_value="False",
            description="Whether to enable Gazebo simulation.",
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            "use_mock_hardware",
            default_value="False",
            description="Indicate whether robot is running with mock hardware mirroring command to its states.",
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            "findvp_wrist_camera", default_value="false",
            description="Include FindVP's calibrated wrist D415 collision link in local MoveIt.",
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument("launch_rviz", default_value="True", description="Launch RViz?")
    )

    return LaunchDescription(declared_arguments + [OpaqueFunction(function=launch_setup)])
