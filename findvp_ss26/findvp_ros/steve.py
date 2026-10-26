"""Steve deployment conventions matched to the student workspace README.

Only MoveIt/RViz are launched locally. The onboard driver owns hardware and TF;
dashboard program startup remains an explicit operator action.
"""

STEVE_MOVEIT_ARGUMENTS = {
    "robot_type": "mpo_700",
    "arm_type": "ur5e",
    "prefix": "ur5e",
    "launch_rviz": "True",
    "use_sim_time": "False",
    "use_gz": "False",
    "use_mock_hardware": "False",
    "use_ur_dc": "False",
    "gripper_type": "",
    "findvp_wrist_camera": "true",
}
