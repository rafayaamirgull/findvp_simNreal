from glob import glob
from setuptools import find_packages, setup


package_name = "findvp_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("tests", "tests.*")),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/urdf", glob("urdf/*.xacro")),
    ],
    install_requires=["setuptools", "numpy", "Pillow", "PyYAML"],
    zip_safe=False,
    author="FindVP project",
    description="ROS 2 real-robot deployment adapter for FindVP",
    license="Proprietary",
    entry_points={
        "console_scripts": [
            "findvp_capture = findvp_ros.capture_node:main",
            "findvp_localize = findvp_ros.localize_cli:main",
            "findvp_select_target = findvp_ros.target_selection:main",
            "findvp_survey = findvp_ros.survey_node:main",
            "findvp_record_pose = findvp_ros.record_pose_node:main",
            "findvp_search = findvp_ros.search_node:main",
            "findvp_register_scene = findvp_ros.scene_cli:main",
            "findvp_seed_target = findvp_ros.seed_target_cli:main",
            "findvp_snapshot = findvp_ros.snapshot_cli:main",
        ]
    },
)
