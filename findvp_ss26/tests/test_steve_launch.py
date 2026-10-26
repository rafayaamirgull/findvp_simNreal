"""Inspect launch composition without starting any nodes or connecting to Steve."""
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("launch.actions")
from launch import LaunchContext
from launch.actions import IncludeLaunchDescription
from launch.utilities import perform_substitutions, normalize_to_list_of_substitutions

from findvp_ros.steve import STEVE_MOVEIT_ARGUMENTS


def test_wrapper_only_includes_steves_moveit_with_pinned_real_robot_arguments():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("findvp_steve_launch", root / "launch/steve_moveit.launch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    description = module.generate_launch_description()
    assert len(description.entities) == 1
    included = description.entities[0]
    assert isinstance(included, IncludeLaunchDescription)
    context = LaunchContext()
    def resolve(value):
        return perform_substitutions(context, normalize_to_list_of_substitutions(value))
    arguments = {resolve(key): resolve(value)
                 for key, value in included.launch_arguments}
    assert arguments == STEVE_MOVEIT_ARGUMENTS
