from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import yaml

from sample.curobo_motion import (
    CuRoboMotionPlanner,
    CuRoboMotionPlanningError,
    _normalize_trajectory_positions,
    validate_collision_aware_robot_config,
)
from sample.workcell_config import ARM_JOINT_ORDER


class FakeTensor:
    def __init__(self, value) -> None:
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class CuRoboMotionPureTests(unittest.TestCase):
    def test_v1_warmup_only_exercises_joint_space_planning(self) -> None:
        class FakeRetract:
            def view(self, *_shape):
                return self

            def clone(self):
                return np.zeros((1, len(ARM_JOINT_ORDER)), dtype=float)

        class FakeJointState:
            def __init__(self, position, joint_names):
                self.position = np.asarray(position, dtype=float)
                self.joint_names = list(joint_names)

            @classmethod
            def from_position(cls, position, joint_names):
                return cls(position, joint_names)

            def clone(self):
                return type(self)(self.position.copy(), self.joint_names)

        class FakePlanConfig:
            def __init__(self, **kwargs):
                self.options = kwargs

        robot_module = ModuleType("curobo.types.robot")
        robot_module.JointState = FakeJointState
        motion_module = ModuleType("curobo.wrap.reacher.motion_gen")
        motion_module.MotionGenPlanConfig = FakePlanConfig
        calls = []
        fake_planner = SimpleNamespace(
            rollout_fn=SimpleNamespace(
                joint_names=ARM_JOINT_ORDER,
                dynamics_model=SimpleNamespace(retract_config=FakeRetract()),
            ),
            plan_single_js=lambda start, goal, config: calls.append(
                (start, goal, config)
            ),
        )

        with patch.dict(
            "sys.modules",
            {
                "curobo.types.robot": robot_module,
                "curobo.wrap.reacher.motion_gen": motion_module,
            },
        ):
            CuRoboMotionPlanner._warmup_v1_joint_space(fake_planner)

        self.assertEqual(3, len(calls))
        self.assertTrue(all(not call[2].options["enable_graph"] for call in calls))
        self.assertTrue(
            all(call[2].options["enable_finetune_trajopt"] for call in calls)
        )
        np.testing.assert_allclose(calls[0][0].position, np.zeros((1, 6)))
        np.testing.assert_allclose(
            calls[0][1].position,
            np.asarray([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        )

    def test_legacy_ik_only_config_is_rejected_for_motion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "robot.yml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "robot_cfg": {
                            "kinematics": {
                                "collision_link_names": [],
                                "collision_spheres": None,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(CuRoboMotionPlanningError, "IK-only"):
                validate_collision_aware_robot_config(config_path)

    def test_collision_aware_v2_config_builds_a_lazy_planner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "robot.yml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "kinematics": {
                            "collision_link_names": ["ur5e_forearm_link"],
                            "collision_spheres": {"ur5e_forearm_link": [[0, 0, 0, 0.05]]},
                        }
                    }
                ),
                encoding="utf-8",
            )
            planner = CuRoboMotionPlanner(
                robot_config_path=config_path,
                scene_model={"cuboid": {"table": {"dims": [1, 1, 0.1]}}},
                expected_joint_names=ARM_JOINT_ORDER,
                clearance_m=0.005,
            )

        self.assertEqual("v2", planner.api_generation)
        self.assertIsNone(planner._planner)

    def test_trajectory_normalization_squeezes_batch_and_reorders_joints(self) -> None:
        canonical = np.asarray(
            (
                (0, 1, 2, 3, 4, 5),
                (10, 11, 12, 13, 14, 15),
            ),
            dtype=float,
        )
        reversed_names = tuple(reversed(ARM_JOINT_ORDER))
        reversed_axis = [ARM_JOINT_ORDER.index(name) for name in reversed_names]
        result = _normalize_trajectory_positions(
            FakeTensor(canonical[:, reversed_axis][None, ...]),
            reversed_names,
            ARM_JOINT_ORDER,
        )

        np.testing.assert_allclose(result, canonical)

    def test_motion_endpoints_are_validated_before_runtime_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "robot.yml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "kinematics": {
                            "collision_link_names": ["link"],
                            "collision_spheres": {"link": [[0, 0, 0, 0.05]]},
                        }
                    }
                ),
                encoding="utf-8",
            )
            planner = CuRoboMotionPlanner(
                robot_config_path=config_path,
                scene_model={"cuboid": {"table": {"dims": [1, 1, 0.1]}}},
                expected_joint_names=ARM_JOINT_ORDER,
                clearance_m=0.005,
            )
            with self.assertRaisesRegex(ValueError, "Start state"):
                planner.plan([0.0] * 5, [0.0] * 6)
            with self.assertRaisesRegex(ValueError, "finite"):
                planner.plan([0.0] * 6, [0.0] * 5 + [float("nan")])

    def test_v1_motion_rejects_unverified_warp_version_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "robot.yml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "robot_cfg": {
                            "kinematics": {
                                "collision_link_names": ["link"],
                                "collision_spheres": {
                                    "link": [[0, 0, 0, 0.05]]
                                },
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            planner = CuRoboMotionPlanner(
                robot_config_path=config_path,
                scene_model={"cuboid": {"table": {"dims": [1, 1, 0.1]}}},
                expected_joint_names=ARM_JOINT_ORDER,
                clearance_m=0.005,
            )
            with patch("sample.curobo_motion.version", return_value="1.15.0"):
                with self.assertRaisesRegex(
                    CuRoboMotionPlanningError, "requires the tested warp-lang 1.5.1"
                ):
                    planner._initialize_v1()


if __name__ == "__main__":
    unittest.main()
