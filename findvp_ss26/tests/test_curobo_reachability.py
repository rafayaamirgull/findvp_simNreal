from __future__ import annotations

import importlib.metadata
import math
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest import mock

import numpy as np
import yaml

from sample.curobo_reachability import (
    CUROBO_API_V1,
    CUROBO_API_V2,
    CuRoboIKCandidateBackend,
    CuRoboRuntimeProbe,
    CuRoboUnavailableError,
    DEFAULT_CUROBO_TOOL_FRAME,
    CUROBO_V1_UR5E_JOINT_MAP,
    CUROBO_V1_UR5E_LINK_MAP,
    _parse_version_triplet,
    add_curobo_v1_reference_collision_model,
    build_curobo_scene_model,
    curobo_v1_rotation_metric_from_degrees,
    curobo_v1_rotation_metric_to_degrees,
    detect_curobo_robot_config_api,
    generate_curobo_camera_urdf,
    pack_curobo_goalset,
    probe_curobo_runtime,
    require_curobo_runtime,
    write_curobo_v1_ik_robot_config,
)
from sample.workcell_config import (
    ARM_JOINT_ORDER,
    CAMERA_BODY_HALF_EXTENTS,
    CAMERA_OPTICAL_FROM_PARENT,
    DEFAULT_ROBOT_URDF,
    HOME_JOINT_POSITIONS_RAD,
    KEPT_JOINT_NAMES,
    KEPT_LINK_NAMES,
    ROBOT_BASE_XYZ,
    STEVE_DESCRIPTION_ROOT,
    TABLE_CENTER_XYZ,
    TABLE_LEG_HALF_EXTENTS,
    TABLE_LEG_POSITIONS,
    TABLE_SIZE_XYZ,
    WORKPIECE_BASE_XYZ,
)
from sample.prepare_curobo_robot import (
    EXPECTED_COLLISION_LINKS,
    finalize_curobo_robot_config,
)
from sample.viewpoint_reachability import (
    IKGoal,
    IKProblem,
    ViewpointCandidate,
    quaternion_xyzw_to_matrix,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def rpy_to_matrix(rpy):
    roll, pitch, yaw = rpy
    cx, sx = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(yaw), math.sin(yaw)
    rotation_x = np.asarray(((1, 0, 0), (0, cx, -sx), (0, sx, cx)), dtype=float)
    rotation_y = np.asarray(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)), dtype=float)
    rotation_z = np.asarray(((cz, -sz, 0), (sz, cz, 0), (0, 0, 1)), dtype=float)
    return rotation_z @ rotation_y @ rotation_x


def write_v1_test_config(directory: Path) -> tuple[Path, Path]:
    """Write a minimal existing URDF and its host-safe V1 IK-only config."""

    camera_urdf_path = directory / "ur5e_camera_curobo.urdf"
    camera_urdf_path.write_text(
        '<robot name="v1_schema_fixture"><link name="ur5e_base_link"/></robot>\n',
        encoding="utf-8",
    )
    config_path = directory / "ur5e_camera_curobo.yml"
    write_curobo_v1_ik_robot_config(
        config_path=config_path,
        camera_urdf_path=camera_urdf_path,
        joint_names=ARM_JOINT_ORDER,
        retract_config=[HOME_JOINT_POSITIONS_RAD[name] for name in ARM_JOINT_ORDER],
    )
    return config_path, camera_urdf_path


def make_ik_problem(index: int, goal_count: int = 2) -> IKProblem:
    goals = []
    for goal_index in range(goal_count):
        viewpoint = ViewpointCandidate(
            index=index,
            position_xyz=(0.3 + 0.01 * index, -0.1, 0.4),
            orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            radius_m=0.4,
            azimuth_deg=0.0,
            elevation_deg=30.0,
            roll_deg=90.0 * goal_index,
        )
        goals.append(
            IKGoal(
                camera_viewpoint=viewpoint,
                tool_position_xyz=(0.0, 0.0, 0.0),
                tool_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            )
        )
    return IKProblem(sampled_viewpoint=goals[0].camera_viewpoint, goals=tuple(goals))


def nested_mapping_keys(value) -> set[str]:
    keys = set()
    if isinstance(value, dict):
        keys.update(value)
        for child in value.values():
            keys.update(nested_mapping_keys(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            keys.update(nested_mapping_keys(child))
    return keys


class CuRoboPureAdapterTests(unittest.TestCase):
    def test_v1_container_exposes_cuda_and_nvidia_x11_capabilities(self) -> None:
        dockerfile = (REPOSITORY_ROOT / "Dockerfile.curobo-v1").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display",
            dockerfile,
        )
        self.assertIn("mesa-utils", dockerfile)
        self.assertIn("x11-utils", dockerfile)

    def test_v1_reference_spheres_are_mapped_only_after_geometry_verification(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            target_urdf = directory / "target.urdf"
            generate_curobo_camera_urdf(
                source_urdf=DEFAULT_ROBOT_URDF,
                output_urdf=target_urdf,
                kept_link_names=KEPT_LINK_NAMES,
                kept_joint_names=KEPT_JOINT_NAMES,
                package_description_root=STEVE_DESCRIPTION_ROOT,
                tool_from_optical_camera=CAMERA_OPTICAL_FROM_PARENT,
                camera_body_half_extents=CAMERA_BODY_HALF_EXTENTS,
            )
            target_config = directory / "target.yml"
            write_curobo_v1_ik_robot_config(
                config_path=target_config,
                camera_urdf_path=target_urdf,
                joint_names=ARM_JOINT_ORDER,
                retract_config=[
                    HOME_JOINT_POSITIONS_RAD[name] for name in ARM_JOINT_ORDER
                ],
            )

            reverse_links = {
                target: reference for reference, target in CUROBO_V1_UR5E_LINK_MAP.items()
            }
            reverse_joints = {
                target: reference
                for reference, target in CUROBO_V1_UR5E_JOINT_MAP.items()
            }
            reference_tree = ET.parse(target_urdf)
            reference_root = reference_tree.getroot()
            for link in reference_root.findall("link"):
                if link.attrib.get("name") in reverse_links:
                    link.attrib["name"] = reverse_links[link.attrib["name"]]
            for joint in reference_root.findall("joint"):
                if joint.attrib.get("name") in reverse_joints:
                    joint.attrib["name"] = reverse_joints[joint.attrib["name"]]
                for relation in (joint.find("parent"), joint.find("child")):
                    if relation is not None and relation.attrib.get("link") in reverse_links:
                        relation.attrib["link"] = reverse_links[relation.attrib["link"]]
            reference_urdf = directory / "reference.urdf"
            reference_tree.write(reference_urdf, encoding="utf-8", xml_declaration=True)

            reference_collision_links = [
                "shoulder_link",
                "upper_arm_link",
                "forearm_link",
                "wrist_1_link",
                "wrist_2_link",
                "wrist_3_link",
                "tool0",
            ]
            reference_config = directory / "reference.yml"
            reference_config.write_text(
                yaml.safe_dump(
                    {
                        "robot_cfg": {
                            "kinematics": {
                                "collision_link_names": reference_collision_links,
                                "collision_spheres": {
                                    name: [{"center": [0, 0, 0], "radius": 0.05}]
                                    for name in reference_collision_links
                                },
                                "self_collision_ignore": {
                                    "upper_arm_link": ["shoulder_link"]
                                },
                                "self_collision_buffer": {"wrist_3_link": 0.01},
                            }
                        }
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            add_curobo_v1_reference_collision_model(
                config_path=target_config,
                reference_config_path=reference_config,
                reference_urdf_path=reference_urdf,
                camera_body_half_extents=CAMERA_BODY_HALF_EXTENTS,
            )
            result = yaml.safe_load(target_config.read_text(encoding="utf-8"))

            reference_tool_joint = next(
                joint
                for joint in reference_root.findall("joint")
                if joint.attrib.get("name") == "flange-tool0"
            )
            tool_origin = reference_tool_joint.find("origin")
            original_tool_xyz = tool_origin.attrib["xyz"]
            tool_origin.attrib["xyz"] = "0.01 0 0"
            reference_tree.write(reference_urdf, encoding="utf-8", xml_declaration=True)
            with self.assertRaisesRegex(RuntimeError, "kinematics differ"):
                add_curobo_v1_reference_collision_model(
                    config_path=target_config,
                    reference_config_path=reference_config,
                    reference_urdf_path=reference_urdf,
                    camera_body_half_extents=CAMERA_BODY_HALF_EXTENTS,
                )
            tool_origin.attrib["xyz"] = original_tool_xyz

            reference_shoulder = next(
                link
                for link in reference_root.findall("link")
                if link.attrib.get("name") == "shoulder_link"
            )
            reference_shoulder.find("collision/origin").attrib["xyz"] = "0.1 0 0"
            reference_tree.write(reference_urdf, encoding="utf-8", xml_declaration=True)
            with self.assertRaisesRegex(RuntimeError, "geometry differs"):
                add_curobo_v1_reference_collision_model(
                    config_path=target_config,
                    reference_config_path=reference_config,
                    reference_urdf_path=reference_urdf,
                    camera_body_half_extents=CAMERA_BODY_HALF_EXTENTS,
                )

        kinematics = result["robot_cfg"]["kinematics"]
        self.assertIn("ur5e_camera_body", kinematics["collision_link_names"])
        self.assertEqual(
            {"ur5e_wrist_3_link"},
            set(kinematics["self_collision_ignore"]["ur5e_camera_body"]),
        )
        camera_sphere = kinematics["collision_spheres"]["ur5e_camera_body"][0]
        self.assertAlmostEqual(
            float(np.linalg.norm(CAMERA_BODY_HALF_EXTENTS)), camera_sphere["radius"]
        )
        self.assertTrue(result["findvp_adapter"]["curobo_collision_check"])

    def test_backend_contract_returns_every_optimized_seed(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path, _ = write_v1_test_config(Path(temporary_directory))
            backend = CuRoboIKCandidateBackend(
                robot_config_path=config_path,
                scene_model={},
                world_from_base=np.eye(4),
                expected_joint_names=ARM_JOINT_ORDER,
                num_seeds=7,
            )

        self.assertEqual(7, backend.num_seeds)
        self.assertEqual(7, backend.return_seeds)
        self.assertEqual(CUROBO_API_V1, backend.api_generation)
        self.assertEqual("curobo_v1_legacy_goalset_ik", backend.backend_name)

    def test_v1_config_schema_is_absolute_detectable_and_has_no_collision_model(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path, camera_urdf_path = write_v1_test_config(
                Path(temporary_directory)
            )
            config_text = config_path.read_text(encoding="utf-8")
            data = yaml.safe_load(config_text)
            detected_api = detect_curobo_robot_config_api(config_path)

        self.assertEqual(CUROBO_API_V1, detected_api)
        self.assertEqual({"robot_cfg", "findvp_adapter"}, set(data))
        self.assertNotIn("kinematics", data)
        kinematics = data["robot_cfg"]["kinematics"]
        self.assertEqual("ur5e_base_link", kinematics["base_link"])
        self.assertEqual(DEFAULT_CUROBO_TOOL_FRAME, kinematics["ee_link"])
        self.assertNotIn("tool_frames", kinematics)
        self.assertEqual(camera_urdf_path.resolve(), Path(kinematics["urdf_path"]))
        self.assertTrue(Path(kinematics["urdf_path"]).is_absolute())
        self.assertEqual(
            camera_urdf_path.resolve().parent,
            Path(kinematics["asset_root_path"]),
        )
        self.assertTrue(Path(kinematics["asset_root_path"]).is_absolute())
        self.assertEqual([], kinematics["collision_link_names"])
        self.assertIsNone(kinematics["collision_spheres"])
        self.assertEqual({}, kinematics["extra_collision_spheres"])
        self.assertEqual({}, kinematics["self_collision_ignore"])
        self.assertEqual({}, kinematics["self_collision_buffer"])
        self.assertFalse(kinematics["load_meshes"])
        self.assertNotIn("center:", config_text)
        self.assertNotIn("radius:", config_text)
        self.assertTrue(
            {"center", "radius"}.isdisjoint(nested_mapping_keys(kinematics))
        )

        cspace = kinematics["cspace"]
        self.assertEqual(ARM_JOINT_ORDER, cspace["joint_names"])
        self.assertNotIn("default_joint_position", cspace)
        np.testing.assert_allclose(
            cspace["retract_config"],
            [HOME_JOINT_POSITIONS_RAD[name] for name in ARM_JOINT_ORDER],
        )
        self.assertFalse(data["findvp_adapter"]["curobo_collision_check"])
        self.assertEqual(
            "pybullet", data["findvp_adapter"]["authoritative_endpoint_validator"]
        )

    def test_robot_builder_output_is_pinned_to_workcell_invariants(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "robot.yml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "kinematics": {
                            "base_link": "ur5e_base_link",
                            "tool_frames": [DEFAULT_CUROBO_TOOL_FRAME],
                            "collision_link_names": sorted(EXPECTED_COLLISION_LINKS),
                            "self_collision_ignore": {
                                "ur5e_camera_body": ["ur5e_wrist_3_link"]
                            },
                            "collision_sphere_buffer": 0.0,
                            "cspace": {
                                "joint_names": list(ARM_JOINT_ORDER),
                                "default_joint_position": [0.0] * 6,
                            },
                        }
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            self.assertEqual(CUROBO_API_V2, detect_curobo_robot_config_api(config_path))
            finalize_curobo_robot_config(config_path)
            result = yaml.safe_load(config_path.read_text(encoding="utf-8"))

            unsafe = dict(result)
            unsafe["kinematics"]["self_collision_ignore"][
                "ur5e_camera_body"
            ].append("ur5e_wrist_2_link")
            config_path.write_text(
                yaml.safe_dump(unsafe, sort_keys=False), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "Unsafe camera"):
                finalize_curobo_robot_config(config_path)

        self.assertEqual(0.005, result["kinematics"]["collision_sphere_buffer"])
        np.testing.assert_allclose(
            result["kinematics"]["cspace"]["default_joint_position"],
            [0.0, -1.57, 1.57, -1.57, -1.57, 0.0],
        )

    def test_default_pybullet_module_import_does_not_load_curobo(self):
        repository_root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "import sample.pybullet_tabletop_ur5e_eye_in_hand; "
                    "assert not any(n == 'curobo' or n.startswith('curobo.') "
                    "for n in sys.modules); "
                    "print('lazy')"
                ),
            ],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("lazy", completed.stdout)

    def test_prepare_module_import_does_not_load_pybullet(self):
        repository_root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "import sample.prepare_curobo_robot; "
                    "assert not any(n == 'pybullet' or n.startswith('pybullet.') "
                    "for n in sys.modules); "
                    "print('pure')"
                ),
            ],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("pure", completed.stdout)

    def test_backend_caches_successful_runtime_validation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path, _ = write_v1_test_config(Path(temporary_directory))
            backend = CuRoboIKCandidateBackend(
                robot_config_path=config_path,
                scene_model={},
                world_from_base=np.eye(4),
                expected_joint_names=ARM_JOINT_ORDER,
            )
            sentinel_probe = mock.Mock()
            with mock.patch(
                "sample.curobo_reachability.require_curobo_runtime",
                return_value=sentinel_probe,
            ) as runtime_check:
                first = backend._require_runtime_once()
                second = backend._require_runtime_once()

        self.assertIs(sentinel_probe, first)
        self.assertIs(first, second)
        runtime_check.assert_called_once_with(expected_api=CUROBO_API_V1)

    def test_runtime_rejects_a_detected_gpu_that_cannot_execute_a_cuda_tensor(self):
        runtime = CuRoboRuntimeProbe(
            compatible=True,
            api_generation=CUROBO_API_V1,
            python_version="3.8.10",
            nvidia_driver_version="535.261.03",
            curobo_version="0.7.8",
            issues=(),
        )
        fake_torch = mock.Mock()
        fake_torch.version.cuda = "12.2"
        fake_torch.cuda.is_available.return_value = True
        fake_torch.cuda.get_device_properties.return_value = mock.Mock(
            total_memory=6 * 1024**3,
            major=8,
            minor=6,
        )
        fake_torch.zeros.side_effect = RuntimeError("synthetic CUDA kernel failure")

        with mock.patch(
            "sample.curobo_reachability.probe_curobo_runtime",
            return_value=runtime,
        ), mock.patch(
            "sample.curobo_reachability.importlib.import_module",
            return_value=mock.Mock(),
        ), mock.patch.dict(sys.modules, {"torch": fake_torch}):
            with self.assertRaisesRegex(
                CuRoboUnavailableError, "tensor execution smoke test"
            ):
                require_curobo_runtime(expected_api=CUROBO_API_V1)

        fake_torch.zeros.assert_called_once_with(1, device="cuda:0")

    def test_v1_runtime_requires_sm86_and_synchronizes_device_zero(self):
        runtime = CuRoboRuntimeProbe(
            compatible=True,
            api_generation=CUROBO_API_V1,
            python_version="3.8.10",
            nvidia_driver_version="535.261.03",
            curobo_version="0.7.8",
            issues=(),
        )

        for compute, accepted in (((8, 0), False), ((8, 6), True)):
            with self.subTest(compute=compute):
                fake_torch = mock.Mock()
                fake_torch.version.cuda = "12.2"
                fake_torch.cuda.is_available.return_value = True
                fake_torch.cuda.get_device_properties.return_value = mock.Mock(
                    total_memory=6 * 1024**3,
                    major=compute[0],
                    minor=compute[1],
                )
                cuda_tensor = mock.Mock()
                fake_torch.zeros.return_value = cuda_tensor

                with mock.patch(
                    "sample.curobo_reachability.probe_curobo_runtime",
                    return_value=runtime,
                ), mock.patch(
                    "sample.curobo_reachability.importlib.import_module",
                    return_value=mock.Mock(),
                ), mock.patch.dict(sys.modules, {"torch": fake_torch}):
                    if accepted:
                        selected = require_curobo_runtime(expected_api=CUROBO_API_V1)
                    else:
                        with self.assertRaisesRegex(
                            CuRoboUnavailableError, "need at least 8.6"
                        ):
                            require_curobo_runtime(expected_api=CUROBO_API_V1)

                if accepted:
                    self.assertIs(runtime, selected)
                    fake_torch.zeros.assert_called_once_with(1, device="cuda:0")
                    cuda_tensor.square_.assert_called_once_with()
                    fake_torch.cuda.synchronize.assert_called_once_with(device=0)
                else:
                    fake_torch.zeros.assert_not_called()

    def test_v1_runtime_probe_accepts_python38_and_r535(self):
        for python_version in ("3.8.10", "3.10.14"):
            with self.subTest(python_version=python_version), mock.patch(
                "sample.curobo_reachability.platform.python_version",
                return_value=python_version,
            ), mock.patch(
                "sample.curobo_reachability._read_nvidia_driver_version",
                return_value="535.261.03",
            ), mock.patch(
                "sample.curobo_reachability.importlib.metadata.version",
                return_value="0.7.8",
            ):
                probe = probe_curobo_runtime(expected_api=CUROBO_API_V1)

            self.assertTrue(probe.compatible)
            self.assertEqual(CUROBO_API_V1, probe.api_generation)
            self.assertEqual((), probe.issues)
            self.assertEqual("0.7.8", probe.curobo_version)

    def test_v1_runtime_probe_reports_only_a_missing_package_on_current_host(self):
        package_missing = importlib.metadata.PackageNotFoundError()
        with mock.patch(
            "sample.curobo_reachability.platform.python_version", return_value="3.8.10"
        ), mock.patch(
            "sample.curobo_reachability._read_nvidia_driver_version",
            return_value="535.261.03",
        ), mock.patch(
            "sample.curobo_reachability.importlib.metadata.version",
            side_effect=package_missing,
        ):
            probe = probe_curobo_runtime(expected_api=CUROBO_API_V1)

        self.assertFalse(probe.compatible)
        self.assertEqual(CUROBO_API_V1, probe.api_generation)
        self.assertEqual(1, len(probe.issues))
        self.assertIn("not installed", probe.issues[0])

    def test_v2_runtime_probe_rejects_python38_and_r535(self):
        with mock.patch(
            "sample.curobo_reachability.platform.python_version", return_value="3.8.10"
        ), mock.patch(
            "sample.curobo_reachability._read_nvidia_driver_version",
            return_value="535.261.03",
        ), mock.patch(
            "sample.curobo_reachability.importlib.metadata.version",
            return_value="0.8.1",
        ):
            probe = probe_curobo_runtime(expected_api=CUROBO_API_V2)

        self.assertFalse(probe.compatible)
        self.assertEqual(CUROBO_API_V2, probe.api_generation)
        self.assertEqual(2, len(probe.issues))
        self.assertIn("Python 3.8.10", probe.issues[0])
        self.assertIn("535.261.03", probe.issues[1])
        self.assertEqual((0, 8, 0), _parse_version_triplet("v0.8.0-no-tag"))

    def test_v1_runtime_probe_rejects_unsupported_version_and_api_mismatch(self):
        for installed_version, expected_fragment in (
            ("0.7.7", "unsupported"),
            ("0.8.1", "uses V2 APIs"),
        ):
            with self.subTest(installed_version=installed_version), mock.patch(
                "sample.curobo_reachability.platform.python_version",
                return_value="3.8.10",
            ), mock.patch(
                "sample.curobo_reachability._read_nvidia_driver_version",
                return_value="535.261.03",
            ), mock.patch(
                "sample.curobo_reachability.importlib.metadata.version",
                return_value=installed_version,
            ):
                probe = probe_curobo_runtime(expected_api=CUROBO_API_V1)

            self.assertFalse(probe.compatible)
            self.assertEqual(CUROBO_API_V1, probe.api_generation)
            self.assertEqual(1, len(probe.issues))
            self.assertIn(expected_fragment, probe.issues[0])

    def test_v1_rotation_metric_matches_sine_half_angle_and_round_trips(self):
        self.assertAlmostEqual(
            math.sin(math.radians(0.5)),
            curobo_v1_rotation_metric_from_degrees(1.0),
            places=15,
        )
        self.assertAlmostEqual(
            5.7319679651977244,
            curobo_v1_rotation_metric_to_degrees(0.05),
            places=12,
        )
        for angle_deg in (0.0, 1.0, 45.0, 90.0, 180.0):
            with self.subTest(angle_deg=angle_deg):
                metric = curobo_v1_rotation_metric_from_degrees(angle_deg)
                self.assertAlmostEqual(
                    angle_deg,
                    curobo_v1_rotation_metric_to_degrees(metric),
                    places=12,
                )
        for invalid in (-1.0, 181.0, float("nan"), float("inf")):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    curobo_v1_rotation_metric_from_degrees(invalid)

    def test_result_array_parser_normalizes_shapes_and_reorders_joints(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path, _ = write_v1_test_config(Path(temporary_directory))
            backend = CuRoboIKCandidateBackend(
                robot_config_path=config_path,
                scene_model={},
                world_from_base=np.eye(4),
                expected_joint_names=ARM_JOINT_ORDER,
                num_seeds=2,
            )

            problems = [make_ik_problem(40), make_ik_problem(41)]
            canonical_joint_positions = np.asarray(
                (
                    ((10, 11, 12, 13, 14, 15), (20, 21, 22, 23, 24, 25)),
                    ((30, 31, 32, 33, 34, 35), (40, 41, 42, 43, 44, 45)),
                ),
                dtype=float,
            )
            result_joint_names = tuple(reversed(ARM_JOINT_ORDER))
            result_axis = [ARM_JOINT_ORDER.index(name) for name in result_joint_names]
            result_joint_positions = canonical_joint_positions[..., result_axis]
            success = np.asarray(((True, False), (True, True)), dtype=bool)
            position_error = np.asarray(((0.001, 0.2), (0.002, 0.003)))
            rotation_error = np.asarray(
                (
                    (
                        curobo_v1_rotation_metric_from_degrees(1.0),
                        curobo_v1_rotation_metric_from_degrees(10.0),
                    ),
                    (
                        curobo_v1_rotation_metric_from_degrees(2.0),
                        curobo_v1_rotation_metric_from_degrees(3.0),
                    ),
                )
            )
            goalset_index = np.asarray(((1, 0), (0, 1)), dtype=np.int64)[..., None]

            results = backend._problem_results_from_arrays(
                problems=problems,
                success=success,
                joint_positions=result_joint_positions,
                position_error=position_error,
                rotation_error=rotation_error,
                goalset_index=goalset_index,
                result_joint_names=result_joint_names,
                rotation_error_units="sin_half_angle",
            )

            with self.assertRaisesRegex(RuntimeError, "Unexpected CuRobo IK result shapes"):
                backend._problem_results_from_arrays(
                    problems=problems,
                    success=success[:, :1],
                    joint_positions=result_joint_positions,
                    position_error=position_error,
                    rotation_error=rotation_error,
                    goalset_index=goalset_index,
                    result_joint_names=result_joint_names,
                    rotation_error_units="sin_half_angle",
                )

        self.assertEqual([40, 41], [result.viewpoint_index for result in results])
        self.assertEqual([1, 2], [len(result.candidates) for result in results])
        self.assertEqual([2, 2], [result.attempted_solutions for result in results])
        self.assertEqual(
            {"ik_solver_no_solution": 1}, results[0].solver_failure_counts
        )
        self.assertIsNone(results[1].solver_failure_counts)
        first_candidate = results[0].candidates[0]
        self.assertEqual(1, first_candidate.goal_index)
        np.testing.assert_allclose(
            canonical_joint_positions[0, 0], first_candidate.joint_positions_rad
        )
        self.assertEqual(
            CUROBO_API_V1, first_candidate.solver_provenance["api_generation"]
        )
        self.assertEqual(0, first_candidate.solver_provenance["returned_branch_index"])
        self.assertEqual(
            "sin_half_angle",
            first_candidate.solver_provenance["solver_rotation_error_units"],
        )
        self.assertAlmostEqual(
            1.0, first_candidate.solver_provenance["solver_rotation_error_deg"]
        )
        np.testing.assert_allclose(
            canonical_joint_positions[1, 1],
            results[1].candidates[1].joint_positions_rad,
        )

    def test_goalset_packing_is_viewpoint_major_base_relative_and_wxyz(self):
        def problem(index, positions):
            goals = []
            for goal_index, position in enumerate(positions):
                viewpoint = ViewpointCandidate(
                    index=index,
                    position_xyz=position,
                    orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
                    radius_m=0.4,
                    azimuth_deg=0.0,
                    elevation_deg=30.0,
                    roll_deg=90.0 * goal_index,
                )
                goals.append(
                    IKGoal(
                        camera_viewpoint=viewpoint,
                        tool_position_xyz=(0.0, 0.0, 0.0),
                        tool_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
                    )
                )
            return IKProblem(sampled_viewpoint=goals[0].camera_viewpoint, goals=tuple(goals))

        problems = [
            problem(10, ((1.1, 2.2, 3.3), (1.4, 2.5, 3.6))),
            problem(11, ((-0.1, 0.2, 1.3), (-0.4, 0.5, 1.6))),
        ]
        angle = math.radians(30.0)
        world_from_base = np.eye(4)
        world_from_base[:3, :3] = np.asarray(
            (
                (math.cos(angle), -math.sin(angle), 0.0),
                (math.sin(angle), math.cos(angle), 0.0),
                (0.0, 0.0, 1.0),
            )
        )
        world_from_base[:3, 3] = (0.3, -0.2, 0.75)

        batch = pack_curobo_goalset(problems, world_from_base)

        self.assertEqual((2, 2, 3), batch.positions_base_m.shape)
        self.assertEqual((2, 2, 4), batch.quaternions_wxyz.shape)
        base_from_world = np.linalg.inv(world_from_base)
        for batch_index, ik_problem in enumerate(problems):
            for goal_index, goal in enumerate(ik_problem.goals):
                expected = base_from_world @ np.asarray(
                    (
                        (1.0, 0.0, 0.0, goal.camera_viewpoint.position_xyz[0]),
                        (0.0, 1.0, 0.0, goal.camera_viewpoint.position_xyz[1]),
                        (0.0, 0.0, 1.0, goal.camera_viewpoint.position_xyz[2]),
                        (0.0, 0.0, 0.0, 1.0),
                    )
                )
                np.testing.assert_allclose(
                    batch.positions_base_m[batch_index, goal_index], expected[:3, 3], atol=1e-6
                )
                w, x, y, z = batch.quaternions_wxyz[batch_index, goal_index]
                np.testing.assert_allclose(
                    quaternion_xyzw_to_matrix((x, y, z, w)), expected[:3, :3], atol=1e-6
                )

    def test_scene_matches_pybullet_in_robot_base_coordinates(self):
        tabletop_underside_z = TABLE_CENTER_XYZ[2] - TABLE_SIZE_XYZ[2] * 0.5
        for leg_position in TABLE_LEG_POSITIONS:
            self.assertAlmostEqual(leg_position[2] - TABLE_LEG_HALF_EXTENTS[2], 0.0)
            self.assertAlmostEqual(
                leg_position[2] + TABLE_LEG_HALF_EXTENTS[2],
                tabletop_underside_z,
            )

        with tempfile.TemporaryDirectory() as temporary_directory:
            mesh_path = Path(temporary_directory) / "workpiece.obj"
            mesh_path.touch()
            scene = build_curobo_scene_model(
                workpiece_mesh_path=mesh_path,
                robot_base_xyz=ROBOT_BASE_XYZ,
                table_size_xyz=TABLE_SIZE_XYZ,
                table_center_xyz=TABLE_CENTER_XYZ,
                table_leg_half_extents=TABLE_LEG_HALF_EXTENTS,
                table_leg_positions_xyz=TABLE_LEG_POSITIONS,
                workpiece_base_xyz=WORKPIECE_BASE_XYZ,
            )

        self.assertEqual(5, len(scene["cuboid"]))
        self.assertEqual(list(TABLE_SIZE_XYZ), scene["cuboid"]["tabletop"]["dims"])
        np.testing.assert_allclose(
            scene["cuboid"]["tabletop"]["pose"],
            [0.25, 0.0, -0.025, 1.0, 0.0, 0.0, 0.0],
        )
        np.testing.assert_allclose(
            scene["mesh"]["workpiece"]["pose"],
            [0.45, -0.10, 0.0, 1.0, 0.0, 0.0, 0.0],
        )
        for obstacle in list(scene["cuboid"].values()) + list(scene["mesh"].values()):
            self.assertEqual(7, len(obstacle["pose"]))
            self.assertTrue(np.all(np.isfinite(obstacle["pose"])))

    def test_generated_urdf_has_one_base_root_and_exact_camera_extrinsic(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "ur5e_camera.urdf"
            generate_curobo_camera_urdf(
                source_urdf=DEFAULT_ROBOT_URDF,
                output_urdf=output_path,
                kept_link_names=KEPT_LINK_NAMES,
                kept_joint_names=KEPT_JOINT_NAMES,
                package_description_root=STEVE_DESCRIPTION_ROOT,
                tool_from_optical_camera=CAMERA_OPTICAL_FROM_PARENT,
                camera_body_half_extents=CAMERA_BODY_HALF_EXTENTS,
            )
            root = ET.parse(output_path).getroot()

        links = root.findall("link")
        joints = root.findall("joint")
        child_names = {joint.find("child").attrib["link"] for joint in joints}
        root_names = {link.attrib["name"] for link in links} - child_names
        self.assertEqual(14, len(links))
        self.assertEqual(13, len(joints))
        self.assertEqual({"ur5e_base_link"}, root_names)
        self.assertEqual(6, sum(joint.attrib["type"] == "revolute" for joint in joints))
        self.assertNotIn("world", {link.attrib["name"] for link in links})

        optical_joint = next(
            joint
            for joint in joints
            if joint.find("child").attrib["link"] == DEFAULT_CUROBO_TOOL_FRAME
        )
        origin = optical_joint.find("origin")
        xyz = np.fromstring(origin.attrib["xyz"], sep=" ")
        rpy = np.fromstring(origin.attrib["rpy"], sep=" ")
        np.testing.assert_allclose(xyz, CAMERA_OPTICAL_FROM_PARENT[:3, 3], atol=1e-12)
        np.testing.assert_allclose(
            rpy_to_matrix(rpy), CAMERA_OPTICAL_FROM_PARENT[:3, :3], atol=1e-12
        )

        camera_link = next(
            link for link in links if link.attrib["name"] == "ur5e_camera_body"
        )
        box_size = np.fromstring(
            camera_link.find("collision/geometry/box").attrib["size"], sep=" "
        )
        np.testing.assert_allclose(box_size, 2.0 * np.asarray(CAMERA_BODY_HALF_EXTENTS))
        base_collision_link = next(
            link for link in links if link.attrib["name"] == "ur5e_base_link_inertia"
        )
        self.assertIsNone(base_collision_link.find("collision"))
        for mesh in root.iter("mesh"):
            self.assertNotIn("/workspace", mesh.attrib["filename"])
            self.assertTrue(Path(mesh.attrib["filename"]).exists())


if __name__ == "__main__":
    unittest.main()
