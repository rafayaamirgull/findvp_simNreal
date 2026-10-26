import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

PYBULLET_AVAILABLE = importlib.util.find_spec("pybullet") is not None

from sample.viewpoint_reachability import (
    DEFAULT_COLLISION_CLEARANCE_M,
    IKCandidate,
    IKProblemResult,
    ReachabilityConfig,
    ViewpointCandidate,
    matrix_to_quaternion_xyzw as matrix_to_quat_xyzw,
    quaternion_angular_distance_deg,
    quaternion_xyzw_to_matrix,
)
from sample.curobo_reachability import (
    CUROBO_API_V1,
    CuRoboRuntimeProbe,
    CuRoboUnavailableError,
)
from sample.workcell_config import (
    CAMERA_BODY_HALF_EXTENTS,
    CAMERA_COLLISION_FROM_PARENT,
    CAMERA_OPTICAL_FROM_PARENT,
    CAMERA_PARENT_LINK_NAME,
    DEFAULT_ROBOT_URDF,
    DEFAULT_WORKPIECE_NAME,
    DEFAULT_WORKPIECE_PLY,
    END_EFFECTOR_LINK_NAME,
    TABLE_TOP_Z,
    resolve_workpiece_ply,
)

if PYBULLET_AVAILABLE:
    from sample.pybullet_tabletop_ur5e_eye_in_hand import (
        CapabilityMap,
        IKBackendSelection,
        InteractiveSceneCamera,
        ReachabilityReport,
        RejectedViewpoint,
        TrajectoryPlayer,
        UR5EyeInHandPyBulletEnv,
        build_argument_parser,
        build_selected_candidate_backend,
        convert_ply_to_obj,
        generate_capability_map,
        load_capability_map,
        load_ply_geometry,
        load_ply_mesh,
        project_world_point_to_screen,
        reachability_config_from_args,
        resolve_ik_backend,
        run_viewpoint_reachability,
        save_reachability_report,
        select_nearest_viewpoint_from_screen,
        validate_cli_arguments,
        visualize_capability_map,
    )
    from sample.capture_four_views import (
        build_argument_parser as build_capture_argument_parser,
    )
    from sample.curobo_motion import CuRoboMotionPlan


@unittest.skipUnless(PYBULLET_AVAILABLE, "PyBullet is not installed")
class WorkpieceSelectionTests(unittest.TestCase):
    @staticmethod
    def _write_binary_textured_ply(path: Path) -> None:
        header = (
            b"ply\n"
            b"format binary_little_endian 1.0\n"
            b"element vertex 4\n"
            b"property float x\n"
            b"property float y\n"
            b"property float z\n"
            b"element face 2\n"
            b"property list uchar int vertex_indices\n"
            b"element multi_texture_vertex 4\n"
            b"property uchar tx\n"
            b"property float u\n"
            b"property float v\n"
            b"element multi_texture_face 2\n"
            b"property uchar tx\n"
            b"property uint tn\n"
            b"property list uchar int texture_vertex_indices\n"
            b"end_header\n"
        )
        vertices = np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
            ),
            dtype="<f4",
        )
        geometry_faces = b"".join(
            b"\x03" + np.asarray(indices, dtype="<i4").tobytes()
            for indices in ((0, 1, 2), (0, 2, 3))
        )
        texture_vertices = np.array(
            [
                (0, 0.0, 0.0),
                (0, 1.0, 0.0),
                (0, 1.0, 1.0),
                (0, 0.0, 1.0),
            ],
            dtype=np.dtype([("tx", "u1"), ("u", "<f4"), ("v", "<f4")]),
        )
        texture_face_dtype = np.dtype(
            [
                ("tx", "u1"),
                ("tn", "<u4"),
                ("count", "u1"),
                ("indices", "<i4", (3,)),
            ]
        )
        texture_faces = np.array(
            [(0, 0, 3, (0, 1, 2)), (0, 1, 3, (0, 2, 3))],
            dtype=texture_face_dtype,
        )
        path.write_bytes(
            header
            + vertices.tobytes()
            + geometry_faces
            + texture_vertices.tobytes()
            + texture_faces.tobytes()
        )

    def test_both_simulation_entry_points_default_to_obj_000020(self) -> None:
        for parser in (build_argument_parser(), build_capture_argument_parser()):
            args = parser.parse_args([])
            selected = resolve_workpiece_ply(args.object, args.workpiece_ply)
            self.assertEqual(DEFAULT_WORKPIECE_NAME, selected.stem)
            self.assertEqual(DEFAULT_WORKPIECE_PLY, selected)

    def test_object_flag_accepts_a_bundled_name_with_spaces(self) -> None:
        for parser in (build_argument_parser(), build_capture_argument_parser()):
            args = parser.parse_args(["--object", "Christmas Bear"])
            self.assertEqual(
                "Christmas Bear.ply",
                resolve_workpiece_ply(args.object, args.workpiece_ply).name,
            )

    def test_binary_little_endian_ply_geometry_is_loaded(self) -> None:
        header = (
            b"ply\n"
            b"format binary_little_endian 1.0\n"
            b"element vertex 4\n"
            b"property float x\n"
            b"property float y\n"
            b"property float z\n"
            b"element face 2\n"
            b"property list uchar int vertex_indices\n"
            b"end_header\n"
        )
        vertices = np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
            dtype="<f4",
        )
        faces = b"".join(
            b"\x03" + np.asarray(indices, dtype="<i4").tobytes()
            for indices in ((0, 1, 2), (0, 2, 3))
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            ply_path = Path(temporary_directory) / "binary.ply"
            ply_path.write_bytes(header + vertices.tobytes() + faces)
            loaded_vertices, loaded_triangles = load_ply_geometry(ply_path)

        np.testing.assert_allclose(vertices, loaded_vertices)
        np.testing.assert_array_equal(((0, 1, 2), (0, 2, 3)), loaded_triangles)

    def test_ascii_ply_vertex_colours_are_loaded(self) -> None:
        contents = """ply
format ascii 1.0
element vertex 3
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
element face 1
property list uchar int vertex_indices
end_header
0 0 0 255 0 0
1 0 0 0 255 0
0 1 0 0 0 255
3 0 1 2
"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            ply_path = Path(temporary_directory) / "coloured.ply"
            ply_path.write_text(contents, encoding="ascii")
            vertices, triangles, colours = load_ply_mesh(ply_path)

        np.testing.assert_allclose(
            ((0, 0, 0), (1, 0, 0), (0, 1, 0)), vertices
        )
        np.testing.assert_array_equal(((0, 1, 2),), triangles)
        np.testing.assert_array_equal(
            ((255, 0, 0), (0, 255, 0), (0, 0, 255)), colours
        )

    def test_artec_uv_texture_is_preserved_in_generated_pybullet_obj(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            ply_path = root / "textured.ply"
            texture_path = root / "textured_0.png"
            self._write_binary_textured_ply(ply_path)
            Image.new("RGB", (4, 4), (20, 80, 160)).save(texture_path)

            output_obj, selected_texture, *_ = convert_ply_to_obj(
                ply_path,
                root / "generated",
                1.0,
                0.0,
            )

            self.assertEqual(texture_path, selected_texture)
            contents = output_obj.read_text(encoding="utf-8")
            self.assertEqual(4, contents.count("\nvt "))
            self.assertIn("\nvt 0.00000000 1.00000000\n", contents)
            self.assertIn("\nvt 1.00000000 0.00000000\n", contents)
            self.assertIn("\nf 1/1 2/2 3/3\n", contents)
            self.assertIn("\nf 1/1 3/3 4/4\n", contents)

    def test_artec_uv_texture_reports_a_missing_companion_bitmap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            ply_path = root / "textured.ply"
            self._write_binary_textured_ply(ply_path)

            with self.assertRaisesRegex(
                FileNotFoundError,
                "textured_0.png.*Artec PLY exports store the bitmap separately",
            ):
                convert_ply_to_obj(ply_path, root / "generated", 1.0, 0.0)


@unittest.skipUnless(PYBULLET_AVAILABLE, "PyBullet is not installed")
class PyBulletReachabilityIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.env = UR5EyeInHandPyBulletEnv(
            robot_urdf=DEFAULT_ROBOT_URDF,
            workpiece_ply=DEFAULT_WORKPIECE_PLY,
            workpiece_unit_scale=0.001,
            workpiece_rotation_z_deg=0.0,
            gui=False,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.env.disconnect()

    def setUp(self) -> None:
        self.env.reset_to_home_pose()

    def test_workpiece_vertex_colours_are_attached_as_a_texture(self) -> None:
        self.assertIsNotNone(self.env.generated_workpiece_texture)
        self.assertTrue(self.env.generated_workpiece_texture.is_file())
        self.assertGreaterEqual(self.env.workpiece_texture_id, 0)
        self.assertIn(
            "\nvt ",
            self.env.generated_workpiece_obj.read_text(encoding="utf-8"),
        )

    def test_joint_commands_require_exact_finite_values_and_valid_blend(self) -> None:
        original_joints = self.env.get_joint_positions()
        with self.assertRaisesRegex(ValueError, "exactly 6"):
            self.env.command_joint_positions([0.0] * 5)
        with self.assertRaisesRegex(ValueError, "finite"):
            self.env.command_joint_positions([0.0] * 5 + [float("nan")])
        with self.assertRaisesRegex(ValueError, "blend"):
            self.env.command_joint_positions([0.0] * 6, blend=-0.1)
        np.testing.assert_allclose(self.env.get_joint_positions(), original_joints)

    def test_dense_joint_trajectory_is_revalidated_and_entry_pose_is_restored(self) -> None:
        original = np.asarray(self.env.get_joint_positions(), dtype=float)
        positions = np.stack((original, original, original))

        result = self.env.validate_joint_trajectory(
            positions,
            DEFAULT_COLLISION_CLEARANCE_M,
        )

        self.assertTrue(result.valid)
        self.assertGreaterEqual(result.samples_checked, 3)
        self.assertGreaterEqual(
            result.minimum_clearance_m, DEFAULT_COLLISION_CLEARANCE_M
        )
        np.testing.assert_allclose(self.env.get_joint_positions(), original)

        mismatched = positions.copy()
        mismatched[0, 0] += 0.5
        rejected = self.env.validate_joint_trajectory(
            mismatched,
            DEFAULT_COLLISION_CLEARANCE_M,
        )
        self.assertFalse(rejected.valid)
        self.assertEqual("start_state_mismatch", rejected.failure_reason)
        np.testing.assert_allclose(self.env.get_joint_positions(), original)

    def test_screen_projection_selection_and_trajectory_player_are_deterministic(self) -> None:
        identity_column_major = tuple(np.eye(4).reshape(-1, order="F"))
        self.assertEqual(
            (320.0, 240.0),
            project_world_point_to_screen(
                (0.0, 0.0, 0.0),
                identity_column_major,
                identity_column_major,
                640,
                480,
            ),
        )
        viewpoint = ViewpointCandidate(
            index=12,
            position_xyz=(0.0, 0.0, 0.0),
            orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            radius_m=0.4,
            azimuth_deg=0.0,
            elevation_deg=30.0,
        )
        report = ReachabilityReport(
            sampled=[viewpoint],
            reachable=[],
            rejected=[
                RejectedViewpoint(
                    viewpoint=viewpoint,
                    furthest_failure_stage="ik_solver_no_solution",
                    attempted_solutions=1,
                    failure_counts={"ik_solver_no_solution": 1},
                )
            ],
            object_center_xyz=(0.0, 0.0, 0.0),
            start_joint_positions_rad=[0.0] * 6,
            elapsed_sec=0.0,
        )
        camera_info = (
            640,
            480,
            identity_column_major,
            identity_column_major,
        )
        self.assertIs(
            viewpoint,
            select_nearest_viewpoint_from_screen(report, 326.0, 235.0, camera_info),
        )
        self.assertIsNone(
            select_nearest_viewpoint_from_screen(report, 500.0, 400.0, camera_info)
        )

        plan = CuRoboMotionPlan(
            joint_names=tuple(f"joint_{index}" for index in range(6)),
            joint_positions_rad=np.asarray(([0.0] * 6, [1.0] * 6)),
            interpolation_dt_s=2.0,
            planning_time_s=0.1,
            used_graph=False,
            status="success",
        )
        player = TrajectoryPlayer(plan=plan, start_time=10.0)
        midpoint, complete = player.sample(11.0)
        np.testing.assert_allclose(midpoint, [0.5] * 6)
        self.assertFalse(complete)
        endpoint, complete = player.sample(12.0)
        np.testing.assert_allclose(endpoint, [1.0] * 6)
        self.assertTrue(complete)

    def test_scene_camera_supports_modifier_free_orbit_pan_and_clicks(self) -> None:
        identity_column_major = tuple(np.eye(4).reshape(-1, order="F"))

        class FakePhysics:
            KEY_IS_DOWN = 1
            KEY_WAS_TRIGGERED = 2
            KEY_WAS_RELEASED = 4

            def __init__(self) -> None:
                self.camera_info = (
                    640,
                    480,
                    identity_column_major,
                    identity_column_major,
                    (0.0, 0.0, 1.0),
                    (0.0, 1.0, 0.0),
                    (1.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0),
                    45.0,
                    -35.0,
                    2.0,
                    (0.0, 0.0, 0.0),
                )
                self.camera_resets: list[dict[str, object]] = []

            def getDebugVisualizerCamera(self):
                return self.camera_info

            def resetDebugVisualizerCamera(self, **kwargs) -> None:
                self.camera_resets.append(kwargs)
                self.camera_info = (
                    *self.camera_info[:8],
                    kwargs["cameraYaw"],
                    kwargs["cameraPitch"],
                    kwargs["cameraDistance"],
                    tuple(kwargs["cameraTargetPosition"]),
                )

        physics = FakePhysics()
        camera = InteractiveSceneCamera(physics)

        clicks = camera.process(
            [
                (
                    2,
                    100,
                    100,
                    0,
                    physics.KEY_IS_DOWN | physics.KEY_WAS_TRIGGERED,
                ),
                (1, 120, 110, -1, 0),
                (2, 120, 110, 0, physics.KEY_WAS_RELEASED),
            ]
        )
        self.assertEqual([], clicks)
        self.assertEqual(39.0, physics.camera_resets[-1]["cameraYaw"])
        self.assertEqual(-38.0, physics.camera_resets[-1]["cameraPitch"])

        clicks = camera.process(
            [
                (
                    2,
                    200,
                    210,
                    0,
                    physics.KEY_IS_DOWN | physics.KEY_WAS_TRIGGERED,
                ),
                (2, 200, 210, 0, physics.KEY_WAS_RELEASED),
            ]
        )
        self.assertEqual([(200.0, 210.0)], clicks)

        camera.process(
            [
                (
                    2,
                    300,
                    300,
                    1,
                    physics.KEY_IS_DOWN | physics.KEY_WAS_TRIGGERED,
                ),
                (1, 310, 320, -1, 0),
                (2, 310, 320, 1, physics.KEY_WAS_RELEASED),
            ]
        )
        np.testing.assert_allclose(
            physics.camera_resets[-1]["cameraTargetPosition"],
            (-0.03, 0.0, 0.06),
        )

    def test_environment_operations_are_scoped_to_their_pybullet_client(self) -> None:
        first_positions = self.env.get_joint_positions()
        other = UR5EyeInHandPyBulletEnv(
            robot_urdf=DEFAULT_ROBOT_URDF,
            workpiece_ply=DEFAULT_WORKPIECE_PLY,
            workpiece_unit_scale=0.001,
            workpiece_rotation_z_deg=0.0,
            gui=False,
        )
        try:
            self.assertNotEqual(self.env.client_id, other.client_id)
            other.command_joint_positions([0.1] * 6, blend=1.0)
            np.testing.assert_allclose(self.env.get_joint_positions(), first_positions)
            np.testing.assert_allclose(other.get_joint_positions(), [0.1] * 6)
        finally:
            other.disconnect()

        self.assertTrue(self.env.physics.isConnected())

    def test_wrist_camera_interface_is_rgb_only(self) -> None:
        capture = self.env.capture_wrist_camera(width=64, height=48)

        self.assertEqual((48, 64, 3), capture.rgb.shape)
        self.assertEqual(np.uint8, capture.rgb.dtype)
        self.assertFalse(hasattr(capture, "depth"))
        self.assertFalse(hasattr(capture, "segmentation"))

    def test_camera_target_to_tool_target_round_trip_matches_fk(self) -> None:
        camera_position, camera_rotation = self.env.get_wrist_camera_pose()
        current_camera = ViewpointCandidate(
            index=0,
            position_xyz=tuple(camera_position),
            orientation_xyzw=matrix_to_quat_xyzw(camera_rotation),
            radius_m=0.0,
            azimuth_deg=0.0,
            elevation_deg=0.0,
        )

        target_tool_position, target_tool_orientation = self.env.camera_viewpoint_to_tool_pose(
            current_camera
        )
        actual_tool_position, actual_tool_orientation = self.env.get_link_pose(
            END_EFFECTOR_LINK_NAME
        )

        np.testing.assert_allclose(target_tool_position, actual_tool_position, atol=1e-10)
        self.assertLess(
            quaternion_angular_distance_deg(
                target_tool_orientation, actual_tool_orientation
            ),
            1e-8,
        )
        np.testing.assert_allclose(
            self.env.get_tool_from_camera_transform(),
            CAMERA_OPTICAL_FROM_PARENT,
            atol=5e-7,
        )

    def test_home_configuration_passes_collision_clearance_gate(self) -> None:
        collision_free, clearance, reason = self.env._configuration_clearance(
            DEFAULT_COLLISION_CLEARANCE_M
        )
        self.assertTrue(collision_free)
        self.assertIsNone(reason)
        self.assertGreaterEqual(clearance, DEFAULT_COLLISION_CLEARANCE_M)

    def test_external_candidate_backend_uses_the_shared_pybullet_validator(self) -> None:
        camera_position, camera_rotation = self.env.get_wrist_camera_pose()
        current_camera = ViewpointCandidate(
            index=7,
            position_xyz=tuple(camera_position),
            orientation_xyzw=matrix_to_quat_xyzw(camera_rotation),
            radius_m=0.0,
            azimuth_deg=0.0,
            elevation_deg=0.0,
        )
        original_joints = self.env.get_joint_positions()
        zero_roll_rotation = quaternion_xyzw_to_matrix(
            current_camera.with_roll(0.0).orientation_xyzw
        )
        relative_roll = zero_roll_rotation.T @ camera_rotation
        matching_roll_deg = math.degrees(
            math.atan2(float(relative_roll[1, 0]), float(relative_roll[0, 0]))
        )

        class FakeCandidateBackend:
            backend_name = "fake_batch_candidate_backend"
            metadata = {"purpose": "shared-validator-test"}

            def solve(self, problems, start_joint_positions_rad, config):
                self.assertions = (
                    len(problems),
                    len(problems[0].goals),
                    list(start_joint_positions_rad),
                    config.position_tolerance_m,
                )
                return [
                    IKProblemResult(
                        viewpoint_index=problems[0].sampled_viewpoint.index,
                        candidates=(
                            IKCandidate(
                                goal_index=0,
                                joint_positions_rad=tuple(start_joint_positions_rad),
                                solver_provenance={"returned_branch_index": 0},
                            ),
                        ),
                        attempted_solutions=32,
                    )
                ]

        backend = FakeCandidateBackend()
        config = ReachabilityConfig(
            roll_angles_deg=(matching_roll_deg,),
            seed_count=1,
            refinement_steps=0,
            require_line_of_sight=False,
        )
        report = self.env.filter_reachable_viewpoints(
            [current_camera],
            object_center_xyz=self.env.get_workpiece_center(),
            config=config,
            candidate_backend=backend,
        )

        self.assertEqual((1, 1), backend.assertions[:2])
        np.testing.assert_allclose(backend.assertions[2], original_joints)
        self.assertEqual("fake_batch_candidate_backend", report.solver_backend)
        self.assertEqual("pybullet_endpoint_validation", report.validation_backend)
        self.assertEqual({"purpose": "shared-validator-test"}, report.backend_metadata)
        self.assertEqual(1, len(report.reachable))
        self.assertEqual(32, report.reachable[0].attempted_solutions)
        self.assertIsNone(report.reachable[0].seed_index)
        self.assertEqual(
            {"returned_branch_index": 0}, report.reachable[0].solver_provenance
        )
        np.testing.assert_allclose(self.env.get_joint_positions(), original_joints)

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "accepted.json"
            save_reachability_report(report, config, self.env, output_path)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
        record = payload["viewpoints"][0]
        self.assertTrue(record["reachable"])
        self.assertEqual("fake_batch_candidate_backend", payload["backend"])
        self.assertEqual("pybullet_endpoint_validation", payload["validation_backend"])
        self.assertIsNone(record["seed_index"])
        self.assertEqual(
            {"returned_branch_index": 0}, record["solver_provenance"]
        )
        self.assertEqual(6, len(record["joint_positions_rad"]))

    def test_backend_exception_restores_entry_joint_state(self) -> None:
        camera_position, camera_rotation = self.env.get_wrist_camera_pose()
        current_camera = ViewpointCandidate(
            index=9,
            position_xyz=tuple(camera_position),
            orientation_xyzw=matrix_to_quat_xyzw(camera_rotation),
            radius_m=0.0,
            azimuth_deg=0.0,
            elevation_deg=0.0,
        )
        original_joints = self.env.get_joint_positions()

        class MutatingFailingBackend:
            backend_name = "mutating_failing_backend"
            metadata = {}

            def solve(inner_self, problems, start_joint_positions_rad, config):
                self.env.command_joint_positions([0.1] * 6, blend=1.0)
                raise RuntimeError("synthetic backend failure")

        with self.assertRaisesRegex(RuntimeError, "synthetic backend failure"):
            self.env.filter_reachable_viewpoints(
                [current_camera],
                self.env.get_workpiece_center(),
                ReachabilityConfig(
                    roll_angles_deg=(0.0,),
                    seed_count=1,
                    refinement_steps=0,
                    require_line_of_sight=False,
                ),
                candidate_backend=MutatingFailingBackend(),
            )

        np.testing.assert_allclose(self.env.get_joint_positions(), original_joints)

    def test_camera_collision_proxy_is_anchored_behind_optical_center(self) -> None:
        lower, upper = self.env.physics.getAABB(self.env.camera_marker_body_id)
        actual_collision_center = 0.5 * (
            np.asarray(lower, dtype=float) + np.asarray(upper, dtype=float)
        )
        visible_body_center = np.asarray(
            self.env.physics.getBasePositionAndOrientation(
                self.env.camera_marker_body_id
            )[0],
            dtype=float,
        )
        world_from_parent = self.env.get_link_transform(CAMERA_PARENT_LINK_NAME)
        expected_collision_center = (
            world_from_parent @ CAMERA_COLLISION_FROM_PARENT
        )[:3, 3]
        optical_center, _ = self.env.get_wrist_camera_pose()

        np.testing.assert_allclose(
            actual_collision_center, expected_collision_center, atol=1e-10
        )
        np.testing.assert_allclose(visible_body_center, actual_collision_center, atol=1e-10)
        self.assertAlmostEqual(
            CAMERA_BODY_HALF_EXTENTS[2],
            float(np.linalg.norm(actual_collision_center - optical_center)),
            places=10,
        )

    def test_non_finite_safety_thresholds_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ReachabilityConfig(clearance_m=float("nan")).validate()

        parser = build_argument_parser()
        for option in (
            "--collision-clearance",
            "--ik-position-tolerance",
            "--ik-orientation-tolerance",
            "--view-radius-min",
            "--motion-interpolation-dt",
        ):
            with self.subTest(option=option):
                args = parser.parse_args(["--headless", "--plan-viewpoints", option, "nan"])
                with self.assertRaises(ValueError):
                    validate_cli_arguments(args)

        invalid_attempts = parser.parse_args(
            ["--headless", "--plan-viewpoints", "--motion-plan-attempts", "0"]
        )
        with self.assertRaisesRegex(ValueError, "motion-plan-attempts"):
            validate_cli_arguments(invalid_attempts)

        missing_config = parser.parse_args(
            ["--headless", "--plan-viewpoints", "--ik-backend", "curobo"]
        )
        with self.assertRaisesRegex(ValueError, "--curobo-robot-config"):
            validate_cli_arguments(missing_config)

        self.assertEqual(
            "auto",
            parser.parse_args(["--headless", "--plan-viewpoints"]).ik_backend,
        )

    def test_camera_feed_is_enabled_by_default_and_can_be_disabled(self) -> None:
        parser = build_argument_parser()

        self.assertTrue(parser.parse_args([]).show_camera_feed)
        self.assertTrue(parser.parse_args(["--show-camera-feed"]).show_camera_feed)
        self.assertFalse(parser.parse_args(["--no-camera-feed"]).show_camera_feed)

    def test_capability_map_is_enabled_by_default_and_can_be_disabled(self) -> None:
        parser = build_argument_parser()

        self.assertTrue(parser.parse_args([]).show_capability_map)
        self.assertTrue(parser.parse_args(["--show-capability-map"]).show_capability_map)
        self.assertFalse(parser.parse_args(["--no-capability-map"]).show_capability_map)

    def test_capability_map_load_and_visualization(self) -> None:
        world_from_tcp = self.env.get_link_transform(END_EFFECTOR_LINK_NAME).copy()
        world_from_tcp[:3, 3] = self.env.get_workpiece_center()
        world_from_base = np.eye(4)
        world_from_base[2, 3] = TABLE_TOP_Z
        tcp_from_base = np.linalg.inv(world_from_tcp) @ world_from_base
        arrays = {
            "workspace_xyz_m": np.asarray(((0.4, 0.0, 0.3),)),
            "reachability": np.asarray((1.0,)),
            "manipulability": np.asarray((0.2,)),
            "inverse_reachability": tcp_from_base[None, :, :],
            "inverse_manipulability": np.asarray((0.2,)),
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "capability.npz"
            np.savez(path, **arrays)
            loaded = load_capability_map(path)

        self.assertIsInstance(loaded, CapabilityMap)
        self.assertEqual((1, 1), visualize_capability_map(self.env, loaded))

        entry_joints = self.env.get_joint_positions()
        generated = generate_capability_map(self.env, samples=100, seed=2)
        self.assertGreater(len(generated.workspace_xyz_m), 0)
        self.assertEqual(
            len(generated.inverse_reachability),
            len(generated.inverse_manipulability),
        )
        np.testing.assert_allclose(self.env.get_joint_positions(), entry_joints)

    def test_auto_backend_prefers_v1_and_uses_gpu_search_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "robot.yml"
            config_path.write_text("robot_cfg: {}\n", encoding="utf-8")
            runtime = CuRoboRuntimeProbe(
                compatible=True,
                api_generation=CUROBO_API_V1,
                python_version="3.8.10",
                nvidia_driver_version="535.261.03",
                curobo_version="0.7.8",
                issues=(),
            )
            with mock.patch(
                "sample.pybullet_tabletop_ur5e_eye_in_hand.require_curobo_runtime",
                return_value=runtime,
            ) as runtime_check:
                selection = resolve_ik_backend(
                    "auto", config_path, message_sink=lambda message: None
                )

        args = build_argument_parser().parse_args(["--headless", "--plan-viewpoints"])
        config = reachability_config_from_args(args, selection.effective_backend)
        self.assertTrue(selection.uses_curobo)
        self.assertEqual("prefer_curobo_v1_then_pybullet", selection.policy)
        self.assertFalse(selection.report_metadata()["automatic_fallback_used"])
        self.assertEqual((0.0,), config.roll_angles_deg)
        self.assertEqual(32, config.seed_count)
        self.assertEqual(0, config.refinement_steps)
        runtime_check.assert_called_once_with(expected_api=CUROBO_API_V1)

        unresolved_args = build_argument_parser().parse_args(
            ["--headless", "--plan-viewpoints"]
        )
        with self.assertRaisesRegex(ValueError, "effective IK backend"):
            reachability_config_from_args(unresolved_args)

    def test_auto_backend_falls_back_only_for_runtime_unavailability(self) -> None:
        messages = []
        with mock.patch(
            "sample.pybullet_tabletop_ur5e_eye_in_hand.require_curobo_runtime",
            side_effect=CuRoboUnavailableError("CUDA device is unavailable"),
        ) as runtime_check:
            selection = resolve_ik_backend("auto", None, message_sink=messages.append)

        args = build_argument_parser().parse_args(["--headless", "--plan-viewpoints"])
        config = reachability_config_from_args(args, selection.effective_backend)
        self.assertFalse(selection.uses_curobo)
        self.assertEqual("pybullet", selection.effective_backend)
        self.assertEqual(3, config.seed_count)
        self.assertEqual(1, config.refinement_steps)
        self.assertIn("CUDA device is unavailable", selection.fallback_reason)
        self.assertIn("falling back", messages[0])
        runtime_check.assert_called_once_with(expected_api=CUROBO_API_V1)

    def test_explicit_backend_modes_do_not_silently_fallback(self) -> None:
        with mock.patch(
            "sample.pybullet_tabletop_ur5e_eye_in_hand.require_curobo_runtime"
        ) as runtime_check:
            selection = resolve_ik_backend(
                "pybullet", None, message_sink=lambda message: None
            )
        self.assertEqual("pybullet", selection.effective_backend)
        runtime_check.assert_not_called()

        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "robot.yml"
            config_path.write_text("robot_cfg: {}\n", encoding="utf-8")
            with mock.patch(
                "sample.pybullet_tabletop_ur5e_eye_in_hand.require_curobo_runtime",
                side_effect=CuRoboUnavailableError("no CUDA"),
            ):
                with self.assertRaisesRegex(CuRoboUnavailableError, "no CUDA"):
                    resolve_ik_backend(
                        "curobo", config_path, message_sink=lambda message: None
                    )

        with self.assertRaisesRegex(
            FileNotFoundError, "CuRobo robot config not found"
        ):
            resolve_ik_backend(
                "curobo",
                Path("/definitely/missing/curobo-v2.yml"),
                message_sink=lambda message: None,
            )

    def test_cli_validation_ignores_knobs_for_a_forced_other_backend(self) -> None:
        parser = build_argument_parser()
        pybullet_args = parser.parse_args(
            [
                "--headless",
                "--plan-viewpoints",
                "--ik-backend",
                "pybullet",
                "--curobo-seeds",
                "0",
            ]
        )
        validate_cli_arguments(pybullet_args)

        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "robot.yml"
            config_path.write_text("robot_cfg: {}\n", encoding="utf-8")
            curobo_args = parser.parse_args(
                [
                    "--headless",
                    "--plan-viewpoints",
                    "--ik-backend",
                    "curobo",
                    "--curobo-robot-config",
                    str(config_path),
                    "--ik-seeds",
                    "0",
                    "--ik-refinement-steps",
                    "-1",
                ]
            )
            validate_cli_arguments(curobo_args)

        for option in ("--ik-seeds", "--curobo-seeds"):
            auto_args = parser.parse_args(
                ["--headless", "--plan-viewpoints", option, "0"]
            )
            with self.subTest(option=option), self.assertRaises(ValueError):
                validate_cli_arguments(auto_args)

    def test_selected_backend_factory_builds_only_the_resolved_curobo_backend(
        self,
    ) -> None:
        pybullet_selection = IKBackendSelection(
            requested_backend="auto",
            effective_backend="pybullet",
            policy="prefer_curobo_v1_then_pybullet",
            fallback_reason="no CUDA",
        )
        with mock.patch(
            "sample.pybullet_tabletop_ur5e_eye_in_hand.build_curobo_scene_model"
        ) as scene_builder, mock.patch(
            "sample.pybullet_tabletop_ur5e_eye_in_hand.CuRoboIKCandidateBackend"
        ) as backend_class:
            self.assertIsNone(
                build_selected_candidate_backend(self.env, pybullet_selection)
            )
        scene_builder.assert_not_called()
        backend_class.assert_not_called()

        runtime = CuRoboRuntimeProbe(
            compatible=True,
            api_generation=CUROBO_API_V1,
            python_version="3.10.12",
            nvidia_driver_version="535.261.03",
            curobo_version="0.7.8",
            issues=(),
        )
        config_path = Path("/tmp/curobo-v1.yml")
        curobo_selection = IKBackendSelection(
            requested_backend="auto",
            effective_backend="curobo",
            policy="prefer_curobo_v1_then_pybullet",
            curobo_config_path=config_path,
            runtime_probe=runtime,
        )
        scene_model = {"world": "sentinel"}
        backend = mock.Mock()
        with mock.patch(
            "sample.pybullet_tabletop_ur5e_eye_in_hand.build_curobo_scene_model",
            return_value=scene_model,
        ) as scene_builder, mock.patch(
            "sample.pybullet_tabletop_ur5e_eye_in_hand.CuRoboIKCandidateBackend",
            return_value=backend,
        ) as backend_class:
            selected = build_selected_candidate_backend(
                self.env, curobo_selection, curobo_seeds=17
            )

        self.assertIs(backend, selected)
        scene_builder.assert_called_once()
        backend_class.assert_called_once_with(
            robot_config_path=config_path,
            scene_model=scene_model,
            world_from_base=mock.ANY,
            expected_joint_names=mock.ANY,
            num_seeds=17,
        )

    def test_auto_mode_does_not_hide_config_errors_after_v1_preflight(self) -> None:
        runtime = CuRoboRuntimeProbe(
            compatible=True,
            api_generation=CUROBO_API_V1,
            python_version="3.10.12",
            nvidia_driver_version="535.261.03",
            curobo_version="0.7.8",
            issues=(),
        )
        with tempfile.TemporaryDirectory() as temporary_directory, mock.patch(
            "sample.pybullet_tabletop_ur5e_eye_in_hand.require_curobo_runtime",
            return_value=runtime,
        ):
            missing_config = Path(temporary_directory) / "missing.yml"
            with self.assertRaises(FileNotFoundError):
                resolve_ik_backend(
                    "auto", missing_config, message_sink=lambda message: None
                )

            v2_config = Path(temporary_directory) / "v2.yml"
            v2_config.write_text("kinematics: {}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "only prefers CuRobo V1"):
                resolve_ik_backend(
                    "auto", v2_config, message_sink=lambda message: None
                )

    def test_report_schema_is_finite_and_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "viewpoints.json"
            report = run_viewpoint_reachability(
                env=self.env,
                count=1,
                radius_min_m=0.30,
                radius_max_m=0.50,
                elevation_min_deg=12.0,
                elevation_max_deg=82.0,
                config=ReachabilityConfig(
                    roll_angles_deg=(0.0,),
                    seed_count=1,
                    refinement_steps=1,
                    require_line_of_sight=False,
                ),
                output_path=output_path,
                visualize=False,
                backend_selection_metadata={
                    "requested_backend": "auto",
                    "effective_backend": "pybullet",
                    "automatic_fallback_used": True,
                    "fallback_reason": "test CUDA unavailable",
                },
            )

            payload = json.loads(
                output_path.read_text(encoding="utf-8"),
                parse_constant=lambda value: self.fail(f"Non-finite JSON value: {value}"),
            )

        self.assertEqual(1, len(report.sampled))
        self.assertEqual(1, payload["sampled_count"])
        self.assertEqual("pybullet_multistart_ik", payload["backend"])
        self.assertEqual("pybullet_endpoint_validation", payload["validation_backend"])
        self.assertEqual(1, payload["backend_metadata"]["roll_goal_count"])
        self.assertEqual(
            "auto", payload["backend_metadata"]["selection"]["requested_backend"]
        )
        self.assertEqual(
            "test CUDA unavailable",
            payload["backend_metadata"]["selection"]["fallback_reason"],
        )
        self.assertEqual(6, len(payload["robot"]["joint_names"]))
        self.assertEqual(4, len(payload["world"]["table_leg_positions_xyz_m"]))
        self.assertEqual(0.30, payload["sampling"]["requested_radius_min_m"])
        self.assertGreaterEqual(
            payload["sampling"]["effective_radius_min_m"],
            payload["sampling"]["requested_radius_min_m"],
        )


if __name__ == "__main__":
    unittest.main()
