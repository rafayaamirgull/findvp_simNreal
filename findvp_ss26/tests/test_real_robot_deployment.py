import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from findvp_ros.configuration import load_named_joint_pose, load_robot_configuration
from findvp_ros.geometry import camera_pose_to_tool_pose, fixed_joint_transform_from_urdf, transform_matrix
from findvp_ros.geometry import camera_mount_from_urdf
from findvp_ros.image_conversion import ros_image_to_pil
from findvp_ros.manifest import (
    append_observation,
    camera_matrix_from_info,
    make_observation_record,
    ordered_joint_positions,
    ordered_joint_velocities,
)
from findvp_ros.triangle_search import hemisphere_triangles, subdivide_spherical_triangle
from findvp_ros.steve import STEVE_MOVEIT_ARGUMENTS
from findvp_ros.search_node import _argument_parser, _validate_arguments
from findvp_ros.scene_cli import known_center, scene_geometry
from findvp_ros.seed_target_cli import main as seed_target
from findvp_ros.search_node import _load_localization
from findvp_ros.runtime import collision_primitive_center
from findvp_ros.steve_workcell import KNOWN_CARTON_DIMS_M, known_carton_keepout_radius
from PIL import Image
import xml.etree.ElementTree as ET


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_collision_center_uses_object_pose_and_primitive_pose():
    def pose(x, y, yaw_quaternion):
        return SimpleNamespace(
            position=SimpleNamespace(x=x, y=y, z=0.0),
            orientation=SimpleNamespace(x=0.0, y=0.0,
                                        z=yaw_quaternion[0], w=yaw_quaternion[1]),
        )

    quarter_turn = (np.sqrt(0.5), np.sqrt(0.5))
    obj = SimpleNamespace(
        pose=pose(0.2, 0.3, quarter_turn),
        primitive_poses=[pose(0.1, 0.0, (0.0, 1.0))],
    )
    target_from_header = transform_matrix((1.0, 2.0, 0.0),
                                          (0.0, 0.0, *quarter_turn))
    np.testing.assert_allclose(
        collision_primitive_center(obj, target_from_header),
        (0.6, 2.2, 0.0), atol=1e-10,
    )


def test_fixed_urdf_mount_transform_keeps_cabinet_offset_and_yaw():
    model = ET.fromstring(
        '<robot><joint name="cabinet_joint" type="fixed">'
        '<parent link="base_link"/><child link="cabinet_link"/>'
        '<origin xyz="0.072 0 0.35" rpy="0 0 0"/></joint>'
        '<joint name="arm_joint" type="fixed">'
        '<parent link="cabinet_link"/><child link="ur5ebase_link"/>'
        '<origin xyz="0.133 0 0.416" rpy="0 0 -1.5707963267948966"/>'
        '</joint></robot>'
    )
    base_from_arm = (
        fixed_joint_transform_from_urdf(model, "base_link", "cabinet_link")
        @ fixed_joint_transform_from_urdf(model, "cabinet_link", "ur5ebase_link")
    )
    np.testing.assert_allclose(base_from_arm[:3, 3], (.205, 0.0, .766))
    np.testing.assert_allclose(base_from_arm[:3, :3],
                               ((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
                               atol=1e-10)


def test_steve_moveit_launch_arguments_match_robot_configuration():
    config = load_robot_configuration(REPO_ROOT / "config" / "steve_real.yaml")
    assert STEVE_MOVEIT_ARGUMENTS["robot_type"] == "mpo_700"
    assert STEVE_MOVEIT_ARGUMENTS["arm_type"] == "ur5e"
    prefix = STEVE_MOVEIT_ARGUMENTS["prefix"]
    assert prefix == "ur5e"
    assert config.tool_frame == prefix + "tool0"
    assert all(name.startswith(prefix) for name in config.joint_names)
    for flag in ("use_sim_time", "use_gz", "use_mock_hardware"):
        assert STEVE_MOVEIT_ARGUMENTS[flag] == "False"


def test_steve_configuration_matches_live_names_and_handeye():
    config = load_robot_configuration(REPO_ROOT / "config" / "steve_real.yaml")
    assert config.planning_frame == "ur5ebase_link"
    assert config.camera_frame.endswith("_handeye")
    assert config.joint_names[0] == "ur5eshoulder_pan_joint"
    assert all("ur5e_" not in name for name in config.joint_names)
    assert np.allclose(config.tool_from_camera[:3, 3], [0.0348268212, -0.0039878706, 0.0655307955])


def test_robot_mounted_config_uses_same_calibration_without_odom():
    old = load_robot_configuration(REPO_ROOT / "config" / "steve_real.yaml")
    mounted = load_robot_configuration(REPO_ROOT / "config" / "steve_robot_table.yaml")
    assert old.base_tracking_mode == "odom"
    assert mounted.base_tracking_mode == "robot_mounted"
    assert mounted.planning_frame == old.planning_frame
    assert np.allclose(mounted.tool_from_camera, old.tool_from_camera)


def test_known_box_seed_preserves_reference_and_scene_center(tmp_path):
    image = tmp_path / "target.png"
    mask = tmp_path / "mask.png"
    Image.new("RGB", (8, 8), "white").save(image)
    Image.new("L", (8, 8), 255).save(mask)
    output = tmp_path / "known.json"
    config = REPO_ROOT / "config" / "steve_robot_table.yaml"
    seed_target(["--config", str(config), "--target-image", str(image),
                 "--target-mask", str(mask), "--target-center", ".18", ".30", ".0635",
                 "--output-json", str(output)])
    report = json.loads(output.read_text())
    assert report["object_prior_used"] is True
    assert report["base_reference"]["mode"] == "robot_mounted"
    np.testing.assert_allclose(_load_localization(output, "ur5ebase_link"), [.18, .30, .0635])
    assert known_center(output, "ur5ebase_link") == pytest.approx((.18, .30, .0635))
    with pytest.raises(ValueError, match="frame differs"):
        known_center(output, "map")
    geometry = scene_geometry([.18, .30, .0635], .03)
    assert geometry[-1][0] == "findvp_target_keepout"
    expected = 0.5 * np.linalg.norm(KNOWN_CARTON_DIMS_M) + .03
    assert geometry[-1][3][0] == pytest.approx(expected)
    assert geometry[-1][3][0] == pytest.approx(known_carton_keepout_radius(.03))


def test_known_box_keepout_uses_half_diagonal_and_rejects_bad_uncertainty():
    assert known_carton_keepout_radius(0.0) == pytest.approx(
        0.5 * np.linalg.norm(KNOWN_CARTON_DIMS_M)
    )
    with pytest.raises(ValueError, match="finite and nonnegative"):
        known_carton_keepout_radius(-0.001)


def test_named_start_pose_is_ordered_by_configured_joint_names(tmp_path):
    joints = ("joint_b", "joint_a")
    path = tmp_path / "start.yaml"
    path.write_text(
        "name: table_view_start\n"
        "joint_positions_rad:\n"
        "  joint_a: 1.0\n"
        "  joint_b: 2.0\n",
        encoding="utf-8",
    )
    pose = load_named_joint_pose(path, joints)
    assert pose.name == "table_view_start"
    assert pose.joint_positions == (2.0, 1.0)
    with pytest.raises(ValueError, match="joint names differ"):
        load_named_joint_pose(path, ("joint_a", "joint_c"))


def test_hardware_execution_requires_recorded_start_pose():
    args = _argument_parser().parse_args([
        "--config", "robot.yaml",
        "--localization", "localization.json",
        "--target-image", "target.png",
        "--output-dir", "output",
        "--capture-manifest", "observations.json",
        "--required-scene-object", "table",
        "--execute",
        "--acknowledge-model-reviewed",
    ])
    with pytest.raises(ValueError, match="operator-recorded --start-pose"):
        _validate_arguments(args)
    args.start_pose = Path("table_view_start.yaml")
    with pytest.raises(ValueError, match="--target-mask"):
        _validate_arguments(args)
    args.target_mask = Path("target_mask.png")
    _validate_arguments(args)


def test_three_view_mode_rejects_other_candidate_counts():
    args = _argument_parser().parse_args([
        "--config", "robot.yaml",
        "--localization", "localization.json",
        "--target-image", "target.png",
        "--output-dir", "output",
        "--capture-manifest", "observations.json",
        "--three-view-only",
        "--views", "4",
    ])
    with pytest.raises(ValueError, match="requires --views 3"):
        _validate_arguments(args)
    args.views = 3
    _validate_arguments(args)


def test_camera_target_is_converted_to_tool_target():
    base_from_camera = transform_matrix((0.4, -0.1, 0.6), (0, 0, 0, 1))
    tool_from_camera = transform_matrix((0.03, 0, 0.06), (0, 0, 0, 1))
    result = camera_pose_to_tool_pose(base_from_camera, tool_from_camera)
    assert np.allclose(result[:3, 3], [0.37, -0.1, 0.54])
    assert np.allclose(result @ tool_from_camera, base_from_camera)


def test_camera_mount_urdf_transform_is_validated():
    model = ET.fromstring('<robot><joint type="fixed"><parent link="tool"/>'
                               '<child link="camera"/><origin xyz=".1 .2 .3" rpy="0 0 0"/>'
                               '</joint></robot>')
    np.testing.assert_allclose(camera_mount_from_urdf(model, "tool", "camera")[:3, 3],
                               [.1, .2, .3])
    with pytest.raises(ValueError, match="attached directly"):
        camera_mount_from_urdf(model, "wrong_tool", "camera")


def test_joint_states_are_mapped_by_name_not_array_order():
    message = SimpleNamespace(
        name=["j3", "j1", "j2"],
        position=[3.0, 1.0, 2.0],
        velocity=[0.3, 0.1, 0.2],
    )
    assert ordered_joint_positions(message, ("j1", "j2", "j3")) == [1.0, 2.0, 3.0]
    assert ordered_joint_velocities(message, ("j1", "j2", "j3")) == [0.1, 0.2, 0.3]
    with pytest.raises(ValueError, match="missing"):
        ordered_joint_positions(message, ("j1", "j4"))


def test_real_observation_manifest_contains_no_object_prior(tmp_path):
    record = make_observation_record(
        image="view.png",
        label="first",
        intrinsics=np.asarray([[600, 0, 320], [0, 600, 240], [0, 0, 1]]),
        distortion_coefficients=[0, 0, 0, 0, 0],
        planning_from_camera=np.eye(4),
        stamp_ns=42,
        camera_frame="calibrated_camera",
        image_header_frame="driver_camera",
        joint_names=("a", "b"),
        joint_positions_rad=[0.1, 0.2],
    )
    path = tmp_path / "observations.json"
    assert append_observation(path, "robot_base", record) == 0
    assert append_observation(path, "robot_base", {**record, "label": "second"}) == 1
    document = json.loads(path.read_text())
    assert document["estimate_input"] == "calibrated_rgb_only"
    assert len(document["observations"]) == 2
    assert "object" not in json.dumps(document).lower()
    with pytest.raises(ValueError, match="coordinate frame"):
        append_observation(path, "map", record)


def test_camera_info_and_padded_bgr_image_conversion():
    info = SimpleNamespace(k=[600, 0, 1, 0, 610, 1, 0, 0, 1])
    assert camera_matrix_from_info(info)[1, 1] == 610
    # Two BGR pixels followed by two row-padding bytes.
    message = SimpleNamespace(
        encoding="bgr8", width=2, height=1, step=8,
        data=bytes([3, 2, 1, 30, 20, 10, 0, 0]),
    )
    image = ros_image_to_pil(message)
    assert image.mode == "RGB"
    assert list(image.getdata()) == [(1, 2, 3), (10, 20, 30)]


def test_ros_triangle_search_matches_object_agnostic_geometry():
    triangles, viewpoints = hemisphere_triangles((0.4, 0.0, 0.3), 0.4, 3, 10.0)
    assert len(triangles) == len(viewpoints) == 3
    assert all(np.isclose(np.linalg.norm(np.asarray(v.position_xyz) - [0.4, 0, 0.3]), 0.4)
               for v in viewpoints)
    children, child_views = subdivide_spherical_triangle(
        triangles[0], (0.4, 0.0, 0.3), 0.4
    )
    assert len(children) == len(child_views) == 3
    assert all(child["triangle_path"][:1] == [1] for child in children)
