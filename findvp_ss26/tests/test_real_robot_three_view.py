"""Hardware-free checks for real_robot/ (Steve three-view run)."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "real_robot"))

from geometry import (  # noqa: E402
    box_mask,
    camera_to_tool0,
    load_handeye,
    load_scene,
    look_at_matrix,
    matrix_to_pose,
    pose_to_matrix,
    triangle_views,
    world_cuboids,
)
from ur_ik import fk_dh, ik_dh, unwrap_near  # noqa: E402

K = np.array([[615.0, 0.0, 320.0], [0.0, 615.0, 240.0], [0.0, 0.0, 1.0]])


def test_analytic_ik_recovers_every_random_configuration():
    rng = np.random.default_rng(1)
    for q in rng.uniform(-math.pi, math.pi, (300, 6)):
        T = fk_dh(q)
        solutions = ik_dh(T)
        assert solutions
        for s in solutions:
            assert np.abs(fk_dh(s) - T).max() < 1e-9
        assert any(np.abs(np.arctan2(np.sin(s - q), np.cos(s - q))).max() < 1e-6 for s in solutions)


def test_unwrap_prefers_the_turn_closest_to_the_live_joint():
    limits = np.full(6, 2 * math.pi)
    q = unwrap_near(np.array([3.0, 0, 0, 0, 0, -3.0]), np.array([-3.0, 0, 0, 0, 0, 3.0]), -limits, limits)
    assert q[0] == pytest.approx(3.0 - 2 * math.pi)
    assert q[5] == pytest.approx(-3.0 + 2 * math.pi)


def test_three_triangle_views_look_at_the_carton_upright():
    scene = load_scene()
    center = np.asarray(scene["object"]["center_xyz_m"])
    views = triangle_views(scene, 0.40, 0.0)
    assert [v["triangle"] for v in views] == [1, 2, 3]
    for view, azimuth in zip(views, (60.0, 180.0, 300.0)):
        T = view["T_base_camera"]
        forward = center - T[:3, 3]
        assert np.linalg.norm(forward) == pytest.approx(0.40)
        assert T[:3, 2] @ forward / np.linalg.norm(forward) == pytest.approx(1.0)
        assert T[2, 1] < 0.0  # optical +Y (image down) points towards the deck
        assert view["azimuth_deg"] == pytest.approx(azimuth)
        assert view["elevation_deg"] == pytest.approx(45.0, abs=1e-6)


def test_camera_to_tool0_inverts_the_hand_eye():
    T_tool0_camera = load_handeye()
    T_camera = look_at_matrix([0.3, 0.5, 0.35], [0.18, 0.3, 0.06])
    T_tool0 = camera_to_tool0(T_camera, T_tool0_camera)
    assert np.abs(T_tool0 @ T_tool0_camera - T_camera).max() < 1e-12
    t, q = matrix_to_pose(T_tool0)
    assert np.abs(pose_to_matrix(t, q) - T_tool0).max() < 1e-9


def test_projected_box_mask_is_centred_when_looking_at_the_carton():
    scene = load_scene()
    center = np.asarray(scene["object"]["center_xyz_m"])
    T = look_at_matrix(center + [0.0, -0.3, 0.3], center)
    mask = box_mask(scene["object"], T, K, 640, 480)
    rows, cols = np.nonzero(mask)
    assert 0.01 < mask.mean() < 0.3
    assert abs(rows.mean() - 240) < 25 and abs(cols.mean() - 320) < 25
    behind = look_at_matrix(center + [0.0, -0.3, 0.3], center + [0.0, -0.6, 0.6])
    assert not box_mask(scene["object"], behind, K, 640, 480).any()


def test_world_pads_the_carton_by_its_collision_margin():
    scene = load_scene()
    carton = world_cuboids(scene)[0]
    margin = scene["object"]["collision_margin_m"]
    assert carton["dims"] == pytest.approx([d + 2 * margin for d in scene["object"]["dims_xyz_m"]])
    assert carton["pose"][:3] == pytest.approx(scene["object"]["center_xyz_m"])


@pytest.mark.parametrize("yaw_deg", [-82.9, -30.0, 0.0, 45.0, 89.0])
@pytest.mark.parametrize("camera_azimuth", [200.0, 270.0, 300.0, 20.0])
def test_locate_recovers_centre_and_yaw_from_two_clicked_corners(yaw_deg, camera_azimuth):
    from geometry import locate_box_from_bottom_edge, pixel_to_plane, project_points, spherical_view

    obj = {"center_xyz_m": [0.21, 0.33, 0.0635], "yaw_deg": yaw_deg, "dims_xyz_m": [0.031, 0.09, 0.127]}
    T = spherical_view(obj["center_xyz_m"], 0.4, camera_azimuth, 35.0)
    yaw = math.radians(yaw_deg)
    local_x = np.array([math.cos(yaw), math.sin(yaw), 0.0])  # thin axis = broad-face normal
    local_y = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    center = np.array([0.21, 0.33, 0.0])
    towards_camera = (T[:3, 3] - center) / np.linalg.norm(T[:3, 3] - center)
    if abs(local_x @ towards_camera) < 0.3:
        pytest.skip("the broad face is seen edge-on from this camera; nobody would click it")
    side = 1.0 if local_x @ towards_camera > 0 else -1.0  # broad face towards the camera
    corners = [center + side * 0.0155 * local_x + s * 0.045 * local_y for s in (-1.0, 1.0)]
    uv, _ = project_points(np.asarray(corners), T, K)
    points = [pixel_to_plane(p, T, K, 0.0) for p in uv]
    found = locate_box_from_bottom_edge(points[0], points[1], obj, T[:3, 3])
    assert found["face"] == "broad"
    assert found["center_xy_m"] == pytest.approx([0.21, 0.33], abs=1e-6)
    wrapped = (found["yaw_deg"] - yaw_deg + 90.0) % 180.0 - 90.0
    assert abs(wrapped) < 1e-4
