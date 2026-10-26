"""Pure geometry for the Steve three-view run (NumPy only, no ROS, no CuRobo).

Imported by both the ROS-side runner (dev container) and the CuRobo planner
server (findvp-curobo image), so both sides build exactly the same world.

Conventions (identical to the friend's verified full_pipeline):
  * every pose is expressed in the arm base frame ``ur5ebase_link``;
  * the camera frame is ROS/OpenCV optical: +Z forward, +X right, +Y down;
  * quaternions are xyzw everywhere except the CuRobo cuboid poses, which are
    ``[x, y, z, qw, qx, qy, qz]``;
  * ``T_a_b`` maps coordinates from frame ``b`` into frame ``a``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

CANONICAL_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
PREFIX = "ur5e"
BASE_FRAME = "ur5ebase_link"
TOOL0_FRAME = "ur5etool0"
CAMERA_FRAME = "d415_calib_color_optical_frame_handeye"

REAL_ROBOT_DIR = Path(__file__).resolve().parent
DEFAULT_SCENE = REAL_ROBOT_DIR / "scene.json"
DEFAULT_HANDEYE = REAL_ROBOT_DIR.parents[1] / "handeye_result.json"


# --------------------------------------------------------------------------- rotations


def quat_xyzw_to_matrix(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        raise ValueError("zero quaternion")
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quat_xyzw(R) -> list[float]:
    R = np.asarray(R, dtype=float)
    trace = float(np.trace(R))
    if trace > 0.0:
        s = 2.0 * math.sqrt(trace + 1.0)
        w, x = 0.25 * s, (R[2, 1] - R[1, 2]) / s
        y, z = (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w, x = (R[2, 1] - R[1, 2]) / s, 0.25 * s
        y, z = (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w, x = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s
        y, z = 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w, x = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s
        y, z = (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    q /= np.linalg.norm(q)
    if q[3] < 0.0:
        q = -q
    return q.tolist()


def pose_to_matrix(t, q_xyzw) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = quat_xyzw_to_matrix(q_xyzw)
    T[:3, 3] = np.asarray(t, dtype=float)
    return T


def matrix_to_pose(T) -> tuple[list[float], list[float]]:
    T = np.asarray(T, dtype=float)
    return T[:3, 3].tolist(), matrix_to_quat_xyzw(T[:3, :3])


def rotation_angle_deg(Ra, Rb) -> float:
    cos = (float(np.trace(np.asarray(Ra).T @ np.asarray(Rb))) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def look_at_matrix(eye, target) -> np.ndarray:
    """``T_base_camera`` for an optical camera at ``eye`` looking at ``target``.

    Same formula as the friend's ``camera_lookat_quaternion_xyzw`` and FindVP's
    ``look_at_rotation_matrix``: image "up" (-Y) points as close to world +Z as
    possible, so no view is rolled upside down.
    """

    eye = np.asarray(eye, dtype=float)
    z = np.asarray(target, dtype=float) - eye
    z /= np.linalg.norm(z)
    up = np.array([0.0, 0.0, 1.0])
    if abs(float(z @ up)) > 0.98:
        up = np.array([0.0, 1.0, 0.0])
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, y, z])
    T[:3, 3] = eye
    return T


# --------------------------------------------------------------------------- files


def load_json(path) -> dict:
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, document) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_handeye(path=DEFAULT_HANDEYE) -> np.ndarray:
    """``T_tool0_camera`` from the friend's Tsai calibration file."""

    data = load_json(path)
    return pose_to_matrix(data["translation_m"], data["quaternion_xyzw"])


def load_scene(path=DEFAULT_SCENE, *, center=None, yaw_deg=None) -> dict:
    scene = load_json(path)
    obj = scene["object"]
    if center is not None:
        obj["center_xyz_m"] = [float(v) for v in center]
    if yaw_deg is not None:
        obj["yaw_deg"] = float(yaw_deg)
    if scene.get("frame") != BASE_FRAME:
        raise ValueError(f"scene frame must be {BASE_FRAME}, got {scene.get('frame')!r}")
    for key in ("center_xyz_m", "dims_xyz_m"):
        values = np.asarray(obj[key], dtype=float)
        if values.shape != (3,) or not np.all(np.isfinite(values)):
            raise ValueError(f"object.{key} must contain three finite numbers")
    if min(obj["dims_xyz_m"]) <= 0.0:
        raise ValueError("object dimensions must be positive")
    return scene


# --------------------------------------------------------------------------- world


def _yaw_quat_wxyz(yaw_deg: float) -> list[float]:
    half = math.radians(float(yaw_deg)) / 2.0
    return [math.cos(half), 0.0, 0.0, math.sin(half)]


def world_cuboids(scene: dict) -> list[dict]:
    """Collision cuboids for CuRobo: ``{name, pose=[x,y,z,qw,qx,qy,qz], dims}``.

    The object is padded by ``collision_margin_m`` on every side, the static
    obstacles by their own ``margin_m`` (same rule as the friend's pipeline).
    """

    obj = scene["object"]
    margin = float(obj.get("collision_margin_m", 0.02))
    cuboids = [
        {
            "name": obj.get("name", "object"),
            "pose": [*map(float, obj["center_xyz_m"]), *_yaw_quat_wxyz(obj.get("yaw_deg", 0.0))],
            "dims": [float(d) + 2.0 * margin for d in obj["dims_xyz_m"]],
        }
    ]
    for obstacle in scene.get("obstacles", []):
        pad = float(obstacle.get("margin_m", 0.0))
        cuboids.append(
            {
                "name": obstacle["name"],
                "pose": [
                    *map(float, obstacle["center_xyz_m"]),
                    *map(float, obstacle.get("quat_wxyz", [1, 0, 0, 0])),
                ],
                "dims": [float(d) + 2.0 * pad for d in obstacle["dims_xyz_m"]],
            }
        )
    return cuboids


# --------------------------------------------------------------------------- views


def camera_to_tool0(T_base_camera, T_tool0_camera) -> np.ndarray:
    """``T_base_tool0 = T_base_camera @ inv(T_tool0_camera)``."""

    return np.asarray(T_base_camera) @ np.linalg.inv(np.asarray(T_tool0_camera))


def spherical_view(center, radius_m, azimuth_deg, elevation_deg) -> np.ndarray:
    """Camera pose on the sphere around ``center`` looking at ``center``."""

    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    direction = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    center = np.asarray(center, dtype=float)
    return look_at_matrix(center + radius_m * direction, center)


def view_angles(position, center) -> dict:
    d = np.asarray(position, dtype=float) - np.asarray(center, dtype=float)
    r = float(np.linalg.norm(d))
    return {
        "radius_m": r,
        "azimuth_deg": math.degrees(math.atan2(d[1], d[0])) % 360.0,
        "elevation_deg": math.degrees(math.asin(d[2] / r)),
    }


def direction_angle_deg(position_a, position_b, center) -> float:
    """Angle between two viewing directions seen from the object centre."""

    a = np.asarray(position_a, dtype=float) - np.asarray(center, dtype=float)
    b = np.asarray(position_b, dtype=float) - np.asarray(center, dtype=float)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def triangle_views(scene: dict, radius_m: float | None = None, azimuth_offset_deg: float | None = None):
    """FindVP's level-0 spherical triangles (3 per hemisphere) and their camera poses.

    Uses FindVP's own ``hemisphere_triangles`` so the real robot visits the
    same centroids as the simulator.  Each returned record carries the
    triangle metadata plus ``T_base_camera`` (look-at, optical convention).
    """

    from findvp_ros.triangle_search import hemisphere_triangles

    search = scene.get("search", {})
    radius = float(search.get("radius_m", 0.40) if radius_m is None else radius_m)
    offset = float(search.get("azimuth_offset_deg", 0.0) if azimuth_offset_deg is None else azimuth_offset_deg)
    center = np.asarray(scene["object"]["center_xyz_m"], dtype=float)
    triangles, _ = hemisphere_triangles(center, radius, 3, offset)
    views = []
    for triangle in triangles:
        position = np.asarray(triangle["centroid_xyz_m"], dtype=float)
        views.append(
            {
                "triangle": int(triangle["triangle"]),
                "triangle_record": triangle,
                "T_base_camera": look_at_matrix(position, center),
                **view_angles(position, center),
            }
        )
    return views


def scale_view(T_base_camera, center, radius_m) -> np.ndarray:
    """Same viewing direction, different distance (unreachable-view fallback)."""

    center = np.asarray(center, dtype=float)
    direction = np.asarray(T_base_camera)[:3, 3] - center
    direction /= np.linalg.norm(direction)
    return look_at_matrix(center + radius_m * direction, center)


# --------------------------------------------------------------------------- projection


def box_corners(obj: dict, margin_m: float = 0.0) -> np.ndarray:
    """8 x 3 corners of the (optionally padded) object box in the base frame."""

    half = np.asarray(obj["dims_xyz_m"], dtype=float) / 2.0 + float(margin_m)
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
    yaw = math.radians(float(obj.get("yaw_deg", 0.0)))
    Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    return (signs * half) @ Rz.T + np.asarray(obj["center_xyz_m"], dtype=float)


def project_points(points_base, T_base_camera, K):
    """Pixel coordinates (u, v) and depth for base-frame points."""

    T_camera_base = np.linalg.inv(np.asarray(T_base_camera, dtype=float))
    P = np.asarray(points_base, dtype=float) @ T_camera_base[:3, :3].T + T_camera_base[:3, 3]
    depth = P[:, 2]
    K = np.asarray(K, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * P[:, 0] / depth + K[0, 2]
        v = K[1, 1] * P[:, 1] / depth + K[1, 2]
    return np.column_stack([u, v]), depth


def convex_hull(points) -> np.ndarray:
    """Andrew's monotone chain; returns hull vertices counter-clockwise."""

    pts = sorted(set(map(tuple, np.asarray(points, dtype=float).tolist())))
    if len(pts) <= 2:
        return np.asarray(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1])


def box_outline(obj: dict, T_base_camera, K, margin_m: float | None = None):
    """Image polygon of the projected object box, or ``None`` if it is behind the camera."""

    margin = float(obj.get("mask_margin_m", 0.005) if margin_m is None else margin_m)
    uv, depth = project_points(box_corners(obj, margin), T_base_camera, K)
    if np.any(depth < 0.05):
        return None
    return convex_hull(uv)


def pixel_to_plane(uv, T_base_camera, K, plane_z: float):
    """Base-frame point where the camera ray through pixel ``uv`` meets the plane z = ``plane_z``."""

    T = np.asarray(T_base_camera, dtype=float)
    ray = T[:3, :3] @ (np.linalg.inv(np.asarray(K, dtype=float)) @ np.array([uv[0], uv[1], 1.0]))
    if abs(ray[2]) < 1e-9:
        return None
    scale = (plane_z - T[2, 3]) / ray[2]
    if scale <= 0.0:
        return None
    return T[:3, 3] + scale * ray


def locate_box_from_bottom_edge(p1, p2, obj: dict, camera_position) -> dict:
    """Carton centre/yaw from the two bottom corners of one of its vertical faces.

    ``p1``/``p2`` lie on the deck. The clicked edge is matched to the footprint
    side it is closest to in length (``dims[1]`` = broad face, ``dims[0]`` = thin
    face); the centre lies half the other side behind it, away from the camera.
    """

    thin, mid = (float(d) for d in obj["dims_xyz_m"][:2])
    p1, p2 = np.asarray(p1, dtype=float), np.asarray(p2, dtype=float)
    edge = (p2 - p1)[:2]
    length = float(np.linalg.norm(edge))
    if length < 1e-4:
        raise ValueError("the two clicked corners are the same point")
    broad = abs(length - mid) <= abs(length - thin)
    expected, depth = (mid, thin) if broad else (thin, mid)
    normal = np.array([-edge[1], edge[0]]) / length
    middle = (p1[:2] + p2[:2]) / 2.0
    if normal @ (middle - np.asarray(camera_position, dtype=float)[:2]) < 0.0:
        normal = -normal
    center_xy = middle + normal * depth / 2.0
    yaw = math.degrees(math.atan2(edge[1], edge[0])) - (90.0 if broad else 0.0)
    yaw = (yaw + 90.0) % 180.0 - 90.0  # the box is symmetric under 180 deg
    return {
        "center_xy_m": center_xy.tolist(),
        "yaw_deg": yaw,
        "face": "broad" if broad else "thin",
        "edge_length_m": length,
        "expected_edge_m": expected,
    }


def box_mask(obj: dict, T_base_camera, K, width: int, height: int, margin_m: float | None = None) -> np.ndarray:
    """Boolean object mask from the known box geometry and the camera pose.

    This replaces SAM for this known carton: the hand-eye calibration is
    ~3-4 mm accurate, so the projected box silhouette is a tight,
    deterministic mask for both the target and every candidate image.
    """

    from PIL import Image, ImageDraw

    polygon = box_outline(obj, T_base_camera, K, margin_m)
    canvas = Image.new("L", (int(width), int(height)), 0)
    if polygon is not None and len(polygon) >= 3:
        ImageDraw.Draw(canvas).polygon([tuple(p) for p in polygon.tolist()], fill=255)
    return np.asarray(canvas) > 0
