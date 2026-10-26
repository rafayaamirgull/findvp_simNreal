import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from sample.visualize_end_effector_capture_poses import (
    BEST_VIEW_COLOR,
    VIEW_COLORS,
    BestView,
    build_parser,
    camera_frustum_geometry,
    draw_camera_frustums,
    draw_capture_poses,
    draw_projected_images,
    load_capture_poses,
    parse_xwininfo_window_bounds,
    resolve_best_view,
    sphere_avoiding_trajectory,
)


class _FakePhysics:
    GEOM_BOX = 1
    GEOM_SPHERE = 2

    def __init__(self) -> None:
        self.visual_shapes = []
        self.bodies = []
        self.lines = []
        self.labels = []
        self.point_clouds = []

    def createVisualShape(self, **kwargs):
        self.visual_shapes.append(kwargs)
        return 7

    def createMultiBody(self, **kwargs):
        self.bodies.append(kwargs)
        return 11

    def addUserDebugLine(self, start, end, **kwargs):
        self.lines.append((start, end, kwargs))

    def addUserDebugText(self, text, position, **kwargs):
        self.labels.append((text, position, kwargs))

    def addUserDebugPoints(self, points, colors, **kwargs):
        self.point_clouds.append((points, colors, kwargs))


class _FakeEnvironment:
    def __init__(self) -> None:
        self.physics = _FakePhysics()


class CapturePoseLoadingTests(unittest.TestCase):
    def test_loads_one_pose_per_captured_image(self) -> None:
        report = {
            "captures": [
                {
                    "image_filename": "object_view_01.png",
                    "joint_positions_rad": [0.0] * 6,
                    "end_effector_pose_world": {
                        "position_xyz_m": [0.4, 0.1, 0.8],
                        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                    "camera_pose_world": {
                        "position_xyz_m": [0.42, 0.1, 0.82],
                        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                }
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture_poses.json"
            path.write_text(json.dumps(report), encoding="utf-8")

            loaded = load_capture_poses(path)

            self.assertEqual("object_view_01.png", loaded["captures"][0]["image_filename"])

    def test_rejects_an_empty_capture_list(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture_poses.json"
            path.write_text('{"captures": []}', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "at least one capture"):
                load_capture_poses(path)

    def test_rejects_a_capture_without_an_optical_camera_pose(self) -> None:
        report = {
            "captures": [
                {
                    "image_filename": "object_view_01.png",
                    "joint_positions_rad": [0.0] * 6,
                    "end_effector_pose_world": {
                        "position_xyz_m": [0.4, 0.1, 0.8],
                        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                }
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture_poses.json"
            path.write_text(json.dumps(report), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Invalid capture-pose entry"):
                load_capture_poses(path)

    def test_draws_numbered_rectangle_instead_of_xyz_axes(self) -> None:
        environment = _FakeEnvironment()
        report = {
            "captures": [
                {
                    "picture_number": 4,
                    "image_filename": "object_level_01_view_01.png",
                    "refinement_level": 1,
                    "end_effector_pose_world": {
                        "position_xyz_m": [0.4, 0.1, 0.8],
                        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                }
            ]
        }

        draw_capture_poses(
            environment,
            report,
            rectangle_width=0.08,
            rectangle_height=0.06,
            rectangle_thickness=0.002,
        )

        self.assertEqual(1, len(environment.physics.visual_shapes))
        self.assertEqual(1, len(environment.physics.bodies))
        self.assertEqual(4, len(environment.physics.lines))
        self.assertEqual("Picture 4", environment.physics.labels[0][0])

    def test_camera_frustum_uses_optical_positive_z_and_image_aspect(self) -> None:
        origin, corners = camera_frustum_geometry(
            (0.4, 0.1, 0.8),
            (0.0, 0.0, 0.0, 1.0),
            0.12,
            image_width=640,
            image_height=480,
        )

        np.testing.assert_allclose((0.4, 0.1, 0.8), origin)
        np.testing.assert_allclose(0.92, corners[:, 2])
        width = float(np.linalg.norm(corners[1] - corners[0]))
        height = float(np.linalg.norm(corners[3] - corners[0]))
        self.assertAlmostEqual(4.0 / 3.0, width / height)

    def test_draws_unique_frustums_and_emphasizes_the_best_view(self) -> None:
        environment = _FakeEnvironment()
        report = {
            "captures": [
                {
                    "picture_number": 1,
                    "camera_pose_world": {
                        "position_xyz_m": [0.4, 0.1, 0.8],
                        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                },
                {
                    "picture_number": 2,
                    "camera_pose_world": {
                        "position_xyz_m": [0.5, 0.1, 0.8],
                        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                },
            ]
        }

        best_view = BestView(
            capture_index=1,
            picture_number=2,
            image_filename="view_02.png",
            score=0.9762,
            source="summary.json",
        )
        line_count = draw_camera_frustums(
            environment,
            report,
            depth_m=0.12,
            line_width=3.0,
            show_labels=True,
            best_view=best_view,
        )

        self.assertEqual(16, line_count)
        self.assertEqual(
            VIEW_COLORS[0],
            environment.physics.lines[0][2]["lineColorRGB"],
        )
        self.assertEqual(
            BEST_VIEW_COLOR,
            environment.physics.lines[8][2]["lineColorRGB"],
        )
        self.assertEqual(7.5, environment.physics.lines[8][2]["lineWidth"])
        self.assertEqual(
            "BEST VIEW | Picture 2 | score=0.9762",
            environment.physics.labels[1][0],
        )
        self.assertEqual(0.75, environment.physics.labels[0][2]["textSize"])
        self.assertEqual(1.05, environment.physics.labels[1][2]["textSize"])

    def test_resolves_global_best_capture_from_sibling_refinement_summary(self) -> None:
        report = {
            "captures": [
                {"picture_number": 1, "image_filename": "object_view_01.png"},
                {
                    "picture_number": 2,
                    "image_filename": "object_radial_inside_01.png",
                },
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture_path = root / "object_end_effector_capture_poses.json"
            summary_path = root / "object_refinement_summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "final_best": {
                            "best_image": "/workspace/output/object_radial_inside_01.png",
                            "best_score": 0.9761879,
                        }
                    }
                ),
                encoding="utf-8",
            )

            best_view = resolve_best_view(report, capture_path)

        self.assertEqual(1, best_view.capture_index)
        self.assertEqual(2, best_view.picture_number)
        self.assertAlmostEqual(0.9761879, best_view.score)
        self.assertEqual(str(summary_path.resolve()), best_view.source)

    def test_tool_rectangles_use_unique_colors_and_enlarge_the_best(self) -> None:
        environment = _FakeEnvironment()
        captures = []
        for picture_number in (1, 2):
            captures.append(
                {
                    "picture_number": picture_number,
                    "image_filename": f"view_{picture_number:02d}.png",
                    "stage": "angular_refinement",
                    "refinement_level": 0,
                    "triangle_path": [picture_number],
                    "end_effector_pose_world": {
                        "position_xyz_m": [0.4 + picture_number * 0.1, 0.1, 0.8],
                        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                }
            )
        best_view = BestView(1, 2, "view_02.png", 0.95, "summary.json")

        draw_capture_poses(
            environment,
            {"captures": captures},
            rectangle_width=0.08,
            rectangle_height=0.06,
            rectangle_thickness=0.002,
            best_view=best_view,
        )

        first_color = environment.physics.visual_shapes[0]["rgbaColor"][:3]
        second_shape = environment.physics.visual_shapes[1]
        self.assertEqual(list(VIEW_COLORS[0]), first_color)
        self.assertEqual(list(BEST_VIEW_COLOR), second_shape["rgbaColor"][:3])
        self.assertAlmostEqual(0.046, second_shape["halfExtents"][0])
        self.assertTrue(environment.physics.labels[1][0].startswith("BEST VIEW"))

    def test_compact_visualization_defaults_remain_cli_adjustable(self) -> None:
        args = build_parser().parse_args(["--captures", "captures.json"])

        self.assertEqual(0.08, args.frustum_depth)
        self.assertEqual(0.079, args.image_depth)
        self.assertEqual(0.045, args.rectangle_width)
        self.assertEqual(0.034, args.rectangle_height)
        self.assertEqual(0.75, args.label_size)
        self.assertEqual(1.05, args.best_label_size)
        self.assertEqual(1.15, args.best_view_scale)
        self.assertEqual(48, args.gif_frames)
        self.assertEqual(12.0, args.gif_fps)
        self.assertEqual(360.0, args.gif_orbit_degrees)
        self.assertEqual(800, args.gif_width)

    def test_parses_largest_pybullet_x11_window_for_gif_capture(self) -> None:
        tree_output = """
            0x100 "Bullet Physics ExampleBrowser": ()  640x480+0+0  +50+60
            0x200 "Bullet Physics ExampleBrowser using OpenGL3+": ()  1024x768+10+45  +2610+708
        """

        bounds = parse_xwininfo_window_bounds(tree_output)

        self.assertEqual((2610, 708, 3634, 1476), bounds)

    def test_sphere_avoiding_trajectory_uses_surface_arc_when_needed(self) -> None:
        path = sphere_avoiding_trajectory(
            (-2.0, 0.0, 1.0),
            (2.0, 0.0, 1.0),
            (0.0, 0.0, 1.0),
            sphere_radius_m=1.0,
            maximum_step_m=0.1,
            minimum_height_m=-1.0,
        )

        self.assertEqual("sphere_arc", path.mode)
        radii = np.linalg.norm(
            path.points_xyz_m - np.asarray((0.0, 0.0, 1.0)), axis=1
        )
        self.assertGreaterEqual(float(np.min(radii)), 1.0 - 1e-9)
        self.assertAlmostEqual(2.0 + math.pi, path.length_m)

    def test_sphere_avoiding_trajectory_keeps_clear_line_when_possible(self) -> None:
        path = sphere_avoiding_trajectory(
            (-1.0, 2.0, 1.0),
            (1.0, 2.0, 1.0),
            (0.0, 0.0, 1.0),
            sphere_radius_m=1.0,
            maximum_step_m=0.2,
        )

        self.assertEqual("line", path.mode)
        self.assertAlmostEqual(2.0, path.length_m)

    def test_projects_sampled_capture_pixels_onto_camera_plane(self) -> None:
        environment = _FakeEnvironment()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "view.png"
            pixels = np.asarray(
                [
                    [[255, 0, 0], [0, 255, 0]],
                    [[0, 0, 255], [255, 255, 255]],
                ],
                dtype=np.uint8,
            )
            Image.fromarray(pixels).save(image_path)
            report_path = root / "captures.json"
            report = {
                "captures": [
                    {
                        "image_filename": image_path.name,
                        "camera_pose_world": {
                            "position_xyz_m": [0.0, 0.0, 0.0],
                            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                        },
                    }
                ]
            }

            displayed, skipped = draw_projected_images(
                environment,
                report,
                report_path,
                depth_m=0.1,
                sample_step_px=1,
                point_size=2.0,
                flip_images=False,
            )

        self.assertEqual((1, 0), (displayed, skipped))
        points, colors, _ = environment.physics.point_clouds[0]
        self.assertEqual(4, len(points))
        self.assertEqual([1.0, 0.0, 0.0], colors[0])
        self.assertTrue(all(abs(point[2] - 0.1) < 1e-12 for point in points))


if __name__ == "__main__":
    unittest.main()
