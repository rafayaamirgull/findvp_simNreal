"""Register Steve's measured top deck, fixed box, and a conservative target keepout.

The target centre is an operator measurement used only for collision avoidance.
FindVP still localizes and scores the target from RGB images.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from .runtime import collision_primitive_center, pose_matrix

from .steve_workcell import (
    FRONT_BOX_CENTER_M,
    FRONT_BOX_DIMS_M,
    SUPPORT_CENTER_M,
    SUPPORT_DIMS_M,
    known_carton_keepout_radius,
)


def scene_geometry(target_center, uncertainty):
    """Dimensions copied from Steve NBV's 2026-09-21 measured workcell."""
    if len(target_center) != 3 or not all(math.isfinite(v) for v in target_center):
        raise ValueError("--target-center requires finite X Y Z in ur5ebase_link")
    if not math.isfinite(uncertainty) or uncertainty < 0.0:
        raise ValueError("--uncertainty must be finite and nonnegative")
    # This is planning geometry only; RGB still chooses the best viewpoint.
    target_radius = known_carton_keepout_radius(uncertainty)
    return (
        ("findvp_support", "box", SUPPORT_CENTER_M, SUPPORT_DIMS_M),
        ("findvp_front_box", "box", FRONT_BOX_CENTER_M, FRONT_BOX_DIMS_M),
        ("findvp_target_keepout", "sphere", tuple(target_center), (target_radius,)),
    )


def known_center(path, frame):
    document = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if document.get("estimate_kind") != "known_object_center":
        raise ValueError("--localization must be a known-object-centre seed")
    if document.get("coordinate_frame") != frame:
        raise ValueError("--localization frame differs from the scene frame")
    center = document.get("center_world_m")
    if not isinstance(center, list) or len(center) != 3 or not all(
        isinstance(value, (float, int)) and math.isfinite(value) for value in center
    ):
        raise ValueError("--localization has no finite three-dimensional centre")
    return tuple(float(value) for value in center)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    center_source = parser.add_mutually_exclusive_group()
    center_source.add_argument("--target-center", nargs=3, type=float,
                               metavar=("X", "Y", "Z"),
                               help="Measured approximate box centre in ur5ebase_link, metres")
    center_source.add_argument("--localization", type=Path,
                               help="Use the centre stored in a known-object-centre seed JSON")
    parser.add_argument("--uncertainty", type=float, default=.03,
                        help="Additional target placement uncertainty in metres")
    parser.add_argument("--frame", default="ur5ebase_link")
    parser.add_argument("--check", action="store_true",
                        help="Read the current MoveIt scene and verify all three known-box objects")
    args = parser.parse_args(argv)
    center = (known_center(args.localization, args.frame) if args.localization else args.target_center)
    if not args.check and center is None:
        parser.error("--target-center or --localization is required when registering the scene")
    geometry = scene_geometry(center, args.uncertainty) if not args.check else None

    import rclpy
    from geometry_msgs.msg import Pose
    from moveit_msgs.msg import CollisionObject, PlanningScene, PlanningSceneComponents
    from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
    from rclpy.time import Time
    from shape_msgs.msg import SolidPrimitive
    from tf2_ros import Buffer, TransformException, TransformListener

    rclpy.init()
    node = rclpy.create_node("findvp_register_scene")
    try:
        if args.check:
            client = node.create_client(GetPlanningScene, "/get_planning_scene")
            if not client.wait_for_service(timeout_sec=5.0):
                raise RuntimeError("/get_planning_scene unavailable; start exactly one local MoveIt")
            request = GetPlanningScene.Request()
            request.components.components = PlanningSceneComponents.WORLD_OBJECT_GEOMETRY
            future = client.call_async(request)
            rclpy.spin_until_future_complete(node, future, timeout_sec=8.0)
            if not future.done() or future.exception() is not None:
                raise RuntimeError("could not read MoveIt's current planning scene")
            objects = {item.id: item for item in future.result().scene.world.collision_objects}
            required = ("findvp_support", "findvp_front_box", "findvp_target_keepout")
            missing = [name for name in required if name not in objects or not objects[name].primitives]
            tf_buffer = Buffer()
            # Keep the subscription alive until the scene-frame lookups finish.
            node._findvp_tf_listener = TransformListener(tf_buffer, node)

            def center_in_scene_frame(obj):
                source_frame = obj.header.frame_id
                if not source_frame:
                    raise RuntimeError(f"collision object {obj.id} has no coordinate frame")
                if source_frame == args.frame:
                    target_from_header = np.eye(4)
                else:
                    deadline = time.monotonic() + 5.0
                    while True:
                        try:
                            transform = tf_buffer.lookup_transform(args.frame, source_frame, Time())
                            break
                        except TransformException as error:
                            if time.monotonic() >= deadline:
                                raise RuntimeError(
                                    f"cannot transform collision object {obj.id} from "
                                    f"{source_frame} to {args.frame}: {error}"
                                ) from error
                            rclpy.spin_once(node, timeout_sec=0.1)
                    target_from_header = pose_matrix(transform.transform)
                return collision_primitive_center(obj, target_from_header)

            geometry = {}
            for name in required:
                obj = objects.get(name)
                if obj is None or not obj.primitives or not obj.primitive_poses:
                    continue
                geometry[name] = {
                    "frame": args.frame,
                    "source_frame": obj.header.frame_id,
                    "center": center_in_scene_frame(obj).tolist(),
                    "dimensions": list(obj.primitives[0].dimensions),
                }
            print(json.dumps({"present": sorted(objects), "missing_required": missing,
                              "geometry": geometry}, indent=2))
            if missing:
                raise RuntimeError("current MoveIt scene is missing required geometry: " + ", ".join(missing))
            if center is not None:
                keepout = objects["findvp_target_keepout"]
                if not keepout.primitive_poses or (
                    keepout.primitives[0].type != SolidPrimitive.SPHERE or
                    not keepout.primitives[0].dimensions
                ):
                    raise RuntimeError("current target keepout is not a positioned sphere")
                keepout_center = center_in_scene_frame(keepout)
                offset = math.dist(center, keepout_center)
                if offset + known_carton_keepout_radius(0.0) > keepout.primitives[0].dimensions[0]:
                    raise RuntimeError(
                        f"target keepout does not enclose localization centre {center}; "
                        f"scene centre is {keepout_center.tolist()}"
                    )
            return
        client = node.create_client(ApplyPlanningScene, "/apply_planning_scene")
        if not client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("/apply_planning_scene unavailable; start exactly one local MoveIt")
        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        for name, kind, center, dimensions in geometry:
            obj = CollisionObject()
            obj.header.frame_id = args.frame
            obj.id = name
            obj.operation = CollisionObject.ADD
            primitive = SolidPrimitive()
            primitive.type = SolidPrimitive.BOX if kind == "box" else SolidPrimitive.SPHERE
            primitive.dimensions = list(dimensions)
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = center
            pose.orientation.w = 1.0
            obj.primitives = [primitive]
            obj.primitive_poses = [pose]
            scene.world.collision_objects.append(obj)
        request = ApplyPlanningScene.Request()
        request.scene = scene
        future = client.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=8.0)
        if not future.done() or future.exception() is not None or not future.result().success:
            raise RuntimeError("MoveIt did not confirm the scene update")
        print(json.dumps({"frame": args.frame, "objects": [
            {"id": name, "shape": kind, "center": center, "dimensions": dimensions}
            for name, kind, center, dimensions in geometry]}, indent=2))
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
