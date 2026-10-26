"""Save one fresh RGB camera frame as a target-view reference photograph."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

from .configuration import load_robot_configuration
from .image_conversion import ros_image_to_pil
from .runtime import require_recent, stamp_ns


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args(argv)
    config = load_robot_configuration(args.config)
    destination = args.output.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")

    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image

    rclpy.init()
    node = rclpy.create_node("findvp_snapshot")
    received = []
    node.create_subscription(Image, config.image_topic, received.append, qos_profile_sensor_data)
    deadline = time.monotonic() + args.timeout
    try:
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.1)
            if received:
                message = received.pop()
                received.clear()
                try:
                    require_recent(stamp_ns(message), node.get_clock().now().nanoseconds,
                                   1.0, "RGB image")
                    if message.header.frame_id != config.image_frame:
                        raise ValueError("RGB frame does not match the configured optical frame")
                except (RuntimeError, ValueError):
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("xb") as stream:
                    ros_image_to_pil(message).convert("RGB").save(stream, format="PNG")
                print(f"Saved fresh RGB target view: {destination}")
                return
        raise RuntimeError(f"no fresh RGB frame received on {config.image_topic}")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
