"""Convert common ROS sensor_msgs/Image encodings without requiring cv_bridge."""

from __future__ import annotations

import numpy as np
from PIL import Image


def ros_image_to_pil(message) -> Image.Image:
    encoding = str(message.encoding).lower()
    formats = {
        "rgb8": (3, "RGB", None),
        "bgr8": (3, "RGB", (2, 1, 0)),
        "rgba8": (4, "RGBA", None),
        "bgra8": (4, "RGBA", (2, 1, 0, 3)),
        "mono8": (1, "L", None),
    }
    if encoding not in formats:
        raise ValueError(
            f"unsupported image encoding {message.encoding!r}; expected one of {sorted(formats)}"
        )
    channels, mode, order = formats[encoding]
    width, height, step = int(message.width), int(message.height), int(message.step)
    if width <= 0 or height <= 0 or step < width * channels:
        raise ValueError("ROS image dimensions/step are invalid")
    raw = np.frombuffer(message.data, dtype=np.uint8)
    if raw.size < step * height:
        raise ValueError("ROS image data is shorter than height * step")
    rows = raw[: step * height].reshape(height, step)
    pixels = rows[:, : width * channels]
    if channels == 1:
        array = pixels.reshape(height, width)
    else:
        array = pixels.reshape(height, width, channels)
        if order is not None:
            array = array[..., list(order)]
    return Image.fromarray(np.ascontiguousarray(array), mode=mode)
