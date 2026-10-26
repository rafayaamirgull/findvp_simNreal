"""Select the full reference object once, without assuming an object class."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .runtime import file_digest


def polygon_mask(size, vertices):
    points = np.asarray(vertices, dtype=float)
    width, height = size
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 3
            or not np.isfinite(points).all()):
        raise ValueError("select at least three finite polygon vertices")
    if ((points < 0).any() or (points[:, 0] >= width).any()
            or (points[:, 1] >= height).any()):
        raise ValueError("polygon vertices must lie inside the reference image")
    image = Image.new("L", size, 0)
    ImageDraw.Draw(image).polygon([tuple(p) for p in points], fill=255)
    mask = np.asarray(image) > 0
    if mask.sum() < 16:
        raise ValueError("selected object is too small")
    return mask


def load_reference_mask(path, image_path):
    path, image_path = Path(path).expanduser().resolve(), Path(image_path).expanduser().resolve()
    with Image.open(image_path) as reference, Image.open(path) as source:
        mask = np.asarray(source.convert("L"))
        if mask.shape != (reference.height, reference.width):
            raise ValueError("reference mask dimensions do not match the target image")
    if not np.isin(mask, (0, 255)).all() or mask.sum() == 0:
        raise ValueError("reference mask must contain a nonempty white object on a black background")
    metadata = path.with_suffix(".json")
    if metadata.exists():
        if json.loads(metadata.read_text()).get("target_sha256") != file_digest(image_path):
            raise ValueError("reference image changed since target selection; select the object again")
    return mask > 0


def validate_target_provenance(document, image_path, mask_path):
    if document.get("target_sha256") != file_digest(image_path):
        raise ValueError("localization does not match the target image; rerun findvp_localize")
    expected = document.get("target_mask_sha256")
    actual = file_digest(mask_path) if mask_path is not None else None
    if expected != actual:
        raise ValueError("localization does not match the target mask; rerun findvp_localize")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--polygon", type=float, nargs="+",
                        help="Optional headless annotation: x1 y1 x2 y2 x3 y3 ... in original pixels")
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    metadata = output.with_suffix(".json")
    if output.suffix.lower() != ".png":
        parser.error("--output must be a PNG mask")
    if output.exists() or metadata.exists():
        raise FileExistsError(f"refusing to overwrite target selection: {output}")
    with Image.open(args.image) as source:
        reference = source.convert("RGB")
    vertices = args.polygon
    if vertices is not None:
        if len(vertices) % 2:
            parser.error("--polygon requires x/y pairs")
        vertices = np.asarray(vertices).reshape(-1, 2).tolist()
    else:
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            parser.error("no display; run in the graphical container or supply --polygon")
        import matplotlib
        matplotlib.use("QtAgg")
        import matplotlib.pyplot as plt
        from matplotlib.widgets import PolygonSelector
        fig, axes = plt.subplots()
        axes.imshow(reference)
        axes.set_title("Outline the WHOLE object, click first point to close. Enter: save. Esc: cancel.")
        selected = {"vertices": None, "accepted": False}
        def choose(value):
            selected["vertices"] = value
        selector = PolygonSelector(axes, choose)
        def key(event):
            if event.key == "enter" and selected["vertices"] is not None:
                selected["accepted"] = True
                plt.close(fig)
            elif event.key == "escape":
                plt.close(fig)
        fig.canvas.mpl_connect("key_press_event", key)
        plt.show()
        selector.disconnect_events()
        if not selected["accepted"]:
            raise SystemExit("Target selection cancelled; no files written")
        vertices = selected["vertices"]
    mask = polygon_mask(reference.size, vertices)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        Image.fromarray(mask.astype(np.uint8) * 255).save(stream, format="PNG")
    with metadata.open("x", encoding="utf-8") as stream:
        json.dump({"target_image": str(args.image.resolve()),
                   "target_sha256": file_digest(args.image), "polygon_xy": vertices}, stream, indent=2)
    print(f"Reference mask saved: {output}. Inspect it before localization.")


if __name__ == "__main__":
    main()
