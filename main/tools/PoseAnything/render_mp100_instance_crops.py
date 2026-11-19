#!/usr/bin/env python3
"""Build an instance-level MP-100 RGB/pose dataset.

The original ``images`` tree is kept unchanged.  Each unique annotation is
cropped with a small amount of surrounding context and written to three
parallel trees:

* ``crop_images``: the RGB instance crop;
* ``crop_images_pose_black``: only that instance's pose on black;
* ``crop_images_pose_overlay``: the same pose over the RGB crop.

The renderer reuses the canonical MP-100 pose style and coordinate handling
from :mod:`render_mp100_pose_visualizations`: EXIF orientation is applied
before annotation coordinates are scaled, bones are white, and joints are
red.  Annotation ids are embedded in filenames so multiple instances from one
source image remain distinct and exactly paired across all three trees.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps

from render_mp100_pose_visualizations import (
    DEFAULT_INTERNAL_SCALE,
    DEFAULT_JOINT_RADIUS,
    DEFAULT_LINE_WIDTH,
    JOINT_COLOR,
    UNIFORM_BONE_COLOR,
    bone_color,
    draw_capsule,
    load_records,
)


DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[3] / "output" / "PoseAnything_MP100"
DEFAULT_CONTEXT_SCALE = 1.35
REFERENCE_LONG_EDGE = 480.0
DEFAULT_WORKERS = max(1, min(40, os.cpu_count() or 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--rgb-dirname", default="crop_images")
    parser.add_argument("--black-dirname", default="crop_images_pose_black")
    parser.add_argument("--overlay-dirname", default="crop_images_pose_overlay")
    parser.add_argument("--context-scale", type=float, default=DEFAULT_CONTEXT_SCALE)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--prune-stale", action="store_true", help="Remove generated files for rejected annotations.")
    parser.add_argument("--limit", type=int, default=None, help="Optional instance limit for smoke tests.")
    return parser.parse_args()


def instance_relative_paths(file_name: str, annotation_id: int) -> tuple[Path, Path]:
    source = Path(file_name)
    stem = f"{source.stem}__ann_{int(annotation_id):09d}"
    rgb = source.parent / f"{stem}{source.suffix.lower()}"
    pose = source.parent / f"{stem}.png"
    return rgb, pose


def expanded_crop_box(
    bbox: list[float],
    coordinate_scale_x: float,
    coordinate_scale_y: float,
    image_width: int,
    image_height: int,
    context_scale: float,
) -> tuple[int, int, int, int]:
    x, y, width, height = (float(value) for value in bbox[:4])
    x *= coordinate_scale_x
    width *= coordinate_scale_x
    y *= coordinate_scale_y
    height *= coordinate_scale_y
    width = max(width, 2.0)
    height = max(height, 2.0)
    crop_width = min(float(image_width), width * context_scale)
    crop_height = min(float(image_height), height * context_scale)
    center_x = x + width / 2.0
    center_y = y + height / 2.0
    left = min(max(center_x - crop_width / 2.0, 0.0), image_width - crop_width)
    top = min(max(center_y - crop_height / 2.0, 0.0), image_height - crop_height)
    right = left + crop_width
    bottom = top + crop_height
    integer_box = (
        max(0, int(math.floor(left))),
        max(0, int(math.floor(top))),
        min(image_width, int(math.ceil(right))),
        min(image_height, int(math.ceil(bottom))),
    )
    if integer_box[2] <= integer_box[0] or integer_box[3] <= integer_box[1]:
        raise ValueError(f"Invalid crop box {integer_box} from bbox={bbox}")
    return integer_box


def save_rgb(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        image.save(path, format="JPEG", quality=95, subsampling=0)
    else:
        image.save(path, format="PNG", optimize=False, compress_level=1)


def annotation_has_visible_bone(record: dict[str, Any], annotation: dict[str, Any]) -> bool:
    raw_keypoints = annotation.get("keypoints") or []
    if not raw_keypoints or len(raw_keypoints) % 3:
        return False
    visibility = [raw_keypoints[index + 2] for index in range(0, len(raw_keypoints), 3)]
    return any(
        start < len(visibility)
        and end < len(visibility)
        and visibility[start] > 0
        and visibility[end] > 0
        for start, end in record["skeleton"]
    )


def prune_stale_files(root: Path, expected: set[Path]) -> int:
    removed = 0
    for path in root.rglob("*"):
        if path.is_file() and path.relative_to(root) not in expected:
            path.unlink()
            removed += 1
    for directory in sorted((path for path in root.rglob("*") if path.is_dir()), reverse=True):
        try:
            directory.rmdir()
        except OSError:
            pass
    return removed


def render_instance(task: tuple[Any, ...]) -> dict[str, Any]:
    (
        record,
        annotation,
        dataset_root_text,
        rgb_root_text,
        black_root_text,
        overlay_root_text,
        context_scale,
        overwrite,
    ) = task
    dataset_root = Path(dataset_root_text)
    rgb_root = Path(rgb_root_text)
    black_root = Path(black_root_text)
    overlay_root = Path(overlay_root_text)
    annotation_id = int(annotation["id"])
    rgb_relative, pose_relative = instance_relative_paths(record["file_name"], annotation_id)
    rgb_path = rgb_root / rgb_relative
    black_path = black_root / pose_relative
    overlay_path = overlay_root / rgb_relative

    with Image.open(dataset_root / "images" / record["file_name"]) as source_image:
        source_rgb = ImageOps.exif_transpose(source_image).convert("RGB")
    image_width, image_height = source_rgb.size
    annotation_width = max(1, int(record["annotation_width"]))
    annotation_height = max(1, int(record["annotation_height"]))
    coordinate_scale_x = image_width / annotation_width
    coordinate_scale_y = image_height / annotation_height
    bbox = annotation.get("bbox")
    if not bbox or len(bbox) < 4:
        raise ValueError(f"annotation {annotation_id} has no valid bbox")
    crop_box = expanded_crop_box(
        bbox,
        coordinate_scale_x,
        coordinate_scale_y,
        image_width,
        image_height,
        context_scale,
    )
    crop_rgb = source_rgb.crop(crop_box)
    crop_width, crop_height = crop_rgb.size

    result = {
        "status": "skipped",
        "sample_id": f"mp100::{Path(record['file_name']).with_suffix('').as_posix()}::ann_{annotation_id}",
        "source_file_name": record["file_name"],
        "annotation_id": annotation_id,
        "category_id": int(record["category_id"]),
        "category_name": record["category_name"],
        "bbox_annotation_xywh": [float(value) for value in bbox[:4]],
        "crop_box_source_ltrb": list(crop_box),
        "source_size": [image_width, image_height],
        "annotation_size": [annotation_width, annotation_height],
        "crop_size": [crop_width, crop_height],
        "rgb": str(rgb_relative),
        "pose_black": str(pose_relative),
        "pose_overlay": str(rgb_relative),
    }
    if not overwrite and rgb_path.is_file() and black_path.is_file() and overlay_path.is_file():
        return result

    scale = DEFAULT_INTERNAL_SCALE
    geometry_scale = max(crop_width, crop_height) / REFERENCE_LONG_EDGE
    line_width = max(0.5, DEFAULT_LINE_WIDTH * geometry_scale)
    joint_radius_output = max(0.5, DEFAULT_JOINT_RADIUS * geometry_scale)
    render_size = (crop_width * scale, crop_height * scale)
    overlay_canvas = crop_rgb.resize(render_size, Image.Resampling.LANCZOS).convert("RGBA")
    black_canvas = Image.new("RGBA", render_size, (0, 0, 0, 255))
    overlay_draw = ImageDraw.Draw(overlay_canvas)
    black_draw = ImageDraw.Draw(black_canvas)
    raw_keypoints = annotation.get("keypoints", [])
    points = [
        (
            (float(raw_keypoints[index]) * coordinate_scale_x - crop_box[0]) * scale,
            (float(raw_keypoints[index + 1]) * coordinate_scale_y - crop_box[1]) * scale,
            float(raw_keypoints[index + 2]),
        )
        for index in range(0, len(raw_keypoints), 3)
    ]
    skeleton = record["skeleton"]
    for bone_index, (start, end) in enumerate(skeleton):
        if start >= len(points) or end >= len(points):
            continue
        x1, y1, visibility1 = points[start]
        x2, y2, visibility2 = points[end]
        if visibility1 <= 0 or visibility2 <= 0:
            continue
        rgba = (*bone_color(bone_index, len(skeleton)), 255)
        draw_capsule(overlay_draw, (x1, y1), (x2, y2), rgba, line_width * scale)
        draw_capsule(black_draw, (x1, y1), (x2, y2), rgba, line_width * scale)

    joint_radius = joint_radius_output * scale
    for x, y, visibility in points:
        if visibility <= 0:
            continue
        bounds = (x - joint_radius, y - joint_radius, x + joint_radius, y + joint_radius)
        overlay_draw.ellipse(bounds, fill=(*JOINT_COLOR, 255))
        black_draw.ellipse(bounds, fill=(*JOINT_COLOR, 255))

    rgb_output = crop_rgb
    overlay_output = overlay_canvas.resize((crop_width, crop_height), Image.Resampling.LANCZOS).convert("RGB")
    black_output = black_canvas.resize((crop_width, crop_height), Image.Resampling.LANCZOS).convert("RGB")
    save_rgb(rgb_output, rgb_path)
    save_rgb(overlay_output, overlay_path)
    black_path.parent.mkdir(parents=True, exist_ok=True)
    black_output.save(black_path, format="PNG", optimize=False, compress_level=1)
    result["status"] = "rendered"
    return result


def main() -> None:
    args = parse_args()
    if args.context_scale < 1.0:
        raise ValueError("--context-scale must be at least 1.0")
    dataset_root = args.dataset_root.resolve()
    rgb_root = dataset_root / args.rgb_dirname
    black_root = dataset_root / args.black_dirname
    overlay_root = dataset_root / args.overlay_dirname
    rgb_root.mkdir(parents=True, exist_ok=True)
    black_root.mkdir(parents=True, exist_ok=True)
    overlay_root.mkdir(parents=True, exist_ok=True)

    all_instances = [
        (record, annotation)
        for record in load_records(dataset_root)
        for annotation in record["annotations"]
    ]
    instances = [
        (record, annotation)
        for record, annotation in all_instances
        if annotation_has_visible_bone(record, annotation)
    ]
    if args.limit is not None:
        instances = instances[: max(0, args.limit)]
    tasks = [
        (
            record,
            annotation,
            str(dataset_root),
            str(rgb_root),
            str(black_root),
            str(overlay_root),
            float(args.context_scale),
            bool(args.overwrite),
        )
        for record, annotation in instances
    ]

    results: list[dict[str, Any]] = []
    rendered = 0
    skipped = 0
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        for index, result in enumerate(executor.map(render_instance, tasks, chunksize=8), 1):
            results.append(result)
            rendered += result["status"] == "rendered"
            skipped += result["status"] == "skipped"
            if index % 500 == 0 or index == len(tasks):
                print(f"[{index}/{len(tasks)}] rendered={rendered} skipped={skipped}", flush=True)

    removed_stale = {"rgb": 0, "pose_black": 0, "pose_overlay": 0}
    if args.prune_stale:
        expected_rgb: set[Path] = set()
        expected_pose: set[Path] = set()
        for record, annotation in instances:
            rgb_relative, pose_relative = instance_relative_paths(record["file_name"], annotation["id"])
            expected_rgb.add(rgb_relative)
            expected_pose.add(pose_relative)
        removed_stale = {
            "rgb": prune_stale_files(rgb_root, expected_rgb),
            "pose_black": prune_stale_files(black_root, expected_pose),
            "pose_overlay": prune_stale_files(overlay_root, expected_rgb),
        }

    manifest = {
        "format": "PoseAnything MP-100 instance-crop dataset manifest v1",
        "dataset_root": str(dataset_root),
        "source_images": "images",
        "rgb_crops": args.rgb_dirname,
        "pose_black_crops": args.black_dirname,
        "pose_overlay_crops": args.overlay_dirname,
        "context_scale": float(args.context_scale),
        "style": {
            "bone_color_rgb": list(UNIFORM_BONE_COLOR),
            "joint_color_rgb": list(JOINT_COLOR),
            "sizing": "relative_to_crop_long_edge_for_480px_letterbox",
            "reference_long_edge_px": REFERENCE_LONG_EDGE,
            "reference_line_width_px": DEFAULT_LINE_WIDTH,
            "reference_joint_radius_px": DEFAULT_JOINT_RADIUS,
            "internal_scale": DEFAULT_INTERNAL_SCALE,
        },
        "instances": len(results),
        "annotations_seen": len(all_instances),
        "annotations_without_visible_bone": len(all_instances) - len(instances),
        "rendered": rendered,
        "skipped": skipped,
        "removed_stale": removed_stale,
        "items": results,
    }
    manifest_path = dataset_root / "instance_crop_pose_visualization_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
