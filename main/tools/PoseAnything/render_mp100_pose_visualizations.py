#!/usr/bin/env python3
"""Render MP-100 instance-compatible pose visualizations.

The selected default style uses a single, learnable visual grammar:

* white skeleton lines and red joints on a black canvas;
* line and joint sizes proportional to the image long edge;
* 5.0 px limbs and 5.625 px joint radii after 480 px letterbox normalization.

Two paired image trees are generated beside ``images`` while preserving every
relative category/path stem:

* ``images_pose_overlay``: skeleton over RGB, preserving the source suffix;
* ``images_pose_black``: lossless PNG skeleton over a pure black canvas.

The command is incremental by default. Existing pairs are skipped unless
``--overwrite`` is supplied.
"""

from __future__ import annotations

import argparse
import colorsys
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps


DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[3] / "output" / "PoseAnything_MP100"
DEFAULT_OVERLAY_DIRNAME = "images_pose_overlay"
DEFAULT_BLACK_DIRNAME = "images_pose_black"
DEFAULT_INTERNAL_SCALE = 2
DEFAULT_LINE_WIDTH = 5.0
DEFAULT_JOINT_RADIUS = 5.625
REFERENCE_LONG_EDGE = 480.0
DEFAULT_WORKERS = max(1, min(40, (os.cpu_count() or 1)))


def color_from_env(name: str, default: tuple[int, int, int] | None = None) -> tuple[int, int, int] | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    values = tuple(int(value.strip()) for value in raw.split(","))
    if len(values) != 3 or any(value < 0 or value > 255 for value in values):
        raise ValueError(f"{name} must be R,G,B with values in [0,255], got {raw!r}")
    return values


UNIFORM_BONE_COLOR = color_from_env("MP100_UNIFORM_BONE_COLOR", (246, 246, 246))
JOINT_COLOR = color_from_env("MP100_JOINT_COLOR", (255, 48, 64))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--overlay-dirname", default=DEFAULT_OVERLAY_DIRNAME)
    parser.add_argument("--black-dirname", default=DEFAULT_BLACK_DIRNAME)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dimension-mismatch-only",
        action="store_true",
        help="Render only records whose downloaded image size differs from the annotation size.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test image limit.")
    parser.add_argument(
        "--distinct-category-limit",
        type=int,
        default=None,
        help="Select the first record from this many distinct categories for a compact style preview.",
    )
    return parser.parse_args()


def bone_color(index: int, count: int) -> tuple[int, int, int]:
    if UNIFORM_BONE_COLOR is not None:
        return UNIFORM_BONE_COLOR
    hue = (index / max(count, 1) + 0.02) % 1.0
    rgb = colorsys.hsv_to_rgb(hue, 0.78, 1.0)
    return tuple(round(channel * 255) for channel in rgb)


def draw_capsule(
    draw: ImageDraw.ImageDraw,
    p1: tuple[float, float],
    p2: tuple[float, float],
    color: tuple[int, int, int, int],
    width: float,
) -> None:
    integer_width = max(1, round(width))
    radius = integer_width / 2
    draw.line((p1, p2), fill=color, width=integer_width)
    for x, y in (p1, p2):
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def load_records(dataset_root: Path) -> list[dict[str, Any]]:
    images: dict[int, dict[str, Any]] = {}
    annotations: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
    categories: dict[int, dict[str, Any]] = {}

    annotation_paths = sorted((dataset_root / "Data").glob("mp100_split*.json"))
    if not annotation_paths:
        raise FileNotFoundError(f"No mp100_split*.json under {dataset_root / 'Data'}")

    for annotation_path in annotation_paths:
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        for image in payload.get("images", []):
            images[image["id"]] = image
        for annotation in payload.get("annotations", []):
            annotations[annotation["image_id"]][annotation["id"]] = annotation
        for category in payload.get("categories", []):
            categories[category["id"]] = category

    records = []
    for image_id, image in sorted(images.items(), key=lambda item: item[1]["file_name"]):
        source = dataset_root / "images" / image["file_name"]
        image_annotations = list(annotations.get(image_id, {}).values())
        if not source.is_file() or not image_annotations:
            continue

        category_name = image["file_name"].split("/", 1)[0]
        category_ids = {annotation["category_id"] for annotation in image_annotations}
        category = next(
            (
                categories[category_id]
                for category_id in category_ids
                if categories[category_id].get("name") == category_name
            ),
            categories[image_annotations[0]["category_id"]],
        )
        if not category.get("skeleton"):
            continue

        records.append(
            {
                "image_id": image_id,
                "file_name": image["file_name"],
                "annotation_width": int(image["width"]),
                "annotation_height": int(image["height"]),
                "category_id": category["id"],
                "category_name": category.get("name", category_name),
                "skeleton": category["skeleton"],
                "annotations": image_annotations,
            }
        )
    return records


def output_relative_paths(file_name: str) -> tuple[Path, Path]:
    source_relative = Path(file_name)
    return source_relative, source_relative.with_suffix(".png")


def render_record(task: tuple[Any, ...]) -> dict[str, Any]:
    (
        record,
        dataset_root_text,
        overlay_root_text,
        black_root_text,
        overwrite,
    ) = task
    dataset_root = Path(dataset_root_text)
    overlay_root = Path(overlay_root_text)
    black_root = Path(black_root_text)
    source_path = dataset_root / "images" / record["file_name"]
    overlay_relative, black_relative = output_relative_paths(record["file_name"])
    overlay_path = overlay_root / overlay_relative
    black_path = black_root / black_relative

    if not overwrite and overlay_path.is_file() and black_path.is_file():
        return {"status": "skipped", "file_name": record["file_name"]}

    with Image.open(source_path) as source_image:
        source_rgb = ImageOps.exif_transpose(source_image).convert("RGB")
    width, height = source_rgb.size
    annotation_width = max(1, int(record["annotation_width"]))
    annotation_height = max(1, int(record["annotation_height"]))
    coordinate_scale_x = width / annotation_width
    coordinate_scale_y = height / annotation_height
    scale = DEFAULT_INTERNAL_SCALE
    geometry_scale = max(width, height) / REFERENCE_LONG_EDGE
    line_width = max(0.5, DEFAULT_LINE_WIDTH * geometry_scale)
    joint_radius_output = max(0.5, DEFAULT_JOINT_RADIUS * geometry_scale)
    render_size = (width * scale, height * scale)

    overlay_canvas = source_rgb.resize(render_size, Image.Resampling.LANCZOS).convert("RGBA")
    black_canvas = Image.new("RGBA", render_size, (0, 0, 0, 255))
    overlay_draw = ImageDraw.Draw(overlay_canvas)
    black_draw = ImageDraw.Draw(black_canvas)
    skeleton = record["skeleton"]

    for annotation in record["annotations"]:
        raw_keypoints = annotation.get("keypoints", [])
        if not raw_keypoints or len(raw_keypoints) % 3:
            continue
        points = [
            (
                float(raw_keypoints[index]) * coordinate_scale_x * scale,
                float(raw_keypoints[index + 1]) * coordinate_scale_y * scale,
                float(raw_keypoints[index + 2]),
            )
            for index in range(0, len(raw_keypoints), 3)
        ]

        for bone_index, (start, end) in enumerate(skeleton):
            if start >= len(points) or end >= len(points):
                continue
            x1, y1, visibility1 = points[start]
            x2, y2, visibility2 = points[end]
            if visibility1 <= 0 or visibility2 <= 0:
                continue
            rgba = (*bone_color(bone_index, len(skeleton)), 255)
            draw_capsule(
                overlay_draw,
                (x1, y1),
                (x2, y2),
                rgba,
                line_width * scale,
            )
            draw_capsule(
                black_draw,
                (x1, y1),
                (x2, y2),
                rgba,
                line_width * scale,
            )

        joint_radius = joint_radius_output * scale
        for x, y, visibility in points:
            if visibility <= 0:
                continue
            bounds = (x - joint_radius, y - joint_radius, x + joint_radius, y + joint_radius)
            overlay_draw.ellipse(bounds, fill=(*JOINT_COLOR, 255))
            black_draw.ellipse(bounds, fill=(*JOINT_COLOR, 255))

    overlay_output = overlay_canvas.resize((width, height), Image.Resampling.LANCZOS).convert("RGB")
    black_output = black_canvas.resize((width, height), Image.Resampling.LANCZOS).convert("RGB")
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    black_path.parent.mkdir(parents=True, exist_ok=True)
    if overlay_path.suffix.lower() in {".jpg", ".jpeg"}:
        overlay_output.save(overlay_path, format="JPEG", quality=95, subsampling=0)
        legacy_overlay_png = overlay_path.with_suffix(".png")
        if legacy_overlay_png != overlay_path and legacy_overlay_png.is_file():
            legacy_overlay_png.unlink()
    else:
        overlay_output.save(overlay_path, format="PNG", optimize=False, compress_level=1)
    black_output.save(black_path, format="PNG", optimize=False, compress_level=1)

    return {
        "status": "rendered",
        "file_name": record["file_name"],
        "overlay": str(overlay_path),
        "black": str(black_path),
        "width": width,
        "height": height,
        "annotation_width": annotation_width,
        "annotation_height": annotation_height,
        "coordinate_scale_x": coordinate_scale_x,
        "coordinate_scale_y": coordinate_scale_y,
        "instances": len(record["annotations"]),
    }


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    overlay_root = dataset_root / args.overlay_dirname
    black_root = dataset_root / args.black_dirname
    overlay_root.mkdir(parents=True, exist_ok=True)
    black_root.mkdir(parents=True, exist_ok=True)

    records = load_records(dataset_root)
    if args.distinct_category_limit is not None:
        selected = []
        seen_categories = set()
        for record in records:
            category_key = record["file_name"].split("/", 1)[0]
            if category_key in seen_categories:
                continue
            selected.append(record)
            seen_categories.add(category_key)
            if len(selected) >= max(0, args.distinct_category_limit):
                break
        records = selected
    if args.dimension_mismatch_only:
        mismatched_records = []
        for record in records:
            source_path = dataset_root / "images" / record["file_name"]
            with Image.open(source_path) as source_image:
                actual_size = ImageOps.exif_transpose(source_image).size
            annotation_size = (record["annotation_width"], record["annotation_height"])
            if actual_size != annotation_size:
                mismatched_records.append(record)
        records = mismatched_records
    if args.limit is not None:
        records = records[: max(0, args.limit)]

    tasks = [
        (
            record,
            str(dataset_root),
            str(overlay_root),
            str(black_root),
            args.overwrite,
        )
        for record in records
    ]
    rendered = 0
    skipped = 0
    failures: list[dict[str, str]] = []
    results: list[dict[str, Any]] = []
    workers = max(1, args.workers)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for index, result in enumerate(executor.map(render_record, tasks, chunksize=8), 1):
            results.append(result)
            rendered += result["status"] == "rendered"
            skipped += result["status"] == "skipped"
            if index % 500 == 0 or index == len(tasks):
                print(
                    f"[{index}/{len(tasks)}] rendered={rendered} skipped={skipped} failures={len(failures)}",
                    flush=True,
                )

    manifest = {
        "format": "PoseAnything MP-100 white-bone red-joint visualization manifest v2",
        "dataset_root": str(dataset_root),
        "source_images": "images",
        "overlay_images": args.overlay_dirname,
        "black_images": args.black_dirname,
        "style": {
            "palette": "uniform white bones",
            "bone_color_rgb": list(UNIFORM_BONE_COLOR),
            "joint_color_rgb": list(JOINT_COLOR),
            "internal_scale": DEFAULT_INTERNAL_SCALE,
            "sizing": "relative_to_image_long_edge_for_480px_letterbox",
            "reference_long_edge_px": REFERENCE_LONG_EDGE,
            "reference_line_width_px": DEFAULT_LINE_WIDTH,
            "reference_joint_radius_px": DEFAULT_JOINT_RADIUS,
        },
        "output_format": {
            "overlay": "source suffix (JPEG quality 95 or PNG)",
            "black": "PNG",
        },
        "relative_path_rule": {
            "overlay": "same relative file_name as source",
            "black": "source relative file_name with suffix replaced by .png",
        },
        "records": len(records),
        "selection": "dimension_mismatch_only" if args.dimension_mismatch_only else "all",
        "rendered": rendered,
        "skipped": skipped,
        "failures": failures,
        "items": results,
    }
    manifest_path = dataset_root / "pose_visualization_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
