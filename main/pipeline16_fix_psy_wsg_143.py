#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite ShapeNetSem WSG channel 143 as a per-pixel material-region mask "
            "projected from the fixed bird camera used by pipeline16."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "output" / "ShapeNetSem_results",
        help="ShapeNetSem_results directory.",
    )
    parser.add_argument(
        "--wsg-dir",
        type=Path,
        default=None,
        help="WSG root directory. Defaults to <dataset-dir>/wsg.",
    )
    parser.add_argument(
        "--mapping-path",
        type=Path,
        default=None,
        help=(
            "Path to the material-region mapping JSON. "
            "Defaults to <dataset-dir>/material_channel_143_region_mapping.json."
        ),
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=143,
        help="Target channel to rewrite. Default: 143.",
    )
    parser.add_argument(
        "--alpha-threshold",
        type=float,
        default=20.0,
        help="Only pixels with channel-3 alpha greater than this are considered foreground.",
    )
    parser.add_argument(
        "--camera-distance",
        type=float,
        default=2.0,
        help="Bird camera distance. Matches blender_script_shapenet.py.",
    )
    parser.add_argument(
        "--lens-mm",
        type=float,
        default=35.0,
        help="Camera lens in mm. Matches blender_script_shapenet.py.",
    )
    parser.add_argument(
        "--sensor-width-mm",
        type=float,
        default=32.0,
        help="Camera sensor width in mm. Matches blender_script_shapenet.py.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(os.cpu_count() or 1, 8),
        help="Number of worker threads.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N WSG files for testing.",
    )
    parser.add_argument(
        "--uuids",
        nargs="*",
        default=None,
        help="Optional explicit UUID list to process.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Compute mappings/stats but do not rewrite WSG files.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create .bak143region backups before in-place modification.",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
        help="Optional directory for overlay/debug PNGs and stats JSON.",
    )
    parser.add_argument(
        "--projection-mode",
        type=str,
        default="auto",
        choices=["auto", "orig", "flip_x", "flip_y", "flip_xy"],
        help="How to align projected material regions to the rendered image.",
    )
    parser.add_argument(
        "--mask-source",
        type=str,
        default="geometry",
        choices=["geometry", "blender"],
        help="Use geometric reprojection or a Blender material-id pass to build channel 143.",
    )
    parser.add_argument(
        "--render-python",
        type=Path,
        default=Path("/research/cbim/vast/sf895/miniforge3/envs/render/bin/python"),
        help="Python executable from the render environment with bpy installed.",
    )
    parser.add_argument(
        "--material-pass-script",
        type=Path,
        default=Path(__file__).resolve().parent / "utils" / "blender_material_region_pass.py",
        help="Helper script that renders the material region pass using bpy.",
    )
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_wsg_array(path):
    data = np.load(path, allow_pickle=True)
    if "wsg_data" in data:
        key = "wsg_data"
    elif "arr_0" in data:
        key = "arr_0"
    else:
        key = data.files[0]
    return data[key], key


def save_wsg_array(arr, key, output_path):
    output_path = Path(output_path)
    base_path = output_path.with_suffix("")
    np.savez_compressed(base_path, **{key: arr.astype(np.float32)})
    npz_path = base_path.with_suffix(".npz")
    if output_path.exists():
        output_path.unlink()
    npz_path.rename(output_path)


def find_metadata_txt(wsg_path):
    candidates = sorted(wsg_path.parent.glob("*.txt"))
    return candidates[0] if candidates else None


def natural_key(text):
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", text)]


def resolve_obj_path(dataset_dir, wsg_path, metadata):
    uuid = wsg_path.parent.name
    primary = dataset_dir / "models-OBJ" / "models" / f"{uuid}.obj"
    if primary.exists():
        return primary

    additional = metadata.get("additional_data", {})
    obj_path = additional.get("obj_path")
    if isinstance(obj_path, str) and obj_path.strip():
        candidates = [
            dataset_dir / Path(obj_path).name,
            wsg_path.parent / Path(obj_path).name,
            (dataset_dir / obj_path).resolve(),
        ]
        for candidate in candidates:
            if Path(candidate).exists():
                return Path(candidate)

    return None


def parse_vertex_index(token, vertex_count):
    raw = token.split("/")[0]
    idx = int(raw)
    if idx < 0:
        return vertex_count + idx
    return idx - 1


def parse_obj_with_materials(obj_path):
    vertices = []
    triangles = []
    triangle_materials = []
    current_material = "unknown"

    with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("v "):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("usemtl "):
                current_material = line.split(maxsplit=1)[1].strip()
            elif line.startswith("f "):
                parts = line.split()[1:]
                if len(parts) < 3:
                    continue
                face = [parse_vertex_index(token, len(vertices)) for token in parts]
                for i in range(1, len(face) - 1):
                    triangles.append([face[0], face[i], face[i + 1]])
                    triangle_materials.append(current_material)

    if not vertices or not triangles:
        return None, None, None

    return (
        np.asarray(vertices, dtype=np.float32),
        np.asarray(triangles, dtype=np.int32),
        triangle_materials,
    )


def normalize_vertices(vertices):
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    bbox_size = bbox_max - bbox_min
    max_extent = float(np.max(bbox_size))
    if max_extent <= 0:
        return vertices.copy()
    scale = 1.0 / max_extent
    center = (bbox_min + bbox_max) / 2.0
    return vertices * scale - center * scale


def build_bird_camera(distance):
    camera_position = np.asarray(
        [distance * 0.577, distance * 0.577, distance * 0.816], dtype=np.float32
    )
    target = np.zeros(3, dtype=np.float32)
    forward = target - camera_position
    forward /= np.linalg.norm(forward)

    world_up = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(forward, world_up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-8:
        right = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        right /= right_norm

    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    return camera_position, right, up, forward


def project_vertices(vertices, width, height, distance, lens_mm, sensor_width_mm):
    camera_position, right, up, forward = build_bird_camera(distance)
    relative = vertices - camera_position[None, :]

    x_cam = relative @ right
    y_cam = relative @ up
    z_cam = relative @ forward

    focal_x = (lens_mm / sensor_width_mm) * width
    focal_y = (lens_mm / sensor_width_mm) * height
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0

    pixels = np.full((vertices.shape[0], 2), np.nan, dtype=np.float32)
    valid = z_cam > 1e-6
    pixels[valid, 0] = focal_x * (x_cam[valid] / z_cam[valid]) + cx
    pixels[valid, 1] = cy - focal_y * (y_cam[valid] / z_cam[valid])
    return pixels, z_cam, valid


def edge_function(ax, ay, bx, by, px, py):
    return (px - ax) * (by - ay) - (py - ay) * (bx - ax)


def rasterize_material_regions(
    pixels,
    depths,
    triangles,
    triangle_region_ids,
    alpha_mask,
):
    height, width = alpha_mask.shape
    label_map = np.full((height, width), -1, dtype=np.int32)
    depth_buffer = np.full((height, width), np.inf, dtype=np.float32)

    for tri_index, triangle in enumerate(triangles):
        pts = pixels[triangle]
        tri_depths = depths[triangle]
        if np.isnan(pts).any() or np.any(tri_depths <= 1e-6):
            continue

        min_x = max(int(np.floor(np.min(pts[:, 0]))), 0)
        max_x = min(int(np.ceil(np.max(pts[:, 0]))), width - 1)
        min_y = max(int(np.floor(np.min(pts[:, 1]))), 0)
        max_y = min(int(np.ceil(np.max(pts[:, 1]))), height - 1)
        if min_x > max_x or min_y > max_y:
            continue

        alpha_slice = alpha_mask[min_y : max_y + 1, min_x : max_x + 1]
        if not alpha_slice.any():
            continue

        x = np.arange(min_x, max_x + 1, dtype=np.float32) + 0.5
        y = np.arange(min_y, max_y + 1, dtype=np.float32) + 0.5
        grid_x, grid_y = np.meshgrid(x, y)

        x0, y0 = pts[0]
        x1, y1 = pts[1]
        x2, y2 = pts[2]
        area = edge_function(x0, y0, x1, y1, x2, y2)
        if abs(area) < 1e-8:
            continue

        w0 = edge_function(x1, y1, x2, y2, grid_x, grid_y) / area
        w1 = edge_function(x2, y2, x0, y0, grid_x, grid_y) / area
        w2 = edge_function(x0, y0, x1, y1, grid_x, grid_y) / area

        inside = (
            (w0 >= -1e-6)
            & (w1 >= -1e-6)
            & (w2 >= -1e-6)
            & alpha_slice
        )
        if not inside.any():
            continue

        tri_depth = w0 * tri_depths[0] + w1 * tri_depths[1] + w2 * tri_depths[2]
        current_depth = depth_buffer[min_y : max_y + 1, min_x : max_x + 1]
        update = inside & (tri_depth < current_depth)
        if not update.any():
            continue

        current_depth[update] = tri_depth[update]
        depth_buffer[min_y : max_y + 1, min_x : max_x + 1] = current_depth

        current_label = label_map[min_y : max_y + 1, min_x : max_x + 1]
        current_label[update] = triangle_region_ids[tri_index]
        label_map[min_y : max_y + 1, min_x : max_x + 1] = current_label

    return label_map


def apply_projection_mode(pixels, width, height, mode):
    transformed = pixels.copy()
    if mode in {"flip_x", "flip_xy"}:
        transformed[:, 0] = (width - 1) - transformed[:, 0]
    if mode in {"flip_y", "flip_xy"}:
        transformed[:, 1] = (height - 1) - transformed[:, 1]
    return transformed


def choose_best_label_map(pixels, depths, triangles, triangle_region_ids, alpha_mask, projection_mode):
    height, width = alpha_mask.shape
    if projection_mode != "auto":
        transformed = apply_projection_mode(pixels, width, height, projection_mode)
        label_map = rasterize_material_regions(
            transformed, depths, triangles, triangle_region_ids, alpha_mask
        )
        labeled_pixels = int(((label_map >= 0) & alpha_mask).sum())
        return label_map, projection_mode, labeled_pixels

    best = None
    for mode in ["orig", "flip_x", "flip_y", "flip_xy"]:
        transformed = apply_projection_mode(pixels, width, height, mode)
        label_map = rasterize_material_regions(
            transformed, depths, triangles, triangle_region_ids, alpha_mask
        )
        labeled_pixels = int(((label_map >= 0) & alpha_mask).sum())
        if best is None or labeled_pixels > best[2]:
            best = (label_map, mode, labeled_pixels)
    return best


def create_region_mapping(material_names):
    ordered = sorted(set(material_names), key=natural_key)
    material_to_region_id = {name: idx for idx, name in enumerate(ordered)}
    region_id_to_material = {str(idx): name for name, idx in material_to_region_id.items()}
    return material_to_region_id, region_id_to_material


def render_material_regions_with_blender(
    obj_path,
    resolution,
    render_python,
    material_pass_script,
    camera_distance,
):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        output_png = tmpdir / "material_regions.png"
        output_json = tmpdir / "material_regions.json"
        cmd = [
            str(render_python),
            str(material_pass_script),
            "--object_path",
            str(obj_path),
            "--output_png",
            str(output_png),
            "--output_json",
            str(output_json),
            "--resolution",
            str(resolution),
            "--fixed_camera",
            "bird",
            "--camera_distance",
            str(camera_distance),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Material pass failed for {obj_path}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )

        metadata = load_json(output_json)
        rgba = np.array(Image.open(output_png).convert("RGBA"), dtype=np.uint8)
        return rgba, metadata


def decode_region_ids_from_rgba(rgba, region_id_to_rgb, foreground_mask):
    label_map = np.full(foreground_mask.shape, -1, dtype=np.int32)
    if not foreground_mask.any():
        return label_map

    rgb = rgba[:, :, :3].astype(np.int16)
    region_items = sorted((int(k), v) for k, v in region_id_to_rgb.items())
    palette = np.asarray([item[1] for item in region_items], dtype=np.int16)
    region_ids = np.asarray([item[0] for item in region_items], dtype=np.int32)

    foreground_rgb = rgb[foreground_mask]
    distances = np.sum((foreground_rgb[:, None, :] - palette[None, :, :]) ** 2, axis=2)
    nearest = np.argmin(distances, axis=1)
    label_map[foreground_mask] = region_ids[nearest]
    return label_map


def make_palette(size):
    base_colors = np.asarray(
        [
            [230, 57, 70],
            [69, 123, 157],
            [42, 157, 143],
            [244, 162, 97],
            [233, 196, 106],
            [141, 153, 174],
            [239, 71, 111],
            [6, 214, 160],
            [17, 138, 178],
            [131, 56, 236],
            [255, 107, 107],
            [77, 144, 142],
            [188, 108, 37],
            [58, 134, 255],
            [255, 183, 3],
        ],
        dtype=np.uint8,
    )
    if size <= len(base_colors):
        return base_colors[:size]
    repeat = int(np.ceil(size / len(base_colors)))
    return np.tile(base_colors, (repeat, 1))[:size]


def write_debug_outputs(debug_dir, uuid, wsg_array, label_map, alpha_mask, region_id_to_material):
    debug_dir.mkdir(parents=True, exist_ok=True)
    rgb = np.clip(wsg_array[:, :, :3], 0, 255).astype(np.uint8)
    overlay = rgb.copy()
    region_rgb = np.zeros_like(rgb)

    region_ids = sorted(int(key) for key in region_id_to_material.keys())
    palette = make_palette(len(region_ids))
    for palette_index, region_id in enumerate(region_ids):
        mask = label_map == region_id
        color = palette[palette_index]
        region_rgb[mask] = color
        overlay[mask] = (0.55 * overlay[mask] + 0.45 * color).astype(np.uint8)

    alpha_vis = np.zeros_like(rgb)
    alpha_vis[alpha_mask] = 255

    Image.fromarray(rgb).save(debug_dir / f"{uuid}_base_rgb.png")
    Image.fromarray(region_rgb).save(debug_dir / f"{uuid}_channel143_region_ids.png")
    Image.fromarray(overlay).save(debug_dir / f"{uuid}_channel143_overlay.png")
    Image.fromarray(alpha_vis).save(debug_dir / f"{uuid}_channel143_alpha_mask.png")


def process_single_wsg(task):
    (
        wsg_path,
        dataset_dir,
        channel,
        alpha_threshold,
        camera_distance,
        lens_mm,
        sensor_width_mm,
        report_only,
        make_backup,
        debug_dir,
        projection_mode,
        mask_source,
        render_python,
        material_pass_script,
    ) = task

    txt_path = find_metadata_txt(wsg_path)
    if txt_path is None:
        return {"path": str(wsg_path), "changed": False, "error": "missing_metadata"}

    metadata = load_json(txt_path)
    obj_path = resolve_obj_path(dataset_dir, wsg_path, metadata)
    if obj_path is None:
        return {"path": str(wsg_path), "changed": False, "error": "missing_obj"}

    vertices, triangles, triangle_materials = parse_obj_with_materials(obj_path)
    if vertices is None:
        return {"path": str(wsg_path), "changed": False, "error": "empty_obj"}

    wsg_array, key = load_wsg_array(wsg_path)
    height, width = wsg_array.shape[:2]
    alpha_mask = wsg_array[:, :, 3] > alpha_threshold

    foreground_pixels = int(alpha_mask.sum())

    if mask_source == "blender":
        rgba, blender_meta = render_material_regions_with_blender(
            obj_path=obj_path,
            resolution=width,
            render_python=render_python,
            material_pass_script=material_pass_script,
            camera_distance=camera_distance,
        )
        material_to_region_id = {
            str(name): int(region_id)
            for name, region_id in blender_meta["material_name_to_region_id"].items()
        }
        region_id_to_material = {
            str(region_id): str(name)
            for region_id, name in blender_meta["region_id_to_material_name"].items()
        }
        label_map = decode_region_ids_from_rgba(
            rgba=rgba,
            region_id_to_rgb=blender_meta["region_id_to_rgb"],
            foreground_mask=alpha_mask,
        )
        chosen_projection_mode = "blender_pass"
        labeled_pixels = int(((label_map >= 0) & alpha_mask).sum())
    else:
        material_to_region_id, region_id_to_material = create_region_mapping(triangle_materials)
        triangle_region_ids = np.asarray(
            [material_to_region_id[name] for name in triangle_materials], dtype=np.int32
        )
        normalized_vertices = normalize_vertices(vertices)
        pixels, depths, valid_vertices = project_vertices(
            normalized_vertices,
            width=width,
            height=height,
            distance=camera_distance,
            lens_mm=lens_mm,
            sensor_width_mm=sensor_width_mm,
        )
        if not valid_vertices.any():
            return {"path": str(wsg_path), "changed": False, "error": "no_valid_projection"}

        label_map, chosen_projection_mode, labeled_pixels = choose_best_label_map(
            pixels=pixels,
            depths=depths,
            triangles=triangles,
            triangle_region_ids=triangle_region_ids,
            alpha_mask=alpha_mask,
            projection_mode=projection_mode,
        )

    unlabeled_foreground = foreground_pixels - labeled_pixels
    unique_region_ids = np.unique(label_map[label_map >= 0]).tolist()

    object_mapping = {
        "object_uuid": wsg_path.parent.name,
        "obj_path": str(obj_path),
        "camera": "bird",
        "channel": channel,
        "material_name_to_region_id": material_to_region_id,
        "region_id_to_material_name": region_id_to_material,
        "region_count": len(material_to_region_id),
        "alpha_threshold": alpha_threshold,
        "projection_mode": chosen_projection_mode,
        "mask_source": mask_source,
        "alpha_foreground_pixels": foreground_pixels,
        "labeled_foreground_pixels": labeled_pixels,
        "unlabeled_foreground_pixels": unlabeled_foreground,
    }

    if debug_dir is not None:
        write_debug_outputs(
            debug_dir=debug_dir,
            uuid=wsg_path.parent.name,
            wsg_array=wsg_array,
            label_map=label_map,
            alpha_mask=alpha_mask,
            region_id_to_material=region_id_to_material,
        )
        save_json(
            {
                **object_mapping,
                "projected_region_ids_present": unique_region_ids,
            },
            debug_dir / f"{wsg_path.parent.name}_channel143_stats.json",
        )

    if report_only:
        return {
            "path": str(wsg_path),
            "changed": False,
            "mapping": object_mapping,
        }

    if make_backup:
        backup_path = wsg_path.with_suffix(wsg_path.suffix + ".bak143region")
        if not backup_path.exists():
            shutil.copy2(wsg_path, backup_path)

    wsg_array[:, :, channel] = -1.0
    write_mask = alpha_mask & (label_map >= 0)
    wsg_array[write_mask, channel] = label_map[write_mask].astype(np.float32)
    save_wsg_array(wsg_array, key, wsg_path)

    return {
        "path": str(wsg_path),
        "changed": True,
        "mapping": object_mapping,
    }


def main():
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    wsg_dir = (args.wsg_dir or (dataset_dir / "wsg")).resolve()
    mapping_path = (
        args.mapping_path
        or (dataset_dir / "material_channel_143_region_mapping.json")
    ).resolve()
    debug_dir = args.debug_dir.resolve() if args.debug_dir is not None else None

    if args.uuids:
        wsg_paths = [wsg_dir / uuid / f"{uuid}.wsg" for uuid in args.uuids]
        wsg_paths = [path for path in wsg_paths if path.exists()]
    else:
        wsg_paths = sorted(wsg_dir.glob("*/*.wsg"))

    if args.limit is not None:
        wsg_paths = wsg_paths[: args.limit]

    if not wsg_paths:
        print(f"No WSG files found under {wsg_dir}", flush=True)
        return 1

    tasks = [
        (
            wsg_path,
            dataset_dir,
            args.channel,
            args.alpha_threshold,
            args.camera_distance,
            args.lens_mm,
            args.sensor_width_mm,
            args.report_only,
            not args.no_backup,
            debug_dir,
            args.projection_mode,
            args.mask_source,
            args.render_python.resolve(),
            args.material_pass_script.resolve(),
        )
        for wsg_path in wsg_paths
    ]

    if mapping_path.exists():
        mapping_data = load_json(mapping_path)
    else:
        mapping_data = {"objects": {}}
    mapping_data.setdefault("objects", {})

    processed = 0
    changed = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        for result in executor.map(process_single_wsg, tasks):
            processed += 1
            if result.get("mapping"):
                mapping_data["objects"][result["mapping"]["object_uuid"]] = result["mapping"]
            if result.get("changed"):
                changed += 1
            if result.get("error"):
                errors += 1
                print(
                    f"Skipped {Path(result['path']).name}: {result['error']}",
                    flush=True,
                )
            if processed % 10 == 0 or processed == len(tasks):
                print(f"Processed {processed}/{len(tasks)} WSG files", flush=True)

    save_json(mapping_data, mapping_path)
    print(f"Saved mapping: {mapping_path}", flush=True)
    print(f"Changed WSG files: {changed}", flush=True)
    print(f"Errors: {errors}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
