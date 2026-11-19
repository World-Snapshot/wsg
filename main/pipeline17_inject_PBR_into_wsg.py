#!/usr/bin/env python3
"""
Generate MatSynth WSG files from local parquet source data.

This pipeline does not use existing WSG files as the source of truth.
Instead, it reads MatSynth parquet rows, decodes the material maps, renders
preview RGB/alpha, and writes fresh WSG files plus crop_frame PNGs.

Key behaviors:
- Channels 0-2: rendered RGB preview
- Channel 3: rendered alpha driven by opacity when available
- Channels 36-38, 161-173: control maps written to WSG
- Basecolor and diffuse are mutually exclusive per sample:
  one is retained, the other is written as -1
- Variation generation supports:
  - combinatorial: closer to the current dataset scale
  - single_factor: easier to inspect and debug
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
from itertools import product
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image


RGB_CHANNELS = {
    "front": (0, 1, 2),
    "normal": (36, 37, 38),
    "basecolor": (161, 162, 163),
    "diffuse": (164, 165, 166),
    "specular": (170, 171, 172),
}

SCALAR_CHANNELS = {
    "height": 167,
    "roughness": 168,
    "metallic": 169,
    "opacity": 173,
}

LEVEL_VALUES_7 = [0.0, 1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0, 4.0 / 6.0, 5.0 / 6.0, 1.0]
BRIGHTNESS_VALUES = LEVEL_VALUES_7
ABSOLUTE_VALUES = LEVEL_VALUES_7
SPECULAR_SCALE_VALUES = LEVEL_VALUES_7
OPACITY_VALUES = LEVEL_VALUES_7


def parse_args():
    root = Path(__file__).resolve().parent.parent / "output" / "MatSynth_results"
    parser = argparse.ArgumentParser(description="Generate MatSynth WSG files from parquet.")
    parser.add_argument(
        "--parquet-dir",
        type=Path,
        default=root / "data",
        help="Directory containing local MatSynth parquet shards.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "wsg_regenerated",
        help="Directory to write generated WSG folders.",
    )
    parser.add_argument(
        "--reference-wsg-dir",
        type=Path,
        default=root / "wsg",
        help="Optional existing WSG directory used only for material subset and count hints.",
    )
    parser.add_argument(
        "--use-reference-materials",
        action="store_true",
        help="Limit generation to material directory names present under --reference-wsg-dir.",
    )
    parser.add_argument(
        "--use-reference-counts",
        action="store_true",
        help="Use existing per-material aug counts from --reference-wsg-dir when available.",
    )
    parser.add_argument(
        "--materials",
        nargs="*",
        default=None,
        help="Optional explicit material names to generate.",
    )
    parser.add_argument(
        "--limit-materials",
        type=int,
        default=None,
        help="Only generate the first N matched materials.",
    )
    parser.add_argument(
        "--variation-mode",
        choices=["combinatorial", "single_factor"],
        default="combinatorial",
        help="How to create control-map variants.",
    )
    parser.add_argument(
        "--max-augs-per-material",
        type=int,
        default=None,
        help="Optional cap on generated aug files per material.",
    )
    parser.add_argument(
        "--levels-json",
        type=str,
        default=None,
        help="Optional JSON list overriding the shared 7-level schedule, e.g. '[0,0.1,0.2,...,1]'.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=300,
        help="Output WSG width/height.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel worker count.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in the output directory.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Plan outputs without writing files.",
    )
    parser.add_argument(
        "--render-backend",
        choices=["blender", "numpy"],
        default="numpy",
        help="Use the lightweight numpy renderer for bulk generation or Blender GPU rendering for spot checks.",
    )
    parser.add_argument(
        "--render-python",
        type=Path,
        default=Path("/research/cbim/vast/sf895/miniforge3/envs/render/bin/python"),
        help="Python executable for the render environment.",
    )
    parser.add_argument(
        "--blend-file",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "output" / "MatSynth_results" / "scripts" / "render_ambient.blend",
        help="Blend file used for GPU rendering.",
    )
    parser.add_argument(
        "--render-helper",
        type=Path,
        default=Path(__file__).resolve().parent / "utils" / "matsynth_blender_render.py",
        help="bpy helper that renders a material preview from temporary maps.",
    )
    parser.add_argument(
        "--render-batch-helper",
        type=Path,
        default=Path(__file__).resolve().parent / "utils" / "matsynth_blender_render_batch.py",
        help="bpy helper that batch-renders many variants in one Blender session.",
    )
    parser.add_argument(
        "--temp-root",
        type=Path,
        default=Path("/tmp/matsynth_pipeline17"),
        help="Temporary directory used to stage variant maps before rendering.",
    )
    parser.add_argument(
        "--render-samples",
        type=int,
        default=32,
        help="Cycles sample count for Blender rendering.",
    )
    parser.add_argument(
        "--render-batch-size",
        type=int,
        default=32,
        help="How many variants to render per long-lived Blender batch before writing outputs.",
    )
    parser.add_argument(
        "--gpu-count",
        type=int,
        default=8,
        help="Number of GPUs to shard material renders across.",
    )
    parser.add_argument(
        "--compute-device",
        choices=["OPTIX", "CUDA"],
        default="OPTIX",
        help="Cycles device type.",
    )
    return parser.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def decode_image_struct(image_struct, target_size: int, is_rgb: bool):
    if image_struct is None:
        return None
    if isinstance(image_struct, dict):
        raw = image_struct.get("bytes")
    else:
        raw = None
    if not raw:
        return None

    with Image.open(BytesIO(raw)) as img:
        if is_rgb:
            img = img.convert("RGB")
            if img.size != (target_size, target_size):
                img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
            arr = np.asarray(img, dtype=np.float32) / 255.0
            return arr

        arr = np.asarray(img)
        if img.size != (target_size, target_size):
            pil_mode = "I;16" if arr.dtype == np.uint16 else "L"
            scalar_img = Image.fromarray(arr, mode=pil_mode)
            scalar_img = scalar_img.resize((target_size, target_size), Image.Resampling.LANCZOS)
            arr = np.asarray(scalar_img)
        if arr.dtype == np.uint16:
            arr = arr.astype(np.float32) / 65535.0
        else:
            arr = arr.astype(np.float32) / 255.0
        return np.clip(arr, 0.0, 1.0)


def decode_material_row(row: dict, target_size: int):
    return {
        "basecolor": decode_image_struct(row.get("basecolor"), target_size, True),
        "diffuse": decode_image_struct(row.get("diffuse"), target_size, True),
        "normal": decode_image_struct(row.get("normal"), target_size, True),
        "specular": decode_image_struct(row.get("specular"), target_size, True),
        "roughness": decode_image_struct(row.get("roughness"), target_size, False),
        "metallic": decode_image_struct(row.get("metallic"), target_size, False),
        "height": decode_image_struct(row.get("height") or row.get("displacement"), target_size, False),
        "opacity": decode_image_struct(row.get("opacity"), target_size, False),
    }


def save_wsg_array(arr: np.ndarray, output_path: Path):
    output_path = Path(output_path)
    base_path = output_path.with_suffix("")
    np.savez_compressed(base_path, wsg_data=arr.astype(np.float16))
    npz_path = base_path.with_suffix(".npz")
    if output_path.exists():
        output_path.unlink()
    npz_path.rename(output_path)


def write_crop_frame(output_path: Path, rgb: np.ndarray, alpha: np.ndarray):
    rgb_uint8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    alpha_uint8 = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    if np.all(alpha_uint8 == 255):
        Image.fromarray(rgb_uint8, mode="RGB").save(output_path)
    else:
        rgba = np.dstack([rgb_uint8, alpha_uint8])
        Image.fromarray(rgba, mode="RGBA").save(output_path)


def save_rgb_png(data: np.ndarray, output_path: Path):
    arr = np.clip(data * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(output_path)


def save_scalar_png(data: np.ndarray, output_path: Path, bit16: bool = False):
    if bit16:
        arr = np.clip(data * 65535.0, 0, 65535).astype(np.uint16)
        Image.fromarray(arr, mode="I;16").save(output_path)
    else:
        arr = np.clip(data * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(arr, mode="L").save(output_path)


def scalar_constant_value(data: np.ndarray | None, atol: float = 1e-6):
    if data is None:
        return None
    arr = np.asarray(data)
    if arr.ndim != 2:
        return None
    first = float(arr[0, 0])
    if np.allclose(arr, first, atol=atol):
        return first
    return None


def rgb_constant_luma(data: np.ndarray | None, atol: float = 1e-6):
    if data is None:
        return None
    arr = np.asarray(data)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return None
    first = arr[0, 0].astype(np.float32)
    if np.allclose(arr, first, atol=atol):
        return float(0.2126 * first[0] + 0.7152 * first[1] + 0.0722 * first[2])
    return None


def stage_variant_maps(rendered_maps: dict, task_name: str, variant_index: int, size: int, temp_root: Path):
    variant_dir = temp_root / task_name / f"variant_{variant_index:05d}"
    ensure_dir(variant_dir)

    albedo = rendered_maps["basecolor"] if rendered_maps["basecolor"] is not None else rendered_maps["diffuse"]
    save_rgb_png(albedo, variant_dir / "albedo.png")
    save_rgb_png(rendered_maps["normal"], variant_dir / "normal.png")
    save_scalar_png(rendered_maps["roughness"], variant_dir / "roughness.png")
    save_scalar_png(rendered_maps["metallic"], variant_dir / "metallic.png")
    save_scalar_png(rendered_maps["height"] if rendered_maps["height"] is not None else np.ones((size, size), dtype=np.float32), variant_dir / "height.png", bit16=True)
    if rendered_maps["specular"] is not None:
        save_rgb_png(rendered_maps["specular"], variant_dir / "specular.png")
    else:
        save_rgb_png(np.full((size, size, 3), 0.04, dtype=np.float32), variant_dir / "specular.png")
    if rendered_maps["opacity"] is not None:
        save_scalar_png(rendered_maps["opacity"], variant_dir / "opacity.png")
    return variant_dir, {
        "roughness_constant": scalar_constant_value(rendered_maps["roughness"]),
        "metallic_constant": scalar_constant_value(rendered_maps["metallic"]),
        "opacity_constant": scalar_constant_value(rendered_maps["opacity"]),
        "specular_constant": rgb_constant_luma(rendered_maps["specular"]),
    }


def run_blender_batch(material_name: str, render_jobs: list[dict], render_python: Path, blend_file: Path, render_batch_helper: Path, render_samples: int, compute_device: str, gpu_id: int):
    jobs_path = Path(render_jobs[0]["jobs_path"])
    jobs_path.write_text(json.dumps(render_jobs, indent=2, ensure_ascii=False), encoding="utf-8")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cmd = [
        str(render_python),
        str(render_batch_helper),
        "--blend-file",
        str(blend_file),
        "--jobs-json",
        str(jobs_path),
        "--samples",
        str(render_samples),
        "--compute-device",
        compute_device,
    ]
    subprocess.run(cmd, check=True, env=env)


def load_render_output(output_png: Path):
    with Image.open(output_png) as img:
        img = img.convert("RGBA")
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr[:, :, 0:3], arr[:, :, 3]


def ensure_rgb(data: np.ndarray | None, size: int, fill: float):
    if data is None:
        return np.full((size, size, 3), fill, dtype=np.float32)
    return np.clip(data.astype(np.float32), 0.0, 1.0)


def ensure_scalar(data: np.ndarray | None, size: int, fill: float):
    if data is None:
        return np.full((size, size), fill, dtype=np.float32)
    return np.clip(data.astype(np.float32), 0.0, 1.0)


def decode_normal_map(normal: np.ndarray):
    decoded = normal * 2.0 - 1.0
    lengths = np.sqrt(np.sum(decoded ** 2, axis=-1, keepdims=True))
    return decoded / (lengths + 1e-8)


def render_enhanced_pbr(albedo, normal, roughness, metallic, height=None, specular=None, opacity=None):
    normal_decoded = decode_normal_map(normal)
    light_dir = np.array([0.5, 0.5, 1.0], dtype=np.float32)
    light_dir /= np.linalg.norm(light_dir)
    view_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    halfway = light_dir + view_dir
    halfway /= np.linalg.norm(halfway)

    ndotl = np.maximum(0.0, np.sum(normal_decoded * light_dir, axis=-1))
    ndoth = np.maximum(0.0, np.sum(normal_decoded * halfway, axis=-1))

    shininess = np.maximum(1.0, (1.0 - roughness) * 128.0)
    spec_power = np.power(ndoth, shininess)

    metallic_3d = metallic[:, :, None]
    diffuse_color = albedo * (1.0 - metallic_3d)
    if specular is not None:
        specular_color = np.clip(specular, 0.0, 1.0)
    else:
        specular_color = albedo * metallic_3d + np.ones_like(albedo) * (1.0 - metallic_3d) * 0.04

    diffuse_term = diffuse_color * (0.3 + ndotl[:, :, None])
    specular_term = specular_color * spec_power[:, :, None] * (1.0 - roughness[:, :, None])
    final_color = diffuse_term + specular_term

    if height is not None:
        ao = 0.5 + 0.5 * np.clip(height, 0.0, 1.0)
        final_color *= ao[:, :, None]

    final_color = final_color / (final_color + 1.0)
    final_color = np.power(np.clip(final_color, 0.0, 1.0), 1.0 / 2.2)
    alpha = np.clip(opacity, 0.0, 1.0) if opacity is not None else np.ones((albedo.shape[0], albedo.shape[1]), dtype=np.float32)
    return final_color.astype(np.float32), alpha.astype(np.float32)


def build_material_record(row: dict, target_size: int):
    metadata = row.get("metadata") or {}
    maps = decode_material_row(row, target_size)
    maps["name"] = row["name"]
    maps["category_id"] = int(row.get("category", -1))
    maps["metadata"] = metadata
    return maps


def get_reference_material_names(reference_wsg_dir: Path):
    if not reference_wsg_dir.exists():
        return []
    return sorted(path.name for path in reference_wsg_dir.iterdir() if path.is_dir())


def get_reference_aug_count(reference_wsg_dir: Path, material_name: str):
    material_dir = reference_wsg_dir / material_name
    if not material_dir.exists():
        return None
    return sum(1 for _ in material_dir.glob(f"{material_name}_aug_*.wsg"))


def build_name_to_parquet_map(parquet_dir: Path):
    mapping = {}
    for parquet_path in sorted(parquet_dir.glob("*.parquet")):
        table = pq.read_table(parquet_path, columns=["name"])
        for name in table.column("name").to_pylist():
            mapping[name] = parquet_path
    return mapping


def load_selected_rows(parquet_dir: Path, material_names: list[str]):
    name_to_parquet = build_name_to_parquet_map(parquet_dir)
    grouped_names = {}
    for name in material_names:
        parquet_path = name_to_parquet.get(name)
        if parquet_path is None:
            continue
        grouped_names.setdefault(parquet_path, set()).add(name)

    rows = {}
    for parquet_path, wanted_names in grouped_names.items():
        table = pq.read_table(parquet_path)
        for row in table.to_pylist():
            name = row["name"]
            if name not in wanted_names:
                continue
            row["__parquet_name__"] = parquet_path.name
            rows[name] = row
    return rows, name_to_parquet


def combinatorial_total(has_specular: bool, has_opacity: bool):
    total = len(BRIGHTNESS_VALUES) * len(ABSOLUTE_VALUES) * len(ABSOLUTE_VALUES)
    total *= len(SPECULAR_SCALE_VALUES) if has_specular else 1
    total *= len(OPACITY_VALUES) if has_opacity else 1
    return total


def resolve_level_values(levels_json: str | None):
    if not levels_json:
        return {
            "brightness": BRIGHTNESS_VALUES,
            "absolute": ABSOLUTE_VALUES,
            "specular": SPECULAR_SCALE_VALUES,
            "opacity": OPACITY_VALUES,
        }
    parsed = json.loads(levels_json)
    if not isinstance(parsed, list) or not parsed:
        raise ValueError("--levels-json must decode to a non-empty JSON list")
    values = [float(x) for x in parsed]
    return {
        "brightness": values,
        "absolute": values,
        "specular": values,
        "opacity": values,
    }


def traversal_order(values: list[float]):
    center = (len(values) - 1) / 2.0
    ordered_indices = sorted(
        range(len(values)),
        key=lambda idx: (abs(idx - center), -idx),
    )
    return [values[idx] for idx in ordered_indices]


def closest_value(values: list[float], target: float):
    return min(values, key=lambda value: (abs(value - target), value))


def variant_sort_metadata(variant: dict, anchor: dict):
    changed = []
    param_order = ["albedo_source", "brightness", "roughness", "metallic", "specular_scale", "opacity"]
    for key in param_order:
        if variant.get(key) != anchor.get(key):
            changed.append(key)

    distance = 0.0
    for key in ("brightness", "roughness", "metallic", "specular_scale", "opacity"):
        value = variant.get(key)
        anchor_value = anchor.get(key)
        if value is None or anchor_value is None:
            continue
        distance += abs(float(value) - float(anchor_value))

    return {
        "changed_params": changed,
        "delta_count": len(changed),
        "priority_group": "single_parameter" if len(changed) == 1 else ("anchor" if len(changed) == 0 else "multi_parameter"),
        "distance_from_anchor": distance,
    }


def build_combinatorial_variants(has_basecolor: bool, has_diffuse: bool, has_specular: bool, has_opacity: bool, level_values: dict):
    default_albedo = "basecolor" if has_basecolor else "diffuse"
    albedo_options = [default_albedo]
    if has_basecolor and has_diffuse:
        alt = "diffuse" if default_albedo == "basecolor" else "basecolor"
        albedo_options.append(alt)

    brightness_values = traversal_order(level_values["brightness"])
    roughness_values = traversal_order(level_values["absolute"])
    metallic_values = traversal_order(level_values["absolute"])
    specular_values = traversal_order(level_values["specular"]) if has_specular else [1.0]
    opacity_values = traversal_order(level_values["opacity"]) if has_opacity else [None]

    anchor = {
        "albedo_source": default_albedo,
        "brightness": closest_value(level_values["brightness"], 1.0),
        "roughness": closest_value(level_values["absolute"], 0.5),
        "metallic": closest_value(level_values["absolute"], 0.5),
        "specular_scale": closest_value(level_values["specular"], 1.0) if has_specular else 1.0,
        "opacity": closest_value(level_values["opacity"], 1.0) if has_opacity else None,
    }

    variants = []
    for albedo_source, brightness, roughness, metallic, specular_scale, opacity in product(
        albedo_options,
        brightness_values,
        roughness_values,
        metallic_values,
        specular_values,
        opacity_values,
    ):
        variant = {
            "mode": "combinatorial",
            "albedo_source": albedo_source,
            "brightness": brightness,
            "roughness": roughness,
            "metallic": metallic,
            "specular_scale": specular_scale,
            "opacity": opacity,
        }
        variant.update(variant_sort_metadata(variant, anchor))
        variants.append(variant)

    variants.sort(
        key=lambda item: (
            item["delta_count"],
            item["distance_from_anchor"],
            tuple(item["changed_params"]),
            item["albedo_source"],
            item["brightness"],
            item["roughness"],
            item["metallic"],
            item["specular_scale"],
            -1.0 if item["opacity"] is None else item["opacity"],
        )
    )
    return variants


def single_factor_variants(has_basecolor: bool, has_diffuse: bool, has_specular: bool, has_opacity: bool, level_values: dict):
    variants = []
    if has_basecolor and has_diffuse:
        variants.append({
            "mode": "single_factor",
            "family": "albedo_source",
            "albedo_source": "diffuse",
            "brightness": 1.0,
            "roughness": None,
            "metallic": None,
            "specular_scale": 1.0,
            "opacity": None,
        })

    for value in level_values["brightness"]:
        variants.append({
            "mode": "single_factor",
            "family": "brightness",
            "albedo_source": "basecolor" if has_basecolor else "diffuse",
            "brightness": value,
            "roughness": None,
            "metallic": None,
            "specular_scale": 1.0,
            "opacity": None,
        })

    for family in ("roughness", "metallic"):
        for value in level_values["absolute"]:
            variants.append({
                "mode": "single_factor",
                "family": family,
                "albedo_source": "basecolor" if has_basecolor else "diffuse",
                "brightness": 1.0,
                "roughness": value if family == "roughness" else None,
                "metallic": value if family == "metallic" else None,
                "specular_scale": 1.0,
                "opacity": None,
            })

    if has_specular:
        for value in level_values["specular"]:
            variants.append({
                "mode": "single_factor",
                "family": "specular",
                "albedo_source": "basecolor" if has_basecolor else "diffuse",
                "brightness": 1.0,
                "roughness": None,
                "metallic": None,
                "specular_scale": value,
                "opacity": None,
            })

    if has_opacity:
        for value in level_values["opacity"]:
            variants.append({
                "mode": "single_factor",
                "family": "opacity",
                "albedo_source": "basecolor" if has_basecolor else "diffuse",
                "brightness": 1.0,
                "roughness": None,
                "metallic": None,
                "specular_scale": 1.0,
                "opacity": value,
            })
    return variants


def build_variant_list(material: dict, variation_mode: str, aug_limit: int | None, level_values: dict):
    has_basecolor = material["basecolor"] is not None
    has_diffuse = material["diffuse"] is not None
    has_specular = material["specular"] is not None
    has_opacity = material["opacity"] is not None
    default_albedo = "basecolor" if has_basecolor else "diffuse"

    variants = [{
        "mode": "original",
        "family": "original",
        "albedo_source": default_albedo,
        "brightness": 1.0,
        "roughness": None,
        "metallic": None,
        "specular_scale": 1.0,
        "opacity": None,
        "changed_params": [],
        "delta_count": 0,
        "priority_group": "original",
        "distance_from_anchor": 0.0,
    }]

    if variation_mode == "single_factor":
        for variant in single_factor_variants(has_basecolor, has_diffuse, has_specular, has_opacity, level_values):
            changed = []
            if variant["albedo_source"] != default_albedo:
                changed.append("albedo_source")
            if variant["brightness"] != 1.0:
                changed.append("brightness")
            if variant["roughness"] is not None:
                changed.append("roughness")
            if variant["metallic"] is not None:
                changed.append("metallic")
            if variant["specular_scale"] != 1.0:
                changed.append("specular_scale")
            if variant["opacity"] is not None:
                changed.append("opacity")
            variant["changed_params"] = changed
            variant["delta_count"] = len(changed)
            variant["priority_group"] = "single_parameter" if len(changed) == 1 else "single_factor"
            variant["distance_from_anchor"] = float(len(changed))
            variants.append(variant)
    else:
        combinatorial_variants = build_combinatorial_variants(
            has_basecolor,
            has_diffuse,
            has_specular,
            has_opacity,
            level_values,
        )
        count = len(combinatorial_variants) if aug_limit is None else min(len(combinatorial_variants), aug_limit)
        variants.extend(combinatorial_variants[:count])

    if aug_limit is not None and variation_mode == "single_factor":
        variants = variants[: aug_limit + 1]
    return variants


def apply_variant(material: dict, variant: dict, size: int):
    basecolor = ensure_rgb(material["basecolor"], size, 1.0)
    diffuse = ensure_rgb(material["diffuse"], size, 1.0) if material["diffuse"] is not None else None
    normal = ensure_rgb(material["normal"], size, 0.5)
    normal[:, :, 2] = np.maximum(normal[:, :, 2], 0.5)
    roughness = ensure_scalar(material["roughness"], size, 0.5)
    metallic = ensure_scalar(material["metallic"], size, 0.0)
    height_map = ensure_scalar(material["height"], size, 1.0) if material["height"] is not None else None
    specular = ensure_rgb(material["specular"], size, 0.04) if material["specular"] is not None else None
    opacity = ensure_scalar(material["opacity"], size, 1.0) if material["opacity"] is not None else None

    if variant["albedo_source"] == "diffuse" and diffuse is not None:
        albedo_source = np.clip(diffuse * variant["brightness"], 0.0, 1.0)
        basecolor_out = None
        diffuse_out = albedo_source
    else:
        albedo_source = np.clip(basecolor * variant["brightness"], 0.0, 1.0)
        basecolor_out = albedo_source
        diffuse_out = None if material["diffuse"] is not None else None

    roughness_out = np.full_like(roughness, variant["roughness"], dtype=np.float32) if variant["roughness"] is not None else roughness
    metallic_out = np.full_like(metallic, variant["metallic"], dtype=np.float32) if variant["metallic"] is not None else metallic
    specular_out = np.clip(specular * variant["specular_scale"], 0.0, 1.0) if specular is not None else None
    opacity_out = np.full_like(opacity, variant["opacity"], dtype=np.float32) if (opacity is not None and variant["opacity"] is not None) else opacity

    rgb, alpha = render_enhanced_pbr(
        albedo=albedo_source,
        normal=normal,
        roughness=roughness_out,
        metallic=metallic_out,
        height=height_map,
        specular=specular_out,
        opacity=opacity_out,
    )

    return {
        "rgb": rgb,
        "alpha": alpha,
        "basecolor": basecolor_out,
        "diffuse": diffuse_out,
        "normal": normal,
        "height": height_map,
        "roughness": roughness_out,
        "metallic": metallic_out,
        "specular": specular_out,
        "opacity": opacity_out,
    }


def build_wsg_array(rendered_maps: dict, size: int):
    arr = np.full((size, size, 256), -1.0, dtype=np.float16)
    # WSG RGB channels use image-space [0, 255]. MatSynth render maps are
    # produced in [0, 1], while material/PBR map channels below intentionally
    # remain in [0, 1].
    arr[:, :, 0:3] = np.clip(rendered_maps["rgb"] * 255.0, 0.0, 255.0).astype(np.float16)
    arr[:, :, 3] = rendered_maps["alpha"].astype(np.float16)

    if rendered_maps["normal"] is not None:
        arr[:, :, list(RGB_CHANNELS["normal"])] = rendered_maps["normal"].astype(np.float16)
    if rendered_maps["basecolor"] is not None:
        arr[:, :, list(RGB_CHANNELS["basecolor"])] = rendered_maps["basecolor"].astype(np.float16)
    if rendered_maps["diffuse"] is not None:
        arr[:, :, list(RGB_CHANNELS["diffuse"])] = rendered_maps["diffuse"].astype(np.float16)
    if rendered_maps["specular"] is not None:
        arr[:, :, list(RGB_CHANNELS["specular"])] = rendered_maps["specular"].astype(np.float16)

    for key in ("height", "roughness", "metallic", "opacity"):
        value = rendered_maps[key]
        if value is not None:
            arr[:, :, SCALAR_CHANNELS[key]] = value.astype(np.float16)

    return arr


def write_info_file(material_dir: Path, material: dict, parquet_name: str, variants: list[dict]):
    metadata = material["metadata"] or {}
    info_path = material_dir / f"{material['name']}_info.txt"
    lines = [
        "=" * 60,
        "PBR Material Information (Generated From Local Parquet)",
        "=" * 60,
        "",
        f"Material Name: {material['name']}",
        f"Source File: {parquet_name}",
        f"Category ID: {material['category_id']}",
        f"Category Name: {metadata.get('category', '')}",
        f"Description: {metadata.get('description', '')}",
        f"Physical Size: {metadata.get('physical_size', '')}",
        "",
        "WSG Channel Usage:",
        "  Channel   0-  2: RGB render preview",
        "  Channel   3: alpha render preview",
        "  Channel  36- 38: NORMAL",
        "  Channel 161-163: BASECOLOR or -1 when diffuse is active",
        "  Channel 164-166: DIFFUSE or -1 when basecolor is active",
        "  Channel 167: HEIGHT",
        "  Channel 168: ROUGHNESS",
        "  Channel 169: METALLIC",
        "  Channel 170-172: SPECULAR",
        "  Channel 173: OPACITY",
        "",
        f"Generated Variants: {len(variants)}",
    ]
    info_path.write_text("\n".join(lines), encoding="utf-8")


def process_material(task):
    (
        row,
        output_dir_str,
        size,
        variation_mode,
        aug_limit,
        overwrite,
        report_only,
        reference_aug_count,
        level_values,
        render_backend,
        render_python,
        blend_file,
        render_helper,
        render_batch_helper,
        temp_root,
        render_samples,
        render_batch_size,
        compute_device,
        gpu_id,
    ) = task
    output_dir = Path(output_dir_str)
    material = build_material_record(row, size)
    material_dir = output_dir / material["name"]
    crop_dir = material_dir / "crop_frame"

    if not report_only:
        ensure_dir(material_dir)
        ensure_dir(crop_dir)

    target_aug_limit = aug_limit
    if target_aug_limit is None and reference_aug_count is not None:
        target_aug_limit = reference_aug_count

    variants = build_variant_list(material, variation_mode, target_aug_limit, level_values)
    manifest = []
    written = 0
    pending_outputs = []
    render_jobs = []
    jobs_dir = Path(temp_root) / material["name"]
    if not report_only and render_backend == "blender":
        ensure_dir(jobs_dir)

    def flush_render_batch():
        nonlocal written, render_jobs, pending_outputs
        if not render_jobs:
            return
        run_blender_batch(
            material["name"],
            sorted(render_jobs, key=lambda job: (job["hdri_index"], job["hdri_rotation"])),
            Path(render_python),
            Path(blend_file),
            Path(render_batch_helper),
            render_samples,
            compute_device,
            gpu_id,
        )
        for rendered_maps, wsg_path, png_path, output_png in pending_outputs:
            rgb, alpha = load_render_output(output_png)
            rendered_maps["rgb"] = rgb
            rendered_maps["alpha"] = alpha
            arr = build_wsg_array(rendered_maps, size)
            save_wsg_array(arr, wsg_path)
            write_crop_frame(png_path, rendered_maps["rgb"], rendered_maps["alpha"])
            written += 1
        render_jobs = []
        pending_outputs = []

    for index, variant in enumerate(variants):
        if index == 0:
            file_stem = material["name"]
        else:
            file_stem = f"{material['name']}_aug_{index:02d}"

        wsg_path = material_dir / f"{file_stem}.wsg"
        png_path = crop_dir / f"{file_stem}.png"

        manifest.append({
            "file_stem": file_stem,
            "variant": variant,
        })

        if report_only:
            continue
        if not overwrite and wsg_path.exists() and png_path.exists():
            continue

        rendered_maps = apply_variant(material, variant, size)
        if render_backend == "blender":
            variant_dir, render_meta = stage_variant_maps(rendered_maps, material["name"], index, size, Path(temp_root))
            output_png = variant_dir / "render.png"
            render_jobs.append({
                "maps_dir": str(variant_dir),
                "output": str(output_png),
                "size": size,
                "hdri_index": index % 5,
                "hdri_rotation": (index * (math.pi / 7.0)) % (2.0 * math.pi),
                "jobs_path": str(jobs_dir / "render_jobs.json"),
                **render_meta,
            })
            pending_outputs.append((rendered_maps, wsg_path, png_path, output_png))
            if len(render_jobs) >= render_batch_size:
                flush_render_batch()
        else:
            arr = build_wsg_array(rendered_maps, size)
            save_wsg_array(arr, wsg_path)
            write_crop_frame(png_path, rendered_maps["rgb"], rendered_maps["alpha"])
            written += 1

    if not report_only and render_backend == "blender" and render_jobs:
        flush_render_batch()

    if not report_only:
        write_info_file(material_dir, material, row["__parquet_name__"], variants)
        (material_dir / f"{material['name']}_manifest.json").write_text(
            json.dumps(
                {
                    "material_name": material["name"],
                    "parquet_file": row["__parquet_name__"],
                    "variation_mode": variation_mode,
                    "variant_count": len(variants),
                    "variants": manifest,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    return {
        "material_name": material["name"],
        "variant_count": len(variants),
        "written": written,
    }


def main():
    args = parse_args()
    parquet_dir = args.parquet_dir.resolve()
    name_to_parquet = build_name_to_parquet_map(parquet_dir)
    level_values = resolve_level_values(args.levels_json)
    selected_names = set(args.materials or [])
    if args.use_reference_materials:
        selected_names.update(get_reference_material_names(args.reference_wsg_dir.resolve()))
    if not selected_names:
        selected_names = set(name_to_parquet.keys())

    material_names = sorted(name for name in selected_names if name in name_to_parquet)
    if args.limit_materials is not None:
        material_names = material_names[: args.limit_materials]

    if not material_names:
        print("No matching MatSynth materials found.")
        return 1

    source_rows, _ = load_selected_rows(parquet_dir, material_names)

    output_dir = args.output_dir.resolve()
    if not args.report_only:
        ensure_dir(output_dir)

    tasks = []
    for material_name in material_names:
        reference_aug_count = None
        if args.use_reference_counts:
            reference_aug_count = get_reference_aug_count(args.reference_wsg_dir.resolve(), material_name)
        tasks.append((
            source_rows[material_name],
            str(output_dir),
            args.size,
            args.variation_mode,
            args.max_augs_per_material,
            args.overwrite,
            args.report_only,
            reference_aug_count,
            level_values,
            args.render_backend,
            str(args.render_python.resolve()),
            str(args.blend_file.resolve()),
            str(args.render_helper.resolve()),
            str(args.render_batch_helper.resolve()),
            str(args.temp_root.resolve()),
            args.render_samples,
            args.render_batch_size,
            args.compute_device,
            (len(tasks) % max(1, args.gpu_count)),
        ))

    processed = 0
    total_written = 0
    total_variants = 0

    if args.workers <= 1:
        results = map(process_material, tasks)
    else:
        executor = ProcessPoolExecutor(max_workers=args.workers)
        results = executor.map(process_material, tasks)

    try:
        for result in results:
            processed += 1
            total_written += result["written"]
            total_variants += result["variant_count"]
            print(
                f"[{processed}/{len(tasks)}] {result['material_name']} "
                f"variants={result['variant_count']} written={result['written']}",
                flush=True,
            )
    finally:
        if args.workers > 1:
            executor.shutdown()

    print(f"Processed materials: {processed}")
    print(f"Planned/generated variants: {total_variants}")
    print(f"Written files: {total_written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
