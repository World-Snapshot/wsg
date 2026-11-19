#!/usr/bin/env python3
import argparse
import colorsys
import importlib.util
import json
import re
from pathlib import Path

import bpy


def natural_key(text):
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", text)]


def parse_args():
    parser = argparse.ArgumentParser(description="Render a material-region ID pass for a ShapeNet OBJ.")
    parser.add_argument("--object_path", type=str, required=True)
    parser.add_argument("--output_png", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=720)
    parser.add_argument("--fixed_camera", type=str, default="bird", choices=["front", "bird", "default"])
    parser.add_argument("--camera_distance", type=float, default=2.0)
    return parser.parse_args()

def get_scene_mesh_objects():
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def region_id_to_rgba(region_id):
    hue = ((region_id * 0.61803398875) % 1.0)
    saturation = 0.85
    value = 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return (r, g, b, 1.0)


def assign_region_materials(material_names):
    material_names = sorted(set(material_names), key=natural_key)
    material_name_to_region_id = {name: idx for idx, name in enumerate(material_names)}

    for material in bpy.data.materials:
        region_id = material_name_to_region_id[material.name]
        rgba = region_id_to_rgba(region_id)
        material.use_nodes = True
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        nodes.clear()
        output = nodes.new(type="ShaderNodeOutputMaterial")
        emission = nodes.new(type="ShaderNodeEmission")
        emission.inputs["Color"].default_value = rgba
        emission.inputs["Strength"].default_value = 1.0
        links.new(emission.outputs["Emission"], output.inputs["Surface"])
        if hasattr(material, "blend_method"):
            material.blend_method = "OPAQUE"
        if hasattr(material, "shadow_method"):
            material.shadow_method = "OPAQUE"

    return material_name_to_region_id


def ensure_unknown_material():
    unknown = bpy.data.materials.get("unknown")
    if unknown is None:
        unknown = bpy.data.materials.new(name="unknown")
    return unknown


def fill_missing_material_slots():
    for obj in get_scene_mesh_objects():
        if len(obj.data.materials) == 0:
            unknown = ensure_unknown_material()
            obj.data.materials.append(unknown)


def configure_render(scene, output_png, resolution):
    render = scene.render
    render.engine = "CYCLES"
    render.image_settings.file_format = "PNG"
    render.image_settings.color_mode = "RGBA"
    render.image_settings.color_depth = "8"
    render.resolution_x = resolution
    render.resolution_y = resolution
    render.resolution_percentage = 100
    render.film_transparent = True
    render.filepath = output_png

    scene.view_settings.view_transform = "Standard"
    scene.display_settings.display_device = "sRGB"
    scene.sequencer_colorspace_settings.name = "sRGB"


def configure_lightweight_mask_cycles(scene):
    scene.cycles.device = "CPU"
    scene.cycles.samples = 1
    scene.cycles.use_denoising = False
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.max_bounces = 0
    scene.cycles.diffuse_bounces = 0
    scene.cycles.glossy_bounces = 0
    scene.cycles.transparent_max_bounces = 0
    scene.cycles.transmission_bounces = 0
    scene.cycles.volume_bounces = 0
    scene.cycles.caustics_reflective = False
    scene.cycles.caustics_refractive = False
    scene.cycles.filter_width = 0.01
    if hasattr(scene.cycles, "pixel_filter_type"):
        scene.cycles.pixel_filter_type = "BOX"
    scene.render.use_compositing = False
    scene.render.use_sequencer = False


def load_base_script():
    script_path = Path(__file__).resolve().parent / "blender_script_shapenet.py"
    spec = importlib.util.spec_from_file_location("blender_script_shapenet_base", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.context = bpy.context
    module.scene = bpy.context.scene
    return module


def main():
    args = parse_args()
    output_png = Path(args.output_png)
    output_json = Path(args.output_json)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    base = load_base_script()
    base.reset_scene()
    scene = bpy.context.scene
    camera = scene.objects["Camera"]
    scene.camera = camera
    camera.data.lens = 35
    camera.data.sensor_width = 32
    camera.constraints.clear()
    cam_constraint = camera.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"
    empty = bpy.data.objects.new("Empty", None)
    scene.collection.objects.link(empty)
    cam_constraint.target = empty
    base.load_object(args.object_path)
    fill_missing_material_slots()
    base.normalize_scene()
    base.set_fixed_camera(args.fixed_camera, args.camera_distance)

    material_names = [material.name for material in bpy.data.materials]
    material_name_to_region_id = assign_region_materials(material_names)
    configure_render(scene, str(output_png), args.resolution)
    configure_lightweight_mask_cycles(scene)

    bpy.ops.render.render(write_still=True)

    metadata = {
        "object_path": args.object_path,
        "output_png": str(output_png),
        "resolution": args.resolution,
        "fixed_camera": args.fixed_camera,
        "camera_distance": args.camera_distance,
        "material_name_to_region_id": material_name_to_region_id,
        "region_id_to_material_name": {
            str(region_id): name for name, region_id in material_name_to_region_id.items()
        },
        "region_id_to_rgb": {
            str(region_id): [int(round(channel * 255)) for channel in region_id_to_rgba(region_id)[:3]]
            for name, region_id in material_name_to_region_id.items()
        },
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
