#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import bpy


def parse_args():
    parser = argparse.ArgumentParser(description="Batch render MatSynth variants with one bpy session.")
    parser.add_argument("--blend-file", required=True)
    parser.add_argument("--jobs-json", required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--compute-device", choices=["OPTIX", "CUDA"], default="OPTIX")
    return parser.parse_args()


def set_cycles_devices(compute_device: str, samples: int):
    prefs = bpy.context.preferences
    cycles_prefs = prefs.addons["cycles"].preferences

    try:
        cycles_prefs.compute_device_type = compute_device
    except TypeError:
        cycles_prefs.compute_device_type = "CUDA"

    cycles_prefs.get_devices()
    for device in cycles_prefs.devices:
        device.use = device.type in {"OPTIX", "CUDA"}

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"
    scene.cycles.samples = max(1, int(samples))
    scene.cycles.use_denoising = False
    scene.render.use_persistent_data = True


def load_image_into_node(node_name: str, image_path: Path, colorspace: str):
    nodes = bpy.data.materials["BSDFPlane"].node_tree.nodes
    node = nodes.get(node_name)
    if node is None:
        raise RuntimeError(f"Missing node: {node_name}")

    image_name = f"{node_name}_Image"
    image = bpy.data.images.get(image_name)
    if image is None:
        image = bpy.data.images.load(str(image_path), check_existing=True)
        image.name = image_name
    else:
        image.filepath = str(image_path)
        image.reload()
    image.colorspace_settings.name = colorspace
    node.image = image


def remove_links(links, socket):
    for link in list(socket.links):
        links.remove(link)


def set_scalar_input(nodes, links, input_name: str, value: float):
    bsdf = nodes["Principled BSDF"]
    remove_links(links, bsdf.inputs[input_name])
    bsdf.inputs[input_name].default_value = float(value)


def ensure_rgb_to_bw(nodes, links, tex_node_name: str, bw_node_name: str, image_path: Path, output_input_name: str):
    bsdf = nodes["Principled BSDF"]
    tex_node = nodes.get(tex_node_name)
    if tex_node is None:
        tex_node = nodes.new("ShaderNodeTexImage")
        tex_node.name = tex_node_name

    bw_node = nodes.get(bw_node_name)
    if bw_node is None:
        bw_node = nodes.new("ShaderNodeRGBToBW")
        bw_node.name = bw_node_name

    image_name = f"{tex_node_name}_Image"
    image = bpy.data.images.get(image_name)
    if image is None:
        image = bpy.data.images.load(str(image_path), check_existing=True)
        image.name = image_name
    else:
        image.filepath = str(image_path)
        image.reload()
    image.colorspace_settings.name = "Non-Color"
    tex_node.image = image

    remove_links(links, bw_node.inputs["Color"])
    remove_links(links, bsdf.inputs[output_input_name])

    links.new(tex_node.outputs["Color"], bw_node.inputs["Color"])
    links.new(bw_node.outputs["Val"], bsdf.inputs[output_input_name])


def set_opacity(nodes, links, opacity_path: Path | None, opacity_constant: float | None):
    bsdf = nodes["Principled BSDF"]
    remove_links(links, bsdf.inputs["Alpha"])
    if opacity_constant is not None:
        bsdf.inputs["Alpha"].default_value = float(opacity_constant)
        return
    if opacity_path is None or not opacity_path.exists():
        bsdf.inputs["Alpha"].default_value = 1.0
        return
    ensure_rgb_to_bw(nodes, links, "Opacity", "Opacity To BW", opacity_path, "Alpha")


def set_specular(nodes, links, specular_path: Path | None, specular_constant: float | None):
    if specular_constant is not None:
        set_scalar_input(nodes, links, "Specular IOR Level", specular_constant)
        return
    if specular_path is None or not specular_path.exists():
        set_scalar_input(nodes, links, "Specular IOR Level", 0.04)
        return
    ensure_rgb_to_bw(nodes, links, "Specular", "Specular To BW", specular_path, "Specular IOR Level")


def configure_scene_for_job(size: int, hdri_index: int, hdri_rotation: float):
    scene = bpy.context.scene
    scene.render.resolution_x = size
    scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True

    world_nodes = bpy.data.worlds["World"].node_tree.nodes
    world_links = bpy.data.worlds["World"].node_tree.links
    bg_node = world_nodes["Background"]
    for link in list(bg_node.inputs["Color"].links):
        world_links.remove(link)
    env_node = world_nodes[f"Env {hdri_index % 5}"]
    world_links.new(env_node.outputs["Color"], bg_node.inputs["Color"])
    world_nodes["HDRIRotation"].outputs[0].default_value = hdri_rotation


def configure_material_graph():
    material = bpy.data.materials["BSDFPlane"]
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    output = nodes["Principled Output"]
    bsdf = nodes["Principled BSDF"]
    normal_map = nodes["Normal Map"]
    height = nodes["Height"]
    scale = nodes.get("Scale")

    # Use bump instead of true displacement to avoid rebuilding displaced geometry per variant.
    remove_links(links, output.inputs["Displacement"])

    bump = nodes.get("Height Bump")
    if bump is None:
        bump = nodes.new("ShaderNodeBump")
        bump.name = "Height Bump"

    if scale is not None:
        bump.inputs["Distance"].default_value = float(scale.outputs[0].default_value)
    bump.inputs["Strength"].default_value = 1.0

    remove_links(links, bump.inputs["Height"])
    remove_links(links, bump.inputs["Normal"])
    remove_links(links, bsdf.inputs["Normal"])
    links.new(height.outputs["Color"], bump.inputs["Height"])
    links.new(normal_map.outputs["Normal"], bump.inputs["Normal"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])


def set_scalar_or_map(nodes, links, node_name: str, input_name: str, image_path: Path, constant: float | None):
    if constant is not None:
        set_scalar_input(nodes, links, input_name, constant)
        return
    load_image_into_node(node_name, image_path, "Non-Color")


def render_job(job: dict):
    maps_dir = Path(job["maps_dir"])
    output_path = Path(job["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    configure_scene_for_job(job["size"], job["hdri_index"], job["hdri_rotation"])

    load_image_into_node("Base Color", maps_dir / "albedo.png", "sRGB")
    load_image_into_node("Normal", maps_dir / "normal.png", "Non-Color")
    load_image_into_node("Height", maps_dir / "height.png", "Non-Color")

    material = bpy.data.materials["BSDFPlane"]
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    set_scalar_or_map(nodes, links, "Roughness", "Roughness", maps_dir / "roughness.png", job.get("roughness_constant"))
    set_scalar_or_map(nodes, links, "Metallic", "Metallic", maps_dir / "metallic.png", job.get("metallic_constant"))
    set_specular(nodes, links, maps_dir / "specular.png", job.get("specular_constant"))
    set_opacity(nodes, links, maps_dir / "opacity.png", job.get("opacity_constant"))

    bpy.context.scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    args = parse_args()
    jobs = json.loads(Path(args.jobs_json).read_text(encoding="utf-8"))

    bpy.ops.wm.open_mainfile(filepath=args.blend_file)
    set_cycles_devices(args.compute_device, args.samples)
    configure_material_graph()

    for idx, job in enumerate(jobs, start=1):
        render_job(job)
        print(f"Rendered {idx}/{len(jobs)} -> {job['output']}", flush=True)
