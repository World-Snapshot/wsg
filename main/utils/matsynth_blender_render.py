#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import bpy


def parse_args():
    parser = argparse.ArgumentParser(description="Render a MatSynth material preview with Blender bpy.")
    parser.add_argument("--blend-file", required=True)
    parser.add_argument("--maps-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, default=300)
    parser.add_argument("--hdri-index", type=int, default=0)
    parser.add_argument("--hdri-rotation", type=float, default=0.0)
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
    scene.cycles.use_denoising = True


def ensure_image(node_name: str, image_path: Path, colorspace: str):
    nodes = bpy.data.materials["BSDFPlane"].node_tree.nodes
    node = nodes.get(node_name)
    if node is None:
        raise RuntimeError(f"Missing node: {node_name}")

    existing = bpy.data.images.get(f"{node_name}_Image")
    if existing is not None:
        bpy.data.images.remove(existing)

    image = bpy.data.images.load(str(image_path))
    image.colorspace_settings.name = colorspace
    image.name = f"{node_name}_Image"
    node.image = image


def ensure_specular_link(image_path: Path):
    material = bpy.data.materials["BSDFPlane"]
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    bsdf = nodes["Principled BSDF"]

    specular_node = nodes.get("Specular")
    if specular_node is None:
        specular_node = nodes.new("ShaderNodeTexImage")
        specular_node.name = "Specular"
        specular_node.location = (-800, -500)

    rgb_to_bw = nodes.get("Specular To BW")
    if rgb_to_bw is None:
        rgb_to_bw = nodes.new("ShaderNodeRGBToBW")
        rgb_to_bw.name = "Specular To BW"
        rgb_to_bw.location = (-550, -500)

    existing = bpy.data.images.get("Specular_Image")
    if existing is not None:
        bpy.data.images.remove(existing)

    image = bpy.data.images.load(str(image_path))
    image.colorspace_settings.name = "Non-Color"
    image.name = "Specular_Image"
    specular_node.image = image

    for link in list(bsdf.inputs["Specular IOR Level"].links):
        links.remove(link)
    for link in list(rgb_to_bw.inputs["Color"].links):
        links.remove(link)

    links.new(specular_node.outputs["Color"], rgb_to_bw.inputs["Color"])
    links.new(rgb_to_bw.outputs["Val"], bsdf.inputs["Specular IOR Level"])


def ensure_opacity_link(image_path: Path | None):
    material = bpy.data.materials["BSDFPlane"]
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    bsdf = nodes["Principled BSDF"]

    for link in list(bsdf.inputs["Alpha"].links):
        links.remove(link)

    if image_path is None:
        bsdf.inputs["Alpha"].default_value = 1.0
        return

    opacity_node = nodes.get("Opacity")
    if opacity_node is None:
        opacity_node = nodes.new("ShaderNodeTexImage")
        opacity_node.name = "Opacity"
        opacity_node.location = (-800, -650)

    rgb_to_bw = nodes.get("Opacity To BW")
    if rgb_to_bw is None:
        rgb_to_bw = nodes.new("ShaderNodeRGBToBW")
        rgb_to_bw.name = "Opacity To BW"
        rgb_to_bw.location = (-550, -650)

    existing = bpy.data.images.get("Opacity_Image")
    if existing is not None:
        bpy.data.images.remove(existing)

    image = bpy.data.images.load(str(image_path))
    image.colorspace_settings.name = "Non-Color"
    image.name = "Opacity_Image"
    opacity_node.image = image

    for link in list(rgb_to_bw.inputs["Color"].links):
        links.remove(link)

    links.new(opacity_node.outputs["Color"], rgb_to_bw.inputs["Color"])
    links.new(rgb_to_bw.outputs["Val"], bsdf.inputs["Alpha"])


def configure_scene(size: int, hdri_index: int, hdri_rotation: float):
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


if __name__ == "__main__":
    args = parse_args()
    bpy.ops.wm.open_mainfile(filepath=args.blend_file)

    set_cycles_devices(args.compute_device, args.samples)
    configure_scene(args.size, args.hdri_index, args.hdri_rotation)

    maps_dir = Path(args.maps_dir)
    ensure_image("Base Color", maps_dir / "albedo.png", "sRGB")
    ensure_image("Normal", maps_dir / "normal.png", "Non-Color")
    ensure_image("Roughness", maps_dir / "roughness.png", "Non-Color")
    ensure_image("Metallic", maps_dir / "metallic.png", "Non-Color")
    ensure_image("Height", maps_dir / "height.png", "Non-Color")
    ensure_specular_link(maps_dir / "specular.png")
    opacity_path = maps_dir / "opacity.png"
    ensure_opacity_link(opacity_path if opacity_path.exists() else None)

    bpy.context.scene.render.filepath = args.output
    bpy.ops.render.render(write_still=True)
