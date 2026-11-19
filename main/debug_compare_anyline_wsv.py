#!/usr/bin/env python3
"""Compare the current OpenCV Canny edge extraction with Anyline on WSV frames."""

import argparse
import io
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from huggingface_hub import hf_hub_download
from skimage import morphology


DEFAULT_WSVS = [
    "/research/cbim/vast/sf895/code/wsg/output/AVIS_val_results/FSiUB0FRHqY-1694875030796/wsv/merge.wsv",
    "/research/cbim/vast/sf895/code/wsg/output/AVIS_val_results/00007637/wsv/merge.wsv",
    "/research/cbim/vast/sf895/code/wsg/output/AVIS_val_results/00008920/wsv/merge.wsv",
    "/research/cbim/vast/sf895/code/wsg/output/AVIS_val_results/00000258/wsv/merge.wsv",
]
DEFAULT_OUTPUT_DIR = (
    "/research/cbim/vast/sf895/code/WSMs/WSM_v1/debug/anyline_wsv_comparison"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wsvs", nargs="*", default=DEFAULT_WSVS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--controlnet-aux-src",
        default="/tmp/comfyui_controlnet_aux/src",
        help="Directory containing the custom_controlnet_aux package.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resolution", type=int, default=1280)
    return parser.parse_args()


def import_anyline_dependencies(controlnet_aux_src):
    src = str(Path(controlnet_aux_src).resolve())
    if src not in sys.path:
        sys.path.insert(0, src)
    from custom_controlnet_aux.lineart_standard import LineartStandardDetector
    from custom_controlnet_aux.teed import TEDDetector
    from custom_controlnet_aux.teed.ted import TED

    return LineartStandardDetector, TEDDetector, TED


def load_middle_rgb(wsv_path):
    with zipfile.ZipFile(wsv_path, "r") as archive:
        members = sorted(
            member
            for member in archive.namelist()
            if member.startswith("index_") and member.endswith(".npy")
        )
        if not members:
            raise ValueError(f"No WSG frames found in {wsv_path}")
        member = members[len(members) // 2]
        frame = np.load(io.BytesIO(archive.read(member)), allow_pickle=False)
    rgb = np.nan_to_num(frame[..., 0:3], nan=0.0, posinf=255.0, neginf=0.0)
    return np.clip(rgb, 0, 255).astype(np.uint8), member


def canny_edge(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return cv2.Canny(gray, threshold1=50, threshold2=150)


def intensity_mask(image, lower_bound=0, upper_bound=1):
    mask = image[:, :, 0]
    mask = np.where((mask >= lower_bound) & (mask <= upper_bound), mask, 0)
    return np.repeat(mask[:, :, None], 3, axis=2)


def combine_layers(base_layer, top_layer):
    mask = top_layer.astype(bool)
    combined = 1 - (1 - top_layer) * (1 - base_layer)
    return base_layer * (~mask) + combined * mask


class AnylineDetector:
    def __init__(self, controlnet_aux_src, device, resolution):
        lineart_cls, ted_detector_cls, ted_cls = import_anyline_dependencies(
            controlnet_aux_src
        )
        checkpoint = hf_hub_download(
            repo_id="TheMistoAI/MistoLine",
            filename="MTEED.pth",
            subfolder="Anyline",
        )
        model = ted_cls()
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        model.eval()
        self.mteed = ted_detector_cls(model=model).to(device)
        self.lineart = lineart_cls()
        self.resolution = resolution

    def __call__(self, rgb):
        mteed = self.mteed(
            rgb,
            detect_resolution=self.resolution,
            output_type="np",
        ).astype(np.float32) / 255.0
        lineart = self.lineart(
            rgb,
            guassian_sigma=2,
            intensity_threshold=3,
            detect_resolution=self.resolution,
            output_type="np",
        ).astype(np.float32) / 255.0
        lineart = intensity_mask(lineart)
        cleaned = morphology.remove_small_objects(
            lineart.astype(bool), min_size=36, connectivity=1
        )
        lineart *= cleaned
        return np.clip(combine_layers(mteed, lineart) * 255.0, 0, 255).astype(
            np.uint8
        )


def fit_panel(image, size=(640, 360)):
    image = Image.fromarray(image).convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    panel = Image.new("RGB", size, "white")
    panel.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return panel


def labeled_panel(image, label, size=(640, 360)):
    header = 42
    panel = Image.new("RGB", (size[0], size[1] + header), "white")
    panel.paste(fit_panel(image, size), (0, header))
    ImageDraw.Draw(panel).text((12, 12), label, fill="black", font=ImageFont.load_default())
    return panel


def save_comparison(output_dir, sample_name, member, rgb, canny, anyline):
    output_dir.mkdir(parents=True, exist_ok=True)
    canny_rgb = np.repeat(canny[:, :, None], 3, axis=2)
    anyline_resized = cv2.resize(
        anyline, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_AREA
    )
    panels = [
        labeled_panel(rgb, f"RGB 0-2 | {member}"),
        labeled_panel(anyline_resized, "Anyline (new)"),
        labeled_panel(canny_rgb, "pipeline09 Canny (current)"),
    ]
    comparison = Image.new("RGB", (panels[0].width * 3, panels[0].height), "white")
    for index, panel in enumerate(panels):
        comparison.paste(panel, (index * panel.width, 0))
    comparison.save(output_dir / f"{sample_name}_comparison.png")
    Image.fromarray(rgb).save(output_dir / f"{sample_name}_rgb.png")
    Image.fromarray(canny_rgb).save(output_dir / f"{sample_name}_canny.png")
    Image.fromarray(anyline_resized).save(output_dir / f"{sample_name}_anyline.png")
    return comparison


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    detector = AnylineDetector(args.controlnet_aux_src, args.device, args.resolution)
    comparisons = []
    for index, wsv in enumerate(args.wsvs):
        wsv_path = Path(wsv)
        rgb, member = load_middle_rgb(wsv_path)
        sample_name = f"{index:02d}_{wsv_path.parents[1].name}"
        print(f"[{index + 1}/{len(args.wsvs)}] {wsv_path} {member} {rgb.shape}")
        comparisons.append(
            save_comparison(
                output_dir,
                sample_name,
                member,
                rgb,
                canny_edge(rgb),
                detector(rgb),
            )
        )
    overview = Image.new(
        "RGB",
        (comparisons[0].width, sum(image.height for image in comparisons)),
        "white",
    )
    y = 0
    for image in comparisons:
        overview.paste(image, (0, y))
        y += image.height
    overview.save(output_dir / "all_four_wsv_comparisons.png")
    print(f"Wrote comparisons to {output_dir}")


if __name__ == "__main__":
    main()
