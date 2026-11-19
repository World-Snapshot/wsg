#!/usr/bin/env python3
"""
Inject three black-background pose visualizations into WSV frame channels.

The WSV file is a compressed NumPy archive whose ``index_*`` keys contain
individual WSG frames shaped ``(height, width, channels)``.  This pipeline
reconstructs the already-injected pose data in each frame, invokes the
existing visualization functions, and stores the RGB visualization images in
the new v1 pose visualization channels:

    DWPose:    channels 67-69
    MediaPipe: channels 99-101
    SMPLer-X:  channels 90-92

No other frame channels may change.  Missing pose data produces a valid black
visualization image (three zero-valued channels).
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import tempfile
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from PIL import Image
from tqdm import tqdm


DWPOSE_CHANNELS = slice(67, 70)
MEDIAPIPE_CHANNELS = slice(99, 102)
SMPLERX_CHANNELS = slice(90, 93)
SMPLERX_RESERVED_CHANNEL = 93
TARGET_SLICES = (DWPOSE_CHANNELS, MEDIAPIPE_CHANNELS, SMPLERX_CHANNELS)
UNCHANGED_SLICES = (slice(0, 67), slice(70, 90), slice(94, 99), slice(102, None))
UNCHANGED_SLICES_WITH_LEGACY_CLEAR = (
    slice(0, 67), slice(70, 90), slice(94, 99), slice(102, 105), slice(108, None)
)

DEFAULT_DATASET_DIRS = (
    Path("../output/AVIS_train_results"),
    Path("../output/AVIS_val_results"),
    Path("../output/AVIS_test_results"),
)

SMPLERX_PERSON_CHANNEL = 72
SMPLERX_JOINT_ID_CHANNEL = 73
SMPLERX_JOINT_CHANNELS = (74, 75, 76)
SMPLERX_MODEL_ROOT = Path("/research/cbim/vast/sf895/code/wsg/common/utils/human_model_files")
_SMPLERX_MODEL = None
_SMPLERX_DEVICE = None
_SMPLERX_RENDERER = None
_SMPLERX_RENDERER_SIZE = None
_VERBOSE_RENDER = False


def _parse_target_poses(value: str | Iterable[str]) -> Set[str]:
    if isinstance(value, str):
        raw = [part.strip().lower() for part in value.split(",")]
    else:
        raw = [str(part).strip().lower() for part in value]
    poses = {part for part in raw if part}
    if not poses or "all" in poses:
        return {"dwpose", "mediapipe", "smplerx"}
    invalid = poses - {"dwpose", "mediapipe", "smplerx"}
    if invalid:
        raise ValueError(f"Unsupported pose target(s): {sorted(invalid)}")
    return poses


def _model_device():
    if _SMPLERX_DEVICE is not None:
        return _SMPLERX_DEVICE
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def configure_gpu(gpu_id: int | None) -> None:
    """Bind one standalone worker to one CUDA/EGL GPU before model loading."""
    global _SMPLERX_DEVICE
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    if gpu_id is None:
        _SMPLERX_DEVICE = "cpu"
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["EGL_DEVICE_ID"] = str(gpu_id)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(f"GPU {gpu_id} requested, but CUDA is not available")
    _SMPLERX_DEVICE = "cuda:0"
    print(f"Worker GPU binding: CUDA_VISIBLE_DEVICES={gpu_id}, EGL_DEVICE_ID={gpu_id}")


def _as_rgb_uint8(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Normalize reader visualization output to an ``H x W x 3`` RGB image."""
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected a three-dimensional visualization, got shape {image.shape}")
    if image.shape == (3, height, width):
        image = np.transpose(image, (1, 2, 0))
    if image.shape != (height, width, 3):
        raise ValueError(
            f"Visualization shape {image.shape} does not match frame size {(height, width, 3)}"
        )
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating) and image.size and np.nanmax(image) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def render_dwpose(
    frame: np.ndarray,
    line_scale: float = 3.0,
    point_scale: float = 2.0,
) -> Tuple[np.ndarray, bool]:
    from pipeline01_read_dwpose_from_wsg import (
        draw_pose,
        extract_dwpose_from_wsg,
        process_pose_data,
    )

    height, width = frame.shape[:2]
    pose = extract_dwpose_from_wsg(frame)
    if pose is None or pose.get("num_persons", 0) == 0:
        return np.zeros((height, width, 3), dtype=np.uint8), False
    prepared = process_pose_data(pose, height, width)
    image = draw_pose(
        pose=prepared,
        H=height,
        W=width,
        include_body=True,
        include_hand=True,
        include_face=True,
        line_scale=line_scale,
        point_scale=point_scale,
    )
    return _as_rgb_uint8(image, height, width), True


def render_mediapipe(frame: np.ndarray) -> Tuple[np.ndarray, bool]:
    from pipeline03_read_mediapipe_from_wsg import (
        draw_pose_on_black_background,
        extract_mediapipe_from_wsg,
    )

    height, width = frame.shape[:2]
    pose = extract_mediapipe_from_wsg(frame)
    if pose is None:
        return np.zeros((height, width, 3), dtype=np.uint8), False
    image_bgr = draw_pose_on_black_background(pose, width, height)
    image_rgb = image_bgr[:, :, ::-1]
    return _as_rgb_uint8(image_rgb, height, width), True


def _recover_vector_for_id(
    frame: np.ndarray,
    person_mask: np.ndarray,
    joint_id: int,
    count: int = 1,
) -> np.ndarray:
    output = np.zeros((count, 3), dtype=np.float32)
    valid = person_mask & (frame[:, :, SMPLERX_JOINT_ID_CHANNEL] == joint_id)
    if np.any(valid):
        values = frame[:, :, list(SMPLERX_JOINT_CHANNELS)][valid][0].astype(np.float32)
        output[0] = values
    return output


def _recover_betas_or_expression(
    frame: np.ndarray,
    person_mask: np.ndarray,
    type_id: int,
) -> np.ndarray:
    output = np.zeros((1, 10), dtype=np.float32)
    type_channel = frame[:, :, 77]
    valid_type = person_mask & (type_channel == type_id)
    if not np.any(valid_type):
        return output
    for offset, channel in enumerate(range(78, 88)):
        values = frame[:, :, channel][valid_type].astype(np.float32)
        values = values[np.isfinite(values) & (values != -1)]
        if values.size:
            output[0, offset] = float(values[0])
    return output


def extract_smplerx_model_parameters(frame: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
    """Decode trusted SMPL-X pose vectors and use a neutral body for visualization."""
    person_channel = frame[:, :, SMPLERX_PERSON_CHANNEL]
    valid = person_channel != -1
    if not np.any(valid):
        return {}

    person_ids = sorted(int(value) for value in np.unique(person_channel[valid]))
    results: Dict[str, Dict[str, np.ndarray]] = {}
    for person_id in person_ids:
        person_mask = person_channel == person_id
        person = {
            "global_orient": _recover_vector_for_id(frame, person_mask, 0),
            "body_pose": np.vstack(
                [_recover_vector_for_id(frame, person_mask, joint_id) for joint_id in range(1, 22)]
            ).astype(np.float32),
            "left_hand_pose": np.vstack(
                [_recover_vector_for_id(frame, person_mask, joint_id) for joint_id in range(22, 37)]
            ).astype(np.float32),
            "right_hand_pose": np.vstack(
                [_recover_vector_for_id(frame, person_mask, joint_id) for joint_id in range(37, 52)]
            ).astype(np.float32),
            "jaw_pose": _recover_vector_for_id(frame, person_mask, 52),
            "leye_pose": _recover_vector_for_id(frame, person_mask, 53),
            "reye_pose": _recover_vector_for_id(frame, person_mask, 54),
            "transl": _recover_vector_for_id(frame, person_mask, 55),
        }
        person["betas"] = _recover_betas_or_expression(frame, person_mask, 0)
        person["expression"] = _recover_betas_or_expression(frame, person_mask, 1)
        results[f"person_{person_id}"] = person
    return results


def _get_smplerx_model():
    global _SMPLERX_MODEL
    if _SMPLERX_MODEL is None:
        import smplx
        import torch

        device = torch.device(_model_device())

        _SMPLERX_MODEL = smplx.create(
            str(SMPLERX_MODEL_ROOT),
            model_type="smplx",
            gender="neutral",
            ext="npz",
            num_betas=10,
            num_expression_coeffs=10,
            use_pca=False,
            use_face_contour=True,
            create_global_orient=False,
            create_body_pose=False,
            create_left_hand_pose=False,
            create_right_hand_pose=False,
            create_jaw_pose=False,
            create_leye_pose=False,
            create_reye_pose=False,
            create_betas=False,
            create_expression=False,
            create_transl=False,
        ).to(device).eval()
    return _SMPLERX_MODEL


def _create_full_smplerx_mesh(person: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    import torch

    model = _get_smplerx_model()
    device = torch.device(_model_device())
    with torch.no_grad():
        output = model(
            betas=torch.from_numpy(person["betas"]).to(device),
            body_pose=torch.from_numpy(person["body_pose"].reshape(1, -1)).to(device),
            global_orient=torch.from_numpy(person["global_orient"]).to(device),
            right_hand_pose=torch.from_numpy(person["right_hand_pose"].reshape(1, -1)).to(device),
            left_hand_pose=torch.from_numpy(person["left_hand_pose"].reshape(1, -1)).to(device),
            jaw_pose=torch.from_numpy(person["jaw_pose"]).to(device),
            leye_pose=torch.from_numpy(person["leye_pose"]).to(device),
            reye_pose=torch.from_numpy(person["reye_pose"]).to(device),
            expression=torch.from_numpy(person["expression"]).to(device),
            transl=torch.from_numpy(person["transl"]).to(device),
            return_verts=True,
        )
    return output.vertices[0].cpu().numpy(), np.asarray(model.faces)


def _estimate_camera_from_pose_region(
    frame: np.ndarray,
    person_id: int,
    vertices: np.ndarray,
    width: int,
    height: int,
) -> Tuple[float, float, float, float]:
    """Align the neutral body to the visible person region without undoing crop."""
    person_mask = frame[:, :, SMPLERX_PERSON_CHANNEL] == person_id
    ys, xs = np.where(person_mask)
    if xs.size < 4:
        return 1500.0, 1500.0, width / 2.0, height / 2.0

    target_x1, target_x2 = float(xs.min()), float(xs.max())
    target_y1, target_y2 = float(ys.min()), float(ys.max())
    target_width = max(target_x2 - target_x1, 1.0)
    target_height = max(target_y2 - target_y1, 1.0)

    z = vertices[:, 2].astype(np.float64)
    valid = np.isfinite(vertices).all(axis=1) & (np.abs(z) > 1e-5)
    if np.count_nonzero(valid) < 100:
        return 1500.0, 1500.0, (target_x1 + target_x2) / 2.0, (target_y1 + target_y2) / 2.0

    projected_x = vertices[valid, 0].astype(np.float64) / z[valid]
    projected_y = vertices[valid, 1].astype(np.float64) / z[valid]
    mesh_x1, mesh_x2 = np.quantile(projected_x, [0.01, 0.99])
    mesh_y1, mesh_y2 = np.quantile(projected_y, [0.01, 0.99])
    mesh_width = max(float(mesh_x2 - mesh_x1), 1e-6)
    mesh_height = max(float(mesh_y2 - mesh_y1), 1e-6)

    touches_left = target_x1 <= 1.0
    touches_right = target_x2 >= width - 2.0
    touches_top = target_y1 <= 1.0
    touches_bottom = target_y2 >= height - 2.0

    # A clipped region does not describe the full-body extent along that axis.
    # Estimate scale from an un-clipped axis and always use one focal length so
    # the neutral body cannot be stretched horizontally or vertically.
    x_scale = target_width / mesh_width
    y_scale = target_height / mesh_height
    if (touches_top or touches_bottom) and not (touches_left or touches_right):
        focal = x_scale
    elif (touches_left or touches_right) and not (touches_top or touches_bottom):
        focal = y_scale
    else:
        focal = min(x_scale, y_scale)
    focal *= 0.92

    mesh_center_x = (mesh_x1 + mesh_x2) / 2.0
    mesh_center_y = (mesh_y1 + mesh_y2) / 2.0
    if touches_left and not touches_right:
        cx = target_x2 - focal * mesh_x2
    elif touches_right and not touches_left:
        cx = target_x1 - focal * mesh_x1
    else:
        cx = (target_x1 + target_x2) / 2.0 - focal * mesh_center_x

    if touches_top and not touches_bottom:
        cy = target_y2 - focal * mesh_y2
    elif touches_bottom and not touches_top:
        cy = target_y1 - focal * mesh_y1
    else:
        cy = (target_y1 + target_y2) / 2.0 - focal * mesh_center_y

    if not np.all(np.isfinite([focal, cx, cy])) or focal <= 0:
        return 1500.0, 1500.0, width / 2.0, height / 2.0
    return float(focal), float(focal), float(cx), float(cy)



def _expanded_person_bbox_mask(
    frame: np.ndarray,
    person_id: int,
    margin_fraction: float,
) -> np.ndarray:
    height, width = frame.shape[:2]
    person_mask = frame[:, :, SMPLERX_PERSON_CHANNEL] == person_id
    ys, xs = np.where(person_mask)
    mask = np.zeros((height, width), dtype=bool)
    if xs.size < 4:
        return mask
    x1, x2 = float(xs.min()), float(xs.max())
    y1, y2 = float(ys.min()), float(ys.max())
    bw = max(x2 - x1 + 1.0, 1.0)
    bh = max(y2 - y1 + 1.0, 1.0)
    mx = bw * float(margin_fraction)
    my = bh * float(margin_fraction)
    ix1 = max(0, int(np.floor(x1 - mx)))
    iy1 = max(0, int(np.floor(y1 - my)))
    ix2 = min(width, int(np.ceil(x2 + mx)) + 1)
    iy2 = min(height, int(np.ceil(y2 + my)) + 1)
    if ix1 < ix2 and iy1 < iy2:
        mask[iy1:iy2, ix1:ix2] = True
    return mask


def _render_full_smplerx_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    camera: Tuple[float, float, float, float],
    height: int,
    width: int,
) -> np.ndarray:
    import pyrender
    import trimesh

    fx, fy, cx, cy = camera
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0]))
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.1,
        roughnessFactor=0.4,
        alphaMode="OPAQUE",
        emissiveFactor=(0.2, 0.2, 0.2),
        baseColorFactor=(0.7, 0.7, 0.7, 1.0),
    )
    scene = pyrender.Scene(bg_color=(0, 0, 0, 0), ambient_light=(0.3, 0.3, 0.3))
    scene.add(pyrender.Mesh.from_trimesh(mesh, material=material), "mesh")
    camera_model = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, zfar=1e12)
    camera_pose = np.eye(4)
    scene.add(camera_model, pose=camera_pose)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.0), pose=camera_pose)
    renderer = _get_smplerx_renderer(width, height)
    rgba, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    return rgba[:, :, :3]


def _delete_smplerx_renderer() -> None:
    global _SMPLERX_RENDERER, _SMPLERX_RENDERER_SIZE
    if _SMPLERX_RENDERER is not None:
        _SMPLERX_RENDERER.delete()
        _SMPLERX_RENDERER = None
        _SMPLERX_RENDERER_SIZE = None


def _get_smplerx_renderer(width: int, height: int):
    global _SMPLERX_RENDERER, _SMPLERX_RENDERER_SIZE
    import pyrender

    size = (width, height)
    if _SMPLERX_RENDERER_SIZE != size:
        _delete_smplerx_renderer()
        _SMPLERX_RENDERER = pyrender.OffscreenRenderer(
            viewport_width=width, viewport_height=height, point_size=1.0
        )
        _SMPLERX_RENDERER_SIZE = size
    return _SMPLERX_RENDERER


atexit.register(_delete_smplerx_renderer)


def render_smplerx(
    frame: np.ndarray,
    visible_clamp: str = "none",
    visible_margin_fraction: float = 0.2,
) -> Tuple[np.ndarray, bool]:
    if visible_clamp not in {"none", "bbox"}:
        raise ValueError(f"Unsupported SMPLer-X visible clamp mode: {visible_clamp}")
    height, width = frame.shape[:2]
    people = extract_smplerx_model_parameters(frame)
    if not people:
        return np.zeros((height, width, 3), dtype=np.uint8), False
    rendered = np.zeros((height, width, 3), dtype=np.uint8)
    for person_key, person in people.items():
        person_id = int(person_key.split("_")[-1])
        vertices, faces = _create_full_smplerx_mesh(person)
        camera = _estimate_camera_from_pose_region(frame, person_id, vertices, width, height)
        if _VERBOSE_RENDER:
            print(
                f"  {person_key}: regenerated SMPL-X mesh "
                f"({vertices.shape[0]} vertices, {faces.shape[0]} faces), "
                f"camera fx={camera[0]:.1f}, fy={camera[1]:.1f}, "
                f"cx={camera[2]:.1f}, cy={camera[3]:.1f}"
            )
        image = _render_full_smplerx_mesh(vertices, faces, camera, height, width)
        mask = np.any(image != 0, axis=2)
        if visible_clamp == "bbox":
            visible_mask = _expanded_person_bbox_mask(frame, person_id, visible_margin_fraction)
            mask &= visible_mask
        rendered[mask] = image[mask]
    return _as_rgb_uint8(rendered, height, width), True


def assert_only_pose_visualization_channels_changed(
    before: np.ndarray, after: np.ndarray, clear_legacy_smplerx_vis: bool = False
) -> None:
    """Reject updates outside pose visualization channels and reserved channel 93."""
    if before.shape != after.shape:
        raise ValueError(f"Frame shape changed from {before.shape} to {after.shape}")
    slices = UNCHANGED_SLICES_WITH_LEGACY_CLEAR if clear_legacy_smplerx_vis else UNCHANGED_SLICES
    for channels in slices:
        if not np.array_equal(before[:, :, channels], after[:, :, channels], equal_nan=True):
            raise RuntimeError(f"Non-target WSV channels changed in slice {channels}")


def inject_frame(
    frame: np.ndarray,
    dwpose_line_scale: float = 3.0,
    dwpose_point_scale: float = 2.0,
    verify_integrity: bool = True,
    clear_legacy_smplerx_vis: bool = False,
    target_poses: Set[str] | None = None,
    smplerx_visible_clamp: str = "none",
    smplerx_visible_margin_fraction: float = 0.2,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, bool]]:
    if frame.ndim != 3 or frame.shape[2] < 108:
        raise ValueError(f"Expected WSV frame with at least 108 channels, got shape {frame.shape}")

    original = frame if verify_integrity else None
    updated = frame.copy() if verify_integrity else frame
    target_poses = target_poses or {"dwpose", "mediapipe", "smplerx"}
    height, width = frame.shape[:2]

    if "dwpose" in target_poses:
        dwpose_rgb, has_dwpose = render_dwpose(frame, dwpose_line_scale, dwpose_point_scale)
    else:
        dwpose_rgb = _as_rgb_uint8(np.where(frame[:, :, DWPOSE_CHANNELS] == -1, 0, frame[:, :, DWPOSE_CHANNELS]), height, width)
        has_dwpose = False
    if "mediapipe" in target_poses:
        mediapipe_rgb, has_mediapipe = render_mediapipe(frame)
    else:
        mediapipe_rgb = _as_rgb_uint8(np.where(frame[:, :, MEDIAPIPE_CHANNELS] == -1, 0, frame[:, :, MEDIAPIPE_CHANNELS]), height, width)
        has_mediapipe = False
    if "smplerx" in target_poses:
        smplerx_rgb, has_smplerx = render_smplerx(
            frame,
            visible_clamp=smplerx_visible_clamp,
            visible_margin_fraction=smplerx_visible_margin_fraction,
        )
    else:
        smplerx_rgb = _as_rgb_uint8(np.where(frame[:, :, SMPLERX_CHANNELS] == -1, 0, frame[:, :, SMPLERX_CHANNELS]), height, width)
        has_smplerx = False

    if "dwpose" in target_poses:
        updated[:, :, DWPOSE_CHANNELS] = dwpose_rgb.astype(updated.dtype, copy=False)
    if "mediapipe" in target_poses:
        updated[:, :, MEDIAPIPE_CHANNELS] = mediapipe_rgb.astype(updated.dtype, copy=False)
    if "smplerx" in target_poses:
        if has_smplerx:
            updated[:, :, SMPLERX_CHANNELS] = smplerx_rgb.astype(updated.dtype, copy=False)
        else:
            updated[:, :, SMPLERX_CHANNELS] = -1
        updated[:, :, SMPLERX_RESERVED_CHANNEL] = -1
    if clear_legacy_smplerx_vis:
        updated[:, :, 105:108] = -1

    if verify_integrity:
        assert_only_pose_visualization_channels_changed(
            original, updated, clear_legacy_smplerx_vis
        )
    previews = {
        "dwpose": dwpose_rgb,
        "mediapipe": mediapipe_rgb,
        "smplerx_rerendered": smplerx_rgb,
        "smplerx_stored_90_92": np.where(
            updated[:, :, SMPLERX_CHANNELS] == -1, 0, updated[:, :, SMPLERX_CHANNELS]
        ).astype(np.uint8),
    }
    present = {"dwpose": has_dwpose, "mediapipe": has_mediapipe, "smplerx": has_smplerx}
    return updated, previews, present


def discover_wsv_files(inputs: Iterable[Path]) -> List[Path]:
    files: List[Path] = []
    for input_path in inputs:
        if input_path.is_file():
            if input_path.suffix.lower() != ".wsv":
                raise ValueError(f"Input file is not a .wsv file: {input_path}")
            files.append(input_path)
        elif input_path.is_dir():
            # The AVIS result directories are symlinks to the large data volume.
            resolved = input_path.resolve()
            if input_path.name.startswith("AVIS_") and input_path.name.endswith("_results"):
                files.extend(sorted(resolved.glob("*/wsv/merge.wsv")))
            else:
                files.extend(sorted(resolved.rglob("*.wsv")))
        else:
            print(f"Skip missing input: {input_path}")
    return sorted(set(path.resolve() for path in files))


def _write_wsv_array(archive: ZipFile, key: str, array: np.ndarray) -> None:
    """Write one NumPy array member without retaining other WSV frames in RAM."""
    with archive.open(f"{key}.npy", mode="w", force_zip64=True) as member:
        np.lib.format.write_array(member, np.asanyarray(array), allow_pickle=True)


@contextmanager
def _atomic_wsv_writer(output_path: Path, compresslevel: int):
    """Yield a streaming NPZ writer and replace the target only after success."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".wsv",
        dir=str(output_path.parent),
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        with ZipFile(
            temp_path,
            mode="w",
            compression=ZIP_DEFLATED,
            compresslevel=compresslevel,
            allowZip64=True,
        ) as archive:
            yield archive
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _save_previews(preview_dir: Path, wsv_path: Path, frame_key: str, previews: Dict[str, np.ndarray]) -> None:
    relative_name = f"{wsv_path.parent.parent.name}_{wsv_path.stem}_{frame_key}"
    target_dir = preview_dir / relative_name
    target_dir.mkdir(parents=True, exist_ok=True)
    for pose_name, image in previews.items():
        Image.fromarray(image, mode="RGB").save(target_dir / f"{pose_name}.png")


def _completion_marker_path(state_dir: Path, input_path: Path) -> Path:
    digest = hashlib.sha1(str(input_path.resolve()).encode("utf-8")).hexdigest()[:16]
    return state_dir / f"{digest}_{input_path.parent.parent.name}.done.json"


def _count_wsv_frame_members(input_path: Path) -> int:
    """Count frame members from ZIP metadata without inflating WSV arrays."""
    with ZipFile(input_path) as archive:
        return sum(
            1
            for member in archive.namelist()
            if Path(member).stem.startswith("index_") and member.endswith(".npy")
        )


def _save_completion_marker(
    state_dir: Path,
    input_path: Path,
    output_path: Path,
    counts: Dict[str, int],
    elapsed_seconds: float,
    gpu_id: int | None,
) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    marker = _completion_marker_path(state_dir, input_path)
    fd, temp_name = tempfile.mkstemp(prefix=f".{marker.name}.", suffix=".tmp", dir=str(state_dir))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(
                {
                    "input": str(input_path.resolve()),
                    "output": str(output_path.resolve()),
                    "counts": counts,
                    "elapsed_seconds": elapsed_seconds,
                    "gpu_id": gpu_id,
                    "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                stream,
                indent=2,
            )
        os.replace(temp_name, marker)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def process_wsv(
    input_path: Path,
    output_path: Path | None,
    preview_dir: Path | None = None,
    max_frames: int | None = None,
    dwpose_line_scale: float = 3.0,
    dwpose_point_scale: float = 2.0,
    progress_bar=None,
    compresslevel: int = 1,
    verify_integrity: bool = True,
    clear_legacy_smplerx_vis: bool = False,
    target_poses: Set[str] | None = None,
    smplerx_visible_clamp: str = "none",
    smplerx_visible_margin_fraction: float = 0.2,
) -> Dict[str, int]:
    print(f"Processing WSV: {input_path}")
    counts = {"frames": 0, "dwpose": 0, "mediapipe": 0, "smplerx": 0}

    output_context = (
        _atomic_wsv_writer(output_path, compresslevel) if output_path is not None else nullcontext(None)
    )
    with np.load(input_path, allow_pickle=True) as archive, output_context as output_archive:
        frame_keys = sorted(
            key
            for key in archive.files
            if key.startswith("index_")
        )
        if max_frames is not None:
            frame_keys = frame_keys[:max_frames]
        selected = set(frame_keys)

        for key in archive.files:
            if key not in selected:
                if output_archive is not None:
                    _write_wsv_array(output_archive, key, archive[key])
                if progress_bar is not None:
                    progress_bar.update(1)
                continue
            frame = archive[key]
            if frame.ndim != 3 or frame.shape[2] < 108:
                if output_archive is not None:
                    _write_wsv_array(output_archive, key, frame)
                if progress_bar is not None:
                    progress_bar.update(1)
                continue
            updated, previews, present = inject_frame(
                frame,
                dwpose_line_scale,
                dwpose_point_scale,
                verify_integrity,
                clear_legacy_smplerx_vis,
                target_poses,
                smplerx_visible_clamp,
                smplerx_visible_margin_fraction,
            )
            if output_archive is not None:
                _write_wsv_array(output_archive, key, updated)
            counts["frames"] += 1
            for pose_name, available in present.items():
                counts[pose_name] += int(available)
            if preview_dir is not None:
                _save_previews(preview_dir, input_path, key, previews)
            if progress_bar is not None:
                progress_bar.update(1)

    if counts["frames"] == 0:
        raise RuntimeError(f"No frame arrays found in WSV file: {input_path}")

    print(
        "  Rendered "
        f"{counts['frames']} frame(s); pose found "
        f"DWPose={counts['dwpose']}, MediaPipe={counts['mediapipe']}, SMPLer-X={counts['smplerx']}"
    )
    if output_path is not None:
        print(f"  Saved injected WSV: {output_path}")
    else:
        print("  Preview only: WSV data was not written.")
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inject DWPose, MediaPipe, and SMPLer-X black-background visualizations into WSV."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="WSV files or directories. With no input, scans AVIS_train/val/test_results.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output WSV path for a single input file. Without it, files are updated in place.",
    )
    parser.add_argument(
        "--preview-dir",
        type=Path,
        help="Optional directory for black-background PNG previews generated during injection.",
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Generate PNG previews without writing any WSV data. Requires --preview-dir.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Process only the first N frame keys in each WSV, intended for smoke testing.",
    )
    parser.add_argument(
        "--dwpose-line-scale",
        type=float,
        default=3.0,
        help="DWPose body and hand line-width multiplier (default: 3).",
    )
    parser.add_argument(
        "--dwpose-point-scale",
        type=float,
        default=2.0,
        help="DWPose body, hand, and face point-radius multiplier (default: 2).",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        help="Bind SMPL-X reconstruction and EGL rendering to this physical GPU index.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split the discovered WSV list into this many deterministic worker shards.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Process only this zero-based shard; use one process per GPU.",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path("../output/pipeline30_pose_vis_state"),
        help="Directory for per-WSV completion markers used by --skip-completed.",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip files that already have a completion marker in --state-dir.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Display a tqdm per-frame progress bar suitable for redirected worker logs.",
    )
    parser.add_argument(
        "--verbose-render",
        action="store_true",
        help="Print full SMPL-X camera information for every rendered person.",
    )
    parser.add_argument(
        "--compresslevel",
        type=int,
        choices=range(1, 10),
        default=1,
        metavar="{1..9}",
        help="DEFLATE compression level for rewritten WSV files (default: 1 for throughput).",
    )
    parser.add_argument(
        "--clear-legacy-smplerx-vis",
        action="store_true",
        help="Set deprecated channels 105-107 to -1 while injecting; disabled by default.",
    )
    parser.add_argument(
        "--skip-integrity-check",
        action="store_true",
        help=(
            "Production fast path: overwrite only the nine target channels in the loaded frame "
            "without copying and comparing all non-target channels."
        ),
    )
    parser.add_argument(
        "--target-poses",
        default="all",
        help="Comma-separated pose visualizations to refresh: all,dwpose,mediapipe,smplerx.",
    )
    parser.add_argument(
        "--smplerx-visible-clamp",
        choices=("none", "bbox"),
        default="none",
        help="Clamp SMPLer-X rendered mesh to the stored visible person bbox; use bbox for v2 repair.",
    )
    parser.add_argument(
        "--smplerx-visible-margin-fraction",
        type=float,
        default=0.2,
        help="BBox expansion fraction used by --smplerx-visible-clamp=bbox.",
    )
    return parser.parse_args()


def main() -> int:
    global _VERBOSE_RENDER
    args = parse_args()
    if args.preview_only and args.preview_dir is None:
        raise ValueError("--preview-only requires --preview-dir")
    if args.preview_only and args.output is not None:
        raise ValueError("--preview-only cannot be combined with --output")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be between 0 and --num-shards - 1")
    if args.output is not None and args.num_shards != 1:
        raise ValueError("--output cannot be combined with sharded processing")
    if args.max_frames is not None and args.skip_completed:
        raise ValueError("--max-frames cannot be combined with --skip-completed")
    _VERBOSE_RENDER = args.verbose_render
    target_poses = _parse_target_poses(args.target_poses)
    configure_gpu(args.gpu_id)
    input_roots = args.inputs if args.inputs else list(DEFAULT_DATASET_DIRS)
    wsv_files = discover_wsv_files(input_roots)
    if not wsv_files:
        print("No WSV files found.")
        return 1
    if args.output is not None and len(wsv_files) != 1:
        raise ValueError("--output may only be used when exactly one WSV file is selected")
    all_file_count = len(wsv_files)
    wsv_files = wsv_files[args.shard_index::args.num_shards]
    print(
        f"Worker shard {args.shard_index}/{args.num_shards}: "
        f"selected {len(wsv_files)} of {all_file_count} WSV file(s)"
    )

    pending_files = [
        path
        for path in wsv_files
        if not (args.skip_completed and _completion_marker_path(args.state_dir, path).exists())
    ]
    progress_bar = None
    if args.progress:
        print(f"Counting frames for ETA across {len(pending_files)} pending WSV file(s)...")
        pending_frames = sum(_count_wsv_frame_members(path) for path in pending_files)
        progress_bar = tqdm(
            total=pending_frames,
            desc=f"GPU {args.gpu_id if args.gpu_id is not None else 'CPU'} shard {args.shard_index}",
            unit="frame",
            dynamic_ncols=True,
            mininterval=2.0,
        )
    totals = {"files": 0, "skipped": 0, "frames": 0, "dwpose": 0, "mediapipe": 0, "smplerx": 0}
    for file_index, input_path in enumerate(wsv_files, start=1):
        if args.skip_completed and _completion_marker_path(args.state_dir, input_path).exists():
            totals["skipped"] += 1
            print(f"[{file_index}/{len(wsv_files)}] Skip completed: {input_path}")
            continue
        print(f"[{file_index}/{len(wsv_files)}] Start: {input_path}")
        if args.preview_only:
            output_path = None
        else:
            output_path = args.output.resolve() if args.output else input_path
        started = time.monotonic()
        counts = process_wsv(
            input_path,
            output_path,
            args.preview_dir,
            args.max_frames,
            args.dwpose_line_scale,
            args.dwpose_point_scale,
            progress_bar,
            args.compresslevel,
            not args.skip_integrity_check,
            args.clear_legacy_smplerx_vis,
            target_poses,
            args.smplerx_visible_clamp,
            args.smplerx_visible_margin_fraction,
        )
        elapsed_seconds = time.monotonic() - started
        if output_path is not None and args.max_frames is None:
            _save_completion_marker(
                args.state_dir,
                input_path,
                output_path,
                counts,
                elapsed_seconds,
                args.gpu_id,
            )
        print(
            f"[{file_index}/{len(wsv_files)}] Done in {elapsed_seconds:.1f}s: "
            f"{input_path} frames={counts['frames']}"
        )
        totals["files"] += 1
        for key in ("frames", "dwpose", "mediapipe", "smplerx"):
            totals[key] += counts[key]

    if progress_bar is not None:
        progress_bar.close()
    print(
        "Complete: "
        f"files={totals['files']}, skipped={totals['skipped']}, frames={totals['frames']}, "
        f"DWPose={totals['dwpose']}, MediaPipe={totals['mediapipe']}, SMPLer-X={totals['smplerx']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
