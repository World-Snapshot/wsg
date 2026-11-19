#!/usr/bin/env python3
"""
Fix AVIS semantic channel-195 injection for legacy lexicographic crop_frame order.

Older AVIS preprocessing copied source frames with plain string sorting. For
non-zero-padded AVIS frame names this produced crop_frame order such as
1, 10, 11, ..., 19, 2, 20, ... . Channel 195 was later injected with the
assumption that index_000000NN maps to AVIS segmentations[NN-1], so semantic
masks can be temporally permuted while RGB-derived modalities remain aligned
with crop_frame.

This script minimally fixes one or more WSV files by re-injecting only channel
195 using the same legacy lexicographic order. All non-195 channels are verified
unchanged. It can also emit before/after visualization frames and a full video
for inspection.
"""

from __future__ import annotations

import argparse
import os
import io
import json
import re
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

import pipeline29_inject_avis_semantic_instance_into_wsv as p29


CHANNEL_195 = 195
EMPTY_VALUE = -1.0


PALETTE = {
    1: (230, 25, 75),
    2: (60, 180, 75),
    3: (255, 225, 25),
    4: (0, 130, 200),
    5: (245, 130, 48),
    6: (145, 30, 180),
    7: (70, 240, 240),
    8: (240, 50, 230),
    9: (210, 245, 60),
    10: (250, 190, 212),
    11: (0, 128, 128),
    12: (220, 190, 255),
    13: (170, 110, 40),
    14: (255, 250, 200),
    15: (128, 0, 0),
    16: (170, 255, 195),
    17: (128, 128, 0),
    18: (255, 215, 180),
    19: (0, 0, 128),
    20: (128, 128, 128),
    21: (255, 255, 255),
    22: (0, 0, 0),
    23: (128, 255, 128),
    24: (255, 128, 128),
    25: (128, 128, 255),
    26: (255, 255, 128),
    27: (255, 128, 255),
    28: (128, 255, 255),
    29: (192, 192, 192),
    30: (64, 64, 64),
    31: (192, 0, 0),
    32: (0, 192, 0),
    33: (0, 0, 192),
    34: (192, 192, 0),
    35: (192, 0, 192),
    36: (0, 192, 192),
    37: (64, 0, 0),
    38: (0, 64, 0),
    39: (0, 0, 64),
    40: (64, 64, 0),
}


def frame_index_from_key(key: str) -> Optional[int]:
    match = re.fullmatch(r"index_(\d+)", key)
    if not match:
        return None
    return max(0, int(match.group(1)) - 1)


def video_name_from_wsv(wsv_path: Path) -> str:
    if wsv_path.parent.name == "wsv":
        return wsv_path.parent.parent.name
    return wsv_path.stem


def load_avis_json(json_path: Path) -> dict:
    with json_path.open("r") as f:
        return json.load(f)


def find_video_record(avis: dict, video_name: str) -> dict:
    videos = list(avis.get("videos", []))
    # Prefer the actual AVIS frame directory name. Numeric WSV folders such as
    # 00000016 can collide with zero-padded AVIS video ids from unrelated clips.
    candidate_names = [video_name]
    if video_name.endswith("_new"):
        candidate_names.append(video_name[:-4])
    for candidate in candidate_names:
        for video in videos:
            file_names = video.get("file_names", [])
            if not file_names:
                continue
            parent = file_names[0].split("/", 1)[0]
            if parent == candidate:
                return video
    for video in videos:
        try:
            video_id = int(video.get("id"))
        except (TypeError, ValueError):
            continue
        if str(video.get("id")) == video_name or f"{video_id:08d}" == video_name:
            return video
    raise KeyError(f"Could not find AVIS video record for {video_name}")


def build_legacy_crop_to_json_index(file_names: Sequence[str]) -> List[int]:
    """Return mapping from crop_frame zero-based index to AVIS JSON frame index."""
    original_index = {name: idx for idx, name in enumerate(file_names)}
    lexicographic = sorted(file_names)
    return [original_index[name] for name in lexicographic]


def build_frame_instances(avis: dict, video_id: int) -> List[List[p29.FrameInstance]]:
    length = 0
    for video in avis.get("videos", []):
        if int(video["id"]) == video_id:
            length = int(video.get("length", len(video.get("file_names", []))))
            break
    frames: List[List[p29.FrameInstance]] = [[] for _ in range(length)]
    for ann in avis.get("annotations", []):
        if int(ann["video_id"]) != video_id:
            continue
        category_id = int(ann["category_id"])
        segmentations = ann.get("segmentations", [])
        areas = ann.get("areas", [])
        if len(frames) < len(segmentations):
            frames.extend([] for _ in range(len(segmentations) - len(frames)))
        for frame_idx, seg in enumerate(segmentations):
            if not seg:
                continue
            area_hint = 0.0
            if frame_idx < len(areas) and areas[frame_idx] is not None:
                try:
                    area_hint = float(areas[frame_idx])
                except (TypeError, ValueError):
                    area_hint = 0.0
            frames[frame_idx].append(
                p29.FrameInstance(
                    category_id=category_id,
                    segmentation=seg,
                    area_hint=area_hint,
                )
            )
    return frames


def save_npz_member(zf: zipfile.ZipFile, key: str, arr: np.ndarray) -> None:
    buf = io.BytesIO()
    np.save(buf, arr)
    zf.writestr(f"{key}.npy", buf.getvalue())


def fix_wsv_channel_195(
    source_wsv: Path,
    target_wsv: Path,
    avis_json: Path,
    *,
    compression: int = zipfile.ZIP_DEFLATED,
    compresslevel: int = 6,
) -> Dict[str, int]:
    avis = load_avis_json(avis_json)
    video_name = video_name_from_wsv(source_wsv)
    video = find_video_record(avis, video_name)
    file_names = video.get("file_names", [])
    crop_to_json = build_legacy_crop_to_json_index(file_names)
    frame_instances = build_frame_instances(avis, int(video["id"]))

    target_wsv.parent.mkdir(parents=True, exist_ok=True)
    changed_frames = 0
    unchanged_other_channels = 0
    total_instances = 0
    total_pixels = 0

    with np.load(source_wsv, allow_pickle=True) as npz:
        keys = list(npz.files)
        with zipfile.ZipFile(
            target_wsv,
            "w",
            compression=compression,
            compresslevel=compresslevel,
        ) as zf:
            for key in tqdm(keys, desc=f"fix ch195 {video_name}"):
                arr = np.array(npz[key])
                frame_idx = frame_index_from_key(key)
                if frame_idx is None or frame_idx >= len(crop_to_json):
                    save_npz_member(zf, key, arr)
                    continue

                before = arr.copy()
                json_idx = crop_to_json[frame_idx]
                instances = frame_instances[json_idx] if json_idx < len(frame_instances) else []
                instance_count, pixel_count = p29.inject_frame_channel_195(
                    arr,
                    instances,
                    verify_unchanged=False,
                    clear_207_255=False,
                )
                total_instances += instance_count
                total_pixels += pixel_count
                changed_frames += 1

                if arr.shape != before.shape:
                    raise RuntimeError(f"{key}: shape changed from {before.shape} to {arr.shape}")
                other = np.ones(arr.shape[-1], dtype=bool)
                other[CHANNEL_195] = False
                if not np.array_equal(arr[..., other], before[..., other]):
                    raise RuntimeError(f"{key}: non-195 channels changed")
                unchanged_other_channels += 1
                save_npz_member(zf, key, arr)

    return {
        "frames": changed_frames,
        "verified_non195": unchanged_other_channels,
        "instances": total_instances,
        "pixels": total_pixels,
        "mapping_changed": sum(i != j for i, j in enumerate(crop_to_json)),
        "mapping_total": len(crop_to_json),
    }



def atomic_replace(src: Path, dst: Path) -> None:
    os.replace(src, dst)


def reindex_wsv_frames_and_channel_195(
    source_wsv: Path,
    target_wsv: Path,
    avis_json: Path,
    *,
    compression: int = zipfile.ZIP_DEFLATED,
    compresslevel: int = 6,
) -> Dict[str, int]:
    """Write a WSV whose index_N frames follow AVIS JSON/natural frame order."""
    avis = load_avis_json(avis_json)
    video_name = video_name_from_wsv(source_wsv)
    video = find_video_record(avis, video_name)
    file_names = video.get("file_names", [])
    crop_to_json = build_legacy_crop_to_json_index(file_names)
    json_to_crop = {json_idx: crop_idx for crop_idx, json_idx in enumerate(crop_to_json)}
    frame_instances = build_frame_instances(avis, int(video["id"]))

    target_wsv.parent.mkdir(parents=True, exist_ok=True)
    moved_frames = 0
    total_instances = 0
    total_pixels = 0
    mapping_changed = 0

    with np.load(source_wsv, allow_pickle=True) as npz:
        keys = list(npz.files)
        frame_keys = [key for key in keys if frame_index_from_key(key) is not None]
        other_keys = [key for key in keys if frame_index_from_key(key) is None]
        key_by_old_crop = {}
        for key in frame_keys:
            idx = frame_index_from_key(key)
            if idx is not None:
                key_by_old_crop[idx] = key
        with zipfile.ZipFile(target_wsv, "w", compression=compression, compresslevel=compresslevel) as zf:
            for json_idx in range(len(file_names)):
                old_crop_idx = json_to_crop.get(json_idx)
                if old_crop_idx is None:
                    continue
                old_key = key_by_old_crop.get(old_crop_idx)
                if old_key is None:
                    continue
                new_key = f"index_{json_idx + 1:08d}"
                arr = np.array(npz[old_key])
                instances = frame_instances[json_idx] if json_idx < len(frame_instances) else []
                instance_count, pixel_count = p29.inject_frame_channel_195(
                    arr,
                    instances,
                    verify_unchanged=False,
                    clear_207_255=False,
                )
                save_npz_member(zf, new_key, arr)
                moved_frames += 1
                total_instances += instance_count
                total_pixels += pixel_count
                if old_key != new_key:
                    mapping_changed += 1
            for key in other_keys:
                save_npz_member(zf, key, np.asarray(npz[key]))

    return {
        "frames": moved_frames,
        "instances": total_instances,
        "pixels": total_pixels,
        "mapping_changed": mapping_changed,
        "mapping_total": len(file_names),
    }


def discover_wsvs(dataset_root: Path) -> List[Path]:
    return sorted(dataset_root.glob("*/wsv/merge.wsv"))


def read_completed_sources(log_path: Optional[Path]) -> set[str]:
    if log_path is None or not log_path.exists():
        return set()
    completed: set[str] = set()
    for raw in log_path.read_text(errors="ignore").splitlines():
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            continue
        try:
            record = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            continue
        source = record.get("source")
        if source:
            completed.add(str(Path(source).resolve()))
    return completed


def _reindex_one_worker(args: Tuple[str, str, bool, int]) -> Dict[str, object]:
    source_s, json_s, inplace, compresslevel = args
    source = Path(source_s)
    json_path = Path(json_s)
    if inplace:
        target = source.with_name(source.name + ".pipeline30b.tmp")
    else:
        target = source.with_name(source.stem + ".fixed_order.wsv")
    stats = reindex_wsv_frames_and_channel_195(source, target, json_path, compresslevel=compresslevel)
    if inplace:
        atomic_replace(target, source)
        final_target = source
    else:
        final_target = target
    return {"source": str(source), "target": str(final_target), **stats}

def rgb_from_frame(frame: np.ndarray) -> np.ndarray:
    rgb = frame[..., :3].astype(np.float32)
    if rgb.size and (rgb.min() < 0.0 or rgb.max() <= 2.0):
        rgb = (rgb + 1.0) * 127.5
    return np.clip(rgb, 0, 255).astype(np.uint8)


def edge_from_frame(frame: np.ndarray) -> np.ndarray:
    if frame.shape[-1] <= 40:
        return np.zeros((*frame.shape[:2], 3), dtype=np.uint8)
    edge = frame[..., 40].astype(np.float32)
    if edge.size and edge.max() <= 1.5:
        edge = edge * 255.0
    edge = np.clip(edge, 0, 255).astype(np.uint8)
    return np.repeat(edge[..., None], 3, axis=2)


def normal_from_frame(frame: np.ndarray) -> np.ndarray:
    if frame.shape[-1] <= 38:
        return np.zeros((*frame.shape[:2], 3), dtype=np.uint8)
    normal = frame[..., 36:39].astype(np.float32)
    if normal.size and normal.min() < 0.0:
        normal = (normal + 1.0) * 127.5
    elif normal.size and normal.max() <= 1.5:
        normal = normal * 255.0
    return np.clip(normal, 0, 255).astype(np.uint8)


def semantic_from_frame(frame: np.ndarray) -> np.ndarray:
    if frame.shape[-1] <= CHANNEL_195:
        return np.zeros((*frame.shape[:2], 3), dtype=np.uint8)
    sem = np.rint(frame[..., CHANNEL_195].astype(np.float32)).astype(np.int32)
    out = np.zeros((*sem.shape, 3), dtype=np.uint8)
    for cls in np.unique(sem):
        if cls <= 0:
            continue
        out[sem == cls] = PALETTE.get(int(cls), (255, 255, 255))
    return out


def label_image(arr: np.ndarray, text: str) -> Image.Image:
    im = Image.fromarray(arr, "RGB")
    draw = ImageDraw.Draw(im)
    draw.rectangle((0, 0, im.width, 24), fill=(0, 0, 0))
    draw.text((5, 5), text, fill=(255, 255, 255))
    return im


def resize_tile(im: Image.Image, size: Tuple[int, int]) -> Image.Image:
    out = Image.new("RGB", size, (0, 0, 0))
    tmp = im.copy()
    tmp.thumbnail(size, Image.Resampling.BILINEAR)
    out.paste(tmp, ((size[0] - tmp.width) // 2, (size[1] - tmp.height) // 2))
    return out


def make_panel(before: np.ndarray, after: np.ndarray, key: str, tile_size: Tuple[int, int]) -> Image.Image:
    tiles = [
        label_image(rgb_from_frame(before), f"{key} RGB"),
        label_image(edge_from_frame(before), "edge ch40"),
        label_image(normal_from_frame(before), "normal ch36-38"),
        label_image(semantic_from_frame(before), "semantic before"),
        label_image(semantic_from_frame(after), "semantic fixed"),
    ]
    resized = [resize_tile(tile, tile_size) for tile in tiles]
    panel = Image.new("RGB", (tile_size[0] * len(resized), tile_size[1]), (0, 0, 0))
    for idx, tile in enumerate(resized):
        panel.paste(tile, (idx * tile_size[0], 0))
    return panel


def write_mp4(frames: List[Path], output_mp4: Path, fps: int) -> None:
    try:
        import imageio.v2 as imageio
    except Exception as exc:
        print(f"imageio unavailable; skipped mp4 ({exc})")
        return
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output_mp4, fps=fps, codec="libx264", quality=8) as writer:
        for path in frames:
            writer.append_data(np.asarray(Image.open(path).convert("RGB")))


def visualize_before_after(
    before_wsv: Path,
    after_wsv: Path,
    output_dir: Path,
    *,
    fps: int = 8,
    tile_width: int = 256,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    with np.load(before_wsv, allow_pickle=True) as before_npz, np.load(after_wsv, allow_pickle=True) as after_npz:
        keys = [k for k in before_npz.files if frame_index_from_key(k) is not None]
        keys = sorted(keys, key=lambda k: frame_index_from_key(k) or 0)
        for key in tqdm(keys, desc="visualize before/after"):
            before = np.asarray(before_npz[key])
            after = np.asarray(after_npz[key])
            if before.shape != after.shape:
                raise RuntimeError(f"{key}: before/after shape mismatch")
            other = np.ones(before.shape[-1], dtype=bool)
            other[CHANNEL_195] = False
            if not np.array_equal(before[..., other], after[..., other]):
                raise RuntimeError(f"{key}: non-195 channels changed in fixed WSV")
            h, w = before.shape[:2]
            tile_h = max(1, int(tile_width * h / max(w, 1)))
            panel = make_panel(before, after, key, (tile_width, tile_h))
            out = frame_dir / f"{key}.png"
            panel.save(out)
            saved.append(out)
    if saved:
        # Contact sheet for quick inspection.
        thumbs = [resize_tile(Image.open(p).convert("RGB"), (tile_width * 5, max(1, Image.open(p).height))) for p in saved[:12]]
        sheet = Image.new("RGB", (tile_width * 5, sum(t.height for t in thumbs)), (0, 0, 0))
        y = 0
        for tile in thumbs:
            sheet.paste(tile, (0, y))
            y += tile.height
        sheet.save(output_dir / "summary_first12.png")
        write_mp4(saved, output_dir / "before_after_full_video.mp4", fps=fps)



def _fix195_one_worker(args: Tuple[str, str, bool, int]) -> Dict[str, object]:
    source_s, json_s, inplace, compresslevel = args
    source = Path(source_s)
    json_path = Path(json_s)
    if inplace:
        target = source.with_name(source.name + ".fix195.tmp")
    else:
        target = source.with_name(source.stem + ".fix195.wsv")
    stats = fix_wsv_channel_195(source, target, json_path, compresslevel=compresslevel)
    if inplace:
        atomic_replace(target, source)
        final_target = source
    else:
        final_target = target
    return {"source": str(source), "target": str(final_target), **stats}


def split_json_for_dataset(dataset_root: Path) -> Path:
    name = dataset_root.name
    if name == "AVIS_train_results":
        return dataset_root / "train.json"
    if name == "AVIS_val_results":
        return dataset_root / "val.json"
    if name == "AVIS_test_results":
        return dataset_root / "test.json"
    for candidate in ("train.json", "val.json", "test.json"):
        path = dataset_root / candidate
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not infer AVIS json for {dataset_root}")


def run_batch(
    dataset_roots: Sequence[Path],
    mode: str,
    *,
    workers: int,
    inplace: bool,
    compresslevel: int,
    limit: Optional[int],
    dry_run: bool,
    skip_completed_log: Optional[Path],
) -> None:
    completed_sources = read_completed_sources(skip_completed_log)
    jobs: List[Tuple[str, str, bool, int]] = []
    for root in dataset_roots:
        json_path = split_json_for_dataset(root)
        wsvs = discover_wsvs(root)
        if limit is not None:
            wsvs = wsvs[:limit]
        for wsv in wsvs:
            if str(wsv.resolve()) in completed_sources:
                continue
            jobs.append((str(wsv), str(json_path), inplace, compresslevel))
    print(json.dumps({
        "mode": mode,
        "datasets": [str(p) for p in dataset_roots],
        "jobs": len(jobs),
        "skipped_completed": len(completed_sources),
        "workers": workers,
        "inplace": inplace,
        "dry_run": dry_run,
    }, indent=2), flush=True)
    if dry_run:
        for job in jobs[:20]:
            print("DRY", job[0], "json=", job[1])
        return
    worker_fn = _fix195_one_worker if mode == "fix-195" else _reindex_one_worker
    done = 0
    totals = {"frames": 0, "mapping_changed": 0, "mapping_total": 0, "instances": 0, "pixels": 0}
    with ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = [ex.submit(worker_fn, job) for job in jobs]
        for fut in as_completed(futures):
            result = fut.result()
            done += 1
            for key in totals:
                if key in result:
                    totals[key] += int(result[key])
            print(json.dumps({"done": done, "total": len(jobs), **result}, sort_keys=True), flush=True)
    print(json.dumps({"done": done, "total": len(jobs), **totals}, indent=2, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["fix-195", "reindex-wsv"], default="fix-195")
    parser.add_argument("--source-wsv", type=Path)
    parser.add_argument("--avis-json", type=Path)
    parser.add_argument("--target-wsv", type=Path)
    parser.add_argument("--dataset-root", type=Path, action="append", default=[])
    parser.add_argument("--all-avis", action="store_true", help="Process AVIS_train/val/test under --base-output-dir.")
    parser.add_argument("--base-output-dir", type=Path, default=Path("/common/users/sf895/wave2pixel"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-completed-log", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--inplace", action="store_true", help="Atomically replace each source WSV after a temp file is fully written.")
    parser.add_argument("--compresslevel", type=int, default=6)
    parser.add_argument("--visualize-dir", type=Path)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--tile-width", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.all_avis or args.dataset_root:
        roots = list(args.dataset_root)
        if args.all_avis:
            roots.extend([
                args.base_output_dir / "AVIS_train_results",
                args.base_output_dir / "AVIS_val_results",
                args.base_output_dir / "AVIS_test_results",
            ])
        roots = list(dict.fromkeys(root.resolve() for root in roots))
        run_batch(
            roots,
            args.mode,
            workers=args.workers,
            inplace=args.inplace,
            compresslevel=args.compresslevel,
            limit=args.limit,
            dry_run=args.dry_run,
            skip_completed_log=args.skip_completed_log,
        )
        return

    if args.source_wsv is None or args.avis_json is None or args.target_wsv is None:
        raise SystemExit("single-file mode requires --source-wsv, --avis-json, and --target-wsv")
    if args.mode == "fix-195":
        stats = fix_wsv_channel_195(args.source_wsv, args.target_wsv, args.avis_json, compresslevel=args.compresslevel)
    else:
        stats = reindex_wsv_frames_and_channel_195(args.source_wsv, args.target_wsv, args.avis_json, compresslevel=args.compresslevel)
    print(json.dumps(stats, indent=2, sort_keys=True))
    if args.visualize_dir:
        visualize_before_after(args.source_wsv, args.target_wsv, args.visualize_dir, fps=args.fps, tile_width=args.tile_width)


if __name__ == "__main__":
    main()
