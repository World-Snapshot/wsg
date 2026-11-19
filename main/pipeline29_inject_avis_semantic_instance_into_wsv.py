#!/usr/bin/env python3
"""
Pipeline 29: inject AVIS semantic instance categories into WSV channel 195.

Channel 195 is the 196th WSG/WSV channel. This pipeline fills it with AVIS
category ids only inside AVIS instance masks and leaves all other pixels as -1.

Optional dangerous cleanup:
  --clear-207-255 clears channels 207-255 to -1 in every processed frame.

The input annotation files are COCO/YTVIS-style AVIS JSON files:
  videos[*].file_names
  categories[*].id/name
  annotations[*].category_id
  annotations[*].segmentations[frame_index]

For each frame:
  1. reset channel 195 to -1
  2. rasterize every valid AVIS instance polygon for that frame
  3. optionally clear channels 207-255 to -1 when explicitly requested
  4. write that instance's integer category_id into channel 195

If masks overlap, larger masks are written first and smaller masks later, so
small foreground objects are less likely to be erased.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm


CHANNEL_195 = 195
WIRELESS_CLEAR_START = 207
WIRELESS_CLEAR_END = 256
EMPTY_VALUE = -1.0


class TeeLogger:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> None:
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def format_seconds(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    hours = seconds / 3600.0
    if hours >= 1:
        return f"{hours:.2f}h"
    minutes = seconds / 60.0
    if minutes >= 1:
        return f"{minutes:.1f}m"
    return f"{seconds:.1f}s"


DEFAULT_SUBSETS = {
    "train": {
        "results_dir": "../output/AVIS_train_results",
        "json_path": "../output/AVIS_train_results/train.json",
    },
    "val": {
        "results_dir": "../output/AVIS_val_results",
        "json_path": "../output/AVIS_val_results/val.json",
    },
    "test": {
        "results_dir": "../output/AVIS_test_results",
        "json_path": "../output/AVIS_test_results/test.json",
    },
}


@dataclass(frozen=True)
class FrameInstance:
    category_id: int
    segmentation: object
    area_hint: float


@dataclass
class AvisIndex:
    categories: Dict[int, str]
    video_id_by_name: Dict[str, int]
    frame_instances_by_video: Dict[int, List[List[FrameInstance]]]


def load_avis_index(json_path: Path) -> AvisIndex:
    with json_path.open("r") as f:
        avis = json.load(f)

    categories = {
        int(cat["id"]): str(cat.get("name", cat["id"]))
        for cat in avis.get("categories", [])
    }

    video_id_by_name: Dict[str, int] = {}
    frame_instances_by_video: Dict[int, List[List[FrameInstance]]] = {}

    for video in avis.get("videos", []):
        video_id = int(video["id"])
        file_names = video.get("file_names", [])
        if not file_names:
            continue

        first = file_names[0]
        parent = first.split("/", 1)[0]
        stem = Path(first).stem
        aliases = {
            parent,
            stem,
            str(video_id),
            f"{video_id:08d}",
        }
        for alias in aliases:
            video_id_by_name.setdefault(alias, video_id)

        length = int(video.get("length", len(file_names)))
        frame_instances_by_video[video_id] = [[] for _ in range(length)]

    for ann in avis.get("annotations", []):
        video_id = int(ann["video_id"])
        category_id = int(ann["category_id"])
        segmentations = ann.get("segmentations", [])
        areas = ann.get("areas", [])

        if video_id not in frame_instances_by_video:
            frame_instances_by_video[video_id] = [[] for _ in range(len(segmentations))]
        frames = frame_instances_by_video[video_id]
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
                FrameInstance(
                    category_id=category_id,
                    segmentation=seg,
                    area_hint=area_hint,
                )
            )

    return AvisIndex(
        categories=categories,
        video_id_by_name=video_id_by_name,
        frame_instances_by_video=frame_instances_by_video,
    )


def discover_wsv_files(results_dir: Path) -> List[Path]:
    return sorted(results_dir.glob("*/wsv/merge.wsv"))


def resolve_video_id(video_dir_name: str, avis_index: AvisIndex) -> Optional[int]:
    if video_dir_name in avis_index.video_id_by_name:
        return avis_index.video_id_by_name[video_dir_name]

    # Fallback for rare naming drift: match if either side contains the other.
    for alias, video_id in avis_index.video_id_by_name.items():
        if video_dir_name in alias or alias in video_dir_name:
            return video_id
    return None


def frame_index_from_wsv_key(key: str) -> Optional[int]:
    if not key.startswith("index_"):
        return None
    try:
        frame_number = int(key.split("_", 1)[1])
    except ValueError:
        return None
    # WSV keys are one-based frame file names such as index_00000001.
    return max(0, frame_number - 1)


def rasterize_polygon_segmentation(segmentation: object, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)

    if isinstance(segmentation, dict):
        # The current AVIS files observed here use polygon lists. Dict/RLE
        # support is intentionally conservative to avoid silently decoding
        # incompatible formats without pycocotools.
        return mask

    if not isinstance(segmentation, list) or not segmentation:
        return mask

    polygons: List[Sequence[float]]
    if segmentation and isinstance(segmentation[0], list):
        polygons = segmentation  # type: ignore[assignment]
    else:
        polygons = [segmentation]  # type: ignore[list-item]

    for poly in polygons:
        points: List[Tuple[float, float]] = []
        if not poly:
            continue

        if isinstance(poly[0], list):
            for point in poly:  # type: ignore[index]
                if len(point) >= 2:
                    points.append((float(point[0]), float(point[1])))
        else:
            for i in range(0, len(poly), 2):  # type: ignore[arg-type]
                if i + 1 < len(poly):  # type: ignore[arg-type]
                    points.append((float(poly[i]), float(poly[i + 1])))  # type: ignore[index]

        if len(points) < 3:
            continue

        pts = np.asarray(points, dtype=np.float32)
        pts[:, 0] = np.clip(pts[:, 0], 0, max(0, width - 1))
        pts[:, 1] = np.clip(pts[:, 1], 0, max(0, height - 1))
        cv2.fillPoly(mask, [np.rint(pts).astype(np.int32)], 1)

    return mask


def inject_frame_channel_195(
    frame: np.ndarray,
    instances: Sequence[FrameInstance],
    verify_unchanged: bool = True,
    clear_207_255: bool = False,
) -> Tuple[int, int]:
    if frame.ndim != 3 or frame.shape[2] <= CHANNEL_195:
        raise ValueError(f"Expected HxWx256-like WSG frame, got shape {frame.shape}")

    height, width = frame.shape[:2]

    # Cheap deterministic guard: the injection code is allowed to touch only
    # channel 195. This samples a few pixels over all other channels before and
    # after the write, catching accidental broad writes without copying a full
    # frame.
    sample_before = None
    sample_y = None
    sample_x = None
    other_channels = None
    if verify_unchanged:
        sample_y = np.linspace(0, height - 1, num=min(8, height), dtype=np.int64)
        sample_x = np.linspace(0, width - 1, num=min(8, width), dtype=np.int64)
        if clear_207_255:
            protected = set(range(WIRELESS_CLEAR_START, min(WIRELESS_CLEAR_END, frame.shape[2])))
            protected.add(CHANNEL_195)
            other_channels = np.asarray(
                [idx for idx in range(frame.shape[2]) if idx not in protected],
                dtype=np.int64,
            )
        else:
            other_channels = np.r_[0:CHANNEL_195, CHANNEL_195 + 1:frame.shape[2]]
        sample_before = frame[np.ix_(sample_y, sample_x, other_channels)].copy()

    frame[:, :, CHANNEL_195] = np.asarray(EMPTY_VALUE, dtype=frame.dtype)
    if clear_207_255:
        frame[:, :, WIRELESS_CLEAR_START:WIRELESS_CLEAR_END] = np.asarray(EMPTY_VALUE, dtype=frame.dtype)

    valid_instances = 0
    written_pixels = 0

    # Write larger regions first; smaller masks can overwrite overlaps.
    sorted_instances = sorted(instances, key=lambda item: item.area_hint, reverse=True)
    for instance in sorted_instances:
        mask = rasterize_polygon_segmentation(instance.segmentation, height, width)
        if not np.any(mask):
            continue

        pixels = mask > 0
        frame[pixels, CHANNEL_195] = np.asarray(float(instance.category_id), dtype=frame.dtype)
        valid_instances += 1
        written_pixels += int(np.sum(pixels))

    if verify_unchanged and sample_before is not None:
        sample_after = frame[np.ix_(sample_y, sample_x, other_channels)]
        if not np.array_equal(sample_before, sample_after):
            raise RuntimeError("Safety check failed: a channel other than 195 changed")

    return valid_instances, written_pixels


def zip_compression_settings(compression: str, compress_level: int) -> Tuple[int, Optional[int]]:
    if compression == "stored":
        return zipfile.ZIP_STORED, None
    if compression == "deflated":
        return zipfile.ZIP_DEFLATED, compress_level
    raise ValueError(f"Unsupported compression mode: {compression}")


def completion_marker_path(state_dir: Path, wsv_path: Path) -> Path:
    digest = hashlib.sha1(str(wsv_path.resolve()).encode("utf-8")).hexdigest()[:16]
    video_name = wsv_path.parent.parent.name
    return state_dir / f"{digest}_{video_name}.done.json"


def save_completion_marker(
    state_dir: Path,
    wsv_path: Path,
    counts: Dict[str, int],
    elapsed_seconds: float,
    subset_name: str,
    shard_index: int,
    num_shards: int,
) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    marker = completion_marker_path(state_dir, wsv_path)
    fd, temp_name = tempfile.mkstemp(prefix=f".{marker.name}.", suffix=".tmp", dir=str(state_dir))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(
                {
                    "input": str(wsv_path.resolve()),
                    "subset": subset_name,
                    "counts": counts,
                    "elapsed_seconds": elapsed_seconds,
                    "shard_index": shard_index,
                    "num_shards": num_shards,
                    "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                stream,
                indent=2,
            )
        os.replace(temp_name, marker)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def write_wsv_streaming(
    source_wsv: Path,
    target_wsv: Path,
    avis_index: AvisIndex,
    video_id: int,
    compression: str = "deflated",
    compress_level: int = 6,
    verify_unchanged: bool = True,
    clear_207_255: bool = False,
    progress_label: str = "",
    progress_frames: int = 10,
    progress_seconds: float = 60.0,
) -> Tuple[int, int, int, int]:
    frames = avis_index.frame_instances_by_video.get(video_id, [])
    changed_frames = 0
    total_instances = 0
    total_pixels = 0
    processed_frames = 0
    started = time.time()
    last_log = started

    with np.load(source_wsv, allow_pickle=True) as npz:
        keys = list(npz.files)
        frame_keys = [key for key in keys if frame_index_from_wsv_key(key) is not None]
        total_frame_keys = len(frame_keys)
        target_wsv.parent.mkdir(parents=True, exist_ok=True)
        zip_method, zip_level = zip_compression_settings(compression, compress_level)
        zip_kwargs = {"compression": zip_method}
        if zip_level is not None:
            zip_kwargs["compresslevel"] = zip_level
        with zipfile.ZipFile(target_wsv, mode="w", **zip_kwargs) as zf:
            for key in keys:
                arr = np.array(npz[key])
                frame_idx = frame_index_from_wsv_key(key)
                if frame_idx is not None and frame_idx < len(frames):
                    instance_count, pixel_count = inject_frame_channel_195(
                        arr,
                        frames[frame_idx],
                        verify_unchanged=verify_unchanged,
                        clear_207_255=clear_207_255,
                    )
                    changed_frames += 1
                    total_instances += instance_count
                    total_pixels += pixel_count
                if frame_idx is not None:
                    processed_frames += 1
                    now = time.time()
                    should_log = (
                        processed_frames == 1
                        or processed_frames == total_frame_keys
                        or (progress_frames > 0 and processed_frames % progress_frames == 0)
                        or (progress_seconds > 0 and now - last_log >= progress_seconds)
                    )
                    if should_log:
                        elapsed = now - started
                        avg = elapsed / max(processed_frames, 1)
                        remaining = max(0, total_frame_keys - processed_frames)
                        eta = avg * remaining
                        temp_size = target_wsv.stat().st_size if target_wsv.exists() else 0
                        print(
                            f"  [frame-progress] {progress_label}: "
                            f"{processed_frames}/{total_frame_keys} "
                            f"temp={temp_size / (1024**3):.2f}GiB "
                            f"elapsed={format_seconds(elapsed)} eta={format_seconds(eta)}"
                        )
                        last_log = now

                buffer = io.BytesIO()
                np.lib.format.write_array(buffer, arr, allow_pickle=True)
                zf.writestr(f"{key}.npy", buffer.getvalue())

    return changed_frames, total_instances, total_pixels, processed_frames


def process_wsv_file(
    wsv_path: Path,
    avis_index: AvisIndex,
    dry_run: bool = False,
    skip_existing: bool = False,
    compression: str = "deflated",
    compress_level: int = 6,
    verify_unchanged: bool = True,
    clear_207_255: bool = False,
    progress_frames: int = 10,
    progress_seconds: float = 60.0,
) -> Tuple[str, str, Dict[str, int], float]:
    started = time.time()
    video_dir_name = wsv_path.parent.parent.name
    video_id = resolve_video_id(video_dir_name, avis_index)
    if video_id is None:
        return "skip", f"{video_dir_name}: no matching AVIS video", {}, 0.0

    frame_instances = avis_index.frame_instances_by_video.get(video_id, [])
    annotated_frames = sum(1 for items in frame_instances if items)
    instance_refs = sum(len(items) for items in frame_instances)
    if dry_run:
        return (
            "success",
            f"{video_dir_name}: video_id={video_id}, annotated_frames={annotated_frames}, instance_refs={instance_refs}",
            {"annotated_frames": annotated_frames, "instance_refs": instance_refs},
            0.0,
        )

    if skip_existing:
        with np.load(wsv_path, allow_pickle=True) as npz:
            first_key = sorted(npz.files)[0] if npz.files else None
            if first_key is not None:
                first_frame = npz[first_key]
                if first_frame.ndim == 3 and first_frame.shape[2] > CHANNEL_195:
                    if np.any(first_frame[:, :, CHANNEL_195] != EMPTY_VALUE):
                        return "skip", f"{video_dir_name}: channel 195 already has data", {}, 0.0

    with tempfile.NamedTemporaryFile(
        prefix=f"{wsv_path.stem}.pipeline29.",
        suffix=".wsv",
        dir=str(wsv_path.parent),
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)

    try:
        changed_frames, total_instances, total_pixels, processed_frames = write_wsv_streaming(
            wsv_path,
            tmp_path,
            avis_index,
            video_id,
            compression=compression,
            compress_level=compress_level,
            verify_unchanged=verify_unchanged,
            clear_207_255=clear_207_255,
            progress_label=video_dir_name,
            progress_frames=progress_frames,
            progress_seconds=progress_seconds,
        )
        os.replace(tmp_path, wsv_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    elapsed = time.time() - started
    counts = {
        "frames": processed_frames,
        "changed_frames": changed_frames,
        "instances": total_instances,
        "pixels": total_pixels,
    }
    return (
        "success",
        f"{video_dir_name}: frames={changed_frames}, instances={total_instances}, pixels={total_pixels}",
        counts,
        elapsed,
    )


def process_subset(
    subset_name: str,
    results_dir: Path,
    json_path: Path,
    dry_run: bool = False,
    limit_videos: Optional[int] = None,
    skip_existing: bool = False,
    compression: str = "deflated",
    compress_level: int = 6,
    verify_unchanged: bool = True,
    clear_207_255: bool = False,
    num_shards: int = 1,
    shard_index: int = 0,
    state_dir: Path = Path("../output/pipeline29_avis_semantic_state"),
    skip_completed: bool = True,
    progress_frames: int = 10,
    progress_seconds: float = 60.0,
) -> None:
    print("=" * 72)
    print(f"Pipeline 29: AVIS semantic instance -> channel 195 [{subset_name}]")
    print(f"Results dir: {results_dir}")
    print(f"AVIS JSON:   {json_path}")
    print(f"Mode:        {'dry-run' if dry_run else 'write in-place'}")
    print(f"Channel:     {CHANNEL_195} only; all other channels are guarded")
    if clear_207_255:
        print(f"DANGER:      clearing channels {WIRELESS_CLEAR_START}-{WIRELESS_CLEAR_END - 1} to -1")
    print(f"Compression: {compression}" + (f" level {compress_level}" if compression == "deflated" else ""))

    avis_index = load_avis_index(json_path)
    print(f"Categories:  {len(avis_index.categories)}")
    print(f"Videos:      {len(avis_index.video_id_by_name)} aliases")

    wsv_files = discover_wsv_files(results_dir)
    if limit_videos is not None:
        wsv_files = wsv_files[:limit_videos]
    print(f"WSV files:   {len(wsv_files)}")
    all_file_count = len(wsv_files)
    wsv_files = wsv_files[shard_index::num_shards]
    print(f"Shard:       {shard_index}/{num_shards}, selected {len(wsv_files)} of {all_file_count} WSV files")
    print(f"State dir:   {state_dir}")

    counts = {"success": 0, "skip": 0, "fail": 0}
    start_time = time.time()
    processed = 0
    last_eta_log = 0.0
    for wsv_path in tqdm(wsv_files, desc=f"{subset_name} shard {shard_index} WSV"):
        item_start = time.time()
        if skip_completed and completion_marker_path(state_dir, wsv_path).exists() and not dry_run:
            status, message, file_counts, file_elapsed = "skip", f"{wsv_path.parent.parent.name}: completion marker exists", {}, 0.0
        else:
            status = message = None
            file_counts = {}
            file_elapsed = 0.0
        try:
            if status is None:
                status, message, file_counts, file_elapsed = process_wsv_file(
                    wsv_path,
                    avis_index,
                    dry_run=dry_run,
                    skip_existing=skip_existing,
                    compression=compression,
                    compress_level=compress_level,
                    verify_unchanged=verify_unchanged,
                    clear_207_255=clear_207_255,
                    progress_frames=progress_frames,
                    progress_seconds=progress_seconds,
                )
                if status == "success" and not dry_run:
                    save_completion_marker(
                        state_dir,
                        wsv_path,
                        file_counts,
                        file_elapsed,
                        subset_name,
                        shard_index,
                        num_shards,
                    )
        except Exception as exc:
            status, message, file_counts, file_elapsed = "fail", f"{wsv_path}: {exc}", {}, 0.0
        counts[status] = counts.get(status, 0) + 1
        processed += 1
        elapsed = time.time() - start_time
        item_elapsed = time.time() - item_start
        avg = elapsed / max(processed, 1)
        remaining = max(0, len(wsv_files) - processed)
        eta = avg * remaining
        if status != "success" or dry_run:
            print(f"  [{status}] {message}")
        now = time.time()
        if processed == 1 or processed == len(wsv_files) or now - last_eta_log >= 60:
            print(
                f"  [progress] {subset_name}: {processed}/{len(wsv_files)} "
                f"last={format_seconds(item_elapsed)} avg={format_seconds(avg)} "
                f"elapsed={format_seconds(elapsed)} eta={format_seconds(eta)} "
                f"eta_hours={eta / 3600.0:.2f}"
            )
            last_eta_log = now

    total_elapsed = time.time() - start_time
    print(f"Done [{subset_name}]: {counts}, elapsed={format_seconds(total_elapsed)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inject AVIS category ids into WSV channel 195 for train/val/test subsets."
    )
    parser.add_argument(
        "--base-output-dir",
        type=Path,
        default=Path("../output"),
        help="Base output dir containing AVIS_*_results directories.",
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=["train", "val", "test"],
        choices=sorted(DEFAULT_SUBSETS.keys()),
        help="AVIS subsets to process.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report matched videos and annotation counts; do not rewrite WSV files.",
    )
    parser.add_argument(
        "--limit-videos",
        type=int,
        default=None,
        help="Limit number of WSV files per subset for smoke tests.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a WSV when the first frame already has channel 195 data.",
    )
    parser.add_argument(
        "--compression",
        choices=["deflated", "stored"],
        default="deflated",
        help="WSV zip compression. 'deflated' keeps files smaller; 'stored' is faster but much larger.",
    )
    parser.add_argument(
        "--compress-level",
        type=int,
        default=6,
        help="Deflate compression level. Level 6 keeps good compression; lower it for quick smoke tests.",
    )
    parser.add_argument(
        "--no-verify-unchanged",
        action="store_true",
        help="Disable the cheap sampled guard that verifies non-195 channels are unchanged.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs"),
        help="Directory for pipeline29 run logs, relative to wsg/main unless absolute.",
    )
    parser.add_argument(
        "--no-log-file",
        action="store_true",
        help="Disable tee logging to a timestamped log file.",
    )
    parser.add_argument(
        "--clear-207-255",
        action="store_true",
        help="DANGEROUS: clear channels 207-255 to -1 in every processed frame.",
    )
    parser.add_argument(
        "--yes-clear-207-255",
        action="store_true",
        help="Confirm --clear-207-255 non-interactively for batch/tmux runs.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split each subset's WSV list into this many deterministic worker shards.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Process only this zero-based shard. Launch one tmux process per shard for parallelism.",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path("../output/pipeline29_avis_semantic_state"),
        help="Directory for per-WSV completion markers used by --skip-completed.",
    )
    parser.add_argument(
        "--no-skip-completed",
        action="store_true",
        help="Disable completion-marker resume behavior.",
    )
    parser.add_argument(
        "--progress-frames",
        type=int,
        default=10,
        help="Log frame-level progress every N frames inside each WSV. Use 0 to disable frame-count logging.",
    )
    parser.add_argument(
        "--progress-seconds",
        type=float,
        default=60.0,
        help="Log frame-level progress at least this often while rewriting a large WSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be between 0 and --num-shards - 1")
    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)

    log_file = None
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    if not args.no_log_file:
        log_dir = args.log_dir if args.log_dir.is_absolute() else script_dir / args.log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        subset_tag = "_".join(args.subsets)
        mode_tag = "dryrun" if args.dry_run else "write"
        shard_tag = f"shard{args.shard_index:02d}of{args.num_shards:02d}"
        log_path = log_dir / f"pipeline29_{subset_tag}_{mode_tag}_{shard_tag}_{timestamp}.log"
        log_file = log_path.open("w", buffering=1)
        sys.stdout = TeeLogger(sys.stdout, log_file)  # type: ignore[assignment]
        sys.stderr = TeeLogger(sys.stderr, log_file)  # type: ignore[assignment]
        print(f"Log file: {log_path}")

    if args.clear_207_255:
        warning = (
            "DANGEROUS OPERATION: this run will set WSV channels 207-255 to -1 "
            "for every processed frame. This destroys existing wireless sensing "
            "channels in the selected AVIS subsets."
        )
        print("\n" + "!" * 72)
        print(warning)
        print("!" * 72)
        if not args.yes_clear_207_255:
            response = input("Type CLEAR_207_255 to continue: ").strip()
            if response != "CLEAR_207_255":
                print("Abort: confirmation did not match.")
                if log_file is not None:
                    sys.stdout = original_stdout
                    sys.stderr = original_stderr
                    log_file.close()
                return

    for subset in args.subsets:
        results_dir = args.base_output_dir / f"AVIS_{subset}_results"
        json_path = results_dir / f"{subset}.json"
        process_subset(
            subset,
            results_dir=results_dir,
            json_path=json_path,
            dry_run=args.dry_run,
            limit_videos=args.limit_videos,
            skip_existing=args.skip_existing,
            compression=args.compression,
            compress_level=args.compress_level,
            verify_unchanged=not args.no_verify_unchanged,
            clear_207_255=args.clear_207_255,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
            state_dir=args.state_dir,
            skip_completed=not args.no_skip_completed,
            progress_frames=args.progress_frames,
            progress_seconds=args.progress_seconds,
        )

    if log_file is not None:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


if __name__ == "__main__":
    main()
