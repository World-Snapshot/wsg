#!/usr/bin/env python3
"""Inject frame-aligned AVIS audio into merged WSV archives.

The historical version of this pipeline selected only the largest annotated
instance in each frame and wrote a possibly truncated payload into per-frame
``.wsg`` files.  That loses the correspondence between mixed audio and all
simultaneously sounding regions.  The canonical AVIS cache is now a merged
``wsv/merge.wsv`` archive, so this pipeline updates that archive directly.

For every frame this script:

* rasterizes every AVIS instance mask (the same implementation as pipeline 29),
* writes every instance category to channel 195,
* writes the frame-aligned mixed-audio payload to channels 175--184 over the
  union of all valid instance masks, and
* refuses to write a partial payload when the mask is too small.

All non-audio/non-category channels are preserved.  Each WSV is rewritten to a
temporary file and atomically replaced only after the complete archive succeeds.
"""

from __future__ import annotations

import argparse
import io
import math
import os
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import signal
from scipy.io import wavfile
from tqdm import tqdm

import pipeline29_inject_avis_semantic_instance_into_wsv as avis_semantic


AUDIO_START = 175
AUDIO_END = 185
CATEGORY_CHANNEL = 195
EMPTY_VALUE = -1.0
TARGET_SAMPLE_RATE = 8_000
HEADER_SIZE = 14
MAGIC = np.asarray([0xF0, 0x0F, 0xF0, 0x0F], dtype=np.uint8)


@dataclass
class FrameInjectionStats:
    annotated_instances: int = 0
    mask_pixels: int = 0
    audio_bytes: int = 0
    capacity_failures: int = 0
    missing_audio_segments: int = 0


@dataclass
class ArchiveInjectionStats:
    frames: int = 0
    annotated_frames: int = 0
    injected_audio_frames: int = 0
    instances: int = 0
    mask_pixels: int = 0
    capacity_failures: int = 0
    missing_audio_segments: int = 0


def pcm_to_float32(audio: np.ndarray) -> np.ndarray:
    """Convert PCM/float WAV samples to mono float32 without gain normalization."""
    samples = np.asarray(audio)
    if samples.ndim == 2:
        samples = samples.astype(np.float32).mean(axis=1)

    if np.issubdtype(samples.dtype, np.floating):
        return np.clip(samples.astype(np.float32), -1.0, 1.0)
    if samples.dtype == np.uint8:
        return (samples.astype(np.float32) - 128.0) / 128.0
    if np.issubdtype(samples.dtype, np.signedinteger):
        info = np.iinfo(samples.dtype)
        scale = float(max(abs(info.min), abs(info.max)))
        return np.clip(samples.astype(np.float32) / scale, -1.0, 1.0)
    raise TypeError(f"Unsupported WAV dtype: {samples.dtype}")


def frame_audio_8bit(
    audio: np.ndarray,
    source_rate: int,
    frame_index: int,
    frame_seconds: float = 1.0,
    target_rate: int = TARGET_SAMPLE_RATE,
) -> Optional[np.ndarray]:
    """Return one frame-aligned int8 segment without per-frame normalization."""
    if source_rate <= 0 or frame_seconds <= 0:
        raise ValueError("source_rate and frame_seconds must be positive")

    start = int(round(frame_index * frame_seconds * source_rate))
    stop = int(round((frame_index + 1) * frame_seconds * source_rate))
    if start >= len(audio):
        return None

    expected_source = max(1, stop - start)
    segment = np.asarray(audio[start:min(stop, len(audio))], dtype=np.float32)
    if len(segment) < expected_source:
        segment = np.pad(segment, (0, expected_source - len(segment)))

    target_samples = int(round(frame_seconds * target_rate))
    divisor = math.gcd(source_rate, target_rate)
    resampled = signal.resample_poly(
        segment,
        up=target_rate // divisor,
        down=source_rate // divisor,
    )
    if len(resampled) < target_samples:
        resampled = np.pad(resampled, (0, target_samples - len(resampled)))
    else:
        resampled = resampled[:target_samples]

    # Preserve relative loudness between frames.  The old injector normalized
    # every second independently, which changed the meaning of the audio cue.
    return np.rint(np.clip(resampled, -1.0, 1.0) * 127.0).astype(np.int8)


def create_audio_payload(audio_8bit: np.ndarray, frame_index: int) -> np.ndarray:
    """Create the reader-compatible 14-byte header followed by uint8 audio."""
    audio_i8 = np.asarray(audio_8bit, dtype=np.int8).reshape(-1)
    length = int(len(audio_i8))
    checksum = int(np.sum(audio_i8.astype(np.int64))) & 0xFFFFFFFF
    header = np.concatenate(
        [
            MAGIC,
            np.asarray([frame_index & 0xFF, (frame_index >> 8) & 0xFF], dtype=np.uint8),
            np.asarray([(length >> shift) & 0xFF for shift in (0, 8, 16, 24)], dtype=np.uint8),
            np.asarray([(checksum >> shift) & 0xFF for shift in (0, 8, 16, 24)], dtype=np.uint8),
        ]
    )
    return np.concatenate([header, (audio_i8.astype(np.int16) + 128).astype(np.uint8)])


def bytes_to_storage_values(payload: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Encode bytes so the existing float16 reader recovers every byte exactly.

    The reader decodes ``int((value / 100 + .5) * 255)``.  Encoding the center
    of each byte's interval avoids the off-by-one errors produced by the old
    lower-bound mapping after float16 rounding.
    """
    byte_values = np.asarray(payload, dtype=np.uint8)
    encoded = ((byte_values.astype(np.float32) + 0.5) / 255.0 - 0.5) * 100.0
    stored = encoded.astype(dtype)
    decoded = np.clip(((stored.astype(np.float32) / 100.0 + 0.5) * 255.0).astype(np.int32), 0, 255)
    if not np.array_equal(decoded.astype(np.uint8), byte_values):
        raise RuntimeError("Audio byte encoding is not lossless for the WSV dtype")
    return stored


def decode_storage_values(values: np.ndarray) -> np.ndarray:
    decoded = np.clip(
        ((np.asarray(values, dtype=np.float32) / 100.0 + 0.5) * 255.0).astype(np.int32),
        0,
        255,
    )
    return decoded.astype(np.uint8)


def rasterize_instances(
    instances: Sequence[avis_semantic.FrameInstance],
    height: int,
    width: int,
) -> Tuple[np.ndarray, int]:
    """Return the union of every valid AVIS instance mask."""
    union = np.zeros((height, width), dtype=bool)
    valid_instances = 0
    for instance in instances:
        mask = avis_semantic.rasterize_polygon_segmentation(
            instance.segmentation,
            height,
            width,
        )
        if np.any(mask):
            union |= mask.astype(bool)
            valid_instances += 1
    return union, valid_instances


def inject_frame(
    frame: np.ndarray,
    instances: Sequence[avis_semantic.FrameInstance],
    audio_8bit: Optional[np.ndarray],
    frame_index: int,
    verify_unchanged: bool = True,
) -> FrameInjectionStats:
    """Inject category/mask correspondence and one complete audio payload."""
    if frame.ndim != 3 or frame.shape[2] <= CATEGORY_CHANNEL:
        raise ValueError(f"Expected HxWx256-like frame, got {frame.shape}")

    height, width = frame.shape[:2]
    protected_before = None
    protected_channels = np.r_[0:AUDIO_START, AUDIO_END:CATEGORY_CHANNEL, CATEGORY_CHANNEL + 1:frame.shape[2]]
    sample_y = np.linspace(0, height - 1, min(8, height), dtype=np.int64)
    sample_x = np.linspace(0, width - 1, min(8, width), dtype=np.int64)
    if verify_unchanged:
        protected_before = frame[np.ix_(sample_y, sample_x, protected_channels)].copy()

    # Channel 195 is reconstructed from all AVIS instances, using the canonical
    # overlap policy from pipeline 29 (large masks first, small masks last).
    semantic_instances, _ = avis_semantic.inject_frame_channel_195(
        frame,
        instances,
        verify_unchanged=False,
    )
    union_mask, valid_instances = rasterize_instances(instances, height, width)
    if semantic_instances != valid_instances:
        raise RuntimeError("Semantic and audio mask rasterization disagree")

    frame[:, :, AUDIO_START:AUDIO_END] = np.asarray(EMPTY_VALUE, dtype=frame.dtype)
    mask_pixels = int(np.count_nonzero(union_mask))
    stats = FrameInjectionStats(
        annotated_instances=valid_instances,
        mask_pixels=mask_pixels,
    )

    if mask_pixels > 0 and audio_8bit is None:
        stats.missing_audio_segments = 1
    elif mask_pixels > 0 and audio_8bit is not None:
        payload = create_audio_payload(audio_8bit, frame_index)
        capacity = mask_pixels * (AUDIO_END - AUDIO_START)
        if capacity < len(payload):
            # Never create a corrupt but superficially non-empty frame.
            stats.capacity_failures = 1
        else:
            repeated = np.resize(payload, capacity)
            stored = bytes_to_storage_values(repeated, frame.dtype).reshape(mask_pixels, AUDIO_END - AUDIO_START)
            frame[union_mask, AUDIO_START:AUDIO_END] = stored
            recovered = decode_storage_values(frame[union_mask, AUDIO_START:AUDIO_END].reshape(-1)[: len(payload)])
            if not np.array_equal(recovered, payload):
                raise RuntimeError("Audio payload failed immediate read-back verification")
            stats.audio_bytes = int(len(payload))

    if verify_unchanged and protected_before is not None:
        protected_after = frame[np.ix_(sample_y, sample_x, protected_channels)]
        if not np.array_equal(protected_before, protected_after):
            raise RuntimeError("Safety check failed: an unrelated WSV channel changed")
    return stats


def find_audio_file(video_dir: Path) -> Optional[Path]:
    audio_dir = video_dir / "WAVAudios"
    preferred = audio_dir / f"{video_dir.name}.wav"
    if preferred.is_file():
        return preferred
    candidates = sorted(audio_dir.glob("*.wav"))
    return candidates[0] if candidates else None


def write_array(zf: zipfile.ZipFile, key: str, array: np.ndarray) -> None:
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, array, allow_pickle=True)
    zf.writestr(f"{key}.npy", buffer.getvalue())


def rewrite_wsv(
    wsv_path: Path,
    avis_index: avis_semantic.AvisIndex,
    video_id: int,
    audio: np.ndarray,
    source_rate: int,
    frame_seconds: float,
    compression: str,
    compress_level: int,
    verify_unchanged: bool,
) -> ArchiveInjectionStats:
    stats = ArchiveInjectionStats()
    annotated = avis_index.frame_instances_by_video.get(video_id, [])
    method, level = avis_semantic.zip_compression_settings(compression, compress_level)
    zip_kwargs: Dict[str, object] = {"compression": method}
    if level is not None:
        zip_kwargs["compresslevel"] = level

    with tempfile.NamedTemporaryFile(
        prefix=f"{wsv_path.stem}.pipeline18.",
        suffix=".wsv",
        dir=str(wsv_path.parent),
        delete=False,
    ) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        with np.load(wsv_path, allow_pickle=True) as source, zipfile.ZipFile(temp_path, "w", **zip_kwargs) as target:
            for key in source.files:
                array = np.array(source[key])
                frame_index = avis_semantic.frame_index_from_wsv_key(key)
                if frame_index is not None:
                    stats.frames += 1
                    instances = annotated[frame_index] if frame_index < len(annotated) else []
                    segment = frame_audio_8bit(
                        audio,
                        source_rate,
                        frame_index,
                        frame_seconds=frame_seconds,
                    )
                    frame_stats = inject_frame(
                        array,
                        instances,
                        segment,
                        frame_index,
                        verify_unchanged=verify_unchanged,
                    )
                    if frame_stats.annotated_instances:
                        stats.annotated_frames += 1
                    if frame_stats.audio_bytes:
                        stats.injected_audio_frames += 1
                    stats.instances += frame_stats.annotated_instances
                    stats.mask_pixels += frame_stats.mask_pixels
                    stats.capacity_failures += frame_stats.capacity_failures
                    stats.missing_audio_segments += frame_stats.missing_audio_segments
                write_array(target, key, array)
        os.replace(temp_path, wsv_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    return stats


def process_subset(args: argparse.Namespace, subset: str) -> None:
    results_dir = args.base_output_dir / f"AVIS_{subset}_results"
    json_path = results_dir / f"{subset}.json"
    avis_index = avis_semantic.load_avis_index(json_path)
    wsv_files = avis_semantic.discover_wsv_files(results_dir)
    wsv_files = wsv_files[args.shard_index :: args.num_shards]
    if args.limit_videos is not None:
        wsv_files = wsv_files[: args.limit_videos]

    print(f"[{subset}] JSON={json_path} WSV={len(wsv_files)} shard={args.shard_index}/{args.num_shards}")
    totals = ArchiveInjectionStats()
    failures = 0
    for wsv_path in tqdm(wsv_files, desc=f"AVIS {subset} audio"):
        video_dir = wsv_path.parent.parent
        video_id = avis_semantic.resolve_video_id(video_dir.name, avis_index)
        audio_path = find_audio_file(video_dir)
        if video_id is None or audio_path is None:
            failures += 1
            print(f"  [skip] {video_dir.name}: video_id={video_id}, audio={audio_path}")
            continue

        if args.dry_run:
            frame_lists = avis_index.frame_instances_by_video.get(video_id, [])
            print(
                f"  [dry-run] {video_dir.name}: audio={audio_path.name}, "
                f"annotated_frames={sum(bool(x) for x in frame_lists)}, "
                f"instances={sum(len(x) for x in frame_lists)}"
            )
            continue

        source_rate, raw_audio = wavfile.read(audio_path)
        audio = pcm_to_float32(raw_audio)
        item = rewrite_wsv(
            wsv_path,
            avis_index,
            video_id,
            audio,
            int(source_rate),
            args.frame_seconds,
            args.compression,
            args.compress_level,
            not args.no_verify_unchanged,
        )
        for field in totals.__dataclass_fields__:
            setattr(totals, field, getattr(totals, field) + getattr(item, field))
        print(
            f"  [ok] {video_dir.name}: frames={item.frames}, annotated={item.annotated_frames}, "
            f"audio={item.injected_audio_frames}, instances={item.instances}, "
            f"small_masks={item.capacity_failures}, missing_audio={item.missing_audio_segments}"
        )

    if not args.dry_run:
        print(f"[{subset}] totals={totals} missing_video_or_wav={failures}")


def self_test() -> None:
    """Fast synthetic test for byte fidelity, multi-mask semantics and guards."""
    every_byte = np.arange(256, dtype=np.uint8)
    encoded = bytes_to_storage_values(every_byte, np.dtype(np.float16))
    assert np.array_equal(decode_storage_values(encoded), every_byte)

    frame = np.full((64, 64, 256), -1.0, dtype=np.float16)
    frame[:, :, 0] = np.float16(0.25)
    instances = [
        avis_semantic.FrameInstance(2, [[2, 2, 31, 2, 31, 31, 2, 31]], 900),
        avis_semantic.FrameInstance(5, [[32, 32, 61, 32, 61, 61, 32, 61]], 900),
    ]
    audio = np.arange(TARGET_SAMPLE_RATE, dtype=np.int16)
    audio = ((audio % 255) - 127).astype(np.int8)
    result = inject_frame(frame, instances, audio, frame_index=3)
    assert result.annotated_instances == 2
    assert result.audio_bytes == HEADER_SIZE + TARGET_SAMPLE_RATE
    assert set(np.unique(frame[:, :, CATEGORY_CHANNEL]).astype(int)) == {-1, 2, 5}
    assert np.all(frame[:, :, 0] == np.float16(0.25))
    assert int(np.count_nonzero(np.any(frame[:, :, AUDIO_START:AUDIO_END] != EMPTY_VALUE, axis=2))) == result.mask_pixels
    print("pipeline18 self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-output-dir", type=Path, default=Path("../output"))
    parser.add_argument("--subsets", nargs="+", choices=["train", "val", "test"], default=["train", "val", "test"])
    parser.add_argument("--frame-seconds", type=float, default=1.0, help="AVIS frame duration; official caches use 1 FPS.")
    parser.add_argument("--compression", choices=["deflated", "stored"], default="deflated")
    parser.add_argument("--compress-level", type=int, default=6)
    parser.add_argument("--limit-videos", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-verify-unchanged", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("Require num-shards >= 1 and 0 <= shard-index < num-shards")
    if not 0 <= args.compress_level <= 9:
        parser.error("compress-level must be in [0, 9]")
    return args


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    started = time.time()
    for subset in args.subsets:
        process_subset(args, subset)
    print(f"Done in {(time.time() - started) / 60.0:.1f} minutes")


if __name__ == "__main__":
    main()
