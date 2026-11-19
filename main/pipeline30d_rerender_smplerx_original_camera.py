#!/usr/bin/env python3
"""Re-render SMPLer-X visualization channels from original SMPLer-X outputs.

This repairs WSV channels 90-92 without re-running SMPLer-X inference and without
modifying the stored SMPLer-X parameter channels 72-89.  It reads each video's
existing results_smplerx/smplx/*.npz and results_smplerx/meta/*.json files and
renders with the original focal/princpt camera from SMPLer-X metadata.

Only channels 90-92 and reserved channel 93 are rewritten.
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import sys
import tempfile
import time
from datetime import datetime
from contextlib import contextmanager, nullcontext
from pathlib import Path
import shutil
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from PIL import Image
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import pipeline30_fix_and_inject_three_pose_vis as pose_vis  # noqa: E402

SMPLERX_CHANNELS = slice(90, 93)
SMPLERX_RESERVED_CHANNEL = 93
DEFAULT_ROOTS = (
    Path('/common/users/sf895/wave2pixel/AVIS_train_results'),
    Path('/common/users/sf895/wave2pixel/AVIS_val_results'),
    Path('/common/users/sf895/wave2pixel/AVIS_test_results'),
)


def discover_wsv_files(inputs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in inputs:
        if path.is_file():
            files.append(path.resolve())
        elif path.is_dir():
            files.extend(sorted(path.resolve().glob('*/wsv/merge.wsv')))
        else:
            print(f'skip missing input: {path}')
    return sorted(set(files))


def parse_skip_mtime_after(value: str | None) -> float | None:
    if not value:
        return None
    for fmt in ('%Y%m%d_%H%M%S', '%Y-%m-%dT%H:%M:%S'):
        try:
            return datetime.strptime(value, fmt).timestamp()
        except ValueError:
            pass
    return float(value)


def filter_by_output_mtime(files: list[Path], cutoff: float | None) -> tuple[list[Path], int]:
    if cutoff is None:
        return files, 0
    kept: list[Path] = []
    skipped = 0
    for path in files:
        try:
            if path.stat().st_mtime >= cutoff:
                skipped += 1
                continue
        except FileNotFoundError:
            pass
        kept.append(path)
    return kept, skipped


def frame_index_from_key(key: str) -> int | None:
    if not key.startswith('index_'):
        return None
    try:
        return int(key.split('_', 1)[1])
    except ValueError:
        return None


def person_from_npz(npz_path: Path) -> dict[str, np.ndarray]:
    data = np.load(npz_path, allow_pickle=False)

    def arr(name: str, shape: tuple[int, ...]) -> np.ndarray:
        if name in data.files:
            return np.asarray(data[name]).astype(np.float32, copy=False)
        return np.zeros(shape, dtype=np.float32)

    return {
        'global_orient': arr('global_orient', (1, 3)),
        'body_pose': arr('body_pose', (21, 3)),
        'left_hand_pose': arr('left_hand_pose', (15, 3)),
        'right_hand_pose': arr('right_hand_pose', (15, 3)),
        'jaw_pose': arr('jaw_pose', (1, 3)),
        'leye_pose': arr('leye_pose', (1, 3)),
        'reye_pose': arr('reye_pose', (1, 3)),
        'transl': arr('transl', (1, 3)),
        'betas': arr('betas', (1, 10)),
        'expression': arr('expression', (1, 10)),
    }


def render_frame_from_original_smplerx(
    video_dir: Path,
    frame_index: int,
    height: int,
    width: int,
    min_detector_score: float,
) -> tuple[np.ndarray, bool, int]:
    smpl_dir = video_dir / 'results_smplerx'
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    rendered_people = 0
    if not smpl_dir.is_dir():
        return canvas, False, 0
    for npz_path in sorted((smpl_dir / 'smplx').glob(f'{frame_index:05d}_*.npz')):
        meta_path = smpl_dir / 'meta' / f'{npz_path.stem}.json'
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except Exception as exc:
            print(f'  skip bad meta {meta_path}: {exc}')
            continue
        score = float(meta.get('detector_score') or 0.0)
        if score < min_detector_score:
            continue
        try:
            person = person_from_npz(npz_path)
            vertices, faces = pose_vis._create_full_smplerx_mesh(person)
        except Exception as exc:
            print(f'  skip bad smplx {npz_path}: {exc}')
            continue
        focal = meta.get('focal') or [1500.0, 1500.0]
        princpt = meta.get('princpt') or [width / 2.0, height / 2.0]
        camera = (float(focal[0]), float(focal[1]), float(princpt[0]), float(princpt[1]))
        try:
            image = pose_vis._render_full_smplerx_mesh(vertices, faces, camera, height, width)
        except Exception as exc:
            print(f'  render failed {npz_path}: {exc}')
            continue
        mask = np.any(image != 0, axis=2)
        canvas[mask] = image[mask]
        rendered_people += 1
    return canvas, rendered_people > 0, rendered_people


def _write_wsv_array(archive: ZipFile, key: str, array: np.ndarray) -> None:
    with archive.open(f'{key}.npy', mode='w', force_zip64=True) as member:
        np.lib.format.write_array(member, np.asanyarray(array), allow_pickle=True)


@contextmanager
def _atomic_wsv_writer(output_path: Path, compresslevel: int, temp_dir: Path | None = None, overwrite_from_temp: bool = False):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_parent = temp_dir or output_path.parent
    temp_parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f'.{output_path.name}.', suffix='.wsv', dir=str(temp_parent))
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        with ZipFile(temp_path, mode='w', compression=ZIP_DEFLATED, compresslevel=compresslevel, allowZip64=True) as archive:
            yield archive
        if overwrite_from_temp:
            # Used when the source filesystem has too little quota for same-dir
            # atomic replacement. The full archive is built on temp_dir first,
            # then copied over the existing WSV so the source filesystem does
            # not need to hold both copies at once.
            shutil.copyfile(temp_path, output_path)
        else:
            os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def save_preview(preview_dir: Path, wsv_path: Path, key: str, frame: np.ndarray, render: np.ndarray) -> None:
    rgb = frame[:, :, :3].astype(np.float32)
    rgb = np.where(rgb == -1, 0, rgb)
    if rgb.size and np.nanmax(rgb) <= 1.0:
        rgb *= 255.0
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    old = frame[:, :, SMPLERX_CHANNELS].astype(np.float32)
    old = np.where(old == -1, 0, old)
    if old.size and np.nanmax(old) <= 1.0:
        old *= 255.0
    old = np.clip(old, 0, 255).astype(np.uint8)
    tiles = [rgb, old, render]
    width = 240
    images = []
    for arr in tiles:
        im = Image.fromarray(arr, 'RGB')
        h = max(1, int(round(width * im.height / max(im.width, 1))))
        images.append(im.resize((width, h), Image.Resampling.BILINEAR))
    out = Image.new('RGB', (width * len(images), max(im.height for im in images)), (0, 0, 0))
    x = 0
    for im in images:
        out.paste(im, (x, 0))
        x += width
    safe = f'{wsv_path.parent.parent.name}_{key}.png'
    preview_dir.mkdir(parents=True, exist_ok=True)
    out.save(preview_dir / safe)


def process_wsv(
    wsv_path: Path,
    output_path: Path | None,
    preview_dir: Path | None,
    preview_only: bool,
    max_frames: int | None,
    min_detector_score: float,
    compresslevel: int,
    temp_dir: Path | None,
    overwrite_from_temp: bool,
) -> dict[str, int | str]:
    video_dir = wsv_path.parent.parent
    counts = {'frames': 0, 'rendered_frames': 0, 'people': 0, 'missing': 0}
    output_context = nullcontext(None) if preview_only else _atomic_wsv_writer(
        output_path or wsv_path, compresslevel, temp_dir=temp_dir, overwrite_from_temp=overwrite_from_temp
    )
    with np.load(wsv_path, allow_pickle=False) as archive, output_context as writer:
        frame_keys = [key for key in archive.files if frame_index_from_key(key) is not None]
        frame_keys = sorted(frame_keys, key=lambda key: frame_index_from_key(key) or 0)
        if max_frames is not None:
            frame_keys = frame_keys[:max_frames]
        selected = set(frame_keys)
        for key in archive.files:
            frame = archive[key]
            if key not in selected or frame.ndim != 3 or frame.shape[-1] <= SMPLERX_RESERVED_CHANNEL:
                if writer is not None:
                    _write_wsv_array(writer, key, frame)
                continue
            frame_index = frame_index_from_key(key)
            assert frame_index is not None
            render, has_render, people = render_frame_from_original_smplerx(
                video_dir, frame_index, frame.shape[0], frame.shape[1], min_detector_score
            )
            updated = frame.copy()
            if has_render:
                updated[:, :, SMPLERX_CHANNELS] = render.astype(updated.dtype, copy=False)
                counts['rendered_frames'] += 1
                counts['people'] += int(people)
            else:
                updated[:, :, SMPLERX_CHANNELS] = -1
                counts['missing'] += 1
            updated[:, :, SMPLERX_RESERVED_CHANNEL] = -1
            if preview_dir is not None:
                save_preview(preview_dir, wsv_path, key, frame, render)
            if writer is not None:
                _write_wsv_array(writer, key, updated)
            counts['frames'] += 1
    counts['path'] = str(wsv_path)
    return counts


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument('inputs', nargs='*', type=Path, help='WSV files or AVIS result dirs. Default scans AVIS train/val/test.')
    ap.add_argument('--gpu-id', type=int)
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--shard-index', type=int, default=0)
    ap.add_argument('--preview-dir', type=Path)
    ap.add_argument('--preview-only', action='store_true')
    ap.add_argument('--max-files', type=int)
    ap.add_argument('--max-frames', type=int)
    ap.add_argument('--min-detector-score', type=float, default=0.7)
    ap.add_argument('--compresslevel', type=int, default=1, choices=range(1, 10), metavar='{1..9}')
    ap.add_argument('--temp-dir', type=Path, help='Directory for full temporary WSV archives.')
    ap.add_argument('--skip-mtime-after', help='Skip WSVs whose output mtime is >= this time (YYYYMMDD_HHMMSS, ISO, or epoch).')
    ap.add_argument(
        '--overwrite-from-temp',
        action='store_true',
        help='Write temp archive outside the source filesystem, then overwrite output in place. Saves source quota but is not atomic.',
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError('bad shard settings')
    pose_vis.configure_gpu(args.gpu_id)
    roots = args.inputs or list(DEFAULT_ROOTS)
    files = discover_wsv_files(roots)
    all_count = len(files)
    skip_cutoff = parse_skip_mtime_after(args.skip_mtime_after)
    files, skipped_mtime = filter_by_output_mtime(files, skip_cutoff)
    if skipped_mtime:
        print(f'skip mtime-complete WSV files: {skipped_mtime} / {all_count}')
    files = files[args.shard_index::args.num_shards]
    if args.max_files is not None:
        files = files[:args.max_files]
    print(f'shard {args.shard_index}/{args.num_shards}: selected {len(files)} of {all_count} WSV files')
    totals = {'files': 0, 'frames': 0, 'rendered_frames': 0, 'people': 0, 'missing': 0}
    started = time.monotonic()
    for path in tqdm(files, desc=f'smpl original camera gpu {args.gpu_id}', dynamic_ncols=True):
        result = process_wsv(
            path,
            None,
            args.preview_dir,
            args.preview_only,
            args.max_frames,
            args.min_detector_score,
            args.compresslevel,
            args.temp_dir,
            args.overwrite_from_temp,
        )
        totals['files'] += 1
        for key in ('frames', 'rendered_frames', 'people', 'missing'):
            totals[key] += int(result[key])
    elapsed = time.monotonic() - started
    print(f'complete elapsed={elapsed:.1f}s totals={totals}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
