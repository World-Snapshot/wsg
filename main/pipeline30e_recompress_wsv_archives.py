#!/usr/bin/env python3
"""Recompress WSV/NPZ archives without changing array contents."""
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from tqdm import tqdm

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


def parse_time(value: str | None) -> float | None:
    if not value:
        return None
    for fmt in ('%Y%m%d_%H%M%S', '%Y-%m-%dT%H:%M:%S'):
        try:
            return datetime.strptime(value, fmt).timestamp()
        except ValueError:
            pass
    return float(value)


def filter_mtime(files: list[Path], after: float | None, before: float | None) -> tuple[list[Path], int]:
    kept: list[Path] = []
    skipped = 0
    for path in files:
        try:
            mt = path.stat().st_mtime
        except FileNotFoundError:
            skipped += 1
            continue
        if after is not None and mt < after:
            skipped += 1
            continue
        if before is not None and mt >= before:
            skipped += 1
            continue
        kept.append(path)
    return kept, skipped


@contextmanager
def writer(output_path: Path, compresslevel: int, temp_dir: Path):
    temp_dir.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f'.{output_path.name}.', suffix='.wsv', dir=str(temp_dir))
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        with ZipFile(temp_path, mode='w', compression=ZIP_DEFLATED, compresslevel=compresslevel, allowZip64=True) as archive:
            yield archive
        shutil.copyfile(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def write_array(archive: ZipFile, key: str, array: np.ndarray) -> None:
    with archive.open(f'{key}.npy', mode='w', force_zip64=True) as member:
        np.lib.format.write_array(member, np.asanyarray(array), allow_pickle=True)


def recompress_one(path: Path, temp_dir: Path, compresslevel: int) -> tuple[int, int, int]:
    old_size = path.stat().st_size
    arrays = 0
    with np.load(path, allow_pickle=False) as src, writer(path, compresslevel, temp_dir) as dst:
        for key in src.files:
            write_array(dst, key, src[key])
            arrays += 1
    new_size = path.stat().st_size
    return old_size, new_size, arrays


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument('inputs', nargs='*', type=Path, help='WSV files or AVIS result dirs. Default scans AVIS train/val/test.')
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--shard-index', type=int, default=0)
    ap.add_argument('--mtime-after', help='Only process files whose mtime is >= this time.')
    ap.add_argument('--mtime-before', help='Only process files whose mtime is < this time.')
    ap.add_argument('--compresslevel', type=int, default=6, choices=range(1, 10), metavar='{1..9}')
    ap.add_argument('--temp-dir', type=Path, required=True)
    ap.add_argument('--max-files', type=int)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError('bad shard settings')
    roots = args.inputs or list(DEFAULT_ROOTS)
    files = discover_wsv_files(roots)
    all_count = len(files)
    files, skipped_mtime = filter_mtime(files, parse_time(args.mtime_after), parse_time(args.mtime_before))
    files = files[args.shard_index::args.num_shards]
    if args.max_files is not None:
        files = files[:args.max_files]
    print(f'shard {args.shard_index}/{args.num_shards}: selected {len(files)} of {all_count} WSV files (skipped_mtime={skipped_mtime})')
    started = time.monotonic()
    old_total = 0
    new_total = 0
    arrays_total = 0
    bad_files = 0
    for path in tqdm(files, desc=f'recompress level {args.compresslevel} shard {args.shard_index}', dynamic_ncols=True):
        try:
            old_size, new_size, arrays = recompress_one(path, args.temp_dir, args.compresslevel)
        except Exception as exc:
            bad_files += 1
            print(f'skip bad archive: {path} :: {type(exc).__name__}: {exc}', flush=True)
            continue
        old_total += old_size
        new_total += new_size
        arrays_total += arrays
    elapsed = time.monotonic() - started
    saved = old_total - new_total
    print(
        f'complete elapsed={elapsed:.1f}s files={len(files)} arrays={arrays_total} '
        f'old_gib={old_total/1024**3:.3f} new_gib={new_total/1024**3:.3f} saved_gib={saved/1024**3:.3f} bad_files={bad_files}'
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
