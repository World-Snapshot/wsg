#!/usr/bin/env python3
"""Repair WSV archives accidentally rewritten with float32 frame arrays.

The script reads completion lists produced by pipeline02_inject_smplerx_into_wsg.py
and rewrites each listed ``wsv/merge.wsv`` archive frame-by-frame, converting
floating arrays back to float16.  It does not reinject or rerender anything.
"""

from __future__ import annotations

import argparse
import io
import os
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from tqdm import tqdm


def _read_frame(archive: ZipFile, member: str) -> np.ndarray:
    with archive.open(member, "r") as handle:
        return np.load(handle, allow_pickle=False)


def _write_frame(archive: ZipFile, member: str, array: np.ndarray, dtype: np.dtype) -> None:
    if np.issubdtype(array.dtype, np.floating):
        array = array.astype(dtype, copy=False)
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    archive.writestr(member, buffer.getvalue())


def rewrite_wsv_dtype(path: Path, dtype: np.dtype, compresslevel: int) -> tuple[bool, dict[str, int]]:
    tmp_fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.dtype.", suffix=".wsv", dir=str(path.parent))
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    counts: dict[str, int] = {}
    try:
        with ZipFile(path, "r") as src, ZipFile(
            tmp_path,
            "w",
            compression=ZIP_DEFLATED,
            compresslevel=compresslevel,
            allowZip64=True,
        ) as dst:
            for member in sorted(src.namelist()):
                if member.endswith(".npy"):
                    array = _read_frame(src, member)
                    counts[str(array.dtype)] = counts.get(str(array.dtype), 0) + 1
                    _write_frame(dst, member, array, dtype)
                else:
                    dst.writestr(member, src.read(member))
        os.replace(tmp_path, path)
        return True, counts
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def load_completion_samples(dataset_dir: Path, tag: str) -> list[Path]:
    completion = dataset_dir / f"processed_video_name_by_smplerx_wsv_inject_{tag}.txt"
    if not completion.exists():
        return []
    samples = []
    for line in completion.read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if name:
            samples.append(dataset_dir / name)
    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-output-dir", type=Path, default=Path("../output"))
    parser.add_argument("--datasets", nargs="+", default=["AVIS_train", "AVIS_val", "AVIS_test"])
    parser.add_argument("--completion-tag", default="trusted_v2")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--compresslevel", type=int, default=3)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = np.dtype(args.dtype)
    targets: list[Path] = []
    for dataset in args.datasets:
        dataset_dir = args.base_output_dir / f"{dataset}_results"
        targets.extend(load_completion_samples(dataset_dir, args.completion_tag))
    targets = targets[args.shard_index::args.num_shards]
    print(f"repair targets={len(targets)} shard={args.shard_index}/{args.num_shards} dtype={dtype}")
    for sample_dir in tqdm(targets, desc="repair WSV dtype", unit="sample"):
        wsv_path = sample_dir / "wsv" / "merge.wsv"
        if not wsv_path.exists():
            print(f"missing {wsv_path}")
            continue
        if args.dry_run:
            print(wsv_path)
            continue
        ok, counts = rewrite_wsv_dtype(wsv_path, dtype, args.compresslevel)
        print(f"{'ok' if ok else 'fail'} {wsv_path} source_dtypes={counts}")


if __name__ == "__main__":
    main()
