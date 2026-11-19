#!/usr/bin/env python3
"""Fix MatSynth WSG RGB channel scale.

MatSynth pipeline17 historically wrote rendered preview RGB channels 0-2 in
[0, 1]. WSG RGB channels are normalized elsewhere as image-space [0, 255], so
existing files need channels 0-2 multiplied by 255 while PBR channels stay
unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/common/users/sf895/wave2pixel/MatSynth_results/wsg"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--threshold", type=float, default=1.5, help="Only fix files whose valid RGB max is <= threshold.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--report", type=Path, default=Path("/research/cbim/vast/sf895/code/WSMs/WSM_v1/train_log_sparse/matsynth_rgb_scale_fix_report.jsonl"))
    return ap.parse_args()


def iter_files(root: Path, limit: int = 0) -> list[Path]:
    files = sorted(root.glob("*/*.wsg"))
    if limit and limit > 0:
        files = files[:limit]
    return files


def process_one(path: Path, threshold: float, dry_run: bool) -> dict:
    try:
        with np.load(path, allow_pickle=False) as payload:
            key = "wsg_data" if "wsg_data" in payload.files else payload.files[0]
            arr = payload[key]
        rgb = arr[..., 0:3]
        valid = rgb[rgb >= 0]
        if valid.size == 0:
            return {"path": str(path), "status": "no_valid_rgb"}
        old_min = float(valid.min())
        old_max = float(valid.max())
        if old_max > threshold:
            return {"path": str(path), "status": "already_scaled", "old_min": old_min, "old_max": old_max}
        if dry_run:
            return {"path": str(path), "status": "would_fix", "old_min": old_min, "old_max": old_max}
        fixed = np.array(arr, copy=True)
        mask = fixed[..., 0:3] >= 0
        fixed_rgb = fixed[..., 0:3].astype(np.float32)
        fixed_rgb[mask] = np.clip(fixed_rgb[mask] * 255.0, 0.0, 255.0)
        fixed[..., 0:3] = fixed_rgb.astype(fixed.dtype, copy=False)
        tmp = path.with_name(path.name + f".tmp.{os.getpid()}.npz")
        np.savez_compressed(tmp, **{key: fixed.astype(np.float16, copy=False)})
        tmp.replace(path)
        new_valid = fixed[..., 0:3][fixed[..., 0:3] >= 0]
        return {
            "path": str(path),
            "status": "fixed",
            "old_min": old_min,
            "old_max": old_max,
            "new_min": float(new_valid.min()),
            "new_max": float(new_valid.max()),
        }
    except Exception as exc:
        return {"path": str(path), "status": "error", "error": repr(exc)}


def main() -> None:
    args = parse_args()
    files = iter_files(args.root, args.limit)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    counts = {}
    with args.report.open("w", encoding="utf-8") as handle:
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futs = [ex.submit(process_one, p, args.threshold, args.dry_run) for p in files]
            for i, fut in enumerate(as_completed(futs), 1):
                rec = fut.result()
                counts[rec["status"]] = counts.get(rec["status"], 0) + 1
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if i % 500 == 0 or i == len(files):
                    print(json.dumps({"done": i, "total": len(files), "counts": counts}, ensure_ascii=False), flush=True)
    print(json.dumps({"done": len(files), "total": len(files), "counts": counts, "report": str(args.report)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
