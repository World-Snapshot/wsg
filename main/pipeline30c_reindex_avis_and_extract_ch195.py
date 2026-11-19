#!/usr/bin/env python3
"""One-pass AVIS WSV repair plus channel-195 VAE-input extraction.

For each AVIS WSV this does the expensive NPZ read once:
  1. inject fixed semantic channel 195 using the legacy lexicographic crop index;
  2. write final WSV in natural AVIS JSON frame order;
  3. extract cropped raw channel-195 planes for the existing old channel-195
     latent-cache sample_ids, so VAE encoding can run without reopening WSVs.

If a WSV is listed in --already-reindexed-log, it is assumed to have already been
rewritten by pipeline30b --mode reindex-wsv. In that case this script only
extracts channel-195 planes by mapping old cache sample_ids to natural frame keys.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
WSM_ROOT = Path('/research/cbim/vast/sf895/code/WSMs/WSM_v1')
if str(WSM_ROOT) not in sys.path:
    sys.path.insert(0, str(WSM_ROOT))

import pipeline30b_fix_avis_semantic_lexicographic_order as p30b  # noqa: E402
from src.dataset.wsg_dataset import apply_resize_crop_info  # noqa: E402

CHANNEL = 195
SAMPLE_RE = re.compile(r"(?P<prefix>.*?/AVIS_(?P<split>train|val|test)_results/(?P<video>.*?)/wsv/merge\.wsv)::index_(?P<a>\d+)-index_(?P<b>\d+)$")


def save_npz_member(zf: zipfile.ZipFile, key: str, arr: np.ndarray) -> None:
    buf = io.BytesIO()
    np.save(buf, arr)
    zf.writestr(f"{key}.npy", buf.getvalue())


def split_json_for_dataset(dataset_root: Path) -> Path:
    return p30b.split_json_for_dataset(dataset_root)


def discover_wsvs(dataset_root: Path) -> list[Path]:
    return sorted(dataset_root.glob('*/wsv/merge.wsv'))


def read_completed_sources(log_path: Optional[Path]) -> set[str]:
    if log_path is None or not log_path.exists():
        return set()
    completed = set()
    for raw in log_path.read_text(errors='ignore').splitlines():
        start = raw.find('{')
        end = raw.rfind('}')
        if start < 0 or end <= start:
            continue
        try:
            record = json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            continue
        source = record.get('source')
        if source:
            completed.add(str(Path(source).resolve()))
    return completed


def parse_old_index(sample_id: str) -> Optional[int]:
    m = SAMPLE_RE.fullmatch(str(sample_id))
    if not m:
        return None
    return int(m.group('a'))


def load_extraction_requests(cache_root: Path) -> dict[str, list[dict[str, Any]]]:
    requests: dict[str, list[dict[str, Any]]] = defaultdict(list)
    shard_dir = cache_root / 'shards' / '195'
    for shard_path in sorted(shard_dir.glob('shard_*.pt')):
        payload = torch.load(shard_path, map_location='cpu', weights_only=False)
        sample_ids = list(payload.get('sample_ids') or [])
        sample_sources = list(payload.get('sample_sources') or [{} for _ in sample_ids])
        crop_infos = list(payload.get('crop_infos') or [{} for _ in sample_ids])
        if not (len(sample_ids) == len(sample_sources) == len(crop_infos)):
            raise RuntimeError(f'metadata length mismatch in {shard_path}')
        for pos, (sample_id, source, crop_info) in enumerate(zip(sample_ids, sample_sources, crop_infos)):
            if not isinstance(source, dict):
                continue
            source_path = source.get('path')
            old_index = parse_old_index(str(sample_id))
            if not source_path or old_index is None:
                continue
            requests[str(Path(source_path).resolve())].append({
                'sample_id': str(sample_id),
                'old_index': int(old_index),
                'crop_info': crop_info,
                'shard_name': shard_path.name,
                'position': int(pos),
            })
    return requests


def part_path_for(extract_dir: Path, source: Path) -> Path:
    digest = hashlib.sha1(str(source.resolve()).encode('utf-8')).hexdigest()[:16]
    video = source.parent.parent.name if source.parent.name == 'wsv' else source.stem
    return extract_dir / f'{video}_{digest}.pt'


def extract_plane(arr: np.ndarray, crop_info: dict) -> np.ndarray:
    cropped = apply_resize_crop_info(arr, crop_info)
    if CHANNEL >= cropped.shape[-1]:
        return np.full(cropped.shape[:2], -1, dtype=np.int16)
    plane = np.rint(cropped[..., CHANNEL]).astype(np.int16, copy=False)
    return plane


def save_extraction_part(extract_dir: Path, source: Path, records: list[dict[str, Any]], planes: list[np.ndarray]) -> None:
    if not records:
        return
    extract_dir.mkdir(parents=True, exist_ok=True)
    out = part_path_for(extract_dir, source)
    if out.exists():
        return
    raw = torch.from_numpy(np.stack(planes, axis=0).astype(np.int16, copy=False))
    tmp = out.with_suffix(out.suffix + '.tmp')
    torch.save({'source': str(source), 'records': records, 'raw_ch195': raw}, tmp)
    os.replace(tmp, out)


def process_one(args: tuple[str, str, str, bool, int, list[dict[str, Any]], bool]) -> dict[str, Any]:
    source_s, json_s, extract_dir_s, already_reindexed, compresslevel, requests, dry_run = args
    source = Path(source_s)
    json_path = Path(json_s)
    extract_dir = Path(extract_dir_s)
    avis = p30b.load_avis_json(json_path)
    video_name = p30b.video_name_from_wsv(source)
    video = p30b.find_video_record(avis, video_name)
    file_names = video.get('file_names', [])
    crop_to_json = p30b.build_legacy_crop_to_json_index(file_names)  # zero-based old crop -> zero-based natural json
    json_to_crop = {json_idx: crop_idx for crop_idx, json_idx in enumerate(crop_to_json)}
    frame_instances = p30b.build_frame_instances(avis, int(video['id']))

    req_by_old = defaultdict(list)
    for req in requests:
        req_by_old[int(req['old_index'])].append(req)

    records: list[dict[str, Any]] = []
    planes: list[np.ndarray] = []
    if already_reindexed:
        target = source.with_name(source.name + '.pipeline30c.tmp')
        moved_frames = 0
        total_instances = 0
        total_pixels = 0
        with np.load(source, allow_pickle=True) as npz:
            keys = list(npz.files)
            frame_keys = [key for key in keys if p30b.frame_index_from_key(key) is not None]
            other_keys = [key for key in keys if p30b.frame_index_from_key(key) is None]
            with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=compresslevel) as zf:
                for key in sorted(frame_keys, key=lambda k: p30b.frame_index_from_key(k) or 0):
                    natural_zero = (p30b.frame_index_from_key(key) or 0)
                    arr = np.array(npz[key])
                    instances = frame_instances[natural_zero] if natural_zero < len(frame_instances) else []
                    instance_count, pixel_count = p30b.p29.inject_frame_channel_195(
                        arr,
                        instances,
                        verify_unchanged=False,
                        clear_207_255=False,
                    )
                    # Old cache sample_ids still refer to the old lexicographic index;
                    # map those requests to this natural frame.
                    for old_zero, json_zero in enumerate(crop_to_json):
                        if json_zero != natural_zero:
                            continue
                        for req in req_by_old.get(old_zero + 1, []):
                            planes.append(extract_plane(arr, req['crop_info']))
                            records.append({k: req[k] for k in ('sample_id', 'old_index', 'shard_name', 'position')})
                    save_npz_member(zf, key, arr)
                    moved_frames += 1
                    total_instances += instance_count
                    total_pixels += pixel_count
                for key in other_keys:
                    save_npz_member(zf, key, np.asarray(npz[key]))
        os.replace(target, source)
        save_extraction_part(extract_dir, source, records, planes)
        return {
            'source': str(source),
            'already_reindexed': True,
            'frames': moved_frames,
            'instances': total_instances,
            'pixels': total_pixels,
            'extracted': len(records),
            'rewritten': True,
            'reinject_reindexed': True,
        }

    target = source.with_name(source.name + '.pipeline30c.tmp')
    moved_frames = 0
    total_instances = 0
    total_pixels = 0
    mapping_changed = 0
    if dry_run:
        return {'source': str(source), 'already_reindexed': False, 'frames': len(file_names), 'extract_requests': len(requests), 'dry_run': True}

    with np.load(source, allow_pickle=True) as npz:
        keys = list(npz.files)
        frame_keys = [key for key in keys if p30b.frame_index_from_key(key) is not None]
        other_keys = [key for key in keys if p30b.frame_index_from_key(key) is None]
        key_by_old_crop = {}
        for key in frame_keys:
            idx = p30b.frame_index_from_key(key)
            if idx is not None:
                key_by_old_crop[idx] = key
        with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=compresslevel) as zf:
            for json_idx in range(len(file_names)):
                old_crop_idx = json_to_crop.get(json_idx)
                if old_crop_idx is None:
                    continue
                old_key = key_by_old_crop.get(old_crop_idx)
                if old_key is None:
                    continue
                new_key = f'index_{json_idx + 1:08d}'
                arr = np.array(npz[old_key])
                instances = frame_instances[json_idx] if json_idx < len(frame_instances) else []
                instance_count, pixel_count = p30b.p29.inject_frame_channel_195(
                    arr,
                    instances,
                    verify_unchanged=False,
                    clear_207_255=False,
                )
                old_index = old_crop_idx + 1
                for req in req_by_old.get(old_index, []):
                    planes.append(extract_plane(arr, req['crop_info']))
                    records.append({k: req[k] for k in ('sample_id', 'old_index', 'shard_name', 'position')})
                save_npz_member(zf, new_key, arr)
                moved_frames += 1
                total_instances += instance_count
                total_pixels += pixel_count
                if old_key != new_key:
                    mapping_changed += 1
            for key in other_keys:
                save_npz_member(zf, key, np.asarray(npz[key]))
    os.replace(target, source)
    save_extraction_part(extract_dir, source, records, planes)
    return {
        'source': str(source),
        'already_reindexed': False,
        'frames': moved_frames,
        'instances': total_instances,
        'pixels': total_pixels,
        'mapping_changed': mapping_changed,
        'mapping_total': len(file_names),
        'extracted': len(records),
        'rewritten': True,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cache-root', type=Path, default=Path('/research/cbim/vast/sf895/code/WSMs/WSM_v1/data/wsg_flux2_frame_latents_dense_multidataset'))
    ap.add_argument('--extract-dir', type=Path, required=True)
    ap.add_argument('--dataset-root', type=Path, action='append', default=[])
    ap.add_argument('--all-avis', action='store_true')
    ap.add_argument('--base-output-dir', type=Path, default=Path('/common/users/sf895/wave2pixel'))
    ap.add_argument('--already-reindexed-log', type=Path)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--compresslevel', type=int, default=6)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    roots = list(args.dataset_root)
    if args.all_avis:
        roots.extend([
            args.base_output_dir / 'AVIS_train_results',
            args.base_output_dir / 'AVIS_val_results',
            args.base_output_dir / 'AVIS_test_results',
        ])
    roots = list(dict.fromkeys(root.resolve() for root in roots))
    requests_by_source = load_extraction_requests(args.cache_root)
    completed = read_completed_sources(args.already_reindexed_log)
    jobs = []
    for root in roots:
        json_path = split_json_for_dataset(root)
        wsvs = discover_wsvs(root)
        if args.limit is not None:
            wsvs = wsvs[:args.limit]
        for wsv in wsvs:
            source_key = str(wsv.resolve())
            jobs.append((source_key, str(json_path), str(args.extract_dir), source_key in completed, int(args.compresslevel), requests_by_source.get(source_key, []), bool(args.dry_run)))
    print(json.dumps({
        'jobs': len(jobs),
        'workers': args.workers,
        'completed_reindexed': len(completed),
        'sources_with_extract_requests': len(requests_by_source),
        'extract_dir': str(args.extract_dir),
        'dry_run': args.dry_run,
    }, indent=2), flush=True)
    done = 0
    totals = {'rewritten': 0, 'already_reindexed': 0, 'extracted': 0}
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [ex.submit(process_one, job) for job in jobs]
        for fut in as_completed(futs):
            result = fut.result()
            done += 1
            totals['rewritten'] += int(bool(result.get('rewritten')))
            totals['already_reindexed'] += int(bool(result.get('already_reindexed')))
            totals['extracted'] += int(result.get('extracted', 0))
            print(json.dumps({'done': done, 'total': len(jobs), **result}, sort_keys=True), flush=True)
    print(json.dumps({'done': done, 'total': len(jobs), **totals}, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
