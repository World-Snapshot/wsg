#!/usr/bin/env python3
"""
Pipeline 33: Upload datasets to Hugging Face with streaming tar packaging
"""

import os
import sys
import time
import argparse
from pathlib import Path
from huggingface_hub import HfApi
import json
import hashlib
import subprocess
import threading
import psutil
from datetime import datetime
import pty
import select
import io
from typing import List, Tuple, Optional
import tempfile

class HuggingFaceLargeUploader:
    def __init__(self, repo_id, repo_type="dataset", token=None):
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.api = HfApi(token=token)
        self.progress_file = "upload_progress.json"
        self.chunk_size = 15 * 1024 * 1024 * 1024  # 15GB per chunk
        self.temp_dir = Path(
            os.environ.get(
                "WSG_HF_UPLOAD_TMP",
                "/common/users/sf895/wsg_hf_upload_tmp",
            )
        )
        
    def ensure_repo_exists(self):
        """确保仓库存在，如果不存在则创建"""
        try:
            # 检查仓库是否存在
            self.api.repo_info(repo_id=self.repo_id, repo_type=self.repo_type)
            print(f"✓ 仓库 {self.repo_id} 已存在")
        except Exception:
            # 仓库不存在，创建新仓库
            print(f"仓库 {self.repo_id} 不存在，正在创建...")
            try:
                # 创建仓库
                self.api.create_repo(
                    repo_id=self.repo_id,
                    repo_type=self.repo_type,
                    private=False
                )
                print(f"✓ 成功创建仓库 {self.repo_id}")
                
                # 创建 LICENSE 文件设置 MIT 许可证
                license_content = """MIT License

Copyright (c) 2025 World-Snapshot

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""
                
                # 上传 LICENSE 文件
                self.api.upload_file(
                    path_or_fileobj=license_content.encode('utf-8'),
                    path_in_repo="LICENSE",
                    repo_id=self.repo_id,
                    repo_type=self.repo_type,
                    commit_message="Add MIT License"
                )
                print(f"✓ 添加 MIT 许可证")
                
            except Exception as e:
                print(f"✗ 创建仓库失败: {e}")
                raise
        
    def load_progress(self):
        """Load upload progress from file"""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_progress(self, progress):
        """Save upload progress to file"""
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
    
    def get_folder_info(self, folder_path):
        """Get folder size and file count"""
        total_size = 0
        file_count = 0
        
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.isfile(file_path):
                    total_size += os.path.getsize(file_path)
                    file_count += 1
                    
        return total_size, file_count
    
    def format_size(self, size):
        """Format file size in human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.2f} {unit}"
            size /= 1024
        return f"{size:.2f} TB"
    
    @staticmethod
    def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
        """Hash a large shard without loading it into memory."""
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(chunk_size):
                digest.update(chunk)
        return digest.hexdigest()

    def load_remote_json(self, path_in_repo: str) -> dict:
        """Load a small JSON manifest if it already exists remotely."""
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError

        try:
            local_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=path_in_repo,
                repo_type=self.repo_type,
                token=self.api.token,
            )
        except EntryNotFoundError:
            return {}
        with open(local_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def upload_raw_pt_modalities(
        self,
        cache_path: Path,
        cache_name: str,
        modalities: List[str],
        force_replace: bool = False,
        dry_run: bool = False,
    ) -> bool:
        """Upload modality-scoped PT shards without wrapping them in tar files."""
        from huggingface_hub import CommitOperationAdd

        cache_path = cache_path.resolve()
        shards_root = cache_path / "shards"
        if not shards_root.is_dir():
            raise FileNotFoundError(f"Missing shards directory: {shards_root}")

        available = sorted(path.name for path in shards_root.iterdir() if path.is_dir())
        selected = available if modalities == ["all"] else modalities
        unknown = sorted(set(selected) - set(available))
        if unknown:
            raise ValueError(f"Unknown modalities for {cache_name}: {', '.join(unknown)}")

        modality_files = {
            modality: sorted((shards_root / modality).glob("*.pt"))
            for modality in selected
        }
        for modality, pt_files in modality_files.items():
            if not pt_files:
                raise RuntimeError(
                    f"No PT shards found for modality {modality}: {shards_root / modality}"
                )
        total_bytes = sum(
            path.stat().st_size
            for pt_files in modality_files.values()
            for path in pt_files
        )
        processed_bytes = 0
        started_at = time.monotonic()

        print(f"Raw PT repository: {self.repo_id}")
        print(f"Cache: {cache_path}")
        print(f"Modalities: {', '.join(selected)}")
        print(f"Total PT data: {self.format_size(total_bytes)}")
        print(f"Force replace: {force_replace}")

        for modality_index, modality in enumerate(selected, 1):
            pt_files = modality_files[modality]
            modality_bytes = sum(path.stat().st_size for path in pt_files)
            print(
                f"\n=== Modality {modality_index}/{len(selected)}: {modality} "
                f"({self.format_size(modality_bytes)}) ==="
            )

            manifest_path = f"manifests/{cache_name}/{modality}.sha256.json"
            remote_manifest = (
                {} if force_replace or dry_run else self.load_remote_json(manifest_path)
            )
            remote_entries = remote_manifest.get("files", {})
            local_entries = {}
            changed_files = []

            print(f"\n[{modality}] hashing {len(pt_files)} PT shard(s)...")
            for pt_path in pt_files:
                repo_path = f"{cache_name}/shards/{modality}/{pt_path.name}"
                entry = {
                    "sha256": self.sha256_file(pt_path),
                    "size": pt_path.stat().st_size,
                }
                local_entries[repo_path] = entry
                if force_replace or remote_entries.get(repo_path) != entry:
                    changed_files.append((pt_path, repo_path))

            manifest = {
                "format": "wsm-raw-pt-modality-manifest-v1",
                "wsg_version": "1.8",
                "cache": cache_name,
                "modality": modality,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "files": local_entries,
            }
            manifest_bytes = json.dumps(
                manifest, ensure_ascii=False, indent=2
            ).encode("utf-8")

            changed_size = sum(path.stat().st_size for path, _ in changed_files)
            print(
                f"[{modality}] changed={len(changed_files)}/{len(pt_files)} "
                f"upload={self.format_size(changed_size)}"
            )
            if dry_run:
                for _, repo_path in changed_files:
                    print(f"  would upload: {repo_path}")
            elif not changed_files and remote_manifest:
                print(f"[{modality}] remote manifest matches; nothing to upload")
            else:
                operations = [
                    CommitOperationAdd(path_in_repo=repo_path, path_or_fileobj=str(path))
                    for path, repo_path in changed_files
                ]
                operations.append(
                    CommitOperationAdd(path_in_repo=manifest_path, path_or_fileobj=manifest_bytes)
                )
                commit_message = (
                    f"Update raw PT modality {modality}: "
                    f"{len(changed_files)}/{len(pt_files)} shard(s)"
                )
                for attempt in range(1, 4):
                    try:
                        commit = self.api.create_commit(
                            repo_id=self.repo_id,
                            repo_type=self.repo_type,
                            operations=operations,
                            commit_message=commit_message,
                        )
                        break
                    except Exception:
                        if attempt == 3:
                            raise
                        wait_seconds = attempt * 30
                        print(
                            f"[{modality}] upload attempt {attempt}/3 failed; "
                            f"retrying in {wait_seconds}s"
                        )
                        time.sleep(wait_seconds)
                print(f"[{modality}] committed: {commit.commit_url}")

            processed_bytes += modality_bytes
            elapsed = max(time.monotonic() - started_at, 1e-6)
            progress = processed_bytes / total_bytes if total_bytes else 1.0
            throughput = processed_bytes / elapsed
            eta_seconds = (total_bytes - processed_bytes) / throughput if throughput else 0
            print(
                f"[overall] {modality_index}/{len(selected)} "
                f"{progress:.1%} ({self.format_size(processed_bytes)} / "
                f"{self.format_size(total_bytes)}) "
                f"average={self.format_size(throughput)}/s "
                f"eta={self.format_duration(eta_seconds)}"
            )

        return True

    def upload_raw_file_group(
        self,
        root_path: Path,
        repo_root: str,
        files: List[Path],
        manifest_name: str,
        force_replace: bool = False,
        dry_run: bool = False,
    ) -> bool:
        """Upload a small, explicitly selected group of training-cache files."""
        from huggingface_hub import CommitOperationAdd

        root_path = root_path.resolve()
        files = sorted(path.resolve() for path in files if path.is_file())
        if not files:
            raise RuntimeError(f"No files selected for {repo_root}/{manifest_name}")

        manifest_path = f"manifests/{repo_root}/{manifest_name}.sha256.json"
        remote_manifest = (
            {} if force_replace or dry_run else self.load_remote_json(manifest_path)
        )
        remote_entries = remote_manifest.get("files", {})
        local_entries = {}
        changed_files = []
        total_size = sum(path.stat().st_size for path in files)

        print(
            f"\n=== Support group {repo_root}/{manifest_name}: "
            f"{len(files)} file(s), {self.format_size(total_size)} ==="
        )
        for local_path in files:
            relative_path = local_path.relative_to(root_path).as_posix()
            path_in_repo = f"{repo_root}/{relative_path}"
            entry = {
                "sha256": self.sha256_file(local_path),
                "size": local_path.stat().st_size,
            }
            local_entries[path_in_repo] = entry
            if force_replace or remote_entries.get(path_in_repo) != entry:
                changed_files.append((local_path, path_in_repo))

        changed_size = sum(path.stat().st_size for path, _ in changed_files)
        print(
            f"[{manifest_name}] changed={len(changed_files)}/{len(files)} "
            f"upload={self.format_size(changed_size)}"
        )
        if dry_run:
            for _, path_in_repo in changed_files:
                print(f"  would upload: {path_in_repo}")
            return True
        if not changed_files and remote_manifest:
            print(f"[{manifest_name}] remote manifest matches; nothing to upload")
            return True

        manifest = {
            "format": "wsm-raw-file-group-manifest-v1",
            "wsg_version": "1.8",
            "root": repo_root,
            "group": manifest_name,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "files": local_entries,
        }
        operations = [
            CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=str(local_path))
            for local_path, path_in_repo in changed_files
        ]
        operations.append(
            CommitOperationAdd(
                path_in_repo=manifest_path,
                path_or_fileobj=json.dumps(
                    manifest, ensure_ascii=False, indent=2
                ).encode("utf-8"),
            )
        )
        commit_message = (
            f"Update raw cache group {repo_root}/{manifest_name}: "
            f"{len(changed_files)}/{len(files)} file(s)"
        )
        for attempt in range(1, 4):
            try:
                commit = self.api.create_commit(
                    repo_id=self.repo_id,
                    repo_type=self.repo_type,
                    operations=operations,
                    commit_message=commit_message,
                )
                break
            except Exception:
                if attempt == 3:
                    raise
                wait_seconds = attempt * 30
                print(
                    f"[{manifest_name}] upload attempt {attempt}/3 failed; "
                    f"retrying in {wait_seconds}s"
                )
                time.sleep(wait_seconds)
        print(f"[{manifest_name}] committed: {commit.commit_url}")
        return True

    @staticmethod
    def format_duration(seconds: float) -> str:
        seconds = max(int(seconds), 0)
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours}h{minutes:02d}m"
        if minutes:
            return f"{minutes}m{secs:02d}s"
        return f"{secs}s"

    def get_remote_files(self, folder_name):
        """从 HuggingFace 获取指定文件夹下的所有文件"""
        try:
            remote_files = set()
            
            # 列出仓库中的文件
            repo_files = self.api.list_repo_files(
                repo_id=self.repo_id,
                repo_type=self.repo_type
            )
            
            # 筛选出属于当前文件夹的文件
            folder_prefix = f"{folder_name}/"
            for file_path in repo_files:
                if file_path.startswith(folder_prefix):
                    # 提取相对路径
                    rel_path = file_path[len(folder_prefix):]
                    remote_files.add(rel_path)
            
            return remote_files
            
        except Exception as e:
            print(f"警告: 无法获取远程文件列表: {e}")
            return set()
    
    def sync_delete_remote_files(self, folder_name, local_files, remote_files, dry_run=False):
        """删除远程存在但本地不存在的文件"""
        from huggingface_hub import CommitOperationDelete
        
        # 找出需要删除的文件（远程有但本地没有的）
        files_to_delete = remote_files - local_files
        
        if not files_to_delete:
            print("  没有需要删除的远程文件")
            return True
        
        print(f"\n⚠️ 发现 {len(files_to_delete)} 个远程文件在本地不存在")
        
        if dry_run:
            print("[模拟运行] 将删除以下远程文件:")
            for i, file in enumerate(sorted(files_to_delete)[:10]):
                print(f"  - {folder_name}/{file}")
            if len(files_to_delete) > 10:
                print(f"  ... 还有 {len(files_to_delete) - 10} 个文件")
            return True
        
        # 确认删除
        print("\n将删除以下远程文件:")
        for i, file in enumerate(sorted(files_to_delete)[:10]):
            print(f"  - {folder_name}/{file}")
        if len(files_to_delete) > 10:
            print(f"  ... 还有 {len(files_to_delete) - 10} 个文件")
        
        confirm = input("\n确认删除这些文件？ (yes/no): ")
        if confirm.lower() != 'yes':
            print("取消删除操作")
            return True
        
        # 准备删除操作
        delete_operations = []
        for file_path in files_to_delete:
            delete_operations.append(
                CommitOperationDelete(
                    path_in_repo=f"{folder_name}/{file_path}"
                )
            )
        
        # 分批删除（避免一次删除太多）
        batch_size = 100
        total_batches = (len(delete_operations) + batch_size - 1) // batch_size
        
        for i in range(0, len(delete_operations), batch_size):
            batch = delete_operations[i:i+batch_size]
            batch_num = i // batch_size + 1
            
            print(f"\n删除批次 {batch_num}/{total_batches} ({len(batch)} 个文件)...")
            
            try:
                commit_info = self.api.create_commit(
                    repo_id=self.repo_id,
                    repo_type=self.repo_type,
                    operations=batch,
                    commit_message=f"Delete {len(batch)} files from {folder_name} (batch {batch_num}/{total_batches})"
                )
                print(f"✓ 批次 {batch_num} 删除成功")
                time.sleep(1)  # 避免太快
            except Exception as e:
                print(f"✗ 批次 {batch_num} 删除失败: {e}")
                return False
        
        print(f"\n✓ 成功删除 {len(files_to_delete)} 个远程文件")
        return True
    
    def monitor_process_status(self, pid, stop_event):
        """监控进程状态的线程函数"""
        try:
            proc = psutil.Process(pid)
            start_time = time.time()
            
            while not stop_event.is_set():
                try:
                    cpu_percent = proc.cpu_percent(interval=1)
                    mem_info = proc.memory_info()
                    runtime = int(time.time() - start_time)
                    
                    # 输出状态到单独的日志文件
                    with open('upload_status.log', 'a') as f:
                        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - "
                               f"PID: {pid}, CPU: {cpu_percent:.1f}%, "
                               f"内存: {self.format_size(mem_info.rss)}, "
                               f"运行时间: {runtime}秒\n")
                    
                    time.sleep(10)
                except psutil.NoSuchProcess:
                    break
        except Exception as e:
            print(f"监控线程错误: {e}")
    
    def create_streaming_tar_chunks(self, folder_path: Path, dataset_name: str):
        """
        创建流式tar包并分块上传，不占用本地存储
        """
        print(f"\n开始为 {dataset_name} 创建流式 tar 包...")
        print(f"临时tar目录: {self.temp_dir}")
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        remote_files = set(self.api.list_repo_files(repo_id=self.repo_id, repo_type=self.repo_type))
        
        # 收集所有文件
        all_files = []
        total_size = 0
        
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.isfile(file_path):
                    file_size = os.path.getsize(file_path)
                    # 保留完整的相对路径，包括数据集文件夹名
                    rel_path = os.path.relpath(file_path, folder_path.parent)
                    all_files.append((file_path, rel_path, file_size))
                    total_size += file_size
        
        print(f"总文件数: {len(all_files)}")
        print(f"总大小: {self.format_size(total_size)}")
        
        # 计算需要的分块数
        num_chunks = (total_size + self.chunk_size - 1) // self.chunk_size
        print(f"预计分块数: {num_chunks}")
        
        # 分块处理。manifest 记录每个源文件落在哪个 tar part，便于线上定位样本
        # 而不必下载全部分块逐个查找。
        chunk_num = 0
        current_chunk_files = []
        current_chunk_size = 0
        manifest_entries = []
        part_summaries = []
        
        for file_path, rel_path, file_size in all_files:
            # 如果添加这个文件会超过块大小，先上传当前块
            if current_chunk_size + file_size > self.chunk_size and current_chunk_files:
                chunk_num += 1
                # 去掉 _results 后缀用于tar文件命名
                base_name = dataset_name.replace('_results', '')
                tar_filename = f"{base_name}_part{chunk_num:03d}.tar.zst"
                
                print(f"\n上传分块 {chunk_num}/{num_chunks}: {tar_filename}")
                print(f"包含 {len(current_chunk_files)} 个文件, 大小: {self.format_size(current_chunk_size)}")
                part_summaries.append({
                    "part": tar_filename,
                    "file_count": len(current_chunk_files),
                    "total_size": current_chunk_size,
                })
                for chunk_file_path, chunk_rel_path, chunk_file_size in current_chunk_files:
                    manifest_entries.append({
                        "part": tar_filename,
                        "path": chunk_rel_path,
                        "size": chunk_file_size,
                    })
                
                # 创建并上传tar。已存在的远程分块会跳过，便于断点恢复。
                success = self.upload_streaming_tar(
                    current_chunk_files,
                    tar_filename,
                    folder_path,
                    remote_files=remote_files,
                )
                if not success:
                    print(f"分块 {chunk_num} 上传失败")
                    return False
                remote_files.add(tar_filename)
                
                # 重置当前块
                current_chunk_files = []
                current_chunk_size = 0
            
            current_chunk_files.append((file_path, rel_path, file_size))
            current_chunk_size += file_size
        
        # 上传最后一个块
        if current_chunk_files:
            chunk_num += 1
            # 去掉 _results 后缀用于tar文件命名
            base_name = dataset_name.replace('_results', '')
            tar_filename = f"{base_name}_part{chunk_num:03d}.tar.zst"
            
            print(f"\n上传分块 {chunk_num}/{num_chunks}: {tar_filename}")
            print(f"包含 {len(current_chunk_files)} 个文件, 大小: {self.format_size(current_chunk_size)}")
            part_summaries.append({
                "part": tar_filename,
                "file_count": len(current_chunk_files),
                "total_size": current_chunk_size,
            })
            for chunk_file_path, chunk_rel_path, chunk_file_size in current_chunk_files:
                manifest_entries.append({
                    "part": tar_filename,
                    "path": chunk_rel_path,
                    "size": chunk_file_size,
                })
            
            success = self.upload_streaming_tar(
                current_chunk_files,
                tar_filename,
                folder_path,
                remote_files=remote_files,
            )
            if not success:
                print(f"分块 {chunk_num} 上传失败")
                return False
            remote_files.add(tar_filename)

        if not self.upload_part_manifest(dataset_name, folder_path, total_size, manifest_entries, part_summaries):
            return False
        
        print(f"\n✓ {dataset_name} 所有分块上传完成！")
        return True

    def upload_part_manifest(
        self,
        dataset_name: str,
        folder_path: Path,
        total_size: int,
        manifest_entries: List[dict],
        part_summaries: List[dict],
    ) -> bool:
        """Upload small path-to-part indices for later sample lookup."""
        from huggingface_hub import CommitOperationAdd

        base_name = dataset_name.replace('_results', '')
        manifest_path = self.temp_dir / f"{base_name}_parts_manifest.jsonl"
        summary_path = self.temp_dir / f"{base_name}_parts_summary.json"
        try:
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            with open(manifest_path, "w", encoding="utf-8") as f:
                for entry in manifest_entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            summary = {
                "dataset_name": dataset_name,
                "source_path": str(folder_path),
                "repo_id": self.repo_id,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "file_count": len(manifest_entries),
                "total_size": total_size,
                "parts": part_summaries,
            }
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

            self.api.create_commit(
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                operations=[
                    CommitOperationAdd(
                        path_in_repo=manifest_path.name,
                        path_or_fileobj=str(manifest_path),
                    ),
                    CommitOperationAdd(
                        path_in_repo=summary_path.name,
                        path_or_fileobj=str(summary_path),
                    ),
                ],
                commit_message=f"Upload {dataset_name} part manifest",
            )
            print(f"✓ 上传 part manifest: {manifest_path.name}, {summary_path.name}")
            return True
        except Exception as e:
            print(f"✗ 上传 {dataset_name} part manifest 失败: {e}")
            return False
        finally:
            manifest_path.unlink(missing_ok=True)
            summary_path.unlink(missing_ok=True)
    
    def upload_streaming_tar(
        self,
        files: List[Tuple[str, str]],
        tar_filename: str,
        base_path: Path,
        remote_files: Optional[set] = None,
    ) -> bool:
        """
        创建临时tar并上传到HuggingFace。

        早期版本使用 io.BytesIO() 持有 15GB 分块，容易被系统 OOM kill。
        这里把临时tar写到 /common/users/sf895 下，上传结束后删除。
        """
        if remote_files is not None and tar_filename in remote_files:
            print(f"✓ {tar_filename} 已存在于远程仓库，跳过")
            return True

        temp_tar_path: Optional[Path] = None
        try:
            from huggingface_hub import CommitOperationAdd

            self.temp_dir.mkdir(parents=True, exist_ok=True)
            fd, temp_name = tempfile.mkstemp(
                prefix=f"{tar_filename}.",
                suffix=".tmp",
                dir=str(self.temp_dir),
            )
            os.close(fd)
            temp_tar_path = Path(temp_name)

            print(f"正在创建稀疏临时tar: {temp_tar_path}")
            file_list_path = temp_tar_path.with_suffix(temp_tar_path.suffix + ".files")
            try:
                with open(file_list_path, "wb") as file_list:
                    for _, arcname, _ in files:
                        file_list.write(os.fsencode(arcname))
                        file_list.write(b"\0")

                subprocess.run(
                    [
                        "tar",
                        "--sparse",
                        "--zstd",
                        "--null",
                        "--files-from",
                        str(file_list_path),
                        "--create",
                        "--file",
                        str(temp_tar_path),
                    ],
                    cwd=str(base_path.parent),
                    check=True,
                )
            finally:
                file_list_path.unlink(missing_ok=True)

            apparent_size = temp_tar_path.stat().st_size
            allocated_size = temp_tar_path.stat().st_blocks * 512
            print(
                "临时tar大小: "
                f"logical={self.format_size(apparent_size)}, "
                f"allocated={self.format_size(allocated_size)}"
            )
            
            # 创建上传操作
            operations = [
                CommitOperationAdd(
                    path_in_repo=tar_filename,
                    path_or_fileobj=str(temp_tar_path)
                )
            ]
            
            # 执行上传
            print(f"正在上传 {tar_filename}...")
            
            commit_info = self.api.create_commit(
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                operations=operations,
                commit_message=f"Upload {tar_filename}"
            )
            
            print(f"✓ {tar_filename} 上传成功")
            return True
            
        except Exception as e:
            print(f"✗ 上传 {tar_filename} 失败: {e}")
            return False
        finally:
            if temp_tar_path is not None and temp_tar_path.exists():
                try:
                    temp_tar_path.unlink()
                    print(f"✓ 已删除临时tar: {temp_tar_path}")
                except Exception as cleanup_error:
                    print(f"警告: 删除临时tar失败 {temp_tar_path}: {cleanup_error}")

    def upload_large_folder_in_batches(self, folder_path, folder_name, folder_hash, progress):
        """分批上传大文件夹 - 直接上传，达到20k文件时提交"""
        # 检查是否有未完成的文件
        upload_progress_key = f"{folder_hash}_files"
        uploaded_files = set(progress.get(upload_progress_key, {}).get('uploaded_files', []))
        
        # 从 HuggingFace 获取已存在的文件
        print("正在检查 HuggingFace 上的已有文件...")
        remote_files = self.get_remote_files(folder_name)
        print(f"远程已有文件数: {len(remote_files)}")
        
        # 合并本地记录和远程文件
        uploaded_files.update(remote_files)
        
        # 收集所有文件
        print("正在扫描本地文件...")
        all_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                rel_path = os.path.relpath(os.path.join(root, file), folder_path)
                if rel_path not in uploaded_files:
                    all_files.append(rel_path)
        
        remaining_files = len(all_files)
        total_uploaded = len(uploaded_files)
        total_files = remaining_files + total_uploaded
        
        print(f"文件总数: {total_files}")
        print(f"已上传: {total_uploaded}")
        print(f"待上传: {remaining_files}")
        
        if remaining_files == 0:
            print(f"✓ {folder_name} 所有文件已上传")
            return True
        
        # 分批上传 - 减小批次大小以确保成功
        batch_size = 5000  # 从20000减少到5000，更安全
        current_batch = []
        batch_num = (total_uploaded // batch_size) + 1
        
        print(f"\n将分 {(remaining_files + batch_size - 1) // batch_size} 批上传，每批最多 {batch_size} 个文件")
        
        for i, rel_path in enumerate(all_files):
            current_batch.append(rel_path)
            
            # 达到批次大小或是最后一个文件时，执行上传
            if len(current_batch) >= batch_size or i == len(all_files) - 1:
                print(f"\n{'='*50}")
                print(f"上传批次 {batch_num} - 文件数: {len(current_batch)}")
                print(f"{'='*50}")
                
                # 直接上传当前批次
                success = self.upload_file_batch_direct(
                    folder_path, folder_name, current_batch, 
                    batch_num, uploaded_files
                )
                
                if success:
                    # 更新已上传文件列表
                    uploaded_files.update(current_batch)
                    
                    # 保存进度
                    if upload_progress_key not in progress:
                        progress[upload_progress_key] = {}
                    progress[upload_progress_key]['uploaded_files'] = list(uploaded_files)
                    progress[upload_progress_key]['last_update'] = time.strftime('%Y-%m-%d %H:%M:%S')
                    progress[upload_progress_key]['total_uploaded'] = len(uploaded_files)
                    self.save_progress(progress)
                    
                    print(f"✓ 批次 {batch_num} 上传成功 - 总进度: {len(uploaded_files)}/{total_files}")
                    
                    # 重置当前批次
                    current_batch = []
                    batch_num += 1
                else:
                    print(f"✗ 批次 {batch_num} 上传失败")
                    return False
        
        # 所有文件上传完成
        progress[folder_hash] = {
            'folder': folder_name,
            'path': str(folder_path),
            'completed': True,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_files': total_files
        }
        self.save_progress(progress)
        
        print(f"\n✓ {folder_name} 所有文件上传完成！")
        return True
    
    def upload_file_batch_direct(self, folder_path, folder_name, file_list, batch_num, uploaded_files):
        """直接上传一批文件到HuggingFace"""
        try:
            from huggingface_hub import CommitOperationAdd
            
            print(f"准备上传 {len(file_list)} 个文件...")
            
            # 准备上传操作
            operations = []
            for i, rel_path in enumerate(file_list):
                if (i + 1) % 1000 == 0:
                    print(f"  准备文件 {i + 1}/{len(file_list)}...")
                
                local_path = os.path.join(folder_path, rel_path)
                path_in_repo = f"{folder_name}/{rel_path}"
                
                # 创建上传操作
                operations.append(
                    CommitOperationAdd(
                        path_in_repo=path_in_repo,
                        path_or_fileobj=local_path
                    )
                )
            
            # 创建提交
            print(f"\n开始上传批次 {batch_num} ({len(operations)} 个文件)...")
            
            commit_message = f"Upload {folder_name} batch {batch_num} - {len(operations)} files"
            
            # 执行上传，带重试机制
            max_retries = 3
            for retry in range(max_retries):
                try:
                    print(f"正在提交批次 {batch_num}... (尝试 {retry + 1}/{max_retries})")
                    
                    commit_info = self.api.create_commit(
                        repo_id=self.repo_id,
                        repo_type=self.repo_type,
                        operations=operations,
                        commit_message=commit_message
                    )
                    
                    print(f"✓ 批次 {batch_num} 上传成功")
                    print(f"  Commit URL: {commit_info.commit_url}")
                    
                    # 成功后等待一下，避免太快
                    time.sleep(2)
                    
                    return True
                    
                except Exception as e:
                    if retry < max_retries - 1:
                        wait_time = (retry + 1) * 10
                        print(f"批次 {batch_num} 提交失败: {e}")
                        print(f"等待 {wait_time} 秒后重试...")
                        time.sleep(wait_time)
                    else:
                        raise
            
        except Exception as e:
            print(f"\n批次 {batch_num} 上传失败: {e}")
            return False

    
    def upload_folder_with_progress(self, folder_path, folder_name, dry_run=False, sync_delete=False):
        """Upload a single folder with better progress monitoring"""
        print(f"\n{'='*60}")
        print(f"正在处理: {folder_name}")
        print(f"同步删除: {'启用' if sync_delete else '禁用'}")
        print(f"{'='*60}")
        
        # Resolve symlinks to get the actual path
        actual_path = Path(folder_path).resolve()
        print(f"源路径: {folder_path}")
        print(f"实际路径: {actual_path}")
        
        # Get folder information
        total_size, file_count = self.get_folder_info(folder_path)
        print(f"文件夹大小: {self.format_size(total_size)}")
        print(f"文件总数: {file_count}")
        
        if dry_run:
            print(f"[模拟运行] 将会上传 {folder_name} 到 {self.repo_id}")
            return True
        
        # Check if folder was already uploaded
        progress = self.load_progress()
        folder_hash = hashlib.md5(str(folder_path).encode()).hexdigest()
        
        # 检查是否需要分批上传
        if file_count > 20000:
            print(f"\n⚠️ 检测到文件数量 ({file_count}) 超过 20k，将自动分批上传")
            return self.upload_large_folder_in_batches(folder_path, folder_name, folder_hash, progress)
        
        # 检查远程是否已存在完整文件夹
        print("正在检查远程文件...")
        remote_files = self.get_remote_files(folder_name)
        
        if folder_hash in progress and progress[folder_hash].get('completed', False):
            print(f"✓ {folder_name} 已经上传过（本地记录），跳过...")
            return True
        
        # 计算本地文件数
        local_files = set()
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                rel_path = os.path.relpath(os.path.join(root, file), folder_path)
                local_files.add(rel_path)
        
        # 如果启用了同步删除，先处理删除
        if sync_delete and len(remote_files) > 0:
            print("\n检查需要删除的远程文件...")
            if not self.sync_delete_remote_files(folder_name, local_files, remote_files, dry_run):
                print("删除远程文件失败，中止上传")
                return False
        
        # 检查是否所有文件都已在远程
        missing_files = local_files - remote_files
        if len(missing_files) == 0 and len(local_files) > 0:
            print(f"✓ {folder_name} 所有文件都已在远程存在，跳过上传...")
            # 更新本地进度记录
            progress[folder_hash] = {
                'folder': folder_name,
                'path': str(folder_path),
                'completed': True,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'remote_verified': True
            }
            self.save_progress(progress)
            return True
        elif len(remote_files) > 0:
            print(f"远程已有 {len(remote_files)} 个文件，缺少 {len(missing_files)} 个文件")
        
        # Start upload using CLI command
        print(f"\n开始上传到 {self.repo_id}/{folder_name}...")
        print("大文件夹可能需要较长时间，进度将显示在下方。\n")
        print("提示: 可以查看 upload_status.log 文件了解详细进度")
        print("使用命令: tail -f upload_status.log")
        
        try:
            # 使用环境变量强制显示进度
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            env['FORCE_COLOR'] = '1'
            
            # Use huggingface-cli upload command with path_in_repo
            cmd = [
                "huggingface-cli", "upload",
                self.repo_id,  # Repository ID
                str(actual_path),  # Local path 
                folder_name,  # Path in repo (destination folder)
                "--repo-type", self.repo_type,
                "--commit-message", f"Upload {folder_name}"
            ]
            
            # 使用pty来捕获完整的终端输出
            master_fd, slave_fd = pty.openpty()
            
            # Run the command with PTY
            process = subprocess.Popen(
                cmd,
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                env=env
            )
            
            # 关闭slave端
            os.close(slave_fd)
            
            # 启动监控线程
            stop_event = threading.Event()
            monitor_thread = threading.Thread(
                target=self.monitor_process_status,
                args=(process.pid, stop_event)
            )
            monitor_thread.start()
            
            # 读取输出
            output_buffer = ""
            while True:
                try:
                    # 检查是否有数据可读
                    r, _, _ = select.select([master_fd], [], [], 0.1)
                    if r:
                        data = os.read(master_fd, 1024).decode('utf-8', errors='replace')
                        if not data:
                            break
                        
                        # 打印并保存输出
                        sys.stdout.write(data)
                        sys.stdout.flush()
                        output_buffer += data
                        
                        # 同时保存到日志文件
                        with open('upload_detail.log', 'a') as f:
                            f.write(data)
                    
                    # 检查进程是否结束
                    if process.poll() is not None:
                        break
                        
                except Exception as e:
                    print(f"\n读取输出时出错: {e}")
                    break
            
            # 停止监控线程
            stop_event.set()
            monitor_thread.join()
            
            # 关闭master端
            os.close(master_fd)
            
            # Wait for process to complete
            return_code = process.wait()
            
            if return_code == 0:
                # Mark as completed
                progress[folder_hash] = {
                    'folder': folder_name,
                    'path': str(folder_path),
                    'completed': True,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                self.save_progress(progress)
                
                print(f"\n✓ 成功上传 {folder_name}")
                return True
            else:
                raise Exception(f"上传命令失败，返回码: {return_code}")
            
        except Exception as e:
            print(f"\n✗ 上传 {folder_name} 时出错: {str(e)}")
            
            # Save partial progress
            progress[folder_hash] = {
                'folder': folder_name,
                'path': str(folder_path),
                'completed': False,
                'error': str(e),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            self.save_progress(progress)
            return False
    
    def upload_all_folders(self, folders, dry_run=False, sync_delete=False):
        """Upload multiple folders with overall progress"""
        print("\n" + "="*60)
        print(f"HUGGING FACE 大数据集上传工具")
        print(f"仓库: {self.repo_id}")
        print(f"待处理文件夹数: {len(folders)}")
        print(f"运行模式: {'模拟运行' if dry_run else '实际执行'}")
        print(f"同步删除: {'启用' if sync_delete else '禁用'}")
        print("="*60)
        
        if sync_delete:
            print("\n⚠️ 警告: 同步删除已启用！")
            print("这将删除远程存在但本地不存在的文件。")
            print("请确保您了解这个操作的后果。\n")
        
        success_count = 0
        
        for i, (folder_path, folder_name) in enumerate(folders, 1):
            print(f"\n[{i}/{len(folders)}] 处理 {folder_name}...")
            
            if self.upload_folder_with_progress(folder_path, folder_name, dry_run, sync_delete):
                success_count += 1
            else:
                print(f"处理 {folder_name} 失败。可以稍后重试。")
        
        # Summary
        print("\n" + "="*60)
        print("上传总结")
        print(f"文件夹总数: {len(folders)}")
        print(f"成功上传: {success_count}")
        print(f"失败: {len(folders) - success_count}")
        print("="*60)
        
        if not dry_run and success_count == len(folders):
            print("\n✓ 所有文件夹上传成功！")
            # Clean up progress file
            if os.path.exists(self.progress_file):
                os.remove(self.progress_file)
        elif not dry_run:
            print("\n⚠ 部分上传失败。再次运行脚本可以重试。")

def print_usage():
    """打印使用说明"""
    print("""
使用方法:
1. 直接运行（自动扫描并上传所有数据集）:
   python pipeline33_upload_to_huggingface.py

2. 指定特定数据集上传:
   python pipeline33_upload_to_huggingface.py AVIS_test
   python pipeline33_upload_to_huggingface.py AVIS_val,AVIS_test

功能特性:
- 自动扫描 ../output 目录下的所有 *_results 文件夹
- 为每个数据集创建独立的 HuggingFace 仓库
- 使用流式 tar 打包，不占用本地存储空间
- 自动分块，每块最大 15GB
- 如果仓库不存在会自动创建（MIT 许可证）
- 支持断点续传

注意: 
- 修改代码中的 dry_run = False 来实际执行上传
- 修改代码中的 use_tar = True/False 来选择打包或原始文件上传
- 建议先用 dry_run = True 模式查看将要上传的内容
""")

def show_upload_tips():
    """显示上传提示信息"""
    print("\n" + "="*60)
    print("上传监控提示:")
    print("1. 查看实时进度: tail -f upload_detail.log")
    print("2. 查看系统状态: tail -f upload_status.log") 
    print("3. 查看上传记录: cat upload_progress.json")
    print("4. 运行监控面板: bash ../tools/upload_monitor/monitor_upload.sh")
    print("5. 持续监控: watch -n 10 bash ../tools/upload_monitor/monitor_upload.sh")
    print("="*60 + "\n")

def scan_output_directory(base_path: Path, allowed_names: Optional[set] = None) -> List[Tuple[Path, str]]:
    """扫描output目录下的所有数据集"""
    datasets = []
    
    print(f"扫描目录: {base_path}")
    
    # 扫描所有_results结尾的文件夹
    for item in base_path.iterdir():
        if item.is_dir() and (
            item.name.endswith('_results')
            or (allowed_names is not None and item.name in allowed_names)
        ):
            if allowed_names is not None and item.name not in allowed_names:
                continue
            dataset_name = item.name  # 保留完整名称，包含 _results 后缀
            datasets.append((item, dataset_name))
    
    # 按名称排序
    datasets.sort(key=lambda x: x[1])
    
    print(f"发现 {len(datasets)} 个数据集:")
    for path, name in datasets:
        size, count = get_folder_info_static(path)
        print(f"  - {name}: {count} 个文件, {format_size_static(size)}")
    
    return datasets

def get_folder_info_static(folder_path):
    """静态函数版本的 get_folder_info"""
    total_size = 0
    file_count = 0
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.isfile(file_path):
                total_size += os.path.getsize(file_path)
                file_count += 1
                
    return total_size, file_count

def format_size_static(size):
    """静态函数版本的 format_size"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "datasets",
        nargs="?",
        help="Comma-separated legacy dataset names",
    )
    parser.add_argument(
        "--raw-pt",
        action="store_true",
        help="Upload modality-scoped PT shards instead of tar archives",
    )
    parser.add_argument(
        "--repo-id",
        default="World-Snapshot/WSM_v1_Latent_PT_WSG_v1.8_for_Flux2",
        help="Destination Hugging Face dataset repository for --raw-pt",
    )
    parser.add_argument(
        "--cache-name",
        default="wsg_flux2_frame_latents_dense_multidataset",
        help="Cache directory under WSM_v1/data used by --raw-pt",
    )
    parser.add_argument(
        "--modalities",
        default="all",
        help="Comma-separated modality directory names, or all",
    )
    parser.add_argument(
        "--force-replace",
        action="store_true",
        help="Replace every selected PT shard even when its manifest hash matches",
    )
    parser.add_argument(
        "--include-training-support",
        action="store_true",
        help="Also upload XPose, caption embeddings, reference frames, and cache metadata",
    )
    parser.add_argument(
        "--support-only",
        action="store_true",
        help="Skip the primary dense shards and upload only training support caches",
    )
    parser.add_argument(
        "--include-sparse",
        action="store_true",
        help="Also upload sparse datasets, derived caches, and audio semantic PT latents",
    )
    parser.add_argument(
        "--include-custom-pose-support",
        action="store_true",
        help="Also upload MP-100 prompts, metadata, and text-embedding shards",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    """主函数，处理命令行参数"""
    args = parse_args()

    if args.raw_pt:
        data_root = Path(__file__).resolve().parents[2] / "WSMs/WSM_v1/data"
        cache_path = data_root / args.cache_name
        modalities = [item.strip() for item in args.modalities.split(",") if item.strip()]
        if not modalities:
            raise SystemExit("--modalities must not be empty")
        uploader = HuggingFaceLargeUploader(args.repo_id, repo_type="dataset")
        if not args.dry_run:
            uploader.ensure_repo_exists()
        success = True
        if not args.support_only:
            success = uploader.upload_raw_pt_modalities(
                cache_path=cache_path,
                cache_name=args.cache_name,
                modalities=modalities,
                force_replace=args.force_replace,
                dry_run=args.dry_run,
            )

        if args.include_training_support or args.support_only:
            xpose_name = "xpose_flux2_frame_latents_sample10pct_2frames"
            xpose_path = data_root / xpose_name
            success = uploader.upload_raw_pt_modalities(
                cache_path=xpose_path,
                cache_name=xpose_name,
                modalities=["all"],
                force_replace=args.force_replace,
                dry_run=args.dry_run,
            ) and success
            success = uploader.upload_raw_file_group(
                root_path=xpose_path,
                repo_root=xpose_name,
                files=[xpose_path / "prompts.pt"],
                manifest_name="root_pt",
                force_replace=args.force_replace,
                dry_run=args.dry_run,
            ) and success

            caption_name = "wsg_flux2_caption_embeddings_all_datasets"
            caption_path = data_root / caption_name
            success = uploader.upload_raw_file_group(
                root_path=caption_path,
                repo_root=caption_name,
                files=list(caption_path.glob("*.pt")),
                manifest_name="caption_shards",
                force_replace=args.force_replace,
                dry_run=args.dry_run,
            ) and success

            reference_dir = cache_path / "reference_frames"
            success = uploader.upload_raw_file_group(
                root_path=cache_path,
                repo_root=args.cache_name,
                files=list(reference_dir.glob("*.pt")),
                manifest_name="reference_frames",
                force_replace=args.force_replace,
                dry_run=args.dry_run,
            ) and success
            metadata_files = [
                path for path in cache_path.iterdir()
                if path.is_file() and path.suffix in {".json", ".yaml", ".yml"}
            ]
            success = uploader.upload_raw_file_group(
                root_path=cache_path,
                repo_root=args.cache_name,
                files=metadata_files,
                manifest_name="metadata",
                force_replace=args.force_replace,
                dry_run=args.dry_run,
            ) and success

        if args.include_sparse:
            sparse_path = data_root / "sparse_structure"
            sparse_groups = {
                "datasets": list((sparse_path / "datasets").rglob("*.npz")),
                "derived_cache": list((sparse_path / "cache").rglob("*.npz")),
                "audio_semantic_latents": [
                    path for path in
                    (sparse_path / "audio_semantic_overlay_flux2_latents").rglob("*")
                    if path.is_file()
                ],
            }
            for group_name, group_files in sparse_groups.items():
                success = uploader.upload_raw_file_group(
                    root_path=sparse_path,
                    repo_root="sparse_structure",
                    files=group_files,
                    manifest_name=group_name,
                    force_replace=args.force_replace,
                    dry_run=args.dry_run,
                ) and success

        if args.include_custom_pose_support:
            pose_cache_name = "mp100_universal_pose_flux2_instance_crop_latents"
            pose_cache_path = data_root / pose_cache_name
            success = uploader.upload_raw_file_group(
                root_path=pose_cache_path,
                repo_root=pose_cache_name,
                files=[
                    pose_cache_path / "prompts.pt",
                    pose_cache_path / "dataset_index.json",
                ],
                manifest_name="training_metadata",
                force_replace=args.force_replace,
                dry_run=args.dry_run,
            ) and success

            pose_text_name = "mp100_universal_pose_flux2_text_embeddings"
            pose_text_path = data_root / pose_text_name
            success = uploader.upload_raw_file_group(
                root_path=pose_text_path,
                repo_root=pose_text_name,
                files=list(pose_text_path.glob("*.pt")),
                manifest_name="text_embedding_shards",
                force_replace=args.force_replace,
                dry_run=args.dry_run,
            ) and success

        raise SystemExit(0 if success else 1)

    # 设置基础路径
    base_path = Path("../output")
    
    # 配置要上传的数据集白名单（用户可以在这里修改）
    # 只有在这个列表中的数据集才会被上传
    # 注意：这里使用的是完整的文件夹名称，包含 "_results" 后缀
    upload_whitelist = [
        "AVIS_test_results",
        "AVIS_train_results", 
        "AVIS_val_results",
        "PoseAnything_MP100",
    ]

    repo_name_map = {
        "AVIS_train_results": "WSV_AVIS_train_results_v1.6",
        "AVIS_val_results": "WSV_AVIS_val_results_v1.6",
        "AVIS_test_results": "WSV_AVIS_test_results_v1.6",
        "PoseAnything_MP100": "PoseAnything_MP100",
        "WSM_v1_Latent_WSG_v1.6_for_Flux2": "WSM_v1_Latent_WSG_v1.6_for_Flux2",
    }
    
    # 扫描output目录下的白名单数据集。避免在无关的大型结果目录上花费很久
    # 统计大小，尤其是这些目录经常是指向公共存储的 symlink。
    requested_names = (
        {name.strip() for name in args.datasets.split(",") if name.strip()}
        if args.datasets
        else set(upload_whitelist)
    )
    available_datasets = scan_output_directory(base_path, allowed_names=requested_names)
    latent_dataset_name = "WSM_v1_Latent_WSG_v1.6_for_Flux2"
    latent_data_path = Path("../../WSMs/WSM_v1/data")
    if latent_data_path.exists():
        available_datasets.append((latent_data_path, latent_dataset_name))
    
    if not available_datasets:
        print("错误: 在 ../output 目录下未找到任何数据集")
        sys.exit(1)
    
    # 过滤白名单中的数据集
    filtered_datasets = []
    for path, name in available_datasets:
        if name in upload_whitelist:
            filtered_datasets.append((path, name))
    
    print(f"\n根据白名单过滤后，将上传 {len(filtered_datasets)} 个数据集:")
    for path, name in filtered_datasets:
        print(f"  ✓ {name}")
    
    # 显示被排除的数据集
    excluded = []
    for path, name in available_datasets:
        if name not in upload_whitelist:
            excluded.append(name)
    
    if excluded:
        print(f"\n以下数据集不在白名单中，将被跳过:")
        for name in excluded:
            print(f"  ✗ {name}")
    
    # 配置运行模式
    dry_run = args.dry_run
    sync_delete = False  # True: 删除远程存在但本地不存在的文件，False: 只上传不删除
    use_tar = True  # True: 使用流式tar打包上传，False: 使用原始文件上传
    
    # 解析命令行参数（可选）
    if args.datasets:
        # 支持命令行覆盖数据集名称
        dataset_names = args.datasets.split(',')
        # 过滤出指定的数据集
        folders = []
        for path, name in available_datasets:
            if name in dataset_names:
                folders.append((path, name))
        if not folders:
            print(f"错误: 未找到指定的数据集: {', '.join(dataset_names)}")
            sys.exit(1)
    else:
        # 使用白名单过滤后的数据集
        folders = filtered_datasets
        # Upload the cached WSV latents after the three AVIS splits.
        folders.append((latent_data_path, latent_dataset_name))
        
    if not folders:
        print("\n错误: 没有要上传的数据集")
        sys.exit(1)
    
    # The streaming-tar path uses the Python huggingface_hub API directly and
    # does not require huggingface-cli. Only the raw-folder upload fallback
    # shells out to the CLI.
    if not use_tar:
        try:
            subprocess.run(["huggingface-cli", "version"], capture_output=True, check=True)
        except:
            print("错误: huggingface-cli 未找到。请安装:")
            print("  pip install -U 'huggingface_hub[cli]'")
            sys.exit(1)
    
    # 显示上传提示
    show_upload_tips()
    
    # 处理每个数据集
    for folder_path, dataset_name in folders:
        print(f"\n{'='*80}")
        print(f"处理数据集: {dataset_name}")
        print(f"{'='*80}")
        
        # 为每个数据集创建独立的仓库
        repo_id = f"World-Snapshot/{repo_name_map.get(dataset_name, dataset_name)}"
        repo_type = "dataset"
        
        # 创建上传器
        uploader = HuggingFaceLargeUploader(repo_id, repo_type)
        
        try:
            # 确保仓库存在
            uploader.ensure_repo_exists()
            
            if use_tar:
                # 使用流式tar打包上传
                success = uploader.create_streaming_tar_chunks(folder_path, dataset_name)
                if success:
                    print(f"\n✓ {dataset_name} 上传成功！")
                else:
                    print(f"\n✗ {dataset_name} 上传失败！")
            else:
                # 使用原始文件上传方式
                uploader.upload_all_folders([(folder_path, dataset_name)], dry_run, sync_delete)
                
        except KeyboardInterrupt:
            print("\n\n⚠ 上传已中断。进度已保存。")
            print("再次运行脚本即可继续上传。")
            sys.exit(0)
        except Exception as e:
            print(f"\n✗ 处理 {dataset_name} 时出错: {str(e)}")
            continue
    
    print("\n" + "="*80)
    print("所有数据集处理完成！")
    print("="*80)

if __name__ == '__main__':
    main()
