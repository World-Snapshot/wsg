#!/usr/bin/env python3
"""
OpenVidHD 高速下载和提取工具
使用 huggingface_hub 的 snapshot_download 实现高速下载
"""

import os
import zipfile
import pandas as pd
import json
import time
import pickle
from datetime import datetime
from tqdm.auto import tqdm  # 使用 tqdm.auto 而不是直接 tqdm
from huggingface_hub import snapshot_download, hf_hub_download
import shutil

class ProgressManager:
    """进度管理器"""
    def __init__(self, progress_file="download_progress_fast.pkl"):
        self.progress_file = progress_file
        self.progress = {
            'completed_parts': set(),
            'extracted_videos': set(),
            'failed_parts': set(),
            'total_extracted': 0,
            'start_time': None,
            'last_update': None
        }
        self.load_progress()
    
    def load_progress(self):
        """加载之前的进度"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'rb') as f:
                    self.progress = pickle.load(f)
                print(f"已加载之前的进度:")
                print(f"  已完成部分: {len(self.progress['completed_parts'])} 个")
                print(f"  已提取视频: {len(self.progress['extracted_videos'])} 个")
                print(f"  失败部分: {len(self.progress['failed_parts'])} 个")
                if self.progress['start_time']:
                    print(f"  开始时间: {self.progress['start_time']}")
            except Exception as e:
                print(f"加载进度文件失败: {e}，将重新开始")
                self.reset_progress()
        else:
            self.reset_progress()
    
    def save_progress(self):
        """保存当前进度"""
        self.progress['last_update'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(self.progress_file, 'wb') as f:
                pickle.dump(self.progress, f)
        except Exception as e:
            print(f"保存进度失败: {e}")
    
    def reset_progress(self):
        """重置进度"""
        self.progress = {
            'completed_parts': set(),
            'extracted_videos': set(),
            'failed_parts': set(),
            'total_extracted': 0,
            'start_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'last_update': None
        }
        self.save_progress()
    
    def mark_part_completed(self, part_name):
        """标记part为已完成"""
        self.progress['completed_parts'].add(part_name)
        self.save_progress()
    
    def mark_part_failed(self, part_name):
        """标记part为失败"""
        self.progress['failed_parts'].add(part_name)
        self.save_progress()
    
    def mark_video_extracted(self, video_name):
        """标记视频为已提取"""
        self.progress['extracted_videos'].add(video_name)
        self.progress['total_extracted'] = len(self.progress['extracted_videos'])
        self.save_progress()
    
    def is_part_completed(self, part_name):
        """检查part是否已完成"""
        return part_name in self.progress['completed_parts']
    
    def is_video_extracted(self, video_name):
        """检查视频是否已提取"""
        return video_name in self.progress['extracted_videos']
    
    def get_remaining_parts(self, all_parts):
        """获取剩余需要处理的parts"""
        return [p for p in all_parts if not self.is_part_completed(p)]
    
    def print_summary(self):
        """打印进度摘要"""
        print(f"\n=== 进度摘要 ===")
        print(f"已完成部分: {len(self.progress['completed_parts'])}")
        print(f"已提取视频: {len(self.progress['extracted_videos'])}")
        print(f"失败部分: {len(self.progress['failed_parts'])}")
        if self.progress['failed_parts']:
            print(f"失败的部分: {list(self.progress['failed_parts'])}")

def download_with_hf_hub(part_name, cache_dir="./hf_cache"):
    """使用 huggingface_hub 高速下载"""
    part_num = int(part_name.replace('part', ''))
    print(f"开始高速下载: OpenVidHD_part_{part_num}.zip")
    
    try:
        # 对于分割的文件，需要下载多个部分
        part_num = int(part_name.replace('part', ''))
        
        if part_num <= 14:  # part1-14 是完整文件
            # 下载单个文件
            file_path = hf_hub_download(
                repo_id="nkp37/OpenVid-1M",
                filename=f"OpenVidHD/OpenVidHD_part_{part_num}.zip",
                repo_type="dataset",
                cache_dir=cache_dir,
                resume_download=True,
                local_dir_use_symlinks=False
            )
            return [file_path]
        else:  # part15+ 是分割文件
            # 下载分割的部分
            downloaded_parts = []
            part_suffixes = ['_part_aa', '_part_ab']
            
            for suffix in part_suffixes:
                filename = f"OpenVidHD/OpenVidHD_part_{part_num}{suffix}"
                print(f"  下载分片: {filename}")
                
                file_path = hf_hub_download(
                    repo_id="nkp37/OpenVid-1M",
                    filename=filename,
                    repo_type="dataset",
                    cache_dir=cache_dir,
                    resume_download=True,
                    local_dir_use_symlinks=False
                )
                downloaded_parts.append(file_path)
            
            return downloaded_parts
    
    except Exception as e:
        print(f"下载失败: {e}")
        return None

def merge_split_files(part_files, output_path):
    """合并分割的文件"""
    print(f"合并文件到: {output_path}")
    
    try:
        with open(output_path, 'wb') as outfile:
            for part_path in part_files:
                print(f"  合并: {os.path.basename(part_path)}")
                with open(part_path, 'rb') as infile:
                    shutil.copyfileobj(infile, outfile)
        return True
    except Exception as e:
        print(f"合并失败: {e}")
        return False

def extract_videos_from_zip(zip_path, target_videos, output_dir, progress_manager):
    """从ZIP文件中直接提取指定的视频文件"""
    extracted_count = 0
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # 获取ZIP中的所有文件名
            zip_files = set(zip_ref.namelist())
            
            # 创建进度条
            videos_to_extract = [v for v in target_videos if not progress_manager.is_video_extracted(v)]
            
            with tqdm.auto.tqdm(total=len(videos_to_extract), desc="提取视频") as pbar:
                for target_video in videos_to_extract:
                    if target_video in zip_files:
                        # 直接提取到目标目录
                        zip_ref.extract(target_video, output_dir)
                        
                        # 如果提取的文件在子文件夹中，移动到根目录
                        extracted_path = os.path.join(output_dir, target_video)
                        final_path = os.path.join(output_dir, os.path.basename(target_video))
                        
                        if os.path.dirname(target_video):  # 如果有子目录
                            if os.path.exists(extracted_path):
                                os.rename(extracted_path, final_path)
                                # 删除空的子目录
                                sub_dir = os.path.join(output_dir, os.path.dirname(target_video))
                                if os.path.exists(sub_dir) and not os.listdir(sub_dir):
                                    os.rmdir(sub_dir)
                        
                        # 验证文件是否成功提取
                        if os.path.exists(final_path) and os.path.getsize(final_path) > 0:
                            progress_manager.mark_video_extracted(target_video)
                            extracted_count += 1
                    
                    pbar.update(1)
    
    except zipfile.BadZipFile:
        print(f"错误: {zip_path} 不是有效的ZIP文件")
    except Exception as e:
        print(f"提取时出错: {str(e)}")
    
    return extracted_count

def get_videos_for_part(part_name, filtered_videos):
    """获取指定part中需要的视频列表"""
    with open('OpenVidHD.json', 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    for item in json_data:
        if isinstance(item, dict) and part_name in item:
            part_videos = set(item[part_name])
            needed_videos = filtered_videos.intersection(part_videos)
            return list(needed_videos)
    
    return []

def process_part(part_name, filtered_videos, output_dir, progress_manager):
    """处理单个part"""
    # 获取该part需要的视频
    needed_videos = get_videos_for_part(part_name, filtered_videos)
    
    if not needed_videos:
        print(f"Part {part_name} 中没有需要的视频，跳过")
        progress_manager.mark_part_completed(part_name)
        return 0
    
    print(f"\n=== 处理 {part_name} ===")
    print(f"需要提取 {len(needed_videos)} 个视频")
    
    # 确保临时目录存在
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # 使用 huggingface_hub 下载
        downloaded_files = download_with_hf_hub(part_name)
        
        if not downloaded_files:
            progress_manager.mark_part_failed(part_name)
            return 0
        
        # 确定最终的ZIP文件路径
        if len(downloaded_files) == 1:
            # 单个文件，直接使用
            zip_path = downloaded_files[0]
        else:
            # 多个文件，需要合并
            zip_path = os.path.join(temp_dir, f"OpenVidHD_{part_name}.zip")
            if not merge_split_files(downloaded_files, zip_path):
                progress_manager.mark_part_failed(part_name)
                return 0
        
        # 从ZIP中提取需要的视频
        print(f"从ZIP中提取视频...")
        extracted_count = extract_videos_from_zip(zip_path, needed_videos, output_dir, progress_manager)
        
        print(f"成功提取 {extracted_count} 个视频")
        progress_manager.mark_part_completed(part_name)
        
        # 如果是合并的文件，删除它以节省空间
        if len(downloaded_files) > 1 and os.path.exists(zip_path):
            os.remove(zip_path)
            print(f"已删除临时合并文件")
        
        # 清理 HF 缓存中的原始下载文件
        for file_path in downloaded_files:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"已删除缓存文件: {os.path.basename(file_path)}")
                except Exception as e:
                    print(f"删除缓存文件失败: {e}")
        
        # 尝试清理空的缓存目录
        try:
            cache_dir = "./hf_cache"
            if os.path.exists(cache_dir):
                # 递归删除空目录
                for root, dirs, files in os.walk(cache_dir, topdown=False):
                    if not files and not dirs:
                        os.rmdir(root)
        except Exception as e:
            print(f"清理缓存目录失败: {e}")
        
        return extracted_count
        
    except Exception as e:
        print(f"处理 {part_name} 时出错: {str(e)}")
        progress_manager.mark_part_failed(part_name)
        return 0

def main():
    """主函数"""
    print("OpenVidHD 高速下载提取工具")
    print("=" * 50)
    
    # 初始化进度管理器
    progress_manager = ProgressManager()
    
    # 读取筛选后的CSV
    filtered_csv = "filtered_OpenVidHD.csv"
    if not os.path.exists(filtered_csv):
        print(f"错误: 找不到文件 {filtered_csv}")
        print("请先运行筛选脚本生成此文件")
        return
    
    df = pd.read_csv(filtered_csv)
    filtered_videos = set(df['video'].tolist())
    print(f"总共需要提取 {len(filtered_videos)} 个视频")
    
    # 创建输出目录
    output_dir = "./OpenVidHD-camera"
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")
    
    # 获取part分布信息
    if os.path.exists('part_distribution_analysis.csv'):
        part_df = pd.read_csv('part_distribution_analysis.csv')
        all_parts = part_df[part_df['matching_videos'] > 0]['part'].tolist()
    else:
        # 如果没有分布分析文件，处理所有part
        all_parts = [f"part{i}" for i in range(1, 99)]
    
    # 获取剩余需要处理的parts
    remaining_parts = progress_manager.get_remaining_parts(all_parts)
    
    print(f"总共需要处理 {len(all_parts)} 个part")
    print(f"剩余需要处理 {len(remaining_parts)} 个part")
    
    if not remaining_parts:
        print("所有part都已处理完成！")
        progress_manager.print_summary()
        return
    
    # 显示当前进度
    progress_manager.print_summary()
    
    total_extracted = progress_manager.progress['total_extracted']
    processed_parts = len(progress_manager.progress['completed_parts'])
    
    # 开始处理
    start_time = time.time()
    
    for i, part_name in enumerate(remaining_parts):
        print(f"\n进度: {processed_parts + 1}/{len(all_parts)} - 已提取 {total_extracted} 个视频")
        
        extracted = process_part(part_name, filtered_videos, output_dir, progress_manager)
        
        total_extracted += extracted
        processed_parts += 1
        
        # 显示速度信息
        elapsed = time.time() - start_time
        speed = total_extracted / elapsed if elapsed > 0 else 0
        eta = (len(filtered_videos) - total_extracted) / speed if speed > 0 else 0
        
        print(f"当前已处理 {processed_parts}/{len(all_parts)} 个part")
        print(f"总共提取了 {total_extracted} 个视频")
        print(f"平均速度: {speed:.1f} 个视频/秒")
        print(f"预计剩余时间: {eta/60:.1f} 分钟")
        
        # 每处理几个part后暂停一下，避免过于频繁的请求
        if (i + 1) % 5 == 0:
            print("暂停10秒...")
            time.sleep(10)
    
    print(f"\n=== 处理完成 ===")
    progress_manager.print_summary()
    print(f"输出目录: {output_dir}")
    print(f"总用时: {(time.time() - start_time)/60:.1f} 分钟")
    
    # 清理临时目录和缓存
    if os.path.exists("temp") and not os.listdir("temp"):
        os.rmdir("temp")
        print("已清理临时目录")
    
    # 可选：清理 HF 缓存
    # if os.path.exists("./hf_cache"):
    #     shutil.rmtree("./hf_cache")
    #     print("已清理 HF 缓存")

if __name__ == "__main__":
    # 检查必要的包
    try:
        import pandas
        import tqdm
        import huggingface_hub
    except ImportError:
        print("请安装必要的包:")
        print("pip install pandas tqdm huggingface-hub")
        exit(1)
    
    main()