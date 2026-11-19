import os
import subprocess
import shutil
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import time
import numpy as np
import glob
import sys
sys.path.append('monst3r_inference')

def verify_processed_video(result_dir, video_name):
    """验证视频是否真正被成功处理（有有效的点云数据）"""
    monst3r_output_dir = os.path.join(result_dir, video_name, "results_monst3r")
    
    # 检查输出文件夹是否存在
    if not os.path.exists(monst3r_output_dir):
        return False
    
    # 检查是否有输出文件（如深度图、RGB图像等）
    depth_files = glob.glob(os.path.join(monst3r_output_dir, "depth_*.npy"))
    conf_files = glob.glob(os.path.join(monst3r_output_dir, "conf_*.npy"))
    
    # 检查是否有轨迹文件
    traj_file = os.path.join(monst3r_output_dir, "pred_traj.txt")
    
    if len(depth_files) > 0 and len(conf_files) > 0 and os.path.exists(traj_file):
        # 进一步检查轨迹文件是否有效
        try:
            with open(traj_file, 'r') as f:
                lines = f.readlines()
                if len(lines) > 0:
                    return True
        except:
            pass
    
    return False

def get_processed_videos(result_dir):
    """获取已处理的视频列表，并验证其有效性"""
    record_file = os.path.join(result_dir, "processed_video_name_by_monst3r.txt")
    if not os.path.exists(record_file):
        return set()
    
    with open(record_file, 'r') as f:
        recorded_videos = set(line.strip() for line in f if line.strip())
    
    # 验证每个已记录的视频
    valid_videos = set()
    invalid_videos = set()
    
    print(f"[MONST3R] Verifying {len(recorded_videos)} previously processed videos...")
    
    for video_name in recorded_videos:
        if verify_processed_video(result_dir, video_name):
            valid_videos.add(video_name)
        else:
            invalid_videos.add(video_name)
            print(f"[MONST3R] Invalid processing found for {video_name}, will reprocess")
    
    # 如果有无效的视频，更新记录文件
    if invalid_videos:
        print(f"[MONST3R] Found {len(invalid_videos)} invalid videos, updating record file")
        with open(record_file, 'w') as f:
            for video_name in sorted(valid_videos):
                f.write(f"{video_name}\n")
    
    print(f"[MONST3R] Verification complete: {len(valid_videos)} valid, {len(invalid_videos)} invalid")
    return valid_videos

def record_video_processed(result_dir, video_name):
    record_file = os.path.join(result_dir, "processed_video_name_by_monst3r.txt")
    # 如果文件不存在则创建
    if not os.path.exists(record_file):
        with open(record_file, 'w') as f:
            pass
    # 写入视频名
    with open(record_file, 'a') as f:
        f.write(f"{video_name}\n")

def get_available_gpus(force_all=False):
    """获取可用的GPU列表"""
    try:
        if force_all:
            # 强制使用所有GPU
            result = subprocess.run(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, check=True)
            all_gpus = [int(line.strip()) for line in result.stdout.strip().split('\n') if line.strip()]
            print(f"[MONST3R] Force using all {len(all_gpus)} GPUs: {all_gpus}")
            return all_gpus
        
        # 检测空闲的GPU
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, check=True)
        
        available_gpus = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = line.split(',')
                gpu_id = int(parts[0].strip())
                memory_used = int(parts[1].strip())
                
                # 如果GPU内存使用少于1GB，认为是可用的
                if memory_used < 1000:
                    available_gpus.append(gpu_id)
                else:
                    print(f"[MONST3R] GPU {gpu_id} is busy (using {memory_used}MB), skipping")
        
        # 如果没有空闲GPU，至少使用GPU 1-7
        return available_gpus if available_gpus else list(range(1, 8))
    except Exception as e:
        # 默认使用所有GPU除了GPU 0
        print("Warning: Cannot detect GPUs automatically, using GPUs 1-7")
        return list(range(1, 8))

def count_video_frames(crop_frame_dir):
    """快速计算视频帧数"""
    count = 0
    for ext in ['.jpg', '.jpeg', '.png']:
        count += len([f for f in os.listdir(crop_frame_dir) if f.lower().endswith(ext)])
    return count

def process_single_video(video_name, result_dir, gpu_id):
    """在指定GPU上处理单个视频"""
    crop_frame_dir = os.path.join(result_dir, video_name, "crop_frame")
    monst3r_output_dir = os.path.join(result_dir, video_name, "results_monst3r")
    print(f"\n[MONST3R] Processing video: {video_name} on GPU {gpu_id}")

    # 检查帧数
    frame_count = count_video_frames(crop_frame_dir)
    print(f"[MONST3R] Video has {frame_count} frames")
    
    # 创建输出目录
    os.makedirs(monst3r_output_dir, exist_ok=True)
    
    # 调用 MonST3R demo.py
    cmd = [
        "python", "demo.py",
        "--input", os.path.abspath(crop_frame_dir),
        "--output_dir", os.path.abspath(monst3r_output_dir),
        "--seq_name", video_name,
        "--not_batchify",  # 使用内存高效模式
        "--batch_size", "8",  # 减小批次大小以适应内存
        "--num_frames", str(min(frame_count, 100)),  # 限制最大帧数
        "--weights", "checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth"
    ]
    
    # 设置环境变量指定GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
    
    print(f"[MONST3R] Running on GPU {gpu_id} with CUDA_VISIBLE_DEVICES={gpu_id}")
    
    try:
        subprocess.run(cmd, cwd="monst3r_inference", check=True, env=env, timeout=1200)
    except Exception as e:
        print(f"[MONST3R] Error processing {video_name}: {e}")
        return False

    # 验证处理结果
    if verify_processed_video(result_dir, video_name):
        # 记录处理成功
        record_video_processed(result_dir, video_name)
        print(f"[MONST3R] Successfully processed and verified: {video_name}")
        return True
    else:
        print(f"[MONST3R] WARNING: Processing completed but verification failed for {video_name}")
        # 删除无效的结果
        if os.path.exists(monst3r_output_dir):
            shutil.rmtree(monst3r_output_dir)
        return False

def main():
    dataset_name = "test_dataset"  # 改为实际的数据集名称
    result_dir = f"../output/{dataset_name}_results"
    
    # 获取所有视频
    all_video_names = [name for name in os.listdir(result_dir)
                       if os.path.isdir(os.path.join(result_dir, name, "crop_frame"))]
    
    print(f"[MONST3R] Total videos found: {len(all_video_names)}")
    
    # 获取并验证已处理的视频
    processed_videos = get_processed_videos(result_dir)
    
    # 过滤出需要处理的视频
    video_names = [name for name in all_video_names if name not in processed_videos]
    
    print(f"[MONST3R] Valid processed videos: {len(processed_videos)}")
    print(f"[MONST3R] Videos to be processed: {len(video_names)}")
    
    if not video_names:
        print("[MONST3R] All videos have been successfully processed!")
        return

    # 检查 MonST3R 模型文件是否存在
    model_path = "monst3r_inference/checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth"
    if not os.path.exists(model_path):
        print(f"[MONST3R] ERROR: Model file not found at {model_path}")
        print("[MONST3R] Please download the model first:")
        print("  cd monst3r_inference/data && bash download_ckpt.sh")
        return

    # 获取可用的GPU
    available_gpus = get_available_gpus(force_all=True)  # 使用所有GPU
    num_gpus = len(available_gpus)
    
    print(f"[MONST3R] Found {num_gpus} available GPUs: {available_gpus}")
    
    if num_gpus == 0:
        print("[MONST3R] ERROR: No available GPUs found!")
        return
    
    # 使用所有可用的GPU，但为了内存考虑，限制并发数
    max_workers = min(num_gpus, 4)  # 最多同时运行4个进程，因为MonST3R需要更多内存
    print(f"[MONST3R] Using {max_workers} workers concurrently")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        # 分配视频到不同的GPU
        for i, video_name in enumerate(video_names):
            gpu_id = available_gpus[i % num_gpus]
            future = executor.submit(process_single_video, video_name, result_dir, gpu_id)
            futures.append((future, video_name))
            
            # 避免同时启动太多进程
            if i < max_workers:
                time.sleep(10)  # 10秒间隔启动，给GPU足够的初始化时间
        
        # 收集结果
        for future, video_name in futures:
            try:
                success = future.result()
                if success:
                    print(f"[MONST3R] Completed: {video_name}")
                else:
                    print(f"[MONST3R] Failed: {video_name}")
            except Exception as e:
                print(f"[MONST3R] Exception processing {video_name}: {e}")

if __name__ == "__main__":
    main()