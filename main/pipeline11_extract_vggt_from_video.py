import os
import subprocess
import shutil
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import time

def ensure_images_subdir(crop_frame_dir):
    images_dir = os.path.join(crop_frame_dir, "images")
    if not os.path.exists(images_dir):
        os.makedirs(images_dir, exist_ok=True)
        # 只软链图片，避免重复拷贝
        for file in os.listdir(crop_frame_dir):
            if file.lower().endswith(('.jpg', '.png')):
                src = os.path.join(crop_frame_dir, file)
                dst = os.path.join(images_dir, file)
                if not os.path.exists(dst):
                    os.symlink(os.path.abspath(src), dst)
    return images_dir

def verify_processed_video(result_dir, video_name):
    """验证视频是否真正被成功处理（有有效的点云数据）"""
    vggt_output_dir = os.path.join(result_dir, video_name, "results_vggt_sparse")
    points_file = os.path.join(vggt_output_dir, "points.ply")
    
    if not os.path.exists(points_file):
        return False
    
    # 检查文件大小，如果太小说明没有有效数据
    file_size = os.path.getsize(points_file)
    if file_size <= 300:  # 空文件通常是238字节
        return False
    
    # 检查PLY文件头部信息
    try:
        with open(points_file, 'rb') as f:
            # 读取前1000字节并转换为字符串（处理二进制文件）
            header_bytes = f.read(1000)
            try:
                content = header_bytes.decode('utf-8', errors='ignore')
            except:
                content = header_bytes.decode('latin-1', errors='ignore')
            
            # 查找vertex数量
            if "element vertex 0" in content:
                return False
            
            # 检查是否有有效的vertex数量
            import re
            vertex_match = re.search(r'element vertex (\d+)', content)
            if vertex_match:
                vertex_count = int(vertex_match.group(1))
                if vertex_count > 0:
                    return True
                else:
                    return False
            
    except Exception as e:
        print(f"Error reading PLY file {points_file}: {e}")
        return False
    
    # 如果文件足够大但找不到vertex信息，假设是有效的
    return file_size > 1000

def get_processed_videos(result_dir):
    """获取已处理的视频列表，并验证其有效性"""
    record_file = os.path.join(result_dir, "processed_video_name_by_vggt.txt")
    if not os.path.exists(record_file):
        return set()
    
    with open(record_file, 'r') as f:
        recorded_videos = set(line.strip() for line in f if line.strip())
    
    # 验证每个已记录的视频
    valid_videos = set()
    invalid_videos = set()
    
    print(f"[VERIFY] Verifying {len(recorded_videos)} previously processed videos...")
    
    for video_name in recorded_videos:
        if verify_processed_video(result_dir, video_name):
            valid_videos.add(video_name)
        else:
            invalid_videos.add(video_name)
            print(f"[VERIFY] Invalid processing found for {video_name}, will reprocess")
    
    # 如果有无效的视频，更新记录文件
    if invalid_videos:
        print(f"[VERIFY] Found {len(invalid_videos)} invalid videos, updating record file")
        with open(record_file, 'w') as f:
            for video_name in sorted(valid_videos):
                f.write(f"{video_name}\n")
    
    print(f"[VERIFY] Verification complete: {len(valid_videos)} valid, {len(invalid_videos)} invalid")
    return valid_videos

def record_video_processed(result_dir, video_name):
    record_file = os.path.join(result_dir, "processed_video_name_by_vggt.txt")
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
            print(f"[VGGT] Force using all {len(all_gpus)} GPUs: {all_gpus}")
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
                    print(f"[VGGT] GPU {gpu_id} is busy (using {memory_used}MB), skipping")
        
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

def process_single_video(video_name, result_dir, gpu_id, use_smart_processing=True):
    """在指定GPU上处理单个视频"""
    crop_frame_dir = os.path.join(result_dir, video_name, "crop_frame")
    vggt_output_dir = os.path.join(result_dir, video_name, "results_vggt_sparse")
    print(f"\n[VGGT] Processing video: {video_name} on GPU {gpu_id}")

    # 检查帧数
    frame_count = count_video_frames(crop_frame_dir)
    batch_size = 60  # 每批次的最大帧数（V100 32GB可以处理60帧）
    
    if use_smart_processing and frame_count > batch_size:
        print(f"[VGGT] Video has {frame_count} frames, using multi-batch processing (batch_size={batch_size})")
        # 使用多批次处理器
        from vggt_inference.multi_batch_vggt_processor import process_video_in_batches
        return process_video_in_batches(video_name, result_dir, gpu_id, batch_size=batch_size)
    
    # 确保 images 子目录存在
    ensure_images_subdir(crop_frame_dir)

    # 调用VGGT demo_colmap.py
    cmd = [
        "python", "demo_colmap.py",
        "--scene_dir", os.path.abspath(crop_frame_dir)
    ]
    
    # 设置环境变量指定GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
    
    print(f"[VGGT] Running on GPU {gpu_id} with CUDA_VISIBLE_DEVICES={gpu_id}")
    
    try:
        subprocess.run(cmd, cwd="vggt_inference", check=True, env=env, timeout=600)
    except Exception as e:
        print(f"[VGGT] Error processing {video_name}: {e}")
        return False

    # 移动 sparse 结果到 results_vggt_sparse
    sparse_dir = os.path.join(crop_frame_dir, "sparse")
    if os.path.exists(sparse_dir):
        if os.path.exists(vggt_output_dir):
            shutil.rmtree(vggt_output_dir)
        shutil.move(sparse_dir, vggt_output_dir)
        print(f"[VGGT] Saved VGGT results to {vggt_output_dir}")
        
        # 验证处理结果
        if verify_processed_video(result_dir, video_name):
            # 记录处理成功
            record_video_processed(result_dir, video_name)
            print(f"[VGGT] Successfully processed and verified: {video_name}")
            return True
        else:
            print(f"[VGGT] WARNING: Processing completed but verification failed for {video_name}")
            # 删除无效的结果
            if os.path.exists(vggt_output_dir):
                shutil.rmtree(vggt_output_dir)
            return False
    else:
        print(f"[VGGT] Warning: No sparse result found for {video_name}")
        return False

def main():
    dataset_name = "OpenVidHD-camera"  # 改为实际的数据集名称
    result_dir = f"../output/{dataset_name}_results"
    
    # 获取所有视频
    all_video_names = [name for name in os.listdir(result_dir)
                       if os.path.isdir(os.path.join(result_dir, name, "crop_frame"))]
    
    print(f"[VGGT] Total videos found: {len(all_video_names)}")
    
    # 获取并验证已处理的视频
    processed_videos = get_processed_videos(result_dir)
    
    # 过滤出需要处理的视频
    video_names = [name for name in all_video_names if name not in processed_videos]
    
    print(f"[VGGT] Valid processed videos: {len(processed_videos)}")
    print(f"[VGGT] Videos to be processed: {len(video_names)}")
    
    if not video_names:
        print("[VGGT] All videos have been successfully processed!")
        return

    # 获取可用的GPU
    # 现在GPU 0已经清理，可以使用所有GPU
    available_gpus = get_available_gpus(force_all=True)  # 使用所有8个GPU
    num_gpus = len(available_gpus)
    
    print(f"[VGGT] Found {num_gpus} available GPUs: {available_gpus}")
    
    if num_gpus == 0:
        print("[VGGT] ERROR: No available GPUs found!")
        return
    
    # 使用所有可用的GPU
    max_workers = num_gpus  # 使用所有可用的GPU
    print(f"[VGGT] Using all {max_workers} available GPUs concurrently")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        # 分配视频到不同的GPU
        for i, video_name in enumerate(video_names):
            gpu_id = available_gpus[i % num_gpus]
            future = executor.submit(process_single_video, video_name, result_dir, gpu_id)
            futures.append((future, video_name))
            
            # 避免同时启动太多进程
            if i < max_workers:
                time.sleep(5)  # 5秒间隔启动
        
        # 收集结果
        for future, video_name in futures:
            try:
                success = future.result()
                if success:
                    print(f"[VGGT] Completed: {video_name}")
                else:
                    print(f"[VGGT] Failed: {video_name}")
            except Exception as e:
                print(f"[VGGT] Exception processing {video_name}: {e}")

if __name__ == "__main__":
    main()