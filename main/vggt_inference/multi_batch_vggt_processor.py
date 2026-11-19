#!/usr/bin/env python
"""
多批次VGGT处理器：将长视频分成多个批次，保持第一帧相同用于对齐
"""
import os
import subprocess
import glob
import shutil
import numpy as np

def count_images(crop_frame_dir):
    """计算目录中的图像数量"""
    image_count = 0
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_count += len(glob.glob(os.path.join(crop_frame_dir, ext)))
    return image_count

def get_sorted_images(crop_frame_dir):
    """获取排序后的图像列表"""
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(glob.glob(os.path.join(crop_frame_dir, ext)))
    return sorted(image_files)

def process_video_in_batches(video_name, result_dir, gpu_id, batch_size=60):
    """
    将视频分成多个批次处理，每个批次包含第一帧用于对齐
    """
    crop_frame_dir = os.path.join(result_dir, video_name, "crop_frame")
    vggt_output_dir = os.path.join(result_dir, video_name, "results_vggt_sparse")
    
    # 获取所有图像
    all_images = get_sorted_images(crop_frame_dir)
    total_frames = len(all_images)
    
    if total_frames == 0:
        print(f"[VGGT] No images found in {crop_frame_dir}")
        return False
    
    print(f"\n[VGGT] Processing {video_name} with {total_frames} frames on GPU {gpu_id}")
    
    # 如果帧数少于批次大小，直接处理
    if total_frames <= batch_size:
        return process_single_batch(crop_frame_dir, vggt_output_dir, gpu_id)
    
    # 计算需要的批次数
    # 每个批次包含：第一帧 + (batch_size-1)个新帧
    frames_per_batch = batch_size - 1  # 除了第一帧外的新帧数
    num_batches = (total_frames - 1 + frames_per_batch - 1) // frames_per_batch
    
    print(f"[VGGT] Splitting into {num_batches} batches (batch_size={batch_size})")
    
    # 存储每个批次的结果
    batch_results = []
    
    for batch_idx in range(num_batches):
        # 创建批次目录
        batch_dir = os.path.join(crop_frame_dir, f"batch_{batch_idx}")
        os.makedirs(batch_dir, exist_ok=True)
        
        # 计算这个批次要处理的帧
        start_idx = 1 + batch_idx * frames_per_batch  # 跳过第一帧，从第二帧开始
        end_idx = min(start_idx + frames_per_batch, total_frames)
        
        print(f"[VGGT] Batch {batch_idx+1}/{num_batches}: frames 0 + {start_idx}-{end_idx-1}")
        
        # 复制图像到批次目录
        # 始终包含第一帧
        batch_images = [all_images[0]] + all_images[start_idx:end_idx]
        
        for i, src_image in enumerate(batch_images):
            dst_image = os.path.join(batch_dir, f"{i:08d}.jpg")
            shutil.copy2(src_image, dst_image)
        
        # 确保images子目录存在
        images_dir = os.path.join(batch_dir, "images")
        if not os.path.exists(images_dir):
            os.makedirs(images_dir, exist_ok=True)
            for file in os.listdir(batch_dir):
                if file.lower().endswith(('.jpg', '.png')):
                    src = os.path.join(batch_dir, file)
                    dst = os.path.join(images_dir, file)
                    if not os.path.exists(dst):
                        os.symlink(os.path.abspath(src), dst)
        
        # 处理这个批次
        batch_output_dir = os.path.join(batch_dir, "sparse")
        success = process_single_batch(batch_dir, batch_output_dir, gpu_id)
        
        if success and os.path.exists(batch_output_dir):
            batch_results.append({
                'batch_idx': batch_idx,
                'batch_dir': batch_dir,
                'output_dir': batch_output_dir,
                'start_frame': start_idx,
                'end_frame': end_idx
            })
        else:
            print(f"[VGGT] Failed to process batch {batch_idx}")
            # 清理失败的批次
            shutil.rmtree(batch_dir)
    
    # 合并所有批次的结果
    if len(batch_results) == num_batches:
        print(f"[VGGT] All {num_batches} batches processed successfully, merging results...")
        success = merge_batch_results(batch_results, vggt_output_dir, all_images[0])
        
        # 清理批次目录
        for result in batch_results:
            shutil.rmtree(result['batch_dir'])
        
        return success
    else:
        print(f"[VGGT] Only {len(batch_results)}/{num_batches} batches succeeded")
        return False

def process_single_batch(input_dir, output_dir, gpu_id):
    """处理单个批次"""
    cmd = [
        "python", "demo_colmap.py",
        "--scene_dir", os.path.abspath(input_dir),
        "--conf_thres_value", "1.0"
    ]
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    try:
        subprocess.run(cmd, cwd="vggt_inference", check=True, env=env, timeout=600)
        return True
    except Exception as e:
        print(f"[VGGT] Error processing batch: {e}")
        return False

def merge_batch_results(batch_results, output_dir, first_frame_path):
    """
    合并多个批次的结果
    由于所有批次都使用相同的第一帧，相机位姿应该是对齐的
    """
    print("[VGGT] Merging batch results...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 合并PLY文件中的点云
    all_points = []
    all_colors = []
    
    for i, result in enumerate(batch_results):
        ply_file = os.path.join(result['output_dir'], 'points.ply')
        if os.path.exists(ply_file):
            # 读取PLY文件
            import trimesh
            try:
                point_cloud = trimesh.load(ply_file)
                points = np.array(point_cloud.vertices)
                # 获取颜色，trimesh可能将颜色存储在visual.vertex_colors中
                if hasattr(point_cloud, 'colors') and point_cloud.colors is not None:
                    colors = np.array(point_cloud.colors)
                elif hasattr(point_cloud, 'visual') and hasattr(point_cloud.visual, 'vertex_colors'):
                    colors = np.array(point_cloud.visual.vertex_colors)[:, :3]  # 只取RGB，忽略alpha
                else:
                    colors = None
                
                # 如果是第一个批次，保留所有点
                # 对于后续批次，跳过第一帧的点（避免重复）
                if i > 0:
                    # 简单估计：假设点均匀分布在各帧
                    # 实际上应该根据points3D.bin来精确过滤
                    num_frames_in_batch = len(os.listdir(result['batch_dir'])) - 1  # 减去images目录
                    if num_frames_in_batch > 1:
                        points_per_frame = len(points) // num_frames_in_batch
                        points = points[points_per_frame:]  # 跳过第一帧的点
                        if colors is not None:
                            colors = colors[points_per_frame:]
                
                all_points.append(points)
                if colors is not None:
                    all_colors.append(colors)
                    
            except Exception as e:
                print(f"[VGGT] Error reading PLY from batch {i}: {e}")
    
    # 保存合并的点云
    if all_points:
        merged_points = np.vstack(all_points)
        merged_colors = np.vstack(all_colors) if all_colors else None
        
        merged_ply_path = os.path.join(output_dir, 'points.ply')
        merged_cloud = trimesh.PointCloud(vertices=merged_points, colors=merged_colors)
        merged_cloud.export(merged_ply_path)
        print(f"[VGGT] Saved merged point cloud with {len(merged_points)} points")
    
    # 复制其他COLMAP文件（cameras.bin, images.bin等）从第一个批次
    if batch_results:
        first_batch_output = batch_results[0]['output_dir']
        for filename in ['cameras.bin', 'images.bin', 'points3D.bin']:
            src = os.path.join(first_batch_output, filename)
            if os.path.exists(src):
                dst = os.path.join(output_dir, filename)
                shutil.copy2(src, dst)
    
    return True

if __name__ == "__main__":
    # 测试
    import sys
    if len(sys.argv) > 1:
        video_name = sys.argv[1]
        result_dir = "../output/OpenVidHD-camera_results"
        gpu_id = 0
        process_video_in_batches(video_name, result_dir, gpu_id, batch_size=60)