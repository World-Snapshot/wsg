import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
from PIL import Image
import json
import urllib.request
import multiprocessing
from multiprocessing import Process
import signal
import atexit
import subprocess

# 注意：不再强制设置spawn方法，让系统自动选择

# 配置数据集名称
data_set_name = "test_dataset"
#data_set_name = "AVIS_train"
#data_set_name = "ncslgr_mp4"
#data_set_name = "ncslgr_mp4_small"
#data_set_name = "ASLLRP_utterances_small"
#data_set_name = "ASLLRP_utterances"
video_dir = f"../input/{data_set_name}"
result_dir = f"../output/{data_set_name}_results"
output_subdir = "results_mediapipe_segmentation"
framerate = 1  # 设置输出视频的帧率
MAX_CONCURRENT_PROCESSES = 8  # 使用CPU模式，可以安全地使用多进程

# MediaPipe Multiclass Selfie Segmentation模型信息
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
MODEL_PATH = "./models/selfie_multiclass_256x256.tflite"

# Segmentation类别定义
SEGMENTATION_CLASSES = {
    0: "background",
    1: "hair", 
    2: "body-skin",
    3: "face-skin",
    4: "clothes",
    5: "others"  # accessories
}

# 颜色映射 (BGR格式)
SEGMENTATION_COLORS = {
    0: (0, 0, 0),       # 背景 - 黑色
    1: (255, 0, 0),     # 头发 - 蓝色
    2: (0, 255, 255),   # 身体皮肤 - 黄色
    3: (255, 255, 0),   # 脸部皮肤 - 青色
    4: (0, 255, 0),     # 衣服 - 绿色
    5: (255, 0, 255),   # 配饰 - 洋红色
}

def download_model():
    """下载模型文件到指定位置"""
    if not os.path.exists(MODEL_PATH):
        print(f"模型文件不存在，开始下载: {MODEL_URL}")
        # 创建models目录
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        
        # 下载模型文件
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print(f"模型下载完成: {MODEL_PATH}")
        except Exception as e:
            print(f"模型下载失败: {e}")
            raise
    else:
        print(f"模型文件已存在: {MODEL_PATH}")

def save_segmentation_data(segmentation_mask, frame_idx, base_dir, frame_cv2):
    """保存分割数据和可视化结果"""
    try:
        print(f"\n[Save] Starting to save segmentation data for frame {frame_idx}")
        
        # 创建所需的目录
        frame_name = f"{frame_idx:08d}"
        data_dir = os.path.join(base_dir, "results_mediapipe_segmentation")
        img_dir = os.path.join(data_dir, "img")
        npz_dir = os.path.join(data_dir, "npz")
        overlay_dir = os.path.join(base_dir, "mediapipe_segmentation_overlay_img")
        
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(npz_dir, exist_ok=True)
        os.makedirs(overlay_dir, exist_ok=True)

        # 保存NPZ数据
        npz_path = os.path.join(npz_dir, f"{frame_name}.npz")
        np.savez(npz_path, segmentation_mask=segmentation_mask)
        
        # 创建彩色可视化
        height, width = segmentation_mask.shape
        colored_segmentation = np.zeros((height, width, 3), dtype=np.uint8)
        
        for class_id, color in SEGMENTATION_COLORS.items():
            colored_segmentation[segmentation_mask == class_id] = color
        
        # 保存彩色分割图
        cv2.imwrite(os.path.join(img_dir, f"{frame_name}.jpg"), colored_segmentation)
        
        # 创建叠加图像
        overlay = cv2.addWeighted(frame_cv2, 0.5, colored_segmentation, 0.7, 0)
        cv2.imwrite(os.path.join(overlay_dir, f"{frame_name}.png"), overlay)
        
        print(f"[Save] Successfully saved segmentation data and visualizations for frame {frame_idx}")
        return True

    except Exception as e:
        print(f"[Save] Error in save_segmentation_data for frame {frame_idx}:")
        import traceback
        print(traceback.format_exc())
        return False

def process_video_with_mediapipe_segmentation(rank, video_name, crop_frame_path, output_base_dir, framerate=30):
    """使用MediaPipe进行分割处理"""
    # 注意：CUDA_VISIBLE_DEVICES 应该在进程启动前就设置好
    
    print(f"GPU {rank}: MediaPipe Segmentation - 开始处理视频: {video_name}")
    
    try:
        # 初始化MediaPipe分割模型
        print(f"GPU {rank}: 初始化MediaPipe分割模型...")
        # 注意：由于多GPU环境下的兼容性问题，改用CPU模式
        base_options = mp_tasks.BaseOptions(
            model_asset_path=MODEL_PATH,
            delegate=mp_tasks.BaseOptions.Delegate.CPU  # 使用CPU避免GPU批大小错误
        )
        options = vision.ImageSegmenterOptions(
            base_options=base_options,
            output_category_mask=True,
            output_confidence_masks=False
        )
        segmenter = vision.ImageSegmenter.create_from_options(options)
        print(f"GPU {rank}: 分割模型初始化完成")
        
        # 读取crop_frame目录中的图片
        import glob
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(crop_frame_path, ext)))
        image_files.sort()  # 按文件名排序
        
        frame_count = len(image_files)
        if frame_count == 0:
            print(f"警告: {crop_frame_path} 中没有找到图片")
            return
        
        print(f"GPU {rank}: 找到 {frame_count} 张图片需要处理")
        
        # 处理每一帧
        for frame_idx, image_path in enumerate(image_files):
            try:
                # 读取图像
                frame = cv2.imread(image_path)
                if frame is None:
                    print(f"GPU {rank}: 警告 - 无法读取图片 {image_path}")
                    continue
                
                # 显示处理进度
                if (frame_idx + 1) % 10 == 0 or frame_idx == 0 or frame_idx == frame_count - 1:
                    print(f"GPU {rank}: 已处理 {frame_idx + 1}/{frame_count} 帧 ({(frame_idx + 1)/frame_count*100:.1f}%)")
                
                # 转换为RGB格式
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
                
                # 执行分割
                segmentation_result = segmenter.segment(mp_image)
                
                if segmentation_result.category_mask is not None:
                    # 获取分割mask
                    segmentation_mask = segmentation_result.category_mask.numpy_view()
                    
                    # 保存结果
                    success = save_segmentation_data(
                        segmentation_mask, 
                        frame_idx + 1,  # 帧索引从1开始
                        output_base_dir, 
                        frame
                    )
                    
                    if not success:
                        print(f"GPU {rank}: 帧 {frame_idx + 1} 保存失败")
                else:
                    print(f"GPU {rank}: 帧 {frame_idx + 1} 没有检测到分割结果")
                    
            except Exception as e:
                print(f"GPU {rank}: 处理帧 {frame_idx + 1} 时出错: {str(e)}")
                import traceback
                print(traceback.format_exc())
                continue
        
        # 生成可视化视频
        print(f"GPU {rank}: 生成可视化视频...")
        try:
            video_output_dir = output_base_dir
            
            # 生成彩色分割视频
            img_dir = os.path.join(video_output_dir, "results_mediapipe_segmentation/img")
            if os.path.exists(img_dir) and os.listdir(img_dir):
                vis_path = os.path.join(video_output_dir, "visualization_mediapipe_segmentation.mp4")
                cmd = f'ffmpeg -y -framerate {framerate} -i {img_dir}/%08d.jpg -c:v libx264 -pix_fmt yuv420p {vis_path}'
                subprocess.run(cmd, shell=True, check=True)
                print(f"GPU {rank}: 创建分割可视化视频")

            # 生成叠加视频
            overlay_dir = os.path.join(video_output_dir, "mediapipe_segmentation_overlay_img")
            if os.path.exists(overlay_dir) and os.listdir(overlay_dir):
                overlay_path = os.path.join(video_output_dir, "visualization_mediapipe_segmentation_overlap.mp4")
                cmd = f'ffmpeg -y -framerate {framerate} -i {overlay_dir}/%08d.png -c:v libx264 -pix_fmt yuv420p {overlay_path}'
                subprocess.run(cmd, shell=True, check=True)
                print(f"GPU {rank}: 创建叠加可视化视频")

        except Exception as e:
            print(f"GPU {rank}: 生成视频时出错: {str(e)}")
            
    except Exception as e:
        print(f"GPU {rank}: 处理视频 {video_name} 时出错: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise
    finally:
        if 'segmenter' in locals():
            segmenter.close()
            print(f"GPU {rank}: 关闭分割器")

def process_videos_distributed(rank, world_size, video_paths, result_dir, framerate=30):
    """分布式处理视频"""
    # 已处理视频记录文件路径
    processed_file_path = os.path.join(result_dir, "processed_video_name_by_mediapipe_segmentation.txt")
    
    for i, (video_name, crop_frame_path) in enumerate(video_paths):
        if i % world_size == rank:
            print(f"GPU {rank} 正在处理视频: {video_name}")
            
            try:
                # 检查是否已经有结果
                segmentation_results_dir = os.path.join(result_dir, video_name, "results_mediapipe_segmentation")
                if os.path.exists(os.path.join(segmentation_results_dir, "npz")):
                    print(f"GPU {rank}: 视频 {video_name} 已有分割结果，跳过")
                    continue
                
                folder_path = os.path.join(result_dir, video_name)
                process_video_with_mediapipe_segmentation(rank, video_name, crop_frame_path, folder_path, framerate)
                
                # 处理完成后，将视频名称添加到已处理列表
                with open(processed_file_path, 'a') as f:
                    f.write(video_name + '\n')
                print(f"GPU {rank}: 视频 {video_name} 处理完成并已记录")
                
            except Exception as e:
                print(f"GPU {rank}: 处理视频 {video_name} 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue

def worker_wrapper(rank, world_size, video_paths, result_dir, framerate):
    """Worker进程包装函数，在进程开始时设置环境变量"""
    # 设置环境变量，限制GPU使用
    os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'  # 每个GPU使用独立的线程
    
    print(f"Worker {rank} started with CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    
    # 添加延迟，避免同时初始化
    import time
    time.sleep(rank * 0.5)  # 每个rank延迟0.5秒
    
    # 然后再导入需要GPU的库
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision
    
    # 调用实际的处理函数
    process_videos_distributed(rank, world_size, video_paths, result_dir, framerate)

# 全局变量存储进程列表
processes = []

def cleanup():
    """清理函数，确保所有子进程被正确终止"""
    global processes
    for p in processes:
        if p.is_alive():
            p.terminate()
            p.join()

# 注册清理函数
atexit.register(cleanup)

def signal_handler(signum, frame):
    """信号处理函数"""
    print("\n接收到中断信号，正在清理...")
    cleanup()
    exit(0)

# 注册信号处理
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def load_processed_videos(processed_file_path):
    """加载已处理的视频列表"""
    if os.path.exists(processed_file_path):
        with open(processed_file_path, 'r', encoding='utf-8') as f:
            return set(line.strip() for line in f if line.strip())
    return set()

def main():
    # 首先检查并下载模型
    download_model()
    
    # 创建result_dir目录（如果不存在）
    os.makedirs(result_dir, exist_ok=True)
    
    # 已处理视频记录文件路径
    processed_file_path = os.path.join(result_dir, "processed_video_name_by_mediapipe_segmentation.txt")
    
    # 加载已处理的视频列表
    processed_videos = load_processed_videos(processed_file_path)
    print(f"已处理的视频数量: {len(processed_videos)}")
    
    # 收集需要处理的视频路径
    video_paths = []
    
    # 从output目录检查需要处理的视频
    if os.path.exists(result_dir):
        for video_name in os.listdir(result_dir):
            video_path = os.path.join(result_dir, video_name)
            if os.path.isdir(video_path):
                # 检查是否已经处理过
                if video_name in processed_videos:
                    continue
                    
                crop_frame_path = os.path.join(video_path, "crop_frame")
                if os.path.exists(crop_frame_path):
                    video_paths.append((video_name, crop_frame_path))
                else:
                    print(f"跳过: {video_name}，未找到crop_frame目录")
    else:
        print(f"结果目录不存在: {result_dir}")
        return
    
    if not video_paths:
        print("没有找到需要处理的视频")
        return
    
    print(f"找到 {len(video_paths)} 个视频需要MediaPipe分割处理")
    
    # 获取可用GPU数量
    try:
        # 使用nvidia-smi获取GPU数量
        result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
        if result.returncode == 0:
            num_gpus = len(result.stdout.strip().split('\n'))
        else:
            num_gpus = 0
    except:
        num_gpus = 0
    
    if num_gpus == 0:
        print("没有找到GPU，使用CPU处理")
        num_gpus = 1
    
    # 限制并发进程数
    actual_processes = min(num_gpus, MAX_CONCURRENT_PROCESSES)
    print(f"找到 {num_gpus} 个GPU，但限制使用 {actual_processes} 个进程以避免GPU冲突")
    
    if actual_processes == 1:
        # 单进程串行处理所有视频
        print("使用单进程串行处理模式")
        worker_wrapper(0, 1, video_paths, result_dir, framerate)
    else:
        # 多进程处理
        print(f"使用 {actual_processes} 个进程并行处理")
        global processes
        processes = []
        
        for rank in range(actual_processes):
            p = Process(target=worker_wrapper, args=(rank, actual_processes, video_paths, result_dir, framerate))
            p.start()
            processes.append(p)
        
        # 等待所有进程完成
        try:
            for p in processes:
                p.join()
        except KeyboardInterrupt:
            print("\n接收到中断信号，正在终止进程...")
            cleanup()
    
    print("\n所有MediaPipe分割处理任务完成！")

if __name__ == '__main__':
    main()