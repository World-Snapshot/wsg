import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
from PIL import Image
import json
from tqdm import tqdm
import urllib.request
import multiprocessing
from multiprocessing import Process
import signal
import atexit
import subprocess

# 配置数据集名称
data_set_name = "test_dataset"
#data_set_name = "AVIS_val"
video_dir = f"../input/{data_set_name}"
result_dir = f"../output/{data_set_name}_results"
output_subdir = "results_mediapipe"
set_fps = 1  # 设置处理视频的帧率

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32)
]

# Model下载链接和本地路径
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
MODEL_PATH = "./models/pose_landmarker_heavy.task"

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

def process_video_with_mediapipe(rank, video_name, crop_frame_path, output_base_dir, fps=1):
    """使用指定GPU处理单个视频"""
    # 设置环境变量，确保MediaPipe使用正确的GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
    output_dir = os.path.join(output_base_dir, video_name, output_subdir)
    npz_dir = os.path.join(output_dir, "npz")
    img_dir = os.path.join(output_dir, "img")
    json_dir = os.path.join(output_dir, "json")
    os.makedirs(npz_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    print(f"GPU {rank}: MediaPipe - 创建输出目录...")
    
    # 初始化mediapipe pose模型
    print(f"GPU {rank}: MediaPipe - 在 GPU {rank} 上初始化检测器...")
    base_options = mp_tasks.BaseOptions(
        model_asset_path=MODEL_PATH,
        delegate=mp_tasks.BaseOptions.Delegate.GPU  # 使用GPU
    )
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        num_poses=10,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3
    )
    detector = vision.PoseLandmarker.create_from_options(options)
    print(f"GPU {rank}: MediaPipe - 初始化检测器完成")

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
    
    # 读取第一张图片获取尺寸
    first_img = cv2.imread(image_files[0])
    height, width = first_img.shape[:2]
    frame_interval = 1  # 处理所有图片

    vis_images = []
    save_idx = 1
    print(f"GPU {rank}: MediaPipe - 从crop_frame目录读取了 {frame_count} 张图片")
    
    for frame_idx, image_path in enumerate(image_files):
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"GPU {rank}: 警告 - 无法读取图片 {image_path}")
            continue
        
        # 显示处理进度
        if (frame_idx + 1) % 10 == 0 or frame_idx == 0 or frame_idx == frame_count - 1:
            print(f"GPU {rank}: MediaPipe - 已处理 {frame_idx + 1}/{frame_count} 帧 ({(frame_idx + 1)/frame_count*100:.1f}%)")
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        result = detector.detect(mp_image)
        keypoints_dict = {}
        output_image = frame.copy()
        # 多人检测
        if result.pose_landmarks:
            for idx, pose_landmarks in enumerate(result.pose_landmarks):
                keypoints = []
                for lm in pose_landmarks:
                    keypoints.append([
                        lm.x, lm.y, lm.z, lm.visibility
                    ])
                keypoints = np.array(keypoints)
                keypoints_dict[f'person_{idx}'] = keypoints
                color = (255, 255, 255)  # 线条颜色可自定义(默认白色)
                for connection in POSE_CONNECTIONS:
                    start_idx, end_idx = connection
                    if (start_idx < len(keypoints) and end_idx < len(keypoints) and 
                        keypoints[start_idx][3] > 0.3 and keypoints[end_idx][3] > 0.3):
                        start_point = (int(keypoints[start_idx][0] * width), int(keypoints[start_idx][1] * height))
                        end_point = (int(keypoints[end_idx][0] * width), int(keypoints[end_idx][1] * height))
                        cv2.line(output_image, start_point, end_point, color, 2)
                for x, y, z, v in keypoints:
                    if v > 0.3:
                        cv2.circle(output_image, (int(x * width), int(y * height)), 4, color, -1)
                        cv2.circle(output_image, (int(x * width), int(y * height)), 4, (255, 255, 255), 1)
        # 保存npz
        npz_path = os.path.join(npz_dir, f"{save_idx:08d}.npz")
        np.savez_compressed(npz_path, **keypoints_dict)
        # 保存json
        json_path = os.path.join(json_dir, f"{save_idx:08d}.json")
        with open(json_path, 'w') as f:
            json.dump({k: v.tolist() for k, v in keypoints_dict.items()}, f)
        # 保存图片
        img_path = os.path.join(img_dir, f"{save_idx:08d}.jpg")
        cv2.imwrite(img_path, output_image)
        vis_images.append(img_path)
        save_idx += 1
    print(f"GPU {rank}: MediaPipe - 完成处理 {video_name}")
    detector.close()
    # 生成可视化视频
    if vis_images:
        video_out_path = os.path.join(output_base_dir, video_name, f"visualization_mediapipe.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_out_path, fourcc, fps, (width, height))
        for img_path in vis_images:
            img = cv2.imread(img_path)
            out.write(img)
        out.release()
        print(f"GPU {rank}: MediaPipe - 生成可视化视频: {video_out_path}")

def load_processed_videos(processed_file_path):
    """加载已处理的视频列表"""
    if os.path.exists(processed_file_path):
        with open(processed_file_path, 'r', encoding='utf-8') as f:
            return set(line.strip() for line in f if line.strip())
    return set()

def save_processed_video(processed_file_path, video_name):
    """保存已处理的视频名称到文件"""
    with open(processed_file_path, 'a', encoding='utf-8') as f:
        f.write(video_name + '\n')

def process_videos_distributed(video_paths, result_dir, fps, rank, world_size):
    """分布式处理视频"""
    # 已处理视频记录文件路径
    processed_file_path = os.path.join(result_dir, "processed_video_name_by_mediapipe_pose.txt")
    
    for i, (video_name, crop_frame_path) in enumerate(video_paths):
        if i % world_size == rank:
            print(f"GPU {rank} 正在处理视频: {video_name}")
            
            try:
                # 检查是否已经有MediaPipe结果
                mediapipe_results_dir = os.path.join(result_dir, video_name, output_subdir)
                if os.path.exists(os.path.join(mediapipe_results_dir, "json")):
                    print(f"GPU {rank}: 视频 {video_name} 已有MediaPipe结果，跳过")
                    continue
                
                process_video_with_mediapipe(rank, video_name, crop_frame_path, result_dir, fps)
                
                # 处理完成后，将视频名称添加到已处理列表
                with open(processed_file_path, 'a') as f:
                    f.write(video_name + '\n')
                print(f"GPU {rank}: 视频 {video_name} 处理完成并已记录")
                
            except Exception as e:
                print(f"GPU {rank}: 处理视频 {video_name} 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue

def worker(rank, world_size, video_paths, result_dir, fps):
    """Worker进程函数"""
    process_videos_distributed(video_paths, result_dir, fps, rank, world_size)

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

def main():
    # 首先检查并下载模型
    download_model()
    
    # 创建result_dir目录（如果不存在）
    os.makedirs(result_dir, exist_ok=True)
    
    # 已处理视频记录文件路径
    processed_file_path = os.path.join(result_dir, "processed_video_name_by_mediapipe_pose.txt")
    
    # 加载已处理的视频列表
    processed_videos = load_processed_videos(processed_file_path)
    print(f"已处理的视频数量: {len(processed_videos)}")
    if processed_videos:
        print(f"已处理的视频: {list(processed_videos)}")
    
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
    
    print(f"找到 {len(video_paths)} 个视频需要MediaPipe处理")
    
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
    
    print(f"找到 {num_gpus} 个GPU，运行分布式处理")
    
    # 创建进程
    global processes
    processes = []
    
    for rank in range(num_gpus):
        p = Process(target=worker, args=(rank, num_gpus, video_paths, result_dir, set_fps))
        p.start()
        processes.append(p)
    
    # 等待所有进程完成
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\n接收到中断信号，正在终止进程...")
        cleanup()
    
    print("\n所有MediaPipe处理任务完成！")

if __name__ == '__main__':
    main()