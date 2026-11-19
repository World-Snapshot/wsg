#!/usr/bin/env python3
import os
import sys
import numpy as np
import time
import argparse
import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
import scipy.io as scio
import cv2
import warnings
warnings.filterwarnings('ignore')



def map_lidar_to_image(lidar_xyz: np.ndarray, image_array: np.ndarray):
    """
    优化版本的LiDAR映射
    """
    H, W = image_array.shape[:2]
    lidar_map = np.full((H, W, 3), -1.0, dtype=np.float16)
    
    if len(lidar_xyz) == 0:
        return lidar_map
    
    # 向量化坐标归一化
    x_min, x_max = lidar_xyz[:, 0].min(), lidar_xyz[:, 0].max()
    y_min, y_max = lidar_xyz[:, 1].min(), lidar_xyz[:, 1].max()
    
    if x_max > x_min and y_max > y_min:
        x_norm = (lidar_xyz[:, 0] - x_min) / (x_max - x_min)
        y_norm = (lidar_xyz[:, 1] - y_min) / (y_max - y_min)
        
        u = np.clip((x_norm * (W - 1)).astype(np.int32), 0, W-1)
        v = np.clip((y_norm * (H - 1)).astype(np.int32), 0, H-1)
        
        # 使用更快的唯一坐标查找
        coords = np.column_stack((v, u))
        unique_coords, first_idx = np.unique(coords, axis=0, return_index=True)
        
        # 直接赋值
        lidar_map[unique_coords[:, 0], unique_coords[:, 1]] = lidar_xyz[first_idx]
    
    return lidar_map

def map_radar_to_image(radar_data: np.ndarray, image_array: np.ndarray):
    """
    优化版本的Radar映射
    """
    H, W = image_array.shape[:2]
    radar_map = np.full((H, W, 5), -1.0, dtype=np.float16)
    
    if len(radar_data) == 0:
        return radar_map
    
    # 向量化处理
    x_min, x_max = radar_data[:, 0].min(), radar_data[:, 0].max()
    y_min, y_max = radar_data[:, 1].min(), radar_data[:, 1].max()
    
    if x_max > x_min and y_max > y_min:
        x_norm = (radar_data[:, 0] - x_min) / (x_max - x_min)
        y_norm = (radar_data[:, 1] - y_min) / (y_max - y_min)
        
        u = np.clip((x_norm * (W - 1)).astype(np.int32), 0, W-1)
        v = np.clip((y_norm * (H - 1)).astype(np.int32), 0, H-1)
        
        coords = np.column_stack((v, u))
        unique_coords, first_idx = np.unique(coords, axis=0, return_index=True)
        
        radar_map[unique_coords[:, 0], unique_coords[:, 1]] = radar_data[first_idx]
    
    return radar_map

def extract_wifi_features_to_image_array(csi_amp, image_array, verbose=False):
    """
    csi_amp: np.ndarray of shape (T, F, A) from 'CSIamp'
    image_array: np.ndarray of shape (H, W, 3) from PIL image
    """
    H, W = image_array.shape[:2]
    T, F, A = csi_amp.shape  # (3, 114, 10)

    # === Clean and normalize CSI ===
    csi_amp = np.nan_to_num(csi_amp, nan=np.nanmean(csi_amp))
    min_val, max_val = csi_amp.min(), csi_amp.max()
    if max_val > min_val:
        csi_amp = (csi_amp - min_val) / (max_val - min_val)

    # === 创建10个通道的WiFi特征图 ===
    wifi_map = np.zeros((H, W, A), dtype=np.float16)
    
    # 批量处理所有天线 - 预先转换为float32提高效率
    csi_float32 = csi_amp.astype(np.float32)
    
    for antenna_idx in range(A):
        # 使用cv2.resize进行双线性插值
        interpolated = cv2.resize(csi_float32[:, :, antenna_idx], 
                                 (W, H), 
                                 interpolation=cv2.INTER_LINEAR)
        wifi_map[:, :, antenna_idx] = interpolated.astype(np.float16)
    
    if verbose:
        print(f"    WiFi时频图插值: ({T}, {F}) -> ({H}, {W})")
    
    return wifi_map


def load_wsg_data(wsg_path):
    """加载WSG数据"""
    try:
        data = np.load(wsg_path, allow_pickle=True)
        return data['wsg_data']
    except Exception as e:
        print(f"加载WSG数据失败 {wsg_path}: {str(e)}")
        return None

def save_wsg_data(wsg_array, wsg_path):
    """优化的保存函数"""
    try:
        os.makedirs(os.path.dirname(wsg_path), exist_ok=True)
        
        # 确保路径以.wsg结尾
        if not wsg_path.endswith('.wsg'):
            wsg_path = wsg_path + '.wsg'
        
        # 先保存为.npz文件
        base_path = wsg_path[:-4]  # 去掉.wsg
        npz_path = base_path + '.npz'
        np.savez_compressed(npz_path, wsg_data=wsg_array)
        
        # 重命名为.wsg
        if os.path.exists(wsg_path):
            os.remove(wsg_path)
        os.rename(npz_path, wsg_path)
        
        return True
    except Exception as e:
        return False

def process_single_frame(args):
    """处理单个帧的数据注入"""
    frame_path, experiment_dir, wsg_path, verbose = args
    
    try:
        frame_num = os.path.basename(frame_path).replace('.png', '')
        
        # 使用cv2加载图像（更快）
        rgb = cv2.imread(frame_path)
        if rgb is None:
            return False
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        
        # Create new WSG array (不再检查是否存在，直接创建新的)
        wsg = np.full((h, w, 256), -1.0, dtype=np.float16)
        
        # Inject RGB data
        wsg[:, :, :3] = rgb
        
        # Load and inject LiDAR data (使用np.fromfile更快)
        lidar_path = os.path.join(experiment_dir, 'lidar', frame_num + '.bin')
        if os.path.exists(lidar_path):
            lidar = np.fromfile(lidar_path, dtype=np.float64).reshape(-1, 3)
            if verbose and len(lidar) > 0:
                print(f"    LiDAR: {len(lidar)} 点")
            if len(lidar) > 0:
                wsg[:, :, 207:210] = map_lidar_to_image(lidar, rgb)
        
        # Load and inject radar data (使用np.fromfile更快)
        radar_path = os.path.join(experiment_dir, 'mmwave', frame_num + '.bin')
        if os.path.exists(radar_path):
            radar = np.fromfile(radar_path, dtype=np.float64).reshape(-1, 5)
            if verbose and len(radar) > 0:
                print(f"    Radar: {len(radar)} 点")
            if len(radar) > 0:
                wsg[:, :, 211:216] = map_radar_to_image(radar, rgb)
        
        # Load and inject WiFi CSI data
        wifi_path = os.path.join(experiment_dir, 'wifi-csi', frame_num + '.mat')
        if os.path.exists(wifi_path):
            wifi = scio.loadmat(wifi_path)['CSIamp']
            # 快速清理NaN和归一化
            wifi = np.nan_to_num(wifi, nan=0)
            wifi_min, wifi_max = wifi.min(), wifi.max()
            if wifi_max > wifi_min:
                wifi = (wifi - wifi_min) / (wifi_max - wifi_min)
            
            wsg[:, :, 217:227] = extract_wifi_features_to_image_array(wifi, rgb, verbose)
        
        # Save WSG data
        return save_wsg_data(wsg, wsg_path)
        
    except Exception as e:
        if verbose:
            print(f"  处理帧失败 {frame_num}: {str(e)}")
        return False

def process(mmfi_dataset_dir="MMFi_Dataset_results", output_dir="output", max_workers=None, max_frames_per_video=None):
    """处理MMFi数据集的新格式
    
    Args:
        mmfi_dataset_dir: MMFi数据集路径
        output_dir: 输出路径
        max_workers: 最大工作进程数
        max_frames_per_video: 每个视频最多处理的帧数，-1或None表示处理所有帧
    """
    if max_workers is None:
        max_workers = mp.cpu_count()
    
    # 处理帧数限制参数
    if max_frames_per_video == -1:
        max_frames_per_video = None
    
    # 检查已处理的视频列表
    processed_file = os.path.join(mmfi_dataset_dir, 'processed_video_name_by_mmfi.txt')
    processed_videos = set()
    if os.path.exists(processed_file):
        with open(processed_file, 'r') as f:
            processed_videos = set(line.strip() for line in f if line.strip())
        print(f"已处理的视频数: {len(processed_videos)}")
    
    # Get all experiment folders (E01S01A01 format)
    experiment_folders = [
        f for f in os.listdir(mmfi_dataset_dir)
        if os.path.isdir(os.path.join(mmfi_dataset_dir, f)) and f.startswith('E')
    ]
    experiment_folders.sort()
    
    # 过滤掉已处理的视频
    experiment_folders = [f for f in experiment_folders if f not in processed_videos]
    
    if not experiment_folders:
        print("没有找到需要处理的实验文件夹（所有视频可能已处理）")
        return
    
    print(f"找到 {len(experiment_folders)} 个待处理的实验文件夹")
    
    for experiment_folder in tqdm(experiment_folders, desc="处理实验"):
        experiment_dir = os.path.join(mmfi_dataset_dir, experiment_folder)
        print(f"\n处理实验: {experiment_folder}")
        
        # Check if crop_frame directory exists
        crop_frame_dir = os.path.join(experiment_dir, 'crop_frame')
        if not os.path.exists(crop_frame_dir):
            print(f"  crop_frame目录不存在: {crop_frame_dir}")
            continue
        
        # Create output WSG directory in the same experiment folder
        out_dir = os.path.join(experiment_dir, 'wsg')
        os.makedirs(out_dir, exist_ok=True)
        
        # Get all crop frame images
        crop_files = [f for f in os.listdir(crop_frame_dir) if f.endswith('.png')]
        crop_files.sort()
        
        if not crop_files:
            print(f"  没有找到crop_frame图像文件")
            continue
        
        # 应用帧数限制
        total_frames = len(crop_files)
        if max_frames_per_video is not None and max_frames_per_video > 0:
            crop_files = crop_files[:max_frames_per_video]
            print(f"  找到 {total_frames} 个帧，限制处理前 {len(crop_files)} 个帧")
        else:
            print(f"  找到 {len(crop_files)} 个帧需要处理")
        
        # Prepare tasks
        tasks = []
        verbose = False  # 设置是否显示详细信息
        for crop_file in crop_files:
            crop_path = os.path.join(crop_frame_dir, crop_file)
            wsg_path = os.path.join(out_dir, crop_file.replace('.png', '.wsg'))
            tasks.append((crop_path, experiment_dir, wsg_path, verbose))
        
        # 检查已存在的WSG文件，避免重复处理
        existing_wsg = set(f.replace('.wsg', '.png') for f in os.listdir(out_dir) if f.endswith('.wsg'))
        tasks = [task for task in tasks if os.path.basename(task[0]) not in existing_wsg]
        
        if not tasks:
            print(f"  所有帧已处理")
            # 记录到已处理列表
            with open(processed_file, 'a') as f:
                f.write(experiment_folder + '\n')
            continue
        
        # 分批处理以提高效率
        success_count = 0
        batch_size = 50  # 每批处理50个文件
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 分批提交任务
            for i in range(0, len(tasks), batch_size):
                batch_tasks = tasks[i:i+batch_size]
                batch_futures = [executor.submit(process_single_frame, task) for task in batch_tasks]
                
                # 处理该批次
                for future in tqdm(batch_futures, desc=f"  处理{experiment_folder} [批{i//batch_size+1}]", leave=False):
                    try:
                        if future.result():
                            success_count += 1
                    except Exception as e:
                        if verbose:
                            print(f"  任务失败: {str(e)}")
        
        print(f"  完成! 成功: {success_count}/{len(crop_files)}")
        
        # 如果成功处理了大部分帧，将视频名称添加到已处理列表
        if success_count >= len(crop_files) * 0.9:  # 90%以上成功即认为处理完成
            with open(processed_file, 'a') as f:
                f.write(experiment_folder + '\n')
            print(f"  已记录到已处理列表: {experiment_folder}")


def main():
    """主函数"""
    import multiprocessing as mp
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor
    
    # ==================== 配置区域 ====================
    # 输入配置
    mmfi_path = "../output/MMFi_Dataset_results"  # MMFi数据集路径
    # mmfi_path = "../output/MMFi_Dataset_results"  # 另一个数据集示例
    
    # 输出配置 - 直接在原数据集目录下创建wsg文件夹
    output_path = mmfi_path  # 保存在同一目录结构下
    
    # 处理配置
    max_workers = mp.cpu_count()  # 使用所有CPU核心
    # max_workers = 8  # 可以手动设置工作器数量
    
    # 帧数限制配置
    max_frames_per_video = 40  # 每个视频只处理前40帧，设为-1或None处理所有帧
    
    # ==================== 命令行参数（可选）====================
    parser = argparse.ArgumentParser(description='MMFi到WSG注入器 - 新数据格式')
    parser.add_argument('--mmfi', '-m', type=str, default=None,
                        help='覆盖默认的MMFi数据集路径')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='覆盖默认的输出路径')
    parser.add_argument('--workers', '-w', type=int, default=None,
                        help='覆盖默认的并行工作器数量')
    parser.add_argument('--max-frames', '-f', type=int, default=None,
                        help='每个视频最多处理的帧数，-1表示处理所有帧（默认：40）')
    
    args = parser.parse_args()
    
    # 如果提供了命令行参数，覆盖默认值
    if args.mmfi:
        mmfi_path = args.mmfi
    if args.output:
        output_path = args.output
    else:
        output_path = mmfi_path  # 如果没有指定输出路径，使用输入路径
    if args.workers:
        max_workers = args.workers
    if args.max_frames is not None:
        max_frames_per_video = args.max_frames

    print("=" * 50)
    print("MMFi到WSG注入器")
    print("=" * 50)
    print(f"MMFi数据集路径: {mmfi_path}")
    print(f"输出路径: {output_path}")
    print(f"工作器数量: {max_workers}")
    if max_frames_per_video is not None and max_frames_per_video > 0:
        print(f"帧数限制: 每个视频前 {max_frames_per_video} 帧")
    else:
        print(f"帧数限制: 无（处理所有帧）")
    print("=" * 50)

    start_time = time.time()

    process(mmfi_path, output_path, max_workers, max_frames_per_video)

    duration = time.time() - start_time
    print(f"\n总耗时: {duration:.2f} 秒")


if __name__ == '__main__':
    sys.exit(main())