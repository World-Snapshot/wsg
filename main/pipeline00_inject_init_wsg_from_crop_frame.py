import os
import numpy as np
from PIL import Image
import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import time

# 尝试导入psutil进行系统监控
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️  psutil未安装，系统自动检测功能受限。可通过 'pip install psutil' 安装")


def ensure_dir(path):
    """确保目录存在"""
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def get_optimal_workers():
    """动态获取最优工作器数量"""
    cpu_count = mp.cpu_count()
    
    if PSUTIL_AVAILABLE:
        memory_gb = psutil.virtual_memory().total / (1024**3)
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        
        print(f"系统信息: CPU={cpu_count}核, 内存={memory_gb:.1f}GB, CPU使用率={cpu_percent:.1f}%, 内存使用率={memory_percent:.1f}%")
        
        # 简化的调整策略
        if cpu_percent > 70 or memory_percent > 80:
            optimal_workers = max(1, cpu_count // 2)  # 保守策略
            system_status = 'busy'
        else:
            optimal_workers = min(cpu_count, 32)  # 限制最大32个工作器
            system_status = 'good'
        
        use_multiprocessing = memory_gb > 16 and cpu_count > 8
    else:
        print(f"系统信息: CPU={cpu_count}核")
        optimal_workers = max(1, cpu_count - 1)
        use_multiprocessing = cpu_count > 4
        system_status = 'unknown'
    
    config = {
        'optimal_workers': optimal_workers,
        'conservative_workers': max(1, cpu_count // 2),
        'use_multiprocessing': use_multiprocessing,
        'system_status': system_status
    }
    
    print(f"推荐配置: {optimal_workers}个工作器, {'多进程' if use_multiprocessing else '多线程'}, 系统状态={system_status}")
    return config


def load_image_with_alpha(image_path):
    """加载图像，保持透明度通道"""
    try:
        img = Image.open(image_path)
        
        if img.mode == 'RGBA':
            img_array = np.array(img, dtype=np.uint8)
        elif img.mode == 'RGB':
            img_array = np.array(img, dtype=np.uint8)
        elif img.mode == 'L':
            img_array = np.array(img, dtype=np.uint8)
        else:
            img = img.convert('RGB')
            img_array = np.array(img, dtype=np.uint8)
            
        return img_array
    except Exception as e:
        print(f"加载图像失败 {image_path}: {str(e)}")
        return None


def convert_to_wsg_format(img_array):
    """将图像数组转换为WSG格式（256通道）"""
    height, width = img_array.shape[:2]
    
    # 预分配256通道数组
    wsg_array = np.full((height, width, 256), -1.0, dtype=np.float16)
    
    # 处理通道
    if len(img_array.shape) == 3:
        original_channels = img_array.shape[2]
    else:
        img_array = np.expand_dims(img_array, axis=2)
        original_channels = 1
    
    # 转换为float16
    img_float = img_array.astype(np.float16)
    
    if original_channels >= 3:
        # RGB通道
        wsg_array[:, :, 0:3] = img_float[:, :, 0:3]
        
        if original_channels == 4:
            # Alpha通道归一化
            wsg_array[:, :, 3] = img_float[:, :, 3] / 255.0
        else:
            wsg_array[:, :, 3] = 1.0
            
    elif original_channels == 1:
        # 灰度图
        gray_value = img_float[:, :, 0]
        wsg_array[:, :, 0:3] = np.stack([gray_value, gray_value, gray_value], axis=2)
        wsg_array[:, :, 3] = 1.0
        wsg_array[:, :, 4] = gray_value  # 原始灰度值保存在第5通道
    
    return wsg_array


def save_wsg_file(wsg_array, output_path):
    """保存WSG文件"""
    try:
        # 确保输出路径以.wsg结尾
        if not output_path.endswith('.wsg'):
            output_path = output_path + '.wsg'
        
        # numpy的savez_compressed会自动添加.npz，所以我们要预先移除.wsg
        base_path = output_path[:-4]  # 移除.wsg，得到基础路径
        
        # 保存文件（numpy会自动添加.npz）
        np.savez_compressed(base_path, wsg_data=wsg_array)
        
        # 立即重命名：从 base_path.npz 改为 base_path.wsg
        npz_path = base_path + '.npz'
        wsg_path = base_path + '.wsg'
        
        if os.path.exists(npz_path):
            # 如果目标.wsg文件已存在，先删除
            if os.path.exists(wsg_path):
                os.remove(wsg_path)
            # 重命名
            os.rename(npz_path, wsg_path)
        
        return True
    except Exception as e:
        print(f"保存失败 {output_path}: {str(e)}")
        return False


def process_single_image(args):
    """处理单个图像（用于并行处理）"""
    image_path, output_dir = args
    
    try:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}.wsg")
        
        # 跳过已存在的文件
        if os.path.exists(output_path):
            return True
        
        # 处理图像
        img_array = load_image_with_alpha(image_path)
        if img_array is None:
            return False
            
        wsg_array = convert_to_wsg_format(img_array)
        return save_wsg_file(wsg_array, output_path)
        
    except Exception as e:
        print(f"处理失败 {image_path}: {str(e)}")
        return False


def convert_frames_to_wsg(input_dir, output_dir, max_workers=None, use_multiprocessing=False):
    """转换图像帧到WSG格式"""
    ensure_dir(output_dir)
    
    # 获取所有图像文件
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    if not image_files:
        print(f"在目录 {input_dir} 中没有找到图像文件")
        return False
    
    image_files.sort()
    print(f"找到 {len(image_files)} 个图像文件，使用 {'多进程' if use_multiprocessing else '多线程'}，{max_workers}个工作器")
    
    if max_workers and max_workers > 1:
        # 并行处理
        tasks = [(image_path, output_dir) for image_path in image_files]
        executor_class = ProcessPoolExecutor if use_multiprocessing else ThreadPoolExecutor
        
        success_count = 0
        with executor_class(max_workers=max_workers) as executor:
            futures = [executor.submit(process_single_image, task) for task in tasks]
            
            for future in tqdm(futures, desc="转换图像"):
                try:
                    if future.result():
                        success_count += 1
                except Exception as e:
                    print(f"任务失败: {str(e)}")
    else:
        # 单线程处理
        success_count = 0
        for image_path in tqdm(image_files, desc="转换图像"):
            if process_single_image((image_path, output_dir)):
                success_count += 1
    
    print(f"转换完成! 成功: {success_count}/{len(image_files)}")
    return success_count > 0


def process_dataset(dataset_results_dir, max_workers=None, use_multiprocessing=False):
    """处理整个数据集"""
    if not os.path.exists(dataset_results_dir):
        print(f"数据集目录不存在: {dataset_results_dir}")
        return
    
    video_folders = [f for f in os.listdir(dataset_results_dir) 
                    if os.path.isdir(os.path.join(dataset_results_dir, f))]
    
    if not video_folders:
        print(f"没有找到视频文件夹")
        return
    
    print(f"找到 {len(video_folders)} 个视频文件夹")
    
    total_success = 0
    for video_folder in tqdm(video_folders, desc="处理视频文件夹"):
        crop_frame_dir = os.path.join(dataset_results_dir, video_folder, "crop_frame")
        wsg_dir = os.path.join(dataset_results_dir, video_folder, "wsg")
        
        if not os.path.exists(crop_frame_dir):
            print(f"跳过 {video_folder}: crop_frame目录不存在")
            continue
        
        print(f"\n处理视频: {video_folder}")
        if convert_frames_to_wsg(crop_frame_dir, wsg_dir, max_workers, use_multiprocessing):
            total_success += 1
    
    print(f"\n数据集处理完成! 成功: {total_success}/{len(video_folders)}")


def main():
    """主函数"""
    # ==================== 配置区域 ====================
    dataset_name = "test_dataset"
    #dataset_name = "AVIS_train"
    
    # 并行处理配置
    ENABLE_PARALLEL = True
    AUTO_DETECT_WORKERS = True
    MANUAL_WORKERS = 8  # 手动指定工作器数量
    USE_MULTIPROCESSING = None  # None=自动, True=强制多进程, False=强制多线程
    
    print("=" * 50)
    print("WSG格式转换器")
    print("=" * 50)
    
    # 系统检测
    if AUTO_DETECT_WORKERS:
        print("\n🔍 检测系统资源...")
        config = get_optimal_workers()
        max_workers = config['optimal_workers']
        
        if USE_MULTIPROCESSING is None:
            use_multiprocessing = config['use_multiprocessing']
        else:
            use_multiprocessing = USE_MULTIPROCESSING
            
        print(f"✅ 自动配置: {max_workers}个工作器")
        
        # 系统繁忙时询问
        if config['system_status'] == 'busy':
            print(f"⚠️  系统繁忙，建议使用保守配置: {config['conservative_workers']}个工作器")
            response = input("使用保守配置？(y/N): ").strip().lower()
            if response == 'y':
                max_workers = config['conservative_workers']
    else:
        max_workers = MANUAL_WORKERS if ENABLE_PARALLEL else None
        use_multiprocessing = USE_MULTIPROCESSING if USE_MULTIPROCESSING is not None else False
        print(f"⚙️  手动配置: {max_workers}个工作器")
    
    # 开始处理
    dataset_results_dir = f"../output/{dataset_name}_results"
    print(f"\n📁 数据集: {dataset_name}")
    print(f"📂 处理目录: {dataset_results_dir}")
    print(f"💾 输出格式: .wsg文件（压缩NPZ内容）")
    
    try:
        print(f"\n🚀 开始转换...")
        start_time = time.time()
        
        if ENABLE_PARALLEL:
            process_dataset(dataset_results_dir, max_workers, use_multiprocessing)
        else:
            process_dataset(dataset_results_dir, max_workers=None, use_multiprocessing=False)
        
        duration = time.time() - start_time
        print(f"\n✅ 转换完成! 耗时: {duration:.2f}秒")
        
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()