import os
import numpy as np
import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

def load_wsg_data(wsg_path):
    """加载WSG数据"""
    try:
        # WSG文件实际上是压缩的NPZ格式
        data = np.load(wsg_path, allow_pickle=True)
        return data['wsg_data']
    except Exception as e:
        print(f"加载WSG数据失败 {wsg_path}: {str(e)}")
        return None

def save_wsg_data(wsg_array, wsg_path):
    """保存WSG数据"""
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(wsg_path), exist_ok=True)
        
        # 保存为压缩NPZ格式，但文件扩展名为.wsg
        base_path = wsg_path[:-4] if wsg_path.endswith('.wsg') else wsg_path
        np.savez_compressed(base_path, wsg_data=wsg_array)
        
        # 重命名为.wsg
        npz_path = base_path + '.npz'
        final_wsg_path = base_path + '.wsg'
        
        if os.path.exists(npz_path):
            if os.path.exists(final_wsg_path):
                os.remove(final_wsg_path)
            os.rename(npz_path, final_wsg_path)
        
        return True
    except Exception as e:
        print(f"保存WSG数据失败 {wsg_path}: {str(e)}")
        return False

def inject_segmentation_to_wsg(segmentation_data, wsg_array):
    """
    将MediaPipe segmentation数据注入到WSG数组中
    
    WSG通道分配：
    - channel 103: MediaPipe multi-class segmentation (0-5类别)
    
    MediaPipe分割类别：
    0 - background
    1 - hair
    2 - body-skin
    3 - face-skin
    4 - clothes
    5 - others (accessories)
    """
    if segmentation_data is None or wsg_array is None:
        return False
    
    height, width = wsg_array.shape[:2]
    
    # 记录注入前的状态
    before_inject = np.sum(wsg_array[:, :, 103] != -1)
    print(f"  注入前通道103非-1像素数: {before_inject}")
    
    # 确保分割数据大小匹配
    if segmentation_data.shape != (height, width):
        print(f"  警告: 分割数据尺寸 {segmentation_data.shape} 与WSG尺寸 {(height, width)} 不匹配")
        # 尝试resize
        from PIL import Image
        seg_img = Image.fromarray(segmentation_data.astype(np.uint8))
        seg_img_resized = seg_img.resize((width, height), Image.NEAREST)
        segmentation_data = np.array(seg_img_resized)
    
    # inject segmentation data to channel 103
    wsg_array[:, :, 103] = segmentation_data.astype(np.float32)

    # 记录注入后的状态
    after_inject = np.sum(wsg_array[:, :, 103] != -1)
    print(f"  注入后通道103非-1像素数: {after_inject}")
    print(f"  实际注入像素数: {after_inject - before_inject}")
    
    # 统计各类别像素数
    unique, counts = np.unique(segmentation_data, return_counts=True)
    print(f"  分割类别分布: {dict(zip(unique, counts))}")
    
    return after_inject > before_inject

def process_single_file(args):
    segmentation_path, wsg_path = args
    
    try:
        # 加载数据
        seg_data = np.load(segmentation_path)
        # MediaPipe segmentation数据存储在'segmentation_mask'键中
        if 'segmentation_mask' in seg_data:
            segmentation_mask = seg_data['segmentation_mask']
        else:
            print(f"  跳过: 未找到segmentation_mask数据")
            return False
            
        wsg_array = load_wsg_data(wsg_path)
        
        if segmentation_mask is None:
            print(f"  跳过: Segmentation数据加载失败")
            return False
            
        if wsg_array is None:
            print(f"  跳过: WSG数据加载失败")
            return False
        
        # 注入Segmentation数据
        if inject_segmentation_to_wsg(segmentation_mask, wsg_array):
            # 保存更新后的WSG文件
            if save_wsg_data(wsg_array, wsg_path):
                return True
        
        return False
        
    except Exception as e:
        print(f"  处理文件失败: {str(e)}")
        return False

def inject_segmentation_to_video(video_dir, segmentation_subdir, wsg_subdir, max_workers=4):
    """为单个视频目录注入MediaPipe segmentation数据"""
    segmentation_dir = os.path.join(video_dir, segmentation_subdir)
    wsg_dir = os.path.join(video_dir, wsg_subdir)
    
    # 检查目录是否存在
    if not os.path.exists(segmentation_dir):
        print(f"  Segmentation目录不存在: {segmentation_dir}")
        return False
        
    if not os.path.exists(wsg_dir):
        print(f"  WSG目录不存在: {wsg_dir}")
        return False
    
    # 获取所有NPZ文件
    segmentation_files = glob.glob(os.path.join(segmentation_dir, "*.npz"))
    segmentation_files.sort()
    
    if not segmentation_files:
        print(f"  没有找到Segmentation NPZ文件")
        return False
    
    
    # 准备任务
    tasks = []
    for seg_path in segmentation_files:
        filename = os.path.basename(seg_path)
        wsg_filename = filename.replace('.npz', '.wsg')
        wsg_path = os.path.join(wsg_dir, wsg_filename)
        
        if os.path.exists(wsg_path):
            tasks.append((seg_path, wsg_path))
    
    if not tasks:
        print(f"  没有找到匹配的WSG文件")
        return False
    
    print(f"  找到 {len(tasks)} 个文件需要处理")
    
    # 并行处理
    success_count = 0
    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_single_file, task) for task in tasks]
            
            for i, future in enumerate(tqdm(futures, desc="  注入MediaPipe Segmentation")):
                try:
                    if future.result():
                        success_count += 1
                except Exception as e:
                    print(f"  任务失败: {str(e)}")
    else:
        # 单线程处理
        for task in tqdm(tasks, desc="  注入MediaPipe Segmentation"):
            if process_single_file(task):
                success_count += 1
    
    print(f"  完成! 成功: {success_count}/{len(tasks)}")
    return success_count > 0

def process_dataset(dataset_results_dir, segmentation_subdir, wsg_subdir, max_workers=4):
    """处理整个数据集"""
    if not os.path.exists(dataset_results_dir):
        print(f"数据集目录不存在: {dataset_results_dir}")
        return
    
    # 获取所有视频文件夹
    # video folder == video name
    video_folders = [f for f in os.listdir(dataset_results_dir) 
                    if os.path.isdir(os.path.join(dataset_results_dir, f))]
    
    # 可以选择只处理特定视频，或处理所有视频
    # video_folders = ["human_signer_480"]  # 只处理特定视频
    
    if not video_folders:
        print("没有找到视频文件夹")
        return
    
    print(f"找到 {len(video_folders)} 个视频文件夹")
    
    success_count = 0
    for video_folder in tqdm(video_folders, desc="处理视频"):
        video_dir = os.path.join(dataset_results_dir, video_folder)
        print(f"\n处理视频: {video_folder}")
        
        if inject_segmentation_to_video(video_dir, segmentation_subdir, wsg_subdir, max_workers):
            success_count += 1
    
    print(f"\n数据集处理完成! 成功: {success_count}/{len(video_folders)}")

def main():
    """主函数"""
    # ==================== 配置区域 ====================
    # 数据集配置
    dataset_name = "test_dataset"
    #dataset_name = "AVIS_train"
    base_output_dir = "../output"
    
    # 路径配置
    segmentation_subdir = "results_mediapipe_segmentation/npz"  # MediaPipe segmentation的NPZ文件路径
    wsg_subdir = "wsg"                                          # WSG文件的相对路径
    
    # 处理配置
    max_workers = mp.cpu_count() // 2     # 使用一半的CPU核心
    
    # 构建完整路径
    dataset_results_dir = os.path.join(base_output_dir, f"{dataset_name}_results")
    
    print("=" * 50)
    print("MediaPipe Segmentation到WSG注入器")
    print("=" * 50)
    print(f"数据集: {dataset_name}")
    print(f"数据集路径: {dataset_results_dir}")
    print(f"Segmentation子路径: {segmentation_subdir}")
    print(f"WSG子路径: {wsg_subdir}")
    print(f"目标通道: 103")
    print(f"工作器数量: {max_workers}")
    print("=" * 50)
    
    try:
        process_dataset(dataset_results_dir, segmentation_subdir, wsg_subdir, max_workers)
        print("\n✅ 所有注入任务完成!")
        
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()