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

def inject_curvature_to_wsg(curvature_data, wsg_array):
    """
    将Curvature数据注入到WSG数组中
    
    WSG通道分配：
    - channel 39: curvature
    """
    if curvature_data is None or wsg_array is None:
        return False
    
    height, width = wsg_array.shape[:2]
    
    # 记录注入前的状态
    before_inject = np.sum(wsg_array[:, :, 39] != -1)
    print(f"  注入前通道39非-1像素数: {before_inject}")
    
    # inject Curvature
    wsg_array[:, :, 39] = curvature_data

    # 记录注入后的状态
    after_inject = np.sum(wsg_array[:, :, 39] != -1)
    print(f"  注入后通道39非-1像素数: {after_inject}")
    print(f"  实际注入像素数: {after_inject - before_inject}")
    
    return after_inject > before_inject

def process_single_file(args):
    curvature_path, wsg_path = args
    
    try:
        # 加载数据
        curvature_data = np.load(curvature_path)['curvature_data']
        
        wsg_array = load_wsg_data(wsg_path)
        
        if curvature_data is None:
            print(f"  跳过: Curvature 数据加载失败")
            return False
            
        if wsg_array is None:
            print(f"  跳过: WSG数据加载失败")
            return False
        
        # 注入curvature数据
        if inject_curvature_to_wsg(curvature_data, wsg_array):
            # 保存更新后的WSG文件
            if save_wsg_data(wsg_array, wsg_path):
                return True
        
        return False
        
    except Exception as e:
        print(f"  处理文件失败: {str(e)}")
        return False

def inject_curvature_to_video_batch(video_dir, curvature_subdir, wsg_subdir, max_workers=4):
    """批量处理视频目录中的所有文件，优化I/O"""
    curvature_dir = os.path.join(video_dir, curvature_subdir)
    wsg_dir = os.path.join(video_dir, wsg_subdir)
    
    # 获取所有文件对
    curvature_files = sorted(glob.glob(os.path.join(curvature_dir, "*.npz")))
    if not curvature_files:
        print(f"  没有找到curvature NPZ文件")
        return False
    
    # 批量处理 - 预加载多个文件到内存
    batch_size = min(50, len(curvature_files))  # 每批处理50个文件
    total_success = 0
    
    for batch_start in range(0, len(curvature_files), batch_size):
        batch_end = min(batch_start + batch_size, len(curvature_files))
        batch_files = curvature_files[batch_start:batch_end]
        
        # 预加载批次数据
        batch_data = []
        for curvature_path in batch_files:
            filename = os.path.basename(curvature_path)
            wsg_filename = filename.replace('.npz', '.wsg')
            wsg_path = os.path.join(wsg_dir, wsg_filename)
            
            if os.path.exists(wsg_path):
                try:
                    # 加载曲率数据
                    curvature_data = np.load(curvature_path, mmap_mode='r')['curvature_data']
                    batch_data.append({
                        'curvature_path': curvature_path,
                        'wsg_path': wsg_path,
                        'curvature_data': curvature_data
                    })
                except Exception as e:
                    print(f"  加载失败 {curvature_path}: {e}")
        
        # 并行处理批次
        if batch_data:
            if max_workers > 1:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = []
                    for item in batch_data:
                        future = executor.submit(process_batch_item, item)
                        futures.append(future)
                    
                    for future in futures:
                        if future.result():
                            total_success += 1
            else:
                for item in batch_data:
                    if process_batch_item(item):
                        total_success += 1
        
        print(f"  处理批次 {batch_start//batch_size + 1}/{(len(curvature_files) + batch_size - 1)//batch_size}")
    
    print(f"  完成! 成功: {total_success}/{len(curvature_files)}")
    return total_success > 0

def process_batch_item(item):
    """处理批次中的单个项目"""
    try:
        wsg_array = load_wsg_data(item['wsg_path'])
        if wsg_array is None:
            return False
            
        # 直接注入已加载的数据
        wsg_array[:, :, 39] = item['curvature_data']
        
        # 异步保存
        return save_wsg_data(wsg_array, item['wsg_path'])
    except Exception as e:
        print(f"  处理失败 {item['wsg_path']}: {e}")
        return False

def inject_curvature_to_video(video_dir, curvature_subdir, wsg_subdir, max_workers=4):
    """为单个视频目录注入curvature数据（向后兼容）"""
    return inject_curvature_to_video_batch(video_dir, curvature_subdir, wsg_subdir, max_workers)
    
def inject_curvature_to_video_original(video_dir, curvature_subdir, wsg_subdir, max_workers=4):
    """原始版本 - 为单个视频目录注入curvature数据"""
    curvature_dir = os.path.join(video_dir, curvature_subdir)
    wsg_dir = os.path.join(video_dir, wsg_subdir)
    
    # 检查目录是否存在
    if not os.path.exists(curvature_dir):
        print(f"  curvature目录不存在: {curvature_dir}")
        return False
        
    if not os.path.exists(wsg_dir):
        print(f"  WSG目录不存在: {wsg_dir}")
        return False
    
    # 获取所有NPZ文件
    curvature_files = glob.glob(os.path.join(curvature_dir, "*.npz"))
    curvature_files.sort()
    
    if not curvature_files:
        print(f"  没有找到curvature NPZ文件")
        return False
    
    
    # 准备任务
    tasks = []
    for curvature_path in curvature_files:
        filename = os.path.basename(curvature_path)
        wsg_filename = filename.replace('.npz', '.wsg')
        wsg_path = os.path.join(wsg_dir, wsg_filename)
        
        if os.path.exists(wsg_path):
            tasks.append((curvature_path, wsg_path))
    
    if not tasks:
        print(f"  没有找到匹配的WSG文件")
        return False
    
    print(f"  找到 {len(tasks)} 个文件需要处理")
    
    # 并行处理
    success_count = 0
    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_single_file, task) for task in tasks]
            
            for i, future in enumerate(tqdm(futures, desc="  注入Curvature")):
                try:
                    if future.result():
                        success_count += 1
                except Exception as e:
                    print(f"  任务失败: {str(e)}")
    else:
        # 单线程处理
        for task in tqdm(tasks, desc="  注入Curvature"):
            if process_single_file(task):
                success_count += 1
    
    print(f"  完成! 成功: {success_count}/{len(tasks)}")
    return success_count > 0

def process_dataset(dataset_results_dir, feature_subdir, wsg_subdir, max_workers=4):
    """处理整个数据集"""
    if not os.path.exists(dataset_results_dir):
        print(f"数据集目录不存在: {dataset_results_dir}")
        return
    
    # 获取所有视频文件夹
    # video folder == video name
    video_folders = [f for f in os.listdir(dataset_results_dir) 
                    if os.path.isdir(os.path.join(dataset_results_dir, f))]
    #video_folders = ["human_signer_480"]
    
    if not video_folders:
        print("没有找到视频文件夹")
        return
    
    print(f"找到 {len(video_folders)} 个视频文件夹")
    
    success_count = 0
    for video_folder in tqdm(video_folders, desc="处理视频"):
        video_dir = os.path.join(dataset_results_dir, video_folder)
        print(f"\n处理视频: {video_folder}")
        
        if inject_curvature_to_video(video_dir, feature_subdir, wsg_subdir, max_workers):
            success_count += 1
    
    print(f"\n数据集处理完成! 成功: {success_count}/{len(video_folders)}")

def main():
    """主函数"""
    # ==================== 配置区域 ====================
    # 数据集配置
    #dataset_name = "test_dataset"
    dataset_name = "AVIS_train"
    base_output_dir = "../output"
    # 路径配置
    curvature_subdir = "results_curvature/npz"  
    wsg_subdir = "wsg"                    # WSG文件的相对路径
    
    # 处理配置
    max_workers = mp.cpu_count()     # 使用一半的CPU核心
    
    # 构建完整路径
    dataset_results_dir = os.path.join(base_output_dir, f"{dataset_name}_results")
    
    print("=" * 50)
    print("Curvature到WSG注入器")
    print("=" * 50)
    print(f"数据集: {dataset_name}")
    print(f"数据集路径: {dataset_results_dir}")
    print(f"curvature子路径: {curvature_subdir}")
    print(f"WSG子路径: {wsg_subdir}")
    print(f"工作器数量: {max_workers}")
    print("=" * 50)
    
    try:
        process_dataset(dataset_results_dir, curvature_subdir, wsg_subdir, max_workers)
        print("\n✅ 所有注入任务完成!")
        
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()