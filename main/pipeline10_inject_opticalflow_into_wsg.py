import os
import numpy as np
import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import time

def load_wsg_data(wsg_path):
    """加载WSG数据"""
    try:
        # WSG文件实际上是压缩的NPZ格式
        # 使用mmap_mode='r'可以减少内存使用
        data = np.load(wsg_path, allow_pickle=True, mmap_mode='r')
        return data['wsg_data'].copy()  # 复制以便修改
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

def inject_opticalflow_to_wsg(feature_data, wsg_array, verbose=False):
    """
    将opticalflow数据注入到WSG数组中
    
    WSG通道分配：
    - channel 41-42: optical flow x, y
    """
    if feature_data is None or wsg_array is None:
        return False
    
    # 直接注入，不计算统计信息（除非verbose模式）
    if verbose:
        before_inject = np.sum(wsg_array[:, :, 41:43] != -1)
    
    # inject opticalflow - 使用更快的内存复制
    wsg_array[:, :, 41:43] = feature_data
    
    if verbose:
        after_inject = np.sum(wsg_array[:, :, 41:43] != -1)
        print(f"  注入前/后通道41-42非-1像素数: {before_inject} / {after_inject}")
        print(f"  实际注入像素数: {after_inject - before_inject}")
        return after_inject > before_inject
    
    return True

def process_single_file(args):
    opticalflow_path, wsg_path = args
    
    try:
        # 加载数据
        opticalflow_data = np.load(opticalflow_path, mmap_mode='r')['flow_data']
        wsg_array = load_wsg_data(wsg_path)
        
        if opticalflow_data is None or wsg_array is None:
            return False
        
        # 注入opticalflow数据（关闭verbose以提高速度）
        if inject_opticalflow_to_wsg(opticalflow_data, wsg_array, verbose=False):
            # 保存更新后的WSG文件
            if save_wsg_data(wsg_array, wsg_path):
                return True
        
        return False
        
    except Exception as e:
        # 仅在出错时打印
        print(f"  处理文件失败 {opticalflow_path}: {str(e)}")
        return False

def process_batch_files(tasks_batch):
    """批量处理文件，减少进程开销"""
    success_count = 0
    for task in tasks_batch:
        if process_single_file(task):
            success_count += 1
    return success_count

def inject_opticalflow_to_video(video_dir, opticalflow_subdir, wsg_subdir, max_workers=4):
    """为单个视频目录注入opticalflow数据"""
    opticalflow_dir = os.path.join(video_dir, opticalflow_subdir)
    wsg_dir = os.path.join(video_dir, wsg_subdir)
    
    # 检查目录是否存在
    if not os.path.exists(opticalflow_dir):
        print(f"  OpticalFlow目录不存在: {opticalflow_dir}")
        return False
        
    if not os.path.exists(wsg_dir):
        print(f"  WSG目录不存在: {wsg_dir}")
        return False
    
    # 获取所有NPZ文件
    opticalflow_files = glob.glob(os.path.join(opticalflow_dir, "*.npz"))
    opticalflow_files.sort()
    
    if not opticalflow_files:
        print(f"  没有找到Optical Flow NPZ文件")
        return False
    
    # 准备任务
    tasks = []
    for opticalflow_path in opticalflow_files:
        filename = os.path.basename(opticalflow_path)
        wsg_filename = filename.replace('.npz', '.wsg')
        wsg_path = os.path.join(wsg_dir, wsg_filename)
        
        if os.path.exists(wsg_path):
            tasks.append((opticalflow_path, wsg_path))
    
    if not tasks:
        print(f"  没有找到匹配的WSG文件")
        return False
    
    print(f"  找到 {len(tasks)} 个文件需要处理")
    
    # 记录开始时间
    start_time = time.time()
    
    # 将任务分批处理，每批处理多个文件
    batch_size = max(1, len(tasks) // (max_workers * 4))  # 每个worker处理4批
    task_batches = [tasks[i:i + batch_size] for i in range(0, len(tasks), batch_size)]
    
    # 并行处理批次
    success_count = 0
    if max_workers > 1 and len(task_batches) > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_batch_files, batch) for batch in task_batches]
            
            with tqdm(total=len(tasks), desc="  注入opticalflow") as pbar:
                for future in futures:
                    try:
                        batch_success = future.result()
                        success_count += batch_success
                        pbar.update(len(task_batches[0]))  # 更新进度条
                    except Exception as e:
                        print(f"  批次处理失败: {str(e)}")
    else:
        # 单线程处理
        for task in tqdm(tasks, desc="  注入opticalflow"):
            if process_single_file(task):
                success_count += 1
    
    elapsed_time = time.time() - start_time
    print(f"  完成! 成功: {success_count}/{len(tasks)} (耗时: {elapsed_time:.1f}秒)")
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
        
        if inject_opticalflow_to_video(video_dir, feature_subdir, wsg_subdir, max_workers):
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
    opticalflow_subdir = "results_opticalflow/npz"  
    wsg_subdir = "wsg"                    # WSG文件的相对路径
    
    # 处理配置
    max_workers = mp.cpu_count()     # 使用一半的CPU核心
    
    # 构建完整路径
    dataset_results_dir = os.path.join(base_output_dir, f"{dataset_name}_results")
    
    print("=" * 50)
    print("Optical Flow到WSG注入器")
    print("=" * 50)
    print(f"数据集: {dataset_name}")
    print(f"数据集路径: {dataset_results_dir}")
    print(f"Optical Flow子路径: {opticalflow_subdir}")
    print(f"WSG子路径: {wsg_subdir}")
    print(f"工作器数量: {max_workers}")
    print("=" * 50)
    
    try:
        process_dataset(dataset_results_dir, opticalflow_subdir, wsg_subdir, max_workers)
        print("\n✅ 所有注入任务完成!")
        
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()