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

def inject_depth_to_wsg(depth_data, wsg_array):
    """
    将relative depth数据注入到WSG数组中
    
    WSG通道分配：
    - channel 32: relative depth
    - channel 33: relative depth confidence (not used)
    """
    if depth_data is None or wsg_array is None:
        return False
    
    height, width = wsg_array.shape[:2]
    
    # 记录注入前的状态
    before_inject = np.sum(wsg_array[:, :, 32] != -1)
    print(f"  注入前通道32非-1像素数: {before_inject}")
    
    # inject relative depth
    wsg_array[:, :, 32] = depth_data

    # 记录注入后的状态
    after_inject = np.sum(wsg_array[:, :, 32] != -1)
    print(f"  注入后通道32非-1像素数: {after_inject}")
    print(f"  实际注入像素数: {after_inject - before_inject}")
    
    return after_inject > before_inject

def process_single_file(args):
    primedepth_path, wsg_path = args
    
    try:
        # 加载数据
        primedepth_data = np.load(primedepth_path)['depth_data']
        wsg_array = load_wsg_data(wsg_path)
        
        if primedepth_data is None:
            print(f"  跳过: Relative Depth数据加载失败")
            return False
            
        if wsg_array is None:
            print(f"  跳过: WSG数据加载失败")
            return False
        
        # 注入DWPose数据
        if inject_depth_to_wsg(primedepth_data, wsg_array):
            # 保存更新后的WSG文件
            if save_wsg_data(wsg_array, wsg_path):
                return True
        
        return False
        
    except Exception as e:
        print(f"  处理文件失败: {str(e)}")
        return False

def inject_primedepth_to_video(video_dir, primedepth_subdir, wsg_subdir, max_workers=4):
    """为单个视频目录注入DWPose数据"""
    primedepth_dir = os.path.join(video_dir, primedepth_subdir)
    wsg_dir = os.path.join(video_dir, wsg_subdir)
    
    # 检查目录是否存在
    if not os.path.exists(primedepth_dir):
        print(f"  PrimeDepth目录不存在: {primedepth_dir}")
        return False
        
    if not os.path.exists(wsg_dir):
        print(f"  WSG目录不存在: {wsg_dir}")
        return False
    
    # 获取所有NPZ文件
    primedepth_files = glob.glob(os.path.join(primedepth_dir, "*.npz"))
    primedepth_files.sort()
    
    if not primedepth_files:
        print(f"  没有找到DWPose NPZ文件")
        return False
    
    
    # 准备任务
    tasks = []
    for primedepth_path in primedepth_files:
        filename = os.path.basename(primedepth_path)
        wsg_filename = filename.replace('.npz', '.wsg')
        wsg_path = os.path.join(wsg_dir, wsg_filename)
        
        if os.path.exists(wsg_path):
            tasks.append((primedepth_path, wsg_path))
    
    if not tasks:
        print(f"  没有找到匹配的WSG文件")
        return False
    
    print(f"  找到 {len(tasks)} 个文件需要处理")
    
    # 并行处理
    success_count = 0
    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_single_file, task) for task in tasks]
            
            for i, future in enumerate(tqdm(futures, desc="  注入Relative Depth")):
                try:
                    if future.result():
                        success_count += 1
                except Exception as e:
                    print(f"  任务失败: {str(e)}")
    else:
        # 单线程处理
        for task in tqdm(tasks, desc="  注入Relative Depth"):
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
    # video_folders = ["human_signer_480"]  # 注释掉硬编码的测试代码
    
    if not video_folders:
        print("没有找到视频文件夹")
        return
    
    print(f"找到 {len(video_folders)} 个视频文件夹")
    
    success_count = 0
    for video_folder in tqdm(video_folders, desc="处理视频"):
        video_dir = os.path.join(dataset_results_dir, video_folder)
        print(f"\n处理视频: {video_folder}")
        
        if inject_primedepth_to_video(video_dir, feature_subdir, wsg_subdir, max_workers):
            success_count += 1
    
    print(f"\n数据集处理完成! 成功: {success_count}/{len(video_folders)}")

def main():
    """主函数"""
    # ==================== 配置区域 ====================
    # 数据集配置
    dataset_name = "test_dataset"
    #dataset_name = "AVIS_test"
    base_output_dir = "../output"
    # 路径配置
    primedepth_subdir = "results_primedepth_depth/npz"  # PrimeDepth NPZ文件的相对路径
    wsg_subdir = "wsg"                    # WSG文件的相对路径
    
    # 处理配置
    max_workers = mp.cpu_count()      # 使用一半的CPU核心
    
    # 构建完整路径
    dataset_results_dir = os.path.join(base_output_dir, f"{dataset_name}_results")
    
    print("=" * 50)
    print("PrimeDepth到WSG注入器")
    print("=" * 50)
    print(f"数据集: {dataset_name}")
    print(f"数据集路径: {dataset_results_dir}")
    print(f"Relative Depth子路径: {primedepth_subdir}")
    print(f"WSG子路径: {wsg_subdir}")
    print(f"工作器数量: {max_workers}")
    print("=" * 50)
    
    try:
        process_dataset(dataset_results_dir, primedepth_subdir, wsg_subdir, max_workers)
        print("\n✅ 所有注入任务完成!")
        
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()