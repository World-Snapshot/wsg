import os
import sys
import numpy as np
import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp


def load_processed_video_names(dataset_results_dir):
    record_path = os.path.join(dataset_results_dir, "processed_video_name_by_normal.txt")
    if not os.path.exists(record_path):
        return None
    with open(record_path, "r") as f:
        return {line.strip() for line in f if line.strip()}


def is_video_normal_complete(video_dir, metricdepth_subdir):
    frames_dir = os.path.join(video_dir, "crop_frame")
    metricdepth_dir = os.path.join(video_dir, metricdepth_subdir)
    if not os.path.isdir(frames_dir) or not os.path.isdir(metricdepth_dir):
        return False

    frame_count = len(
        [f for f in os.listdir(frames_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    )
    npz_count = len([f for f in os.listdir(metricdepth_dir) if f.endswith(".npz")])
    return frame_count > 0 and npz_count == frame_count

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

def inject_normal_to_wsg(feature_data, wsg_array):
    """
    将normals数据注入到WSG数组中
    
    WSG通道分配：
    - channel 36-38: normal x, y, z
    """
    if feature_data is None or wsg_array is None:
        return False
    
    expected_shape = wsg_array[:, :, 36:39].shape
    if feature_data.shape != expected_shape:
        print(f"  形状不匹配: normals={feature_data.shape}, target={expected_shape}")
        return False
    
    # 记录注入前的状态
    before_inject = np.sum(wsg_array[:, :, 36:39] != -1)
    print(f"  注入前通道36-38非-1像素数: {before_inject}")
    
    # inject normals
    wsg_array[:, :, 36:39] = feature_data

    # 记录注入后的状态
    after_inject = np.sum(wsg_array[:, :, 36:39] != -1)
    print(f"  注入后通道36-38非-1像素数: {after_inject}")
    print(f"  实际注入像素数: {after_inject - before_inject}")
    
    return True

def process_single_file(args):
    metricdepth_path, wsg_path, delete_npz_after_inject = args
    
    try:
        # 加载数据
        metricdepth_data = np.load(metricdepth_path)['normal_data']
        wsg_array = load_wsg_data(wsg_path)
        
        if metricdepth_data is None:
            print(f"  跳过: normals数据加载失败")
            return False
            
        if wsg_array is None:
            print(f"  跳过: WSG数据加载失败")
            return False
        
        # 注入normals数据
        if inject_normal_to_wsg(metricdepth_data, wsg_array):
            # 保存更新后的WSG文件
            if save_wsg_data(wsg_array, wsg_path):
                if delete_npz_after_inject:
                    try:
                        os.remove(metricdepth_path)
                    except Exception as e:
                        print(f"  注入成功但删除NPZ失败 {metricdepth_path}: {str(e)}")
                return True
        
        return False
        
    except Exception as e:
        print(f"  处理文件失败: {str(e)}")
        return False

def inject_normal_to_video(
    video_dir,
    metricdepth_subdir,
    wsg_subdir,
    max_workers=4,
    delete_npz_after_inject=False,
):
    """为单个视频目录注入normals数据"""
    metricdepth_dir = os.path.join(video_dir, metricdepth_subdir)
    wsg_dir = os.path.join(video_dir, wsg_subdir)
    
    # 检查目录是否存在
    if not os.path.exists(metricdepth_dir):
        print(f"  Normal目录不存在: {metricdepth_dir}")
        return False
        
    if not os.path.exists(wsg_dir):
        print(f"  WSG目录不存在: {wsg_dir}")
        return False
    
    # 获取所有NPZ文件
    metricdepth_files = glob.glob(os.path.join(metricdepth_dir, "*.npz"))
    metricdepth_files.sort()
    
    if not metricdepth_files:
        print(f"  没有找到normals NPZ文件")
        return False
    
    
    # 准备任务
    tasks = []
    for metricdepth_path in metricdepth_files:
        filename = os.path.basename(metricdepth_path)
        wsg_filename = filename.replace('.npz', '.wsg')
        wsg_path = os.path.join(wsg_dir, wsg_filename)
        
        if os.path.exists(wsg_path):
            tasks.append((metricdepth_path, wsg_path, delete_npz_after_inject))
    
    if not tasks:
        print(f"  没有找到匹配的WSG文件")
        return False
    
    print(f"  找到 {len(tasks)} 个文件需要处理")
    
    # 并行处理
    success_count = 0
    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_single_file, task) for task in tasks]
            
            for i, future in enumerate(tqdm(futures, desc="  注入normals")):
                try:
                    if future.result():
                        success_count += 1
                except Exception as e:
                    print(f"  任务失败: {str(e)}")
    else:
        # 单线程处理
        for task in tqdm(tasks, desc="  注入normals"):
            if process_single_file(task):
                success_count += 1
    
    print(f"  完成! 成功: {success_count}/{len(tasks)}")
    return success_count > 0

def process_dataset(
    dataset_results_dir,
    feature_subdir,
    wsg_subdir,
    max_workers=4,
    delete_npz_after_inject=False,
    use_processed_record=True,
):
    """处理整个数据集"""
    if not os.path.exists(dataset_results_dir):
        print(f"数据集目录不存在: {dataset_results_dir}")
        return
    
    # 获取所有视频文件夹
    # video folder == video name
    video_folders = [
        f for f in os.listdir(dataset_results_dir)
        if os.path.isdir(os.path.join(dataset_results_dir, f))
    ]

    processed_videos = None
    if use_processed_record:
        processed_videos = load_processed_video_names(dataset_results_dir)
        if processed_videos is None:
            print("未找到 processed_video_name_by_normal.txt，将按完整性检查处理全部视频目录")
        else:
            before_filter = len(video_folders)
            video_folders = [f for f in video_folders if f in processed_videos]
            print(f"根据 processed_video_name_by_normal.txt 过滤: {before_filter} -> {len(video_folders)}")

    complete_video_folders = []
    incomplete_video_folders = []
    for video_folder in video_folders:
        video_dir = os.path.join(dataset_results_dir, video_folder)
        if is_video_normal_complete(video_dir, feature_subdir):
            complete_video_folders.append(video_folder)
        else:
            incomplete_video_folders.append(video_folder)
    video_folders = complete_video_folders

    if not video_folders:
        print("没有找到视频文件夹")
        return

    if incomplete_video_folders:
        print(f"跳过 {len(incomplete_video_folders)} 个未完整提取的视频目录")

    print(f"找到 {len(video_folders)} 个视频文件夹")
    
    success_count = 0
    for video_folder in tqdm(video_folders, desc="处理视频"):
        video_dir = os.path.join(dataset_results_dir, video_folder)
        print(f"\n处理视频: {video_folder}")
        
        if inject_normal_to_video(
            video_dir,
            feature_subdir,
            wsg_subdir,
            max_workers,
            delete_npz_after_inject,
        ):
            success_count += 1
    
    print(f"\n数据集处理完成! 成功: {success_count}/{len(video_folders)}")

def main():
    """主函数"""
    # ==================== 配置区域 ====================
    # 数据集配置
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("WSG_DATASET_NAME", "AVIS_test")
    base_output_dir = "../output"
    # 路径配置
    metricdepth_subdir = "results_normal/npz"  
    wsg_subdir = "wsg"                    # WSG文件的相对路径
    
    # 处理配置
    max_workers = mp.cpu_count() 
    delete_npz_after_inject = False
    if len(sys.argv) > 2:
        delete_npz_after_inject = sys.argv[2].lower() in {"1", "true", "yes", "y", "delete"}
    use_processed_record = True
    if len(sys.argv) > 3:
        use_processed_record = sys.argv[3].lower() not in {"0", "false", "no", "n"}
    
    # 构建完整路径
    dataset_results_dir = os.path.join(base_output_dir, f"{dataset_name}_results")
    
    print("=" * 50)
    print("normals到WSG注入器")
    print("=" * 50)
    print(f"数据集: {dataset_name}")
    print(f"数据集路径: {dataset_results_dir}")
    print(f"normals子路径: {metricdepth_subdir}")
    print(f"WSG子路径: {wsg_subdir}")
    print(f"工作器数量: {max_workers}")
    print(f"注入后删除NPZ: {delete_npz_after_inject}")
    print(f"使用 processed_video_name_by_normal.txt: {use_processed_record}")
    print("=" * 50)
    
    try:
        process_dataset(
            dataset_results_dir,
            metricdepth_subdir,
            wsg_subdir,
            max_workers,
            delete_npz_after_inject,
            use_processed_record,
        )
        print("\n✅ 所有注入任务完成!")
        
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
