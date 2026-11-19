import os
import numpy as np
import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

# mediapipe 33点的keypoint映射
MEDIAPIPE_KEYPOINT_MAPPING = {f"pose_{i}": i for i in range(33)}

# 注入到WSG的通道
PERSON_ID_CHANNEL = 95
KEYPOINT_ID_CHANNEL = 96
Z_CHANNEL = 97
SCORE_CHANNEL = 98

def load_mediapipe_data(npz_path):
    try:
        data = np.load(npz_path, allow_pickle=True)
        return dict(data)
    except Exception as e:
        print(f"加载mediapipe数据失败 {npz_path}: {str(e)}")
        return None

def load_wsg_data(wsg_path):
    try:
        data = np.load(wsg_path, allow_pickle=True)
        return data['wsg_data']
    except Exception as e:
        print(f"加载WSG数据失败 {wsg_path}: {str(e)}")
        return None

def save_wsg_data(wsg_array, wsg_path):
    try:
        os.makedirs(os.path.dirname(wsg_path), exist_ok=True)
        base_path = wsg_path[:-4] if wsg_path.endswith('.wsg') else wsg_path
        np.savez_compressed(base_path, wsg_data=wsg_array)
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

def inject_mediapipe_to_wsg(mediapipe_data, wsg_array, keypoint_mapping, confidence_threshold=0.1, clear_existing=True):
    """
    将mediapipe数据注入到WSG数组中
    
    Args:
        mediapipe_data: mediapipe检测结果
        wsg_array: WSG数组
        keypoint_mapping: 关键点映射
        confidence_threshold: 置信度阈值
        clear_existing: 是否清除现有的mediapipe数据
    """
    if mediapipe_data is None or wsg_array is None:
        return False
    
    height, width = wsg_array.shape[:2]
    
    # 如果需要，先清除现有的mediapipe数据
    if clear_existing:
        print(f"    [INFO] 清除现有的mediapipe数据...")
        wsg_array[:, :, PERSON_ID_CHANNEL] = -1
        wsg_array[:, :, KEYPOINT_ID_CHANNEL] = -1
        wsg_array[:, :, SCORE_CHANNEL] = -1
        wsg_array[:, :, Z_CHANNEL] = -1
    
    before_inject = np.sum(wsg_array[:, :, PERSON_ID_CHANNEL] != -1)
    injection_count = 0
    
    # 多人处理
    for person_id, keypoints in mediapipe_data.items():
        if not isinstance(keypoints, np.ndarray) or keypoints.shape[1] < 4:
            continue
            
        person_num = int(person_id.split('_')[-1])
        
        for kp_idx in range(min(33, keypoints.shape[0])):
            x, y, z, v = keypoints[kp_idx]
            
            if v < confidence_threshold:
                continue
                
            px = int(x * width)
            py = int(y * height)
            
            if 0 <= px < width and 0 <= py < height:
                # 注入数据
                wsg_array[py, px, PERSON_ID_CHANNEL] = person_num
                wsg_array[py, px, KEYPOINT_ID_CHANNEL] = kp_idx
                wsg_array[py, px, SCORE_CHANNEL] = float(v)
                wsg_array[py, px, Z_CHANNEL] = float(z)
                injection_count += 1
    
    after_inject = np.sum(wsg_array[:, :, PERSON_ID_CHANNEL] != -1)
    
    print(f"    [INFO] 注入统计: 注入前={before_inject}, 注入后={after_inject}, 新增={after_inject-before_inject}, 总注入次数={injection_count}")
    
    # 修改成功判断条件：只要注入了数据就算成功
    return injection_count > 0

def process_single_file(args):
    mediapipe_path, wsg_path, keypoint_mapping, confidence_threshold, clear_existing = args
    try:
        mediapipe_data = load_mediapipe_data(mediapipe_path)
        wsg_array = load_wsg_data(wsg_path)
        
        if mediapipe_data is None or wsg_array is None:
            return False
        
        if inject_mediapipe_to_wsg(mediapipe_data, wsg_array, keypoint_mapping, confidence_threshold, clear_existing):
            return save_wsg_data(wsg_array, wsg_path)
        return False
    except Exception as e:
        print(f"处理文件失败: {str(e)}")
        return False

def inject_mediapipe_to_video(video_dir, mediapipe_subdir, wsg_subdir, max_workers=4, confidence_threshold=0.1, clear_existing=True):
    mediapipe_dir = os.path.join(video_dir, mediapipe_subdir, "npz")
    wsg_dir = os.path.join(video_dir, wsg_subdir)
    
    if not os.path.exists(mediapipe_dir):
        print(f"  mediapipe目录不存在: {mediapipe_dir}")
        return False
    if not os.path.exists(wsg_dir):
        print(f"  WSG目录不存在: {wsg_dir}")
        return False
    
    mediapipe_files = glob.glob(os.path.join(mediapipe_dir, "*.npz"))
    mediapipe_files.sort()
    
    if not mediapipe_files:
        print(f"  没有找到mediapipe NPZ文件")
        return False
    
    keypoint_mapping = MEDIAPIPE_KEYPOINT_MAPPING
    tasks = []
    
    for mediapipe_path in mediapipe_files:
        filename = os.path.basename(mediapipe_path)
        wsg_filename = filename.replace('.npz', '.wsg')
        wsg_path = os.path.join(wsg_dir, wsg_filename)
        
        if os.path.exists(wsg_path):
            tasks.append((mediapipe_path, wsg_path, keypoint_mapping, confidence_threshold, clear_existing))
    
    if not tasks:
        print(f"  没有找到匹配的WSG文件")
        return False
    
    print(f"  找到 {len(tasks)} 个文件需要处理")
    
    success_count = 0
    
    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_single_file, task) for task in tasks]
            for future in tqdm(futures, desc="  注入mediapipe"):
                try:
                    if future.result():
                        success_count += 1
                except Exception as e:
                    print(f"任务失败: {str(e)}")
    else:
        for task in tqdm(tasks, desc="  注入mediapipe"):
            if process_single_file(task):
                success_count += 1
    
    print(f"  完成! 成功: {success_count}/{len(tasks)}")
    return success_count > 0

def process_dataset(dataset_results_dir, mediapipe_subdir, wsg_subdir, max_workers=4, confidence_threshold=0.1, clear_existing=True):
    if not os.path.exists(dataset_results_dir):
        print(f"数据集目录不存在: {dataset_results_dir}")
        return
    
    video_folders = [f for f in os.listdir(dataset_results_dir) if os.path.isdir(os.path.join(dataset_results_dir, f))]
    
    if not video_folders:
        print("没有找到视频文件夹")
        return
    
    print(f"找到 {len(video_folders)} 个视频文件夹")
    
    success_count = 0
    for video_folder in tqdm(video_folders, desc="处理视频"):
        video_dir = os.path.join(dataset_results_dir, video_folder)
        print(f"\n处理视频: {video_folder}")
        
        if inject_mediapipe_to_video(video_dir, mediapipe_subdir, wsg_subdir, max_workers, confidence_threshold, clear_existing):
            success_count += 1
    
    print(f"\n数据集处理完成! 成功: {success_count}/{len(video_folders)}")

def main():
    dataset_name = "test_dataset"
    #dataset_name = "AVIS_test"
    base_output_dir = "../output"
    mediapipe_subdir = "results_mediapipe"  # mediapipe输出的子目录
    wsg_subdir = "wsg"                      # WSG文件的子目录
    max_workers = mp.cpu_count() // 2
    confidence_threshold = 0.1
    clear_existing = False  # 是否清除现有的mediapipe数据
    
    dataset_results_dir = os.path.join(base_output_dir, f"{dataset_name}_results")
    
    print("=" * 50)
    print("修复版本 - mediapipe到WSG注入器")
    print("=" * 50)
    print(f"数据集: {dataset_name}")
    print(f"数据集路径: {dataset_results_dir}")
    print(f"mediapipe子路径: {mediapipe_subdir}")
    print(f"WSG子路径: {wsg_subdir}")
    print(f"工作器数量: {max_workers}")
    print(f"置信度阈值: {confidence_threshold}")
    print(f"清除现有数据: {'是' if clear_existing else '否'}")
    print("=" * 50)
    
    try:
        process_dataset(dataset_results_dir, mediapipe_subdir, wsg_subdir, max_workers, confidence_threshold, clear_existing)
        print("\n✅ 所有注入任务完成!")
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()