import os
import numpy as np
import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp


def load_dwpose_data(npz_path):
    """加载DWPose NPZ数据"""
    try:
        data = np.load(npz_path, allow_pickle=True)
        return dict(data)
    except Exception as e:
        print(f"加载DWPose数据失败 {npz_path}: {str(e)}")
        return None


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


def get_dwpose_keypoint_mapping():
    """
    定义DWPose关键点到WSG Keypoint ID的映射
    DWPose包含：身体18点 + 面部68点 + 左手21点 + 右手21点 = 128点
    """
    mapping = {}
    current_id = 0
    
    # 身体关键点 (0-17)
    body_parts = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'neck'
    ]
    
    for i, part in enumerate(body_parts):
        mapping[f'body_{i}'] = current_id
        current_id += 1
    
    # 面部关键点 (18-85)
    for i in range(68):
        mapping[f'face_{i}'] = current_id
        current_id += 1
    
    # 左手关键点 (86-106)
    for i in range(21):
        mapping[f'left_hand_{i}'] = current_id
        current_id += 1
    
    # 右手关键点 (107-127)
    for i in range(21):
        mapping[f'right_hand_{i}'] = current_id
        current_id += 1
    
    return mapping


def inject_dwpose_to_wsg(dwpose_data, wsg_array, keypoint_mapping, confidence_threshold=0.1):
    """
    将DWPose数据注入到WSG数组中
    
    WSG通道分配：
    - 通道64: Person ID
    - 通道65: Keypoint ID  
    - 通道66: Score
    """
    if dwpose_data is None or wsg_array is None:
        return False
    
    height, width = wsg_array.shape[:2]
    
    # 记录注入前的状态
    before_inject = np.sum(wsg_array[:, :, 64] != -1)
    print(f"  注入前通道64非-1像素数: {before_inject}")
    
    # 检查数据格式：新格式(有num_persons)还是旧格式(直接存储数据)
    if 'num_persons' in dwpose_data:
        # 新格式：多人数据
        num_persons = dwpose_data['num_persons']
        print(f"  检测到 {num_persons} 个人 (新格式)")
        
        for person_id in range(num_persons):
            person_key = f'person_{person_id}'
            if person_key not in dwpose_data:
                continue
                
            # 关键修复：正确解析numpy标量数组中的字典
            person_data_raw = dwpose_data[person_key]
            if isinstance(person_data_raw, np.ndarray) and person_data_raw.shape == ():
                # 从numpy标量数组中提取字典
                person_data = person_data_raw.item()
            else:
                person_data = person_data_raw
                
            if isinstance(person_data, dict):
                process_person_data(person_data, person_id, wsg_array, keypoint_mapping, confidence_threshold, height, width)
            else:
                print(f"    警告: person_{person_id} 数据格式不正确: {type(person_data)}")
    
    else:
        # 旧格式：单人数据，直接存储在根级别
        print(f"  检测到 1 个人 (旧格式，默认为person_0)")
        person_id = 0
        person_data = dwpose_data  # 整个dwpose_data就是person数据
        process_person_data(person_data, person_id, wsg_array, keypoint_mapping, confidence_threshold, height, width)
    
    # 记录注入后的状态
    after_inject = np.sum(wsg_array[:, :, 64] != -1)
    print(f"  注入后通道64非-1像素数: {after_inject}")
    print(f"  实际注入像素数: {after_inject - before_inject}")
    
    return after_inject > before_inject


def process_person_data(person_data, person_id, wsg_array, keypoint_mapping, confidence_threshold, height, width):
    """处理单个人的姿态数据 - 简化版，直接注入所有关键点"""
    
    # 处理身体关键点
    body_keypoints = None
    body_scores = None
    
    if 'body_keypoints' in person_data and 'body_scores' in person_data:
        # 新格式
        body_keypoints = person_data['body_keypoints']
        body_scores = person_data['body_scores']
    elif 'bodies' in person_data and 'body_scores' in person_data:
        # 旧格式
        bodies = person_data['bodies']
        if len(bodies.shape) == 2 and bodies.shape[0] == 18:
            body_keypoints = bodies
        elif len(bodies.shape) == 3:
            body_keypoints = bodies[0] if bodies.shape[0] > 0 else None
        
        body_scores_raw = person_data['body_scores']
        if len(body_scores_raw.shape) == 2:
            body_scores = body_scores_raw[0]
        else:
            body_scores = body_scores_raw
    
    if body_keypoints is not None and body_scores is not None:
        valid_count = 0
        for kp_idx in range(len(body_keypoints)):
            x, y = body_keypoints[kp_idx]
            score = body_scores[kp_idx] if kp_idx < len(body_scores) else 0.0
            
            # 坐标转换
            px = int(x * width)
            py = int(y * height)
            
            # 只检查边界，不管分数是什么
            if 0 <= px < width and 0 <= py < height:
                keypoint_id = keypoint_mapping[f'body_{kp_idx}']
                
                # 直接注入，替换-1
                wsg_array[py, px, 64] = person_id
                wsg_array[py, px, 65] = keypoint_id
                wsg_array[py, px, 66] = float(score)
                valid_count += 1
        
        print(f"    身体关键点注入: {valid_count}/{len(body_keypoints)}")
    
    # 处理面部关键点
    face_keypoints = None
    face_scores = None
    
    if 'face_keypoints' in person_data and 'face_scores' in person_data:
        # 新格式
        face_keypoints = person_data['face_keypoints']
        face_scores = person_data['face_scores']
    elif 'faces' in person_data and 'faces_scores' in person_data:
        # 旧格式
        faces = person_data['faces']
        faces_scores = person_data['faces_scores']
        
        if len(faces.shape) == 3 and faces.shape[0] > 0:
            face_keypoints = faces[0]
        elif len(faces.shape) == 2:
            face_keypoints = faces
            
        if len(faces_scores.shape) == 2 and faces_scores.shape[0] > 0:
            face_scores = faces_scores[0]
        elif len(faces_scores.shape) == 1:
            face_scores = faces_scores
    
    if face_keypoints is not None and face_scores is not None:
        valid_count = 0
        for kp_idx in range(len(face_keypoints)):
            x, y = face_keypoints[kp_idx]
            score = face_scores[kp_idx] if kp_idx < len(face_scores) else 0.0
            
            px = int(x * width)
            py = int(y * height)
            
            if 0 <= px < width and 0 <= py < height:
                keypoint_id = keypoint_mapping[f'face_{kp_idx}']
                
                wsg_array[py, px, 64] = person_id
                wsg_array[py, px, 65] = keypoint_id
                wsg_array[py, px, 66] = float(score)
                valid_count += 1
        
        print(f"    面部关键点注入: {valid_count}/{len(face_keypoints)}")
    
    # 处理左手关键点
    left_hand_keypoints = None
    left_hand_scores = None
    
    if 'left_hand_keypoints' in person_data and 'left_hand_scores' in person_data:
        # 新格式
        left_hand_keypoints = person_data['left_hand_keypoints']
        left_hand_scores = person_data['left_hand_scores']
    elif 'hands' in person_data and 'hands_scores' in person_data:
        # 旧格式
        hands = person_data['hands']
        hands_scores = person_data['hands_scores']
        
        if len(hands.shape) == 3 and hands.shape[0] >= 1:
            left_hand_keypoints = hands[0]
        if len(hands_scores.shape) == 2 and hands_scores.shape[0] >= 1:
            left_hand_scores = hands_scores[0]
    
    if left_hand_keypoints is not None and left_hand_scores is not None:
        valid_count = 0
        for kp_idx in range(len(left_hand_keypoints)):
            x, y = left_hand_keypoints[kp_idx]
            score = left_hand_scores[kp_idx] if kp_idx < len(left_hand_scores) else 0.0
            
            px = int(x * width)
            py = int(y * height)
            
            if 0 <= px < width and 0 <= py < height:
                keypoint_id = keypoint_mapping[f'left_hand_{kp_idx}']
                
                wsg_array[py, px, 64] = person_id
                wsg_array[py, px, 65] = keypoint_id
                wsg_array[py, px, 66] = float(score)
                valid_count += 1
        
        print(f"    左手关键点注入: {valid_count}/{len(left_hand_keypoints)}")
    
    # 处理右手关键点
    right_hand_keypoints = None
    right_hand_scores = None
    
    if 'right_hand_keypoints' in person_data and 'right_hand_scores' in person_data:
        # 新格式
        right_hand_keypoints = person_data['right_hand_keypoints']
        right_hand_scores = person_data['right_hand_scores']
    elif 'hands' in person_data and 'hands_scores' in person_data:
        # 旧格式
        hands = person_data['hands']
        hands_scores = person_data['hands_scores']
        
        if len(hands.shape) == 3 and hands.shape[0] >= 2:
            right_hand_keypoints = hands[1]
        if len(hands_scores.shape) == 2 and hands_scores.shape[0] >= 2:
            right_hand_scores = hands_scores[1]
    
    if right_hand_keypoints is not None and right_hand_scores is not None:
        valid_count = 0
        for kp_idx in range(len(right_hand_keypoints)):
            x, y = right_hand_keypoints[kp_idx]
            score = right_hand_scores[kp_idx] if kp_idx < len(right_hand_scores) else 0.0
            
            px = int(x * width)
            py = int(y * height)
            
            if 0 <= px < width and 0 <= py < height:
                keypoint_id = keypoint_mapping[f'right_hand_{kp_idx}']
                
                wsg_array[py, px, 64] = person_id
                wsg_array[py, px, 65] = keypoint_id
                wsg_array[py, px, 66] = float(score)
                valid_count += 1
        
        print(f"    右手关键点注入: {valid_count}/{len(right_hand_keypoints)}")


def process_single_file(args):
    """处理单个文件的DWPose注入"""
    dwpose_path, wsg_path, keypoint_mapping, confidence_threshold = args
    
    try:
        # 加载数据
        dwpose_data = load_dwpose_data(dwpose_path)
        wsg_array = load_wsg_data(wsg_path)
        
        if dwpose_data is None:
            print(f"  跳过: DWPose数据加载失败")
            return False
            
        if wsg_array is None:
            print(f"  跳过: WSG数据加载失败")
            return False
        
        # 注入DWPose数据
        if inject_dwpose_to_wsg(dwpose_data, wsg_array, keypoint_mapping, confidence_threshold):
            # 保存更新后的WSG文件
            if save_wsg_data(wsg_array, wsg_path):
                return True
        
        return False
        
    except Exception as e:
        print(f"  处理文件失败: {str(e)}")
        return False


def inject_dwpose_to_video(video_dir, dwpose_subdir, wsg_subdir, max_workers=4, confidence_threshold=0.1):
    """为单个视频目录注入DWPose数据"""
    dwpose_dir = os.path.join(video_dir, dwpose_subdir)
    wsg_dir = os.path.join(video_dir, wsg_subdir)
    
    # 检查目录是否存在
    if not os.path.exists(dwpose_dir):
        print(f"  DWPose目录不存在: {dwpose_dir}")
        return False
        
    if not os.path.exists(wsg_dir):
        print(f"  WSG目录不存在: {wsg_dir}")
        return False
    
    # 获取所有NPZ文件
    dwpose_files = glob.glob(os.path.join(dwpose_dir, "*.npz"))
    dwpose_files.sort()
    
    if not dwpose_files:
        print(f"  没有找到DWPose NPZ文件")
        return False
    
    # 获取关键点映射
    keypoint_mapping = get_dwpose_keypoint_mapping()
    
    # 准备任务
    tasks = []
    for dwpose_path in dwpose_files:
        filename = os.path.basename(dwpose_path)
        wsg_filename = filename.replace('.npz', '.wsg')
        wsg_path = os.path.join(wsg_dir, wsg_filename)
        
        if os.path.exists(wsg_path):
            tasks.append((dwpose_path, wsg_path, keypoint_mapping, confidence_threshold))
    
    if not tasks:
        print(f"  没有找到匹配的WSG文件")
        return False
    
    print(f"  找到 {len(tasks)} 个文件需要处理")
    
    # 并行处理
    success_count = 0
    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_single_file, task) for task in tasks]
            
            for i, future in enumerate(tqdm(futures, desc="  注入DWPose")):
                try:
                    if future.result():
                        success_count += 1
                except Exception as e:
                    print(f"  任务失败: {str(e)}")
    else:
        # 单线程处理
        for task in tqdm(tasks, desc="  注入DWPose"):
            if process_single_file(task):
                success_count += 1
    
    print(f"  完成! 成功: {success_count}/{len(tasks)}")
    return success_count > 0


def process_dataset(dataset_results_dir, dwpose_subdir, wsg_subdir, max_workers=4, confidence_threshold=0.1):
    """处理整个数据集"""
    if not os.path.exists(dataset_results_dir):
        print(f"数据集目录不存在: {dataset_results_dir}")
        return
    
    # 获取所有视频文件夹
    video_folders = [f for f in os.listdir(dataset_results_dir) 
                    if os.path.isdir(os.path.join(dataset_results_dir, f))]
    
    if not video_folders:
        print("没有找到视频文件夹")
        return
    
    print(f"找到 {len(video_folders)} 个视频文件夹")
    
    success_count = 0
    for video_folder in tqdm(video_folders, desc="处理视频"):
        video_dir = os.path.join(dataset_results_dir, video_folder)
        print(f"\n处理视频: {video_folder}")
        
        if inject_dwpose_to_video(video_dir, dwpose_subdir, wsg_subdir, max_workers, confidence_threshold):
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
    dwpose_subdir = "results_dwpose/npz"  # DWPose NPZ文件的相对路径
    wsg_subdir = "wsg"                    # WSG文件的相对路径
    
    # 处理配置
    max_workers = mp.cpu_count() // 2     # 使用一半的CPU核心
    confidence_threshold = 0.1            # 关键点置信度阈值（现在不使用，但保留接口）
    
    # 构建完整路径
    dataset_results_dir = os.path.join(base_output_dir, f"{dataset_name}_results")
    
    print("=" * 50)
    print("DWPose到WSG注入器")
    print("=" * 50)
    print(f"数据集: {dataset_name}")
    print(f"数据集路径: {dataset_results_dir}")
    print(f"DWPose子路径: {dwpose_subdir}")
    print(f"WSG子路径: {wsg_subdir}")
    print(f"工作器数量: {max_workers}")
    print(f"置信度阈值: {confidence_threshold} (当前不使用)")
    print("=" * 50)
    
    try:
        process_dataset(dataset_results_dir, dwpose_subdir, wsg_subdir, max_workers, confidence_threshold)
        print("\n✅ 所有注入任务完成!")
        
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()