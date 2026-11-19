import os
import subprocess
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import json
from easy_dwpose import DWposeDetector
from PIL import Image
from utils.util import read_frames
from utils.draw_dw_lib import draw_pose, process_pose_data

# 设置分布式训练环境变量
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"

def setup(rank, world_size):
    """设置分布式训练环境"""
    try:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '23457'  # 使用不同的端口避免冲突
        dist.init_process_group(
            "gloo",
            rank=rank, 
            world_size=world_size
        )
        return True
    except Exception as e:
        print(f"初始化进程组失败: {str(e)}")
        return False

def cleanup():
    """清理分布式训练环境"""
    dist.destroy_process_group()

def save_dwpose_data(pose_data, frame_idx, json_dir, npy_dir, npz_dir, img_dir, frame_pil, detector):
    """保存DWpose数据到各种格式 - 完全使用新的数据结构"""
    try:
        frame_name = f"{frame_idx:08d}"
        
        # 检测人数
        num_persons = pose_data['faces'].shape[0] if 'faces' in pose_data else 0
        
        # 完全使用新的数据结构
        new_pose_data = {
            'num_persons': num_persons
        }
        
        if num_persons > 0:
            # 重新整理身体关键点数据
            bodies_reshaped = pose_data['bodies'].reshape(num_persons, 18, 2)
            
            # 为每个人添加单独的数据
            for person_id in range(num_persons):
                new_pose_data[f'person_{person_id}'] = {
                    'body_keypoints': bodies_reshaped[person_id],
                    'body_scores': pose_data['body_scores'][person_id],
                    'face_keypoints': pose_data['faces'][person_id],
                    'face_scores': pose_data['faces_scores'][person_id],
                    'left_hand_keypoints': pose_data['hands'][person_id * 2],
                    'left_hand_scores': pose_data['hands_scores'][person_id * 2],
                    'right_hand_keypoints': pose_data['hands'][person_id * 2 + 1],
                    'right_hand_scores': pose_data['hands_scores'][person_id * 2 + 1],
                }
        
        # 保存NPY文件 - 只保存新格式
        try:
            np.save(os.path.join(npy_dir, f"{frame_name}.npy"), new_pose_data)
        except Exception as e:
            print(f"保存NPY文件失败 (帧 {frame_idx}): {str(e)}")
        
        # 保存NPZ文件 - 只保存新格式
        try:
            np.savez(os.path.join(npz_dir, f"{frame_name}.npz"), **new_pose_data)
        except Exception as e:
            print(f"保存NPZ文件失败 (帧 {frame_idx}): {str(e)}")
        
        # 保存可视化图像
        try:
            # 使用新的可视化方法
            try:
                width, height = frame_pil.size
                processed_pred = process_pose_data(new_pose_data, height, width)
                
                vis_img = draw_pose(
                    pose=processed_pred,
                    H=height,
                    W=width,
                    include_body=True,
                    include_hand=True,
                    include_face=True
                )
                
                # 处理通道顺序
                if isinstance(vis_img, np.ndarray):
                    if vis_img.shape[0] == 3 and len(vis_img.shape) == 3:
                        vis_img = np.transpose(vis_img, (1, 2, 0))
                    
                    if vis_img.dtype != np.uint8:
                        vis_img = (vis_img * 255).astype(np.uint8)
                    
                    pred_img = Image.fromarray(vis_img)
                else:
                    raise ValueError(f"draw_pose返回了无效类型: {type(vis_img)}")
                    
            except Exception as viz_error:
                print(f"警告: 新可视化方法失败 ({str(viz_error)}), 使用原始方法")
                pred_img = detector(frame_pil, output_type="pil", include_hands=True, include_face=True)
            
            pred_img.save(os.path.join(img_dir, f"{frame_name}.jpg"))
            
        except Exception as e:
            print(f"保存姿势可视化图像失败 (帧 {frame_idx}): {str(e)}")

        # 创建叠加可视化图像
        try:
            overlay_dir = os.path.join(os.path.dirname(os.path.dirname(img_dir)), "dwpose_overlay_img")
            os.makedirs(overlay_dir, exist_ok=True)

            processed_pred = process_pose_data(new_pose_data, height, width)
            transparent_vis = draw_pose(
                pose=processed_pred,
                H=height,
                W=width,
                include_body=True,
                include_hand=True,
                include_face=True,
                transparent=True
            )

            if isinstance(transparent_vis, np.ndarray):
                if transparent_vis.shape[0] == 4 and len(transparent_vis.shape) == 3:
                    transparent_vis = np.transpose(transparent_vis, (1, 2, 0))
                transparent_img = Image.fromarray(transparent_vis, 'RGBA')
            
            frame_rgba = frame_pil.convert('RGBA')
            overlay_img = Image.alpha_composite(frame_rgba, transparent_img)
            
            overlay_path = os.path.join(overlay_dir, f"{frame_name}.png")
            overlay_img.save(overlay_path)

        except Exception as e:
            print(f"创建叠加可视化失败 (帧 {frame_idx}): {str(e)}")

        # 保存JSON文件 - 只保存新格式
        try:
            pose_data_json = {}
            for k, v in new_pose_data.items():
                if isinstance(v, np.ndarray):
                    pose_data_json[k] = v.tolist()
                elif isinstance(v, (np.int32, np.int64)):
                    pose_data_json[k] = int(v)
                elif isinstance(v, (np.float32, np.float64)):
                    pose_data_json[k] = float(v)
                elif isinstance(v, dict):
                    # 处理person_id字典
                    person_dict = {}
                    for pk, pv in v.items():
                        if isinstance(pv, np.ndarray):
                            person_dict[pk] = pv.tolist()
                        else:
                            person_dict[pk] = pv
                    pose_data_json[k] = person_dict
                else:
                    pose_data_json[k] = v
            
            json_path = os.path.join(json_dir, f"{frame_name}.json")
            with open(json_path, 'w') as f:
                json.dump(pose_data_json, f)
                
        except Exception as e:
            print(f"保存JSON文件失败 (帧 {frame_idx}): {str(e)}")
            
    except Exception as e:
        print(f"save_dwpose_data发生一般错误 (帧 {frame_idx}): {str(e)}")

def process_video_with_dwpose(rank, crop_frame_dir, output_base_dir, fps):
    """使用指定GPU处理crop_frame文件夹中的图片"""
    json_dir = os.path.join(output_base_dir, "json")
    npy_dir = os.path.join(output_base_dir, "npy")
    npz_dir = os.path.join(output_base_dir, "npz")
    img_dir = os.path.join(output_base_dir, "img")
    
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(npy_dir, exist_ok=True)
    os.makedirs(npz_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    print(f"GPU {rank}: DWpose - 创建输出目录完成")

    device = f"cuda:{rank}"
    detector = DWposeDetector(device=device)
    print(f"GPU {rank}: DWpose - 在 {device} 上初始化检测器完成")

    # 读取crop_frame目录中的图片
    import glob
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(crop_frame_dir, ext)))
    image_files.sort()  # 按文件名排序
    
    total_frames = len(image_files)
    print(f"GPU {rank}: DWpose - 从crop_frame目录读取了 {total_frames} 张图片")

    for i, image_path in enumerate(image_files):
        frame_pil = Image.open(image_path)
        # 获取文件名（不含扩展名）作为帧编号
        frame_name = os.path.splitext(os.path.basename(image_path))[0]
        frame_idx = int(frame_name) if frame_name.isdigit() else i + 1
        
        pose_data = detector(frame_pil, draw_pose=False, include_hands=True, include_face=True)
        save_dwpose_data(pose_data, frame_idx, json_dir, npy_dir, npz_dir, img_dir, frame_pil, detector)

        if (i + 1) % 10 == 0 or i == 0 or i == total_frames - 1:
            print(f"GPU {rank}: DWpose - 已处理 {i + 1}/{total_frames} 帧 ({(i + 1)/total_frames*100:.1f}%)")

    # 生成可视化视频
    visualization_path = os.path.join(os.path.dirname(output_base_dir), "visualization_dwpose.mp4")
    print(f"GPU {rank}: DWpose - 创建可视化视频...")
    try:
        cmd = f'ffmpeg -y -framerate {fps} -i {img_dir}/%08d.jpg -c:v libx264 -pix_fmt yuv420p {visualization_path}'
        subprocess.run(cmd, shell=True, check=True)
        print(f"GPU {rank}: DWpose - 成功创建可视化视频: {visualization_path}")
    except Exception as e:
        print(f"GPU {rank}: DWpose - 创建可视化视频失败: {str(e)}")

    # 生成叠加可视化视频
    overlay_dir = os.path.join(os.path.dirname(output_base_dir), "dwpose_overlay_img")
    visualization_overlap_path = os.path.join(os.path.dirname(output_base_dir), "visualization_dwpose_overlap.mp4")
    print(f"GPU {rank}: DWpose - 创建叠加可视化视频...")
    try:
        cmd = f'ffmpeg -y -framerate {fps} -i {overlay_dir}/%08d.png -c:v libx264 -pix_fmt yuv420p {visualization_overlap_path}'
        subprocess.run(cmd, shell=True, check=True)
        print(f"GPU {rank}: DWpose - 成功创建叠加可视化视频: {visualization_overlap_path}")
    except Exception as e:
        print(f"GPU {rank}: DWpose - 创建叠加可视化视频失败: {str(e)}")

def process_videos_distributed(rank, world_size, video_paths, dataset_name, fps):
    """分布式处理视频"""
    if not setup(rank, world_size):
        return
        
    result_dir = f"../output/{dataset_name}_results"
    
    for i, (video_name, crop_frame_path) in enumerate(video_paths):
        if i % world_size == rank:
            print(f"GPU {rank} 正在处理视频: {video_name}")
            
            try:
                folder_path = f"{result_dir}/{video_name}"
                dwpose_results_dir = f"{folder_path}/results_dwpose"
                
                if not os.path.exists(os.path.join(dwpose_results_dir, "json")):
                    os.makedirs(dwpose_results_dir, exist_ok=True)
                    print(f"GPU {rank}: 创建目录 {dwpose_results_dir}")
                    process_video_with_dwpose(rank, crop_frame_path, dwpose_results_dir, fps)
                    print(f"GPU {rank}: 完成DWpose处理 {video_name}")
                    
                    with open(f"{result_dir}/processed_video_name_by_dwpose.txt", "a") as f:
                        f.write(f"{video_name}\n")
                else:
                    print(f"GPU {rank}: {video_name} 已经处理过，跳过")
                            
            except Exception as e:
                print(f"GPU {rank}: 处理视频 {video_name} 时发生错误: {str(e)}")
                continue

    cleanup()

def main():
    """主函数"""
    # 配置数据集名称
    #dataset_name = "AVIS_train"
    dataset_name = "test_dataset"
    # dataset_name = "how2sign_test"
    # dataset_name = "ncslgr_mp4"
    # dataset_name = "ncslgr_mp4_small"
    # dataset_name = "ASLLRP_batch_full_video_v1_1_small"
    # dataset_name = "ASLLRP_utterances_small"
    
    result_dir = f"../output/{dataset_name}_results"

    fps = 1

    # 收集需要处理的视频
    video_paths = []
    processed_dwpose = set()
    
    # 读取已处理的视频列表
    dwpose_log = f"{result_dir}/processed_video_name_by_dwpose.txt"
    
    if os.path.exists(dwpose_log):
        with open(dwpose_log, 'r') as f:
            processed_dwpose = set(f.read().splitlines())

    # 从output目录检查需要DWpose处理的视频
    if os.path.exists(result_dir):
        for video_name in os.listdir(result_dir):
            video_path = os.path.join(result_dir, video_name)
            if os.path.isdir(video_path):
                crop_frame_path = os.path.join(video_path, "crop_frame")
                
                if os.path.exists(crop_frame_path) and video_name not in processed_dwpose:
                    video_paths.append((video_name, crop_frame_path))
                    print(f"添加视频进行DWpose处理: {video_name}")
    else:
        print(f"结果目录不存在: {result_dir}")
        return

    # 运行DWpose处理
    if video_paths:
        print(f"找到 {len(video_paths)} 个视频需要DWpose处理")
        
        world_size = torch.cuda.device_count()
        if world_size > 1:
            print(f"找到 {world_size} 个GPU，运行分布式处理")
            mp.spawn(
                process_videos_distributed,
                args=(world_size, video_paths, dataset_name, fps),
                nprocs=world_size,
                join=True
            )
        else:
            print("只有一个GPU可用，运行单GPU模式")
            process_videos_distributed(0, 1, video_paths, dataset_name, fps)
    else:
        print("所有视频都已经进行过DWpose处理")

if __name__ == '__main__':
    main()