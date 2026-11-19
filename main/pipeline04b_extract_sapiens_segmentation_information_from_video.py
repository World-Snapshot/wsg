import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import shutil
import subprocess
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from utils.util import get_fps, read_frames, save_videos_from_pil
from PIL import Image
import numpy as np
import json
from typing import List, Dict, Any, Optional, Tuple
import cv2

from sapiens_inference.normal import SapiensNormal, SapiensNormalType, draw_normal_map
from sapiens_inference.pose import SapiensPoseEstimation, SapiensPoseEstimationType
from sapiens_inference.segmentation import SapiensSegmentation, SapiensSegmentationType, draw_segmentation_map
from sapiens_inference.depth import SapiensDepth, SapiensDepthType, draw_depth_map
from sapiens_inference import SapiensPredictor, SapiensConfig, SapiensDepthType, SapiensNormalType
from sapiens_inference.common import TaskType

# Environment variables for distributed processing
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"

def setup(rank, world_size):
    """Setup distributed training"""
    try:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '23456'
        dist.init_process_group(
            "gloo",
            rank=rank, 
            world_size=world_size
        )
        return True
    except Exception as e:
        print(f"Failed to initialize process group: {str(e)}")
        return False

def cleanup():
    dist.destroy_process_group()

def process_video_with_sapiens(rank, frames_dir, output_base_dir, framerate=30):
    """Process frames with all Sapiens models using specific GPU"""
    device = f"cuda:{rank}"
    dtype = torch.float16

    try:
        # Initialize combined predictor for all tasks
        config = SapiensConfig()
        config.dtype = dtype
        config.device = device
        config.depth_type = SapiensDepthType.DEPTH_1B
        config.normal_type = SapiensNormalType.NORMAL_1B
        config.segmentation_type = SapiensSegmentationType.SEGMENTATION_1B
        
        # 初始化 predictor 和单独的 pose estimator
        predictor = SapiensPredictor(config)
        pose_estimator = SapiensPoseEstimation(
            type=SapiensPoseEstimationType.POSE_ESTIMATION_1B,
            device=device,
            dtype=dtype
        )

        print(f"[GPU {rank}] Initialized all Sapiens models on {device}")

        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png'))])
        total_frames = len(frame_files)
        print(f"[GPU {rank}] Found {total_frames} frames to process")

        # 创建所需目录
        data_types = ["segmentation"]
        for data_type in data_types:
            os.makedirs(os.path.join(output_base_dir, f"results_sapiens_{data_type}/img"), exist_ok=True)
            os.makedirs(os.path.join(output_base_dir, f"results_sapiens_{data_type}/npz"), exist_ok=True)
            os.makedirs(os.path.join(output_base_dir, f"sapiens_{data_type}_overlay_img"), exist_ok=True)

        for i, frame_file in enumerate(frame_files):
            try:
                frame_path = os.path.join(frames_dir, frame_file)
                frame_pil = Image.open(frame_path)
                frame_cv2 = cv2.imread(frame_path)
                
                print(f"\n[GPU {rank}] Processing frame {i+1}/{total_frames}")

                # 获取所有预测结果
                img_shape = frame_cv2.shape[:2]
                person_boxes = [[0, 0, img_shape[1], img_shape[0]]]  # 处理整张图片
                
                save_results = []
                frame_idx = i + 1

                # 处理 pose 估计
                try:
                    # 获取pose结果和关键点
                    pose_result_orig, keypoints = pose_estimator(frame_cv2)
                    
                    # 打印调试信息
                    print(f"[DEBUG] Pose result shape: {pose_result_orig.shape if pose_result_orig is not None else 'None'}")
                    print(f"[DEBUG] Keypoints length: {len(keypoints) if keypoints is not None else 'None'}")
                    
                    # 创建黑色背景的pose图像
                    black_bg = np.zeros_like(frame_cv2)
                    # 确保pose_result_orig不是None且是有效的numpy数组
                    if pose_result_orig is not None and isinstance(pose_result_orig, np.ndarray):
                        pose_mask = cv2.cvtColor(pose_result_orig, cv2.COLOR_BGR2GRAY) > 0
                        pose_result_black_bg = black_bg.copy()
                        pose_result_black_bg[pose_mask] = pose_result_orig[pose_mask]
                        
                        print("[DEBUG] Created black background pose image")
                        
                        # 准备保存数据
                        pose_data = {
                            'pose': pose_result_black_bg,
                            'pose_overlay': pose_result_orig,
                            'keypoints': keypoints
                        }
                        
                        # 保存pose结果
                        success = save_sapiens_data(
                            pose_data,
                            frame_idx, 
                            output_base_dir, 
                            frame_pil, 
                            "pose"
                        )
                        save_results.append(('pose', success))
                        print(f"[DEBUG] Pose save result: {success}")
                    else:
                        print("[ERROR] Invalid pose result format")
                        save_results.append(('pose', False))
                        
                except Exception as pose_err:
                    print(f"[GPU {rank}] Error in pose estimation: {str(pose_err)}")
                    print(traceback.format_exc())
                    save_results.append(('pose', False))
                
                # 获取所有预测结果
                segmentation_maps = []
                normal_maps = []
                depth_maps = []
                
                for box in person_boxes:
                    crop = frame_cv2[box[1]:box[3], box[0]:box[2]]
                    
                    if predictor.segmentation_predictor:
                        seg_map = predictor.segmentation_predictor(crop)
                        segmentation_maps.append(seg_map)
                    
                    if predictor.has_normal and predictor.normal_predictor:
                        normal_map = predictor.normal_predictor(crop)
                        normal_maps.append(normal_map)
                        
                    if predictor.has_depth and predictor.depth_predictor:
                        depth_map = predictor.depth_predictor(crop)
                        depth_maps.append(depth_map)

                save_results = []
                frame_idx = i + 1
                
                # 处理分割结果
                if segmentation_maps:
                    success = save_sapiens_data(
                        segmentation_maps[0], frame_idx, output_base_dir, frame_pil, "segmentation"
                    )
                    save_results.append(('segmentation', success))
                    
                    # 获取分割遮罩用于其他模态
                    segmentation_mask = segmentation_maps[0] > 0

                # 处理法线图结果
                if normal_maps:
                    normal_data = normal_maps[0]
                    if 'segmentation_mask' in locals():
                        # 使用分割遮罩
                        normal_data = normal_data * segmentation_mask[..., None]
                    success = save_sapiens_data(
                        normal_data, frame_idx, output_base_dir, frame_pil, "normal"
                    )
                    save_results.append(('normal', success))

                # 处理深度图结果
                if depth_maps:
                    depth_data = depth_maps[0]
                    if 'segmentation_mask' in locals():
                        # 使用分割遮罩
                        depth_data = depth_data * segmentation_mask
                    success = save_sapiens_data(
                        depth_data, frame_idx, output_base_dir, frame_pil, "depth"
                    )
                    save_results.append(('depth', success))

                # 打印保存结果
                print(f"[GPU {rank}] Frame {frame_idx} save results:")
                for result_type, success in save_results:
                    print(f"- {result_type}: {'Success' if success else 'Failed'}")

                if frame_idx % 10 == 0:
                    print(f"[GPU {rank}] Processed {frame_idx}/{total_frames} frames ({frame_idx/total_frames*100:.1f}%)")

            except Exception as e:
                print(f"[GPU {rank}] Error processing frame {i + 1}: {str(e)}")
                import traceback
                print(traceback.format_exc())
                continue

        # 生成可视化视频
        for data_type in data_types:
            try:
                video_output_dir = output_base_dir
                
                # 生成可视化视频
                img_dir = os.path.join(video_output_dir, f"results_sapiens_{data_type}/img")
                if os.path.exists(img_dir) and os.listdir(img_dir):
                    vis_path = os.path.join(video_output_dir, f"visualization_sapiens_{data_type}.mp4")
                    cmd = f'ffmpeg -y -framerate {framerate} -i {img_dir}/%08d.jpg -c:v libx264 -pix_fmt yuv420p {vis_path}'
                    subprocess.run(cmd, shell=True, check=True)
                    print(f"[GPU {rank}] Created visualization video for {data_type}")

                # 生成overlay视频
                overlay_dir = os.path.join(video_output_dir, f"sapiens_{data_type}_overlay_img")
                if os.path.exists(overlay_dir) and os.listdir(overlay_dir):
                    overlay_path = os.path.join(video_output_dir, f"visualization_sapiens_{data_type}_overlap.mp4")
                    cmd = f'ffmpeg -y -framerate {framerate} -i {overlay_dir}/%08d.png -c:v libx264 -pix_fmt yuv420p {overlay_path}'
                    subprocess.run(cmd, shell=True, check=True)
                    print(f"[GPU {rank}] Created overlay video for {data_type}")

            except Exception as e:
                print(f"[GPU {rank}] Error creating visualization videos for {data_type}: {str(e)}")
                print(traceback.format_exc())

    except Exception as e:
        print(f"[GPU {rank}] Error initializing models: {str(e)}")
        print(traceback.format_exc())
        raise

def save_sapiens_data(frame_data, frame_idx, base_dir, frame_pil, data_type):
    """Save processed data and visualizations for each Sapiens model type"""
    try:
        print(f"\n[Save] Starting to save {data_type} data for frame {frame_idx}")
        
        # 创建所需的目录
        frame_name = f"{frame_idx:08d}"
        data_dir = os.path.join(base_dir, f"results_sapiens_{data_type}")
        img_dir = os.path.join(data_dir, "img")
        npz_dir = os.path.join(data_dir, "npz")
        overlay_dir = os.path.join(base_dir, f"sapiens_{data_type}_overlay_img")
        
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(npz_dir, exist_ok=True)
        os.makedirs(overlay_dir, exist_ok=True)

        # 转换输入图像
        frame_cv2 = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        npz_path = os.path.join(npz_dir, f"{frame_name}.npz")

        if data_type == "pose":
            if not isinstance(frame_data, dict):
                print(f"[Save] ERROR: Pose data should be a dict but got {type(frame_data)}")
                return False
            
            # 先保存可视化
            cv2.imwrite(os.path.join(img_dir, f"{frame_name}.jpg"), frame_data['pose'])
            overlay = cv2.addWeighted(frame_cv2, 0.7, frame_data['pose_overlay'], 0.3, 0)
            cv2.imwrite(os.path.join(overlay_dir, f"{frame_name}.png"), overlay)
            
            # 再保存NPZ数据
            np.savez(npz_path, **frame_data)
            
        elif data_type == "normal":
            # 先生成可视化
            normal_map = draw_normal_map(frame_data)
            if normal_map.shape[:2] != frame_cv2.shape[:2]:
                normal_map = cv2.resize(normal_map, (frame_cv2.shape[1], frame_cv2.shape[0]))
            
            cv2.imwrite(os.path.join(img_dir, f"{frame_name}.jpg"), normal_map)
            overlay = cv2.addWeighted(frame_cv2, 0.3, normal_map, 0.8, 0)
            cv2.imwrite(os.path.join(overlay_dir, f"{frame_name}.png"), overlay)
            
            # 检查normal_map是否正确生成
            if normal_map is not None and np.any(normal_map != 0):
                # 从normal_map反推原始数据
                normal_data = normal_map.astype(np.float32)
                normal_data = normal_data / 255.0 * 2 - 1  # 反归一化
                np.savez(npz_path, normal_data=normal_data)

        elif data_type == "depth":
            # 先生成可视化
            depth_colored = draw_depth_map(frame_data)
            if depth_colored.shape[:2] != frame_cv2.shape[:2]:
                depth_colored = cv2.resize(depth_colored, (frame_cv2.shape[1], frame_cv2.shape[0]))
            
            cv2.imwrite(os.path.join(img_dir, f"{frame_name}.jpg"), depth_colored)
            overlay = cv2.addWeighted(frame_cv2, 0.5, depth_colored, 0.5, 0)
            cv2.imwrite(os.path.join(overlay_dir, f"{frame_name}.png"), overlay)
            
            # 检查depth_colored是否正确生成
            if depth_colored is not None and np.any(depth_colored != 128):  # 128是背景值
                # 保存depth数据
                depth_data = frame_data.astype(np.float32)
                np.savez(npz_path, depth_data=depth_data)

        elif data_type == "segmentation":
            # 先生成可视化
            seg_vis = draw_segmentation_map(frame_data)
            if seg_vis.shape[:2] != frame_cv2.shape[:2]:
                seg_vis = cv2.resize(seg_vis, (frame_cv2.shape[1], frame_cv2.shape[0]))
            
            cv2.imwrite(os.path.join(img_dir, f"{frame_name}.jpg"), seg_vis)
            overlay = cv2.addWeighted(frame_cv2, 0.5, seg_vis, 0.7, 0)
            cv2.imwrite(os.path.join(overlay_dir, f"{frame_name}.png"), overlay)
            
            # 检查seg_vis是否正确生成
            if seg_vis is not None and np.any(seg_vis != 0):
                # 保存分割数据
                seg_data = frame_data.astype(np.float32)
                np.savez(npz_path, segmentation_data=seg_data)
        
        print(f"[Save] Successfully saved {data_type} data and visualizations for frame {frame_idx}")
        return True

    except Exception as e:
        print(f"[Save] Error in save_sapiens_data for frame {frame_idx}, type {data_type}:")
        import traceback
        print(traceback.format_exc())
        return False

def process_videos_distributed(rank, world_size, video_dirs, dataset_name, framerate=30):
    """Process videos with Sapiens models distributed across GPUs"""
    if not setup(rank, world_size):
        return
        
    result_dir = f"../output/{dataset_name}_results"
    
    for i, (video_name, frames_dir) in enumerate(video_dirs):
        if i % world_size == rank:
            print(f"GPU {rank} processing video: {video_name}")
            
            try:
                # Process with Sapiens models
                folder_path = f"{result_dir}/{video_name}"
                os.makedirs(folder_path, exist_ok=True)
                process_video_with_sapiens(rank, frames_dir, folder_path, framerate)
                print(f"GPU {rank}: Completed processing for {video_name}")
                
                with open(f"{result_dir}/processed_video_name_by_sapiens.txt", "a") as f:
                    f.write(f"{video_name}\n")
                    
            except Exception as e:
                print(f"GPU {rank}: Error processing video {video_name}: {str(e)}")
                continue

    cleanup()

def main():
    #dataset_name = "test_dataset"
    dataset_name = "AVIS_val"
    #dataset_name = "ncslgr_mp4"
    #dataset_name = "ncslgr_mp4_small"
    #dataset_name = "ASLLRP_utterances_small"
    #dataset_name = "ASLLRP_utterances"
    framerate = 1  # Default framerate, can be modified as needed
    result_dir = f"../output/{dataset_name}_results"
    os.makedirs(result_dir, exist_ok=True)

    # Collect videos that need processing
    video_dirs = []
    processed_videos = set()
    
    # Read processed videos list
    processed_log = f"{result_dir}/processed_video_name_by_sapiens.txt"
    if os.path.exists(processed_log):
        with open(processed_log, 'r') as f:
            processed_videos = set(f.read().splitlines())

    # Check for videos needing processing
    for video_name in os.listdir(result_dir):
        frames_dir = os.path.join(result_dir, video_name, "crop_frame")
        if os.path.isdir(frames_dir) and video_name not in processed_videos:
            video_dirs.append((video_name, frames_dir))
            print(f"Adding video for processing: {video_name}")

    if video_dirs:
        print(f"Found {len(video_dirs)} videos to process")
        
        world_size = torch.cuda.device_count()
        if world_size > 1:
            print(f"Found {world_size} GPUs, running distributed processing")
            mp.spawn(
                process_videos_distributed,
                args=(world_size, video_dirs, dataset_name, framerate),
                nprocs=world_size,
                join=True
            )
        else:
            print("Only one GPU available, running in single GPU mode")
            process_videos_distributed(0, 1, video_dirs, dataset_name, framerate)
    else:
        print("All videos have been processed.")

if __name__ == '__main__':
    main()