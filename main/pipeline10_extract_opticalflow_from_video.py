import os
import cv2
import numpy as np
import torch
from PIL import Image
from typing import Tuple
from multiprocessing.dummy import Pool
import time

class OpticalFlowProcessor:
    def __init__(self, config_path: str, device: str = "auto", result_dir: str = None):
        self.config_path = config_path
        
        # 自动检测可用设备
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
                print("CUDA可用，使用GPU")
            else:
                self.device = "cpu"
                print("CUDA不可用，切换到CPU")
        elif device.startswith("cuda:"):
            # 设置特定的GPU
            gpu_id = int(device.split(":")[1])
            if gpu_id < torch.cuda.device_count():
                self.device = device
                torch.cuda.set_device(gpu_id)
                print(f"使用指定的GPU: {device}")
            else:
                print(f"GPU {gpu_id} 不可用，切换到CPU")
                self.device = "cpu"
        else:
            # 验证指定的设备是否可用
            if device == "cuda" and not torch.cuda.is_available():
                print("CUDA不可用，切换到CPU")
                self.device = "cpu"
            else:
                self.device = device
        
        self.result_dir = result_dir
        
        # Initialize record files
        if result_dir:
            self.depth_record_file = os.path.join(result_dir, "processed_video_name_by_opticalflow.txt")
            if not os.path.exists(self.depth_record_file):
                with open(self.depth_record_file, 'w') as f:
                    pass

        print(f"使用设备: {self.device}")
        print("Initializing Optical Flow (flownetsd) model...")
        self.initialize_depth_model()

    # install the depth_pro
    # cd opticalflow_inference, then run the following command:
    # pip install -e .
    # source get_pretrained_models.sh
    # cd ..
    def initialize_depth_model(self):
        """Initialize the optical flow model"""
        try:            
            # 创建推理引擎时传入正确的设备
            import ptlflow
            self.engine = ptlflow.get_model('flownetsd', ckpt_path='things')
            self.engine.to(self.device)
            
            # 如果InferenceEngine内部有模型加载，我们可能需要处理设备映射
            # 这取决于InferenceEngine的具体实现
            print(f"Initialized optical flow model on {self.device}")
            
        except Exception as e:
            error_msg = str(e)
            print(f"Error initializing depth model: {error_msg}")
            raise

    def is_video_processed(self, video_name: str) -> bool:
        """检查视频是否已经被处理过"""
        if os.path.exists(self.depth_record_file):
            with open(self.depth_record_file, 'r') as f:
                processed_videos = set(f.read().splitlines())
                return video_name in processed_videos
        return False

    def record_video_processed(self, video_name: str):
        """记录视频已被处理"""
        with open(self.depth_record_file, 'a') as f:
            f.write(f"{video_name}\n")

    def process_opticalflow(self, frame_path: str, frame2_path: str) -> Tuple[np.ndarray, Image.Image]:
        """Process optical flow for a single frame"""
        try:
            from ptlflow.utils import flow_utils
            from ptlflow.utils.io_adapter import IOAdapter
            image_paths = [frame_path, frame2_path]
            images = [cv2.imread(path) for path in image_paths]
            io_adapter = IOAdapter(self.engine, images[0].shape[:2])
            inputs = io_adapter.prepare_inputs(images)
            inputs['images'] = inputs['images'].to(self.device)
            predictions = self.engine(inputs)
            # flow is 5d, shape (1, 1, 2, H, W).
            flows = predictions['flows']
            flows = flows.detach().cpu()
            flow_map = flows.squeeze().permute(1, 2, 0).numpy()
            flow_rgb = flow_utils.flow_to_rgb(flows)
            flow_rgb = flow_rgb[0, 0].permute(1, 2, 0)
            flow_rgb = flow_rgb.detach().cpu().numpy()
            # flow_map = flow_rgb.detach().cpu().numpy()
            
            return flow_map, flow_rgb

        except Exception as e:
            print(f"Error processing normal for frame {frame_path}: {str(e)}")
            raise

    def create_visualization_video(self, output_dir: str, total_frames: int):
        """Create visualization videos using OpenCV"""
        try:
            # 检查第一帧是否存在
            first_frame_path = os.path.join(output_dir, "results_opticalflow", "img", "00000001.jpg")
            print(f"Checking first frame path: {first_frame_path}")
            if not os.path.exists(first_frame_path):
                print(f"Error: Could not find first frame at {first_frame_path}")
                return
            else:
                print("First frame exists!")

            # 获取帧大小
            first_frame = cv2.imread(first_frame_path)
            if first_frame is None:
                print("Error: Could not read first frame")
                return
            frame_size = (first_frame.shape[1], first_frame.shape[0])
            print(f"Frame size: {frame_size}")

            # 视频编码器设置
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 30

            # 处理两种类型的视频：深度图和叠加深度图
            video_types = ['opticalflow', 'opticalflow_overlay']
            
            for video_type in video_types:
                input_dir = {
                    'opticalflow': os.path.join(output_dir, "results_opticalflow/img"),
                    'opticalflow_overlay': os.path.join(output_dir, "opticalflow_overlay_img")
                }[video_type]

                print(f"\nProcessing {video_type}")
                print(f"Input directory: {input_dir}")
                
                if not os.path.exists(input_dir):
                    print(f"Skipping {video_type} - input directory not found")
                    continue
                
                # 检查输入目录中的文件
                files = sorted(os.listdir(input_dir))
                print(f"Found {len(files)} files in input directory")
                if files:
                    print(f"First few files: {files[:5]}")

                output_path = os.path.join(output_dir, f"visualization_{video_type}.mp4")
                print(f"Output path: {output_path}")
                
                # 创建视频写入器
                out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
                if not out.isOpened():
                    print(f"Error: Could not create video writer for {video_type}")
                    continue

                frames_processed = 0
                try:
                    # 逐帧处理
                    for frame_num in range(1, total_frames + 1):
                        frame_path = os.path.join(
                            input_dir, 
                            f"{frame_num:08d}.{'png' if 'overlay' in video_type else 'jpg'}"
                        )
                        
                        if not os.path.exists(frame_path):
                            print(f"Warning: Frame {frame_num} not found at {frame_path}")
                            continue

                        frame = cv2.imread(frame_path)
                        if frame is not None:
                            out.write(frame)
                            frames_processed += 1
                        else:
                            print(f"Warning: Could not read frame {frame_num} for {video_type}")

                        if frame_num % 100 == 0:
                            print(f"Processing {video_type}: {frame_num}/{total_frames} frames")

                    print(f"Successfully processed {frames_processed} frames for {video_type}")

                except Exception as e:
                    print(f"Error processing {video_type}: {str(e)}")
                    import traceback
                    print(traceback.format_exc())

                finally:
                    # 确保视频写入器正确释放
                    out.release()
                    print(f"Video writer released for {video_type}")

        except Exception as e:
            print(f"Error in create_visualization_video: {str(e)}")
            import traceback
            print(traceback.format_exc())

    def process_video_frames(self, frames_dir: str, output_dir: str, video_name: str):
        """Process all frames in a directory for depth estimation"""
        if not self.result_dir:
            raise ValueError("result_dir must be set to track video processing")

        # 检查是否需要处理
        # if self.is_video_processed(video_name):
        #     print(f"Video {video_name} has already been processed by opticalflow")
        #     return

        # 创建输出目录
        depth_data_dir = os.path.join(output_dir, "results_opticalflow")
        depth_img_dir = os.path.join(depth_data_dir, "img")
        depth_npz_dir = os.path.join(depth_data_dir, "npz")
        depth_overlay_dir = os.path.join(output_dir, "opticalflow_overlay_img")
        
        # 创建所有必要的目录
        for directory in [depth_img_dir, depth_npz_dir, depth_overlay_dir]:
            os.makedirs(directory, exist_ok=True)

        # 获取所有帧文件
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png'))])
        total_frames = len(frame_files)
        print(f"Found {total_frames} frames to process")

        try:
            for i, frame_file in enumerate(frame_files):
                try:
                    frame_path = os.path.join(frames_dir, frame_file)
                    frame_name = f"{i+1:08d}"
                    # the optical flow is computed between two consecutive frames
                    # for the last frame, we use the same frame as the second frame
                    if i != len(frame_files) - 1:
                        frame2_path = os.path.join(frames_dir, frame_files[i+1])
                    else:
                        frame2_path = os.path.join(frames_dir, frame_files[i-1])
                    
                    # 读取原始帧用于叠加
                    frame_pil = Image.open(frame_path)
                    frame_cv2 = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

                    # 处理深度
                    flow_map, flow_rgb = self.process_opticalflow(frame_path, frame2_path)
                    flow_vis = cv2.cvtColor(flow_rgb, cv2.COLOR_RGB2BGR)
                    flow_color = (flow_vis * 255).clip(0, 255).astype(np.uint8)
                    
                    # 保存深度相关输出
                    # 保存深度可视化图像
                    cv2.imwrite(os.path.join(depth_img_dir, f"{frame_name}.jpg"), flow_color)
                    
                    # 保存原始深度数据
                    np.savez(os.path.join(depth_npz_dir, f"{frame_name}.npz"), flow_data=flow_map)
                    
                    # 创建深度叠加效果
                    depth_overlay = cv2.addWeighted(frame_cv2, 0.5, flow_color, 0.5, 0)
                    cv2.imwrite(os.path.join(depth_overlay_dir, f"{frame_name}.png"), depth_overlay)

                    if (i + 1) % 10 == 0:
                        print(f"Processed {i+1}/{total_frames} frames ({(i+1)/total_frames*100:.1f}%)")

                except Exception as e:
                    print(f"Error processing frame {frame_file}: {str(e)}")
                    continue

            # 创建视频文件
            self.create_visualization_video(output_dir, total_frames)

            # 记录处理完成
            self.record_video_processed(video_name)

        except Exception as e:
            print(f"Error processing video {video_name}: {str(e)}")
            import traceback
            print(traceback.format_exc())

def process_videos_on_gpu(args):
    """在指定GPU上处理多个视频"""
    gpu_id, video_list, config_path, result_dir = args
    
    print(f"[GPU {gpu_id}] Starting processing {len(video_list)} videos")
    
    # 初始化处理器，使用指定的GPU
    try:
        processor = OpticalFlowProcessor(
            config_path=config_path,
            device=f"cuda:{gpu_id}",
            result_dir=result_dir
        )
    except Exception as e:
        print(f"[GPU {gpu_id}] Error initializing processor: {str(e)}")
        return
    
    # 处理分配的视频列表
    for video_idx, video_name in enumerate(video_list):
        try:
            print(f"[GPU {gpu_id}] Processing video {video_idx+1}/{len(video_list)}: {video_name}")
            frames_dir = os.path.join(result_dir, video_name, "crop_frame")
            output_dir = os.path.join(result_dir, video_name)
            
            # 检查是否已经有NPZ文件
            npz_dir = os.path.join(output_dir, "results_opticalflow", "npz")
            if os.path.exists(npz_dir):
                npz_files = [f for f in os.listdir(npz_dir) if f.endswith('.npz')]
                if npz_files:
                    print(f"[GPU {gpu_id}] [SKIP] Video {video_name} already has {len(npz_files)} NPZ files")
                    continue
            
            # 检查输入目录
            if not os.path.isdir(frames_dir):
                print(f"[GPU {gpu_id}] Error: Frames directory not found for video {video_name}")
                continue
                
            # 检查帧文件
            frame_files = [f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png'))]
            if not frame_files:
                print(f"[GPU {gpu_id}] Error: No frame files found in {frames_dir}")
                continue
                
            print(f"[GPU {gpu_id}] Found {len(frame_files)} frames in {frames_dir}")
            
            # 处理视频帧
            start_time = time.time()
            processor.process_video_frames(frames_dir, output_dir, video_name)
            elapsed_time = time.time() - start_time
            print(f"[GPU {gpu_id}] Completed processing video: {video_name} in {elapsed_time:.1f}s")
            
        except Exception as e:
            print(f"[GPU {gpu_id}] Error processing video {video_name}: {str(e)}")
            import traceback
            print(traceback.format_exc())
            continue
    
    print(f"[GPU {gpu_id}] Completed all assigned videos")

def main():
    """主函数"""
    # Configuration
    config_path = "./PrimeDepth_inference/configs/inference.yaml"
    dataset_name = "test_dataset"
    #dataset_name = "AVIS_train"
    result_dir = f"../output/{dataset_name}_results"
    
    # 创建输出目录
    os.makedirs(result_dir, exist_ok=True)

    print(f"Processing dataset: {dataset_name}")
    print(f"Output directory: {result_dir}")
    
    # 检测可用的GPU数量
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs available")
    
    if num_gpus == 0:
        print("No GPUs available, falling back to single CPU mode")
        # 原始的单机处理代码
        processor = OpticalFlowProcessor(
            config_path=config_path,
            device="cpu",
            result_dir=result_dir
        )
        # ... 原始处理代码
        return

    # 获取要处理的视频列表
    try:
        video_names = [name for name in os.listdir(result_dir) 
                      if os.path.isdir(os.path.join(result_dir, name, "crop_frame"))]
        # test on 480 only
        #video_names = ["human_signer_480"]
        print(f"Found {len(video_names)} videos to process")
    except Exception as e:
        print(f"Error getting video list: {str(e)}")
        return
    
    if not video_names:
        print("No videos found to process")
        return
    
    # 将视频分配给不同的GPU
    videos_per_gpu = len(video_names) // num_gpus + (1 if len(video_names) % num_gpus > 0 else 0)
    gpu_tasks = []
    
    for gpu_id in range(num_gpus):
        start_idx = gpu_id * videos_per_gpu
        end_idx = min((gpu_id + 1) * videos_per_gpu, len(video_names))
        gpu_video_list = video_names[start_idx:end_idx]
        
        if gpu_video_list:  # 确保有视频要处理
            gpu_tasks.append((gpu_id, gpu_video_list, config_path, result_dir))
    
    print(f"\nDistributing {len(video_names)} videos across {len(gpu_tasks)} GPUs")
    for gpu_id, video_list, _, _ in gpu_tasks:
        print(f"GPU {gpu_id}: {len(video_list)} videos")
    
    # 使用线程池并行处理
    start_time = time.time()
    with Pool(processes=len(gpu_tasks)) as pool:
        pool.map(process_videos_on_gpu, gpu_tasks)
    
    elapsed_time = time.time() - start_time
    print(f"\nAll Optical Flow processing completed in {elapsed_time:.1f} seconds!")
    print(f"Average time per video: {elapsed_time/len(video_names):.1f} seconds")

if __name__ == '__main__':
    main()