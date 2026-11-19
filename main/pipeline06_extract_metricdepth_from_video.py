import os
import cv2
import numpy as np
import torch
from PIL import Image
from typing import Tuple, List
import multiprocessing as mp
from multiprocessing import Process, Queue
import time
from tqdm import tqdm
import shutil

# 优化设置
torch.backends.cudnn.benchmark = True      # 选择最快的内核
torch.set_float32_matmul_precision('high')  # 使用高精度矩阵乘法

class MetricDepthProcessor:
    def __init__(self, config_path: str, device: str = "auto", result_dir: str = None, batch_size: int = 1, gpu_id: int = None):
        """Initialize MetricDepth processor"""
        self.config_path = config_path
        self.gpu_id = gpu_id
        
        # 自动检测可用设备
        if device == "auto":
            if torch.cuda.is_available():
                if gpu_id is not None:
                    self.device = f"cuda:{gpu_id}"
                    print(f"使用GPU {gpu_id}")
                else:
                    self.device = "cuda"
                    print("CUDA可用，使用GPU")
            else:
                self.device = "cpu"
                print("CUDA不可用，切换到CPU")
        else:
            # 验证指定的设备是否可用
            if device == "cuda" and not torch.cuda.is_available():
                print("CUDA不可用，切换到CPU")
                self.device = "cpu"
            else:
                if gpu_id is not None and device == "cuda":
                    self.device = f"cuda:{gpu_id}"
                else:
                    self.device = device
        
        self.result_dir = result_dir
        self.batch_size = batch_size
        
        # Initialize record files
        if result_dir:
            self.depth_record_file = os.path.join(result_dir, "processed_video_name_by_metricdepth.txt")
            if not os.path.exists(self.depth_record_file):
                with open(self.depth_record_file, 'w') as f:
                    pass

        print(f"使用设备: {self.device}")
        print("Initializing MetricDepth model...")
        self.initialize_depth_model()

    # install the depth_pro
    # cd MetricDepth_inference, then run the following command:
    # pip install -e .
    # source get_pretrained_models.sh
    # cd ..
    def initialize_depth_model(self):
        """Initialize the depth estimation model"""
        try:
            import depth_pro
            
            # 创建推理引擎时传入正确的设备
            self.depth_engine, self.transform = depth_pro.create_model_and_transforms()
            self.depth_engine.eval()
            
            # 将模型移动到GPU
            if self.device.startswith("cuda"):
                self.depth_engine = self.depth_engine.to(self.device)
                print(f"Model moved to {self.device}")
                
                # 使用 torch.compile 优化模型
                print("Compiling model with torch.compile...")
                self.depth_engine = torch.compile(self.depth_engine, mode="reduce-overhead")
                print("Model compiled successfully")
            
            # 检查模型是否在GPU上
            if hasattr(self.depth_engine, 'parameters'):
                first_param = next(self.depth_engine.parameters())
                print(f"Model device: {first_param.device}")
            
            print(f"Initialized depth estimation model on {self.device}")
            
            # 预热模型（对 torch.compile 很重要）
            if self.device.startswith("cuda"):
                print("Warming up model...")
                dummy_input = torch.randn(1, 3, 1536, 1536).to(self.device)
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        _ = self.depth_engine.infer(dummy_input)
                print("Model warmed up")
            
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

    @torch.no_grad()
    def process_depth(self, frame_path: str) -> torch.Tensor:
        """Process depth for a single frame"""
        try:
            import depth_pro
            from torch.cuda.amp import autocast
            
            image, _, f_px = depth_pro.load_rgb(frame_path)
            image = self.transform(image)
            
            # 确保输入是 batch 格式 (1, C, H, W)
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            
            # 将输入移动到GPU
            if self.device.startswith("cuda"):
                image = image.to(self.device)
            
            # 使用混合精度推理
            if self.device.startswith("cuda"):
                with autocast():
                    prediction = self.depth_engine.infer(image, f_px=f_px)
            else:
                prediction = self.depth_engine.infer(image, f_px=f_px)
                
            return prediction["depth"]
        except Exception as e:
            print(f"\nError processing depth for frame {frame_path}: {str(e)}")
            raise

    def create_visualization_video(self, output_dir: str, total_frames: int, target_fps: int):
        """Create visualization videos using OpenCV"""
        try:
            # 检查第一帧是否存在
            first_frame_path = os.path.join(output_dir, "results_metricdepth_depth", "img", "00000001.jpg")
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
            fps = target_fps

            # 处理两种类型的视频：深度图和叠加深度图
            video_types = ['metricdepth', 'metricdepth_depth_overlay']
            
            for video_type in video_types:
                input_dir = {
                    'metricdepth': os.path.join(output_dir, "results_metricdepth_depth/img"),
                    'metricdepth_depth_overlay': os.path.join(output_dir, "metricdepth_depth_overlay_img")
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

    def process_video_frames(self, frames_dir: str, output_dir: str, video_name: str, target_fps: int, max_frames: int = None, remove_intermediate_files: bool = False):
        """Process all frames in a directory for depth estimation"""
        if not self.result_dir:
            raise ValueError("result_dir must be set to track video processing")

        # 检查是否需要处理
        if self.is_video_processed(video_name):
            print(f"Video {video_name} has already been processed by MetricDepth")
            return

        # 创建输出目录
        depth_data_dir = os.path.join(output_dir, "results_metricdepth_depth")
        depth_img_dir = os.path.join(depth_data_dir, "img")
        depth_npz_dir = os.path.join(depth_data_dir, "npz")
        depth_overlay_dir = os.path.join(output_dir, "metricdepth_depth_overlay_img")
        
        # 创建所有必要的目录
        for directory in [depth_img_dir, depth_npz_dir, depth_overlay_dir]:
            os.makedirs(directory, exist_ok=True)

        # 获取所有帧文件
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png'))])
        
        # 限制处理的帧数（用于测试）
        if max_frames is not None and max_frames > 0:
            frame_files = frame_files[:max_frames]
            print(f"Processing only first {max_frames} frames for testing")
        
        total_frames = len(frame_files)
        print(f"Found {total_frames} frames to process")

        try:
            import time
            start_time = time.time()
            
            for i, frame_file in enumerate(frame_files):
                try:
                    frame_start = time.time()
                    frame_path = os.path.join(frames_dir, frame_file)
                    frame_name = f"{i+1:08d}"
                    
                    # 读取原始帧用于叠加
                    frame_pil = Image.open(frame_path)
                    if i == 0:
                        print(f"\nImage size: {frame_pil.size}")
                    frame_cv2 = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

                    # 处理深度
                    print(f"Processing frame {i+1}/{total_frames}...", end='', flush=True)
                    depth = self.process_depth(frame_path)
                    depth = depth.squeeze().detach().cpu().numpy()
                    depth_norm = (depth - np.nanmin(depth)) / (np.nanmax(depth) - np.nanmin(depth))
                    depth_vis = (depth_norm * 255).astype(np.uint8)
                    depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                    
                    frame_time = time.time() - frame_start
                    elapsed_time = time.time() - start_time
                    avg_time_per_frame = elapsed_time / (i + 1)
                    remaining_frames = total_frames - (i + 1)
                    eta_seconds = remaining_frames * avg_time_per_frame
                    eta_minutes = eta_seconds / 60
                    
                    # 显示GPU内存使用情况（第一帧）
                    if i == 0 and self.device.startswith("cuda"):
                        import torch
                        if self.gpu_id is not None:
                            gpu_mem = torch.cuda.memory_allocated(self.gpu_id) / 1024**3
                            gpu_mem_reserved = torch.cuda.memory_reserved(self.gpu_id) / 1024**3
                        else:
                            gpu_mem = torch.cuda.memory_allocated() / 1024**3
                            gpu_mem_reserved = torch.cuda.memory_reserved() / 1024**3
                        print(f" done! ({frame_time:.2f}s, GPU: {gpu_mem:.1f}/{gpu_mem_reserved:.1f}GB, ETA: {eta_minutes:.1f} min)")
                    else:
                        print(f" done! ({frame_time:.2f}s, ETA: {eta_minutes:.1f} min)")

                    depth_ssi = depth
                    
                    # 保存深度相关输出
                    # 保存深度可视化图像
                    cv2.imwrite(os.path.join(depth_img_dir, f"{frame_name}.jpg"), np.array(depth_color))
                    
                    # 保存原始深度数据
                    np.savez(os.path.join(depth_npz_dir, f"{frame_name}.npz"), depth_data=depth_ssi)
                    
                    # 创建深度叠加效果
                    depth_overlay = cv2.addWeighted(frame_cv2, 0.5, np.array(depth_color), 0.5, 0)
                    cv2.imwrite(os.path.join(depth_overlay_dir, f"{frame_name}.png"), depth_overlay)


                except Exception as e:
                    print(f"Error processing frame {frame_file}: {str(e)}")
                    continue

            # 显示视频处理完成的总时间
            total_time = time.time() - start_time
            print(f"\nVideo processing completed in {total_time/60:.1f} minutes")
            
            # 创建视频文件
            print("\nCreating visualization videos...")
            self.create_visualization_video(output_dir, total_frames, target_fps)

            # 记录处理完成
            self.record_video_processed(video_name)
            
            # 删除中间文件（如果指定）
            if remove_intermediate_files:
                print("\nRemoving intermediate visualization files...")
                try:
                    # 删除 metricdepth_depth_overlay_img 文件夹
                    if os.path.exists(depth_overlay_dir):
                        shutil.rmtree(depth_overlay_dir)
                        print(f"Removed {depth_overlay_dir}")
                    
                    # 删除 results_metricdepth_depth/img 文件夹
                    if os.path.exists(depth_img_dir):
                        shutil.rmtree(depth_img_dir)
                        print(f"Removed {depth_img_dir}")
                except Exception as e:
                    print(f"Warning: Error removing intermediate files: {str(e)}")

        except Exception as e:
            print(f"Error processing video {video_name}: {str(e)}")
            import traceback
            print(traceback.format_exc())

def process_videos_on_gpu(gpu_id: int, video_list: List[tuple], config_path: str, 
                         result_dir: str, target_fps: int, progress_queue: Queue, 
                         remove_intermediate_files: bool = False):
    """Process a list of videos on a specific GPU"""
    
    # 设置当前进程使用的GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    print(f"GPU {gpu_id}: Starting worker process")
    
    try:
        # 初始化处理器，每个GPU只初始化一次
        processor = MetricDepthProcessor(
            config_path=config_path,
            device="cuda",
            result_dir=result_dir,
            gpu_id=0  # 因为设置了CUDA_VISIBLE_DEVICES，所以这里用0
        )
        print(f"GPU {gpu_id}: MetricDepth processor initialized successfully")
        
        # 处理分配的视频
        for video_info in video_list:
            video_idx, total_videos, video_name, frames_dir, output_dir = video_info
            
            print(f"\nGPU {gpu_id}: Processing video {video_idx}/{total_videos}: {video_name}")
            
            try:
                # 检查输入目录
                if not os.path.isdir(frames_dir):
                    print(f"GPU {gpu_id}: Error - Frames directory not found for video {video_name}")
                    progress_queue.put(('error', gpu_id, video_name))
                    continue
                
                # 检查帧文件
                frame_files = [f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png'))]
                if not frame_files:
                    print(f"GPU {gpu_id}: Error - No frame files found in {frames_dir}")
                    progress_queue.put(('error', gpu_id, video_name))
                    continue
                
                print(f"GPU {gpu_id}: Found {len(frame_files)} frames for {video_name}")
                
                # 处理视频
                start_time = time.time()
                processor.process_video_frames(frames_dir, output_dir, video_name, target_fps, 
                                             remove_intermediate_files=remove_intermediate_files)
                processing_time = time.time() - start_time
                
                print(f"GPU {gpu_id}: Completed {video_name} in {processing_time/60:.1f} minutes")
                progress_queue.put(('completed', gpu_id, video_name, processing_time))
                
            except Exception as e:
                print(f"GPU {gpu_id}: Error processing video {video_name}: {str(e)}")
                import traceback
                print(traceback.format_exc())
                progress_queue.put(('error', gpu_id, video_name))
    
    except Exception as e:
        print(f"GPU {gpu_id}: Fatal error in worker process: {str(e)}")
        import traceback
        print(traceback.format_exc())
    
    print(f"GPU {gpu_id}: Worker process finished")

def main():
    """主函数 - 支持多GPU并行处理"""
    # Configuration
    config_path = "./PrimeDepth_inference/configs/inference.yaml"
    dataset_name = "test_dataset"
    #dataset_name = "RealCam-Vid"
    #dataset_name = "AVIS_train"

    result_dir = f"../output/{dataset_name}_results"
    target_fps = 1  # 目标帧率
    use_multi_gpu = True  # 是否使用多GPU
    remove_intermediate_files = True  # 是否删除中间可视化文件
    
    # 创建输出目录
    os.makedirs(result_dir, exist_ok=True)

    print(f"Processing dataset: {dataset_name}")
    print(f"Output directory: {result_dir}")
    
    # 检查GPU数量
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No CUDA devices available, falling back to CPU processing")
        use_multi_gpu = False
    elif num_gpus == 1:
        print("Only 1 GPU available, using single GPU processing")
        use_multi_gpu = False
    else:
        print(f"Found {num_gpus} GPUs available for processing")
    
    # 获取要处理的视频列表
    try:
        all_video_names = [name for name in os.listdir(result_dir) 
                          if os.path.isdir(os.path.join(result_dir, name, "crop_frame"))]
        
        if not all_video_names:
            print("No videos found to process")
            return
        
        print(f"Found {len(all_video_names)} total videos")
        
        # 读取已处理的视频列表
        processed_videos = set()
        depth_record_file = os.path.join(result_dir, "processed_video_name_by_metricdepth.txt")
        if os.path.exists(depth_record_file):
            with open(depth_record_file, 'r') as f:
                processed_videos = set(f.read().splitlines())
            print(f"Found {len(processed_videos)} already processed videos")
        
        # 过滤出未处理的视频
        video_names = [name for name in all_video_names if name not in processed_videos]
        
        if not video_names:
            print("All videos have been processed!")
            return
            
        print(f"Found {len(video_names)} videos to process (excluding already processed)")
    except Exception as e:
        print(f"Error getting video list: {str(e)}")
        return

    # 准备视频信息列表
    video_list = []
    for idx, video_name in enumerate(video_names, 1):
        frames_dir = os.path.join(result_dir, video_name, "crop_frame")
        output_dir = os.path.join(result_dir, video_name)
        video_list.append((idx, len(video_names), video_name, frames_dir, output_dir))
    
    if use_multi_gpu and num_gpus > 1:
        # 多GPU并行处理
        print(f"\nUsing {num_gpus} GPUs for parallel processing")
        
        # 设置多进程启动方法
        mp.set_start_method('spawn', force=True)
        
        # 分配视频到各个GPU
        videos_per_gpu = len(video_list) // num_gpus
        gpu_assignments = []
        
        for gpu_id in range(num_gpus):
            start_idx = gpu_id * videos_per_gpu
            if gpu_id == num_gpus - 1:  # 最后一个GPU处理剩余的视频
                end_idx = len(video_list)
            else:
                end_idx = start_idx + videos_per_gpu
            
            gpu_videos = video_list[start_idx:end_idx]
            gpu_assignments.append(gpu_videos)
            print(f"GPU {gpu_id}: assigned {len(gpu_videos)} videos")
        
        # 创建进度队列
        progress_queue = mp.Queue()
        
        # 启动GPU进程
        processes = []
        for gpu_id in range(num_gpus):
            if gpu_assignments[gpu_id]:  # 只有分配了视频的GPU才启动进程
                p = Process(target=process_videos_on_gpu,
                           args=(gpu_id, gpu_assignments[gpu_id], config_path, 
                                result_dir, target_fps, progress_queue, 
                                remove_intermediate_files))
                p.start()
                processes.append(p)
                print(f"Started process for GPU {gpu_id}")
        
        # 监控进度
        completed_count = 0
        error_count = 0
        total_videos = len(video_list)
        
        print("\nProcessing videos...")
        pbar = tqdm(total=total_videos, desc="Overall Progress")
        
        while completed_count + error_count < total_videos:
            try:
                status = progress_queue.get(timeout=1)
                
                if status[0] == 'completed':
                    completed_count += 1
                    pbar.update(1)
                    pbar.set_postfix({'completed': completed_count, 'errors': error_count})
                elif status[0] == 'error':
                    error_count += 1
                    pbar.update(1)
                    pbar.set_postfix({'completed': completed_count, 'errors': error_count})
            except:
                # 检查进程是否还在运行
                if all(not p.is_alive() for p in processes):
                    break
        
        pbar.close()
        
        # 等待所有进程完成
        for p in processes:
            p.join()
        
        print(f"\nMulti-GPU processing completed: {completed_count} successful, {error_count} errors")
        
    else:
        # 单GPU或CPU处理
        print("\nUsing single GPU/CPU processing")
        
        # Initialize MetricDepth processor
        try:
            print("Initializing MetricDepth processor...")
            processor = MetricDepthProcessor(
                config_path=config_path,
                device="auto",
                result_dir=result_dir
            )
            print("Successfully initialized MetricDepth processor")
        except Exception as e:
            print(f"Error initializing MetricDepth processor: {str(e)}")
            return
        
        # 处理每个视频
        for video_idx, video_name in enumerate(video_names, 1):
            try:
                print(f"\nProcessing video {video_idx}/{len(video_names)}: {video_name}")
                frames_dir = os.path.join(result_dir, video_name, "crop_frame")
                output_dir = os.path.join(result_dir, video_name)
                
                # 检查输入目录
                if not os.path.isdir(frames_dir):
                    print(f"Error: Frames directory not found for video {video_name}")
                    continue
                    
                # 检查帧文件
                frame_files = [f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png'))]
                if not frame_files:
                    print(f"Error: No frame files found in {frames_dir}")
                    continue
                    
                print(f"Found {len(frame_files)} frames in {frames_dir}")
                
                # 处理视频帧
                processor.process_video_frames(frames_dir, output_dir, video_name, target_fps,
                                             remove_intermediate_files=remove_intermediate_files)
                print(f"Completed processing video: {video_name}")
                
            except Exception as e:
                print(f"Error processing video {video_name}: {str(e)}")
                import traceback
                print(traceback.format_exc())
                continue

    print("\nAll MetricDepth processing completed!")

if __name__ == '__main__':
    main()