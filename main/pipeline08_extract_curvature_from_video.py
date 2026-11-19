import os
import cv2
import numpy as np
import torch
from PIL import Image
from typing import Tuple
import threading
import time
try:
    import multiprocessing.dummy as mp_dummy
except ImportError:
    mp_dummy = None

class CurvatureProcessor:
    def __init__(self, config_path: str, device: str = "auto", result_dir: str = None, gpu_id: int = None):
        self.config_path = config_path
        
        # 设备设置
        if gpu_id is not None and torch.cuda.is_available():
            # 指定GPU设备
            self.device = f"cuda:{gpu_id}"
            torch.cuda.set_device(gpu_id)
            print(f"使用指定GPU: {gpu_id}")
        elif device == "auto":
            if torch.cuda.is_available():
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
                self.device = device
        
        self.result_dir = result_dir
        
        # Initialize record files
        if result_dir:
            self.depth_record_file = os.path.join(result_dir, "processed_video_name_by_curvature.txt")
            if not os.path.exists(self.depth_record_file):
                with open(self.depth_record_file, 'w') as f:
                    pass

        print(f"使用设备: {self.device}")
        print("Initializing Curvature (cv2) model...")
        self.initialize_depth_model()

    # install the depth_pro
    # cd curvature_inference, then run the following command:
    # pip install -e .
    # source get_pretrained_models.sh
    # cd ..
    def initialize_depth_model(self):
        """Initialize the depth estimation model"""
        try:            
            # 创建推理引擎时传入正确的设备\
            # no model is needed, use cv2 based method
            self.engine = None
            print(f"Initialized depth estimation model on {self.device}")
            
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

    def process_curvature(self, frame_path: str) -> Tuple[np.ndarray, Image.Image]:
        """Process curvature for a single frame"""
        try:
            if self.device != "cpu" and torch.cuda.is_available():
                # GPU加速版本
                rgb_image = cv2.imread(frame_path)
                gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
                
                # 转换为PyTorch张量并移到GPU
                gray_tensor = torch.from_numpy(gray_image).float().unsqueeze(0).unsqueeze(0).to(self.device)
                
                # 高斯模糊（使用PyTorch）
                gaussian_kernel = torch.tensor([[1, 4, 7, 4, 1],
                                               [4, 16, 26, 16, 4],
                                               [7, 26, 41, 26, 7],
                                               [4, 16, 26, 16, 4],
                                               [1, 4, 7, 4, 1]], dtype=torch.float32).to(self.device)
                gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
                gaussian_kernel = gaussian_kernel.view(1, 1, 5, 5)
                
                gray_blur = torch.nn.functional.conv2d(gray_tensor, gaussian_kernel, padding=2)
                
                # Laplacian算子
                laplacian_kernel = torch.tensor([[0, 1, 0],
                                                [1, -4, 1],
                                                [0, 1, 0]], dtype=torch.float32).to(self.device)
                laplacian_kernel = laplacian_kernel.view(1, 1, 3, 3)
                
                curvature_tensor = torch.nn.functional.conv2d(gray_blur, laplacian_kernel, padding=1)
                
                # 转回numpy
                curvature_map = curvature_tensor.squeeze().cpu().numpy()
                
            else:
                # CPU版本（原始代码）
                rgb_image = cv2.imread(frame_path)
                rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
                gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
                gray_blur = cv2.GaussianBlur(gray_image, (5, 5), 1)
                curvature_map = cv2.Laplacian(gray_blur, cv2.CV_64F)
            
            return curvature_map

        except Exception as e:
            print(f"Error processing curvature for frame {frame_path}: {str(e)}")
            raise

    def create_visualization_video(self, output_dir: str, total_frames: int):
        """Create visualization videos using OpenCV"""
        try:
            # 检查第一帧是否存在
            first_frame_path = os.path.join(output_dir, "results_curvature", "img", "00000001.jpg")
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
            video_types = ['curvature', 'curvature_overlay']
            
            for video_type in video_types:
                input_dir = {
                    'curvature': os.path.join(output_dir, "results_curvature/img"),
                    'curvature_overlay': os.path.join(output_dir, "curvature_overlay_img")
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

        # 创建输出目录路径
        depth_data_dir = os.path.join(output_dir, "results_curvature")
        depth_img_dir = os.path.join(depth_data_dir, "img")
        depth_npz_dir = os.path.join(depth_data_dir, "npz")
        depth_overlay_dir = os.path.join(output_dir, "curvature_overlay_img")
        
        # 检查 NPZ 文件夹是否存在且不为空
        if os.path.exists(depth_npz_dir):
            npz_files = [f for f in os.listdir(depth_npz_dir) if f.endswith('.npz')]
            if npz_files:
                print(f"[SKIP] Video {video_name} already has {len(npz_files)} NPZ files, skipping processing")
                return
        
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
                    
                    # 读取原始帧用于叠加
                    frame_pil = Image.open(frame_path)
                    frame_cv2 = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

                    # 处理深度
                    curvature_map = self.process_curvature(frame_path)
                    curvature_norm = cv2.normalize(curvature_map, None, 0, 255, cv2.NORM_MINMAX)
                    curvature_vis = curvature_norm.astype(np.uint8)
                    if curvature_vis.shape != frame_cv2.shape:
                        curvature_vis = cv2.resize(curvature_vis, (frame_cv2.shape[1], frame_cv2.shape[0]))
                    
                    cv2.imwrite(os.path.join(depth_img_dir, f"{frame_name}.jpg"), curvature_vis)
                    
                    # 异步保存原始深度数据
                    npz_path = os.path.join(depth_npz_dir, f"{frame_name}.npz")
                    save_thread = threading.Thread(
                        target=lambda: np.savez(npz_path, curvature_data=curvature_map.copy())
                    )
                    save_thread.start()
                    
                    # 创建深度叠加效果
                    # 将灰度图转换为3通道
                    curvature_vis_3ch = cv2.cvtColor(curvature_vis, cv2.COLOR_GRAY2BGR)
                    depth_overlay = cv2.addWeighted(frame_cv2, 0.5, curvature_vis_3ch, 0.5, 0)
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
    """Process a batch of videos on a specific GPU"""
    gpu_id, video_batch, dataset_name, result_dir, config_path, processed_videos_lock = args
    
    print(f"GPU {gpu_id}: Processing {len(video_batch)} videos")
    success_count = 0
    
    # Create processor instance for this GPU
    try:
        processor = CurvatureProcessor(
            config_path=config_path,
            device="cuda",
            result_dir=result_dir,
            gpu_id=gpu_id
        )
    except Exception as e:
        print(f"GPU {gpu_id}: Error initializing Curvature Processor: {str(e)}")
        return 0
    
    for video_name in video_batch:
        try:
            frames_dir = os.path.join(result_dir, video_name, "crop_frame")
            output_dir = os.path.join(result_dir, video_name)
            
            # Check if input directory exists
            if not os.path.isdir(frames_dir):
                print(f"GPU {gpu_id}: Error: Frames directory not found for video {video_name}")
                continue
                
            # Check frame files
            frame_files = [f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png'))]
            if not frame_files:
                print(f"GPU {gpu_id}: Error: No frame files found in {frames_dir}")
                continue
                
            # 检查 NPZ 文件夹是否存在且不为空
            npz_dir = os.path.join(output_dir, "results_curvature", "npz")
            if os.path.exists(npz_dir):
                npz_files = [f for f in os.listdir(npz_dir) if f.endswith('.npz')]
                if npz_files:
                    print(f"GPU {gpu_id}: [SKIP] Video {video_name} already has {len(npz_files)} NPZ files")
                    success_count += 1  # 计为成功，因为已经处理过
                    continue
            
            print(f"GPU {gpu_id}: Processing {video_name} with {len(frame_files)} frames")
            
            # Process video frames
            processor.process_video_frames(frames_dir, output_dir, video_name)
            success_count += 1
            print(f"GPU {gpu_id}: Completed processing video: {video_name}")
            
        except Exception as e:
            print(f"GPU {gpu_id}: Error processing video {video_name}: {str(e)}")
            import traceback
            print(traceback.format_exc())
            continue
    
    print(f"GPU {gpu_id}: Completed {success_count}/{len(video_batch)} videos")
    return success_count

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

    # Initialize curvature processor
    try:
        print("Initializing Curvature Processor...")
        processor = CurvatureProcessor(
            config_path=config_path,
            device="auto",  # 改为自动检测设备
            result_dir=result_dir
        )
        print("Successfully initialized Curvature Processor")
    except Exception as e:
        print(f"Error initializing Curvature Processor: {str(e)}")
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

    # Check for multi-GPU support
    if torch.cuda.is_available() and torch.cuda.device_count() > 1 and mp_dummy is not None:
        # Multi-GPU processing
        num_gpus = torch.cuda.device_count()
        print(f"Using {num_gpus} GPUs for parallel processing")
        
        # Split videos into batches for each GPU
        videos_per_gpu = len(video_names) // num_gpus
        video_batches = []
        
        for i in range(num_gpus):
            start_idx = i * videos_per_gpu
            if i == num_gpus - 1:
                # Last GPU gets remaining videos
                batch = video_names[start_idx:]
            else:
                batch = video_names[start_idx:start_idx + videos_per_gpu]
            video_batches.append(batch)
        
        # Progress tracking
        start_time = time.time()
        processed_videos_lock = threading.Lock()
        
        # Prepare arguments for each GPU
        gpu_args = []
        for gpu_id, batch in enumerate(video_batches):
            if batch:  # Only add if batch is not empty
                gpu_args.append((
                    gpu_id, batch, dataset_name, result_dir, config_path, processed_videos_lock
                ))
        
        print(f"Starting multi-GPU processing with {len(gpu_args)} GPU workers...")
        
        # Process on multiple GPUs in parallel
        with mp_dummy.Pool(len(gpu_args)) as pool:
            results = pool.map(process_videos_on_gpu, gpu_args)
        
        # Final results
        total_elapsed = time.time() - start_time
        total_success = sum(results)
        
        print(f"\nMulti-GPU processing complete!")
        print(f"Total processed: {total_success}/{len(video_names)} videos")
        print(f"Processing time: {total_elapsed/60:.1f} minutes")
        print(f"Average rate: {total_success/(total_elapsed/60):.1f} videos/minute")
        
    else:
        # Single-GPU or CPU processing (fallback)
        if torch.cuda.is_available():
            print("Using single GPU processing")
        else:
            print("Using CPU processing")
            
        # Initialize single processor
        try:
            processor = CurvatureProcessor(
                config_path=config_path,
                device="auto",
                result_dir=result_dir
            )
        except Exception as e:
            print(f"Error initializing Curvature Processor: {str(e)}")
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
                
                # 检查 NPZ 文件夹是否存在且不为空
                npz_dir = os.path.join(output_dir, "results_curvature", "npz")
                if os.path.exists(npz_dir):
                    npz_files = [f for f in os.listdir(npz_dir) if f.endswith('.npz')]
                    if npz_files:
                        print(f"[SKIP] Video {video_name} already has {len(npz_files)} NPZ files")
                        continue
                
                # 处理视频帧
                processor.process_video_frames(frames_dir, output_dir, video_name)
                print(f"Completed processing video: {video_name}")
                
            except Exception as e:
                print(f"Error processing video {video_name}: {str(e)}")
                import traceback
                print(traceback.format_exc())
                continue

    print("\nAll curvature processing completed!")

if __name__ == '__main__':
    main()