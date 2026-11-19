import os
import sys
import cv2
import time
import tempfile
import math
import multiprocessing as mp
import numpy as np
import torch
from PIL import Image

try:
    from decord import VideoReader, cpu
except ImportError as exc:
    raise ImportError("decord is required for pipeline07_extract_normals_from_video.py") from exc


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
NORMALCRAFTER_DIR = os.path.join(CURRENT_DIR, "NormalCrafter_inference")
MODEL_CACHE_DIR = os.path.join(CURRENT_DIR, "models")
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
os.environ.setdefault("HF_HOME", MODEL_CACHE_DIR)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(MODEL_CACHE_DIR, "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(MODEL_CACHE_DIR, "hub"))
if NORMALCRAFTER_DIR not in sys.path:
    sys.path.insert(0, NORMALCRAFTER_DIR)

from diffusers import AutoencoderKLTemporalDecoder
from diffusers.training_utils import set_seed
from normalcrafter.normal_crafter_ppl import NormalCrafterPipeline
from normalcrafter.unet import DiffusersUNetSpatioTemporalConditionModelNormalCrafter


class NormalProcessor:
    def __init__(
        self,
        device: str = "auto",
        result_dir: str = None,
        gpu_id: int = None,
        max_res: int = 1024,
        process_length: int = -1,
        target_fps: int = -1,
        window_size: int = 14,
        time_step_size: int = 10,
        decode_chunk_size: int = 7,
        seed: int = 42,
        unet_path: str = "Yanrui95/NormalCrafter",
        pretrain_path: str = "stabilityai/stable-video-diffusion-img2vid-xt",
        save_visualizations: bool = False,
    ):
        if gpu_id is not None and torch.cuda.is_available():
            self.device = f"cuda:{gpu_id}"
            torch.cuda.set_device(gpu_id)
        elif device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            if device.startswith("cuda") and not torch.cuda.is_available():
                print("CUDA unavailable, falling back to CPU")
                self.device = "cpu"
            else:
                self.device = device

        self.result_dir = result_dir
        self.max_res = max_res
        self.process_length = process_length
        self.target_fps = target_fps
        self.window_size = window_size
        self.time_step_size = time_step_size
        self.decode_chunk_size = decode_chunk_size
        self.seed = seed
        self.unet_path = unet_path
        self.pretrain_path = pretrain_path
        self.save_visualizations = save_visualizations
        self.weight_dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.cache_dir = MODEL_CACHE_DIR

        if result_dir:
            self.depth_record_file = os.path.join(result_dir, "processed_video_name_by_normal.txt")
            if not os.path.exists(self.depth_record_file):
                with open(self.depth_record_file, "w") as f:
                    f.write("")
            self.processed_videos = self._load_processed_videos()
        else:
            self.processed_videos = set()

        print(f"Using device: {self.device}")
        print("Initializing NormalCrafter model...")
        self.initialize_normal_model()

    def initialize_normal_model(self):
        unet = DiffusersUNetSpatioTemporalConditionModelNormalCrafter.from_pretrained(
            self.unet_path,
            subfolder="unet",
            low_cpu_mem_usage=False,
            cache_dir=self.cache_dir,
            torch_dtype=self.weight_dtype,
        )
        vae = AutoencoderKLTemporalDecoder.from_pretrained(
            self.unet_path,
            subfolder="vae",
            cache_dir=self.cache_dir,
            torch_dtype=self.weight_dtype,
        )

        self.pipe = NormalCrafterPipeline.from_pretrained(
            self.pretrain_path,
            unet=unet,
            vae=vae,
            torch_dtype=self.weight_dtype,
            variant="fp16" if self.weight_dtype == torch.float16 else None,
            cache_dir=self.cache_dir,
        )
        self.pipe.to(self.device)

        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("Enabled xformers memory efficient attention")
        except Exception as exc:
            print(f"xformers not enabled: {exc}")

    def is_video_processed(self, video_name: str) -> bool:
        return video_name in self.processed_videos

    def _load_processed_videos(self):
        if not os.path.exists(self.depth_record_file):
            return set()
        with open(self.depth_record_file, "r") as f:
            return {line.strip() for line in f if line.strip()}

    def record_video_processed(self, video_name: str):
        if video_name in self.processed_videos:
            return
        with open(self.depth_record_file, "a") as f:
            f.write(f"{video_name}\n")
        self.processed_videos.add(video_name)

    def read_video_frames(self, video_path: str):
        print(f"==> processing video: {video_path}")
        vid = VideoReader(video_path, ctx=cpu(0))
        first_frame = vid.get_batch([0]).asnumpy()[0]
        original_height, original_width = first_frame.shape[:2]

        if max(original_height, original_width) > self.max_res:
            scale = self.max_res / max(original_height, original_width)
            height = max(1, round(original_height * scale))
            width = max(1, round(original_width * scale))
        else:
            height = original_height
            width = original_width

        vid = VideoReader(video_path, ctx=cpu(0), width=width, height=height)

        source_fps = float(vid.get_avg_fps())
        fps = source_fps if self.target_fps == -1 else float(self.target_fps)
        stride = max(round(source_fps / fps), 1)
        frame_indices = list(range(0, len(vid), stride))
        if self.process_length != -1 and self.process_length < len(frame_indices):
            frame_indices = frame_indices[: self.process_length]

        print(f"==> original shape: {(len(vid), original_height, original_width)}")
        print(f"==> resized shape: {(len(vid), height, width)}")
        print(f"==> using stride {stride}, final frames: {len(frame_indices)}")

        frames = vid.get_batch(frame_indices).asnumpy().astype(np.uint8)
        frames = [Image.fromarray(frame) for frame in frames]
        return frames, fps

    def infer_video_normals(self, video_path: str):
        frames, fps = self.read_video_frames(video_path)
        set_seed(self.seed)
        num_frames = len(frames)
        if num_frames == 0:
            raise ValueError(f"No frames were decoded from {video_path}")

        effective_window_size = min(self.window_size, num_frames)
        effective_time_step_size = max(1, min(self.time_step_size, effective_window_size))
        effective_decode_chunk_size = max(1, min(self.decode_chunk_size, num_frames))

        with torch.inference_mode():
            normal_sequence = self.pipe(
                frames,
                decode_chunk_size=effective_decode_chunk_size,
                time_step_size=effective_time_step_size,
                window_size=effective_window_size,
            ).frames[0]

        normal_sequence = np.asarray(normal_sequence, dtype=np.float32)
        if normal_sequence.ndim != 4 or normal_sequence.shape[-1] != 3:
            raise ValueError(f"Unexpected normal output shape: {normal_sequence.shape}")

        normal_sequence = np.clip(normal_sequence, -1.0, 1.0)
        return normal_sequence, fps

    def build_temp_video_from_frames(self, frames_dir: str):
        frame_files = sorted(
            f for f in os.listdir(frames_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        if not frame_files:
            raise FileNotFoundError(f"No frames found in {frames_dir}")

        first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
        if first_frame is None:
            raise ValueError(f"Failed to read first frame in {frames_dir}")

        temp_dir = tempfile.mkdtemp(prefix="normalcrafter_")
        temp_video_path = os.path.join(temp_dir, "input.mp4")
        writer = cv2.VideoWriter(
            temp_video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            30,
            (first_frame.shape[1], first_frame.shape[0]),
        )
        for frame_file in frame_files:
            frame = cv2.imread(os.path.join(frames_dir, frame_file))
            if frame is not None:
                writer.write(frame)
        writer.release()
        return temp_video_path, temp_dir

    def create_visualization_video(self, output_dir: str, total_frames: int, fps: float):
        first_frame_path = os.path.join(output_dir, "results_normal", "img", "00000001.jpg")
        if not os.path.exists(first_frame_path):
            print(f"Missing first visualization frame: {first_frame_path}")
            return

        first_frame = cv2.imread(first_frame_path)
        if first_frame is None:
            print(f"Failed to read first visualization frame: {first_frame_path}")
            return

        frame_size = (first_frame.shape[1], first_frame.shape[0])
        output_specs = {
            "normal": (os.path.join(output_dir, "results_normal", "img"), "jpg"),
            "normal_overlay": (os.path.join(output_dir, "normal_overlay_img"), "png"),
        }

        for name, (input_dir, extension) in output_specs.items():
            if not os.path.isdir(input_dir):
                continue
            output_path = os.path.join(output_dir, f"visualization_{name}.mp4")
            writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps if fps > 0 else 30,
                frame_size,
            )
            if not writer.isOpened():
                print(f"Failed to create video writer for {output_path}")
                continue

            for frame_num in range(1, total_frames + 1):
                frame_path = os.path.join(input_dir, f"{frame_num:08d}.{extension}")
                frame = cv2.imread(frame_path)
                if frame is not None:
                    writer.write(frame)
            writer.release()

    def process_video_frames(self, frames_dir: str, output_dir: str, video_name: str):
        if not self.result_dir:
            raise ValueError("result_dir must be set to track video processing")

        if self.is_video_processed(video_name):
            print(f"[SKIP] Video {video_name} already recorded as completed")
            return

        normal_data_dir = os.path.join(output_dir, "results_normal")
        normal_npz_dir = os.path.join(normal_data_dir, "npz")
        os.makedirs(normal_npz_dir, exist_ok=True)

        normal_img_dir = os.path.join(normal_data_dir, "img")
        normal_overlay_dir = os.path.join(output_dir, "normal_overlay_img")
        if self.save_visualizations:
            os.makedirs(normal_img_dir, exist_ok=True)
            os.makedirs(normal_overlay_dir, exist_ok=True)

        existing_npz = [f for f in os.listdir(normal_npz_dir) if f.endswith(".npz")]
        if existing_npz:
            print(f"[SKIP] Video {video_name} already has {len(existing_npz)} NPZ files")
            return

        frame_files = sorted(
            f for f in os.listdir(frames_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        if not frame_files:
            raise FileNotFoundError(f"No frame files found in {frames_dir}")

        original_video_path = os.path.join(output_dir, "crop_original_video.mp4")
        video_path = original_video_path
        temp_video_path = None
        temp_video_dir = None
        if not os.path.exists(video_path):
            temp_video_path, temp_video_dir = self.build_temp_video_from_frames(frames_dir)
            video_path = temp_video_path

        try:
            try:
                normal_sequence, fps = self.infer_video_normals(video_path)
            except Exception:
                if video_path == original_video_path:
                    print(f"Falling back to crop_frame for {video_name}")
                    temp_video_path, temp_video_dir = self.build_temp_video_from_frames(frames_dir)
                    video_path = temp_video_path
                    normal_sequence, fps = self.infer_video_normals(video_path)
                else:
                    raise
            total_frames = min(len(frame_files), len(normal_sequence))
            if total_frames != len(frame_files):
                print(
                    f"Warning: frame count mismatch for {video_name}: "
                    f"frames_dir={len(frame_files)}, normal_sequence={len(normal_sequence)}"
                )

            for i in range(total_frames):
                frame_file = frame_files[i]
                frame_path = os.path.join(frames_dir, frame_file)
                frame_name = f"{i + 1:08d}"

                normal_map = normal_sequence[i]
                if self.save_visualizations:
                    frame_pil = Image.open(frame_path).convert("RGB")
                    frame_cv2 = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
                    target_height, target_width = frame_cv2.shape[:2]
                else:
                    with Image.open(frame_path) as frame_pil:
                        target_width, target_height = frame_pil.size

                if normal_map.shape[:2] != (target_height, target_width):
                    normal_map = cv2.resize(
                        normal_map,
                        (target_width, target_height),
                        interpolation=cv2.INTER_LINEAR,
                    )
                normal_map = np.clip(normal_map.astype(np.float32), -1.0, 1.0)

                if self.save_visualizations:
                    normal_vis = ((normal_map + 1.0) / 2.0).clip(0, 1)
                    normal_vis = (normal_vis * 255).astype(np.uint8)
                    normal_color = cv2.cvtColor(normal_vis, cv2.COLOR_RGB2BGR)
                    normal_overlay = cv2.addWeighted(frame_cv2, 0.5, normal_color, 0.5, 0)

                    cv2.imwrite(os.path.join(normal_img_dir, f"{frame_name}.jpg"), normal_color)
                    cv2.imwrite(os.path.join(normal_overlay_dir, f"{frame_name}.png"), normal_overlay)
                np.savez(os.path.join(normal_npz_dir, f"{frame_name}.npz"), normal_data=normal_map)

                if (i + 1) % 10 == 0 or i == 0 or i + 1 == total_frames:
                    print(f"Processed {i + 1}/{total_frames} frames ({(i + 1) / total_frames * 100:.1f}%)")

            if self.save_visualizations:
                self.create_visualization_video(output_dir, total_frames, fps)
            self.record_video_processed(video_name)
        finally:
            if temp_video_dir and os.path.isdir(temp_video_dir):
                for root, dirs, files in os.walk(temp_video_dir, topdown=False):
                    for name in files:
                        os.remove(os.path.join(root, name))
                    for name in dirs:
                        os.rmdir(os.path.join(root, name))
                os.rmdir(temp_video_dir)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def process_videos_on_gpu(gpu_id, video_batch, result_dir, processor_kwargs, result_queue):
    print(f"GPU {gpu_id}: Processing {len(video_batch)} videos")

    try:
        processor = NormalProcessor(
            device="cuda",
            result_dir=result_dir,
            gpu_id=gpu_id,
            **processor_kwargs,
        )
    except Exception as exc:
        print(f"GPU {gpu_id}: Failed to initialize NormalCrafter: {exc}")
        result_queue.put((gpu_id, 0))
        return

    success_count = 0
    for video_name in video_batch:
        if processor.is_video_processed(video_name):
            print(f"GPU {gpu_id}: Skipping completed {video_name}")
            continue
        frames_dir = os.path.join(result_dir, video_name, "crop_frame")
        output_dir = os.path.join(result_dir, video_name)
        try:
            processor.process_video_frames(frames_dir, output_dir, video_name)
            success_count += 1
            print(f"GPU {gpu_id}: Completed {video_name}")
        except Exception as exc:
            print(f"GPU {gpu_id}: Error processing {video_name}: {exc}")
    result_queue.put((gpu_id, success_count))


def run_multi_gpu(video_names, result_dir, processor_kwargs):
    num_gpus = min(torch.cuda.device_count(), len(video_names))
    if num_gpus <= 1:
        return None

    print(f"Using {num_gpus} GPUs for parallel processing")
    videos_per_gpu = math.ceil(len(video_names) / num_gpus)
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    processes = []

    for gpu_id in range(num_gpus):
        start_idx = gpu_id * videos_per_gpu
        end_idx = min(start_idx + videos_per_gpu, len(video_names))
        batch = video_names[start_idx:end_idx]
        if not batch:
            continue
        process = ctx.Process(
            target=process_videos_on_gpu,
            args=(gpu_id, batch, result_dir, processor_kwargs, result_queue),
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    total_success = 0
    while not result_queue.empty():
        _, success_count = result_queue.get()
        total_success += success_count
    return total_success


def main():
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("WSG_DATASET_NAME", "test_dataset")
    result_dir = f"../output/{dataset_name}_results"
    enable_multi_gpu = True

    processor_kwargs = {
        "max_res": 1024,
        "process_length": -1,
        "target_fps": -1,
        "window_size": 14,
        "time_step_size": 10,
        "decode_chunk_size": 7,
        "seed": 42,
        "save_visualizations": False,
    }

    os.makedirs(result_dir, exist_ok=True)
    print(f"Processing dataset: {dataset_name}")
    print(f"Output directory: {result_dir}")

    video_names = [
        name for name in os.listdir(result_dir)
        if os.path.isdir(os.path.join(result_dir, name, "crop_frame"))
    ]
    print(f"Found {len(video_names)} videos to process")

    if enable_multi_gpu and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        start_time = time.time()
        total_success = run_multi_gpu(video_names, result_dir, processor_kwargs)
        if total_success is not None:
            elapsed = time.time() - start_time
            print(f"Processed {total_success}/{len(video_names)} videos in {elapsed / 60:.1f} minutes")
            return

    try:
        processor = NormalProcessor(device="auto", result_dir=result_dir, **processor_kwargs)
    except Exception as exc:
        print(f"Failed to initialize NormalCrafter: {exc}")
        return

    for idx, video_name in enumerate(video_names, 1):
        frames_dir = os.path.join(result_dir, video_name, "crop_frame")
        output_dir = os.path.join(result_dir, video_name)
        print(f"\nProcessing video {idx}/{len(video_names)}: {video_name}")
        try:
            processor.process_video_frames(frames_dir, output_dir, video_name)
            print(f"Completed processing video: {video_name}")
        except Exception as exc:
            print(f"Error processing video {video_name}: {exc}")

    print("\nAll normals processing completed!")


if __name__ == "__main__":
    main()
