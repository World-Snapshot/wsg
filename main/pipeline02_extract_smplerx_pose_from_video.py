import os
import argparse
import fcntl
import subprocess
import torch
import torch.multiprocessing as mp
from tqdm import tqdm


def process_video_with_smplerx(rank, crop_frame_path, video_name, folder_path, fps):
    """使用指定GPU处理crop_frame目录中的图片"""
    try:
        folder_path_results_smplerx = f"{folder_path}/results_smplerx"
        os.makedirs(folder_path_results_smplerx, exist_ok=True)
        print(f"GPU {rank}: SMPLer-X - 为 {video_name} 创建输出目录")
        
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # 设置使用的GPU
        # 注意：现在传递folder_path而不是video_path，因为extract_smplerx_slurm.sh会从crpo_frame目录读取图片
        cmd = [
            "sh",
            "extract_smplerx_slurm.sh",
            folder_path,
            "mp4",
            str(fps),
            "smpler_x_h32",
            folder_path,
        ]
        print(f"GPU {rank}: SMPLer-X - 运行命令: {cmd}")
        
        # 使用环境变量设置GPU
        my_env = os.environ.copy()
        my_env["CUDA_VISIBLE_DEVICES"] = str(rank)
        
        process = subprocess.run(
            cmd,
            env=my_env,
        )
        
        if process.returncode != 0:
            print(f"GPU {rank}: SMPLer-X - 处理失败，返回码={process.returncode}")
            return False
            
        print(f"GPU {rank}: SMPLer-X - 成功处理 {video_name}")
        return True
        
    except Exception as e:
        print(f"GPU {rank}: SMPLer-X - 处理 {video_name} 时发生错误: {str(e)}")
        return False

def append_completed_sample(path, video_name):
    with open(path, "a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(f"{video_name}\n")
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f, fcntl.LOCK_UN)


def process_videos_distributed(rank, world_size, video_paths, dataset_name, fps, smplerx_log):
    """分布式处理视频"""
    result_dir = f"../output/{dataset_name}_results"
    assigned_videos = video_paths[rank::world_size]

    with tqdm(
        assigned_videos,
        desc=f"{dataset_name} GPU {rank}",
        unit="sample",
        position=rank,
        dynamic_ncols=True,
    ) as progress:
        for video_name, crop_frame_path in progress:
            progress.set_postfix_str(video_name[:32])
            print(f"GPU {rank} 正在处理视频: {video_name}")
            
            try:
                folder_path = f"{result_dir}/{video_name}"
                
                # 重新读取已处理的视频列表（每次处理前都重新读取，确保最新状态）
                processed_smplerx = set()
                if os.path.exists(smplerx_log):
                    with open(smplerx_log, 'r') as f:
                        processed_smplerx = set(f.read().splitlines())
                
                # 检查视频是否已在txt文件中记录为已处理
                if video_name not in processed_smplerx:
                    print(f"GPU {rank}: 开始SMPLer-X处理 {video_name} (不在已处理列表中)")
                    success = process_video_with_smplerx(rank, crop_frame_path, video_name, folder_path, fps)
                    print(f"GPU {rank}: SMPLer-X处理 {'成功' if success else '失败'} - {video_name}")
                    
                    # 只有处理成功才写入txt文件
                    if success:
                        try:
                            append_completed_sample(smplerx_log, video_name)
                            print(f"GPU {rank}: 已将 {video_name} 添加到已处理列表")
                        except Exception as e:
                            print(f"GPU {rank}: 写入已处理列表失败: {str(e)}")
                else:
                    print(f"GPU {rank}: {video_name} 已在已处理列表中，跳过")
                            
            except Exception as e:
                print(f"GPU {rank}: 处理视频 {video_name} 时发生错误: {str(e)}")
                continue


def process_dataset(dataset_name, fps, completion_tag, workers):
    result_dir = f"../output/{dataset_name}_results"
    video_paths = []
    processed_smplerx = set()
    smplerx_log = f"{result_dir}/processed_video_name_by_smplerx_{completion_tag}.txt"
    
    if os.path.exists(smplerx_log):
        with open(smplerx_log, "r", encoding="utf-8") as f:
            processed_smplerx = set(f.read().splitlines())
        print(f"{dataset_name}: 已完成 trusted 提取样本数量: {len(processed_smplerx)}")

    # 从output目录检查需要SMPLer-X处理的视频
    if os.path.exists(result_dir):
        for video_name in sorted(os.listdir(result_dir)):
            video_path = os.path.join(result_dir, video_name)
            if os.path.isdir(video_path):
                crop_frame_path = os.path.join(video_path, "crop_frame")
                
                # 检查crop_frame目录是否存在和视频名是否在已处理列表中
                if os.path.exists(crop_frame_path):
                    if video_name not in processed_smplerx:
                        video_paths.append((video_name, crop_frame_path))
    else:
        print(f"结果目录不存在: {result_dir}")
        return

    if video_paths:
        print(f"{dataset_name}: 找到 {len(video_paths)} 个样本需要 trusted SMPLer-X 提取")
        # The coordinator may run in a CPU-only/base environment while each
        # extraction subprocess uses the dedicated smplerx CUDA environment.
        world_size = min(int(workers), len(video_paths))
        if world_size > 1:
            print(f"找到 {world_size} 个GPU，运行分布式处理")
            mp.spawn(
                process_videos_distributed,
                args=(world_size, video_paths, dataset_name, fps, smplerx_log),
                nprocs=world_size,
                join=True
            )
        else:
            print("只有一个GPU可用，运行单GPU模式")
            process_videos_distributed(0, 1, video_paths, dataset_name, fps, smplerx_log)
    else:
        print(f"{dataset_name}: 所有样本都已经完成 trusted SMPLer-X 提取")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["AVIS_train", "AVIS_val", "AVIS_test"],
        help="Dataset prefixes under ../output, without the _results suffix.",
    )
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--completion-tag",
        default="trusted_v2",
        help="Versioned completion log suffix used for safe resume.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    for dataset_name in args.datasets:
        process_dataset(dataset_name, args.fps, args.completion_tag, args.workers)

if __name__ == '__main__':
    main()
