import os
import sys
import subprocess
from utils.preprocess_video import preprocess_videos, ensure_dir


def main(dataset_name="test_dataset", square_crop=False, target_fps=1, quality_preset="medium", target_resolution=None):
    """
    增强版视频预处理脚本 - 支持质量和分辨率控制
    
    Args:
        dataset_name (str): 数据集名称
        square_crop (bool): 是否进行方形裁剪
        target_fps (int): 目标帧率
        quality_preset (str): 质量预设 - "high", "medium", "low", "ultra_low"
        target_resolution (int): 目标分辨率（像素），None为保持原分辨率
    """
    # 配置数据集名称选项
    #dataset_name = "test_dataset"
    # dataset_name = "ASLLRP"
    # dataset_name = "how2sign_test"
    # dataset_name = "ncslgr_mp4"
    # dataset_name = "ncslgr_mp4_small"
    # dataset_name = "ASLLRP_batch_full_video_v1_1_small"
    # dataset_name = "ASLLRP_utterances_small"
    #dataset_name = "OpenVidHD-camera"  # 新增大数据集
    
    # === 配置处理参数 ===
    #square_crop = True      # 是否进行方形裁剪，False为不裁剪保持原始比例
    #target_fps = 8           # 目标帧率（推荐：1-30）
    # quality_preset = "medium" # 质量预设："high", "medium", "low", "ultra_low"
    # target_resolution = None  # 目标分辨率（像素），如：512, 256, None为原分辨率
    
    # === 常用配置组合示例 ===
    
    # 1. 高质量处理（适合小数据集）
    # quality_preset = "high"
    # target_resolution = None  # 保持原分辨率
    
    # 2. 平衡质量与空间（推荐）
    # quality_preset = "medium"
    # target_resolution = 512
    
    # 3. 节省空间（适合大数据集）
    # quality_preset = "low" 
    # target_resolution = 384
    
    # 3.5 最大压缩（适合超大数据集或存储受限）
    #quality_preset = "low"
    #target_resolution = 256

    # 4. 最大压缩（适合超大数据集或存储受限）
    # quality_preset = "ultra_low"
    # target_resolution = 256
    
    # 5. 特殊用途：极低帧率 + 小分辨率（用于快速原型测试）
    # target_fps = 1
    # quality_preset = "low"
    # target_resolution = 224
    
    # 设置输入和输出目录
    video_dir = ensure_dir(f"../input/{dataset_name}")
    result_dir = ensure_dir(f"../output/{dataset_name}_results")
    
    # 显示配置信息
    crop_status = "方形裁剪" if square_crop else "保持原始比例"
    resolution_info = f"{target_resolution}px" if target_resolution else "原分辨率"
    
    print("=" * 60)
    print("增强版视频预处理")
    print("=" * 60)
    print(f"数据集名称: {dataset_name}")
    print(f"输入目录: {video_dir}")
    print(f"输出目录: {result_dir}")
    print(f"处理模式: {crop_status}")
    print(f"目标帧率: {target_fps} FPS")
    print(f"质量预设: {quality_preset}")
    print(f"目标分辨率: {resolution_info}")
    
    # 估算压缩效果
    compression_estimates = {
        ("high", None): "文件大小约为原始的 80-100%",
        ("medium", None): "文件大小约为原始的 40-60%", 
        ("low", None): "文件大小约为原始的 20-40%",
        ("ultra_low", None): "文件大小约为原始的 10-20%",
        ("medium", 512): "文件大小约为原始的 15-25%",
        ("low", 384): "文件大小约为原始的 8-15%",
        ("ultra_low", 256): "文件大小约为原始的 3-8%"
    }
    
    estimate_key = (quality_preset, target_resolution)
    if estimate_key in compression_estimates:
        print(f"预估效果: {compression_estimates[estimate_key]}")
    else:
        print("预估效果: 根据设置自定义压缩")
    
    print("=" * 60)
    
    # 执行视频预处理
    try:
        preprocess_videos(
            video_dir=video_dir, 
            dataset_name=dataset_name, 
            square_crop=square_crop, 
            fps=target_fps,
            quality_preset=quality_preset,
            target_resolution=target_resolution
        )
        print("\n" + "=" * 60)
        print("视频预处理完成!")
        
        # 统计处理结果
        processed_count = 0
        total_size = 0
        
        for video_file in os.listdir(video_dir):
            if video_file.endswith(".mp4"):
                video_name = os.path.splitext(video_file)[0]
                output_video_path = os.path.join(result_dir, video_name, "crop_original_video.mp4")
                
                if os.path.exists(output_video_path):
                    processed_count += 1
                    total_size += os.path.getsize(output_video_path)
        
        total_size_gb = total_size / (1024**3)
        
        print(f"成功处理了 {processed_count} 个视频")
        print(f"输出视频总大小: {total_size_gb:.2f} GB")
        print("=" * 60)
        
    except Exception as e:
        print(f"预处理过程中发生错误: {str(e)}")


def print_usage():
    """打印使用说明"""
    print("""
使用方法:
1. 基本用法（使用默认参数）:
   python script.py

2. 指定数据集:
   python script.py dataset_name

3. 完整参数:
   python script.py dataset_name square_crop fps quality_preset resolution

参数说明:
- dataset_name: 数据集名称 (如: "test_dataset")
- square_crop: true/false (是否方形裁剪)
- fps: 帧率 (如: 30, 15, 1)
- quality_preset: high/medium/low/ultra_low (质量预设)
- resolution: 目标分辨率像素 (如: 512, 256, 或 none 保持原分辨率)

示例:
python script.py OpenVidHD false 1 low 384
python script.py test_dataset true 30 medium 512
python script.py my_dataset false 15 ultra_low 256

推荐配置:
- 小数据集: medium质量 + 原分辨率
- 大数据集: low质量 + 512分辨率  
- 超大数据集: ultra_low质量 + 256分辨率
""")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help', 'help']:
            print_usage()
            sys.exit(0)
            
        dataset_name = sys.argv[1]
        square_crop = sys.argv[2].lower() == 'true' if len(sys.argv) > 2 else False
        target_fps = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        quality_preset = sys.argv[4] if len(sys.argv) > 4 else "medium"
        
        # 处理分辨率参数
        if len(sys.argv) > 5:
            resolution_str = sys.argv[5].lower()
            target_resolution = None if resolution_str in ['none', 'null', 'original'] else int(resolution_str)
        else:
            target_resolution = None
            
        main(dataset_name, square_crop, target_fps, quality_preset, target_resolution)
    else:
        main()