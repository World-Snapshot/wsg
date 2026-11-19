import os
import sys
import shutil
import glob
from typing import List, Set


def delete_items_in_dataset(dataset_name: str, items_to_delete: List[str], dry_run: bool = False) -> tuple[int, int]:
    """
    删除指定数据集中每个视频文件夹里的特定文件和目录
    
    Args:
        dataset_name (str): 数据集名称 (如 "AVIS_val")
        items_to_delete (List[str]): 要删除的文件/目录列表
        dry_run (bool): 是否只显示将要删除的内容而不实际删除
    
    Returns:
        tuple[int, int]: (处理的文件夹数, 删除的项目数)
    """
    # 设置数据集路径
    dataset_path = os.path.join("../output", f"{dataset_name}_results")
    
    if not os.path.exists(dataset_path):
        print(f"错误: 数据集路径不存在: {dataset_path}")
        return 0, 0
    
    # 统计信息
    processed_folders = 0
    deleted_items = 0
    
    print("=" * 60)
    print(f"数据集清理工具 - {'模拟运行' if dry_run else '实际运行'}")
    print("=" * 60)
    print(f"数据集名称: {dataset_name}")
    print(f"数据集路径: {dataset_path}")
    print(f"要删除的项目: {', '.join(items_to_delete)}")
    print("=" * 60)
    
    # 遍历数据集中的每个视频文件夹
    for video_folder in sorted(os.listdir(dataset_path)):
        video_folder_path = os.path.join(dataset_path, video_folder)
        
        # 跳过非目录项
        if not os.path.isdir(video_folder_path):
            continue
        
        processed_folders += 1
        folder_deleted_items = []
        
        # 对每个要删除的项目进行处理
        for item in items_to_delete:
            # 支持通配符匹配
            if '*' in item:
                # 使用glob匹配文件
                pattern = os.path.join(video_folder_path, item)
                matched_items = glob.glob(pattern)
                
                for matched_path in matched_items:
                    matched_name = os.path.basename(matched_path)
                    if dry_run:
                        if os.path.isdir(matched_path):
                            folder_deleted_items.append(f"[目录] {matched_name}")
                        else:
                            folder_deleted_items.append(f"[文件] {matched_name}")
                    else:
                        try:
                            if os.path.isdir(matched_path):
                                shutil.rmtree(matched_path)
                                folder_deleted_items.append(f"[目录] {matched_name}")
                            else:
                                os.remove(matched_path)
                                folder_deleted_items.append(f"[文件] {matched_name}")
                        except Exception as e:
                            print(f"  错误: 无法删除 {matched_path}: {str(e)}")
            else:
                # 精确匹配
                item_path = os.path.join(video_folder_path, item)
                
                if os.path.exists(item_path):
                    if dry_run:
                        if os.path.isdir(item_path):
                            folder_deleted_items.append(f"[目录] {item}")
                        else:
                            folder_deleted_items.append(f"[文件] {item}")
                    else:
                        try:
                            if os.path.isdir(item_path):
                                shutil.rmtree(item_path)
                                folder_deleted_items.append(f"[目录] {item}")
                            else:
                                os.remove(item_path)
                                folder_deleted_items.append(f"[文件] {item}")
                        except Exception as e:
                            print(f"  错误: 无法删除 {item_path}: {str(e)}")
        
        # 显示每个文件夹的处理结果
        if folder_deleted_items:
            deleted_items += len(folder_deleted_items)
            print(f"\n{video_folder}:")
            for item in folder_deleted_items:
                print(f"  - {item}")
    
    # 显示统计结果
    print("\n" + "=" * 60)
    print(f"处理完成!")
    print(f"处理的视频文件夹数: {processed_folders}")
    print(f"{'将要删除' if dry_run else '已删除'}的项目总数: {deleted_items}")
    print("=" * 60)
    
    return processed_folders, deleted_items


def main():
    """主函数，处理命令行参数"""
    # 配置数据集名称选项
    #dataset_name = "AVIS_val"
    dataset_name = "MMFi_Dataset"
    #dataset_name = "AVIS_test"
    #dataset_name = "AVIS_train"
    # dataset_name = "test_dataset"
    # dataset_name = "ASLLRP"
    # dataset_name = "OpenVidHD-camera"
    
    # 配置要删除的项目
    items_to_delete = [
        "edge_overlay_img",
        "results_primedepth_depth/img",
        "primedepth_depth_overlay_img",
        "dwpose_overlay_img",
        "mediapipe_segmentation_overlay_img",
        "normal_overlay_img",
        "opticalflow_overlay_img",
        "curvature_overlay_img",
        "results_curvature/img",
        "results_curvature/npz", #npz文件
        "results_dwpose/img",
        "results_dwpose/json",
        "results_dwpose/npy",
        "results_dwpose/npz", #npz文件
        "results_edge/img",
        "results_edge/npz",
        "results_mediapipe/img",
        "results_mediapipe/json",
        "results_mediapipe/npz", #npz文件
        "results_mediapipe_segmentation/img",
        "results_mediapipe_segmentation/npz", #npz文件
        "results_metricdepth_depth/npz", #npz文件
        "results_normal/img",
        "results_normal/npz", #npz文件
        "results_opticalflow/img",
        "results_opticalflow/npz", #npz文件
        "results_primedepth_depth/npz", #npz文件
        "results_sapiens_depth/npz",
        "results_sapiens_normal/npz",
        "results_sapiens_pose/npz",
        "results_sapiens_depth/img",
        "results_sapiens_normal/img",
        "results_sapiens_pose/img",
        "results_sapiens_segmentation/img",
        "results_smplerx",
        "sapiens_depth_overlay_img",
        "sapiens_normal_overlay_img",
        "sapiens_pose_overlay_img",
        "sapiens_segmentation_overlay_img",
        "visualization_primedepth_depth_overlay.mp4",
        "visualization_primedepth.mp4", 
        "visualization_dwpose_overlap.mp4", 
        "visualization_dwpose.mp4", 
        "visualization_mediapipe_segmentation_overlap.mp4", 
        "visualization_mediapipe_segmentation.mp4", 
        "visualization_mediapipe.mp4", 
        "visualization_metricdepth_depth_overlay.mp4", 
        "visualization_metricdepth.mp4", 
        "visualization_normal_overlay.mp4", 
        "visualization_normal.mp4", 
        "visualization_sapiens_segmentation_overlap.mp4", 
        "visualization_sapiens_segmentation_overlap.mp4",
        "depth",
        "infra1",
        "infra2",
        "lidar",
        "mmwave",
        "wifi-csi",
        "ground_truth.npy",
        "rgb",
    ]
    



    # 其他常用配置选项（取消注释使用）
    # items_to_delete = [
    #     "visualization_dwpose.mp4",
    #     "visualization_dwpose_overlap.mp4",
    #     "results_dwpose",
    #     "dwpose_overlay_img"
    # ]
    
    # items_to_delete = [
    #     "visualization_mediapipe.mp4",
    #     "visualization_mediapipe_pose.mp4",
    #     "visualization_mediapipe_pose_overlay.mp4",
    #     "results_mediapipe",
    #     "results_mediapipe_pose",
    #     "mediapipe_pose_overlay_img"
    # ]
    
    # items_to_delete = [
    #     "visualization_smplerx_overlap.mp4",
    #     "results_smplerx",
    #     "smplerx_overlay_img"
    # ]
    
    # 删除所有可视化和结果文件
    # items_to_delete = [
    #     "visualization_*.mp4",
    #     "results_*",
    #     "*_overlay_img"
    # ]
    
    # 配置运行模式
    dry_run = False  # True: 只显示将要删除的内容，False: 实际执行删除
    
    # 解析命令行参数（可选）
    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help']:
            print_usage()
            sys.exit(0)
        # 支持命令行覆盖数据集名称
        dataset_name = sys.argv[1]
        if len(sys.argv) > 2:
            # 支持命令行指定删除项目
            items_to_delete = sys.argv[2:]
    
    # 执行删除操作
    delete_items_in_dataset(dataset_name, items_to_delete, dry_run)


def print_usage():
    """打印使用说明"""
    print("""
使用方法:
1. 直接运行（使用代码中配置的默认值）:
   python pipeline30_delete_middle_files.py

2. 指定数据集名称（覆盖代码中的配置）:
   python pipeline30_delete_middle_files.py AVIS_test

3. 指定数据集和删除项目:
   python pipeline30_delete_middle_files.py AVIS_val visualization_*.mp4 results_*

注意: 
- 修改代码中的 dry_run = False 来实际执行删除
- 支持通配符 * 来匹配多个文件
- 建议先用 dry_run = True 模式查看将要删除的内容

推荐用法：
直接修改代码中的配置，然后运行 python pipeline30_delete_middle_files.py
""")


if __name__ == '__main__':
    main()