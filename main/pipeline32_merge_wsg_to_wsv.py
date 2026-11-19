#!/usr/bin/env python3
"""
Pipeline 32: 合并WSG文件到WSV格式
将每个实验文件夹下的所有.wsg文件合并成一个大的.wsv文件
"""

import os
import sys
import numpy as np
import glob
from pathlib import Path
from tqdm import tqdm
import shutil
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed


def discover_experiment_folders(dataset_path):
    """发现数据集下所有包含 wsg 文件夹的一级视频/实验目录。"""
    experiment_folders = []
    for item in os.listdir(dataset_path):
        item_path = os.path.join(dataset_path, item)
        if not os.path.isdir(item_path):
            continue
        wsg_folder = os.path.join(item_path, 'wsg')
        if os.path.isdir(wsg_folder):
            experiment_folders.append((item, wsg_folder))
    experiment_folders.sort(key=lambda x: x[0])
    return experiment_folders

def get_frame_number(filename):
    """从文件名中提取帧号"""
    try:
        # 假设文件名格式为 frameXXX.wsg 或 XXX.wsg
        base = os.path.basename(filename).replace('.wsg', '')
        if base.startswith('frame'):
            base = base.replace('frame', '')
        return int(base)
    except:
        return -1

def merge_wsg_to_wsv(wsg_folder, output_wsv_path, delete_wsg=True):
    """
    将一个文件夹下的所有WSG文件合并成一个WSV文件
    
    Args:
        wsg_folder: 包含.wsg文件的文件夹路径
        output_wsv_path: 输出的.wsv文件路径
        delete_wsg: 是否在合并后删除原始.wsg文件
    
    Returns:
        bool: 是否成功
    """
    try:
        # 获取所有wsg文件
        wsg_files = glob.glob(os.path.join(wsg_folder, '*.wsg'))
        if not wsg_files:
            return False
        
        # 按帧号排序
        wsg_files_with_numbers = [(f, get_frame_number(f)) for f in wsg_files]
        wsg_files_with_numbers = [f for f in wsg_files_with_numbers if f[1] >= 0]
        wsg_files_with_numbers.sort(key=lambda x: x[1])
        
        if not wsg_files_with_numbers:
            print(f"  警告: 无法解析文件名中的帧号")
            return False
        
        # 创建一个字典来存储所有数据
        merged_data = {}
        
        # 读取每个wsg文件并添加到字典中
        for wsg_file, frame_num in wsg_files_with_numbers:
            try:
                # 加载wsg文件（实际上是.npz格式）
                data = np.load(wsg_file, allow_pickle=True)
                
                # 使用索引格式存储，例如 index_00000001
                index_key = f"index_{frame_num:08d}"
                
                # 假设wsg文件中有一个主要的数组（通常叫'wsg_data'）
                if 'wsg_data' in data:
                    merged_data[index_key] = data['wsg_data']
                else:
                    # 如果没有'wsg_data'，尝试获取第一个数组
                    keys = list(data.keys())
                    if keys:
                        merged_data[index_key] = data[keys[0]]
                
                data.close()
                
            except Exception as e:
                print(f"  警告: 无法读取 {wsg_file}: {e}")
                continue
        
        if not merged_data:
            print(f"  错误: 没有成功读取任何WSG文件")
            return False
        
        # 创建输出目录
        os.makedirs(os.path.dirname(output_wsv_path), exist_ok=True)
        
        # 保存合并后的数据为.wsv文件
        np.savez_compressed(output_wsv_path.replace('.wsv', ''), **merged_data)
        
        # 重命名为.wsv
        if os.path.exists(output_wsv_path.replace('.wsv', '.npz')):
            os.rename(output_wsv_path.replace('.wsv', '.npz'), output_wsv_path)
        
        print(f"  ✓ 成功合并 {len(merged_data)} 帧到 {output_wsv_path}")
        
        # 如果需要，删除原始wsg文件
        if delete_wsg:
            for wsg_file, _ in wsg_files_with_numbers:
                try:
                    os.remove(wsg_file)
                except:
                    pass
            
            # 删除空的wsg文件夹
            try:
                os.rmdir(wsg_folder)
                print(f"  ✓ 已删除原始wsg文件夹")
            except:
                print(f"  警告: 无法删除wsg文件夹，可能还有其他文件")
        
        return True
        
    except Exception as e:
        print(f"  错误: {e}")
        return False


def process_single_experiment(task):
    exp_name, wsg_folder, dry_run, delete_wsg = task
    wsv_path = os.path.join(os.path.dirname(wsg_folder), 'wsv', 'merge.wsv')

    if os.path.exists(wsv_path):
        return ("skip", exp_name, "已存在WSV文件，跳过")

    wsg_files = glob.glob(os.path.join(wsg_folder, '*.wsg'))
    if not wsg_files:
        return ("skip", exp_name, "未找到WSG文件，跳过")

    if dry_run:
        return ("success", exp_name, f"[模拟] 将合并 {len(wsg_files)} 个WSG到 {wsv_path}")

    ok = merge_wsg_to_wsv(wsg_folder, wsv_path, delete_wsg)
    if ok:
        return ("success", exp_name, f"成功合并 {len(wsg_files)} 个WSG到 {wsv_path}")
    return ("fail", exp_name, "合并失败")


def process_dataset(dataset_path, dry_run=False, delete_wsg=True, workers=1):
    """
    处理整个数据集，将每个实验的wsg文件合并成wsv
    
    Args:
        dataset_path: MMFi_Dataset_results的路径
        dry_run: 是否只是模拟运行
        delete_wsg: 是否删除原始wsg文件
        workers: 并行worker数量。默认保守使用低并发，避免打爆存储读取上限
    """
    print(f"\n{'='*60}")
    print(f"WSG到WSV合并工具")
    print(f"数据集路径: {dataset_path}")
    print(f"模式: {'模拟运行' if dry_run else '实际执行'}")
    print(f"删除原始文件: {'是' if delete_wsg else '否'}")
    print(f"Worker数量: {workers}")
    print(f"{'='*60}\n")
    
    # 获取所有实验/视频文件夹
    experiment_folders = discover_experiment_folders(dataset_path)
    
    if not experiment_folders:
        print("未找到任何包含wsg文件夹的实验")
        return
    
    print(f"找到 {len(experiment_folders)} 个实验需要处理\n")
    
    # 统计信息
    success_count = 0
    skip_count = 0
    fail_count = 0
    
    tasks = [(exp_name, wsg_folder, dry_run, delete_wsg) for exp_name, wsg_folder in experiment_folders]

    if workers <= 1:
        for task in tqdm(tasks, desc="处理实验"):
            status, exp_name, message = process_single_experiment(task)
            print(f"\n处理 {exp_name}:")
            print(f"  {message}")
            if status == "success":
                success_count += 1
            elif status == "skip":
                skip_count += 1
            else:
                fail_count += 1
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_exp = {executor.submit(process_single_experiment, task): task[0] for task in tasks}
            for future in tqdm(as_completed(future_to_exp), total=len(future_to_exp), desc="处理实验"):
                exp_name = future_to_exp[future]
                try:
                    status, _, message = future.result()
                except Exception as e:
                    status, message = "fail", f"并行处理异常: {e}"
                print(f"\n处理 {exp_name}:")
                print(f"  {message}")
                if status == "success":
                    success_count += 1
                elif status == "skip":
                    skip_count += 1
                else:
                    fail_count += 1
    
    # 打印总结
    print(f"\n{'='*60}")
    print(f"处理完成!")
    print(f"成功: {success_count}")
    print(f"跳过: {skip_count}")
    print(f"失败: {fail_count}")
    print(f"总计: {len(experiment_folders)}")
    print(f"{'='*60}\n")

def main():
    # ==================== 配置区域 ====================
    # 选择要处理的数据集
    # dataset_names = ["AVIS_val", "AVIS_test", "AVIS_train"]  # 处理所有AVIS数据集
    # dataset_names = ["AVIS_val"]  # 只处理val
    # dataset_names = ["AVIS_test"]  # 只处理test
    # dataset_names = ["AVIS_train"]  # 只处理train
    dataset_names = ["MMFi_Dataset"]  # 只处理MMFi数据集
    
    # 运行配置
    dry_run = False  # True: 模拟运行，False: 实际执行
    delete_wsg = True  # True: 删除原始WSG文件，False: 保留
    workers = 4  # 保守并行度：实验级低并发，避免打爆存储读取上限
    
    # 基础路径
    base_path = "../output"
    
    # ==================== 执行处理 ====================
    print(f"\n{'='*60}")
    print("Pipeline 32: WSG到WSV合并工具")
    print(f"数据集: {', '.join(dataset_names)}")
    print(f"模式: {'模拟运行' if dry_run else '实际执行'}")
    print(f"删除原始文件: {'是' if delete_wsg else '否'}")
    print(f"Worker数量: {workers}")
    print(f"{'='*60}\n")
    
    # 处理每个数据集
    for dataset_name in dataset_names:
        # 构建数据集路径
        if dataset_name == "MMFi_Dataset":
            dataset_path = os.path.join(base_path, "MMFi_Dataset_results")
        else:
            dataset_path = os.path.join(base_path, f"{dataset_name}_results")
        
        # 检查路径是否存在
        if not os.path.exists(dataset_path):
            print(f"警告: 数据集路径不存在: {dataset_path}")
            continue
        
        print(f"\n处理数据集: {dataset_name}")
        print(f"路径: {dataset_path}")
        
        # 执行处理
        process_dataset(
            dataset_path=dataset_path,
            dry_run=dry_run,
            delete_wsg=delete_wsg,
            workers=workers,
        )
    
    print("\n完成! 可以使用以下命令验证WSV文件:")
    print("python -c \"import numpy as np; data = np.load('path/to/merge.wsv'); print(list(data.keys())[:5])\"")

if __name__ == '__main__':
    main()
