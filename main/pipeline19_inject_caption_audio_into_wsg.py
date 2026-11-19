#!/usr/bin/env python3
"""
Pipeline 19: 字幕音频注入工具
将视频字幕转换为音频并注入到WSG文件的全帧区域
使用通道185-194存储字幕音频（10个通道）
"""

import os
import numpy as np
import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import threading
import queue
import librosa
from scipy.io import wavfile
import csv
import tempfile
import subprocess
import warnings
warnings.filterwarnings('ignore')

# 全局异步保存队列和线程
save_queue = None
save_thread = None

def init_async_save():
    """初始化异步保存系统"""
    global save_queue, save_thread
    save_queue = queue.Queue(maxsize=100)  # 限制队列大小防止内存溢出
    save_thread = threading.Thread(target=async_save_worker, daemon=True)
    save_thread.start()

def async_save_worker():
    """异步保存工作线程"""
    while True:
        try:
            # 获取保存任务
            task = save_queue.get(timeout=1)
            if task is None:  # 结束信号
                break
                
            wsg_array, wsg_path = task
            
            # 执行保存
            try:
                save_wsg_data_sync(wsg_array, wsg_path)
            except Exception as e:
                print(f"异步保存失败 {wsg_path}: {str(e)}")
            finally:
                save_queue.task_done()
                
        except queue.Empty:
            continue
        except Exception as e:
            print(f"异步保存线程错误: {str(e)}")

def shutdown_async_save():
    """关闭异步保存系统"""
    global save_queue, save_thread
    if save_queue:
        save_queue.put(None)  # 发送结束信号
    if save_thread and save_thread.is_alive():
        save_thread.join(timeout=5)

def load_wsg_data(wsg_path):
    """加载WSG数据（优化内存使用）"""
    try:
        # WSG文件实际上是压缩的NPZ格式
        # 使用mmap_mode减少内存使用
        data = np.load(wsg_path, allow_pickle=True, mmap_mode='r')
        # 复制数据以确保可以修改
        wsg_array = np.array(data['wsg_data'])
        data.close()  # 显式关闭文件
        return wsg_array
    except Exception as e:
        print(f"加载WSG数据失败 {wsg_path}: {str(e)}")
        return None

def save_wsg_data_sync(wsg_array, wsg_path):
    """同步保存WSG数据（被异步工作线程调用）"""
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(wsg_path), exist_ok=True)
        
        # 保存为压缩NPZ格式，但文件扩展名为.wsg
        base_path = wsg_path[:-4] if wsg_path.endswith('.wsg') else wsg_path
        np.savez_compressed(base_path, wsg_data=wsg_array)
        
        # 重命名为.wsg
        npz_path = base_path + '.npz'
        final_wsg_path = base_path + '.wsg'
        
        if os.path.exists(npz_path):
            if os.path.exists(final_wsg_path):
                os.remove(final_wsg_path)
            os.rename(npz_path, final_wsg_path)
        
        return True
    except Exception as e:
        print(f"保存WSG数据失败 {wsg_path}: {str(e)}")
        return False

def save_wsg_data(wsg_array, wsg_path, async_save=True):
    """保存WSG数据（支持异步模式）"""
    global save_queue
    
    if async_save and save_queue is not None:
        try:
            # 复制数组以避免在异步保存时被修改
            wsg_array_copy = wsg_array.copy()
            # 非阻塞放入队列
            save_queue.put_nowait((wsg_array_copy, wsg_path))
            return True
        except queue.Full:
            print(f"警告: 异步保存队列已满，回退到同步保存")
            return save_wsg_data_sync(wsg_array, wsg_path)
    else:
        # 同步保存
        return save_wsg_data_sync(wsg_array, wsg_path)

def process_audio_segment(audio_segment, sr=16000, target_sr=8000):
    """处理1秒音频段，压缩为8kHz 8bit格式"""
    # 重采样到8kHz
    if sr != target_sr:
        num_samples = int(len(audio_segment) * target_sr / sr)
        audio_8k = np.interp(
            np.linspace(0, len(audio_segment) - 1, num_samples),
            np.arange(len(audio_segment)),
            audio_segment
        )
    else:
        audio_8k = audio_segment
    
    # 确保长度为8000样本（1秒）
    if len(audio_8k) > target_sr:
        audio_8k = audio_8k[:target_sr]
    elif len(audio_8k) < target_sr:
        audio_8k = np.pad(audio_8k, (0, target_sr - len(audio_8k)), 'constant')
    
    # 归一化
    max_val = np.max(np.abs(audio_8k))
    if max_val > 0:
        audio_8k = audio_8k / max_val
    
    # 转换为8bit
    audio_8bit = (audio_8k * 127).astype(np.int8)
    
    return audio_8bit

def create_caption_audio_pattern(audio_8bit, video_id):
    """创建结构化字幕音频数据"""
    # Header: 魔数(4) + 视频ID(2) + 长度(4) + 校验和(4) = 14字节
    magic = np.array([0xF0, 0x0F, 0xF0, 0x0F], dtype=np.uint8)
    video_bytes = np.array([video_id & 0xFF, (video_id >> 8) & 0xFF], dtype=np.uint8)
    
    # 使用4字节长度字段
    audio_length = len(audio_8bit)
    length_bytes = np.array([
        audio_length & 0xFF,
        (audio_length >> 8) & 0xFF,
        (audio_length >> 16) & 0xFF,
        (audio_length >> 24) & 0xFF
    ], dtype=np.uint8)
    
    # 校验和
    sum_val = np.sum(audio_8bit.astype(np.int32))
    checksum = int(sum_val) & 0xFFFFFFFF
    checksum_bytes = np.array([
        checksum & 0xFF,
        (checksum >> 8) & 0xFF,
        (checksum >> 16) & 0xFF,
        (checksum >> 24) & 0xFF
    ], dtype=np.uint8)
    
    header = np.concatenate([magic, video_bytes, length_bytes, checksum_bytes])
    
    # 转换音频数据
    audio_uint8 = (audio_8bit.astype(np.int16) + 128).astype(np.uint8)
    
    return np.concatenate([header, audio_uint8])

def text_to_speech(text, output_path, use_gtts=True):
    """将文本转换为语音"""
    try:
        if use_gtts:
            # 使用Google Text-to-Speech
            from gtts import gTTS
            tts = gTTS(text=text, lang='en', slow=False)
            mp3_path = output_path.replace('.wav', '.mp3')
            tts.save(mp3_path)
            
            # 转换mp3到wav (16kHz)
            subprocess.run([
                'ffmpeg', '-i', mp3_path,
                '-acodec', 'pcm_s16le', '-ac', '1', '-ar', '16000',
                output_path, '-y'
            ], capture_output=True, check=True)
            
            # 删除临时mp3文件
            if os.path.exists(mp3_path):
                os.remove(mp3_path)
            
            return os.path.exists(output_path)
        else:
            # 使用系统TTS (macOS示例)
            subprocess.run([
                'say', '-o', output_path.replace('.wav', '.aiff'), text
            ], check=True)
            
            # 转换到wav
            subprocess.run([
                'ffmpeg', '-i', output_path.replace('.wav', '.aiff'),
                '-acodec', 'pcm_s16le', '-ac', '1', '-ar', '16000',
                output_path, '-y'
            ], capture_output=True, check=True)
            
            if os.path.exists(output_path.replace('.wav', '.aiff')):
                os.remove(output_path.replace('.wav', '.aiff'))
            
            return os.path.exists(output_path)
    except Exception as e:
        print(f"TTS转换失败: {str(e)}")
        return False

def inject_caption_to_wsg(caption_text, wsg_array, video_id, check_existing=True, audio_cache=None):
    """
    将字幕音频注入到WSG数组的全帧区域（优化版）
    
    WSG通道分配：
    - channels 185-194: 存储压缩后的字幕音频数据（10个通道）
    """
    try:
        height, width = wsg_array.shape[:2]
        total_pixels = height * width
        
        # 检查是否已存在数据
        if check_existing:
            # 快速检查：只检查第一个像素
            if wsg_array[0, 0, 185] != -1.0:
                # print(f"  跳过：已包含字幕音频数据")  # 减少输出提高速度
                return True  # 返回True表示"已完成"
        
        # 使用向量化操作清空整个图像的字幕音频通道
        wsg_array[:, :, 185:195] = np.float16(-1.0)
        
        # 不使用缓存的structured_data，因为它包含特定video_id的header
        # 生成新的音频
        if audio_cache and 'audio_8bit' in audio_cache:
            # 使用缓存的8bit音频，只需要创建新的header
            audio_8bit = audio_cache['audio_8bit']
            print(f"  使用缓存的音频")
        else:
            # 完全重新生成
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            print(f"  生成TTS音频...")
            if not text_to_speech(caption_text, tmp_path):
                print(f"  TTS生成失败")
                return False
            
            # 加载生成的音频
            try:
                audio, sr = librosa.load(tmp_path, sr=16000, mono=True)
            finally:
                # 清理临时文件
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            
            # 限制音频长度（最多处理40秒）
            max_duration = 40
            max_samples = max_duration * sr
            if len(audio) > max_samples:
                audio = audio[:max_samples]
                print(f"  音频截断到{max_duration}秒")
            
            # 处理音频为8kHz 8bit格式
            audio_duration = len(audio) / sr
            print(f"  音频时长: {audio_duration:.2f}秒")
            
            # 重采样到8kHz（使用librosa的更快方法）
            target_sr = 8000
            if sr != target_sr:
                audio_8k = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            else:
                audio_8k = audio
            
            # 归一化和量化为8bit
            max_val = np.max(np.abs(audio_8k))
            if max_val > 0:
                audio_8k = audio_8k / max_val
            audio_8bit = (audio_8k * 127).astype(np.int8)
            
            print(f"  处理后音频: {len(audio_8bit)} 字节")
            
            # 缓存音频
            if audio_cache is not None:
                audio_cache['audio_8bit'] = audio_8bit
        
        # 创建结构化数据
        structured_data = create_caption_audio_pattern(audio_8bit, video_id)
        total_bytes = len(structured_data)
        
        # 不缓存structured_data，避免header长度问题
        # 每个视频需要用自己的video_id创建header
        
        # 计算存储参数
        channels_per_pixel = 10
        bytes_per_pixel = channels_per_pixel
        
        # 检查是否有足够的空间
        max_storage = total_pixels * bytes_per_pixel
        if total_bytes > max_storage:
            print(f"  警告: 音频数据({total_bytes}字节)超过存储容量({max_storage}字节)")
            structured_data = structured_data[:max_storage]
            total_bytes = max_storage
        
        # 优化版：使用向量化操作但避免临时大数组
        # 将实际音频数据转换并填充
        if total_bytes > 0:
            # 批量转换为float16值
            # 映射到float16友好的范围: byte(0-255) -> float16(-50, 50)
            # 优化：一次性转换所有数据
            normalized_audio = (structured_data.astype(np.float32) * (100.0 / 255.0)) - 50.0
            
            # 创建填充数据，只创建需要的大小
            # 如果数据不够填满所有空间，用-1.0填充
            if total_bytes < max_storage:
                # 创建完整数组并填充
                full_data = np.full(max_storage, -1.0, dtype=np.float16)
                full_data[:total_bytes] = normalized_audio.astype(np.float16)
            else:
                # 数据刚好够或超过，直接转换
                full_data = normalized_audio[:max_storage].astype(np.float16)
            
            # 使用reshape直接写入（这是最快的方式）
            wsg_array[:, :, 185:195] = full_data.reshape(height, width, channels_per_pixel)
        else:
            # 没有数据时，清空通道
            wsg_array[:, :, 185:195] = np.float16(-1.0)
        
        print(f"  注入{total_bytes}字节字幕音频到全帧")
        print(f"  使用{min(total_bytes // bytes_per_pixel + 1, total_pixels)}个像素")
        
        return True
        
    except Exception as e:
        print(f"  字幕音频处理失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def load_video_captions(csv_path):
    """加载视频字幕"""
    captions = {}
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_name = row['video_name']
                caption = row['caption']
                captions[video_name] = caption
    except Exception as e:
        print(f"加载字幕文件失败: {str(e)}")
    return captions

def process_single_file(args):
    """处理单个WSG文件（优化版）"""
    wsg_path, caption_text, video_id, check_existing, audio_cache = args
    
    try:
        # 加载WSG数据
        wsg_array = load_wsg_data(wsg_path)
        if wsg_array is None:
            return False, f"WSG数据加载失败: {wsg_path}"
        
        # 注入字幕音频（传递音频缓存）
        if inject_caption_to_wsg(caption_text, wsg_array, video_id, check_existing, audio_cache):
            # 保存更新后的WSG文件
            if save_wsg_data(wsg_array, wsg_path):
                return True, None
        
        return False, "注入或保存失败"
        
    except Exception as e:
        return False, f"处理文件失败: {str(e)}"

def process_batch_files(batch_args):
    """批量处理多个WSG文件（优化版：批量读取和保存）"""
    batch_wsg_paths, caption_text, video_id, check_existing, audio_cache = batch_args
    
    success_count = 0
    errors = []
    batch_size = 5  # 每批处理5个文件
    
    # 首次生成音频并缓存
    if not audio_cache or 'audio_8bit' not in audio_cache:
        # 使用第一个文件测试注入以生成缓存
        test_array = np.zeros((100, 100, 256), dtype=np.float16)
        audio_cache_temp = {} if not audio_cache else audio_cache
        inject_caption_to_wsg(caption_text, test_array, video_id, False, audio_cache_temp)
        if audio_cache is not None:
            audio_cache.update(audio_cache_temp)
    
    # 分批处理文件
    for i in range(0, len(batch_wsg_paths), batch_size):
        batch_paths = batch_wsg_paths[i:i+batch_size]
        batch_arrays = []
        valid_paths = []
        
        # 批量读取
        for wsg_path in batch_paths:
            try:
                wsg_array = load_wsg_data(wsg_path)
                if wsg_array is not None:
                    # 快速检查是否需要处理
                    if check_existing and wsg_array[0, 0, 185] != -1.0:
                        success_count += 1
                        continue
                    batch_arrays.append(wsg_array)
                    valid_paths.append(wsg_path)
                else:
                    errors.append(f"WSG数据加载失败: {wsg_path}")
            except Exception as e:
                errors.append(f"加载失败 {wsg_path}: {str(e)}")
        
        # 批量处理
        processed_arrays = []
        for idx, wsg_array in enumerate(batch_arrays):
            try:
                if inject_caption_to_wsg(caption_text, wsg_array, video_id, False, audio_cache):
                    processed_arrays.append((valid_paths[idx], wsg_array))
                else:
                    errors.append(f"注入失败: {valid_paths[idx]}")
            except Exception as e:
                errors.append(f"处理失败 {valid_paths[idx]}: {str(e)}")
        
        # 批量保存
        for wsg_path, wsg_array in processed_arrays:
            try:
                if save_wsg_data(wsg_array, wsg_path):
                    success_count += 1
                else:
                    errors.append(f"保存失败: {wsg_path}")
            except Exception as e:
                errors.append(f"保存失败 {wsg_path}: {str(e)}")
    
    return success_count, errors

def inject_caption_to_video(video_dir, wsg_subdir, video_name, caption_text, video_id, max_workers=4, force_overwrite=False, use_async_save=True):
    """为单个视频的所有帧注入相同的字幕音频（优化版）"""
    wsg_dir = os.path.join(video_dir, wsg_subdir)
    
    # 初始化异步保存（如果需要）
    if use_async_save and save_queue is None:
        init_async_save()
    
    # 检查目录是否存在
    if not os.path.exists(wsg_dir):
        print(f"  WSG目录不存在: {wsg_dir}")
        return False
    
    # 获取所有WSG文件
    wsg_files = glob.glob(os.path.join(wsg_dir, "*.wsg"))
    wsg_files.sort()
    
    if not wsg_files:
        print(f"  没有找到WSG文件")
        return False
    
    print(f"  找到 {len(wsg_files)} 个WSG文件需要处理")
    
    # 在非强制覆盖模式下，先检查第一帧是否已有字幕音频数据
    if not force_overwrite and len(wsg_files) > 0:
        first_wsg = wsg_files[0]
        try:
            wsg_array = load_wsg_data(first_wsg)
            if wsg_array is not None:
                # 检查字幕音频通道(185-194)是否有非-1的值
                has_caption_data = np.any(wsg_array[:, :, 185:195] != -1.0)
                
                if has_caption_data:
                    print(f"  跳过: 检测到第一帧已有字幕音频数据（使用 force_overwrite=True 强制覆盖）")
                    return True  # 返回True表示"已处理"，避免被计为失败
        except Exception as e:
            print(f"  警告: 无法检查第一帧数据: {e}")
    
    # 创建音频缓存（所有帧共享同一音频）
    audio_cache = {}
    
    # 使用批量处理优化I/O
    batch_size = 10  # 每个工作线程处理10个文件
    use_batch_mode = len(wsg_files) > 20  # 文件多于20个时使用批量模式
    
    success_count = 0
    failed_count = 0
    
    if use_batch_mode:
        # 批量模式：将文件分组
        batches = []
        for i in range(0, len(wsg_files), batch_size):
            batch = wsg_files[i:i+batch_size]
            batches.append((batch, caption_text, video_id, not force_overwrite, audio_cache.copy()))
        
        print(f"  使用批量模式: {len(batches)} 批次，每批 {batch_size} 文件")
        
        if max_workers > 1 and len(batches) > 1:
            with ThreadPoolExecutor(max_workers=min(max_workers, len(batches))) as executor:
                futures = [executor.submit(process_batch_files, batch) for batch in batches]
                
                with tqdm(total=len(wsg_files), desc="  注入字幕音频") as pbar:
                    for future in futures:
                        try:
                            batch_success, batch_errors = future.result()
                            success_count += batch_success
                            failed_count += len(batch_errors)
                            pbar.update(batch_success + len(batch_errors))
                            
                            # 显示前3个错误
                            for i, error in enumerate(batch_errors[:3]):
                                print(f"    错误: {error}")
                        except Exception as e:
                            print(f"    批次处理异常: {str(e)}")
                            pbar.update(batch_size)
        else:
            # 单线程批量处理
            for batch in tqdm(batches, desc="  处理批次"):
                batch_success, batch_errors = process_batch_files(batch)
                success_count += batch_success
                failed_count += len(batch_errors)
    else:
        # 常规模式：逐个文件处理
        tasks = []
        for wsg_path in wsg_files:
            tasks.append((wsg_path, caption_text, video_id, not force_overwrite, audio_cache))
        
        if max_workers > 1 and len(tasks) > 10:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(process_single_file, task) for task in tasks]
                
                with tqdm(total=len(futures), desc="  注入字幕音频") as pbar:
                    for future in futures:
                        try:
                            success, error = future.result()
                            if success:
                                success_count += 1
                            else:
                                failed_count += 1
                                if error and failed_count <= 3:
                                    print(f"    错误: {error}")
                        except Exception as e:
                            failed_count += 1
                            if failed_count <= 3:
                                print(f"    任务异常: {str(e)}")
                        pbar.update(1)
        else:
            # 单线程处理
            for task in tqdm(tasks, desc="  注入字幕音频"):
                success, error = process_single_file(task)
                if success:
                    success_count += 1
                elif error and failed_count < 3:
                    print(f"    错误: {error}")
                    failed_count += 1
    
    print(f"  完成! 成功: {success_count}/{len(wsg_files)}")
    return success_count > 0

def process_dataset(dataset_results_dir, captions_csv_path, wsg_subdir, max_workers=4, force_overwrite=False):
    """处理整个数据集（优化版）"""
    if not os.path.exists(dataset_results_dir):
        print(f"数据集目录不存在: {dataset_results_dir}")
        return
    
    # 加载字幕
    print(f"加载字幕文件: {captions_csv_path}")
    captions = load_video_captions(captions_csv_path)
    print(f"  找到 {len(captions)} 个视频字幕")
    
    # 获取所有视频文件夹
    video_folders = [f for f in os.listdir(dataset_results_dir) 
                    if os.path.isdir(os.path.join(dataset_results_dir, f))]
    
    # 过滤出有字幕的视频
    videos_with_captions = [v for v in video_folders if v in captions]
    
    if not videos_with_captions:
        print("没有找到匹配的视频文件夹")
        return
    
    print(f"找到 {len(videos_with_captions)} 个有字幕的视频文件夹")
    
    # 创建全局音频缓存字典（为相同字幕复用音频）
    global_audio_cache = {}
    
    success_count = 0
    for idx, video_folder in enumerate(tqdm(videos_with_captions, desc="处理视频")):
        video_dir = os.path.join(dataset_results_dir, video_folder)
        caption = captions[video_folder]
        video_id = idx  # 简单使用索引作为ID
        
        print(f"\n处理视频: {video_folder}")
        print(f"  字幕: {caption[:100]}...")  # 只显示前100个字符
        
        # 检查全局缓存中是否有相同字幕的音频
        cache_key = hash(caption) % (2**32)  # 使用字幕的hash作为缓存键
        if cache_key in global_audio_cache:
            print(f"  使用全局缓存的音频 (cache_key: {cache_key})")
            audio_cache = global_audio_cache[cache_key].copy()  # 复制缓存避免并发问题
        else:
            audio_cache = {}
            global_audio_cache[cache_key] = audio_cache
        
        if inject_caption_to_video(video_dir, wsg_subdir, video_folder, caption, video_id, max_workers, force_overwrite):
            success_count += 1
            
        # 清理过大的缓存（保持最多10个条目）
        if len(global_audio_cache) > 10:
            # 删除最旧的条目
            oldest_key = list(global_audio_cache.keys())[0]
            del global_audio_cache[oldest_key]
    
    print(f"\n数据集处理完成! 成功: {success_count}/{len(videos_with_captions)}")

def main():
    """主函数"""
    # ==================== 配置区域 ====================
    # 数据集配置
    dataset_name = "AVIS_train"
    base_output_dir = "../output"
    
    # 字幕文件路径
    captions_csv_path = "../output/AVIS_train_results/video_captions.csv"
    
    # 路径配置
    wsg_subdir = "wsg"  # WSG文件的相对路径
    
    # 处理配置
    max_workers = min(mp.cpu_count() // 2, 6)  # 限制并发数
    force_overwrite = False  # 是否强制覆盖已有数据（默认False以支持断点续传）
    
    # 构建完整路径
    dataset_results_dir = os.path.join(base_output_dir, f"{dataset_name}_results")
    
    print("=" * 50)
    print("字幕音频到WSG注入器 (Pipeline 19)")
    print("=" * 50)
    print(f"数据集: {dataset_name}")
    print(f"数据集路径: {dataset_results_dir}")
    print(f"字幕文件: {captions_csv_path}")
    print(f"WSG子路径: {wsg_subdir}")
    print(f"工作器数量: {max_workers}")
    print(f"强制覆盖: {'是' if force_overwrite else '否'}")
    print("=" * 50)
    
    try:
        # 初始化异步保存系统
        print("初始化异步保存系统...")
        init_async_save()
        
        process_dataset(dataset_results_dir, captions_csv_path, wsg_subdir, max_workers, force_overwrite)
        
        # 等待所有异步保存完成
        if save_queue:
            print("\n等待异步保存完成...")
            save_queue.join()
        
        print("\n✅ 所有注入任务完成!")
        
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # 关闭异步保存系统
        shutdown_async_save()


if __name__ == '__main__':
    main()