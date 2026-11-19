#!/usr/bin/env python3
import os
import sys
import numpy as np
from tqdm import tqdm
import time
import argparse
import gc
import psutil
import json
import csv
import subprocess
import tempfile
from collections import defaultdict
from PIL import Image
from pathlib import Path
from multiprocessing import Pool, cpu_count, current_process
import logging
from datetime import datetime
import functools


def setup_logging(dataset_name):
    """设置日志记录"""
    log_dir = f"../output/{dataset_name}_results"
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志文件名（包含时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"injection_log_{timestamp}.txt")
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - [%(processName)s] - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"日志文件: {log_file}")
    return logger


class BoneDataPacker:
    @staticmethod
    def pack_weights(weights):
        weights = np.array(weights, dtype=np.float32)
        if len(weights) < 4:
            padded_weights = np.zeros(4, dtype=np.float32)
            padded_weights[:len(weights)] = weights[:4]
            weights = padded_weights
        elif len(weights) > 4:
            weights = weights[:4]

        weight_sum = weights.sum()
        if weight_sum > 0:
            weights = weights / weight_sum

        w = np.clip(weights * 255, 0, 255).astype(np.uint8)
        packed = (int(w[0]) << 24) | (int(w[1]) << 16) | (int(w[2]) << 8) | int(w[3])
        return np.frombuffer(np.array([packed], dtype=np.uint32), dtype=np.float32)[0]

    @staticmethod
    def pack_indices(indices):
        indices = np.array(indices, dtype=np.int32)
        if len(indices) < 4:
            padded_indices = np.zeros(4, dtype=np.int32)
            padded_indices[:len(indices)] = indices[:4]
            indices = padded_indices
        elif len(indices) > 4:
            indices = indices[:4]

        idx = np.clip(indices, 0, 255).astype(np.uint8)
        packed = (int(idx[0]) << 24) | (int(idx[1]) << 16) | (int(idx[2]) << 8) | int(idx[3])
        return np.frombuffer(np.array([packed], dtype=np.uint32), dtype=np.float32)[0]

    @staticmethod
    def pack_skeleton_meta(joint_count, bone_count, root_index):
        joint_count = min(joint_count, 1023)
        bone_count = min(bone_count, 1023)
        root_index = min(root_index, 1023)
        packed = (joint_count << 22) | (bone_count << 12) | (root_index << 2)
        return np.frombuffer(np.array([packed], dtype=np.uint32), dtype=np.float32)[0]


class CompleteSkeletonWSGInjector:
    def __init__(self, enable_color_injection=False, category_mapping_path=None):
        """
        初始化注入器

        Args:
            enable_color_injection (bool): 是否启用RGB颜色注入到WSG通道0、1、2
                                          False - 不注入颜色，通道0、1、2保持-1.0（默认）
                                          True - 注入颜色数据
            category_mapping_path (str): 类别映射JSON文件路径
        """
        self.wsg_channels = 256
        # 宽度和高度将根据具体的输入图片或需求动态设置
        self.default_wsg_width = 720
        self.default_wsg_height = 720
        self.bone_packer = BoneDataPacker()
        self.enable_color_injection = enable_color_injection
        
        # 加载类别映射
        self.category_to_id = {}
        if category_mapping_path and os.path.exists(category_mapping_path):
            with open(category_mapping_path, 'r') as f:
                mapping_data = json.load(f)
                self.category_to_id = mapping_data['category_to_id']
                if current_process().name == 'MainProcess':
                    print(f"🎯 加载了 {len(self.category_to_id)} 个类别映射")
        else:
            if current_process().name == 'MainProcess':
                print(f"⚠️ 未加载类别映射，将使用默认object_id")

        # 🎨 颜色功能状态提示（只在主进程打印）
        if current_process().name == 'MainProcess':
            if self.enable_color_injection:
                print("🎨 颜色注入功能: 启用 - RGB数据将写入WSG通道0、1、2")
            else:
                print("⚫ 颜色注入功能: 禁用 - 通道0、1、2将保持-1.0（默认）")
        
        # 查找Blender渲染组件
        self.blender_path, self.blender_script = self._find_blender_components()
        if current_process().name == 'MainProcess':
            if self.blender_path and self.enable_color_injection:
                print(f"🎨 找到Blender渲染器: {self.blender_path}")
                print(f"📝 渲染脚本: {self.blender_script}")
            elif self.enable_color_injection:
                print("⚠️ 未找到Blender，将使用简单投影作为RGB内容")
    
    def _find_blender_components(self):
        """查找Blender和渲染脚本"""
        # 首先检查是否有bpy模块
        has_bpy = False
        try:
            import bpy
            has_bpy = True
        except ImportError:
            pass
            
        # 查找Blender
        blender_paths = ["blender", "/usr/bin/blender", "/usr/local/bin/blender"]
        blender_path = None
        if has_bpy:
            blender_path = "bpy"  # 使用bpy模块标记
        else:
            for path in blender_paths:
                try:
                    result = subprocess.run(["which", path], capture_output=True, text=True)
                    if result.returncode == 0:
                        blender_path = path
                        break
                except Exception:
                    continue
        
        # 查找Blender脚本
        script_paths = [
            "utils/blender_script.py",
            "main/utils/blender_script.py",
            "../main/utils/blender_script.py",
            "blender_script.py"
        ]
        blender_script = None
        for path in script_paths:
            if os.path.exists(path):
                blender_script = path
                break
                
        return blender_path, blender_script
    
    def render_3d_model_to_rgb(self, model_data, render_size=720):
        """
        使用Blender渲染3D模型生成RGB图像
        Returns: numpy array of shape (H, W, 3) with RGB values in [0, 255]
        """
        if not self.blender_path or not self.blender_script:
            return None
            
        # 创建临时glTF文件
        with tempfile.TemporaryDirectory() as temp_dir:
            # 先将模型数据导出为glTF
            gltf_path = os.path.join(temp_dir, "temp_model.gltf")
            if not self._export_to_gltf(model_data, gltf_path):
                return None
            
            # 设置渲染输出目录
            render_dir = os.path.join(temp_dir, "renders")
            os.makedirs(render_dir, exist_ok=True)
            
            # 构建渲染命令
            if self.blender_path == "bpy":
                cmd = [
                    sys.executable,
                    self.blender_script,
                    "--object_path", gltf_path,
                    "--output_dir", render_dir,
                    "--engine", "CYCLES",
                    "--num_renders", "1",  # 只渲染一个视角
                    "--resolution", str(render_size),
                    "--fixed_camera", "front"  # 使用正向相机，匹配XY投影
                ]
            else:
                cmd = [
                    self.blender_path,
                    "--background",
                    "--python", self.blender_script,
                    "--",
                    "--object_path", gltf_path,
                    "--output_dir", render_dir,
                    "--engine", "CYCLES",
                    "--num_renders", "1",
                    "--resolution", str(render_size),
                    "--fixed_camera", "front"  # 使用正向相机，匹配XY投影
                ]
            
            # 执行渲染
            try:
                result = subprocess.run(cmd, capture_output=True, timeout=120)
                if result.returncode != 0:
                    return None
                    
                # 查找渲染结果
                png_files = list(Path(render_dir).glob("*.png"))
                if png_files:
                    # 读取第一个渲染结果
                    img = Image.open(png_files[0])
                    img = img.convert('RGB')
                    return np.array(img)
                    
            except Exception as e:
                print(f"    ⚠️ Blender渲染失败: {e}")
                
        return None
    
    def _export_to_gltf(self, model_data, gltf_path):
        """将模型数据导出为glTF格式"""
        try:
            vertices = model_data['vertices']
            faces = model_data['faces']
            normals = model_data.get('normals', [])
            
            # 创建简单的glTF结构
            vertices_flat = np.array(vertices, dtype=np.float32).flatten()
            normals_flat = np.array(normals, dtype=np.float32).flatten() if len(normals) > 0 else None
            indices_flat = np.array(faces, dtype=np.uint32).flatten()
            
            # 计算边界
            vertices_array = np.array(vertices)
            min_pos = vertices_array.min(axis=0).tolist()
            max_pos = vertices_array.max(axis=0).tolist()
            
            # 构建glTF JSON
            gltf = {
                "asset": {"version": "2.0"},
                "scenes": [{"nodes": [0]}],
                "scene": 0,
                "nodes": [{"mesh": 0}],
                "meshes": [{
                    "primitives": [{
                        "attributes": {"POSITION": 0},
                        "indices": 1
                    }]
                }],
                "accessors": [
                    {  # 位置
                        "bufferView": 0,
                        "componentType": 5126,  # FLOAT
                        "count": len(vertices),
                        "type": "VEC3",
                        "min": min_pos,
                        "max": max_pos
                    },
                    {  # 索引
                        "bufferView": 1,
                        "componentType": 5125,  # UNSIGNED_INT
                        "count": len(indices_flat),
                        "type": "SCALAR"
                    }
                ],
                "bufferViews": [
                    {
                        "buffer": 0,
                        "byteOffset": 0,
                        "byteLength": len(vertices_flat) * 4
                    },
                    {
                        "buffer": 0,
                        "byteOffset": len(vertices_flat) * 4,
                        "byteLength": len(indices_flat) * 4
                    }
                ],
                "buffers": []
            }
            
            # 添加法线（如果有）
            if normals_flat is not None:
                gltf["meshes"][0]["primitives"][0]["attributes"]["NORMAL"] = 2
                gltf["accessors"].append({
                    "bufferView": 2,
                    "componentType": 5126,
                    "count": len(normals),
                    "type": "VEC3"
                })
                gltf["bufferViews"].append({
                    "buffer": 0,
                    "byteOffset": len(vertices_flat) * 4 + len(indices_flat) * 4,
                    "byteLength": len(normals_flat) * 4
                })
            
            # 创建二进制数据
            binary_data = vertices_flat.tobytes() + indices_flat.tobytes()
            if normals_flat is not None:
                binary_data += normals_flat.tobytes()
            
            # 写入二进制文件
            bin_path = gltf_path.replace('.gltf', '.bin')
            with open(bin_path, 'wb') as f:
                f.write(binary_data)
            
            # 更新buffer信息
            gltf["buffers"] = [{
                "uri": os.path.basename(bin_path),
                "byteLength": len(binary_data)
            }]
            
            # 写入glTF JSON
            with open(gltf_path, 'w') as f:
                json.dump(gltf, f, indent=2)
            
            return True
        except Exception as e:
            print(f"    ⚠️ 导出glTF失败: {e}")
            return False

    def extract_vertex_colors(self, model_dict, vertex_count):
        """🎨 从NPZ数据中提取顶点颜色信息"""
        # 如果颜色注入被禁用，直接返回None
        if not self.enable_color_injection:
            print(f"    ⚫ 颜色注入已禁用，跳过颜色提取")
            return None

        color_keys = [
            'colors', 'vertex_colors', 'color', 'rgb', 'rgba',
            'diffuse_color', 'base_color', 'material_color', 'vertex_color',
            'point_colors', 'mesh_colors'
        ]

        vertex_colors = None

        for key in color_keys:
            if key in model_dict:
                color_data = model_dict[key]
                if isinstance(color_data, np.ndarray) and len(color_data) > 0:
                    print(f"    🎨 找到颜色数据: {key}, 形状: {color_data.shape}")
                    vertex_colors = color_data
                    break

        if vertex_colors is None:
            print(f"    ⚠️ 未找到颜色数据，生成基于几何的颜色")
            return self.generate_default_colors(model_dict, vertex_count)

        vertex_colors = np.array(vertex_colors, dtype=np.float32)

        if len(vertex_colors) != vertex_count:
            if len(vertex_colors) > vertex_count:
                vertex_colors = vertex_colors[:vertex_count]
                print(f"    ✂️ 截断颜色数据到 {vertex_count} 个顶点")
            else:
                print(f"    🔄 扩展颜色数据从 {len(vertex_colors)} 到 {vertex_count}")
                vertex_colors = self.extend_colors(vertex_colors, vertex_count)

        if vertex_colors.ndim == 1:
            vertex_colors = np.tile(vertex_colors.reshape(-1, 1), (1, 3))
        elif vertex_colors.ndim == 2:
            if vertex_colors.shape[1] == 1:
                vertex_colors = np.tile(vertex_colors, (1, 3))
            elif vertex_colors.shape[1] == 4:
                vertex_colors = vertex_colors[:, :3]
            elif vertex_colors.shape[1] == 3:
                pass
            else:
                print(f"    ❌ 不支持的颜色格式: {vertex_colors.shape}，使用默认颜色")
                return self.generate_default_colors(model_dict, vertex_count)

        if vertex_colors.max() > 1.0:
            vertex_colors = vertex_colors / 255.0
            print(f"    🔧 颜色值从[0,255]归一化到[0,1]")

        vertex_colors = np.clip(vertex_colors, 0.0, 1.0)
        print(f"    ✅ 成功提取 {len(vertex_colors)} 个顶点的RGB颜色")
        return vertex_colors

    def generate_default_colors(self, model_dict, vertex_count):
        """生成默认的顶点颜色"""
        # 如果颜色注入被禁用，返回None
        if not self.enable_color_injection:
            return None

        vertices = model_dict.get('vertices', np.array([]))
        normals = model_dict.get('normals', np.array([]))

        if len(normals) == vertex_count:
            normals_array = np.array(normals)
            colors = (normals_array + 1.0) / 2.0
            colors = np.abs(colors)
            print(f"    🌈 基于法向量生成颜色")
        elif len(vertices) == vertex_count:
            vertices_array = np.array(vertices)
            vertices_normalized = vertices_array - vertices_array.min(axis=0)
            vertices_range = vertices_array.max(axis=0) - vertices_array.min(axis=0)
            vertices_range[vertices_range == 0] = 1
            colors = vertices_normalized / vertices_range
            print(f"    🗺️ 基于顶点位置生成颜色")
        else:
            np.random.seed(42)
            colors = np.random.rand(vertex_count, 3)
            print(f"    🎲 生成随机颜色")

        return colors.astype(np.float32)
    
    def save_rendered_image_as_png(self, rendered_rgb, output_dir, filename):
        """保存Blender渲染的RGB图像为PNG文件"""
        try:
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            # rendered_rgb应该是float32格式，范围[0,255]
            # 确保在正确范围内并转换为uint8
            rgb_clipped = np.clip(rendered_rgb, 0, 255)
            img_array = rgb_clipped.astype(np.uint8)
            
            # 保存为PNG
            img = Image.fromarray(img_array)
            png_path = os.path.join(output_dir, filename)
            img.save(png_path)
            
            print(f"    🎨 保存Blender渲染图像: {png_path} ({img_array.shape[0]}x{img_array.shape[1]})")
            return True
            
        except Exception as e:
            print(f"    ❌ 保存Blender渲染图像失败: {e}")
            return False
    
    def save_color_as_png(self, vertices, colors, output_dir, filename, img_size=512):
        """根据顶点位置投影，将颜色保存为PNG图片"""
        if colors is None or len(colors) == 0 or vertices is None:
            return False
            
        try:
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            # 将颜色转换为0-255范围
            colors_uint8 = (colors * 255).astype(np.uint8)
            
            # 创建图像数组（白色背景）
            img_array = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
            
            # 计算顶点的边界框
            vertices_array = np.array(vertices)
            min_coords = vertices_array.min(axis=0)
            max_coords = vertices_array.max(axis=0)
            range_coords = max_coords - min_coords
            
            # 防止除零
            range_coords[range_coords == 0] = 1
            
            # 选择投影平面（使用XY平面，Z朝向观察者）
            # 也可以选择其他投影方式
            margin = 20  # 边距
            scale = (img_size - 2 * margin) / max(range_coords[0], range_coords[1])
            
            # 投影每个顶点并绘制
            for i, vertex in enumerate(vertices_array):
                # 归一化坐标到[0,1]
                norm_x = (vertex[0] - min_coords[0]) / range_coords[0]
                norm_y = (vertex[1] - min_coords[1]) / range_coords[1]
                
                # 转换到图像坐标
                img_x = int(margin + norm_x * (img_size - 2 * margin))
                img_y = int(margin + (1 - norm_y) * (img_size - 2 * margin))  # Y轴翻转
                
                # 确保坐标在图像范围内
                img_x = max(0, min(img_size - 1, img_x))
                img_y = max(0, min(img_size - 1, img_y))
                
                # 绘制一个小区域（3x3）而不是单个像素，让点更明显
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        px = img_x + dx
                        py = img_y + dy
                        if 0 <= px < img_size and 0 <= py < img_size:
                            img_array[py, px] = colors_uint8[i]
            
            # 保存为PNG
            # 不需要指定mode参数，PIL会自动识别
            img = Image.fromarray(img_array)
            png_path = os.path.join(output_dir, filename)
            img.save(png_path)
            
            print(f"    📸 保存颜色投影图: {png_path} ({img_size}x{img_size})")
            return True
            
        except Exception as e:
            print(f"    ❌ 保存颜色投影图失败: {e}")
            return False

    def extend_colors(self, colors, target_count):
        """扩展颜色数组到目标长度"""
        if len(colors) == 0:
            return np.random.rand(target_count, 3).astype(np.float32)

        repeat_count = target_count // len(colors)
        remainder = target_count % len(colors)

        extended_colors = np.tile(colors, (repeat_count, 1))
        if remainder > 0:
            extended_colors = np.vstack([extended_colors, colors[:remainder]])

        return extended_colors

    def generate_safe_object_id(self, uuid, category_label=None):
        """根据类别标签生成object_id"""
        if category_label and self.category_to_id:
            # 处理多个类别的情况（用逗号分隔）
            categories = [c.strip().strip('"').lower() for c in category_label.split(',')]
            
            # 优先选择最具体的类别
            for cat in categories:
                if cat in self.category_to_id:
                    object_id = float(self.category_to_id[cat])
                    if current_process().name == 'MainProcess':
                        print(f"    ✅ 使用类别 '{cat}' 的ID: {object_id}")
                    return object_id
            
            # 如果没有找到匹配，尝试查看是否是类别变体
            # 例如 "robot" vs "robotic", "mythology" vs "mythological"
            for cat in categories:
                # 查找相似的类别
                for known_cat in self.category_to_id:
                    if cat in known_cat or known_cat in cat:
                        object_id = float(self.category_to_id[known_cat])
                        if current_process().name == 'MainProcess':
                            print(f"    ✅ 使用相似类别 '{known_cat}' 的ID: {object_id} (原始: '{cat}')")
                        return object_id
            
            # 如果没有找到匹配，返回默认值
            if current_process().name == 'MainProcess':
                print(f"    ⚠️ 未找到类别 '{category_label}' 的映射，使用UUID哈希值")
        
        # 默认方法：使用UUID的哈希值
        try:
            # 使用float32后可以使用更大的范围
            # 但仍然保持合理的范围，避免过大的数值
            return float((abs(hash(str(uuid))) % 999999) + 1)
        except:
            import random
            return float(random.randint(1, 10000))

    def apply_objaverse_xl_normalization(self, vertices, joints=None):
        """应用与Objaverse-XL相同的标准化"""
        vertices = np.array(vertices)

        if len(vertices) == 0:
            return vertices, joints, None

        all_points = [vertices]
        if joints is not None and len(joints) > 0:
            joints_array = np.array(joints)
            all_points.append(joints_array)

        combined_points = np.vstack(all_points)
        bbox_min = combined_points.min(axis=0)
        bbox_max = combined_points.max(axis=0)

        bbox_size = bbox_max - bbox_min
        scale_factor = 1.0 / np.max(bbox_size)

        bbox_center = (bbox_min + bbox_max) / 2

        vertices_scaled = vertices * scale_factor
        vertices_normalized = vertices_scaled - (bbox_center * scale_factor)

        joints_normalized = None
        if joints is not None and len(joints) > 0:
            joints_scaled = joints_array * scale_factor
            joints_normalized = joints_scaled - (bbox_center * scale_factor)

        normalization_info = {
            'scale_factor': scale_factor,
            'bbox_center': bbox_center,
            'original_bbox_min': bbox_min,
            'original_bbox_max': bbox_max,
            'original_bbox_size': bbox_size,
            'normalized': True
        }

        return vertices_normalized, joints_normalized, normalization_info

    def extract_skeleton_structure(self, model_dict):
        joints = model_dict.get('joints', np.array([]))
        bones = model_dict.get('bones', np.array([]))
        joint_names = model_dict.get('joint_names', [])
        root_index = model_dict.get('root_index', 0)

        if len(joints) == 0 or len(bones) == 0:
            return None

        parent_mapping = {}
        for bone in bones:
            if len(bone) >= 2:
                start_joint = int(bone[0])
                end_joint = int(bone[1])
                parent_mapping[end_joint] = start_joint

        for joint_id in range(len(joints)):
            if joint_id not in parent_mapping:
                parent_mapping[joint_id] = -1

        return {
            'joints': joints,
            'bones': bones,
            'joint_names': joint_names,
            'root_index': root_index,
            'parent_mapping': parent_mapping,
            'joint_count': len(joints),
            'bone_count': len(bones)
        }

    def extract_enhanced_bone_data(self, model_dict, vertices):
        vertex_count = len(vertices)

        if ('skinning_weights_value' in model_dict and
                'skinning_weights_row' in model_dict and
                'skinning_weights_col' in model_dict and
                'skinning_weights_shape' in model_dict):

            try:
                values = np.array(model_dict['skinning_weights_value'])
                rows = np.array(model_dict['skinning_weights_row'])
                cols = np.array(model_dict['skinning_weights_col'])

                vertex_bone_weights = [[1.0, 0.0, 0.0, 0.0] for _ in range(vertex_count)]
                vertex_bone_indices = [[0, 0, 0, 0] for _ in range(vertex_count)]

                vertex_data = defaultdict(list)
                for i in range(len(values)):
                    vertex_data[rows[i]].append((values[i], cols[i]))

                for vertex_id, data in vertex_data.items():
                    if vertex_id >= vertex_count:
                        continue

                    data.sort(reverse=True, key=lambda x: x[0])
                    weights = [0.0, 0.0, 0.0, 0.0]
                    indices = [0, 0, 0, 0]

                    for i, (weight, joint_id) in enumerate(data[:4]):
                        weights[i] = weight
                        indices[i] = int(joint_id)

                    weight_sum = sum(weights)
                    if weight_sum > 0:
                        weights = [w / weight_sum for w in weights]

                    vertex_bone_weights[vertex_id] = weights
                    vertex_bone_indices[vertex_id] = indices

                return vertex_bone_weights, vertex_bone_indices

            except Exception as e:
                pass

        return ([[1.0, 0.0, 0.0, 0.0] for _ in range(vertex_count)],
                [[0, 0, 0, 0] for _ in range(vertex_count)])

    def extract_model_data(self, model_dict):
        vertices = model_dict.get('vertices', np.array([]))
        faces = model_dict.get('faces', np.array([]))
        normals = model_dict.get('normals', np.array([]))
        uuid = model_dict.get('uuid', f'model_{hash(str(model_dict)) % 10000}')
        
        # 获取类别标签
        category_label = model_dict.get('category_label', '')

        vertex_bone_weights, vertex_bone_indices = self.extract_enhanced_bone_data(model_dict, vertices)
        skeleton_info = self.extract_skeleton_structure(model_dict)

        # 🎨 提取顶点颜色（受enable_color_injection控制）
        vertex_colors = self.extract_vertex_colors(model_dict, len(vertices))

        joints = skeleton_info['joints'] if skeleton_info else None
        normalized_vertices, normalized_joints, norm_info = self.apply_objaverse_xl_normalization(
            vertices, joints
        )

        if skeleton_info and normalized_joints is not None:
            skeleton_info['joints'] = normalized_joints

        uvs = np.array([])
        if len(normalized_vertices) > 0:
            x_range = normalized_vertices[:, 0].max() - normalized_vertices[:, 0].min() + 1e-6
            y_range = normalized_vertices[:, 1].max() - normalized_vertices[:, 1].min() + 1e-6
            uvs = np.column_stack([
                (normalized_vertices[:, 0] - normalized_vertices[:, 0].min()) / x_range,
                (normalized_vertices[:, 1] - normalized_vertices[:, 1].min()) / y_range
            ])

        return {
            'uuid': uuid,
            'vertices': normalized_vertices,
            'faces': faces,
            'normals': normals,
            'uvs': uvs,
            'vertex_colors': vertex_colors,  # 🎨 可能为None（如果禁用颜色注入）
            'vertex_bone_weights': vertex_bone_weights,
            'vertex_bone_indices': vertex_bone_indices,
            'skeleton_info': skeleton_info,
            'normalization_info': norm_info,
            'original_model_dict': model_dict,
            'category_label': category_label
        }

    def validate_faces_simple(self, faces, vertex_count):
        valid_faces = []
        for face in faces:
            if (len(face) >= 3 and
                    all(0 <= v < vertex_count for v in face[:3]) and
                    len(set(face[:3])) == 3):
                valid_faces.append(face[:3])
        return valid_faces

    def generate_normals_fast(self, vertices, faces):
        if len(vertices) == 0 or len(faces) == 0:
            return np.zeros_like(vertices)

        normals = np.zeros_like(vertices, dtype=np.float32)
        for face in faces:
            try:
                v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
                edge1 = v1 - v0
                edge2 = v2 - v0
                face_normal = np.cross(edge1, edge2)
                norm = np.linalg.norm(face_normal)
                if norm > 1e-8:
                    face_normal = face_normal / norm
                    normals[face[0]] += face_normal
                    normals[face[1]] += face_normal
                    normals[face[2]] += face_normal
            except:
                continue

        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1
        normals = normals / norms
        return normals

    def create_complete_skeleton_wsg(self, model_data, save_color_path=None, model_index=None):
        try:
            vertices = model_data['vertices']
            faces = model_data['faces']
            normals = model_data['normals']
            uvs = model_data['uvs']
            vertex_colors = model_data.get('vertex_colors', None)  # 🎨 可能为None
            vertex_bone_weights = model_data.get('vertex_bone_weights', [])
            vertex_bone_indices = model_data.get('vertex_bone_indices', [])
            skeleton_info = model_data.get('skeleton_info')
            uuid = model_data['uuid']

            if len(vertices) == 0:
                return None

            # 不再需要检查float16限制，因为现在使用float32
            # 但仍然可以设置一个合理的上限来避免内存问题
            max_vertices = 1000000  # 100万顶点的合理上限
            if len(vertices) > max_vertices:
                print(f"    ⚠️ 跳过模型: 顶点数 {len(vertices)} 超过上限 {max_vertices}")
                return None
            
            valid_faces = self.validate_faces_simple(faces, len(vertices))

            if len(normals) != len(vertices):
                normals = self.generate_normals_fast(vertices, valid_faces)

            has_bone_data = len(vertex_bone_weights) > 0 and len(vertex_bone_indices) > 0
            if not has_bone_data:
                vertex_bone_weights = [[1.0, 0.0, 0.0, 0.0] for _ in range(len(vertices))]
                vertex_bone_indices = [[0, 0, 0, 0] for _ in range(len(vertices))]

            # 🎨 颜色数据处理（受控制参数影响）
            has_color_data = self.enable_color_injection and vertex_colors is not None
            if has_color_data and len(vertex_colors) != len(vertices):
                print(f"    ⚠️ 颜色数据长度 {len(vertex_colors)} 与顶点数 {len(vertices)} 不匹配，禁用颜色注入")
                has_color_data = False
                vertex_colors = None
            
            # 🎨 如果启用颜色注入，尝试使用Blender渲染获取RGB图像
            rendered_rgb = None
            wsg_width = self.default_wsg_width
            wsg_height = self.default_wsg_height
            
            if self.enable_color_injection:
                print("    🎨 尝试使用Blender渲染3D模型...")
                rendered_rgb = self.render_3d_model_to_rgb(model_data, wsg_width)
                if rendered_rgb is not None:
                    print(f"    ✅ Blender渲染成功: {rendered_rgb.shape}")
                    wsg_height, wsg_width = rendered_rgb.shape[:2]
                else:
                    print("    ⚠️ Blender渲染失败，将使用简单投影")
            
            # 如果有颜色数据并且提供了保存路径，保存颜色为PNG
            # 但是如果有Blender渲染的图像，优先保存渲染图像
            if save_color_path is not None and model_index is not None:
                crop_frame_dir = os.path.join(save_color_path, 'crop_frame')
                png_filename = f"{model_index+1:08d}.png"  # 00000001.png 格式
                
                # 先尝试保存Blender渲染的图像（如果有的话）
                if rendered_rgb is not None:
                    self.save_rendered_image_as_png(rendered_rgb, crop_frame_dir, png_filename)
                elif has_color_data:
                    # 如果没有渲染图像但有颜色数据，保存简单投影
                    self.save_color_as_png(vertices, vertex_colors, crop_frame_dir, png_filename)

            # 获取category_label
            category_label = model_data.get('category_label', '')
            object_id = self.generate_safe_object_id(uuid, category_label)

            vertex_count = len(vertices)
            face_count = len(valid_faces)
            joint_count = 0
            bone_count = 0

            if skeleton_info:
                joint_count = len(skeleton_info['joints'])
                bone_count = len(skeleton_info['bones'])

            # 使用float32后，不再需要担心65504的限制
            # 只需要检查合理的上限
            total_records = vertex_count + face_count + joint_count + bone_count
            
            # 创建WSG图像数组
            # 使用float32保存，避免精度损失
            wsg_array = np.full((wsg_height, wsg_width, self.wsg_channels), -1.0, dtype=np.float32)
            
            # 如果有渲染的RGB图像，先填充到通道0-2
            if rendered_rgb is not None:
                wsg_array[:, :, 0:3] = rendered_rgb.astype(np.float32)
            
            # 计算顶点的边界框和归一化参数
            vertices_array = np.array(vertices)
            min_coords = vertices_array.min(axis=0)
            max_coords = vertices_array.max(axis=0)
            range_coords = max_coords - min_coords
            range_coords[range_coords == 0] = 1  # 防止除零
            
            margin = 50  # 边距，给模型留出一些空间

            # 创建占用掩码，跟踪哪些像素已被使用
            occupied_mask = np.zeros((wsg_height, wsg_width), dtype=bool)
            
            # 🎨 存储顶点数据（使用XY投影）
            stored_vertices = 0
            collision_vertices = 0
            
            for vertex_id in range(vertex_count):
                vertex = vertices[vertex_id]
                
                # 计算顶点在图像上的投影位置（使用XY平面）
                norm_x = (vertex[0] - min_coords[0]) / range_coords[0]
                norm_y = (vertex[1] - min_coords[1]) / range_coords[1]
                
                # 转换到图像坐标
                img_x = int(margin + norm_x * (wsg_width - 2 * margin))
                img_y = int(margin + (1 - norm_y) * (wsg_height - 2 * margin))  # Y轴翻转
                
                # 确保坐标在范围内
                img_x = max(0, min(wsg_width - 1, img_x))
                img_y = max(0, min(wsg_height - 1, img_y))
                
                # 尝试存储顶点
                stored = False
                
                # 首先尝试原始位置
                if not occupied_mask[img_y, img_x]:
                    stored = True
                    target_x, target_y = img_x, img_y
                else:
                    # 如果原始位置被占用，在附近寻找空位
                    collision_vertices += 1
                    for radius in [1, 2, 3, 4]:  # 搜索半径
                        if stored:
                            break
                        # 只检查边框像素以提高效率
                        for dy in range(-radius, radius + 1):
                            if stored:
                                break
                            for dx in range(-radius, radius + 1):
                                # 只检查边框上的点
                                if abs(dx) == radius or abs(dy) == radius:
                                    tx, ty = img_x + dx, img_y + dy
                                    if 0 <= tx < wsg_width and 0 <= ty < wsg_height and not occupied_mask[ty, tx]:
                                        stored = True
                                        target_x, target_y = tx, ty
                                        break
                
                # 存储顶点数据
                if stored:
                    img_x, img_y = target_x, target_y
                    occupied_mask[img_y, img_x] = True
                    stored_vertices += 1
                    
                    # 在投影位置存储数据
                    # Object/Animal Information Domain [111-125]
                    wsg_array[img_y, img_x, 111] = object_id
                    wsg_array[img_y, img_x, 112] = 1  # Class Value = 1 (Vertex Data)
                    wsg_array[img_y, img_x, 113] = vertex_id
                    
                    # 顶点坐标
                    wsg_array[img_y, img_x, 114] = vertex[0]
                    wsg_array[img_y, img_x, 115] = vertex[1]
                    wsg_array[img_y, img_x, 116] = vertex[2]
                    
                    # 🎨 RGB颜色存储到通道 0, 1, 2
                    if has_color_data:
                        wsg_array[img_y, img_x, 0] = vertex_colors[vertex_id][0]  # R
                        wsg_array[img_y, img_x, 1] = vertex_colors[vertex_id][1]  # G
                        wsg_array[img_y, img_x, 2] = vertex_colors[vertex_id][2]  # B
                    
                    # 法向量 Basic Depth & Normals [36-38]
                    wsg_array[img_y, img_x, 36] = normals[vertex_id][0]
                    wsg_array[img_y, img_x, 37] = normals[vertex_id][1]
                    wsg_array[img_y, img_x, 38] = normals[vertex_id][2]
                    
                    # 骨骼权重和索引
                    if vertex_id < len(vertex_bone_weights) and vertex_id < len(vertex_bone_indices):
                        weights = vertex_bone_weights[vertex_id]
                        indices = vertex_bone_indices[vertex_id]
                        wsg_array[img_y, img_x, 117] = self.bone_packer.pack_weights(weights)
                        wsg_array[img_y, img_x, 118] = self.bone_packer.pack_indices(indices)
                    else:
                        wsg_array[img_y, img_x, 117] = self.bone_packer.pack_weights([1.0, 0.0, 0.0, 0.0])
                        wsg_array[img_y, img_x, 118] = self.bone_packer.pack_indices([0, 0, 0, 0])
                    
                    # UV坐标
                    if len(uvs) > vertex_id:
                        wsg_array[img_y, img_x, 120] = uvs[vertex_id][0]
                        wsg_array[img_y, img_x, 121] = uvs[vertex_id][1]

            # 存储面数据 - 使用三角形光栅化填充
            for face_id, face in enumerate(valid_faces):
                
                # 获取三个顶点的投影坐标
                v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
                
                # 计算投影坐标
                def project_vertex(v):
                    norm_x = (v[0] - min_coords[0]) / range_coords[0]
                    norm_y = (v[1] - min_coords[1]) / range_coords[1]
                    img_x = int(margin + norm_x * (wsg_width - 2 * margin))
                    img_y = int(margin + (1 - norm_y) * (wsg_height - 2 * margin))
                    return max(0, min(wsg_width - 1, img_x)), max(0, min(wsg_height - 1, img_y))
                
                x0, y0 = project_vertex(v0)
                x1, y1 = project_vertex(v1)
                x2, y2 = project_vertex(v2)
                
                # 计算三角形的边界框
                min_x = max(0, min(x0, x1, x2))
                max_x = min(wsg_width - 1, max(x0, x1, x2))
                min_y = max(0, min(y0, y1, y2))
                max_y = min(wsg_height - 1, max(y0, y1, y2))
                
                # 使用三角形光栅化填充
                for y in range(min_y, max_y + 1):
                    for x in range(min_x, max_x + 1):
                        # 计算重心坐标
                        def sign(p1, p2, p3):
                            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
                        
                        d1 = sign((x, y), (x0, y0), (x1, y1))
                        d2 = sign((x, y), (x1, y1), (x2, y2))
                        d3 = sign((x, y), (x2, y2), (x0, y0))
                        
                        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
                        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
                        
                        if not (has_neg and has_pos):  # 点在三角形内
                            # 只在没有顶点数据的像素上存储面数据
                            if wsg_array[y, x, 112] != 1:  # 不是顶点
                                wsg_array[y, x, 111] = object_id
                                wsg_array[y, x, 112] = 2  # Class Value = 2 (Face Data)
                                wsg_array[y, x, 113] = face_id  # Face ID
                                wsg_array[y, x, 114] = float(face[0])  # Vertex 1 index
                                wsg_array[y, x, 115] = float(face[1])  # Vertex 2 index
                                wsg_array[y, x, 116] = float(face[2])  # Vertex 3 index

            # 存储关节数据 - 投影到2D位置
            if skeleton_info and joint_count > 0:
                joints = skeleton_info['joints']
                parent_mapping = skeleton_info['parent_mapping']
                root_index = skeleton_info['root_index']

                packed_meta = self.bone_packer.pack_skeleton_meta(joint_count, bone_count, root_index)

                for joint_id in range(joint_count):
                    if joint_id >= len(joints):
                        break
                        
                    joint_pos = joints[joint_id]
                    
                    # 投影关节到2D
                    norm_x = (joint_pos[0] - min_coords[0]) / range_coords[0]
                    norm_y = (joint_pos[1] - min_coords[1]) / range_coords[1]
                    
                    img_x = int(margin + norm_x * (wsg_width - 2 * margin))
                    img_y = int(margin + (1 - norm_y) * (wsg_height - 2 * margin))
                    
                    img_x = max(0, min(wsg_width - 1, img_x))
                    img_y = max(0, min(wsg_height - 1, img_y))
                    
                    # 在投影位置存储关节数据
                    wsg_array[img_y, img_x, 111] = object_id
                    wsg_array[img_y, img_x, 112] = 3  # Class Value = 3 (Joint Data)
                    wsg_array[img_y, img_x, 113] = joint_id
                    
                    # 关节3D坐标
                    wsg_array[img_y, img_x, 114] = joint_pos[0]
                    wsg_array[img_y, img_x, 115] = joint_pos[1]
                    wsg_array[img_y, img_x, 116] = joint_pos[2]
                    
                    # 元数据只存储在第一个关节 (移到117)
                    if joint_id == 0:
                        wsg_array[img_y, img_x, 117] = packed_meta
                    
                    # 父关节ID (移到118)
                    parent_id = parent_mapping.get(joint_id, -1)
                    wsg_array[img_y, img_x, 118] = float(parent_id)

            # 存储骨骼数据 - 在骨骼中点存储
            if skeleton_info and bone_count > 0:
                bones = skeleton_info['bones']
                joints = skeleton_info['joints']

                for bone_id in range(bone_count):
                    if bone_id >= len(bones):
                        break
                        
                    bone = bones[bone_id]
                    start_joint = int(bone[0]) if len(bone) > 0 else 0
                    end_joint = int(bone[1]) if len(bone) > 1 else 0

                    if start_joint < len(joints) and end_joint < len(joints):
                        start_pos = np.array(joints[start_joint])
                        end_pos = np.array(joints[end_joint])
                        
                        # 计算骨骼中点
                        bone_center = (start_pos + end_pos) / 2.0
                        bone_length = np.linalg.norm(end_pos - start_pos)
                        
                        # 投影到2D
                        norm_x = (bone_center[0] - min_coords[0]) / range_coords[0]
                        norm_y = (bone_center[1] - min_coords[1]) / range_coords[1]
                        
                        img_x = int(margin + norm_x * (wsg_width - 2 * margin))
                        img_y = int(margin + (1 - norm_y) * (wsg_height - 2 * margin))
                        
                        img_x = max(0, min(wsg_width - 1, img_x))
                        img_y = max(0, min(wsg_height - 1, img_y))
                        
                        # 存储骨骼数据
                        wsg_array[img_y, img_x, 111] = object_id
                        wsg_array[img_y, img_x, 112] = 4  # Class Value = 4 (Bone Data)
                        wsg_array[img_y, img_x, 113] = bone_id
                        
                        wsg_array[img_y, img_x, 114] = float(start_joint)
                        wsg_array[img_y, img_x, 115] = float(end_joint)
                        wsg_array[img_y, img_x, 116] = bone_length

            # 🎨 状态报告
            color_status = f", {vertex_count} 顶点含RGB颜色" if has_color_data else ", 无颜色数据"
            collision_info = f" ({collision_vertices} 碰撞, {stored_vertices} 已存储)" if collision_vertices > 0 else ""
            print(
                f"    ✅ WSG数组创建成功: {vertex_count} 顶点{collision_info}{color_status}, {face_count} 面, {joint_count} 关节, {bone_count} 骨骼")
            return wsg_array

        except Exception as e:
            print(f"    ❌ 创建WSG时出错: {e}")
            return None

    def create_metadata_txt(self, model_dict, uuid, normalization_info, output_dir, index):
        """创建包含NPZ中所有非几何数据的TXT文件"""
        try:
            metadata = {
                'uuid': uuid,
                'model_index': index,
                'timestamp': time.time(),
                'color_injection_enabled': self.enable_color_injection,  # 🎨 记录颜色注入状态
                'joint_names': model_dict.get('joint_names', []),
                'root_index': model_dict.get('root_index', 0),
                'original_data': {
                    'pc_w_norm': self._serialize_array(model_dict.get('pc_w_norm', [])),
                    'skinning_weights_shape': model_dict.get('skinning_weights_shape', None),
                },
                'normalization_info': {
                    'scale_factor': float(normalization_info.get('scale_factor', 1.0)) if normalization_info else 1.0,
                    'bbox_center': self._serialize_array(
                        normalization_info.get('bbox_center', [0, 0, 0])) if normalization_info else [0, 0, 0],
                    'original_bbox_min': self._serialize_array(
                        normalization_info.get('original_bbox_min', [0, 0, 0])) if normalization_info else [0, 0, 0],
                    'original_bbox_max': self._serialize_array(
                        normalization_info.get('original_bbox_max', [0, 0, 0])) if normalization_info else [0, 0, 0],
                    'original_bbox_size': self._serialize_array(
                        normalization_info.get('original_bbox_size', [0, 0, 0])) if normalization_info else [0, 0, 0],
                    'normalized': bool(normalization_info.get('normalized', False)) if normalization_info else False
                },
                'statistics': {
                    'vertex_count': len(model_dict.get('vertices', [])),
                    'face_count': len(model_dict.get('faces', [])),
                    'joint_count': len(model_dict.get('joints', [])),
                    'bone_count': len(model_dict.get('bones', [])),
                    'has_skinning': ('skinning_weights_value' in model_dict),
                    'has_normals': ('normals' in model_dict),
                    'has_colors': self._has_color_data(model_dict)
                },
                'additional_data': {}
            }

            excluded_keys = {'vertices', 'faces', 'normals', 'joints', 'bones',
                             'skinning_weights_value', 'skinning_weights_row',
                             'skinning_weights_col', 'skinning_weights_shape'}

            for key, value in model_dict.items():
                if key not in excluded_keys:
                    try:
                        if isinstance(value, np.ndarray):
                            if value.size < 1000:
                                metadata['additional_data'][key] = self._serialize_array(value)
                            else:
                                metadata['additional_data'][key] = f"<large_array_shape_{value.shape}>"
                        elif isinstance(value, (list, tuple)) and len(value) < 100:
                            metadata['additional_data'][key] = list(value)
                        elif isinstance(value, (str, int, float, bool)):
                            metadata['additional_data'][key] = value
                        else:
                            metadata['additional_data'][key] = str(value)
                    except:
                        metadata['additional_data'][key] = f"<serialization_error_{type(value)}>"

            txt_path = os.path.join(output_dir, f"{uuid}_{index:04d}.txt")
            with open(txt_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            return True

        except Exception as e:
            print(f"创建元数据TXT文件失败: {e}")
            return False

    def _has_color_data(self, model_dict):
        """检查模型是否包含颜色数据"""
        color_keys = [
            'colors', 'vertex_colors', 'color', 'rgb', 'rgba',
            'diffuse_color', 'base_color', 'material_color', 'vertex_color'
        ]
        return any(key in model_dict for key in color_keys)

    def _serialize_array(self, arr):
        """将numpy数组序列化为可JSON化的格式"""
        try:
            if isinstance(arr, np.ndarray):
                if arr.size == 0:
                    return []
                elif arr.size == 1:
                    return float(arr.item())
                else:
                    return arr.tolist()
            elif isinstance(arr, (list, tuple)):
                return list(arr)
            else:
                return arr
        except:
            return str(arr)

    def save_wsg_file(self, wsg_array, output_path):
        """保存WSG文件（720x720x256图像格式）"""
        try:
            if not output_path.endswith('.wsg'):
                output_path = output_path + '.wsg'

            # 检查通道数是否正确
            if wsg_array.shape[2] != self.wsg_channels:
                print(f"    ⚠️ WSG数组通道数不正确: {wsg_array.shape[2]}，期望: {self.wsg_channels}")
                return False

            # 保持float32精度，避免精度损失
            # 注意：虽然文件会更大，但压缩算法会处理重复的-1值
            wsg_array_float32 = wsg_array.astype(np.float32)
            
            # 保存为压缩的NPZ格式
            base_path = output_path[:-4]
            np.savez_compressed(base_path, wsg_data=wsg_array_float32)

            # 重命名为.wsg
            npz_path = base_path + '.npz'
            wsg_path = base_path + '.wsg'

            if os.path.exists(npz_path):
                if os.path.exists(wsg_path):
                    os.remove(wsg_path)
                os.rename(npz_path, wsg_path)

            print(f"    💾 WSG文件已保存: {wsg_path} (形状: {wsg_array.shape})")
            return True
        except Exception as e:
            print(f"保存WSG文件失败: {e}")
            return False


def process_single_model(args):
    """
    处理单个模型的函数（用于多进程）
    """
    (model_idx, model_dict, npz_output_dir, npz_name_without_ext, 
     npz_start_index, enable_color_injection, category_to_id, category_mapping_path) = args
    
    try:
        logger = logging.getLogger(__name__)
        
        # 先获取UUID来构造文件路径
        uuid = model_dict.get('uuid', 'unknown')
        
        # 使用UUID作为文件夹名
        model_folder = os.path.join(npz_output_dir, uuid)
        
        # 生成WSG文件路径（文件名保持原样，包含索引以便排序）
        wsg_output_path = os.path.join(model_folder, f"{uuid}_{npz_name_without_ext}_{npz_start_index:04d}.wsg")
        
        # 检查WSG文件是否已存在
        if os.path.exists(wsg_output_path):
            logger.info(f"跳过模型 {npz_start_index} - WSG文件已存在: {os.path.basename(wsg_output_path)}")
            return 'skipped', npz_start_index  # 返回跳过状态
        
        # 创建注入器实例（每个进程独立）
        injector = CompleteSkeletonWSGInjector(
            enable_color_injection=enable_color_injection,
            category_mapping_path=category_mapping_path
        )
        # 如果已经有category_to_id，直接使用（避免重复加载）
        if category_to_id:
            injector.category_to_id = category_to_id
        
        logger.info(f"处理模型 {model_idx + 1}")
        
        model_data = injector.extract_model_data(model_dict)
        uuid = model_data['uuid']
        
        # 创建模型文件夹
        os.makedirs(model_folder, exist_ok=True)
        
        wsg_array = injector.create_complete_skeleton_wsg(model_data, model_folder, npz_start_index)
        
        wsg_success = False
        txt_success = False
        
        if wsg_array is not None:
            wsg_success = injector.save_wsg_file(wsg_array, wsg_output_path)
            if wsg_success:
                logger.info(f"WSG文件保存成功: {os.path.basename(wsg_output_path)}")
        
        # 生成对应的TXT元数据文件
        if wsg_success:
            txt_success = injector.create_metadata_txt(
                model_dict,
                uuid,
                model_data.get('normalization_info'),
                model_folder,
                npz_start_index
            )
            if txt_success:
                logger.info(f"TXT文件保存成功: 模型 {npz_start_index}")
        
        del wsg_array, model_data
        gc.collect()
        
        return wsg_success and txt_success, npz_start_index
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"处理模型 {npz_start_index} 时出错: {e}")
        return False, npz_start_index


def batch_process_npz_to_complete_skeleton_wsg(dataset_name, max_models=None,
                                               enable_color_injection=False,
                                               specific_files=None,
                                               max_models_per_category=False,
                                               num_workers=None):
    """
    批量处理NPZ文件，生成WSG+TXT组合

    Args:
        dataset_name: 数据集名称（如 "Articulation"）
        max_models: 每个文件的最大模型数
        enable_color_injection: 是否启用RGB颜色注入到WSG通道0、1、2（默认False）
        specific_files: 指定要处理的文件列表及顺序，None表示处理所有文件
        max_models_per_category: 如果为True，max_models变成每个类别的最大模型数
        num_workers: 并行处理的进程数，None表示使用所有CPU核心
    """
    # 使用固定的路径配置
    base_output_dir = "../output"
    result_dir = f"{base_output_dir}/{dataset_name}_results"
    input_path = result_dir
    
    # 加载类别元数据
    uuid_to_category = {}
    meta_csv_path = os.path.join(result_dir, "meta_Articulation_XL_2.0.csv")
    if os.path.exists(meta_csv_path):
        print(f"📊 加载类别元数据: {meta_csv_path}")
        with open(meta_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                uuid = row.get('uuid', '')
                category = row.get('category_label', 'unknown').lower()
                uuid_to_category[uuid] = category
        print(f"   ✅ 加载了 {len(uuid_to_category)} 个模型的类别信息")
    else:
        print(f"⚠️ 未找到类别元数据文件: {meta_csv_path}")
    
    # 类别映射文件路径
    category_mapping_path = os.path.join(result_dir, "category_id_mapping.json")

    # 🎨 使用颜色控制参数和类别映射创建注入器
    injector = CompleteSkeletonWSGInjector(enable_color_injection=enable_color_injection, 
                                          category_mapping_path=category_mapping_path)

    if not os.path.exists(input_path):
        print(f"❌ 输入目录不存在: {input_path}")
        print(f"💡 请确保NPZ文件位于: {input_path}")
        return False

    # 获取NPZ文件列表
    if specific_files:
        # 使用指定的文件列表
        npz_files = specific_files
        print(f"📋 使用指定的文件列表: {npz_files}")
        # 检查文件是否存在
        missing_files = [f for f in npz_files if not os.path.exists(os.path.join(input_path, f))]
        if missing_files:
            print(f"⚠️ 以下文件不存在: {missing_files}")
            npz_files = [f for f in npz_files if f not in missing_files]
    else:
        # 自动查找所有NPZ文件
        npz_files = [f for f in os.listdir(input_path) if f.endswith('.npz')]
        if not npz_files:
            print(f"❌ 在 {input_path} 中没有找到NPZ文件")
            print(f"💡 请先运行 objaverse_downloader.py 生成NPZ文件")
            return False

    print(f"🎯 读取目录: {input_path}")
    print(f"📁 将处理的NPZ文件: {npz_files}")

    total_success = 0
    total_processed = 0
    total_skipped = 0

    # 🎨 根据颜色注入状态显示不同的提示
    color_mode_text = "带RGB颜色的" if enable_color_injection else "无颜色的"
    print(f"🎨 开始处理 {len(npz_files)} 个NPZ文件，生成{color_mode_text}WSG+TXT组合...")

    for npz_file in npz_files:
        npz_path = os.path.join(input_path, npz_file)
        
        # 为每个NPZ文件创建独立的输出目录
        npz_name_without_ext = os.path.splitext(npz_file)[0]
        npz_output_dir = os.path.join(result_dir, npz_name_without_ext)
        os.makedirs(npz_output_dir, exist_ok=True)
        
        # 记录这个NPZ文件处理的起始索引
        npz_start_index = 0
        
        try:
            print(f"\n📂 处理文件: {npz_file}")
            print(f"📁 输出目录: {npz_output_dir}")
            data = np.load(npz_path, allow_pickle=True)

            if 'arr_0' in data.files:
                models = data['arr_0']
            elif len(data.files) == 1:
                models = data[data.files[0]]
            else:
                print(f"⚠️ 跳过 {npz_file}: 未知数据格式")
                continue

            # 处理模型限制
            if max_models_per_category and max_models:
                # 按类别限制模型数量
                print(f"  📊 按类别筛选模型（每类最多{max_models}个）...")
                
                # 创建类别分组
                category_models = {}
                
                # 打印第一个模型的字段，了解数据结构
                if len(models) > 0:
                    print(f"    📋 模型字段示例: {list(models[0].keys())}")
                
                for idx, model in enumerate(models):
                    # 通过UUID获取模型的类别
                    category = 'unknown'
                    if 'uuid' in model:
                        model_uuid = model['uuid']
                        category = uuid_to_category.get(model_uuid, 'unknown')
                    
                    # 如果没有在CSV中找到，尝试其他字段
                    if category == 'unknown':
                        if 'category' in model:
                            category = model['category'].lower()
                        elif 'object_category' in model:
                            category = model['object_category'].lower()
                        elif 'class' in model:
                            category = model['class'].lower()
                    
                    if category not in category_models:
                        category_models[category] = []
                    category_models[category].append((idx, model))
                
                # 从每个类别选择最多max_models个模型
                selected_models = []
                for category, model_list in sorted(category_models.items()):
                    selected_count = min(len(model_list), max_models)
                    selected = model_list[:selected_count]
                    selected_models.extend(selected)
                    print(f"    类别 '{category}': {len(model_list)} 个模型，选择 {selected_count} 个")
                
                # 按原始索引排序，保持顺序
                selected_models.sort(key=lambda x: x[0])
                models = [model for idx, model in selected_models]
                
                print(f"  📊 共选择 {len(models)} 个模型（来自 {len(category_models)} 个类别）")
            elif max_models and len(models) > max_models:
                # 简单限制总数
                models = models[:max_models]
                print(f"  📊 限制为前 {max_models} 个模型")
            else:
                print(f"  📊 找到 {len(models)} 个模型")

            # 准备多进程处理的参数
            process_args = []
            for i, model_dict in enumerate(models):
                # 将类别信息添加到model_dict中
                if 'uuid' in model_dict:
                    model_uuid = model_dict['uuid']
                    category = uuid_to_category.get(model_uuid, 'unknown')
                    model_dict['category_label'] = category
                
                args = (i, model_dict, npz_output_dir, npz_name_without_ext, 
                       npz_start_index + i, enable_color_injection, injector.category_to_id, 
                       category_mapping_path)
                process_args.append(args)
            
            # 决定使用的进程数
            if num_workers is None:
                num_workers = min(cpu_count(), 16)  # 最多使用16个进程
            
            print(f"  🚀 使用 {num_workers} 个进程并行处理...")
            
            # 使用进程池处理
            with Pool(processes=num_workers) as pool:
                # 使用imap_unordered获取更好的进度显示
                results = []
                with tqdm(total=len(models), desc=f"处理 {npz_file}") as pbar:
                    for result in pool.imap_unordered(process_single_model, process_args):
                        status, model_idx = result
                        if status == 'skipped':
                            total_skipped += 1
                            total_success += 1  # 计为成功，因为文件已存在
                        elif status:
                            total_success += 1
                        total_processed += 1
                        pbar.update(1)
                        results.append(result)
            
            # 更新下一个NPZ文件的起始索引
            npz_start_index += len(models)

            del data, models
            gc.collect()

        except Exception as e:
            print(f"❌ 处理文件 {npz_file} 时出错: {e}")
            continue

    logger = logging.getLogger(__name__)
    
    print(f"\n🎉 处理完成！")
    print(f"✅ 成功转换: {total_success}/{total_processed} 个模型")
    if total_skipped > 0:
        print(f"⏩ 跳过已存在: {total_skipped} 个模型")
        print(f"🆕 新建模型: {total_success - total_skipped} 个")

    # 🎨 根据颜色注入状态显示不同的完成信息
    if enable_color_injection:
        print(f"🎨 所有模型都包含RGB颜色数据（通道0、1、2）")
    else:
        print(f"⚫ 所有模型不包含RGB颜色数据（通道0、1、2保持-1.0）")

    print(f"📁 输出根目录: {result_dir}")
    print(f"📂 每个NPZ文件都有独立的目录，其中每个模型有独立的子文件夹")
    
    # 记录到日志
    logger.info("=" * 50)
    logger.info("处理完成统计")
    logger.info(f"成功转换: {total_success}/{total_processed} 个模型")
    logger.info(f"跳过已存在: {total_skipped} 个模型")
    logger.info(f"新建模型: {total_success - total_skipped} 个")
    logger.info(f"成功率: {total_success/max(1, total_processed)*100:.1f}%")
    logger.info("=" * 50)
    
    return total_success > 0


def main():
    # ==================== 配置区域 ====================
    # 数据集配置
    dataset_name = "Articulation"  # 可以改为其他数据集
    # dataset_name = "test_dataset"
    
    # 处理配置
    max_models = 100  # 每个文件的最大模型数，None表示处理所有
    max_models_per_category = True  # 如果为True，max_models变成每个类别的最大模型数
    enable_color_injection = True  # 是否注入RGB颜色
    num_workers = 16  # 使用的CPU核心数，None表示自动选择（你有32核心）
    
    # 📋 指定处理文件顺序（小文件放在前面）
    # None表示自动找到所有NPZ文件
    process_files = [
        'articulation_xlv2_test.npz',              # 测试集，通常较小
        'articulation_xlv2_diverse_pose_test.npz', # 多姿态测试集
        'articulation_xlv2_train.npz',             # 训练集，通常较大
        'articulation_xlv2_diverse_pose_train.npz' # 多姿态训练集
    ]
    # process_files = None  # 注释这一行使用自动查找
    
    # 设置日志
    logger = setup_logging(dataset_name)
    
    # ==================== 开始处理 ====================
    print("=" * 50)
    print("3D对象到WSG注入器")
    print("=" * 50)
    print(f"数据集: {dataset_name}")
    print(f"数据集路径: ../output/{dataset_name}_results")
    print(f"输出路径: ../output/{dataset_name}_results/[npz文件名]/")
    if max_models_per_category and max_models:
        print(f"每个类别最大模型数: {max_models} (预计总共 {max_models * 54} 个模型)")
    else:
        print(f"每个文件最大模型数: {max_models if max_models else '全部'}")
    print(f"颜色注入: {'启用' if enable_color_injection else '禁用'}")
    print(f"并行处理: {num_workers if num_workers else '自动'} 个进程")
    print("=" * 50)
    
    # 记录到日志
    logger.info("=" * 50)
    logger.info("开始3D对象到WSG注入")
    logger.info(f"数据集: {dataset_name}")
    logger.info(f"每个类别最大模型数: {max_models}" if max_models_per_category else f"每个文件最大模型数: {max_models}")
    logger.info(f"并行进程数: {num_workers if num_workers else '自动'}")
    logger.info("=" * 50)
    
    try:
        color_mode = "带RGB颜色控制的" if enable_color_injection else "无颜色数据的"
        print(f"\n🎨 开始批量处理，生成{color_mode}WSG+TXT配对文件...")
        success = batch_process_npz_to_complete_skeleton_wsg(
            dataset_name,
            max_models,
            enable_color_injection=enable_color_injection,
            specific_files=process_files,
            max_models_per_category=max_models_per_category,
            num_workers=num_workers
        )
        return 0 if success else 1
    except Exception as e:
        print(f"❌ 程序错误: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())