#!/usr/bin/env python3

# Force immediate output
import sys
print("🚀 Starting SMPLer-X injection script...", flush=True)
sys.stdout.flush()

print("📦 Loading dependencies...", flush=True)
import argparse
import io
import os
import numpy as np
import glob
import json
import zipfile
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
import threading
import time
import queue
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except:
    CUDA_AVAILABLE = False
print("✅ Dependencies loaded successfully!", flush=True)
if CUDA_AVAILABLE:
    print(f"🚀 GPU ACCELERATION ENABLED: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️  GPU not available, using CPU mode")
    print("   Recommended environment: conda activate smplerx")


# SMPLer-X joint ID mapping
SMPLERX_JOINT_MAPPING = {
    'global_orient': 0,
    'body_pose': list(range(1, 22)),  # 21 body joints
    'left_hand_pose': list(range(22, 37)),  # 15 left hand joints
    'right_hand_pose': list(range(37, 52)),  # 15 right hand joints
    'jaw_pose': 52,
    'leye_pose': 53,
    'reye_pose': 54,
    'transl': 55
}

# WSG channel definitions
PERSON_ID_CHANNEL = 72
JOINT_ID_CHANNEL = 73
JOINT_X_CHANNEL = 74
JOINT_Y_CHANNEL = 75
JOINT_Z_CHANNEL = 76
BETAS_EXPRESSION_TYPE_CHANNEL = 77  # 0=betas, 1=expression
BETAS_EXPRESSION_CHANNELS = list(range(78, 88))  # 10 values
BBOX_CHANNEL = 88
BBOX_MMDET_CHANNEL = 89
SMPLERX_VISUALIZATION_CHANNELS = slice(90, 93)
SMPLERX_RESERVED_CHANNEL = 93
# Legacy aliases only exist for unreachable historical full-mesh code below.
VERTEX_INDEX_CHANNEL = 90
VERTEX_X_CHANNEL = 91
VERTEX_Y_CHANNEL = 92
VERTEX_Z_CHANNEL = 93
SMPLERX_END_CHANNEL = 94
MIN_DETECTOR_SCORE = float(os.environ.get("SMPLERX_MIN_DETECTOR_SCORE", "0.7"))
ALLOW_LEGACY_MISSING_SCORE = os.environ.get("SMPLERX_ALLOW_LEGACY_MISSING_SCORE", "0") == "1"
LIGHTWEIGHT_SMPLERX = True
CLEAR_LEGACY_SMPLERX_VIS = os.environ.get("SMPLERX_CLEAR_LEGACY_VIS", "0") == "1"
SMPLERX_RENDER_LOCK = threading.Lock()


def is_trusted_smplerx_meta(meta):
    score = meta.get("detector_score")
    if score is None:
        if not ALLOW_LEGACY_MISSING_SCORE:
            print("    [SKIP] Missing detector_score in SMPLer-X metadata")
            return False
        return True
    if float(score) < MIN_DETECTOR_SCORE:
        print(f"    [SKIP] detector_score={float(score):.3f} < {MIN_DETECTOR_SCORE:.3f}")
        return False
    return True


def inject_smplerx_visualization(wsg_array):
    """Render the trusted neutral SMPL-X pose and store it in channels 90-92."""
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    from pipeline30_fix_and_inject_three_pose_vis import render_smplerx

    visible_clamp = os.environ.get("SMPLERX_VIS_CLAMP", "bbox")
    visible_margin = float(os.environ.get("SMPLERX_VIS_MARGIN_FRACTION", "0.2"))
    with SMPLERX_RENDER_LOCK:
        visualization, has_smplerx = render_smplerx(
            wsg_array,
            visible_clamp=visible_clamp,
            visible_margin_fraction=visible_margin,
        )
    if has_smplerx:
        wsg_array[:, :, SMPLERX_VISUALIZATION_CHANNELS] = visualization.astype(
            wsg_array.dtype, copy=False
        )
    else:
        wsg_array[:, :, SMPLERX_VISUALIZATION_CHANNELS] = -1
    wsg_array[:, :, SMPLERX_RESERVED_CHANNEL] = -1
    return has_smplerx


def load_obj_vertices(obj_path):
    """Load vertices from OBJ file"""
    try:
        vertices = []
        with open(obj_path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        vertices.append([x, y, z])
        return np.array(vertices) if vertices else None
    except Exception as e:
        print(f"Failed to load OBJ vertices {obj_path}: {str(e)}")
        return None


def load_smplerx_data(npz_path, meta_path, vertices_path=None, mesh_path=None):
    """Load SMPLer-X data including parameters, metadata and vertices"""
    try:
        # Load NPZ parameters
        params = np.load(npz_path, allow_pickle=True)
        
        # Load metadata
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        if LIGHTWEIGHT_SMPLERX:
            return params, meta, None

        # Legacy full-mesh mode only.
        vertices = None
        
        # First try: vertices from NPZ file
        if vertices_path and os.path.exists(vertices_path):
            vertices_data = np.load(vertices_path, allow_pickle=True)
            if 'vertices' in vertices_data:
                vertices = vertices_data['vertices']
        
        # Second try: vertices from mesh OBJ file
        if vertices is None and mesh_path and os.path.exists(mesh_path):
            vertices = load_obj_vertices(mesh_path)
        
        return params, meta, vertices
    except Exception as e:
        print(f"Failed to load SMPLer-X data {npz_path}: {str(e)}")
        return None, None, None


def load_wsg_data(wsg_path):
    """Load WSG data"""
    try:
        data = np.load(wsg_path, allow_pickle=True)
        return data['wsg_data']
    except Exception as e:
        print(f"Failed to load WSG data {wsg_path}: {str(e)}")
        return None


class AsyncFileWriter:
    """Asynchronous file writer to handle I/O bottleneck"""
    def __init__(self, max_workers=4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.futures = []
        self.save_queue = queue.Queue()
        self.writer_thread = threading.Thread(target=self._writer_worker, daemon=True)
        self.writer_thread.start()
        
    def _writer_worker(self):
        """Worker thread that handles file writing"""
        while True:
            try:
                item = self.save_queue.get(timeout=1)
                if item is None:  # Shutdown signal
                    break
                wsg_array, wsg_path = item
                self._save_wsg_data_sync(wsg_array, wsg_path)
                self.save_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in writer thread: {e}")
    
    def _save_wsg_data_sync(self, wsg_array, wsg_path):
        """Synchronous save function"""
        try:
            os.makedirs(os.path.dirname(wsg_path), exist_ok=True)
            base_path = wsg_path[:-4] if wsg_path.endswith('.wsg') else wsg_path
            np.savez_compressed(base_path, wsg_data=wsg_array)
            npz_path = base_path + '.npz'
            final_wsg_path = base_path + '.wsg'
            if os.path.exists(npz_path):
                if os.path.exists(final_wsg_path):
                    os.remove(final_wsg_path)
                os.rename(npz_path, final_wsg_path)
        except Exception as e:
            print(f"Failed to save WSG data {wsg_path}: {str(e)}")
    
    def save_async(self, wsg_array, wsg_path):
        """Add file to save queue"""
        # Make a copy to avoid data corruption
        wsg_copy = wsg_array.copy()
        self.save_queue.put((wsg_copy, wsg_path))
        
    def wait_all(self):
        """Wait for all pending saves to complete"""
        self.save_queue.join()
        
    def shutdown(self):
        """Shutdown the writer thread"""
        self.save_queue.put(None)
        self.writer_thread.join()


def save_wsg_data(wsg_array, wsg_path):
    """Save WSG data"""
    try:
        os.makedirs(os.path.dirname(wsg_path), exist_ok=True)
        base_path = wsg_path[:-4] if wsg_path.endswith('.wsg') else wsg_path
        np.savez_compressed(base_path, wsg_data=wsg_array)
        npz_path = base_path + '.npz'
        final_wsg_path = base_path + '.wsg'
        if os.path.exists(npz_path):
            if os.path.exists(final_wsg_path):
                os.remove(final_wsg_path)
            os.rename(npz_path, final_wsg_path)
        return True
    except Exception as e:
        print(f"Failed to save WSG data {wsg_path}: {str(e)}")
        return False


def inject_smplerx_to_wsg(params, meta, vertices, wsg_array, clear_existing=True, clear_all_smplerx=False):
    """
    Inject SMPLer-X data into WSG array
    """
    if params is None or meta is None or wsg_array is None:
        return False
    if not is_trusted_smplerx_meta(meta):
        return False
    if LIGHTWEIGHT_SMPLERX:
        if clear_existing or clear_all_smplerx:
            wsg_array[:, :, PERSON_ID_CHANNEL:SMPLERX_END_CHANNEL] = -1
        inject_joints_and_metadata(
            wsg_array, params, meta, int(meta.get("bbox_id", 0)), trusted_checked=True
        )
        return True
    
    height, width = wsg_array.shape[:2]
    
    # Get person_id
    person_id = meta.get('bbox_id', 0)
    
    # Only check for first person to avoid redundant checks
    if not clear_existing and not clear_all_smplerx and person_id == 0:
        # Quick check if any SMPLer-X data exists in the frame
        has_smplerx_data = np.any(wsg_array[:, :, PERSON_ID_CHANNEL] != -1)
        if has_smplerx_data:
            print(f"    [SKIP] Frame already contains SMPLer-X data, skipping entire frame")
            return True  # Return True to indicate frame was already processed
    
    # Clear existing SMPLer-X data globally if requested
    if clear_all_smplerx:
        print(f"    [STEP 1/6] Clearing ALL existing SMPLer-X data...")
        import time
        start_time = time.time()
        wsg_array[:, :, PERSON_ID_CHANNEL:VERTEX_Z_CHANNEL+1] = -1
        print(f"    [STEP 1/6] ✓ Cleared in {time.time()-start_time:.2f}s")
    elif clear_existing:
        print(f"    [STEP 1/6] Clearing existing SMPLer-X data...")
        import time
        start_time = time.time()
        wsg_array[:, :, PERSON_ID_CHANNEL:VERTEX_Z_CHANNEL+1] = -1
        print(f"    [STEP 1/6] ✓ Cleared in {time.time()-start_time:.2f}s")
    
    # Get bounding boxes
    bbox = meta.get('bbox', None)
    bbox_mmdet = meta.get('bbox_mmdet', None)
    
    if bbox is None:
        print(f"    [WARN] No bbox information found")
        return False
    
    # Use mmdet bbox for storing information (more precise human region)
    if bbox_mmdet:
        # Use mmdet bbox as primary storage region
        x1 = max(0, int(bbox_mmdet[0]))
        y1 = max(0, int(bbox_mmdet[1]))
        x2 = min(width, int(bbox_mmdet[0] + bbox_mmdet[2]))
        y2 = min(height, int(bbox_mmdet[1] + bbox_mmdet[3]))
        print(f"    [INFO] Using mmdet bbox for information storage: ({x1},{y1})-({x2},{y2})")
    else:
        # Fallback to main bbox if mmdet not available
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(width, int(bbox[0] + bbox[2]))
        y2 = min(height, int(bbox[1] + bbox[3]))
        print(f"    [INFO] Using main bbox for information storage: ({x1},{y1})-({x2},{y2})")
    
    # Calculate bbox corners for marking
    bbox_x1 = max(0, int(bbox[0]))
    bbox_y1 = max(0, int(bbox[1]))
    bbox_x2 = min(width, int(bbox[0] + bbox[2]))
    bbox_y2 = min(height, int(bbox[1] + bbox[3]))
    
    bbox_mmdet_x1, bbox_mmdet_y1, bbox_mmdet_x2, bbox_mmdet_y2 = x1, y1, x2, y2
    if bbox_mmdet:
        bbox_mmdet_x1 = max(0, int(bbox_mmdet[0]))
        bbox_mmdet_y1 = max(0, int(bbox_mmdet[1]))
        bbox_mmdet_x2 = min(width, int(bbox_mmdet[0] + bbox_mmdet[2]))
        bbox_mmdet_y2 = min(height, int(bbox_mmdet[1] + bbox_mmdet[3]))
    
    # Collect all joint parameters
    print(f"    [STEP 2/6] Collecting joint parameters...")
    import time
    start_time = time.time()
    joint_data = []
    
    # global_orient
    if 'global_orient' in params:
        orient = params['global_orient'].reshape(-1, 3)
        for i in range(orient.shape[0]):
            joint_data.append((SMPLERX_JOINT_MAPPING['global_orient'], orient[i]))
    
    # body_pose
    if 'body_pose' in params:
        body = params['body_pose'].reshape(-1, 3)
        for i in range(min(21, body.shape[0])):
            joint_data.append((SMPLERX_JOINT_MAPPING['body_pose'][i], body[i]))
    
    # left_hand_pose
    if 'left_hand_pose' in params:
        lhand = params['left_hand_pose'].reshape(-1, 3)
        for i in range(min(15, lhand.shape[0])):
            joint_data.append((SMPLERX_JOINT_MAPPING['left_hand_pose'][i], lhand[i]))
    
    # right_hand_pose
    if 'right_hand_pose' in params:
        rhand = params['right_hand_pose'].reshape(-1, 3)
        for i in range(min(15, rhand.shape[0])):
            joint_data.append((SMPLERX_JOINT_MAPPING['right_hand_pose'][i], rhand[i]))
    
    # jaw_pose
    if 'jaw_pose' in params:
        jaw = params['jaw_pose'].reshape(-1, 3)
        if jaw.shape[0] > 0:
            joint_data.append((SMPLERX_JOINT_MAPPING['jaw_pose'], jaw[0]))
    
    # leye_pose
    if 'leye_pose' in params:
        leye = params['leye_pose'].reshape(-1, 3)
        if leye.shape[0] > 0:
            joint_data.append((SMPLERX_JOINT_MAPPING['leye_pose'], leye[0]))
    
    # reye_pose
    if 'reye_pose' in params:
        reye = params['reye_pose'].reshape(-1, 3)
        if reye.shape[0] > 0:
            joint_data.append((SMPLERX_JOINT_MAPPING['reye_pose'], reye[0]))
    
    # transl
    if 'transl' in params:
        transl = params['transl'].reshape(-1, 3)
        if transl.shape[0] > 0:
            joint_data.append((SMPLERX_JOINT_MAPPING['transl'], transl[0]))
    
    print(f"    [STEP 2/6] ✓ Collected {len(joint_data)} joint parameters in {time.time()-start_time:.2f}s")
    
    # Distribute joint data evenly within bbox region
    print(f"    [STEP 3/6] Distributing joint data in bbox...")
    start_time = time.time()
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    bbox_area = bbox_width * bbox_height
    
    if bbox_area > 0 and len(joint_data) > 0:
        # OPTIMIZED: Use numpy operations instead of python loops
        # Create coordinate grids
        yy, xx = np.mgrid[y1:y2, x1:x2]
        pixels_y = yy.ravel()
        pixels_x = xx.ravel()
        total_pixels = len(pixels_y)
        
        # Get existing person IDs for all pixels at once
        existing_persons = wsg_array[pixels_y, pixels_x, PERSON_ID_CHANNEL]
        
        # Find available pixels (not occupied by smaller ID person)
        available_mask = (existing_persons == -1) | (existing_persons >= person_id)
        available_indices = np.where(available_mask)[0]
        
        if len(available_indices) > 0:
            # Prepare joint data for batch assignment
            num_joints = len(joint_data)
            num_available = len(available_indices)
            
            # Repeat joint data cyclically to fill all available pixels
            repeat_count = (num_available + num_joints - 1) // num_joints
            
            # Batch prepare all data
            joint_ids = []
            joint_xs = []
            joint_ys = []
            joint_zs = []
            
            for _ in range(repeat_count):
                for joint_id, joint_values in joint_data:
                    joint_ids.append(joint_id)
                    joint_xs.append(joint_values[0])
                    joint_ys.append(joint_values[1])
                    joint_zs.append(joint_values[2])
            
            # Trim to exact number needed
            joint_ids = joint_ids[:num_available]
            joint_xs = joint_xs[:num_available]
            joint_ys = joint_ys[:num_available]
            joint_zs = joint_zs[:num_available]
            
            # Batch assign to available pixels
            selected_y = pixels_y[available_indices]
            selected_x = pixels_x[available_indices]
            
            wsg_array[selected_y, selected_x, PERSON_ID_CHANNEL] = person_id
            wsg_array[selected_y, selected_x, JOINT_ID_CHANNEL] = joint_ids
            wsg_array[selected_y, selected_x, JOINT_X_CHANNEL] = joint_xs
            wsg_array[selected_y, selected_x, JOINT_Y_CHANNEL] = joint_ys
            wsg_array[selected_y, selected_x, JOINT_Z_CHANNEL] = joint_zs
    
    print(f"    [STEP 3/6] ✓ Distributed joint data in {time.time()-start_time:.2f}s")
    
    # Inject betas and expression parameters
    print(f"    [STEP 4/6] Injecting betas and expression parameters...")
    start_time = time.time()
    # Store betas in first row, expression in second row of bbox region
    if 'betas' in params and y1 < y2:
        betas = params['betas'].reshape(-1)[:10]  # Max 10 values
        for i, value in enumerate(betas):
            if x1 + i < x2:
                wsg_array[y1, x1 + i, BETAS_EXPRESSION_TYPE_CHANNEL] = 0  # betas
                wsg_array[y1, x1 + i, BETAS_EXPRESSION_CHANNELS[i]] = value
    
    if 'expression' in params and y1 + 1 < y2:
        expression = params['expression'].reshape(-1)[:10]  # Max 10 values
        for i, value in enumerate(expression):
            if x1 + i < x2:
                wsg_array[y1 + 1, x1 + i, BETAS_EXPRESSION_TYPE_CHANNEL] = 1  # expression
                wsg_array[y1 + 1, x1 + i, BETAS_EXPRESSION_CHANNELS[i]] = value
    
    print(f"    [STEP 4/6] ✓ Injected betas/expression in {time.time()-start_time:.2f}s")
    
    print(f"    [STEP 5/6] Marking bbox corners...")
    start_time = time.time()
    # Mark bbox corners (main bbox)
    bbox_corners = [
        (bbox_x1, bbox_y1), (bbox_x2-1, bbox_y1),  # Top two corners
        (bbox_x1, bbox_y2-1), (bbox_x2-1, bbox_y2-1)  # Bottom two corners
    ]
    for x, y in bbox_corners:
        if 0 <= x < width and 0 <= y < height:
            wsg_array[y, x, BBOX_CHANNEL] = 1
    
    # Mark bbox_mmdet corners
    bbox_mmdet_corners = [
        (bbox_mmdet_x1, bbox_mmdet_y1), (bbox_mmdet_x2-1, bbox_mmdet_y1),
        (bbox_mmdet_x1, bbox_mmdet_y2-1), (bbox_mmdet_x2-1, bbox_mmdet_y2-1)
    ]
    for x, y in bbox_mmdet_corners:
        if 0 <= x < width and 0 <= y < height:
            wsg_array[y, x, BBOX_MMDET_CHANNEL] = 1
    
    print(f"    [STEP 5/6] ✓ Marked bbox corners in {time.time()-start_time:.2f}s")
    
    # Inject vertex data with indices (keep original coordinates)
    if vertices is not None:
        print(f"    [STEP 6/6] 🚀 STARTING BATCH VERTEX INJECTION: {vertices.shape[0]} vertices...")
        vertex_start_time = time.time()
        
        # Get camera parameters
        focal = meta.get('focal', [1000, 1000])
        princpt = meta.get('princpt', [width/2, height/2])
        
        # Clear SMPLer-X vertex channels for this person across the entire image
        print(f"    [STEP 6a/6] Clearing existing vertex data for person {person_id}...")
        clear_start_time = time.time()
        
        # Clear all SMPLer-X vertex data for this specific person across the entire image
        # OPTIMIZED: Single mask operation and batch clearing
        person_mask = (wsg_array[:, :, PERSON_ID_CHANNEL] == person_id)
        
        # Clear all channels at once for better cache efficiency
        if np.any(person_mask):
            # Find the indices where mask is True
            y_indices, x_indices = np.where(person_mask)
            
            # Batch clear all relevant channels
            wsg_array[y_indices, x_indices, VERTEX_INDEX_CHANNEL:VERTEX_Z_CHANNEL+1] = -1
            wsg_array[y_indices, x_indices, PERSON_ID_CHANNEL] = -1
        
        cleared_pixels = np.sum(person_mask)
        print(f"    [STEP 6a/6] ✓ Cleared {cleared_pixels} pixels in {time.time()-clear_start_time:.2f}s")
        
        # Additional safety: Clear any orphaned vertex data
        orphaned_mask = (wsg_array[:, :, VERTEX_INDEX_CHANNEL] != -1) & \
                       (wsg_array[:, :, PERSON_ID_CHANNEL] == -1)
        
        if np.sum(orphaned_mask) > 0:
            print(f"    [INFO] Found {np.sum(orphaned_mask)} orphaned vertex pixels, clearing...")
            wsg_array[:, :, VERTEX_INDEX_CHANNEL][orphaned_mask] = -1
            wsg_array[:, :, VERTEX_X_CHANNEL][orphaned_mask] = -1
            wsg_array[:, :, VERTEX_Y_CHANNEL][orphaned_mask] = -1  
            wsg_array[:, :, VERTEX_Z_CHANNEL][orphaned_mask] = -1
        
        print(f"    [STEP 6b/6] 🔄 BATCH PROJECTING ALL VERTICES...")
        storage_start_time = time.time()
        
        # BATCH PROCESSING: Project all vertices at once using vectorized operations
        # Filter vertices with positive Z values
        valid_mask = vertices[:, 2] > 0
        valid_vertices = vertices[valid_mask]
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_vertices) > 0:
            if CUDA_AVAILABLE:
                # GPU ACCELERATED PROJECTION
                with torch.no_grad():
                    # Move to GPU
                    device = torch.cuda.current_device()
                    valid_vertices_gpu = torch.from_numpy(valid_vertices).float().cuda(device)
                    focal_gpu = torch.tensor(focal, dtype=torch.float32).cuda(device)
                    princpt_gpu = torch.tensor(princpt, dtype=torch.float32).cuda(device)
                    
                    # Batch projection on GPU
                    px_float = focal_gpu[0] * valid_vertices_gpu[:, 0] / valid_vertices_gpu[:, 2] + princpt_gpu[0]
                    py_float = focal_gpu[1] * valid_vertices_gpu[:, 1] / valid_vertices_gpu[:, 2] + princpt_gpu[1]
                    
                    # Add deterministic jittering on GPU
                    torch.manual_seed(42)
                    jitter_x = torch.rand(len(valid_vertices), device='cuda') * 3.0 - 1.5
                    jitter_y = torch.rand(len(valid_vertices), device='cuda') * 3.0 - 1.5
                    
                    # Convert to integer pixel coordinates
                    px_coords = torch.round(px_float + jitter_x).int().cpu().numpy()
                    py_coords = torch.round(py_float + jitter_y).int().cpu().numpy()
            else:
                # CPU fallback with optimized numpy
                # Batch projection using numpy vectorization
                px_float = focal[0] * valid_vertices[:, 0] / valid_vertices[:, 2] + princpt[0]
                py_float = focal[1] * valid_vertices[:, 1] / valid_vertices[:, 2] + princpt[1]
                
                # Add deterministic jittering to reduce overlaps
                np.random.seed(42)  # Fixed seed for reproducibility
                jitter_x = np.random.uniform(-1.5, 1.5, size=len(valid_vertices))
                jitter_y = np.random.uniform(-1.5, 1.5, size=len(valid_vertices))
                
                # Convert to integer pixel coordinates
                px_coords = np.round(px_float + jitter_x).astype(int)
                py_coords = np.round(py_float + jitter_y).astype(int)
            
            # Create bounds mask for valid pixel locations
            bounds_mask = (px_coords >= 0) & (px_coords < width) & \
                         (py_coords >= 0) & (py_coords < height)
            
            # Filter to only valid projections
            valid_px = px_coords[bounds_mask]
            valid_py = py_coords[bounds_mask]
            valid_vertex_indices = valid_indices[bounds_mask]
            valid_vertex_data = valid_vertices[bounds_mask]
            
            print(f"    [STEP 6c/6] 🔄 BATCH STORING VERTICES...")
            
            # Create a mask for occupied pixels
            occupied_mask = np.zeros((height, width), dtype=bool)
            occupied_mask[wsg_array[:, :, VERTEX_INDEX_CHANNEL] != -1] = True
            
            stored_count = 0
            failed_count = 0
            
            # Process in batches to avoid memory issues
            batch_size = 5000  # Increased batch size for better performance
            num_batches = (len(valid_px) + batch_size - 1) // batch_size
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(valid_px))
                
                # Progress reporting - less frequent for better performance
                if batch_idx % 2 == 0 or batch_idx == num_batches - 1:
                    elapsed = time.time() - storage_start_time
                    percent = (start_idx / len(valid_px)) * 100
                    rate = start_idx / elapsed if elapsed > 0 else 0
                    eta = (len(valid_px) - start_idx) / rate if rate > 0 else 0
                    print(f"      Progress: {percent:.1f}% ({start_idx}/{len(valid_px)}) - Rate: {rate:.0f} v/s - ETA: {eta:.1f}s")
                
                # Process batch
                for i in range(start_idx, end_idx):
                    px, py = valid_px[i], valid_py[i]
                    vertex_idx = valid_vertex_indices[i]
                    vertex = valid_vertex_data[i]
                    
                    # Fast check and store
                    if not occupied_mask[py, px]:
                        wsg_array[py, px, VERTEX_INDEX_CHANNEL] = vertex_idx
                        wsg_array[py, px, VERTEX_X_CHANNEL] = vertex[0]
                        wsg_array[py, px, VERTEX_Y_CHANNEL] = vertex[1]
                        wsg_array[py, px, VERTEX_Z_CHANNEL] = vertex[2]
                        if wsg_array[py, px, PERSON_ID_CHANNEL] == -1:
                            wsg_array[py, px, PERSON_ID_CHANNEL] = person_id
                        occupied_mask[py, px] = True
                        stored_count += 1
                    else:
                        # Try nearby pixels with smaller search radius for speed
                        stored = False
                        for radius in [1, 2, 3]:
                            if stored:
                                break
                            for dy in range(-radius, radius + 1):
                                if stored:
                                    break
                                for dx in range(-radius, radius + 1):
                                    if abs(dx) == radius or abs(dy) == radius:  # Only check border
                                        tx, ty = px + dx, py + dy
                                        if 0 <= tx < width and 0 <= ty < height and not occupied_mask[ty, tx]:
                                            wsg_array[ty, tx, VERTEX_INDEX_CHANNEL] = vertex_idx
                                            wsg_array[ty, tx, VERTEX_X_CHANNEL] = vertex[0]
                                            wsg_array[ty, tx, VERTEX_Y_CHANNEL] = vertex[1]
                                            wsg_array[ty, tx, VERTEX_Z_CHANNEL] = vertex[2]
                                            if wsg_array[ty, tx, PERSON_ID_CHANNEL] == -1:
                                                wsg_array[ty, tx, PERSON_ID_CHANNEL] = person_id
                                            occupied_mask[ty, tx] = True
                                            stored = True
                                            stored_count += 1
                                            break
                        
                        if not stored:
                            failed_count += 1
        
        total_vertex_time = time.time() - vertex_start_time
        storage_time = time.time() - storage_start_time
        print(f"    [STEP 6/6] ✅ BATCH VERTEX INJECTION COMPLETED!")
        print(f"               - Stored: {stored_count}/{vertices.shape[0]} vertices ({stored_count/vertices.shape[0]*100:.1f}%)")
        print(f"               - Failed: {vertices.shape[0] - stored_count} vertices")
        print(f"               - Total time: {total_vertex_time:.2f}s")
        print(f"               - Storage time: {storage_time:.2f}s") 
        print(f"               - Rate: {vertices.shape[0]/storage_time:.0f} vertices/second")
    
    print(f"    [INFO] SMPLer-X data injection completed, person_id={person_id}")
    return True


def process_single_file(args):
    """Process single file for SMPLer-X injection"""
    frame_idx, smplerx_dir, wsg_path, clear_existing, num_persons = args
    
    try:
        wsg_array = load_wsg_data(wsg_path)
        if wsg_array is None:
            return False
        if clear_existing and LIGHTWEIGHT_SMPLERX:
            wsg_array[:, :, PERSON_ID_CHANNEL:SMPLERX_END_CHANNEL] = -1
            if CLEAR_LEGACY_SMPLERX_VIS:
                wsg_array[:, :, 105:108] = -1
        
        success_count = 0
        
        # Process all persons for this frame
        frame_already_processed = False
        for bbox_id in range(num_persons):
            base_name = f"{frame_idx:05d}_{bbox_id}"
            npz_path = os.path.join(smplerx_dir, "smplx", f"{base_name}.npz")
            meta_path = os.path.join(smplerx_dir, "meta", f"{base_name}.json")
            vertices_path = os.path.join(smplerx_dir, "vertices", f"{base_name}.npz")
            mesh_path = os.path.join(smplerx_dir, "mesh", f"{base_name}.obj")
            
            # Check if required files exist
            if not os.path.exists(npz_path) or not os.path.exists(meta_path):
                continue
            
            # Load data
            params, meta, vertices = load_smplerx_data(npz_path, meta_path, vertices_path, mesh_path)
            if params is None or meta is None:
                continue
            
            # Inject SMPLer-X data (only clear on first person)
            result = inject_smplerx_to_wsg(params, meta, vertices, wsg_array, clear_existing and bbox_id == 0, clear_all_smplerx=False)
            
            if result:
                success_count += 1
        
        # In lightweight clear mode, empty frames must also be saved so stale
        # legacy SMPL vertices and false-positive people are removed.
        if success_count > 0:
            inject_smplerx_visualization(wsg_array)
        if success_count > 0 or (clear_existing and LIGHTWEIGHT_SMPLERX):
            return save_wsg_data(wsg_array, wsg_path)
        
        return False
        
    except Exception as e:
        print(f"  Failed to process file: {str(e)}")
        return False


def _smplerx_person_ids_for_frame(smplerx_dir, frame_idx):
    pattern = os.path.join(smplerx_dir, "smplx", f"{frame_idx:05d}_*.npz")
    person_ids = []
    for path in glob.glob(pattern):
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            person_ids.append(int(stem.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return sorted(set(person_ids))


def inject_smplerx_to_frame(frame_idx, smplerx_dir, wsg_array, clear_existing=True):
    """Inject trusted SMPLer-X data into an in-memory WSG frame."""
    if clear_existing and LIGHTWEIGHT_SMPLERX:
        wsg_array[:, :, PERSON_ID_CHANNEL:SMPLERX_END_CHANNEL] = -1
        if CLEAR_LEGACY_SMPLERX_VIS:
            wsg_array[:, :, 105:108] = -1

    success_count = 0
    person_ids = _smplerx_person_ids_for_frame(smplerx_dir, frame_idx)
    for person_id in person_ids:
        base_name = f"{frame_idx:05d}_{person_id}"
        npz_path = os.path.join(smplerx_dir, "smplx", f"{base_name}.npz")
        meta_path = os.path.join(smplerx_dir, "meta", f"{base_name}.json")
        vertices_path = os.path.join(smplerx_dir, "vertices", f"{base_name}.npz")
        mesh_path = os.path.join(smplerx_dir, "mesh", f"{base_name}.obj")
        if not os.path.exists(npz_path) or not os.path.exists(meta_path):
            continue
        params, meta, vertices = load_smplerx_data(npz_path, meta_path, vertices_path, mesh_path)
        if params is None or meta is None:
            continue
        result = inject_smplerx_to_wsg(
            params,
            meta,
            vertices,
            wsg_array,
            clear_existing and success_count == 0,
            clear_all_smplerx=False,
        )
        if result:
            success_count += 1

    if success_count > 0:
        inject_smplerx_visualization(wsg_array)
    elif clear_existing and LIGHTWEIGHT_SMPLERX:
        wsg_array[:, :, SMPLERX_VISUALIZATION_CHANNELS] = -1
        wsg_array[:, :, SMPLERX_RESERVED_CHANNEL] = -1
    return success_count


def _read_wsv_frame(archive, member):
    with archive.open(member, "r") as frame_file:
        return np.load(frame_file, allow_pickle=False)


def _write_wsv_frame(output_archive, member, frame):
    buffer = io.BytesIO()
    np.save(buffer, np.asanyarray(frame), allow_pickle=False)
    output_archive.writestr(member, buffer.getvalue())


def _frame_index_from_wsv_member(member):
    name = os.path.basename(member)
    if not name.startswith("index_") or not name.endswith(".npy"):
        return None
    try:
        return int(name[len("index_"):-len(".npy")])
    except ValueError:
        return None


def inject_smplerx_to_wsv(video_dir, smplerx_subdir="results_smplerx", wsv_subdir="wsv", clear_existing=True, compresslevel=3):
    """Stream-inject trusted SMPLer-X data into a WSV merge archive."""
    smplerx_dir = os.path.join(video_dir, smplerx_subdir)
    wsv_path = os.path.join(video_dir, wsv_subdir, "merge.wsv")
    if not os.path.exists(smplerx_dir):
        print(f"  SMPLer-X directory does not exist: {smplerx_dir}")
        return False
    if not os.path.exists(wsv_path):
        print(f"  WSV file does not exist: {wsv_path}")
        return False

    tmp_path = f"{wsv_path}.tmp.{os.getpid()}"
    changed_frames = 0
    total_frames = 0
    try:
        with zipfile.ZipFile(wsv_path, "r") as src, zipfile.ZipFile(
            tmp_path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=compresslevel,
        ) as dst:
            members = sorted(src.namelist())
            for member in tqdm(members, desc=f"  Inject WSV {os.path.basename(video_dir)}", leave=False):
                frame_idx = _frame_index_from_wsv_member(member)
                if frame_idx is None:
                    dst.writestr(member, src.read(member))
                    continue
                total_frames += 1
                frame = _read_wsv_frame(src, member)
                success_count = inject_smplerx_to_frame(frame_idx, smplerx_dir, frame, clear_existing=clear_existing)
                if success_count > 0 or (clear_existing and LIGHTWEIGHT_SMPLERX):
                    changed_frames += 1
                _write_wsv_frame(dst, member, frame)
        os.replace(tmp_path, wsv_path)
        print(f"  WSV injection complete: {os.path.basename(video_dir)} changed_frames={changed_frames}/{total_frames}")
        return True
    except Exception as e:
        print(f"  Failed to inject WSV {wsv_path}: {e}")
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        return False


def inject_joints_and_metadata(wsg_array, params, meta, person_id, trusted_checked=False):
    """Inject trusted pose rotations and boxes without shape coefficients or vertices."""
    if not trusted_checked and not is_trusted_smplerx_meta(meta):
        return False
    height, width = wsg_array.shape[:2]
    
    # Get bounding boxes
    bbox = meta.get('bbox', None)
    bbox_mmdet = meta.get('bbox_mmdet', None)
    
    if bbox is None:
        return False
    
    # Use mmdet bbox for storing information (more precise human region)
    if bbox_mmdet:
        x1 = max(0, int(bbox_mmdet[0]))
        y1 = max(0, int(bbox_mmdet[1]))
        x2 = min(width, int(bbox_mmdet[0] + bbox_mmdet[2]))
        y2 = min(height, int(bbox_mmdet[1] + bbox_mmdet[3]))
    else:
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(width, int(bbox[0] + bbox[2]))
        y2 = min(height, int(bbox[1] + bbox[3]))
    
    # Collect joint data
    joint_data = []
    
    # Add all joint types
    if 'global_orient' in params:
        orient = params['global_orient'].reshape(-1, 3)
        for i in range(orient.shape[0]):
            joint_data.append((SMPLERX_JOINT_MAPPING['global_orient'], orient[i]))
    
    if 'body_pose' in params:
        body = params['body_pose'].reshape(-1, 3)
        for i in range(min(21, body.shape[0])):
            joint_data.append((SMPLERX_JOINT_MAPPING['body_pose'][i], body[i]))
    
    if 'left_hand_pose' in params:
        lhand = params['left_hand_pose'].reshape(-1, 3)
        for i in range(min(15, lhand.shape[0])):
            joint_data.append((SMPLERX_JOINT_MAPPING['left_hand_pose'][i], lhand[i]))
    
    if 'right_hand_pose' in params:
        rhand = params['right_hand_pose'].reshape(-1, 3)
        for i in range(min(15, rhand.shape[0])):
            joint_data.append((SMPLERX_JOINT_MAPPING['right_hand_pose'][i], rhand[i]))
    
    if 'jaw_pose' in params:
        jaw = params['jaw_pose'].reshape(-1, 3)
        if jaw.shape[0] > 0:
            joint_data.append((SMPLERX_JOINT_MAPPING['jaw_pose'], jaw[0]))
    
    if 'leye_pose' in params:
        leye = params['leye_pose'].reshape(-1, 3)
        if leye.shape[0] > 0:
            joint_data.append((SMPLERX_JOINT_MAPPING['leye_pose'], leye[0]))
    
    if 'reye_pose' in params:
        reye = params['reye_pose'].reshape(-1, 3)
        if reye.shape[0] > 0:
            joint_data.append((SMPLERX_JOINT_MAPPING['reye_pose'], reye[0]))
    
    if 'transl' in params:
        transl = params['transl'].reshape(-1, 3)
        if transl.shape[0] > 0:
            joint_data.append((SMPLERX_JOINT_MAPPING['transl'], transl[0]))
    
    # Distribute joint data in bbox
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    
    if bbox_width > 0 and bbox_height > 0 and len(joint_data) > 0:
        # Use numpy operations for efficiency
        yy, xx = np.mgrid[y1:y2, x1:x2]
        pixels_y = yy.ravel()
        pixels_x = xx.ravel()
        
        # Get existing person IDs
        existing_persons = wsg_array[pixels_y, pixels_x, PERSON_ID_CHANNEL]
        
        # Find available pixels
        available_mask = (existing_persons == -1) | (existing_persons >= person_id)
        available_indices = np.where(available_mask)[0]
        
        if len(available_indices) > 0:
            num_joints = len(joint_data)
            num_available = len(available_indices)
            repeat_count = (num_available + num_joints - 1) // num_joints
            
            # Prepare data
            joint_ids = []
            joint_xs = []
            joint_ys = []
            joint_zs = []
            
            for _ in range(repeat_count):
                for joint_id, joint_values in joint_data:
                    joint_ids.append(joint_id)
                    joint_xs.append(joint_values[0])
                    joint_ys.append(joint_values[1])
                    joint_zs.append(joint_values[2])
            
            # Trim to exact number
            joint_ids = joint_ids[:num_available]
            joint_xs = joint_xs[:num_available]
            joint_ys = joint_ys[:num_available]
            joint_zs = joint_zs[:num_available]
            
            # Batch assign
            selected_y = pixels_y[available_indices]
            selected_x = pixels_x[available_indices]
            
            wsg_array[selected_y, selected_x, PERSON_ID_CHANNEL] = person_id
            wsg_array[selected_y, selected_x, JOINT_ID_CHANNEL] = joint_ids
            wsg_array[selected_y, selected_x, JOINT_X_CHANNEL] = joint_xs
            wsg_array[selected_y, selected_x, JOINT_Y_CHANNEL] = joint_ys
            wsg_array[selected_y, selected_x, JOINT_Z_CHANNEL] = joint_zs
    
    # Mark bbox corners
    bbox_x1 = max(0, int(bbox[0]))
    bbox_y1 = max(0, int(bbox[1]))
    bbox_x2 = min(width, int(bbox[0] + bbox[2]))
    bbox_y2 = min(height, int(bbox[1] + bbox[3]))
    
    bbox_corners = [
        (bbox_x1, bbox_y1), (bbox_x2-1, bbox_y1),
        (bbox_x1, bbox_y2-1), (bbox_x2-1, bbox_y2-1)
    ]
    
    for x, y in bbox_corners:
        if 0 <= x < width and 0 <= y < height:
            wsg_array[y, x, BBOX_CHANNEL] = 1
    
    # Mark mmdet bbox corners if available
    if bbox_mmdet:
        mmdet_x1 = max(0, int(bbox_mmdet[0]))
        mmdet_y1 = max(0, int(bbox_mmdet[1]))
        mmdet_x2 = min(width, int(bbox_mmdet[0] + bbox_mmdet[2]))
        mmdet_y2 = min(height, int(bbox_mmdet[1] + bbox_mmdet[3]))
        
        mmdet_corners = [
            (mmdet_x1, mmdet_y1), (mmdet_x2-1, mmdet_y1),
            (mmdet_x1, mmdet_y2-1), (mmdet_x2-1, mmdet_y2-1)
        ]
        
        for x, y in mmdet_corners:
            if 0 <= x < width and 0 <= y < height:
                wsg_array[y, x, BBOX_MMDET_CHANNEL] = 1
    return True


def inject_smplerx_to_video_gpu_batch_async(video_dir, smplerx_subdir, wsg_subdir, clear_existing=True, gpu_id=0):
    """GPU-optimized batch processing with async file saving"""
    if LIGHTWEIGHT_SMPLERX:
        return inject_smplerx_to_video(
            video_dir, smplerx_subdir, wsg_subdir, max_workers=mp.cpu_count(),
            clear_existing=clear_existing,
        )
    if not CUDA_AVAILABLE:
        return inject_smplerx_to_video(video_dir, smplerx_subdir, wsg_subdir, max_workers=mp.cpu_count(), clear_existing=clear_existing)
    
    # Set specific GPU for this process
    torch.cuda.set_device(gpu_id)
    
    smplerx_dir = os.path.join(video_dir, smplerx_subdir)
    wsg_dir = os.path.join(video_dir, wsg_subdir)
    
    if not os.path.exists(smplerx_dir) or not os.path.exists(wsg_dir):
        return False
    
    # Get all WSG files
    wsg_files = sorted(glob.glob(os.path.join(wsg_dir, "*.wsg")))
    if not wsg_files:
        return False
    
    # Quick check for existing data
    if not clear_existing:
        print(f"  Checking if video already has SMPLer-X data...")
        sample_indices = [0, len(wsg_files)//4, len(wsg_files)//2, 3*len(wsg_files)//4, -1]
        for idx in sample_indices:
            if idx < len(wsg_files):
                sample_wsg = wsg_files[idx if idx >= 0 else len(wsg_files) + idx]
                sample_array = load_wsg_data(sample_wsg)
                if sample_array is not None and np.any(sample_array[:, :, PERSON_ID_CHANNEL] != -1):
                    print(f"  [SKIP] Video already contains SMPLer-X data, skipping entire video!")
                    return True
    
    print(f"  [GPU ASYNC MODE] Processing {len(wsg_files)} frames with async I/O...")
    
    # Initialize async writer with more workers for I/O
    writer = AsyncFileWriter(max_workers=8)
    
    # Process in batches to keep memory usage reasonable
    batch_size = 20  # Process 20 frames at a time
    success_count = 0
    
    start_time = time.time()
    
    for batch_start in range(0, len(wsg_files), batch_size):
        batch_end = min(batch_start + batch_size, len(wsg_files))
        batch_wsg_files = wsg_files[batch_start:batch_end]
        
        # Load all frames in batch
        batch_data = []
        for wsg_path in batch_wsg_files:
            wsg_array = load_wsg_data(wsg_path)
            if wsg_array is None:
                continue
                
            frame_idx = int(os.path.basename(wsg_path).replace('.wsg', ''))
            
            # Find all person files for this frame
            smplx_pattern = os.path.join(smplerx_dir, "smplx", f"{frame_idx:05d}_*.npz")
            person_files = sorted(glob.glob(smplx_pattern))
            
            frame_data = {
                'wsg_path': wsg_path,
                'wsg_array': wsg_array,
                'frame_idx': frame_idx,
                'persons': []
            }
            
            # Collect all person data for this frame
            for person_file in person_files:
                base_name = os.path.basename(person_file).replace('.npz', '')
                person_id = int(base_name.split('_')[1]) if '_' in base_name else 0
                
                # Load data
                npz_path = person_file
                meta_path = os.path.join(smplerx_dir, "meta", f"{base_name}.json")
                vertices_path = os.path.join(smplerx_dir, "vertices", f"{base_name}.npz")
                mesh_path = os.path.join(smplerx_dir, "mesh", f"{base_name}.obj")
                
                params, meta, vertices = load_smplerx_data(npz_path, meta_path, vertices_path, mesh_path)
                if params is not None and meta is not None:
                    frame_data['persons'].append({
                        'person_id': person_id,
                        'params': params,
                        'meta': meta,
                        'vertices': vertices
                    })
            
            if frame_data['persons']:
                batch_data.append(frame_data)
        
        # Process batch on GPU
        if batch_data:
            # Collect all vertices for GPU projection
            all_vertices_batch = []
            for frame_data in batch_data:
                height, width = frame_data['wsg_array'].shape[:2]
                
                # Clear if needed
                if clear_existing:
                    frame_data['wsg_array'][:, :, PERSON_ID_CHANNEL:VERTEX_Z_CHANNEL+1] = -1
                
                for person_data in frame_data['persons']:
                    # Inject joints and metadata first
                    inject_joints_and_metadata(frame_data['wsg_array'], 
                                             person_data['params'], 
                                             person_data['meta'], 
                                             person_data['person_id'])
                    
                    # Prepare vertices for batch projection
                    vertices = person_data['vertices']
                    if vertices is not None:
                        all_vertices_batch.append({
                            'vertices': vertices,
                            'person_id': person_data['person_id'],
                            'meta': person_data['meta'],
                            'frame_data': frame_data
                        })
            
            # GPU batch projection
            if all_vertices_batch:
                print(f"  Processing batch {batch_start//batch_size + 1}/{(len(wsg_files) + batch_size - 1)//batch_size}...")
                
                with torch.no_grad():
                    for vertex_data in all_vertices_batch:
                        vertices = vertex_data['vertices']
                        meta = vertex_data['meta']
                        frame_data = vertex_data['frame_data']
                        person_id = vertex_data['person_id']
                        wsg_array = frame_data['wsg_array']
                        height, width = wsg_array.shape[:2]
                        
                        # Get camera parameters
                        cam_trans = meta.get('cam_trans', [0, 0, 0])
                        focal = meta.get('focal', [5000, 5000])
                        princpt = meta.get('princpt', [width/2, height/2])
                        
                        # Project vertices on GPU
                        vertices_gpu = torch.from_numpy(vertices).float().cuda(gpu_id)
                        cam_trans_gpu = torch.tensor(cam_trans).float().cuda(gpu_id)
                        
                        # Apply camera translation
                        vertices_cam = vertices_gpu + cam_trans_gpu
                        
                        # Project to 2D
                        x = vertices_cam[:, 0] * focal[0] / vertices_cam[:, 2] + princpt[0]
                        y = vertices_cam[:, 1] * focal[1] / vertices_cam[:, 2] + princpt[1]
                        
                        # Convert to pixel coordinates
                        px_coords = x.long().cpu().numpy()
                        py_coords = y.long().cpu().numpy()
                        
                        # Filter valid projections
                        bounds_mask = (px_coords >= 0) & (px_coords < width) & \
                                    (py_coords >= 0) & (py_coords < height)
                        
                        valid_indices = np.where(bounds_mask)[0]
                        
                        # Store vertices efficiently
                        occupied_mask = wsg_array[:, :, VERTEX_INDEX_CHANNEL] != -1
                        stored_count = 0
                        
                        for idx in valid_indices:
                            px, py = px_coords[idx], py_coords[idx]
                            if not occupied_mask[py, px]:
                                wsg_array[py, px, VERTEX_INDEX_CHANNEL] = idx
                                wsg_array[py, px, VERTEX_X_CHANNEL] = vertices[idx, 0]
                                wsg_array[py, px, VERTEX_Y_CHANNEL] = vertices[idx, 1]
                                wsg_array[py, px, VERTEX_Z_CHANNEL] = vertices[idx, 2]
                                if wsg_array[py, px, PERSON_ID_CHANNEL] == -1:
                                    wsg_array[py, px, PERSON_ID_CHANNEL] = person_id
                                occupied_mask[py, px] = True
                                stored_count += 1
                        
                        print(f"    Frame {frame_data['frame_idx']}, Person {person_id}: Stored {stored_count}/{len(vertices)} vertices")
            
            # Async save all frames in batch
            for frame_data in batch_data:
                writer.save_async(frame_data['wsg_array'], frame_data['wsg_path'])
                success_count += 1
    
    # Wait for all saves to complete
    print(f"  Waiting for async saves to complete...")
    writer.wait_all()
    writer.shutdown()
    
    elapsed = time.time() - start_time
    print(f"  [GPU ASYNC] Processed {success_count} frames in {elapsed:.1f}s ({success_count/elapsed:.1f} frames/sec)")
    
    return success_count > 0


def inject_smplerx_to_video_gpu_batch(video_dir, smplerx_subdir, wsg_subdir, clear_existing=True, gpu_id=0):
    """GPU-optimized batch processing for entire video"""
    if LIGHTWEIGHT_SMPLERX:
        return inject_smplerx_to_video(
            video_dir, smplerx_subdir, wsg_subdir, max_workers=mp.cpu_count(),
            clear_existing=clear_existing,
        )
    if not CUDA_AVAILABLE:
        # Fallback to original method if no GPU
        return inject_smplerx_to_video(video_dir, smplerx_subdir, wsg_subdir, max_workers=mp.cpu_count(), clear_existing=clear_existing)
    
    # Set specific GPU for this process
    torch.cuda.set_device(gpu_id)
    
    smplerx_dir = os.path.join(video_dir, smplerx_subdir)
    wsg_dir = os.path.join(video_dir, wsg_subdir)
    
    if not os.path.exists(smplerx_dir) or not os.path.exists(wsg_dir):
        return False
    
    # Get all WSG files
    wsg_files = sorted(glob.glob(os.path.join(wsg_dir, "*.wsg")))
    if not wsg_files:
        return False
    
    # Quick check for existing data
    if not clear_existing:
        print(f"  Checking if video already has SMPLer-X data...")
        sample_indices = [0, len(wsg_files)//4, len(wsg_files)//2, 3*len(wsg_files)//4, -1]
        for idx in sample_indices:
            if idx < len(wsg_files):
                sample_wsg = wsg_files[idx if idx >= 0 else len(wsg_files) + idx]
                sample_array = load_wsg_data(sample_wsg)
                if sample_array is not None and np.any(sample_array[:, :, PERSON_ID_CHANNEL] != -1):
                    print(f"  [SKIP] Video already contains SMPLer-X data, skipping entire video!")
                    return True
    
    print(f"  [GPU BATCH MODE] Collecting all vertices for {len(wsg_files)} frames...")
    
    # First pass: collect all vertices and metadata
    all_vertices_data = []  # List of (frame_idx, person_id, vertices, meta, wsg_path)
    total_vertices = 0
    
    for wsg_path in wsg_files:
        frame_idx = int(os.path.basename(wsg_path).replace('.wsg', ''))
        
        # Find all person files for this frame
        smplx_pattern = os.path.join(smplerx_dir, "smplx", f"{frame_idx:05d}_*.npz")
        person_files = sorted(glob.glob(smplx_pattern))
        
        for person_file in person_files:
            # Extract person ID from filename
            base_name = os.path.basename(person_file).replace('.npz', '')
            person_id = int(base_name.split('_')[1]) if '_' in base_name else 0
            
            # Load data
            meta_path = os.path.join(smplerx_dir, "meta", f"{base_name}.json")
            vertices_path = os.path.join(smplerx_dir, "vertices", f"{base_name}.npz")
            mesh_path = os.path.join(smplerx_dir, "mesh", f"{base_name}.obj")
            
            params, meta, vertices = load_smplerx_data(person_file, meta_path, vertices_path, mesh_path)
            
            if params is not None and meta is not None and vertices is not None:
                all_vertices_data.append({
                    'frame_idx': frame_idx,
                    'person_id': person_id,
                    'vertices': vertices,
                    'meta': meta,
                    'params': params,
                    'wsg_path': wsg_path
                })
                total_vertices += len(vertices)
    
    if not all_vertices_data:
        print(f"  No valid SMPLer-X data found")
        return True
    
    print(f"  Collected {len(all_vertices_data)} person instances across {len(wsg_files)} frames")
    print(f"  Total vertices to process: {total_vertices:,}")
    
    # GPU batch projection - process ALL vertices at once
    print(f"  [GPU BATCH] Projecting all {total_vertices:,} vertices on GPU...")
    projection_start = time.time()
    
    # Collect all vertices and camera params for batch processing
    all_vertices_list = []
    all_focal_list = []
    all_princpt_list = []
    all_data_indices = []  # Track which vertices belong to which person
    
    for idx, data in enumerate(all_vertices_data):
        vertices = data['vertices']
        meta = data['meta']
        focal = meta.get('focal', [1000, 1000])
        princpt = meta.get('princpt', [640, 360])
        
        valid_mask = vertices[:, 2] > 0
        valid_vertices = vertices[valid_mask]
        
        if len(valid_vertices) > 0:
            all_vertices_list.append(valid_vertices)
            all_focal_list.extend([focal] * len(valid_vertices))
            all_princpt_list.extend([princpt] * len(valid_vertices))
            all_data_indices.extend([idx] * len(valid_vertices))
    
    if all_vertices_list:
        # Concatenate all vertices
        all_vertices_np = np.vstack(all_vertices_list)
        all_focal_np = np.array(all_focal_list)
        all_princpt_np = np.array(all_princpt_list)
        
        # Single GPU operation for ALL vertices
        with torch.no_grad():
            # Move everything to GPU at once
            vertices_gpu = torch.from_numpy(all_vertices_np).float().cuda(gpu_id)
            focal_gpu = torch.from_numpy(all_focal_np).float().cuda(gpu_id)
            princpt_gpu = torch.from_numpy(all_princpt_np).float().cuda(gpu_id)
            
            # Batch projection
            px_float = focal_gpu[:, 0] * vertices_gpu[:, 0] / vertices_gpu[:, 2] + princpt_gpu[:, 0]
            py_float = focal_gpu[:, 1] * vertices_gpu[:, 1] / vertices_gpu[:, 2] + princpt_gpu[:, 1]
            
            # Add jittering for all vertices
            torch.manual_seed(42)
            jitter_x = torch.rand(len(vertices_gpu), device=f'cuda:{gpu_id}') * 3.0 - 1.5
            jitter_y = torch.rand(len(vertices_gpu), device=f'cuda:{gpu_id}') * 3.0 - 1.5
            
            px_coords = torch.round(px_float + jitter_x).int()
            py_coords = torch.round(py_float + jitter_y).int()
            
            # Transfer back to CPU in one operation
            px_coords_cpu = px_coords.cpu().numpy()
            py_coords_cpu = py_coords.cpu().numpy()
        
        # Reorganize projections by person
        all_projections = []
        current_idx = 0
        
        for idx, data in enumerate(all_vertices_data):
            vertices = data['vertices']
            valid_mask = vertices[:, 2] > 0
            valid_vertices = vertices[valid_mask]
            valid_indices = np.where(valid_mask)[0]
            
            if len(valid_vertices) > 0:
                # Extract projections for this person
                end_idx = current_idx + len(valid_vertices)
                px_coords = px_coords_cpu[current_idx:end_idx]
                py_coords = py_coords_cpu[current_idx:end_idx]
                current_idx = end_idx
                
                all_projections.append({
                    'data': data,
                    'valid_vertices': valid_vertices,
                    'valid_indices': valid_indices,
                    'px_coords': px_coords,
                    'py_coords': py_coords
                })
    else:
        # Old code path for compatibility
        all_projections = []
        with torch.no_grad():
            for data in all_vertices_data:
                vertices = data['vertices']
                meta = data['meta']
                
                # Get camera parameters
                focal = meta.get('focal', [1000, 1000])
                princpt = meta.get('princpt', [640, 360])
                
                # Filter valid vertices
                valid_mask = vertices[:, 2] > 0
                valid_vertices = vertices[valid_mask]
                valid_indices = np.where(valid_mask)[0]
                
                if len(valid_vertices) > 0:
                    # Move to GPU and project
                    vertices_gpu = torch.from_numpy(valid_vertices).float().cuda(gpu_id)
                    focal_gpu = torch.tensor(focal, dtype=torch.float32).cuda(gpu_id)
                    princpt_gpu = torch.tensor(princpt, dtype=torch.float32).cuda(gpu_id)
                    
                    # Batch projection
                    px_float = focal_gpu[0] * vertices_gpu[:, 0] / vertices_gpu[:, 2] + princpt_gpu[0]
                    py_float = focal_gpu[1] * vertices_gpu[:, 1] / vertices_gpu[:, 2] + princpt_gpu[1]
                    
                    # Add jittering
                    torch.manual_seed(42 + data['person_id'])
                    jitter_x = torch.rand(len(valid_vertices), device=f'cuda:{gpu_id}') * 3.0 - 1.5
                    jitter_y = torch.rand(len(valid_vertices), device=f'cuda:{gpu_id}') * 3.0 - 1.5
                    
                    px_coords = torch.round(px_float + jitter_x).int().cpu().numpy()
                    py_coords = torch.round(py_float + jitter_y).int().cpu().numpy()
                    
                    all_projections.append({
                        'data': data,
                        'valid_vertices': valid_vertices,
                        'valid_indices': valid_indices,
                        'px_coords': px_coords,
                        'py_coords': py_coords
                    })
    
    print(f"  [GPU BATCH] Projection completed in {time.time() - projection_start:.2f}s")
    
    # Second pass: write projections back to WSG files
    print(f"  Writing projections back to WSG files...")
    
    # Group by frame
    frame_groups = {}
    for proj in all_projections:
        wsg_path = proj['data']['wsg_path']
        if wsg_path not in frame_groups:
            frame_groups[wsg_path] = []
        frame_groups[wsg_path].append(proj)
    
    # Process each frame
    success_count = 0
    for wsg_path, projections in frame_groups.items():
        wsg_array = load_wsg_data(wsg_path)
        if wsg_array is None:
            continue
        
        height, width = wsg_array.shape[:2]
        
        # Clear if needed
        if clear_existing:
            wsg_array[:, :, PERSON_ID_CHANNEL:VERTEX_Z_CHANNEL+1] = -1
        
        # Process all persons in this frame
        for proj in projections:
            data = proj['data']
            person_id = data['person_id']
            params = data['params']
            meta = data['meta']
            
            # First inject joints and other data (non-vertex)
            inject_joints_and_metadata(wsg_array, params, meta, person_id)
            
            # Then inject pre-computed vertex projections
            px_coords = proj['px_coords']
            py_coords = proj['py_coords']
            valid_vertices = proj['valid_vertices']
            valid_indices = proj['valid_indices']
            
            # Create bounds mask
            bounds_mask = (px_coords >= 0) & (px_coords < width) & \
                         (py_coords >= 0) & (py_coords < height)
            
            # GPU-accelerated vertex storage with collision detection
            if CUDA_AVAILABLE:
                with torch.no_grad():
                    # Move arrays to GPU for faster processing
                    occupied_gpu = torch.from_numpy(wsg_array[:, :, VERTEX_INDEX_CHANNEL] != -1).cuda(gpu_id)
                    px_gpu = torch.from_numpy(px_coords).cuda(gpu_id)
                    py_gpu = torch.from_numpy(py_coords).cuda(gpu_id)
                    bounds_mask_gpu = torch.from_numpy(bounds_mask).cuda(gpu_id)
                    
                    # Find valid storage locations on GPU
                    valid_storage = torch.zeros(len(px_coords), dtype=torch.bool, device=f'cuda:{gpu_id}')
                    
                    # Process in batches to avoid memory issues
                    batch_size = 10000
                    for start in range(0, len(px_coords), batch_size):
                        end = min(start + batch_size, len(px_coords))
                        batch_bounds = bounds_mask_gpu[start:end]
                        batch_px = px_gpu[start:end]
                        batch_py = py_gpu[start:end]
                        
                        # Check occupancy for this batch
                        for i in range(end - start):
                            if batch_bounds[i]:
                                if not occupied_gpu[batch_py[i], batch_px[i]]:
                                    valid_storage[start + i] = True
                                    occupied_gpu[batch_py[i], batch_px[i]] = True
                    
                    # Get valid indices back to CPU
                    valid_storage_cpu = valid_storage.cpu().numpy()
                    valid_indices_to_store = np.where(valid_storage_cpu)[0]
                    
                    # Batch store all valid vertices
                    for i in valid_indices_to_store:
                        px, py = px_coords[i], py_coords[i]
                        vertex_idx = valid_indices[i]
                        vertex = valid_vertices[i]
                        
                        wsg_array[py, px, VERTEX_INDEX_CHANNEL] = vertex_idx
                        wsg_array[py, px, VERTEX_X_CHANNEL] = vertex[0]
                        wsg_array[py, px, VERTEX_Y_CHANNEL] = vertex[1]
                        wsg_array[py, px, VERTEX_Z_CHANNEL] = vertex[2]
                        if wsg_array[py, px, PERSON_ID_CHANNEL] == -1:
                            wsg_array[py, px, PERSON_ID_CHANNEL] = person_id
                    
                    stored_count = len(valid_indices_to_store)
            else:
                # CPU fallback
                stored_count = 0
                occupied_mask = wsg_array[:, :, VERTEX_INDEX_CHANNEL] != -1
                
                for i in np.where(bounds_mask)[0]:
                    px, py = px_coords[i], py_coords[i]
                    if not occupied_mask[py, px]:
                        vertex_idx = valid_indices[i]
                        vertex = valid_vertices[i]
                        
                        wsg_array[py, px, VERTEX_INDEX_CHANNEL] = vertex_idx
                        wsg_array[py, px, VERTEX_X_CHANNEL] = vertex[0]
                        wsg_array[py, px, VERTEX_Y_CHANNEL] = vertex[1]
                        wsg_array[py, px, VERTEX_Z_CHANNEL] = vertex[2]
                        if wsg_array[py, px, PERSON_ID_CHANNEL] == -1:
                            wsg_array[py, px, PERSON_ID_CHANNEL] = person_id
                        occupied_mask[py, px] = True
                        stored_count += 1
            
            print(f"    Frame {os.path.basename(wsg_path)}, Person {person_id}: Stored {stored_count} vertices")
        
        # Save frame
        if save_wsg_data(wsg_array, wsg_path):
            success_count += 1
    
    print(f"  [GPU BATCH] Complete! Success: {success_count}/{len(frame_groups)}")
    return True


def inject_smplerx_to_video(video_dir, smplerx_subdir, wsg_subdir, max_workers=4, clear_existing=True):
    """Inject SMPLer-X data for single video directory"""
    smplerx_dir = os.path.join(video_dir, smplerx_subdir)
    wsg_dir = os.path.join(video_dir, wsg_subdir)
    
    if not os.path.exists(smplerx_dir):
        print(f"  SMPLer-X directory does not exist: {smplerx_dir}")
        return False
    
    if not os.path.exists(wsg_dir):
        print(f"  WSG directory does not exist: {wsg_dir}")
        return False
    
    # Get all WSG files
    wsg_files = glob.glob(os.path.join(wsg_dir, "*.wsg"))
    wsg_files.sort()
    
    if not wsg_files:
        print(f"  No WSG files found")
        return False
    
    # Quick check: Sample a few frames to see if video is already processed
    if not clear_existing:
        print(f"  Checking if video already has SMPLer-X data...")
        sample_indices = [0, len(wsg_files)//4, len(wsg_files)//2, 3*len(wsg_files)//4, -1]
        has_smplerx = False
        
        for idx in sample_indices:
            if idx < len(wsg_files):
                sample_wsg = wsg_files[idx if idx >= 0 else len(wsg_files) + idx]
                sample_array = load_wsg_data(sample_wsg)
                if sample_array is not None:
                    # Check if this frame has SMPLer-X data
                    if np.any(sample_array[:, :, PERSON_ID_CHANNEL] != -1):
                        has_smplerx = True
                        break
        
        if has_smplerx:
            print(f"  [SKIP] Video already contains SMPLer-X data, skipping entire video!")
            # Don't record here - let process_dataset handle all recording to avoid duplicates
            return True  # Return True to indicate video was already processed
    
    # Prepare tasks - process all bbox files for each frame
    tasks = []
    for wsg_path in wsg_files:
        filename = os.path.basename(wsg_path)
        # Extract frame index from WSG filename
        frame_idx = int(filename.replace('.wsg', ''))
        
        # Find all bbox files for this frame
        smplx_pattern = os.path.join(smplerx_dir, "smplx", f"{frame_idx:05d}_*.npz")
        bbox_files = glob.glob(smplx_pattern)
        
        if bbox_files:
            tasks.append((frame_idx, smplerx_dir, wsg_path, clear_existing, len(bbox_files)))
        else:
            tasks.append((frame_idx, smplerx_dir, wsg_path, clear_existing, 1))
    
    print(f"  Found {len(tasks)} files to process")
    
    # Parallel processing
    success_count = 0
    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_single_file, task) for task in tasks]
            for future in tqdm(futures, desc="  Injecting SMPLer-X"):
                try:
                    if future.result():
                        success_count += 1
                except Exception as e:
                    print(f"  Task failed: {str(e)}")
    else:
        for task in tqdm(tasks, desc="  Injecting SMPLer-X"):
            if process_single_file(task):
                success_count += 1
    
    print(f"  Complete! Success: {success_count}/{len(tasks)}")
    
    # Return True if any frames were processed OR if this is an empty video (no SMPLer-X data)
    # This ensures empty videos are also marked as processed
    return True  # Always return True to mark video as checked


def process_videos_on_gpu(args):
    """Process a batch of videos on a specific GPU"""
    gpu_id, video_batch, dataset_results_dir, smplerx_subdir, wsg_subdir, clear_existing, processed_list_path, processed_videos_lock = args
    
    print(f"  GPU {gpu_id}: Processing {len(video_batch)} videos")
    success_count = 0
    
    for video_folder in video_batch:
        video_dir = os.path.join(dataset_results_dir, video_folder)
        try:
            result = inject_smplerx_to_video_gpu_batch_async(video_dir, smplerx_subdir, wsg_subdir, clear_existing, gpu_id)
            if result:
                success_count += 1
                
                # Thread-safe recording
                with processed_videos_lock:
                    try:
                        with open(processed_list_path, 'a') as f:
                            f.write(f"{video_folder}\n")
                    except Exception as e:
                        print(f"Failed to update processed videos list: {str(e)}")
        except Exception as e:
            print(f"  GPU {gpu_id}: Error processing {video_folder}: {str(e)}")
    
    print(f"  GPU {gpu_id}: Completed {success_count}/{len(video_batch)} videos")
    return success_count


def process_dataset(dataset_results_dir, smplerx_subdir, wsg_subdir, max_workers=4, clear_existing=True):
    """Process entire dataset"""
    if not os.path.exists(dataset_results_dir):
        print(f"Dataset directory does not exist: {dataset_results_dir}")
        return
    
    # Check for processed videos list
    processed_list_path = os.path.join(dataset_results_dir, "processed_video_names_by_smplerx_inject.txt")
    processed_videos = set()
    processed_videos_lock = threading.Lock()  # Thread lock for safe concurrent access
    
    if clear_existing and os.path.exists(processed_list_path):
        # Clear the processed list if we're clearing existing data
        print(f"Clearing processed videos list due to clear_existing=True")
        try:
            os.remove(processed_list_path)
        except Exception as e:
            print(f"Failed to clear processed videos list: {str(e)}")
    
    if os.path.exists(processed_list_path) and not clear_existing:
        print(f"Loading processed videos list from: {processed_list_path}")
        try:
            with open(processed_list_path, 'r') as f:
                processed_videos = set(line.strip() for line in f if line.strip())
            print(f"Found {len(processed_videos)} already processed videos")
        except Exception as e:
            print(f"Failed to load processed videos list: {str(e)}")
    
    video_folders = [f for f in os.listdir(dataset_results_dir) 
                    if os.path.isdir(os.path.join(dataset_results_dir, f))]
    
    if not video_folders:
        print("No video folders found")
        return
    
    print(f"Found {len(video_folders)} video folders")
    
    # Filter out already processed videos
    if processed_videos:
        video_folders = [v for v in video_folders if v not in processed_videos]
        print(f"After filtering: {len(video_folders)} videos to process")
    
    success_count = 0
    newly_processed_videos = []
    
    # Define a function to safely record processed videos
    def record_processed_video(video_name):
        with processed_videos_lock:
            if video_name not in processed_videos:
                try:
                    with open(processed_list_path, 'a') as f:
                        f.write(f"{video_name}\n")
                    processed_videos.add(video_name)
                    return True
                except Exception as e:
                    print(f"Failed to update processed videos list: {str(e)}")
                    return False
            return False
    
    if LIGHTWEIGHT_SMPLERX and CUDA_AVAILABLE and torch.cuda.device_count() > 1:
        print(
            "Lightweight neutral-body rendering uses one GPU per injector process. "
            "Run multiple dataset shards as separate processes to use multiple GPUs safely."
        )

    if CUDA_AVAILABLE and torch.cuda.device_count() > 1 and not LIGHTWEIGHT_SMPLERX:
        # Multi-GPU processing
        num_gpus = torch.cuda.device_count()
        print(f"Using {num_gpus} GPUs for parallel processing")
        
        # Progress tracking
        total_videos = len(video_folders)
        initial_processed = len(processed_videos)
        total_to_process = len(video_folders) + initial_processed
        start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"PROGRESS TRACKING:")
        print(f"Total videos in dataset: {total_to_process}")
        print(f"Already processed: {initial_processed}")
        print(f"Videos to process: {total_videos}")
        print(f"{'='*60}\n")
        
        # Split videos into batches for each GPU
        videos_per_gpu = len(video_folders) // num_gpus
        video_batches = []
        
        for i in range(num_gpus):
            start_idx = i * videos_per_gpu
            if i == num_gpus - 1:
                # Last GPU gets remaining videos
                batch = video_folders[start_idx:]
            else:
                batch = video_folders[start_idx:start_idx + videos_per_gpu]
            video_batches.append(batch)
        
        # Create process pool for GPUs
        import multiprocessing.dummy as mp_dummy
        processed_videos_lock = threading.Lock()
        
        # Progress monitor thread
        progress_stop = threading.Event()
        
        def progress_monitor():
            """Monitor progress and estimate completion time"""
            last_count = initial_processed
            while not progress_stop.is_set():
                time.sleep(30)  # Update every 30 seconds
                
                # Count currently processed videos
                with processed_videos_lock:
                    current_count = len(processed_videos)
                
                if current_count > last_count:
                    elapsed = time.time() - start_time
                    videos_processed_this_session = current_count - initial_processed
                    rate = videos_processed_this_session / elapsed if elapsed > 0 else 0
                    remaining = total_to_process - current_count
                    eta_seconds = remaining / rate if rate > 0 else 0
                    
                    # Format time
                    eta_hours = eta_seconds / 3600
                    eta_days = eta_hours / 24
                    
                    print(f"\n{'='*60}")
                    print(f"PROGRESS UPDATE [{time.strftime('%Y-%m-%d %H:%M:%S')}]")
                    print(f"Processed: {current_count}/{total_to_process} ({current_count/total_to_process*100:.1f}%)")
                    print(f"This session: {videos_processed_this_session} videos in {elapsed/3600:.1f} hours")
                    print(f"Rate: {rate*60:.1f} videos/minute")
                    if eta_days > 1:
                        print(f"ETA: {eta_days:.1f} days ({eta_hours:.1f} hours)")
                    else:
                        print(f"ETA: {eta_hours:.1f} hours")
                    print(f"{'='*60}\n")
                    
                    last_count = current_count
        
        # Start progress monitor
        progress_thread = threading.Thread(target=progress_monitor, daemon=True)
        progress_thread.start()
        
        # Prepare arguments for each GPU
        gpu_args = []
        for gpu_id, batch in enumerate(video_batches):
            if batch:  # Only add if batch is not empty
                gpu_args.append((
                    gpu_id, batch, dataset_results_dir, smplerx_subdir, 
                    wsg_subdir, clear_existing, processed_list_path, processed_videos_lock
                ))
        
        # Process on multiple GPUs in parallel
        with mp_dummy.Pool(len(gpu_args)) as pool:
            results = pool.map(process_videos_on_gpu, gpu_args)
        
        # Stop progress monitor
        progress_stop.set()
        
        # Final progress report
        final_count = len(processed_videos)
        total_elapsed = time.time() - start_time
        videos_processed_this_session = final_count - initial_processed
        
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS:")
        print(f"Total processed: {final_count}/{total_to_process}")
        print(f"This session: {videos_processed_this_session} videos in {total_elapsed/3600:.1f} hours")
        print(f"Average rate: {videos_processed_this_session/(total_elapsed/60):.1f} videos/minute")
        print(f"{'='*60}\n")
        
        success_count = sum(results)
        newly_processed_videos = []  # We don't track this in multi-GPU mode
        
    else:
        # Single GPU or CPU processing
        for video_folder in tqdm(video_folders, desc="Processing videos"):
            video_dir = os.path.join(dataset_results_dir, video_folder)
            print(f"\nProcessing video: {video_folder}")
            
            # Use GPU batch mode if available
            if CUDA_AVAILABLE:
                result = inject_smplerx_to_video_gpu_batch(video_dir, smplerx_subdir, wsg_subdir, clear_existing)
            else:
                result = inject_smplerx_to_video(video_dir, smplerx_subdir, wsg_subdir, max_workers, clear_existing)
            
            if result:
                success_count += 1
                newly_processed_videos.append(video_folder)
                
                # Update processed videos list after each successful video
                record_processed_video(video_folder)
    
    print(f"\nDataset processing complete! Success: {success_count}/{len(video_folders)}")
    print(f"Newly processed videos: {len(newly_processed_videos)}")


def _dataset_video_folders(dataset_results_dir):
    if not os.path.exists(dataset_results_dir):
        return []
    return sorted(
        name for name in os.listdir(dataset_results_dir)
        if os.path.isdir(os.path.join(dataset_results_dir, name))
    )


def process_wsv_dataset(
    dataset_results_dir,
    smplerx_subdir,
    wsv_subdir,
    clear_existing=True,
    completion_tag="trusted_v2",
    num_shards=1,
    shard_index=0,
    limit=None,
    compresslevel=3,
):
    processed_list_path = os.path.join(
        dataset_results_dir,
        f"processed_video_name_by_smplerx_wsv_inject_{completion_tag}.txt",
    )
    processed_videos = set()
    if os.path.exists(processed_list_path):
        with open(processed_list_path, "r", encoding="utf-8") as f:
            processed_videos = set(line.strip() for line in f if line.strip())

    all_folders = _dataset_video_folders(dataset_results_dir)
    all_pending = [name for name in all_folders if name not in processed_videos]
    pending = all_pending[shard_index::num_shards]
    if limit is not None:
        pending = pending[:limit]

    print("=" * 60)
    print(f"WSV SMPLer-X injection dataset: {dataset_results_dir}")
    print(
        f"all={len(all_folders)} processed={len(processed_videos)} "
        f"all_pending={len(all_pending)} shard={shard_index}/{num_shards} pending={len(pending)}"
    )
    print(f"completion: {processed_list_path}")
    print(f"clear_existing={clear_existing} clear_legacy_105_107={CLEAR_LEGACY_SMPLERX_VIS}")
    print("=" * 60)

    success_count = 0
    start_time = time.time()
    with tqdm(pending, desc=f"WSV inject {os.path.basename(dataset_results_dir)} shard {shard_index}/{num_shards}", unit="sample") as progress:
        for video_name in progress:
            video_dir = os.path.join(dataset_results_dir, video_name)
            progress.set_postfix_str(video_name[:32])
            ok = inject_smplerx_to_wsv(
                video_dir,
                smplerx_subdir=smplerx_subdir,
                wsv_subdir=wsv_subdir,
                clear_existing=clear_existing,
                compresslevel=compresslevel,
            )
            if ok:
                success_count += 1
                with open(processed_list_path, "a", encoding="utf-8") as f:
                    f.write(f"{video_name}\n")
            elapsed = time.time() - start_time
            if success_count > 0 and elapsed > 0:
                rate = success_count / elapsed
                remaining = len(pending) - success_count
                eta_hours = remaining / rate / 3600 if rate > 0 else 0
                progress.set_postfix_str(f"{video_name[:20]} eta={eta_hours:.2f}h")

    print(f"WSV dataset complete: success={success_count}/{len(pending)}")
    return success_count


def parse_args():
    parser = argparse.ArgumentParser(description="Inject trusted SMPLer-X outputs into WSG/WSV files.")
    parser.add_argument("--datasets", nargs="+", default=["AVIS_train", "AVIS_val", "AVIS_test"])
    parser.add_argument("--base-output-dir", default="../output")
    parser.add_argument("--smplerx-subdir", default="results_smplerx")
    parser.add_argument("--wsg-subdir", default="wsg")
    parser.add_argument("--wsv-subdir", default="wsv")
    parser.add_argument("--mode", choices=["wsv", "wsg"], default="wsv")
    parser.add_argument("--completion-tag", default="trusted_v2")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=mp.cpu_count())
    parser.add_argument("--compresslevel", type=int, default=3)
    parser.add_argument("--clear-existing", action="store_true", default=True)
    parser.add_argument("--no-clear-existing", dest="clear_existing", action="store_false")
    parser.add_argument("--clear-legacy-smplerx-vis", action="store_true")
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    if args.clear_legacy_smplerx_vis:
        global CLEAR_LEGACY_SMPLERX_VIS
        CLEAR_LEGACY_SMPLERX_VIS = True

    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")

    print("=" * 50)
    print("SMPLer-X to WSG/WSV Injector")
    print("=" * 50)
    print(f"Datasets: {args.datasets}")
    print(f"Mode: {args.mode}")
    print(f"Shard: {args.shard_index}/{args.num_shards}")
    print(f"Clear existing data: {'Yes' if args.clear_existing else 'No'}")
    print(f"Clear legacy 105-107: {'Yes' if CLEAR_LEGACY_SMPLERX_VIS else 'No'}")
    print("=" * 50)

    try:
        for dataset_name in args.datasets:
            dataset_results_dir = os.path.join(args.base_output_dir, f"{dataset_name}_results")
            if args.mode == "wsv":
                process_wsv_dataset(
                    dataset_results_dir,
                    args.smplerx_subdir,
                    args.wsv_subdir,
                    clear_existing=args.clear_existing,
                    completion_tag=args.completion_tag,
                    num_shards=args.num_shards,
                    shard_index=args.shard_index,
                    limit=args.limit,
                    compresslevel=args.compresslevel,
                )
            else:
                process_dataset(
                    dataset_results_dir,
                    args.smplerx_subdir,
                    args.wsg_subdir,
                    args.max_workers,
                    args.clear_existing,
                )
        print("\n✅ All injection tasks completed!")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
