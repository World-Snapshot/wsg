import os
import shutil
from typing import List
import requests
from tqdm import tqdm
from enum import Enum
from huggingface_hub import hf_hub_download, hf_hub_url

from torchvision import transforms


class TaskType(Enum):
    DEPTH = "depth"
    NORMAL = "normal"
    SEG = "seg"
    POSE = "pose"


def download(url: str, filename: str):
    with open(filename, 'wb') as f:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0))

            # tqdm has many interesting parameters. Feel free to experiment!
            tqdm_params = {
                'total': total,
                'miniters': 1,
                'unit': 'B',
                'unit_scale': True,
                'unit_divisor': 1024,
            }
            with tqdm(**tqdm_params) as pb:
                for chunk in r.iter_content(chunk_size=8192):
                    pb.update(len(chunk))
                    f.write(chunk)


def download_hf_model(model_name: str, task_type: TaskType, model_dir: str = 'models'):
    # 根据任务类型构建正确的子目录路径
    if task_type == TaskType.NORMAL:
        subdir = 'sapiens-normal-2b-torchscript'
        model_file = 'sapiens_2b_normal_render_people_epoch_70_torchscript.pt2'
    elif task_type == TaskType.DEPTH:
        subdir = 'sapiens-depth-2b-torchscript'
        model_file = 'sapiens_2b_render_people_epoch_25_torchscript.pt2'
    elif task_type == TaskType.POSE:
        subdir = 'sapiens-pose-1b-torchscript'
        model_file = 'sapiens_1b_goliath_best_goliath_AP_639_torchscript.pt2'
    elif task_type == TaskType.SEG:
        subdir = 'sapiens-seg-1b-torchscript'
        model_file = 'sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2'
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    # 构建完整的模型路径
    path = os.path.join(model_dir, subdir, model_file)
    
    if os.path.exists(path):
        return path
    else:
        raise FileNotFoundError(f"Model not found at {path}. Please make sure the model file exists in the correct directory.")

from typing import Tuple
def create_preprocessor(input_size: Tuple[int, int],
                        mean: List[float] = (0.485, 0.456, 0.406),
                        std: List[float] = (0.229, 0.224, 0.225)):
    return transforms.Compose([transforms.ToPILImage(),
                               transforms.Resize(input_size),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=mean, std=std),
                               transforms.Lambda(lambda x: x.unsqueeze(0))
                               ])

def pose_estimation_preprocessor(input_size: Tuple[int, int],
                        mean: List[float] = (0.485, 0.456, 0.406),
                        std: List[float] = (0.229, 0.224, 0.225)):
    
    return transforms.Compose([transforms.ToPILImage(),
                               transforms.Resize(input_size),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=mean, std=std),
                               ])