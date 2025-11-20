import cv2
import numpy as np
from pathlib import Path

def collect_images_from_folder(folder_path, recursive=True):
    """收集文件夹中的所有图像"""
    folder = Path(folder_path)
    
    if not folder.exists():
        return []
    
    image_files = []
    
    if recursive:
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend(folder.rglob(f'*{ext}'))
            image_files.extend(folder.rglob(f'*{ext.upper()}'))
    else:
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend(folder.glob(f'*{ext}'))
            image_files.extend(folder.glob(f'*{ext.upper()}'))
    
    return [str(f) for f in image_files]


def load_images_batch(image_paths):
    """批量加载图像"""
    images = []
    valid_paths = []
    
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            images.append(img)
            valid_paths.append(path)
    
    return images, valid_paths

