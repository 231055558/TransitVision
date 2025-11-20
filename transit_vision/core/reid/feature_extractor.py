import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import pickle
from pathlib import Path

class RectScale:
    def __init__(self, height, width, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.interpolation = interpolation
        
    def __call__(self, img):
        w, h = img.size
        if h == self.height and w == self.width:
            return img
        return img.resize((self.width, self.height), self.interpolation)


class ReIDFeatureExtractor:
    def __init__(self, model_path, cfg_path, device='cuda'):
        self.device = device
        self.model = self._load_model(model_path, cfg_path)
        self.transform = self._build_transform()
    
    def _load_model(self, model_path, cfg_path):
        cfg = pickle.load(open(cfg_path, 'rb'))
        
        from .pose2id_model import make_model
        
        model = make_model(cfg, num_class=751, camera_num=0, view_num=1)
        model.load_param(model_path)
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def _build_transform(self):
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        return transforms.Compose([
            transforms.ToPILImage(),
            RectScale(256, 128),
            transforms.ToTensor(),
            normalize
        ])
    
    def extract_single(self, image):
        """从单张图像提取特征"""
        if isinstance(image, str):
            image = cv2.imread(image)
        
        if image is None:
            return None
        
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            feat = self.model(img_tensor,
                            cam_label=torch.zeros(1, dtype=torch.long).to(self.device),
                            view_label=torch.ones(1, dtype=torch.long).to(self.device))
        
        return feat.cpu()
    
    def extract_batch(self, images, batch_size=8):
        """从图像列表批量提取特征"""
        features = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            batch_tensors = []
            
            for img in batch:
                if isinstance(img, str):
                    img = cv2.imread(img)
                
                if img is None:
                    continue
                
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_tensor = self.transform(img_rgb)
                batch_tensors.append(img_tensor)
            
            if len(batch_tensors) == 0:
                continue
            
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            
            with torch.no_grad():
                feat = self.model(batch_tensor,
                                cam_label=torch.zeros(batch_tensor.shape[0], dtype=torch.long).to(self.device),
                                view_label=torch.ones(batch_tensor.shape[0], dtype=torch.long).to(self.device))
                features.append(feat.cpu())
        
        if len(features) == 0:
            return None
        
        return torch.cat(features, dim=0)

