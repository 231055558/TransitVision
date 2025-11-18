import cv2
import numpy as np
from ultralytics import YOLO
from ...data_structures import Door

class DoorSegmentor:
    def __init__(self, model_path, device_config):
        self.model = YOLO(model_path)
        self.model.to(device_config.device)
        self.device = device_config
    
    def detect(self, frame, conf=0.75, min_area=5000):
        results = self.model.predict(
            source=frame,
            conf=conf,
            verbose=False,
            device=self.device.device_str
        )
        
        if results[0].masks is None or len(results[0].masks) == 0:
            return None
        
        h, w = frame.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        
        for mask in results[0].masks.data.cpu().numpy():
            mask_resized = cv2.resize(mask, (w, h))
            mask_binary = (mask_resized > 0.5).astype(np.uint8)
            combined_mask = cv2.bitwise_or(combined_mask, mask_binary)
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            combined_mask, connectivity=8
        )
        
        final_mask = np.zeros((h, w), dtype=np.uint8)
        
        for i in range(1, num_labels):
            x, y, w_bbox, h_bbox, area = stats[i]
            bbox_area = w_bbox * h_bbox
            
            if bbox_area >= min_area:
                final_mask[labels == i] = 255
        
        if np.sum(final_mask) == 0:
            return None
        
        return Door(final_mask)
    
    def detect_with_angle(self, frame, conf=0.3):
        results = self.model.predict(
            source=frame,
            conf=conf,
            verbose=False,
            device=self.device.device_str
        )
        
        if results[0].masks is None or len(results[0].masks) == 0:
            return None
        
        mask = results[0].masks.data[0].cpu().numpy()
        h, w = frame.shape[:2]
        mask_resized = cv2.resize(mask, (w, h))
        mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel)
        mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask_binary, connectivity=8
        )
        
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            max_area = areas.max()
            
            new_mask = np.zeros_like(mask_binary)
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area >= max_area * 0.1:
                    new_mask[labels == i] = 255
            
            mask_binary = new_mask
        
        points = cv2.findNonZero(mask_binary)
        if points is None:
            return None
        
        angle = self._calc_angle(points)
        rotation_angle = -angle
        
        M = cv2.getRotationMatrix2D((w // 2, h // 2), rotation_angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2
        
        mask_rotated = cv2.warpAffine(mask_binary, M, (new_w, new_h), borderValue=0)
        
        ys, xs = np.where(mask_rotated > 0)
        if len(xs) == 0:
            return None
        
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        
        return Door(mask_rotated, bbox, angle)
    
    def _calc_angle(self, points):
        def calc_ratio(pts, ang):
            center = pts.mean(axis=0)[0]
            rad = np.radians(ang)
            cos_a, sin_a = np.cos(rad), np.sin(rad)
            
            rotated = pts.copy().astype(np.float32)
            for i in range(len(rotated)):
                x, y = rotated[i][0] - center
                rotated[i][0][0] = x * cos_a - y * sin_a + center[0]
                rotated[i][0][1] = x * sin_a + y * cos_a + center[1]
            
            x_min, y_min = rotated.min(axis=0)[0]
            x_max, y_max = rotated.max(axis=0)[0]
            bbox_area = (x_max - x_min) * (y_max - y_min)
            
            return len(pts) / bbox_area if bbox_area > 0 else 0
        
        def check_upright(pts, ang):
            center = pts.mean(axis=0)[0]
            rad = np.radians(ang)
            cos_a, sin_a = np.cos(rad), np.sin(rad)
            
            rotated = pts.copy().astype(np.float32)
            for i in range(len(rotated)):
                x, y = rotated[i][0] - center
                rotated[i][0][0] = x * cos_a - y * sin_a + center[0]
                rotated[i][0][1] = x * sin_a + y * cos_a + center[1]
            
            x_min, y_min = rotated.min(axis=0)[0]
            x_max, y_max = rotated.max(axis=0)[0]
            h = y_max - y_min
            third = h / 3
            
            y_coords = rotated[:, 0, 1]
            upper_cnt = np.sum((y_coords >= y_min) & (y_coords <= y_min + third))
            lower_cnt = np.sum((y_coords >= y_max - third) & (y_coords <= y_max))
            
            w = x_max - x_min
            upper_ratio = upper_cnt / (w * third) if w * third > 0 else 0
            lower_ratio = lower_cnt / (w * third) if w * third > 0 else 0
            
            return upper_ratio > lower_ratio
        
        ratios = []
        angles = []
        
        for ang in range(0, 180, 5):
            ratio = calc_ratio(points, ang)
            ratios.append(ratio)
            angles.append(ang)
            
            if len(ratios) >= 5:
                idx = len(ratios) - 3
                if (ratios[idx] > ratios[idx-1] and ratios[idx] > ratios[idx-2] and 
                    ratios[idx] > ratios[idx+1] and ratios[idx] > ratios[idx+2]):
                    break
        
        best_idx = ratios.index(max(ratios))
        best_angle = angles[best_idx]
        
        if not check_upright(points, best_angle):
            best_angle = (best_angle + 180) % 360
        
        return best_angle

