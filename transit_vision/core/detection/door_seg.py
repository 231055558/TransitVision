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

