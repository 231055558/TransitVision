import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from ...data_structures import Person

class PersonSegTracker:
    def __init__(self, model_path, tracker_config, device_config):
        self.model = YOLO(model_path)
        self.model.to(device_config.device)
        self.tracker_config = tracker_config
        self.device = device_config
        self.tracks = defaultdict(lambda: Person(0))
    
    def track(self, frame, conf=0.3):
        results = self.model.track(
            source=frame,
            tracker=self.tracker_config,
            classes=0,
            conf=conf,
            persist=True,
            verbose=False,
            device=self.device.device_str
        )
        
        detections = []
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()
            
            masks = None
            if results[0].masks is not None:
                masks = results[0].masks.data.cpu().numpy()
            
            for i, track_id in enumerate(track_ids):
                box = boxes[i]
                conf_val = confs[i]
                mask = masks[i] if masks is not None else None
                
                if mask is not None:
                    h, w = frame.shape[:2]
                    mask_resized = cv2.resize(mask, (w, h))
                    mask_binary = (mask_resized > 0.5).astype(np.uint8)
                else:
                    mask_binary = None
                
                detections.append({
                    'id': track_id,
                    'box': box.tolist(),
                    'mask': mask_binary,
                    'conf': float(conf_val)
                })
        
        return detections
    
    def track_video(self, video_reader, conf=0.3):
        frame_idx = 0
        all_tracks = defaultdict(lambda: Person(0))
        
        for frame in video_reader:
            detections = self.track(frame, conf)
            
            for det in detections:
                track_id = det['id']
                if all_tracks[track_id].id == 0:
                    all_tracks[track_id].id = track_id
                
                all_tracks[track_id].add_detection(
                    frame_idx,
                    det['box'],
                    det['mask'],
                    det['conf']
                )
            
            frame_idx += 1
        
        return dict(all_tracks)
    
    def reset(self):
        self.tracks.clear()

