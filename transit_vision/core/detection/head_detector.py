from typing import List, Dict
from ultralytics import YOLO

class HeadDetector:
    def __init__(self, model_path: str, tracker_config: str = 'botsort.yaml'):
        self.model = YOLO(model_path)
        self.tracker_config = tracker_config

    def detect_and_track(self, frame, conf: float = 0.6, device=None) -> List[Dict]:
        results = self.model.track(
            source=frame,
            tracker=self.tracker_config,
            classes=0,
            conf=conf,
            persist=True,
            verbose=False,
            device=device
        )

        detections = []
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            
            for i, track_id in enumerate(track_ids):
                box = boxes[i]
                x1, y1, x2, y2 = map(int, box)
                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                
                detections.append({
                    'id': track_id,
                    'box': [x1, y1, x2, y2],
                    'center': center
                })
        
        return detections

