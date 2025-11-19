import cv2
import numpy as np
from typing import Dict
from ..data_structures.person import Person
from ..data_structures.door import Door

def bbox_overlaps_mask(box, door_mask):
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2 = min(door_mask.shape[1], x2)
    y2 = min(door_mask.shape[0], y2)
    
    roi = door_mask[y1:y2, x1:x2]
    return np.any(roi > 0)

def check_door_entry(mask, box, door_mask, threshold=0.85):
    if not bbox_overlaps_mask(box, door_mask):
        return False
    
    mask_points = cv2.findNonZero(mask)
    if mask_points is None or len(mask_points) == 0:
        return False
    
    x1, y1, x2, y2 = box
    bbox_height = y2 - y1
    upper_threshold = y1 + bbox_height * 4 / 5
    
    upper_points = [pt[0] for pt in mask_points if pt[0][1] <= upper_threshold]
    
    if len(upper_points) == 0:
        return False
    
    inside_count = sum(1 for x, y in upper_points 
                      if 0 <= y < door_mask.shape[0] and 0 <= x < door_mask.shape[1] 
                      and door_mask[y, x] > 0)
    
    ratio = inside_count / len(upper_points)
    return ratio >= threshold

def is_reliable_track(frames, min_frames=5, max_gap=2):
    if len(frames) < min_frames:
        return False
    
    count = 1
    for i in range(1, len(frames)):
        if frames[i] - frames[i-1] <= max_gap:
            count += 1
            if count >= min_frames:
                return True
        else:
            count = 1
    
    return False

def check_alighting_action(status_list):
    for i in range(1, len(status_list)):
        if status_list[i-1] == 0 and status_list[i] == 1:
            return i
    return -1

def filter_alighting_passengers(tracks, door_mask):
    valid = {}
    
    for track_id, person in tracks.items():
        if not is_reliable_track(person.frames):
            continue
        
        status = []
        for mask, box in zip(person.masks, person.boxes):
            entered = check_door_entry(mask, box, door_mask)
            status.append(1 if entered else 0)
        
        trigger_idx = check_alighting_action(status)
        if trigger_idx >= 0:
            person.trigger_frame = person.frames[trigger_idx]
            valid[track_id] = person
    
    return valid

# 类式API包装器（统一接口）
class AlightingCounterV1:
    def __init__(self, config: dict = None):
        self.threshold = 0.85
        self.min_frames = 5
        self.passenger_count = 0
        if config:
            alighting_config = config.get('alighting_counter', {})
            self.threshold = alighting_config.get('door_entry_threshold', 0.85)
    
    def reset(self):
        self.passenger_count = 0
    
    def update_counts(self, frame_idx: int, persons: Dict[int, Person], door: Door):
        # 将Person对象转换为tracks格式
        tracks = {pid: p for pid, p in persons.items() if not getattr(p, 'has_counted', False)}
        
        if door is None or door.mask is None:
            return
        
        # 使用原逻辑过滤
        valid = filter_alighting_passengers(tracks, door.mask)
        
        # 更新计数
        for pid, person in valid.items():
            if not getattr(person, 'has_counted', False):
                self.passenger_count += 1
                person.has_counted = True
                print(f"[V1-Frame {frame_idx}] Person {person.id} counted as alighting.")
    
    def get_count(self) -> int:
        return self.passenger_count
