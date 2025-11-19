import cv2
import numpy as np
from collections import defaultdict
from ..data_structures import Person

def calc_overlap_ratio(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    
    return intersection / area1 if area1 > 0 else 0.0

def check_boarding_pattern(boxes, door_bbox):
    if len(boxes) < 7:
        return False
    
    overlap_ratios = [calc_overlap_ratio(box, door_bbox) for box in boxes]
    
    in_threshold = 0.85
    out_threshold = 0.40
    
    status = []
    for ratio in overlap_ratios:
        if ratio >= in_threshold:
            status.append(1)
        elif ratio <= out_threshold:
            status.append(0)
        else:
            status.append(status[-1] if status else 0)
    
    first_in = next((i for i, s in enumerate(status) if s == 1), -1)
    if first_in == -1 or first_in > 3:
        return False
    
    last_in = next((i for i in range(len(status)-1, -1, -1) if status[i] == 1), -1)
    
    exit_idx = next((i for i in range(last_in+1, len(status)) if status[i] == 0), -1)
    if exit_idx == -1:
        return False
    
    door_center_x = (door_bbox[0] + door_bbox[2]) / 2
    exit_center_x = (boxes[exit_idx][0] + boxes[exit_idx][2]) / 2
    
    if exit_center_x <= door_center_x:
        return False
    
    # 高度检查: 过滤车窗外经过的行人
    door_height = door_bbox[3] - door_bbox[1]
    if door_height == 0:
        return False
    
    inside_boxes = [boxes[i] for i in range(exit_idx, len(boxes)) 
                   if (boxes[i][0] + boxes[i][2]) / 2 > door_center_x]
    
    if not inside_boxes:
        return False
    
    avg_height = np.mean([box[3] - box[1] for box in inside_boxes])
    
    # 车内平均高度需达到门框高度的48%
    if avg_height < door_height * 0.48:
        return False
    
    return True

def is_reliable_track(frames, min_frames=7, max_gap=2):
    if len(frames) < min_frames:
        return False
    
    count = 1
    for i in range(1, len(frames)):
        gap = frames[i] - frames[i-1]
        if gap <= max_gap + 1:
            count += 1
            if count >= min_frames:
                return True
        else:
            count = 1
    
    return False

def filter_boarding_passengers(tracks, door_bbox):
    valid = {}
    
    for track_id, person in tracks.items():
        if not is_reliable_track(person.frames):
            continue
        
        if check_boarding_pattern(person.boxes, door_bbox):
            valid[track_id] = person
    
    return valid
