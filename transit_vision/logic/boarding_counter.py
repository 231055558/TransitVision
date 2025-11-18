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
    # 至少7帧
    if len(boxes) < 7:
        return False
    
    # 计算与门框重合度
    overlap_ratios = [calc_overlap_ratio(box, door_bbox) for box in boxes]
    
    # 状态判定: 1=门内(>=85%), 0=门外(<=40%)
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
    
    # 找第一个进门帧
    first_in = next((i for i, s in enumerate(status) if s == 1), -1)
    if first_in == -1 or first_in > 3:  # 未进门或前面帧数过多
        return False
    
    # 找最后一个在门内帧
    last_in = next((i for i in range(len(status)-1, -1, -1) if status[i] == 1), -1)
    
    # 检查是否离开门框
    exit_idx = next((i for i in range(last_in+1, len(status)) if status[i] == 0), -1)
    if exit_idx == -1:
        return False
    
    # 检查离开时在门框右侧
    door_center_x = (door_bbox[0] + door_bbox[2]) / 2
    exit_center_x = (boxes[exit_idx][0] + boxes[exit_idx][2]) / 2
    
    return exit_center_x > door_center_x

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
    # tracks: {id: Person}
    valid = {}
    
    for track_id, person in tracks.items():
        # 轨迹可靠性检查
        if not is_reliable_track(person.frames):
            continue
        
        # 上车模式检查
        if check_boarding_pattern(person.boxes, door_bbox):
            valid[track_id] = person
    
    return valid

