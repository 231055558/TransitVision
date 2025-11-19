import cv2
import numpy as np

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
