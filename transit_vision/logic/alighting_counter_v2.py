import cv2
import numpy as np

def check_door_entry(mask, door_mask, threshold=0.5):
    if door_mask is None or mask is None:
        return False
    
    contours, _ = cv2.findContours(door_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    
    door_polygon = max(contours, key=cv2.contourArea)
    mask_points = cv2.findNonZero(mask)
    
    if mask_points is None:
        return False
    
    inside_count = sum(1 for point in mask_points 
                      if cv2.pointPolygonTest(door_polygon, tuple(point[0].astype(float)), False) >= 0)
    
    ratio = inside_count / len(mask_points)
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

def filter_alighting_passengers(tracks, door_mask, threshold=0.5):
    valid = {}
    
    for track_id, person in tracks.items():
        if not is_reliable_track(person.frames):
            continue
        
        status = []
        for mask in person.masks:
            entered = check_door_entry(mask, door_mask, threshold)
            status.append(1 if entered else 0)
        
        trigger_idx = check_alighting_action(status)
        if trigger_idx >= 0:
            person.trigger_frame = person.frames[trigger_idx]
            valid[track_id] = person
    
    return valid
