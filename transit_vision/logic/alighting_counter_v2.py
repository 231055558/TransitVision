import cv2
import numpy as np
from collections import deque

def bbox_overlaps_polygon(box, polygon_points):
    if polygon_points is None or len(polygon_points) < 3:
        return False
    
    x1, y1, x2, y2 = box
    polygon_np = np.array(polygon_points, np.int32)
    poly_x_min, poly_y_min = polygon_np.min(axis=0)
    poly_x_max, poly_y_max = polygon_np.max(axis=0)
    
    if x2 < poly_x_min or x1 > poly_x_max or y2 < poly_y_min or y1 > poly_y_max:
        return False
    
    return True

def check_door_entry(mask, box, polygon_points, threshold=0.5):
    if polygon_points is None or len(polygon_points) < 3:
        return False
    
    if not bbox_overlaps_polygon(box, polygon_points):
        return False
    
    mask_points = cv2.findNonZero(mask)
    if mask_points is None:
        return False
    
    x1, y1, x2, y2 = map(int, box)
    upper_4_5_y_limit = y1 + (y2 - y1)
    
    upper_points = [p[0] for p in mask_points if p[0][1] <= upper_4_5_y_limit]
    
    if not upper_points:
        return False
    
    polygon_np = np.array(polygon_points, np.int32)
    inside_count = sum(1 for point in upper_points 
                      if cv2.pointPolygonTest(polygon_np, (float(point[0]), float(point[1])), False) >= 0)
    
    ratio = inside_count / len(upper_points)
    return ratio >= threshold

def check_box_overlap_with_mask(body_box, door_mask):
    if body_box is None or door_mask is None:
        return False
    
    h, w = door_mask.shape[:2]
    bx1, by1, bx2, by2 = map(int, body_box)
    bx1, by1 = max(0, bx1), max(0, by1)
    bx2, by2 = min(w - 1, bx2), min(h - 1, by2)
    
    if bx1 >= bx2 or by1 >= by2:
        return False
    
    door_mask_in_box = door_mask[by1:by2, bx1:bx2]
    return np.any(door_mask_in_box > 0)

def filter_alighting_passengers(tracks, door_mask, threshold=0.5, grace_period=6):
    # 从门掩码提取多边形点
    contours, _ = cv2.findContours(door_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {}
    
    door_polygon = max(contours, key=cv2.contourArea)
    polygon_points = door_polygon.squeeze().tolist()
    
    # 追踪状态
    track_state = {}
    valid = {}
    
    # 收集所有帧
    all_frames = set()
    for person in tracks.values():
        all_frames.update(person.frames)
    
    if not all_frames:
        return {}
    
    all_frames = sorted(all_frames)
    
    # 逐帧处理
    for frame_idx in all_frames:
        # 更新当前帧出现的ID
        current_ids = set()
        
        for track_id, person in tracks.items():
            if frame_idx in person.frames:
                current_ids.add(track_id)
                
                if track_id not in track_state:
                    track_state[track_id] = {
                        'has_intent': False,
                        'has_counted': False,
                        'last_seen_frame': frame_idx,
                        'last_box': None,
                        'inside_history': []
                    }
                
                state = track_state[track_id]
                state['last_seen_frame'] = frame_idx
                
                idx = person.frames.index(frame_idx)
                box = person.boxes[idx]
                mask = person.masks[idx]
                state['last_box'] = box
                
                # 检查门内状态
                if not state['has_intent'] and mask is not None:
                    is_in = check_door_entry(mask, box, polygon_points, threshold)
                    state['inside_history'].append(1 if is_in else 0)
                    
                    if len(state['inside_history']) > 20:
                        state['inside_history'].pop(0)
                    
                    # 检测0→1转变
                    if len(state['inside_history']) >= 2 and \
                       state['inside_history'][-2] == 0 and \
                       state['inside_history'][-1] == 1:
                        state['has_intent'] = True
        
        # 检查消失的ID
        for track_id, state in track_state.items():
            if state['has_counted'] or not state['has_intent']:
                continue
            
            if track_id not in current_ids:
                disappeared = frame_idx - state['last_seen_frame']
                
                if disappeared >= grace_period:
                    if check_box_overlap_with_mask(state['last_box'], door_mask):
                        person = tracks[track_id]
                        
                        # 检查可靠性
                        frames = person.frames
                        if len(frames) < 5:
                            continue
                        
                        count = 1
                        reliable = False
                        for i in range(1, len(frames)):
                            if frames[i] - frames[i-1] <= 2:
                                count += 1
                                if count >= 5:
                                    reliable = True
                                    break
                            else:
                                count = 1
                        
                        if reliable:
                            person.trigger_frame = state['last_seen_frame']
                            valid[track_id] = person
                            state['has_counted'] = True
    
    return valid
