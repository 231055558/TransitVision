import cv2
import numpy as np

def polygon_to_local_mask(polygon, box):
    """
    将多边形转换为局部掩码 (高效内存方案)
    
    Args:
        polygon: np.ndarray, shape (N, 1, 2) 或 (N, 2), 全局坐标
        box: [x1, y1, x2, y2]
    
    Returns:
        local_mask: np.ndarray, shape (h, w), 局部坐标掩码
    """
    x1, y1, x2, y2 = map(int, box)
    w, h = x2 - x1, y2 - y1
    
    if w <= 0 or h <= 0:
        return None
    
    # 转换为局部坐标
    if polygon.ndim == 3:
        polygon = polygon.squeeze()
    
    local_poly = polygon - np.array([x1, y1])
    
    # 创建局部掩码
    local_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(local_mask, [local_poly.astype(np.int32)], 255)
    
    return local_mask

def check_door_entry_v3(person_polygon, box, door_polygon, threshold=0.5):
    """
    V3版本: 使用矩阵位运算检测进门 (O(1) 复杂度)
    
    Args:
        person_polygon: 人物多边形 (全局坐标)
        box: 人物bbox [x1, y1, x2, y2]
        door_polygon: 门多边形 (全局坐标)
        threshold: 重叠阈值
    
    Returns:
        bool: 是否进入门区域
    """
    if person_polygon is None or door_polygon is None:
        return False
    
    if len(person_polygon) < 3 or len(door_polygon) < 3:
        return False
    
    x1, y1, x2, y2 = map(int, box)
    
    # 快速bbox预检
    door_poly_np = door_polygon.squeeze() if door_polygon.ndim == 3 else door_polygon
    poly_x_min, poly_y_min = door_poly_np.min(axis=0)
    poly_x_max, poly_y_max = door_poly_np.max(axis=0)
    
    if x2 < poly_x_min or x1 > poly_x_max or y2 < poly_y_min or y1 > poly_y_max:
        return False
    
    # 计算上4/5区域
    h = y2 - y1
    upper_h = int(h * 4 / 5)
    
    if upper_h <= 0:
        return False
    
    upper_box = [x1, y1, x2, y1 + upper_h]
    
    # 1. 转换人物多边形到局部掩码 (只处理上4/5区域)
    person_local = polygon_to_local_mask(person_polygon, upper_box)
    if person_local is None:
        return False
    
    # 2. 转换门多边形到局部掩码
    door_local = polygon_to_local_mask(door_polygon, upper_box)
    if door_local is None:
        return False
    
    # 3. 矩阵位运算 (核心优化)
    intersection = cv2.bitwise_and(person_local, door_local)
    overlap_pixels = cv2.countNonZero(intersection)
    person_pixels = cv2.countNonZero(person_local)
    
    if person_pixels == 0:
        return False
    
    ratio = overlap_pixels / person_pixels
    return ratio >= threshold

def check_box_overlap_with_polygon(body_box, door_polygon):
    """
    检查bbox是否与门多边形重叠
    
    Args:
        body_box: [x1, y1, x2, y2]
        door_polygon: 门多边形
    
    Returns:
        bool: 是否重叠
    """
    if body_box is None or door_polygon is None:
        return False
    
    if len(door_polygon) < 3:
        return False
    
    x1, y1, x2, y2 = map(int, body_box)
    
    # 快速bbox预检
    door_poly_np = door_polygon.squeeze() if door_polygon.ndim == 3 else door_polygon
    poly_x_min, poly_y_min = door_poly_np.min(axis=0)
    poly_x_max, poly_y_max = door_poly_np.max(axis=0)
    
    if x2 < poly_x_min or x1 > poly_x_max or y2 < poly_y_min or y1 > poly_y_max:
        return False
    
    # 检查bbox四个角点是否在多边形内
    corners = [
        (x1, y1), (x2, y1),
        (x1, y2), (x2, y2)
    ]
    
    for corner in corners:
        if cv2.pointPolygonTest(door_poly_np.astype(np.float32), corner, False) >= 0:
            return True
    
    return False

def filter_alighting_passengers(tracks, door_mask, threshold=0.5, grace_period=6):
    """
    V3版本: 过滤下车乘客 (多边形输入 + 矩阵位运算)
    
    Args:
        tracks: dict, {track_id: Person}, Person.mask_polygons 存储多边形
        door_mask: 门掩码 (用于提取门多边形)
        threshold: 进门判定阈值
        grace_period: 消失宽限期
    
    Returns:
        dict: 下车乘客 {track_id: Person}
    """
    # 从门掩码提取多边形
    contours, _ = cv2.findContours(door_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {}
    
    door_polygon = max(contours, key=cv2.contourArea)
    
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
                state['last_box'] = box
                
                # 检查门内状态 (使用多边形)
                if not state['has_intent'] and len(person.mask_polygons) > idx:
                    person_polygon = person.mask_polygons[idx]
                    
                    if person_polygon is not None and len(person_polygon) > 0:
                        is_in = check_door_entry_v3(person_polygon, box, door_polygon, threshold)
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
                    if check_box_overlap_with_polygon(state['last_box'], door_polygon):
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

