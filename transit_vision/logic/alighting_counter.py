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

def check_door_entry_v3(person_polygon, box, door_mask, threshold=0.5):
    """
    V3版本: 使用矩阵位运算检测进门 (O(1) 复杂度)
    
    Args:
        person_polygon: 人物多边形 (全局坐标)
        box: 人物bbox [x1, y1, x2, y2]
        door_mask: 门掩码 (全局掩码，保持原始精度)
        threshold: 重叠阈值
    
    Returns:
        bool: 是否进入门区域
    """
    if person_polygon is None or door_mask is None:
        return False
    
    if len(person_polygon) < 3:
        return False
    
    x1, y1, x2, y2 = map(int, box)
    
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
    
    # 2. 提取门掩码的局部区域
    door_local = door_mask[y1:y1+upper_h, x1:x2]
    if door_local.size == 0:
        return False
    
    # 确保尺寸一致
    if door_local.shape != person_local.shape:
        return False
    
    # 3. 矩阵位运算 (核心优化)
    intersection = cv2.bitwise_and(person_local, door_local)
    overlap_pixels = cv2.countNonZero(intersection)
    person_pixels = cv2.countNonZero(person_local)
    
    if person_pixels == 0:
        return False
    
    ratio = overlap_pixels / person_pixels
    return ratio >= threshold

def check_box_overlap_with_mask(body_box, door_mask):
    """
    检查bbox是否与门掩码重叠
    
    Args:
        body_box: [x1, y1, x2, y2]
        door_mask: 门掩码
    
    Returns:
        bool: 是否重叠
    """
    if body_box is None or door_mask is None:
        return False
    
    x1, y1, x2, y2 = map(int, body_box)
    
    # 确保坐标在有效范围内
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(door_mask.shape[1], x2)
    y2 = min(door_mask.shape[0], y2)
    
    if x2 <= x1 or y2 <= y1:
        return False
    
    # 检查bbox区域内是否有门掩码
    box_region = door_mask[y1:y2, x1:x2]
    return np.any(box_region > 0)

def get_upper_region_mask(person_polygon, box, frame_shape):
    """
    获取上4/5区域的全局掩码 (用于可视化调试)
    
    Args:
        person_polygon: 人物多边形
        box: [x1, y1, x2, y2]
        frame_shape: (height, width)
    
    Returns:
        mask: 全局掩码
    """
    if person_polygon is None or len(person_polygon) < 3:
        return None
    
    x1, y1, x2, y2 = map(int, box)
    h = y2 - y1
    upper_h = int(h * 4 / 5)
    
    if upper_h <= 0:
        return None
    
    upper_box = [x1, y1, x2, y1 + upper_h]
    
    person_local = polygon_to_local_mask(person_polygon, upper_box)
    if person_local is None:
        return None
    
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    mask[y1:y1+upper_h, x1:x2] = person_local
    
    return mask

def get_overlap_region_mask(person_polygon, box, door_mask, frame_shape):
    """
    获取上4/5区域与门框重叠的全局掩码 (用于可视化调试)
    
    Args:
        person_polygon: 人物多边形
        box: [x1, y1, x2, y2]
        door_mask: 门掩码 (全局掩码)
        frame_shape: (height, width)
    
    Returns:
        mask: 全局掩码
    """
    if person_polygon is None or door_mask is None:
        return None
    
    if len(person_polygon) < 3:
        return None
    
    x1, y1, x2, y2 = map(int, box)
    h = y2 - y1
    upper_h = int(h * 4 / 5)
    
    if upper_h <= 0:
        return None
    
    upper_box = [x1, y1, x2, y1 + upper_h]
    
    person_local = polygon_to_local_mask(person_polygon, upper_box)
    if person_local is None:
        return None
    
    # 提取门掩码的局部区域
    door_local = door_mask[y1:y1+upper_h, x1:x2]
    if door_local.size == 0 or door_local.shape != person_local.shape:
        return None
    
    intersection = cv2.bitwise_and(person_local, door_local)
    
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    mask[y1:y1+upper_h, x1:x2] = intersection
    
    return mask

def filter_alighting_passengers(tracks, door_mask, threshold=0.5, grace_period=6):
    """
    V3版本: 过滤下车乘客 (人物多边形 + 门掩码 + 矩阵位运算)
    
    Args:
        tracks: dict, {track_id: Person}, Person.mask_polygons 存储多边形
        door_mask: 门掩码 (直接使用，保持精度)
        threshold: 进门判定阈值
        grace_period: 消失宽限期
    
    Returns:
        dict: 下车乘客 {track_id: Person}
    """
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
                
                # 检查门内状态 (使用人物多边形 + 门掩码)
                if not state['has_intent'] and len(person.mask_polygons) > idx:
                    person_polygon = person.mask_polygons[idx]
                    
                    if person_polygon is not None and len(person_polygon) > 0:
                        is_in = check_door_entry_v3(person_polygon, box, door_mask, threshold)
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


