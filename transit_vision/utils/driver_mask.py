import cv2
import numpy as np
from collections import defaultdict
from .image_ops import rotate_frame

def extract_driver_mask(video_reader, tracker, angle=0):
    # 收集所有ID的掩码数据
    all_tracks = defaultdict(lambda: {'frames': [], 'masks': []})
    frame_idx = 0
    
    first_frame = None
    h, w = None, None
    
    for frame in video_reader:
        if first_frame is None:
            first_frame = frame
            rotated = rotate_frame(frame, angle)
            h, w = rotated.shape[:2]
        
        rotated = rotate_frame(frame, angle)
        detections = tracker.track(rotated)
        
        for det in detections:
            track_id = det['id']
            if det['mask'] is not None:
                all_tracks[track_id]['frames'].append(frame_idx)
                all_tracks[track_id]['masks'].append(det['mask'])
        
        frame_idx += 1
        
        # 提前退出：找到第一个满足连续7帧的ID
        for tid in sorted(all_tracks.keys()):
            if _is_reliable_track(all_tracks[tid]['frames']):
                if len(all_tracks[tid]['frames']) >= 7:
                    return _calc_driver_bbox(all_tracks[tid]['masks'], h, w)
    
    return None

def _is_reliable_track(frames, min_frames=7, max_gap=2):
    if len(frames) < min_frames:
        return False
    
    count = 1
    for i in range(1, len(frames)):
        if frames[i] - frames[i-1] <= max_gap + 1:
            count += 1
            if count >= min_frames:
                return True
        else:
            count = 1
    
    return False

def _calc_driver_bbox(masks, h, w):
    # 选择中间5帧(从7帧中掐头去尾)
    total = len(masks)
    if total < 7:
        return None
    
    indices = np.linspace(0, total-1, 7, dtype=int).tolist()
    selected_indices = indices[1:-1]
    
    # 合并掩码
    combined = np.zeros((h, w), dtype=np.uint8)
    for idx in selected_indices:
        mask_binary = (masks[idx] > 0.5).astype(np.uint8) * 255
        combined = cv2.bitwise_or(combined, mask_binary)
    
    # 去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    
    # 计算bbox
    ys, xs = np.where(combined > 0)
    if len(xs) == 0:
        return None
    
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

