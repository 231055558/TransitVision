import cv2
import numpy as np

def calc_door_angle(mask_binary):
    points = cv2.findNonZero(mask_binary)
    if points is None or len(points) == 0:
        return None
    
    def calc_ratio(pts, ang):
        center = pts.mean(axis=0)[0]
        rad = np.radians(ang)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        
        rotated = pts.copy().astype(np.float32)
        for i in range(len(rotated)):
            x, y = rotated[i][0] - center
            rotated[i][0][0] = x * cos_a - y * sin_a + center[0]
            rotated[i][0][1] = x * sin_a + y * cos_a + center[1]
        
        x_min, y_min = rotated.min(axis=0)[0]
        x_max, y_max = rotated.max(axis=0)[0]
        bbox_area = (x_max - x_min) * (y_max - y_min)
        
        return len(pts) / bbox_area if bbox_area > 0 else 0
    
    def check_upright(pts, ang):
        center = pts.mean(axis=0)[0]
        rad = np.radians(ang)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        
        rotated = pts.copy().astype(np.float32)
        for i in range(len(rotated)):
            x, y = rotated[i][0] - center
            rotated[i][0][0] = x * cos_a - y * sin_a + center[0]
            rotated[i][0][1] = x * sin_a + y * cos_a + center[1]
        
        x_min, y_min = rotated.min(axis=0)[0]
        x_max, y_max = rotated.max(axis=0)[0]
        h = y_max - y_min
        third = h / 3
        
        y_coords = rotated[:, 0, 1]
        upper_cnt = np.sum((y_coords >= y_min) & (y_coords <= y_min + third))
        lower_cnt = np.sum((y_coords >= y_max - third) & (y_coords <= y_max))
        
        w = x_max - x_min
        upper_ratio = upper_cnt / (w * third) if w * third > 0 else 0
        lower_ratio = lower_cnt / (w * third) if w * third > 0 else 0
        
        return upper_ratio > lower_ratio
    
    ratios = []
    angles = []
    
    for ang in range(0, 180, 5):
        ratio = calc_ratio(points, ang)
        ratios.append(ratio)
        angles.append(ang)
        
        if len(ratios) >= 5:
            idx = len(ratios) - 3
            if (ratios[idx] > ratios[idx-1] and ratios[idx] > ratios[idx-2] and 
                ratios[idx] > ratios[idx+1] and ratios[idx] > ratios[idx+2]):
                break
    
    best_idx = ratios.index(max(ratios))
    best_angle = angles[best_idx]
    
    if not check_upright(points, best_angle):
        best_angle = (best_angle + 180) % 360
    
    return best_angle

def calc_rotated_bbox(mask_binary, angle, frame_shape):
    h, w = frame_shape[:2]
    center = (w // 2, h // 2)
    rotation_angle = -angle
    
    M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    
    mask_rotated = cv2.warpAffine(mask_binary, M, (new_w, new_h), borderValue=0)
    
    ys, xs = np.where(mask_rotated > 0)
    if len(xs) == 0:
        return None
    
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

