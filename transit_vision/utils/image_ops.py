import cv2
import numpy as np

def rotate_frame(frame, angle):
    if angle == 0:
        return frame
    
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    
    return cv2.warpAffine(frame, M, (new_w, new_h), borderValue=(0, 0, 0))

def apply_mask(frame, mask_bbox):
    result = frame.copy()
    x1, y1, x2, y2 = map(int, mask_bbox)
    result[y1:y2, x1:x2] = 0
    return result

def denoise_mask(mask_binary, kernel_size=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    denoised = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel)
    denoised = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
    return denoised

def filter_connected_components(mask_binary, min_ratio=0.1):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_binary, connectivity=8
    )
    
    if num_labels <= 1:
        return mask_binary
    
    areas = stats[1:, cv2.CC_STAT_AREA]
    max_area = areas.max()
    
    new_mask = np.zeros_like(mask_binary)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= max_area * min_ratio:
            new_mask[labels == i] = 255
    
    return new_mask

