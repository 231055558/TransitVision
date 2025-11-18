from ..utils import denoise_mask, filter_connected_components, calc_door_angle, calc_rotated_bbox

def preprocess_front_door(door_seg, frame):
    # 获取门掩码
    mask = door_seg.mask
    
    # 去噪
    denoised = denoise_mask(mask)
    
    # 过滤连通域
    filtered = filter_connected_components(denoised)
    
    # 计算角度
    angle = calc_door_angle(filtered)
    if angle is None:
        return None, None
    
    # 计算旋转后bbox
    bbox = calc_rotated_bbox(filtered, angle, frame.shape)
    if bbox is None:
        return None, None
    
    return angle, bbox

def preprocess_rear_door(door_seg):
    # 获取门掩码
    mask = door_seg.mask
    
    # 去噪
    denoised = denoise_mask(mask)
    
    # 过滤连通域
    filtered = filter_connected_components(denoised)
    
    return filtered

