import cv2
from pathlib import Path
from .frame_selector import select_frame_indices
from .image_ops import rotate_frame

def save_bbox_crops(video_path, track_id, frames, boxes, output_dir, rotation_angle=0):
    if len(frames) < 7:
        return 0
    
    cap = cv2.VideoCapture(str(video_path))
    
    # 选择帧
    selected_frames = select_frame_indices(frames)
    selected_indices = [frames.index(f) for f in selected_frames]
    selected_boxes = [boxes[i] for i in selected_indices]
    
    # 创建输出目录
    id_dir = Path(output_dir) / f"id_{track_id}"
    id_dir.mkdir(parents=True, exist_ok=True)
    
    saved_count = 0
    for frame_idx, box in zip(selected_frames, selected_boxes):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        if rotation_angle != 0:
            frame = rotate_frame(frame, rotation_angle)
        
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        
        crop = frame[y1:y2, x1:x2]
        if crop.size > 0:
            save_path = id_dir / f"frame_{saved_count}.jpg"
            cv2.imwrite(str(save_path), crop)
            saved_count += 1
    
    cap.release()
    return saved_count

