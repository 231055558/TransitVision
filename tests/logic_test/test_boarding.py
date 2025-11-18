import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transit_vision.utils import VideoReader, read_first_frame, DeviceConfig, rotate_frame
from transit_vision.core.detection import PersonSegTracker, DoorSegmentor
from transit_vision.logic import preprocess_front_door, filter_boarding_passengers
import cv2
import numpy as np

VIDEO_PATH = "/mnt/mydisk/My_project/bus_down/reid_mark/od_1021/36路/8-6163/2025-10-20-08-41_8-6163_杨家门_up.mp4"
PERSON_MODEL = "/mnt/mydisk/My_project/bus_down/yolo11x-seg.pt"
DOOR_MODEL = "/mnt/mydisk/My_project/bus_down/front_door.pt"
TRACKER_CONFIG = str(Path(__file__).parent.parent.parent / "configs" / "botsort_seg.yaml")
DEVICE_CONFIG = str(Path(__file__).parent.parent.parent / "configs" / "device_debug.yaml")
OUTPUT_DIR = Path(__file__).parent / "output"

def test_boarding_logic():
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print("=== Boarding Logic Test ===")
    
    device_cfg = DeviceConfig(DEVICE_CONFIG)
    
    # 1. 检测门并预处理
    print("\n1. Door detection & preprocessing...")
    door_seg = DoorSegmentor(DOOR_MODEL, device_cfg)
    first_frame = read_first_frame(VIDEO_PATH)
    door = door_seg.detect(first_frame, conf=0.3)
    
    if door is None:
        print("✗ No door detected")
        return
    
    angle, door_bbox = preprocess_front_door(door, first_frame)
    if angle is None:
        print("✗ Preprocessing failed")
        return
    
    print(f"✓ Angle: {angle}°, Bbox: {door_bbox}")
    
    # 2. 追踪人员(旋转视频)
    print("\n2. Person tracking (rotated video)...")
    person_tracker = PersonSegTracker(PERSON_MODEL, TRACKER_CONFIG, device_cfg)
    
    rotation_angle = -angle
    all_tracks = {}
    frame_idx = 0
    
    with VideoReader(VIDEO_PATH) as reader:
        for frame in reader:
            rotated = rotate_frame(frame, rotation_angle)
            detections = person_tracker.track(rotated)
            
            for det in detections:
                tid = det['id']
                if tid not in all_tracks:
                    all_tracks[tid] = type('Person', (), {
                        'id': tid, 'frames': [], 'boxes': [], 'masks': [], 'confs': []
                    })()
                
                p = all_tracks[tid]
                p.frames.append(frame_idx)
                p.boxes.append(det['box'])
                p.masks.append(det['mask'])
                p.confs.append(det['conf'])
            
            frame_idx += 1
            if frame_idx % 50 == 0:
                print(f"  Processed {frame_idx} frames")
    
    print(f"✓ Total tracks: {len(all_tracks)}")
    
    # 3. 上客判定
    print("\n3. Boarding filtering...")
    boarding = filter_boarding_passengers(all_tracks, door_bbox)
    
    print(f"✓ Boarding passengers: {len(boarding)}")
    for tid, person in sorted(boarding.items()):
        print(f"  ID {tid}: {len(person)} frames")
    
    # 4. 可视化(带帧定格)
    print("\n4. Visualization...")
    
    # 找出所有上车判定时刻(轨迹最后一帧)
    trigger_frames = {person.frames[-1]: tid for tid, person in boarding.items()}
    
    with VideoReader(VIDEO_PATH) as reader:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_path = str(OUTPUT_DIR / "boarding_result.mp4")
        
        first_rot = rotate_frame(next(iter(reader)), rotation_angle)
        h, w = first_rot.shape[:2]
        out = cv2.VideoWriter(out_path, fourcc, reader.fps, (w, h))
        
        reader.seek(0)
        frame_idx = 0
        
        for frame in reader:
            rotated = rotate_frame(frame, rotation_angle)
            
            # 绘制门框
            x1, y1, x2, y2 = map(int, door_bbox)
            cv2.rectangle(rotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # 绘制检测结果
            for tid, person in boarding.items():
                if frame_idx in person.frames:
                    idx = person.frames.index(frame_idx)
                    box = person.boxes[idx]
                    mask = person.masks[idx]
                    
                    color = (0, 255, 0)
                    
                    if mask is not None:
                        mask_color = np.zeros_like(rotated)
                        mask_color[mask > 0] = color
                        rotated = cv2.addWeighted(rotated, 1.0, mask_color, 0.4, 0)
                    
                    bx1, by1, bx2, by2 = map(int, box)
                    cv2.rectangle(rotated, (bx1, by1), (bx2, by2), color, 2)
                    cv2.putText(rotated, f"BOARD:{tid}", (bx1, by1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            out.write(rotated)
            
            # 判定时刻定格0.5s
            if frame_idx in trigger_frames:
                freeze_frames = int(reader.fps * 0.5)
                for _ in range(freeze_frames):
                    out.write(rotated)
            
            frame_idx += 1
        
        out.release()
    
    print(f"\nOutput: {out_path}")

if __name__ == "__main__":
    test_boarding_logic()

