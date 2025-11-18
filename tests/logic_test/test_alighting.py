import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transit_vision.utils import VideoReader, read_first_frame, DeviceConfig
from transit_vision.core.detection import PersonSegTracker, DoorSegmentor
from transit_vision.logic import preprocess_rear_door, filter_alighting_passengers
import cv2
import numpy as np

VIDEO_PATH = "/mnt/mydisk/My_project/bus_down/reid_mark/od_1021/36路/8-6161/2025-10-20-07-01_8-6161_杨家门_down.mp4"
PERSON_MODEL = "/mnt/mydisk/My_project/bus_down/yolo11x-seg.pt"
DOOR_MODEL = "/mnt/mydisk/My_project/bus_down/front_door.pt"
TRACKER_CONFIG = str(Path(__file__).parent.parent.parent / "configs" / "botsort_seg.yaml")
DEVICE_CONFIG = str(Path(__file__).parent.parent.parent / "configs" / "device_debug.yaml")
OUTPUT_DIR = Path(__file__).parent / "output"

def test_alighting_logic():
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print("=== Alighting Logic Test ===")
    
    device_cfg = DeviceConfig(DEVICE_CONFIG)
    
    # 1. 检测门并预处理
    print("\n1. Door detection & preprocessing...")
    door_seg = DoorSegmentor(DOOR_MODEL, device_cfg)
    first_frame = read_first_frame(VIDEO_PATH)
    door = door_seg.detect(first_frame)
    
    if door is None:
        print("✗ No door detected")
        return
    
    door_mask = preprocess_rear_door(door)
    print(f"✓ Door mask area: {np.sum(door_mask > 0)}")
    
    # 2. 追踪人员
    print("\n2. Person tracking...")
    person_tracker = PersonSegTracker(PERSON_MODEL, TRACKER_CONFIG, device_cfg)
    
    all_tracks = {}
    frame_idx = 0
    
    with VideoReader(VIDEO_PATH) as reader:
        for frame in reader:
            detections = person_tracker.track(frame)
            
            for det in detections:
                tid = det['id']
                if tid not in all_tracks:
                    all_tracks[tid] = type('Person', (), {
                        'id': tid, 'frames': [], 'boxes': [], 'masks': [], 'confs': [],
                        'trigger_frame': None
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
    
    # 3. 下客判定
    print("\n3. Alighting filtering...")
    alighting = filter_alighting_passengers(all_tracks, door_mask)
    
    print(f"✓ Alighting passengers: {len(alighting)}")
    for tid, person in sorted(alighting.items()):
        print(f"  ID {tid}: {len(person)} frames, trigger={person.trigger_frame}")
    
    # 4. 可视化
    print("\n4. Visualization...")
    with VideoReader(VIDEO_PATH) as reader:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_path = str(OUTPUT_DIR / "alighting_result.mp4")
        out = cv2.VideoWriter(out_path, fourcc, reader.fps, (reader.width, reader.height))
        
        frame_idx = 0
        
        for frame in reader:
            # 绘制门掩码
            mask_color = np.zeros_like(frame)
            mask_color[door_mask > 0] = [0, 255, 255]
            frame = cv2.addWeighted(frame, 0.85, mask_color, 0.15, 0)
            
            # 绘制检测结果
            for tid, person in alighting.items():
                if frame_idx in person.frames:
                    idx = person.frames.index(frame_idx)
                    box = person.boxes[idx]
                    mask = person.masks[idx]
                    
                    color = (0, 255, 0)
                    
                    if mask is not None:
                        mask_color = np.zeros_like(frame)
                        mask_color[mask > 0] = color
                        frame = cv2.addWeighted(frame, 1.0, mask_color, 0.5, 0)
                    
                    bx1, by1, bx2, by2 = map(int, box)
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
                    cv2.putText(frame, f"ALIGHT:{tid}", (bx1, by1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            out.write(frame)
            frame_idx += 1
        
        out.release()
    
    print(f"\nOutput: {out_path}")

if __name__ == "__main__":
    test_alighting_logic()

