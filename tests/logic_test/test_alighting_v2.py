import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transit_vision.utils import VideoReader, read_first_frame, DeviceConfig
from transit_vision.core.detection import PersonSegTracker, DoorSegmentor
from transit_vision.logic import preprocess_rear_door
from transit_vision.logic.alighting_counter_v2 import filter_alighting_passengers
import cv2
import numpy as np

DOOR_VIDEO = "/mnt/mydisk/My_project/bus_down/reid_mark/od_1021/36路/8-6161/2025-10-20-07-01_8-6161_杨家门_down.mp4"
TEST_VIDEO = "/mnt/mydisk/My_project/od_identification/bus_data/拥堵视频-1011/8-2-8116-002.mp4"
PERSON_MODEL = "/mnt/mydisk/My_project/bus_down/yolo11x-seg.pt"
DOOR_MODEL = "/mnt/mydisk/My_project/bus_down/front_door.pt"
TRACKER_CONFIG = str(Path(__file__).parent.parent.parent / "configs" / "botsort_seg.yaml")
DEVICE_CONFIG = str(Path(__file__).parent.parent.parent / "configs" / "device_debug.yaml")
OUTPUT_DIR = Path(__file__).parent / "output"

def test_alighting_v2():
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print("=== V2 Test ===")
    
    device_cfg = DeviceConfig(DEVICE_CONFIG)
    
    # 1. 从门检测视频获取门掩码
    print(f"\n1. Door detection from: {Path(DOOR_VIDEO).name}")
    door_seg = DoorSegmentor(DOOR_MODEL, device_cfg)
    door_frame = read_first_frame(DOOR_VIDEO)
    door = door_seg.detect(door_frame)
    
    if door is None:
        print("✗ No door")
        return
    
    door_mask = preprocess_rear_door(door)
    print(f"✓ Door mask area: {np.sum(door_mask > 0)}")
    
    # 2. 从测试视频进行人员追踪
    print(f"\n2. Person tracking from: {Path(TEST_VIDEO).name}")
    person_tracker = PersonSegTracker(PERSON_MODEL, TRACKER_CONFIG, device_cfg)
    all_tracks = person_tracker.track_video(VideoReader(TEST_VIDEO))
    
    print(f"✓ Total tracks: {len(all_tracks)}")
    
    # 3. 下客判定
    print(f"\n3. Alighting detection...")
    alighting = filter_alighting_passengers(all_tracks, door_mask, threshold=0.5, grace_period=6)
    print(f"✓ Alighting passengers: {len(alighting)}")
    
    for tid, person in sorted(alighting.items()):
        print(f"  ID {tid}: {len(person)} frames")
    
    # 4. 可视化
    print(f"\n4. Generating visualization...")
    trigger_frames = {person.trigger_frame: tid for tid, person in alighting.items() 
                      if person.trigger_frame is not None}
    
    with VideoReader(TEST_VIDEO) as reader:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_path = str(OUTPUT_DIR / "alighting_v2.mp4")
        out = cv2.VideoWriter(out_path, fourcc, reader.fps, (reader.width, reader.height))
        
        frame_idx = 0
        for frame in reader:
            mask_color = np.zeros_like(frame)
            mask_color[door_mask > 0] = [0, 255, 255]
            frame = cv2.addWeighted(frame, 0.85, mask_color, 0.15, 0)
            
            cv2.putText(frame, f"V2 Count: {len(alighting)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
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
                    cv2.putText(frame, f"ID:{tid}", (bx1, by1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            out.write(frame)
            
            if frame_idx in trigger_frames:
                for _ in range(int(reader.fps * 0.5)):
                    out.write(frame)
            
            frame_idx += 1
        
        out.release()
    
    print(f"\n✓ Output: {out_path}")
    print(f"✓ Total alighting count: {len(alighting)}")

if __name__ == "__main__":
    test_alighting_v2()

