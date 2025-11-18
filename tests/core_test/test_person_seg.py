import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transit_vision.utils import VideoReader, DeviceConfig
from transit_vision.core.detection import PersonSegTracker
import cv2
import numpy as np

VIDEO_PATH = "/mnt/mydisk/My_project/bus_down/reid_mark/od_1021/36路/8-6163/2025-10-20-08-41_8-6163_杨家门_up.mp4"
MODEL_PATH = "/mnt/mydisk/My_project/bus_down/yolo11x-seg.pt"
TRACKER_CONFIG = str(Path(__file__).parent.parent.parent / "configs" / "botsort_seg.yaml")
DEVICE_CONFIG = str(Path(__file__).parent.parent.parent / "configs" / "device_debug.yaml")
OUTPUT_DIR = Path(__file__).parent / "output"

def test_person_track():
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print("=== PersonSegTracker Test ===")
    
    device_cfg = DeviceConfig(DEVICE_CONFIG)
    tracker = PersonSegTracker(MODEL_PATH, TRACKER_CONFIG, device_cfg)
    
    with VideoReader(VIDEO_PATH) as reader:
        print(f"Video: {reader.width}x{reader.height}, {reader.frame_count} frames")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_path = str(OUTPUT_DIR / "person_track.mp4")
        out = cv2.VideoWriter(out_path, fourcc, reader.fps, (reader.width, reader.height))
        
        frame_idx = 0
        all_tracks = {}
        
        for frame in reader:
            detections = tracker.track(frame)
            
            for det in detections:
                track_id = det['id']
                if track_id not in all_tracks:
                    all_tracks[track_id] = []
                all_tracks[track_id].append(frame_idx)
                
                box = det['box']
                mask = det['mask']
                conf = det['conf']
                
                if mask is not None:
                    color = (int((track_id * 50) % 255), 
                            int((track_id * 100) % 255),
                            int((track_id * 150) % 255))
                    
                    mask_color = np.zeros_like(frame)
                    mask_color[mask > 0] = color
                    frame = cv2.addWeighted(frame, 1.0, mask_color, 0.4, 0)
                
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID:{track_id} {conf:.2f}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            out.write(frame)
            frame_idx += 1
            
            if frame_idx % 50 == 0:
                print(f"Processed {frame_idx} frames")
        
        out.release()
        
        print(f"\n✓ Total {len(all_tracks)} tracks detected")
        for track_id, frames in sorted(all_tracks.items()):
            print(f"  ID {track_id}: {len(frames)} frames")
        
        print(f"\nOutput: {out_path}")

def test_person_track_video():
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print("\n=== PersonSegTracker.track_video Test ===")
    
    device_cfg = DeviceConfig(DEVICE_CONFIG)
    tracker = PersonSegTracker(MODEL_PATH, TRACKER_CONFIG, device_cfg)
    
    with VideoReader(VIDEO_PATH) as reader:
        tracks = tracker.track_video(reader)
    
    print(f"✓ Total {len(tracks)} tracks")
    for track_id, person in sorted(tracks.items()):
        print(f"  ID {track_id}: {len(person)} frames")

if __name__ == "__main__":
    test_person_track()
    test_person_track_video()

