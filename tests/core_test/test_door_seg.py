import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transit_vision.utils import read_first_frame, DeviceConfig
from transit_vision.core.detection import DoorSegmentor
import cv2
import numpy as np

VIDEO_UP = "/mnt/mydisk/My_project/bus_down/reid_mark/od_1021/36路/8-6163/2025-10-20-08-41_8-6163_杨家门_up.mp4"
VIDEO_DOWN = "/mnt/mydisk/My_project/bus_down/reid_mark/od_1021/36路/8-6161/2025-10-20-07-01_8-6161_杨家门_down.mp4"
MODEL_PATH = "/mnt/mydisk/My_project/bus_down/front_door.pt"
DEVICE_CONFIG = str(Path(__file__).parent.parent.parent / "configs" / "device_debug.yaml")
OUTPUT_DIR = Path(__file__).parent / "output"

def test_door_detect():
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print("=== DoorSegmentor.detect Test (down) ===")
    
    device_cfg = DeviceConfig(DEVICE_CONFIG)
    detector = DoorSegmentor(MODEL_PATH, device_cfg)
    
    frame = read_first_frame(VIDEO_DOWN)
    print(f"Frame shape: {frame.shape}")
    
    door = detector.detect(frame)
    
    if door is None:
        print("✗ No door detected")
        return
    
    print(f"✓ Door detected")
    print(f"  Bbox: {door.bbox}")
    print(f"  Area: {door.area}")
    print(f"  Center: {door.center}")
    
    result = frame.copy()
    mask_color = np.zeros_like(frame)
    mask_color[door.mask > 0] = [0, 0, 255]
    result = cv2.addWeighted(result, 0.7, mask_color, 0.3, 0)
    
    if door.bbox:
        x1, y1, x2, y2 = door.bbox
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 3)
    
    out_path = OUTPUT_DIR / "door_detect.jpg"
    cv2.imwrite(str(out_path), result)
    print(f"\nOutput: {out_path}")

def test_door_detect_angle():
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print("\n=== DoorSegmentor.detect_with_angle Test (up) ===")
    
    device_cfg = DeviceConfig(DEVICE_CONFIG)
    detector = DoorSegmentor(MODEL_PATH, device_cfg)
    
    frame = read_first_frame(VIDEO_UP)
    print(f"Frame shape: {frame.shape}")
    
    door = detector.detect_with_angle(frame)
    
    if door is None:
        print("✗ No door detected")
        return
    
    print(f"✓ Door detected")
    print(f"  Angle: {door.angle}°")
    print(f"  Bbox (rotated): {door.bbox}")
    print(f"  Area: {door.area}")
    
    result = np.zeros_like(door.mask)
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    mask_color = np.zeros((door.mask.shape[0], door.mask.shape[1], 3), dtype=np.uint8)
    mask_color[door.mask > 0] = [0, 0, 255]
    result = cv2.addWeighted(result, 0.7, mask_color, 0.3, 0)
    
    if door.bbox:
        x1, y1, x2, y2 = door.bbox
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 3)
    
    out_path = OUTPUT_DIR / "door_angle.jpg"
    cv2.imwrite(str(out_path), result)
    print(f"\nOutput: {out_path}")

if __name__ == "__main__":
    test_door_detect()
    test_door_detect_angle()

