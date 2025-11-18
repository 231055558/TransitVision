import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transit_vision.utils import VideoReader, read_first_frame
import cv2

VIDEO_PATH = "/mnt/mydisk/My_project/bus_down/reid_mark/od_1021/36路/8-6163/2025-10-20-08-41_8-6163_杨家门_up.mp4"
OUTPUT_DIR = Path(__file__).parent / "output"

def test_video_reader():
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print("=== VideoReader Test ===")
    
    with VideoReader(VIDEO_PATH) as reader:
        print(f"FPS: {reader.fps}")
        print(f"Size: {reader.width}x{reader.height}")
        print(f"Frames: {reader.frame_count}")
        
        ret, frame = reader.read()
        if ret:
            cv2.imwrite(str(OUTPUT_DIR / "frame_0.jpg"), frame)
            print("✓ frame_0 saved")
        
        reader.seek(100)
        ret, frame = reader.read()
        if ret:
            cv2.imwrite(str(OUTPUT_DIR / "frame_100.jpg"), frame)
            print("✓ frame_100 saved")
    
    print("\n=== read_first_frame Test ===")
    
    frame = read_first_frame(VIDEO_PATH)
    cv2.imwrite(str(OUTPUT_DIR / "first_frame.jpg"), frame)
    print("✓ first_frame saved")
    print(f"Shape: {frame.shape}")
    
    print(f"\nOutput: {OUTPUT_DIR}")

if __name__ == "__main__":
    test_video_reader()

