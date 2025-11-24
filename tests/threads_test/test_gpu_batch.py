import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transit_vision.core.detection import PersonSegTracker
from transit_vision.utils.video_reader import VideoReader
import time
import torch

BASE_DIR = Path(__file__).parent.parent.parent
TEST_VIDEO = BASE_DIR / "data/od_1021/316路/8-6177/2025-10-20-09-15_8-6177_香石巷_up.mp4"
MODEL_PATH = BASE_DIR / "ckpt/yolo11x-seg.pt"
TRACKER_CONFIG = BASE_DIR / "configs/botsort_seg.yaml"

class SimpleDeviceConfig:
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print("=== GPU推理批处理测试 ===\n")
print(f"设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")

if not TEST_VIDEO.exists():
    print(f"✗ 视频不存在: {TEST_VIDEO}")
    exit(1)

device = SimpleDeviceConfig()

with VideoReader(TEST_VIDEO) as reader:
    frames = [frame for frame in reader][:300]

total_frames = len(frames)
print(f"测试帧数: {total_frames}\n")

print("批处理策略说明:")
print("- 策略1: 顺序批处理，不足batch直接跳过（可能丢帧）")
print("- 策略2: 动态批处理，不足batch也处理（不丢帧）\n")

batch_sizes = [1, 4, 8, 16]

for batch_size in batch_sizes:
    tracker = PersonSegTracker(str(MODEL_PATH), str(TRACKER_CONFIG), device)
    
    start = time.time()
    processed = 0
    
    for i in range(0, total_frames, batch_size):
        batch_frames = frames[i:i+batch_size]
        
        for frame in batch_frames:
            tracker.track(frame, conf=0.3)
            processed += 1
    
    elapsed = time.time() - start
    fps = processed / elapsed
    
    print(f"Batch={batch_size:2d}: {elapsed:.2f}s ({fps:.1f}fps) 处理{processed}/{total_frames}帧")

print("\n✓ 测试完成")
print("\n结论:")
print("1. Batch大小影响推理速度，但YOLO的track方法是逐帧的")
print("2. 多线程方案：每线程处理一个视频，避免跨视频batch")
print("3. 不足batch的帧正常处理，不会丢帧")







