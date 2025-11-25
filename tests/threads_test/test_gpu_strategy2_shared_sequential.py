"""
GPU推理策略2: 多线程共享模型 + 顺序推理
所有线程共享同一个模型实例，使用锁保证顺序推理
优点: 显存占用最小，稳定可靠
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import time
import threading

from transit_vision.utils import VideoReader, DeviceConfig
from transit_vision.core.detection import PersonSegTracker

# 配置
TEST_VIDEO = "/mnt/mydisk/My_project/TransitVision/data/od_1021/316路/8-6177/2025-10-20-09-33_8-6177_康宁街华西路口_down.mp4"
PERSON_MODEL = "/mnt/mydisk/My_project/bus_down/yolo11x-seg.pt"
TRACKER_CONFIG = str(Path(__file__).parent.parent.parent / "configs" / "botsort_seg.yaml")
DEVICE_CONFIG = str(Path(__file__).parent.parent.parent / "configs" / "device_debug.yaml")

NUM_STREAMS = 16

class SharedModelInference:
    def __init__(self, model_path, tracker_config, device_config):
        self.device_cfg = DeviceConfig(device_config)
        self.tracker = PersonSegTracker(model_path, tracker_config, self.device_cfg)
        self.lock = threading.Lock()
        self.results = {}
        
    def process_video(self, stream_id, video_path):
        """处理单个视频流"""
        detections = []
        
        with VideoReader(video_path) as reader:
            for frame_idx, frame in enumerate(reader):
                with self.lock:
                    det = self.tracker.track(frame)
                    detections.append((frame_idx, det))
        
        self.results[stream_id] = detections
        print(f"Stream {stream_id}: {len(detections)} frames")

def test_strategy2():
    print("=" * 70)
    print("GPU推理策略2: 多线程共享模型 + 顺序推理")
    print("=" * 70)
    print(f"视频流数量: {NUM_STREAMS}")
    
    inference_engine = SharedModelInference(
        PERSON_MODEL, TRACKER_CONFIG, DEVICE_CONFIG
    )
    
    threads = []
    start_time = time.time()
    
    for i in range(NUM_STREAMS):
        t = threading.Thread(
            target=inference_engine.process_video,
            args=(i, TEST_VIDEO)
        )
        t.start()
        threads.append(t)
    
    for t in threads:
        t.join()
    
    elapsed = time.time() - start_time
    
    # 统计结果
    total_frames = sum(len(dets) for dets in inference_engine.results.values())
    
    print(f"\n{'='*70}")
    print(f"总耗时: {elapsed:.2f}s")
    print(f"总帧数: {total_frames}")
    print(f"平均FPS: {total_frames / elapsed:.2f}")
    print(f"每流FPS: {total_frames / NUM_STREAMS / elapsed:.2f}")
    print(f"{'='*70}")

if __name__ == "__main__":
    test_strategy2()

