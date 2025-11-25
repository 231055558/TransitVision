"""
GPU推理策略1: 多线程共享模型 + 动态批处理
16个视频流同时推理，每次收集16帧组成batch进行推理
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import time
import threading
import queue
import numpy as np
from collections import defaultdict

from transit_vision.utils import VideoReader, DeviceConfig
from transit_vision.core.detection import PersonSegTracker

# 配置
TEST_VIDEO = "/mnt/mydisk/My_project/TransitVision/data/od_1021/316路/8-6177/2025-10-20-09-33_8-6177_康宁街华西路口_down.mp4"
PERSON_MODEL = "/mnt/mydisk/My_project/bus_down/yolo11x-seg.pt"
TRACKER_CONFIG = str(Path(__file__).parent.parent.parent / "configs" / "botsort_seg.yaml")
DEVICE_CONFIG = str(Path(__file__).parent.parent.parent / "configs" / "device_debug.yaml")

NUM_STREAMS = 16
BATCH_SIZE = 16

class DynamicBatchInference:
    def __init__(self, model_path, tracker_config, device_config, batch_size=16):
        self.device_cfg = DeviceConfig(device_config)
        self.tracker = PersonSegTracker(model_path, tracker_config, self.device_cfg)
        self.batch_size = batch_size
        self.frame_queue = queue.Queue(maxsize=batch_size * 2)
        self.result_queues = {}
        self.lock = threading.Lock()
        self.running = True
        
    def inference_worker(self):
        """推理线程：收集batch后批量推理"""
        while self.running:
            batch_data = []
            
            # 收集batch
            for _ in range(self.batch_size):
                try:
                    item = self.frame_queue.get(timeout=0.1)
                    if item is None:
                        break
                    batch_data.append(item)
                except queue.Empty:
                    break
            
            if not batch_data:
                continue
            
            # 批量推理
            frames = [item['frame'] for item in batch_data]
            
            with self.lock:
                # YOLO track_batch需要逐帧调用track
                results = []
                for frame in frames:
                    det = self.tracker.track(frame)
                    results.append(det)
            
            # 分发结果
            for item, det in zip(batch_data, results):
                stream_id = item['stream_id']
                frame_idx = item['frame_idx']
                self.result_queues[stream_id].put((frame_idx, det))
    
    def stream_worker(self, stream_id, video_path):
        """视频流线程：读取帧并提交到队列"""
        self.result_queues[stream_id] = queue.Queue()
        
        with VideoReader(video_path) as reader:
            for frame_idx, frame in enumerate(reader):
                self.frame_queue.put({
                    'stream_id': stream_id,
                    'frame_idx': frame_idx,
                    'frame': frame
                })
        
        # 标记结束
        self.frame_queue.put(None)

def test_strategy1():
    print("=" * 70)
    print("GPU推理策略1: 多线程共享模型 + 动态批处理")
    print("=" * 70)
    print(f"视频流数量: {NUM_STREAMS}")
    print(f"批处理大小: {BATCH_SIZE}")
    
    inference_engine = DynamicBatchInference(
        PERSON_MODEL, TRACKER_CONFIG, DEVICE_CONFIG, BATCH_SIZE
    )
    
    # 启动推理线程
    inference_thread = threading.Thread(target=inference_engine.inference_worker)
    inference_thread.start()
    
    # 启动视频流线程
    stream_threads = []
    start_time = time.time()
    
    for i in range(NUM_STREAMS):
        t = threading.Thread(
            target=inference_engine.stream_worker,
            args=(i, TEST_VIDEO)
        )
        t.start()
        stream_threads.append(t)
    
    # 等待所有视频流完成
    for t in stream_threads:
        t.join()
    
    inference_engine.running = False
    inference_thread.join()
    
    elapsed = time.time() - start_time
    
    # 统计结果
    total_frames = 0
    for stream_id in range(NUM_STREAMS):
        q = inference_engine.result_queues[stream_id]
        frame_count = q.qsize()
        total_frames += frame_count
    
    print(f"\n{'='*70}")
    print(f"总耗时: {elapsed:.2f}s")
    print(f"总帧数: {total_frames}")
    print(f"平均FPS: {total_frames / elapsed:.2f}")
    print(f"每流FPS: {total_frames / NUM_STREAMS / elapsed:.2f}")
    print(f"{'='*70}")

if __name__ == "__main__":
    test_strategy1()

