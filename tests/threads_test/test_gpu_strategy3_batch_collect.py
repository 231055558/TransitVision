"""
GPU推理策略3: 单线程批处理
先用多线程读取所有视频帧，然后单线程批量推理
优点: 批处理效率高，GPU利用率高
缺点: 需要缓存所有帧，内存占用大
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import time
import threading
import queue

from transit_vision.utils import VideoReader, DeviceConfig
from transit_vision.core.detection import PersonSegTracker

# 配置
TEST_VIDEO = "/mnt/mydisk/My_project/TransitVision/data/od_1021/316路/8-6177/2025-10-20-09-33_8-6177_康宁街华西路口_down.mp4"
PERSON_MODEL = "/mnt/mydisk/My_project/bus_down/yolo11x-seg.pt"
TRACKER_CONFIG = str(Path(__file__).parent.parent.parent / "configs" / "botsort_seg.yaml")
DEVICE_CONFIG = str(Path(__file__).parent.parent.parent / "configs" / "device_debug.yaml")

NUM_STREAMS = 16
BATCH_SIZE = 16

def read_video_frames(stream_id, video_path, frame_queue):
    """读取视频帧到队列"""
    with VideoReader(video_path) as reader:
        for frame_idx, frame in enumerate(reader):
            frame_queue.put({
                'stream_id': stream_id,
                'frame_idx': frame_idx,
                'frame': frame
            })
    print(f"Stream {stream_id}: 读取完成")

def test_strategy3():
    print("=" * 70)
    print("GPU推理策略3: 单线程批处理")
    print("=" * 70)
    print(f"视频流数量: {NUM_STREAMS}")
    print(f"批处理大小: {BATCH_SIZE}")
    
    # 阶段1: 多线程读取视频帧
    print("\n阶段1: 读取视频帧...")
    frame_queue = queue.Queue(maxsize=1000)
    threads = []
    
    read_start = time.time()
    for i in range(NUM_STREAMS):
        t = threading.Thread(
            target=read_video_frames,
            args=(i, TEST_VIDEO, frame_queue)
        )
        t.start()
        threads.append(t)
    
    for t in threads:
        t.join()
    
    read_time = time.time() - read_start
    total_frames = frame_queue.qsize()
    print(f"读取完成: {total_frames} 帧, 耗时 {read_time:.2f}s")
    
    # 阶段2: 单线程批量推理
    print("\n阶段2: GPU批量推理...")
    device_cfg = DeviceConfig(DEVICE_CONFIG)
    tracker = PersonSegTracker(PERSON_MODEL, TRACKER_CONFIG, device_cfg)
    
    results = []
    batch_frames = []
    batch_meta = []
    
    infer_start = time.time()
    
    while not frame_queue.empty():
        try:
            item = frame_queue.get_nowait()
            batch_frames.append(item['frame'])
            batch_meta.append((item['stream_id'], item['frame_idx']))
            
            if len(batch_frames) >= BATCH_SIZE:
                # 批量推理
                for frame in batch_frames:
                    det = tracker.track(frame)
                    results.append(det)
                
                batch_frames = []
                batch_meta = []
        except queue.Empty:
            break
    
    # 处理剩余帧
    if batch_frames:
        for frame in batch_frames:
            det = tracker.track(frame)
            results.append(det)
    
    infer_time = time.time() - infer_start
    
    print(f"\n{'='*70}")
    print(f"读取耗时: {read_time:.2f}s")
    print(f"推理耗时: {infer_time:.2f}s")
    print(f"总耗时: {read_time + infer_time:.2f}s")
    print(f"总帧数: {total_frames}")
    print(f"推理FPS: {total_frames / infer_time:.2f}")
    print(f"总体FPS: {total_frames / (read_time + infer_time):.2f}")
    print(f"{'='*70}")

if __name__ == "__main__":
    test_strategy3()

