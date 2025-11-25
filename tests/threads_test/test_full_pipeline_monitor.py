"""
完整流程系统监控测试
监控多线程完整流程的内存占用和CPU使用情况
检测内存泄露和溢出风险
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import time
import threading
import psutil
import os
from collections import defaultdict

from transit_vision.utils import VideoReader, read_first_frame, DeviceConfig
from transit_vision.core.detection import PersonSegTracker, DoorSegmentor
from transit_vision.logic import preprocess_rear_door, filter_alighting_passengers

# 配置
DOOR_VIDEO = "/mnt/mydisk/My_project/TransitVision/data/od_1021/316路/8-6177/2025-10-20-08-49_8-6177_半道红_down.mp4"
TEST_VIDEO = "/mnt/mydisk/My_project/TransitVision/data/od_1021/316路/8-6177/2025-10-20-09-33_8-6177_康宁街华西路口_down.mp4"
PERSON_MODEL = "/mnt/mydisk/My_project/bus_down/yolo11x-seg.pt"
DOOR_MODEL = "/mnt/mydisk/My_project/bus_down/front_door.pt"
TRACKER_CONFIG = str(Path(__file__).parent.parent.parent / "configs" / "botsort_seg.yaml")
DEVICE_CONFIG = str(Path(__file__).parent.parent.parent / "configs" / "device_debug.yaml")

NUM_STREAMS = 8

class SystemMonitor:
    def __init__(self, interval=0.5):
        self.interval = interval
        self.running = False
        self.process = psutil.Process(os.getpid())
        self.memory_samples = []
        self.cpu_samples = []
        self.thread = None
        
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()
        
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
    
    def _monitor(self):
        while self.running:
            mem_mb = self.process.memory_info().rss / 1024 / 1024
            cpu_percent = self.process.cpu_percent(interval=None)
            
            self.memory_samples.append(mem_mb)
            self.cpu_samples.append(cpu_percent)
            
            time.sleep(self.interval)
    
    def get_stats(self):
        if not self.memory_samples:
            return None
        
        return {
            'memory': {
                'min': min(self.memory_samples),
                'max': max(self.memory_samples),
                'avg': sum(self.memory_samples) / len(self.memory_samples),
                'samples': len(self.memory_samples)
            },
            'cpu': {
                'min': min(self.cpu_samples),
                'max': max(self.cpu_samples),
                'avg': sum(self.cpu_samples) / len(self.cpu_samples),
                'samples': len(self.cpu_samples)
            }
        }

class FullPipelineWorker:
    def __init__(self, door_mask, person_model, tracker_config, device_config):
        self.door_mask = door_mask
        self.device_cfg = DeviceConfig(device_config)
        self.tracker = PersonSegTracker(person_model, tracker_config, self.device_cfg)
        self.lock = threading.Lock()
        self.results = {}
        
    def process_video(self, stream_id, video_path):
        """完整流程: 读取 -> 检测 -> 逻辑分析"""
        # 1. 追踪
        all_tracks = {}
        frame_idx = 0
        
        with VideoReader(video_path) as reader:
            for frame in reader:
                with self.lock:
                    detections = self.tracker.track(frame)
                
                for det in detections:
                    tid = det['id']
                    if tid not in all_tracks:
                        from transit_vision.data_structures import Person
                        all_tracks[tid] = Person(tid)
                    
                    all_tracks[tid].add_detection(
                        frame_idx,
                        det['box'],
                        det['polygon'],
                        det['conf']
                    )
                
                frame_idx += 1
        
        # 2. 逻辑分析
        alighting = filter_alighting_passengers(
            all_tracks, self.door_mask,
            threshold=0.5, grace_period=6
        )
        
        self.results[stream_id] = {
            'total_tracks': len(all_tracks),
            'alighting': len(alighting),
            'frames': frame_idx
        }
        
        print(f"Stream {stream_id}: {frame_idx}帧, {len(all_tracks)}人, {len(alighting)}下车")

def test_full_pipeline():
    print("=" * 70)
    print("完整流程系统监控测试")
    print("=" * 70)
    print(f"视频流数量: {NUM_STREAMS}")
    print(f"监控间隔: 0.5s")
    
    # 初始化
    print("\n初始化...")
    device_cfg = DeviceConfig(DEVICE_CONFIG)
    
    # 获取门掩码
    print("获取门掩码...")
    door_seg = DoorSegmentor(DOOR_MODEL, device_cfg)
    door_frame = read_first_frame(DOOR_VIDEO)
    door = door_seg.detect(door_frame)
    door_mask = preprocess_rear_door(door)
    
    # 创建工作器
    worker = FullPipelineWorker(
        door_mask, PERSON_MODEL, TRACKER_CONFIG, DEVICE_CONFIG
    )
    
    # 启动监控
    monitor = SystemMonitor(interval=0.5)
    monitor.start()
    
    print(f"\n开始处理 {NUM_STREAMS} 个视频流...")
    start_time = time.time()
    
    # 启动处理线程
    threads = []
    for i in range(NUM_STREAMS):
        t = threading.Thread(
            target=worker.process_video,
            args=(i, TEST_VIDEO)
        )
        t.start()
        threads.append(t)
    
    # 等待完成
    for t in threads:
        t.join()
    
    elapsed = time.time() - start_time
    
    # 停止监控
    monitor.stop()
    stats = monitor.get_stats()
    
    # 统计结果
    total_frames = sum(r['frames'] for r in worker.results.values())
    total_tracks = sum(r['total_tracks'] for r in worker.results.values())
    total_alighting = sum(r['alighting'] for r in worker.results.values())
    
    print(f"\n{'='*70}")
    print("处理结果")
    print(f"{'='*70}")
    print(f"总耗时: {elapsed:.2f}s")
    print(f"总帧数: {total_frames}")
    print(f"总追踪数: {total_tracks}")
    print(f"总下车数: {total_alighting}")
    print(f"平均FPS: {total_frames / elapsed:.2f}")
    
    print(f"\n{'='*70}")
    print("系统资源监控")
    print(f"{'='*70}")
    
    if stats:
        print(f"\n内存占用 (MB):")
        print(f"  最小: {stats['memory']['min']:.1f}")
        print(f"  最大: {stats['memory']['max']:.1f}")
        print(f"  平均: {stats['memory']['avg']:.1f}")
        print(f"  增量: {stats['memory']['max'] - stats['memory']['min']:.1f}")
        
        print(f"\nCPU使用率 (%):")
        print(f"  最小: {stats['cpu']['min']:.1f}")
        print(f"  最大: {stats['cpu']['max']:.1f}")
        print(f"  平均: {stats['cpu']['avg']:.1f}")
        
        print(f"\n采样次数: {stats['memory']['samples']}")
        
        # 内存泄露检测
        mem_growth = stats['memory']['max'] - stats['memory']['min']
        if mem_growth > 1000:
            print(f"\n⚠ 警告: 内存增长 {mem_growth:.1f}MB，可能存在内存泄露")
        else:
            print(f"\n✓ 内存增长 {mem_growth:.1f}MB，在正常范围")
    
    # CPU核心使用情况
    cpu_count = psutil.cpu_count()
    cpu_percent_per_core = psutil.cpu_percent(interval=1, percpu=True)
    
    print(f"\nCPU核心数: {cpu_count}")
    print(f"各核心使用率: {[f'{p:.1f}%' for p in cpu_percent_per_core]}")
    
    print(f"\n{'='*70}")

if __name__ == "__main__":
    test_full_pipeline()

