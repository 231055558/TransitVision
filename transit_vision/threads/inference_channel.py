"""
推理通道模块
接收输入通道的视频任务，进行GPU推理
"""
import threading
import queue
from pathlib import Path
from typing import Optional
from ..utils import VideoReader, DeviceConfig
from ..core.detection import PersonSegTracker
from ..data_structures import VideoTask

class InferenceChannel:
    """推理通道管理器"""
    def __init__(self, model_path: str, tracker_config: str, device_config: DeviceConfig, 
                 batch_size=128, queue_size=200):
        self.batch_size = batch_size
        self.input_queue = queue.Queue(maxsize=queue_size)
        self.output_queue = queue.Queue(maxsize=queue_size)
        self.tracker = PersonSegTracker(model_path, tracker_config, device_config)
        self.inference_lock = threading.Lock()
        self.workers = []
        self.running = False
        self.stats = {
            'total_inputs': 0,
            'processed': 0,
            'total_frames': 0
        }
        
    def submit_task(self, task: VideoTask):
        """提交推理任务"""
        self.input_queue.put(task)
        self.stats['total_inputs'] += 1
    
    def _inference_worker(self, worker_id: int):
        """推理工作线程"""
        while self.running:
            try:
                task = self.input_queue.get(timeout=0.5)
                if task is None:
                    break
                
                result = self._process_task(task)
                self.output_queue.put(result)
                
                self.stats['processed'] += 1
                self.input_queue.task_done()
                
            except queue.Empty:
                continue
    
    def _process_task(self, task: VideoTask):
        """处理单个视频任务"""
        all_tracks = {}
        frame_count = 0
        
        with VideoReader(task.video_path) as reader:
            frames_batch = []
            frame_indices = []
            
            for frame_idx, frame in enumerate(reader):
                frames_batch.append(frame)
                frame_indices.append(frame_idx)
                frame_count += 1
                
                # 达到batch大小或视频结束时推理
                if len(frames_batch) >= self.batch_size:
                    self._inference_batch(frames_batch, frame_indices, all_tracks)
                    frames_batch = []
                    frame_indices = []
            
            # 处理剩余帧
            if frames_batch:
                self._inference_batch(frames_batch, frame_indices, all_tracks)
        
        self.stats['total_frames'] += frame_count
        
        return {
            'task': task,
            'tracks': all_tracks,
            'frame_count': frame_count
        }
    
    def _inference_batch(self, frames, frame_indices, all_tracks):
        """批量推理"""
        with self.inference_lock:
            for frame_idx, frame in zip(frame_indices, frames):
                detections = self.tracker.track(frame)
                
                for det in detections:
                    track_id = det['id']
                    if track_id not in all_tracks:
                        from ..data_structures import Person
                        all_tracks[track_id] = Person(track_id)
                    
                    all_tracks[track_id].add_detection(
                        frame_idx,
                        det['box'],
                        det['polygon'],
                        det['conf']
                    )
    
    def start(self, num_workers=4):
        """启动推理通道"""
        self.running = True
        for i in range(num_workers):
            worker = threading.Thread(target=self._inference_worker, args=(i,))
            worker.start()
            self.workers.append(worker)
    
    def stop(self):
        """停止推理通道"""
        self.running = False
        
        for _ in range(len(self.workers)):
            self.input_queue.put(None)
        
        for worker in self.workers:
            worker.join()
        
        self.workers.clear()
    
    def get_result(self, timeout=1.0):
        """获取推理结果"""
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_stats(self):
        """获取统计信息"""
        return {
            'total_inputs': self.stats['total_inputs'],
            'processed': self.stats['processed'],
            'total_frames': self.stats['total_frames'],
            'pending': self.input_queue.qsize(),
            'output_queue': self.output_queue.qsize()
        }

class MultiDirectionInferenceChannel:
    """多方向推理通道(up/down分离)"""
    def __init__(self, model_path: str, tracker_config: str, device_config: DeviceConfig,
                 batch_size=128, num_workers=4):
        self.up_channel = InferenceChannel(
            model_path, tracker_config, device_config, batch_size
        )
        self.down_channel = InferenceChannel(
            model_path, tracker_config, device_config, batch_size
        )
        self.num_workers = num_workers
    
    def start(self):
        """启动所有通道"""
        self.up_channel.start(self.num_workers)
        self.down_channel.start(self.num_workers)
    
    def stop(self):
        """停止所有通道"""
        self.up_channel.stop()
        self.down_channel.stop()
    
    def submit_task(self, task: VideoTask):
        """根据方向提交任务"""
        if task.direction == 'up':
            self.up_channel.submit_task(task)
        elif task.direction == 'down':
            self.down_channel.submit_task(task)
    
    def get_result(self, direction: str, timeout=1.0):
        """获取指定方向的结果"""
        if direction == 'up':
            return self.up_channel.get_result(timeout)
        elif direction == 'down':
            return self.down_channel.get_result(timeout)
        return None
    
    def get_total_stats(self):
        """获取总统计"""
        up_stats = self.up_channel.get_stats()
        down_stats = self.down_channel.get_stats()
        
        return {
            'up': up_stats,
            'down': down_stats,
            'total_inputs': up_stats['total_inputs'] + down_stats['total_inputs'],
            'total_processed': up_stats['processed'] + down_stats['processed'],
            'total_frames': up_stats['total_frames'] + down_stats['total_frames']
        }

