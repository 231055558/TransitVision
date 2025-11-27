"""
下客流程三线程流水线
使用有界阻塞队列防止内存溢出
"""
import threading
import queue
import numpy as np
from collections import defaultdict

from ..utils.video_reader import VideoReader
from ..core.detection import PersonSegTracker
from ..logic.alighting_counter import filter_alighting_passengers
from ..data_structures import Person

class VideoReadThread(threading.Thread):
    """视频读取线程"""
    def __init__(self, video_path, frame_queue, max_queue_size=30):
        super().__init__(daemon=True)
        self.video_path = video_path
        self.frame_queue = frame_queue
        self.running = True
    
    def run(self):
        reader = VideoReader(self.video_path)
        frame_idx = 0
        
        for frame in reader:
            if not self.running:
                break
            
            self.frame_queue.put((frame_idx, frame))
            frame_idx += 1
        
        self.frame_queue.put(None)
        reader.release()
    
    def stop(self):
        self.running = False

class InferenceThread(threading.Thread):
    """GPU推理线程 - 逐帧推理并传递多边形结果"""
    def __init__(self, frame_queue, result_queue, person_seg_tracker):
        super().__init__(daemon=True)
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.person_seg_tracker = person_seg_tracker
        self.running = True
    
    def run(self):
        while self.running:
            try:
                item = self.frame_queue.get(timeout=1)
                
                if item is None:
                    self.result_queue.put(None)
                    break
                
                frame_idx, frame = item
                
                # 逐帧推理 (PersonSegTracker.track 已输出多边形)
                detections = self.person_seg_tracker.track(frame)
                
                self.result_queue.put((frame_idx, detections))
                
            except queue.Empty:
                continue
    
    def stop(self):
        self.running = False

class LogicThread(threading.Thread):
    """逻辑处理线程 - 收集追踪结果和计数"""
    def __init__(self, result_queue, door_mask):
        super().__init__(daemon=True)
        self.result_queue = result_queue
        self.door_mask = door_mask
        self.tracks = defaultdict(lambda: Person(0))
        self.running = True
    
    def run(self):
        while self.running:
            try:
                item = self.result_queue.get(timeout=1)
                
                if item is None:
                    break
                
                frame_idx, detections = item
                
                # detections 已包含 track_id, box, polygon, conf
                if detections:
                    for det in detections:
                        track_id = det['id']
                        box = det['box']
                        polygon = det['polygon']
                        conf = det['conf']
                        
                        if track_id not in self.tracks:
                            self.tracks[track_id] = Person(track_id)
                        
                        self.tracks[track_id].add_detection(frame_idx, box, polygon, conf)
                
            except queue.Empty:
                continue
    
    def get_alighting_passengers(self):
        """获取下车乘客"""
        return filter_alighting_passengers(dict(self.tracks), self.door_mask)
    
    def stop(self):
        self.running = False

class AlightingPipeline:
    """下客流程流水线管理器"""
    def __init__(self, video_path, door_mask, person_seg_tracker, queue_size=30):
        self.video_path = video_path
        self.door_mask = door_mask
        self.person_seg_tracker = person_seg_tracker
        self.queue_size = queue_size
        
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.result_queue = queue.Queue(maxsize=queue_size)
        
        self.video_thread = VideoReadThread(video_path, self.frame_queue)
        self.inference_thread = InferenceThread(
            self.frame_queue, self.result_queue, person_seg_tracker
        )
        self.logic_thread = LogicThread(
            self.result_queue, door_mask
        )
    
    def start(self):
        """启动流水线"""
        self.video_thread.start()
        self.inference_thread.start()
        self.logic_thread.start()
    
    def wait_completion(self):
        """等待完成"""
        self.video_thread.join()
        self.inference_thread.join()
        self.logic_thread.join()
    
    def get_result(self):
        """获取结果"""
        return self.logic_thread.get_alighting_passengers()
    
    def stop(self):
        """停止流水线"""
        self.video_thread.stop()
        self.inference_thread.stop()
        self.logic_thread.stop()
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()

