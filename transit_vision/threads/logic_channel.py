"""
逻辑运算通道模块
处理推理结果，执行上下车逻辑判定
"""
import queue
import threading
from ..utils import VideoReader, read_first_frame, rotate_frame, DeviceConfig
from ..core.detection import PersonSegTracker, DoorSegmentor
from ..logic import preprocess_front_door, preprocess_rear_door
from ..logic import filter_boarding_passengers, filter_alighting_passengers
from ..data_structures import Person

class LogicChannel:
    """单方向逻辑处理通道"""
    def __init__(self, direction: str, person_model: str, tracker_config: str, 
                 door_model: str, device_config: DeviceConfig, batch_size=128, shared_lock=None,
                 recalc_door_per_video=False):
        self.direction = direction
        self.batch_size = batch_size
        self.person_model = person_model
        self.tracker_config = tracker_config
        self.device_config = device_config
        self.door_segmentor = DoorSegmentor(door_model, device_config)
        self.inference_lock = shared_lock if shared_lock else threading.Lock()
        self.door_cache = {}
        self.recalc_door_per_video = recalc_door_per_video
        self.workers = []
        self.running = False
        self.input_queue = queue.Queue(maxsize=200)
        self.output_queue = queue.Queue(maxsize=200)
        self.stats = {
            'total_inputs': 0,
            'processed': 0,
            'boarding_count': 0,
            'alighting_count': 0
        }
    
    def _logic_worker(self, worker_id: int):
        """逻辑处理工作线程"""
        while self.running:
            try:
                task = self.input_queue.get(timeout=0.5)
                if task is None:
                    break
                
                result = self._process_logic(task)
                self.output_queue.put(result)
                
                self.stats['processed'] += 1
                self.input_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Logic worker {worker_id} error: {e}")
                import traceback
                traceback.print_exc()
                self.input_queue.task_done()
    
    def _extract_line_id(self, bus_id: str) -> str:
        """从bus_id提取线路标识作为缓存键"""
        parts = bus_id.split('_')
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"
        return bus_id
    
    def _get_door_info(self, video_path: str, bus_id: str):
        """获取门信息(带条件缓存)"""
        cache_key = self._extract_line_id(bus_id)
        
        # 如果不需要每次重算，且缓存中有，直接返回
        if not self.recalc_door_per_video and cache_key in self.door_cache:
            return self.door_cache[cache_key]
        
        # 检测门框
        first_frame = read_first_frame(video_path)
        door = self.door_segmentor.detect(first_frame, conf=0.3)
        
        if door is None:
            return None
        
        if self.direction == 'up':
            angle, door_bbox = preprocess_front_door(door, first_frame)
            if angle is None:
                return None
            door_info = {'angle': angle, 'bbox': door_bbox}
        else:
            door_mask = preprocess_rear_door(door)
            door_info = {'mask': door_mask}
        
        # 如果不需要每次重算，保存到缓存
        if not self.recalc_door_per_video:
            self.door_cache[cache_key] = door_info
        
        return door_info
    
    def _inference_video(self, video_path: str, rotation_angle=None):
        """推理视频(每个视频创建独立tracker,确保零状态污染)"""
        tracker = PersonSegTracker(self.person_model, self.tracker_config, self.device_config)
        all_tracks = {}
        
        with VideoReader(video_path) as reader:
            frames_batch = []
            frame_indices = []
            
            for frame_idx, frame in enumerate(reader):
                if rotation_angle is not None:
                    frame = rotate_frame(frame, rotation_angle)
                
                frames_batch.append(frame)
                frame_indices.append(frame_idx)
                
                if len(frames_batch) >= self.batch_size:
                    self._inference_batch(tracker, frames_batch, frame_indices, all_tracks)
                    frames_batch = []
                    frame_indices = []
            
            if frames_batch:
                self._inference_batch(tracker, frames_batch, frame_indices, all_tracks)
        
        return all_tracks
    
    def _inference_batch(self, tracker, frames, frame_indices, all_tracks):
        """批量推理(复用推理通道的实现方式)"""
        with self.inference_lock:
            for frame_idx, frame in zip(frame_indices, frames):
                detections = tracker.track(frame)
                
                for det in detections:
                    tid = det['id']
                    if tid not in all_tracks:
                        all_tracks[tid] = Person(tid)
                    
                    all_tracks[tid].add_detection(
                        frame_idx,
                        det['box'],
                        det['polygon'],
                        det['conf']
                    )
    
    def _process_logic(self, task):
        """处理单个任务：门检测 → 推理 → 逻辑判定"""
        print(f"  [逻辑-{self.direction}] 开始处理: {task.bus_id}")
        
        door_info = self._get_door_info(task.video_path, task.bus_id)
        
        if door_info is None:
            print(f"  [逻辑-{self.direction}] ✗ 未检测到门: {task.bus_id}")
            return {
                'task': task,
                'valid_passengers': {},
                'count': 0,
                'door_info': None
            }
        
        rotation_angle = None
        if self.direction == 'up':
            rotation_angle = -door_info['angle']
        
        tracks = self._inference_video(task.video_path, rotation_angle)
        
        if self.direction == 'up':
            valid = filter_boarding_passengers(tracks, door_info['bbox'])
            self.stats['boarding_count'] += len(valid)
            print(f"  [逻辑-{self.direction}] ✓ {task.bus_id}: {len(tracks)}人追踪 -> {len(valid)}人上车")
        else:
            valid = filter_alighting_passengers(tracks, door_info['mask'])
            self.stats['alighting_count'] += len(valid)
            print(f"  [逻辑-{self.direction}] ✓ {task.bus_id}: {len(tracks)}人追踪 -> {len(valid)}人下车")
        
        return {
            'task': task,
            'valid_passengers': valid,
            'count': len(valid),
            'door_info': door_info  # 返回门框信息供后续使用
        }
    
    def submit_task(self, task):
        """提交视频任务"""
        self.input_queue.put(task)
        self.stats['total_inputs'] += 1
    
    def start(self, num_workers=4):
        """启动逻辑通道"""
        self.running = True
        for i in range(num_workers):
            worker = threading.Thread(target=self._logic_worker, args=(i,))
            worker.start()
            self.workers.append(worker)
    
    def stop(self):
        """停止逻辑通道"""
        self.running = False
        for _ in range(len(self.workers)):
            self.input_queue.put(None)
        for worker in self.workers:
            worker.join()
        self.workers.clear()
    
    def get_result(self, timeout=1.0):
        """获取逻辑结果"""
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_stats(self):
        """获取统计信息"""
        return {
            'total_inputs': self.stats['total_inputs'],
            'processed': self.stats['processed'],
            'boarding_count': self.stats['boarding_count'],
            'alighting_count': self.stats['alighting_count'],
            'pending': self.input_queue.qsize(),
            'output_queue': self.output_queue.qsize()
        }

class MultiDirectionLogicChannel:
    """多方向逻辑通道"""
    def __init__(self, person_model: str, tracker_config: str, door_model: str, 
                 device_config: DeviceConfig, batch_size=128, num_workers=4,
                 recalc_door_up=False, recalc_door_down=False):
        # 创建全局推理锁，强制所有通道串行使用GPU/CPU
        self.global_lock = threading.Lock()
        
        self.up_channel = LogicChannel(
            'up', person_model, tracker_config, door_model, device_config, 
            batch_size, shared_lock=self.global_lock, recalc_door_per_video=recalc_door_up
        )
        self.down_channel = LogicChannel(
            'down', person_model, tracker_config, door_model, device_config,
            batch_size, shared_lock=self.global_lock, recalc_door_per_video=recalc_door_down
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
    
    def submit_task(self, task):
        """根据方向提交视频任务"""
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
            'total_boarding': up_stats['boarding_count'],
            'total_alighting': down_stats['alighting_count']
        }

