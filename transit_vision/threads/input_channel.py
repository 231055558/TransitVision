"""
输入通道模块
负责接收多路视频输入，分配到线程池，管理视频流
"""
import threading
import queue
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
from ..data_structures import VideoTask, ProcessedVideo

@dataclass
class StationInput:
    """单站输入数据"""
    station_id: int
    station_name: str
    up_video: Path
    down_video: Path
    bus_id: str

class InputChannel:
    """输入通道管理器"""
    def __init__(self, num_workers=32, queue_size=100):
        self.num_workers = num_workers
        self.input_queue = queue.Queue(maxsize=queue_size)
        self.output_queues = {
            'up': queue.Queue(maxsize=queue_size),
            'down': queue.Queue(maxsize=queue_size)
        }
        self.workers = []
        self.running = False
        self.stats = {
            'total_inputs': 0,
            'processed': 0,
            'up_count': 0,
            'down_count': 0
        }
        
    def load_station_data(self, data_dir: Path, max_stations: Optional[int] = None):
        """加载站点数据"""
        data_dir = Path(data_dir)
        stations = []
        
        station_id = 1
        while True:
            up_video = data_dir / f"{station_id}_up.mp4"
            down_video = data_dir / f"{station_id}_down.mp4"
            
            if not up_video.exists() or not down_video.exists():
                break
            
            stations.append(StationInput(
                station_id=station_id,
                station_name=f"站点{station_id}",
                up_video=up_video,
                down_video=down_video,
                bus_id=f"bus_{station_id}"
            ))
            
            station_id += 1
            if max_stations and station_id > max_stations:
                break
        
        return stations
    
    def submit_station(self, station: StationInput):
        """提交单站数据(up+down同时提交)"""
        # 提交上车视频
        up_task = VideoTask(
            bus_id=f"{station.bus_id}_up",
            video_path=str(station.up_video),
            direction='up',
            station_id=station.station_id,
            station_name=station.station_name
        )
        self.input_queue.put(('up', up_task))
        
        # 提交下车视频
        down_task = VideoTask(
            bus_id=f"{station.bus_id}_down",
            video_path=str(station.down_video),
            direction='down',
            station_id=station.station_id,
            station_name=station.station_name
        )
        self.input_queue.put(('down', down_task))
        
        self.stats['total_inputs'] += 2
    
    def _worker(self, worker_id: int):
        """工作线程：从输入队列获取任务并分发"""
        while self.running:
            try:
                item = self.input_queue.get(timeout=0.5)
                if item is None:
                    break
                
                direction, task = item
                
                # 分发到对应的输出队列
                self.output_queues[direction].put(task)
                
                self.stats['processed'] += 1
                if direction == 'up':
                    self.stats['up_count'] += 1
                else:
                    self.stats['down_count'] += 1
                
                self.input_queue.task_done()
                
            except queue.Empty:
                continue
    
    def start(self):
        """启动输入通道"""
        self.running = True
        for i in range(self.num_workers):
            worker = threading.Thread(target=self._worker, args=(i,))
            worker.start()
            self.workers.append(worker)
    
    def stop(self):
        """停止输入通道"""
        self.running = False
        
        # 发送停止信号
        for _ in range(self.num_workers):
            self.input_queue.put(None)
        
        # 等待所有工作线程结束
        for worker in self.workers:
            worker.join()
        
        self.workers.clear()
    
    def get_output_task(self, direction: str, timeout=1.0):
        """从输出队列获取任务"""
        try:
            return self.output_queues[direction].get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_stats(self):
        """获取统计信息"""
        return {
            'total_inputs': self.stats['total_inputs'],
            'processed': self.stats['processed'],
            'up_count': self.stats['up_count'],
            'down_count': self.stats['down_count'],
            'pending': self.input_queue.qsize(),
            'up_queue': self.output_queues['up'].qsize(),
            'down_queue': self.output_queues['down'].qsize()
        }

class MultiLineInputChannel:
    """多线路输入通道(模拟32线同时输入)"""
    def __init__(self, num_lines=32, workers_per_line=2):
        self.num_lines = num_lines
        self.channels = [
            InputChannel(num_workers=workers_per_line)
            for _ in range(num_lines)
        ]
        
    def load_and_replicate_data(self, data_dir: Path, max_stations: Optional[int] = None):
        """加载数据并复制到所有线路"""
        # 加载基础数据
        base_stations = self.channels[0].load_station_data(data_dir, max_stations)
        return base_stations
    
    def submit_station_to_all_lines(self, station: StationInput):
        """将单站数据提交到所有线路"""
        for line_id, channel in enumerate(self.channels):
            # 为每条线路创建独立的任务
            station_copy = StationInput(
                station_id=station.station_id,
                station_name=station.station_name,
                up_video=station.up_video,
                down_video=station.down_video,
                bus_id=f"line{line_id}_{station.bus_id}"
            )
            channel.submit_station(station_copy)
    
    def start_all(self):
        """启动所有线路"""
        for channel in self.channels:
            channel.start()
    
    def stop_all(self):
        """停止所有线路"""
        for channel in self.channels:
            channel.stop()
    
    def get_total_stats(self):
        """获取所有线路的统计信息"""
        total = {
            'total_inputs': 0,
            'processed': 0,
            'up_count': 0,
            'down_count': 0,
            'pending': 0,
            'up_queue': 0,
            'down_queue': 0
        }
        
        for channel in self.channels:
            stats = channel.get_stats()
            for key in total:
                total[key] += stats[key]
        
        return total

