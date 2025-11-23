import threading
import queue
from pathlib import Path
from ..data_structures import VideoTask, ProcessedVideo
from ..utils.video_reader import VideoReader, read_first_frame
from ..core.detection import DoorSegmentor
from ..utils.angle_calc import calc_door_angle

class VideoCaptureWorker(threading.Thread):
    def __init__(self, task_queue, result_queue, worker_id, 
                 shared_door_seg=None, inference_lock=None):
        super().__init__(daemon=True)
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.worker_id = worker_id
        self.running = True
        self.door_seg = shared_door_seg
        self.inference_lock = inference_lock
    
    def run(self):
        while self.running:
            try:
                task = self.task_queue.get(timeout=1)
                if task is None:
                    break
                
                result = self._process_task(task)
                self.result_queue.put(result)
                self.task_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker {self.worker_id} error: {e}")
                self.task_queue.task_done()
    
    def _process_task(self, task: VideoTask) -> ProcessedVideo:
        door_mask = None
        door_angle = None
        
        reader = VideoReader(task.video_path)
        fps = reader.fps
        width = reader.width
        height = reader.height
        frame_count = reader.frame_count
        reader.release()
        
        if task.need_door_seg or task.need_angle_calc:
            first_frame = read_first_frame(task.video_path)
            
            if task.need_door_seg and self.door_seg:
                if self.inference_lock:
                    with self.inference_lock:
                        door_obj = self.door_seg.detect(first_frame)
                else:
                    door_obj = self.door_seg.detect(first_frame)
                
                if door_obj:
                    door_mask = door_obj.mask
                    
                    if task.need_angle_calc:
                        door_angle = calc_door_angle(door_mask)
        
        return ProcessedVideo(
            bus_id=task.bus_id,
            video_path=task.video_path,
            door_mask=door_mask,
            door_angle=door_angle,
            fps=fps,
            width=width,
            height=height,
            frame_count=frame_count
        )
    
    def stop(self):
        self.running = False

class VideoCaptureThreadPool:
    def __init__(self, num_workers=4, door_model_path=None, device_config=None):
        self.num_workers = num_workers
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.workers = []
        self.inference_lock = threading.Lock()
        
        shared_door_seg = None
        if door_model_path and device_config:
            shared_door_seg = DoorSegmentor(door_model_path, device_config)
        
        for i in range(num_workers):
            worker = VideoCaptureWorker(
                self.task_queue, 
                self.result_queue, 
                i,
                shared_door_seg,
                self.inference_lock
            )
            self.workers.append(worker)
    
    def start(self):
        for worker in self.workers:
            worker.start()
    
    def submit(self, task: VideoTask):
        self.task_queue.put(task)
    
    def submit_batch(self, tasks: list):
        for task in tasks:
            self.submit(task)
    
    def get_result(self, timeout=None):
        return self.result_queue.get(timeout=timeout)
    
    def get_all_results(self, count, timeout=None):
        results = []
        for _ in range(count):
            try:
                result = self.get_result(timeout=timeout)
                results.append(result)
            except queue.Empty:
                break
        return results
    
    def wait_completion(self):
        self.task_queue.join()
    
    def shutdown(self):
        for _ in self.workers:
            self.task_queue.put(None)
        
        for worker in self.workers:
            worker.join()
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.shutdown()







