import numpy as np
import collections
from typing import List, Dict

from ..data_structures.door import Door

def smooth_detection(current_count: int, history: collections.deque) -> int:
    # 对检测数量进行平滑处理，减少突变
    history.append(current_count)
    if len(history) < 3:
        return current_count
    
    weights = np.linspace(0.3, 1.0, len(history))
    return int(np.average(list(history), weights=weights))

def calculate_crowd_status(occupancy_rate: float, near_door_count: int, config: dict) -> (str, tuple, str):
    # 根据配置计算当前的拥挤状态
    seat_vacancy_rate = 1.0 - occupancy_rate

    if near_door_count >= config['near_door_severe_threshold']:
        return "4", (0, 0, 255), f"车门堵塞({near_door_count}人)"
    if near_door_count >= config['near_door_crowded_threshold']:
        return "3", (0, 255, 255), f"车门聚集({near_door_count}人)"
    if seat_vacancy_rate <= config['crowded_seat_rate']:
        return "3", (0, 255, 255), f"空座率≤{config['crowded_seat_rate']:.0%}"
    if config['comfortable_seat_rate_min'] <= seat_vacancy_rate <= (1.0 - config['crowded_seat_rate']):
        return "2", (255, 0, 0), f"舒适"
    if seat_vacancy_rate > config['idle_seat_rate']:
        return "1", (0, 255, 0), f"空闲"
    
    if occupancy_rate < 0.3:
        return "1", (0, 255, 0), f"载客率<30%"
    elif occupancy_rate < 0.7:
        return "2", (255, 0, 0), f"载客率<70%"
    else:
        return "3", (0, 255, 255), f"载客率≥70%"

class OccupancyAnalyzer:
    def __init__(self, config: dict):
        self.config = config['occupancy_analyzer']
        self.people_history = collections.deque(maxlen=10)
        self.near_door_history = collections.deque(maxlen=10)

    def reset(self):
        self.people_history.clear()
        self.near_door_history.clear()

    def analyze(self, head_detections: List[Dict], door: Door) -> Dict:
        # 分析单帧的检测结果并返回拥挤度状态
        current_frame_people = len(head_detections)

        near_door_count = 0
        if door and door.anchor:
            for det in head_detections:
                head_center = det.get('center') 
                if head_center:
                    dist = np.linalg.norm(np.array(head_center) - np.array(door.anchor))
                    if dist < self.config['door_distance_threshold']:
                        near_door_count += 1

        smoothed_people = smooth_detection(current_frame_people, self.people_history)
        smoothed_near_door = smooth_detection(near_door_count, self.near_door_history)

        occupancy_rate = smoothed_people / self.config['max_people']
        status, color, reason = calculate_crowd_status(
            occupancy_rate, smoothed_near_door, self.config
        )

        analysis_result = {
            'person_count': smoothed_people,
            'near_door_count': smoothed_near_door,
            'occupancy_rate': occupancy_rate,
            'status_level': status,
            'status_color': color,
            'status_reason': reason
        }

        return analysis_result
