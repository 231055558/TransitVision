import numpy as np
import cv2
from typing import Dict

from ..data_structures.person import Person
from ..data_structures.door import Door

def check_door_entry(person: Person, door: Door, threshold: float) -> bool:
    #检查乘客的最新掩码是否有指定比例进入车门区域

    if door.mask is None or person.last_mask is None:
        return False

    contours, _ = cv2.findContours(door.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    door_polygon = max(contours, key=cv2.contourArea)

    mask_points = cv2.findNonZero(person.last_mask)
    if mask_points is None:
        return False

    inside_count = 0
    for point in mask_points:
        p = point[0]
        if cv2.pointPolygonTest(door_polygon, (float(p[0]), float(p[1])), False) >= 0:
            inside_count += 1
            
    ratio = inside_count / len(mask_points)
    return ratio >= threshold

def check_box_overlap_with_mask(box: list, door_mask: np.ndarray) -> bool:
    #检查边界框是否与车门掩码区域有像素级重叠
    if box is None or door_mask is None: 
        return False
    h, w = door_mask.shape[:2]
    bx1, by1, bx2, by2 = map(int, box)
    bx1, by1 = max(0, bx1), max(0, by1)
    bx2, by2 = min(w - 1, bx2), min(h - 1, by2)
    if bx1 >= bx2 or by1 >= by2: 
        return False
    door_mask_in_box_region = door_mask[by1:by2, bx1:bx2]
    return np.any(door_mask_in_box_region > 0)

class AlightingCounter:
    def __init__(self, config: dict):
        alighting_config = config['alighting_counter']
        self.grace_period_frames = alighting_config['grace_period_frames']
        self.door_entry_threshold = alighting_config['door_entry_threshold']
        self.passenger_count = 0

    def reset(self):
        self.passenger_count = 0

    def update_counts(self, frame_idx: int, persons: Dict[int, Person], door: Door):

        # 处理当前帧的追踪数据并更新下客计数

        if door is None:
            return

        for person in persons.values():
            if person.has_disembark_intent or person.has_counted:
                continue

            is_in_door = check_door_entry(person, door, self.door_entry_threshold)
            status = 1 if is_in_door else 0
            person.inside_door_history.append(status)

            if len(person.inside_door_history) >= 2 and \
               person.inside_door_history[-2] == 0 and \
               person.inside_door_history[-1] == 1:
                person.has_disembark_intent = True

        for person in persons.values():
            if not person.has_disembark_intent or person.has_counted:
                continue
            
            frames_disappeared = frame_idx - person.last_seen_frame
            if frames_disappeared >= self.grace_period_frames:
                if check_box_overlap_with_mask(person.last_box, door.mask):
                    self.passenger_count += 1
                    person.has_counted = True
                    print(f"[Frame {frame_idx}] Person {person.id} counted as alighting.")

    def get_count(self) -> int:
        # 返回当前的总下车人数
        return self.passenger_count
