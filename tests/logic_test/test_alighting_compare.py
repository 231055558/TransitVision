import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transit_vision.utils import VideoReader, read_first_frame, DeviceConfig
from transit_vision.core.detection import PersonSegTracker, DoorSegmentor
from transit_vision.logic import preprocess_rear_door
import cv2
import numpy as np

# ===== 版本选择 =====
# 切换版本: 修改这里的导入即可
USE_VERSION = "v1"  # 可选: "v1" 或 "v2"

if USE_VERSION == "v1":
    from transit_vision.logic.alighting_counter_v1 import AlightingCounterV1 as AlightingCounter
    VERSION_NAME = "V1 (原版本 - 函数式逻辑)"
elif USE_VERSION == "v2":
    from transit_vision.logic.alighting_counter import AlightingCounter
    VERSION_NAME = "V2 (新版本 - 类式API)"
else:
    raise ValueError(f"未知版本: {USE_VERSION}")

VIDEO_PATH = "/mnt/mydisk/My_project/od_identification/bus_data/拥堵视频-1011/8-2-6712-001.mp4"
PERSON_MODEL = "/mnt/mydisk/My_project/bus_down/yolo11x-seg.pt"
DOOR_MODEL = "/mnt/mydisk/My_project/bus_down/front_door.pt"
TRACKER_CONFIG = str(Path(__file__).parent.parent.parent / "configs" / "botsort_seg.yaml")
DEVICE_CONFIG = str(Path(__file__).parent.parent.parent / "configs" / "device_debug.yaml")
OUTPUT_DIR = Path(__file__).parent / "output"

def test_alighting_compare():
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print("="*70)
    print(f"下客识别对比测试 - 使用版本: {VERSION_NAME}")
    print("="*70)
    
    device_cfg = DeviceConfig(DEVICE_CONFIG)
    
    # 1. 检测门并预处理
    print("\n1. Door detection & preprocessing...")
    door_seg = DoorSegmentor(DOOR_MODEL, device_cfg)
    first_frame = read_first_frame(VIDEO_PATH)
    door = door_seg.detect(first_frame)
    
    if door is None:
        print("✗ No door detected")
        return
    
    door.mask = preprocess_rear_door(door)
    print(f"✓ Door mask area: {np.sum(door.mask > 0)}")
    
    # 2. 初始化计数器
    config = {'alighting_counter': {'door_entry_threshold': 0.85}}
    counter = AlightingCounter(config)
    
    # 3. 追踪人员
    print("\n2. Person tracking...")
    person_tracker = PersonSegTracker(PERSON_MODEL, TRACKER_CONFIG, device_cfg)
    
    all_persons = {}
    frame_idx = 0
    
    with VideoReader(VIDEO_PATH) as reader:
        for frame in reader:
            detections = person_tracker.track(frame)
            
            # 更新Person对象
            current_frame_ids = set()
            for det in detections:
                tid = det['id']
                current_frame_ids.add(tid)
                
                if tid not in all_persons:
                    from transit_vision.data_structures import Person
                    all_persons[tid] = Person(tid)
                
                p = all_persons[tid]
                p.add_detection(frame_idx, det['box'], det['mask'], det['conf'])
            
            # 更新计数器
            counter.update_counts(frame_idx, all_persons, door)
            
            frame_idx += 1
            if frame_idx % 50 == 0:
                print(f"  Processed {frame_idx} frames, Count: {counter.get_count()}")
    
    total_count = counter.get_count()
    print(f"\n✓ Total alighting passengers: {total_count}")
    
    # 4. 可视化
    print("\n3. Visualization...")
    counted_ids = {pid for pid, p in all_persons.items() if getattr(p, 'has_counted', False)}
    trigger_frames = {getattr(p, 'trigger_frame', p.frames[-1] if p.frames else 0): pid 
                      for pid, p in all_persons.items() if pid in counted_ids}
    
    with VideoReader(VIDEO_PATH) as reader:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_path = str(OUTPUT_DIR / f"alighting_result_{USE_VERSION}.mp4")
        out = cv2.VideoWriter(out_path, fourcc, reader.fps, (reader.width, reader.height))
        
        frame_idx = 0
        
        for frame in reader:
            # 绘制门掩码
            mask_color = np.zeros_like(frame)
            mask_color[door.mask > 0] = [0, 255, 255]
            frame = cv2.addWeighted(frame, 0.85, mask_color, 0.15, 0)
            
            # 显示版本信息
            cv2.putText(frame, f"Version: {USE_VERSION.upper()}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Count: {counter.get_count()}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 绘制检测结果
            for tid, person in all_persons.items():
                if frame_idx in person.frames:
                    idx = person.frames.index(frame_idx)
                    box = person.boxes[idx]
                    mask = person.masks[idx]
                    
                    is_counted = tid in counted_ids
                    color = (0, 255, 0) if is_counted else (255, 0, 0)
                    
                    if mask is not None:
                        mask_color = np.zeros_like(frame)
                        mask_color[mask > 0] = color
                        frame = cv2.addWeighted(frame, 1.0, mask_color, 0.5, 0)
                    
                    bx1, by1, bx2, by2 = map(int, box)
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
                    label = f"{'ALIGHT' if is_counted else 'TRACK'}:{tid}"
                    cv2.putText(frame, label, (bx1, by1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            out.write(frame)
            
            # 判定时刻定格0.5s
            if frame_idx in trigger_frames:
                freeze_frames = int(reader.fps * 0.5)
                for _ in range(freeze_frames):
                    out.write(frame)
            
            frame_idx += 1
        
        out.release()
    
    print(f"\nOutput: {out_path}")
    print(f"Total alighting count: {total_count}")
    print("="*70)

if __name__ == "__main__":
    test_alighting_compare()

