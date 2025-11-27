"""
提取ReID测试数据脚本
从视频中提取上下车乘客的bbox截图，用于ReID测试
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from transit_vision.threads import MultiLineInputChannel, MultiDirectionLogicChannel
from transit_vision.utils import DeviceConfig, VideoReader, rotate_frame

DATA_DIR = Path(__file__).parent.parent / "data" / "close_loop_od"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "reid_dataset"
PERSON_MODEL = "/mnt/mydisk/My_project/bus_down/yolo11x-seg.pt"
DOOR_MODEL = "/mnt/mydisk/My_project/bus_down/front_door.pt"
TRACKER_CONFIG = str(Path(__file__).parent.parent / "configs" / "botsort_seg.yaml")
DEVICE_CONFIG = str(Path(__file__).parent.parent / "configs" / "device_debug.yaml")

NUM_LINES = 2
BATCH_SIZE = 64
NUM_WORKERS = 2
MAX_STATIONS = 3

def select_frames(total_frames, n=7):
    """从总帧数中均匀选择n帧,然后掐头去尾保留中间n-2帧"""
    if total_frames <= n:
        indices = list(range(total_frames))
    else:
        indices = np.linspace(0, total_frames-1, n, dtype=int).tolist()
    
    if len(indices) > 2:
        return indices[1:-1]
    return indices

def save_person_crops(video_path, person, line_id, station_id, person_idx, 
                     output_dir, rotation_angle=None):
    """保存单个人的bbox截图"""
    if len(person.frames) < 7:
        return 0
    
    selected_indices = select_frames(len(person.frames))
    
    with VideoReader(video_path) as reader:
        saved_count = 0
        for idx in selected_indices:
            frame_num = person.frames[idx]
            box = person.boxes[idx]
            
            reader.seek(frame_num)
            frame = next(reader)
            
            if rotation_angle is not None:
                frame = rotate_frame(frame, rotation_angle)
            
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                filename = f"{line_id}_{station_id}_{person_idx}_{saved_count}.png"
                save_path = output_dir / filename
                cv2.imwrite(str(save_path), crop)
                saved_count += 1
    
    return saved_count

def extract_reid_data():
    print("=" * 70)
    print("ReID数据提取脚本")
    print("=" * 70)
    print(f"数据目录: {DATA_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"并行线路数: {NUM_LINES}")
    print(f"测试站点数: {MAX_STATIONS}")
    print()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    device_cfg = DeviceConfig(DEVICE_CONFIG)
    
    print("创建输入通道...")
    input_channel = MultiLineInputChannel(num_lines=NUM_LINES, workers_per_line=2)
    stations = input_channel.load_and_replicate_data(DATA_DIR, MAX_STATIONS)
    print(f"✓ 加载 {len(stations)} 个站点")
    
    print("\n创建逻辑通道...")
    logic_channel = MultiDirectionLogicChannel(
        PERSON_MODEL, TRACKER_CONFIG, DOOR_MODEL, device_cfg,
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )
    print(f"✓ 逻辑通道已创建")
    
    print("\n启动所有通道...")
    input_channel.start_all()
    logic_channel.start()
    print("✓ 所有通道已启动")
    
    print(f"\n开始处理站点数据...")
    print("=" * 70)
    
    import time
    
    station_results = {}
    
    for station in stations:
        station_id = station.station_id - 1
        print(f"\n[站点 {station.station_id}: {station.station_name}]")
        
        input_channel.submit_station_to_all_lines(station)
        print(f"  提交到输入通道: {NUM_LINES * 2} 任务")
        
        time.sleep(0.5)
        
        initial_logic_stats = logic_channel.get_total_stats()
        initial_logic_processed = initial_logic_stats['total_processed']
        
        task_map = {}
        submitted = 0
        for line_id in range(NUM_LINES):
            channel = input_channel.channels[line_id]
            
            up_task = channel.get_output_task('up', timeout=1.0)
            down_task = channel.get_output_task('down', timeout=1.0)
            
            if up_task:
                logic_channel.submit_task(up_task)
                task_map[f"{line_id}_up"] = up_task
                submitted += 1
            if down_task:
                logic_channel.submit_task(down_task)
                task_map[f"{line_id}_down"] = down_task
                submitted += 1
        
        print(f"  提交到逻辑通道: {submitted} 任务")
        
        print(f"  等待逻辑处理完成...")
        wait_start = time.time()
        check_count = 0
        while True:
            stats = logic_channel.get_total_stats()
            
            check_count += 1
            if check_count % 4 == 0:
                elapsed = time.time() - wait_start
                print(f"    [逻辑 {elapsed:.1f}s] 已处理 {stats['total_processed']}/{initial_logic_processed + submitted} 任务")
            
            if stats['total_processed'] >= initial_logic_processed + submitted:
                print(f"    ✓ 逻辑处理完成")
                break
            time.sleep(0.5)
        
        results = []
        for _ in range(submitted):
            for direction in ['up', 'down']:
                result = logic_channel.get_result(direction, timeout=0.1)
                if result:
                    results.append(result)
        
        station_results[station_id] = {
            'tasks': task_map,
            'results': results
        }
        
        print(f"  收集到 {len(results)} 个结果")
    
    print("\n停止所有通道...")
    input_channel.stop_all()
    logic_channel.stop()
    print("✓ 所有通道已停止")
    
    print(f"\n{'='*70}")
    print("开始提取图片...")
    print(f"{'='*70}")
    
    total_saved = 0
    person_global_idx = {}
    
    for station_id, data in sorted(station_results.items()):
        print(f"\n[站点 {station_id}]")
        
        for result in data['results']:
            task = result['task']
            passengers = result['valid_passengers']
            direction = task.direction
            
            line_id = None
            for lid in range(NUM_LINES):
                task_key = f"{lid}_{direction}"
                if task_key in data['tasks'] and data['tasks'][task_key].bus_id == task.bus_id:
                    line_id = lid
                    break
            
            if line_id is None:
                continue
            
            if len(passengers) == 0:
                continue
            
            rotation_angle = None
            if direction == 'up':
                from transit_vision.logic import preprocess_front_door
                from transit_vision.core.detection import DoorSegmentor
                from transit_vision.utils import read_first_frame
                
                door_seg = DoorSegmentor(DOOR_MODEL, device_cfg)
                first_frame = read_first_frame(task.video_path)
                door = door_seg.detect(first_frame, conf=0.3)
                if door:
                    angle, _ = preprocess_front_door(door, first_frame)
                    if angle is not None:
                        rotation_angle = -angle
            
            print(f"  线路{line_id} {direction.upper()}: {len(passengers)}人")
            
            key = f"{line_id}_{station_id}"
            if key not in person_global_idx:
                person_global_idx[key] = 0
            
            for track_id, person in sorted(passengers.items()):
                person_idx = person_global_idx[key]
                saved = save_person_crops(
                    task.video_path, person, line_id, station_id, 
                    person_idx, OUTPUT_DIR, rotation_angle
                )
                if saved > 0:
                    print(f"    ID {track_id} -> person_{person_idx}: 保存 {saved} 张图片")
                    total_saved += saved
                    person_global_idx[key] += 1
                else:
                    print(f"    ID {track_id}: 帧数不足(<7), 跳过")
    
    print(f"\n{'='*70}")
    print("提取完成!")
    print(f"{'='*70}")
    print(f"总共保存: {total_saved} 张图片")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"{'='*70}")

if __name__ == "__main__":
    extract_reid_data()

