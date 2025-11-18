import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transit_vision.utils import (
    VideoReader, read_first_frame, DeviceConfig, rotate_frame,
    extract_driver_mask, save_bbox_crops
)
from transit_vision.core.detection import PersonSegTracker, DoorSegmentor
from transit_vision.logic import (
    preprocess_front_door, preprocess_rear_door,
    filter_boarding_passengers, filter_alighting_passengers
)

PERSON_MODEL = "/mnt/mydisk/My_project/bus_down/yolo11x-seg.pt"
DOOR_MODEL = "/mnt/mydisk/My_project/bus_down/front_door.pt"
TRACKER_CONFIG = str(Path(__file__).parent.parent.parent / "configs" / "botsort_seg.yaml")
DEVICE_CONFIG = str(Path(__file__).parent.parent.parent / "configs" / "device_debug.yaml")

def process_up_video(video_path, output_dir, person_tracker, angle, door_bbox, driver_bbox=None):
    print(f"  [UP] {video_path.name}")
    
    rotation_angle = -angle
    all_tracks = {}
    frame_idx = 0
    
    with VideoReader(video_path) as reader:
        for frame in reader:
            rotated = rotate_frame(frame, rotation_angle)
            
            # 应用司机掩码
            if driver_bbox:
                x1, y1, x2, y2 = driver_bbox
                rotated[y1:y2, x1:x2] = 0
            
            detections = person_tracker.track(rotated)
            
            for det in detections:
                tid = det['id']
                if tid not in all_tracks:
                    all_tracks[tid] = type('Person', (), {
                        'id': tid, 'frames': [], 'boxes': [], 'masks': [], 'confs': []
                    })()
                
                p = all_tracks[tid]
                p.frames.append(frame_idx)
                p.boxes.append(det['box'])
                p.masks.append(det['mask'])
                p.confs.append(det['conf'])
            
            frame_idx += 1
    
    # 上客判定
    boarding = filter_boarding_passengers(all_tracks, door_bbox)
    
    if len(boarding) == 0:
        print(f"    → 无上车乘客")
        output_dir.mkdir(parents=True, exist_ok=True)
        return
    
    print(f"    → {len(boarding)} 个上车乘客")
    
    # 保存截图
    saved = 0
    for tid, person in boarding.items():
        count = save_bbox_crops(video_path, tid, person.frames, person.boxes, 
                                output_dir, rotation_angle)
        if count > 0:
            print(f"      ID {tid}: {count} 张")
            saved += 1
        else:
            print(f"      ID {tid}: 跳过(<7帧)")
    
    if saved == 0:
        print(f"    → 无有效ID")

def process_down_video(video_path, output_dir, person_tracker, door_mask):
    print(f"  [DOWN] {video_path.name}")
    
    all_tracks = {}
    frame_idx = 0
    
    with VideoReader(video_path) as reader:
        for frame in reader:
            detections = person_tracker.track(frame)
            
            for det in detections:
                tid = det['id']
                if tid not in all_tracks:
                    all_tracks[tid] = type('Person', (), {
                        'id': tid, 'frames': [], 'boxes': [], 'masks': [], 'confs': [],
                        'trigger_frame': None
                    })()
                
                p = all_tracks[tid]
                p.frames.append(frame_idx)
                p.boxes.append(det['box'])
                p.masks.append(det['mask'])
                p.confs.append(det['conf'])
            
            frame_idx += 1
    
    # 下客判定
    alighting = filter_alighting_passengers(all_tracks, door_mask)
    
    if len(alighting) == 0:
        print(f"    → 无下车乘客")
        output_dir.mkdir(parents=True, exist_ok=True)
        return
    
    print(f"    → {len(alighting)} 个下车乘客")
    
    # 保存截图
    saved = 0
    for tid, person in alighting.items():
        count = save_bbox_crops(video_path, tid, person.frames, person.boxes, output_dir)
        if count > 0:
            print(f"      ID {tid}: {count} 张")
            saved += 1
        else:
            print(f"      ID {tid}: 跳过(<7帧)")
    
    if saved == 0:
        print(f"    → 无有效ID")

def process_dataset(data_root, output_root):
    data_path = Path(data_root)
    output_path = Path(output_root)
    
    print("="*70)
    print("批量处理程序启动")
    print("="*70)
    
    # 加载模型
    print("\n加载模型...")
    device_cfg = DeviceConfig(DEVICE_CONFIG)
    door_seg = DoorSegmentor(DOOR_MODEL, device_cfg)
    person_tracker = PersonSegTracker(PERSON_MODEL, TRACKER_CONFIG, device_cfg)
    print("✓ 模型加载完成")
    
    # 遍历线路
    for line_dir in sorted(data_path.iterdir()):
        if not line_dir.is_dir() or line_dir.name.startswith('.'):
            continue
        
        print(f"\n{'='*70}")
        print(f"线路: {line_dir.name}")
        print(f"{'='*70}")
        
        # 读取站台列表
        station_file = line_dir / "station_list.txt"
        if not station_file.exists():
            print(f"  ✗ 未找到 station_list.txt")
            continue
        
        with open(station_file, 'r', encoding='utf-8') as f:
            stations = [line.strip() for line in f if line.strip()]
        
        print(f"  站台数: {len(stations)}")
        print(f"  站台列表: {', '.join(stations)}")
        
        # 遍历车辆
        for vehicle_dir in sorted(line_dir.iterdir()):
            if not vehicle_dir.is_dir() or vehicle_dir.name in ['station_list.txt', '.DS_Store']:
                continue
            
            print(f"\n  车牌: {vehicle_dir.name}")
            
            # 车辆级缓存
            vehicle_front_angle = None
            vehicle_front_bbox = None
            vehicle_driver_bbox = None
            vehicle_rear_mask = None
            
            # 按站台顺序处理
            for station in stations:
                # 处理down视频
                down_videos = list(vehicle_dir.glob(f"*_{station}_down.mp4"))
                for video_path in down_videos:
                    # 检测后车门(首次)
                    if vehicle_rear_mask is None:
                        print(f"    检测后车门(缓存)...")
                        first_frame = read_first_frame(video_path)
                        door = door_seg.detect(first_frame)
                        
                        if door is None:
                            print(f"    ✗ 未检测到后车门")
                            vehicle_rear_mask = False
                        else:
                            vehicle_rear_mask = preprocess_rear_door(door)
                            print(f"    ✓ 后车门检测完成")
                    
                    if vehicle_rear_mask is False:
                        continue
                    
                    output_dir = output_path / line_dir.name / vehicle_dir.name / video_path.stem
                    try:
                        person_tracker.reset()
                        process_down_video(video_path, output_dir, person_tracker, vehicle_rear_mask)
                    except Exception as e:
                        print(f"    ✗ 处理失败: {e}")
                        output_dir.mkdir(parents=True, exist_ok=True)
                
                # 处理up视频
                up_videos = list(vehicle_dir.glob(f"*_{station}_up.mp4"))
                for video_path in up_videos:
                    # 检测前车门(首次)
                    if vehicle_front_angle is None:
                        print(f"    检测前车门角度和bbox(缓存)...")
                        first_frame = read_first_frame(video_path)
                        door = door_seg.detect(first_frame, conf=0.3)
                        
                        if door is None:
                            print(f"    ✗ 未检测到前车门")
                            vehicle_front_angle = False
                            break
                        
                        angle, bbox = preprocess_front_door(door, first_frame)
                        if angle is None:
                            print(f"    ✗ 预处理失败")
                            vehicle_front_angle = False
                            break
                        
                        vehicle_front_angle = angle
                        vehicle_front_bbox = bbox
                        print(f"    ✓ 角度: {angle}°, Bbox: {bbox}")
                    
                    # 提取司机掩码(首次)
                    if vehicle_driver_bbox is None and vehicle_front_angle is not False:
                        print(f"    提取司机掩码(缓存)...")
                        try:
                            with VideoReader(video_path) as reader:
                                person_tracker.reset()
                                vehicle_driver_bbox = extract_driver_mask(
                                    reader, person_tracker, -vehicle_front_angle
                                )
                            
                            if vehicle_driver_bbox:
                                print(f"    ✓ 司机掩码: {vehicle_driver_bbox}")
                            else:
                                print(f"    → 未检测到司机")
                                vehicle_driver_bbox = False
                        except Exception as e:
                            print(f"    ✗ 司机掩码提取失败: {e}")
                            vehicle_driver_bbox = False
                    
                    if vehicle_front_angle is False:
                        continue
                    
                    output_dir = output_path / line_dir.name / vehicle_dir.name / video_path.stem
                    try:
                        person_tracker.reset()
                        driver_mask = vehicle_driver_bbox if vehicle_driver_bbox is not False else None
                        process_up_video(video_path, output_dir, person_tracker, 
                                       vehicle_front_angle, vehicle_front_bbox, driver_mask)
                    except Exception as e:
                        print(f"    ✗ 处理失败: {e}")
                        output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("批量处理完成!")
    print(f"输出目录: {output_path}")
    print(f"{'='*70}")

if __name__ == "__main__":
    data_root = "/mnt/mydisk/My_project/TransitVision/data/od_1021"
    output_root = "/mnt/mydisk/My_project/TransitVision/output/reid_crops"
    
    process_dataset(data_root, output_root)

