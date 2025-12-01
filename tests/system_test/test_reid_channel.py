"""
ReID通道系统测试
测试完整流程：输入 → 逻辑通道 → ReID匹配
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import time
import cv2
import torch
import numpy as np
import yaml
from collections import defaultdict
from transit_vision.threads import MultiLineInputChannel, MultiDirectionLogicChannel
from transit_vision.utils import DeviceConfig, VideoReader, rotate_frame, read_first_frame
from transit_vision.core.reid import ReIDFeatureExtractor, compute_avg_similarity, greedy_matching
from transit_vision.logic import preprocess_front_door, preprocess_rear_door
from transit_vision.core.detection import DoorSegmentor

PROJECT_ROOT = Path(__file__).parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "close_loop_od"
REID_DATA_DIR = PROJECT_ROOT / "data" / "reid_features"
PERSON_MODEL = "/mnt/mydisk/My_project/bus_down/yolo11x-seg.pt"
DOOR_MODEL = "/mnt/mydisk/My_project/bus_down/front_door.pt"
TRACKER_CONFIG = str(PROJECT_ROOT / "configs" / "botsort_seg.yaml")
DEVICE_CONFIG = str(PROJECT_ROOT / "configs" / "device_debug.yaml")
LOGIC_CONFIG = str(PROJECT_ROOT / "configs" / "logic_config.yaml")
REID_MODEL = "/mnt/mydisk/My_project/TransitVision/ckpt/pose2id/transformer_20.pth"
REID_CFG = "/mnt/mydisk/My_project/TransitVision/tests/pose2id_scheme/Pose2ID/IPG/cfg_transreid.pkl"

# 加载逻辑配置
with open(LOGIC_CONFIG, 'r', encoding='utf-8') as f:
    logic_cfg = yaml.safe_load(f)

NUM_LINES = 1
BATCH_SIZE = 64
NUM_WORKERS = 2
MAX_STATIONS = 21

MATCH_THRESHOLD = 0.45
FINAL_MATCH_THRESHOLD = 0.40
RECALC_DOOR_PER_VIDEO_UP = False
RECALC_DOOR_PER_VIDEO_DOWN = False
DOOR_DETECTION_CONF = logic_cfg['alighting_counter']['person_conf']

def select_frames(total_frames, n=7):
    """从总帧数中均匀选择n帧,然后掐头去尾保留中间n-2帧"""
    if total_frames <= n:
        indices = list(range(total_frames))
    else:
        indices = np.linspace(0, total_frames-1, n, dtype=int).tolist()
    
    if len(indices) > 2:
        return indices[1:-1]
    return indices


def extract_person_images(video_path, person, rotation_angle=None):
    """提取单个人的bbox截图列表"""
    if len(person.frames) < 7:
        return []
    
    selected_indices = select_frames(len(person.frames))
    images = []
    
    with VideoReader(video_path) as reader:
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
                images.append(crop)
    
    return images


def save_person_images(images, line_id, station_id, person_idx, direction, output_dir):
    """保存人员图片到文件夹"""
    person_dir = output_dir / f"line_{line_id}" / direction / f"station_{station_id}" / f"person_{person_idx}"
    person_dir.mkdir(parents=True, exist_ok=True)
    
    for i, img in enumerate(images):
        save_path = person_dir / f"{i}.png"
        cv2.imwrite(str(save_path), img)
    
    return person_dir


class PassengerDatabase:
    """乘客数据库，管理在车乘客的特征和状态"""
    def __init__(self):
        self.boarding_passengers = {}  # {person_idx: {'features': tensor, 'station_id': int, 'status': str, 'images_dir': Path}}
        self.alighting_passengers = {}  # {person_idx: {'features': tensor, 'station_id': int, 'matched_to': int or None, 'images_dir': Path}}
        self.next_boarding_id = 0
        self.next_alighting_id = 0
    
    def add_boarding(self, features, station_id, images_dir):
        """添加上车乘客"""
        person_id = self.next_boarding_id
        self.boarding_passengers[person_id] = {
            'features': features,
            'station_id': station_id,
            'status': 'on_bus',
            'images_dir': images_dir
        }
        self.next_boarding_id += 1
        return person_id
    
    def add_alighting(self, features, station_id, images_dir):
        """添加下车乘客"""
        person_id = self.next_alighting_id
        self.alighting_passengers[person_id] = {
            'features': features,
            'station_id': station_id,
            'matched_to': None,
            'images_dir': images_dir
        }
        self.next_alighting_id += 1
        return person_id
    
    def get_on_bus_passengers(self):
        """获取所有在车上的乘客"""
        return {pid: data for pid, data in self.boarding_passengers.items() if data['status'] == 'on_bus'}
    
    def mark_matched(self, boarding_id, alighting_id):
        """标记匹配成功"""
        self.boarding_passengers[boarding_id]['status'] = 'matched'
        self.alighting_passengers[alighting_id]['matched_to'] = boarding_id
    
    def get_unmatched_boarding(self):
        """获取未匹配的上车乘客"""
        return {pid: data for pid, data in self.boarding_passengers.items() if data['status'] == 'on_bus'}
    
    def get_unmatched_alighting(self):
        """获取未匹配的下车乘客"""
        return {pid: data for pid, data in self.alighting_passengers.items() if data['matched_to'] is None}


def match_alighting_to_boarding(alighting_features, on_bus_passengers, threshold=0.45):
    """将下车乘客与在车乘客匹配
    
    Returns:
        matched_boarding_id: 匹配到的上车乘客ID，None表示未匹配
        max_similarity: 最高相似度
    """
    if len(on_bus_passengers) == 0:
        return None, 0.0
    
    max_sim = 0.0
    best_match = None
    
    for boarding_id, boarding_data in on_bus_passengers.items():
        boarding_features = boarding_data['features']
        sim = compute_avg_similarity(alighting_features, boarding_features)
        
        if sim > max_sim:
            max_sim = sim
            best_match = boarding_id
    
    if max_sim >= threshold:
        return best_match, max_sim
    
    return None, max_sim


def final_cross_matching(database, threshold=0.40):
    """最终交叉匹配所有未匹配的乘客"""
    unmatched_boarding = database.get_unmatched_boarding()
    unmatched_alighting = database.get_unmatched_alighting()
    
    if len(unmatched_boarding) == 0 or len(unmatched_alighting) == 0:
        return []
    
    boarding_ids = list(unmatched_boarding.keys())
    alighting_ids = list(unmatched_alighting.keys())
    
    boarding_feats = torch.cat([unmatched_boarding[pid]['features'] for pid in boarding_ids], dim=0)
    alighting_feats = torch.cat([unmatched_alighting[pid]['features'] for pid in alighting_ids], dim=0)
    
    similarity_matrix = torch.matmul(alighting_feats, boarding_feats.t()).numpy()
    
    matches, _, _ = greedy_matching(similarity_matrix, threshold=threshold)
    
    final_matches = []
    for alight_idx, board_idx, sim in matches:
        alight_id = alighting_ids[alight_idx]
        board_id = boarding_ids[board_idx]
        database.mark_matched(board_id, alight_id)
        final_matches.append((alight_id, board_id, sim))
    
    return final_matches


def test_reid_channel():
    print("=" * 70)
    print("ReID通道系统测试")
    print("=" * 70)
    print(f"数据目录: {DATA_DIR}")
    print(f"ReID特征保存: {REID_DATA_DIR}")
    print(f"并行线路数: {NUM_LINES}")
    print(f"测试站点数: {MAX_STATIONS}")
    print(f"匹配阈值: {MATCH_THRESHOLD}")
    print(f"最终匹配阈值: {FINAL_MATCH_THRESHOLD}")
    print(f"上车门框重算: {RECALC_DOOR_PER_VIDEO_UP}")
    print(f"下车门框重算: {RECALC_DOOR_PER_VIDEO_DOWN}")
    print()
    
    REID_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    device_cfg = DeviceConfig(DEVICE_CONFIG)
    
    print("加载ReID模型...")
    reid_extractor = ReIDFeatureExtractor(REID_MODEL, REID_CFG)
    print("✓ ReID模型已加载")
    
    print("\n创建输入通道...")
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
    
    door_seg = DoorSegmentor(DOOR_MODEL, device_cfg)
    
    line_databases = [PassengerDatabase() for _ in range(NUM_LINES)]
    
    door_cache_up = {}
    door_cache_down = {}
    
    for station in stations:
        station_id = station.station_id - 1
        print(f"\n[站点 {station.station_id}: {station.station_name}]")
        
        input_channel.submit_station_to_all_lines(station)
        
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
        
        wait_start = time.time()
        while True:
            stats = logic_channel.get_total_stats()
            if stats['total_processed'] >= initial_logic_processed + submitted:
                break
            time.sleep(0.5)
        
        results = []
        for _ in range(submitted):
            for direction in ['up', 'down']:
                result = logic_channel.get_result(direction, timeout=0.1)
                if result:
                    results.append(result)
        
        down_results = [r for r in results if r['task'].direction == 'down']
        up_results = [r for r in results if r['task'].direction == 'up']
        
        for line_id in range(NUM_LINES):
            db = line_databases[line_id]
            
            boarding_ids = []
            alighting_ids = []
            
            for result in down_results:
                task = result['task']
                passengers = result['valid_passengers']
                
                task_line_id = None
                for lid in range(NUM_LINES):
                    if f"{lid}_down" in task_map and task_map[f"{lid}_down"].bus_id == task.bus_id:
                        task_line_id = lid
                        break
                
                if task_line_id != line_id:
                    continue
                
                door_mask = None
                if RECALC_DOOR_PER_VIDEO_DOWN:
                    from transit_vision.utils import read_first_frame
                    from transit_vision.logic import preprocess_rear_door
                    first_frame = read_first_frame(task.video_path)
                    door = door_seg.detect(first_frame, conf=DOOR_DETECTION_CONF)
                    if door:
                        door_mask = preprocess_rear_door(door)
                else:
                    cache_key = f"{line_id}_down"
                    if cache_key not in door_cache_down:
                        from transit_vision.utils import read_first_frame
                        from transit_vision.logic import preprocess_rear_door
                        first_frame = read_first_frame(task.video_path)
                        door = door_seg.detect(first_frame, conf=DOOR_DETECTION_CONF)
                        if door:
                            door_mask = preprocess_rear_door(door)
                            door_cache_down[cache_key] = door_mask
                    else:
                        door_mask = door_cache_down[cache_key]
                
                for track_id, person in passengers.items():
                    images = extract_person_images(task.video_path, person, rotation_angle=None)
                    
                    if len(images) == 0:
                        continue
                    
                    person_idx = db.next_alighting_id
                    images_dir = save_person_images(images, line_id, station_id, person_idx, 'alighting', REID_DATA_DIR)
                    
                    features = reid_extractor.extract_batch(images, batch_size=8)
                    if features is None:
                        continue
                    
                    on_bus = db.get_on_bus_passengers()
                    matched_boarding_id, max_sim = match_alighting_to_boarding(features, on_bus, threshold=MATCH_THRESHOLD)
                    
                    alight_id = db.add_alighting(features, station_id, images_dir)
                    
                    if matched_boarding_id is not None:
                        db.mark_matched(matched_boarding_id, alight_id)
                        alighting_ids.append(f"{alight_id}(首次匹配→{matched_boarding_id}, {max_sim:.3f})")
                    else:
                        alighting_ids.append(f"{alight_id}(未匹配, 最高{max_sim:.3f})")
            
            for result in up_results:
                task = result['task']
                passengers = result['valid_passengers']
                
                task_line_id = None
                for lid in range(NUM_LINES):
                    if f"{lid}_up" in task_map and task_map[f"{lid}_up"].bus_id == task.bus_id:
                        task_line_id = lid
                        break
                
                if task_line_id != line_id:
                    continue
                
                rotation_angle = None
                door_bbox = None
                if RECALC_DOOR_PER_VIDEO_UP:
                    from transit_vision.utils import read_first_frame
                    first_frame = read_first_frame(task.video_path)
                    door = door_seg.detect(first_frame, conf=DOOR_DETECTION_CONF)
                    if door:
                        angle, bbox = preprocess_front_door(door, first_frame)
                        if angle is not None:
                            rotation_angle = -angle
                            door_bbox = bbox
                else:
                    cache_key = f"{line_id}_up"
                    if cache_key not in door_cache_up:
                        from transit_vision.utils import read_first_frame
                        first_frame = read_first_frame(task.video_path)
                        door = door_seg.detect(first_frame, conf=DOOR_DETECTION_CONF)
                        if door:
                            angle, bbox = preprocess_front_door(door, first_frame)
                            if angle is not None:
                                rotation_angle = -angle
                                door_bbox = bbox
                                door_cache_up[cache_key] = (rotation_angle, door_bbox)
                    else:
                        rotation_angle, door_bbox = door_cache_up[cache_key]
                
                for track_id, person in passengers.items():
                    images = extract_person_images(task.video_path, person, rotation_angle=rotation_angle)
                    
                    if len(images) == 0:
                        continue
                    
                    person_idx = db.next_boarding_id
                    images_dir = save_person_images(images, line_id, station_id, person_idx, 'boarding', REID_DATA_DIR)
                    
                    features = reid_extractor.extract_batch(images, batch_size=8)
                    if features is None:
                        continue
                    
                    board_id = db.add_boarding(features, station_id, images_dir)
                    boarding_ids.append(f"{board_id}")
            
            if len(boarding_ids) > 0 or len(alighting_ids) > 0:
                print(f"  线路{line_id}:")
                if len(boarding_ids) > 0:
                    print(f"    上车: {', '.join(boarding_ids)}")
                if len(alighting_ids) > 0:
                    print(f"    下车: {', '.join(alighting_ids)}")
    
    print("\n停止所有通道...")
    input_channel.stop_all()
    logic_channel.stop()
    print("✓ 所有通道已停止")
    
    print(f"\n{'='*70}")
    print("最终交叉匹配")
    print(f"{'='*70}")
    
    for line_id in range(NUM_LINES):
        db = line_databases[line_id]
        
        print(f"\n线路{line_id}:")
        
        unmatched_boarding_before = len(db.get_unmatched_boarding())
        unmatched_alighting_before = len(db.get_unmatched_alighting())
        
        print(f"  交叉匹配前: 未匹配上车{unmatched_boarding_before}人, 未匹配下车{unmatched_alighting_before}人")
        
        final_matches = final_cross_matching(db, threshold=FINAL_MATCH_THRESHOLD)
        
        if len(final_matches) > 0:
            print(f"  交叉匹配结果:")
            for alight_id, board_id, sim in final_matches:
                board_station = db.boarding_passengers[board_id]['station_id']
                alight_station = db.alighting_passengers[alight_id]['station_id']
                print(f"    下车{alight_id}(站点{alight_station}) ← 上车{board_id}(站点{board_station}), 相似度{sim:.3f}")
        
        unmatched_boarding_after = len(db.get_unmatched_boarding())
        unmatched_alighting_after = len(db.get_unmatched_alighting())
        
        print(f"  交叉匹配后: 未匹配上车{unmatched_boarding_after}人, 未匹配下车{unmatched_alighting_after}人")
        
        total_boarding = len(db.boarding_passengers)
        total_alighting = len(db.alighting_passengers)
        matched_count = total_boarding - unmatched_boarding_after
        
        print(f"\n  最终统计:")
        print(f"    总上车: {total_boarding}人")
        print(f"    总下车: {total_alighting}人")
        print(f"    成功闭环: {matched_count}人")
        print(f"    未匹配上车: {unmatched_boarding_after}人")
        print(f"    未匹配下车: {unmatched_alighting_after}人")
        
        if total_boarding > 0:
            closure_rate = matched_count / total_boarding * 100
            print(f"    闭环率: {closure_rate:.1f}%")
    
    print(f"\n{'='*70}")
    print("测试完成")
    print(f"{'='*70}")

if __name__ == "__main__":
    test_reid_channel()

