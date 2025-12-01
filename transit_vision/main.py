"""
TransitVision 主程序
完整的公交客流OD识别系统
"""
import sys
from pathlib import Path
import time
import cv2
import torch
import numpy as np
import yaml
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from transit_vision.threads import MultiLineInputChannel, MultiDirectionLogicChannel
from transit_vision.utils import DeviceConfig, VideoReader, rotate_frame, read_first_frame
from transit_vision.core.reid import ReIDFeatureExtractor, compute_avg_similarity, greedy_matching
from transit_vision.logic import preprocess_front_door, preprocess_rear_door, calculate_occupancy
from transit_vision.core.detection import DoorSegmentor


class SystemConfig:
    """系统配置管理"""
    def __init__(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.cfg = yaml.safe_load(f)
        
        # 加载逻辑配置
        logic_config_path = PROJECT_ROOT / self.cfg['configs']['logic_config']
        with open(logic_config_path, 'r', encoding='utf-8') as f:
            self.logic_cfg = yaml.safe_load(f)
        
        self.data_dir = PROJECT_ROOT / self.cfg['data']['input_dir']
        self.reid_features_dir = PROJECT_ROOT / self.cfg['data']['reid_features_dir']
        
        self.person_model = self.cfg['models']['person_model']
        self.door_model = self.cfg['models']['door_model']
        self.reid_model = self.cfg['models']['reid_model']
        self.reid_cfg = self.cfg['models']['reid_cfg']
        
        self.tracker_config = str(PROJECT_ROOT / self.cfg['configs']['tracker_config'])
        self.device_config = str(PROJECT_ROOT / self.cfg['configs']['device_config'])
        
        self.num_lines = self.cfg['system']['num_lines']
        self.workers_per_line = self.cfg['system']['workers_per_line']
        self.logic_workers = self.cfg['system']['logic_workers']
        self.batch_size = self.cfg['system']['batch_size']
        self.max_stations = self.cfg['system']['max_stations']
        
        self.match_threshold = self.cfg['reid']['match_threshold']
        self.final_match_threshold = self.cfg['reid']['final_match_threshold']
        self.feature_batch_size = self.cfg['reid']['feature_batch_size']
        self.min_frames = self.cfg['reid']['min_frames']
        self.select_frames = self.cfg['reid']['select_frames']
        
        self.recalc_door_up = self.cfg['door']['recalc_per_video_up']
        self.recalc_door_down = self.cfg['door']['recalc_per_video_down']
        
        # 从logic_config加载算法参数
        self.door_conf = self.logic_cfg['alighting_counter']['person_conf']
        self.door_entry_threshold = self.logic_cfg['alighting_counter']['door_entry_threshold']
        self.grace_period = self.logic_cfg['alighting_counter']['grace_period_frames']
        self.max_people = self.logic_cfg['occupancy_analyzer']['max_people']
        
        self.save_features = self.cfg['output']['save_features']
        self.verbose = self.cfg['output']['verbose']
        self.output_dir = PROJECT_ROOT / self.cfg['output'].get('output_dir', 'output/results')


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


class PassengerDatabase:
    """乘客数据库，管理在车乘客的特征和状态"""
    def __init__(self):
        self.boarding_passengers = {}
        self.alighting_passengers = {}
        self.next_boarding_id = 0
        self.next_alighting_id = 0
    
    def add_boarding(self, features, station_id, images, middle_image):
        """添加上车乘客"""
        person_id = self.next_boarding_id
        self.boarding_passengers[person_id] = {
            'features': features,
            'station_id': station_id,
            'status': 'on_bus',
            'images': images,
            'middle_image': middle_image
        }
        self.next_boarding_id += 1
        return person_id
    
    def add_alighting(self, features, station_id, images, middle_image):
        """添加下车乘客"""
        person_id = self.next_alighting_id
        self.alighting_passengers[person_id] = {
            'features': features,
            'station_id': station_id,
            'matched_to': None,
            'match_type': 'unmatched',  # 'immediate', 'final', 'unmatched'
            'images': images,
            'middle_image': middle_image
        }
        self.next_alighting_id += 1
        return person_id
    
    def get_on_bus_passengers(self):
        """获取所有在车上的乘客"""
        return {pid: data for pid, data in self.boarding_passengers.items() if data['status'] == 'on_bus'}
    
    def mark_matched(self, boarding_id, alighting_id, match_type='immediate'):
        """标记匹配成功"""
        self.boarding_passengers[boarding_id]['status'] = 'matched'
        self.alighting_passengers[alighting_id]['matched_to'] = boarding_id
        self.alighting_passengers[alighting_id]['match_type'] = match_type
    
    def get_unmatched_boarding(self):
        """获取未匹配的上车乘客"""
        return {pid: data for pid, data in self.boarding_passengers.items() if data['status'] == 'on_bus'}
    
    def get_unmatched_alighting(self):
        """获取未匹配的下车乘客"""
        return {pid: data for pid, data in self.alighting_passengers.items() if data['matched_to'] is None}


def match_alighting_to_boarding(alighting_features, on_bus_passengers, threshold=0.45):
    """将下车乘客与在车乘客匹配"""
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
        database.mark_matched(board_id, alight_id, match_type='final')
        final_matches.append((alight_id, board_id, sim))
    
    return final_matches


def save_station_results(station_id, station_name, boarding_data, alighting_data, occupancy_info, output_dir):
    """保存单站结果"""
    station_dir = output_dir / f"station_{station_id:02d}_{station_name}"
    station_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存上车信息
    if len(boarding_data) > 0:
        boarding_dir = station_dir / "boarding"
        boarding_dir.mkdir(exist_ok=True)
        
        boarding_info = []
        for person_id, middle_img in boarding_data:
            img_path = boarding_dir / f"person_{person_id}.png"
            cv2.imwrite(str(img_path), middle_img)
            boarding_info.append({
                'person_id': person_id,
                'image': str(img_path.relative_to(output_dir))
            })
        
        with open(station_dir / "boarding_info.json", 'w', encoding='utf-8') as f:
            json.dump(boarding_info, f, ensure_ascii=False, indent=2)
    
    # 保存下车信息
    if len(alighting_data) > 0:
        alighting_dir = station_dir / "alighting"
        alighting_dir.mkdir(exist_ok=True)
        
        alighting_info = []
        for person_id, middle_img, matched, match_status in alighting_data:
            img_path = alighting_dir / f"person_{person_id}.png"
            cv2.imwrite(str(img_path), middle_img)
            alighting_info.append({
                'person_id': person_id,
                'image': str(img_path.relative_to(output_dir)),
                'matched_boarding_id': matched if matched is not None else -1,
                'match_status': match_status  # 'immediate' or 'pending'
            })
        
        with open(station_dir / "alighting_info.json", 'w', encoding='utf-8') as f:
            json.dump(alighting_info, f, ensure_ascii=False, indent=2)
    
    # 保存拥挤度信息
    with open(station_dir / "occupancy_info.json", 'w', encoding='utf-8') as f:
        json.dump(occupancy_info, f, ensure_ascii=False, indent=2)
    
    print(f"  ✓ 站点 {station_id} 结果已保存到: {station_dir}")


def update_final_match_status(output_dir, database, num_stations):
    """更新最终匹配状态"""
    for station_id in range(num_stations):
        station_dirs = list(output_dir.glob(f"station_{station_id:02d}_*"))
        if len(station_dirs) == 0:
            continue
        
        station_dir = station_dirs[0]
        alighting_json = station_dir / "alighting_info.json"
        
        if not alighting_json.exists():
            continue
        
        with open(alighting_json, 'r', encoding='utf-8') as f:
            alighting_info = json.load(f)
        
        # 更新匹配状态
        updated = False
        for person_data in alighting_info:
            person_id = person_data['person_id']
            if person_id in database.alighting_passengers:
                alight_data = database.alighting_passengers[person_id]
                if alight_data['match_type'] == 'final':
                    person_data['match_status'] = 'final'
                    person_data['matched_boarding_id'] = alight_data['matched_to']
                    updated = True
                elif alight_data['match_type'] == 'unmatched':
                    person_data['match_status'] = 'unmatched'
                    updated = True
        
        if updated:
            with open(alighting_json, 'w', encoding='utf-8') as f:
                json.dump(alighting_info, f, ensure_ascii=False, indent=2)


def save_final_summary(database, occupancy_history, station_names, output_dir):
    """保存最终汇总"""
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(exist_ok=True)
    
    # 1. 保存匹配成功的乘客信息
    matched_passengers = []
    for board_id, board_data in database.boarding_passengers.items():
        if board_data['status'] == 'matched':
            # 找到对应的下车乘客
            alight_id = None
            for a_id, a_data in database.alighting_passengers.items():
                if a_data['matched_to'] == board_id:
                    alight_id = a_id
                    break
            
            if alight_id is not None:
                a_data = database.alighting_passengers[alight_id]
                
                # 保存上车图片
                board_img_path = summary_dir / f"matched_boarding_{board_id}.png"
                cv2.imwrite(str(board_img_path), board_data['middle_image'])
                
                # 保存下车图片
                alight_img_path = summary_dir / f"matched_alighting_{alight_id}.png"
                cv2.imwrite(str(alight_img_path), a_data['middle_image'])
                
                matched_passengers.append({
                    'boarding_id': board_id,
                    'boarding_station': board_data['station_id'],
                    'boarding_station_name': station_names[board_data['station_id']],
                    'boarding_image': str(board_img_path.relative_to(output_dir)),
                    'alighting_id': alight_id,
                    'alighting_station': a_data['station_id'],
                    'alighting_station_name': station_names[a_data['station_id']],
                    'alighting_image': str(alight_img_path.relative_to(output_dir)),
                    'match_type': a_data['match_type']
                })
    
    with open(summary_dir / "matched_passengers.json", 'w', encoding='utf-8') as f:
        json.dump(matched_passengers, f, ensure_ascii=False, indent=2)
    
    # 2. 保存站点统计
    station_stats = []
    for station_id in range(len(station_names)):
        boarding_count = sum(1 for p in database.boarding_passengers.values() if p['station_id'] == station_id)
        alighting_count = sum(1 for p in database.alighting_passengers.values() if p['station_id'] == station_id)
        
        station_stats.append({
            'station_id': station_id,
            'station_name': station_names[station_id],
            'boarding_count': boarding_count,
            'alighting_count': alighting_count,
            'occupancy': occupancy_history[station_id] if station_id < len(occupancy_history) else {}
        })
    
    with open(summary_dir / "station_statistics.json", 'w', encoding='utf-8') as f:
        json.dump(station_stats, f, ensure_ascii=False, indent=2)
    
    # 3. 绘制车内人数折线图
    plt.figure(figsize=(12, 6))
    station_indices = list(range(len(occupancy_history)))
    passenger_counts = [occ['passenger_count'] for occ in occupancy_history]
    
    plt.plot(station_indices, passenger_counts, marker='o', linewidth=2, markersize=8)
    plt.xlabel('站点', fontsize=12)
    plt.ylabel('车内人数', fontsize=12)
    plt.title('车内人数变化', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(station_indices, [f"站{i}" for i in station_indices], rotation=45)
    plt.tight_layout()
    plt.savefig(summary_dir / "passenger_count_curve.png", dpi=150)
    plt.close()
    
    # 4. 保存总体统计
    total_stats = {
        'total_boarding': len(database.boarding_passengers),
        'total_alighting': len(database.alighting_passengers),
        'matched_count': sum(1 for p in database.boarding_passengers.values() if p['status'] == 'matched'),
        'unmatched_boarding': len(database.get_unmatched_boarding()),
        'unmatched_alighting': len(database.get_unmatched_alighting()),
        'closure_rate': 0.0
    }
    
    if total_stats['total_boarding'] > 0:
        total_stats['closure_rate'] = total_stats['matched_count'] / total_stats['total_boarding'] * 100
    
    with open(summary_dir / "total_statistics.json", 'w', encoding='utf-8') as f:
        json.dump(total_stats, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 最终汇总已保存到: {summary_dir}")
    print(f"  - 匹配成功乘客: {total_stats['matched_count']}人")
    print(f"  - 车内人数折线图: passenger_count_curve.png")
    print(f"  - 站点统计: station_statistics.json")


def run_system(config_path):
    """运行完整系统"""
    cfg = SystemConfig(config_path)
    
    print("=" * 70)
    print("TransitVision 公交客流OD识别系统")
    print("=" * 70)
    print(f"数据目录: {cfg.data_dir}")
    print(f"输出目录: {cfg.output_dir}")
    print(f"并行线路数: {cfg.num_lines}")
    print(f"最大站点数: {cfg.max_stations}")
    print(f"匹配阈值: {cfg.match_threshold}")
    print(f"最终匹配阈值: {cfg.final_match_threshold}")
    print()
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = cfg.output_dir / f"run_{timestamp}"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    
    device_cfg = DeviceConfig(cfg.device_config)
    
    print("加载ReID模型...")
    reid_extractor = ReIDFeatureExtractor(cfg.reid_model, cfg.reid_cfg)
    print("✓ ReID模型已加载")
    
    print("\n创建输入通道...")
    input_channel = MultiLineInputChannel(num_lines=cfg.num_lines, workers_per_line=cfg.workers_per_line)
    stations = input_channel.load_and_replicate_data(cfg.data_dir, cfg.max_stations)
    print(f"✓ 加载 {len(stations)} 个站点")
    
    print("\n创建逻辑通道...")
    logic_channel = MultiDirectionLogicChannel(
        cfg.person_model, cfg.tracker_config, cfg.door_model, device_cfg,
        batch_size=cfg.batch_size, num_workers=cfg.logic_workers,
        recalc_door_up=cfg.recalc_door_up, recalc_door_down=cfg.recalc_door_down
    )
    print(f"✓ 逻辑通道已创建 (上车门框重算:{cfg.recalc_door_up}, 下车门框重算:{cfg.recalc_door_down})")
    
    print("\n启动所有通道...")
    input_channel.start_all()
    logic_channel.start()
    print("✓ 所有通道已启动")
    
    print(f"\n开始处理站点数据...")
    print("=" * 70)
    
    line_databases = [PassengerDatabase() for _ in range(cfg.num_lines)]
    
    # 记录每条线路的车内人数和拥挤度历史
    occupancy_histories = [[] for _ in range(cfg.num_lines)]
    station_names_list = [[] for _ in range(cfg.num_lines)]
    
    for station in stations:
        station_id = station.station_id - 1
        print(f"\n[站点 {station.station_id}: {station.station_name}]")
        
        input_channel.submit_station_to_all_lines(station)
        
        time.sleep(0.5)
        
        initial_logic_stats = logic_channel.get_total_stats()
        initial_logic_processed = initial_logic_stats['total_processed']
        
        task_map = {}
        submitted = 0
        for line_id in range(cfg.num_lines):
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
        
        for line_id in range(cfg.num_lines):
            db = line_databases[line_id]
            station_names_list[line_id].append(station.station_name)
            
            boarding_data = []
            alighting_data = []
            
            # 处理下车乘客
            for result in down_results:
                task = result['task']
                passengers = result['valid_passengers']
                
                task_line_id = None
                for lid in range(cfg.num_lines):
                    if f"{lid}_down" in task_map and task_map[f"{lid}_down"].bus_id == task.bus_id:
                        task_line_id = lid
                        break
                
                if task_line_id != line_id:
                    continue
                
                for track_id, person in passengers.items():
                    images = extract_person_images(task.video_path, person, rotation_angle=None)
                    
                    if len(images) == 0:
                        continue
                    
                    middle_image = images[len(images) // 2]
                    
                    features = reid_extractor.extract_batch(images, batch_size=cfg.feature_batch_size)
                    if features is None:
                        continue
                    
                    on_bus = db.get_on_bus_passengers()
                    matched_boarding_id, max_sim = match_alighting_to_boarding(features, on_bus, threshold=cfg.match_threshold)
                    
                    alight_id = db.add_alighting(features, station_id, images, middle_image)
                    
                    if matched_boarding_id is not None:
                        db.mark_matched(matched_boarding_id, alight_id, match_type='immediate')
                        alighting_data.append((alight_id, middle_image, matched_boarding_id, 'immediate'))
                    else:
                        alighting_data.append((alight_id, middle_image, None, 'pending'))
            
            # 处理上车乘客
            for result in up_results:
                task = result['task']
                passengers = result['valid_passengers']
                door_info = result.get('door_info')
                
                task_line_id = None
                for lid in range(cfg.num_lines):
                    if f"{lid}_up" in task_map and task_map[f"{lid}_up"].bus_id == task.bus_id:
                        task_line_id = lid
                        break
                
                if task_line_id != line_id:
                    continue
                
                # 从door_info中获取旋转角度
                rotation_angle = None
                if door_info and 'angle' in door_info:
                    rotation_angle = -door_info['angle']
                
                for track_id, person in passengers.items():
                    images = extract_person_images(task.video_path, person, rotation_angle=rotation_angle)
                    
                    if len(images) == 0:
                        continue
                    
                    middle_image = images[len(images) // 2]
                    
                    features = reid_extractor.extract_batch(images, batch_size=cfg.feature_batch_size)
                    if features is None:
                        continue
                    
                    board_id = db.add_boarding(features, station_id, images, middle_image)
                    boarding_data.append((board_id, middle_image))
            
            # 计算车内人数和拥挤度
            current_on_bus = len(db.get_on_bus_passengers())
            occupancy_rate = current_on_bus / cfg.max_people
            
            # 简单的拥挤度判断
            if occupancy_rate < 0.3:
                occupancy_level = "空闲"
                occupancy_color = "green"
            elif occupancy_rate < 0.7:
                occupancy_level = "舒适"
                occupancy_color = "yellow"
            else:
                occupancy_level = "拥挤"
                occupancy_color = "red"
            
            occupancy_info = {
                'passenger_count': current_on_bus,
                'occupancy_rate': occupancy_rate,
                'occupancy_level': occupancy_level,
                'occupancy_color': occupancy_color,
                'boarding_count': len(boarding_data),
                'alighting_count': len(alighting_data)
            }
            
            occupancy_histories[line_id].append(occupancy_info)
            
            # 保存本站结果
            line_output_dir = run_output_dir / f"line_{line_id}"
            save_station_results(station_id, station.station_name, boarding_data, alighting_data, occupancy_info, line_output_dir)
            
            if len(boarding_data) > 0 or len(alighting_data) > 0:
                print(f"  线路{line_id}: 上车{len(boarding_data)}人, 下车{len(alighting_data)}人, 车内{current_on_bus}人, 拥挤度:{occupancy_level}")
    
    print("\n停止所有通道...")
    input_channel.stop_all()
    logic_channel.stop()
    print("✓ 所有通道已停止")
    
    print(f"\n{'='*70}")
    print("最终交叉匹配")
    print(f"{'='*70}")
    
    for line_id in range(cfg.num_lines):
        db = line_databases[line_id]
        
        print(f"\n线路{line_id}:")
        
        unmatched_boarding_before = len(db.get_unmatched_boarding())
        unmatched_alighting_before = len(db.get_unmatched_alighting())
        
        print(f"  交叉匹配前: 未匹配上车{unmatched_boarding_before}人, 未匹配下车{unmatched_alighting_before}人")
        
        final_matches = final_cross_matching(db, threshold=cfg.final_match_threshold)
        
        if len(final_matches) > 0:
            print(f"  交叉匹配成功: {len(final_matches)}对")
        
        # 更新所有站点的最终匹配状态
        line_output_dir = run_output_dir / f"line_{line_id}"
        update_final_match_status(line_output_dir, db, len(station_names_list[line_id]))
        
        # 保存最终汇总
        save_final_summary(db, occupancy_histories[line_id], station_names_list[line_id], line_output_dir)
        
        unmatched_boarding_after = len(db.get_unmatched_boarding())
        unmatched_alighting_after = len(db.get_unmatched_alighting())
        
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
    print("系统运行完成")
    print(f"所有结果已保存到: {run_output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='TransitVision 公交客流OD识别系统')
    parser.add_argument('--config', type=str, 
                       default='configs/system_config.yaml',
                       help='系统配置文件路径')
    
    args = parser.parse_args()
    
    config_path = PROJECT_ROOT / args.config
    
    if not config_path.exists():
        print(f"错误: 配置文件不存在: {config_path}")
        exit(1)
    
    run_system(config_path)
