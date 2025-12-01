import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transit_vision.utils import VideoReader, read_first_frame, DeviceConfig
from transit_vision.core.detection import PersonSegTracker, DoorSegmentor
from transit_vision.logic import (
    preprocess_rear_door, 
    filter_alighting_passengers,
    get_upper_region_mask,
    get_overlap_region_mask
)
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

DOOR_VIDEO = "/mnt/mydisk/My_project/TransitVision/data/close_loop_od/13_down.mp4"
TEST_VIDEO = "/mnt/mydisk/My_project/TransitVision/data/close_loop_od/13_down.mp4"
PERSON_MODEL = "/mnt/mydisk/My_project/bus_down/yolo11x-seg.pt"
DOOR_MODEL = "/mnt/mydisk/My_project/bus_down/front_door.pt"
TRACKER_CONFIG = str(Path(__file__).parent.parent.parent / "configs" / "botsort_seg.yaml")
DEVICE_CONFIG = str(Path(__file__).parent.parent.parent / "configs" / "device_debug.yaml")
OUTPUT_DIR = Path(__file__).parent / "output"

OVERLAP_THRESHOLD = 0.5
GRACE_PERIOD = 6

def polygon_to_local_mask(polygon, box):
    """将多边形转换为局部掩码"""
    x1, y1, x2, y2 = map(int, box)
    w, h = x2 - x1, y2 - y1
    
    if w <= 0 or h <= 0:
        return None
    
    if polygon.ndim == 3:
        polygon = polygon.squeeze()
    
    local_poly = polygon - np.array([x1, y1])
    local_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(local_mask, [local_poly.astype(np.int32)], 255)
    
    return local_mask

def calculate_overlap_ratio(person_polygon, box, door_mask):
    """计算上4/5区域与门框的重叠比例"""
    if person_polygon is None or door_mask is None:
        return 0.0
    
    if len(person_polygon) < 3:
        return 0.0
    
    x1, y1, x2, y2 = map(int, box)
    h = y2 - y1
    upper_h = int(h * 4 / 5)
    
    if upper_h <= 0:
        return 0.0
    
    upper_box = [x1, y1, x2, y1 + upper_h]
    
    person_local = polygon_to_local_mask(person_polygon, upper_box)
    if person_local is None:
        return 0.0
    
    # 提取门掩码的局部区域
    door_local = door_mask[y1:y1+upper_h, x1:x2]
    if door_local.size == 0 or door_local.shape != person_local.shape:
        return 0.0
    
    intersection = cv2.bitwise_and(person_local, door_local)
    overlap_pixels = cv2.countNonZero(intersection)
    person_pixels = cv2.countNonZero(person_local)
    
    if person_pixels == 0:
        return 0.0
    
    return overlap_pixels / person_pixels

def check_box_overlap_with_mask(body_box, door_mask):
    """检查bbox是否与门掩码重叠"""
    if body_box is None or door_mask is None:
        return False
    
    x1, y1, x2, y2 = map(int, body_box)
    
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(door_mask.shape[1], x2)
    y2 = min(door_mask.shape[0], y2)
    
    if x2 <= x1 or y2 <= y1:
        return False
    
    box_region = door_mask[y1:y2, x1:x2]
    return np.any(box_region > 0)

def analyze_alighting_logic(tracks, door_mask, threshold=0.5, grace_period=6):
    """详细分析下车逻辑，返回每个ID的判定信息"""
    track_state = {}
    valid = {}
    analysis_info = {}
    
    all_frames = set()
    for person in tracks.values():
        all_frames.update(person.frames)
    
    if not all_frames:
        return {}, {}
    
    all_frames = sorted(all_frames)
    
    for frame_idx in all_frames:
        current_ids = set()
        
        for track_id, person in tracks.items():
            if frame_idx in person.frames:
                current_ids.add(track_id)
                
                if track_id not in track_state:
                    track_state[track_id] = {
                        'has_intent': False,
                        'has_counted': False,
                        'last_seen_frame': frame_idx,
                        'last_box': None,
                        'inside_history': []
                    }
                    analysis_info[track_id] = {
                        'overlap_ratios': [],
                        'frame_indices': [],
                        'has_intent': False,
                        'intent_frame': None,
                        'disappeared_frames': 0,
                        'last_box_overlap': False,
                        'reliable': False,
                        'min_frames': False,
                        'final_decision': 'unknown',
                        'fail_reason': []
                    }
                
                state = track_state[track_id]
                info = analysis_info[track_id]
                state['last_seen_frame'] = frame_idx
                
                idx = person.frames.index(frame_idx)
                box = person.boxes[idx]
                state['last_box'] = box
                
                if len(person.mask_polygons) > idx:
                    person_polygon = person.mask_polygons[idx]
                    
                    if person_polygon is not None and len(person_polygon) > 0:
                        overlap_ratio = calculate_overlap_ratio(person_polygon, box, door_mask)
                        info['overlap_ratios'].append(overlap_ratio)
                        info['frame_indices'].append(frame_idx)
                        
                        is_in = overlap_ratio >= threshold
                        state['inside_history'].append(1 if is_in else 0)
                        
                        if len(state['inside_history']) > 20:
                            state['inside_history'].pop(0)
                        
                        if not state['has_intent'] and len(state['inside_history']) >= 2 and \
                           state['inside_history'][-2] == 0 and \
                           state['inside_history'][-1] == 1:
                            state['has_intent'] = True
                            info['has_intent'] = True
                            info['intent_frame'] = frame_idx
        
        for track_id, state in track_state.items():
            if state['has_counted'] or not state['has_intent']:
                continue
            
            if track_id not in current_ids:
                disappeared = frame_idx - state['last_seen_frame']
                analysis_info[track_id]['disappeared_frames'] = disappeared
                
                if disappeared >= grace_period:
                    last_box_overlap = check_box_overlap_with_mask(state['last_box'], door_mask)
                    analysis_info[track_id]['last_box_overlap'] = last_box_overlap
                    
                    if last_box_overlap:
                        person = tracks[track_id]
                        frames = person.frames
                        
                        analysis_info[track_id]['min_frames'] = len(frames) >= 5
                        
                        if len(frames) < 5:
                            analysis_info[track_id]['final_decision'] = 'rejected'
                            analysis_info[track_id]['fail_reason'].append(f'帧数不足({len(frames)}<5)')
                            state['has_counted'] = True
                            continue
                        
                        count = 1
                        reliable = False
                        for i in range(1, len(frames)):
                            if frames[i] - frames[i-1] <= 2:
                                count += 1
                                if count >= 5:
                                    reliable = True
                                    break
                            else:
                                count = 1
                        
                        analysis_info[track_id]['reliable'] = reliable
                        
                        if reliable:
                            person.trigger_frame = state['last_seen_frame']
                            valid[track_id] = person
                            state['has_counted'] = True
                            analysis_info[track_id]['final_decision'] = 'accepted'
                        else:
                            analysis_info[track_id]['final_decision'] = 'rejected'
                            analysis_info[track_id]['fail_reason'].append('连续性不足(无5帧连续)')
                            state['has_counted'] = True
                    else:
                        analysis_info[track_id]['final_decision'] = 'rejected'
                        analysis_info[track_id]['fail_reason'].append('最后位置未与门框重叠')
                        state['has_counted'] = True
    
    for track_id in analysis_info:
        if analysis_info[track_id]['final_decision'] == 'unknown':
            if not analysis_info[track_id]['has_intent']:
                analysis_info[track_id]['final_decision'] = 'rejected'
                analysis_info[track_id]['fail_reason'].append('未检测到进门意图')
            elif analysis_info[track_id]['disappeared_frames'] < grace_period:
                analysis_info[track_id]['final_decision'] = 'rejected'
                analysis_info[track_id]['fail_reason'].append(f'消失时间不足({analysis_info[track_id]["disappeared_frames"]}<{grace_period})')
    
    return valid, analysis_info

def plot_overlap_analysis(analysis_info, total_frames, threshold, output_path):
    """绘制重叠比例随帧数变化的散点图"""
    ids_with_overlap = {tid: info for tid, info in analysis_info.items() 
                        if len(info['overlap_ratios']) > 0}
    
    if not ids_with_overlap:
        print("  ⚠ 没有ID与门框重叠")
        return
    
    plt.figure(figsize=(14, 8))
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(ids_with_overlap)))
    
    for idx, (track_id, info) in enumerate(sorted(ids_with_overlap.items())):
        frames = info['frame_indices']
        ratios = info['overlap_ratios']
        
        label = f"ID {track_id}"
        if info['final_decision'] == 'accepted':
            label += " (下车)"
        else:
            label += " (非下车)"
        
        plt.scatter(frames, ratios, c=[colors[idx]], label=label, s=50, alpha=0.7)
    
    plt.axhline(y=threshold, color='r', linestyle='--', linewidth=2, label=f'阈值 ({threshold})')
    
    plt.xlabel('帧数', fontsize=12)
    plt.ylabel('重叠占比 (上4/5区域)', fontsize=12)
    plt.title('ID重叠占比随帧数变化', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, total_frames)
    plt.ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ 重叠分析图: {output_path}")

def generate_decision_summary(analysis_info, output_path):
    """生成判定摘要图"""
    ids_with_overlap = {tid: info for tid, info in analysis_info.items() 
                        if len(info['overlap_ratios']) > 0}
    
    if not ids_with_overlap:
        return
    
    accepted_ids = [tid for tid, info in ids_with_overlap.items() 
                    if info['final_decision'] == 'accepted']
    rejected_ids = [tid for tid, info in ids_with_overlap.items() 
                    if info['final_decision'] == 'rejected']
    
    fig, ax = plt.subplots(figsize=(12, max(8, len(ids_with_overlap) * 0.5)))
    
    y_pos = 0
    y_labels = []
    y_ticks = []
    
    ax.text(0.5, y_pos, '下车判定摘要', fontsize=16, fontweight='bold', 
            ha='center', transform=ax.transData)
    y_pos -= 1.5
    
    if accepted_ids:
        ax.text(0.1, y_pos, f'✓ 判定下车 ({len(accepted_ids)}人):', 
                fontsize=12, fontweight='bold', color='green', transform=ax.transData)
        y_pos -= 1
        
        for tid in sorted(accepted_ids):
            info = ids_with_overlap[tid]
            text = f"  ID {tid}: 进门意图(帧{info['intent_frame']}), "
            text += f"消失{info['disappeared_frames']}帧, 连续性✓"
            ax.text(0.15, y_pos, text, fontsize=10, transform=ax.transData)
            y_pos -= 0.8
    
    y_pos -= 0.5
    
    if rejected_ids:
        ax.text(0.1, y_pos, f'✗ 判定非下车 ({len(rejected_ids)}人):', 
                fontsize=12, fontweight='bold', color='red', transform=ax.transData)
        y_pos -= 1
        
        for tid in sorted(rejected_ids):
            info = ids_with_overlap[tid]
            reasons = ', '.join(info['fail_reason']) if info['fail_reason'] else '未知原因'
            text = f"  ID {tid}: {reasons}"
            ax.text(0.15, y_pos, text, fontsize=10, transform=ax.transData)
            y_pos -= 0.8
    
    ax.set_xlim(0, 1)
    ax.set_ylim(y_pos - 1, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ 判定摘要图: {output_path}")

def test_alighting_logic():
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print("=== Alighting Logic Test ===")
    
    device_cfg = DeviceConfig(DEVICE_CONFIG)
    
    # 1. 门检测
    print(f"\n1. Door detection from: {Path(DOOR_VIDEO).name}")
    door_seg = DoorSegmentor(DOOR_MODEL, device_cfg)
    door_frame = read_first_frame(DOOR_VIDEO)
    door = door_seg.detect(door_frame)
    
    if door is None:
        print("✗ No door")
        return
    
    door_mask = preprocess_rear_door(door)
    print(f"✓ Door mask area: {np.sum(door_mask > 0)}")
    
    # 2. 人员追踪
    print(f"\n2. Person tracking from: {Path(TEST_VIDEO).name}")
    person_tracker = PersonSegTracker(PERSON_MODEL, TRACKER_CONFIG, device_cfg)
    all_tracks = person_tracker.track_video(VideoReader(TEST_VIDEO))
    
    print(f"✓ Total tracks: {len(all_tracks)}")
    
    # 获取视频总帧数
    with VideoReader(TEST_VIDEO) as reader:
        total_frames = reader.frame_count
    
    # 3. 下客判定（详细分析）
    print(f"\n3. Alighting detection with detailed analysis...")
    alighting, analysis_info = analyze_alighting_logic(all_tracks, door_mask, 
                                                       threshold=OVERLAP_THRESHOLD, 
                                                       grace_period=GRACE_PERIOD)
    print(f"✓ Alighting passengers: {len(alighting)}")
    
    for tid, person in sorted(alighting.items()):
        print(f"  ID {tid}: {len(person)} frames")
    
    # 4. 生成分析图表
    print(f"\n4. Generating analysis plots...")
    
    plot_overlap_analysis(analysis_info, total_frames, OVERLAP_THRESHOLD, 
                         OUTPUT_DIR / "overlap_analysis.png")
    
    generate_decision_summary(analysis_info, OUTPUT_DIR / "decision_summary.png")
    
    # 5. 可视化视频（标记所有ID + 上4/5区域 + 重叠区域）
    print(f"\n5. Generating visualization video...")
    trigger_frames = {person.trigger_frame: tid for tid, person in alighting.items() 
                      if person.trigger_frame is not None}
    
    with VideoReader(TEST_VIDEO) as reader:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_path = str(OUTPUT_DIR / "alighting_result.mp4")
        out = cv2.VideoWriter(out_path, fourcc, reader.fps, (reader.width, reader.height))
        
        frame_idx = 0
        for frame in reader:
            # 1. 绘制门框（黄色半透明）
            mask_color = np.zeros_like(frame)
            mask_color[door_mask > 0] = [0, 255, 255]
            frame = cv2.addWeighted(frame, 0.85, mask_color, 0.15, 0)
            
            # 2. 绘制所有ID的可视化信息
            for tid, person in all_tracks.items():
                if frame_idx in person.frames:
                    idx = person.frames.index(frame_idx)
                    box = person.boxes[idx]
                    polygon = person.mask_polygons[idx] if idx < len(person.mask_polygons) else None
                    
                    is_alighting = tid in alighting
                    color = (0, 255, 0) if is_alighting else (255, 100, 100)
                    
                    if polygon is not None and len(polygon) > 0:
                        # 2.1 绘制完整掩码（浅色）
                        mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                        cv2.fillPoly(mask, [polygon], 255)
                        
                        mask_color = np.zeros_like(frame)
                        mask_color[mask > 0] = color
                        frame = cv2.addWeighted(frame, 1.0, mask_color, 0.2, 0)
                        
                        # 2.2 绘制上4/5区域（蓝色边框）
                        upper_mask = get_upper_region_mask(polygon, box, frame.shape)
                        if upper_mask is not None:
                            contours, _ = cv2.findContours(upper_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            cv2.drawContours(frame, contours, -1, (255, 200, 0), 2)
                        
                        # 2.3 绘制重叠区域（红色高亮）
                        overlap_mask = get_overlap_region_mask(polygon, box, door_mask, frame.shape)
                        if overlap_mask is not None and np.any(overlap_mask > 0):
                            mask_color = np.zeros_like(frame)
                            mask_color[overlap_mask > 0] = [0, 0, 255]
                            frame = cv2.addWeighted(frame, 1.0, mask_color, 0.6, 0)
                            
                            # 计算并显示重叠比例
                            if tid in analysis_info and len(analysis_info[tid]['frame_indices']) > 0:
                                frame_pos = analysis_info[tid]['frame_indices'].index(frame_idx) if frame_idx in analysis_info[tid]['frame_indices'] else -1
                                if frame_pos >= 0:
                                    overlap_ratio = analysis_info[tid]['overlap_ratios'][frame_pos]
                                    bx1, by1, bx2, by2 = map(int, box)
                                    cv2.putText(frame, f"{overlap_ratio:.2f}", (bx1, by2+20),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    # 2.4 绘制bbox
                    bx1, by1, bx2, by2 = map(int, box)
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
                    
                    # 2.5 绘制ID标签
                    label = f"ID:{tid}"
                    if is_alighting:
                        label += " (下车)"
                    
                    cv2.putText(frame, label, (bx1, by1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # 3. 绘制统计信息
            cv2.putText(frame, f"Alighting: {len(alighting)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Frame: {frame_idx}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 4. 绘制图例
            legend_y = frame.shape[0] - 120
            cv2.rectangle(frame, (10, legend_y), (300, frame.shape[0] - 10), (0, 0, 0), -1)
            cv2.putText(frame, "Legend:", (20, legend_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, "Yellow: Door", (20, legend_y + 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(frame, "Cyan: Upper 4/5 region", (20, legend_y + 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
            cv2.putText(frame, "Red: Overlap region", (20, legend_y + 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            out.write(frame)
            
            if frame_idx in trigger_frames:
                for _ in range(int(reader.fps * 0.5)):
                    out.write(frame)
            
            frame_idx += 1
        
        out.release()
    
    print(f"  ✓ 可视化视频: {out_path}")
    
    print(f"\n{'='*60}")
    print("分析完成")
    print(f"{'='*60}")
    print(f"总追踪ID数: {len(all_tracks)}")
    print(f"与门框重叠ID数: {len([tid for tid, info in analysis_info.items() if len(info['overlap_ratios']) > 0])}")
    print(f"判定下车人数: {len(alighting)}")
    print(f"\n输出文件:")
    print(f"  - 重叠分析图: {OUTPUT_DIR / 'overlap_analysis.png'}")
    print(f"  - 判定摘要图: {OUTPUT_DIR / 'decision_summary.png'}")
    print(f"  - 可视化视频: {OUTPUT_DIR / 'alighting_result.mp4'}")

if __name__ == "__main__":
    test_alighting_logic()
