"""
下客逻辑V2 vs V3性能对比测试
使用真实视频数据对比两个版本的性能差异
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transit_vision.utils import VideoReader, read_first_frame, DeviceConfig
from transit_vision.core.detection import PersonSegTracker, DoorSegmentor
from transit_vision.logic import preprocess_rear_door
from transit_vision.logic.alighting_counter_v2 import filter_alighting_passengers as filter_v2
from transit_vision.logic.alighting_counter_v3 import filter_alighting_passengers as filter_v3
import cv2
import numpy as np
import time

DOOR_VIDEO = "/mnt/mydisk/My_project/TransitVision/data/od_1021/316路/8-6177/2025-10-20-08-49_8-6177_半道红_down.mp4"
TEST_VIDEO = "/mnt/mydisk/My_project/TransitVision/data/od_1021/316路/8-6177/2025-10-20-09-17_8-6177_地铁西文街站C2口_down.mp4"
PERSON_MODEL = "/mnt/mydisk/My_project/bus_down/yolo11x-seg.pt"
DOOR_MODEL = "/mnt/mydisk/My_project/bus_down/front_door.pt"
TRACKER_CONFIG = str(Path(__file__).parent.parent.parent / "configs" / "botsort_seg.yaml")
DEVICE_CONFIG = str(Path(__file__).parent.parent.parent / "configs" / "device_debug.yaml")

def test_compare():
    print("=" * 70)
    print("下客逻辑算法性能对比测试: V2 vs V3 (真实数据)")
    print("=" * 70)
    
    device_cfg = DeviceConfig(DEVICE_CONFIG)
    
    # 1. 获取门掩码
    print(f"\n1. 门检测: {Path(DOOR_VIDEO).name}")
    door_seg = DoorSegmentor(DOOR_MODEL, device_cfg)
    door_frame = read_first_frame(DOOR_VIDEO)
    door = door_seg.detect(door_frame)
    
    if door is None:
        print("✗ 未检测到门")
        return
    
    door_mask = preprocess_rear_door(door)
    print(f"✓ 门掩码面积: {np.sum(door_mask > 0)}")
    
    # 2. 人员追踪
    print(f"\n2. 人员追踪: {Path(TEST_VIDEO).name}")
    person_tracker = PersonSegTracker(PERSON_MODEL, TRACKER_CONFIG, device_cfg)
    
    print("   正在追踪...")
    all_tracks = person_tracker.track_video(VideoReader(TEST_VIDEO))
    print(f"✓ 总追踪数: {len(all_tracks)}")
    
    # 将追踪结果转换为两种格式
    print("\n3. 准备测试数据...")
    
    # V2格式: 使用masks
    tracks_v2 = {}
    for tid, person in all_tracks.items():
        from transit_vision.data_structures import Person
        p = Person(tid)
        p.frames = person.frames.copy()
        p.boxes = person.boxes.copy()
        p.confs = person.confs.copy()
        
        # 将多边形转换为掩码 (V2需要)
        for polygon in person.mask_polygons:
            if polygon is not None and len(polygon) > 0:
                mask = np.zeros((1080, 1920), dtype=np.uint8)
                cv2.fillPoly(mask, [polygon], 255)
                p.masks.append(mask)
            else:
                p.masks.append(None)
        
        tracks_v2[tid] = p
    
    # V3格式: 直接使用原始数据
    tracks_v3 = all_tracks
    
    print(f"✓ V2格式数据准备完成")
    print(f"✓ V3格式数据准备完成")
    
    # 4. 性能测试
    print("\n" + "=" * 70)
    print("开始性能测试")
    print("=" * 70)
    
    # 测试V2
    print("\n[V2 - 逐点计算]")
    start = time.time()
    result_v2 = filter_v2(tracks_v2, door_mask, threshold=0.5, grace_period=6)
    time_v2 = time.time() - start
    print(f"  耗时: {time_v2:.4f}s")
    print(f"  下车人数: {len(result_v2)}")
    for tid in sorted(result_v2.keys()):
        print(f"    ID {tid}: {len(result_v2[tid])} 帧")
    
    # 测试V3
    print("\n[V3 - 矩阵位运算]")
    start = time.time()
    result_v3 = filter_v3(tracks_v3, door_mask, threshold=0.5, grace_period=6)
    time_v3 = time.time() - start
    print(f"  耗时: {time_v3:.4f}s")
    print(f"  下车人数: {len(result_v3)}")
    for tid in sorted(result_v3.keys()):
        print(f"    ID {tid}: {len(result_v3[tid])} 帧")
    
    # 5. 结果对比
    print("\n" + "=" * 70)
    print("性能对比结果")
    print("=" * 70)
    
    speedup = time_v2 / time_v3 if time_v3 > 0 else 0
    
    print(f"V2 耗时: {time_v2:.4f}s")
    print(f"V3 耗时: {time_v3:.4f}s")
    print(f"加速比: {speedup:.2f}x")
    print(f"时间节省: {(time_v2 - time_v3):.4f}s ({(1 - time_v3/time_v2)*100:.1f}%)")
    
    # 检查结果一致性
    print("\n" + "=" * 70)
    print("结果一致性检查")
    print("=" * 70)
    
    v2_ids = set(result_v2.keys())
    v3_ids = set(result_v3.keys())
    
    if v2_ids == v3_ids:
        print(f"✓ 检测到的ID完全一致: {v2_ids}")
    else:
        print(f"⚠ ID不完全一致")
        print(f"  V2独有: {v2_ids - v3_ids}")
        print(f"  V3独有: {v3_ids - v2_ids}")
        print(f"  共同: {v2_ids & v3_ids}")
    
    print("\n" + "=" * 70)
    print("✓ 测试完成")
    print("=" * 70)

if __name__ == "__main__":
    test_compare()

