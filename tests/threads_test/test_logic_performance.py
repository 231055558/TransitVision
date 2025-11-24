"""
上车下车逻辑算法多线程性能测试
测试纯逻辑算法的CPU并行效率
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import time
import threading
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from transit_vision.logic.boarding_counter import filter_boarding_passengers
from transit_vision.logic.alighting_counter_v2 import filter_alighting_passengers
from transit_vision.data_structures import Person

def generate_mock_tracks(num_tracks=30, num_frames=50):
    """生成模拟追踪数据 - 只存储bbox，不存储大掩码"""
    tracks = {}
    for i in range(num_tracks):
        person = Person(i)
        for frame_idx in range(0, num_frames, 2):
            x = 500 + np.random.randint(-50, 50)
            y = 300 + np.random.randint(-50, 50)
            w = 80 + np.random.randint(-10, 10)
            h = 150 + np.random.randint(-20, 20)
            box = [x, y, x+w, y+h]
            
            # 只生成小掩码(人物大小)，不是全图
            mask = np.random.randint(0, 2, (h, w), dtype=np.uint8) * 255
            person.add_detection(frame_idx, box, mask, 0.9)
        tracks[i] = person
    return tracks

def generate_door_mask():
    """生成门掩码"""
    mask = np.zeros((720, 1280), dtype=np.uint8)
    mask[200:600, 400:700] = 255
    return mask

def generate_door_bbox():
    """生成门框bbox"""
    return [400, 200, 700, 600]

print("=== 上车下车逻辑算法多线程性能测试 ===\n")

# 生成测试数据
print("生成测试数据...")
num_videos = 16
all_tracks = [generate_mock_tracks(30, 50) for _ in range(num_videos)]
door_mask = generate_door_mask()
door_bbox = generate_door_bbox()
print(f"✓ 生成 {num_videos} 个视频的模拟数据\n")

# ==================== 测试1: 上车逻辑 ====================
print("=" * 60)
print("测试1: 上车逻辑算法 (boarding_counter)")
print("=" * 60)

def process_boarding_worker(tracks, idx):
    result = filter_boarding_passengers(tracks, door_bbox)
    return idx, result

boarding_results = []

for num_threads in [1, 2, 4, 8]:
    if num_threads == 1:
        print(f"\n[单线程]")
        start = time.time()
        
        results = []
        for tracks in tqdm(all_tracks, desc="处理", bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}'):
            result = filter_boarding_passengers(tracks, door_bbox)
            results.append(result)
        
        elapsed = time.time() - start
        
    else:
        print(f"\n[{num_threads} 线程]")
        start = time.time()
        
        results = [None] * len(all_tracks)
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {executor.submit(process_boarding_worker, tracks, i): i 
                      for i, tracks in enumerate(all_tracks)}
            
            with tqdm(total=len(all_tracks), desc="处理", 
                     bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}') as pbar:
                for future in as_completed(futures):
                    idx, result = future.result()
                    results[idx] = result
                    pbar.update(1)
        
        elapsed = time.time() - start
    
    avg_time = elapsed / len(all_tracks)
    speedup = boarding_results[0][1] / elapsed if boarding_results else 1.0
    
    boarding_results.append((num_threads, elapsed, avg_time, speedup))
    print(f"  总耗时: {elapsed:.2f}s")
    print(f"  平均: {avg_time:.3f}s/视频")
    print(f"  加速比: {speedup:.2f}x")

print("\n上车逻辑性能对比:")
print(f"{'线程数':<8} {'总耗时':<12} {'平均耗时':<15} {'加速比':<10}")
print("-" * 50)
for threads, total, avg, speedup in boarding_results:
    print(f"{threads:<8} {total:>8.2f}s    {avg:>8.3f}s/视频    {speedup:>6.2f}x")

# ==================== 测试2: 下车逻辑 ====================
print("\n" + "=" * 60)
print("测试2: 下车逻辑算法 (alighting_counter_v2)")
print("=" * 60)

def process_alighting_worker(tracks, idx):
    result = filter_alighting_passengers(tracks, door_mask)
    return idx, result

alighting_results = []

for num_threads in [1, 2, 4, 8]:
    if num_threads == 1:
        print(f"\n[单线程]")
        start = time.time()
        
        results = []
        for tracks in tqdm(all_tracks, desc="处理", bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}'):
            result = filter_alighting_passengers(tracks, door_mask)
            results.append(result)
        
        elapsed = time.time() - start
        
    else:
        print(f"\n[{num_threads} 线程]")
        start = time.time()
        
        results = [None] * len(all_tracks)
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {executor.submit(process_alighting_worker, tracks, i): i 
                      for i, tracks in enumerate(all_tracks)}
            
            with tqdm(total=len(all_tracks), desc="处理", 
                     bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}') as pbar:
                for future in as_completed(futures):
                    idx, result = future.result()
                    results[idx] = result
                    pbar.update(1)
        
        elapsed = time.time() - start
    
    avg_time = elapsed / len(all_tracks)
    speedup = alighting_results[0][1] / elapsed if alighting_results else 1.0
    
    alighting_results.append((num_threads, elapsed, avg_time, speedup))
    print(f"  总耗时: {elapsed:.2f}s")
    print(f"  平均: {avg_time:.3f}s/视频")
    print(f"  加速比: {speedup:.2f}x")

print("\n下车逻辑性能对比:")
print(f"{'线程数':<8} {'总耗时':<12} {'平均耗时':<15} {'加速比':<10}")
print("-" * 50)
for threads, total, avg, speedup in alighting_results:
    print(f"{threads:<8} {total:>8.2f}s    {avg:>8.3f}s/视频    {speedup:>6.2f}x")

print("\n" + "=" * 60)
print("总结:")
print("=" * 60)
print("1. 此测试仅测试纯逻辑算法的CPU并行效率")
print("2. 不包含GPU推理，不会产生大量掩码数据")
print("3. 真实下客流程应使用AlightingPipeline(有界队列)")
print("4. AlightingPipeline可防止GPU推理产生的掩码内存溢出")
