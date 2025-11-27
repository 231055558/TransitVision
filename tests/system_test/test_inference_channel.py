"""
推理通道模块系统测试
测试多线路视频输入的GPU推理
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import time
from transit_vision.threads import MultiLineInputChannel, MultiDirectionInferenceChannel
from transit_vision.utils import DeviceConfig

# 配置
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "close_loop_od"
PERSON_MODEL = "/mnt/mydisk/My_project/bus_down/yolo11x-seg.pt"
TRACKER_CONFIG = str(Path(__file__).parent.parent.parent / "configs" / "botsort_seg.yaml")
DEVICE_CONFIG = str(Path(__file__).parent.parent.parent / "configs" / "device_debug.yaml")

NUM_LINES = 4
BATCH_SIZE = 64 # 减小batch size以避免CPU NMS超时
NUM_WORKERS = 2
MAX_STATIONS = 3

def test_inference_channel():
    print("=" * 70)
    print("推理通道模块系统测试")
    print("=" * 70)
    print(f"数据目录: {DATA_DIR}")
    print(f"并行线路数: {NUM_LINES}")
    print(f"推理批大小: {BATCH_SIZE}")
    print(f"推理工作线程: {NUM_WORKERS}")
    print(f"测试站点数: {MAX_STATIONS}")
    print()
    
    # 初始化设备配置
    device_cfg = DeviceConfig(DEVICE_CONFIG)
    
    # 创建输入通道
    print("创建输入通道...")
    input_channel = MultiLineInputChannel(num_lines=NUM_LINES, workers_per_line=2)
    stations = input_channel.load_and_replicate_data(DATA_DIR, MAX_STATIONS)
    print(f"✓ 加载 {len(stations)} 个站点")
    
    # 创建推理通道
    print("\n创建推理通道...")
    inference_channel = MultiDirectionInferenceChannel(
        PERSON_MODEL, TRACKER_CONFIG, device_cfg,
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )
    print(f"✓ 推理通道已创建 (batch={BATCH_SIZE}, workers={NUM_WORKERS})")
    
    # 启动通道
    print("\n启动通道...")
    input_channel.start_all()
    inference_channel.start()
    print("✓ 所有通道已启动")
    
    # 逐站处理
    print(f"\n开始处理站点数据...")
    print("=" * 70)
    
    start_time = time.time()
    
    for station in stations:
        print(f"\n[站点 {station.station_id}: {station.station_name}]")
        
        # 提交到输入通道
        input_channel.submit_station_to_all_lines(station)
        print(f"  提交到输入通道: {NUM_LINES * 2} 任务")
        
        # 等待输入通道处理
        time.sleep(0.5)
        
        # 记录当前状态
        initial_stats = inference_channel.get_total_stats()
        initial_processed = initial_stats['total_processed']

        # 从输入通道获取任务并提交到推理通道
        submitted = 0
        for line_id in range(NUM_LINES):
            channel = input_channel.channels[line_id]
            
            up_task = channel.get_output_task('up', timeout=1.0)
            down_task = channel.get_output_task('down', timeout=1.0)
            
            if up_task:
                inference_channel.submit_task(up_task)
                submitted += 1
            if down_task:
                inference_channel.submit_task(down_task)
                submitted += 1
        
        print(f"  提交到推理通道: {submitted} 任务")
        
        # 等待推理处理
        print(f"  等待推理完成...")
        wait_start = time.time()
        check_count = 0
        while True:
            stats = inference_channel.get_total_stats()
            
            check_count += 1
            if check_count % 4 == 0:
                elapsed = time.time() - wait_start
                print(f"    [{elapsed:.1f}s] 已处理 {stats['total_processed']}/{initial_processed + submitted} 视频, "
                      f"{stats['total_frames']} 帧")
            
            if stats['total_processed'] >= initial_processed + submitted:
                print(f"    ✓ 推理完成")
                break
            if time.time() - wait_start > 120:
                print(f"  ⚠ 推理超时")
                break
            time.sleep(0.5)
            
        # 获取统计
        inf_stats = inference_channel.get_total_stats()
        
        print(f"\n  推理进度:")
        print(f"    总输入: {inf_stats['total_inputs']}")
        print(f"    已处理: {inf_stats['total_processed']}")
        print(f"    总帧数: {inf_stats['total_frames']}")
        print(f"    上车: {inf_stats['up']['processed']} 视频, {inf_stats['up']['total_frames']} 帧")
        print(f"    下车: {inf_stats['down']['processed']} 视频, {inf_stats['down']['total_frames']} 帧")
        
        # 获取推理结果示例
        print(f"\n  推理结果示例:")
        for direction in ['up', 'down']:
            result = inference_channel.get_result(direction, timeout=0.1)
            if result:
                task = result['task']
                tracks = result['tracks']
                frame_count = result['frame_count']
                print(f"    {direction.upper():4s}: {task.bus_id:30s} | "
                      f"{frame_count}帧, {len(tracks)}人")
        
        print(f"\n{'='*70}")
    
    # 等待所有推理完成
    print("\n等待所有推理完成...")
    time.sleep(3)
    
    elapsed = time.time() - start_time
    
    # 最终统计
    final_stats = inference_channel.get_total_stats()
    
    print(f"\n{'='*70}")
    print("最终统计")
    print(f"{'='*70}")
    print(f"总耗时: {elapsed:.2f}s")
    print(f"总输入任务: {final_stats['total_inputs']}")
    print(f"已处理任务: {final_stats['total_processed']}")
    print(f"总推理帧数: {final_stats['total_frames']}")
    print(f"\n上车通道:")
    print(f"  视频数: {final_stats['up']['processed']}")
    print(f"  帧数: {final_stats['up']['total_frames']}")
    print(f"\n下车通道:")
    print(f"  视频数: {final_stats['down']['processed']}")
    print(f"  帧数: {final_stats['down']['total_frames']}")
    
    # 性能指标
    if elapsed > 0:
        fps = final_stats['total_frames'] / elapsed
        print(f"\n性能指标:")
        print(f"  平均FPS: {fps:.2f}")
        print(f"  视频吞吐: {final_stats['total_processed'] / elapsed:.2f} 视频/秒")
    
    # 验证
    expected_videos = len(stations) * NUM_LINES * 2
    print(f"\n验证:")
    print(f"  预期视频数: {expected_videos}")
    print(f"  实际处理数: {final_stats['total_processed']}")
    
    if final_stats['total_processed'] == expected_videos:
        print("  ✓ 所有视频已正确处理")
    else:
        print(f"  ⚠ 处理不完整，差异: {expected_videos - final_stats['total_processed']}")
    
    # 停止通道
    print(f"\n停止所有通道...")
    input_channel.stop_all()
    inference_channel.stop()
    print("✓ 所有通道已停止")
    
    print(f"\n{'='*70}")
    print("测试完成")
    print(f"{'='*70}")

if __name__ == "__main__":
    test_inference_channel()

