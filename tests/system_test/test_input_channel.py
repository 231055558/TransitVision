"""
输入通道模块系统测试
测试多线路视频输入的分配和管理
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import time
from transit_vision.threads import MultiLineInputChannel

# 配置
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "close_loop_od"
NUM_LINES = 32
MAX_STATIONS = 5  # 测试前5站

def test_input_channel():
    print("=" * 70)
    print("输入通道模块系统测试")
    print("=" * 70)
    print(f"数据目录: {DATA_DIR}")
    print(f"并行线路数: {NUM_LINES}")
    print(f"测试站点数: {MAX_STATIONS}")
    print()
    
    # 创建多线路输入通道
    multi_channel = MultiLineInputChannel(num_lines=NUM_LINES, workers_per_line=2)
    
    # 加载站点数据
    print("加载站点数据...")
    stations = multi_channel.load_and_replicate_data(DATA_DIR, MAX_STATIONS)
    print(f"✓ 加载 {len(stations)} 个站点")
    
    for station in stations:
        print(f"  站点{station.station_id}: {station.station_name}")
        print(f"    上车视频: {station.up_video.name}")
        print(f"    下车视频: {station.down_video.name}")
    
    # 启动所有线路
    print(f"\n启动 {NUM_LINES} 条线路...")
    multi_channel.start_all()
    print("✓ 所有线路已启动")
    
    # 逐站提交数据
    print(f"\n开始处理站点数据...")
    print("=" * 70)
    
    for station in stations:
        print(f"\n[站点 {station.station_id}: {station.station_name}]")
        
        # 提交到所有线路
        submit_start = time.time()
        multi_channel.submit_station_to_all_lines(station)
        submit_time = time.time() - submit_start
        
        print(f"  提交耗时: {submit_time:.4f}s")
        print(f"  提交任务数: {NUM_LINES * 2} (每线路up+down)")
        
        # 等待处理
        time.sleep(0.5)
        
        # 获取统计信息
        stats = multi_channel.get_total_stats()
        
        print(f"\n  处理进度:")
        print(f"    总输入: {stats['total_inputs']}")
        print(f"    已处理: {stats['processed']}")
        print(f"    待处理: {stats['pending']}")
        print(f"    上车队列: {stats['up_queue']}")
        print(f"    下车队列: {stats['down_queue']}")
        
        # 输出格式示例
        print(f"\n  输出格式示例:")
        for line_id in range(min(3, NUM_LINES)):
            channel = multi_channel.channels[line_id]
            
            # 尝试获取输出任务
            up_task = channel.get_output_task('up', timeout=0.1)
            down_task = channel.get_output_task('down', timeout=0.1)
            
            if up_task:
                print(f"    线路{line_id} UP  : {up_task.bus_id} | {Path(up_task.video_path).name}")
            if down_task:
                print(f"    线路{line_id} DOWN: {down_task.bus_id} | {Path(down_task.video_path).name}")
        
        print(f"\n{'='*70}")
    
    # 等待所有任务完成
    print("\n等待所有任务处理完成...")
    time.sleep(2)
    
    # 最终统计
    final_stats = multi_channel.get_total_stats()
    
    print(f"\n{'='*70}")
    print("最终统计")
    print(f"{'='*70}")
    print(f"总输入任务: {final_stats['total_inputs']}")
    print(f"已处理任务: {final_stats['processed']}")
    print(f"上车任务: {final_stats['up_count']}")
    print(f"下车任务: {final_stats['down_count']}")
    print(f"待处理: {final_stats['pending']}")
    
    # 验证
    expected_total = len(stations) * NUM_LINES * 2
    print(f"\n预期任务数: {expected_total}")
    print(f"实际处理数: {final_stats['processed']}")
    
    if final_stats['processed'] == expected_total:
        print("✓ 所有任务已正确处理")
    else:
        print(f"⚠ 任务处理不完整，差异: {expected_total - final_stats['processed']}")
    
    # 停止所有线路
    print(f"\n停止所有线路...")
    multi_channel.stop_all()
    print("✓ 所有线路已停止")
    
    print(f"\n{'='*70}")
    print("测试完成")
    print(f"{'='*70}")

if __name__ == "__main__":
    test_input_channel()

