"""
逻辑运算模块系统测试
测试完整流程：输入 → 逻辑通道(门检测+推理+判定)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import time
import yaml
from transit_vision.threads import MultiLineInputChannel, MultiDirectionLogicChannel
from transit_vision.utils import DeviceConfig

PROJECT_ROOT = Path(__file__).parent.parent.parent

# 加载系统配置
SYSTEM_CONFIG = PROJECT_ROOT / "configs" / "system_config.yaml"
with open(SYSTEM_CONFIG, 'r', encoding='utf-8') as f:
    system_cfg = yaml.safe_load(f)

DATA_DIR = PROJECT_ROOT / system_cfg['data']['input_dir']
PERSON_MODEL = system_cfg['models']['person_model']
DOOR_MODEL = system_cfg['models']['door_model']
TRACKER_CONFIG = str(PROJECT_ROOT / system_cfg['configs']['tracker_config'])
DEVICE_CONFIG = str(PROJECT_ROOT / system_cfg['configs']['device_config'])

NUM_LINES = system_cfg['system']['num_lines']
BATCH_SIZE = system_cfg['system']['batch_size']
NUM_WORKERS = system_cfg['system']['logic_workers']
MAX_STATIONS = system_cfg['system']['max_stations']

RECALC_DOOR_UP = system_cfg['door']['recalc_per_video_up']
RECALC_DOOR_DOWN = system_cfg['door']['recalc_per_video_down']

def test_logic_channel():
    print("=" * 70)
    print("逻辑运算模块系统测试")
    print("=" * 70)
    print(f"数据目录: {DATA_DIR}")
    print(f"并行线路数: {NUM_LINES}")
    print(f"测试站点数: {MAX_STATIONS}")
    print(f"上车门框重算: {RECALC_DOOR_UP}")
    print(f"下车门框重算: {RECALC_DOOR_DOWN}")
    print()
    
    device_cfg = DeviceConfig(DEVICE_CONFIG)
    
    print("创建输入通道...")
    input_channel = MultiLineInputChannel(num_lines=NUM_LINES, workers_per_line=2)
    stations = input_channel.load_and_replicate_data(DATA_DIR, MAX_STATIONS)
    print(f"✓ 加载 {len(stations)} 个站点")
    
    print("\n创建逻辑通道...")
    logic_channel = MultiDirectionLogicChannel(
        PERSON_MODEL, TRACKER_CONFIG, DOOR_MODEL, device_cfg,
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
        recalc_door_up=RECALC_DOOR_UP, recalc_door_down=RECALC_DOOR_DOWN
    )
    print(f"✓ 逻辑通道已创建")
    
    print("\n启动所有通道...")
    input_channel.start_all()
    logic_channel.start()
    print("✓ 所有通道已启动")
    
    print(f"\n开始处理站点数据...")
    print("=" * 70)
    
    start_time = time.time()
    
    for station in stations:
        print(f"\n[站点 {station.station_id}: {station.station_name}]")
        
        input_channel.submit_station_to_all_lines(station)
        print(f"  提交到输入通道: {NUM_LINES * 2} 任务")
        
        time.sleep(0.5)
        
        initial_logic_stats = logic_channel.get_total_stats()
        initial_logic_processed = initial_logic_stats['total_processed']
        
        submitted = 0
        for line_id in range(NUM_LINES):
            channel = input_channel.channels[line_id]
            
            up_task = channel.get_output_task('up', timeout=1.0)
            down_task = channel.get_output_task('down', timeout=1.0)
            
            if up_task:
                logic_channel.submit_task(up_task)
                submitted += 1
            if down_task:
                logic_channel.submit_task(down_task)
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
                print(f"    [逻辑 {elapsed:.1f}s] 已处理 {stats['total_processed']}/{initial_logic_processed + submitted} 任务, "
                      f"上车:{stats['total_boarding']} 下车:{stats['total_alighting']}")
            
            if stats['total_processed'] >= initial_logic_processed + submitted:
                print(f"    ✓ 逻辑处理完成")
                break
            # if time.time() - wait_start > 120:
            #     print(f"  ⚠ 逻辑处理超时")
            #     break
            time.sleep(0.5)
        
        logic_stats = logic_channel.get_total_stats()
        
        print(f"\n  逻辑处理结果:")
        print(f"    总输入: {logic_stats['total_inputs']}")
        print(f"    已处理: {logic_stats['total_processed']}")
        print(f"    上车人数: {logic_stats['total_boarding']}")
        print(f"    下车人数: {logic_stats['total_alighting']}")
        
        print(f"\n  详细结果示例:")
        for direction in ['up', 'down']:
            result = logic_channel.get_result(direction, timeout=0.1)
            if result:
                task = result['task']
                count = result['count']
                passengers = result['valid_passengers']
                print(f"    {direction.upper():4s}: {task.bus_id:30s} | "
                      f"{count}人 (track_ids: {list(passengers.keys())[:5]})")
        
        print(f"\n{'='*70}")
    
    print("\n等待所有处理完成...")
    time.sleep(2)
    
    elapsed = time.time() - start_time
    
    final_stats = logic_channel.get_total_stats()
    
    print(f"\n{'='*70}")
    print("最终统计")
    print(f"{'='*70}")
    print(f"总耗时: {elapsed:.2f}s")
    print(f"总输入任务: {final_stats['total_inputs']}")
    print(f"已处理任务: {final_stats['total_processed']}")
    print(f"\n上车通道:")
    print(f"  处理数: {final_stats['up']['processed']}")
    print(f"  上车人数: {final_stats['total_boarding']}")
    print(f"\n下车通道:")
    print(f"  处理数: {final_stats['down']['processed']}")
    print(f"  下车人数: {final_stats['total_alighting']}")
    
    expected_tasks = len(stations) * NUM_LINES * 2
    print(f"\n验证:")
    print(f"  预期任务数: {expected_tasks}")
    print(f"  实际处理数: {final_stats['total_processed']}")
    
    if final_stats['total_processed'] == expected_tasks:
        print("  ✓ 所有任务已正确处理")
    else:
        print(f"  ⚠ 处理不完整，差异: {expected_tasks - final_stats['total_processed']}")
    
    print(f"\n停止所有通道...")
    input_channel.stop_all()
    logic_channel.stop()
    print("✓ 所有通道已停止")
    
    print(f"\n{'='*70}")
    print("测试完成")
    print(f"{'='*70}")

if __name__ == "__main__":
    test_logic_channel()

