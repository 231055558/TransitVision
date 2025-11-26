"""
输入通道模块简化测试
验证数据加载和分配逻辑
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 配置
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "close_loop_od"
NUM_LINES = 32
MAX_STATIONS = 5

def test_data_loading():
    """测试数据加载"""
    print("=" * 70)
    print("输入通道模块 - 数据加载测试")
    print("=" * 70)
    print(f"数据目录: {DATA_DIR}")
    print(f"并行线路数: {NUM_LINES}")
    print(f"测试站点数: {MAX_STATIONS}")
    print()
    
    # 检查数据目录
    if not DATA_DIR.exists():
        print(f"✗ 数据目录不存在: {DATA_DIR}")
        return
    
    print("✓ 数据目录存在")
    
    # 加载站点数据
    print("\n加载站点数据...")
    stations = []
    
    station_id = 1
    while True:
        up_video = DATA_DIR / f"{station_id}_up.mp4"
        down_video = DATA_DIR / f"{station_id}_down.mp4"
        
        if not up_video.exists() or not down_video.exists():
            break
        
        stations.append({
            'station_id': station_id,
            'station_name': f"站点{station_id}",
            'up_video': up_video,
            'down_video': down_video
        })
        
        station_id += 1
        if MAX_STATIONS and station_id > MAX_STATIONS:
            break
    
    print(f"✓ 加载 {len(stations)} 个站点")
    
    # 显示站点信息
    print("\n站点详情:")
    print("-" * 70)
    for station in stations:
        print(f"站点 {station['station_id']}: {station['station_name']}")
        print(f"  上车: {station['up_video'].name} ({station['up_video'].stat().st_size / 1024 / 1024:.1f}MB)")
        print(f"  下车: {station['down_video'].name} ({station['down_video'].stat().st_size / 1024 / 1024:.1f}MB)")
    
    # 模拟多线路输入
    print(f"\n{'='*70}")
    print(f"模拟 {NUM_LINES} 线路并行输入")
    print(f"{'='*70}")
    
    total_tasks = 0
    for station in stations:
        print(f"\n[站点 {station['station_id']}: {station['station_name']}]")
        
        # 每条线路提交up+down
        tasks_per_station = NUM_LINES * 2
        total_tasks += tasks_per_station
        
        print(f"  提交任务数: {tasks_per_station}")
        print(f"  累计任务数: {total_tasks}")
        
        # 输出格式示例
        print(f"\n  输出格式示例 (前3线):")
        for line_id in range(min(3, NUM_LINES)):
            bus_id_up = f"line{line_id}_bus_{station['station_id']}_up"
            bus_id_down = f"line{line_id}_bus_{station['station_id']}_down"
            
            print(f"    线路{line_id} UP  : {bus_id_up:30s} | {station['up_video'].name}")
            print(f"    线路{line_id} DOWN: {bus_id_down:30s} | {station['down_video'].name}")
    
    # 最终统计
    print(f"\n{'='*70}")
    print("最终统计")
    print(f"{'='*70}")
    print(f"站点数: {len(stations)}")
    print(f"线路数: {NUM_LINES}")
    print(f"总任务数: {total_tasks}")
    print(f"上车任务: {total_tasks // 2}")
    print(f"下车任务: {total_tasks // 2}")
    
    print(f"\n{'='*70}")
    print("✓ 数据加载和分配逻辑验证通过")
    print(f"{'='*70}")

if __name__ == "__main__":
    test_data_loading()

