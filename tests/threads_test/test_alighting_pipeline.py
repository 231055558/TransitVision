"""
测试下客流程三线程流水线
验证有界阻塞队列防止内存溢出
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import time
import psutil
import os
from transit_vision.threads import AlightingPipeline

# 配置路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
VIDEO_PATH = PROJECT_ROOT / "tests" / "pose2id_scheme" / "reid_test" / "video" / "test_video.mp4"
PERSON_MODEL = PROJECT_ROOT / "models" / "yolo11x-seg.pt"
CONFIG_PATH = PROJECT_ROOT / "configs" / "device_debug.yaml"
TRACKER_CONFIG_PATH = PROJECT_ROOT / "configs" / "botsort_seg.yaml"

def get_memory_mb():
    """获取当前进程内存占用(MB)"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def generate_mock_door_mask():
    """生成模拟门掩码"""
    import numpy as np
    mask = np.zeros((1080, 1920), dtype=np.uint8)
    mask[300:800, 500:700] = 255
    return mask

print("=== 下客流程三线程流水线测试 ===\n")

if not VIDEO_PATH.exists():
    print(f"错误: 视频文件不存在 {VIDEO_PATH}")
    sys.exit(1)

print(f"视频: {VIDEO_PATH}")
print(f"模型: {PERSON_MODEL}")

# 加载配置
from transit_vision.utils.device import DeviceConfig
device_config = DeviceConfig(str(CONFIG_PATH))

# 初始化分割追踪器
print("\n初始化PersonSegTracker...")
from transit_vision.core.detection import PersonSegTracker
person_seg_tracker = PersonSegTracker(str(PERSON_MODEL), str(TRACKER_CONFIG_PATH), device_config)

# 生成门掩码
door_mask = generate_mock_door_mask()

print("\n开始测试不同batch_size和queue_size的内存占用...")
print("-" * 70)

test_configs = [
    {"queue_size": 10},
    {"queue_size": 20},
    {"queue_size": 30},
]

for config in test_configs:
    queue_size = config["queue_size"]
    
    print(f"\n[queue_size={queue_size}]")
    
    mem_start = get_memory_mb()
    print(f"  初始内存: {mem_start:.1f} MB")
    
    start_time = time.time()
    
    with AlightingPipeline(
        str(VIDEO_PATH),
        door_mask,
        person_seg_tracker,
        queue_size=queue_size
    ) as pipeline:
        pipeline.wait_completion()
        result = pipeline.get_result()
    
    elapsed = time.time() - start_time
    mem_end = get_memory_mb()
    mem_peak = mem_end
    
    print(f"  处理耗时: {elapsed:.2f}s")
    print(f"  结束内存: {mem_end:.1f} MB")
    print(f"  内存增量: {mem_end - mem_start:.1f} MB")
    print(f"  下车人数: {len(result)}")

print("\n" + "=" * 70)
print("测试完成！")
print("\n关键点:")
print("1. 有界队列防止内存无限增长")
print("2. GPU推理快于CPU处理时，推理线程会阻塞等待")
print("3. 内存增量应该保持在合理范围内")
print("4. 不同queue_size的内存占用应该可控")



