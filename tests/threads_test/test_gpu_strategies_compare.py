"""
GPU推理策略对比测试
对比三种策略的性能差异
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import subprocess
import time

strategies = [
    ("策略1: 动态批处理", "test_gpu_strategy1_dynamic_batch.py"),
    ("策略2: 共享顺序推理", "test_gpu_strategy2_shared_sequential.py"),
    ("策略3: 单线程批处理", "test_gpu_strategy3_batch_collect.py"),
]

print("=" * 70)
print("GPU推理策略性能对比测试")
print("=" * 70)

results = []

for name, script in strategies:
    print(f"\n{'='*70}")
    print(f"测试: {name}")
    print(f"{'='*70}")
    
    script_path = Path(__file__).parent / script
    
    start = time.time()
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True
    )
    elapsed = time.time() - start
    
    print(result.stdout)
    if result.stderr:
        print("错误:", result.stderr)
    
    results.append({
        'name': name,
        'time': elapsed,
        'success': result.returncode == 0
    })
    
    time.sleep(2)

print("\n" + "=" * 70)
print("性能对比总结")
print("=" * 70)
print(f"{'策略':<30} {'耗时(s)':<15} {'状态':<10}")
print("-" * 70)

for res in results:
    status = "✓" if res['success'] else "✗"
    print(f"{res['name']:<30} {res['time']:<15.2f} {status:<10}")

print("=" * 70)

