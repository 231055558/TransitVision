import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import math
import random

# ==========================================
# 1. 准备数据
# ==========================================
def get_complex_poly(center, radius, num_points=150):
    """生成高复杂度锯齿多边形"""
    pts = []
    cx, cy = center
    for i in range(num_points):
        angle = math.radians(i * (360 / num_points))
        r = radius + random.randint(-20, 20) 
        x = int(cx + r * math.cos(angle))
        y = int(cy + r * math.sin(angle))
        pts.append([x, y])
    return np.array(pts, np.int32)

def generate_test_data(num_frames=100):
    print(f"1. 正在生成 {num_frames} 帧高复杂度掩码数据...")
    data = []
    width, height = 1280, 720
    
    for i in range(num_frames):
        mask = np.zeros((height, width), dtype=np.uint8)
        center = (200 + i*8, 360)
        poly = get_complex_poly(center, 150, num_points=150)
        cv2.fillPoly(mask, [poly], 255)
        
        # 预先算好 BBox，因为这是检测模型直接给出的，不算在掩码处理耗时里
        x, y, w, h = cv2.boundingRect(poly)
        
        data.append({
            "mask": mask,
            "box": (x, y, w, h)
        })
    return data

# ==========================================
# 2. 核心对比逻辑
# ==========================================
if __name__ == "__main__":
    raw_data = generate_test_data(10000)
    
    print("\n=== 全链路耗时对比 (CPU Cost Comparison) ===")
    print("条件: 100帧, 720P分辨率, 150+顶点复杂多边形")
    
    # --- 方案 A: Bitmap 全链路 ---
    # 流程: Copy(传输) -> Slice(准备计算)
    print("\n[方案 A: 直接传图片]")
    t_start = time.perf_counter()
    
    for item in raw_data:
        mask = item['mask']
        x, y, w, h = item['box']
        
        # 1. 生产者：复制 (模拟存入队列)
        transferred_mask = mask.copy()
        
        # 2. 消费者：ROI 切片 (准备进行 V2 矩阵运算)
        # 这是 V2 算法的第一步：只取局部
        roi_mask = transferred_mask[y:y+h, x:x+w]
        
        # (此时 roi_mask 已经可以直接用于 bitwise_and 了)
        
    t_end = time.perf_counter()
    time_a = (t_end - t_start) * 1000
    
    # --- 方案 B: Polygon 全链路 ---
    # 流程: findContours(压缩) -> zeros+fillPoly(解压/还原)
    print("[方案 B: 传多边形 + 还原]")
    t_start = time.perf_counter()
    
    for item in raw_data:
        mask = item['mask']
        x, y, w, h = item['box']
        
        # 1. 生产者：提取轮廓 (压缩)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            polygon = max(contours, key=cv2.contourArea)
            
            # (模拟传输过程，polygon 极小，传输时间忽略不计)
            
            # 2. 消费者：还原为局部掩码 (解压)
            # 创建局部画布
            restored_roi = np.zeros((h, w), dtype=np.uint8)
            
            # 坐标平移：全局坐标 -> 局部坐标
            # 这一步是必须的，因为我们在小画布上画图
            local_polygon = polygon - np.array([x, y])
            
            # 填充 (Rasterization)
            cv2.fillPoly(restored_roi, [local_polygon], 255)
            
            # (此时 restored_roi 已经可以直接用于 bitwise_and 了)
            
    t_end = time.perf_counter()
    time_b = (t_end - t_start) * 1000

    # ==========================================
    # 3. 结果统计与分析
    # ==========================================
    print("\n" + "="*50)
    print("📊 最终耗时报告")
    print("="*50)
    
    avg_a = time_a / 100
    avg_b = time_b / 100
    
    print(f"方案 A (Bitmap Copy + Slice):")
    print(f"  - 总耗时: {time_a:.2f} ms")
    print(f"  - 单帧耗时: {avg_a:.3f} ms")
    
    print(f"\n方案 B (Contour Extract + FillPoly Restore):")
    print(f"  - 总耗时: {time_b:.2f} ms")
    print(f"  - 单帧耗时: {avg_b:.3f} ms")
    
    diff = avg_b - avg_a
    
    print("-" * 50)
    print(f"结论分析:")
    print(f"1. 纯 CPU 耗时增加: {diff:.3f} ms/帧")
    if diff > 0:
        print(f"   (方案 B 确实比 方案 A 慢)")
    
    print("\n【工程决策依据】")
    print(f"为了节省 1000倍 的内存带宽，我们每帧多支付了 {diff:.3f} ms 的 CPU 时间。")
    print(f"对于 30 FPS 的视频，每帧预算约 33ms。")
    print(f"这个 'CPU 税' 占每帧预算的: {(diff/33)*100:.2f}%")
    print("="*50)
    
    # 绘图
    plt.figure(figsize=(8, 6))
    plt.bar(['A: Bitmap (Full)', 'B: Polygon (Round-Trip)'], [time_a, time_b], color=['gray', 'orange'])
    plt.ylabel('Total Time (ms)')
    plt.title('CPU Time Cost Comparison (100 Frames)\nSimulating Full Producer-Consumer Lifecycle')
    
    # 标数值
    plt.text(0, time_a, f"{time_a:.1f} ms", ha='center', va='bottom', fontsize=12, fontweight='bold')
    plt.text(1, time_b, f"{time_b:.1f} ms", ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("lifecycle_time_benchmark.png")
    print("图表已保存: lifecycle_time_benchmark.png")
    plt.show()