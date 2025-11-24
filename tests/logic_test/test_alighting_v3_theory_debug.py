import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import math
import random

# ==========================================
# 0. 辅助工具：生成复杂场景与对象 (升级版)
# ==========================================

class MovingObject:
    def __init__(self, start_pos, end_pos, size, shape_sides, color, offset_frame=0, duration_one_way=150):
        self.start_pos = np.array(start_pos, dtype=float)
        self.end_pos = np.array(end_pos, dtype=float)
        self.current_pos = np.array(start_pos, dtype=float)
        self.size = size
        self.sides = shape_sides
        self.color = color # BGR
        self.offset = offset_frame
        # 定义单程（从起点到终点）需要多少帧，控制速度
        # 30fps下，150帧=5秒走一趟，比较合理
        self.duration = duration_one_way 

    def update(self, current_frame):
        # 计算相对时间
        t = current_frame - self.offset
        if t < 0:
            return None # 还没出生
        
        # 实现乒乓往复运动 (Ping-Pong)
        # 周期 = 去程 + 回程 = duration * 2
        cycle = self.duration * 2
        phase = t % cycle
        
        if phase < self.duration:
            # 去程 (0 -> 1)
            progress = phase / self.duration
        else:
            # 回程 (1 -> 0)
            progress = 1.0 - ((phase - self.duration) / self.duration)
            
        # 缓动效果 (可选，让运动更自然)
        # progress = -0.5 * (math.cos(math.pi * progress) - 1)
        
        self.current_pos = self.start_pos + (self.end_pos - self.start_pos) * progress
        
        # 增加垂直方向的波浪干扰，增加轨迹复杂度
        self.current_pos[1] += 30 * math.sin(t * 0.05)
        
        return self.get_polygon_points()

    def get_polygon_points(self):
        pts = []
        cx, cy = self.current_pos
        angle_step = 360 / self.sides
        # 让它旋转起来 (随时间变化)
        rotation_angle = (cx + cy) * 0.05
        
        for i in range(self.sides * 2): 
            # 制造星形效果 (奇数点半径大，偶数点半径小)
            r = self.size if i % 2 == 0 else self.size * 0.5
            angle_rad = math.radians(i * angle_step / 2 + rotation_angle)
            x = int(cx + r * math.cos(angle_rad))
            y = int(cy + r * math.sin(angle_rad))
            pts.append([x, y])
        return np.array(pts, np.int32)

def generate_30s_simulation(filename="sim_source_30s.mp4", num_frames=900):
    """
    生成 30秒 (900帧) 的 720P 复杂模拟视频
    """
    width, height = 1280, 720 # 720P
    print(f"1. 正在生成 30秒/720P 源视频 (共 {num_frames} 帧): {filename} ...")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, 30.0, (width, height))
    
    # 定义一个巨大的、形状复杂的门 (模拟车门区域)
    door_pts = np.array([
        [300, 100], [900, 150], [1050, 360], 
        [900, 650], [350, 600], [200, 360]
    ], np.int32)
    door_list = door_pts.tolist()
    
    # 定义 5 个不同的物体，速度各异，来回穿梭
    objects = [
        # 1. 白色五角星，左右横穿，速度中等
        MovingObject((100, 300), (1100, 300), 60, 5, (255, 255, 255), 0, 150),
        # 2. 红色三角形，上下穿梭，速度快
        MovingObject((600, 50), (600, 650), 50, 3, (200, 200, 255), 10, 90),
        # 3. 绿色七边形，斜向运动，速度慢
        MovingObject((1100, 600), (200, 200), 55, 7, (200, 255, 200), 20, 200),
        # 4. 蓝色四角星，反向斜穿
        MovingObject((200, 100), (1000, 600), 45, 4, (255, 200, 200), 5, 120),
        # 5. 青色六边形，快速横扫
        MovingObject((100, 550), (1180, 150), 40, 6, (255, 255, 0), 40, 100),
    ]
    
    data_stream = []
    
    # 模拟进度条
    for i in range(num_frames):
        if i % 100 == 0: print(f"   已生成 {i}/{num_frames} 帧...")
        
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 画门 (蓝色粗线)
        cv2.polylines(frame, [door_pts], True, (255, 100, 0), 3)
        
        frame_objects_data = [] 
        
        for obj in objects:
            poly = obj.update(i)
            if poly is not None:
                # 模拟 YOLO 的输出：每个物体有一个独立的 Mask 和 Box
                mask = np.zeros((height, width), dtype=np.uint8)
                cv2.fillPoly(mask, [poly], 255)
                
                # 画在视频上
                cv2.fillPoly(frame, [poly], obj.color)
                
                # 计算 BBox
                x, y, w, h = cv2.boundingRect(poly)
                x = max(0, x); y = max(0, y)
                w = min(width - x, w); h = min(height - y, h)
                box = [x, y, x + w, y + h]
                
                frame_objects_data.append({
                    "mask": mask,
                    "box": box
                })
        
        out.write(frame)
        
        data_stream.append({
            "frame": frame,
            "objects": frame_objects_data,
            "door_points": door_list
        })
        
    out.release()
    print("   源视频生成完毕。")
    return data_stream

# ==========================================
# 1. V1 算法 (逐点计算 - 性能杀手)
# ==========================================
def process_v1_benchmark(data_stream, filename="result_30s_v1.mp4"):
    print(f"2. 正在运行 V1 (逐点计算) - 这可能需要一点时间 ...")
    h, w, _ = data_stream[0]['frame'].shape
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (w, h))
    
    total_time = 0
    frame_times = []
    
    num_frames = len(data_stream)
    
    for idx, data in enumerate(data_stream):
        if idx % 50 == 0: print(f"   V1 处理进度: {idx}/{num_frames} ...")
        
        frame_vis = data['frame'].copy()
        door_points = data['door_points']
        
        frame_start = time.perf_counter()
        
        # 遍历当前帧的所有物体
        for obj in data['objects']:
            mask = obj['mask']
            box = obj['box']
            x1, y1, x2, y2 = map(int, box)
            
            # --- V1 逻辑 ---
            mask_points = cv2.findNonZero(mask) 
            if mask_points is not None:
                upper_limit = y1 + (y2 - y1) * 4 / 5
                check_points = [p[0] for p in mask_points if p[0][1] <= upper_limit]
                
                if check_points:
                    polygon_np = np.array(door_points, np.int32)
                    
                    # 逐点判断 (只画少量点以节省可视化时间，但计算全量)
                    for pt_idx, pt in enumerate(check_points):
                        # 核心耗时点
                        is_in = cv2.pointPolygonTest(polygon_np, (float(pt[0]), float(pt[1])), False) >= 0
                        
                        # 可视化：只画部分点
                        if pt_idx % 30 == 0:
                            color = (0, 255, 0) if is_in else (0, 0, 255)
                            cv2.circle(frame_vis, (pt[0], pt[1]), 2, color, -1)

        frame_end = time.perf_counter()
        elapsed = (frame_end - frame_start)
        total_time += elapsed
        frame_times.append(elapsed * 1000)
        
        # 显示实时耗时
        cv2.putText(frame_vis, f"V1 Time: {elapsed*1000:.1f} ms", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        out.write(frame_vis)
        
    out.release()
    print("   V1 处理完毕。")
    return total_time, frame_times

# ==========================================
# 2. V2 算法 (矩阵运算 - 性能优化)
# ==========================================
def process_v2_benchmark(data_stream, filename="result_30s_v2.mp4"):
    print(f"3. 正在运行 V2 (矩阵运算) ...")
    h, w, _ = data_stream[0]['frame'].shape
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (w, h))
    
    total_time = 0
    frame_times = []
    
    num_frames = len(data_stream)
    
    for idx, data in enumerate(data_stream):
        if idx % 100 == 0: print(f"   V2 处理进度: {idx}/{num_frames} ...")

        frame_vis = data['frame'].copy()
        door_points = data['door_points']
        
        frame_start = time.perf_counter()
        
        # 遍历当前帧的所有物体
        for obj in data['objects']:
            mask = obj['mask']
            box = obj['box']
            x1, y1, x2, y2 = map(int, box)
            bw, bh = x2 - x1, y2 - y1
            
            # --- V2 逻辑 ---
            if bw > 0 and bh > 0:
                # 1. ROI Crop
                person_roi = mask[y1:y2, x1:x2]
                cut_h = int(bh * 4 / 5)
                
                if cut_h > 0:
                    person_upper = person_roi[:cut_h, :]
                    
                    # 2. 局部栅格化
                    local_poly = np.array(door_points, np.int32) - np.array([x1, y1])
                    door_roi_mask = np.zeros((cut_h, bw), dtype=np.uint8)
                    cv2.fillPoly(door_roi_mask, [local_poly], 255)
                    
                    # 3. 位运算
                    intersection = cv2.bitwise_and(person_upper, door_roi_mask)
                    
                    # 4. 可视化：青色高亮重叠区域
                    # 创建全图遮罩
                    color_mask = np.zeros_like(frame_vis)
                    # 填充局部
                    overlay_roi = color_mask[y1:y1+cut_h, x1:x2]
                    # 青色 BGR: [255, 255, 0]
                    overlay_roi[intersection > 0] = [255, 255, 0]
                    # 叠加
                    frame_vis = cv2.addWeighted(frame_vis, 1.0, color_mask, 0.6, 0)

        frame_end = time.perf_counter()
        elapsed = (frame_end - frame_start)
        total_time += elapsed
        frame_times.append(elapsed * 1000)
        
        cv2.putText(frame_vis, f"V2 Time: {elapsed*1000:.1f} ms", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        out.write(frame_vis)
        
    out.release()
    print("   V2 处理完毕。")
    return total_time, frame_times

# ==========================================
# 3. 结果汇总与绘图
# ==========================================
if __name__ == "__main__":
    # 1. 生成 30秒 720P 复杂数据
    data = generate_30s_simulation()
    
    # 2. 运行对比
    t_v1, times_v1 = process_v1_benchmark(data)
    t_v2, times_v2 = process_v2_benchmark(data)
    
    # 3. 计算倍率
    speedup = t_v1 / t_v2 if t_v2 > 0 else 0
    avg_v1 = np.mean(times_v1)
    avg_v2 = np.mean(times_v2)
    
    print("\n" + "="*50)
    print(f"30秒压力测试完成! (分辨率: 1280x720, 目标: 5个往复运动)")
    print("="*50)
    print(f"V1 (原算法) 总耗时: {t_v1:.4f} 秒 (平均每帧: {avg_v1:.2f} ms)")
    print(f"V2 (优化版) 总耗时: {t_v2:.4f} 秒 (平均每帧: {avg_v2:.2f} ms)")
    print("-" * 50)
    print(f"🚀 性能提速倍率: {speedup:.2f} 倍")
    print("="*50)
    
    # 4. 绘图
    plt.figure(figsize=(14, 7))
    plt.plot(times_v1, color='red', label='V1: Point Test (ms)', alpha=0.5, linewidth=1)
    plt.plot(times_v2, color='blue', label='V2: Matrix Op (ms)', linewidth=2)
    plt.title(f'30s Benchmark (720P, 5 Objects Ping-Pong)\nTotal Speedup: {speedup:.1f}x', fontsize=16)
    plt.ylabel('Processing Time per Frame (ms)', fontsize=12)
    plt.xlabel('Frame Index (Total 900 Frames)', fontsize=12)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 添加文字标注
    plt.text(10, max(times_v1)*0.9, f"V1 Avg: {avg_v1:.1f} ms", color='red', fontsize=12, fontweight='bold')
    plt.text(10, max(times_v1)*0.85, f"V2 Avg: {avg_v2:.1f} ms", color='blue', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("benchmark_30s_result.png")
    print("图表已保存: benchmark_30s_result.png")
    plt.show()