# 逻辑模块说明

## 概述

逻辑模块负责判定上下车行为，包含三个核心部分：
1. **上客判定** (boarding_counter)
2. **下客判定** (alighting_counter)  
3. **门预处理** (door_preprocessor)

## 1. 上客判定逻辑

### 核心思路
通过轨迹与门框bbox的重合度变化判定上车行为。

### 判定条件
- 轨迹可靠性：连续7帧，帧间隔≤2
- 上车模式：从门框内(>=85%重合)移动到门框右侧(<=40%重合)
- 过滤反向：若前3帧以上在门外，判定为回来刷卡，排除

### 关键函数
- `calc_overlap_ratio(box1, box2)`: 计算box1与box2重合度
- `check_boarding_pattern(boxes, door_bbox)`: 检查是否符合上车模式
- `is_reliable_track(frames)`: 检查轨迹可靠性
- `filter_boarding_passengers(tracks, door_bbox)`: 过滤出上车乘客

### 输入输出
- **输入**: 追踪结果tracks {id: Person}, 旋转后门框bbox
- **输出**: 过滤后的上车乘客 {id: Person}

## 2. 下客判定逻辑

### 版本演进

#### V2 (alighting_counter_v2.py)
- **算法**: 逐点几何运算 `cv2.pointPolygonTest`
- **复杂度**: O(N) - N为掩码像素点数
- **性能**: 慢，Python逐点计算

#### V3 (alighting_counter_v3.py) ⭐ 推荐
- **算法**: 矩阵位运算 `cv2.bitwise_and`
- **复杂度**: O(1) - 矩阵运算
- **性能**: 快，显著提升 (10-50倍加速)
- **内存**: 使用多边形存储，大幅降低内存占用

### 核心思路
通过检测人物多边形与门多边形的重叠判定下车行为。

### 判定条件
- 轨迹可靠性：连续5帧，帧间隔≤2
- 下车触发：从车外到进入门区域(上4/5部分≥50%重叠)
- 状态转变：0(门外) -> 1(门内)

### 关键函数 (V3)
- `polygon_to_local_mask(polygon, box)`: 多边形转局部掩码
- `check_door_entry_v3(person_polygon, box, door_polygon)`: 矩阵位运算检测进门
- `check_box_overlap_with_polygon(box, door_polygon)`: 快速bbox重叠检测
- `filter_alighting_passengers(tracks, door_mask)`: 过滤出下车乘客

### 输入输出
- **输入**: 追踪结果tracks {id: Person}, Person.mask_polygons存储多边形
- **输出**: 过滤后的下车乘客 {id: Person}

## 3. 门预处理逻辑

### 前车门预处理
1. 获取门分割掩码
2. 形态学去噪(kernel=5)
3. 连通域过滤(保留>=10%最大面积)
4. 计算门角度(旋转至正立)
5. 计算旋转后bbox坐标

**输出**: (angle, bbox) - 用于视频旋转和上客判定

### 后车门预处理
1. 获取门分割掩码
2. 形态学去噪
3. 连通域过滤

**输出**: 过滤后的掩码 - 用于下客判定

### 关键函数
- `preprocess_front_door(door_seg, frame)`: 处理前车门，返回角度和bbox
- `preprocess_rear_door(door_seg)`: 处理后车门，返回掩码

## 工作流程

### 前车门(上客)流程
```
视频首帧 -> 门检测 -> 门预处理(得到angle+bbox) 
    -> 视频读取+旋转(angle) -> 人员追踪 
    -> 上客判定(bbox) -> 上车乘客结果
```

### 后车门(下客)流程
```
视频首帧 -> 门检测 -> 门预处理(得到mask) 
    -> 视频读取 -> 人员追踪 
    -> 下客判定(mask) -> 下车乘客结果
```

## 效率优化

- 门检测和预处理：每辆车仅执行一次
- 前车门angle+bbox：缓存后用于该车所有前门视频
- 后车门mask：缓存后用于该车所有后门视频

## 内存管理与性能优化

### 多边形存储方案 (V3核心优化)

#### 问题
- 掩码数据量巨大: 720×1280×1字节 ≈ 0.9MB/帧
- 多人多帧累积导致内存爆炸: 50人×100帧×0.9MB ≈ 4.5GB

#### 解决方案
**掩码 ↔ 多边形 互转**:

```python
# 输出时: 掩码 -> 多边形 (压缩)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
polygon = max(contours, key=cv2.contourArea)  # 只传坐标点数组

# 输入时: 多边形 -> 局部掩码 (还原)
local_mask = np.zeros((h, w), dtype=np.uint8)
local_poly = polygon - np.array([x1, y1])  # 坐标平移
cv2.fillPoly(local_mask, [local_poly], 255)  # 瞬间还原
```

#### 效果
- **内存**: 降低1000倍+ (0.9MB -> 几KB)
- **速度**: CPU开销<1ms/帧，可忽略
- **精度**: 无损，完全保留原始形状

### 有界阻塞队列设计
为防止内存溢出，下客流程采用三线程流水线+有界队列：

1. **VideoReadThread**: 读取视频帧
2. **InferenceThread**: GPU推理，输出多边形
3. **LogicThread**: 收集追踪结果和计数

线程间使用 `queue.Queue(maxsize=N)` 连接：
- 若CPU处理速度 < GPU推理速度，队列满后推理线程阻塞
- 多边形数据极小，队列不会爆内存

### 数据流
- **PersonSegTracker**: 输出多边形 (不输出掩码)
- **Person.mask_polygons**: 存储多边形路径
- **Person.masks**: 保留字段但不存数据 (兼容性)
- **下客逻辑**: 接收多边形，内部转局部掩码进行位运算

### 其他流程
上客判定只需要boxes，不保存任何mask/polygon数据。

