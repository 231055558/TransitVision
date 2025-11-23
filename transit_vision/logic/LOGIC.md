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

### 核心思路
通过检测掩码与门掩码的重叠判定下车行为。

### 判定条件
- 轨迹可靠性：连续5帧，帧间隔≤2
- 下车触发：从车外(mask未进门)到进入门区域(mask上4/5部分85%进门)
- 状态转变：0(门外) -> 1(门内)

### 关键函数
- `bbox_overlaps_mask(box, door_mask)`: 快速判断bbox与门掩码是否重叠
- `check_door_entry(mask, box, door_mask)`: 判断mask上4/5区域是否85%进入门
- `check_alighting_action(status_list)`: 检查0->1状态转变
- `filter_alighting_passengers(tracks, door_mask)`: 过滤出下车乘客

### 输入输出
- **输入**: 追踪结果tracks {id: Person}, 后门掩码
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

## 内存管理

### 下客流程的特殊性
下客判定是**唯一需要保存连续掩码结果**的流程，因为需要检测mask与门掩码的重叠状态变化(0→1转变)。

### 有界阻塞队列设计
为防止内存溢出，下客流程采用三线程流水线+有界队列：

1. **VideoReadThread**: 读取视频帧
2. **InferenceThread**: 批量GPU推理，产生掩码
3. **LogicThread**: 跟踪和计数

线程间使用 `queue.Queue(maxsize=N)` 连接：
- 若CPU处理速度 < GPU推理速度，队列满后推理线程阻塞
- 防止掩码数据无限堆积导致内存爆炸

### 其他流程
上客判定、占用率分析等流程**不需要保存连续掩码**，应避免在内存中累积掩码数据。

