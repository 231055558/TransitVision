# 下客识别算法对比测试说明

## 版本说明

### V1 (原版本)
- **文件**: `alighting_counter_v1.py`
- **特点**: 函数式逻辑，基于上4/5掩码区域检测
- **判定方式**: 检查掩码上4/5区域是否85%进入门区域
- **优势**: 逻辑简单，计算快速

### V2 (新版本)  
- **文件**: `alighting_counter_v2.py` (合并后为 `alighting_counter.py`)
- **特点**: 类式API，基于点多边形测试
- **判定方式**: 使用cv2.pointPolygonTest检查掩码点是否在门多边形内
- **优势**: 更精确的几何判定，面向对象设计

## 统一接口

两个版本已统一为相同的类式API：

```python
class AlightingCounter:
    def __init__(self, config: dict)
    def reset(self)
    def update_counts(self, frame_idx: int, persons: Dict[int, Person], door: Door)
    def get_count(self) -> int
```

## 使用对比测试

### 切换版本

编辑 `test_alighting_compare.py` 第10行：

```python
USE_VERSION = "v1"  # 测试V1版本
# 或
USE_VERSION = "v2"  # 测试V2版本
```

### 运行测试

```bash
cd tests/logic_test
python test_alighting_compare.py
```

### 输出结果

- **终端输出**: 处理进度和最终计数
- **视频文件**: `output/alighting_result_v1.mp4` 或 `output/alighting_result_v2.mp4`
- **可视化**: 
  - 绿色: 已计为下车
  - 红色: 仅追踪中
  - 左上角显示版本和计数

## 性能对比指标

建议从以下维度对比：

1. **准确率**: 实际下车人数 vs 检测人数
2. **误报率**: 非下车乘客被误判的数量
3. **漏检率**: 下车乘客未被检测的数量  
4. **处理速度**: FPS和总处理时间
5. **稳定性**: 多次运行结果的一致性

## 测试数据

当前测试视频：
- `杨家门_down.mp4` - 典型下车场景

建议增加测试：
- 拥挤场景
- 快速下车场景
- 边缘情况（司机移动等）

