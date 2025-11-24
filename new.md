整个系统针对现在现存的问题我经过深刻的思考想到了如下的系统性解决方案：
1.针对算法冗余速度极慢，这里主要存在的问题就是基于python逐像素计算带来巨大的计算耗时

解决方案是将原方案O(N)的逐点几何运算转化为新方案O(1)的矩阵位运算

将原方案遍历人的每一个像素点，逐个判断是否在门的几何多边形内

修改为在内存中画出门的局部形状，将人像矩阵与门矩阵叠在一起做“与”运算，直接统计重叠像素

这样的方案在精度没有损失的情况下带来了巨大的速度提升。

```markdown
# 1. 局部 ROI 切片 (不处理全图)
person_roi = person_mask[y1:y2, x1:x2]
# 2. 局部栅格化门掩码
cv2.fillPoly(door_roi, [door_poly - [x1, y1]], 1)
# 3. 极速核心：位与运算 + 非零统计
overlap = cv2.countNonZero(cv2.bitwise_and(person_roi, door_roi))
```

2.针对系统多线程出现的内存泄漏或溢出，需要将掩码数据暂转路径数据减少内存占用，在计算时实时转化为掩码进行计算。

这样的计算经过实验不会带来巨大的时间开销但是可以极大降低内存占用。

```markdown
# 掩码 -> 多边形
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
queue.put(max(contours, key=cv2.contourArea)) # 只传坐标点数组
# 多边形 -> 局部掩码
local_mask = np.zeros((h, w), dtype=np.uint8)
# 坐标平移后瞬间还原填色
cv2.fillPoly(local_mask, [received_polygon - [x1, y1]], 1)
```

3.要保持接口输入输出的一致性，即为了解决存储开销将掩码转化为路径，而进行位与运算是基于掩码的，那么就需要在这之前先将路径转化为掩码，这一件事需要被考虑为这个算法的一部分。