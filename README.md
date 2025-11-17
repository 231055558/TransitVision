## 文件结构说明
```
TransitVision/
├── transit_vision/           # 核心源代码包
├── configs/                  # 配置文件目录
├── data/                     # 数据样本和测试视频
├── ckpt/                     # 存放模型权重文件
├── output/                   # 存放运行结果 (日志, 可视化视频, 统计数据)
├── scripts/                  # 辅助脚本 (如数据预处理, 模型评估)
├── tests/                    # 单元测试和集成测试
├── README.md                 # 项目说明文档
└── requirements.txt          # 项目依赖库
```

## transit_vision 核心源代码包详解
```
transit_vision/
├── __init__.py
|
├── main.py                   # 程序主入口，负责启动和协调所有线程
|
├── core/                     # 核心算法模块
│   ├── __init__.py
│   ├── detection/            # 分割与检测模块
│   │   ├── __init__.py
│   │   ├── person_seg.py  # 封装YOLOv11实例分割模型
│   │   └── door_seg.py     # 封装YOLOv11门框检测模型
│   │
│   ├── reid/                 # 重识别模块
│   │   ├── __init__.py
│   │   └── feature_similarity.py # 封装PASS或您选择的ReID模型
│   │
│   └── prediction/           # 客流预估模块 (暂时留白)
│       ├── __init__.py
│       └── flow_predictor.py
|
├── logic/                    # 业务逻辑处理模块
│   ├── __init__.py
│   ├── alighting_counter.py    # 下客识别与计数逻辑
│   ├── occupancy_analyzer.py   # 满载率分析逻辑
│   ├── od_matcher.py           # 客流OD匹配与管理逻辑
│   └── boarding_counter.py     # 上客识别与计数逻辑
|
├── threads/                  # 多线程管理模块
│   ├── __init__.py
│   ├── video_capture_thread.py # 视频读取线程
│   ├── inference_thread.py     # 推理计算线程
│   └── visualization_thread.py # 可视化结果生成线程
|
├── utils/                    # 通用工具函数模块（还没）
│   ├── __init__.py
│   ├── video_utils.py        # 视频读写、编解码相关工具
│   ├── image_utils.py        # 图像处理、预处理工具
│   ├── visualization.py      # 绘制BBox, Mask, 轨迹等可视化函数
│   └── config_loader.py      # 加载和解析配置文件
|
└── data_structures/          # 自定义数据结构（还没）
    ├── __init__.py
    ├── track.py              # 定义Track对象 (包含ID, bbox, mask, feature, history等)
    └── frame_data.py         # 定义FrameData对象，用于在线程间传递数据
```