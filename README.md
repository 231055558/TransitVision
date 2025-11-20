## 文件结构说明
```
TransitVision/
├── transit_vision/           # 核心源代码包
├── configs/                  # 配置文件目录
├── data/                     # 数据样本和测试视频
├── ckpt/                     # 存放模型权重文件
├── output/                   # 存放运行结果 (日志, 可视化视频, 统计数据)
├── scripts/                  # 辅助脚本 (单线程批量处理)
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
│   │   ├── person_seg.py     # 人员分割追踪(PersonSegTracker)
│   │   └── door_seg.py       # 门检测分割(DoorSegmentor)
│   │
│   ├── reid/                 # 重识别模块(基于Pose2ID)
│   │   ├── __init__.py
│   │   ├── feature_extractor.py  # ReID特征提取器
│   │   ├── nfc.py                # 邻域特征中心化
│   │   ├── matcher.py            # 特征匹配算法
│   │   └── pose2id_model/        # Pose2ID模型代码
│   │
│   └── prediction/           # 客流预估模块 (暂时留白)
│       ├── __init__.py
│       └── flow_predictor.py
|
├── logic/                    # 业务逻辑处理模块
│   ├── __init__.py
│   ├── alighting_counter.py    # 下客识别与计数逻辑
│   ├── boarding_counter.py     # 上客识别与计数逻辑
│   ├── door_preprocessor.py    # 门预处理逻辑
│   ├── occupancy_analyzer.py   # 满载率分析逻辑
│   └── od_matcher.py           # 客流OD匹配与管理逻辑
|
├── threads/                  # 多线程管理模块
│   ├── __init__.py
│   ├── video_capture_thread.py # 视频读取线程
│   ├── inference_thread.py     # 推理计算线程
│   └── visualization_thread.py # 可视化结果生成线程
|
├── utils/                    # 通用工具函数模块
│   ├── __init__.py
│   ├── video_reader.py       # 视频读取工具
│   ├── device.py             # 设备配置管理
│   ├── image_ops.py          # 图像操作(旋转、掩码、去噪)
│   ├── angle_calc.py         # 角度计算工具
│   ├── driver_mask.py        # 司机掩码提取
│   ├── frame_selector.py     # 帧选择工具
│   ├── bbox_saver.py         # bbox截图保存
│   └── reid_utils.py         # ReID工具函数
|
└── data_structures/          # 自定义数据结构
    ├── __init__.py
    ├── person.py             # 定义Person对象(ID, frames, boxes, masks等)
    └── door.py               # 定义Door对象(mask, bbox, angle等)
```

## 代码风格规范

**左倾风格**: 代码追求简洁高效，命名精炼，避免冗余
- 变量/函数名简短达意
- 避免过度抽象和嵌套
- 优先使用列表推导和函数式写法

**注释规范**: 仅在逻辑复杂处添加必要注释
- 核心算法逻辑: 简要说明关键步骤
- 工具函数: 无需注释，代码即文档
- 阈值参数: 标注含义(如"门内高度需达48%")

## 开发规则

**严格禁止**:
- 禁止创建任何代码外文件(md/txt/sh等)，除非明确要求
- 禁止为测试目的创建临时代码文件
- 禁止修改README.md，除非明确要求
- 所有临时测试直接在命令行完成或测试后立即删除代码文件

**允许操作**:
- 修改现有代码文件
- 创建明确要求的代码文件
- 在对话中使用命令行测试