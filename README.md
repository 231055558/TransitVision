## 文件结构说明
```
TransitVision/
├── transit_vision/           # 核心源代码包
├── configs/                  # 配置文件目录
├── data/                     # 数据样本和测试视频
│   ├── close_loop_od/        # 闭环OD测试数据
│   ├── reid_test/            # ReID测试数据
│   ├── reid_dataset/         # ReID数据集(脚本生成)
│   └── reid_features/        # ReID特征数据(测试生成)
├── ckpt/                     # 存放模型权重文件
├── output/                   # 存放运行结果
├── scripts/                  # 辅助脚本
│   └── extract_reid_data.py  # ReID数据提取脚本
├── tests/                    # 测试目录
│   ├── core_test/            # 核心算法测试
│   ├── logic_test/           # 业务逻辑测试
│   ├── threads_test/         # 多线程性能测试
│   ├── system_test/          # 系统模块化测试
│   ├── utils_test/           # 工具函数测试
│   └── pose2id_scheme/       # ReID相关测试
├── README.md
└── requirements.txt
```

## transit_vision 核心源代码包
```
transit_vision/
├── __init__.py
├── main.py
│
├── core/                     # 核心算法模块
│   ├── __init__.py
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── person_seg.py     # 人员分割追踪
│   │   ├── door_seg.py       # 门检测分割
│   │   └── head_detector.py  # 头部检测
│   └── reid/                 # 重识别模块
│       ├── __init__.py
│       ├── feature_extractor.py
│       ├── feature_similarity.py
│       ├── matcher.py
│       ├── nfc.py
│       └── pose2id_model/
│
├── logic/                    # 业务逻辑模块
│   ├── __init__.py
│   ├── alighting_counter.py  # 下客逻辑
│   ├── boarding_counter.py   # 上客逻辑
│   ├── door_preprocessor.py  # 门预处理
│   ├── occupancy_analyzer.py # 满载率分析
│   └── od_matcher.py         # OD匹配
│
├── threads/                  # 多线程模块
│   ├── __init__.py
│   ├── input_channel.py      # 输入通道
│   ├── inference_channel.py  # 推理通道
│   ├── logic_channel.py      # 逻辑运算通道
│   └── alighting_pipeline.py # 下客流水线
│
├── utils/                    # 工具模块
│   ├── __init__.py
│   ├── video_reader.py
│   ├── device.py
│   ├── image_ops.py
│   ├── angle_calc.py
│   ├── driver_mask.py
│   ├── frame_selector.py
│   ├── bbox_saver.py
│   ├── config_loader.py
│   └── reid_utils.py
│
└── data_structures/          # 数据结构
    ├── __init__.py
    ├── person.py
    ├── door.py
    └── video_task.py
```

## tests 测试目录
```
tests/
├── core_test/
│   ├── test_person_seg.py
│   ├── test_door_seg.py
│   └── test_reid.py
│
├── logic_test/
│   ├── test_alighting.py
│   ├── test_boarding.py
│   └── test_occupancy.py
│
├── threads_test/
│   ├── test_gpu_strategy2_shared_sequential.py
│   ├── test_alighting_pipeline.py
│   ├── test_full_pipeline_monitor.py
│   ├── test_gpu_batch.py
│   └── test_logic_performance.py
│
├── system_test/
│   ├── test_input_channel.py
│   ├── test_inference_channel.py
│   ├── test_logic_channel.py
│   └── test_reid_channel.py
│
├── utils_test/
│   ├── test_video_reader.py
│   └── test_device.py
│
└── pose2id_scheme/
    └── (ReID相关测试和数据)
```

## 代码风格规范（严格遵守）

**左倾风格**: 代码追求简洁高效，命名精炼，避免冗余
- 变量/函数名简短达意
- 避免过度抽象和嵌套
- 优先使用列表推导和函数式写法

**注释规范**: 仅在逻辑复杂处添加必要注释
- 核心算法逻辑: 简要说明关键步骤
- 工具函数: 无需注释，代码即文档
- 阈值参数: 标注含义

## 开发规则

**严格禁止**:
- 禁止创建任何代码外文件(md/txt/sh等)，除非明确要求
- 禁止为测试目的创建临时代码文件
- 禁止修改README.md，除非明确要求
- 所有临时测试直接在命令行完成或测试后立即删除代码文件
- 禁止主动进行git操作(包括提交)，除非我明确要求

**允许操作**:
- 修改现有代码文件
- 创建明确要求的代码文件
- 根据要求完成后更新README的文件目录
- 在对话中使用命令行测试
