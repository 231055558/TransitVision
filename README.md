# TransitVision: Advanced Passenger OD Recognition System

[English](README.md) | [中文](README_CN.md)

TransitVision is a professional-grade solution for public transportation passenger analytics. It provides end-to-end Passenger **Origin-Destination (OD) recognition**, real-time **Occupancy Analysis**, and high-precision **Boarding/Alighting counting** tailored for extreme transit environments.

## 🚀 Core Contributions & Highlights

Our system is specifically engineered to overcome the unique challenges of bus-mounted surveillance:

- **Public Transport Specific Data Augmentation**: Specialized image preprocessing and augmentation pipelines to handle **extreme lighting variations**, **color distortion**, and **severe camera vibration** common in moving vehicles.
- **Fine-tuned YOLO11x-Seg & ReID**: Deeply optimized YOLO11x segmentation models for precise person/door masking and custom ReID hyperparameters to ensure robustness under low-quality imaging.
- **Pose2ID Integration**: We introduced a **Pose2ID** scheme that utilizes pose-aware feature extraction to significantly improve ReID accuracy across different camera perspectives (e.g., matching a front-door boarding view with a rear-door alighting view), effectively eliminating "camera view bias."
- **Industrial Deployment**: Successfully applied in urban public transport systems for transit network optimization and station throughput analysis.

---

## 🧠 Algorithmic Logic

TransitVision employs three sophisticated logic channels to ensure high accuracy in complex scenarios:

### 1. Adaptive Door Preprocessing (`door_preprocessor.py`)
Rather than using static regions of interest (ROI), our system dynamically segments the bus doors:
- **Denoising & CC Filtering**: Uses Connected Component (CC) analysis to isolate the door structure from background noise.
- **Geometric Correction**: Automatically calculates the door's rotation angle and applies affine transformations to align the coordinate system, ensuring consistent spatial logic regardless of camera installation tilt.

### 2. Pattern-based Boarding Recognition (`boarding_counter.py`)
To distinguish between actual passengers and pedestrians walking outside the vehicle:
- **Spatial Transition Analysis**: Tracks the "Overlap Ratio" between the person's bbox and the door mask. It looks for a specific pattern (Outside -> Inside -> Door Center Crossing).
- **Height-based Filtering**: Implements a physical constraint check. Passengers inside the bus must meet a relative height threshold (avg height > 48% of door height) to filter out false positives from the street level.

### 3. V3 Alighting Logic with Bitwise Optimization (`alighting_counter.py`)
For the high-density environment of a rear exit:
- **Matrix Bitwise Detection**: Instead of simple bbox checks, we use **Person Polygon + Door Mask** bitwise AND operations. This $O(1)$ complexity approach allows for pixel-perfect entry detection.
- **Grace Period Mechanism**: A "Disappearance Grace Period" logic is used to handle tracking fragmentation. If a passenger disappears while overlapping with the door mask after showing "intent to alight," they are counted, providing resilience against occlusion.

### 4. End-to-End OD Matching (`main.py`)
- **Passenger Database**: Maintains a lifecycle-aware database of "On-bus" passengers.
- **Dual-Stage Matching**: 
    1. **Immediate Match**: Real-time matching using ReID features during alighting.
    2. **Final Cross-Matching**: A global greedy matching optimization performed at the end of the trip to close the loop for "pending" passengers, maximizing the **Closure Rate**.

---

## 📂 Project Structure

```text
TransitVision/
├── transit_vision/           # Core Package
│   ├── core/                 # AI Modules (YOLO11-Seg, ReID, Pose2ID)
│   ├── logic/                # Business Logic (Boarding, Alighting, OD)
│   ├── threads/              # Parallel Processing Pipelines
│   ├── utils/                # Geometry & Image Helpers
│   └── main.py               # System Entry Point
├── configs/                  # System & Model Configurations
├── scripts/                  # Data Extraction & Training Scripts
├── tests/                    # Comprehensive Test Suite
└── README.md
```

---

## 📊 System Outputs

The system generates a rich set of analytics for every run:
- **Matched Passenger JSON**: Linked boarding and alighting stations for every recognized individual.
- **Occupancy Curves**: Visual representation of bus fullness across the entire route.
- **Station Statistics**: Detailed boarding/alighting counts per station.
- **Visual Evidence**: Captured images of every boarding and alighting event for auditing.

---

## 🛠️ Technical Stack
- **Vision**: YOLOv11x-seg, PyTorch, OpenCV
- **Tracking**: BoT-SORT (Segmentation-aware)
- **ReID**: Pose2ID Transformer / ResNet-based Feature Extractors
- **Architecture**: Multi-threaded Producer-Consumer Pipeline

---
*Generated with TransitVision — Precision Analytics for Modern Urban Mobility.*
