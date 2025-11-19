import argparse
import cv2
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
import torch
from PIL import Image, ImageDraw, ImageFont

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from transit_vision.utils.video_reader import VideoReader
from transit_vision.utils.config_loader import load_config
from transit_vision.utils import DeviceConfig
from transit_vision.core.detection.door_seg import DoorSegmentor
from transit_vision.core.detection.head_detector import HeadDetector
from transit_vision.logic.occupancy_analyzer import OccupancyAnalyzer

def put_chinese_text(img, text, position, font_scale=0.7, color=(255, 255, 255)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    font_size = int(font_scale * 35)
    font_paths = [
        "simhei.ttf", "msyh.ttc", "simsun.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simsun.ttc"
    ]
    
    font = None
    for font_path in font_paths:
        try:
            font = ImageFont.truetype(font_path, font_size)
            break
        except IOError:
            continue
    
    if font is None:
        try:
            font = ImageFont.load_default()
        except IOError:
            print("警告: 无法加载字体")
            return img

    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def main(args):
    config = load_config(args.config)
    if not config:
        print(f"错误: 无法加载配置 '{args.config}'")
        return

    try:
        device_config_path = 'configs/device_debug.yaml'
        device_cfg = DeviceConfig(device_config_path)
        print(f"设备: {device_cfg.device}")

        video_reader = VideoReader(args.video)
        door_segmentor = DoorSegmentor(config['models']['door_segmentation'], device_cfg)
        head_detector = HeadDetector(config['models']['head_detection'])
        occupancy_analyzer = OccupancyAnalyzer(config)
    except Exception as e:
        print(f"初始化失败: {e}")
        return

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{Path(args.video).stem}_occupancy_test.mp4"
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), 
                             video_reader.fps, (video_reader.width, video_reader.height))

    ret, first_frame = video_reader.read()
    if not ret:
        print("错误: 无法读取第一帧")
        video_reader.release()
        writer.release()
        return
    
    print("检测车门...")
    door_obj = door_segmentor.detect(first_frame)
    if door_obj is None:
        print("警告: 未检测到车门")
    else:
        print(f"车门锚点: {door_obj.anchor}")

    video_reader.seek(0)

    print(f"分析视频: {args.video}")
    for frame in tqdm(video_reader, total=video_reader.frame_count, desc="处理"):
        head_detections = head_detector.detect_and_track(frame, conf=config['occupancy_analyzer']['head_conf_threshold'], device=device_cfg.device)

        analysis_result = occupancy_analyzer.analyze(head_detections, door_obj)

        vis_frame = frame.copy()
        status_color = analysis_result['status_color']
        
        if door_obj and door_obj.anchor:
            door_overlay = np.zeros_like(vis_frame, np.uint8)
            door_overlay[door_obj.mask > 0] = (255, 0, 0)
            vis_frame = cv2.addWeighted(vis_frame, 1.0, door_overlay, 0.3, 0)
            cv2.circle(vis_frame, door_obj.anchor, 10, (0, 0, 255), -1)
            door_dist_threshold = config['occupancy_analyzer']['door_distance_threshold']
            cv2.circle(vis_frame, door_obj.anchor, door_dist_threshold, (0, 0, 255), 2)

        for det in head_detections:
            x1, y1, x2, y2 = det['box']
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.rectangle(vis_frame, (20, 20), (70, 70), status_color, -1)

        occupancy_rate = analysis_result['occupancy_rate']
        seat_vacancy_rate = 1.0 - occupancy_rate
        
        text_x = video_reader.width - 400
        info_text1 = f"总人数: {analysis_result['person_count']} | 空座率: {seat_vacancy_rate:.0%}"
        info_text2 = f"门前(<{config['occupancy_analyzer']['door_distance_threshold']}px): {analysis_result['near_door_count']}"
        info_text3 = f"状态: {analysis_result['status_reason']}"

        vis_frame = put_chinese_text(vis_frame, info_text1, (text_x, 30), color=(0, 255, 255))
        vis_frame = put_chinese_text(vis_frame, info_text2, (text_x, 65), color=(0, 165, 255))
        vis_frame = put_chinese_text(vis_frame, info_text3, (text_x, 100), color=status_color)

        writer.write(vis_frame)

    writer.release()
    video_reader.release()
    print(f"\n完成: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='拥挤度分析测试')
    parser.add_argument('-v', '--video', type=str, required=True, help='输入视频路径')
    parser.add_argument('-c', '--config', type=str, default='configs/logic_config.yaml', help='配置文件路径')
    parser.add_argument('-o', '--output', type=str, default='output', help='输出目录')
    
    args = parser.parse_args()
    main(args)

