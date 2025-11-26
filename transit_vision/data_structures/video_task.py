from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np

@dataclass
class VideoTask:
    video_path: Path
    bus_id: str
    direction: str = 'unknown'
    station_id: int = 0
    station_name: str = ''
    need_door_seg: bool = False
    need_angle_calc: bool = False
    first_frame: Optional[np.ndarray] = None
    door_mask: Optional[np.ndarray] = None
    door_angle: Optional[float] = None
    
    def __post_init__(self):
        if isinstance(self.video_path, str):
            self.video_path = Path(self.video_path)
        
        if not self.video_path.exists():
            raise FileNotFoundError(f"video not found: {self.video_path}")

@dataclass
class ProcessedVideo:
    bus_id: str
    video_path: Path
    door_mask: Optional[np.ndarray] = None
    door_angle: Optional[float] = None
    fps: int = 30
    width: int = 0
    height: int = 0
    frame_count: int = 0



