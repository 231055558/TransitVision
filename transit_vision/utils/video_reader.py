import cv2
from pathlib import Path

class VideoReader:
    def __init__(self, path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"video not found: {path}")
        
        ext = self.path.suffix.lower()
        if ext not in ['.mp4', '.avi']:
            raise ValueError(f"unsupported format: {ext}")
        
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise IOError(f"failed to open: {path}")
        
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def read(self):
        ret, frame = self.cap.read()
        return ret, frame
    
    def seek(self, frame_idx):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    
    def release(self):
        self.cap.release()
    
    def __iter__(self):
        return self
    
    def __next__(self):
        ret, frame = self.read()
        if not ret:
            raise StopIteration
        return frame
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.release()

def read_first_frame(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"failed to open: {video_path}")
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise IOError(f"failed to read frame: {video_path}")
    
    return frame

