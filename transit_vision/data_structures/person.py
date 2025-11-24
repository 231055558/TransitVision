class Person:
    def __init__(self, track_id):
        self.id = track_id
        self.frames = []
        self.boxes = []
        self.masks = []  # 保留字段但不存数据(兼容性)
        self.mask_polygons = []  # 存储多边形路径(内存优化)
        self.confs = []
        self.trigger_frame = None
    
    def add_detection(self, frame_idx, box, mask_polygon=None, conf=None):
        self.frames.append(frame_idx)
        self.boxes.append(box)
        self.mask_polygons.append(mask_polygon)
        if conf is not None:
            self.confs.append(conf)
    
    @property
    def trajectory(self):
        if len(self.boxes) == 0:
            return []
        return [[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2] for box in self.boxes]
    
    @property
    def last_box(self):
        return self.boxes[-1] if self.boxes else None
    
    @property
    def last_mask(self):
        return self.masks[-1] if self.masks else None
    
    @property
    def last_polygon(self):
        return self.mask_polygons[-1] if self.mask_polygons else None
    
    def __len__(self):
        return len(self.frames)
