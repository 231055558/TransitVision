from collections import deque

class Person:
    def __init__(self, track_id):
        self.id = track_id
        self.frames = []
        self.boxes = []
        self.masks = []
        self.confs = []

        self.has_disembark_intent = False
        self.has_counted = False
        self.inside_door_history = deque(maxlen=20)

    def add_detection(self, frame_idx, box, mask=None, conf=None):
        self.frames.append(frame_idx)
        self.boxes.append(box)
        if mask is not None:
            self.masks.append(mask)
        if conf is not None:
            self.confs.append(conf)

    @property
    def trajectory(self):
        if not self.boxes:
            return []
        return [[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2] for box in self.boxes]

    @property
    def last_box(self):
        return self.boxes[-1] if self.boxes else None

    @property
    def last_mask(self):
        return self.masks[-1] if self.masks else None

    @property
    def last_seen_frame(self):
        return self.frames[-1] if self.frames else -1

    def __len__(self):
        return len(self.frames)