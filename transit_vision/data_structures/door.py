import numpy as np

class Door:
    def __init__(self, mask, bbox=None, angle=None):
        self.mask = mask
        self.bbox = bbox or self._calc_bbox(mask)
        self.angle = angle
    
    def _calc_bbox(self, mask):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None
        return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    
    @property
    def center(self):
        if self.bbox is None:
            return None
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    @property
    def area(self):
        return np.sum(self.mask > 0) if self.mask is not None else 0

