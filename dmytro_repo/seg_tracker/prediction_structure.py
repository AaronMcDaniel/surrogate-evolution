from dataclasses import dataclass
from typing import List

@dataclass
class Box:
    cx: float
    cy: float
    w: float
    h: float

    @property
    def left(self):
        return self.cx - self.w / 2

    @property
    def right(self):
        return self.cx + self.w / 2

    @property
    def top(self):
        return self.cy - self.h / 2

    @property
    def bottom(self):
        return self.cy + self.h / 2


@dataclass
class GtItem(Box):
    item_id: str
    distance: float
    matched_conf: float = 0.0


@dataclass
class DetectedItem(Box):
    item_id: str
    distance: float

    confidence: float
    dx: float
    dy: float

    is_matched_planned: bool = False
    is_matched_unplanned: bool = False
    matched_item_id: str = ''
    track_id: int = -1
    add_to_submit: bool = False


@dataclass
class FrameItems:
    predicted: List[DetectedItem]
    gt_planned: List[GtItem]
    gt_unplanned: List[GtItem]

    frame_img_fn: str = ''
    frame_img_prev_fn: str = ''

    transform_dx: float = 0
    transform_dy: float = 0
    transform_angle: float = 0
    transform_error: float = 0

