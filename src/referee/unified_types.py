# -*- coding: utf-8 -*-
"""
统一的模型观测 / 事件数据结构。

设计原则（对应方案文档第 7 节）：
- 适配器只输出「逐帧观测 Observation」，绝不直接输出「谁得分」；
- 所有坐标统一还原到原始图像像素坐标（左上角原点，x 向右，y 向下）；
- 没有检测到球时也必须输出一条 visible=False 的观测，center_px 为 None，
  不允许填上一次的位置或用 (0,0) 表示丢失；
- model 自带的事件能力（如 TTNet 的 bounce / net）放在 event_scores 里，
  它们是「候选分数」而非「已确认事件」，必须经过事件提取与去重才能变成 Event。

这样 YOLO / TrackNet / TTNet 可以随时切换或组合，而计分规则完全不变。
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import math


# ---------------------------------------------------------------- 事件类型
class EventType:
    """状态机认识的统一事件类型。"""

    BOUNCE = 'BOUNCE'        # 球落台（在台面上弹一下）
    NET_HIT = 'NET_HIT'      # 擦网
    HIT = 'HIT'              # 球拍击球（由轨迹方向反转推断）
    SERVE = 'SERVE'          # 发球
    BALL_LOST = 'BALL_LOST'  # 球丢失（仅用于断段，绝不直接判失分）


class Side:
    """球台两侧。A 固定为图像左侧球员，B 为右侧球员。"""

    A = 'A'
    B = 'B'
    UNKNOWN = 'UNKNOWN'

    @staticmethod
    def opposite(side: str) -> str:
        if side == Side.A:
            return Side.B
        if side == Side.B:
            return Side.A
        return Side.UNKNOWN


# ---------------------------------------------------------------- 观测
@dataclass
class BallObservation:
    """单帧的球观测。"""

    visible: bool
    center_px: Optional[List[float]] = None      # [x, y]，原始图像像素坐标
    bbox_xyxy: Optional[List[int]] = None        # 无检测框时为 None
    confidence: Optional[float] = None           # 模型原始分数语义，需在使用前校准

    def __post_init__(self):
        if not self.visible:
            # 丢失时必须全部为 None，防止下游把占位值当成真实坐标
            self.center_px = None
            self.bbox_xyxy = None
            self.confidence = None


@dataclass
class Observation:
    """单帧统一观测。适配器的唯一输出类型。"""

    frame_id: int
    timestamp_s: float
    source: str                                  # yolo / tracknet / ttnet
    image_size: List[int]                        # [width, height] 原始分辨率
    ball: BallObservation
    event_scores: Dict[str, Optional[float]] = field(
        default_factory=lambda: {'bounce': None, 'net_hit': None})
    # 以下为可选扩展，不属于统一接口的最小集合
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'frame_id': self.frame_id,
            'timestamp_s': round(self.timestamp_s, 4),
            'source': self.source,
            'image_size': self.image_size,
            'ball': {
                'visible': self.ball.visible,
                'center_px': self.ball.center_px,
                'bbox_xyxy': self.ball.bbox_xyxy,
                'confidence': (round(self.ball.confidence, 4)
                               if self.ball.confidence is not None else None),
            },
            'event_scores': {k: (round(v, 4) if v is not None else None)
                             for k, v in self.event_scores.items()},
            'extras': self.extras,
        }


# ---------------------------------------------------------------- 事件
@dataclass
class Event:
    """事件处理层输出，状态机的唯一输入类型。"""

    event_id: str
    timestamp_s: float
    type: str                       # EventType.*
    side: str = Side.UNKNOWN        # 事件发生在哪一侧台面
    position_px: Optional[List[int]] = None
    confidence: Optional[float] = None
    sources: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'event_id': self.event_id,
            'timestamp_s': round(self.timestamp_s, 4),
            'type': self.type,
            'side': self.side,
            'position_px': self.position_px,
            'confidence': (round(self.confidence, 4)
                           if self.confidence is not None else None),
            'sources': list(self.sources),
        }


# ---------------------------------------------------------------- 工具
def distance(a: Optional[List[float]], b: Optional[List[float]]) -> float:
    """两点欧氏距离；任一点缺失时返回 inf，便于统一做阈值判断。"""
    if a is None or b is None:
        return math.inf
    return math.hypot(a[0] - b[0], a[1] - b[1])
