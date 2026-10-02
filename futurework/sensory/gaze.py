"""
视线追踪层：把人"正在看哪里"转成可操作的锚点。

视线在自然交互里是最省力的指代方式——人几乎不抬手就能说"这里"。
本模块提供屏幕归一化坐标、注视停留检测（dwell）与界面元素吸附，
并支持"无眼动仪硬件时"退化为鼠标/头部朝向近似。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from futurework.types import GazeSignal, GroundingTarget

# 最小停留时间（毫秒）与移动阈值：达到即可判定为"有意指向"
DWELL_THRESHOLD_MS = 450.0
DWELL_MOVE_TOLERANCE = 0.06


@dataclass
class UIElement:
    """可被视线吸附的界面元素。"""

    element_id: str
    title: str
    bbox: Tuple[float, float, float, float]  # (x, y, w, h)，归一化
    window_title: str = ""

    def contains(self, x: float, y: float) -> bool:
        bx, by, bw, bh = self.bbox
        return bx <= x <= bx + bw and by <= y <= by + bh


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


class GazeTracker:
    """
    视线追踪器：坐标平滑 + 停留判定 + 元素吸附。

    真实系统里原始眼动数据抖动很大，因此先做指数平滑，再判定停留，
    最后在注册的 UI 元素列表中做命中测试。
    """

    SMOOTHING = 0.35

    def __init__(self, smoothing: Optional[float] = None) -> None:
        self._alpha = self.SMOOTHING if smoothing is None else smoothing
        self._sx: Optional[float] = None
        self._sy: Optional[float] = None
        self._anchor: Optional[Tuple[float, float]] = None
        self._anchor_since: Optional[float] = None
        self._elements: List[UIElement] = []
        self._last: Optional[GazeSignal] = None

    def register_element(self, element: UIElement) -> None:
        self._elements.append(element)

    def register_elements(self, elements: Sequence[UIElement]) -> None:
        self._elements.extend(elements)

    # ------------------------------------------------------------------
    def update(
        self,
        x: float,
        y: float,
        *,
        confidence: float = 0.9,
        timestamp: Optional[float] = None,
        screen_w: float = 1920.0,
        screen_h: float = 1080.0,
    ) -> GazeSignal:
        now_ts = timestamp if timestamp is not None else time.time()
        sx = max(0.0, min(1.0, x))
        sy = max(0.0, min(1.0, y))

        if self._sx is None:
            self._sx, self._sy = sx, sy
        else:
            self._sx += self._alpha * (sx - self._sx)
            self._sy += self._alpha * (sy - self._sy)

        dwell_ms = 0.0
        if self._anchor is not None and _dist(self._anchor, (self._sx, self._sy)) <= DWELL_MOVE_TOLERANCE:
            dwell_ms = (now_ts - (self._anchor_since or now_ts)) * 1000.0
        else:
            self._anchor = (self._sx, self._sy)
            self._anchor_since = now_ts
            dwell_ms = 0.0

        hit = self.hit_test(self._sx, self._sy)
        signal = GazeSignal(
            screen_x=round(self._sx, 4),
            screen_y=round(self._sy, 4),
            confidence=confidence,
            fixation_duration_ms=round(dwell_ms, 1),
            target_window_title=hit.window_title if hit else None,
            target_element_id=hit.element_id if hit else None,
            timestamp=now_ts,
        )
        self._last = signal
        return signal

    def hit_test(self, x: float, y: float) -> Optional[UIElement]:
        """返回视点命中的界面元素（后注册的优先，便于覆盖层优先）。"""
        for element in reversed(self._elements):
            if element.contains(x, y):
                return element
        return None

    def is_dwelling(self, min_ms: float = DWELL_THRESHOLD_MS) -> bool:
        return self._last is not None and self._last.fixation_duration_ms >= min_ms

    def to_target(self, signal: Optional[GazeSignal] = None) -> Optional[GroundingTarget]:
        """把当前视线转成认知层可用的锚点对象。"""
        sig = signal or self._last
        if sig is None:
            return None
        box = 0.03
        return GroundingTarget(
            target_type="element" if sig.target_element_id else "coordinate",
            target_id=sig.target_element_id or f"gaze_{sig.screen_x:.3f}_{sig.screen_y:.3f}",
            label=sig.target_window_title or "视线焦点",
            bounding_box=(sig.screen_x - box / 2, sig.screen_y - box / 2, box, box),
            metadata={"dwell_ms": sig.fixation_duration_ms, "confidence": sig.confidence},
        )


@dataclass
class SimulatedEyeTracker:
    """脚本化眼动源（离线演示/测试）。"""

    samples: List[Tuple[float, float, float]] = field(default_factory=list)  # (x, y, dwell_ms)

    def stream(self) -> List[GazeSignal]:
        signals: List[GazeSignal] = []
        for x, y, dwell in self.samples:
            signals.append(
                GazeSignal(
                    screen_x=x,
                    screen_y=y,
                    confidence=0.88,
                    fixation_duration_ms=dwell,
                    timestamp=time.time(),
                )
            )
        return signals


def head_direction_to_screen(
    yaw: float,
    pitch: float,
    *,
    fov_deg: float = 60.0,
    yaw_range: float = 30.0,
    pitch_range: float = 20.0,
) -> Tuple[float, float]:
    """
    无眼动仪时的降级方案：把头部朝向线性映射到屏幕归一化坐标。

    精度不如真眼动，但配合"转头看哪"的操作习惯已足够实用。
    """
    nx = 0.5 + (max(-1.0, min(1.0, yaw / yaw_range))) * (fov_deg / 180.0)
    ny = 0.5 - (max(-1.0, min(1.0, pitch / pitch_range))) * (fov_deg / 180.0)
    return max(0.0, min(1.0, nx)), max(0.0, min(1.0, ny))