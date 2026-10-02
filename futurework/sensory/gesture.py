"""
手势感知层：手部关键点 → 语义手势 → 空间指向。

手势在自然交互中承担三种角色：
1. **指令**：捏合=确认、张开手掌=停止、竖拇指=通过、握拳=撤销；
2. **指向**：手指点向屏幕某处，为"把这个""移到那里"提供坐标；
3. **修饰**：手势与语音同时出现时加强/否定语音意图（融合层处理）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from futurework.types import GestureType, GroundingTarget, HandGestureSignal

# 21 个手部关键点的常用子集（MediaPipe Hands 拓扑）
LANDMARK_TIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
FINGER_PIPS = {"thumb": 2, "index": 6, "middle": 10, "ring": 14, "pinky": 18}


@dataclass
class Landmark:
    """单个手部关键点（归一化坐标）。"""

    x: float
    y: float
    z: float = 0.0


def _dist(a: Landmark, b: Landmark) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def _angle(a: Landmark, b: Landmark, c: Landmark) -> float:
    """三点夹角（度），用于判断手指伸展/弯曲。"""
    v1 = (a.x - b.x, a.y - b.y, a.z - b.z)
    v2 = (c.x - b.x, c.y - b.y, c.z - b.z)
    dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
    n1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2)
    n2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2 + v2[2] ** 2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cos = max(-1.0, min(1.0, dot / (n1 * n2)))
    return math.degrees(math.acos(cos))


class GestureInterpreter:
    """
    基于几何规则的手势识别器。

    规则化实现的好处：无 GPU 也能跑、可单元测试、行为可预测；
    真实部署时可替换为小型 CNN 分类器，但保持 ``classify`` 接口不变。
    """

    # 指尖到掌心的平均距离阈值（相对手掌尺寸）
    EXTENDED_RATIO = 1.35
    FOLDED_RATIO = 0.75
    PINCH_MAX_DISTANCE = 0.055
    SWIPE_MIN_DISPLACEMENT = 0.22
    SWIPE_MAX_DURATION_S = 1.2

    def __init__(self) -> None:
        self._last: Optional[HandGestureSignal] = None
        self._trail: List[Tuple[float, Tuple[float, float, float]]] = []

    # ------------------------------------------------------------------
    def classify(self, landmarks: Sequence[Landmark], *, hand: str = "right") -> HandGestureSignal:
        """
        从 21 个关键点判定手势。

        :raises ValueError: 关键点数量不足时立即报错，由融合层捕获并降级。
        """
        if len(landmarks) < 21:
            raise ValueError(f"hand landmarks need 21 points, got {len(landmarks)}")

        palm = _palm_center(landmarks)
        palm_size = max(1e-6, _palm_size(landmarks))

        extended = {
            name: _dist(landmarks[idx], palm) / palm_size >= self.EXTENDED_RATIO
            for name, idx in LANDMARK_TIPS.items()
        }
        folded = {
            name: _dist(landmarks[idx], palm) / palm_size <= self.FOLDED_RATIO
            for name, idx in LANDMARK_TIPS.items()
        }

        pinch_dist = _dist(landmarks[4], landmarks[8])
        index_tip = landmarks[8]
        thumb_tip = landmarks[4]

        # 四指（食指到小指）的状态——拇指单独处理，避免与握拳混淆
        others = ("index", "middle", "ring", "pinky")
        others_extended = any(extended[n] for n in others)
        others_folded = sum(1 for n in others if folded[n])
        # 竖拇指与握拳的可靠区别是拇指是否伸展：竖拇指时拇指伸出远离掌心，
        # 握拳时拇指横搭在指背上紧贴掌心。只看拇指的 y 坐标不可靠——
        # 握拳时拇指同样位于掌心上方，会把每一次握拳都读成"赞"。
        thumb_stick_out = extended["thumb"]

        gesture = GestureType.IDLE
        point_direction: Optional[Tuple[float, float, float]] = None
        confidence = 0.55

        # 判定顺序即优先级：捏合 > 单指指向 > 竖拇指 > 倒拇指 > 握拳 > 张开手掌
        if pinch_dist < self.PINCH_MAX_DISTANCE * palm_size * 2:
            gesture, confidence = GestureType.PINCH, 0.94

        elif extended["index"] and not any(extended[n] for n in ("middle", "ring", "pinky")):
            gesture, confidence = GestureType.POINT, 0.9
            direction = (
                index_tip.x - landmarks[6].x,
                index_tip.y - landmarks[6].y,
                index_tip.z - landmarks[6].z,
            )
            norm = math.sqrt(sum(c * c for c in direction)) or 1.0
            point_direction = tuple(c / norm for c in direction)

        elif not others_extended and others_folded >= 3 and thumb_stick_out and thumb_tip.y < palm.y:
            gesture, confidence = GestureType.THUMBS_UP, 0.86

        elif not others_extended and others_folded >= 3 and not thumb_stick_out and thumb_tip.y > palm.y:
            gesture, confidence = GestureType.THUMBS_DOWN, 0.78

        elif not others_extended and others_folded >= 3:
            gesture, confidence = GestureType.FIST, 0.88

        elif all(extended.values()):
            gesture, confidence = GestureType.OPEN_PALM, 0.85

        signal = HandGestureSignal(
            gesture=gesture,
            confidence=confidence,
            hand=hand,
            position_3d=(palm.x, palm.y, palm.z),
            pinch_distance=pinch_dist,
            point_direction=point_direction,
        )
        self._update_velocity(signal)
        self._last = signal
        return signal

    def classify_swipe(self, landmarks: Sequence[Landmark], *, timestamp: Optional[float] = None, hand: str = "right") -> HandGestureSignal:
        """在 ``classify`` 基础上追加挥动（swipe）与双指滚动识别。"""
        signal = self.classify(landmarks, hand=hand)
        now_ts = timestamp if timestamp is not None else signal.timestamp
        self._trail.append((now_ts, signal.position_3d))
        self._trail = self._trail[-16:]

        if len(self._trail) >= 2:
            t0, p0 = self._trail[0]
            t1, p1 = self._trail[-1]
            dt = t1 - t0
            if dt <= self.SWIPE_MAX_DURATION_S:
                dx, dy = p1[0] - p0[0], p1[1] - p0[1]
                if abs(dx) > abs(dy):
                    if dx >= self.SWIPE_MIN_DISPLACEMENT:
                        signal.gesture = GestureType.SWIPE_RIGHT
                    elif -dx >= self.SWIPE_MIN_DISPLACEMENT:
                        signal.gesture = GestureType.SWIPE_LEFT
                else:
                    if dy >= self.SWIPE_MIN_DISPLACEMENT:
                        signal.gesture = GestureType.SWIPE_DOWN
                    elif -dy >= self.SWIPE_MIN_DISPLACEMENT:
                        signal.gesture = GestureType.SWIPE_UP
                signal.velocity = ((p1[0] - p0[0]) / max(dt, 1e-3), (p1[1] - p0[1]) / max(dt, 1e-3), 0.0)

        # 捏合距离的变化量映射为缩放
        if signal.pinch_distance is not None and self._last and self._last.pinch_distance:
            ratio = self._last.pinch_distance / max(signal.pinch_distance, 1e-6)
            if ratio > 1.4:
                signal.gesture = GestureType.ZOOM_OUT
            elif ratio < 0.7:
                signal.gesture = GestureType.ZOOM_IN
        return signal

    def _update_velocity(self, signal: HandGestureSignal) -> None:
        if self._last is not None:
            dt = max(signal.timestamp - self._last.timestamp, 1e-3)
            signal.velocity = (
                (signal.position_3d[0] - self._last.position_3d[0]) / dt,
                (signal.position_3d[1] - self._last.position_3d[1]) / dt,
                0.0,
            )


@dataclass
class SimulatedHandTracker:
    """脚本化手势源，用于离线演示与回归测试。"""

    gestures: List[HandGestureSignal] = field(default_factory=list)

    def stream(self) -> List[HandGestureSignal]:
        return list(self.gestures)


class PointingResolver:
    """
    指向解析器：把"指向手势 + 视线焦点"解析为屏幕坐标与窗口目标。

    真实交互中人常用"手指 + 视线"共同指代目标，因此融合两者：
    指尖方向决定粗定位，视线方向决定细定位，二者加权。
    """

    GAZE_WEIGHT = 0.6
    HAND_WEIGHT = 0.4

    def __init__(self) -> None:
        self._recent: List[HandGestureSignal] = []

    def feed(self, gesture: HandGestureSignal) -> None:
        self._recent.append(gesture)
        self._recent = [g for g in self._recent if gesture.timestamp - g.timestamp < 2.0]

    def active_pointing(self) -> Optional[HandGestureSignal]:
        for gesture in reversed(self._recent):
            if gesture.gesture in (GestureType.POINT, GestureType.PINCH):
                return gesture
        return None

    def resolve(
        self,
        *,
        screen_w: float,
        screen_h: float,
        gaze: Optional[Tuple[float, float]] = None,
    ) -> Optional[GroundingTarget]:
        """
        解析当前指向目标。

        :returns: 归一化边界框 ``(x, y, w, h)``，无指向时返回 ``None``。
        """
        pointing = self.active_pointing()
        if pointing is None:
            return None

        hand_x, hand_y, hand_z = pointing.position_3d
        if gaze is not None:
            gx, gy = gaze
            x = self.GAZE_WEIGHT * gx + self.HAND_WEIGHT * hand_x
            y = self.GAZE_WEIGHT * gy + self.HAND_WEIGHT * hand_y
        else:
            x, y = hand_x, hand_y

        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        # 目标框大小随手的 Z 深度变化：手越近，框越大（越精细的操作）
        box = max(0.02, min(0.25, 0.05 + 0.08 * (1.0 - hand_z)))
        return GroundingTarget(
            target_type="coordinate",
            target_id=f"pt_{int(x * screen_w)}_{int(y * screen_h)}",
            label="指向目标",
            bounding_box=(x, y, box, box),
            metadata={"source": "gesture+gaze" if gaze else "gesture", "confidence": pointing.confidence},
        )


def _palm_center(landmarks: Sequence[Landmark]) -> Landmark:
    """掌心（腕点 + 四个掌指关节）的质心。返回 Landmark 以便直接参与距离计算。"""
    ids = (0, 5, 9, 13, 17)
    n = len(ids)
    return Landmark(
        sum(landmarks[i].x for i in ids) / n,
        sum(landmarks[i].y for i in ids) / n,
        sum(landmarks[i].z for i in ids) / n,
    )


def _palm_size(landmarks: Sequence[Landmark]) -> float:
    return max(1e-6, _dist(landmarks[0], landmarks[9]))


def gesture_to_action(gesture: GestureType) -> str:
    """手势 → 认知层动作语义的映射表。"""
    mapping: Dict[GestureType, str] = {
        GestureType.PINCH: "confirm",
        GestureType.THUMBS_UP: "approve",
        GestureType.THUMBS_DOWN: "reject",
        GestureType.PALM_STOP: "stop",
        GestureType.OPEN_PALM: "stop",
        GestureType.FIST: "undo",
        GestureType.SWIPE_LEFT: "previous",
        GestureType.SWIPE_RIGHT: "next",
        GestureType.SWIPE_UP: "scroll_up",
        GestureType.SWIPE_DOWN: "scroll_down",
        GestureType.POINT: "select",
        GestureType.WAVE: "greet",
    }
    return mapping.get(gesture, gesture.value)