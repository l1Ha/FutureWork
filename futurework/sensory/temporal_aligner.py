"""
异步时间窗口对齐与多模态流式感知循环 (Temporal Alignment & Streaming Perceptual Loop)

物理现实中各传感器采样率与传输延迟完全异步：
- 摄像头手势/表情: 30 fps (~33ms)
- 眼动仪/视线: 60 - 250 Hz (4ms - 16ms)
- 麦克风/ASR: 流式分块音频或 VAD 语音断句 (~100-300ms)

人在说"把这个移过去"时：
- "这个"发音时刻为 t0
- 手指指向并在屏幕锚定的时刻通常在 t0 - 200ms 至 t0 + 250ms 之间
- 点头或表情反馈通常在 t0 之后 100ms ~ 500ms

本模块通过高斯核时间权重窗口实现多模态事件在时序上的自动对齐与同步截取。
"""

from __future__ import annotations

import asyncio
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Tuple

from futurework.types import (
    FacialSignal,
    GazeSignal,
    HandGestureSignal,
    HeadPoseSignal,
    MultimodalFrame,
    SpeechSignal,
)


@dataclass
class TimestampedEvent:
    timestamp: float
    channel: str  # "speech", "gesture", "facial", "head_pose", "gaze"
    payload: Any


class TemporalAlignmentBuffer:
    """
    高精度时间滑动缓冲区，维护各个独立感知通道的时间序列。
    """

    def __init__(self, window_seconds: float = 4.0) -> None:
        self.window_seconds = window_seconds
        self._lock = threading.RLock()
        self._speech_buffer: Deque[Tuple[float, SpeechSignal]] = deque()
        self._gesture_buffer: Deque[Tuple[float, List[HandGestureSignal]]] = deque()
        self._facial_buffer: Deque[Tuple[float, FacialSignal]] = deque()
        self._pose_buffer: Deque[Tuple[float, HeadPoseSignal]] = deque()
        self._gaze_buffer: Deque[Tuple[float, GazeSignal]] = deque()

    def push_speech(self, signal: SpeechSignal, ts: Optional[float] = None) -> None:
        now = ts or signal.timestamp or time.time()
        with self._lock:
            self._speech_buffer.append((now, signal))
            self._prune(now)

    def push_gestures(self, gestures: List[HandGestureSignal], ts: Optional[float] = None) -> None:
        now = ts or (gestures[0].timestamp if gestures else time.time())
        with self._lock:
            self._gesture_buffer.append((now, gestures))
            self._prune(now)

    def push_facial(self, signal: FacialSignal, ts: Optional[float] = None) -> None:
        now = ts or signal.timestamp or time.time()
        with self._lock:
            self._facial_buffer.append((now, signal))
            self._prune(now)

    def push_head_pose(self, signal: HeadPoseSignal, ts: Optional[float] = None) -> None:
        now = ts or signal.timestamp or time.time()
        with self._lock:
            self._pose_buffer.append((now, signal))
            self._prune(now)

    def push_gaze(self, signal: GazeSignal, ts: Optional[float] = None) -> None:
        now = ts or signal.timestamp or time.time()
        with self._lock:
            self._gaze_buffer.append((now, signal))
            self._prune(now)

    def align_frame(
        self,
        anchor_timestamp: Optional[float] = None,
        *,
        sigma_seconds: float = 0.35,
    ) -> MultimodalFrame:
        """
        以 anchor_timestamp 为中心，使用高斯时间权重核提取各个通道最相关的特征片段，
        构造时间对齐的统一 MultimodalFrame。
        """
        with self._lock:
            t_center = anchor_timestamp or time.time()

            # 1. 语音信号：取离中心最近的 final 信号或最新信号
            best_speech = self._select_closest(self._speech_buffer, t_center, sigma_seconds)

            # 2. 手势信号：取时间窗口内显著度最高（乘高斯时间衰减）的手势列表
            best_gestures = self._select_best_gestures(self._gesture_buffer, t_center, sigma_seconds)

            # 3. 面部表情：取最靠近且置信度最高的表情
            best_facial = self._select_closest(self._facial_buffer, t_center, sigma_seconds)

            # 4. 头部姿态：取窗口内具有 nod/shake 或最接近的姿态
            best_pose = self._select_best_head_pose(self._pose_buffer, t_center, sigma_seconds)

            # 5. 视线信号：取最接近或停留时间最长的视线点
            best_gaze = self._select_closest(self._gaze_buffer, t_center, sigma_seconds)

            return MultimodalFrame(
                timestamp=t_center,
                speech=best_speech,
                gestures=best_gestures or [],
                facial=best_facial,
                head_pose=best_pose,
                gaze=best_gaze,
                camera_active=bool(best_facial or best_pose or best_gestures),
                microphone_active=bool(best_speech),
            )

    # ------------------------------------------------------------------
    # 辅助算子
    # ------------------------------------------------------------------
    def _gaussian_weight(self, t: float, t_center: float, sigma: float) -> float:
        dt = t - t_center
        return math.exp(-(dt * dt) / (2.0 * sigma * sigma))

    def _select_closest(self, buf: Deque[Tuple[float, Any]], t_center: float, sigma: float) -> Optional[Any]:
        if not buf:
            return None
        # 寻找与中心时间最贴合且高斯权重不低于阈值的项
        best_item = None
        best_weight = 0.05
        for ts, item in buf:
            w = self._gaussian_weight(ts, t_center, sigma)
            if w > best_weight:
                best_weight = w
                best_item = item
        return best_item

    def _select_best_gestures(self, buf: Deque[Tuple[float, List[HandGestureSignal]]], t_center: float, sigma: float) -> List[HandGestureSignal]:
        if not buf:
            return []
        best_list: List[HandGestureSignal] = []
        max_score = 0.0
        for ts, g_list in buf:
            w = self._gaussian_weight(ts, t_center, sigma)
            if not g_list:
                continue
            # 综合置信度 * 时间窗口贴合度
            score = max(g.confidence for g in g_list) * w
            if score > max_score:
                max_score = score
                best_list = g_list
        return best_list

    def _select_best_head_pose(self, buf: Deque[Tuple[float, HeadPoseSignal]], t_center: float, sigma: float) -> Optional[HeadPoseSignal]:
        if not buf:
            return None
        # 点头或摇头动作往往持续 200~400ms，在时间窗口内优先保留带有手势的姿态
        gesture_items = []
        for ts, pose in buf:
            w = self._gaussian_weight(ts, t_center, sigma)
            if w > 0.05:
                if pose.nod or pose.shake:
                    return pose
                gesture_items.append((w, pose))
        if gesture_items:
            gesture_items.sort(key=lambda x: x[0], reverse=True)
            return gesture_items[0][1]
        return None

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        for b in (self._speech_buffer, self._gesture_buffer, self._facial_buffer, self._pose_buffer, self._gaze_buffer):
            while b and b[0][0] < cutoff:
                b.popleft()

    def clear(self) -> None:
        with self._lock:
            for b in (self._speech_buffer, self._gesture_buffer, self._facial_buffer, self._pose_buffer, self._gaze_buffer):
                b.clear()


class StreamingPerceptionLoop:
    """
    异步多模态持续感知流循环：
    后台监听多传感器并发数据输入，发生意图或语音断句时触发对齐与下游推理回调。
    """

    def __init__(
        self,
        buffer: Optional[TemporalAlignmentBuffer] = None,
        on_aligned_frame: Optional[Callable[[MultimodalFrame], None]] = None,
    ) -> None:
        self.buffer = buffer or TemporalAlignmentBuffer()
        self.on_aligned_frame = on_aligned_frame
        self._running = False
        self._lock = threading.Lock()

    def feed_speech(self, signal: SpeechSignal) -> Optional[MultimodalFrame]:
        self.buffer.push_speech(signal)
        # 语音完成断句 (is_final) 时是最高频的感知对齐触发点
        if signal.is_final and signal.transcript.strip():
            frame = self.buffer.align_frame(signal.timestamp)
            if self.on_aligned_frame:
                self.on_aligned_frame(frame)
            return frame
        return None

    def feed_gestures(self, gestures: List[HandGestureSignal]) -> None:
        self.buffer.push_gestures(gestures)

    def feed_facial(self, signal: FacialSignal) -> None:
        self.buffer.push_facial(signal)

    def feed_head_pose(self, signal: HeadPoseSignal) -> Optional[MultimodalFrame]:
        self.buffer.push_head_pose(signal)
        # 突发点头或摇头也是独立触发点
        if signal.nod or signal.shake:
            frame = self.buffer.align_frame(signal.timestamp)
            if self.on_aligned_frame:
                self.on_aligned_frame(frame)
            return frame
        return None

    def feed_gaze(self, signal: GazeSignal) -> None:
        self.buffer.push_gaze(signal)
