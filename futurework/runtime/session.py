"""
会话门面：用户与 FutureWork 交互的唯一入口。

把"怎么拼多模态输入"这件麻烦事收敛为几个自然的方法名::

    session = FutureWorkSession(workdir="~/proj")

    session.say("打开浏览器")                    # 只用语音
    session.say("把这个移到那里", gesture=POINT)  # 语音 + 手势（指哪）
    session.nod()                                # 点头授权，无需说话
    session.pinch()                              # 捏合确认
    session.shake()                              # 摇头拒绝

这正是人与人的交互方式：大部分时间用语言，偶尔用手势补充，
在需要确认时点点头就够了。
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from futurework.runtime.orchestrator import EventType, InteractionEvent, Orchestrator, TurnResult
from futurework.sensory.facial import HeadPoseAnalyzer
from futurework.sensory.gaze import GazeTracker, UIElement
from futurework.sensory.gesture import GestureInterpreter, PointingResolver
from futurework.sensory.temporal_aligner import StreamingPerceptionLoop, TemporalAlignmentBuffer
from futurework.tools.registry import ToolRegistry
from futurework.types import (
    EmotionType,
    ExecutionStatus,
    FacialSignal,
    GazeSignal,
    GestureType,
    GroundingTarget,
    HandGestureSignal,
    IntentCategory,
    SpeechSignal,
    SystemFeedback,
)


class FutureWorkSession:
    """一次人机交互会话。"""

    def __init__(
        self,
        *,
        workdir: Optional[str] = None,
        registry: Optional[ToolRegistry] = None,
        orchestrator: Optional[Orchestrator] = None,
    ) -> None:
        self.orchestrator = orchestrator or Orchestrator(workdir=workdir, registry=registry)
        self.head_pose = HeadPoseAnalyzer()
        self.gaze = GazeTracker()
        self.pointing = PointingResolver()
        self.gestures = GestureInterpreter()
        self.temporal_buffer = TemporalAlignmentBuffer()
        self.streaming_loop = StreamingPerceptionLoop(
            buffer=self.temporal_buffer,
            on_aligned_frame=self._on_aligned_frame,
        )
        self._last_gaze: Optional[GazeSignal] = None
        self._turns: List[TurnResult] = []

    def _on_aligned_frame(self, frame) -> None:
        """流式时间窗口对齐触发的回调。"""
        # 将对齐的多模态帧封装为交互事件触发
        event = InteractionEvent(
            speech=frame.speech,
            gestures=frame.gestures,
            facial=frame.facial,
            head_pose=frame.head_pose,
            gaze=frame.gaze,
        )
        self._run(event)

    def push_speech_stream(self, transcript: str, *, is_final: bool = True, timestamp: Optional[float] = None) -> Optional[TurnResult]:
        """
        向多模态流中推入实时语音识别文本片段。
        当 is_final=True 时，自动结合过去 400ms 内出现的手势、视线和表情进行高斯加权时间对齐，并触发执行！
        """
        signal = SpeechSignal(transcript=transcript, is_final=is_final, timestamp=timestamp or time.time())
        frame = self.streaming_loop.feed_speech(signal)
        return self._turns[-1] if frame and self._turns else None

    def push_gesture_stream(self, gesture: GestureType, position_3d: Tuple[float, float, float] = (0.5, 0.5, 0.5), timestamp: Optional[float] = None) -> None:
        """向多模态流中推入实时手势识别事件。"""
        sig = HandGestureSignal(gesture=gesture, position_3d=position_3d, timestamp=timestamp or time.time())
        self.streaming_loop.feed_gestures([sig])

    def push_gaze_stream(self, x: float, y: float, timestamp: Optional[float] = None) -> None:
        """向多模态流中推入实时眼动注视坐标。"""
        sig = self.gaze.update(x, y, timestamp=timestamp or time.time())
        self.streaming_loop.feed_gaze(sig)

    def push_facial_stream(self, emotion: EmotionType, timestamp: Optional[float] = None) -> None:
        """向多模态流中推入实时表情识别结果。"""
        sig = FacialSignal(primary_emotion=emotion, timestamp=timestamp or time.time())
        self.streaming_loop.feed_facial(sig)

    # ==================================================================
    # 便捷交互方法
    # ==================================================================
    def say(
        self,
        text: str,
        *,
        gesture: Optional[GestureType] = None,
        look_at: Optional[Tuple[float, float]] = None,
        expression: Optional[EmotionType] = None,
        gesture_position: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        confidence: float = 1.0,
    ) -> TurnResult:
        """
        说话，并可同时带上手势、视线、表情。

        :param gesture: 同时做出的手势（如 ``POINT`` 指向你说的"这个"）
        :param look_at: 注视的屏幕归一化坐标
        :param expression: 此刻的表情（会影响是否追问）
        """
        gestures: List[HandGestureSignal] = []
        if gesture is not None:
            signal = HandGestureSignal(gesture=gesture, position_3d=gesture_position, confidence=0.95)
            gestures.append(signal)
            self.pointing.feed(signal)

        gaze_signal: Optional[GazeSignal] = None
        if look_at is not None:
            gaze_signal = self.gaze.update(look_at[0], look_at[1])
            self._last_gaze = gaze_signal

        facial = None
        if expression is not None:
            facial = FacialSignal(primary_emotion=expression, attention_score=0.95, confidence=0.9)

        return self._run(InteractionEvent(
            speech=SpeechSignal(transcript=text, confidence=confidence),
            gestures=gestures,
            facial=facial,
            gaze=gaze_signal,
        ))

    def nod(self, *, confidence: float = 0.95) -> TurnResult:
        """点头——自然交互中最普遍的"同意"。"""
        pose = self.head_pose.update(pitch=14.0, yaw=0.0, confidence=confidence, nod=True)
        return self._run(InteractionEvent(
            speech=SpeechSignal(transcript="", confidence=0.0),
            head_pose=pose,
        ))

    def shake(self, *, confidence: float = 0.95) -> TurnResult:
        """摇头——"不同意 / 不是这个"。"""
        pose = self.head_pose.update(pitch=0.0, yaw=18.0, confidence=confidence, shake=True)
        return self._run(InteractionEvent(
            speech=SpeechSignal(transcript="", confidence=0.0),
            head_pose=pose,
        ))

    def pinch(self, *, confidence: float = 0.94) -> TurnResult:
        """捏合手势——确认当前操作。"""
        signal = HandGestureSignal(gesture=GestureType.PINCH, confidence=confidence)
        self.pointing.feed(signal)
        return self._run(InteractionEvent(gestures=[signal]))

    def stop(self) -> TurnResult:
        """张开手掌——"停"。安全停止信号，优先级最高。"""
        signal = HandGestureSignal(gesture=GestureType.PALM_STOP, confidence=0.98)
        return self._run(InteractionEvent(gestures=[signal]))

    def point(self, x: float, y: float, z: float = 0.5) -> HandGestureSignal:
        """手势指向某个屏幕位置（用于给后续指令提供锚点）。"""
        signal = HandGestureSignal(
            gesture=GestureType.POINT,
            position_3d=(x, y, z),
            point_direction=(0.0, 0.0, -1.0),
            confidence=0.9,
        )
        self.pointing.feed(signal)
        self._last_gaze = self.gaze.update(x, y)
        return signal

    def register_ui(self, elements: Sequence[UIElement]) -> None:
        """注册界面元素，让视线能被吸附到具体控件。"""
        self.gaze.register_elements(elements)

    def on_feedback(self, callback: Callable[[SystemFeedback, TurnResult], None]) -> Callable[[], None]:
        """
        订阅反馈流——接 TTS / 屏幕提示 / 触觉反馈都走这里。

        回调收到完整的 :class:`SystemFeedback`（含语音文本、提示音、告警级别）
        以及本轮的 :class:`TurnResult`。返回取消订阅的函数。

        ::

            session.on_feedback(lambda fb, turn: speak(fb.speech_audio_text or fb.text))
        """
        def _dispatch(event) -> None:
            payload = event.payload or {}
            raw = payload.get("feedback")
            feedback = SystemFeedback(**raw) if raw else SystemFeedback(text=payload.get("text", ""))
            callback(feedback, self._turns[-1] if self._turns else None)

        return self.orchestrator.bus.subscribe(EventType.RESPONSE.value, _dispatch)

    # 旧名字保留，避免破坏既有调用方
    feedback = on_feedback

    # ==================================================================
    def undo(self) -> Dict[str, Any]:
        """撤销上一步。"""
        return self.orchestrator.undo_last()

    def health(self) -> Dict[str, Any]:
        return self.orchestrator.health()

    @property
    def turns(self) -> List[TurnResult]:
        return list(self._turns)

    @property
    def last(self) -> Optional[TurnResult]:
        return self._turns[-1] if self._turns else None

    # ==================================================================
    def _run(self, event: InteractionEvent) -> TurnResult:
        result = self.orchestrator.interact(event)
        self._turns.append(result)
        return result


class TextSession(FutureWorkSession):
    """
    纯文本会话：无需任何硬件，适合脚本、测试、CI。

    支持中文/英文口语化表达，并支持 ``>`` 前缀标记手势（演示用）::

        TextSession().run("打开浏览器")
        TextSession().run("把这个移到那里 [POINT@0.3,0.4]")
    """

    def run(self, text: str) -> TurnResult:
        gesture, position = _parse_inline_gesture(text)
        clean = _strip_inline_gesture(text)
        return self.say(clean, gesture=gesture, gesture_position=position)


def _as_gesture(name: str) -> Optional[GestureType]:
    """手势名 → 枚举。不区分大小写（``POINT`` / ``point`` 都接受）。"""
    token = name.strip().lower()
    try:
        return GestureType(token)
    except ValueError:
        return None


def _parse_inline_gesture(text: str) -> Tuple[Optional[GestureType], Tuple[float, float, float]]:
    """解析 ``[POINT@0.3,0.4]`` 形式的内联手势标记。"""
    if "[" not in text or "]" not in text:
        return None, (0.5, 0.5, 0.5)
    inside = text[text.index("[") + 1 : text.index("]")]
    if "@" not in inside:
        return _as_gesture(inside), (0.5, 0.5, 0.5)
    name, coords = inside.split("@", 1)
    try:
        gesture = _as_gesture(name)
        parts = [float(p) for p in coords.split(",")]
        while len(parts) < 3:
            parts.append(0.5)
        return gesture, (parts[0], parts[1], parts[2])
    except (ValueError, IndexError):
        return None, (0.5, 0.5, 0.5)


def _strip_inline_gesture(text: str) -> str:
    if "[" not in text or "]" not in text:
        return text.strip()
    return (text[: text.index("[")] + text[text.index("]") + 1 :]).strip()