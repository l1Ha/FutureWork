"""
面部表情与头部姿态感知层。

在自然工作流中，面部表情是"认知负荷"与"授权意愿"的隐性通道：
皱眉（困惑）→ 主动追问；挑眉/点头 → 确认；摇头 → 拒绝；长时间紧锁眉头
→ 判断用户已疲劳并主动降噪。这些都通过 ``FacialExpressionAnalyzer`` 暴露。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from futurework.types import EmotionType, FacialSignal, HeadPoseSignal

# FACS Action Units 及其情绪含义（AUs 为面部动作编码单元标准）
AU_TO_EMOTION: Dict[str, EmotionType] = {
    "AU1": EmotionType.SURPRISED,   # 内眉抬起
    "AU4": EmotionType.FRUSTRATED,  # 眉头下压
    "AU6": EmotionType.SATISFIED,   # 脸颊抬起
    "AU7": EmotionType.SATISFIED,   # 眼睑收紧
    "AU9": EmotionType.NEUTRAL,     # 皱眉
    "AU12": EmotionType.SATISFIED,  # 嘴角上扬
    "AU15": EmotionType.FRUSTRATED, # 嘴角下撇
    "AU25": EmotionType.FOCUSED,    # 唇抿紧
    "AU26": EmotionType.SURPRISED,  # 双唇张开
    "AU43": EmotionType.FOCUSED,    # 眉头微皱
}

EMOTION_AUS: Dict[EmotionType, List[str]] = {
    EmotionType.NEUTRAL: [],
    EmotionType.FOCUSED: ["AU25", "AU43"],
    EmotionType.CONFUSED: ["AU4", "AU9"],
    EmotionType.FRUSTRATED: ["AU4", "AU15"],
    EmotionType.SATISFIED: ["AU6", "AU12"],
    EmotionType.SURPRISED: ["AU1", "AU26"],
    EmotionType.TIRED: ["AU43"],
}

# 组合表情：单个 AU 不足以定性，组合才有明确含义。
# 皱眉（AU4）单独出现是"不满"，叠加上唇皱（AU9）才是典型的"困惑"——
# 这正是需要多模态融合而非单点判断的典型场景。
AU_COMBINATIONS: List[tuple] = [
    ({"AU4", "AU9"}, EmotionType.CONFUSED),
    ({"AU1", "AU4"}, EmotionType.CONFUSED),
    ({"AU4", "AU15"}, EmotionType.FRUSTRATED),
    ({"AU6", "AU12"}, EmotionType.SATISFIED),
    ({"AU1", "AU26"}, EmotionType.SURPRISED),
    ({"AU25", "AU43"}, EmotionType.FOCUSED),
]


@dataclass
class Landmark3D:
    """面部关键点（欧拉角 + 归一化坐标）。"""

    x: float
    y: float
    z: float = 0.0


class FacialExpressionAnalyzer:
    """
    基于 FACS 动作单元的表情解析器（无需深度模型即可运行）。

    ``analyze`` 接收 AU 激活强度字典，输出带置信度的 ``FacialSignal``；
    置信度会随"最活跃 AU 的强度"上升，避免小抖动被误判为情绪。
    """

    AMBIGUOUS_THRESHOLD = 0.7
    ATTENTION_DECAY = 0.02

    def __init__(self) -> None:
        self._attention = 0.95
        self._streak: List[EmotionType] = []

    def analyze(
        self,
        action_units: Dict[str, float],
        *,
        blink: bool = False,
        face_detected: bool = True,
    ) -> FacialSignal:
        scores: Dict[EmotionType, float] = {emotion: 0.0 for emotion in EmotionType}
        cleaned: Dict[str, float] = {}
        for au, raw in action_units.items():
            intensity = max(0.0, min(1.0, float(raw)))
            cleaned[au.upper()] = round(intensity, 3)
            emotion = AU_TO_EMOTION.get(au.upper())
            if emotion and emotion is not EmotionType.NEUTRAL:
                scores[emotion] = max(scores[emotion], intensity)
                scores[emotion] += 0.15 * intensity  # 多 AU 共振加权

        # 组合表情优先于单 AU 判定
        active = {au for au, v in cleaned.items() if v >= 0.3}
        for combo, emotion in AU_COMBINATIONS:
            if combo <= active:
                strength = min(cleaned[au] for au in combo)
                scores[emotion] = max(scores[emotion], strength + 0.2)

        scores = {k: min(1.0, v) for k, v in scores.items()}
        peak = max(scores.values()) if scores else 0.0

        if not face_detected or peak < 0.25:
            primary = EmotionType.NEUTRAL
        else:
            primary = max(scores, key=lambda k: scores[k])

        self._attention = self._update_attention(primary, blink)
        self._streak.append(primary)
        if len(self._streak) > 8:
            self._streak = self._streak[-8:]

        if primary is EmotionType.FOCUSED:
            confidence = min(0.95, 0.6 + 0.4 * peak)
        else:
            confidence = min(0.92, 0.45 + 0.55 * peak)

        return FacialSignal(
            primary_emotion=primary,
            emotion_scores={k.value: round(v, 3) for k, v in scores.items()},
            action_units=cleaned,
            eye_blink=blink,
            attention_score=round(self._attention, 3),
            confidence=round(confidence, 3),
        )

    def classify_blink(
        self,
        landmarks: Sequence[Landmark3D],
        *,
        ear_threshold: float = 0.21,
    ) -> bool:
        """
        基于 EAR（眼睛纵横比）判断眨眼。

        EAR = (‖p2−p6‖ + ‖p3−p5‖) / (2·‖p1−p4‖)，低于阈值即闭眼。
        """
        # 索引 1..6 即 EAR 六点（共需 7 个点，索引 0 不用）
        if len(landmarks) < 7:
            return False
        try:
            ear = (
                _d(landmarks[2], landmarks[6]) + _d(landmarks[3], landmarks[5])
            ) / (2.0 * max(1e-6, _d(landmarks[1], landmarks[4])))
        except (IndexError, TypeError):
            return False
        return ear < ear_threshold

    def _update_attention(self, emotion: EmotionType, blink: bool) -> float:
        if emotion in (EmotionType.FOCUSED, EmotionType.NEUTRAL):
            self._attention = min(1.0, self._attention + 0.05)
        elif emotion in (EmotionType.TIRED, EmotionType.FRUSTRATED):
            self._attention = max(0.1, self._attention - 0.1)
        else:
            self._attention = max(0.1, self._attention - self.ATTENTION_DECAY)
        if blink:
            self._attention = max(0.1, self._attention - 0.01)
        return self._attention


class HeadPoseAnalyzer:
    """
    头部姿态分析器：连续俯仰/偏航信号 → 离散"点头/摇头"。

    点头/摇头是自然界最普遍的确认/否认手势，因此这里用滑动窗口
    的极值检测判定，比单帧阈值稳健得多。
    """

    NOD_THRESHOLD_DEG = 9.0
    SHAKE_THRESHOLD_DEG = 12.0
    MIN_SAMPLES = 4

    def __init__(self, window: int = 12) -> None:
        self._pitch: List[float] = []
        self._yaw: List[float] = []
        self._window = window

    def update(
        self,
        pitch: float,
        yaw: float,
        roll: float = 0.0,
        *,
        confidence: float = 0.95,
        nod: bool = False,
        shake: bool = False,
    ) -> HeadPoseSignal:
        """
        更新姿态并返回离散手势判定。

        :param nod: 显式指定点头。连续跟踪需要窗口内出现足够的俯仰摆幅；
            而用户刻意的一次点头只有单帧——这类确定性输入应直接告知，
            不该被窗口规则否决。
        :param shake: 同理，用于显式的摇头。
        """
        self._pitch.append(pitch)
        self._yaw.append(yaw)
        if len(self._pitch) > self._window:
            self._pitch = self._pitch[-self._window :]
            self._yaw = self._yaw[-self._window :]

        detected_nod, detected_shake = self._detect()
        return HeadPoseSignal(
            pitch=pitch,
            yaw=yaw,
            roll=roll,
            nod=nod or detected_nod,
            shake=shake or detected_shake,
            confidence=confidence,
        )

    def _detect(self) -> tuple[bool, bool]:
        if len(self._pitch) < self.MIN_SAMPLES:
            return False, False
        nod = (max(self._pitch) - min(self._pitch)) >= self.NOD_THRESHOLD_DEG
        shake = (max(self._yaw) - min(self._yaw)) >= self.SHAKE_THRESHOLD_DEG
        return nod, shake


@dataclass
class SimulatedCamera:
    """脚本化摄像头源（离线演示/测试）。"""

    frames: List[FacialSignal] = field(default_factory=list)
    head_poses: List[HeadPoseSignal] = field(default_factory=list)

    def stream(self) -> Dict[str, List]:
        return {"facial": list(self.frames), "head_pose": list(self.head_poses)}


def _d(a: Landmark3D, b: Landmark3D) -> float:
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2) ** 0.5


def detect_emotion_sequence(frames: Sequence[FacialSignal]) -> Optional[EmotionType]:
    """
    从连续表情序列中提取主导情绪（需至少半数帧一致才算可信）。

    用于"用户情绪持续低落 → 自动放慢节奏/主动确认"的策略。
    """
    if not frames:
        return None
    counts: Dict[EmotionType, int] = {}
    for frame in frames:
        if frame.confidence < 0.4:
            continue
        counts[frame.primary_emotion] = counts.get(frame.primary_emotion, 0) + 1
    if not counts:
        return None
    dominant, hits = max(counts.items(), key=lambda kv: kv[1])
    if hits * 2 < len(frames):
        return None
    return dominant