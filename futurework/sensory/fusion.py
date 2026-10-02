"""
多模态融合层：把同时发生的多个信号聚合成一个可信意图。

这是"人机像人与人一样交流"的技术核心。人和人交流时，从不要求对方
一次只说或只做一个动作——人会边说"把这个挪过去"、边伸手指屏幕、
边皱眉表达疑虑。融合引擎必须复刻这种"并发信号 → 单一意图"的能力，
并处理三类真实冲突：

1. **同意**   —— 语音"是" + 点头 → 高置信确认；
2. **拒绝**   —— 语音"好" + 摇头 → 冲突，置信度下调并追问；
3. **注意力** —— 视线离开屏幕 + 情绪低 → 暂停执行，避免打扰。
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple

from futurework.types import (
    EmotionType,
    FacialSignal,
    GazeSignal,
    GestureType,
    GroundingTarget,
    HandGestureSignal,
    HeadPoseSignal,
    ModalityType,
    MultimodalFrame,
    SpeechSignal,
)

# 各模态的基础权重：语音是显性意图主通道，视线/手势提供隐式锚点
DEFAULT_WEIGHTS: Dict[ModalityType, float] = {
    ModalityType.SPEECH: 0.45,
    ModalityType.GESTURE: 0.2,
    ModalityType.FACIAL: 0.08,
    ModalityType.GAZE: 0.15,
    ModalityType.HEAD_POSE: 0.12,
}


@dataclass
class FusedSignal:
    """融合产物：统一意图线索包，供认知层消费。"""

    confidence: float
    modalities_used: List[ModalityType]
    agreeing_modalities: List[ModalityType] = field(default_factory=list)
    conflicting_modalities: List[ModalityType] = field(default_factory=list)
    target: Optional[GroundingTarget] = None
    affective_state: EmotionType = EmotionType.NEUTRAL
    attention_score: float = 1.0
    requires_confirmation: bool = False
    ambiguity_score: float = 0.0
    rationale: List[str] = field(default_factory=list)

    def explain(self) -> str:
        used = "+".join(m.value for m in self.modalities_used)
        base = f"置信度 {self.confidence:.2f}｜模态 [{used}]"
        if self.agreeing_modalities:
            base += f"｜协同 {'/'.join(m.value for m in self.agreeing_modalities)}"
        if self.conflicting_modalities:
            base += f"｜冲突 {'/'.join(m.value for m in self.conflicting_modalities)}"
        if self.affective_state is not EmotionType.NEUTRAL:
            base += f"｜情绪 {self.affective_state.value}"
        if self.target:
            base += f"｜锚点 {self.target.label or self.target.target_id}"
        return base


@dataclass
class WorkingMemory:
    """
    工作记忆：短时缓冲 + 显著度评分 + 衰减。

    刻意复现人类工作记忆的三个特性：
    * **7±2 组块容量**：超量时按显著度淘汰，而非静默丢弃；
    * **时间衰减**：久远的意图自动淡出，但未完成的显著项保留；
    * **情绪增益**：负面情绪（困惑/挫败）提升显著性，防止忘记澄清。
    """

    CAPACITY = 7
    RETENTION_SECONDS = 45.0

    def __init__(self, capacity: int = CAPACITY, retention: float = RETENTION_SECONDS) -> None:
        self._items: Deque[Tuple[float, float, str, float]] = deque(maxlen=capacity * 3)
        self.capacity = capacity
        self.retention = retention

    def remember(self, key: str, salience: float) -> None:
        now = time.time()
        boost = 1.25 if salience < 0.4 else 1.0  # 低置信项更易被记住→触发澄清
        self._items.append((now, max(0.0, min(1.5, salience * boost)), key, salience))
        self._prune(now)

    def recall(self, key_prefix: str = "", limit: int = CAPACITY) -> List[str]:
        now = time.time()
        self._prune(now)
        scored = [
            (s, k)
            for ts, s, k, raw in self._items
            if k.startswith(key_prefix)
        ]
        scored.sort(key=lambda kv: kv[0], reverse=True)
        return [k for _, k in scored[:limit]]

    def contains(self, key: str) -> bool:
        return any(k == key for _, _, k, _ in self._items)

    def clear(self) -> None:
        self._items.clear()

    def _prune(self, now: float) -> None:
        while self._items and now - self._items[0][0] > self.retention:
            self._items.popleft()


class MultimodalFusionEngine:
    """
    多模态融合引擎：置信度加权 + 冲突仲裁 + 锚点解析。

    使用方式::

        engine = MultimodalFusionEngine()
        frame = engine.assemble(
            speech=SpeechSignal(transcript="把这个挪到那里"),
            gestures=[...], gaze=..., facial=..., head_pose=...
        )
        fused = engine.fuse(frame)
    """

    def __init__(
        self,
        weights: Optional[Dict[ModalityType, float]] = None,
        *,
        conflict_penalty: float = 0.25,
        attention_gate: float = 0.35,
        agreement_bonus: float = 0.12,
    ) -> None:
        self.weights = dict(weights or DEFAULT_WEIGHTS)
        self.conflict_penalty = conflict_penalty
        self.attention_gate = attention_gate
        self.agreement_bonus = agreement_bonus
        self.memory = WorkingMemory()
        self._frames: Deque[MultimodalFrame] = deque(maxlen=64)
        self._alerted_low_attention = False

    # ------------------------------------------------------------------
    def assemble(
        self,
        *,
        speech: Optional[SpeechSignal] = None,
        gestures: Optional[Sequence[HandGestureSignal]] = None,
        facial: Optional[FacialSignal] = None,
        head_pose: Optional[HeadPoseSignal] = None,
        gaze: Optional[GazeSignal] = None,
        environment_noise_level: float = 0.1,
    ) -> MultimodalFrame:
        """把各模态的"当下"打包成一个时间对齐的多模态帧。"""
        frame = MultimodalFrame(
            speech=speech,
            gestures=list(gestures or []),
            facial=facial,
            head_pose=head_pose,
            gaze=gaze,
            environment_noise_level=environment_noise_level,
            camera_active=facial is not None or head_pose is not None,
            microphone_active=speech is not None,
        )
        self._frames.append(frame)
        return frame

    def recent_frames(self, window_seconds: float = 3.0) -> List[MultimodalFrame]:
        now = time.time()
        return [f for f in self._frames if now - f.timestamp <= window_seconds]

    # ------------------------------------------------------------------
    def fuse(self, frame: MultimodalFrame) -> FusedSignal:
        """
        融合单帧 → 统一意图线索。

        仲裁规则（按优先级）：
        1. **安全优先**：手掌/握拳等停止手势无条件覆盖语音；
        2. **显性优先**：语音为最终意图的骨架，其余模态只做增强与锚定；
        3. **冲突降权**：点头 vs 摇头、肯定词 vs 否定词同时出现 → 降权并追问。
        """
        used: List[ModalityType] = []
        agreeing: List[ModalityType] = []
        conflicting: List[ModalityType] = []
        rationale: List[str] = []
        weighted_sum = 0.0
        weight_total = 0.0

        # ---- 语音：意图骨架 -------------------------------------------
        speech = frame.speech
        speech_polarity = _polarity_of(speech.transcript) if speech else None
        if speech and speech.is_final:
            used.append(ModalityType.SPEECH)
            eff_conf = speech.confidence * (1.0 - 0.3 * frame.environment_noise_level)
            weighted_sum += self.weights[ModalityType.SPEECH] * eff_conf
            weight_total += self.weights[ModalityType.SPEECH]
            if speech.energy > 0.9:
                rationale.append("语音语气强调，置信度上调")

        # ---- 头部姿态：点头/摇头与语音极性的一致性 -------------------
        pose = frame.head_pose
        if pose:
            used.append(ModalityType.HEAD_POSE)
            weighted_sum += self.weights[ModalityType.HEAD_POSE] * pose.confidence
            weight_total += self.weights[ModalityType.HEAD_POSE]
            if pose.nod and pose.shake:
                # 点头与摇头同时出现 = 用户自己也在犹豫。
                # 这必须优先于任何语义匹配被检出——否则"点头"分支会先命中，
                # 把明显的动摇当成赞同，系统就会在人还没拿定时动手。
                conflicting.append(ModalityType.HEAD_POSE)
                rationale.append("点头与摇头同时出现 → 你似乎还在犹豫，我先不动手")
            elif pose.nod and speech_polarity == "affirm":
                agreeing.append(ModalityType.HEAD_POSE)
                rationale.append("点头与肯定语义一致 → 确认")
            elif pose.nod and speech_polarity == "deny":
                conflicting.append(ModalityType.HEAD_POSE)
                rationale.append("点头与否定语义冲突 → 需要澄清")
            elif pose.shake and speech_polarity == "deny":
                agreeing.append(ModalityType.HEAD_POSE)
                rationale.append("摇头与否定语义一致 → 取消")
            elif pose.shake and speech_polarity == "affirm":
                conflicting.append(ModalityType.HEAD_POSE)
                rationale.append("摇头与肯定语义冲突 → 需要澄清")

        # ---- 手势：指令覆盖与修饰 -------------------------------------
        gestures = frame.gestures
        dominant = _dominant_gesture(gestures)
        if dominant:
            used.append(ModalityType.GESTURE)
            weighted_sum += self.weights[ModalityType.GESTURE] * dominant.confidence
            weight_total += self.weights[ModalityType.GESTURE]
            if dominant.gesture in (GestureType.PALM_STOP, GestureType.OPEN_PALM):
                rationale.append("张开手掌 → 安全停止信号，优先级最高")
            elif dominant.gesture in (GestureType.PINCH, GestureType.THUMBS_UP):
                if speech_polarity == "affirm":
                    agreeing.append(ModalityType.GESTURE)
                    rationale.append("捏合/点赞与语音肯定一致")
            elif dominant.gesture in (GestureType.THUMBS_DOWN, GestureType.FIST):
                if speech_polarity == "deny":
                    agreeing.append(ModalityType.GESTURE)
                    rationale.append("摇头/握拳与语音否定一致")
            elif dominant.gesture == GestureType.POINT:
                rationale.append("指向手势提供空间锚点")

        # ---- 视线：隐式目标 -------------------------------------------
        gaze = frame.gaze
        if gaze:
            used.append(ModalityType.GAZE)
            weighted_sum += self.weights[ModalityType.GAZE] * gaze.confidence
            weight_total += self.weights[ModalityType.GAZE]
            if gaze.fixation_duration_ms > 450:
                rationale.append("视线停留 → 该区域是注意焦点")

        # ---- 面部情绪：注意力与情绪负担 --------------------------------
        facial = frame.facial
        affective = EmotionType.NEUTRAL
        attention = 1.0
        if facial:
            used.append(ModalityType.FACIAL)
            weighted_sum += self.weights[ModalityType.FACIAL] * facial.confidence
            weight_total += self.weights[ModalityType.FACIAL]
            affective = facial.primary_emotion
            attention = facial.attention_score
            if affective in (EmotionType.CONFUSED, EmotionType.FRUSTRATED):
                rationale.append(f"检测到 {affective.value} → 提高澄清必要性")
            if affective is EmotionType.SATISFIED and dominant and dominant.gesture in (
                GestureType.PINCH,
                GestureType.THUMBS_UP,
            ):
                agreeing.append(ModalityType.FACIAL)

        # ---- 综合置信度 ------------------------------------------------
        confidence = weighted_sum / weight_total if weight_total > 0 else 0.0

        # 跨模态相互印证应提升置信度。单纯做加权平均是不够的：新增模态会
        # 稀释原模态的权重，而"点头 + 说是"恰恰是最强的证据组合，必须加成。
        if agreeing:
            confidence = min(1.0, confidence * (1.0 + self.agreement_bonus * len(agreeing)))

        if conflicting:
            confidence = max(0.05, confidence - self.conflict_penalty * len(conflicting))
        if frame.environment_noise_level > 0.5 and speech:
            confidence *= 1.0 - 0.4 * frame.environment_noise_level
            rationale.append("环境噪声高 → 语音置信度额外衰减")

        ambiguity = 0.0
        if conflicting:
            ambiguity += 0.45
        if speech is None and dominant is None:
            ambiguity += 0.3
        if dominant is not None and dominant.confidence < 0.6:
            ambiguity += 0.15
        if affective in (EmotionType.CONFUSED, EmotionType.SURPRISED):
            ambiguity += 0.2

        target = self._resolve_target(frame, dominant)

        # ---- 注意力闸门：用户没在关注时不贸然执行 ------------------------
        requires_confirmation = bool(conflicting) or ambiguity >= 0.4 or confidence < 0.6
        if attention < self.attention_gate and not self._alerted_low_attention:
            requires_confirmation = True
            rationale.append(f"注意力偏低 ({attention:.2f}) → 先询问而非直接执行")

        fused = FusedSignal(
            confidence=round(min(1.0, confidence), 4),
            modalities_used=used,
            agreeing_modalities=agreeing,
            conflicting_modalities=conflicting,
            target=target,
            affective_state=affective,
            attention_score=round(attention, 3),
            requires_confirmation=requires_confirmation,
            ambiguity_score=round(min(1.0, ambiguity), 3),
            rationale=rationale,
        )
        self.memory.remember(_memory_key(frame, fused), fused.confidence)
        return fused

    # ------------------------------------------------------------------
    def _resolve_target(
        self,
        frame: MultimodalFrame,
        dominant: Optional[HandGestureSignal],
    ) -> Optional[GroundingTarget]:
        """手势指向 + 视线焦点 → 统一锚点。"""
        from futurework.sensory.gaze import head_direction_to_screen

        hand_pos = dominant.position_3d if dominant and dominant.gesture in (
            GestureType.POINT,
            GestureType.PINCH,
        ) else None
        gaze_pos = None
        if frame.gaze:
            gaze_pos = (frame.gaze.screen_x, frame.gaze.screen_y)
            if frame.gaze.target_element_id:
                box = 0.04
                return GroundingTarget(
                    target_type="element",
                    target_id=frame.gaze.target_element_id,
                    label=frame.gaze.target_window_title or frame.gaze.target_element_id,
                    bounding_box=(
                        frame.gaze.screen_x - box / 2,
                        frame.gaze.screen_y - box / 2,
                        box,
                        box,
                    ),
                    metadata={"from": "gaze_dwell"},
                )

        if hand_pos is None and frame.head_pose is not None:
            # 降级方案：仅头部朝向也提供粗粒度锚点
            hx, hy = head_direction_to_screen(frame.head_pose.yaw, frame.head_pose.pitch)
            gaze_pos = (hx, hy)

        if hand_pos is None:
            return None

        hx, hy, hz = hand_pos
        if gaze_pos:
            gx, gy = gaze_pos
            x = 0.6 * gx + 0.4 * hx
            y = 0.6 * gy + 0.4 * hy
            source = "gesture+gaze"
        else:
            x, y, source = hx, hy, "gesture"
        box = max(0.02, min(0.25, 0.05 + 0.08 * (1.0 - hz)))
        return GroundingTarget(
            target_type="coordinate",
            target_id=f"fused_{int(x * 1000)}_{int(y * 1000)}",
            label="融合锚点",
            bounding_box=(max(0.0, min(1.0, x - box / 2)), max(0.0, min(1.0, y - box / 2)), box, box),
            metadata={"source": source},
        )


_AFFIRM_WORDS = (
    "是", "对", "好", "可以", "行", "嗯", "确认", "确定", "没错", "同意", "yes", "ok", "okay", "confirm", "sure", "do it",
)
_DENY_WORDS = (
    "不", "别", "停", "取消", "撤销", "不要", "算了", "错", "no", "stop", "cancel", "undo", "don't", "never",
)


def _polarity_of(text: str) -> Optional[str]:
    """判定语义极性（肯定/否定），用于点头摇头仲裁。"""
    if not text:
        return None
    lowered = text.lower()
    affirm = any(word in lowered for word in _AFFIRM_WORDS)
    deny = any(word in lowered for word in _DENY_WORDS)
    if affirm and not deny:
        return "affirm"
    if deny and not affirm:
        return "deny"
    if affirm and deny:
        # "不要" 里含"不"，"不"也命中否定词；此处的双重命中通常意味着
        # 转折句（如"不好"），保守判定为歧义。
        return None
    return None


def _dominant_gesture(gestures: Sequence[HandGestureSignal]) -> Optional[HandGestureSignal]:
    """从多个同时出现的手势中选出最具指令性的一个。"""
    priority = {
        GestureType.PALM_STOP: 10,
        GestureType.OPEN_PALM: 9,
        GestureType.FIST: 8,
        GestureType.THUMBS_DOWN: 8,
        GestureType.THUMBS_UP: 7,
        GestureType.PINCH: 6,
        GestureType.POINT: 5,
        GestureType.SWIPE_LEFT: 3,
        GestureType.SWIPE_RIGHT: 3,
        GestureType.SWIPE_UP: 3,
        GestureType.SWIPE_DOWN: 3,
        GestureType.WAVE: 2,
        GestureType.IDLE: 0,
    }
    candidates = [g for g in gestures if g.gesture is not GestureType.IDLE]
    if not candidates:
        return None
    return max(candidates, key=lambda g: (priority.get(g.gesture, 1), g.confidence))


def _memory_key(frame: MultimodalFrame, fused: FusedSignal) -> str:
    if frame.speech and frame.speech.transcript:
        return f"speech:{frame.speech.transcript[:24]}"
    if fused.target:
        return f"target:{fused.target.target_id}"
    return f"frame:{frame.frame_id}"