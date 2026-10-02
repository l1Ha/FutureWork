"""
语音感知层：接收自然口语指令，输出结构化语音信号。

要点
----
* **流式识别**：``is_final=False`` 表示识别中的临时结果，可用于实时反馈；
* **韵律特征**：pitch/energy/speaking_rate 承载"强调""急躁""耳语"等意图线索，
  认知层据此做情绪加权；
* **可替换后端**：默认使用 ``SimulatedMicrophone``（脚本回放 + 噪声注入），
  有硬件时可注入 Whisper / 云端 ASR 适配器，接口不变。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, List, Optional, Protocol

from futurework.types import SpeechSignal

# 中文/英文的常见语气词与填充词，识别时剔除以降低认知层噪声
_FILLER_WORDS = re.compile(
    r"(嗯+|呃+|啊+|那个|这个|then\b|um+|uh+|er+)",
    re.IGNORECASE,
)


@dataclass
class Prosody:
    """韵律特征：音高、能量、语速。"""

    pitch_hz: float = 180.0
    energy: float = 0.8
    speaking_rate: float = 1.0

    def to_dict(self) -> dict:
        return {
            "pitch_hz": round(self.pitch_hz, 1),
            "energy": round(self.energy, 3),
            "speaking_rate": round(self.speaking_rate, 3),
        }


class SpeechBackend(Protocol):
    """语音识别后端协议：任何 ASR 引擎实现它即可接入。"""

    def transcribe(self, audio_chunk: bytes, *, is_final: bool) -> SpeechSignal:
        ...


@dataclass
class SimulatedMicrophone:
    """
    离线语音源：按脚本回放文本，模拟真实识别的不确定性。

    ``noise_level`` 模拟嘈杂环境导致的识别错误与置信度下降，
    ``latency`` 模拟识别延迟（毫秒）。
    """

    script: List[str] = field(default_factory=list)
    noise_level: float = 0.0
    latency_ms: float = 120.0
    _cursor: int = 0

    def stream(self) -> Iterator[SpeechSignal]:
        for text in self.script:
            yield self._make_signal(text, is_final=True)

    def push(self, text: str, *, is_final: bool = True, confidence: float = 1.0) -> SpeechSignal:
        """直接注入一段话（测试与手动输入使用）。"""
        return self._make_signal(text, is_final=is_final, confidence_override=confidence)

    def _make_signal(
        self,
        text: str,
        *,
        is_final: bool,
        confidence_override: Optional[float] = None,
    ) -> SpeechSignal:
        cleaned = _FILLER_WORDS.sub("", text).strip()
        confidence = confidence_override if confidence_override is not None else 1.0
        if self.noise_level > 0:
            confidence = max(0.05, confidence * (1.0 - self.noise_level))
        prosody = _estimate_prosody(cleaned)
        return SpeechSignal(
            transcript=cleaned,
            confidence=confidence,
            is_final=is_final,
            pitch_hz=prosody.pitch_hz,
            energy=prosody.energy,
            speaking_rate=prosody.speaking_rate,
        )


class SpeechRecognizer:
    """
    语音识别封装层：清洗文本、归一化、附带韵律与置信度。

    对外只暴露 ``understand`` / ``consume``，屏蔽后端差异。
    """

    def __init__(self, backend: Optional[SpeechBackend] = None) -> None:
        self.backend = backend or SimulatedMicrophone()
        self._history: List[SpeechSignal] = []
        self._listeners: List[Callable[[SpeechSignal], None]] = []

    # ------------------------------------------------------------------
    def on_speech(self, callback: Callable[[SpeechSignal], None]) -> None:
        """注册语音回调（融合引擎、UI 提示均通过它接收）。"""
        self._listeners.append(callback)

    def submit(self, text: str, *, is_final: bool = True, confidence: float = 1.0) -> SpeechSignal:
        """提交一段文本并广播给监听者。"""
        signal = self.backend.push(text, is_final=is_final, confidence=confidence)
        self._dispatch(signal)
        return signal

    def consume(self, signal: SpeechSignal) -> SpeechSignal:
        """接收后端产出的信号，清洗后广播。"""
        cleaned = signal.model_copy(update={"transcript": _FILLER_WORDS.sub("", signal.transcript).strip()})
        self._dispatch(cleaned)
        return cleaned

    def stream(self) -> Iterator[SpeechSignal]:
        """从后端流式读取并广播。"""
        for raw in self.backend.stream():
            signal = self.consume(raw)
            yield signal

    @property
    def history(self) -> List[SpeechSignal]:
        return list(self._history)

    def last_final(self) -> Optional[SpeechSignal]:
        for signal in reversed(self._history):
            if signal.is_final:
                return signal
        return None

    # ------------------------------------------------------------------
    def _dispatch(self, signal: SpeechSignal) -> None:
        self._history.append(signal)
        for listener in self._listeners:
            try:
                listener(signal)
            except Exception:  # 单个监听器异常不得影响感知链路
                continue


_EXCLAMATION = re.compile(r"[!！]")
_QUESTION = re.compile(r"[?？]")


def _estimate_prosody(text: str) -> Prosody:
    """
    从文本启发式估算韵律特征（真实系统由声学模型给出）。

    估算规则：
    * 感叹号 → 音高与能量升高；
    * 问号 → 句尾上扬（音高小幅提高）；
    * 字符长度 → 语速（短促命令更快，长句更慢）；
    * 命令性动词密度 → 能量（祈使句更有力）。
    """
    if not text:
        return Prosody(pitch_hz=180.0, energy=0.5, speaking_rate=1.0)

    length = len(text)
    exclam = bool(_EXCLAMATION.search(text))
    question = bool(_QUESTION.search(text))
    imperative_markers = sum(
        text.count(word) for word in ("打开", "关闭", "新建", "删除", "运行", "打开", "write", "open", "delete", "run")
    )

    pitch = 180.0 + (30.0 if exclam else 0.0) + (14.0 if question else 0.0)
    energy = 0.7 + (0.2 if exclam else 0.0) + min(0.1, 0.01 * imperative_markers)
    energy -= 0.15 if question else 0.0
    energy = max(0.05, min(1.0, energy))
    speaking_rate = 1.0 + (0.25 if exclam else 0.0) - min(0.3, length / 400.0)
    speaking_rate = max(0.4, min(2.0, speaking_rate))
    return Prosody(pitch_hz=pitch, energy=energy, speaking_rate=speaking_rate)


def now() -> float:
    return time.time()