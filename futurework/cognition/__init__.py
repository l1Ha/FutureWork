"""认知层：把融合后的多模态信号理解为可执行意图，并管理多轮对话状态。"""

from __future__ import annotations

from futurework.cognition.intent import (
    IntentParser,
    IntentRule,
    get_default_rules,
)
from futurework.cognition.dialogue import DialogueState, DialogueManager, Clarification

__all__ = [
    "IntentParser",
    "IntentRule",
    "get_default_rules",
    "DialogueState",
    "DialogueManager",
    "Clarification",
]