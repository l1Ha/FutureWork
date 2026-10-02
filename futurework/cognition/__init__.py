"""认知层：把融合后的多模态信号理解为可执行意图，并管理多轮对话状态。"""

from __future__ import annotations

from futurework.cognition.compound import (
    ConditionalFlow,
    Correction,
    detect_correction,
    parse_conditional,
    split_compound,
    split_respecting_quotes,
)
from futurework.cognition.dialogue import Clarification, DialogueManager, DialogueState
from futurework.cognition.intent import (
    IntentParser,
    IntentRule,
    get_default_rules,
)
from futurework.cognition.memory import MemoryEntity, PersistentMemoryStore

__all__ = [
    "IntentParser",
    "IntentRule",
    "get_default_rules",
    "DialogueState",
    "DialogueManager",
    "Clarification",
    "MemoryEntity",
    "PersistentMemoryStore",
    "ConditionalFlow",
    "Correction",
    "parse_conditional",
    "detect_correction",
    "split_compound",
    "split_respecting_quotes",
]