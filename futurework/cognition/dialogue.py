"""
对话状态机：多轮追问、指代消解、悬空指令挂起。

人与人的交流里最常见的摩擦不是"听不懂"，而是**指代**与**打断**：
"把这个挪过去"里的"这个"指什么？说到一半被插话怎么办？
本模块用显式状态机处理这两类问题，保证长流程交互不会丢上下文。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from futurework.types import (
    ExecutionStatus,
    GroundingTarget,
    Intent,
    IntentCategory,
)


class DialogueState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    AWAITING_CLARIFICATION = "awaiting_clarification"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    EXECUTING = "executing"
    SUSPENDED = "suspended"


@dataclass
class Clarification:
    """一次追问请求：问什么、为什么问、可选答案。"""

    question: str
    reason: str
    options: List[str] = field(default_factory=list)
    target_slot: str = ""
    attempts: int = 0

    def is_exhausted(self, max_attempts: int = 3) -> bool:
        return self.attempts >= max_attempts


class DialogueManager:
    """
    对话状态机。

    状态转移::

        IDLE ──意图──▶ EXECUTING ──ok──▶ IDLE
          │                └─need confirm─▶ AWAITING_CONFIRMATION
          ├─不清晰──▶ AWAITING_CLARIFICATION ──补全──▶ EXECUTING
          └─被打断──▶ SUSPENDED ──恢复──▶ AWAITING_CONFIRMATION

    每次追问最多 ``MAX_CLARIFY_ATTEMPTS`` 次，超过则给出兜底方案，
    绝不让用户陷入"反复被问"的死循环。
    """

    MAX_CLARIFY_ATTEMPTS = 3
    PENDING_TTL_SECONDS = 120.0

    def __init__(self, max_clarify_attempts: int = MAX_CLARIFY_ATTEMPTS) -> None:
        self.state = DialogueState.IDLE
        self.max_clarify_attempts = max_clarify_attempts
        self.pending: Optional[Intent] = None
        self.clarification: Optional[Clarification] = None
        self.history: List[Tuple[float, str, str]] = []
        self.reference_table: Dict[str, GroundingTarget] = {}
        self._counter = 0

    # ------------------------------------------------------------------
    def start_turn(self) -> DialogueState:
        self.state = DialogueState.LISTENING
        return self.state

    def register(self, text: str) -> None:
        self._counter += 1
        self.history.append((time.time(), "user", text))
        self.history = self.history[-200:]

    def log_system(self, text: str) -> None:
        self.history.append((time.time(), "system", text))

    # ------------------------------------------------------------------
    def accept(self, intent: Intent) -> DialogueState:
        """
        接收解析后的意图，决定下一步。

        :returns: 转移后的状态。``AWAITING_*`` 表示需要用户补充输入。
        """
        intent = self._resolve_references(intent)

        if intent.category in (IntentCategory.CONFIRMATION,):
            if self.pending is not None:
                self.state = DialogueState.EXECUTING
            return self.state

        if intent.category in (IntentCategory.REJECTION, IntentCategory.CANCEL_UNDO):
            if intent.action == "undo":
                self.pending = intent
                self.state = DialogueState.AWAITING_CONFIRMATION
                self.clarification = Clarification(
                    question="确认撤销上一步操作吗？（点头 / 说是）",
                    reason="撤销可能丢失未保存的更改",
                )
            else:
                self.pending = None
                self.clarification = None
                self.state = DialogueState.IDLE
            return self.state

        if intent.category is IntentCategory.UNKNOWN:
            self.state = DialogueState.AWAITING_CLARIFICATION
            self.clarification = Clarification(
                question=self._build_unknown_question(intent),
                reason="未能识别意图",
                attempts=self.clarification.attempts + 1 if self.clarification else 1,
            )
            return self.state

        missing = self._missing_slots(intent)
        if missing:
            self.pending = intent
            self.state = DialogueState.AWAITING_CLARIFICATION
            self.clarification = Clarification(
                question=f"请补充{missing}：",
                reason="槽位不完整，无法安全执行",
                target_slot=missing,
                attempts=self.clarification.attempts + 1 if self.clarification else 1,
            )
            return self.state

        if intent.requires_confirmation:
            self.pending = intent
            self.state = DialogueState.AWAITING_CONFIRMATION
            self.clarification = Clarification(
                question=self._build_confirmation_question(intent),
                reason="该操作风险较高，需要明确授权",
            )
            return self.state

        self.state = DialogueState.EXECUTING
        self.pending = intent
        return self.state

    # ------------------------------------------------------------------
    def complete(self, status: ExecutionStatus) -> DialogueState:
        """执行结束后的状态流转。"""
        if status in (ExecutionStatus.SUCCESS, ExecutionStatus.FAILED,
                      ExecutionStatus.ROLLED_BACK, ExecutionStatus.CANCELLED):
            self.pending = None
            self.clarification = None
            self.state = DialogueState.IDLE
        return self.state

    def suspend(self, reason: str) -> DialogueState:
        """被打断时挂起当前意图，保留以便恢复。"""
        self.log_system(f"已挂起：{reason}")
        self.state = DialogueState.SUSPENDED
        return self.state

    def resume(self) -> DialogueState:
        """恢复挂起的意图，重新请求确认。"""
        if self.pending is None:
            self.state = DialogueState.IDLE
        else:
            self.state = DialogueState.AWAITING_CONFIRMATION
        return self.state

    def clarification_exhausted(self) -> bool:
        return self.clarification is not None and self.clarification.is_exhausted(self.max_clarify_attempts)

    def fallback_plan(self) -> str:
        """追问超限后的兜底话术：给出可选项而非继续追问。"""
        if self.clarification is None:
            return "我需要重新听一遍，请再说一次。"
        opts = "、".join(self.clarification.options) if self.clarification.options else "继续、取消"
        return f"没能确定你的意思。你可以说：{opts}；或者说「取消」结束本次操作。"

    # ------------------------------------------------------------------
    def _resolve_references(self, intent: Intent) -> Intent:
        """指代消解：把"这个/那个/it/that"替换为已登记的具体目标。"""
        params = dict(intent.parameters)
        changed = False
        for key, value in list(params.items()):
            if isinstance(value, str) and _is_deictic(value):
                resolved = self._lookup_reference(value)
                if resolved is not None:
                    params[key] = resolved.target_id
                    intent.grounded_target = resolved
                    changed = True
        if changed:
            intent.parameters = params
            intent.natural_explanation += "；已用指代消解补全目标"
        return intent

    def remember_target(self, target: GroundingTarget) -> None:
        """登记一次成功交互的目标，供后续指代使用。"""
        self.reference_table[target.target_id] = target
        if len(self.reference_table) > 32:
            oldest = next(iter(self.reference_table))
            self.reference_table.pop(oldest, None)

    def _lookup_reference(self, phrase: str) -> Optional[GroundingTarget]:
        cleaned = _normalize_deictic(phrase)
        if cleaned in self.reference_table:
            return self.reference_table[cleaned]
        for key, target in reversed(list(self.reference_table.items())):
            if cleaned and cleaned in target.label.lower():
                return target
        # 纯指代词（"这个""它"）且当前只有一个候选目标时直接采用——
        # 人对着唯一的文件说"打开这个"就是在指它，不该为此多问一轮。
        if not cleaned:
            candidates = list(self.reference_table.values())
            if len(candidates) == 1:
                return candidates[0]
        return None

    def _missing_slots(self, intent: Intent) -> str:
        required = _REQUIRED_SLOTS.get(intent.action)
        if not required:
            return ""
        params = intent.parameters or {}
        if required in params and params[required]:
            return ""
        if intent.grounded_target is not None and required in ("path", "url", "name", "query"):
            params[required] = intent.grounded_target.target_id
            intent.parameters = params
            return ""
        return required

    def _build_confirmation_question(self, intent: Intent) -> str:
        label = intent.parameters.get("path") or intent.parameters.get("name") or intent.parameters.get("command") or "该操作"
        return f"即将执行：{intent.natural_explanation.split('；')[0]}（{label}）。确认吗？点头 / 说是 / 捏合手势即可。"

    def _build_unknown_question(self, intent: Intent) -> str:
        if intent.grounded_target is not None:
            label = intent.grounded_target.label or intent.grounded_target.target_id
            # 已经看到用户指哪儿了，就该问他"对它做什么"，
            # 而不是假装没看见、重新问一遍"你要干什么"。
            return (
                f"我看到你指向了「{label}」，但不确定要我对它做什么？"
                "可以说：把它移动到某个目录、复制一份、读取它、或者撤销。"
            )
        return "我没太明白，可以换个说法吗？比如：打开某个应用、在终端运行命令、搜索某个网页。"

    def pending_intent(self) -> Optional[Intent]:
        return self.pending


_REQUIRED_SLOTS: Dict[str, str] = {
    "focus_window": "name",
    "launch_app": "app",
    "delete_file": "path",
    "read_file": "path",
    "write_file": "path",
    "append_file": "path",
    "search_in_file": "path",
    "open_url": "url",
    "search_web": "query",
    "run_command": "command",
}


_DEICTIC_WORDS = {
    "这个", "那个", "这些", "那些", "它", "他", "她",
    "this", "that", "it", "these", "those",
}


def _is_deictic(value: str) -> bool:
    v = value.strip().lower()
    return v in _DEICTIC_WORDS or any(v.startswith(w) for w in ("这个", "那个"))


def _normalize_deictic(value: str) -> str:
    v = value.strip().lower()
    for word in ("这个", "那个"):
        v = v.replace(word, "").replace("窗口", "window")
    return v.strip()