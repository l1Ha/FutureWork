"""
撤销栈：让每一次写操作都可回退。

"撤销"在人际交互里是基本礼貌——对方改错了，你可以说"撤销"。
系统在自然交互中同样需要这个能力，否则用户不敢让它放手去做。

两种补偿策略：
* ``COMPENSATE`` —— 执行逆操作（文件恢复快照、git reset）；
* ``NOTIFY``    —— 无法补偿时如实告知（删除文件），绝不用"假装成功"糊弄。
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from futurework.types import ExecutionResult, ExecutionStatus, ToolCommand


class Compensation(str, Enum):
    COMPENSATE = "compensate"  # 可自动回退
    NOTIFY = "notify"          # 无法回退，只能告知
    NONE = "none"              # 只读操作，无需回退


@dataclass
class UndoEntry:
    """一条可撤销记录。"""

    command_id: str
    tool_name: str
    action: str
    args: Dict[str, Any]
    compensation: Compensation
    undo_fn: Optional[Callable[[], bool]] = None
    snapshot: Optional[Any] = None
    created_at: float = field(default_factory=time.time)
    undone: bool = False
    note: str = ""

    def describe(self) -> str:
        if self.compensation is Compensation.COMPENSATE:
            suffix = "（可撤销）"
        elif self.compensation is Compensation.NOTIFY:
            suffix = "（不可撤销）"
        else:
            suffix = ""
        detail = self.args.get("path") or self.args.get("name") or self.args.get("command") or ""
        return f"{self.tool_name}.{self.action} {detail}".strip() + suffix


class UndoManager:
    """
    撤销管理器：记录 → 回退（支持多级） → 查看历史。

    容量有上限（默认 50），超出后丢弃最旧记录——刻意不做无限栈，
    免得长会话内存无界增长。
    """

    def __init__(self, capacity: int = 50) -> None:
        self._entries: List[UndoEntry] = []
        self.capacity = capacity

    # ------------------------------------------------------------------
    def record(
        self,
        command: ToolCommand,
        *,
        undo_fn: Optional[Callable[[], bool]] = None,
        snapshot: Any = None,
        compensation: Compensation = Compensation.NONE,
        note: str = "",
    ) -> Optional[UndoEntry]:
        """
        记账。``Compensation.NONE`` 的只读操作不入栈——

        人的直觉是"撤销刚才那次改动"，不是"撤销刚才那次查看"。
        把只读操作也塞进栈里，会让撤销指向错误的那一步。
        """
        if compensation is Compensation.NONE and undo_fn is None:
            return None

        entry = UndoEntry(
            command_id=command.command_id,
            tool_name=command.tool_name,
            action=command.action,
            args=dict(command.args or {}),
            compensation=compensation,
            undo_fn=undo_fn,
            snapshot=snapshot,
            note=note,
        )
        self._entries.append(entry)
        if len(self._entries) > self.capacity:
            self._entries = self._entries[-self.capacity :]
        return entry

    def record_result(self, command: ToolCommand, result: ExecutionResult, *, undo_fn: Optional[Callable[[], bool]] = None) -> Optional[UndoEntry]:
        """执行后记账：只有成功且可撤销的操作才入栈。"""
        if result.status is not ExecutionStatus.SUCCESS:
            return None
        if undo_fn is None:
            return None
        return self.record(command, undo_fn=undo_fn, compensation=Compensation.COMPENSATE)

    # ------------------------------------------------------------------
    def history(self, limit: int = 10) -> List[str]:
        return [entry.describe() for entry in self._entries[-limit:][::-1]]

    def last(self) -> Optional[UndoEntry]:
        """最近一条尚未回退的记录（含不可补偿的条目，供查看用）。"""
        for entry in reversed(self._entries):
            if not entry.undone:
                return entry
        return None

    def last_undoable(self) -> Optional[UndoEntry]:
        """最近一条真正可回退的记录（跳过不可补偿的）。"""
        for entry in reversed(self._entries):
            if not entry.undone and entry.undo_fn is not None:
                return entry
        return None

    def undo_last(self) -> Dict[str, Any]:
        """回退最近一条可撤销记录。"""
        entry = self.last_undoable()
        if entry is None:
            recent = self.last()
            if recent is not None and recent.compensation is Compensation.NOTIFY:
                return {"ok": False, "message": f"「{recent.describe()}」不可撤销：{recent.note or '该操作没有逆操作'}"}
            return {"ok": False, "message": "没有可撤销的操作"}
        if entry.compensation is Compensation.NOTIFY:
            return {
                "ok": False,
                "message": f"「{entry.describe()}」不可撤销：{entry.note or '该操作没有逆操作'}",
            }
        if entry.undo_fn is None:
            return {"ok": False, "message": f"「{entry.describe()}」没有登记撤销方法"}
        try:
            done = entry.undo_fn()
        except Exception as exc:
            return {"ok": False, "message": f"撤销失败：{type(exc).__name__}: {exc}"}
        if done:
            entry.undone = True
            return {"ok": True, "message": f"已撤销：{entry.describe()}", "command_id": entry.command_id}
        return {"ok": False, "message": f"撤销未成功：{entry.describe()}"}

    def undo_to(self, command_id: str) -> Dict[str, Any]:
        """
        回退到指定命令之前的状态（多级撤销）。

        按逆序回退 ``command_id`` 之后（含）的所有可撤销记录。
        """
        target_index = next(
            (i for i, e in enumerate(self._entries) if e.command_id == command_id),
            None,
        )
        if target_index is None:
            return {"ok": False, "message": f"找不到命令 {command_id}"}

        reverted: List[str] = []
        failures: List[str] = []
        for entry in reversed(self._entries[target_index:]):
            if entry.undone or entry.undo_fn is None:
                continue
            try:
                if entry.undo_fn():
                    entry.undone = True
                    reverted.append(entry.describe())
                else:
                    failures.append(entry.describe())
            except Exception as exc:
                failures.append(f"{entry.describe()} ({type(exc).__name__})")

        if failures:
            return {
                "ok": False,
                "message": f"部分回退失败：{', '.join(failures)}",
                "reverted": reverted,
            }
        return {"ok": True, "message": f"已回退 {len(reverted)} 个操作", "reverted": reverted}

    def clear(self) -> None:
        self._entries.clear()

    def snapshot(self) -> List[Dict[str, Any]]:
        return [
            {
                "command_id": e.command_id,
                "description": e.describe(),
                "compensation": e.compensation.value,
                "undone": e.undone,
            }
            for e in self._entries
        ]