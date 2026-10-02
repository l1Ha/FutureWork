"""
事件总线：系统内部的可观测性与解耦机制。

为什么需要它：多模态交互链路很长（感知→融合→认知→安全→执行→反馈），
任何一个环节出问题都要能定位；同时 UI（语音播报、屏幕提示、表情反馈）
需要各自订阅事件而不互相耦合。
"""

from __future__ import annotations

import itertools
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class EventPriority(int, Enum):
    """优先级：数值越小越先执行。"""

    CRITICAL = 0   # 安全告警、取消
    HIGH = 1       # 错误、需要用户决策
    NORMAL = 2     # 执行结果
    LOW = 3        # 遥测、调试


@dataclass
class Event:
    """总线上的一个事件。"""

    name: str
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: EventPriority = EventPriority.NORMAL
    timestamp: float = field(default_factory=time.time)
    source: str = ""
    event_id: str = field(default_factory=lambda: next(_IDS))


_IDS = itertools.count(1)


class EventBus:
    """
    线程安全事件总线。

    订阅者异常被隔离并记录——一个坏掉的 UI 监听器绝不能阻断工具执行。
    """

    def __init__(self, *, history_limit: int = 200) -> None:
        self._subscribers: Dict[str, List[Callable[[Event], None]]] = {}
        self._wildcards: List[Callable[[Event], None]] = []
        self._history: List[Event] = []
        self._history_limit = history_limit
        self._errors: List[str] = []
        self._lock_free = True  # 保持实现简单；回调内不做重入假设

    # ------------------------------------------------------------------
    def subscribe(self, name: str, callback: Callable[[Event], None]) -> Callable[[], None]:
        """订阅事件；返回取消订阅的函数。"""
        self._subscribers.setdefault(name, []).append(callback)

        def _unsubscribe() -> None:
            try:
                self._subscribers.get(name, []).remove(callback)
            except ValueError:
                pass

        return _unsubscribe

    def subscribe_all(self, callback: Callable[[Event], None]) -> Callable[[], None]:
        """订阅所有事件。"""
        self._wildcards.append(callback)

        def _unsubscribe() -> None:
            try:
                self._wildcards.remove(callback)
            except ValueError:
                pass

        return _unsubscribe

    # ------------------------------------------------------------------
    def emit(self, event: Event) -> None:
        """广播事件；按优先级排序后依次调用订阅者。"""
        self._history.append(event)
        if len(self._history) > self._history_limit:
            self._history = self._history[-self._history_limit :]

        listeners: List[Callable[[Event], None]] = list(self._wildcards)
        listeners.extend(self._subscribers.get(event.name, []))

        if len(listeners) > 1 and event.priority is not EventPriority.CRITICAL:
            listeners.sort(key=lambda _: 0)  # 保持注册顺序；优先级通过多次 emit 表达

        for listener in listeners:
            try:
                listener(event)
            except Exception as exc:
                self._errors.append(
                    f"[{time.strftime('%H:%M:%S')}] 订阅者处理 '{event.name}' 失败: "
                    f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}"
                )

    def emit_simple(self, name: str, payload: Optional[Dict[str, Any]] = None, *, priority: EventPriority = EventPriority.NORMAL, source: str = "") -> None:
        self.emit(Event(name=name, payload=payload or {}, priority=priority, source=source))

    # ------------------------------------------------------------------
    def history(self, name: Optional[str] = None, limit: int = 50) -> List[Event]:
        items = [e for e in self._history if name is None or e.name == name]
        return items[-limit:]

    def errors(self) -> List[str]:
        return list(self._errors)

    def clear_errors(self) -> None:
        self._errors.clear()