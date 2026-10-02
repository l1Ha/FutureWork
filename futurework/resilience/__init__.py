"""
韧性层：让系统在异常情况下依然可用、且始终安全。

五个组件，各司其职：
* ``errors``      —— 统一异常分类，让上层知道"该重试还是该放弃还是该改参数"；
* ``retry``       —— 指数退避重试，仅对瞬态故障生效；
* ``circuit``     —— 熔断器，防止在坏掉的工具上空转；
* ``undo``        —— 撤销栈，让任何操作可回退；
* ``safety``      —— 安全闸门，按风险等级决定自动执行还是征求确认。
"""

from __future__ import annotations

from futurework.resilience.errors import (
    ErrorCategory,
    FutureWorkError,
    classify_exception,
    classify_result,
    human_readable,
)
from futurework.resilience.retry import RetryPolicy, retry_async, with_retry
from futurework.resilience.circuit import CircuitBreaker, CircuitOpenError
from futurework.resilience.undo import UndoManager, UndoEntry
from futurework.resilience.safety import SafetyGate, PermissionDecision, PermissionProfile

__all__ = [
    "ErrorCategory",
    "FutureWorkError",
    "classify_exception",
    "classify_result",
    "human_readable",
    "RetryPolicy",
    "retry_async",
    "with_retry",
    "CircuitBreaker",
    "CircuitOpenError",
    "UndoManager",
    "UndoEntry",
    "SafetyGate",
    "PermissionDecision",
    "PermissionProfile",
]