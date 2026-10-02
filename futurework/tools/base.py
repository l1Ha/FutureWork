"""
适配器基类与能力声明机制。

``ToolAdapter`` 是整个对接核心唯一的契约。它刻意做得很薄——
只有能力声明与一个 ``execute``——这样任何语言实现的对象、
任何远程服务包装器，都能立即接入并被自然语言指挥。
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from futurework.types import ExecutionResult, ExecutionStatus, SafetyLevel


@dataclass(frozen=True)
class Capability:
    """
    能力声明：``action`` + 参数约束 + 安全等级。

    ``validator`` 用于在执行前拦截非法参数（如危险路径、越权命令），
    属于适配器自保机制——适配器最清楚自己能被怎样安全地调用。
    """

    action: str
    params: Dict[str, str] = field(default_factory=dict)  # name -> 简单类型名
    safety_level: SafetyLevel = SafetyLevel.READ_ONLY
    reversible: bool = False
    timeout_seconds: float = 10.0
    validator: Optional[Callable[[Dict[str, Any]], Optional[str]]] = None
    description: str = ""
    aliases: Tuple[str, ...] = ()

    def validate(self, args: Dict[str, Any]) -> Optional[str]:
        """返回 ``None`` 表示通过，否则返回错误说明。"""
        for name, expected in self.params.items():
            if name not in args or args[name] is None:
                if expected.endswith("?"):
                    continue
                return f"缺少必填参数 '{name}'（期望类型 {expected}）"
            value = args[name]
            simple = expected.rstrip("?")
            try:
                if simple == "str" and not isinstance(value, str):
                    return f"参数 '{name}' 应为字符串，实际为 {type(value).__name__}"
                if simple == "int" and not isinstance(value, int):
                    return f"参数 '{name}' 应为整数"
                if simple == "float" and not isinstance(value, (int, float)):
                    return f"参数 '{name}' 应为数值"
                if simple == "bool" and not isinstance(value, bool):
                    return f"参数 '{name}' 应为布尔值"
                if simple == "list" and not isinstance(value, (list, tuple)):
                    return f"参数 '{name}' 应为列表"
            except Exception as exc:  # 校验器自身异常不得放过
                return f"参数校验异常：{exc}"
        if self.validator is not None:
            try:
                return self.validator(args)
            except Exception as exc:
                return f"参数校验器异常：{exc}"
        return None

    def inverse(self, args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """构造逆操作参数；默认不支持撤销，由适配器覆写。"""
        return None


class CapabilityRegistry:
    """单个适配器内部的能力表。"""

    def __init__(self) -> None:
        self._caps: Dict[str, Capability] = {}

    def declare(self, capability: Capability) -> Capability:
        self._caps[capability.action] = capability
        for alias in capability.aliases:
            self._caps[alias] = capability
        return capability

    def capability(self, action: str) -> Optional[Capability]:
        return self._caps.get(action)

    def actions(self) -> List[str]:
        seen: List[str] = []
        for cap in self._caps.values():
            if cap.action not in seen:
                seen.append(cap.action)
        return sorted(seen)

    def __contains__(self, action: str) -> bool:
        return action in self._caps

    def __len__(self) -> int:
        return len(self._caps)


class ToolAdapter(ABC):
    """
    所有工具适配器的基类。

    子类只需：
    1. 在 ``__init__`` 里 ``declare`` 自己的能力；
    2. 实现 ``_dispatch(action, args)``。
    """

    name: str = "base"
    platform_support: Tuple[str, ...] = ("linux", "macos", "windows")
    description: str = ""

    def __init__(self) -> None:
        self.capabilities = CapabilityRegistry()
        self._call_count = 0
        self._fail_count = 0
        self.setup()

    # ------------------------------------------------------------------
    def setup(self) -> None:
        """子类在此声明能力。"""

    def supported_platforms(self) -> Tuple[str, ...]:
        return self.platform_support

    def is_available(self) -> Tuple[bool, str]:
        """返回 ``(是否可用, 原因)``；缺依赖时给出可读原因而非抛异常。"""
        return True, "ok"

    # ------------------------------------------------------------------
    def execute(self, command) -> ExecutionResult:
        """
        统一执行入口：校验 → 执行 → 计时 → 异常归一化。

        任何异常都在此被转换成 ``ExecutionResult(error=...)``，
        保证上层永远不会因适配器问题而崩溃。
        """
        started = time.perf_counter()
        cap = self.capabilities.capability(command.action)
        if cap is None:
            return ExecutionResult(
                command_id=command.command_id,
                status=ExecutionStatus.FAILED,
                error=f"适配器 {self.name} 不支持操作 '{command.action}'；可用操作：{', '.join(self.capabilities.actions())}",
                execution_time_ms=(time.perf_counter() - started) * 1000,
            )

        problem = cap.validate(command.args or {})
        if problem:
            return ExecutionResult(
                command_id=command.command_id,
                status=ExecutionStatus.FAILED,
                error=f"参数不合法：{problem}",
                execution_time_ms=(time.perf_counter() - started) * 1000,
            )

        self._call_count += 1
        try:
            data, details = self._dispatch(command.action, command.args or {})
            details = dict(details or {})
            status = details.pop("_status", ExecutionStatus.SUCCESS)
            rollback = details.pop("_rollback_available", cap.reversible)

            # 统一失败语义：适配器用 ``(None, {"error": ...})`` 报告失败时，
            # 若原样返回，上层会把"没有数据"当成"执行成功"——
            # 这类不一致必须在基类收敛，否则每个适配器都要各写一遍。
            embedded_error = details.pop("error", None)
            if embedded_error and status is ExecutionStatus.SUCCESS:
                status = ExecutionStatus.FAILED
                data = None

            return ExecutionResult(
                command_id=command.command_id,
                status=status,
                data=data,
                error=embedded_error,
                execution_time_ms=(time.perf_counter() - started) * 1000,
                rollback_available=rollback,
                details=details,
            )
        except NotImplementedError as exc:
            self._fail_count += 1
            return ExecutionResult(
                command_id=command.command_id,
                status=ExecutionStatus.FAILED,
                error=f"操作尚未在当前平台实现：{exc}",
                execution_time_ms=(time.perf_counter() - started) * 1000,
            )
        except Exception as exc:
            self._fail_count += 1
            return ExecutionResult(
                command_id=command.command_id,
                status=ExecutionStatus.FAILED,
                error=f"{type(exc).__name__}: {exc}",
                execution_time_ms=(time.perf_counter() - started) * 1000,
            )

    @abstractmethod
    def _dispatch(self, action: str, args: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """子类实现具体执行。返回 ``(data, details)``。"""

    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        return {
            "adapter": self.name,
            "actions": self.capabilities.actions(),
            "calls": self._call_count,
            "failures": self._fail_count,
        }


def safe_join_validator(
    base: str, *, allow_empty: bool = False
) -> Callable[[Dict[str, Any]], Optional[str]]:
    """
    生成一个防止路径穿越（``../``）的参数校验器。

    :param allow_empty: 允许 path 缺省。列目录这类操作说"列出目录"而不说路径
        是完全自然的，适配器会自行落到根目录——此时报错等于无谓地打断用户。
    """

    def _check(args: Dict[str, Any]) -> Optional[str]:
        import os

        target = str(args.get("path", ""))
        if not target:
            return None if allow_empty else "path 不能为空"
        resolved = os.path.abspath(os.path.join(base, target))
        root = os.path.abspath(base)
        if not (resolved == root or resolved.startswith(root + os.sep)):
            return f"路径越界：'{target}' 不在允许的根目录内"
        return None

    return _check


def deny_readonly_validator(args: Dict[str, Any]) -> Optional[str]:
    """示意校验器：拒绝一切写意图参数。"""
    if args.get("readonly", False) is False and "write" in args:
        return "只读操作不接受写入参数"
    return None