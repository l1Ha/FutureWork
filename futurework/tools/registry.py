"""
工具注册中心：意图 → 适配器的路由与生命周期管理。

路由策略是三级回退：
1. 适配器显式声明支持该 ``action`` → 直接用；
2. 别名匹配（``tool_hint``）→ 找到对应适配器；
3. 全部未命中 → 返回结构化的"无可用工具"错误，并列出全部可用能力，
   方便上层直接转成一句人话反馈给用户。

注册中心还负责**可用性巡检**：缺失依赖的适配器不会让整个系统崩溃，
只是被标记为 unavailable，路由时自动跳过。
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from futurework.tools.base import Capability, ToolAdapter
from futurework.tools.browser_adapter import BrowserAdapter
from futurework.tools.filesystem_adapter import FileSystemAdapter
from futurework.tools.productivity_adapter import ProductivitySuiteAdapter
from futurework.tools.system_adapter import SystemAdapter
from futurework.tools.terminal_adapter import EditorAdapter, TerminalAdapter
from futurework.tools.vcs_adapter import VcsAdapter
from futurework.types import SafetyLevel


class ToolRegistry:
    """适配器注册中心。"""

    def __init__(self, adapters: Optional[Sequence[ToolAdapter]] = None) -> None:
        self._adapters: List[ToolAdapter] = []
        self._action_index: Dict[str, str] = {}  # action -> adapter name
        self._aliases: Dict[str, str] = {}      # tool_hint -> adapter name
        self._unavailable: Dict[str, str] = {}  # adapter name -> 原因
        for adapter in adapters or []:
            self.register(adapter)

    # ------------------------------------------------------------------
    def register(self, adapter: ToolAdapter) -> None:
        self._adapters.append(adapter)
        for action in adapter.capabilities.actions():
            self._action_index.setdefault(action, adapter.name)

    def register_hint(self, hint: str, adapter_name: str) -> None:
        """把意图层的 ``tool_hint`` 绑定到某个适配器。"""
        self._aliases[hint] = adapter_name

    def unregister(self, name: str) -> bool:
        before = len(self._adapters)
        self._adapters = [a for a in self._adapters if a.name != name]
        self._action_index = {k: v for k, v in self._action_index.items() if v != name}
        self._aliases = {k: v for k, v in self._aliases.items() if v != name}
        return len(self._adapters) < before

    # ------------------------------------------------------------------
    def adapters(self) -> List[ToolAdapter]:
        return list(self._adapters)

    def get(self, name: str) -> Optional[ToolAdapter]:
        for adapter in self._adapters:
            if adapter.name == name:
                return adapter
        return None

    def capabilities(self) -> Dict[str, List[str]]:
        """所有适配器的能力清单（诊断与"我该说什么"提示用）。"""
        return {a.name: a.capabilities.actions() for a in self._adapters}

    def health_check(self) -> Dict[str, Dict[str, Any]]:
        """
        全量巡检：适配器是否可用、缺什么、建议怎么装。

        这是异常处理的前置防线——先知道什么不可用，
        才能在用户开口之前就给出准确反馈，而不是让操作失败一次。
        """
        report: Dict[str, Dict[str, Any]] = {}
        for adapter in self._adapters:
            try:
                available, reason = adapter.is_available()
            except Exception as exc:
                available, reason = False, f"巡检异常：{type(exc).__name__}: {exc}"
            report[adapter.name] = {
                "available": available,
                "reason": reason,
                "actions": adapter.capabilities.actions(),
                "platforms": adapter.supported_platforms(),
            }
            if not available:
                self._unavailable[adapter.name] = reason
            else:
                self._unavailable.pop(adapter.name, None)
        return report

    def unavailable(self) -> Dict[str, str]:
        return dict(self._unavailable)

    # ------------------------------------------------------------------
    def resolve(self, action: str, tool_hint: str = "") -> Tuple[Optional[ToolAdapter], Optional[Capability], str]:
        """
        路由解析。

        :returns: ``(适配器, 能力, 错误说明)``；错误说明为空串表示成功。
        """
        candidates: List[ToolAdapter] = []

        if tool_hint:
            target = self._aliases.get(tool_hint)
            if target:
                candidate = self.get(target)
                if candidate is not None:
                    candidates.append(candidate)

        for adapter in self._adapters:
            if adapter not in candidates:
                candidates.append(adapter)

        for adapter in candidates:
            capability = adapter.capabilities.capability(action)
            if capability is not None:
                if adapter.name in self._unavailable:
                    continue
                return adapter, capability, ""

        # 命中能力但适配器不可用 → 明确告知缺什么
        blocked = []
        for adapter in candidates:
            if adapter.capabilities.capability(action) is not None and adapter.name in self._unavailable:
                blocked.append(f"{adapter.name}（{self._unavailable[adapter.name]}）")

        available_actions = sorted({a for acts in self.capabilities().values() for a in acts})
        if blocked:
            return None, None, (
                f"操作 '{action}' 需要 {blocked[0]}，但该工具当前不可用。"
            )
        return None, None, (
            f"没有适配器支持操作 '{action}'。当前可用操作：{', '.join(available_actions) or '（无）'}"
        )

    def capability_for(self, action: str) -> Optional[Capability]:
        for adapter in self._adapters:
            capability = adapter.capabilities.capability(action)
            if capability is not None:
                return capability
        return None

    def suggest_safety(self, action: str) -> SafetyLevel:
        capability = self.capability_for(action)
        return capability.safety_level if capability else SafetyLevel.CRITICAL_DESTRUCTIVE

    def stats(self) -> List[Dict[str, Any]]:
        return [adapter.stats() for adapter in self._adapters]


def default_registry(
    workdir: Optional[str] = None,
    *,
    include_mcp: bool = False,
    mcp_command: Optional[List[str]] = None,
) -> ToolRegistry:
    """构建开箱即用的默认注册中心。"""
    root = os.path.abspath(workdir or os.getcwd())
    registry = ToolRegistry([
        SystemAdapter(),
        FileSystemAdapter(root),
        TerminalAdapter(root),
        EditorAdapter(root),
        BrowserAdapter(),
        VcsAdapter(root),
        ProductivitySuiteAdapter(root),
    ])
    registry.register_hint("system", "system")
    registry.register_hint("filesystem", "filesystem")
    registry.register_hint("terminal", "terminal")
    registry.register_hint("editor", "editor")
    registry.register_hint("browser", "browser")
    registry.register_hint("vcs", "vcs")
    registry.register_hint("productivity", "productivity")

    if include_mcp:
        from futurework.tools.mcp_adapter import McpAdapter

        mcp = McpAdapter(mcp_command or [])
        registry.register(mcp)
        registry.register_hint("mcp", "mcp")

    registry.health_check()
    return registry