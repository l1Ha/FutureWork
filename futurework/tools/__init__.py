"""
工具适配层：任何生产力工具的接入点。

设计原则：**能力声明式适配**。适配器不"实现一堆方法"，而是声明
自己支持哪些 ``(action, schema)``，注册中心据此路由与校验；
新增工具 = 新增一个类，无需改动核心代码。

内置适配器覆盖六类生产力场景：
``system``（窗口/应用）、``filesystem``（文件）、``terminal``（终端/代码）、
``editor``（代码编辑器）、``browser``（网页）、``vcs``（版本控制），
以及面向未来的 ``mcp``（Model Context Protocol，任意外部工具接入）。
"""

from __future__ import annotations

from futurework.tools.base import ToolAdapter, Capability, CapabilityRegistry
from futurework.tools.system_adapter import SystemAdapter
from futurework.tools.filesystem_adapter import FileSystemAdapter
from futurework.tools.terminal_adapter import TerminalAdapter, EditorAdapter
from futurework.tools.browser_adapter import BrowserAdapter
from futurework.tools.vcs_adapter import VcsAdapter
from futurework.tools.mcp_adapter import McpAdapter
from futurework.tools.productivity_adapter import ProductivitySuiteAdapter
from futurework.tools.registry import ToolRegistry, default_registry

__all__ = [
    "ToolAdapter",
    "Capability",
    "CapabilityRegistry",
    "SystemAdapter",
    "FileSystemAdapter",
    "TerminalAdapter",
    "EditorAdapter",
    "BrowserAdapter",
    "VcsAdapter",
    "McpAdapter",
    "ProductivitySuiteAdapter",
    "ToolRegistry",
    "default_registry",
]