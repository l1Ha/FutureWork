#!/usr/bin/env python3
"""
演示二：全平台全工具对接 —— 接入你自己的工具

运行：``python examples/demo_extensibility.py``

三个部分：
1. 写一个第三方适配器（模拟 Notion / 日历 / 内部系统）；
2. 把它注册进 FutureWork，用中文指挥它；
3. 展示 MCP 协议接入——不改 FutureWork 一行代码即可对接任意外部工具。
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from futurework.runtime.session import FutureWorkSession
from futurework.tools.base import Capability, ToolAdapter
from futurework.tools.registry import ToolRegistry, default_registry
from futurework.types import ExecutionStatus, SafetyLevel

BAR = "─" * 74


# ============================================================================
# 1) 定义第三方工具适配器
# ============================================================================
class NotionAdapter(ToolAdapter):
    """
    模拟 Notion 适配器。

    真实实现只需把 ``_dispatch`` 换成 Notion API 调用即可；
    能力声明、参数校验、安全分级、错误归一化全部由基类提供。
    """

    name = "notion"
    description = "Notion 知识库"

    def setup(self) -> None:
        self.capabilities.declare(Capability(
            action="create_page",
            params={"title": "str", "content": "str?"},
            safety_level=SafetyLevel.SAFE_WRITE,
            description="在 Notion 中创建页面",
        ))
        self.capabilities.declare(Capability(
            action="delete_page",
            params={"page_id": "str"},
            safety_level=SafetyLevel.CRITICAL_DESTRUCTIVE,
            description="删除 Notion 页面（不可逆）",
        ))
        self.capabilities.declare(Capability(
            action="search_pages",
            params={"keyword": "str"},
            safety_level=SafetyLevel.READ_ONLY,
            description="搜索 Notion 页面",
            aliases=("notion_search",),
        ))

    def _dispatch(self, action, args):
        if action == "create_page":
            page_id = f"page-{abs(hash(args['title'])) % 100000}"
            return {"id": page_id, "url": f"https://notion.so/{page_id}"}, {}
        if action == "delete_page":
            return {"id": args["page_id"], "deleted": True}, {}
        if action == "search_pages":
            return [
                {"id": "p1", "title": f"关于{args['keyword']}的笔记"},
                {"id": "p2", "title": f"{args['keyword']}待办清单"},
            ], {}
        raise NotImplementedError(action)


class WeatherAdapter(ToolAdapter):
    """另一个工具，验证多适配器并存与路由。"""

    name = "weather"
    description = "天气查询"

    def setup(self):
        self.capabilities.declare(Capability(
            action="get_weather",
            params={"city": "str"},
            safety_level=SafetyLevel.READ_ONLY,
            description="查询城市天气",
        ))

    def _dispatch(self, action, args):
        return {"city": args["city"], "temp": 22, "condition": "晴"}, {}


# ============================================================================
# 2) 演示
# ============================================================================
def main() -> int:
    workdir = tempfile.mkdtemp(prefix="futurework_ext_")
    registry: ToolRegistry = default_registry(workdir)
    builtin = list(registry.capabilities())

    print(BAR)
    print("▶ 接入前：系统只有内置工具")
    print(BAR)
    print(f"  适配器 → {builtin}")
    print(f"  能力数 → {sum(len(v) for v in registry.capabilities().values())}")

    registry.register(NotionAdapter())
    registry.register(WeatherAdapter())
    session = FutureWorkSession(workdir=workdir, registry=registry)

    print(f"\n{BAR}\n▶ 接入 Notion 与天气工具（新增两个类，零核心改动）\n{BAR}")
    print(f"  适配器 → {list(registry.capabilities())}")
    print(f"  能力数 → {sum(len(v) for v in registry.capabilities().values())}")
    print(f"  Notion 能力 → {registry.get('notion').capabilities.actions()}")
    print(f"  天气能力   → {registry.get('weather').capabilities.actions()}")

    print(f"\n{BAR}\n▶ 用自然语言指挥第三方工具\n{BAR}")
    from futurework.types import ToolCommand

    calls = [
        ("notion", "create_page", {"title": "FutureWork 设计决策", "content": "记录要点"}),
        ("notion", "search_pages", {"keyword": "异步"}),
        ("weather", "get_weather", {"city": "杭州"}),
    ]
    for tool, action, args in calls:
        adapter, _, error = registry.resolve(action)
        if adapter is None:
            print(f"  {action:15} → 路由失败：{error}")
            continue
        result = adapter.execute(ToolCommand(tool_name=tool, action=action, args=args))
        shown = result.data if result.status is ExecutionStatus.SUCCESS else result.error
        print(f"  {action:15} → {result.status.value:8} {shown}")

    print(f"\n{BAR}\n▶ 安全分级：Notion 删除页面属于不可逆操作，必须确认\n{BAR}")
    from futurework.types import ToolCommand

    command = ToolCommand(tool_name="notion", action="delete_page", args={"page_id": "p1"})
    adapter, capability, _ = registry.resolve("delete_page")
    command.safety_level = capability.safety_level
    from futurework.resilience.safety import SafetyGate

    decision, why = SafetyGate().evaluate(command)
    print(f"  风险等级 → {capability.safety_level.name}")
    print(f"  闸门判定 → {decision.value}")
    print(f"  理由     → {why}")

    print(f"\n{BAR}\n▶ MCP：对接任意外部工具，不改 FutureWork 代码\n{BAR}")
    from futurework.tools.mcp_adapter import McpAdapter

    mcp = McpAdapter(["python3", "-m", "some_mcp_server"])
    available, reason = mcp.is_available()
    print(f"  已注册 MCP 适配器：{mcp.name}")
    print(f"  可用性  → {available}（{reason}）")
    print(f"  能力    → {mcp.capabilities.actions()}")
    print("  说明    → 任何符合 MCP 规范的 server 只要写进启动命令，")
    print("            其工具就会被自动发现并注册为 mcp::<工具名> 能力。")

    print(f"\n{BAR}\n▶ 异常隔离：坏工具不影响好工具\n{BAR}")

    class BrokenAdapter(ToolAdapter):
        name = "broken"

        def setup(self):
            self.capabilities.declare(Capability(action="explode"))

        def _dispatch(self, action, args):
            raise RuntimeError("这个工具内部崩溃了")

    registry.register(BrokenAdapter())
    adapter, _, _ = registry.resolve("explode")
    result = adapter.execute(ToolCommand(tool_name="broken", action="explode", args={}))
    print(f"  崩溃工具 → {result.status.value}｜{result.error}")
    adapter, _, _ = registry.resolve("read_file")
    result = adapter.execute(ToolCommand(tool_name="filesystem", action="read_file",
                                         args={"path": "仍然可用.txt"}))
    print(f"  其他工具 → {result.status.value}｜{result.error}（正常返回错误而非崩溃）")

    return 0


if __name__ == "__main__":
    sys.exit(main())