"""
MCP 适配器：对接 Model Context Protocol，实现"任意工具即插即用"。

这是"全平台全工具"真正可扩展的关键。开发者只需把自己的工具按 MCP 规范
暴露（stdio JSON-RPC），注册到 FutureWork，之后就能被自然语言指挥，
无需改动 FutureWork 一行代码。

协议交互（JSON-RPC 2.0 over stdio）：
    → ``initialize`` / ``tools/list`` / ``tools/call``
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from futurework.tools.base import Capability, ToolAdapter
from futurework.types import ExecutionStatus, SafetyLevel

PROTOCOL_VERSION = "2024-11-05"


class McpAdapter(ToolAdapter):
    """
    MCP 客户端适配器。

    :param command: 启动 MCP server 的命令，如 ``["python", "-m", "my_server"]``
    :param env: 附加环境变量
    :param timeout: 单次 RPC 超时
    """

    name = "mcp"

    def __init__(
        self,
        command: Optional[List[str]] = None,
        *,
        env: Optional[Dict[str, str]] = None,
        timeout: float = 20.0,
        auto_discover: bool = True,
    ) -> None:
        self.command = command or []
        self.env = {**os.environ, **(env or {})}
        self.timeout = timeout
        self._proc: Optional[subprocess.Popen] = None
        self._rpc_id = 0
        self._lock = threading.Lock()
        self._tools_cache: List[Dict[str, Any]] = []
        self._initialized = False
        self.auto_discover = auto_discover
        super().__init__()

    def setup(self) -> None:
        self.capabilities.declare(Capability(
            action="list_mcp_tools",
            safety_level=SafetyLevel.READ_ONLY,
            description="列出 MCP server 提供的工具",
        ))
        self.capabilities.declare(Capability(
            action="call_mcp_tool",
            params={"tool": "str", "arguments": "dict?"},
            safety_level=SafetyLevel.SENSITIVE_MODIFY,
            timeout_seconds=self.timeout,
            description="调用 MCP server 上的工具",
        ))

    def is_available(self) -> Tuple[bool, str]:
        if not self.command:
            return False, "未配置 MCP server 启动命令"
        if shutil_which(self.command[0]) is None and not os.path.exists(self.command[0]):
            return False, f"找不到 MCP server 可执行文件：{self.command[0]}"
        return True, f"command={' '.join(self.command)}"

    # ------------------------------------------------------------------
    def connect(self) -> Tuple[bool, str]:
        """启动 MCP server 并完成 initialize 握手。"""
        if self._initialized:
            return True, "already connected"
        if not self.command:
            return False, "未配置 MCP server 启动命令"
        try:
            self._proc = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self.env,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            return False, f"启动 MCP server 失败：{exc}"

        ok, detail = self._rpc("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "clientInfo": {"name": "futurework", "version": "0.1.0"},
        })
        if not ok:
            self.close()
            return False, f"initialize 失败：{detail}"
        self._initialized = True
        self._rpc("notifications/initialized", {})
        if self.auto_discover:
            self._discover()
        return True, "connected"

    def close(self) -> None:
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=3)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None
        self._initialized = False

    def _discover(self) -> List[Dict[str, Any]]:
        ok, data = self._rpc("tools/list", {})
        if not ok:
            return []
        tools = data.get("tools", []) if isinstance(data, dict) else []
        self._tools_cache = tools
        for tool in tools:
            name = tool.get("name")
            if name:
                self.capabilities.declare(Capability(
                    action=f"mcp::{name}",
                    params={},
                    safety_level=SafetyLevel.SENSITIVE_MODIFY,
                    timeout_seconds=self.timeout,
                    description=tool.get("description", "")[:200],
                ))
        return tools

    # ------------------------------------------------------------------
    def _dispatch(self, action: str, args: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        if action == "list_mcp_tools":
            if not self._tools_cache and self._initialized:
                self._discover()
            return self._tools_cache, {"count": len(self._tools_cache)}

        if action == "call_mcp_tool":
            tool = str(args["tool"])
            payload = args.get("arguments") or {}
            return self._call_tool(tool, payload)

        if action.startswith("mcp::"):
            return self._call_tool(action[5:], args.get("arguments") or {})

        raise NotImplementedError(f"mcp 适配器未实现 '{action}'")

    def _call_tool(self, tool: str, arguments: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        if not self._initialized:
            ok, why = self.connect()
            if not ok:
                return None, {"_status": ExecutionStatus.FAILED, "error": why}
        ok, data = self._rpc("tools/call", {"name": tool, "arguments": arguments})
        if not ok:
            return None, {"_status": ExecutionStatus.FAILED, "error": str(data)}
        if isinstance(data, dict) and data.get("isError"):
            return None, {"_status": ExecutionStatus.FAILED, "error": data.get("content")}
        return data, {"tool": tool}

    # ------------------------------------------------------------------
    def _rpc(self, method: str, params: Optional[Dict[str, Any]]) -> Tuple[bool, Any]:
        if self._proc is None or self._proc.stdin is None or self._proc.stdout is None:
            return False, "MCP server 未启动"
        with self._lock:
            self._rpc_id += 1
            message = {"jsonrpc": "2.0", "id": self._rpc_id, "method": method, "params": params or {}}
            try:
                self._proc.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                return False, f"写入失败：{exc}"

            deadline = time.time() + self.timeout
            while time.time() < deadline:
                line = self._proc.stdout.readline()
                if not line:
                    return False, "MCP server 无响应（进程可能已退出）"
                try:
                    payload = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
                if payload.get("id") != self._rpc_id:
                    continue
                if "error" in payload:
                    return False, payload["error"]
                return True, payload.get("result")

        return False, "超时"

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def shutil_which(name: str) -> Optional[str]:
    import shutil

    return shutil.which(name)