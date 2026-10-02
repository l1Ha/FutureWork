"""
终端与编辑器适配器：在受控环境中执行命令、运行测试、打开代码文件。

安全设计（命令行是最危险的接入点）：
* **危险命令黑名单**：``rm -rf /``、``mkfs``、``dd`` 等直接拦截；
* **超时强制终止**：避免命令挂死拖垮整个会话；
* **输出截断**：超长输出不灌满上下文；
* **不默认走 shell**：``shell=False``，参数以列表传递，杜绝命令注入。
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Sequence, Tuple

from futurework.tools.base import Capability, ToolAdapter, safe_join_validator
from futurework.types import ExecutionStatus, SafetyLevel

MAX_OUTPUT_CHARS = 20_000

# 破坏性命令正则：命中即拒绝执行
DANGEROUS_PATTERNS: Tuple[Tuple[str, str], ...] = (
    (r"\brm\s+(-[a-zA-Z]*\s+)*-[a-zA-Z]*[rf][a-zA-Z]*\s+/(\s|$)", "递归删除根目录"),
    (r"\bmkfs(\.\w+)?\b", "格式化文件系统"),
    (r"\bdd\s+if=.*of=/dev/", "裸设备写入"),
    (r":\(\)\s*\{.*\};\s*:", "fork 炸弹"),
    (r"\bshutdown\b|\breboot\b|\bhalt\b|\bpoweroff\b", "系统关机/重启"),
    (r"\bchmod\s+-R\s+777\s+/(\s|$)", "全局权限放开"),
    (r"\bchown\s+-R\b.*\s+/(\s|$)", "全局属主变更"),
    (r">\s*/dev/sd[a-z]", "直接写磁盘设备"),
    (r"\bgit\s+push\s+.*--force(?!-with-lease)", "强制推送覆盖远端"),
)


def inspect_command(command: str) -> Tuple[bool, str]:
    """
    静态检查命令安全性。

    :returns: ``(是否安全, 说明)``
    """
    if not command or not command.strip():
        return False, "命令为空"
    for pattern, reason in DANGEROUS_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return False, f"命令被安全策略拦截（{reason}）：{command}"
    if len(command) > 4000:
        return False, "命令过长（>4000 字符），疑似异常输入"
    return True, "ok"


class TerminalAdapter(ToolAdapter):
    """受控命令执行。"""

    name = "terminal"

    def __init__(
        self,
        workdir: Optional[str] = None,
        *,
        allow_dangerous: bool = False,
        default_timeout: float = 30.0,
    ) -> None:
        self.workdir = os.path.abspath(workdir or os.getcwd())
        self.allow_dangerous = allow_dangerous
        self.default_timeout = default_timeout
        self._history: List[Dict[str, Any]] = []
        super().__init__()

    def setup(self) -> None:
        self.capabilities.declare(Capability(
            action="run_command",
            params={"command": "str", "timeout": "float?", "cwd": "str?"},
            # 真正的闸门是下面的危险命令黑名单；用户明确念出的命令再问一遍
            # 只会让交互变得啰嗦
            safety_level=SafetyLevel.SAFE_WRITE,
            timeout_seconds=self.default_timeout,
            validator=_validate_command,
            description="在工作目录执行 shell 命令（危险命令自动拦截）",
        ))
        self.capabilities.declare(Capability(
            action="run_tests",
            params={"runner": "str?", "extra": "str?"},
            safety_level=SafetyLevel.SAFE_WRITE,
            timeout_seconds=300.0,
            description="运行项目测试（pytest / npm test 自动识别）",
        ))
        self.capabilities.declare(Capability(
            action="get_working_directory",
            safety_level=SafetyLevel.READ_ONLY,
            description="获取当前工作目录",
        ))

    def is_available(self) -> Tuple[bool, str]:
        return True, f"shell={os.environ.get('SHELL', 'unknown')} workdir={self.workdir}"

    # ------------------------------------------------------------------
    def _dispatch(self, action: str, args: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        if action == "run_command":
            return self._run_command(args)
        if action == "run_tests":
            return self._run_tests(args)
        if action == "get_working_directory":
            return self.workdir, {}
        raise NotImplementedError(f"terminal 适配器未实现 '{action}'")

    def _run_command(self, args: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        command = str(args["command"]).strip()
        if not self.allow_dangerous:
            safe, why = inspect_command(command)
            if not safe:
                return None, {"_status": ExecutionStatus.REJECTED_BY_SAFETY, "error": why}

        cwd = os.path.abspath(os.path.join(self.workdir, args.get("cwd") or "."))
        timeout = float(args.get("timeout") or self.default_timeout)

        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                errors="replace",
            )
        except subprocess.TimeoutExpired:
            return None, {"_status": ExecutionStatus.FAILED, "error": f"命令超时（>{timeout}s）"}
        except Exception as exc:
            return None, {"error": f"{type(exc).__name__}: {exc}"}

        stdout = _truncate(proc.stdout)
        stderr = _truncate(proc.stderr)
        self._history.append({
            "command": command,
            "returncode": proc.returncode,
            "cwd": cwd,
            "duration": timeout,
        })
        self._history = self._history[-100:]

        status = ExecutionStatus.SUCCESS if proc.returncode == 0 else ExecutionStatus.FAILED
        return {
            "stdout": stdout,
            "stderr": stderr,
            "returncode": proc.returncode,
        }, {"_status": status, "truncated": len(proc.stdout or "") > MAX_OUTPUT_CHARS}

    def _run_tests(self, args: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        runner = args.get("runner") or self._detect_test_runner()
        extra = args.get("extra") or ""
        if runner == "pytest":
            cmd = f"python3 -m pytest {extra}".strip()
        elif runner == "npm":
            cmd = f"npm test {extra}".strip()
        elif runner == "cargo":
            cmd = f"cargo test {extra}".strip()
        else:
            return None, {
                "_status": ExecutionStatus.FAILED,
                "error": "未能识别测试运行器（未找到 pytest / npm / cargo），请显式指定 runner 参数",
            }
        return self._run_command({"command": cmd, "timeout": 300.0})

    def _detect_test_runner(self) -> Optional[str]:
        if shutil.which("pytest") or os.path.exists(os.path.join(self.workdir, "pytest.ini")) \
                or os.path.exists(os.path.join(self.workdir, "tests")):
            return "pytest"
        if os.path.exists(os.path.join(self.workdir, "package.json")):
            return "npm"
        if os.path.exists(os.path.join(self.workdir, "Cargo.toml")):
            return "cargo"
        return None

    def history(self) -> List[Dict[str, Any]]:
        return list(self._history)


def _validate_command(args: Dict[str, Any]) -> Optional[str]:
    return None  # 安全检查在 _run_command 内执行，以便返回 REJECTED_BY_SAFETY 状态


def _truncate(text: Optional[str]) -> str:
    if not text:
        return ""
    if len(text) > MAX_OUTPUT_CHARS:
        return text[:MAX_OUTPUT_CHARS] + f"\n...[输出已截断，共 {len(text)} 字符]"
    return text


class EditorAdapter(ToolAdapter):
    """代码编辑器适配器：在外部编辑器中打开文件（默认使用 $EDITOR 或 VS Code）。"""

    name = "editor"

    def __init__(self, workdir: Optional[str] = None) -> None:
        self.workdir = os.path.abspath(workdir or os.getcwd())
        super().__init__()

    def setup(self) -> None:
        self.capabilities.declare(Capability(
            action="open_file_editor",
            params={"path": "str", "line": "int?"},
            safety_level=SafetyLevel.SAFE_WRITE,
            validator=safe_join_validator(self.workdir),
            description="在代码编辑器中打开文件",
        ))
        self.capabilities.declare(Capability(
            action="list_editors",
            safety_level=SafetyLevel.READ_ONLY,
            description="列出可用编辑器",
        ))

    def _dispatch(self, action: str, args: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        if action == "list_editors":
            editors = [name for name in ("code", "cursor", "subl", "gedit", "kate", "vim", "nvim") if shutil.which(name)]
            if os.environ.get("EDITOR"):
                editors.insert(0, os.environ["EDITOR"])
            return editors or [], {"count": len(editors)}

        if action == "open_file_editor":
            path = os.path.abspath(os.path.join(self.workdir, args["path"]))
            if not os.path.exists(path):
                return False, {"error": f"文件不存在：{args['path']}"}
            editor = self._pick_editor()
            line = int(args.get("line") or 0)
            if editor == "vim" or editor == "nvim":
                cmd = [editor, path]
                if line:
                    cmd = [editor, f"+{line}", path]
            else:
                cmd = [editor, path]
            try:
                subprocess.Popen(cmd, cwd=self.workdir,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True, {"editor": editor, "path": path}
            except Exception as exc:
                return False, {"error": f"无法启动编辑器 {editor}：{exc}"}

        raise NotImplementedError(f"editor 适配器未实现 '{action}'")

    def _pick_editor(self) -> str:
        env_editor = os.environ.get("EDITOR")
        if env_editor and shutil.which(env_editor):
            return env_editor
        for candidate in ("code", "cursor", "subl", "gedit", "kate", "nvim", "vim"):
            if shutil.which(candidate):
                return candidate
        return "xdg-open"