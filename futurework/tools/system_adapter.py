"""
系统适配器：窗口管理、应用启动、系统级操作。

跨平台策略：
* **Windows**：``ctypes`` + Win32 API；
* **macOS**：``osascript``；
* **Linux**：``wmctrl`` / ``xdotool`` / ``pyautogui``（按可用性降级）。

所有外部命令调用都做存在性检查，缺失时返回清晰的"如何安装"提示，
而不是抛 ``FileNotFoundError`` 让用户一脸茫然。
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Tuple

from futurework.tools.base import Capability, ToolAdapter
from futurework.types import ExecutionStatus, SafetyLevel

# 常见应用的跨平台别名映射（用户说"浏览器"，系统各有不同可执行名）
APP_ALIASES: Dict[str, Dict[str, str]] = {
    "浏览器": {"linux": "xdg-open", "macos": "open -a Safari", "windows": "start"},
    "浏览器browser": {"linux": "xdg-open", "macos": "open -a Safari", "windows": "start"},
    "终端": {"linux": "gnome-terminal", "macos": "open -a Terminal", "windows": "wt.exe"},
    "文件管理器": {"linux": "nautilus", "macos": "open", "windows": "explorer"},
    "编辑器": {"linux": "code", "macos": "code", "windows": "code"},
    "计算器": {"linux": "gnome-calculator", "macos": "open -a Calculator", "windows": "calc.exe"},
    "音乐": {"linux": "rhythmbox", "macos": "open -a Music", "windows": "wmplayer"},
}

_APP_NAME_MAP = {
    "chrome": {"linux": "google-chrome", "macos": "Google Chrome", "windows": "chrome.exe"},
    "vscode": {"linux": "code", "macos": "Visual Studio Code", "windows": "code"},
    "code": {"linux": "code", "macos": "Visual Studio Code", "windows": "code"},
    "wechat": {"linux": "wechat", "macos": "WeChat", "windows": "WeChat.exe"},
    "qq": {"linux": "qq", "macos": "QQ", "windows": "QQ.exe"},
    "firefox": {"linux": "firefox", "macos": "Firefox", "windows": "firefox.exe"},
}


class SystemAdapter(ToolAdapter):
    """操作系统窗口与应用控制。"""

    name = "system"

    def setup(self) -> None:
        self.capabilities.declare(Capability(
            action="focus_window",
            params={"name": "str"},
            safety_level=SafetyLevel.SAFE_WRITE,
            description="切换焦点到指定名称的窗口",
        ))
        self.capabilities.declare(Capability(
            action="list_windows",
            safety_level=SafetyLevel.READ_ONLY,
            description="列出当前所有窗口",
        ))
        self.capabilities.declare(Capability(
            action="close_window",
            params={"name": "str?"},
            safety_level=SafetyLevel.SENSITIVE_MODIFY,
            description="关闭窗口",
        ))
        self.capabilities.declare(Capability(
            action="minimize_all",
            safety_level=SafetyLevel.SAFE_WRITE,
            description="最小化全部窗口",
        ))
        self.capabilities.declare(Capability(
            action="maximize_window",
            params={"name": "str?"},
            safety_level=SafetyLevel.SAFE_WRITE,
            description="最大化窗口",
        ))
        self.capabilities.declare(Capability(
            action="launch_app",
            params={"app": "str"},
            safety_level=SafetyLevel.SAFE_WRITE,
            description="启动应用程序",
        ))
        self.capabilities.declare(Capability(
            action="system_info",
            safety_level=SafetyLevel.READ_ONLY,
            description="获取系统信息",
        ))

    def is_available(self) -> Tuple[bool, str]:
        system = platform.system().lower()
        if system not in ("linux", "darwin", "windows"):
            return False, f"当前操作系统 '{system}' 暂未内置窗口控制实现"
        return True, "ok"

    # 跨平台共享的动作：无需按操作系统分支
    SHARED_ACTIONS = ("launch_app", "system_info")

    # ------------------------------------------------------------------
    def _dispatch(self, action: str, args: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        if action in self.SHARED_ACTIONS:
            return getattr(self, f"_{action}")(args)

        system = platform.system().lower()
        if system == "darwin":
            handler = getattr(self, f"_darwin_{action}", None)
        elif system == "windows":
            handler = getattr(self, f"_win_{action}", None)
        else:
            handler = getattr(self, f"_linux_{action}", None)

        if handler is None:
            raise NotImplementedError(f"{platform.system()} 上未实现窗口操作 '{action}'")
        return handler(args)

    # ------------------------- 通用工具方法 -------------------------
    @staticmethod
    def _run(cmd: List[str], timeout: float = 5.0) -> Tuple[int, str, str]:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
        except FileNotFoundError:
            return 127, "", f"命令不存在：{cmd[0]}（请先安装对应工具）"
        except subprocess.TimeoutExpired:
            return 124, "", f"命令超时（>{timeout}s）：{' '.join(cmd)}"
        except Exception as exc:
            return 1, "", f"{type(exc).__name__}: {exc}"

    # ------------------------- Linux 实现 -------------------------
    def _linux_list_windows(self, args: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        if shutil.which("wmctrl"):
            code, out, err = self._run(["wmctrl", "-l"])
            if code != 0:
                return [], {"error": err}
            windows = []
            for line in out.splitlines():
                parts = line.split(None, 4)
                if len(parts) >= 5:
                    windows.append({"id": parts[0], "desktop": parts[1], "title": parts[4]})
            return windows, {"count": len(windows)}
        if shutil.which("xdotool"):
            code, out, err = self._run(["xdotool", "search", "--onlyvisible", "--name", ".*"])
            if code != 0:
                return [], {"error": err}
            return [{"id": wid.strip(), "title": ""} for wid in out.splitlines() if wid.strip()], {}
        return [], {"error": "未安装 wmctrl 或 xdotool，无法枚举窗口。请安装：sudo apt install wmctrl xdotool", "hint": True}

    def _linux_focus_window(self, args: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        name = args["name"]
        if shutil.which("wmctrl"):
            code, _, err = self._run(["wmctrl", "-a", name])
            if code == 0:
                return True, {}
            return False, {"error": err or "未找到匹配窗口"}
        if shutil.which("xdotool"):
            code, out, err = self._run(["xdotool", "search", "--name", name])
            if code == 0 and out:
                wid = out.splitlines()[0]
                self._run(["xdotool", "windowactivate", wid])
                return True, {}
            return False, {"error": err or "未找到匹配窗口"}
        return False, {"error": "缺少窗口控制工具 wmctrl/xdotool", "hint": True}

    def _linux_close_window(self, args: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        if shutil.which("wmctrl"):
            code, _, err = self._run(["wmctrl", "-c", args.get("name") or ":ACTIVE:"])
            return code == 0, {"error": err} if code else {}
        return False, {"error": "缺少 wmctrl", "hint": True}

    def _linux_minimize_all(self, args: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        if shutil.which("wmctrl"):
            code, _, err = self._run(["wmctrl", "-r", ":ACTIVE:", "-b", "add,hidden"])
            return code == 0, {"error": err} if code else {}
        return False, {"error": "缺少 wmctrl", "hint": True}

    def _linux_maximize_window(self, args: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        if shutil.which("wmctrl"):
            code, _, err = self._run(["wmctrl", "-r", args.get("name") or ":ACTIVE:", "-b", "add,maximized_vert,maximized_horz"])
            return code == 0, {"error": err} if code else {}
        return False, {"error": "缺少 wmctrl", "hint": True}

    # ------------------------- macOS 实现 -------------------------
    def _darwin_list_windows(self, args: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        script = (
            'tell application "System Events" to get {name, title} of every process '
            "whose background only is false"
        )
        code, out, err = self._run(["osascript", "-e", script])
        if code != 0:
            return [], {"error": err, "hint": "可能需要在 系统设置 → 隐私与安全性 中授权辅助功能"}
        items: List[Dict[str, str]] = []
        for chunk in out.split(", {"):
            if "title=" in chunk or "name=" in chunk:
                title = chunk.split("title=")[-1].strip().strip('"}') if "title=" in chunk else ""
                items.append({"title": title})
        return items, {}

    def _darwin_focus_window(self, args: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        script = f'tell application "{args["name"]}" to activate'
        code, _, err = self._run(["osascript", "-e", script])
        return code == 0, {"error": err} if code else {}

    def _darwin_close_window(self, args: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        name = args.get("name")
        script = f'tell application "{name}" to quit' if name else \
            'tell application "System Events" to keystroke "w" using command down'
        code, _, err = self._run(["osascript", "-e", script])
        return code == 0, {"error": err} if code else {}

    def _darwin_minimize_all(self, args: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        code, _, err = self._run(["osascript", "-e", 'tell application "System Events" to hide every process'])
        return code == 0, {"error": err} if code else {}

    def _darwin_maximize_window(self, args: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        code, _, err = self._run(["osascript", "-e", 'tell application "System Events" to set zoomed of front window to true'])
        return code == 0, {"error": err} if code else {}

    # ------------------------- Windows 实现 -------------------------
    def _win_list_windows(self, args: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        script = (
            "$s = Get-Process | Where-Object {$_.MainWindowTitle -ne ''} | "
            "Select-Object Id, MainWindowTitle; $s | ConvertTo-Json -Compress"
        )
        code, out, err = self._run(["powershell", "-NoProfile", "-Command", script])
        if code != 0:
            return [], {"error": err}
        import json

        try:
            data = json.loads(out or "[]")
        except json.JSONDecodeError:
            return [], {"error": "PowerShell 输出解析失败", "raw": out[:200]}
        if isinstance(data, dict):
            data = [data]
        return [{"id": d.get("Id"), "title": d.get("MainWindowTitle")} for d in data], {}

    def _win_focus_window(self, args: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        script = (
            f"$p = Get-Process | Where-Object {{$_.MainWindowTitle -like '*{args['name']}*'}} | "
            "Select-Object -First 1; if ($p) { (New-Object -ComObject WScript.Shell).AppActivate($p.Id); 'ok' } else { 'notfound' }"
        )
        code, out, err = self._run(["powershell", "-NoProfile", "-Command", script])
        if code != 0:
            return False, {"error": err}
        return out.strip() == "ok", {} if out.strip() == "ok" else {"error": "未找到匹配窗口"}

    def _win_close_window(self, args: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        name = args.get("name")
        script = (
            f"$p = Get-Process | Where-Object {{$_.MainWindowTitle -like '*{name}*'}} | Select-Object -First 1; "
            "if ($p) { $p.CloseMainWindow() }"
        ) if name else "$p = Get-Process | Where-Object {$_.MainWindowHandle -ne 0} | Select-Object -First 1; if ($p) { $p.CloseMainWindow() }"
        code, _, err = self._run(["powershell", "-NoProfile", "-Command", script])
        return code == 0, {"error": err} if code else {}

    def _win_minimize_all(self, args: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        code, _, err = self._run(["powershell", "-NoProfile", "-Command",
                                  "(New-Object -ComObject Shell.Application).MinimizeAll()"])
        return code == 0, {"error": err} if code else {}

    def _win_maximize_window(self, args: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        script = (
            f"$p = Get-Process | Where-Object {{$_.MainWindowTitle -like '*{args.get('name') or ''}*'}} | Select-Object -First 1; "
            "if ($p) { [Win32]::ShowWindow($p.MainWindowHandle, 3) }"
        )
        code, _, err = self._run(["powershell", "-NoProfile", "-Command", script])
        return code == 0, {"error": err} if code else {}

    # ------------------------- 应用启动（跨平台） -------------------------
    def _launch_app(self, args: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        app = str(args["app"]).strip().strip("的")
        system = platform.system().lower()
        key = "linux" if system == "linux" else ("macos" if system == "darwin" else "windows")

        alias = APP_ALIASES.get(app) or APP_ALIASES.get(app.lower()) or _APP_NAME_MAP.get(app.lower())
        if alias:
            cmd = alias[key]
        else:
            cmd = {"linux": app, "macos": f"open -a '{app}'", "windows": f"start {app}"}[key]

        parts = cmd.split()
        executable = parts[0]
        if shutil.which(executable) is None and key == "linux":
            return False, {
                "error": f"找不到应用 '{app}'（可执行文件 {executable} 不在 PATH 中）",
                "hint": "可以用完整路径，或先安装该应用",
            }
        try:
            subprocess.Popen(parts, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True, {"command": cmd}
        except Exception as exc:
            return False, {"error": f"启动失败：{exc}"}

    def _system_info(self, args: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        return {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "python": platform.python_version(),
            "cwd": os.getcwd(),
            "window_tool": "wmctrl" if shutil.which("wmctrl") else ("xdotool" if shutil.which("xdotool") else "none"),
        }, {}