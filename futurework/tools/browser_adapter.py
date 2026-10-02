"""
浏览器适配器：打开网址、搜索、前进后退。

设计要点：默认使用系统默认浏览器打开（零依赖、用户可控），
同时预留 Playwright/CDP 注入点，便于未来做"读页面内容并总结"的闭环。
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple

from futurework.tools.base import Capability, ToolAdapter
from futurework.types import SafetyLevel

SEARCH_ENGINES = {
    "google": "https://www.google.com/search?q={}",
    "bing": "https://www.bing.com/search?q={}",
    "baidu": "https://www.baidu.com/s?wd={}",
    "duckduckgo": "https://duckduckgo.com/?q={}",
}
DEFAULT_SEARCH_ENGINE = "bing"


class BrowserAdapter(ToolAdapter):
    """网页导航与搜索。"""

    name = "browser"

    def __init__(self, search_engine: str = DEFAULT_SEARCH_ENGINE) -> None:
        self.search_engine = search_engine if search_engine in SEARCH_ENGINES else DEFAULT_SEARCH_ENGINE
        self._history: List[Dict[str, str]] = []
        self._back_stack: List[str] = []
        super().__init__()

    def setup(self) -> None:
        self.capabilities.declare(Capability(
            action="open_url",
            params={"url": "str"},
            safety_level=SafetyLevel.SAFE_WRITE,
            validator=_validate_url,
            description="在默认浏览器中打开网址",
        ))
        self.capabilities.declare(Capability(
            action="search_web",
            params={"query": "str", "engine": "str?"},
            safety_level=SafetyLevel.SAFE_WRITE,
            description="搜索网页",
            aliases=("web_search", "google"),
        ))
        self.capabilities.declare(Capability(
            action="browser_back",
            safety_level=SafetyLevel.SAFE_WRITE,
            description="浏览器后退",
            aliases=("go_back", "navigate_back"),
        ))
        self.capabilities.declare(Capability(
            action="history",
            safety_level=SafetyLevel.READ_ONLY,
            description="查看本次会话访问过的网址",
        ))

    def is_available(self) -> Tuple[bool, str]:
        opener = _get_opener()
        if opener is None:
            return False, "未找到可用的浏览器打开方式（xdg-open / open / start）"
        return True, f"opener={opener}"

    # ------------------------------------------------------------------
    def _dispatch(self, action: str, args: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        if action == "open_url":
            return self._open_url(args["url"], args.get("engine"))
        if action == "search_web":
            engine = args.get("engine") or self.search_engine
            template = SEARCH_ENGINES.get(engine, SEARCH_ENGINES[self.search_engine])
            url = template.format(urllib.parse.quote(str(args["query"])))
            return self._open_url(url, engine, query=str(args["query"]))
        if action == "browser_back":
            if not self._back_stack:
                return False, {"error": "没有可返回的历史记录"}
            previous = self._back_stack.pop()
            return self._launch(previous)
        if action == "history":
            return list(self._history), {"count": len(self._history)}
        raise NotImplementedError(f"browser 适配器未实现 '{action}'")

    def _open_url(self, url: str, engine: Optional[str] = None, query: Optional[str] = None) -> Tuple[Any, Dict[str, Any]]:
        normalized = normalize_url(url)
        result, details = self._launch(normalized)
        if result:
            self._back_stack.append(normalized)
            self._history.append({"url": normalized, "engine": engine or "direct", "query": query or ""})
        return result, details

    def _launch(self, url: str) -> Tuple[bool, Dict[str, Any]]:
        opener = _get_opener()
        if opener is None:
            return False, {"error": "无法打开浏览器：系统缺少 xdg-open/open/start"}
        try:
            subprocess.Popen([opener, url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True, {"url": url}
        except Exception as exc:
            return False, {"error": f"打开浏览器失败：{exc}"}


def _get_opener() -> Optional[str]:
    system = platform.system().lower()
    if system == "darwin":
        return "open" if shutil.which("open") else None
    if system == "windows":
        return "cmd" if shutil.which("cmd") else None
    return "xdg-open" if shutil.which("xdg-open") else None


def normalize_url(url: str) -> str:
    """补全协议头；裸域名按 https 处理。"""
    url = url.strip()
    if not url:
        return url
    if url.startswith(("http://", "https://", "file://", "about:")):
        return url
    if url.startswith("localhost") or url.startswith("127.0.0.1") or url.startswith(":"):
        return f"http://{url}"
    if "." in url.split("/")[0]:
        return f"https://{url}"
    return url


def _validate_url(args: Dict[str, Any]) -> Optional[str]:
    url = str(args.get("url", "")).strip()
    if not url:
        return "url 不能为空"
    if not re_match(r"^[\w.-]+\.[a-zA-Z]{2,}", url) and not url.startswith(("http", "file", "about", "localhost")):
        return f"无法识别的网址格式：'{url}'"
    return None


def re_match(pattern: str, value: str) -> bool:
    import re

    return bool(re.search(pattern, value))