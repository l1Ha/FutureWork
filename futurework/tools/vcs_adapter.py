"""
版本控制适配器：Git 常用操作。

只封装最常被自然语言指挥的原子操作（status / diff / log / add / commit），
把 ``git`` 当成外部工具而非内嵌逻辑——这样任何 Git 实现（GitHub/GitLab/
本地/ worktree）都能一致工作。
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Tuple

from futurework.tools.base import Capability, ToolAdapter
from futurework.types import SafetyLevel


class VcsAdapter(ToolAdapter):
    """Git 版本控制。"""

    name = "vcs"

    def __init__(self, repo_path: Optional[str] = None) -> None:
        self.repo_path = os.path.abspath(repo_path or os.getcwd())
        self._last_commit: Optional[str] = None
        super().__init__()

    def setup(self) -> None:
        self.capabilities.declare(Capability(
            action="git_status",
            safety_level=SafetyLevel.READ_ONLY,
            description="查看仓库状态",
            aliases=("status",),
        ))
        self.capabilities.declare(Capability(
            action="git_diff",
            params={"staged": "bool?"},
            safety_level=SafetyLevel.READ_ONLY,
            description="查看改动差异",
            aliases=("diff",),
        ))
        self.capabilities.declare(Capability(
            action="git_log",
            params={"count": "int?"},
            safety_level=SafetyLevel.READ_ONLY,
            description="查看提交历史",
            aliases=("log",),
        ))
        self.capabilities.declare(Capability(
            action="git_commit",
            params={"message": "str", "all": "bool?"},
            safety_level=SafetyLevel.SENSITIVE_MODIFY,
            reversible=True,
            description="提交改动（可回退）",
            aliases=("commit",),
        ))
        self.capabilities.declare(Capability(
            action="git_add",
            params={"paths": "list?"},
            safety_level=SafetyLevel.SAFE_WRITE,
            description="暂存文件",
            aliases=("add", "stage"),
        ))

    def is_available(self) -> Tuple[bool, str]:
        if shutil.which("git") is None:
            return False, "未检测到 git 可执行文件"
        if not os.path.isdir(os.path.join(self.repo_path, ".git")):
            return False, f"{self.repo_path} 不是 Git 仓库"
        return True, f"git@{self.repo_path}"

    # ------------------------------------------------------------------
    def _git(self, *cmd: str, timeout: float = 20.0) -> Tuple[int, str, str]:
        try:
            proc = subprocess.run(
                ["git", *cmd], cwd=self.repo_path, capture_output=True,
                text=True, timeout=timeout, errors="replace",
            )
            return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
        except FileNotFoundError:
            return 127, "", "git 未安装"
        except subprocess.TimeoutExpired:
            return 124, "", f"git 命令超时（>{timeout}s）"
        except Exception as exc:
            return 1, "", f"{type(exc).__name__}: {exc}"

    def _dispatch(self, action: str, args: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        if action == "git_status":
            code, out, err = self._git("status", "--porcelain=v1", "--branch")
            if code != 0:
                return None, {"error": err}
            branch = ""
            entries: List[Dict[str, str]] = []
            for line in out.splitlines():
                if line.startswith("##"):
                    branch = line[2:].strip()
                    continue
                entries.append({"status": line[:2].strip(), "path": line[3:].strip()})
            return {"branch": branch, "changes": entries, "clean": not entries}, {"count": len(entries)}

        if action == "git_diff":
            cmd = ("diff", "--cached") if args.get("staged") else ("diff",)
            code, out, err = self._git(*cmd)
            if code != 0:
                return None, {"error": err}
            return out, {"lines": len(out.splitlines())}

        if action == "git_log":
            count = int(args.get("count") or 10)
            code, out, err = self._git("log", f"-{max(1, min(count, 100))}",
                                       "--pretty=format:%h|%an|%s")
            if code != 0:
                return None, {"error": err}
            commits = []
            for line in out.splitlines():
                parts = line.split("|", 2)
                if len(parts) == 3:
                    commits.append({"hash": parts[0], "author": parts[1], "subject": parts[2]})
            return commits, {"count": len(commits)}

        if action == "git_add":
            paths = args.get("paths") or ["."]
            code, _, err = self._git("add", *paths)
            if code != 0:
                return None, {"error": err}
            return list(paths), {}

        if action == "git_commit":
            message = str(args["message"]).strip()
            if not message:
                return None, {"error": "提交信息不能为空"}
            code, out, err = self._git("rev-parse", "HEAD")
            previous_head = out.strip() if code == 0 else None
            cmd = ["commit", "-m", message]
            if args.get("all"):
                cmd.insert(1, "-a")
            code, out, err = self._git(*cmd)
            if code != 0:
                return None, {"error": err, "previous_head": previous_head}

            # 必须用 rev-parse 取哈希：``git commit`` 的首行是
            # "[main abc1234] message" 这样的人读格式，直接拿去 reset 会失败。
            code, head, _ = self._git("rev-parse", "HEAD")
            self._last_commit = head.strip() if code == 0 else None
            return out.strip(), {
                "_rollback_available": self._last_commit is not None,
                "previous_head": previous_head,
            }

        raise NotImplementedError(f"vcs 适配器未实现 '{action}'")

    def rollback_last_commit(self) -> bool:
        """
        回退最近一次提交（供 UndoManager 使用）。

        撤销首个提交时 ``HEAD~1`` 没有父提交会直接报错，
        此时改用 ``update-ref -d HEAD`` 删掉分支引用——对根提交而言
        效果与 soft reset 相同：改动留在暂存区，工作区不动。
        """
        if not self._last_commit:
            return False
        code, _, _ = self._git("reset", "--soft", "HEAD~1")
        if code == 0:
            return True
        code, _, err = self._git("update-ref", "-d", "HEAD")
        if code != 0:
            self._last_rollback_error = err
        return code == 0