"""
文件系统适配器：读写、追加、删除、搜索、列目录。

安全设计：
* 所有路径经 ``safe_join_validator`` 限定在允许的根目录内，防路径穿越；
* 写入/删除前自动快照，可精确回滚；
* 大文件读取有上限，避免一次把内存撑爆。
"""

from __future__ import annotations

import os
import shutil
from typing import Any, Dict, List, Optional, Tuple

from futurework.tools.base import Capability, ToolAdapter, safe_join_validator
from futurework.types import ExecutionStatus, SafetyLevel

MAX_READ_BYTES = 2 * 1024 * 1024  # 2 MB
MAX_SEARCH_RESULTS = 200


class FileSystemAdapter(ToolAdapter):
    """受限根目录内的文件操作。"""

    name = "filesystem"

    def __init__(self, root: Optional[str] = None) -> None:
        self.root = os.path.abspath(root or os.getcwd())
        self._snapshots: Dict[str, Any] = {}
        super().__init__()
        # "列出目录"而不带路径是完全自然的说法，允许路径缺省
        self.capabilities.declare(Capability(
            action="list_directory",
            params={"path": "str?"},
            safety_level=SafetyLevel.READ_ONLY,
            validator=safe_join_validator(self.root, allow_empty=True),
            description="列出目录内容（不指定路径时列出根目录）",
        ))

    def setup(self) -> None:
        v = safe_join_validator(self.root)
        self.capabilities.declare(Capability(
            action="read_file",
            params={"path": "str"},
            safety_level=SafetyLevel.READ_ONLY,
            validator=v,
            description="读取文本文件内容",
        ))
        self.capabilities.declare(Capability(
            action="write_file",
            params={"path": "str", "content": "str"},
            # 可逆（有快照）就不该弹确认——真实编辑器保存文件也不会问你
            safety_level=SafetyLevel.SAFE_WRITE,
            reversible=True,
            validator=v,
            description="覆盖写入文件（可撤销）",
        ))
        self.capabilities.declare(Capability(
            action="append_file",
            params={"path": "str", "content": "str"},
            safety_level=SafetyLevel.SAFE_WRITE,
            reversible=True,
            validator=v,
            description="追加内容到文件（可撤销）",
        ))
        self.capabilities.declare(Capability(
            action="delete_file",
            params={"path": "str"},
            safety_level=SafetyLevel.CRITICAL_DESTRUCTIVE,
            reversible=True,
            validator=v,
            description="删除文件（需确认，可撤销）",
        ))
        self.capabilities.declare(Capability(
            action="create_directory",
            params={"path": "str"},
            safety_level=SafetyLevel.SAFE_WRITE,
            validator=v,
            description="创建目录",
        ))
        self.capabilities.declare(Capability(
            action="copy_file",
            params={"source": "str", "destination": "str"},
            safety_level=SafetyLevel.SAFE_WRITE,
            reversible=True,
            validator=_validate_transfer(self.root),
            description="复制文件（可撤销）",
        ))
        self.capabilities.declare(Capability(
            action="move_file",
            params={"source": "str", "destination": "str"},
            # 可逆（有快照可搬回去），移动文件不值得每次都打断用户
            safety_level=SafetyLevel.SAFE_WRITE,
            reversible=True,
            validator=_validate_transfer(self.root),
            description="移动文件（可撤销）",
        ))
        self.capabilities.declare(Capability(
            action="list_directory",
            params={"path": "str?"},
            safety_level=SafetyLevel.READ_ONLY,
            validator=v,
            description="列出目录内容",
        ))
        self.capabilities.declare(Capability(
            action="search_in_file",
            params={"path": "str", "keyword": "str"},
            safety_level=SafetyLevel.READ_ONLY,
            validator=v,
            description="在文件中搜索关键词",
        ))

    # ------------------------------------------------------------------
    def _resolve(self, path: str) -> str:
        candidate = os.path.abspath(os.path.join(self.root, path))
        root = self.root
        if not (candidate == root or candidate.startswith(root + os.sep)):
            raise PermissionError(f"拒绝访问根目录之外的路径：'{path}'")
        return candidate

    def _snapshot(self, key: str, path: str) -> None:
        """记录文件状态，供撤销使用。"""
        if os.path.exists(path):
            with open(path, "rb") as fh:
                self._snapshots[key] = ("file", fh.read())
        else:
            self._snapshots[key] = ("absent", None)

    def restore(self, key: str) -> bool:
        """从快照恢复（供 UndoManager 调用）。"""
        state = self._snapshots.get(key)
        if state is None:
            return False
        kind, payload = state
        path = self._snapshots.get(f"{key}::path")
        if not isinstance(path, str):
            return False
        if kind == "file":
            with open(path, "wb") as fh:
                fh.write(payload)
        else:
            if os.path.exists(path):
                os.remove(path)
        return True

    # ------------------------------------------------------------------
    def _dispatch(self, action: str, args: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        path = self._resolve(args.get("path") or ".")

        if action == "read_file":
            if not os.path.exists(path):
                raise FileNotFoundError(f"文件不存在：{args.get('path')}")
            size = os.path.getsize(path)
            if size > MAX_READ_BYTES:
                with open(path, "r", encoding="utf-8", errors="replace") as fh:
                    text = fh.read(MAX_READ_BYTES)
                return text, {"truncated": True, "size": size}
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                return fh.read(), {"size": size}

        if action == "write_file":
            os.makedirs(os.path.dirname(path) or self.root, exist_ok=True)
            self._snapshot(path, path)
            self._snapshots[f"{path}::path"] = path
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(args.get("content", ""))
            return path, {"bytes": len(args.get("content", "").encode()), "_rollback_available": True}

        if action == "append_file":
            os.makedirs(os.path.dirname(path) or self.root, exist_ok=True)
            self._snapshot(path, path)
            self._snapshots[f"{path}::path"] = path
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(args.get("content", ""))
            return path, {"_rollback_available": True}

        if action == "delete_file":
            if not os.path.exists(path):
                raise FileNotFoundError(f"文件不存在：{args.get('path')}")
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
            self._snapshots[f"{path}::path"] = path
            return path, {"_rollback_available": False, "note": "删除已执行但内容不可恢复"}

        if action == "create_directory":
            os.makedirs(path, exist_ok=True)
            return path, {}

        if action in ("copy_file", "move_file"):
            source = self._resolve(args["source"])
            destination = self._resolve(args["destination"])
            if not os.path.exists(source):
                raise FileNotFoundError(f"源文件不存在：{args['source']}")
            if os.path.isdir(destination):
                destination = os.path.join(destination, os.path.basename(source))
            os.makedirs(os.path.dirname(destination) or self.root, exist_ok=True)
            if os.path.abspath(source) == os.path.abspath(destination):
                return destination, {"error": None, "noop": True}

            self._snapshots[f"{destination}::path"] = destination
            self._snapshots[destination] = ("absent", None)
            self._snapshots[f"{source}::move"] = source

            if action == "copy_file":
                shutil.copy2(source, destination)
                return destination, {"_rollback_available": True, "source": source}
            shutil.move(source, destination)
            return destination, {"_rollback_available": True, "source": source}

        if action == "list_directory":
            if not os.path.isdir(path):
                raise NotADirectoryError(f"不是目录：{args.get('path')}")
            entries: List[Dict[str, Any]] = []
            for name in sorted(os.listdir(path)):
                full = os.path.join(path, name)
                entries.append({
                    "name": name,
                    "is_dir": os.path.isdir(full),
                    "size": os.path.getsize(full) if os.path.isfile(full) else None,
                })
            return entries, {"count": len(entries)}

        if action == "search_in_file":
            if not os.path.isfile(path):
                raise FileNotFoundError(f"不是文件：{args.get('path')}")
            keyword = args.get("keyword", "")
            hits: List[Dict[str, Any]] = []
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                for lineno, line in enumerate(fh, 1):
                    if keyword in line:
                        hits.append({"line": lineno, "text": line.rstrip("\n")[:300]})
                        if len(hits) >= MAX_SEARCH_RESULTS:
                            break
            return hits, {"count": len(hits), "truncated": len(hits) >= MAX_SEARCH_RESULTS}

        raise NotImplementedError(f"filesystem 适配器未实现 '{action}'")


def _validate_transfer(root: str):
    """复制/移动的双路径校验：两端都不得越出允许的根目录。"""
    import os

    def _check(args):
        for key in ("source", "destination"):
            value = args.get(key)
            if not value:
                return f"缺少必填参数 '{key}'"
            resolved = os.path.abspath(os.path.join(root, value))
            base = os.path.abspath(root)
            if not (resolved == base or resolved.startswith(base + os.sep)):
                return f"路径越界：'{value}' 不在允许的根目录内"
        return None

    return _check
