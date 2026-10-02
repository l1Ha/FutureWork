"""
办公与通用生产力套件适配器 (Office & Productivity Suite Adapter)

为现代白领、研究人员、工程师的核心桌面办公场景提供自然语言直达接口：
1. **表格与数据分析 (Table & Spreadsheet Analytics)**：
   无需打开复杂电子表格软件，用口语即可对 CSV / TSV 数据完成列过滤、排序、求和、均值统计；
2. **结构化文档与报告生成 (Structured Documents & Reports)**：
   自动生成大纲、插入章节、构建 Markdown / 网页工作报告；
3. **日程与待办清单管理 (Task & Todo Management)**：
   口语快速记录灵感待办、勾选完成、查看当前未结事项。
"""

from __future__ import annotations

import csv
import io
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from futurework.tools.base import Capability, ToolAdapter, safe_join_validator
from futurework.types import ExecutionStatus, SafetyLevel


class ProductivitySuiteAdapter(ToolAdapter):
    """
    通用生产力套件适配器（表格数据、结构化文档报告、待办日程）。
    """

    name = "productivity"
    description = "办公生产力套件：表格统计、文档报告生成、任务待办管理"

    def __init__(self, workdir: Optional[str] = None) -> None:
        self.workdir = os.path.abspath(workdir or os.getcwd())
        self._todos_file = os.path.join(self.workdir, ".futurework", "todos.json")
        self._snapshots: Dict[str, Any] = {}
        super().__init__()

    def setup(self) -> None:
        v = safe_join_validator(self.workdir)
        # 1. 表格操作
        self.capabilities.declare(Capability(
            action="table_aggregate",
            params={"path": "str", "column": "str", "operation": "str?"},
            safety_level=SafetyLevel.READ_ONLY,
            validator=v,
            description="对表格/CSV 数据的指定列进行统计聚合（sum, mean, max, min, count）",
            aliases=("csv_stats", "calc_table"),
        ))
        self.capabilities.declare(Capability(
            action="table_query",
            params={"path": "str", "filter_column": "str?", "filter_value": "str?", "sort_by": "str?", "limit": "int?"},
            safety_level=SafetyLevel.READ_ONLY,
            validator=v,
            description="查询、筛选与排序表格/CSV 行记录",
            aliases=("filter_table", "query_table"),
        ))

        # 2. 结构化文档与报告生成
        self.capabilities.declare(Capability(
            action="generate_report",
            params={"title": "str", "sections": "list?", "output_path": "str?"},
            safety_level=SafetyLevel.SAFE_WRITE,
            reversible=True,
            description="生成结构化工作报告或周报文档（Markdown 格式）",
            aliases=("create_report", "doc_report"),
        ))
        self.capabilities.declare(Capability(
            action="extract_outline",
            params={"path": "str"},
            safety_level=SafetyLevel.READ_ONLY,
            validator=v,
            description="提取文档大纲结构（标题层级与字数概览）",
            aliases=("doc_outline", "outline"),
        ))

        # 3. 待办与任务管理
        self.capabilities.declare(Capability(
            action="todo_add",
            params={"task": "str", "priority": "str?"},
            safety_level=SafetyLevel.SAFE_WRITE,
            reversible=True,
            description="添加一条待办事项",
            aliases=("add_todo", "create_todo"),
        ))
        self.capabilities.declare(Capability(
            action="todo_list",
            params={"status": "str?"},
            safety_level=SafetyLevel.READ_ONLY,
            description="查看待办任务清单（全部、未完成或已完成）",
            aliases=("list_todos", "show_todos"),
        ))
        self.capabilities.declare(Capability(
            action="todo_complete",
            params={"task_id_or_keyword": "str"},
            safety_level=SafetyLevel.SAFE_WRITE,
            reversible=True,
            description="将指定待办任务标记为已完成",
            aliases=("complete_todo", "done_todo"),
        ))

    # ------------------------------------------------------------------
    def _dispatch(self, action: str, args: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        if action == "table_aggregate":
            return self._table_aggregate(args)
        if action == "table_query":
            return self._table_query(args)
        if action == "generate_report":
            return self._generate_report(args)
        if action == "extract_outline":
            return self._extract_outline(args)
        if action == "todo_add":
            return self._todo_add(args)
        if action == "todo_list":
            return self._todo_list(args)
        if action == "todo_complete":
            return self._todo_complete(args)
        raise NotImplementedError(f"productivity 适配器未实现 '{action}'")

    # ------------------------------------------------------------------
    # 表格功能
    # ------------------------------------------------------------------
    def _load_csv(self, rel_path: str) -> List[Dict[str, str]]:
        full_path = os.path.abspath(os.path.join(self.workdir, rel_path))
        if not os.path.isfile(full_path):
            raise FileNotFoundError(f"表格文件不存在：{rel_path}")
        with open(full_path, "r", encoding="utf-8-sig", errors="replace") as f:
            reader = csv.DictReader(f)
            return list(reader)

    def _table_aggregate(self, args: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        path = args["path"]
        col = args["column"].strip()
        op = (args.get("operation") or "sum").strip().lower()

        rows = self._load_csv(path)
        if not rows:
            return None, {"error": "表格为空"}
        if col not in rows[0]:
            avail = ", ".join(rows[0].keys())
            return None, {"error": f"表格中不存在列 '{col}'，当前包含列：{avail}"}

        values: List[float] = []
        for r in rows:
            val_str = r.get(col, "").strip().replace(",", "")
            try:
                values.append(float(val_str))
            except ValueError:
                pass

        if not values and op in ("sum", "mean", "max", "min"):
            return None, {"error": f"列 '{col}' 不包含有效数值，无法计算 {op}"}

        res: Any = None
        if op == "sum":
            res = round(sum(values), 3)
        elif op in ("mean", "avg", "average"):
            res = round(sum(values) / len(values), 3)
        elif op == "max":
            res = max(values)
        elif op == "min":
            res = min(values)
        elif op == "count":
            res = len(rows)
        else:
            return None, {"error": f"不支持的聚合操作 '{op}'，可选：sum, mean, max, min, count"}

        return {
            "file": path,
            "column": col,
            "operation": op,
            "result": res,
            "valid_samples": len(values),
            "total_rows": len(rows),
        }, {"summary": f"{col} 的 {op} 统计值为 {res}"}

    def _table_query(self, args: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        path = args["path"]
        filter_col = args.get("filter_column")
        filter_val = args.get("filter_value")
        sort_by = args.get("sort_by")
        limit = int(args.get("limit") or 20)

        rows = self._load_csv(path)
        filtered = rows
        if filter_col and filter_val is not None:
            filtered = [r for r in filtered if str(filter_val).lower() in str(r.get(filter_col, "")).lower()]

        if sort_by and rows and sort_by in rows[0]:
            try:
                filtered = sorted(filtered, key=lambda r: float(r.get(sort_by, 0)), reverse=True)
            except ValueError:
                filtered = sorted(filtered, key=lambda r: str(r.get(sort_by, "")), reverse=False)

        return filtered[:limit], {"total_matched": len(filtered), "returned": len(filtered[:limit])}

    # ------------------------------------------------------------------
    # 文档大纲与报告
    # ------------------------------------------------------------------
    def _extract_outline(self, args: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        full_path = os.path.abspath(os.path.join(self.workdir, args["path"]))
        if not os.path.isfile(full_path):
            raise FileNotFoundError(f"文件不存在：{args['path']}")

        outline: List[Dict[str, Any]] = []
        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
            for idx, line in enumerate(f, 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    level = len(stripped) - len(stripped.lstrip("#"))
                    title = stripped.lstrip("#").strip()
                    outline.append({"line": idx, "level": level, "title": title})

        return outline, {"headers_count": len(outline)}

    def _generate_report(self, args: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        title = args["title"]
        sections = args.get("sections") or ["项目概述", "工作进展", "指标数据与分析", "下一步规划"]
        out_rel = args.get("output_path") or f"report_{int(time.time())}.md"
        out_full = os.path.abspath(os.path.join(self.workdir, out_rel))

        # 快照以便撤销
        self._snapshot(out_full, out_full)

        lines = [f"# {title}", "", f"> 自动生成于 {time.strftime('%Y-%m-%d %H:%M:%S')}", ""]
        for sec in sections:
            lines.extend([f"## {sec}", "", "（在此填写相关详细内容）", ""])

        os.makedirs(os.path.dirname(out_full), exist_ok=True)
        with open(out_full, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return out_rel, {"_rollback_available": True, "output_path": out_rel}

    # ------------------------------------------------------------------
    # 待办事项管理
    # ------------------------------------------------------------------
    def _load_todos(self) -> List[Dict[str, Any]]:
        if not os.path.isfile(self._todos_file):
            return []
        try:
            with open(self._todos_file, "r", encoding="utf-8") as f:
                return json.load(f).get("todos", [])
        except Exception:
            return []

    def _save_todos(self, todos: List[Dict[str, Any]]) -> None:
        os.makedirs(os.path.dirname(self._todos_file), exist_ok=True)
        self._snapshot(self._todos_file, self._todos_file)
        with open(self._todos_file, "w", encoding="utf-8") as f:
            json.dump({"updated_at": time.time(), "todos": todos}, f, ensure_ascii=False, indent=2)

    def _todo_add(self, args: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        task = args["task"].strip()
        priority = args.get("priority") or "normal"
        todos = self._load_todos()
        item = {
            "id": f"t_{len(todos) + 1}",
            "task": task,
            "priority": priority,
            "completed": False,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        todos.append(item)
        self._save_todos(todos)
        return item, {"_rollback_available": True, "task_id": item["id"]}

    def _todo_list(self, args: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        status = args.get("status")
        todos = self._load_todos()
        if status == "pending":
            todos = [t for t in todos if not t.get("completed")]
        elif status == "completed":
            todos = [t for t in todos if t.get("completed")]
        return todos, {"count": len(todos)}

    def _todo_complete(self, args: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        kw = str(args["task_id_or_keyword"]).strip().lower()
        todos = self._load_todos()
        matched = None
        for t in todos:
            if t["id"].lower() == kw or kw in t["task"].lower():
                t["completed"] = True
                t["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                matched = t
                break
        if matched is None:
            return None, {"error": f"未找到匹配的待办任务：'{args['task_id_or_keyword']}'"}
        self._save_todos(todos)
        return matched, {"_rollback_available": True}

    def _snapshot(self, key: str, path: str) -> None:
        if os.path.exists(path):
            with open(path, "rb") as fh:
                self._snapshots[key] = ("file", fh.read())
        else:
            self._snapshots[key] = ("absent", None)
        self._snapshots[f"{key}::path"] = path

    def restore(self, key: str) -> bool:
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
