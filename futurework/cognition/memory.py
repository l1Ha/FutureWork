"""
持久化记忆与跨会话情境指代消解层。

人类协同的关键特征之一是跨时间的连续性：
"打开刚刚那个文件"、"继续昨天编辑的那个文档"、"把上一次搜索的结果保存下来"。
如果没有持久化记忆，每次重启应用，用户都必须从头重新说明文件路径和目标。

设计：
- ``MemoryEntity``: 记录历史操作中接触过的实体（文件、网页、窗口、代码片段等），
  记录首次出现时间、最近访问时间、访问频次与上下文标签。
- ``PersistentMemoryStore``: 轻量安全的磁盘持久化存储（存放在工作目录或用户目录的 .futurework/memory.json），
  支持跨进程、跨会话恢复。
- 基于时间衰减 (Ebbinghaus decay) 与显著度加权 (Salience) 的召回算法。
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from futurework.types import GroundingTarget


@dataclass
class MemoryEntity:
    """持久化记忆中的实体节点。"""

    entity_id: str
    target_type: str  # "file", "url", "window", "directory", "command"
    label: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    first_seen: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 1
    salience: float = 1.0  # 初始显著度
    context_tags: List[str] = field(default_factory=list)

    def touch(self, salience_boost: float = 0.2) -> None:
        self.last_accessed = time.time()
        self.access_count += 1
        self.salience = min(2.5, self.salience + salience_boost)

    def current_score(self, now: Optional[float] = None) -> float:
        """
        计算当前激活强度（模拟人类遗忘曲线与访问频次加权）。
        Score = Frequency * Salience * exp(-lambda * delta_t_hours)
        """
        current_ts = now or time.time()
        hours = max(0.0, (current_ts - self.last_accessed) / 3600.0)
        # 半衰期约 24 小时 (lambda ≈ 0.0288)
        decay = math.exp(-0.0288 * hours)
        freq_factor = math.log1p(self.access_count)
        return float(self.salience * freq_factor * decay)

    def to_grounding_target(self) -> GroundingTarget:
        return GroundingTarget(
            target_type=self.target_type,
            target_id=self.entity_id,
            label=self.label,
            metadata={**self.metadata, "last_accessed": self.last_accessed, "score": self.current_score()},
        )


class PersistentMemoryStore:
    """
    跨会话持久化记忆存储库。
    """

    def __init__(self, storage_path: Optional[str] = None, *, max_entities: int = 200) -> None:
        if storage_path:
            self.storage_path = os.path.abspath(storage_path)
        else:
            self.storage_path = os.path.join(os.getcwd(), ".futurework", "memory.json")
        self.max_entities = max_entities
        self._entities: Dict[str, MemoryEntity] = {}
        self.load()

    def load(self) -> bool:
        """从磁盘加载历史记忆。"""
        if not os.path.isfile(self.storage_path):
            return False
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._entities.clear()
            for item in data.get("entities", []):
                entity = MemoryEntity(**item)
                self._entities[entity.entity_id] = entity
            return True
        except Exception:
            # 容错降级：不阻断主流程
            return False

    def save(self) -> bool:
        """保存当前记忆至磁盘文件。"""
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            self._prune()
            payload = {
                "version": "1.0",
                "updated_at": time.time(),
                "entities": [asdict(e) for e in self._entities.values()],
            }
            # 原子写入防崩溃损坏
            temp_path = self.storage_path + ".tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(temp_path, self.storage_path)
            return True
        except Exception:
            return False

    def record_target(
        self,
        target: GroundingTarget,
        *,
        context_tags: Optional[List[str]] = None,
        salience: float = 1.0,
    ) -> MemoryEntity:
        """记录或更新一个被交互过的实体。"""
        key = target.target_id
        tags = context_tags or []
        if key in self._entities:
            entity = self._entities[key]
            entity.touch()
            for t in tags:
                if t not in entity.context_tags:
                    entity.context_tags.append(t)
        else:
            entity = MemoryEntity(
                entity_id=key,
                target_type=target.target_type,
                label=target.label or key,
                metadata=target.metadata,
                salience=salience,
                context_tags=tags,
            )
            self._entities[key] = entity

        self.save()
        return entity

    def record_file(self, file_path: str, action: str = "read") -> MemoryEntity:
        """便捷记录文件操作。"""
        target = GroundingTarget(
            target_type="file",
            target_id=file_path,
            label=os.path.basename(file_path),
            metadata={"action": action},
        )
        return self.record_target(target, context_tags=["file", action])

    def query_recent(
        self,
        *,
        target_type: Optional[str] = None,
        context_tag: Optional[str] = None,
        limit: int = 5,
    ) -> List[MemoryEntity]:
        """按记忆激活强度由高到低返回最相关的实体。"""
        candidates = list(self._entities.values())
        if target_type:
            candidates = [c for c in candidates if c.target_type == target_type]
        if context_tag:
            candidates = [c for c in candidates if context_tag in c.context_tags]

        candidates.sort(key=lambda e: e.current_score(), reverse=True)
        return candidates[:limit]

    def resolve_reference(self, phrase: str, target_type_hint: Optional[str] = None) -> Optional[GroundingTarget]:
        """
        基于自然语言指代词和模糊线索查询最佳匹配实体。
        如: "刚才那个文件", "那个文档", "之前的网页", "这个"
        """
        phrase_clean = phrase.strip().lower()

        # 类型提示推断
        inferred_type = target_type_hint
        if any(w in phrase_clean for w in ("文件", "file", "文档", "doc", "txt", "md", "代码")):
            inferred_type = "file"
        elif any(w in phrase_clean for w in ("网页", "网站", "网址", "url", "web")):
            inferred_type = "url"
        elif any(w in phrase_clean for w in ("窗口", "window")):
            inferred_type = "window"

        candidates = self.query_recent(target_type=inferred_type, limit=10)
        if not candidates:
            candidates = self.query_recent(limit=10)

        if not candidates:
            return None

        # 1. 尝试名字子串匹配
        for c in candidates:
            clean_label = c.label.lower()
            if clean_label and clean_label in phrase_clean:
                return c.to_grounding_target()

        # 2. 如果指代词是通用代词（如"刚刚那个"、"那个"、"上一个"、"这个"），返回当前分最高的最近期项
        generic_markers = ("刚才", "刚刚", "上一个", "之前的", "上一次", "那个", "这个", "it", "that", "recent")
        if any(m in phrase_clean for m in generic_markers) or len(phrase_clean) <= 4:
            return candidates[0].to_grounding_target()

        return None

    def _prune(self) -> None:
        """超出上限时剔除得分最低的陈旧条目。"""
        if len(self._entities) <= self.max_entities:
            return
        sorted_entities = sorted(self._entities.values(), key=lambda e: e.current_score())
        excess = len(self._entities) - self.max_entities
        to_remove = [e.entity_id for e in sorted_entities[:excess]]
        for k in to_remove:
            self._entities.pop(k, None)

    def clear(self) -> None:
        """清空记忆。"""
        self._entities.clear()
        if os.path.exists(self.storage_path):
            try:
                os.remove(self.storage_path)
            except OSError:
                pass
