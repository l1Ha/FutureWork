"""
意图解析层：自然语言 + 多模态线索 → 结构化 Intent。

采用"规则 + 语义相似度"的双通道设计，而非把一切都交给大模型：
* **规则通道**保证确定性与可审计性（安全关键操作必须可预测）；
* **语义通道**保证泛化能力（没背过的新说法也能懂）。

两者加权融合；当最高分低于阈值时**不猜**，而是交给对话层追问。
这是自然交互的关键：人不会因为对方猜错而愤怒，但会因为对方瞎猜而愤怒。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Pattern, Sequence, Tuple

from futurework.sensory.fusion import FusedSignal
from futurework.types import (
    EmotionType,
    GestureType,
    GroundingTarget,
    Intent,
    IntentCategory,
    ModalityType,
)


@dataclass
class IntentRule:
    """
    单条意图规则：正则触发 + 槽位提取。

    ``priority`` 越大越先匹配（"删除文件"须先于通用的"删除"命中）。
    """

    category: IntentCategory
    action: str
    patterns: Sequence[str]
    tool_hint: str = ""
    priority: int = 0
    requires_confirmation: bool = False
    safety_hint: str = ""
    slot_patterns: Dict[str, str] = field(default_factory=dict)
    compiled: List[Pattern[str]] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self.compiled = [re.compile(p, re.IGNORECASE) for p in self.patterns]

    def match(self, text: str) -> Optional[Dict[str, str]]:
        for pattern in self.compiled:
            m = pattern.search(text)
            if not m:
                continue
            # 正则里的具名分组优先——这是声明槽位最直观的方式
            slots: Dict[str, str] = {
                name: (value.strip() if isinstance(value, str) else value)
                for name, value in (m.groupdict() or {}).items()
                if value not in (None, "")
            }
            # 未被具名分组覆盖的槽位，用独立正则补充提取
            for slot, slot_re in self.slot_patterns.items():
                if slot in slots:
                    continue
                sm = re.search(slot_re, text, re.IGNORECASE)
                if sm:
                    slots[slot] = sm.group(1) if sm.groups() else sm.group(0)
            return slots
        return None


# 常见顶级域白名单。用它代替 ``[a-z]{2,}`` 是必要的：
# 否则 "open notes.txt" 会被当成网址——任何带后缀的文件名都符合通用域名形态。
_TLD = (
    "com|org|net|edu|gov|int|mil|io|ai|dev|app|co|me|info|biz|tv|cc|xyz|top|site|tech|"
    "online|store|blog|wiki|news|cloud|cn|uk|jp|hk|tw|de|fr|ru|kr|in|br|au|ca|eu|nl|se|no"
)


def get_default_rules() -> List[IntentRule]:
    """
    默认意图规则库（中英双语）。

    覆盖：导航/窗口/文档/代码/浏览器/版本控制/系统控制/确认/拒绝/撤销/查询。

    两处关键设计：
    * **动词写全**：Python 的 ``\\w`` 匹配中日韩字符，若动词写成 ``读``，
      ``读取 a.txt`` 会把"取"当成文件名。动词一律写成完整词。
    * **优先级即消歧**：``查看 git 状态`` 里的"查看"既是 read_file 又是 git_status，
      靠 priority 让更具体的 git 规则先命中。
    """
    return [
        # ---------------- 对话控制（最高优先级，纯语义无歧义） ----------------
        IntentRule(
            category=IntentCategory.CONFIRMATION,
            action="confirm",
            priority=200,
            patterns=[r"^(?:是|对|好|好的|行|嗯|确认|确定|可以|没错|同意|就这样|执行|继续|yes|ok|okay|sure|confirm|do it|go ahead|proceed)[。.!！,\s]*$"],
        ),
        IntentRule(
            category=IntentCategory.REJECTION,
            action="reject",
            priority=200,
            patterns=[r"^(?:不|不是|不用|不要|别|不行|no|nope|don'?t)[。.!！,\s]*$"],
        ),
        IntentRule(
            category=IntentCategory.CANCEL_UNDO,
            action="cancel",
            priority=199,
            patterns=[r"^(?:取消|算了|不用了|停下|停止|stop|cancel|never ?mind)[。.!！,\s]*$"],
        ),
        IntentRule(
            category=IntentCategory.CANCEL_UNDO,
            action="undo",
            priority=199,
            requires_confirmation=True,
            patterns=[r"^(?:撤销|撤回|回退上一步|还原|撤销上一步|undo|revert|roll ?back)[。.!！,\s]*$"],
        ),
        # ---------------- 版本控制（需高于通用"查看"规则） ----------------
        IntentRule(
            category=IntentCategory.CODE_DEV,
            action="git_status",
            tool_hint="vcs",
            priority=180,
            patterns=[
                r"(?:查看|看看|显示|查)\s*(?:一下)?\s*(?:git|仓库)(?:\s*(?:状态|状况|status))?",
                r"^(?:git|repo)\s*status$",
                r"^(?:status|状态)$",
            ],
        ),
        IntentRule(
            category=IntentCategory.CODE_DEV,
            action="git_diff",
            tool_hint="vcs",
            priority=180,
            patterns=[
                r"(?:查看|看看|显示|看)\s*(?:一下)?\s*(?:git)?\s*(?:改动|差异|diff|变更)",
                r"git\s*diff",
            ],
        ),
        IntentRule(
            category=IntentCategory.CODE_DEV,
            action="git_log",
            tool_hint="vcs",
            priority=180,
            patterns=[
                r"(?:查看|看看|显示|看)\s*(?:一下)?\s*(?:git)?\s*(?:提交历史|历史记录|提交记录|log)",
                r"git\s*log",
            ],
        ),
        IntentRule(
            category=IntentCategory.CODE_DEV,
            action="git_commit",
            tool_hint="vcs",
            priority=178,
            patterns=[
                r"(?:提交|commit)(?:代码|改动|文件)?\s*(?:信息|消息)?\s*[:：]?\s*(?P<message>.+)",
                r"^commit\s+(?P<message>.+)",
            ],
        ),
        IntentRule(
            category=IntentCategory.CODE_DEV,
            action="git_add",
            tool_hint="vcs",
            priority=178,
            patterns=[r"^(?:暂存|add)(?:\s+(?P<paths>.+))?$"],
        ),
        # ---------------- 测试 ----------------
        IntentRule(
            category=IntentCategory.CODE_DEV,
            action="run_tests",
            tool_hint="terminal",
            priority=175,
            patterns=[
                r"^(?:跑|运行|执行)(?:一下|下)?\s*(?:所有)?\s*测试(?:用例|套件)?[。.!！,\s]*$",
                r"^run\s+(?:the\s+|all\s+)?tests?[。.!！,\s]*$",
            ],
        ),
        # ---------------- 文件操作（破坏性最高） ----------------
        IntentRule(
            category=IntentCategory.DOCUMENT_EDIT,
            action="delete_file",
            tool_hint="filesystem",
            priority=170,
            requires_confirmation=True,
            safety_hint="destructive",
            patterns=[
                r"^(?:删除|删掉|删|清除|移除)\s*(?:文件)?\s*(?P<path>\S{1,200})",
                r"^(?:delete|remove|rm)\s+(?:the\s+)?(?:file\s+)?(?P<path>\S{1,200})$",
            ],
        ),
        IntentRule(
            category=IntentCategory.DOCUMENT_EDIT,
            action="write_file",
            tool_hint="filesystem",
            priority=165,
            patterns=[
                r"^(?:把|将)?\s*(?P<content>.+?)\s*(?:写入|保存到|写到|存到)\s*(?:文件)?\s*(?P<path>\S{1,200})",
                r"^write\s+(?P<content>.+?)\s+to\s+(?:file\s+)?(?P<path>\S{1,200})",
            ],
        ),
        IntentRule(
            category=IntentCategory.DOCUMENT_EDIT,
            action="append_file",
            tool_hint="filesystem",
            priority=165,
            patterns=[
                r"^(?:追加|附加)\s*(?P<content>.+?)\s*(?:到|至)\s*(?:文件)?\s*(?P<path>\S{1,200})",
                r"^append\s+(?P<content>.+?)\s+to\s+(?P<path>\S{1,200})",
            ],
        ),
        # ---------------- 生产力与办公数据套件 (表格/报告/待办) ----------------
        IntentRule(
            category=IntentCategory.DOCUMENT_EDIT,
            action="table_aggregate",
            tool_hint="productivity",
            priority=172,
            patterns=[
                r"^(?:计算|统计|求)\s*(?P<path>\S+\.(?:csv|tsv))\s*(?:的)?\s*(?P<column>\S+?)\s*(?:列)?\s*(?:的)?\s*(?P<operation>sum|mean|max|min|count|和|平均值|最大值|最小值|行数)$",
                r"^(?:对|在)\s*(?P<path>\S+\.(?:csv|tsv))\s*(?:的)?\s*(?P<column>\S+?)\s*(?:列)?\s*(?:求|计算)\s*(?P<operation>sum|mean|max|min|count|和|平均值|最大值|最小值|行数)$",
            ],
        ),
        IntentRule(
            category=IntentCategory.DOCUMENT_EDIT,
            action="generate_report",
            tool_hint="productivity",
            priority=171,
            patterns=[
                r"^(?:生成|创建|制作)(?:一份)?\s*(?:关于)?\s*(?P<title>\S+?)\s*(?:的)?\s*(?:工作报告|报告|周报|文档)$",
                r"^(?:generate|create)\s+(?:a\s+)?report\s+(?:titled\s+)?(?P<title>\S.*)$",
            ],
        ),
        IntentRule(
            category=IntentCategory.DOCUMENT_EDIT,
            action="extract_outline",
            tool_hint="productivity",
            priority=169,
            patterns=[
                r"^(?:提取|查看|显示|输出)\s*(?P<path>\S+?)\s*(?:的)?\s*(?:大纲|目录结构|结构)$",
                r"^(?:extract|show)\s+(?:the\s+)?outline\s+of\s+(?P<path>\S+)$",
            ],
        ),
        IntentRule(
            category=IntentCategory.DOCUMENT_EDIT,
            action="todo_add",
            tool_hint="productivity",
            priority=168,
            patterns=[
                r"^(?:添加|新建|记下|记录|增加)\s*(?:待办|任务|todo)[:：\s]\s*(?P<task>\S.*)$",
                r"^(?:todo|add todo|todo add)\s+(?P<task>\S.*)$",
            ],
        ),
        IntentRule(
            category=IntentCategory.DOCUMENT_EDIT,
            action="todo_list",
            tool_hint="productivity",
            priority=168,
            patterns=[
                r"^(?:查看|列出|显示|看看)\s*(?:所有)?\s*(?:待办|任务|todos?|todo 清单)$",
                r"^(?:list|show)\s+todos?$",
            ],
        ),
        IntentRule(
            category=IntentCategory.DOCUMENT_EDIT,
            action="todo_complete",
            tool_hint="productivity",
            priority=168,
            patterns=[
                r"^(?:完成|搞定|勾掉|结项)\s*(?:待办|任务)?\s*(?P<task_id_or_keyword>\S.*)$",
                r"^(?:complete|done|finish)\s+todo\s+(?P<task_id_or_keyword>\S.*)$",
            ],
        ),
        IntentRule(
            category=IntentCategory.DOCUMENT_EDIT,
            action="copy_file",
            tool_hint="filesystem",
            priority=166,
            patterns=[
                r"^(?:把)?\s*(?P<source>\S+)\s*(?:复制|拷贝|复刻)(?:一份)?\s*(?:到|至)\s*(?:文件)?\s*(?P<destination>\S+)$",
                r"^(?:copy|duplicate)\s+(?P<source>\S+)\s+(?:to|into)\s+(?P<destination>\S+)$",
            ],
        ),
        IntentRule(
            category=IntentCategory.DOCUMENT_EDIT,
            action="move_file",
            tool_hint="filesystem",
            priority=167,
            patterns=[
                r"^(?:把)?\s*(?P<source>\S+)\s*(?:移动|挪|搬)(?:到|至)\s*(?:文件)?\s*(?P<destination>\S+)$",
                r"^(?:move|rename)\s+(?P<source>\S+)\s+(?:to|into)\s+(?P<destination>\S+)$",
            ],
        ),
        IntentRule(
            category=IntentCategory.DOCUMENT_EDIT,
            action="search_in_file",
            tool_hint="filesystem",
            priority=160,
            patterns=[
                r"^(?:在)?\s*(?P<path>\S{1,200})\s*(?:里面|里面|里|中|内)?\s*(?:查找|搜索|搜|找)\s*(?P<keyword>\S.*)$",
                r"^search\s+(?P<keyword>.+?)\s+in\s+(?P<path>\S{1,200})$",
            ],
        ),
        IntentRule(
            category=IntentCategory.DOCUMENT_EDIT,
            action="read_file",
            tool_hint="filesystem",
            priority=155,
            patterns=[
                r"^(?:读取|读出|读一下|读)\s+(?:文件\s*)?(?P<path>\S{1,200})$",
                r"^(?:读取|读出|读)(?P<path>[A-Za-z0-9._/~\\-]\S{0,199})$",
                r"^(?:读取|读出|读一下|读|打开|查看)\s*(?:一下)?\s*(?:文件\s*)?(?P<path>(?:刚才|刚刚|之前|上一个|这个|那个)\S{0,20})$",
                r"^(?:打开文件|查看文件)\s*(?P<path>\S{1,200})$",
                # "把 TODO.md 看一下" —— 口语里把动作放在对象之后
                r"^(?:把|将)?\s*(?P<path>\S+)\s*(?:看一下|看一眼|看看|瞧一下)$",
                r"^(?:read|view|cat)\s+(?:the\s+)?(?:file\s+)?(?P<path>\S*[./]\S{1,200}|\S+\.[a-z0-9]{1,10})$",
            ],
        ),
        IntentRule(
            category=IntentCategory.DOCUMENT_EDIT,
            action="create_directory",
            tool_hint="filesystem",
            priority=152,
            patterns=[r"^(?:创建|新建|建立)\s*(?:目录|文件夹)\s*(?P<path>\S{1,200})", r"^mkdir\s+(?P<path>\S{1,200})$"],
        ),
        IntentRule(
            category=IntentCategory.DOCUMENT_EDIT,
            action="list_directory",
            tool_hint="filesystem",
            priority=150,
            patterns=[
                r"^(?:列出|看看|显示|查看|浏览)\s*(?:一下)?\s*(?:目录|文件夹)\s*(?P<path>\S{0,200})?$",
                r"^(?:list|show)\s+(?:directory|folder)\s*(?P<path>\S{0,200})?$",
            ],
        ),
        # ---------------- 窗口管理 ----------------
        IntentRule(
            category=IntentCategory.WINDOW_MANAGEMENT,
            action="close_window",
            tool_hint="system",
            priority=145,
            requires_confirmation=True,
            patterns=[
                r"^(?:关闭|关掉)\s*(?:这个|当前|该)?\s*窗口$",
                r"^close\s+(?:this|the current)?\s*window$",
            ],
        ),
        IntentRule(
            category=IntentCategory.WINDOW_MANAGEMENT,
            action="minimize_all",
            tool_hint="system",
            priority=144,
            patterns=[r"^(?:最小化|收起)(?:全部|所有)(?:的)?(?:窗口)?$", r"^minimize\s+all$"],
        ),
        IntentRule(
            category=IntentCategory.WINDOW_MANAGEMENT,
            action="maximize_window",
            tool_hint="system",
            priority=143,
            patterns=[r"^(?:最大化|铺满|全屏)\s*(?:窗口)?$", r"^maximize\s+(?:the\s+)?window$"],
        ),
        IntentRule(
            category=IntentCategory.WINDOW_MANAGEMENT,
            action="list_windows",
            tool_hint="system",
            priority=140,
            patterns=[
                r"^(?:列出|显示|看看|查看)\s*(?:一下)?\s*(?:所有)?\s*窗口$",
                r"^(?:list|show)\s+(?:all\s+)?windows$",
            ],
        ),
        IntentRule(
            category=IntentCategory.WINDOW_MANAGEMENT,
            action="focus_window",
            tool_hint="system",
            priority=138,
            patterns=[
                r"^(?:切换|切到|跳到|激活)\s*(?:到)?\s*(?P<name>.+?)\s*窗口$",
                r"^focus\s+(?:on\s+)?(?:the\s+)?(?P<name>.+?)\s+window$",
            ],
        ),
        # ---------------- 浏览器（URL 形态明确，需高于"打开应用"） ----------------
        IntentRule(
            category=IntentCategory.BROWSER_ACTION,
            action="open_url",
            tool_hint="browser",
            priority=135,
            patterns=[
                rf"^(?:打开|访问|浏览)\s*(?:网址|网站|页面|链接)?\s*(?P<url>(?:https?://)?[a-z0-9][\w-]*(?:\.[\w-]+)*\.(?:{_TLD})(?:/\S*)?|localhost(?::\d+)?(?:\S*)?)\s*$",
                rf"^(?:open|visit|browse|go to)\s+(?P<url>(?:https?://)?[a-z0-9][\w-]*(?:\.[\w-]+)*\.(?:{_TLD})(?:/\S*)?|localhost(?::\d+)?(?:\S*)?)\s*$",
            ],
        ),
        IntentRule(
            category=IntentCategory.BROWSER_ACTION,
            action="browser_back",
            tool_hint="browser",
            priority=134,
            patterns=[r"^(?:后退|返回上一页|回退|上一页)$", r"^(?:go|navigate)\s+back$"],
        ),
        IntentRule(
            category=IntentCategory.BROWSER_ACTION,
            action="search_web",
            tool_hint="browser",
            priority=133,
            patterns=[
                r"^(?:搜索|搜一下|查一下|查询|检索)\s*(?:网页|网上|一下)?\s*(?P<query>\S.*)$",
                r"^(?:search|google|look up)\s+(?:the web\s+)?(?:for\s+)?(?P<query>\S.*)$",
            ],
        ),
        # ---------------- 应用启动（兜底：放在窗口/URL 之后） ----------------
        IntentRule(
            category=IntentCategory.SYSTEM_CONTROL,
            action="launch_app",
            tool_hint="system",
            priority=100,
            patterns=[
                r"^(?:打开|启动|运行|开)\s*(?:一下)?\s*(?:应用|程序|app)?\s*(?P<app>.{1,40})$",
                r"^(?:launch|open|start)\s+(?:the\s+)?(?:app(?:lication)?\s+)?(?P<app>[\w ._+-]{1,40})$",
            ],
        ),
        # ---------------- 终端命令（高于"启动应用"：运行 X 更像命令而非应用名） ----------------
        IntentRule(
            category=IntentCategory.CODE_DEV,
            action="run_command",
            tool_hint="terminal",
            priority=120,
            patterns=[
                r"^(?:在)?(?:终端|命令行|shell|terminal)(?:里|中)?\s*(?:运行|执行|跑)\s+(?P<command>\S.*)$",
                r"^run\s+(?:the\s+)?command\s+(?P<command>\S.*)$",
                r"^(?:运行|执行)\s+(?P<command>\S.*)$",
            ],
        ),
        # ---------------- 应用启动（兜底：放在窗口/URL 之后） ----------------
        IntentRule(
            category=IntentCategory.SYSTEM_CONTROL,
            action="launch_app",
            tool_hint="system",
            priority=100,
            patterns=[
                r"^(?:打开|启动|运行|开)\s*(?:一下)?\s*(?:应用|程序|app)\s*(?P<app>.{1,40})$",
                r"^(?:打开|启动|开)\s*(?P<app>[\w一-鿿 ._+-]{1,40})$",
                r"^(?:launch|open|start)\s+(?:the\s+)?(?:app(?:lication)?\s+)?(?P<app>[\w ._+-]{1,40})$",
            ],
        ),
        IntentRule(
            category=IntentCategory.CODE_DEV,
            action="open_file_editor",
            tool_hint="editor",
            priority=90,
            patterns=[
                r"^(?:编辑|用编辑器打开)\s*(?:文件|代码)?\s*(?P<path>\S{1,200})$",
                r"^edit\s+(?:file\s+)?(?P<path>\S{1,200})$",
            ],
        ),
        # ---------------- 查询 ----------------
        IntentRule(
            category=IntentCategory.QUERY_EXPLAIN,
            action="explain",
            tool_hint="",
            priority=40,
            patterns=[r"^(?:这是什么|什么意思|解释一下|为什么|怎么回事)", r"^(?:explain|why|what (?:does|is))\b"],
        ),
    ]


# 裸动词：用户只说了动作、还没说对象。
# "读取"单独一句是完全自然的话——它不是"听不懂"，而是一个待补全的意图。
# 若按未知指令处理，系统会回答"我没太明白"，把人推回从头再说一遍；
# 正确做法是认出这个动作，然后只追问缺的那一个槽位。
_BARE_VERB_ACTIONS = {
    "读取": "read_file", "读一下": "read_file", "读出": "read_file", "读": "read_file",
    "打开文件": "read_file", "查看文件": "read_file",
    "打开": "launch_app", "启动": "launch_app", "运行应用": "launch_app",
    "删除": "delete_file", "删掉": "delete_file", "移除": "delete_file",
    "重命名": "move_file", "移动": "move_file",
    "复制": "copy_file", "拷贝": "copy_file",
    "创建目录": "create_directory", "新建目录": "create_directory",
    "写入": "write_file", "保存": "write_file",
    "查找": "search_in_file", "搜索": "search_web",
    "关闭": "close_window", "最小化": "minimize_all", "最大化": "maximize_window",
    "切换": "focus_window", "跑测试": "run_tests",
    "read": "read_file", "open": "launch_app", "delete": "delete_file",
    "copy": "copy_file", "move": "move_file", "run": "run_command",
    "search": "search_web", "close": "close_window", "undo": "undo",
}


class IntentParser:
    """
    意图解析器：规则 + 语义通道 + 多模态修饰。

    ``parse`` 接受融合信号与原始文本，输出唯一 ``Intent``；
    若置信度不足，返回 ``IntentCategory.UNKNOWN`` 并附带
    ``clarification_hint``，由对话层决定如何追问。
    """

    MIN_CONFIDENCE = 0.45

    def __init__(
        self,
        rules: Optional[Sequence[IntentRule]] = None,
        *,
        semantic_threshold: float = 0.45,
    ) -> None:
        self.rules = sorted(rules if rules is not None else get_default_rules(), key=lambda r: -r.priority)
        self.semantic_threshold = semantic_threshold
        # 可插拔的语义编码器：提供后自动启用软匹配通道
        self._encoder: Optional[Callable[[str], List[float]]] = None
        self._intent_texts: Dict[str, str] = {}
        self._embeddings: Dict[str, Tuple[IntentCategory, str, List[float]]] = {}

    def set_encoder(self, encoder: Callable[[str], List[float]]) -> None:
        """注入句向量编码器（如 sentence-transformers / API），启用语义通道。"""
        self._encoder = encoder
        if encoder is None:
            self._intent_texts.clear()
            self._embeddings.clear()
            return
        seen: Dict[str, str] = {}
        for rule in self.rules:
            if rule.action not in seen:
                exemplar = _exemplar_for(rule)
                seen[rule.action] = exemplar
                try:
                    self._embeddings[rule.action] = (rule.category, rule.action, encoder(exemplar))
                except Exception:
                    continue
        self._intent_texts = seen

    # ------------------------------------------------------------------
    def parse(self, text: str, fused: Optional[FusedSignal] = None) -> Intent:
        normalized = _normalize(text)
        matched: List[Tuple[IntentRule, Dict[str, str], float]] = []

        for rule in self.rules:
            slots = rule.match(normalized)
            if slots is not None:
                matched.append((rule, slots, 1.0))
        matched.sort(key=lambda item: item[0].priority, reverse=True)

        semantic = self._semantic_match(normalized)

        # 安全优先：张开手掌是"停"，无论内容是否听懂都必须生效。
        # 若把它放在"未命中规则"的分支之后，一次停止手势会被当成"没听懂"而丢弃，
        # 而它恰恰是最不该被丢弃的信号。
        gesture = _gesture_override(fused) if fused is not None else None
        if gesture == "stop":
            return Intent(
                category=IntentCategory.CANCEL_UNDO,
                action="stop",
                confidence=0.97,
                source_modalities=fused.modalities_used if fused else [],
                grounded_target=fused.target if fused else None,
                ambiguity_score=0.0,
                requires_confirmation=False,
                natural_explanation="检测到停止手势，立即中止当前操作",
            )

        if matched:
            rule, slots, base = matched[0]
            confidence = base
            if fused is not None:
                confidence *= _modality_factor(fused)
                if fused.conflicting_modalities:
                    confidence *= 0.6
            category, action = rule.category, rule.action
            params = dict(slots)
            requires_conf = rule.requires_confirmation
            explanation = f"命中规则「{rule.action}」"
        elif semantic is not None:
            category, action, score = semantic
            rule = next((r for r in self.rules if r.action == action), None)
            confidence = score
            if fused is not None:
                confidence *= _modality_factor(fused)
            params = {}
            requires_conf = rule.requires_confirmation if rule else (score < 0.6)
            explanation = f"语义匹配「{action}」({score:.2f})"
        else:
            # 规则未命中，但可能只是一个"只说了动作"���半成品指令
            incomplete = self.detect_incomplete(normalized)
            if incomplete is not None:
                if fused is not None:
                    incomplete.source_modalities = fused.modalities_used
                    incomplete.grounded_target = fused.target
                incomplete.ambiguity_score = 0.5
                return incomplete
            return Intent(
                category=IntentCategory.UNKNOWN,
                action="unknown",
                confidence=0.0,
                source_modalities=fused.modalities_used if fused else [],
                grounded_target=fused.target if fused else None,
                ambiguity_score=1.0 if fused is None else fused.ambiguity_score,
                requires_confirmation=True,
                natural_explanation="未能识别意图，需要澄清",
            )

        ambiguity = fused.ambiguity_score if fused else 0.0
        if ambiguity >= 0.4:
            requires_conf = True

        # 多模态修饰：手势可直接把动作改写为确认/撤销
        # （"stop" 已在本方法开头单独处理——它必须先于规则匹配生效）
        if gesture and gesture != "stop" and category not in (
            IntentCategory.CONFIRMATION,
            IntentCategory.REJECTION,
            IntentCategory.CANCEL_UNDO,
        ):
            explanation += f"；手势修正为「{gesture}」"
            action = gesture
            if gesture in ("confirm", "approve"):
                category = IntentCategory.CONFIRMATION

        return Intent(
            category=category,
            action=action,
            parameters=params,
            confidence=round(min(1.0, confidence), 4),
            source_modalities=fused.modalities_used if fused else [],
            grounded_target=fused.target if fused else None,
            ambiguity_score=ambiguity,
            requires_confirmation=requires_conf,
            natural_explanation=explanation,
        )

    def detect_incomplete(self, text: str) -> Optional[Intent]:
        """
        识别"只说了动作、没说对象"的半成品指令。

        >>> parser.detect_incomplete("读取").action
        'read_file'

        :returns: 识别成功返回缺少槽位的意图，否则返回 ``None``。
        """
        normalized = _normalize(text).strip(" 　。.!！?？")
        if not normalized:
            return None
        # 允许"读取那个""把那个读取一下"这类省略说法
        candidates = [normalized]
        stripped = re.sub(r"^(?:把|将)?\s*(?:那个|这个|它)\s*", "", normalized)
        stripped = re.sub(r"\s*(?:一下|一次)$", "", stripped).strip()
        if stripped and stripped != normalized:
            candidates.append(stripped)

        for candidate in candidates:
            action = _BARE_VERB_ACTIONS.get(candidate) or _BARE_VERB_ACTIONS.get(candidate.lower())
            if not action:
                continue
            rule = next((r for r in self.rules if r.action == action), None)
            return Intent(
                category=rule.category if rule else IntentCategory.UNKNOWN,
                action=action,
                parameters={},
                confidence=0.6,
                requires_confirmation=False,
                natural_explanation=f"识别到动作「{action}」，等待补充对象",
            )
        return None

    # ------------------------------------------------------------------
    def _semantic_match(self, text: str) -> Optional[Tuple[IntentCategory, str, float]]:
        if self._encoder is None or not self._embeddings or not text:
            return None
        try:
            query = self._encoder(text)
        except Exception:
            return None
        best: Optional[Tuple[IntentCategory, str, float]] = None
        for category, action, vector in self._embeddings.values():
            score = _cosine(query, vector)
            if best is None or score > best[2]:
                best = (category, action, score)
        if best and best[2] >= self.semantic_threshold:
            return best
        return None


def _normalize(text: str) -> str:
    """
    文本归一化：全角转半角、压缩空白、统一标点。

    **刻意保留大小写**：文件名在 Linux/macOS 上是大小写敏感的，
    把 ``TODO.md`` 归一化成 ``todo.md`` 会让系统"找不到明明存在的文件"。
    关键字匹配不依赖这里的小写化——所有规则都以 ``re.IGNORECASE`` 编译。
    """
    if not text:
        return ""
    out = []
    for ch in text.strip():
        code = ord(ch)
        if code == 0x3000:
            out.append(" ")
        elif 0xFF01 <= code <= 0xFF5E:
            out.append(chr(code - 0xFEE0))
        else:
            out.append(ch)
    return re.sub(r"\s+", " ", "".join(out)).strip()


def _modality_factor(fused: FusedSignal) -> float:
    """
    多模态一致性因子。

    语音单独出现（只有一种模态）时不做惩罚——人本来就是边打字边说话，
    关键是**不能有冲突**；有协同则加成。
    """
    factor = 1.0
    if fused.conflicting_modalities:
        factor *= 0.5
    if len(fused.agreeing_modalities) >= 2:
        factor *= 1.15
    if fused.affective_state in (EmotionType.CONFUSED, EmotionType.FRUSTRATED):
        factor *= 0.85
    return factor


def _gesture_override(fused: FusedSignal) -> Optional[str]:
    """
    从已融合信号推断手势修正。

    融合引擎未直接暴露手势枚举，这里通过模态存在性 + 理由文本反推，
    保持 fusion → cognition 的单向数据流（不跨层直接访问原始信号）。
    """
    if ModalityType.GESTURE not in fused.modalities_used:
        return None
    rationale = " ".join(fused.rationale)
    if "安全停止" in rationale:
        return "stop"
    if "捏合" in rationale and "一致" in rationale:
        return "confirm"
    if "握拳" in rationale:
        return "undo"
    return None


def _exemplar_for(rule: IntentRule) -> str:
    """为语义通道构造该意图类的代表句。"""
    samples = {
        "focus_window": "切换到浏览器窗口 focus on the browser window",
        "close_window": "关闭这个窗口 close this window",
        "minimize_all": "最小化全部窗口 minimize all",
        "maximize_window": "最大化窗口 maximize the window",
        "list_windows": "列出所有窗口 list all windows",
        "launch_app": "打开应用程序 launch the application",
        "delete_file": "删除文件 delete file",
        "read_file": "读取文件 read file",
        "write_file": "写入文件内容 write content to file",
        "append_file": "追加内容到文件 append content to file",
        "search_in_file": "在文件中查找关键词 search keyword in file",
        "list_directory": "列出目录内容 list directory",
        "run_command": "在终端运行命令 run command in terminal",
        "run_tests": "运行测试 run the tests",
        "open_file_editor": "打开代码文件 open the source file",
        "git_commit": "提交代码 commit the code",
        "open_url": "打开网址 open the url",
        "browser_back": "浏览器后退 go back",
        "search_web": "在网页上搜索 search the web",
        "confirm": "确认执行 confirm yes",
        "reject": "拒绝不要 no reject",
        "cancel": "取消操作 cancel stop",
        "undo": "撤销上一步 undo",
        "copy_file": "复制文件 copy file",
        "move_file": "移动文件 move file",
        "table_aggregate": "统计表格数据 aggregate table",
        "table_query": "查询表格数据 query table",
        "generate_report": "生成工作报告 generate report",
        "extract_outline": "提取文档大纲 extract outline",
        "todo_add": "添加待办事项 add todo",
        "todo_list": "查看所有待办 list todos",
        "todo_complete": "完成待办任务 complete todo",
        "explain": "解释一下这是什么 explain what this is",
    }
    return samples.get(rule.action, rule.action.replace("_", " "))


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)