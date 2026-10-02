"""
复合指令拆分与纠正识别。

这是"像人说话"与"像机器执行"之间最显著的一段差距。
人几乎从不一次只下一个指令：

    "把 a.txt 复制到 b.txt，然后删除 b.txt"      —— 一次说两件事，且第二件依赖第一件
    "打开浏览器并搜索天气"                        —— 复合动作
    "先跑测试，失败了就把日志发我"                —— 带条件的复合

机器要求一次一件事，于是用户要么��成两次交互（打断流畅），
要么以为系统只听懂了前半句（更糟——它真的只执行了前半句）。
本模块提供拆分能力，让一句话能变成一串有序动作。

纠正流程同理。人做完一个动作后说"不对""不是这个"是极其自然的表达，
系统必须理解这是在否定刚才那一步，而不是把它当成一个新指令。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# 纠正信号
# ---------------------------------------------------------------------------
# 必须整句或紧跟短尾触发。"不对劲"里的"对"不该被当成肯定。
# 带更正内容：整句被匹配，但更正部分由捕获组给出。
# 用捕获组而非"删掉匹配项看剩下什么"——后者会把整句一并删空。
_CORRECTION_PATTERNS = (
    (r"^(?:不对|不是|说错了|搞错了)\s*[，,]?\s*(?:应该|其实)?\s*(?:是|要)\s*(?P<fix>\S.*)$", True),
    (r"^(?:应该|其实)\s*[，,]?\s*(?:是|要)\s*(?P<fix>\S.*)$", True),
    # "不对，读取 b.txt" —— 否定之后直接给出替代动作。
    # 必须排在上面那条之后："不对，是 b.txt" 里的"是"属于更正标记，
    # 不能被当成替代指令的开头，否则会把"是 b.txt"整段当成新指令。
    (r"^(?:不对|不是|说错了|搞错了)\s*[，,]\s*(?P<fix>\S.*)$", True),
    (r"^(?:不对|不是这个|不是那个|不是我要的|说错了|错了|搞错了)\s*[。.!！]*$", False),
    (r"^(?:no|no that's wrong|wrong|incorrect|not what i (?:meant|wanted))\s*[.!]*$", False),
    (r"^(?:i )?meant\s+(?P<fix>\S.*)$", True),
)

# 纯否定：用户在对当前状态表态，但没有给出更正内容
_PLAIN_DENIAL_RE = re.compile(
    r"^(?:不对|不行|错|不是|不好|错啦|no|nope)\s*[。.!！]*$", re.IGNORECASE
)


@dataclass
class Correction:
    """一次纠正信号。"""

    text: str
    replacement: Optional[str] = None   # "应该是 b.txt" 里的 "b.txt"
    is_rewrite: bool = False            # 是否给出了更正后的说法

    def describe(self) -> str:
        if self.is_rewrite and self.replacement:
            return f"明白，改成「{self.replacement}」"
        return "明白，那先不动。你想怎么做？"


def detect_correction(text: str) -> Optional[Correction]:
    """
    判断这句话是否在否定/更正刚才的操作。

    区分两种情况：

    * **纯否定**（"不对"）——停手，让用户重新说；
    * **带更正**（"不对，是 b.txt"）——停手并直接采纳新说法。

    :returns: 非纠正语句返回 ``None``。
    """
    if not text:
        return None
    stripped = text.strip()

    for pattern, is_rewrite in _CORRECTION_PATTERNS:
        m = re.match(pattern, stripped, re.IGNORECASE)
        if not m:
            continue
        if is_rewrite:
            fix = (m.groupdict().get("fix") or "").strip(" 。.!！")
            return Correction(text=stripped, replacement=fix or None, is_rewrite=bool(fix))
        return Correction(text=stripped)

    if _PLAIN_DENIAL_RE.match(stripped):
        return Correction(text=stripped)

    return None


# ---------------------------------------------------------------------------
# 复合指令拆分
# ---------------------------------------------------------------------------
# 连接词。刻意保守：只用那些几乎不可能出现在路径或文件名里的词。
# "再"/"并" 单字风险太高（"合并"、"再生文件"），不纳入。
_CONJUNCTIONS = (
    r"然后", r"接着", r"随后", r"之后", r"紧接着",
    r"并且", r"followed by", r"and then", r"then",
    r"最后", r"finally",
)
# 全角/半角都收：语音识别在两种形式间摇摆是常态，
# 只认其中一种会让同样的句子时而能拆、时而不能。
_COMMA_RE = re.compile(r"[，,；;]")

_CONNECTOR_RE = re.compile(
    r"\s*(?:(?:" + "|".join(_CONJUNCTIONS) + r")\s*|[;；]\s*)+",
    re.IGNORECASE,
)

# 「先做 X，再做 Y」里的「先」不是分隔符，而是去掉即可
_LEADING_RE = re.compile(r"^(?:先|首先)\s*")

# 最低长度：太短的片段几乎肯定是误切
_MIN_SEGMENT = 2


def split_compound(text: str) -> List[str]:
    """
    把一句话拆成有序的多个动作。

    >>> split_compound("把 a.txt 复制到 b.txt 然后删除 b.txt")
    ['把 a.txt 复制到 b.txt', '删除 b.txt']
    >>> split_compound("打开浏览器并搜索天气")   # "并" 不作分隔符，避免误切
    ['打开浏览器并搜索天气']

    :returns: 单段文本原样返回 ``[text]``；无法拆分时同样返回单元素列表。
    """
    if not text or not text.strip():
        return []

    normalized = text.strip()
    if _LEADING_RE.match(normalized):
        normalized = _LEADING_RE.sub("", normalized)

    parts = [p.strip(" 　，,。.!！?？") for p in _CONNECTOR_RE.split(normalized)]
    parts = [p for p in parts if p]

    # 逗号也可能分隔两个动作，但风险高（"读取 a.txt，不对" 里的逗号不是分隔符），
    # 因此只在两侧都像完整指令时才切。
    if len(parts) <= 1 and _COMMA_RE.search(normalized):
        by_comma = [p.strip(" 　，,。.!！?？") for p in _COMMA_RE.split(normalized)]
        by_comma = [p for p in by_comma if p]
        if len(by_comma) > 1 and all(_looks_like_instruction(p) for p in by_comma):
            parts = by_comma

    if len(parts) <= 1:
        return [normalized] if normalized else []

    # 若切出过短片段，说明连接词出现在数据（如文件名）里，放弃拆分更安全
    if any(len(p) < _MIN_SEGMENT for p in parts):
        return [normalized]

    return parts


# 指令起始词：用于判断逗号两侧是否各自构成一个完整指令。
# 刻意不加 ``\b`` 词边界：它只在单词类字符之间成立，而"关闭"与"窗口"
# 相邻的两个汉字都是单词类字符，边界不存在，规则会整条失配。
# 改用「按位置锚定 + 中文词优先」的顺序来保证最长匹配。
_VERB_PREFIX_RE = re.compile(
    r"^(?:把|将|打开|启动|运行|执行|跑|读取|读|查看|看看|显示|列出|列|搜索|搜|查找|找|删除|删|删掉"
    r"|关闭|关掉|最小化|最大化|切换|切到|新建|创建|重命名|复制|拷贝|移动|挪|提交|撤销|取消|确认|整理"
    r"|copy|move|delete|open|launch|start|run|read|view|list|show|search|find|close|minimize"
    r"|maximize|focus|create|commit|undo|cancel|yes|no)",
    re.IGNORECASE,
)


# 前导副词/连接词：它们描述"接下来"而非动作本身，
# 判断指令性时应先剥掉。注意不能把它们当动词——
# "再"是动词会把"再生材料.txt"读成一条指令。
_LEADING_ADVERB_RE = re.compile(
    r"^(?:再|又|接着|然后|随后|紧接着|顺带|顺便|接下来|帮我|请你|请|麻烦)\s*",
)


def _looks_like_instruction(fragment: str) -> bool:
    """片段是否像一个能独立成立的指令（剥掉前导副词后以动词开头）。"""
    stripped = _LEADING_ADVERB_RE.sub("", fragment.strip())
    return bool(stripped) and bool(_VERB_PREFIX_RE.match(stripped))


def has_multiple_actions(text: str) -> bool:
    return len(split_compound(text)) > 1


def split_respecting_quotes(text: str) -> List[str]:
    """
    拆分时保护引号与尖括号内容。

    ``把 "然后我们再开会.txt" 移动到 docs`` 不应该被切成两半。
    """
    protected: List[str] = []

    def _stash(m: re.Match) -> str:
        protected.append(m.group(0))
        return f"\x00{len(protected) - 1}\x00"

    guarded = re.sub(r"[「『\"'“”][^」』\"'“”]*[」』\"'“”]|<[^>]*>", _stash, text)
    segments = split_compound(guarded)
    return [_restore(seg, protected) for seg in segments]


def _restore(segment: str, protected: List[str]) -> str:
    def _put(m: re.Match) -> str:
        index = int(m.group(1))
        return protected[index] if 0 <= index < len(protected) else m.group(0)

    return re.sub(r"\x00(\d+)\x00", _put, segment)