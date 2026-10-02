"""认知层测试：意图解析、槽位提取、指代消解、对话状态机。"""

from __future__ import annotations

import pytest

from futurework.cognition.dialogue import Clarification, DialogueManager, DialogueState
from futurework.cognition.intent import IntentParser, _normalize
from futurework.sensory.fusion import _polarity_of
from futurework.sensory.fusion import FusedSignal
from futurework.types import (
    ExecutionStatus,
    GestureType,
    GroundingTarget,
    Intent,
    IntentCategory,
    ModalityType,
)


@pytest.fixture
def parser() -> IntentParser:
    return IntentParser()


# ============================================================================
# 文本归一化
# ============================================================================
class TestNormalization:
    def test_fullwidth_to_halfwidth(self):
        assert _normalize("ＡＢＣ１２３") == "ABC123"

    def test_case_is_preserved(self):
        """回归：归一化不得改大小写，否则 TODO.md 会变成找不到的 todo.md。"""
        assert _normalize("读取 TODO.md") == "读取 TODO.md"
        assert _normalize("Read Notes.TXT") == "Read Notes.TXT"

    def test_collapses_whitespace(self):
        assert _normalize("  打开   浏览器  ") == "打开 浏览器"

    def test_empty_input(self):
        assert _normalize("") == ""


# ============================================================================
# 意图解析
# ============================================================================
class TestIntentParsing:
    @pytest.mark.parametrize("text,action", [
        ("打开浏览器", "launch_app"),
        ("打开 github.com", "open_url"),
        ("打开 https://example.com/docs", "open_url"),
        ("列出目录 .", "list_directory"),
        ("看看目录", "list_directory"),
        ("读取 notes.txt", "read_file"),
        ("删除文件 old.txt", "delete_file"),
        ("把 hello 写入 a.txt", "write_file"),
        ("追加一行到 log.txt", "append_file"),
        ("在 a.txt 里查找 foo", "search_in_file"),
        ("创建目录 build", "create_directory"),
        ("列出所有窗口", "list_windows"),
        ("切换到终端窗口", "focus_window"),
        ("最大化窗口", "maximize_window"),
        ("最小化全部", "minimize_all"),
        ("搜索 python 异步", "search_web"),
        ("后退", "browser_back"),
        ("查看 git 状态", "git_status"),
        ("查看改动", "git_diff"),
        ("提交代码 feat: x", "git_commit"),
        ("跑测试", "run_tests"),
        ("运行 ls -la", "run_command"),
        ("在终端执行 git status", "run_command"),
        ("编辑 main.py", "open_file_editor"),
        ("撤销", "undo"),
        ("取消", "cancel"),
        ("确认", "confirm"),
        ("不要", "reject"),
    ])
    def test_chinese_commands(self, parser, text, action):
        assert parser.parse(text).action == action

    @pytest.mark.parametrize("text,action", [
        ("open browser", "launch_app"),
        ("open example.com", "open_url"),
        ("list directory", "list_directory"),
        ("read notes.txt", "read_file"),
        ("delete file old.txt", "delete_file"),
        ("write hello to a.txt", "write_file"),
        ("search python asyncio", "search_web"),
        ("go back", "browser_back"),
        ("run tests", "run_tests"),
        ("run the command ls -la", "run_command"),
        ("git status", "git_status"),
        ("undo", "undo"),
        ("yes", "confirm"),
        ("no", "reject"),
        ("minimize all", "minimize_all"),
        ("list all windows", "list_windows"),
    ])
    def test_english_commands(self, parser, text, action):
        assert parser.parse(text).action == action

    def test_slot_extraction_path(self, parser):
        intent = parser.parse("读取 /home/user/notes.txt")
        assert intent.parameters["path"] == "/home/user/notes.txt"

    def test_slot_extraction_url(self, parser):
        intent = parser.parse("打开 https://example.com/api")
        assert intent.parameters["url"] == "https://example.com/api"

    def test_slot_extraction_content_and_path(self, parser):
        intent = parser.parse("把 hello world 写入 out.txt")
        assert intent.parameters["path"] == "out.txt"
        assert "hello world" in intent.parameters["content"]

    def test_slot_extraction_command(self, parser):
        intent = parser.parse("运行 git status")
        assert intent.parameters["command"] == "git status"

    def test_verb_is_not_swallowed_into_path(self, parser):
        """回归：Python 的 \\w 匹配汉字，动词不写全会把"取"当成文件名。"""
        intent = parser.parse("读取 notes.txt")
        assert intent.action == "read_file"
        assert intent.parameters["path"] == "notes.txt"

    def test_unrecognized_returns_unknown(self, parser):
        intent = parser.parse("今天天气真不错啊")
        assert intent.category is IntentCategory.UNKNOWN
        assert intent.requires_confirmation is True

    def test_destructive_requires_confirmation(self, parser):
        assert parser.parse("删除文件 x.txt").requires_confirmation is True

    def test_readonly_does_not_require_confirmation(self, parser):
        assert parser.parse("列出目录 .").requires_confirmation is False

    def test_grounded_target_used_when_slot_missing(self, parser):
        fused = FusedSignal(
            confidence=0.9,
            modalities_used=[ModalityType.SPEECH, ModalityType.GAZE],
            target=GroundingTarget(target_type="element", target_id="btn_ok", label="确定"),
        )
        intent = parser.parse("把它按下", fused)
        assert intent.grounded_target is not None

    def test_conflicting_modalities_lower_confidence(self, parser):
        fused = FusedSignal(
            confidence=0.9,
            modalities_used=[ModalityType.SPEECH, ModalityType.HEAD_POSE],
            conflicting_modalities=[ModalityType.HEAD_POSE],
        )
        assert parser.parse("打开浏览器", fused).confidence < 0.9

    def test_multi_modal_agreement_raises_confidence(self, parser):
        fused = FusedSignal(
            confidence=0.95,
            modalities_used=[ModalityType.SPEECH, ModalityType.HEAD_POSE],
            agreeing_modalities=[ModalityType.HEAD_POSE],
        )
        assert parser.parse("打开浏览器", fused).confidence > 0.9

    def test_stop_gesture_overrides_understood_command(self, parser):
        fused = FusedSignal(
            confidence=0.9,
            modalities_used=[ModalityType.SPEECH, ModalityType.GESTURE],
            rationale=["张开手掌 → 安全停止信号，优先级最高"],
        )
        assert parser.parse("运行 ls -la", fused).action == "stop"

    def test_stop_gesture_survives_unrecognised_speech(self, parser):
        """安全信号不能因为"没听懂"而失效。"""
        fused = FusedSignal(
            confidence=0.9,
            modalities_used=[ModalityType.SPEECH, ModalityType.GESTURE],
            rationale=["张开手掌 → 安全停止信号，优先级最高"],
        )
        intent = parser.parse("嗯嗯嗯阿巴阿巴", fused)
        assert intent.action == "stop"
        assert intent.category is IntentCategory.CANCEL_UNDO

    def test_semantic_channel_used_when_no_rule_matches(self, parser):
        """
        注入简易字符袋编码器后，未背过的新说法也能命中。

        真实部署时可换成 sentence-transformers 或大模型 embedding，
        接口与消歧逻辑不变。
        """
        def encoder(text: str):
            v = [0.0, 0.0, 0.0]
            if any(ch in text for ch in ("删", "清", "删掉", "delete", "remove")):
                v[0] = 1.0
            if "file" in text or "文件" in text:
                v[1] = 0.5
            if "browser" in text or "浏览器" in text or "app" in text:
                v[2] = 0.5
            return v

        parser.set_encoder(encoder)
        intent = parser.parse("把桌面上的废纸都清理了")
        assert intent.category is IntentCategory.DOCUMENT_EDIT
        assert intent.action == "delete_file"

    def test_semantic_below_threshold_stays_unknown(self, parser):
        def encoder(text: str):
            # 每个已知样本一个正交基向量，未知文本给零向量 → 余弦恒为 0
            vector = [0.0] * 64
            if text in parser._intent_texts.values():
                vector[hash(text) % 64] = 1.0
            return vector

        parser.set_encoder(encoder)
        intent = parser.parse("zzzz")
        assert intent.category is IntentCategory.UNKNOWN

    def test_rule_match_takes_priority_over_semantic(self, parser):
        def encoder(text: str):
            # 让语义通道强烈指向 DOCUMENT_EDIT
            return [1.0, 0.0]

        parser.set_encoder(encoder)
        intent = parser.parse("列出目录 .")
        assert intent.action == "list_directory"


class TestPolarity:
    @pytest.mark.parametrize("text,expected", [
        ("好的", "affirm"),
        ("确认", "affirm"),
        ("不要", "deny"),
        ("取消吧", "deny"),
        ("打开浏览器", None),
    ])
    def test_polarity(self, text, expected):
        assert _polarity_of(text) == expected


# ============================================================================
# 对话状态机
# ============================================================================
class TestDialogue:
    def test_clear_intent_goes_straight_to_execute(self):
        d = DialogueManager()
        intent = Intent(category=IntentCategory.NAVIGATION, action="list_windows")
        assert d.accept(intent) is DialogueState.EXECUTING

    def test_destructive_intent_awaits_confirmation(self):
        d = DialogueManager()
        intent = Intent(
            category=IntentCategory.DOCUMENT_EDIT, action="delete_file",
            parameters={"path": "a.txt"}, requires_confirmation=True,
        )
        assert d.accept(intent) is DialogueState.AWAITING_CONFIRMATION
        assert d.pending_intent().action == "delete_file"

    def test_unknown_intent_asks_clarification(self):
        d = DialogueManager()
        d.accept(Intent(category=IntentCategory.UNKNOWN, action="unknown"))
        assert d.state is DialogueState.AWAITING_CLARIFICATION

    def test_missing_slot_asks_for_it(self):
        d = DialogueManager()
        d.accept(Intent(category=IntentCategory.DOCUMENT_EDIT, action="read_file", parameters={}))
        assert d.state is DialogueState.AWAITING_CLARIFICATION
        assert "path" in d.clarification.question

    def test_rejection_clears_pending(self):
        d = DialogueManager()
        d.accept(Intent(category=IntentCategory.DOCUMENT_EDIT, action="delete_file",
                         parameters={"path": "a.txt"}, requires_confirmation=True))
        d.accept(Intent(category=IntentCategory.REJECTION, action="reject"))
        assert d.state is DialogueState.IDLE
        assert d.pending_intent() is None

    def test_cancel_clears_pending(self):
        d = DialogueManager()
        d.accept(Intent(category=IntentCategory.DOCUMENT_EDIT, action="delete_file",
                         parameters={"path": "a.txt"}, requires_confirmation=True))
        d.accept(Intent(category=IntentCategory.CANCEL_UNDO, action="cancel"))
        assert d.pending_intent() is None

    def test_undo_itself_requires_confirmation(self):
        d = DialogueManager()
        d.accept(Intent(category=IntentCategory.CANCEL_UNDO, action="undo"))
        assert d.state is DialogueState.AWAITING_CONFIRMATION

    def test_clarification_attempts_increment(self):
        d = DialogueManager()
        for _ in range(3):
            d.accept(Intent(category=IntentCategory.UNKNOWN, action="unknown"))
        assert d.clarification.attempts == 3
        assert d.clarification_exhausted() is True

    def test_fallback_plan_gives_options(self):
        d = DialogueManager()
        for _ in range(3):
            d.accept(Intent(category=IntentCategory.UNKNOWN, action="unknown"))
        plan = d.fallback_plan()
        assert "取消" in plan

    def test_suspend_and_resume(self):
        d = DialogueManager()
        d.accept(Intent(category=IntentCategory.DOCUMENT_EDIT, action="delete_file",
                         parameters={"path": "a.txt"}, requires_confirmation=True))
        assert d.suspend("用户走开了") is DialogueState.SUSPENDED
        assert d.pending_intent() is not None
        assert d.resume() is DialogueState.AWAITING_CONFIRMATION

    def test_complete_resets(self):
        d = DialogueManager()
        d.accept(Intent(category=IntentCategory.DOCUMENT_EDIT, action="delete_file",
                         parameters={"path": "a.txt"}, requires_confirmation=True))
        d.complete(ExecutionStatus.SUCCESS)
        assert d.state is DialogueState.IDLE
        assert d.pending_intent() is None

    def test_deictic_reference_resolved_from_memory(self):
        d = DialogueManager()
        target = GroundingTarget(target_type="file", target_id="report.md", label="报告")
        d.remember_target(target)
        intent = Intent(
            category=IntentCategory.DOCUMENT_EDIT, action="read_file",
            parameters={"path": "这个"},
        )
        d.accept(intent)
        assert intent.parameters["path"] == "report.md"
        assert intent.grounded_target is not None
        assert d.state is DialogueState.EXECUTING

    def test_reference_table_is_bounded(self):
        d = DialogueManager()
        for i in range(50):
            d.remember_target(GroundingTarget(target_id=f"t{i}"))
        assert len(d.reference_table) <= 32