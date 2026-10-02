"""
对话连续性测试：复合指令、纠正、槽位续答。

这三项决定了交互"像不像人"。人和人说话时会一次交代两件事、
会否定刚说完的话、会只补一个词而不是把整句重说一遍——
系统若不支持这些，每一轮都像在跟机器重新学一遍语法。
"""

from __future__ import annotations

import os

import pytest

from futurework.cognition.compound import (
    detect_correction,
    split_compound,
    split_respecting_quotes,
)
from futurework.cognition.intent import IntentParser
from futurework.runtime.session import FutureWorkSession, TextSession
from futurework.types import ExecutionStatus


# ============================================================================
# 复合指令拆分
# ============================================================================
class TestCompoundSplit:
    @pytest.mark.parametrize("text,expected", [
        ("把 a.txt 复制到 b.txt 然后删除 b.txt", ["把 a.txt 复制到 b.txt", "删除 b.txt"]),
        ("先跑测试，然后打开日志", ["跑测试", "打开日志"]),
        ("关闭窗口，再最小化全部", ["关闭窗口", "再最小化全部"]),
        ("关闭这个窗口，顺便读取 README.md", ["关闭这个窗口", "顺便读取 README.md"]),
    ])
    def test_splits_compound(self, text, expected):
        assert split_respecting_quotes(text) == expected

    @pytest.mark.parametrize("text", [
        "读取 README.md",
        "列出目录 .",
        "打开浏览器并搜索天气",       # "并" 不作分隔符，避免误切
        "把 \"然后我们再开会.txt\" 移动到 docs",
        "创建目录 output",
    ])
    def test_does_not_over_split(self, text):
        assert len(split_respecting_quotes(text)) == 1

    def test_empty_input(self):
        assert split_compound("") == []

    def test_leading_first_is_stripped(self):
        assert split_respecting_quotes("先读取 a.txt") == ["读取 a.txt"]


# ============================================================================
# 纠正识别
# ============================================================================
class TestCorrectionDetection:
    @pytest.mark.parametrize("text", ["不对", "不是这个", "说错了", "不行", "no"])
    def test_plain_denial(self, text):
        correction = detect_correction(text)
        assert correction is not None
        assert correction.is_rewrite is False

    @pytest.mark.parametrize("text,fix", [
        ("不对，是 b.txt", "b.txt"),
        ("应该是 y.txt", "y.txt"),
        ("不对,应该是 读取 y.txt", "读取 y.txt"),
        ("I meant b.txt", "b.txt"),
    ])
    def test_correction_with_replacement(self, text, fix):
        correction = detect_correction(text)
        assert correction is not None
        assert correction.is_rewrite is True
        assert correction.replacement == fix

    @pytest.mark.parametrize("text", ["读取 a.txt", "删除文件 x.txt", "确认"])
    def test_not_a_correction(self, text):
        assert detect_correction(text) is None


# ============================================================================
# 半成品意图（裸动词）
# ============================================================================
class TestIncompleteIntent:
    @pytest.fixture
    def parser(self):
        return IntentParser()

    @pytest.mark.parametrize("text,action", [
        ("读取", "read_file"),
        ("读一下", "read_file"),
        ("删除", "delete_file"),
        ("复制", "copy_file"),
        ("open", "launch_app"),
    ])
    def test_detects_bare_verb(self, parser, text, action):
        incomplete = parser.detect_incomplete(text)
        assert incomplete is not None and incomplete.action == action

    @pytest.mark.parametrize("text", ["嗯嗯嗯", "今天天气不错", "", "读取 a.txt"])
    def test_rejects_non_verb(self, parser, text):
        assert parser.detect_incomplete(text) is None

    def test_bare_verb_reaches_parser_as_incomplete(self, parser):
        intent = parser.parse("读取")
        assert intent.action == "read_file"

    def test_verb_does_not_swallow_path(self, parser):
        """回归：``\\w`` 匹配汉字，动词写全会把"取"当成文件名。"""
        intent = parser.parse("读取")
        assert intent.parameters.get("path") is None


# ============================================================================
# 端到端：复合指令
# ============================================================================
class TestCompoundExecution:
    @pytest.fixture
    def session(self, tmp_path) -> TextSession:
        (tmp_path / "a.txt").write_text("hello")
        return TextSession(workdir=str(tmp_path))

    def test_both_actions_executed(self, session, tmp_path):
        turn = session.run("把 a.txt 复制到 c.txt 然后读取 c.txt")
        assert turn.status is ExecutionStatus.SUCCESS
        assert (tmp_path / "c.txt").exists()

    def test_second_action_sees_first_result(self, session, tmp_path):
        """第二段依赖第一段的执行结果——这正是复合指令的意义。"""
        session.run("把 a.txt 复制到 c.txt 然后读取 c.txt")
        assert "hello" in session.last.feedback.text

    def test_stops_on_failure(self, session, tmp_path):
        (tmp_path / "keep.txt").write_text("x")
        turn = session.run("把 不存在.txt 复制到 d.txt 然后删除 keep.txt")
        assert turn.status is ExecutionStatus.FAILED
        # 关键：第一步失败后，绝不能继续执行破坏性的第二步
        assert (tmp_path / "keep.txt").exists()

    def test_stops_at_confirmation(self, session, tmp_path):
        turn = session.run("把 a.txt 复制到 c.txt 然后删除 c.txt")
        assert turn.status is ExecutionStatus.PENDING_CONFIRMATION
        assert (tmp_path / "c.txt").exists()      # 第一步已做
        assert (tmp_path / "c.txt").exists()
        # 第二步未执行——撤销第一��后文件应仍在
        assert turn.needs_confirmation is True

    def test_multimodal_only_applies_to_first_segment(self, session, tmp_path):
        from futurework.types import GestureType

        # 手势锚点若被复用到第二段，会套用到毫不相干的命令上
        turn = session.say("读取 a.txt 然后列出目录 .", gesture=GestureType.POINT,
                           gesture_position=(0.9, 0.9, 0.5))
        assert turn.status is ExecutionStatus.SUCCESS

    def test_single_action_unchanged(self, session):
        assert session.run("列出目录 .").status is ExecutionStatus.SUCCESS


# ============================================================================
# 端到端：纠正
# ============================================================================
class TestCorrectionFlow:
    @pytest.fixture
    def session(self, tmp_path) -> TextSession:
        return TextSession(workdir=str(tmp_path))

    def test_plain_denial_undoes_last_action(self, session, tmp_path):
        session.run("把 hello 写入 z.txt")
        assert (tmp_path / "z.txt").exists()
        turn = session.run("不对")
        assert turn.status is ExecutionStatus.CANCELLED
        assert not (tmp_path / "z.txt").exists()

    def test_denial_invites_next_step(self, session):
        session.run("把 hello 写入 z.txt")
        turn = session.run("不是这个")
        assert "想怎么做" in turn.feedback.text or "先不动" in turn.feedback.text

    def test_rewrite_reapplies_with_new_value(self, session, tmp_path):
        session.run("把 world 写入 w.txt")
        turn = session.run("不对，应该是 a.txt")
        assert (tmp_path / "a.txt").exists()
        assert not (tmp_path / "w.txt").exists()

    def test_rewrite_with_full_instruction(self, session, tmp_path):
        (tmp_path / "src.txt").write_text("data")
        session.run("把 world 写入 w.txt")
        turn = session.run("不对，读取 src.txt")
        assert turn.status is ExecutionStatus.SUCCESS
        assert "data" in turn.feedback.text

    def test_correction_when_nothing_to_undo(self, session):
        turn = session.run("不对")
        assert turn.status is ExecutionStatus.CANCELLED
        assert turn.feedback.text          # 必须给出可读回应，不能崩

    def test_correction_clears_pending(self, session, tmp_path):
        (tmp_path / "x.txt").write_text("x")
        session.run("删除文件 x.txt")
        assert session.orchestrator.dialogue.pending is not None
        session.run("不对")
        assert session.orchestrator.dialogue.pending is None


# ============================================================================
# 端到端：槽位续答
# ============================================================================
class TestSlotContinuation:
    @pytest.fixture
    def session(self, tmp_path) -> TextSession:
        (tmp_path / "a.txt").write_text("hello")
        return TextSession(workdir=str(tmp_path))

    def test_bare_verb_asks_for_missing_slot(self, session):
        turn = session.run("读取")
        assert turn.status is ExecutionStatus.PENDING_CONFIRMATION
        assert "path" in turn.feedback.text

    def test_bare_value_fills_slot(self, session):
        session.run("读取")
        turn = session.run("a.txt")
        assert turn.status is ExecutionStatus.SUCCESS
        assert "hello" in turn.feedback.text

    def test_answer_with_full_sentence_still_works(self, session):
        session.run("读取")
        turn = session.run("读取 a.txt")
        assert turn.status is ExecutionStatus.SUCCESS

    def test_answer_with_new_command_not_treated_as_slot(self, session, tmp_path):
        (tmp_path / "b.txt").write_text("other")
        session.run("读取")
        turn = session.run("读取 b.txt")
        assert "other" in turn.feedback.text

    def test_clarification_state_cleared_after_answer(self, session):
        session.run("读取")
        session.run("a.txt")
        assert session.orchestrator.dialogue.state.value != "awaiting_clarification"


# ============================================================================
# 反馈通道
# ============================================================================
class TestFeedbackChannels:
    def test_callback_receives_full_feedback(self, tmp_path):
        (tmp_path / "a.txt").write_text("x")
        session = TextSession(workdir=str(tmp_path))
        seen = []
        session.on_feedback(lambda fb, turn: seen.append(fb))
        session.run("读取 a.txt")
        assert seen
        assert seen[-1].visual_alert_level == "success"
        assert seen[-1].sound_cue == "success"

    def test_error_feedback_marked_as_error(self, tmp_path):
        session = TextSession(workdir=str(tmp_path))
        seen = []
        session.on_feedback(lambda fb, turn: seen.append(fb))
        session.run("读取 不存在.txt")
        assert seen[-1].visual_alert_level == "error"

    def test_unsubscribe(self, tmp_path):
        session = TextSession(workdir=str(tmp_path))
        seen = []
        unsubscribe = session.on_feedback(lambda fb, turn: seen.append(fb))
        session.run("列出目录 .")
        count = len(seen)
        unsubscribe()
        session.run("列出目录 .")
        assert len(seen) == count

    def test_bad_subscriber_does_not_break_execution(self, tmp_path):
        session = TextSession(workdir=str(tmp_path))
        good = []
        session.on_feedback(lambda fb, turn: (_ for _ in ()).throw(RuntimeError("坏订阅者")))
        session.on_feedback(lambda fb, turn: good.append(fb))
        assert session.run("列出目录 .").status is ExecutionStatus.SUCCESS
        assert good