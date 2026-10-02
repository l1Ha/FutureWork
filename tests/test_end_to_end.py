"""
端到端测试：真实的自然交互流程。

这些用例模拟的是"人怎么跟另一个��协作"，而不是"函数怎么调"：
边说边指、点头确认、摇头拒绝、撤销、被打断后恢复。
"""

from __future__ import annotations

import os

import pytest

from futurework.cognition.dialogue import DialogueState
from futurework.runtime.orchestrator import EventType, InteractionEvent, Orchestrator
from futurework.runtime.session import FutureWorkSession, TextSession, _parse_inline_gesture
from futurework.sensory.gaze import UIElement
from futurework.tools.registry import ToolRegistry
from futurework.types import (
    EmotionType,
    ExecutionStatus,
    FacialSignal,
    GazeSignal,
    GestureType,
    HandGestureSignal,
    SpeechSignal,
)


@pytest.fixture
def session(tmp_path) -> TextSession:
    """所有文件操作限定在 tmp_path 内，绝不触碰真实磁盘。"""
    return TextSession(workdir=str(tmp_path))


# ============================================================================
# 基础流程
# ============================================================================
class TestBasicFlow:
    def test_read_file_e2e(self, session, tmp_path):
        (tmp_path / "notes.txt").write_text("会议记录：周三复盘")
        turn = session.run("读取 notes.txt")
        assert turn.status is ExecutionStatus.SUCCESS
        assert "会议记录" in turn.feedback.text

    def test_write_then_verify_e2e(self, session, tmp_path):
        assert session.run("把 hello 写入 out.txt").status is ExecutionStatus.SUCCESS
        assert (tmp_path / "out.txt").read_text() == "hello"

    def test_list_directory_e2e(self, session, tmp_path):
        (tmp_path / "a.txt").write_text("x")
        (tmp_path / "sub").mkdir()
        turn = session.run("列出目录 .")
        assert turn.status is ExecutionStatus.SUCCESS
        assert "2 个条目" in turn.feedback.text

    def test_search_in_file_e2e(self, session, tmp_path):
        (tmp_path / "code.py").write_text("def foo():\n    pass\n\ndef bar():\n    pass\n")
        turn = session.run("在 code.py 里查找 bar")
        assert turn.status is ExecutionStatus.SUCCESS
        assert "第 4 行" in turn.feedback.text

    def test_run_command_e2e(self, session):
        session.say("运行 echo hello-futurework")
        assert "hello-futurework" in session.last.feedback.text

    def test_system_info_e2e(self, session):
        turn = session.run("列出所有窗口")
        # 无论是否有窗口工具，都不该崩溃，且要说清原因
        assert turn.feedback.text
        assert turn.errors == []

    def test_latency_budget(self, session, tmp_path):
        """交互延迟必须远低于人感知的流畅阈值（约 300ms）。"""
        (tmp_path / "a.txt").write_text("x")
        latencies = [session.run("读取 a.txt").latency_ms for _ in range(5)]
        assert max(latencies) < 300, f"最大延迟 {max(latencies):.0f}ms 超出流畅阈值"


# ============================================================================
# 危险操作确认流
# ============================================================================
class TestConfirmationFlow:
    def test_delete_requires_confirmation(self, session, tmp_path):
        (tmp_path / "tmp.txt").write_text("x")
        turn = session.run("删除文件 tmp.txt")
        assert turn.status is ExecutionStatus.PENDING_CONFIRMATION
        assert (tmp_path / "tmp.txt").exists()      # 关键：没真删

    def test_nod_approves(self, session, tmp_path):
        target = tmp_path / "tmp.txt"
        target.write_text("x")
        session.run("删除文件 tmp.txt")
        turn = session.nod()
        assert turn.status is ExecutionStatus.SUCCESS
        assert not target.exists()

    def test_pinch_approves(self, session, tmp_path):
        target = tmp_path / "tmp.txt"
        target.write_text("x")
        session.run("删除文件 tmp.txt")
        turn = session.pinch()
        assert turn.status is ExecutionStatus.SUCCESS
        assert not target.exists()

    def test_verbal_yes_approves(self, session, tmp_path):
        target = tmp_path / "tmp.txt"
        target.write_text("x")
        session.run("删除文件 tmp.txt")
        turn = session.say("确认")
        assert turn.status is ExecutionStatus.SUCCESS
        assert not target.exists()

    def test_shake_denies(self, session, tmp_path):
        target = tmp_path / "tmp.txt"
        target.write_text("x")
        session.run("删除文件 tmp.txt")
        turn = session.shake()
        assert turn.status is ExecutionStatus.CANCELLED
        assert target.exists()

    def test_stop_gesture_cancels_pending(self, session, tmp_path):
        target = tmp_path / "tmp.txt"
        target.write_text("x")
        session.run("删除文件 tmp.txt")
        turn = session.stop()
        assert turn.status is ExecutionStatus.CANCELLED
        assert target.exists()

    def test_confirmation_is_single_use(self, session, tmp_path):
        (tmp_path / "a.txt").write_text("x")
        (tmp_path / "b.txt").write_text("y")
        session.run("删除文件 a.txt")
        session.nod()
        session.say("确认")           # 再确认一次不应删掉别的文件
        assert (tmp_path / "b.txt").exists()


# ============================================================================
# 撤销流
# ============================================================================
class TestUndoFlow:
    def test_undo_write_restores_original(self, session, tmp_path):
        target = tmp_path / "a.txt"
        target.write_text("原始内容")
        session.run("把 覆盖内容 写入 a.txt")
        assert target.read_text() == "覆盖内容"
        session.undo()
        assert target.read_text() == "原始内容"

    def test_undo_spoken(self, session, tmp_path):
        (tmp_path / "a.txt").write_text("原始")
        session.run("把 新内容 写入 a.txt")
        session.say("撤销")
        assert (tmp_path / "a.txt").read_text() == "原始"

    def test_undo_skips_reads(self, session, tmp_path):
        (tmp_path / "a.txt").write_text("原始")
        session.run("把 新内容 写入 a.txt")
        session.run("读取 a.txt")           # 只读不该被撤销
        session.undo()
        assert (tmp_path / "a.txt").read_text() == "原始"

    def test_delete_is_not_undoable_and_says_so(self, session, tmp_path):
        """删除之后没有可撤销项，且必须明说不可撤销，而不是假装成功。"""
        (tmp_path / "a.txt").write_text("x")
        session.run("删除文件 a.txt")
        session.nod()
        outcome = session.undo()
        assert outcome["ok"] is False
        assert "不可撤销" in outcome["message"]

    def test_undo_targets_latest_change_not_delete(self, session, tmp_path):
        """有一次可撤销的写入时，撤销应作用于它而不是更早的删除。"""
        (tmp_path / "a.txt").write_text("x")
        session.run("删除文件 a.txt")
        session.nod()
        (tmp_path / "b.txt").write_text("原始")
        session.run("把 新 写入 b.txt")
        outcome = session.undo()
        assert outcome["ok"] is True
        assert (tmp_path / "b.txt").read_text() == "原始"


# ============================================================================
# 多模态融合交互
# ============================================================================
class TestMultimodalInteraction:
    def test_speech_plus_pointing_creates_anchor(self, session):
        turn = session.say("把这个挪到那里", gesture=GestureType.POINT, gesture_position=(0.3, 0.4, 0.5))
        assert turn.fused is not None
        assert turn.fused.target is not None

    def test_gaze_target_registered(self, session):
        session.register_ui([UIElement("btn", "提交按钮", (0.4, 0.4, 0.1, 0.1), "编辑器")])
        turn = session.say("点击它", look_at=(0.45, 0.45))
        assert turn.fused.target is not None

    def test_confusion_triggers_question_not_guess(self, session):
        turn = session.say("把这个那样弄一下", expression=EmotionType.CONFUSED)
        assert turn.status is ExecutionStatus.PENDING_CONFIRMATION

    def test_unclear_command_asks_clarification(self, session):
        turn = session.run("嗯嗯嗯那个那个")
        assert turn.status is ExecutionStatus.PENDING_CONFIRMATION
        assert turn.feedback.text   # 必须给出可读的追问

    def test_clarification_loop_terminates(self, session):
        """追问有上限，不能把用户困在循环里。"""
        for _ in range(5):
            turn = session.run("啊啊啊")
            if turn.status is ExecutionStatus.PENDING_CONFIRMATION and "取消" in turn.feedback.text:
                break
        else:
            pytest.fail("追问循环没有兜底出口")
        assert session.orchestrator.dialogue.state is DialogueState.IDLE

    def test_inline_gesture_parsing(self):
        gesture, pos = _parse_inline_gesture("移动这个 [POINT@0.3,0.4,0.2]")
        assert gesture is GestureType.POINT
        assert pos == (0.3, 0.4, 0.2)

    def test_inline_gesture_without_position(self):
        gesture, _ = _parse_inline_gesture("停止 [PALM_STOP]")
        assert gesture is GestureType.PALM_STOP

    def test_inline_gesture_malformed_is_ignored(self):
        gesture, pos = _parse_inline_gesture("随便什么 [GARBAGE@bad]")
        assert gesture is None and pos == (0.5, 0.5, 0.5)


# ============================================================================
# 异常处理
# ============================================================================
class TestErrorHandling:
    def test_missing_file_gives_actionable_advice(self, session):
        turn = session.run("读取 不存在的文件.txt")
        assert turn.status is ExecutionStatus.FAILED
        assert "确认名称或路径" in turn.feedback.text

    def test_dangerous_command_blocked(self, session):
        turn = session.say("在终端执行 rm -rf /")
        assert turn.status in (ExecutionStatus.REJECTED_BY_SAFETY, ExecutionStatus.PENDING_CONFIRMATION)

    def test_path_traversal_blocked(self, session):
        turn = session.run("读取 ../../../etc/passwd")
        assert turn.status is ExecutionStatus.FAILED
        assert "越界" in turn.feedback.text or "不合法" in turn.feedback.text

    def test_adapter_failure_does_not_crash_session(self, session):
        session.run("读取 不存在1.txt")
        turn = session.run("列出目录 .")     # 失败之后仍能继续
        assert turn.status is ExecutionStatus.SUCCESS

    def test_unavailable_tool_explains_why(self, tmp_path):
        """非 Git 目录下问 git 状态，应说清是环境问题而非指令问题。"""
        session = TextSession(workdir=str(tmp_path))
        turn = session.run("查看 git 状态")
        assert "Git 仓库" in turn.feedback.text

    def test_event_bus_isolates_subscriber_errors(self, session):
        calls = []
        session.orchestrator.bus.subscribe("turn.response", lambda e: (_ for _ in ()).throw(RuntimeError("坏订阅者")))
        session.orchestrator.bus.subscribe("turn.response", lambda e: calls.append(e))
        session.run("列出目录 .")
        assert calls                       # 坏订阅者没有影响好订阅者
        assert session.orchestrator.bus.errors()


# ============================================================================
# 可观测性
# ============================================================================
class TestObservability:
    def test_stage_timings_present(self, session, tmp_path):
        (tmp_path / "a.txt").write_text("x")
        turn = session.run("读取 a.txt")
        for stage in ("perceive", "fuse", "interpret", "route", "execute"):
            assert stage in turn.stage_timings

    def test_events_emitted_in_order(self, session, tmp_path):
        (tmp_path / "a.txt").write_text("x")
        seen = []
        for event_type in ("turn.start", "turn.fused", "turn.intent", "turn.executing", "turn.result", "turn.end"):
            session.orchestrator.bus.subscribe(event_type, lambda e: seen.append(e.name))
        session.run("读取 a.txt")
        assert seen == ["turn.start", "turn.fused", "turn.intent", "turn.executing", "turn.result", "turn.end"]

    def test_health_report(self, session):
        health = session.health()
        assert "adapters" in health and "breakers" in health
        assert "filesystem" in health["adapters"]

    def test_capabilities_enumerable(self, session):
        caps = session.orchestrator.registry.capabilities()
        assert "read_file" in caps["filesystem"]
        assert "run_command" in caps["terminal"]

    def test_turn_summary_is_readable(self, session, tmp_path):
        (tmp_path / "a.txt").write_text("x")
        summary = session.run("读取 a.txt").summary()
        assert "success" in summary and "ms" in summary


# ============================================================================
# 自定义工具接入
# ============================================================================
class TestExtensibility:
    def test_third_party_tool_plugs_in(self, session, tmp_path):
        from futurework.tools.base import Capability, ToolAdapter

        class NotionAdapter(ToolAdapter):
            name = "notion"

            def setup(self):
                self.capabilities.declare(Capability(
                    action="create_page",
                    params={"title": "str"},
                    safety_level=SafetyLevel_SAFE,
                ))

            def _dispatch(self, action, args):
                return {"id": "page-123", "title": args["title"]}, {}

        session.orchestrator.registry.register(NotionAdapter())
        assert session.orchestrator.registry.resolve("create_page")[0] is not None

    def test_mcp_adapter_absence_is_graceful(self, tmp_path):
        from futurework.tools.mcp_adapter import McpAdapter

        adapter = McpAdapter([])
        available, why = adapter.is_available()
        assert available is False and "未配置" in why
        result = adapter.execute(__import__(
            "futurework.types", fromlist=["ToolCommand"]
        ).ToolCommand(tool_name="mcp", action="call_mcp_tool", args={"tool": "x"}))
        assert result.status is ExecutionStatus.FAILED


from futurework.types import SafetyLevel as SafetyLevel_SAFE  # noqa: E402  供上面自定义适配器使用


# ============================================================================
# 会话连续性
# ============================================================================
class TestConversationFlow:
    def test_multi_turn_context(self, session, tmp_path):
        (tmp_path / "report.md").write_text("# 季度报告")
        session.run("读取 report.md")
        turn = session.run("把它复制一份")     # 指代上一轮的目标
        assert turn.feedback.text               # 应有回应（追问或执行），而非崩溃

    def test_session_does_not_leak_pending_state(self, session, tmp_path):
        (tmp_path / "a.txt").write_text("x")
        session.run("删除文件 a.txt")
        assert session.orchestrator.dialogue.state is DialogueState.AWAITING_CONFIRMATION
        session.shake()
        assert session.orchestrator.dialogue.state is DialogueState.IDLE
        assert session.orchestrator.dialogue.pending_intent() is None

    def test_turns_are_recorded(self, session):
        session.run("列出目录 .")
        session.run("列出目录 .")
        assert len(session.turns) == 2

    def test_last_turn_accessible(self, session):
        session.run("列出目录 .")
        assert session.last is not None
        assert session.last.turn_id