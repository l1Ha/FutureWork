"""
编排引擎：把「一句话 + 手势 + 视线 + 表情」变成「工具执行 + 自然反馈」。

一次完整交互的九个阶段::

    1 感知 collect      汇总当前所有模态信号
    2 融合 fuse         多模态 → 统一线索（含冲突仲裁）
    3 理解 interpret    线索 + 文本 → 意图 + 参数
    4 对话 dialogue     指代消解、槽位补全、追问或放行
    5 安全 guard        风险分级 → 授权或询问
    6 路由 route        意图动作 → 具体适配器
    7 执行 execute      适配器执行（含重试、熔断、超时）
    8 记账 record       登记撤销信息
    9 反馈 respond      生成多模态反馈（文字/语音/视觉/触觉）

每个阶段都可独立失败且被隔离；失败不会中断链路，而是转成对用户有意义的
反馈继续往下走。这是"流畅"的技术含义。
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from futurework.cognition.compound import (
    ConditionalFlow,
    Correction,
    detect_correction,
    parse_conditional,
    split_respecting_quotes,
)
from futurework.cognition.dialogue import (
    DialogueManager,
    DialogueState,
    _REQUIRED_SLOTS as _REQUIRED_SLOTS_FOR,
)
from futurework.cognition.intent import IntentParser
from futurework.resilience.circuit import BreakerRegistry, CircuitOpenError
from futurework.resilience.errors import ErrorCategory, classify_result, human_readable
from futurework.resilience.retry import RetryPolicy, with_retry
from futurework.resilience.safety import PermissionDecision, SafetyGate, SecurityPolicy
from futurework.resilience.undo import Compensation, UndoManager
from futurework.runtime.event_bus import Event, EventBus, EventPriority
from futurework.sensory.fusion import FusedSignal, MultimodalFusionEngine
from futurework.tools.registry import ToolRegistry, default_registry
from futurework.types import (
    ExecutionResult,
    ExecutionStatus,
    FacialSignal,
    GazeSignal,
    GestureType,
    GroundingTarget,
    HandGestureSignal,
    HeadPoseSignal,
    Intent,
    IntentCategory,
    SafetyLevel,
    SpeechSignal,
    SystemFeedback,
    ToolCommand,
)

# 意图动作 → 工具提示的映射表（认知层与工具层的解耦点）
ACTION_TOOL_HINTS: Dict[str, str] = {
    "focus_window": "system",
    "close_window": "system",
    "minimize_all": "system",
    "maximize_window": "system",
    "list_windows": "system",
    "launch_app": "system",
    "system_info": "system",
    "read_file": "filesystem",
    "write_file": "filesystem",
    "append_file": "filesystem",
    "delete_file": "filesystem",
    "create_directory": "filesystem",
    "copy_file": "filesystem",
    "move_file": "filesystem",
    "list_directory": "filesystem",
    "search_in_file": "filesystem",
    "run_command": "terminal",
    "run_tests": "terminal",
    "get_working_directory": "terminal",
    "open_file_editor": "editor",
    "open_url": "browser",
    "search_web": "browser",
    "browser_back": "browser",
    "history": "browser",
    "git_status": "vcs",
    "git_diff": "vcs",
    "git_log": "vcs",
    "git_commit": "vcs",
    "git_add": "vcs",
    "create_directory": "filesystem",
    "copy_file": "filesystem",
    "move_file": "filesystem",
    "table_aggregate": "productivity",
    "table_query": "productivity",
    "generate_report": "productivity",
    "extract_outline": "productivity",
    "todo_add": "productivity",
    "todo_list": "productivity",
    "todo_complete": "productivity",
    "list_mcp_tools": "mcp",
    "call_mcp_tool": "mcp",
    "list_editors": "editor",
}

# 直接映射到工具动作的对话控制意图
CONTROL_ACTIONS = {"confirm", "reject", "cancel", "undo", "stop"}


class EventType(str, Enum):
    TURN_START = "turn.start"
    PERCEPTION = "turn.perception"
    FUSED = "turn.fused"
    INTENT = "turn.intent"
    CLARIFY = "turn.clarify"
    CONFIRM_REQUIRED = "turn.confirm_required"
    EXECUTING = "turn.executing"
    RESULT = "turn.result"
    RETRY = "turn.retry"
    ROLLBACK = "turn.rollback"
    RESPONSE = "turn.response"
    ERROR = "turn.error"
    TURN_END = "turn.end"


@dataclass
class InteractionEvent:
    """一次交互中收到的多模态输入集合。"""

    speech: Optional[SpeechSignal] = None
    gestures: List[HandGestureSignal] = field(default_factory=list)
    facial: Optional[FacialSignal] = None
    head_pose: Optional[HeadPoseSignal] = None
    gaze: Optional[GazeSignal] = None
    environment_noise_level: float = 0.1

    def is_empty(self) -> bool:
        return not any((self.speech, self.gestures, self.facial, self.head_pose, self.gaze))


@dataclass
class TurnResult:
    """一次完整交互的结果与全链路可观测数据。"""

    turn_id: str
    status: ExecutionStatus
    feedback: SystemFeedback
    intent: Optional[Intent] = None
    fused: Optional[FusedSignal] = None
    result: Optional[ExecutionResult] = None
    executed: bool = False
    needs_confirmation: bool = False
    latency_ms: float = 0.0
    stage_timings: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    undone: bool = False

    @property
    def ok(self) -> bool:
        return self.status is ExecutionStatus.SUCCESS

    def summary(self) -> str:
        stages = " ".join(f"{k}={v:.1f}ms" for k, v in self.stage_timings.items())
        stages_part = f" | {stages}" if stages else ""
        return (
            f"[{self.turn_id}] {self.status.value} | {self.feedback.text}{stages_part} | 总计 {self.latency_ms:.1f}ms"
        )


class Orchestrator:
    """
    交互编排引擎。

    :param workdir: 文件与终端操作的工作根目录
    :param registry: 自定义工具注册中心（默认构建全套内置适配器）
    """

    def __init__(
        self,
        *,
        workdir: Optional[str] = None,
        registry: Optional[ToolRegistry] = None,
        retry_policy: Optional[RetryPolicy] = None,
        undo_capacity: int = 50,
        breaker_defaults: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.workdir = workdir
        self.registry = registry or default_registry(workdir)
        self.fusion = MultimodalFusionEngine()
        self.parser = IntentParser()
        mem_path = os.path.join(workdir, ".futurework", "memory.json") if workdir else None
        self.dialogue = DialogueManager(memory_path=mem_path)
        self.undo = UndoManager(capacity=undo_capacity)
        self.safety = SafetyGate()
        self.policy = SecurityPolicy()
        self.bus = EventBus()
        self.breakers = BreakerRegistry(**(breaker_defaults or {}))
        self.retry_policy = retry_policy or RetryPolicy(max_attempts=3, base_delay=0.12)

        self._last_turn: Optional[TurnResult] = None
        self._turn_count = 0

    # ==================================================================
    # 主入口
    # ==================================================================
    def interact(self, event: InteractionEvent) -> TurnResult:
        """
        处理一次多模态交互（对话层入口）。

        进入九阶段流水线之前，先处理三件"一句话层面"的事——
        它们决定了人机交流像不像人与人交流：

        1. **纠正**："不对""应该是 b.txt"，这是人做完动作后最常说的一句；
        2. **复合拆分**："复制完然后删除"，人从不一次只下一个指令；
        3. 其余交给 :meth:`_interact_single` 原样执行。

        该方法**永不抛异常**：任何内部失败都被转换为带说明的反馈，
        保证对话能继续——这是自然交互的底线要求。
        """
        text = event.speech.transcript if event.speech else ""

        # ---------- 纠正 ----------
        correction = detect_correction(text)
        if correction is not None:
            return self._handle_correction(event, correction)

        # ---------- 条件控制流 (如果...就...否则...) ----------
        conditional = parse_conditional(text) if text else None
        if conditional is not None:
            return self._handle_conditional(event, conditional)

        # ---------- 复合指令 ----------
        segments = split_respecting_quotes(text) if text else []
        if len(segments) > 1:
            return self._handle_compound(event, segments)

        return self._interact_single(event)

    # ==================================================================
    # 纠正
    # ==================================================================
    def _handle_correction(self, event: InteractionEvent, correction: "Correction") -> TurnResult:
        """
        处理"不对 / 应该是 X"。

        有意区分两种结果：

        * **只否定**（"不对"）→ 撤回刚做的那一步，把选择权交回用户。
          人说"不对"时期待的是"回到刚才之前"，而不是"你再说一遍"；
        * **带更正**（"应该是 b.txt"）→ 撤回后直接执行更正后的动作。
        """
        turn_id = uuid.uuid4().hex[:8]
        started = time.perf_counter()
        self._turn_count += 1

        # 必须在撤销之前取：一旦回退，该条目就被标记为 undone，
        # 之后再想拿它来推导更正后的动作就找不到了。
        previous = self.undo.last_undoable()
        undone = self.undo.undo_last()
        self.dialogue.pending = None
        self.dialogue.clarification = None
        self.dialogue.state = DialogueState.IDLE

        if correction.is_rewrite and correction.replacement:
            replacement = correction.replacement
            # 更正内容常常只是一个裸值（"不对，应该是 b.txt"），
            # 它是对刚才那个动作的修正，而不是一条新指令。
            if self.parser.parse(replacement).category is IntentCategory.UNKNOWN and previous is not None:
                slot = _REQUIRED_SLOTS_FOR.get(previous.action, "")
                if slot and slot in previous.args:
                    params = dict(previous.args)
                    params[slot] = replacement
                    rebuilt = _rebuild_utterance(previous.action, params)
                    if rebuilt:
                        replacement = rebuilt

            turn = self._interact_single(
                InteractionEvent(
                    speech=SpeechSignal(transcript=replacement),
                    gestures=event.gestures,
                    facial=event.facial,
                    head_pose=event.head_pose,
                    gaze=event.gaze,
                    environment_noise_level=event.environment_noise_level,
                ),
                quiet=True,
            )
            prefix = f"{undone['message']}。" if undone.get("ok") else ""
            turn.feedback.text = f"{prefix}{turn.feedback.text}"
            turn.latency_ms = (time.perf_counter() - started) * 1000
            self.bus.emit(Event(EventType.TURN_END, {"turn_id": turn_id, "corrected": True}))
            return turn

        if undone.get("ok"):
            message = f"{undone['message']}。想怎么做？"
        else:
            message = f"明白，先不动。{undone.get('message', '')}"

        turn = TurnResult(
            turn_id=turn_id,
            status=ExecutionStatus.CANCELLED,
            feedback=SystemFeedback(text=message, visual_alert_level="info"),
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        self._last_turn = turn
        self.bus.emit(Event(EventType.TURN_END, {"turn_id": turn_id, "corrected": True}))
        return turn

    # ==================================================================
    # 复合指令
    # ==================================================================
    def _handle_compound(self, event: InteractionEvent, segments: List[str]) -> TurnResult:
        """
        顺序执行一句话里的多个动作。

        **遇错即停**：某一步失败或需要确认时，不再继续后面的动作。
        人听到"复制完然后删除"时，默认假设前一步成功了才会做后一步；
        若前一步实际失败还继续执行，就会做出用户没预期的破坏性操作。
        """
        turn_id = uuid.uuid4().hex[:8]
        started = time.perf_counter()
        self._turn_count += 1
        self.bus.emit(Event(EventType.TURN_START, {"turn_id": turn_id, "compound": len(segments)}))

        turns: List[TurnResult] = []
        stopped_at: Optional[int] = None
        for index, segment in enumerate(segments):
            turn = self._interact_single(
                InteractionEvent(
                    speech=SpeechSignal(transcript=segment),
                    # 手势/视线/表情只作用于第一个动作：复用到后续动作上会
                    # 把"刚才指的那个位置"套用到毫不相干的命令上
                    gestures=event.gestures if index == 0 else [],
                    facial=event.facial if index == 0 else None,
                    head_pose=None,
                    gaze=event.gaze if index == 0 else None,
                    environment_noise_level=event.environment_noise_level,
                ),
                quiet=True,
            )
            turns.append(turn)
            if turn.status in (
                ExecutionStatus.FAILED,
                ExecutionStatus.REJECTED_BY_SAFETY,
                ExecutionStatus.PENDING_CONFIRMATION,
                ExecutionStatus.CANCELLED,
            ):
                stopped_at = index
                break

        spoken = "；".join(t.feedback.text for t in turns)
        all_ok = all(t.status is ExecutionStatus.SUCCESS for t in turns)

        if all_ok:
            level, summary = "success", spoken
        elif stopped_at is not None:
            more = f"（第 {stopped_at + 1}/{len(segments)} 步停下）" if stopped_at < len(segments) - 1 else ""
            level = "warning" if turns[stopped_at].status is ExecutionStatus.PENDING_CONFIRMATION else "error"
            summary = f"{spoken}{more}"
        else:
            level, summary = "warning", spoken

        final_status = (
            ExecutionStatus.SUCCESS if all_ok
            else turns[stopped_at].status if stopped_at is not None
            else ExecutionStatus.FAILED
        )

        turn = TurnResult(
            turn_id=turn_id,
            status=final_status,
            feedback=SystemFeedback(text=summary, visual_alert_level=level),
            executed=all_ok,
            needs_confirmation=final_status is ExecutionStatus.PENDING_CONFIRMATION,
            latency_ms=(time.perf_counter() - started) * 1000,
            stage_timings={},
            errors=[e for t in turns for e in t.errors],
        )
        self._last_turn = turn
        self.bus.emit(Event(EventType.TURN_END, {"turn_id": turn_id, "compound": len(segments)}))
        return turn

    # ==================================================================
    # 条件控制流 (如果...就...否则...)
    # ==================================================================
    def _handle_conditional(self, event: InteractionEvent, flow: ConditionalFlow) -> TurnResult:
        """
        执行带条件的自然控制流。

        1. 执行 flow.initial_action（如"跑测试"或"读取 a.txt"）
        2. 判定其结果是否符合 flow.condition_on（"success" 或 "failure"）
        3. 符合则执行 flow.then_branch，不符合且有 flow.else_branch 则执行 else_branch
        4. 统一反馈给用户自然语言执行链路结果
        """
        turn_id = uuid.uuid4().hex[:8]
        started = time.perf_counter()
        self._turn_count += 1
        self.bus.emit(Event(EventType.TURN_START, {"turn_id": turn_id, "conditional": flow.describe()}))

        # 步骤 1：执行前置条件动作
        first_event = InteractionEvent(
            speech=SpeechSignal(transcript=flow.initial_action),
            gestures=event.gestures,
            facial=event.facial,
            head_pose=event.head_pose,
            gaze=event.gaze,
            environment_noise_level=event.environment_noise_level,
        )
        first_turn = self._interact_single(first_event, quiet=True)

        # 步骤 2：判定条件真假
        is_success = (first_turn.status is ExecutionStatus.SUCCESS)
        matched = (is_success if flow.condition_on == "success" else not is_success)

        second_action = None
        branch_name = ""
        if matched:
            second_action = flow.then_branch
            branch_name = "满足条件"
        elif flow.else_branch:
            second_action = flow.else_branch
            branch_name = "未满足条件，进入否则分支"

        # 步骤 3：执行选中的后续分支（若有）
        if second_action:
            second_event = InteractionEvent(
                speech=SpeechSignal(transcript=second_action),
                environment_noise_level=event.environment_noise_level,
            )
            second_turn = self._interact_single(second_event, quiet=True)
            combined_text = f"前置「{flow.initial_action}」{'成功' if is_success else '失败'}（{first_turn.feedback.text}）；{branch_name}，已执行「{second_action}」：{second_turn.feedback.text}"
            final_status = second_turn.status
            level = second_turn.feedback.visual_alert_level
            errors = first_turn.errors + second_turn.errors
            executed = True
            needs_confirmation = (second_turn.status is ExecutionStatus.PENDING_CONFIRMATION)
        else:
            combined_text = f"前置「{flow.initial_action}」{'成功' if is_success else '失败'}（{first_turn.feedback.text}）；条件未满足且无备选操作。"
            final_status = first_turn.status
            level = first_turn.feedback.visual_alert_level
            errors = first_turn.errors
            executed = is_success
            needs_confirmation = False

        turn = TurnResult(
            turn_id=turn_id,
            status=final_status,
            feedback=SystemFeedback(text=combined_text, visual_alert_level=level),
            executed=executed,
            needs_confirmation=needs_confirmation,
            latency_ms=(time.perf_counter() - started) * 1000,
            stage_timings={},
            errors=errors,
        )
        self._last_turn = turn
        self.bus.emit(Event(EventType.TURN_END, {"turn_id": turn_id, "conditional_result": matched}))
        return turn

    # ==================================================================
    # 九阶段流水线
    # ==================================================================
    def _interact_single(self, event: InteractionEvent, *, quiet: bool = False) -> TurnResult:
        """执行单个动作，完整走九阶段流水线。"""
        turn_id = uuid.uuid4().hex[:8]
        started = time.perf_counter()
        timings: Dict[str, float] = {}
        errors: List[str] = []
        self._turn_count += 1

        if not quiet:
            self.bus.emit(Event(EventType.TURN_START, {"turn_id": turn_id}, EventPriority.NORMAL, "orchestrator"))

        def _mark(stage: str, t0: float) -> float:
            timings[stage] = (time.perf_counter() - t0) * 1000
            return time.perf_counter()

        # ---------- 阶段 1：感知 ----------
        t0 = time.perf_counter()
        frame = self.fusion.assemble(
            speech=event.speech,
            gestures=event.gestures,
            facial=event.facial,
            head_pose=event.head_pose,
            gaze=event.gaze,
            environment_noise_level=event.environment_noise_level,
        )
        self.bus.emit(Event(EventType.PERCEPTION, {"frame_id": frame.frame_id}, source="orchestrator"))
        t0 = _mark("perceive", t0)

        # ---------- 阶段 2：融合 ----------
        try:
            fused = self.fusion.fuse(frame)
        except Exception as exc:
            fused = FusedSignal(confidence=0.0, modalities_used=[], rationale=[f"融合异常：{exc}"])
            errors.append(f"fusion: {type(exc).__name__}: {exc}")
        self.bus.emit(Event(EventType.FUSED, {"confidence": fused.confidence, "rationale": fused.rationale}))
        t0 = _mark("fuse", t0)

        # ---------- 阶段 3：理解 ----------
        text = event.speech.transcript if event.speech else ""
        try:
            intent = self.parser.parse(text, fused)
        except Exception as exc:
            intent = Intent(category=IntentCategory.UNKNOWN, action="unknown", requires_confirmation=True)
            errors.append(f"parse: {type(exc).__name__}: {exc}")
        self.bus.emit(Event(EventType.INTENT, {"action": intent.action, "confidence": intent.confidence}))
        t0 = _mark("interpret", t0)

        # ---------- 阶段 4：对话 ----------
        # 状态快照必须早于 start_turn()：它会把状态重置为 LISTENING，
        # 否则"正在等确认"这个事实会在本轮开始时丢失，点头/摇头就再也接不上了。
        was_awaiting_confirmation = self.dialogue.state is DialogueState.AWAITING_CONFIRMATION
        was_awaiting_clarification = self.dialogue.state is DialogueState.AWAITING_CLARIFICATION
        pending_before = self.dialogue.pending

        # ---------- 槽位续答 ----------
        # 系统问"请补充 path"，用户回一个裸值（"x.txt"）——那是答案，不是新指令。
        # 人补充信息时不会重复动词，说"x.txt"是自然的；要求用户把整句重说一遍
        # 会让交互变得非常难用。
        if was_awaiting_clarification and pending_before is not None and intent.category is IntentCategory.UNKNOWN:
            merged = self._merge_slot_answer(pending_before, text)
            if merged is not None:
                intent = merged
                self.dialogue.clarification = None

        self.dialogue.start_turn()
        if text:
            self.dialogue.register(text)

        modal_affirm = SafetyGate.modal_affirmations(
            text,
            bool(event.head_pose and event.head_pose.nod),
            any(g.gesture in (GestureType.PINCH, GestureType.THUMBS_UP) for g in event.gestures),
        )
        modal_deny = SafetyGate.modal_denials(text, bool(event.head_pose and event.head_pose.shake))

        if was_awaiting_confirmation and modal_affirm and (
            intent.action == "confirm" or intent.category is IntentCategory.UNKNOWN
        ):
            # 授权通过：取回挂起的意图，跳过二次确认直接进入执行链路。
            # 点头/捏合时不带任何文字，意图必然是 UNKNOWN——这正是"点头即同意"的含义。
            if pending_before is not None:
                intent = Intent(
                    **{**pending_before.model_dump(), "requires_confirmation": False}
                )
                fused.requires_confirmation = False
                self.dialogue.pending = None
                self.dialogue.clarification = None
                self.dialogue.state = DialogueState.EXECUTING
            self.bus.emit(Event(EventType.INTENT, {"action": intent.action, "authorized": True}))
        elif was_awaiting_confirmation and modal_deny:
            self.dialogue.pending = None
            self.dialogue.clarification = None
            self.dialogue.state = DialogueState.IDLE
            feedback = SystemFeedback(text="好的，已取消，没有执行。", visual_alert_level="info")
            return self._finish(turn_id, started, timings, errors, intent, fused, None, feedback, ExecutionStatus.CANCELLED)
        else:
            try:
                self.dialogue.accept(intent)
            except Exception as exc:
                errors.append(f"dialogue: {type(exc).__name__}: {exc}")

            # 对话控制类意图：拒绝/取消/撤销，不走工具执行
            control = self._handle_control(intent)
            if control is not None:
                feedback, status = control
                return self._finish(turn_id, started, timings, errors, intent, fused, None, feedback, status)

        if self.dialogue.state is DialogueState.AWAITING_CLARIFICATION:
            question = self.dialogue.clarification.question if self.dialogue.clarification else "请再说明一下。"
            self.bus.emit(Event(EventType.CLARIFY, {"question": question}))
            if self.dialogue.clarification_exhausted():
                question = self.dialogue.fallback_plan()
                self.dialogue.state = DialogueState.IDLE
            feedback = SystemFeedback(
                text=question,
                speech_audio_text=question,
                visual_alert_level="warning",
                sound_cue="prompt",
            )
            return self._finish(
                turn_id, started, timings, errors, intent, fused, None, feedback,
                ExecutionStatus.PENDING_CONFIRMATION, needs_confirmation=True,
            )
        t0 = _mark("dialogue", t0)

        # ---------- 阶段 5：安全 ----------
        command = self._build_command(intent, fused)
        if command is None:
            feedback = SystemFeedback(
                text="我没有找到可以执行的具体工具。请先做一次环境巡检，或注册相应的适配器。",
                visual_alert_level="error",
            )
            return self._finish(turn_id, started, timings, errors, intent, fused, None, feedback, ExecutionStatus.FAILED)

        # 6. 路由必须先于 5. 安全：风险等级来自适配器的能力声明，
        #    拿不到 capability 就等于把所有操作都当成只读，安全闸门形同虚设。
        tool_hint = ACTION_TOOL_HINTS.get(intent.action, "")
        adapter, capability, route_error = self.registry.resolve(command.action, tool_hint)
        if adapter is None:
            feedback = SystemFeedback(
                text=route_error,
                speech_audio_text=route_error,
                visual_alert_level="error",
            )
            self.bus.emit(Event(EventType.ERROR, {"error": route_error}, EventPriority.HIGH))
            return self._finish(turn_id, started, timings, errors, intent, fused, None, feedback, ExecutionStatus.FAILED)

        command.tool_name = adapter.name
        if capability is not None:
            command.safety_level = capability.safety_level
            command.timeout_seconds = capability.timeout_seconds
            command.reversible = capability.reversible
        t0 = _mark("route", t0)

        blocked = self.policy.evaluate(command)
        if blocked is not None:
            category, why = blocked
            feedback = SystemFeedback(text=human_readable(category, why), visual_alert_level="error")
            self.bus.emit(Event(EventType.ERROR, {"error": why}, EventPriority.HIGH))
            return self._finish(turn_id, started, timings, errors, intent, fused, None, feedback, ExecutionStatus.REJECTED_BY_SAFETY)

        decision, why = self.safety.evaluate(
            command,
            intent_confidence=intent.confidence,
            user_authorized=modal_affirm,
            authorization_fresh=modal_affirm,
            has_reversible_plan=bool(command.reversible),
        )
        if fused.requires_confirmation and decision is PermissionDecision.ALLOW:
            decision = PermissionDecision.REQUIRE_CONFIRMATION
            why = "你的多模态信号存在歧义，我需要确认一下"

        if decision is PermissionDecision.REQUIRE_CONFIRMATION:
            self.dialogue.pending = intent
            self.dialogue.state = DialogueState.AWAITING_CONFIRMATION
            self.bus.emit(Event(EventType.CONFIRM_REQUIRED, {"action": command.action, "why": why}, EventPriority.HIGH))
            feedback = SystemFeedback(
                text=why,
                speech_audio_text=f"{why}（说「确认」或点头即可继续）",
                visual_alert_level="warning",
                sound_cue="prompt",
            )
            return self._finish(
                turn_id, started, timings, errors, intent, fused, None, feedback,
                ExecutionStatus.PENDING_CONFIRMATION, needs_confirmation=True,
            )
        t0 = _mark("guard", t0)

        # ---------- 阶段 7：执行 ----------
        self.bus.emit(Event(EventType.EXECUTING, {"tool": adapter.name, "action": command.action}))
        breaker = self.breakers.get(adapter.name)
        try:
            result = self._execute_with_resilience(adapter, command, breaker)
        except CircuitOpenError as exc:
            feedback = SystemFeedback(
                text=f"{adapter.name} 暂时不可用（已连续失败，我已暂停调用它）。{exc.retry_after:.0f} 秒后可重试，或换一个方式？",
                visual_alert_level="error",
                sound_cue="error",
            )
            return self._finish(turn_id, started, timings, errors, intent, fused, None, feedback, ExecutionStatus.FAILED)
        t0 = _mark("execute", t0)

        # ---------- 阶段 8：记账 ----------
        self._record_undo(command, result, adapter)
        if result.status is ExecutionStatus.SUCCESS:
            if intent.grounded_target is not None:
                self.dialogue.remember_target(intent.grounded_target)
            else:
                extracted = self._extract_target_from_command(command)
                if extracted is not None:
                    self.dialogue.remember_target(extracted)
        t0 = _mark("record", t0)

        # ---------- 阶段 9：反馈 ----------
        feedback = self._compose_feedback(intent, result)
        self.dialogue.complete(result.status)
        return self._finish(
            turn_id, started, timings, errors, intent, fused, result, feedback, result.status, executed=True,
        )

    # ==================================================================
    # 辅助
    # ==================================================================
    def _execute_with_resilience(self, adapter, command: ToolCommand, breaker) -> ExecutionResult:
        """执行 = 熔断保护 + 重试 + 超时。"""
        on_retry = lambda attempt, exc, delay: self.bus.emit(
            Event(EventType.RETRY, {"tool": adapter.name, "attempt": attempt, "error": str(exc), "delay": delay})
        )

        if breaker.state.value == "open":
            raise CircuitOpenError(adapter.name, breaker.retry_after())

        # 非幂等操作不自动重试，避免重复副作用
        idempotent = command.action in _IDEMPOTENT_ACTIONS

        def _run() -> ExecutionResult:
            return adapter.execute(command)

        try:
            result = with_retry(
                _run, policy=self.retry_policy, idempotent=idempotent, on_retry=on_retry
            )
        except Exception:
            breaker.record_failure()
            raise
        breaker.record_success()
        return result

    def _build_command(self, intent: Intent, fused: FusedSignal) -> Optional[ToolCommand]:
        """意图 → 工具命令（含参数归一化）。"""
        action = intent.action
        if not action or action == "unknown":
            return None
        if action in CONTROL_ACTIONS:
            return None

        args = dict(intent.parameters or {})

        # 补齐缺失的必填参数：用融合锚点兜底
        if fused.target is not None and not any(args.get(k) for k in ("path", "url", "name", "query", "command")):
            if action in ("focus_window", "launch_app"):
                args["name"] = fused.target.target_id
            elif action == "open_url":
                args["url"] = fused.target.target_id

        # 清理占位/空参数
        args = {k: v for k, v in args.items() if v not in (None, "", [])}

        try:
            return ToolCommand(tool_name=ACTION_TOOL_HINTS.get(action, ""), action=action, args=args)
        except Exception:
            return None

    def _merge_slot_answer(self, pending: Intent, text: str) -> Optional[Intent]:
        """
        把用户的裸值回答并入待补全的意图。

        :returns: 合并后的意图；若这段文本不像在回答问题则返回 ``None``。
        """
        slot = self.dialogue.clarification.target_slot if self.dialogue.clarification else None
        if not slot:
            required = _REQUIRED_SLOTS_FOR.get(pending.action)
            slot = required
        if not slot or not text.strip():
            return None

        answer = text.strip().strip("。.!！,，")
        if not answer:
            return None
        # 回答里又带了一个完整动词 → 那是一条新指令，不该塞进槽位
        if self.parser.parse(answer).category is not IntentCategory.UNKNOWN:
            return None

        params = dict(pending.parameters or {})
        params[slot] = answer
        return Intent(
            **{**pending.model_dump(), "parameters": params, "requires_confirmation": False}
        )

    def _extract_target_from_command(self, command: ToolCommand) -> Optional[GroundingTarget]:
        """
        从成功执行的工具命令中提取实体目标，供持久化工作记忆与指代消解。
        """
        action = command.action
        args = command.args or {}
        path = args.get("path") or args.get("destination") or args.get("source")
        if path and isinstance(path, str):
            target_type = "directory" if (action.endswith("directory") or action == "list_directory") else "file"
            return GroundingTarget(
                target_type=target_type,
                target_id=path,
                label=os.path.basename(path) or path,
                metadata={"action": action, "tool": command.tool_name},
            )
        if "url" in args and isinstance(args["url"], str):
            return GroundingTarget(target_type="url", target_id=args["url"], label=args["url"])
        if "name" in args and isinstance(args["name"], str):
            return GroundingTarget(target_type="window", target_id=args["name"], label=args["name"])
        return None

    def _handle_control(self, intent: Intent) -> Optional[Tuple[SystemFeedback, ExecutionStatus]]:
        """
        处理无需调用工具的对话控制意图：拒绝、取消、撤销。

        「确认」不在此处理——它在阶段 4 被提升为待执行意图，
        这样确认后的操作与直接下指令走完全相同的执行链路。
        """
        action = intent.action

        if action == "confirm":
            return SystemFeedback(
                text="目前没有待确认的操作。",
                visual_alert_level="info",
            ), ExecutionStatus.SUCCESS

        if action in ("reject", "cancel", "stop"):
            self.dialogue.pending = None
            self.dialogue.clarification = None
            self.dialogue.state = DialogueState.IDLE
            if action == "stop":
                self.bus.emit(Event(EventType.ERROR, {"reason": "用户手势停止"}))
                return SystemFeedback(
                    text="已停下。",
                    visual_alert_level="info",
                    haptic_pattern="short",
                ), ExecutionStatus.CANCELLED
            return SystemFeedback(text="好的，已取消。", visual_alert_level="info"), ExecutionStatus.CANCELLED

        if action == "undo":
            outcome = self.undo.undo_last()
            level = "success" if outcome["ok"] else "warning"
            feedback = SystemFeedback(text=outcome["message"], visual_alert_level=level)
            status = ExecutionStatus.SUCCESS if outcome["ok"] else ExecutionStatus.FAILED
            if outcome["ok"]:
                self.bus.emit(Event(EventType.ROLLBACK, outcome))
            return feedback, status

        return None

    def _record_undo(self, command: ToolCommand, result: ExecutionResult, adapter) -> None:
        if result.status is not ExecutionStatus.SUCCESS:
            return

        undo_fn: Optional[Callable[[], bool]] = None
        compensation = Compensation.NONE

        if command.action in ("write_file", "append_file", "delete_file"):
            path_key = command.args.get("path", "")
            fs = self.registry.get("filesystem")
            if fs is not None and hasattr(fs, "restore"):
                snapshot_key = _abs_path(fs.root, path_key)
                undo_fn = lambda: fs.restore(snapshot_key)
                compensation = Compensation.COMPENSATE if command.action != "delete_file" else Compensation.NOTIFY

        elif command.action == "git_commit":
            vcs = self.registry.get("vcs")
            if vcs is not None and hasattr(vcs, "rollback_last_commit"):
                undo_fn = vcs.rollback_last_commit
                compensation = Compensation.COMPENSATE

        elif command.action in ("generate_report", "todo_add", "todo_complete"):
            prod = self.registry.get("productivity")
            if prod is not None and hasattr(prod, "restore"):
                if command.action == "generate_report":
                    out_path = result.data if isinstance(result.data, str) else command.args.get("output_path", "")
                    snap_key = _abs_path(prod.workdir, out_path)
                    undo_fn = lambda: prod.restore(snap_key)
                else:
                    undo_fn = lambda: prod.restore(prod._todos_file)
                compensation = Compensation.COMPENSATE

        if undo_fn is None:
            self.undo.record(command, compensation=Compensation.NONE, note="无逆操作")
        elif compensation is Compensation.COMPENSATE:
            self.undo.record(command, undo_fn=undo_fn, compensation=Compensation.COMPENSATE)
        else:
            self.undo.record(command, undo_fn=undo_fn, compensation=Compensation.NOTIFY,
                             note="删除的文件内容无法恢复")

    def _compose_feedback(self, intent: Intent, result: ExecutionResult) -> SystemFeedback:
        """把执行结果转成自然、多通道的反馈。"""
        if result.status is ExecutionStatus.SUCCESS:
            text = _describe_success(intent, result)
            return SystemFeedback(text=text, speech_audio_text=text, visual_alert_level="success", sound_cue="success")
        if result.status is ExecutionStatus.REJECTED_BY_SAFETY:
            return SystemFeedback(
                text=f"我没有执行这个操作：{result.error}",
                visual_alert_level="error",
                sound_cue="error",
            )

        category = classify_result(result)
        detail = result.error or "执行失败"
        return SystemFeedback(
            text=human_readable(category, f"「{intent.action}」{detail}"),
            speech_audio_text=human_readable(category, f"{intent.action} 执行失败"),
            visual_alert_level="error",
            sound_cue="error",
        )

    def _finish(
        self,
        turn_id: str,
        started: float,
        timings: Dict[str, float],
        errors: List[str],
        intent: Optional[Intent],
        fused: Optional[FusedSignal],
        result: Optional[ExecutionResult],
        feedback: SystemFeedback,
        status: ExecutionStatus,
        *,
        executed: bool = False,
        needs_confirmation: bool = False,
        undone: bool = False,
    ) -> TurnResult:
        latency = (time.perf_counter() - started) * 1000
        turn = TurnResult(
            turn_id=turn_id,
            status=status,
            feedback=feedback,
            intent=intent,
            fused=fused,
            result=result,
            executed=executed,
            needs_confirmation=needs_confirmation,
            latency_ms=latency,
            stage_timings=timings,
            errors=errors,
            undone=undone,
        )
        self._last_turn = turn
        self.bus.emit(Event(
            EventType.RESULT,
            {
                "turn_id": turn_id,
                "status": status.value,
                "executed": executed,
                "action": intent.action if intent else None,
                "latency_ms": latency,
                "error": result.error if result else None,
            },
        ))
        self.bus.emit(Event(
            EventType.RESPONSE,
            {
                "text": feedback.text,
                "status": status.value,
                # 完整反馈对象：订阅方（TTS / 屏幕 / 触觉）需要的是
                # speech_audio_text、sound_cue、visual_alert_level 等全部通道，
                # 只给 text 会让这些反馈无处可取
                "feedback": feedback.model_dump(),
            },
        ))
        self.bus.emit(Event(EventType.TURN_END, {"turn_id": turn_id, "latency_ms": latency, "errors": errors}))
        return turn

    # ==================================================================
    # 诊断
    # ==================================================================
    def health(self) -> Dict[str, Any]:
        return {
            "adapters": self.registry.health_check(),
            "breakers": self.breakers.snapshot(),
            "dialogue_state": self.dialogue.state.value,
            "undo_stack": len(self.undo.snapshot()),
            "bus_errors": len(self.bus.errors()),
            "turns": self._turn_count,
        }

    def undo_last(self) -> Dict[str, Any]:
        return self.undo.undo_last()

    def last_turn(self) -> Optional[TurnResult]:
        return self._last_turn


_IDEMPOTENT_ACTIONS = frozenset({
    "read_file", "list_directory", "list_windows", "system_info", "git_status",
    "git_diff", "git_log", "search_in_file", "get_working_directory", "history",
    "table_aggregate", "table_query", "extract_outline", "todo_list",
    "list_mcp_tools", "list_editors",
})


# 用于把"动作 + 已更正槽位"还原成一句可再次解析的话。
# 之所以走文本而���直接构造 Intent，是因为解析层还负责安全判定与路由，
# 绕开它会让更正后的动作跳过同一套检查。
_ACTION_UTTERANCES = {
    "read_file": "读取 {path}",
    "write_file": "把 {content} 写入 {path}",
    "append_file": "把 {content} 追加到 {path}",
    "delete_file": "删除文件 {path}",
    "copy_file": "把 {source} 复制到 {destination}",
    "move_file": "把 {source} 移动到 {destination}",
    "create_directory": "创建目录 {path}",
    "search_in_file": "在 {path} 里查找 {keyword}",
    "launch_app": "打开 {app}",
    "focus_window": "切换到 {name} 窗口",
    "open_url": "打开 {url}",
    "search_web": "搜索 {query}",
    "run_command": "运行 {command}",
    "git_commit": "提交代码 {message}",
}


def _rebuild_utterance(action: str, params: Dict[str, Any]) -> Optional[str]:
    """把更正后的槽位填回动作模板，生成可再次解析的指令文本。"""
    template = _ACTION_UTTERANCES.get(action)
    if not template:
        return None
    try:
        rendered = template.format(**{k: ("" if v is None else v) for k, v in params.items()})
    except (KeyError, IndexError):
        return None
    return rendered if "{" not in rendered else None


def _abs_path(root: str, rel: str) -> str:
    import os

    return os.path.abspath(os.path.join(root, rel))


def _describe_success(intent: Intent, result: ExecutionResult) -> str:
    """把执行结果描述成人话。"""
    data = result.data
    action = intent.action

    if action == "list_windows" and isinstance(data, list):
        return f"当前有 {len(data)} 个窗口：" + "、".join(
            str(w.get("title") or w.get("id")) for w in data[:5]
        ) + ("…" if len(data) > 5 else "")
    if action == "read_file" and isinstance(data, str):
        preview = data.strip().splitlines()
        head = preview[0][:60] if preview else "(空文件)"
        more = f"（共 {len(preview)} 行）" if len(preview) > 1 else ""
        return f"已读取：{head}{more}"
    if action == "search_in_file" and isinstance(data, list):
        return f"找到 {len(data)} 处匹配" + (f"，首处在第 {data[0]['line']} 行" if data else "")
    if action == "list_directory" and isinstance(data, list):
        dirs = sum(1 for e in data if isinstance(e, dict) and e.get("is_dir"))
        return f"{len(data)} 个条目（{dirs} 个目录）"
    if action == "run_command" and isinstance(data, dict):
        rc = data.get("returncode")
        out = (data.get("stdout") or "").strip().splitlines()
        tail = out[-1][:100] if out else "(无输出)"
        return f"命令执行{'成功' if rc == 0 else f'失败(退出码 {rc})'}：{tail}"
    if action == "search_web":
        return f"已在浏览器中搜索：{intent.parameters.get('query', '')}"
    if action == "open_url":
        return f"已打开：{data.get('url') if isinstance(data, dict) else intent.parameters.get('url', '')}"
    if action == "focus_window":
        return f"已切换到窗口：{intent.parameters.get('name', '')}"
    if action == "launch_app":
        return f"已启动：{intent.parameters.get('app', '')}"
    if action == "git_status" and isinstance(data, dict):
        if data.get("clean"):
            return f"仓库 {data.get('branch', '')} 状态干净"
        return f"仓库 {data.get('branch', '')} 有 {len(data.get('changes', []))} 处改动"
    if action in ("write_file", "append_file"):
        return f"已{'写入' if action == 'write_file' else '追加'} {intent.parameters.get('path', '')}"
    if action == "copy_file":
        return f"已复制 {intent.parameters.get('source', '')} → {intent.parameters.get('destination', '')}"
    if action == "move_file":
        return f"已移动 {intent.parameters.get('source', '')} → {intent.parameters.get('destination', '')}"
    if action == "create_directory":
        return f"已创建目录 {intent.parameters.get('path', '')}"
    if action == "delete_file":
        return f"已删除 {intent.parameters.get('path', '')}（此操作不可恢复）"
    if action == "table_aggregate" and isinstance(data, dict):
        return f"已统计 {data.get('file')} 的 {data.get('column')} 列：{data.get('operation')} = {data.get('result')}"
    if action == "table_query" and isinstance(data, list):
        return f"查询到 {len(data)} 条匹配记录"
    if action == "generate_report":
        return f"已自动生成结构化工作报告：{data}"
    if action == "extract_outline" and isinstance(data, list):
        return f"已提取大纲结构，共 {len(data)} 个章节层级"
    if action == "todo_add" and isinstance(data, dict):
        return f"已记录待办「{data.get('task')}」（编号 {data.get('id')}）"
    if action == "todo_list" and isinstance(data, list):
        pending = sum(1 for t in data if not t.get("completed"))
        return f"当前待办任务共 {len(data)} 项（未完成 {pending} 项）"
    if action == "todo_complete" and isinstance(data, dict):
        return f"已将待办「{data.get('task')}」标记为完成"

    return f"已完成：{action}"