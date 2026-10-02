"""韧性层测试：错误分类、重试、熔断、撤销、安全闸门。"""

from __future__ import annotations

import errno
import time

import pytest

from futurework.resilience.circuit import BreakerRegistry, CircuitBreaker, CircuitOpenError, CircuitState
from futurework.resilience.errors import (
    ErrorCategory,
    FutureWorkError,
    classify_exception,
    classify_result,
    human_readable,
    should_retry,
)
from futurework.resilience.retry import RetryPolicy, with_retry
from futurework.resilience.safety import (
    PermissionDecision,
    PermissionProfile,
    SafetyGate,
    SecurityPolicy,
)
from futurework.resilience.undo import Compensation, UndoManager
from futurework.types import ExecutionResult, ExecutionStatus, SafetyLevel, ToolCommand


# ============================================================================
# 错误分类
# ============================================================================
class TestErrorClassification:
    @pytest.mark.parametrize("exc,category", [
        (FileNotFoundError("x"), ErrorCategory.NOT_FOUND),
        (PermissionError("x"), ErrorCategory.PERMISSION),
        (TimeoutError("x"), ErrorCategory.TIMEOUT),
        (NotImplementedError("x"), ErrorCategory.PERMANENT),
        (ConnectionResetError("x"), ErrorCategory.TRANSIENT),
        (RuntimeError("x"), ErrorCategory.INTERNAL),
    ])
    def test_classify_exception(self, exc, category):
        assert classify_exception(exc) is category

    def test_oserror_errno_transient(self):
        exc = OSError(errno.EAGAIN, "resource temporarily unavailable")
        assert classify_exception(exc) is ErrorCategory.TRANSIENT

    def test_oserror_errno_permission(self):
        exc = OSError(errno.EACCES, "denied")
        assert classify_exception(exc) is ErrorCategory.PERMISSION

    def test_fww_error_keeps_its_category(self):
        err = FutureWorkError("x", ErrorCategory.SAFETY)
        assert classify_exception(err) is ErrorCategory.SAFETY

    @pytest.mark.parametrize("text,category", [
        ("文件不存在: a.txt", ErrorCategory.NOT_FOUND),
        ("operation timed out", ErrorCategory.TIMEOUT),
        ("permission denied", ErrorCategory.PERMISSION),
        ("该操作未实现", ErrorCategory.PERMANENT),
    ])
    def test_classify_result(self, text, category):
        assert classify_result(ExecutionResult(command_id="1", status=ExecutionStatus.FAILED, error=text)) is category

    def test_rejected_by_safety_is_safety(self):
        result = ExecutionResult(command_id="1", status=ExecutionStatus.REJECTED_BY_SAFETY, error="危险")
        assert classify_result(result) is ErrorCategory.SAFETY

    def test_human_readable_is_actionable(self):
        for category in ErrorCategory:
            text = human_readable(category)
            assert len(text) > 5, f"{category} 缺少用户可读说明"

    def test_should_retry_policy(self):
        assert should_retry(ErrorCategory.TRANSIENT, 0, 3) is True
        assert should_retry(ErrorCategory.TIMEOUT, 0, 3) is True
        assert should_retry(ErrorCategory.SAFETY, 0, 3) is False
        assert should_retry(ErrorCategory.PERMISSION, 0, 3) is False
        assert should_retry(ErrorCategory.TRANSIENT, 3, 3) is False


# ============================================================================
# 重试
# ============================================================================
class TestRetry:
    def test_succeeds_after_transient_failures(self):
        attempts = {"n": 0}

        def flaky():
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise ConnectionResetError("boom")
            return "ok"

        assert with_retry(flaky, policy=RetryPolicy(max_attempts=3, base_delay=0.001)) == "ok"
        assert attempts["n"] == 3

    def test_does_not_retry_permanent_errors(self):
        attempts = {"n": 0}

        def bad():
            attempts["n"] += 1
            raise ValueError("永久错误")

        with pytest.raises(ValueError):
            with_retry(bad, policy=RetryPolicy(max_attempts=5, base_delay=0.001))
        assert attempts["n"] == 1

    def test_non_idempotent_runs_once(self):
        """非幂等操作重试会产生重复副作用——绝不能重试。"""
        attempts = {"n": 0}

        def write():
            attempts["n"] += 1
            raise ConnectionResetError("写失败了")

        with pytest.raises(ConnectionResetError):
            with_retry(write, policy=RetryPolicy(max_attempts=5, base_delay=0.001), idempotent=False)
        assert attempts["n"] == 1

    def test_gives_up_after_max_attempts(self):
        attempts = {"n": 0}

        def always_fail():
            attempts["n"] += 1
            raise ConnectionResetError("一直失败")

        with pytest.raises(ConnectionResetError):
            with_retry(always_fail, policy=RetryPolicy(max_attempts=2, base_delay=0.001))
        assert attempts["n"] == 2

    def test_total_timeout_bounds_waiting(self):
        policy = RetryPolicy(max_attempts=100, base_delay=0.05, total_timeout=0.12)

        def always_fail():
            raise ConnectionResetError("x")

        start = time.perf_counter()
        with pytest.raises(ConnectionResetError):
            with_retry(always_fail, policy=policy)
        assert time.perf_counter() - start < 1.0

    def test_delay_grows_and_is_capped(self):
        policy = RetryPolicy(base_delay=0.1, multiplier=2.0, max_delay=0.5, jitter=0.0)
        assert policy.delay_for(0) == pytest.approx(0.1)
        assert policy.delay_for(1) == pytest.approx(0.2)
        assert policy.delay_for(9) == pytest.approx(0.5)   # 封顶

    def test_on_retry_callback_fires(self):
        seen = []
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 2:
                raise ConnectionResetError("x")
            return 1

        with_retry(flaky, policy=RetryPolicy(max_attempts=3, base_delay=0.001),
                   on_retry=lambda a, e, d: seen.append((a, str(e))))
        assert seen and seen[0][0] == 1


# ============================================================================
# 熔断
# ============================================================================
class TestCircuitBreaker:
    def test_opens_after_threshold(self):
        breaker = CircuitBreaker(name="t", failure_threshold=3)
        for _ in range(3):
            breaker.record_failure()
        assert breaker.state is CircuitState.OPEN

    def test_open_circuit_fails_fast(self):
        breaker = CircuitBreaker(name="t", failure_threshold=1)
        breaker.record_failure()
        start = time.perf_counter()
        with pytest.raises(CircuitOpenError):
            breaker.call(lambda: time.sleep(5))
        assert time.perf_counter() - start < 0.1   # 立即失败，不等待

    def test_half_open_after_cooldown(self):
        breaker = CircuitBreaker(name="t", failure_threshold=1, recovery_timeout=0.05)
        breaker.record_failure()
        assert breaker.state is CircuitState.OPEN
        time.sleep(0.06)
        assert breaker.state is CircuitState.HALF_OPEN

    def test_recovers_after_success_in_half_open(self):
        breaker = CircuitBreaker(name="t", failure_threshold=1, recovery_timeout=0.05)
        breaker.record_failure()
        time.sleep(0.06)
        breaker.record_success()
        assert breaker.state is CircuitState.CLOSED

    def test_reopens_on_failure_in_half_open(self):
        breaker = CircuitBreaker(name="t", failure_threshold=1, recovery_timeout=0.05)
        breaker.record_failure()
        time.sleep(0.06)
        breaker.record_failure()
        assert breaker.state is CircuitState.OPEN

    def test_single_success_does_not_close(self):
        breaker = CircuitBreaker(name="t", failure_threshold=1, recovery_timeout=0.05)
        breaker.record_failure()
        time.sleep(0.06)
        breaker.record_success()
        breaker.record_failure()   # 又失败了
        assert breaker.state is CircuitState.OPEN

    def test_reset_clears(self):
        breaker = CircuitBreaker(name="t", failure_threshold=1)
        breaker.record_failure()
        breaker.reset()
        assert breaker.state is CircuitState.CLOSED

    def test_registry_manages_multiple(self):
        registry = BreakerRegistry(failure_threshold=2)
        registry.get("a").record_failure()
        registry.get("a").record_failure()
        registry.get("b").record_failure()
        assert registry.snapshot()["a"]["state"] == "open"
        assert registry.snapshot()["b"]["state"] == "closed"


# ============================================================================
# 撤销
# ============================================================================
class TestUndo:
    def _cmd(self, action="write_file", **args):
        return ToolCommand(tool_name="filesystem", action=action, args=args)

    def test_readonly_not_recorded(self):
        """只读操作不入栈——否则撤销会指向"查看"而不是"改动"。"""
        undo = UndoManager()
        undo.record(self._cmd("read_file"))
        assert undo.undo_last()["ok"] is False

    def test_compensatable_undo(self):
        undo = UndoManager()
        undo.record(self._cmd(), undo_fn=lambda: True, compensation=Compensation.COMPENSATE)
        assert undo.undo_last()["ok"] is True

    def test_skips_readonly_entries_when_undoing(self):
        undo = UndoManager()
        undo.record(self._cmd(), undo_fn=lambda: True, compensation=Compensation.COMPENSATE)
        undo.record(self._cmd("read_file"))     # 只读
        assert undo.undo_last()["ok"] is True

    def test_non_compensatable_explains_why(self):
        undo = UndoManager()
        undo.record(self._cmd("delete_file"), undo_fn=lambda: True,
                    compensation=Compensation.NOTIFY, note="文件内容无法恢复")
        outcome = undo.undo_last()
        assert outcome["ok"] is False
        assert "无法恢复" in outcome["message"]

    def test_undo_fn_failure_reported(self):
        undo = UndoManager()
        undo.record(self._cmd(), undo_fn=lambda: False, compensation=Compensation.COMPENSATE)
        assert undo.undo_last()["ok"] is False

    def test_undo_fn_exception_isolated(self):
        def boom():
            raise RuntimeError("撤销时炸了")

        undo = UndoManager()
        undo.record(self._cmd(), undo_fn=boom, compensation=Compensation.COMPENSATE)
        outcome = undo.undo_last()
        assert outcome["ok"] is False and "撤销失败" in outcome["message"]

    def test_undo_todo(self):
        undo = UndoManager()
        cmd_a = self._cmd()
        undo.record(cmd_a, undo_fn=lambda: True, compensation=Compensation.COMPENSATE)
        cmd_b = self._cmd()
        undo.record(cmd_b, undo_fn=lambda: True, compensation=Compensation.COMPENSATE)
        outcome = undo.undo_to(cmd_a.command_id)
        assert outcome["ok"] is True and len(outcome["reverted"]) == 2

    def test_undo_todo_unknown_id(self):
        undo = UndoManager()
        assert undo.undo_to("nope")["ok"] is False

    def test_capacity_enforced(self):
        undo = UndoManager(capacity=3)
        for _ in range(10):
            undo.record(self._cmd(), undo_fn=lambda: True, compensation=Compensation.COMPENSATE)
        assert len(undo.snapshot()) == 3

    def test_undone_entry_not_repeated(self):
        undo = UndoManager()
        undo.record(self._cmd(), undo_fn=lambda: True, compensation=Compensation.COMPENSATE)
        undo.undo_last()
        assert undo.undo_last()["ok"] is False

    def test_record_result_skips_failure(self):
        undo = UndoManager()
        result = ExecutionResult(command_id="x", status=ExecutionStatus.FAILED)
        assert undo.record_result(self._cmd(), result, undo_fn=lambda: True) is None


# ============================================================================
# 安全闸门
# ============================================================================
class TestSafetyGate:
    def _cmd(self, level=SafetyLevel.SAFE_WRITE, action="write_file"):
        return ToolCommand(tool_name="t", action=action, safety_level=level)

    def test_readonly_auto_allowed(self):
        decision, _ = SafetyGate().evaluate(self._cmd(SafetyLevel.READ_ONLY, "list_windows"))
        assert decision is PermissionDecision.ALLOW

    def test_destructive_requires_confirmation(self):
        decision, why = SafetyGate().evaluate(self._cmd(SafetyLevel.CRITICAL_DESTRUCTIVE, "delete_file"))
        assert decision is PermissionDecision.REQUIRE_CONFIRMATION
        assert "确认" in why

    def test_destructive_allowed_after_explicit_authorization(self):
        decision, _ = SafetyGate().evaluate(
            self._cmd(SafetyLevel.CRITICAL_DESTRUCTIVE, "delete_file"),
            intent_confidence=0.99, user_authorized=True, authorization_fresh=True,
        )
        assert decision is PermissionDecision.ALLOW

    def test_low_confidence_even_after_nod_asks_again(self):
        """点了头但我没听懂 —— 此时执行比不执行更危险。"""
        decision, _ = SafetyGate().evaluate(
            self._cmd(SafetyLevel.CRITICAL_DESTRUCTIVE, "delete_file"),
            intent_confidence=0.3, user_authorized=True, authorization_fresh=True,
        )
        assert decision is PermissionDecision.REQUIRE_CONFIRMATION

    def test_reversible_write_auto_allowed(self):
        decision, _ = SafetyGate().evaluate(self._cmd(), has_reversible_plan=True)
        assert decision is PermissionDecision.ALLOW

    def test_profile_auto_allow(self):
        gate = SafetyGate()
        profile = PermissionProfile()
        profile.auto_allow.add("read_file")
        gate.add_profile(profile)
        decision, _ = gate.evaluate(self._cmd(SafetyLevel.SENSITIVE_MODIFY, "read_file"))
        assert decision is PermissionDecision.ALLOW

    def test_profile_always_confirm_overrides_confidence(self):
        gate = SafetyGate()
        profile = PermissionProfile()
        profile.always_confirm.add("delete_file")
        gate.add_profile(profile)
        decision, _ = gate.evaluate(
            self._cmd(SafetyLevel.READ_ONLY, "delete_file"),
            intent_confidence=1.0, user_authorized=True, authorization_fresh=True,
        )
        assert decision is PermissionDecision.REQUIRE_CONFIRMATION

    @pytest.mark.parametrize("text", ["是", "对", "好", "嗯", "确认", "确定", "可以", "yes", "ok", "sure", "执行"])
    def test_modal_affirmations_accepted(self, text):
        assert SafetyGate.modal_affirmations(text) is True

    def test_nod_counts_as_affirmation(self):
        assert SafetyGate.modal_affirmations("", nod=True) is True

    def test_pinch_counts_as_affirmation(self):
        assert SafetyGate.modal_affirmations("", pinch=True) is True

    @pytest.mark.parametrize("text", ["不是", "不要", "取消", "停", "no", "stop"])
    def test_modal_denials_accepted(self, text):
        assert SafetyGate.modal_denials(text) is True

    def test_shake_counts_as_denial(self):
        assert SafetyGate.modal_denials("", shake=True) is True

    def test_random_text_is_neither(self):
        assert SafetyGate.modal_affirmations("打开浏览器") is False
        assert SafetyGate.modal_denials("打开浏览器") is False


class TestSecurityPolicy:
    @pytest.mark.parametrize("command", [
        "rm -rf /",
        "mkfs.ext4",
        ":(){:|:&};:",
        "shutdown now",
    ])
    def test_blocked_commands(self, command):
        policy = SecurityPolicy()
        cmd = ToolCommand(tool_name="terminal", action="run_command", args={"command": command})
        assert policy.is_blocked(cmd) is True

    def test_normal_command_allowed(self):
        cmd = ToolCommand(tool_name="terminal", action="run_command", args={"command": "ls -la"})
        assert SecurityPolicy().is_blocked(cmd) is False

    def test_rate_limit(self):
        policy = SecurityPolicy(max_commands_per_minute=3)
        for _ in range(3):
            assert policy.rate_ok() is True
        assert policy.rate_ok() is False

    def test_rate_limit_window_expires(self):
        policy = SecurityPolicy(max_commands_per_minute=1)
        assert policy.rate_ok(now=1000.0) is True
        assert policy.rate_ok(now=1061.0) is True

    def test_evaluate_returns_reason(self):
        policy = SecurityPolicy()
        cmd = ToolCommand(tool_name="terminal", action="run_command", args={"command": "rm -rf /"})
        category, why = policy.evaluate(cmd)
        assert category is ErrorCategory.SAFETY and why

    def test_evaluate_passes_normal(self):
        policy = SecurityPolicy()
        cmd = ToolCommand(tool_name="terminal", action="run_command", args={"command": "ls"})
        assert policy.evaluate(cmd) is None