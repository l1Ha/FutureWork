"""
安全闸门：按风险等级决定"自动执行"还是"征求确认"。

设计立场：**默认安全，但不打断流畅**。
* 只读操作零打扰——否则用户会被确认弹窗淹没，系统就不好用了；
* 高风险操作必须明确授权，且授权可以来自**任何**模态：
  口头"是"、点头、捏合手势——只要用户表达了明确的同意。

授权凭据支持时效性（``授权一次`` 只对下一次操作有效），
避免"我现在同意了"被误当成永久授权。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from futurework.resilience.errors import ErrorCategory, classify_exception
from futurework.resilience.undo import Compensation
from futurework.types import SafetyLevel, ToolCommand


class PermissionDecision(str, Enum):
    ALLOW = "allow"
    REQUIRE_CONFIRMATION = "require_confirmation"
    DENY = "deny"


@dataclass
class PermissionProfile:
    """
    权限档案：用户/会话级的授权偏好。

    ``auto_allow`` 里的动作永久放行（如"总是允许读文件"）；
    ``always_confirm`` 里的动作永远要问（如删除）。
    """

    name: str = "default"
    auto_allow: set = field(default_factory=set)
    always_confirm: set = field(default_factory=set)
    trust_confidence_threshold: float = 0.9
    trust_nods: bool = True
    trust_pinch: bool = True

    def allows_always(self, action: str) -> bool:
        return action in self.auto_allow

    def confirms_always(self, action: str) -> bool:
        return action in self.always_confirm


@dataclass
class SafetyGate:
    """
    安全闸门。

    判定逻辑（自上而下）：
    1. 显式禁止 → DENY；
    2. 档案中"总是确认" → REQUIRE_CONFIRMATION；
    3. 高风险等级 → REQUIRE_CONFIRMATION（除非用户已明确授权）；
    4. 中风险 + 高置信 + 用户已通过任一模态确认 → ALLOW；
    5. 其余按等级与置信度决定。
    """

    profiles: Dict[str, PermissionProfile] = field(default_factory=lambda: {"default": PermissionProfile()})

    def add_profile(self, profile: PermissionProfile) -> None:
        self.profiles[profile.name] = profile

    def profile(self, name: str = "default") -> PermissionProfile:
        return self.profiles.setdefault(name, PermissionProfile(name=name))

    # ------------------------------------------------------------------
    def evaluate(
        self,
        command: ToolCommand,
        *,
        profile_name: str = "default",
        intent_confidence: float = 1.0,
        user_authorized: bool = False,
        authorization_fresh: bool = False,
        has_reversible_plan: bool = False,
    ) -> Tuple[PermissionDecision, str]:
        """返回 ``(决策, 面向用户的说明)``。"""
        profile = self.profile(profile_name)
        action = command.action

        if profile.allows_always(action) and intent_confidence >= profile.trust_confidence_threshold:
            return PermissionDecision.ALLOW, "该操作已在权限档案中设为自动允许"

        if profile.confirms_always(action):
            return PermissionDecision.REQUIRE_CONFIRMATION, "你在权限设置中要求这类操作必须确认"

        if user_authorized and authorization_fresh:
            if intent_confidence < 0.5:
                return PermissionDecision.REQUIRE_CONFIRMATION, "你已授权，但我对指令的理解不够确定，请再确认一次要做什么"
            return PermissionDecision.ALLOW, "已获得你的明确授权"

        level = command.safety_level
        if level is SafetyLevel.READ_ONLY:
            return PermissionDecision.ALLOW, "只读操作，不影响系统状态"

        if level is SafetyLevel.SAFE_WRITE:
            if has_reversible_plan:
                return PermissionDecision.ALLOW, "可逆操作，已登记撤销"
            if intent_confidence >= 0.95:
                return PermissionDecision.ALLOW, "指令明确，已执行（保持可回退）"
            return PermissionDecision.REQUIRE_CONFIRMATION, "这个操作我理解得不够确定，确认一下吗？"

        if level is SafetyLevel.SENSITIVE_MODIFY:
            if intent_confidence >= profile.trust_confidence_threshold and authorization_fresh:
                return PermissionDecision.ALLOW, "高置信 + 已授权"
            return PermissionDecision.REQUIRE_CONFIRMATION, "这会修改现有内容，确认执行吗？"

        # CRITICAL_DESTRUCTIVE
        if intent_confidence >= 0.95 and authorization_fresh:
            return PermissionDecision.ALLOW, "破坏性操作已获得明确授权"
        return PermissionDecision.REQUIRE_CONFIRMATION, (
            "这是一个不可逆的操作（"
            f"{command.tool_name}.{action}"
            "）。你说「确认」或点头，我才继续。"
        )

    # ------------------------------------------------------------------
    @staticmethod
    def modal_affirmations(speech_text: str = "", nod: bool = False, pinch: bool = False) -> bool:
        """
        判断用户是否通过任一模态表达了确认。

        接受口头肯定、点头、捏合三种方式——这正是"像人一样交流"：
        人不会在电脑前点击"确定"，而是说"嗯"或者点点头。
        """
        if nod or pinch:
            return True
        text = (speech_text or "").strip().lower().rstrip("。.!！")
        affirmations = {
            "是", "对", "好", "行", "嗯", "确认", "确定", "可以", "没错", "同意", "就这样", "执行", "继续",
            "yes", "ok", "okay", "sure", "confirm", "do it", "go ahead", "proceed",
        }
        return text in affirmations

    @staticmethod
    def modal_denials(speech_text: str = "", shake: bool = False) -> bool:
        """判断用户是否表达了拒绝。"""
        if shake:
            return True
        text = (speech_text or "").strip().lower().rstrip("。.!！")
        denials = {"不", "不是", "不", "不用", "不要", "别", "不行", "算了", "取消", "停", "no", "nope", "stop", "cancel"}
        return text in denials


@dataclass
class SecurityPolicy:
    """
    额外的安全约束：命令级黑名单与速率限制。

    黑名单**不在这里另写一份**——直接复用终端适配器的 ``inspect_command``。
    两处各维护一套必然出现"改了一处忘了另一处"，而这类不一致的后果是
    同一条命令走不同路径会得到两种不同的拒绝结果，甚至某一路径完全不拦。
    本层只负责速率限制与跨工具的通用黑名单。
    """

    max_commands_per_minute: int = 120

    _timestamps: List[float] = field(default_factory=list)

    def is_blocked(self, command: ToolCommand) -> bool:
        return self.block_reason(command) is not None

    def block_reason(self, command: ToolCommand) -> Optional[str]:
        """
        返回触发拦截的具体原因。

        只报动作名（``run_command``）对用户毫无意义——他知道自己说了什么，
        他需要知道的是"你这句话里的哪部分让我不敢执行"。
        """
        if command.action in ("run_command", "call_mcp_tool"):
            text = str((command.args or {}).get("command", ""))
            if text:
                # 复用终端层的同一份判定，保证两处结论一致
                from futurework.tools.terminal_adapter import inspect_command

                safe, why = inspect_command(text)
                if not safe:
                    return why
        return None

    def rate_ok(self, now: Optional[float] = None) -> bool:
        ts = now or time.time()
        self._timestamps = [t for t in self._timestamps if ts - t < 60.0]
        if len(self._timestamps) >= self.max_commands_per_minute:
            return False
        self._timestamps.append(ts)
        return True

    def evaluate(self, command: ToolCommand) -> Optional[Tuple[ErrorCategory, str]]:
        """返回 ``None`` 表示放行。"""
        reason = self.block_reason(command)
        if reason is not None:
            return ErrorCategory.SAFETY, reason
        if not self.rate_ok():
            return ErrorCategory.SAFETY, "操作过于频繁，已触发速率限制，请稍候"
        return None