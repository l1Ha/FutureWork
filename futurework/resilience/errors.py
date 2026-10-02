"""
统一异常分类。

为什么需要它：``FileNotFoundError`` 和 ``PermissionError`` 在重试层面
是完全不同的——前者重试一万次也一样，后者可能是临时锁文件。
把异常映射到 7 个类别后，韧性层就能自动决定重试、降级还是要求用户介入。

面向用户的文案（``human_readable``）刻意避免暴露技术细节，
因为交互目标是"像人一样自然"，不是"打印 traceback"。
"""

from __future__ import annotations

import errno
import socket
from enum import Enum
from typing import Any, Optional

from futurework.types import ExecutionResult, ExecutionStatus


class ErrorCategory(str, Enum):
    """异常类别，决定恢复策略。"""

    TRANSIENT = "transient"          # 瞬态：网络抖动、临时资源占用 → 重试
    PERMANENT = "permanent"          # 永久：参数错误、工具不支持 → 放弃并反馈
    NOT_FOUND = "not_found"          # 资源不存在 → 引导用户创建
    PERMISSION = "permission"        # 权限不足 → 引导用户授权
    TIMEOUT = "timeout"              # 超时 → 重试或降级
    SAFETY = "safety"                # 安全策略拦截 → 绝不重试，转为询问用户
    INTERNAL = "internal"            # 适配器内部错误 → 记录并降级


class FutureWorkError(Exception):
    """带分类的统一异常。"""

    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.INTERNAL,
        *,
        recoverable: bool = False,
        suggestion: str = "",
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.category = category
        self.recoverable = recoverable
        self.suggestion = suggestion
        self.cause = cause

    def __str__(self) -> str:
        return f"[{self.category.value}] {self.message}"

    def to_dict(self) -> dict:
        return {
            "message": self.message,
            "category": self.category.value,
            "recoverable": self.recoverable,
            "suggestion": self.suggestion,
        }


# 瞬态 errno：网络类
_TRANSIENT_ERRNOS = {
    errno.EAGAIN, errno.EWOULDBLOCK, errno.EINTR,
    errno.ETIMEDOUT, errno.ECONNRESET, errno.ECONNREFUSED,
    errno.ENETDOWN, errno.ENETUNREACH, errno.EHOSTUNREACH,
}

_PERMISSION_ERRNOS = {errno.EACCES, errno.EPERM}
_NOT_FOUND_ERRNOS = {errno.ENOENT, errno.ENOTDIR}


def classify_exception(exc: BaseException) -> ErrorCategory:
    """把任意异常映射到类别。"""
    if isinstance(exc, FutureWorkError):
        return exc.category
    if isinstance(exc, TimeoutError):
        return ErrorCategory.TIMEOUT
    if isinstance(exc, FileNotFoundError):
        return ErrorCategory.NOT_FOUND
    if isinstance(exc, PermissionError):
        return ErrorCategory.PERMISSION
    if isinstance(exc, NotImplementedError):
        return ErrorCategory.PERMANENT
    if isinstance(exc, (BrokenPipeError, ConnectionError, socket.timeout)):
        return ErrorCategory.TRANSIENT
    if isinstance(exc, OSError):
        code = exc.errno
        if code in _TRANSIENT_ERRNOS:
            return ErrorCategory.TRANSIENT
        if code in _PERMISSION_ERRNOS:
            return ErrorCategory.PERMISSION
        if code in _NOT_FOUND_ERRNOS:
            return ErrorCategory.NOT_FOUND
        return ErrorCategory.INTERNAL
    return ErrorCategory.INTERNAL


def classify_result(result: ExecutionResult) -> ErrorCategory:
    """把执行结果映射到类别。"""
    if result.status is ExecutionStatus.SUCCESS:
        return ErrorCategory.TRANSIENT  # 占位：成功时不会用到
    if result.status is ExecutionStatus.REJECTED_BY_SAFETY:
        return ErrorCategory.SAFETY
    if result.status is ExecutionStatus.CANCELLED:
        return ErrorCategory.PERMANENT
    text = (result.error or "").lower()
    if "timeout" in text or "timed out" in text or "超时" in text:
        return ErrorCategory.TIMEOUT
    if "不存在" in text or "not found" in text or "no such" in text:
        return ErrorCategory.NOT_FOUND
    if "permission" in text or "权限" in text or "access denied" in text:
        return ErrorCategory.PERMISSION
    if any(k in text for k in ("timeout", "temporarily", "busy", "resource", "再试")):
        return ErrorCategory.TRANSIENT
    if "不支持" in text or "未实现" in text or "not supported" in text or "unimplemented" in text:
        return ErrorCategory.PERMANENT
    return ErrorCategory.INTERNAL


# 面向用户的建议：说人话，给可执行的下一步
_SUGGESTIONS: dict = {
    ErrorCategory.TRANSIENT: "资源暂时被占用，我已经自动重试；如果仍失败，稍后再试一次即可。",
    ErrorCategory.TIMEOUT: "这一步耗时超出预期。可以再说一次，或把范围缩小（例如只处理一个文件）。",
    ErrorCategory.NOT_FOUND: "没有找到对应的东西。请确认名称或路径是否正确，或者我帮你列出可选项？",
    ErrorCategory.PERMISSION: "当前权限不足。你可以授权后重试，或告诉我换一个方式。",
    ErrorCategory.PERMANENT: "这个操作在当前环境无法完成。可以换一种说法或换个工具吗？",
    ErrorCategory.INTERNAL: "执行时出了点问题。我已经记录下来，你可以重试或换个方式。",
}

# 安全策略拒绝时使用：措辞刻意**不承诺"确认后可以继续"**。
# 黑名单命中（下载执行、破坏系统）不是"待授权"，而是"不做"；
# 若在这里说"确认就继续"，等于在教用户用一句"是"绕过安全底线。
_SAFETY_REFUSAL = "这是安全底线，我不会执行，换个说法我也不会做——但我可以帮你完成它安全的那部分。"


def human_readable(category: ErrorCategory, detail: str = "") -> str:
    """生成用户能直接听懂、且知道下一步做什么的反馈。"""
    if category is ErrorCategory.SAFETY:
        return f"{detail} {_SAFETY_REFUSAL}".strip() if detail else _SAFETY_REFUSAL
    base = _SUGGESTIONS.get(category, "出错了，请重试。")
    return f"{detail} {base}".strip() if detail else base


def should_retry(category: ErrorCategory, attempt: int, max_attempts: int) -> bool:
    """
    是否值得再试一次。安全类与永久失败绝不重试。

    ``max_attempts`` 是总尝试次数（含第一次），``attempt`` 从 0 计——
    否则 max_attempts=2 实际会跑三次。
    """
    if attempt + 1 >= max_attempts:
        return False
    return category in (ErrorCategory.TRANSIENT, ErrorCategory.TIMEOUT)


def to_fww_error(exc: BaseException, *, context: str = "") -> FutureWorkError:
    """异常 → 统一异常对象。"""
    category = classify_exception(exc)
    message = f"{context}: {exc}" if context else str(exc)
    return FutureWorkError(
        message,
        category,
        recoverable=category in (ErrorCategory.TRANSIENT, ErrorCategory.TIMEOUT),
        suggestion=_SUGGESTIONS.get(category, ""),
        cause=exc,
    )