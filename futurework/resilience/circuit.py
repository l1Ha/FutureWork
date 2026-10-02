"""
熔断器：连续失败的工具先"断电"，避免在坏掉的依赖上空转。

三态机：
    CLOSED（正常）──失败达阈值──▶ OPEN（熔断，快速失败）
        ▲                              │
        └────── 冷却成功一次 ──────────┘
                                    │
                              HALF_OPEN（试探）
                                    │  失败
                                    ▼
                                  OPEN

熔断期间的调用**立即返回**而不真正执行——这在交互层面至关重要：
用户说"打开浏览器"而浏览器已经崩了，正确的体验是 0.1 秒告诉他
"浏览器现在不可用，要不要用别的？"而不是每次都等 10 秒超时。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """熔断打开时抛出。"""

    def __init__(self, name: str, retry_after: float) -> None:
        super().__init__(f"熔断器 '{name}' 处于打开状态，{retry_after:.1f}s 后可重试")
        self.name = name
        self.retry_after = retry_after


@dataclass
class CircuitBreaker:
    """
    熔断器。

    :param failure_threshold: 连续失败多少次后熔断
    :param recovery_timeout: 熔断后多久进入半开状态
    :param success_threshold: 半开状态下连续成功多少次才完全恢复
    """

    name: str = "default"
    failure_threshold: int = 3
    recovery_timeout: float = 20.0
    success_threshold: int = 1

    def __post_init__(self) -> None:
        self._lock = threading.RLock()
        self._failures = 0
        self._successes = 0
        self._opened_at: Optional[float] = None
        self._state = CircuitState.CLOSED
        self._tripped_count = 0

    # ------------------------------------------------------------------
    @property
    def state(self) -> CircuitState:
        with self._lock:
            self._maybe_half_open()
            return self._state

    def allows(self) -> bool:
        """当前是否允许调用。"""
        return self.state is not CircuitState.OPEN

    def retry_after(self) -> float:
        with self._lock:
            if self._state is not CircuitState.OPEN or self._opened_at is None:
                return 0.0
            return max(0.0, self.recovery_timeout - (time.monotonic() - self._opened_at))

    # ------------------------------------------------------------------
    def record_success(self) -> None:
        with self._lock:
            # 必须先做跃迁：冷却期结束后状态仍是 OPEN，
            # 直接比较 HALF_OPEN 会让"试探成功"永远关不掉熔断器。
            self._maybe_half_open()
            self._failures = 0
            self._successes += 1
            if self._state is CircuitState.HALF_OPEN and self._successes >= self.success_threshold:
                self._state = CircuitState.CLOSED
                self._opened_at = None
                self._successes = 0

    def record_failure(self) -> None:
        with self._lock:
            self._maybe_half_open()
            self._failures += 1
            if self._state is CircuitState.HALF_OPEN:
                self._trip()
            elif self._failures >= self.failure_threshold:
                self._trip()

    def reset(self) -> None:
        with self._lock:
            self._failures = 0
            self._successes = 0
            self._opened_at = None
            self._state = CircuitState.CLOSED

    def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        带熔断保护的调用。熔断打开时立即抛 ``CircuitOpenError``（不执行）。
        """
        with self._lock:
            self._maybe_half_open()
            if self._state is CircuitState.OPEN:
                wait = self.recovery_timeout - (time.monotonic() - (self._opened_at or 0.0))
                raise CircuitOpenError(self.name, max(0.0, wait))

        try:
            result = func(*args, **kwargs)
        except Exception:
            self.record_failure()
            raise
        self.record_success()
        return result

    # ------------------------------------------------------------------
    def _maybe_half_open(self) -> None:
        if self._state is CircuitState.OPEN and self._opened_at is not None:
            if time.monotonic() - self._opened_at >= self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._successes = 0

    def _trip(self) -> None:
        self._state = CircuitState.OPEN
        self._opened_at = time.monotonic()
        self._tripped_count += 1

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "consecutive_failures": self._failures,
                "tripped": self._tripped_count,
                "retry_after": round(self.retry_after(), 2),
            }


class BreakerRegistry:
    """按名称管理多个熔断器（每个工具一个）。"""

    def __init__(self, **defaults: Any) -> None:
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._defaults = defaults

    def get(self, name: str) -> CircuitBreaker:
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(name=name, **self._defaults)
        return self._breakers[name]

    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        return {name: breaker.snapshot() for name, breaker in self._breakers.items()}

    def reset_all(self) -> None:
        for breaker in self._breakers.values():
            breaker.reset()