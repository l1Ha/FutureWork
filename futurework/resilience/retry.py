"""
重试策略：指数退避 + 抖动，仅对瞬态故障生效。

关键约束（这些是重试机制最容易做错的地方）：
1. **只重试瞬态错误**——参数错误重试一万次仍是错误；
2. **尊重幂等性**——非幂等操作（写文件、提交）重试可能产生重复副作用，
   因此默认 ``retry_on_non_idempotent=False``；
3. **总时限封顶**——避免多次重试叠加成用户无法接受的等待；
4. **抖动**——避免多个并发任务在同一时刻集体重试形成尖峰。
"""

from __future__ import annotations

import asyncio
import functools
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple, Type

from futurework.resilience.errors import ErrorCategory, classify_exception, should_retry


@dataclass
class RetryPolicy:
    """重试策略配置。"""

    max_attempts: int = 3
    base_delay: float = 0.15
    multiplier: float = 2.0
    max_delay: float = 3.0
    jitter: float = 0.25
    total_timeout: float = 15.0
    retry_categories: Tuple[ErrorCategory, ...] = (
        ErrorCategory.TRANSIENT,
        ErrorCategory.TIMEOUT,
    )

    def delay_for(self, attempt: int) -> float:
        """第 ``attempt`` 次（从 0 计）失败后的等待时长。"""
        raw = self.base_delay * (self.multiplier ** attempt)
        capped = min(self.max_delay, raw)
        if self.jitter > 0:
            spread = capped * self.jitter
            capped = max(0.0, capped + random.uniform(-spread, spread))
        return capped


def should_retry_exception(
    exc: BaseException,
    policy: RetryPolicy,
    attempt: int,
    elapsed: float,
    *,
    idempotent: bool = True,
) -> bool:
    """综合判定是否应重试。"""
    if elapsed > policy.total_timeout:
        return False
    if not idempotent:
        # 非幂等操作（写文件、提交）重试会产生重复副作用，只允许一次尝试
        return False
    category = classify_exception(exc)
    if category not in policy.retry_categories:
        return False
    return should_retry(category, attempt, policy.max_attempts)


def with_retry(
    func: Callable[..., Any],
    *args: Any,
    policy: Optional[RetryPolicy] = None,
    idempotent: bool = True,
    on_retry: Optional[Callable[[int, BaseException, float], None]] = None,
    **kwargs: Any,
) -> Any:
    """
    同步重试包装。

    :raises BaseException: 重试耗尽后抛出最后一次异常。
    """
    active = policy or RetryPolicy()
    started = time.perf_counter()
    attempt = 0
    while True:
        try:
            return func(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - 需捕获后按类别决策
            elapsed = time.perf_counter() - started
            if not should_retry_exception(exc, active, attempt, elapsed, idempotent=idempotent):
                raise
            delay = active.delay_for(attempt)
            if elapsed + delay > active.total_timeout:
                raise
            if on_retry is not None:
                try:
                    on_retry(attempt + 1, exc, delay)
                except Exception:
                    pass
            time.sleep(delay)
            attempt += 1


async def retry_async(
    func: Callable[..., Any],
    *args: Any,
    policy: Optional[RetryPolicy] = None,
    idempotent: bool = True,
    on_retry: Optional[Callable[[int, BaseException, float], None]] = None,
    **kwargs: Any,
) -> Any:
    """异步重试包装。"""
    active = policy or RetryPolicy()
    started = time.perf_counter()
    attempt = 0
    while True:
        try:
            result = func(*args, **kwargs)
            if asyncio.iscoroutine(result):
                result = await result
            return result
        except (asyncio.CancelledError, KeyboardInterrupt):
            raise
        except BaseException as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - started
            if not should_retry_exception(exc, active, attempt, elapsed, idempotent=idempotent):
                raise
            delay = active.delay_for(attempt)
            if elapsed + delay > active.total_timeout:
                raise
            if on_retry is not None:
                try:
                    on_retry(attempt + 1, exc, delay)
                except Exception:
                    pass
            await asyncio.sleep(delay)
            attempt += 1


def retryable(func: Callable[..., Any], policy: Optional[RetryPolicy] = None, idempotent: bool = True) -> Callable[..., Any]:
    """装饰器形式的重试。"""
    active = policy or RetryPolicy()

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return with_retry(func, *args, policy=active, idempotent=idempotent, **kwargs)

    wrapper.retry_policy = active  # type: ignore[attr-defined]
    return wrapper