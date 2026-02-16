"""Retry policy with exponential backoff and jitter.

Used by the orchestrator for retrying failed agent calls within
the bounds allowed by the circuit breaker.
"""

import asyncio
import random
from typing import Any

import structlog

log = structlog.get_logger()


class RetryPolicy:
    """Exponential backoff with jitter for agent retries.

    Implements decorrelated jitter to avoid thundering herd on retries.
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay_sec: float = 1.0,
        max_delay_sec: float = 30.0,
        jitter: bool = True,
    ) -> None:
        self.max_retries = max_retries
        self.base_delay_sec = base_delay_sec
        self.max_delay_sec = max_delay_sec
        self.jitter = jitter

    def calculate_delay(self, attempt: int) -> float:
        """Calculate the delay before the next retry attempt.

        Uses exponential backoff with optional decorrelated jitter.

        Args:
            attempt: The current attempt number (0-based).

        Returns:
            Delay in seconds before the next retry.
        """
        delay = self.base_delay_sec * (2**attempt)
        delay = min(delay, self.max_delay_sec)

        if self.jitter:
            delay = random.uniform(0, delay)  # noqa: S311 — not used for security

        return float(delay)

    async def execute_with_retry(
        self,
        func: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute a function with retry logic.

        Args:
            func: Async callable to execute.
            *args: Positional arguments for the callable.
            **kwargs: Keyword arguments for the callable.

        Returns:
            The return value of the callable.

        Raises:
            The last exception if all retries are exhausted.
        """
        last_exception: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as exc:
                last_exception = exc
                if attempt < self.max_retries:
                    delay = self.calculate_delay(attempt)
                    log.warning(
                        "retry_policy.retrying",
                        attempt=attempt + 1,
                        max_retries=self.max_retries,
                        delay_sec=round(delay, 2),
                        error_type=type(exc).__name__,
                    )
                    await asyncio.sleep(delay)

        if last_exception is not None:
            raise last_exception

        msg = "Retry exhausted without exception or result"
        raise RuntimeError(msg)
