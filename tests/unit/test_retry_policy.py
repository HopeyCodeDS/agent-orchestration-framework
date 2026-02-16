"""Unit tests for the RetryPolicy."""

import pytest

from orchestrator.retry_policy import RetryPolicy


class TestRetryPolicy:
    def test_delay_increases_exponentially(self) -> None:
        policy = RetryPolicy(base_delay_sec=1.0, jitter=False)
        assert policy.calculate_delay(0) == 1.0
        assert policy.calculate_delay(1) == 2.0
        assert policy.calculate_delay(2) == 4.0
        assert policy.calculate_delay(3) == 8.0

    def test_delay_capped_at_max(self) -> None:
        policy = RetryPolicy(base_delay_sec=1.0, max_delay_sec=5.0, jitter=False)
        assert policy.calculate_delay(10) == 5.0

    def test_jitter_produces_bounded_delay(self) -> None:
        policy = RetryPolicy(base_delay_sec=1.0, jitter=True)
        for attempt in range(5):
            delay = policy.calculate_delay(attempt)
            max_expected = min(1.0 * (2**attempt), 30.0)
            assert 0 <= delay <= max_expected

    async def test_execute_with_retry_success(self) -> None:
        policy = RetryPolicy(max_retries=3, base_delay_sec=0.01)
        call_count = 0

        async def succeeding_func() -> str:
            nonlocal call_count
            call_count += 1
            return "success"

        result = await policy.execute_with_retry(succeeding_func)
        assert result == "success"
        assert call_count == 1

    async def test_execute_with_retry_eventual_success(self) -> None:
        policy = RetryPolicy(max_retries=3, base_delay_sec=0.01)
        call_count = 0

        async def eventually_succeeds() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                msg = "not yet"
                raise ValueError(msg)
            return "success"

        result = await policy.execute_with_retry(eventually_succeeds)
        assert result == "success"
        assert call_count == 3

    async def test_execute_with_retry_exhausted(self) -> None:
        policy = RetryPolicy(max_retries=2, base_delay_sec=0.01)

        async def always_fails() -> None:
            msg = "permanent failure"
            raise ValueError(msg)

        with pytest.raises(ValueError, match="permanent failure"):
            await policy.execute_with_retry(always_fails)
