"""Unit tests for the observability module."""

from datetime import datetime, timezone

from observability.metrics import AgentMetrics, ExecutionTimer
from observability.telemetry import AgentTrace, hash_error, hash_payload
from observability.traces import TraceStore


class TestAgentTrace:
    def test_trace_creation(self) -> None:
        now = datetime.now(timezone.utc)
        trace = AgentTrace(
            agent_id="validator",
            state_entered="validating",
            start_time=now,
            end_time=now,
            input_payload_hash="abc123",
            output_confidence=0.95,
        )
        assert trace.agent_id == "validator"
        assert trace.trace_id is not None
        assert trace.span_id is not None

    def test_latency_calculation(self) -> None:
        start = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 1, 1, 0, 0, 1, 500000, tzinfo=timezone.utc)
        trace = AgentTrace(
            agent_id="test",
            state_entered="validating",
            start_time=start,
            end_time=end,
            input_payload_hash="hash",
            output_confidence=0.5,
        )
        assert trace.latency_ms == 1500


class TestHashFunctions:
    def test_hash_payload_deterministic(self) -> None:
        payload = {"key": "value", "number": 42}
        h1 = hash_payload(payload)
        h2 = hash_payload(payload)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_hash_payload_different_inputs(self) -> None:
        assert hash_payload({"a": 1}) != hash_payload({"a": 2})

    def test_hash_error(self) -> None:
        h = hash_error("test error")
        assert len(h) == 64


class TestTraceStore:
    async def test_store_and_retrieve(self) -> None:
        store = TraceStore()
        now = datetime.now(timezone.utc)
        trace = AgentTrace(
            trace_id="trace-1",
            agent_id="validator",
            state_entered="validating",
            start_time=now,
            end_time=now,
            input_payload_hash="hash",
            output_confidence=0.9,
        )

        await store.store(trace)
        traces = await store.get_trace("trace-1")
        assert len(traces) == 1
        assert traces[0].agent_id == "validator"

    async def test_get_agent_traces(self) -> None:
        store = TraceStore()
        now = datetime.now(timezone.utc)

        for i in range(5):
            await store.store(
                AgentTrace(
                    trace_id=f"trace-{i}",
                    agent_id="validator",
                    state_entered="validating",
                    start_time=now,
                    end_time=now,
                    input_payload_hash="hash",
                    output_confidence=0.9,
                )
            )

        traces = await store.get_agent_traces("validator", limit=3)
        assert len(traces) == 3

    async def test_eviction_on_max(self) -> None:
        store = TraceStore(max_traces=3)
        now = datetime.now(timezone.utc)

        for i in range(5):
            await store.store(
                AgentTrace(
                    trace_id=f"trace-{i}",
                    agent_id="test",
                    state_entered="validating",
                    start_time=now,
                    end_time=now,
                    input_payload_hash="hash",
                    output_confidence=0.9,
                )
            )

        stats = store.get_stats()
        assert stats["total_spans"] <= 3


class TestAgentMetrics:
    def test_record_and_retrieve(self) -> None:
        metrics = AgentMetrics()
        metrics.record_execution("validator", 100.0, 0.9, success=True, tokens_used=50)
        metrics.record_execution("validator", 200.0, 0.8, success=True, tokens_used=60)
        metrics.record_execution("validator", 150.0, 0.0, success=False)

        stats = metrics.get_agent_stats("validator")
        assert stats["total_executions"] == 3
        assert stats["success_count"] == 2
        assert stats["failure_count"] == 1
        assert stats["total_tokens"] == 110

    def test_latency_p95(self) -> None:
        metrics = AgentMetrics()
        for i in range(100):
            metrics.record_execution("test", float(i), 0.5, success=True)

        p95 = metrics.get_recent_latency_p95("test")
        assert p95 >= 90.0

    def test_empty_stats(self) -> None:
        metrics = AgentMetrics()
        stats = metrics.get_agent_stats("nonexistent")
        assert stats["total_executions"] == 0


class TestExecutionTimer:
    def test_timer_records_elapsed(self) -> None:
        timer = ExecutionTimer()
        with timer:
            pass
        assert timer.elapsed_ms >= 0.0
