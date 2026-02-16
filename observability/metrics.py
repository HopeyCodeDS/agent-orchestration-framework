"""Custom metrics for agent performance monitoring.

Tracks agent latency, confidence distributions, and execution counts.
Integrates with OpenTelemetry for export to monitoring backends.
"""

import time
from collections import defaultdict
from typing import Any

import structlog

log = structlog.get_logger()


class AgentMetrics:
    """In-memory metrics collection for agent performance monitoring.

    Tracks latency, confidence scores, and success/failure counts per agent.
    Designed for development use — production should use OpenTelemetry metrics SDK.
    """

    def __init__(self) -> None:
        self._latencies: dict[str, list[float]] = defaultdict(list)
        self._confidences: dict[str, list[float]] = defaultdict(list)
        self._success_counts: dict[str, int] = defaultdict(int)
        self._failure_counts: dict[str, int] = defaultdict(int)
        self._total_tokens: dict[str, int] = defaultdict(int)

    def record_execution(
        self,
        agent_id: str,
        latency_ms: float,
        confidence: float,
        success: bool,
        tokens_used: int = 0,
    ) -> None:
        """Record metrics for a single agent execution.

        Args:
            agent_id: The agent that was executed.
            latency_ms: Execution time in milliseconds.
            confidence: Agent confidence score (0.0 to 1.0).
            success: Whether the execution succeeded.
            tokens_used: Number of LLM tokens consumed.
        """
        self._latencies[agent_id].append(latency_ms)
        self._confidences[agent_id].append(confidence)
        self._total_tokens[agent_id] += tokens_used

        if success:
            self._success_counts[agent_id] += 1
        else:
            self._failure_counts[agent_id] += 1

    def get_agent_stats(self, agent_id: str) -> dict[str, Any]:
        """Get aggregated statistics for a specific agent.

        Args:
            agent_id: The agent to get stats for.

        Returns:
            Dict with latency percentiles, confidence stats, and counts.
        """
        latencies = self._latencies.get(agent_id, [])
        confidences = self._confidences.get(agent_id, [])

        return {
            "agent_id": agent_id,
            "total_executions": (
                self._success_counts.get(agent_id, 0) + self._failure_counts.get(agent_id, 0)
            ),
            "success_count": self._success_counts.get(agent_id, 0),
            "failure_count": self._failure_counts.get(agent_id, 0),
            "total_tokens": self._total_tokens.get(agent_id, 0),
            "latency": self._compute_percentiles(latencies) if latencies else {},
            "confidence": self._compute_percentiles(confidences) if confidences else {},
        }

    def get_all_stats(self) -> dict[str, Any]:
        """Get aggregated statistics for all agents."""
        all_agents = set(self._success_counts.keys()) | set(self._failure_counts.keys())
        return {agent_id: self.get_agent_stats(agent_id) for agent_id in sorted(all_agents)}

    def get_recent_latency_p95(self, agent_id: str | None = None) -> float:
        """Get the p95 latency across recent executions.

        Args:
            agent_id: Optional agent to filter by. If None, uses all agents.

        Returns:
            The p95 latency in milliseconds, or 0.0 if no data.
        """
        if agent_id:
            latencies = self._latencies.get(agent_id, [])
        else:
            latencies = [lat for lats in self._latencies.values() for lat in lats]

        if not latencies:
            return 0.0

        sorted_lats = sorted(latencies)
        idx = int(len(sorted_lats) * 0.95)
        return sorted_lats[min(idx, len(sorted_lats) - 1)]

    @staticmethod
    def _compute_percentiles(values: list[float]) -> dict[str, float]:
        """Compute p50, p95, p99 percentiles for a list of values."""
        if not values:
            return {}

        sorted_vals = sorted(values)
        n = len(sorted_vals)

        return {
            "p50": sorted_vals[int(n * 0.50)],
            "p95": sorted_vals[min(int(n * 0.95), n - 1)],
            "p99": sorted_vals[min(int(n * 0.99), n - 1)],
            "mean": sum(sorted_vals) / n,
            "min": sorted_vals[0],
            "max": sorted_vals[-1],
        }


class ExecutionTimer:
    """Context manager for timing agent executions."""

    def __init__(self) -> None:
        self.start_time: float = 0.0
        self.end_time: float = 0.0

    def __enter__(self) -> "ExecutionTimer":
        self.start_time = time.monotonic()
        return self

    def __exit__(self, *args: Any) -> None:
        self.end_time = time.monotonic()

    @property
    def elapsed_ms(self) -> float:
        """Return elapsed time in milliseconds."""
        return (self.end_time - self.start_time) * 1000
