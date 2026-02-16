"""Agent interaction trace storage and retrieval.

Stores AgentTrace records in memory for development. Production systems
should use a persistent trace backend (Jaeger, Tempo, etc.).
"""

from collections import defaultdict
from typing import Any

import structlog

from observability.telemetry import AgentTrace

log = structlog.get_logger()


class TraceStore:
    """In-memory trace storage for development and testing.

    Stores AgentTrace records indexed by trace_id for efficient retrieval.
    Production deployments should replace this with a persistent backend.
    """

    def __init__(self, max_traces: int = 10000) -> None:
        self.max_traces = max_traces
        self._traces: dict[str, list[AgentTrace]] = defaultdict(list)
        self._trace_count: int = 0

    async def store(self, agent_trace: AgentTrace) -> None:
        """Store an agent trace record.

        Args:
            agent_trace: The trace record to store.
        """
        if self._trace_count >= self.max_traces:
            self._evict_oldest()

        self._traces[agent_trace.trace_id].append(agent_trace)
        self._trace_count += 1

        log.debug(
            "trace_store.stored",
            trace_id=agent_trace.trace_id,
            span_id=agent_trace.span_id,
            agent_id=agent_trace.agent_id,
        )

    async def get_trace(self, trace_id: str) -> list[AgentTrace]:
        """Retrieve all spans for a given trace ID.

        Args:
            trace_id: The orchestration trace identifier.

        Returns:
            List of AgentTrace records, ordered by storage time.
        """
        return list(self._traces.get(trace_id, []))

    async def get_agent_traces(self, agent_id: str, limit: int = 100) -> list[AgentTrace]:
        """Retrieve recent traces for a specific agent.

        Args:
            agent_id: The agent to filter by.
            limit: Maximum number of traces to return.

        Returns:
            List of AgentTrace records for the specified agent.
        """
        result: list[AgentTrace] = []
        for spans in self._traces.values():
            for span in spans:
                if span.agent_id == agent_id:
                    result.append(span)
                    if len(result) >= limit:
                        return result
        return result

    def get_stats(self) -> dict[str, Any]:
        """Return trace store statistics for health checks."""
        return {
            "total_traces": len(self._traces),
            "total_spans": self._trace_count,
            "max_traces": self.max_traces,
        }

    def _evict_oldest(self) -> None:
        """Evict the oldest trace to make room for new ones."""
        if not self._traces:
            return

        oldest_key = next(iter(self._traces))
        evicted_spans = len(self._traces[oldest_key])
        del self._traces[oldest_key]
        self._trace_count -= evicted_spans

        log.debug(
            "trace_store.evicted",
            trace_id=oldest_key,
            spans_evicted=evicted_spans,
        )
