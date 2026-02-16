"""AgentTrace schema and OpenTelemetry span management.

Structured spans for agent executions within an orchestration. Payloads
are hashed, never stored raw, for PII protection (ADR-004).
"""

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry import trace
from pydantic import BaseModel, Field


class AgentTrace(BaseModel):
    """Structured span for a single agent execution within an orchestration.

    Attributes:
        trace_id: Correlates the entire orchestration flow.
        span_id: Unique identifier for this agent execution.
        parent_span_id: Links to the parent span (orchestration-level).
        agent_id: The agent that produced this trace.
        state_entered: The orchestration state when this agent was called.
        start_time: When the agent execution started.
        end_time: When the agent execution completed.
        input_payload_hash: SHA-256 hash of the input (no raw data — ADR-004).
        output_confidence: Agent's confidence score for the output.
        tokens_used: Number of LLM tokens consumed.
        tool_calls: List of tool calls made by the agent.
        error_type: Exception class name if an error occurred.
        error_message_hash: SHA-256 hash of the error message (no raw errors — ADR-004).
    """

    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    span_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    parent_span_id: str | None = None
    agent_id: str
    state_entered: str
    start_time: datetime
    end_time: datetime
    input_payload_hash: str
    output_confidence: float = Field(ge=0.0, le=1.0)
    tokens_used: int = Field(default=0, ge=0)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    error_type: str | None = None
    error_message_hash: str | None = None

    @property
    def latency_ms(self) -> int:
        """Calculate execution latency in milliseconds."""
        return int((self.end_time - self.start_time).total_seconds() * 1000)


def hash_payload(payload: dict[str, Any]) -> str:
    """Create a SHA-256 hash of a payload for telemetry (ADR-004).

    Args:
        payload: The data to hash.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()


def hash_error(error_message: str) -> str:
    """Create a SHA-256 hash of an error message for telemetry (ADR-004).

    Args:
        error_message: The error message to hash.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    return hashlib.sha256(error_message.encode()).hexdigest()


class SpanManager:
    """Manages OpenTelemetry spans for agent executions.

    The orchestrator uses this to instrument agent calls without agents
    instrumenting themselves (observability contract).
    """

    def __init__(self, service_name: str = "agent-orchestration-framework") -> None:
        self.tracer = trace.get_tracer(service_name)

    def create_orchestration_span(self, task_id: str) -> trace.Span:
        """Create a top-level span for the entire orchestration.

        Args:
            task_id: The task identifier for this orchestration.

        Returns:
            An OpenTelemetry span.
        """
        return self.tracer.start_span(
            name="orchestration",
            attributes={
                "task.id": task_id,
                "service.name": "orchestrator",
            },
        )

    def create_agent_span(
        self,
        agent_id: str,
        task_id: str,
        state: str,
        parent_context: otel_context.Context | None = None,
    ) -> trace.Span:
        """Create a span for a single agent execution.

        Args:
            agent_id: The agent being executed.
            task_id: The task identifier.
            state: The orchestration state.
            parent_context: Optional parent context for span linking.

        Returns:
            An OpenTelemetry span.
        """
        return self.tracer.start_span(
            name=f"agent.{agent_id}",
            attributes={
                "agent.id": agent_id,
                "task.id": task_id,
                "orchestration.state": state,
            },
            context=parent_context,
        )

    @staticmethod
    def record_agent_result(
        span: trace.Span,
        confidence: float,
        status: str,
        tokens_used: int = 0,
        error_type: str | None = None,
    ) -> None:
        """Record agent execution results on an OpenTelemetry span.

        Args:
            span: The span to annotate.
            confidence: Agent confidence score.
            status: Agent execution status.
            tokens_used: LLM tokens consumed.
            error_type: Exception type if an error occurred.
        """
        span.set_attribute("agent.confidence", confidence)
        span.set_attribute("agent.status", status)
        span.set_attribute("agent.tokens_used", tokens_used)
        if error_type:
            span.set_attribute("agent.error_type", error_type)

    @staticmethod
    def now() -> datetime:
        """Return current UTC datetime for trace timestamps."""
        return datetime.now(timezone.utc)
