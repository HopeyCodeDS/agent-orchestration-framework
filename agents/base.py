"""Abstract base agent interface and shared data models.

Every agent in the framework MUST extend BaseAgent and implement the
execute(), can_handle(), and health_probe() methods. Agents must never
raise exceptions from execute() — return AgentStatus.FAILED instead (ADR-006).
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AgentStatus(Enum):
    """Status of an agent execution."""

    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    ESCALATED = "escalated"


class AgentRequest(BaseModel):
    """Input to an agent execution.

    Attributes:
        task_id: Unique identifier for the orchestration task.
        payload: The data to be processed by the agent.
        context: Shared state accumulated from previous agent executions.
        max_retries: Maximum retry attempts for this agent call.
    """

    task_id: str
    payload: dict[str, Any]
    context: dict[str, Any] = Field(default_factory=dict)
    max_retries: int = Field(default=2, ge=0)


class AgentResponse(BaseModel):
    """Output from an agent execution.

    Attributes:
        agent_id: Identifier of the agent that produced this response.
        status: Execution outcome status.
        output: Structured output data, None on failure.
        confidence_score: Agent's confidence in the output (0.0 to 1.0).
        metadata: Additional data such as token usage, tool calls, timings.
        error: Error description if status is FAILED.
        requires_human_review: Whether the result needs human escalation.
    """

    agent_id: str
    status: AgentStatus
    output: dict[str, Any] | None = None
    confidence_score: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    requires_human_review: bool = False


class BaseAgent(ABC):
    """Abstract base class for all agents in the framework.

    All agents must implement execute(), can_handle(), and health_probe().
    The execute() method must NEVER raise exceptions — return FAILED status instead.
    """

    def __init__(self, agent_id: str, config: dict[str, Any]) -> None:
        self.agent_id = agent_id
        self.config = config

    @abstractmethod
    async def execute(self, request: AgentRequest) -> AgentResponse:
        """Execute the agent's task.

        Must NEVER raise exceptions — return AgentStatus.FAILED with error
        field populated instead (ADR-006).

        Args:
            request: The agent request containing task data and context.

        Returns:
            AgentResponse with status, output, and confidence score.
        """
        ...

    @abstractmethod
    def can_handle(self, request: AgentRequest) -> bool:
        """Determine if this agent can handle the given request.

        Used by the orchestrator for dynamic agent selection.

        Args:
            request: The agent request to evaluate.

        Returns:
            True if this agent can process the request.
        """
        ...

    @abstractmethod
    async def health_probe(self, timeout_sec: int = 2) -> dict[str, Any]:
        """Lightweight health check for the /health endpoint.

        Must complete within timeout_sec. Should NOT perform full execution.

        Args:
            timeout_sec: Maximum time allowed for the health check.

        Returns:
            Dict with at minimum a "healthy" boolean key.
        """
        ...
