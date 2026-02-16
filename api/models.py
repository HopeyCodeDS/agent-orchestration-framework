"""API request and response models — Pydantic v2 models for the REST API."""

from typing import Any

from pydantic import BaseModel, Field


class TaskRequest(BaseModel):
    """Request body for task submission.

    Attributes:
        payload: The task data to process through orchestration.
        priority: Task priority (1=highest, 5=lowest).
        callback_url: Optional webhook for completion notification.
    """

    payload: dict[str, Any]
    priority: int = Field(default=3, ge=1, le=5)
    callback_url: str | None = None


class TaskResponse(BaseModel):
    """Response after task submission (async acknowledgment).

    Attributes:
        task_id: Unique identifier for tracking the task.
        status: Initial status (always "accepted").
    """

    task_id: str
    status: str = "accepted"


class TaskStatusResponse(BaseModel):
    """Response for task status polling.

    Attributes:
        task_id: The task identifier.
        status: Current status (accepted, processing, completed, failed).
        result: Task output if completed.
        error: Error details if failed.
        trace: Execution trace for observability.
    """

    task_id: str
    status: str
    result: dict[str, Any] | None = None
    error: str | None = None
    trace: list[dict[str, Any]] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Response for the health endpoint.

    Attributes:
        status: Overall system health (healthy, degraded, unhealthy).
        agents: Per-agent health status.
        circuits: Per-agent circuit breaker states.
    """

    status: str
    agents: dict[str, Any] = Field(default_factory=dict)
    circuits: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Standard error response.

    Attributes:
        error: Error type or code.
        detail: Human-readable error description.
    """

    error: str
    detail: str
