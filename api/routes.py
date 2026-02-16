"""FastAPI application — async task submission and retrieval.

All route handlers are async. Agent execution uses BackgroundTasks (never awaited
inline). Quota check happens BEFORE task creation. Every state-modifying route
requires API key validation.
"""

import uuid
from typing import Any

import structlog
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException

from api.auth import AuthService
from api.health import HealthChecker
from api.models import (
    ErrorResponse,
    HealthResponse,
    TaskRequest,
    TaskResponse,
    TaskStatusResponse,
)
from orchestrator.state_machine import OrchestrationState, Orchestrator, TaskResult

log = structlog.get_logger()


class TaskStore:
    """In-memory task result store for development.

    In production, this would be backed by Redis or a database.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, dict[str, Any]] = {}

    async def create(self, task_id: str, tenant_id: str, payload: dict[str, Any]) -> None:
        """Create a new task record.

        Args:
            task_id: The task identifier.
            tenant_id: The owning tenant.
            payload: The task payload.
        """
        self._tasks[task_id] = {
            "task_id": task_id,
            "tenant_id": tenant_id,
            "status": "accepted",
            "payload": payload,
            "result": None,
            "error": None,
            "trace": [],
        }

    async def update_result(self, task_id: str, result: TaskResult) -> None:
        """Update a task with its execution result.

        Args:
            task_id: The task identifier.
            result: The orchestration result.
        """
        if task_id not in self._tasks:
            return

        self._tasks[task_id]["status"] = result.status.value
        self._tasks[task_id]["result"] = result.result
        self._tasks[task_id]["error"] = result.error
        self._tasks[task_id]["trace"] = result.trace

    async def get(self, task_id: str, tenant_id: str) -> dict[str, Any] | None:
        """Retrieve a task by ID, scoped to a tenant.

        Args:
            task_id: The task identifier.
            tenant_id: The requesting tenant.

        Returns:
            Task dict if found and authorized, None otherwise.
        """
        task = self._tasks.get(task_id)
        if task is None or task["tenant_id"] != tenant_id:
            return None
        return task


def create_app(
    orchestrator: Orchestrator,
    auth_service: AuthService,
    health_checker: HealthChecker,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        orchestrator: The orchestrator instance.
        auth_service: The authentication service.
        health_checker: The health checker instance.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(
        title="Agent Orchestration Framework",
        description="Production-grade multi-agent AI orchestration API",
        version="0.1.0",
    )
    task_store = TaskStore()

    @app.post(
        "/v1/tasks",
        response_model=TaskResponse,
        responses={401: {"model": ErrorResponse}, 429: {"model": ErrorResponse}},
    )
    async def create_task(
        request: TaskRequest,
        background_tasks: BackgroundTasks,
        x_api_key: str = Header(...),
    ) -> TaskResponse:
        """Submit a task for orchestration.

        Tasks are processed asynchronously. Returns a task_id for polling.
        """
        # Validate API key
        tenant = await auth_service.validate_api_key(x_api_key)
        if tenant is None:
            raise HTTPException(status_code=401, detail="Invalid API key")

        # Quota check BEFORE task creation
        has_quota = await auth_service.check_quota(tenant.tenant_id, tenant.quota_limit)
        if not has_quota:
            raise HTTPException(status_code=429, detail="Quota exceeded")

        # Create task
        task_id = str(uuid.uuid4())
        await task_store.create(task_id, tenant.tenant_id, request.payload)
        await auth_service.record_usage(tenant.tenant_id)

        # Schedule async execution (never await inline)
        background_tasks.add_task(
            _execute_task,
            orchestrator=orchestrator,
            task_store=task_store,
            task_id=task_id,
            payload=request.payload,
        )

        log.info(
            "api.task_created",
            task_id=task_id,
            tenant_id=tenant.tenant_id,
        )

        return TaskResponse(task_id=task_id, status="accepted")

    @app.get(
        "/v1/tasks/{task_id}",
        response_model=TaskStatusResponse,
        responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
    )
    async def get_task(
        task_id: str,
        x_api_key: str = Header(...),
    ) -> TaskStatusResponse:
        """Get the status and result of a submitted task."""
        tenant = await auth_service.validate_api_key(x_api_key)
        if tenant is None:
            raise HTTPException(status_code=401, detail="Invalid API key")

        task = await task_store.get(task_id, tenant.tenant_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Task not found or unauthorized")

        return TaskStatusResponse(
            task_id=task["task_id"],
            status=task["status"],
            result=task["result"],
            error=task["error"],
            trace=task["trace"],
        )

    @app.get("/health", response_model=HealthResponse)
    async def health_check() -> HealthResponse:
        """System health check with agent and circuit breaker status."""
        return await health_checker.check()

    return app


async def _execute_task(
    orchestrator: Orchestrator,
    task_store: TaskStore,
    task_id: str,
    payload: dict[str, Any],
) -> None:
    """Background task that runs orchestration and stores the result.

    Args:
        orchestrator: The orchestrator instance.
        task_store: The task result store.
        task_id: The task identifier.
        payload: The task payload.
    """
    try:
        result = await orchestrator.execute_task({"id": task_id, "payload": payload})
        await task_store.update_result(task_id, result)
    except Exception as exc:
        log.error(
            "api.task_execution_error",
            task_id=task_id,
            error_type=type(exc).__name__,
        )
        # Create a minimal failed result
        failed_result = TaskResult(
            task_id=task_id,
            status=OrchestrationState.FAILED,
            error=f"Internal error: {type(exc).__name__}",
        )
        await task_store.update_result(task_id, failed_result)
