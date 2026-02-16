"""Unit tests for the API layer."""

from typing import Any

from api.auth import AuthService, Tenant
from api.health import HealthChecker
from api.models import (
    ErrorResponse,
    HealthResponse,
    TaskRequest,
    TaskResponse,
    TaskStatusResponse,
)
from api.routes import TaskStore
from orchestrator.circuit_breaker import AgentCircuitBreaker
from orchestrator.state_machine import OrchestrationState, TaskResult
from tests.conftest import FailingAgent, StubAgent


class TestTaskRequest:
    def test_defaults(self) -> None:
        req = TaskRequest(payload={"query": "test"})
        assert req.priority == 3
        assert req.callback_url is None

    def test_custom_priority(self) -> None:
        req = TaskRequest(payload={"query": "test"}, priority=1)
        assert req.priority == 1


class TestTaskResponse:
    def test_accepted_status(self) -> None:
        resp = TaskResponse(task_id="t1")
        assert resp.status == "accepted"


class TestTaskStatusResponse:
    def test_defaults(self) -> None:
        resp = TaskStatusResponse(task_id="t1", status="completed")
        assert resp.result is None
        assert resp.error is None
        assert resp.trace == []


class TestHealthResponse:
    def test_defaults(self) -> None:
        resp = HealthResponse(status="healthy")
        assert resp.agents == {}
        assert resp.circuits == {}


class TestErrorResponse:
    def test_fields(self) -> None:
        resp = ErrorResponse(error="not_found", detail="Task not found")
        assert resp.error == "not_found"


class TestAuthService:
    async def test_register_and_validate(self) -> None:
        service = AuthService()
        tenant = Tenant(tenant_id="t1", name="Test Tenant")
        service.register_api_key("test-key-123", tenant)

        result = await service.validate_api_key("test-key-123")
        assert result is not None
        assert result.tenant_id == "t1"

    async def test_invalid_key_returns_none(self) -> None:
        service = AuthService()
        result = await service.validate_api_key("invalid-key")
        assert result is None

    async def test_empty_key_returns_none(self) -> None:
        service = AuthService()
        result = await service.validate_api_key("")
        assert result is None

    async def test_quota_check(self) -> None:
        service = AuthService()
        assert await service.check_quota("t1", 100) is True

        # Exhaust quota
        for _ in range(100):
            await service.record_usage("t1")
        assert await service.check_quota("t1", 100) is False

    async def test_reset_usage(self) -> None:
        service = AuthService()
        await service.record_usage("t1")
        await service.record_usage("t1")
        assert service.get_usage("t1")["current_usage"] == 2

        await service.reset_usage("t1")
        assert service.get_usage("t1")["current_usage"] == 0

    def test_generate_api_key(self) -> None:
        key = AuthService.generate_api_key("test-tenant")
        assert len(key) == 64  # SHA-256 hex


class TestTenant:
    def test_defaults(self) -> None:
        tenant = Tenant(tenant_id="t1", name="Test")
        assert tenant.tier == "free"
        assert tenant.quota_limit == 100


class TestTaskStore:
    async def test_create_and_get(self) -> None:
        store = TaskStore()
        await store.create("task-1", "tenant-1", {"query": "test"})
        task = await store.get("task-1", "tenant-1")
        assert task is not None
        assert task["status"] == "accepted"

    async def test_get_wrong_tenant(self) -> None:
        store = TaskStore()
        await store.create("task-1", "tenant-1", {"query": "test"})
        task = await store.get("task-1", "tenant-2")
        assert task is None

    async def test_get_nonexistent(self) -> None:
        store = TaskStore()
        task = await store.get("nonexistent", "tenant-1")
        assert task is None

    async def test_update_result(self) -> None:
        store = TaskStore()
        await store.create("task-1", "tenant-1", {"query": "test"})

        result = TaskResult(
            task_id="task-1",
            status=OrchestrationState.COMPLETED,
            result={"answer": "42"},
        )
        await store.update_result("task-1", result)

        task = await store.get("task-1", "tenant-1")
        assert task is not None
        assert task["status"] == "completed"
        assert task["result"] == {"answer": "42"}


class TestHealthChecker:
    async def test_all_healthy(self) -> None:
        agents: dict[str, Any] = {"stub": StubAgent(agent_id="stub")}
        cbs: dict[str, Any] = {
            "stub": AgentCircuitBreaker(agent_id="stub", failure_threshold=3),
        }
        checker = HealthChecker(agents=agents, circuit_breakers=cbs)
        response = await checker.check()
        assert response.status == "healthy"
        assert "stub" in response.agents

    async def test_degraded_with_open_circuit(self) -> None:
        agents: dict[str, Any] = {"stub": StubAgent(agent_id="stub")}
        cb = AgentCircuitBreaker(agent_id="stub", failure_threshold=1, timeout_sec=600)
        # Force circuit open
        cb._record_failure()
        cbs: dict[str, Any] = {"stub": cb}
        checker = HealthChecker(agents=agents, circuit_breakers=cbs)
        response = await checker.check()
        assert response.status == "degraded"

    async def test_unhealthy_agent(self) -> None:
        agents: dict[str, Any] = {"failing": FailingAgent(agent_id="failing")}
        cbs: dict[str, Any] = {
            "failing": AgentCircuitBreaker(agent_id="failing", failure_threshold=3),
        }
        checker = HealthChecker(agents=agents, circuit_breakers=cbs)
        response = await checker.check()
        assert response.status == "unhealthy"
