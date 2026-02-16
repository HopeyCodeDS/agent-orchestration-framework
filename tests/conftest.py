"""Shared test fixtures for the agent orchestration framework."""

from typing import Any

import pytest

from agents.base import AgentRequest, AgentResponse, AgentStatus, BaseAgent
from agents.validation_agent import ValidationAgent
from orchestrator.circuit_breaker import AgentCircuitBreaker


class StubAgent(BaseAgent):
    """A test stub agent that returns configurable responses."""

    def __init__(
        self,
        agent_id: str = "stub_agent",
        config: dict[str, Any] | None = None,
        response: AgentResponse | None = None,
    ) -> None:
        super().__init__(agent_id, config or {})
        self._response = response or AgentResponse(
            agent_id=agent_id,
            status=AgentStatus.COMPLETED,
            output={"result": "stub"},
            confidence_score=0.95,
        )

    async def execute(self, request: AgentRequest) -> AgentResponse:
        return self._response

    def can_handle(self, request: AgentRequest) -> bool:
        return True

    async def health_probe(self, timeout_sec: int = 2) -> dict[str, Any]:
        return {"healthy": True, "agent_id": self.agent_id}


class FailingAgent(BaseAgent):
    """A test agent that always returns FAILED status."""

    def __init__(
        self,
        agent_id: str = "failing_agent",
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(agent_id, config or {})

    async def execute(self, request: AgentRequest) -> AgentResponse:
        return AgentResponse(
            agent_id=self.agent_id,
            status=AgentStatus.FAILED,
            error="Intentional failure for testing",
            confidence_score=0.0,
        )

    def can_handle(self, request: AgentRequest) -> bool:
        return True

    async def health_probe(self, timeout_sec: int = 2) -> dict[str, Any]:
        return {"healthy": False, "agent_id": self.agent_id, "error": "test failure"}


@pytest.fixture
def sample_request() -> AgentRequest:
    """A minimal valid agent request."""
    return AgentRequest(
        task_id="test-task-001",
        payload={"query": "What is the capital of France?", "category": "geography"},
    )


@pytest.fixture
def sample_payload() -> dict[str, Any]:
    """A sample task payload."""
    return {
        "query": "What is the capital of France?",
        "category": "geography",
        "priority": 1,
    }


@pytest.fixture
def validation_config() -> dict[str, Any]:
    """Configuration for the ValidationAgent in tests."""
    return {
        "required_fields": ["query"],
        "field_types": {"query": "str", "priority": "int"},
        "semantic_rules": [
            {"field": "priority", "rule": "min_value", "value": 1},
            {"field": "priority", "rule": "max_value", "value": 5},
        ],
        "pii_detection_enabled": True,
        "timeout_sec": 15,
        "max_retries": 2,
    }


@pytest.fixture
def validation_agent(validation_config: dict[str, Any]) -> ValidationAgent:
    """A configured ValidationAgent instance."""
    return ValidationAgent(config=validation_config)


@pytest.fixture
def stub_agent() -> StubAgent:
    """A stub agent that returns success."""
    return StubAgent()


@pytest.fixture
def failing_agent() -> FailingAgent:
    """A stub agent that always fails."""
    return FailingAgent()


@pytest.fixture
def circuit_breaker() -> AgentCircuitBreaker:
    """A circuit breaker with low thresholds for testing."""
    return AgentCircuitBreaker(
        agent_id="test_agent",
        failure_threshold=3,
        timeout_sec=1,
    )


@pytest.fixture
def orchestration_config() -> dict[str, Any]:
    """Orchestration configuration for tests."""
    return {
        "orchestration": {
            "initial_state": "validating",
            "max_steps": 10,
            "states": {
                "validating": {
                    "agent": "validator",
                    "transitions": {
                        "completed": "completed",
                        "failed": "failed",
                        "escalated": "failed",
                    },
                },
                "completed": {"terminal": True},
                "failed": {"terminal": True},
            },
        }
    }
