"""Unit tests for the BaseAgent contract and data models."""

from agents.base import AgentRequest, AgentResponse, AgentStatus
from tests.conftest import FailingAgent, StubAgent


class TestAgentStatus:
    def test_status_values(self) -> None:
        assert AgentStatus.PENDING.value == "pending"
        assert AgentStatus.EXECUTING.value == "executing"
        assert AgentStatus.COMPLETED.value == "completed"
        assert AgentStatus.FAILED.value == "failed"
        assert AgentStatus.ESCALATED.value == "escalated"


class TestAgentRequest:
    def test_minimal_request(self) -> None:
        request = AgentRequest(task_id="t1", payload={"key": "value"})
        assert request.task_id == "t1"
        assert request.payload == {"key": "value"}
        assert request.context == {}
        assert request.max_retries == 2

    def test_full_request(self) -> None:
        request = AgentRequest(
            task_id="t2",
            payload={"data": 123},
            context={"previous": "output"},
            max_retries=5,
        )
        assert request.context == {"previous": "output"}
        assert request.max_retries == 5

    def test_max_retries_validation(self) -> None:
        import pytest

        with pytest.raises(Exception):  # noqa: B017 — testing pydantic validation
            AgentRequest(task_id="t3", payload={}, max_retries=-1)


class TestAgentResponse:
    def test_successful_response(self) -> None:
        response = AgentResponse(
            agent_id="test",
            status=AgentStatus.COMPLETED,
            output={"result": "data"},
            confidence_score=0.95,
        )
        assert response.status == AgentStatus.COMPLETED
        assert response.confidence_score == 0.95
        assert response.error is None
        assert response.requires_human_review is False

    def test_failed_response(self) -> None:
        response = AgentResponse(
            agent_id="test",
            status=AgentStatus.FAILED,
            error="Something went wrong",
            confidence_score=0.0,
            requires_human_review=True,
        )
        assert response.status == AgentStatus.FAILED
        assert response.error == "Something went wrong"
        assert response.requires_human_review is True

    def test_confidence_bounds(self) -> None:
        import pytest

        with pytest.raises(Exception):  # noqa: B017 — testing pydantic validation
            AgentResponse(agent_id="test", status=AgentStatus.COMPLETED, confidence_score=1.5)
        with pytest.raises(Exception):  # noqa: B017 — testing pydantic validation
            AgentResponse(agent_id="test", status=AgentStatus.COMPLETED, confidence_score=-0.1)


class TestBaseAgentContract:
    async def test_stub_agent_execute(self, sample_request: AgentRequest) -> None:
        agent = StubAgent()
        response = await agent.execute(sample_request)
        assert response.status == AgentStatus.COMPLETED
        assert response.confidence_score == 0.95

    async def test_stub_agent_can_handle(self, sample_request: AgentRequest) -> None:
        agent = StubAgent()
        assert agent.can_handle(sample_request) is True

    async def test_stub_agent_health_probe(self) -> None:
        agent = StubAgent()
        health = await agent.health_probe()
        assert health["healthy"] is True

    async def test_failing_agent_returns_failed_status(self, sample_request: AgentRequest) -> None:
        agent = FailingAgent()
        response = await agent.execute(sample_request)
        assert response.status == AgentStatus.FAILED
        assert response.error is not None

    def test_agent_id_stored(self) -> None:
        agent = StubAgent(agent_id="my_agent")
        assert agent.agent_id == "my_agent"

    def test_agent_config_stored(self) -> None:
        agent = StubAgent(config={"timeout_sec": 30})
        assert agent.config["timeout_sec"] == 30
