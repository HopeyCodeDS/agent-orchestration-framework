"""Unit tests for the AgentCircuitBreaker."""

import pytest

from agents.base import AgentRequest
from orchestrator.circuit_breaker import (
    AgentCircuitBreaker,
    CircuitBreakerOpenError,
    CircuitState,
)
from tests.conftest import FailingAgent, StubAgent


class TestCircuitBreakerStates:
    def test_initial_state_is_closed(self, circuit_breaker: AgentCircuitBreaker) -> None:
        assert circuit_breaker.state == CircuitState.CLOSED

    async def test_stays_closed_on_success(
        self,
        circuit_breaker: AgentCircuitBreaker,
        sample_request: AgentRequest,
    ) -> None:
        agent = StubAgent()
        await circuit_breaker.execute_with_protection(agent, sample_request)
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.failure_count == 0

    async def test_opens_after_threshold_failures(
        self,
        sample_request: AgentRequest,
    ) -> None:
        cb = AgentCircuitBreaker(agent_id="test", failure_threshold=3, timeout_sec=60)
        agent = FailingAgent()

        for _ in range(3):
            await cb.execute_with_protection(agent, sample_request)

        assert cb.state == CircuitState.OPEN
        assert cb.failure_count == 3

    async def test_open_circuit_rejects_requests(
        self,
        sample_request: AgentRequest,
    ) -> None:
        cb = AgentCircuitBreaker(agent_id="test", failure_threshold=2, timeout_sec=600)
        agent = FailingAgent()

        # Trip the circuit
        for _ in range(2):
            await cb.execute_with_protection(agent, sample_request)

        assert cb.state == CircuitState.OPEN

        with pytest.raises(CircuitBreakerOpenError):
            await cb.execute_with_protection(agent, sample_request)

    async def test_half_open_after_timeout(
        self,
        sample_request: AgentRequest,
    ) -> None:
        cb = AgentCircuitBreaker(agent_id="test", failure_threshold=2, timeout_sec=0)
        agent = FailingAgent()

        # Trip the circuit
        for _ in range(2):
            await cb.execute_with_protection(agent, sample_request)

        assert cb.state == CircuitState.OPEN

        # With timeout_sec=0, should immediately go to HALF_OPEN
        await cb.execute_with_protection(agent, sample_request)
        # Still fails so it should re-open
        assert cb.state == CircuitState.OPEN

    async def test_half_open_recovers_on_success(
        self,
        sample_request: AgentRequest,
    ) -> None:
        cb = AgentCircuitBreaker(agent_id="test", failure_threshold=2, timeout_sec=0)
        failing = FailingAgent()
        success = StubAgent()

        # Trip the circuit
        for _ in range(2):
            await cb.execute_with_protection(failing, sample_request)

        assert cb.state == CircuitState.OPEN

        # Recovery attempt succeeds
        await cb.execute_with_protection(success, sample_request)
        assert cb.state == CircuitState.CLOSED  # type: ignore[comparison-overlap]  # state mutated by execute
        assert cb.failure_count == 0

    def test_get_status(self, circuit_breaker: AgentCircuitBreaker) -> None:
        status = circuit_breaker.get_status()
        assert status["state"] == "closed"
        assert status["failure_count"] == 0
        assert status["failure_threshold"] == 3
