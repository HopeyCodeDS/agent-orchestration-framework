"""Circuit breaker pattern — prevents cascading failures across agents.

Each agent gets its own circuit breaker instance (ADR-003). States:
CLOSED (normal) -> OPEN (failing, reject fast) -> HALF_OPEN (probing recovery).
"""

import time
from enum import Enum
from typing import Any

import structlog

from agents.base import AgentRequest, AgentResponse, AgentStatus, BaseAgent

log = structlog.get_logger()


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpenError(Exception):
    """Raised when a circuit breaker is open and rejecting requests."""

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        super().__init__(f"Circuit breaker open for agent: {agent_id}")


class AgentCircuitBreaker:
    """Per-agent circuit breaker that prevents cascading failures.

    Tracks consecutive failures and transitions between CLOSED, OPEN,
    and HALF_OPEN states. When OPEN, requests are rejected immediately.
    After a timeout, one probe request is allowed to test recovery.
    """

    def __init__(
        self,
        agent_id: str,
        failure_threshold: int = 5,
        timeout_sec: int = 60,
    ) -> None:
        self.agent_id = agent_id
        self.failure_threshold = failure_threshold
        self.timeout_sec = timeout_sec
        self.failure_count: int = 0
        self.last_failure_time: float | None = None
        self.state: CircuitState = CircuitState.CLOSED

    async def execute_with_protection(
        self,
        agent: BaseAgent,
        request: AgentRequest,
    ) -> AgentResponse:
        """Execute an agent call with circuit breaker protection.

        Args:
            agent: The agent to execute.
            request: The agent request.

        Returns:
            AgentResponse from the agent.

        Raises:
            CircuitBreakerOpenError: If the circuit is open and not ready to probe.
        """
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                log.info(
                    "circuit_breaker.half_open",
                    agent_id=self.agent_id,
                )
            else:
                raise CircuitBreakerOpenError(self.agent_id)

        response = await agent.execute(request)

        if response.status in (AgentStatus.FAILED, AgentStatus.ESCALATED):
            self._record_failure()
        else:
            self._record_success()

        return response

    def _record_failure(self) -> None:
        """Record a failure and potentially open the circuit."""
        self.failure_count += 1
        self.last_failure_time = time.monotonic()

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            log.warning(
                "circuit_breaker.opened",
                agent_id=self.agent_id,
                failure_count=self.failure_count,
                threshold=self.failure_threshold,
            )

    def _record_success(self) -> None:
        """Record a success and reset the circuit if it was half-open."""
        if self.state == CircuitState.HALF_OPEN:
            log.info(
                "circuit_breaker.closed",
                agent_id=self.agent_id,
            )
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt a reset probe."""
        if self.last_failure_time is None:
            return True
        return (time.monotonic() - self.last_failure_time) >= self.timeout_sec

    def get_status(self) -> dict[str, Any]:
        """Return circuit breaker status for health checks."""
        return {
            "agent_id": self.agent_id,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "timeout_sec": self.timeout_sec,
        }
