"""Health endpoint — detailed health checks with agent and circuit breaker status.

Reports per-agent health, circuit breaker states, and system metrics.
"""

from typing import Any

import structlog

from agents.base import BaseAgent
from api.models import HealthResponse
from orchestrator.circuit_breaker import AgentCircuitBreaker

log = structlog.get_logger()


class HealthChecker:
    """Aggregates health status from all system components.

    Probes each agent and reports circuit breaker states to produce
    a comprehensive health response.
    """

    def __init__(
        self,
        agents: dict[str, BaseAgent],
        circuit_breakers: dict[str, AgentCircuitBreaker],
    ) -> None:
        self.agents = agents
        self.circuit_breakers = circuit_breakers

    async def check(self) -> HealthResponse:
        """Run health checks across all agents and circuit breakers.

        Returns:
            HealthResponse with overall and per-component status.
        """
        agent_statuses: dict[str, Any] = {}
        circuit_statuses: dict[str, Any] = {}

        for agent_id, agent in self.agents.items():
            try:
                status = await agent.health_probe(timeout_sec=2)
                agent_statuses[agent_id] = status
            except Exception as exc:
                log.error(
                    "health.agent_probe_failed",
                    agent_id=agent_id,
                    error_type=type(exc).__name__,
                )
                agent_statuses[agent_id] = {
                    "healthy": False,
                    "error": type(exc).__name__,
                }

        for agent_id, cb in self.circuit_breakers.items():
            circuit_statuses[agent_id] = cb.get_status()

        # Determine overall status
        all_healthy = all(s.get("healthy", False) for s in agent_statuses.values())
        any_open = any(s.get("state") == "open" for s in circuit_statuses.values())

        if all_healthy and not any_open:
            overall = "healthy"
        elif any_open:
            overall = "degraded"
        else:
            overall = "unhealthy"

        return HealthResponse(
            status=overall,
            agents=agent_statuses,
            circuits=circuit_statuses,
        )
