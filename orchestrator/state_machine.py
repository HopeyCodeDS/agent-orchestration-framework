"""Orchestration state machine — manages agent lifecycle and state transitions.

The Orchestrator is the ONLY class that calls agent.execute(). State transitions
are loaded from config/orchestration.yaml. Observability logging happens BEFORE
state transitions (ADR-005).
"""

import asyncio
import hashlib
import json
import uuid
from enum import Enum
from typing import Any

import structlog
import yaml

from agents.base import AgentRequest, AgentResponse, AgentStatus, BaseAgent
from orchestrator.circuit_breaker import AgentCircuitBreaker, CircuitBreakerOpenError

log = structlog.get_logger()


class OrchestrationState(Enum):
    """States in the orchestration pipeline."""

    VALIDATING = "validating"
    RETRIEVING = "retrieving"
    REASONING = "reasoning"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskResult:
    """Result of a completed orchestration task."""

    def __init__(
        self,
        task_id: str,
        status: OrchestrationState,
        result: dict[str, Any] | None = None,
        error: str | None = None,
        trace: list[dict[str, Any]] | None = None,
    ) -> None:
        self.task_id = task_id
        self.status = status
        self.result = result
        self.error = error
        self.trace = trace or []


class Orchestrator:
    """Central orchestrator that coordinates agent execution via a state machine.

    All agent calls go through circuit breakers. State transitions are driven
    by orchestration.yaml configuration. Observability logging occurs before
    every state transition (ADR-005).
    """

    def __init__(
        self,
        agents: list[BaseAgent],
        config: dict[str, Any],
    ) -> None:
        self.agents: dict[str, BaseAgent] = {agent.agent_id: agent for agent in agents}
        self.config = config

        orchestration_config = config.get("orchestration", {})
        self.max_steps: int = orchestration_config.get("max_steps", 10)
        self.state_graph: dict[str, Any] = orchestration_config.get("states", {})

        # Per-agent circuit breakers (ADR-003)
        self.circuit_breakers: dict[str, AgentCircuitBreaker] = {}
        for agent in agents:
            agent_config = self._get_agent_circuit_config(agent.agent_id)
            self.circuit_breakers[agent.agent_id] = AgentCircuitBreaker(
                agent_id=agent.agent_id,
                failure_threshold=agent_config.get("failure_threshold", 5),
                timeout_sec=agent_config.get("timeout_sec", 60),
            )

    @classmethod
    def from_yaml(
        cls,
        agents: list[BaseAgent],
        orchestration_yaml_path: str,
    ) -> "Orchestrator":
        """Create an Orchestrator from a YAML configuration file.

        Args:
            agents: List of agent instances to register.
            orchestration_yaml_path: Path to orchestration.yaml.

        Returns:
            Configured Orchestrator instance.
        """
        with open(orchestration_yaml_path) as f:
            config = yaml.safe_load(f)
        return cls(agents=agents, config=config)

    async def execute_task(self, task: dict[str, Any]) -> TaskResult:
        """Execute a full orchestration pipeline for the given task.

        Args:
            task: Dict containing at minimum an "id" key and task payload.

        Returns:
            TaskResult with final status, result data, and execution trace.
        """
        task_id = task.get("id", str(uuid.uuid4()))
        state = OrchestrationState(
            self.config.get("orchestration", {}).get("initial_state", "validating")
        )
        context: dict[str, Any] = {"task": task, "agent_history": [], "retrieval_attempt": 0}
        trace: list[dict[str, Any]] = []

        log.info(
            "orchestration.started",
            task_id=task_id,
            initial_state=state.value,
        )

        for step in range(self.max_steps):
            # Track retrieval passes so the retrieval agent knows when it is retrying
            if state == OrchestrationState.RETRIEVING:
                context["retrieval_attempt"] = context.get("retrieval_attempt", 0) + 1

            state_config = self.state_graph.get(state.value, {})

            if state_config.get("terminal", False):
                break

            agent_id = state_config.get("agent")
            if agent_id is None or agent_id not in self.agents:
                log.error(
                    "orchestration.no_agent_for_state",
                    task_id=task_id,
                    state=state.value,
                    agent_id=agent_id,
                )
                return TaskResult(
                    task_id=task_id,
                    status=OrchestrationState.FAILED,
                    error=f"No agent configured for state: {state.value}",
                    trace=trace,
                )

            current_agent = self.agents[agent_id]
            request = AgentRequest(
                task_id=task_id,
                payload=task.get("payload", task),
                context=context,
                max_retries=current_agent.config.get("max_retries", 2),
            )

            agent_timeout = current_agent.config.get("timeout_sec", 30)
            response = await self._execute_agent_with_protection(
                current_agent, request, agent_timeout
            )

            # Log BEFORE state transition (ADR-005)
            interaction_record = await self._log_agent_interaction(
                current_agent, request, response, state, step
            )
            trace.append(interaction_record)

            # Determine next state
            next_state = self._determine_next_state(state, response)

            if next_state == OrchestrationState.FAILED:
                log.warning(
                    "orchestration.failed",
                    task_id=task_id,
                    failed_at_state=state.value,
                    agent_id=agent_id,
                    error=response.error,
                )
                return TaskResult(
                    task_id=task_id,
                    status=OrchestrationState.FAILED,
                    error=response.error,
                    trace=trace,
                )

            if next_state == OrchestrationState.COMPLETED:
                log.info("orchestration.completed", task_id=task_id, steps=step + 1)
                return TaskResult(
                    task_id=task_id,
                    status=OrchestrationState.COMPLETED,
                    result=context.get("last_output"),
                    trace=trace,
                )

            state = next_state
            context = self._update_context(context, response)

        log.error("orchestration.max_steps_exceeded", task_id=task_id, max_steps=self.max_steps)
        return TaskResult(
            task_id=task_id,
            status=OrchestrationState.FAILED,
            error=f"Orchestration exceeded maximum steps ({self.max_steps})",
            trace=trace,
        )

    async def _execute_agent_with_protection(
        self,
        agent: BaseAgent,
        request: AgentRequest,
        timeout_sec: int,
    ) -> AgentResponse:
        """Execute an agent with circuit breaker protection and timeout.

        Args:
            agent: The agent to execute.
            request: The agent request.
            timeout_sec: Maximum execution time in seconds.

        Returns:
            AgentResponse — always returns a response, never raises.
        """
        circuit_breaker = self.circuit_breakers[agent.agent_id]

        try:
            response = await asyncio.wait_for(
                circuit_breaker.execute_with_protection(agent, request),
                timeout=timeout_sec,
            )
        except CircuitBreakerOpenError:
            log.warning(
                "orchestration.circuit_breaker_open",
                agent_id=agent.agent_id,
                task_id=request.task_id,
            )
            response = AgentResponse(
                agent_id=agent.agent_id,
                status=AgentStatus.FAILED,
                error=f"Circuit breaker open for agent: {agent.agent_id}",
                confidence_score=0.0,
            )
        except asyncio.TimeoutError:
            log.warning(
                "orchestration.agent_timeout",
                agent_id=agent.agent_id,
                task_id=request.task_id,
                timeout_sec=timeout_sec,
            )
            response = AgentResponse(
                agent_id=agent.agent_id,
                status=AgentStatus.FAILED,
                error=f"Agent timed out after {timeout_sec}s",
                confidence_score=0.0,
            )

        return response

    def _determine_next_state(
        self,
        current_state: OrchestrationState,
        response: AgentResponse,
    ) -> OrchestrationState:
        """Determine the next state based on agent response and state graph.

        Args:
            current_state: The current orchestration state.
            response: The agent's response.

        Returns:
            The next OrchestrationState to transition to.
        """
        state_config = self.state_graph.get(current_state.value, {})
        transitions = state_config.get("transitions", {})

        if response.status == AgentStatus.ESCALATED:
            next_state_str = transitions.get("escalated", "failed")
        elif response.status == AgentStatus.COMPLETED:
            next_state_str = transitions.get("completed", "completed")
        else:
            next_state_str = transitions.get("failed", "failed")

        return OrchestrationState(next_state_str)

    async def _log_agent_interaction(
        self,
        agent: BaseAgent,
        request: AgentRequest,
        response: AgentResponse,
        state: OrchestrationState,
        step: int,
    ) -> dict[str, Any]:
        """Log an agent interaction for observability (ADR-005).

        Hashes payloads instead of logging raw data (ADR-004).

        Args:
            agent: The agent that was executed.
            request: The request sent to the agent.
            response: The response from the agent.
            state: The orchestration state during execution.
            step: The step number in the orchestration.

        Returns:
            Dict record of the interaction for the trace.
        """
        payload_hash = hashlib.sha256(
            json.dumps(request.payload, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

        record = {
            "step": step,
            "state": state.value,
            "agent_id": agent.agent_id,
            "status": response.status.value,
            "confidence_score": response.confidence_score,
            "input_payload_hash": payload_hash,
            "error": response.error,
            "requires_human_review": response.requires_human_review,
        }

        log.info(
            "orchestration.agent_interaction",
            task_id=request.task_id,
            **record,
        )

        return record

    def _update_context(
        self,
        context: dict[str, Any],
        response: AgentResponse,
    ) -> dict[str, Any]:
        """Update shared context with the latest agent response.

        Args:
            context: The current orchestration context.
            response: The agent's response to incorporate.

        Returns:
            Updated context dict.
        """
        context["agent_history"].append(
            {
                "agent_id": response.agent_id,
                "status": response.status.value,
                "confidence_score": response.confidence_score,
                "output": response.output,
            }
        )
        if response.output:
            context["last_output"] = response.output
        return context

    def _get_agent_circuit_config(self, agent_id: str) -> dict[str, Any]:
        """Get circuit breaker config for an agent from the orchestration config."""
        states = self.config.get("orchestration", {}).get("states", {})
        for state_config in states.values():
            if state_config.get("agent") == agent_id:
                result: dict[str, Any] = state_config.get("circuit_breaker", {})
                return result
        return {}
