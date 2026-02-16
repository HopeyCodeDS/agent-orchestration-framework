"""Unit tests for the Orchestrator state machine."""

from typing import Any

from agents.base import AgentResponse, AgentStatus
from orchestrator.state_machine import OrchestrationState, Orchestrator
from tests.conftest import FailingAgent, StubAgent


class TestOrchestrationState:
    def test_state_values(self) -> None:
        assert OrchestrationState.VALIDATING.value == "validating"
        assert OrchestrationState.COMPLETED.value == "completed"
        assert OrchestrationState.FAILED.value == "failed"


class TestOrchestrator:
    async def test_successful_single_agent_flow(
        self,
        orchestration_config: dict[str, Any],
    ) -> None:
        agent = StubAgent(agent_id="validator")
        orchestrator = Orchestrator(agents=[agent], config=orchestration_config)

        result = await orchestrator.execute_task({"id": "task-1", "payload": {"query": "test"}})

        assert result.status == OrchestrationState.COMPLETED
        assert result.task_id == "task-1"
        assert len(result.trace) == 1
        assert result.trace[0]["agent_id"] == "validator"

    async def test_failed_agent_leads_to_failed_state(
        self,
        orchestration_config: dict[str, Any],
    ) -> None:
        agent = FailingAgent(agent_id="validator")
        orchestrator = Orchestrator(agents=[agent], config=orchestration_config)

        result = await orchestrator.execute_task({"id": "task-2", "payload": {"query": "test"}})

        assert result.status == OrchestrationState.FAILED
        assert result.error is not None
        assert len(result.trace) == 1

    async def test_multi_agent_flow(self) -> None:
        config = {
            "orchestration": {
                "initial_state": "validating",
                "max_steps": 10,
                "states": {
                    "validating": {
                        "agent": "validator",
                        "transitions": {
                            "completed": "reasoning",
                            "failed": "failed",
                        },
                    },
                    "reasoning": {
                        "agent": "reasoner",
                        "transitions": {
                            "completed": "completed",
                            "failed": "failed",
                        },
                    },
                    "completed": {"terminal": True},
                    "failed": {"terminal": True},
                },
            }
        }

        validator = StubAgent(agent_id="validator")
        reasoner = StubAgent(agent_id="reasoner")
        orchestrator = Orchestrator(agents=[validator, reasoner], config=config)

        result = await orchestrator.execute_task({"id": "task-3", "payload": {"query": "test"}})

        assert result.status == OrchestrationState.COMPLETED
        assert len(result.trace) == 2
        assert result.trace[0]["agent_id"] == "validator"
        assert result.trace[1]["agent_id"] == "reasoner"

    async def test_missing_agent_for_state_fails(self) -> None:
        config = {
            "orchestration": {
                "initial_state": "validating",
                "max_steps": 10,
                "states": {
                    "validating": {
                        "agent": "nonexistent",
                        "transitions": {"completed": "completed", "failed": "failed"},
                    },
                    "completed": {"terminal": True},
                    "failed": {"terminal": True},
                },
            }
        }

        orchestrator = Orchestrator(agents=[], config=config)
        result = await orchestrator.execute_task({"id": "task-4", "payload": {}})

        assert result.status == OrchestrationState.FAILED
        assert "No agent" in (result.error or "")

    async def test_max_steps_guard(self) -> None:
        """Verify the max_steps guard prevents infinite loops."""
        # Create a config where reasoning loops back to itself
        config = {
            "orchestration": {
                "initial_state": "reasoning",
                "max_steps": 3,
                "states": {
                    "reasoning": {
                        "agent": "reasoner",
                        "transitions": {
                            "completed": "reasoning",  # Loops back
                            "failed": "failed",
                        },
                    },
                    "failed": {"terminal": True},
                },
            }
        }

        reasoner = StubAgent(agent_id="reasoner")
        orchestrator = Orchestrator(agents=[reasoner], config=config)

        result = await orchestrator.execute_task({"id": "task-5", "payload": {}})

        assert result.status == OrchestrationState.FAILED
        assert "maximum steps" in (result.error or "").lower()
        assert len(result.trace) == 3

    async def test_trace_records_all_interactions(
        self,
        orchestration_config: dict[str, Any],
    ) -> None:
        agent = StubAgent(agent_id="validator")
        orchestrator = Orchestrator(agents=[agent], config=orchestration_config)

        result = await orchestrator.execute_task({"id": "task-6", "payload": {"query": "test"}})

        assert len(result.trace) >= 1
        record = result.trace[0]
        assert "step" in record
        assert "state" in record
        assert "agent_id" in record
        assert "confidence_score" in record
        assert "input_payload_hash" in record

    async def test_context_updated_between_agents(self) -> None:
        config = {
            "orchestration": {
                "initial_state": "validating",
                "max_steps": 10,
                "states": {
                    "validating": {
                        "agent": "validator",
                        "transitions": {"completed": "reasoning", "failed": "failed"},
                    },
                    "reasoning": {
                        "agent": "reasoner",
                        "transitions": {"completed": "completed", "failed": "failed"},
                    },
                    "completed": {"terminal": True},
                    "failed": {"terminal": True},
                },
            }
        }

        validator = StubAgent(
            agent_id="validator",
            response=AgentResponse(
                agent_id="validator",
                status=AgentStatus.COMPLETED,
                output={"validated": True},
                confidence_score=0.9,
            ),
        )
        reasoner = StubAgent(agent_id="reasoner")
        orchestrator = Orchestrator(agents=[validator, reasoner], config=config)

        result = await orchestrator.execute_task({"id": "task-7", "payload": {"query": "test"}})

        assert result.status == OrchestrationState.COMPLETED
