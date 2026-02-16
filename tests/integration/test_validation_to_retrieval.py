"""Integration test: Validation -> Retrieval agent handoff."""

from typing import Any

from agents.base import AgentResponse, AgentStatus
from agents.validation_agent import ValidationAgent
from orchestrator.state_machine import OrchestrationState, Orchestrator
from tests.conftest import StubAgent


class TestValidationToRetrieval:
    async def test_successful_handoff(self) -> None:
        """Validated payload flows from validator to retriever."""
        config = {
            "orchestration": {
                "initial_state": "validating",
                "max_steps": 10,
                "states": {
                    "validating": {
                        "agent": "validator",
                        "transitions": {
                            "completed": "retrieving",
                            "failed": "failed",
                        },
                    },
                    "retrieving": {
                        "agent": "retriever",
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

        validation_config: dict[str, Any] = {
            "required_fields": ["query"],
            "field_types": {"query": "str"},
            "semantic_rules": [],
            "pii_detection_enabled": True,
        }

        validator = ValidationAgent(config=validation_config)
        retriever = StubAgent(
            agent_id="retriever",
            response=AgentResponse(
                agent_id="retriever",
                status=AgentStatus.COMPLETED,
                output={"documents": [{"document": "result doc"}]},
                confidence_score=0.85,
            ),
        )

        orchestrator = Orchestrator(agents=[validator, retriever], config=config)
        result = await orchestrator.execute_task(
            {"id": "int-1", "payload": {"query": "What is AI?"}}
        )

        assert result.status == OrchestrationState.COMPLETED
        assert len(result.trace) == 2
        assert result.trace[0]["agent_id"] == "validator"
        assert result.trace[1]["agent_id"] == "retriever"

    async def test_validation_failure_stops_pipeline(self) -> None:
        """Failed validation prevents retrieval from running."""
        config = {
            "orchestration": {
                "initial_state": "validating",
                "max_steps": 10,
                "states": {
                    "validating": {
                        "agent": "validator",
                        "transitions": {
                            "completed": "retrieving",
                            "failed": "failed",
                        },
                    },
                    "retrieving": {
                        "agent": "retriever",
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

        validation_config: dict[str, Any] = {
            "required_fields": ["query"],
            "pii_detection_enabled": True,
        }

        validator = ValidationAgent(config=validation_config)
        retriever = StubAgent(agent_id="retriever")

        orchestrator = Orchestrator(agents=[validator, retriever], config=config)
        # Missing required "query" field
        result = await orchestrator.execute_task({"id": "int-2", "payload": {"data": "no query"}})

        assert result.status == OrchestrationState.FAILED
        assert len(result.trace) == 1  # Only validator ran
        assert result.trace[0]["agent_id"] == "validator"
