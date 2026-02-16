"""Integration test: Full pipeline — Validation -> Retrieval -> Reasoning -> Verification."""

from typing import Any

from agents.base import AgentResponse, AgentStatus
from agents.validation_agent import ValidationAgent
from orchestrator.state_machine import OrchestrationState, Orchestrator
from tests.conftest import StubAgent


class TestFullPipeline:
    async def test_four_agent_pipeline(self) -> None:
        """Full pipeline with all four agents completing successfully."""
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
                            "completed": "reasoning",
                            "failed": "failed",
                        },
                    },
                    "reasoning": {
                        "agent": "reasoner",
                        "transitions": {
                            "completed": "verifying",
                            "failed": "failed",
                        },
                    },
                    "verifying": {
                        "agent": "verifier",
                        "transitions": {
                            "completed": "completed",
                            "failed": "reasoning",
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
            "pii_detection_enabled": True,
        }

        validator = ValidationAgent(config=validation_config)
        retriever = StubAgent(
            agent_id="retriever",
            response=AgentResponse(
                agent_id="retriever",
                status=AgentStatus.COMPLETED,
                output={"documents": [{"document": "Evidence document"}]},
                confidence_score=0.85,
            ),
        )
        reasoner = StubAgent(
            agent_id="reasoner",
            response=AgentResponse(
                agent_id="reasoner",
                status=AgentStatus.COMPLETED,
                output={
                    "conclusion": "Reasoned answer based on evidence",
                    "reasoning_chain": [{"thought": "Step 1", "confidence": 0.8}],
                },
                confidence_score=0.80,
            ),
        )
        verifier = StubAgent(
            agent_id="verifier",
            response=AgentResponse(
                agent_id="verifier",
                status=AgentStatus.COMPLETED,
                output={"verified": True, "verification_score": 0.9},
                confidence_score=0.90,
            ),
        )

        orchestrator = Orchestrator(
            agents=[validator, retriever, reasoner, verifier], config=config
        )
        result = await orchestrator.execute_task(
            {"id": "full-1", "payload": {"query": "What is machine learning?"}}
        )

        assert result.status == OrchestrationState.COMPLETED
        assert len(result.trace) == 4
        assert result.trace[0]["agent_id"] == "validator"
        assert result.trace[1]["agent_id"] == "retriever"
        assert result.trace[2]["agent_id"] == "reasoner"
        assert result.trace[3]["agent_id"] == "verifier"

    async def test_verification_failure_retries_reasoning(self) -> None:
        """Verification failure routes back to reasoning (per orchestration.yaml)."""
        call_count = 0

        class CountingReasonerStub(StubAgent):
            async def execute(self, request: Any) -> AgentResponse:
                nonlocal call_count
                call_count += 1
                return AgentResponse(
                    agent_id="reasoner",
                    status=AgentStatus.COMPLETED,
                    output={"conclusion": f"Attempt {call_count}"},
                    confidence_score=0.7,
                )

        config = {
            "orchestration": {
                "initial_state": "reasoning",
                "max_steps": 5,
                "states": {
                    "reasoning": {
                        "agent": "reasoner",
                        "transitions": {
                            "completed": "verifying",
                            "failed": "failed",
                        },
                    },
                    "verifying": {
                        "agent": "verifier",
                        "transitions": {
                            "completed": "completed",
                            "failed": "reasoning",  # Retry loop
                        },
                    },
                    "completed": {"terminal": True},
                    "failed": {"terminal": True},
                },
            }
        }

        # Verifier fails first time, succeeds second
        verify_calls = 0

        class VerifierStub(StubAgent):
            async def execute(self, request: Any) -> AgentResponse:
                nonlocal verify_calls
                verify_calls += 1
                if verify_calls < 2:
                    return AgentResponse(
                        agent_id="verifier",
                        status=AgentStatus.FAILED,
                        error="Verification failed",
                        confidence_score=0.3,
                    )
                return AgentResponse(
                    agent_id="verifier",
                    status=AgentStatus.COMPLETED,
                    output={"verified": True},
                    confidence_score=0.9,
                )

        reasoner = CountingReasonerStub(agent_id="reasoner")
        verifier = VerifierStub(agent_id="verifier")

        orchestrator = Orchestrator(agents=[reasoner, verifier], config=config)
        result = await orchestrator.execute_task({"id": "retry-1", "payload": {"query": "test"}})

        assert result.status == OrchestrationState.COMPLETED
        assert call_count == 2  # Reasoner called twice
        assert verify_calls == 2  # Verifier called twice
        # Trace: reason -> verify(fail) -> reason -> verify(pass)
        assert len(result.trace) == 4

    async def test_pipeline_context_propagation(self) -> None:
        """Verify context from earlier agents reaches later agents."""
        received_context: dict[str, Any] = {}

        class ContextCapturingAgent(StubAgent):
            async def execute(self, request: Any) -> AgentResponse:
                nonlocal received_context
                received_context = dict(request.context)
                return AgentResponse(
                    agent_id="reasoner",
                    status=AgentStatus.COMPLETED,
                    output={"conclusion": "captured"},
                    confidence_score=0.8,
                )

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

        validation_config: dict[str, Any] = {
            "required_fields": ["query"],
            "pii_detection_enabled": False,
        }

        validator = ValidationAgent(config=validation_config)
        reasoner = ContextCapturingAgent(agent_id="reasoner")

        orchestrator = Orchestrator(agents=[validator, reasoner], config=config)
        await orchestrator.execute_task({"id": "ctx-1", "payload": {"query": "test"}})

        # Reasoner should have received context with validator's history
        assert "agent_history" in received_context
        assert len(received_context["agent_history"]) >= 1
        assert received_context["agent_history"][0]["agent_id"] == "validator"
