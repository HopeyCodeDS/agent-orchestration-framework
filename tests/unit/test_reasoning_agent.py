"""Unit tests for the ReasoningAgent."""

from typing import Any

from agents.base import AgentRequest, AgentStatus
from agents.reasoning_agent import (
    ChainOfThoughtReasoner,
    ReasoningAgent,
    ReasoningStep,
    ThoughtBranch,
    UncertaintyQuantifier,
)


class TestReasoningStep:
    def test_step_creation(self) -> None:
        step = ReasoningStep(step_number=1, thought="Testing", confidence=0.8)
        assert step.step_number == 1
        assert step.thought == "Testing"
        assert step.confidence == 0.8

    def test_confidence_clamped(self) -> None:
        step = ReasoningStep(step_number=1, thought="Test", confidence=1.5)
        assert step.confidence == 1.0

        step2 = ReasoningStep(step_number=1, thought="Test", confidence=-0.5)
        assert step2.confidence == 0.0

    def test_to_dict(self) -> None:
        step = ReasoningStep(step_number=1, thought="Test", confidence=0.7, evidence=["doc1"])
        d = step.to_dict()
        assert d["step_number"] == 1
        assert d["thought"] == "Test"
        assert d["confidence"] == 0.7
        assert d["evidence"] == ["doc1"]


class TestThoughtBranch:
    def test_empty_branch(self) -> None:
        branch = ThoughtBranch(branch_id="b1")
        assert branch.score == 0.0
        assert branch.steps == []

    def test_add_step_updates_score(self) -> None:
        branch = ThoughtBranch(branch_id="b1")
        branch.add_step(ReasoningStep(1, "Step 1", 0.8))
        assert branch.score > 0.0

    def test_score_is_geometric_mean(self) -> None:
        branch = ThoughtBranch(branch_id="b1")
        branch.add_step(ReasoningStep(1, "A", 0.9))
        branch.add_step(ReasoningStep(2, "B", 0.9))
        # Geometric mean of 0.9 and 0.9 = 0.9
        assert abs(branch.score - 0.9) < 0.01

    def test_to_dict(self) -> None:
        branch = ThoughtBranch(branch_id="b1")
        branch.add_step(ReasoningStep(1, "Test", 0.8))
        d = branch.to_dict()
        assert d["branch_id"] == "b1"
        assert len(d["steps"]) == 1


class TestUncertaintyQuantifier:
    def test_empty_steps(self) -> None:
        result = UncertaintyQuantifier.quantify_from_steps([])
        assert result["overall_uncertainty"] == 1.0

    def test_high_confidence_steps(self) -> None:
        steps = [
            ReasoningStep(1, "A", 0.9),
            ReasoningStep(2, "B", 0.8),
        ]
        result = UncertaintyQuantifier.quantify_from_steps(steps)
        assert result["overall_uncertainty"] < 0.3
        assert abs(result["avg_confidence"] - 0.85) < 1e-10

    def test_branch_agreement(self) -> None:
        branches = [ThoughtBranch("b1"), ThoughtBranch("b2")]
        branches[0].add_step(ReasoningStep(1, "A", 0.9))
        branches[1].add_step(ReasoningStep(1, "B", 0.9))
        result = UncertaintyQuantifier.quantify_from_branches(branches)
        assert result["branch_agreement"] > 0.8

    def test_empty_branches(self) -> None:
        result = UncertaintyQuantifier.quantify_from_branches([])
        assert result["branch_agreement"] == 0.0


class TestChainOfThoughtReasoner:
    def test_parse_reasoning_steps(self) -> None:
        reasoner = ChainOfThoughtReasoner({})
        content = (
            "Step 1: Analyze the question (confidence: 0.8)\n"
            "Step 2: Consider evidence (confidence: 0.7)\n"
            "Step 3: Formulate answer (confidence: 0.9)\n"
        )
        steps = reasoner._parse_reasoning_steps(content)
        assert len(steps) == 3
        assert steps[0].confidence == 0.8
        assert steps[2].confidence == 0.9

    def test_parse_no_confidence(self) -> None:
        reasoner = ChainOfThoughtReasoner({})
        content = "Step 1: Just a thought\n"
        steps = reasoner._parse_reasoning_steps(content)
        assert len(steps) == 1
        assert steps[0].confidence == 0.5  # Default

    def test_max_steps_limit(self) -> None:
        reasoner = ChainOfThoughtReasoner({"max_reasoning_steps": 2})
        content = "Step 1: A\nStep 2: B\nStep 3: C\n"
        steps = reasoner._parse_reasoning_steps(content)
        assert len(steps) == 2

    async def test_fallback_reasoning(self) -> None:
        reasoner = ChainOfThoughtReasoner({})
        steps = reasoner._fallback_reasoning("What is AI?", [])
        assert len(steps) >= 2
        assert steps[0].thought.startswith("Analyzing query")

    async def test_fallback_with_documents(self) -> None:
        reasoner = ChainOfThoughtReasoner({})
        steps = reasoner._fallback_reasoning("What is AI?", ["doc1", "doc2"])
        assert len(steps) >= 3  # Extra step for reviewing documents

    def test_extract_documents_from_context(self) -> None:
        context: dict[str, Any] = {
            "last_output": {
                "documents": [
                    {"document": "doc1"},
                    {"document": "doc2"},
                ]
            }
        }
        docs = ChainOfThoughtReasoner._extract_documents(context)
        assert len(docs) == 2
        assert docs[0] == "doc1"


class TestReasoningAgent:
    def _make_config(self) -> dict[str, Any]:
        return {"reasoning_mode": "chain_of_thought"}

    async def test_missing_query_fails(self) -> None:
        agent = ReasoningAgent(config=self._make_config())
        request = AgentRequest(task_id="t1", payload={})
        response = await agent.execute(request)
        assert response.status == AgentStatus.FAILED
        assert "No query" in (response.error or "")

    async def test_cot_reasoning(self) -> None:
        agent = ReasoningAgent(config=self._make_config())
        request = AgentRequest(task_id="t1", payload={"query": "What is AI?"})
        response = await agent.execute(request)
        assert response.status == AgentStatus.COMPLETED
        assert response.output is not None
        assert "conclusion" in response.output
        assert "reasoning_chain" in response.output

    async def test_can_handle_with_query(self) -> None:
        agent = ReasoningAgent(config=self._make_config())
        request = AgentRequest(task_id="t1", payload={"query": "test"})
        assert agent.can_handle(request) is True

    async def test_can_handle_without_query(self) -> None:
        agent = ReasoningAgent(config=self._make_config())
        request = AgentRequest(task_id="t1", payload={})
        assert agent.can_handle(request) is False

    async def test_health_probe(self) -> None:
        agent = ReasoningAgent(config=self._make_config())
        health = await agent.health_probe()
        assert health["healthy"] is True
        assert "chain_of_thought" in health["capabilities"]

    async def test_metadata_includes_uncertainty(self) -> None:
        agent = ReasoningAgent(config=self._make_config())
        request = AgentRequest(task_id="t1", payload={"query": "Explain AI"})
        response = await agent.execute(request)
        assert "uncertainty" in response.metadata
        assert "num_steps" in response.metadata
