"""Unit tests for the VerificationAgent."""

from typing import Any

from agents.base import AgentRequest, AgentStatus
from agents.verification_agent import (
    ConsistencyChecker,
    FactChecker,
    HallucinationDetector,
    VerificationAgent,
)


class TestFactChecker:
    def test_supported_claim(self) -> None:
        checker = FactChecker({})
        result = checker.check_claims(
            "Paris is the capital of France",
            ["Paris is the capital of France and a major European city"],
        )
        assert result["supported"] is True
        assert result["support_score"] > 0.0

    def test_unsupported_claim(self) -> None:
        checker = FactChecker({"support_threshold": 0.5})
        result = checker.check_claims(
            "The moon is made of cheese",
            ["The moon orbits the Earth at a distance of 384,400 km"],
        )
        assert result["support_score"] <= 0.5

    def test_no_sources(self) -> None:
        checker = FactChecker({})
        result = checker.check_claims("Any claim", [])
        assert result["supported"] is False
        assert result["support_score"] == 0.0

    def test_multiple_sources(self) -> None:
        checker = FactChecker({})
        result = checker.check_claims(
            "Python is a programming language",
            [
                "Python is a popular programming language",
                "Java is also a programming language",
            ],
        )
        assert result["support_score"] > 0.0
        assert "best_source_index" in result


class TestHallucinationDetector:
    def test_clean_output(self) -> None:
        detector = HallucinationDetector({})
        result = detector.detect(
            {"conclusion": "The answer is Paris", "reasoning_chain": []},
            ["Paris is the capital of France"],
        )
        assert result["hallucination_score"] < 0.5

    def test_vague_authority_flagged(self) -> None:
        detector = HallucinationDetector({})
        result = detector.detect(
            {
                "conclusion": "As everyone knows, the Earth is flat. Research proves it.",
                "reasoning_chain": [],
            },
            [],
        )
        assert any(f["type"] == "vague_authority" for f in result["flags"])

    def test_ungrounded_reasoning(self) -> None:
        detector = HallucinationDetector({})
        result = detector.detect(
            {
                "conclusion": "Result",
                "reasoning_chain": [
                    {"thought": "completely random unrelated content xyz abc"},
                    {"thought": "more random unrelated stuff qwerty"},
                ],
            },
            ["The capital of France is Paris"],
        )
        # Should flag ungrounded steps
        assert result["hallucination_score"] > 0.0

    def test_grounded_reasoning(self) -> None:
        detector = HallucinationDetector({})
        result = detector.detect(
            {
                "conclusion": "Paris is the capital",
                "reasoning_chain": [
                    {"thought": "Paris is the capital of France"},
                ],
            },
            ["Paris is the capital of France"],
        )
        assert result["hallucination_score"] < 0.5


class TestConsistencyChecker:
    def test_consistent_chain(self) -> None:
        checker = ConsistencyChecker()
        result = checker.check(
            {
                "conclusion": "The answer involves Python programming",
                "reasoning_chain": [
                    {"thought": "Python is a programming language"},
                    {"thought": "Python programming is widely used"},
                ],
            }
        )
        assert result["consistent"] is True
        assert result["consistency_score"] >= 0.6

    def test_empty_chain(self) -> None:
        checker = ConsistencyChecker()
        result = checker.check({"conclusion": "", "reasoning_chain": []})
        assert result["consistency_score"] == 0.5

    def test_conclusion_follows_check(self) -> None:
        checker = ConsistencyChecker()
        follows = checker._conclusion_follows(
            "Python is great for data science",
            [{"thought": "Python is widely used in data science"}],
        )
        assert follows is True

    def test_conclusion_does_not_follow(self) -> None:
        checker = ConsistencyChecker()
        follows = checker._conclusion_follows(
            "xyz abc completely unrelated",
            [{"thought": "Python is a programming language"}],
        )
        assert follows is False


class TestVerificationAgent:
    def _make_config(self) -> dict[str, Any]:
        return {"verification_threshold": 0.3}

    async def test_no_reasoning_output_fails(self) -> None:
        agent = VerificationAgent(config=self._make_config())
        request = AgentRequest(task_id="t1", payload={})
        response = await agent.execute(request)
        assert response.status == AgentStatus.FAILED
        assert "No reasoning output" in (response.error or "")

    async def test_verification_with_reasoning_context(self) -> None:
        agent = VerificationAgent(config=self._make_config())
        request = AgentRequest(
            task_id="t1",
            payload={"query": "test"},
            context={
                "last_output": {
                    "conclusion": "Paris is the capital of France",
                    "reasoning_chain": [
                        {"thought": "France is a country in Europe"},
                        {"thought": "Paris is the capital of France"},
                    ],
                },
            },
        )
        response = await agent.execute(request)
        # Should complete (may pass or fail based on scores)
        assert response.status in (AgentStatus.COMPLETED, AgentStatus.FAILED)
        assert "checks" in response.metadata

    async def test_verification_with_payload_conclusion(self) -> None:
        agent = VerificationAgent(config=self._make_config())
        request = AgentRequest(
            task_id="t1",
            payload={
                "conclusion": "Simple answer",
                "reasoning_chain": [{"thought": "Simple reasoning"}],
            },
        )
        response = await agent.execute(request)
        assert response.status in (AgentStatus.COMPLETED, AgentStatus.FAILED)

    async def test_can_handle_with_reasoning_context(self) -> None:
        agent = VerificationAgent(config=self._make_config())
        request = AgentRequest(
            task_id="t1",
            payload={},
            context={"last_output": {"conclusion": "test", "reasoning_chain": []}},
        )
        assert agent.can_handle(request) is True

    async def test_can_handle_without_reasoning(self) -> None:
        agent = VerificationAgent(config=self._make_config())
        request = AgentRequest(task_id="t1", payload={"data": "something"})
        # No reasoning context and no conclusion in payload — agent cannot handle
        assert agent.can_handle(request) is False

    async def test_health_probe(self) -> None:
        agent = VerificationAgent(config=self._make_config())
        health = await agent.health_probe()
        assert health["healthy"] is True
        assert "fact_checking" in health["capabilities"]
