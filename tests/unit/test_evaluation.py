"""Unit tests for the evaluation harness and metrics."""

from typing import Any

from evaluation.harness import EvaluationHarness, TestCase, TestCaseLoader
from evaluation.metrics import EvaluationMetrics, EvaluationResult


class TestTestCase:
    def test_from_dict(self) -> None:
        data = {
            "test_id": "tc-001",
            "input": {"query": "test"},
            "expected_output": {"conclusion": "answer"},
            "tags": ["easy"],
            "difficulty": "easy",
        }
        tc = TestCase.from_dict(data)
        assert tc.test_id == "tc-001"
        assert tc.input_payload == {"query": "test"}
        assert tc.tags == ["easy"]


class TestTestCaseLoader:
    def test_load_from_nonexistent_dir(self) -> None:
        loader = TestCaseLoader("/nonexistent/path")
        cases = loader.load_all()
        assert cases == []

    def test_load_from_project_test_cases(self) -> None:
        loader = TestCaseLoader("evaluation/test_cases")
        cases = loader.load_all()
        assert len(cases) >= 1
        assert cases[0].test_id == "tc-001"

    def test_load_by_tags(self) -> None:
        loader = TestCaseLoader("evaluation/test_cases")
        cases = loader.load_by_tags(["factual"])
        assert all(any(t in tc.tags for t in ["factual"]) for tc in cases)


class TestEvaluationResult:
    def test_to_dict(self) -> None:
        result = EvaluationResult(
            test_id="tc-001",
            scores={"accuracy": 0.9},
            latency_ms=100.0,
            passed=True,
            trace=[{"agent_id": "validator", "status": "completed", "confidence_score": 0.9}],
        )
        d = result.to_dict()
        assert d["test_id"] == "tc-001"
        assert d["passed"] is True
        assert "agent_contributions" in d

    def test_agent_contributions(self) -> None:
        result = EvaluationResult(
            test_id="tc-001",
            scores={},
            latency_ms=50.0,
            passed=True,
            trace=[
                {
                    "agent_id": "validator",
                    "status": "completed",
                    "confidence_score": 0.95,
                    "step": 0,
                },
                {"agent_id": "reasoner", "status": "completed", "confidence_score": 0.8, "step": 1},
            ],
        )
        contributions = result._analyze_contributions()
        assert "validator" in contributions
        assert "reasoner" in contributions


class TestEvaluationMetrics:
    def test_accuracy_perfect_match(self) -> None:
        metrics = EvaluationMetrics()
        score = metrics.score_accuracy(
            {"key": "value"},
            {"key": "value"},
        )
        assert score == 1.0

    def test_accuracy_no_match(self) -> None:
        metrics = EvaluationMetrics()
        score = metrics.score_accuracy(
            {"a": 1},
            {"b": 2},
        )
        assert score == 0.0

    def test_accuracy_partial_match(self) -> None:
        metrics = EvaluationMetrics()
        score = metrics.score_accuracy(
            {"a": 1, "b": 2},
            {"a": 1, "c": 3},
        )
        assert 0.0 < score < 1.0

    def test_reasoning_fidelity_with_chain(self) -> None:
        metrics = EvaluationMetrics()
        result: dict[str, Any] = {
            "conclusion": "Answer",
            "reasoning_chain": [
                {"thought": "Step 1", "confidence": 0.8},
                {"thought": "Step 2", "confidence": 0.7},
                {"thought": "Step 3", "confidence": 0.9},
            ],
        }
        score = metrics.score_reasoning_fidelity(result, [])
        assert score > 0.5

    def test_reasoning_fidelity_no_chain(self) -> None:
        metrics = EvaluationMetrics()
        score = metrics.score_reasoning_fidelity({}, [])
        assert score == 0.3  # Default for missing chain

    def test_completion_score(self) -> None:
        metrics = EvaluationMetrics()
        from orchestrator.state_machine import OrchestrationState

        assert metrics.score_completion(OrchestrationState.COMPLETED) == 1.0
        assert metrics.score_completion(OrchestrationState.FAILED) == 0.0
        assert metrics.score_completion(None) == 0.0

    def test_latency_score(self) -> None:
        metrics = EvaluationMetrics({"latency_p95_target_ms": 5000})
        assert metrics.score_latency(1000) == 1.0
        assert metrics.score_latency(5000) == 1.0
        assert metrics.score_latency(7000) == 0.5
        assert metrics.score_latency(20000) == 0.0

    def test_confidence_calibration(self) -> None:
        metrics = EvaluationMetrics()
        trace = [
            {"confidence_score": 0.9, "status": "completed"},
            {"confidence_score": 0.1, "status": "failed"},
        ]
        score = metrics.score_confidence_calibration(trace)
        assert score > 0.8  # Well-calibrated


class TestEvaluationHarness:
    async def test_run_empty_evaluation(self) -> None:
        harness = EvaluationHarness(config={})

        async def mock_execute(task: dict[str, Any]) -> Any:
            pass  # pragma: no cover

        result = await harness.run_evaluation(mock_execute)
        assert result["summary"]["total"] == 0

    async def test_run_with_test_cases(self) -> None:
        harness = EvaluationHarness(config={"evaluation": {"thresholds": {}}})

        class MockResult:
            status: Any = None
            result: dict[str, Any] = {}
            trace: list[dict[str, Any]] = []

        from orchestrator.state_machine import OrchestrationState

        mock_result = MockResult()
        mock_result.status = OrchestrationState.COMPLETED
        mock_result.result = {"conclusion": "test answer"}

        async def mock_execute(task: dict[str, Any]) -> MockResult:
            return mock_result

        harness.set_test_cases(
            [
                TestCase(
                    test_id="tc-test",
                    input_payload={"query": "test"},
                    expected_output={"conclusion": "test answer"},
                )
            ]
        )

        result = await harness.run_evaluation(mock_execute)
        assert result["summary"]["total"] == 1
        assert len(result["results"]) == 1
