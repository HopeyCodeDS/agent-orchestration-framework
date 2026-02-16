"""Evaluation harness — automated test runner for orchestration quality.

Runs tasks against ground truth and produces multi-dimensional scoring
including accuracy, reasoning fidelity, retrieval precision, hallucination
detection, and latency metrics.
"""

import json
import time
from pathlib import Path
from typing import Any

import structlog
import yaml

from evaluation.metrics import EvaluationMetrics, EvaluationResult

log = structlog.get_logger()


class TestCase:
    """A single evaluation test case with input, expected output, and metadata.

    Attributes:
        test_id: Unique identifier for this test case.
        input_payload: The task input to run through orchestration.
        expected_output: Ground truth for comparison.
        tags: Optional tags for filtering test cases.
        difficulty: Difficulty level (easy, medium, hard).
    """

    def __init__(
        self,
        test_id: str,
        input_payload: dict[str, Any],
        expected_output: dict[str, Any],
        tags: list[str] | None = None,
        difficulty: str = "medium",
    ) -> None:
        self.test_id = test_id
        self.input_payload = input_payload
        self.expected_output = expected_output
        self.tags = tags or []
        self.difficulty = difficulty

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TestCase":
        """Create a TestCase from a dictionary.

        Args:
            data: Dict with test case fields.

        Returns:
            TestCase instance.
        """
        return cls(
            test_id=data["test_id"],
            input_payload=data["input"],
            expected_output=data["expected_output"],
            tags=data.get("tags", []),
            difficulty=data.get("difficulty", "medium"),
        )


class TestCaseLoader:
    """Loads evaluation test cases from YAML and JSON files."""

    def __init__(self, test_cases_dir: str) -> None:
        self.test_cases_dir = Path(test_cases_dir)

    def load_all(self) -> list[TestCase]:
        """Load all test cases from the configured directory.

        Returns:
            List of TestCase objects.
        """
        test_cases: list[TestCase] = []

        if not self.test_cases_dir.exists():
            log.warning(
                "evaluation.test_cases_dir_missing",
                path=str(self.test_cases_dir),
            )
            return test_cases

        for path in sorted(self.test_cases_dir.iterdir()):
            if path.suffix in (".yaml", ".yml"):
                test_cases.extend(self._load_yaml(path))
            elif path.suffix == ".json":
                test_cases.extend(self._load_json(path))

        log.info("evaluation.loaded_test_cases", count=len(test_cases))
        return test_cases

    def load_by_tags(self, tags: list[str]) -> list[TestCase]:
        """Load test cases filtered by tags.

        Args:
            tags: Tags to filter by (any match).

        Returns:
            Filtered list of TestCase objects.
        """
        all_cases = self.load_all()
        tag_set = set(tags)
        return [tc for tc in all_cases if tag_set & set(tc.tags)]

    @staticmethod
    def _load_yaml(path: Path) -> list[TestCase]:
        """Load test cases from a YAML file.

        Args:
            path: Path to the YAML file.

        Returns:
            List of TestCase objects.
        """
        try:
            with open(path) as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict) or "test_cases" not in data:
                return []
            return [TestCase.from_dict(tc) for tc in data["test_cases"]]
        except Exception as exc:
            log.error(
                "evaluation.yaml_load_error",
                path=str(path),
                error_type=type(exc).__name__,
            )
            return []

    @staticmethod
    def _load_json(path: Path) -> list[TestCase]:
        """Load test cases from a JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            List of TestCase objects.
        """
        try:
            with open(path) as f:
                data = json.load(f)
            if not isinstance(data, dict) or "test_cases" not in data:
                return []
            return [TestCase.from_dict(tc) for tc in data["test_cases"]]
        except Exception as exc:
            log.error(
                "evaluation.json_load_error",
                path=str(path),
                error_type=type(exc).__name__,
            )
            return []


class EvaluationHarness:
    """Automated evaluation harness for orchestration quality.

    Runs test cases through the orchestrator and produces comprehensive
    scoring across multiple dimensions.
    """

    def __init__(
        self,
        config: dict[str, Any],
        test_cases: list[TestCase] | None = None,
    ) -> None:
        self.config = config
        eval_config = config.get("evaluation", {})
        self.thresholds = eval_config.get("thresholds", {})
        self.latency_targets = eval_config.get("latency", {})
        self.metrics = EvaluationMetrics(self.thresholds)
        self._test_cases = test_cases or []

    def set_test_cases(self, test_cases: list[TestCase]) -> None:
        """Set the test cases for evaluation.

        Args:
            test_cases: List of test cases to evaluate.
        """
        self._test_cases = test_cases

    async def run_evaluation(
        self,
        execute_fn: Any,
    ) -> dict[str, Any]:
        """Run all test cases and produce aggregate results.

        Args:
            execute_fn: Async callable that takes a task dict and returns a result.
                Expected signature: async (dict) -> TaskResult

        Returns:
            Dict with individual and aggregate evaluation results.
        """
        results: list[EvaluationResult] = []

        for test_case in self._test_cases:
            result = await self._evaluate_single(test_case, execute_fn)
            results.append(result)

        aggregate = self._aggregate_results(results)

        log.info(
            "evaluation.completed",
            total_cases=len(results),
            passed=aggregate["summary"]["passed"],
            failed=aggregate["summary"]["failed"],
        )

        return aggregate

    async def _evaluate_single(
        self,
        test_case: TestCase,
        execute_fn: Any,
    ) -> EvaluationResult:
        """Evaluate a single test case.

        Args:
            test_case: The test case to evaluate.
            execute_fn: The orchestration function to call.

        Returns:
            EvaluationResult with all scores.
        """
        start_time = time.monotonic()

        try:
            task = {
                "id": test_case.test_id,
                "payload": test_case.input_payload,
            }
            task_result = await execute_fn(task)
            latency_ms = (time.monotonic() - start_time) * 1000

            # Score multiple dimensions
            scores = self.metrics.score_all(
                task_result=task_result,
                expected=test_case.expected_output,
                latency_ms=latency_ms,
            )

            return EvaluationResult(
                test_id=test_case.test_id,
                scores=scores,
                latency_ms=latency_ms,
                passed=self._check_thresholds(scores),
                error=None,
                trace=getattr(task_result, "trace", []),
            )

        except Exception as exc:
            latency_ms = (time.monotonic() - start_time) * 1000
            log.error(
                "evaluation.test_case_error",
                test_id=test_case.test_id,
                error_type=type(exc).__name__,
            )
            return EvaluationResult(
                test_id=test_case.test_id,
                scores={},
                latency_ms=latency_ms,
                passed=False,
                error=f"{type(exc).__name__}: {exc}",
                trace=[],
            )

    def _check_thresholds(self, scores: dict[str, float]) -> bool:
        """Check if scores meet configured thresholds.

        Args:
            scores: Dict of metric name to score value.

        Returns:
            True if all thresholds are met.
        """
        accuracy_min = self.thresholds.get("accuracy_minimum", 0.7)
        if scores.get("accuracy", 0.0) < accuracy_min:
            return False

        hallucination_max: float = self.thresholds.get("hallucination_maximum", 0.2)
        hallucination_score: float = scores.get("hallucination_score", 0.0)
        return hallucination_score <= hallucination_max

    def _aggregate_results(
        self,
        results: list[EvaluationResult],
    ) -> dict[str, Any]:
        """Aggregate individual results into a summary.

        Args:
            results: List of individual evaluation results.

        Returns:
            Dict with summary statistics and individual results.
        """
        if not results:
            return {
                "summary": {"total": 0, "passed": 0, "failed": 0, "pass_rate": 0.0},
                "results": [],
                "metrics": {},
            }

        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed
        latencies = [r.latency_ms for r in results]

        # Aggregate each metric
        all_scores: dict[str, list[float]] = {}
        for result in results:
            for metric, value in result.scores.items():
                all_scores.setdefault(metric, []).append(value)

        metric_averages = {
            metric: sum(values) / len(values) for metric, values in all_scores.items()
        }

        return {
            "summary": {
                "total": len(results),
                "passed": passed,
                "failed": failed,
                "pass_rate": passed / len(results),
            },
            "metrics": metric_averages,
            "latency": {
                "avg_ms": sum(latencies) / len(latencies),
                "max_ms": max(latencies),
                "min_ms": min(latencies),
            },
            "results": [r.to_dict() for r in results],
        }
