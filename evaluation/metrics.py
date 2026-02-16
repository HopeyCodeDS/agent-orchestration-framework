"""Custom evaluation metrics — reasoning fidelity, hallucination scoring, and more.

Multi-dimensional scoring beyond simple accuracy to evaluate the
quality of agent orchestration outputs.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvaluationResult:
    """Result of evaluating a single test case.

    Attributes:
        test_id: The test case identifier.
        scores: Dict of metric name to score value.
        latency_ms: Execution time in milliseconds.
        passed: Whether the test case passed all thresholds.
        error: Error message if evaluation failed.
        trace: Execution trace for agent contribution analysis.
    """

    test_id: str
    scores: dict[str, float]
    latency_ms: float
    passed: bool
    error: str | None = None
    trace: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "test_id": self.test_id,
            "scores": self.scores,
            "latency_ms": round(self.latency_ms, 2),
            "passed": self.passed,
            "error": self.error,
            "agent_contributions": self._analyze_contributions(),
        }

    def _analyze_contributions(self) -> dict[str, Any]:
        """Analyze per-agent contributions from the trace.

        Returns:
            Dict mapping agent_id to contribution metrics.
        """
        contributions: dict[str, Any] = {}
        for record in self.trace:
            agent_id = record.get("agent_id", "unknown")
            contributions[agent_id] = {
                "status": record.get("status"),
                "confidence_score": record.get("confidence_score", 0.0),
                "step": record.get("step"),
            }
        return contributions


class EvaluationMetrics:
    """Multi-dimensional scoring for orchestration evaluation.

    Provides accuracy, reasoning fidelity, retrieval precision,
    hallucination scoring, and latency metrics.
    """

    def __init__(self, thresholds: dict[str, float] | None = None) -> None:
        self.thresholds = thresholds or {}

    def score_all(
        self,
        task_result: Any,
        expected: dict[str, Any],
        latency_ms: float,
    ) -> dict[str, float]:
        """Compute all evaluation metrics.

        Args:
            task_result: The orchestration TaskResult.
            expected: The expected/ground-truth output.
            latency_ms: Execution time in milliseconds.

        Returns:
            Dict of metric name to score value (0.0-1.0).
        """
        result_output = getattr(task_result, "result", {}) or {}
        trace = getattr(task_result, "trace", []) or []
        status = getattr(task_result, "status", None)

        return {
            "accuracy": self.score_accuracy(result_output, expected),
            "reasoning_fidelity": self.score_reasoning_fidelity(result_output, trace),
            "retrieval_precision": self.score_retrieval_precision(result_output, expected),
            "hallucination_score": self.score_hallucination(result_output, expected),
            "completion": self.score_completion(status),
            "latency_score": self.score_latency(latency_ms),
            "confidence_calibration": self.score_confidence_calibration(trace),
        }

    @staticmethod
    def score_accuracy(
        result: dict[str, Any],
        expected: dict[str, Any],
    ) -> float:
        """Score output accuracy against expected ground truth.

        Uses key overlap and value matching for structured outputs.

        Args:
            result: The actual output.
            expected: The expected output.

        Returns:
            Accuracy score between 0.0 and 1.0.
        """
        if not expected:
            return 1.0 if result else 0.0

        if not result:
            return 0.0

        # Check key overlap
        expected_keys = set(expected.keys())
        result_keys = set(result.keys())
        key_overlap = len(expected_keys & result_keys) / len(expected_keys) if expected_keys else 0

        # Check value matches for overlapping keys
        matching_values = 0
        for key in expected_keys & result_keys:
            if _values_match(result[key], expected[key]):
                matching_values += 1

        value_accuracy = matching_values / len(expected_keys) if expected_keys else 0

        return key_overlap * 0.4 + value_accuracy * 0.6

    @staticmethod
    def score_reasoning_fidelity(
        result: dict[str, Any],
        trace: list[dict[str, Any]],
    ) -> float:
        """Score whether the reasoning trace is well-structured and coherent.

        Measures: presence of reasoning chain, step coverage,
        confidence distribution, and conclusion grounding.

        Args:
            result: The output with reasoning data.
            trace: The orchestration trace.

        Returns:
            Reasoning fidelity score between 0.0 and 1.0.
        """
        reasoning_chain = result.get("reasoning_chain", [])
        if not reasoning_chain:
            # Check for nested reasoning output
            for key in ("best_branch", "reasoning"):
                nested = result.get(key, {})
                if isinstance(nested, dict):
                    reasoning_chain = nested.get("steps", [])
                    if reasoning_chain:
                        break

        if not reasoning_chain:
            return 0.3  # No explicit reasoning chain

        # Score based on chain quality
        num_steps = len(reasoning_chain)
        step_score = min(1.0, num_steps / 3)  # At least 3 steps ideal

        # Confidence distribution — penalize uniform or zero confidence
        confidences = [s.get("confidence", 0.5) for s in reasoning_chain]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        conf_spread = max(confidences) - min(confidences) if confidences else 0.0

        conf_score = avg_conf * 0.7 + min(conf_spread * 2, 0.3)

        # Check if conclusion is present
        has_conclusion = bool(result.get("conclusion"))
        conclusion_score = 1.0 if has_conclusion else 0.3

        return step_score * 0.3 + conf_score * 0.4 + conclusion_score * 0.3

    @staticmethod
    def score_retrieval_precision(
        result: dict[str, Any],
        expected: dict[str, Any],
    ) -> float:
        """Score retrieval quality based on document relevance.

        Args:
            result: Output that may contain retrieved documents.
            expected: Expected output with ground truth documents.

        Returns:
            Retrieval precision score between 0.0 and 1.0.
        """
        documents = result.get("documents", [])
        expected_docs = expected.get("expected_documents", [])

        if not expected_docs:
            # No ground truth for retrieval — score based on presence
            return 0.7 if documents else 0.5

        if not documents:
            return 0.0

        # Simple text overlap metric for document matching
        matched = 0
        for expected_doc in expected_docs:
            expected_lower = expected_doc.lower() if isinstance(expected_doc, str) else ""
            for doc in documents:
                doc_text = doc.get("document", str(doc)) if isinstance(doc, dict) else str(doc)
                if expected_lower and expected_lower in doc_text.lower():
                    matched += 1
                    break

        return matched / len(expected_docs)

    @staticmethod
    def score_hallucination(
        result: dict[str, Any],
        expected: dict[str, Any],
    ) -> float:
        """Score hallucination level (lower is better).

        Checks if the result contains claims not present in expected output.

        Args:
            result: The actual output.
            expected: The expected output.

        Returns:
            Hallucination score between 0.0 (none) and 1.0 (fully hallucinated).
        """
        checks = result.get("checks", {})
        if isinstance(checks, dict) and "hallucination" in checks:
            return float(checks["hallucination"].get("hallucination_score", 0.0))

        # Fallback: check for unsupported content
        conclusion = result.get("conclusion", "")
        if not conclusion:
            return 0.0

        expected_text = str(expected)
        conclusion_words = set(conclusion.lower().split())
        expected_words = set(expected_text.lower().split())

        if not conclusion_words:
            return 0.0

        unsupported = conclusion_words - expected_words
        stop_words = {"the", "a", "an", "is", "are", "was", "in", "on", "to", "for", "and", "or"}
        unsupported -= stop_words

        if not conclusion_words - stop_words:
            return 0.0

        content_words = conclusion_words - stop_words
        return len(unsupported) / len(content_words) if content_words else 0.0

    @staticmethod
    def score_completion(status: Any) -> float:
        """Score task completion status.

        Args:
            status: The orchestration status (OrchestrationState or similar).

        Returns:
            1.0 for completed, 0.0 for failed.
        """
        if status is None:
            return 0.0

        status_value = status.value if hasattr(status, "value") else str(status)
        return 1.0 if status_value == "completed" else 0.0

    def score_latency(self, latency_ms: float) -> float:
        """Score latency against configured targets.

        Args:
            latency_ms: Execution time in milliseconds.

        Returns:
            Latency score between 0.0 (too slow) and 1.0 (within target).
        """
        p95_target = self.thresholds.get("latency_p95_target_ms", 5000)
        if latency_ms <= p95_target:
            return 1.0
        if latency_ms <= p95_target * 2:
            return 0.5
        return 0.0

    @staticmethod
    def score_confidence_calibration(trace: list[dict[str, Any]]) -> float:
        """Score how well agent confidence correlates with actual outcomes.

        Args:
            trace: Execution trace with agent confidence scores and statuses.

        Returns:
            Calibration score between 0.0 and 1.0.
        """
        if not trace:
            return 0.5

        calibration_errors: list[float] = []
        for record in trace:
            confidence = record.get("confidence_score", 0.5)
            status = record.get("status", "")
            actual = 1.0 if status == "completed" else 0.0
            calibration_errors.append(abs(confidence - actual))

        avg_error = sum(calibration_errors) / len(calibration_errors)
        return 1.0 - avg_error


def _values_match(actual: Any, expected: Any) -> bool:
    """Check if two values match, with type-flexible comparison.

    Args:
        actual: The actual value.
        expected: The expected value.

    Returns:
        True if values match.
    """
    if actual == expected:
        return True

    # String comparison (case insensitive)
    if isinstance(actual, str) and isinstance(expected, str):
        return actual.lower().strip() == expected.lower().strip()

    # Numeric comparison with tolerance
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return abs(actual - expected) < 0.01

    return False
