"""Validation agent — schema, semantic, PII detection, and data quality scoring.

Implements multi-layered validation with a configurable rule engine.
Never raises from execute() — returns FAILED status instead (ADR-006).
"""

import re
from dataclasses import dataclass, field
from typing import Any

import structlog

from agents.base import AgentRequest, AgentResponse, AgentStatus, BaseAgent

log = structlog.get_logger()


@dataclass
class ValidationCheck:
    """Result of a single validation check."""

    name: str
    passed: bool
    score: float = 1.0
    error: str = ""
    requires_review: bool = False
    details: dict[str, Any] = field(default_factory=dict)


class RuleEngine:
    """Configurable rule engine for multi-layered validation.

    Supports schema validation, semantic checks, PII detection,
    and data quality scoring via pluggable rule sets.
    """

    # Common PII patterns — conservative matches to minimize false positives
    PII_PATTERNS: dict[str, re.Pattern[str]] = {
        "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
        "phone": re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
        "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "credit_card": re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),
    }

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.required_fields: list[str] = config.get("required_fields", [])
        self.field_types: dict[str, str] = config.get("field_types", {})
        self.semantic_rules: list[dict[str, Any]] = config.get("semantic_rules", [])
        self.pii_detection_enabled: bool = config.get("pii_detection_enabled", True)

    def schema_validation(self, payload: dict[str, Any]) -> ValidationCheck:
        """Validate payload against schema requirements (required fields, types)."""
        missing = [f for f in self.required_fields if f not in payload]
        if missing:
            return ValidationCheck(
                name="schema_validation",
                passed=False,
                score=0.0,
                error=f"Missing required fields: {', '.join(missing)}",
            )

        type_errors: list[str] = []
        type_map: dict[str, type | tuple[type, ...]] = {
            "str": str,
            "int": int,
            "float": (int, float),
            "bool": bool,
            "list": list,
        }

        for field_name, expected_type in self.field_types.items():
            if field_name in payload:
                expected = type_map.get(expected_type)
                if expected is not None and not isinstance(payload[field_name], expected):
                    type_errors.append(
                        f"{field_name}: expected {expected_type}, "
                        f"got {type(payload[field_name]).__name__}"
                    )

        if type_errors:
            return ValidationCheck(
                name="schema_validation",
                passed=False,
                score=0.0,
                error=f"Type errors: {'; '.join(type_errors)}",
            )

        return ValidationCheck(name="schema_validation", passed=True, score=1.0)

    def semantic_validation(self, payload: dict[str, Any]) -> ValidationCheck:
        """Validate business logic rules (e.g., value ranges, conditional requirements)."""
        violations: list[str] = []

        for rule in self.semantic_rules:
            rule_field = rule.get("field", "")
            rule_type = rule.get("rule", "")
            value = payload.get(rule_field)

            if value is None:
                continue

            if rule_type == "min_value" and isinstance(value, (int, float)):
                if value < rule["value"]:
                    violations.append(f"{rule_field} must be >= {rule['value']}, got {value}")
            elif rule_type == "max_value" and isinstance(value, (int, float)):
                if value > rule["value"]:
                    violations.append(f"{rule_field} must be <= {rule['value']}, got {value}")
            elif rule_type == "min_length" and isinstance(value, str):
                if len(value) < rule["value"]:
                    violations.append(
                        f"{rule_field} length must be >= {rule['value']}, got {len(value)}"
                    )
            elif (
                rule_type == "one_of"
                and isinstance(rule.get("values"), list)
                and value not in rule["values"]
            ):
                violations.append(f"{rule_field} must be one of {rule['values']}, got {value}")

        if violations:
            return ValidationCheck(
                name="semantic_validation",
                passed=False,
                score=0.0,
                error=f"Semantic violations: {'; '.join(violations)}",
            )

        score = 1.0 if self.semantic_rules else 0.8
        return ValidationCheck(name="semantic_validation", passed=True, score=score)

    def pii_detection(self, payload: dict[str, Any]) -> ValidationCheck:
        """Scan payload values for common PII patterns."""
        if not self.pii_detection_enabled:
            return ValidationCheck(name="pii_detection", passed=True, score=1.0)

        detected: dict[str, list[str]] = {}
        self._scan_for_pii(payload, detected, prefix="")

        if detected:
            return ValidationCheck(
                name="pii_detection",
                passed=False,
                score=0.0,
                error=f"PII detected in fields: {', '.join(detected.keys())}",
                requires_review=True,
                details={"pii_types": detected},
            )

        return ValidationCheck(name="pii_detection", passed=True, score=1.0)

    def _scan_for_pii(
        self,
        data: Any,
        detected: dict[str, list[str]],
        prefix: str,
    ) -> None:
        """Recursively scan data structure for PII patterns."""
        if isinstance(data, dict):
            for key, value in data.items():
                path = f"{prefix}.{key}" if prefix else key
                self._scan_for_pii(value, detected, path)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                self._scan_for_pii(item, detected, f"{prefix}[{i}]")
        elif isinstance(data, str):
            for pii_type, pattern in self.PII_PATTERNS.items():
                if pattern.search(data):
                    detected.setdefault(prefix, []).append(pii_type)

    def data_quality_score(self, payload: dict[str, Any]) -> ValidationCheck:
        """Score the overall data quality of the payload."""
        if not payload:
            return ValidationCheck(
                name="data_quality",
                passed=False,
                score=0.0,
                error="Empty payload",
            )

        total_fields = len(payload)
        non_empty = sum(
            1 for v in payload.values() if v is not None and v != "" and v != [] and v != {}
        )
        completeness = non_empty / total_fields if total_fields > 0 else 0.0

        return ValidationCheck(
            name="data_quality",
            passed=completeness >= 0.5,
            score=completeness,
            error="" if completeness >= 0.5 else f"Low data quality: {completeness:.0%} complete",
            details={"completeness": completeness, "total_fields": total_fields},
        )


class ValidationAgent(BaseAgent):
    """Multi-layered validation agent with configurable rule engine.

    Performs schema validation, semantic checks, PII detection, and
    data quality scoring. Never raises from execute() (ADR-006).
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__("validator", config)
        self.rule_engine = RuleEngine(config)

    async def execute(self, request: AgentRequest) -> AgentResponse:
        """Execute validation checks on the request payload.

        Args:
            request: The agent request containing the payload to validate.

        Returns:
            AgentResponse with validation results and confidence score.
        """
        try:
            checks = [
                self.rule_engine.schema_validation(request.payload),
                self.rule_engine.semantic_validation(request.payload),
                self.rule_engine.pii_detection(request.payload),
                self.rule_engine.data_quality_score(request.payload),
            ]

            failed_checks = [c for c in checks if not c.passed]

            if failed_checks:
                error_msg = "; ".join(c.error for c in failed_checks if c.error)
                return AgentResponse(
                    agent_id=self.agent_id,
                    status=AgentStatus.FAILED,
                    error=f"Validation failed: {error_msg}",
                    confidence_score=0.0,
                    requires_human_review=any(c.requires_review for c in failed_checks),
                    metadata={
                        "checks": {c.name: {"passed": c.passed, "score": c.score} for c in checks}
                    },
                )

            confidence = self._calculate_confidence(checks)
            return AgentResponse(
                agent_id=self.agent_id,
                status=AgentStatus.COMPLETED,
                output={"validated_payload": request.payload},
                confidence_score=confidence,
                metadata={
                    "checks": {c.name: {"passed": c.passed, "score": c.score} for c in checks}
                },
            )

        except Exception as exc:
            log.error(
                "validation_agent.unexpected_error",
                agent_id=self.agent_id,
                task_id=request.task_id,
                error_type=type(exc).__name__,
            )
            return AgentResponse(
                agent_id=self.agent_id,
                status=AgentStatus.FAILED,
                error=f"Unexpected validation error: {type(exc).__name__}",
                confidence_score=0.0,
            )

    def can_handle(self, request: AgentRequest) -> bool:
        """ValidationAgent can handle any request with a non-empty payload."""
        return bool(request.payload)

    async def health_probe(self, timeout_sec: int = 2) -> dict[str, Any]:
        """Return health status for the validation agent."""
        return {
            "healthy": True,
            "agent_id": self.agent_id,
            "checks_available": [
                "schema_validation",
                "semantic_validation",
                "pii_detection",
                "data_quality",
            ],
        }

    @staticmethod
    def _calculate_confidence(checks: list[ValidationCheck]) -> float:
        """Calculate aggregate confidence score from individual check scores."""
        if not checks:
            return 0.0
        return sum(c.score for c in checks) / len(checks)
