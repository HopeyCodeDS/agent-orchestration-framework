"""Unit tests for the ValidationAgent and RuleEngine."""

from typing import Any

from agents.base import AgentRequest, AgentStatus
from agents.validation_agent import RuleEngine, ValidationAgent


class TestRuleEngineSchemaValidation:
    def test_valid_schema(self, validation_config: dict[str, Any]) -> None:
        engine = RuleEngine(validation_config)
        check = engine.schema_validation({"query": "test", "priority": 1})
        assert check.passed is True

    def test_missing_required_field(self, validation_config: dict[str, Any]) -> None:
        engine = RuleEngine(validation_config)
        check = engine.schema_validation({"priority": 1})
        assert check.passed is False
        assert "query" in check.error

    def test_wrong_field_type(self, validation_config: dict[str, Any]) -> None:
        engine = RuleEngine(validation_config)
        check = engine.schema_validation({"query": "test", "priority": "not_an_int"})
        assert check.passed is False
        assert "priority" in check.error


class TestRuleEngineSemanticValidation:
    def test_valid_semantics(self, validation_config: dict[str, Any]) -> None:
        engine = RuleEngine(validation_config)
        check = engine.semantic_validation({"priority": 3})
        assert check.passed is True

    def test_value_below_minimum(self, validation_config: dict[str, Any]) -> None:
        engine = RuleEngine(validation_config)
        check = engine.semantic_validation({"priority": 0})
        assert check.passed is False
        assert "priority" in check.error

    def test_value_above_maximum(self, validation_config: dict[str, Any]) -> None:
        engine = RuleEngine(validation_config)
        check = engine.semantic_validation({"priority": 10})
        assert check.passed is False
        assert "priority" in check.error

    def test_one_of_rule(self) -> None:
        config = {
            "semantic_rules": [
                {"field": "status", "rule": "one_of", "values": ["active", "inactive"]}
            ]
        }
        engine = RuleEngine(config)
        assert engine.semantic_validation({"status": "active"}).passed is True
        assert engine.semantic_validation({"status": "unknown"}).passed is False


class TestRuleEnginePIIDetection:
    def test_no_pii(self, validation_config: dict[str, Any]) -> None:
        engine = RuleEngine(validation_config)
        check = engine.pii_detection({"query": "What is Python?"})
        assert check.passed is True

    def test_email_detected(self, validation_config: dict[str, Any]) -> None:
        engine = RuleEngine(validation_config)
        check = engine.pii_detection({"query": "Contact user@example.com"})
        assert check.passed is False
        assert check.requires_review is True

    def test_ssn_detected(self, validation_config: dict[str, Any]) -> None:
        engine = RuleEngine(validation_config)
        check = engine.pii_detection({"data": "SSN: 123-45-6789"})
        assert check.passed is False

    def test_nested_pii_detected(self, validation_config: dict[str, Any]) -> None:
        engine = RuleEngine(validation_config)
        check = engine.pii_detection({"user": {"contact": {"email": "test@example.com"}}})
        assert check.passed is False

    def test_pii_detection_disabled(self) -> None:
        engine = RuleEngine({"pii_detection_enabled": False})
        check = engine.pii_detection({"email": "user@test.com"})
        assert check.passed is True


class TestRuleEngineDataQuality:
    def test_complete_payload(self, validation_config: dict[str, Any]) -> None:
        engine = RuleEngine(validation_config)
        check = engine.data_quality_score({"a": 1, "b": "text", "c": True})
        assert check.passed is True
        assert check.score == 1.0

    def test_empty_payload(self, validation_config: dict[str, Any]) -> None:
        engine = RuleEngine(validation_config)
        check = engine.data_quality_score({})
        assert check.passed is False

    def test_partial_payload(self, validation_config: dict[str, Any]) -> None:
        engine = RuleEngine(validation_config)
        check = engine.data_quality_score({"a": 1, "b": None, "c": "", "d": []})
        assert check.passed is False
        assert check.details["completeness"] == 0.25


class TestValidationAgent:
    async def test_valid_payload(self, validation_agent: ValidationAgent) -> None:
        request = AgentRequest(
            task_id="t1",
            payload={"query": "What is AI?", "priority": 3},
        )
        response = await validation_agent.execute(request)
        assert response.status == AgentStatus.COMPLETED
        assert response.confidence_score > 0.0
        assert response.output is not None
        assert "validated_payload" in response.output

    async def test_invalid_payload_missing_field(self, validation_agent: ValidationAgent) -> None:
        request = AgentRequest(
            task_id="t2",
            payload={"priority": 3},
        )
        response = await validation_agent.execute(request)
        assert response.status == AgentStatus.FAILED
        assert response.error is not None
        assert "query" in response.error

    async def test_pii_triggers_human_review(self, validation_agent: ValidationAgent) -> None:
        request = AgentRequest(
            task_id="t3",
            payload={"query": "Email me at test@example.com", "priority": 1},
        )
        response = await validation_agent.execute(request)
        assert response.status == AgentStatus.FAILED
        assert response.requires_human_review is True

    async def test_can_handle(self, validation_agent: ValidationAgent) -> None:
        request = AgentRequest(task_id="t4", payload={"data": "something"})
        assert validation_agent.can_handle(request) is True

        empty_request = AgentRequest(task_id="t5", payload={})
        assert validation_agent.can_handle(empty_request) is False

    async def test_health_probe(self, validation_agent: ValidationAgent) -> None:
        health = await validation_agent.health_probe()
        assert health["healthy"] is True
        assert "checks_available" in health

    async def test_metadata_includes_check_results(self, validation_agent: ValidationAgent) -> None:
        request = AgentRequest(
            task_id="t6",
            payload={"query": "Test query", "priority": 2},
        )
        response = await validation_agent.execute(request)
        assert "checks" in response.metadata
        assert "schema_validation" in response.metadata["checks"]
