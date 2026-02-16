"""Unit tests for the config module."""

from config.settings import AppSettings


class TestAppSettings:
    def test_default_values(self) -> None:
        settings = AppSettings()
        assert settings.environment == "development"
        assert settings.debug is False
        assert settings.api_port == 8000
        assert settings.max_orchestration_steps == 10
        assert settings.circuit_breaker_failure_threshold == 5
        assert settings.log_level == "INFO"

    def test_llm_defaults(self) -> None:
        settings = AppSettings()
        assert "anthropic" in settings.default_llm_model
        assert "openai" in settings.llm_fallback_model
        assert settings.llm_max_retries == 3

    def test_vector_db_defaults(self) -> None:
        settings = AppSettings()
        assert settings.vector_db_provider == "chromadb"
        assert settings.qdrant_url is None

    def test_optional_fields(self) -> None:
        settings = AppSettings()
        assert settings.otlp_endpoint is None
        assert settings.qdrant_api_key is None
