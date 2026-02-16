"""Application settings loaded from environment variables and .env files.

Uses pydantic-settings for type-safe configuration. All environment variables
are prefixed with AOF_ to avoid collisions.
"""

from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):
    """Application settings loaded from environment variables and .env files."""

    # Environment
    environment: str = "development"
    debug: bool = False

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # LLM (litellm)
    default_llm_model: str = "anthropic/claude-sonnet-4-5-20250929"
    llm_fallback_model: str = "openai/gpt-4o"
    llm_max_retries: int = 3
    llm_timeout_sec: int = 30

    # Vector DB
    vector_db_provider: str = "chromadb"
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None

    # Orchestration
    max_orchestration_steps: int = 10
    default_agent_timeout_sec: int = 30
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_timeout_sec: int = 60

    # Observability
    otlp_endpoint: str | None = None
    log_level: str = "INFO"

    model_config = {"env_prefix": "AOF_", "env_file": ".env"}
