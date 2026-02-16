"""Structured logging configuration using structlog.

JSON output in production, pretty console in development. Enriched with
trace context via contextvars (ADR-009).
"""

import structlog


def configure_logging(environment: str = "development") -> None:
    """Configure structlog for the application.

    Args:
        environment: One of "development" or "production". Controls output format.
    """
    processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if environment == "production":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
