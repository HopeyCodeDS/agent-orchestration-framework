# Agent Orchestration Framework

A **production-grade multi-agent AI orchestration framework** where specialized agents (validation, retrieval, reasoning, verification) collaborate under centralized coordination to solve complex tasks.

Unlike simple prompt chains, this system implements explicit state machine orchestration, per-agent circuit breakers, full OpenTelemetry observability, and an automated evaluation harness.

## Architecture

```
Client -> API Gateway (FastAPI) -> Orchestrator (State Machine)
                                      |
                    +--------+--------+--------+
                    |        |        |        |
               Validation  Retrieval  Reasoning  Verification
                Agent       Agent      Agent      Agent
                    |        |        |        |
              [Rules DB]  [Vector DB] [Memory]  [Ground Truth]
```

Each agent has: typed request/response contracts, confidence scoring, circuit breaker protection, and health probes.


## Tech Stack

| Category | Technology |
|----------|-----------|
| Runtime | Python 3.10+ |
| API | FastAPI, uvicorn |
| Validation | Pydantic v2 |
| LLM Gateway | litellm (provider-agnostic) |
| Vector DB | ChromaDB (dev) / Qdrant (prod) |
| Observability | OpenTelemetry + structlog |
| Configuration | pydantic-settings + YAML |
| Testing | pytest, deepeval, respx |

## Quickstart

```bash
# 1. Create and activate virtual environment (Python 3.10+)
python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# macOS/Linux
# source .venv/bin/activate

# 2. Install in editable mode with dev tools
pip install -e ".[dev]"

# 3. Run formatters, linters, and tests
black . && ruff check . --fix && mypy . && pytest
```

## Project Structure

```
agents/           # Agent implementations (extend BaseAgent)
orchestrator/     # State machine, circuit breaker, retry policy
observability/    # OpenTelemetry tracing, structlog, metrics
evaluation/       # Automated evaluation harness + scoring
api/              # FastAPI routes, auth, health checks
config/           # YAML configs + pydantic-settings
tests/            # unit/, integration/, e2e/

```

## Key Design Decisions

- **State machine over prompt chaining** — deterministic testing, audit trails, controlled failure recovery
- **Async-first API** — LLM calls are never awaited in route handlers; clients poll for results
- **Circuit breaker per agent** — isolated failure domains; one unhealthy agent doesn't cascade
- **Hashed payloads in telemetry** — PII protection; no sensitive data in log aggregators
- **litellm as LLM gateway** — provider-agnostic with automatic fallback chains

