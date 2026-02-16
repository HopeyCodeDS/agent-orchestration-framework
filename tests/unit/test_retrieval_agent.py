"""Unit tests for the RetrievalAgent."""

from typing import Any

from agents.base import AgentRequest, AgentStatus
from agents.retrieval_agent import (
    QueryDecomposer,
    RelevanceScorer,
    RetrievalAgent,
    VectorStoreClient,
)


class TestQueryDecomposer:
    def test_simple_query_returns_original(self) -> None:
        decomposer = QueryDecomposer({})
        result = decomposer.decompose("What is AI?")
        assert result == ["What is AI?"]

    def test_compound_query_splits(self) -> None:
        decomposer = QueryDecomposer({})
        result = decomposer.decompose("What is AI and how does it work?")
        assert len(result) > 1
        assert "What is AI and how does it work?" in result

    def test_max_sub_queries_respected(self) -> None:
        decomposer = QueryDecomposer({"max_sub_queries": 2})
        result = decomposer.decompose("A and B and C also D additionally E")
        assert len(result) <= 2


class TestRelevanceScorer:
    def test_perfect_similarity(self) -> None:
        scorer = RelevanceScorer({})
        result = {"distance": 0.0, "metadata": {"recency_score": 1.0, "authority_score": 1.0}}
        score = scorer.score(result)
        assert score == 1.0

    def test_no_similarity(self) -> None:
        scorer = RelevanceScorer({})
        result = {"distance": 1.0, "metadata": {}}
        score = scorer.score(result)
        assert 0.0 <= score <= 1.0

    def test_filter_by_threshold(self) -> None:
        scorer = RelevanceScorer({})
        results = [
            {"relevance_score": 0.8},
            {"relevance_score": 0.1},
            {"relevance_score": 0.5},
        ]
        filtered = scorer.filter_by_threshold(results, threshold=0.3)
        assert len(filtered) == 2

    def test_filter_empty_list(self) -> None:
        scorer = RelevanceScorer({})
        assert scorer.filter_by_threshold([], threshold=0.5) == []


class TestVectorStoreClient:
    async def test_health_check_without_init(self) -> None:
        client = VectorStoreClient({})
        health = await client.health_check()
        assert health["healthy"] is False
        assert health["provider"] == "chromadb"

    async def test_query_without_init_returns_empty(self) -> None:
        client = VectorStoreClient({})
        results = await client.query("test query")
        assert results == []


class TestRetrievalAgent:
    def _make_config(self) -> dict[str, Any]:
        return {
            "vector_db_provider": "chromadb",
            "n_results": 5,
            "relevance_threshold": 0.3,
        }

    async def test_missing_query_fails(self) -> None:
        agent = RetrievalAgent(config=self._make_config())
        request = AgentRequest(task_id="t1", payload={})
        response = await agent.execute(request)
        assert response.status == AgentStatus.FAILED
        assert "No query" in (response.error or "")

    async def test_can_handle_with_query(self) -> None:
        agent = RetrievalAgent(config=self._make_config())
        request = AgentRequest(task_id="t1", payload={"query": "test"})
        assert agent.can_handle(request) is True

    async def test_can_handle_without_query(self) -> None:
        agent = RetrievalAgent(config=self._make_config())
        request = AgentRequest(task_id="t1", payload={"data": "something"})
        assert agent.can_handle(request) is False

    async def test_health_probe(self) -> None:
        agent = RetrievalAgent(config=self._make_config())
        health = await agent.health_probe()
        assert health["healthy"] is True
        assert "capabilities" in health

    async def test_execute_empty_store(self) -> None:
        agent = RetrievalAgent(config=self._make_config())
        request = AgentRequest(task_id="t1", payload={"query": "test query"})
        response = await agent.execute(request)
        # Should complete but with no results
        assert response.status == AgentStatus.COMPLETED
        assert response.output is not None
        assert response.output["documents"] == []
