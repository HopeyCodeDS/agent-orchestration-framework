"""Retrieval agent — hybrid RAG with query decomposition and relevance scoring.

Uses litellm for embedding generation and ChromaDB for vector search.
Never raises from execute() — returns FAILED status instead (ADR-006).
"""

from typing import Any

import structlog

from agents.base import AgentRequest, AgentResponse, AgentStatus, BaseAgent

log = structlog.get_logger()


class QueryDecomposer:
    """Decomposes complex queries into sub-queries for improved retrieval.

    Uses structural decomposition (keyword extraction, entity splitting)
    without requiring LLM calls for basic decomposition.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.max_sub_queries: int = config.get("max_sub_queries", 5)

    def decompose(self, query: str) -> list[str]:
        """Break a complex query into simpler sub-queries.

        Args:
            query: The original user query.

        Returns:
            List of sub-queries (always includes the original).
        """
        sub_queries = [query]

        # Split compound questions on common conjunctions
        conjunctions = [" and ", " also ", " additionally "]
        for conj in conjunctions:
            if conj in query.lower():
                parts = query.split(conj, maxsplit=1)
                sub_queries.extend(p.strip() for p in parts if p.strip())

        return sub_queries[: self.max_sub_queries]


class RelevanceScorer:
    """Scores retrieval results for relevance to the original query.

    Uses a combination of vector similarity and metadata signals
    to produce a final relevance score.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.similarity_weight: float = config.get("similarity_weight", 0.7)
        self.recency_weight: float = config.get("recency_weight", 0.15)
        self.authority_weight: float = config.get("authority_weight", 0.15)

    def score(self, result: dict[str, Any]) -> float:
        """Calculate a composite relevance score for a retrieval result.

        Args:
            result: A retrieval result with distance and optional metadata.

        Returns:
            Composite relevance score between 0.0 and 1.0.
        """
        # Convert distance to similarity (lower distance = higher similarity)
        distance = result.get("distance", 1.0)
        similarity = max(0.0, 1.0 - distance)

        recency = result.get("metadata", {}).get("recency_score", 0.5)
        authority = result.get("metadata", {}).get("authority_score", 0.5)

        weighted = (
            similarity * self.similarity_weight
            + recency * self.recency_weight
            + authority * self.authority_weight
        )
        return min(1.0, max(0.0, weighted))

    def filter_by_threshold(
        self,
        results: list[dict[str, Any]],
        threshold: float = 0.3,
    ) -> list[dict[str, Any]]:
        """Filter results below a relevance threshold.

        Args:
            results: Scored retrieval results.
            threshold: Minimum relevance score to keep.

        Returns:
            Filtered list of results meeting the threshold.
        """
        return [r for r in results if r.get("relevance_score", 0.0) >= threshold]


class VectorStoreClient:
    """Abstraction over vector database operations.

    Supports ChromaDB for development and can be extended for Qdrant in production.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.provider: str = config.get("vector_db_provider", "chromadb")
        self.collection_name: str = config.get("collection_name", "default")
        self._client: Any = None
        self._collection: Any = None

    async def initialize(self) -> None:
        """Initialize the vector store client.

        Lazily creates the ChromaDB client and collection.
        """
        if self._client is not None:
            return

        if self.provider == "chromadb":
            try:
                import chromadb

                self._client = chromadb.Client()
                self._collection = self._client.get_or_create_collection(
                    name=self.collection_name
                )
            except ImportError:
                log.warning("retrieval_agent.chromadb_not_available")
                self._client = None

    async def query(
        self,
        query_text: str,
        n_results: int = 5,
    ) -> list[dict[str, Any]]:
        """Query the vector store for similar documents.

        Args:
            query_text: The text to search for.
            n_results: Maximum number of results to return.

        Returns:
            List of result dicts with document, distance, and metadata.
        """
        if self._collection is None:
            return []

        try:
            results = self._collection.query(
                query_texts=[query_text],
                n_results=n_results,
            )
        except Exception as exc:
            log.error(
                "retrieval_agent.vector_query_failed",
                error_type=type(exc).__name__,
            )
            return []

        documents = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        return [
            {
                "document": doc,
                "distance": dist,
                "metadata": meta or {},
            }
            for doc, dist, meta in zip(documents, distances, metadatas)
        ]

    async def add_documents(
        self,
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> None:
        """Add documents to the vector store.

        Args:
            documents: Text documents to add.
            metadatas: Optional metadata for each document.
            ids: Optional document IDs.
        """
        if self._collection is None:
            return

        doc_ids = ids or [f"doc-{i}" for i in range(len(documents))]
        self._collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=doc_ids,
        )

    async def health_check(self) -> dict[str, Any]:
        """Check vector store connectivity.

        Returns:
            Dict with health status.
        """
        return {
            "healthy": self._client is not None,
            "provider": self.provider,
            "collection": self.collection_name,
        }


class RetrievalAgent(BaseAgent):
    """Hybrid retrieval agent with query decomposition and relevance scoring.

    Performs vector search via ChromaDB, decomposes complex queries,
    and scores results for relevance. Never raises from execute() (ADR-006).
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__("retriever", config)
        self.decomposer = QueryDecomposer(config)
        self.scorer = RelevanceScorer(config)
        self.vector_store = VectorStoreClient(config)
        self.n_results: int = config.get("n_results", 5)
        self.relevance_threshold: float = config.get("relevance_threshold", 0.3)

    async def execute(self, request: AgentRequest) -> AgentResponse:
        """Execute retrieval against the vector store.

        Args:
            request: The agent request containing the query payload.

        Returns:
            AgentResponse with retrieved documents and relevance scores.
        """
        try:
            await self.vector_store.initialize()

            query = request.payload.get("query", "")
            if not query:
                return AgentResponse(
                    agent_id=self.agent_id,
                    status=AgentStatus.FAILED,
                    error="No query provided in payload",
                    confidence_score=0.0,
                )

            # Decompose complex queries
            sub_queries = self.decomposer.decompose(query)

            # Retrieve for each sub-query
            all_results: list[dict[str, Any]] = []
            seen_docs: set[str] = set()

            for sub_query in sub_queries:
                results = await self.vector_store.query(
                    query_text=sub_query,
                    n_results=self.n_results,
                )
                for result in results:
                    doc_text = result.get("document", "")
                    if doc_text not in seen_docs:
                        result["relevance_score"] = self.scorer.score(result)
                        all_results.append(result)
                        seen_docs.add(doc_text)

            # Filter by relevance threshold
            relevant_results = self.scorer.filter_by_threshold(
                all_results, self.relevance_threshold
            )

            # Sort by relevance score descending
            relevant_results.sort(key=lambda r: r.get("relevance_score", 0.0), reverse=True)

            # Calculate confidence based on retrieval quality
            confidence = self._calculate_confidence(relevant_results)

            if not relevant_results:
                return AgentResponse(
                    agent_id=self.agent_id,
                    status=AgentStatus.COMPLETED,
                    output={"documents": [], "query_decomposition": sub_queries},
                    confidence_score=max(0.1, confidence),
                    metadata={
                        "total_retrieved": len(all_results),
                        "after_filtering": 0,
                        "sub_queries": len(sub_queries),
                    },
                )

            return AgentResponse(
                agent_id=self.agent_id,
                status=AgentStatus.COMPLETED,
                output={
                    "documents": relevant_results,
                    "query_decomposition": sub_queries,
                },
                confidence_score=confidence,
                metadata={
                    "total_retrieved": len(all_results),
                    "after_filtering": len(relevant_results),
                    "sub_queries": len(sub_queries),
                    "avg_relevance": (
                        sum(r.get("relevance_score", 0.0) for r in relevant_results)
                        / len(relevant_results)
                    ),
                },
            )

        except Exception as exc:
            log.error(
                "retrieval_agent.unexpected_error",
                agent_id=self.agent_id,
                task_id=request.task_id,
                error_type=type(exc).__name__,
            )
            return AgentResponse(
                agent_id=self.agent_id,
                status=AgentStatus.FAILED,
                error=f"Unexpected retrieval error: {type(exc).__name__}",
                confidence_score=0.0,
            )

    def can_handle(self, request: AgentRequest) -> bool:
        """RetrievalAgent handles requests with a query field."""
        return bool(request.payload.get("query"))

    async def health_probe(self, timeout_sec: int = 2) -> dict[str, Any]:
        """Return health status including vector store connectivity."""
        vector_health = await self.vector_store.health_check()
        return {
            "healthy": True,
            "agent_id": self.agent_id,
            "vector_store": vector_health,
            "capabilities": [
                "vector_search",
                "query_decomposition",
                "relevance_scoring",
            ],
        }

    @staticmethod
    def _calculate_confidence(results: list[dict[str, Any]]) -> float:
        """Calculate confidence from retrieval quality signals.

        Args:
            results: Scored retrieval results.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        if not results:
            return 0.1

        scores = [r.get("relevance_score", 0.0) for r in results]
        avg_score = sum(scores) / len(scores)
        top_score = max(scores)

        # Blend average and top score
        return min(1.0, avg_score * 0.6 + top_score * 0.4)
