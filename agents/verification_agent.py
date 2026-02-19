"""Verification agent — fact-checking, hallucination detection, self-consistency.

Validates reasoning output against retrieved sources and checks for
internal consistency. Never raises from execute() — returns FAILED status instead (ADR-006).
"""

from typing import Any

import structlog

from agents.base import AgentRequest, AgentResponse, AgentStatus, BaseAgent

log = structlog.get_logger()


class FactChecker:
    """Checks claims against source documents for factual accuracy.

    When an embedding function is provided (via ``embed_fn``), uses cosine
    similarity over embeddings for semantic matching. Falls back to word-overlap
    when embeddings are unavailable.
    """

    def __init__(self, config: dict[str, Any], embed_fn: Any = None) -> None:
        self.support_threshold: float = config.get("support_threshold", 0.3)
        self.embed_fn = embed_fn  # callable(list[str]) -> list[list[float]] | None

    def check_claims(
        self,
        conclusion: str,
        sources: list[str],
    ) -> dict[str, Any]:
        """Verify a conclusion against source documents.

        Uses embedding-based cosine similarity when ``embed_fn`` is set,
        otherwise falls back to word-overlap scoring.

        Args:
            conclusion: The claim or conclusion to verify.
            sources: List of source document texts.

        Returns:
            Dict with support_score, supported flag, and per-source details.
        """
        if not sources:
            return {
                "support_score": 0.0,
                "supported": False,
                "reason": "No source documents available for verification",
                "source_scores": [],
            }

        source_scores: list[dict[str, Any]] = []

        if self.embed_fn is not None:
            try:
                embeddings = self.embed_fn([conclusion] + sources)
                conclusion_vec = embeddings[0]
                for i, source_vec in enumerate(embeddings[1:]):
                    sim = self._cosine(conclusion_vec, source_vec)
                    source_scores.append({"source_index": i, "similarity_score": round(sim, 3)})

                best_score = max(s["similarity_score"] for s in source_scores)
                avg_score = sum(s["similarity_score"] for s in source_scores) / len(source_scores)
                support_score = best_score * 0.7 + avg_score * 0.3

                return {
                    "support_score": round(support_score, 3),
                    "supported": support_score >= self.support_threshold,
                    "source_scores": source_scores,
                    "best_source_index": max(
                        source_scores, key=lambda s: s["similarity_score"]
                    )["source_index"],
                    "method": "semantic",
                }
            except Exception as exc:
                log.warning(
                    "fact_checker.embedding_failed",
                    error_type=type(exc).__name__,
                    detail=str(exc)[:100],
                )
                # Fall through to word-overlap

        # Word-overlap fallback
        conclusion_words = set(conclusion.lower().split())
        for i, source in enumerate(sources):
            source_words = set(source.lower().split())
            if not conclusion_words:
                overlap = 0.0
            else:
                overlap = len(conclusion_words & source_words) / len(conclusion_words)
            source_scores.append({"source_index": i, "overlap_score": round(overlap, 3)})

        best_score = max(s["overlap_score"] for s in source_scores)
        avg_score = sum(s["overlap_score"] for s in source_scores) / len(source_scores)
        support_score = best_score * 0.7 + avg_score * 0.3

        return {
            "support_score": round(support_score, 3),
            "supported": support_score >= self.support_threshold,
            "source_scores": source_scores,
            "best_source_index": max(source_scores, key=lambda s: s["overlap_score"])[
                "source_index"
            ],
            "method": "word_overlap",
        }

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two embedding vectors.

        Pure-Python implementation — no numpy required.

        Args:
            a: First embedding vector.
            b: Second embedding vector.

        Returns:
            Cosine similarity in [0.0, 1.0] (clamped to avoid float errors).
        """
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        return max(0.0, min(1.0, dot / (na * nb))) if na and nb else 0.0


class HallucinationDetector:
    """Detects potential hallucinations in reasoning output.

    Identifies claims that are not grounded in source material
    or that contain known hallucination patterns.
    """

    # Common patterns that often indicate hallucination
    HALLUCINATION_INDICATORS: list[str] = [
        "as everyone knows",
        "it is well established that",
        "studies have shown",  # Without citation
        "research proves",  # Without citation
        "according to experts",  # Without specific attribution
    ]

    def __init__(self, config: dict[str, Any], embed_fn: Any = None) -> None:
        self.hallucination_threshold: float = config.get("hallucination_threshold", 0.5)
        self.embed_fn = embed_fn

    def detect(
        self,
        reasoning_output: dict[str, Any],
        sources: list[str],
    ) -> dict[str, Any]:
        """Detect potential hallucinations in reasoning output.

        Args:
            reasoning_output: The reasoning agent's output containing conclusion and chain.
            sources: Source documents to ground-truth against.

        Returns:
            Dict with hallucination_score, flagged claims, and analysis.
        """
        conclusion = reasoning_output.get("conclusion", "")
        reasoning_chain = reasoning_output.get("reasoning_chain", [])

        flags: list[dict[str, Any]] = []

        # Check for vague authority claims
        indicator_count = sum(
            1 for indicator in self.HALLUCINATION_INDICATORS if indicator in conclusion.lower()
        )
        if indicator_count > 0:
            flags.append(
                {
                    "type": "vague_authority",
                    "count": indicator_count,
                    "severity": "medium",
                }
            )

        # Check reasoning chain steps for grounding (only meaningful with sources)
        ungrounded_steps = 0
        total_steps = len(reasoning_chain)
        if sources:
            for step in reasoning_chain:
                thought = step.get("thought", "")
                if not self._is_grounded(thought, sources):
                    ungrounded_steps += 1

            if total_steps > 0:
                ungrounded_ratio = ungrounded_steps / total_steps
                if ungrounded_ratio > 0.5:
                    flags.append(
                        {
                            "type": "ungrounded_reasoning",
                            "ungrounded_steps": ungrounded_steps,
                            "total_steps": total_steps,
                            "severity": "high",
                        }
                    )

        # Calculate hallucination score
        score = self._calculate_hallucination_score(indicator_count, ungrounded_steps, total_steps)

        return {
            "hallucination_score": round(score, 3),
            "likely_hallucinated": score >= self.hallucination_threshold,
            "flags": flags,
            "num_flags": len(flags),
        }

    def _is_grounded(self, thought: str, sources: list[str]) -> bool:
        """Check if a thought has any grounding in source documents.

        Uses embedding cosine similarity when ``embed_fn`` is set; otherwise
        falls back to word-overlap on content words.

        Args:
            thought: A reasoning step thought.
            sources: Source documents.

        Returns:
            True if the thought is sufficiently grounded in at least one source.
        """
        if not sources or not thought:
            return False

        if self.embed_fn is not None:
            try:
                embeddings = self.embed_fn([thought] + sources)
                thought_vec = embeddings[0]
                for source_vec in embeddings[1:]:
                    sim = FactChecker._cosine(thought_vec, source_vec)
                    if sim >= 0.3:
                        return True
                return False
            except Exception:
                pass  # Fall through to word-overlap

        # Word-overlap fallback
        thought_words = set(thought.lower().split())
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for"}
        thought_content = thought_words - stop_words

        if not thought_content:
            return True  # No content words to check

        for source in sources:
            source_words = set(source.lower().split()) - stop_words
            if thought_content and source_words:
                overlap = len(thought_content & source_words) / len(thought_content)
                if overlap >= 0.2:
                    return True

        return False

    @staticmethod
    def _calculate_hallucination_score(
        indicator_count: int,
        ungrounded_steps: int,
        total_steps: int,
    ) -> float:
        """Calculate composite hallucination score.

        Args:
            indicator_count: Number of hallucination indicators found.
            ungrounded_steps: Number of reasoning steps without grounding.
            total_steps: Total reasoning steps.

        Returns:
            Score between 0.0 (no hallucination) and 1.0 (likely hallucinated).
        """
        indicator_score = min(1.0, indicator_count * 0.3)
        grounding_score = ungrounded_steps / total_steps if total_steps > 0 else 0.5

        return indicator_score * 0.4 + grounding_score * 0.6


class ConsistencyChecker:
    """Checks internal consistency of reasoning chains.

    Verifies that reasoning steps do not contradict each other
    and that the conclusion follows logically from the chain.
    """

    CONTRADICTION_PAIRS: list[tuple[str, str]] = [
        ("true", "false"),
        ("increase", "decrease"),
        ("more", "less"),
        ("always", "never"),
        ("possible", "impossible"),
        ("likely", "unlikely"),
    ]

    def check(self, reasoning_output: dict[str, Any]) -> dict[str, Any]:
        """Check internal consistency of reasoning.

        Args:
            reasoning_output: The reasoning agent's output.

        Returns:
            Dict with consistency_score and any contradictions found.
        """
        chain = reasoning_output.get("reasoning_chain", [])
        contradictions: list[dict[str, Any]] = []

        # Check for contradictory language between steps
        for i, step_a in enumerate(chain):
            for j, step_b in enumerate(chain):
                if j <= i:
                    continue
                contradiction = self._find_contradiction(
                    step_a.get("thought", ""),
                    step_b.get("thought", ""),
                )
                if contradiction:
                    contradictions.append(
                        {
                            "step_a": i,
                            "step_b": j,
                            "type": contradiction,
                        }
                    )

        # Check conclusion consistency with chain
        conclusion = reasoning_output.get("conclusion", "")
        conclusion_consistent = self._conclusion_follows(conclusion, chain)

        consistency_score = self._calculate_consistency_score(
            len(chain), len(contradictions), conclusion_consistent
        )

        return {
            "consistency_score": round(consistency_score, 3),
            "consistent": consistency_score >= 0.6,
            "contradictions": contradictions,
            "conclusion_follows": conclusion_consistent,
        }

    def _find_contradiction(self, text_a: str, text_b: str) -> str | None:
        """Find contradictory language between two texts.

        Args:
            text_a: First text.
            text_b: Second text.

        Returns:
            Contradiction type string if found, None otherwise.
        """
        a_lower = text_a.lower()
        b_lower = text_b.lower()

        for word_a, word_b in self.CONTRADICTION_PAIRS:
            if (word_a in a_lower and word_b in b_lower) or (
                word_b in a_lower and word_a in b_lower
            ):
                # Only flag if both words appear in context of similar subjects
                a_words = set(a_lower.split())
                b_words = set(b_lower.split())
                # Check for shared context words (crude but effective)
                shared = a_words & b_words - {"the", "a", "an", "is", "are", "it", "and"}
                if len(shared) >= 2:
                    return f"potential_{word_a}_{word_b}_contradiction"

        return None

    @staticmethod
    def _conclusion_follows(conclusion: str, chain: list[dict[str, Any]]) -> bool:
        """Check if conclusion has lexical overlap with the reasoning chain.

        Args:
            conclusion: The final conclusion.
            chain: The reasoning chain steps.

        Returns:
            True if the conclusion appears grounded in the chain.
        """
        if not chain or not conclusion:
            return False

        conclusion_words = set(conclusion.lower().split())
        chain_words: set[str] = set()
        for step in chain:
            chain_words.update(step.get("thought", "").lower().split())

        if not conclusion_words:
            return True

        overlap = len(conclusion_words & chain_words) / len(conclusion_words)
        return overlap >= 0.3

    @staticmethod
    def _calculate_consistency_score(
        total_steps: int,
        num_contradictions: int,
        conclusion_follows: bool,
    ) -> float:
        """Calculate overall consistency score.

        Args:
            total_steps: Total reasoning steps.
            num_contradictions: Number of contradictions found.
            conclusion_follows: Whether conclusion follows from chain.

        Returns:
            Score between 0.0 (inconsistent) and 1.0 (fully consistent).
        """
        if total_steps == 0:
            return 0.5

        # Penalize for contradictions
        contradiction_penalty = min(1.0, num_contradictions * 0.3)
        chain_score = 1.0 - contradiction_penalty

        # Conclusion score
        conclusion_score = 1.0 if conclusion_follows else 0.3

        return chain_score * 0.6 + conclusion_score * 0.4


class VerificationAgent(BaseAgent):
    """Verification agent that fact-checks, detects hallucinations, and checks consistency.

    Validates reasoning output against retrieved sources. Never raises
    from execute() (ADR-006).
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__("verifier", config)

        # Try to use ChromaDB's default embedding model (sentence-transformers/all-MiniLM-L6-v2)
        # for semantic similarity. Falls back gracefully if chromadb is not installed.
        embed_fn: Any = None
        try:
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

            embed_fn = DefaultEmbeddingFunction()
            log.debug("verification_agent.embeddings_enabled", model="all-MiniLM-L6-v2")
        except Exception as exc:
            log.debug(
                "verification_agent.embeddings_unavailable",
                reason=str(exc)[:80],
            )

        self.fact_checker = FactChecker(config, embed_fn=embed_fn)
        self.hallucination_detector = HallucinationDetector(config, embed_fn=embed_fn)
        self.consistency_checker = ConsistencyChecker()
        self.verification_threshold: float = config.get("verification_threshold", 0.5)

    async def execute(self, request: AgentRequest) -> AgentResponse:
        """Execute verification checks on reasoning output.

        Args:
            request: The agent request containing reasoning output in context.

        Returns:
            AgentResponse with verification results.
        """
        try:
            reasoning_output = self._extract_reasoning_output(request)
            sources = self._extract_sources(request)

            # Guard: if the retriever found nothing, verification cannot be meaningful.
            # On the first retrieval attempt, fail and let the adaptive loop re-retrieve.
            # On a second miss, escalate — the knowledge base simply lacks coverage.
            retrieval_confidence = self._extract_retrieval_confidence(request)
            retrieval_attempt = request.context.get("retrieval_attempt", 1)
            if retrieval_confidence <= 0.1 and not sources:
                if retrieval_attempt >= 2:
                    return AgentResponse(
                        agent_id=self.agent_id,
                        status=AgentStatus.ESCALATED,
                        error=(
                            "Knowledge base has no coverage for this query after re-retrieval. "
                            "Human review required."
                        ),
                        confidence_score=0.0,
                        requires_human_review=True,
                        metadata={
                            "retrieval_confidence": retrieval_confidence,
                            "retrieval_attempt": retrieval_attempt,
                        },
                    )
                return AgentResponse(
                    agent_id=self.agent_id,
                    status=AgentStatus.FAILED,
                    error="No relevant documents retrieved; triggering adaptive re-retrieval",
                    confidence_score=0.0,
                    metadata={
                        "retrieval_confidence": retrieval_confidence,
                        "retrieval_attempt": retrieval_attempt,
                    },
                )

            if not reasoning_output:
                return AgentResponse(
                    agent_id=self.agent_id,
                    status=AgentStatus.FAILED,
                    error="No reasoning output found to verify",
                    confidence_score=0.0,
                )

            conclusion = reasoning_output.get("conclusion", "")

            # Run all verification checks
            fact_check = self.fact_checker.check_claims(conclusion, sources)
            hallucination_check = self.hallucination_detector.detect(reasoning_output, sources)
            consistency_check = self.consistency_checker.check(reasoning_output)

            # Aggregate verification score
            verification_score = self._calculate_verification_score(
                fact_check, hallucination_check, consistency_check
            )

            checks = {
                "fact_check": fact_check,
                "hallucination": hallucination_check,
                "consistency": consistency_check,
            }

            # Determine if verification passes
            passed = verification_score >= self.verification_threshold
            needs_review = (
                hallucination_check["likely_hallucinated"] or not consistency_check["consistent"]
            )

            if not passed:
                return AgentResponse(
                    agent_id=self.agent_id,
                    status=AgentStatus.FAILED,
                    error=f"Verification failed (score: {verification_score:.2f})",
                    confidence_score=verification_score,
                    requires_human_review=needs_review,
                    metadata={"checks": checks},
                )

            return AgentResponse(
                agent_id=self.agent_id,
                status=AgentStatus.COMPLETED,
                output={
                    "verified": True,
                    "verification_score": verification_score,
                    "conclusion": conclusion,
                    "checks": checks,
                },
                confidence_score=verification_score,
                requires_human_review=needs_review,
                metadata={"checks": checks},
            )

        except Exception as exc:
            log.error(
                "verification_agent.unexpected_error",
                agent_id=self.agent_id,
                task_id=request.task_id,
                error_type=type(exc).__name__,
            )
            return AgentResponse(
                agent_id=self.agent_id,
                status=AgentStatus.FAILED,
                error=f"Unexpected verification error: {type(exc).__name__}",
                confidence_score=0.0,
            )

    def can_handle(self, request: AgentRequest) -> bool:
        """VerificationAgent handles requests when reasoning output is available."""
        context = request.context
        last_output = context.get("last_output", {})
        if isinstance(last_output, dict):
            return "conclusion" in last_output or "reasoning_chain" in last_output
        return bool(request.payload)

    async def health_probe(self, timeout_sec: int = 2) -> dict[str, Any]:
        """Return health status for the verification agent."""
        return {
            "healthy": True,
            "agent_id": self.agent_id,
            "capabilities": [
                "fact_checking",
                "hallucination_detection",
                "consistency_validation",
            ],
        }

    @staticmethod
    def _extract_reasoning_output(request: AgentRequest) -> dict[str, Any]:
        """Extract reasoning output from request context or payload.

        Args:
            request: The agent request.

        Returns:
            Dict with reasoning output fields.
        """
        # Check context first (from previous agent)
        context = request.context
        last_output = context.get("last_output", {})
        if isinstance(last_output, dict) and "conclusion" in last_output:
            return last_output

        # Fall back to payload
        if "conclusion" in request.payload:
            return request.payload

        return {}

    @staticmethod
    def _extract_sources(request: AgentRequest) -> list[str]:
        """Extract source documents from request context.

        Args:
            request: The agent request.

        Returns:
            List of source document texts.
        """
        sources: list[str] = []

        # Look through agent history for retrieval output
        history = request.context.get("agent_history", [])
        for entry in history:
            if isinstance(entry, dict):
                output = entry.get("output", {})
                if isinstance(output, dict):
                    docs = output.get("documents", [])
                    for doc in docs:
                        if isinstance(doc, dict):
                            sources.append(doc.get("document", ""))
                        elif isinstance(doc, str):
                            sources.append(doc)

        # Also check last_output for documents
        last_output = request.context.get("last_output", {})
        if isinstance(last_output, dict):
            docs = last_output.get("documents", [])
            for doc in docs:
                if isinstance(doc, dict):
                    sources.append(doc.get("document", ""))
                elif isinstance(doc, str):
                    sources.append(doc)

        return [s for s in sources if s]

    @staticmethod
    def _extract_retrieval_confidence(request: AgentRequest) -> float:
        """Extract the retriever's confidence score from agent history.

        Args:
            request: The agent request containing context with agent_history.

        Returns:
            Float confidence score from the most recent retriever entry, or 1.0
            if no retriever entry is found (safe default — assume retrieval was fine).
        """
        history = request.context.get("agent_history", [])
        for entry in reversed(history):
            if isinstance(entry, dict) and entry.get("agent_id") == "retriever":
                return float(entry.get("confidence_score", 1.0))
        return 1.0

    @staticmethod
    def _calculate_verification_score(
        fact_check: dict[str, Any],
        hallucination_check: dict[str, Any],
        consistency_check: dict[str, Any],
    ) -> float:
        """Calculate aggregate verification score.

        When source documents are available, all three checks are weighted:
          fact_check (35%) + hallucination (35%) + consistency (30%)

        When NO sources are available, fact-checking and grounding checks
        are unreliable, so the score shifts to consistency-dominant:
          hallucination_indicators (20%) + consistency (80%)

        Args:
            fact_check: Fact-checking results.
            hallucination_check: Hallucination detection results.
            consistency_check: Consistency check results.

        Returns:
            Score between 0.0 and 1.0.
        """
        fact_score = fact_check.get("support_score", 0.0)
        hallucination_raw = hallucination_check.get("hallucination_score", 0.5)
        hallucination_score = 1.0 - hallucination_raw
        consistency_score = consistency_check.get("consistency_score", 0.5)

        has_sources = bool(fact_check.get("source_scores"))

        if has_sources:
            # Full verification with all three dimensions
            score: float = (
                fact_score * 0.35 + hallucination_score * 0.35 + consistency_score * 0.30
            )
        else:
            # No sources: rely on consistency and hallucination indicators only
            # (grounding checks are meaningless without documents)
            indicator_flags = [
                f
                for f in hallucination_check.get("flags", [])
                if f.get("type") == "vague_authority"
            ]
            indicator_penalty = min(1.0, len(indicator_flags) * 0.3)
            indicator_score = 1.0 - indicator_penalty
            score = indicator_score * 0.20 + consistency_score * 0.80

        return score
