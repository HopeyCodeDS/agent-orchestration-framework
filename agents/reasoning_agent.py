"""Reasoning agent — chain-of-thought, tree-of-thoughts, and uncertainty quantification.

Uses litellm for LLM calls to generate structured reasoning traces.
Never raises from execute() — returns FAILED status instead (ADR-006).
"""

import re
from typing import Any

import structlog

from agents.base import AgentRequest, AgentResponse, AgentStatus, BaseAgent

# Matches "[sources: Doc 1, Doc 3]" or "[source: Doc 2]" at the end of a step.
_SOURCE_PAT = re.compile(r"\[sources?:\s*([^\]]+)\]", re.IGNORECASE)
# Matches "Doc N" labels produced by the prompt template.
_DOC_LABEL_PAT = re.compile(r"doc\s*(\d+)", re.IGNORECASE)

log = structlog.get_logger()


class ReasoningStep:
    """A single step in a reasoning chain.

    Attributes:
        step_number: Position in the chain.
        thought: The reasoning content for this step.
        confidence: Confidence in this particular step (0.0-1.0).
        evidence: Supporting evidence from prior agents.
    """

    def __init__(
        self,
        step_number: int,
        thought: str,
        confidence: float = 0.5,
        evidence: list[str] | None = None,
    ) -> None:
        self.step_number = step_number
        self.thought = thought
        self.confidence = max(0.0, min(1.0, confidence))
        self.evidence = evidence or []

    def to_dict(self) -> dict[str, Any]:
        """Serialize the reasoning step."""
        return {
            "step_number": self.step_number,
            "thought": self.thought,
            "confidence": self.confidence,
            "evidence": self.evidence,
        }


class ThoughtBranch:
    """A branch in the tree-of-thoughts exploration.

    Attributes:
        branch_id: Unique identifier for this branch.
        steps: Reasoning steps in this branch.
        score: Overall score for this branch path.
    """

    def __init__(self, branch_id: str) -> None:
        self.branch_id = branch_id
        self.steps: list[ReasoningStep] = []
        self.score: float = 0.0

    def add_step(self, step: ReasoningStep) -> None:
        """Add a reasoning step to this branch."""
        self.steps.append(step)
        self._recalculate_score()

    def _recalculate_score(self) -> None:
        """Recalculate branch score from step confidences."""
        if not self.steps:
            self.score = 0.0
            return
        # Geometric mean of step confidences penalizes low-confidence steps
        product = 1.0
        for step in self.steps:
            product *= step.confidence
        self.score = product ** (1.0 / len(self.steps))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the thought branch."""
        return {
            "branch_id": self.branch_id,
            "steps": [s.to_dict() for s in self.steps],
            "score": self.score,
        }


class ChainOfThoughtReasoner:
    """Generates linear chain-of-thought reasoning.

    Builds a structured sequence of reasoning steps from the query
    and any retrieved context.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.max_steps: int = config.get("max_reasoning_steps", 5)
        self.model: str = config.get("default_llm_model", "anthropic/claude-sonnet-4-5-20250929")
        self.timeout_sec: int = config.get("llm_timeout_sec", 30)

    async def reason(
        self,
        query: str,
        context: dict[str, Any],
    ) -> list[ReasoningStep]:
        """Generate a chain-of-thought reasoning trace.

        Args:
            query: The question or task to reason about.
            context: Accumulated context from prior agents.

        Returns:
            List of ReasoningStep objects forming the reasoning chain.
        """
        steps: list[ReasoningStep] = []
        documents = self._extract_documents(context)

        try:
            import litellm

            messages = self._build_cot_prompt(query, documents)
            response = await litellm.acompletion(
                model=self.model,
                messages=messages,
                timeout=self.timeout_sec,
            )

            content = response.choices[0].message.content or ""
            steps = self._parse_reasoning_steps(content, documents)

        except ImportError:
            log.warning("reasoning_agent.litellm_not_available")
            steps = self._fallback_reasoning(query, documents)
        except Exception as exc:
            log.error(
                "reasoning_agent.llm_call_failed",
                error_type=type(exc).__name__,
                detail=str(exc)[:200],
            )
            steps = self._fallback_reasoning(query, documents)

        return steps

    def _build_cot_prompt(
        self,
        query: str,
        documents: list[str],
    ) -> list[dict[str, str]]:
        """Build the chain-of-thought prompt messages.

        Args:
            query: The user query.
            documents: Retrieved documents for context.

        Returns:
            List of message dicts for the LLM API.
        """
        context_section = ""
        if documents:
            docs_text = "\n\n".join(f"[Doc {i + 1}]: {d}" for i, d in enumerate(documents))
            context_section = f"\n\nRelevant context:\n{docs_text}"

        return [
            {
                "role": "system",
                "content": (
                    "You are a precise reasoning agent. Think step by step. "
                    "For each step, state your reasoning clearly and rate your "
                    "confidence (0.0-1.0). Format each step as:\n"
                    "Step N: [thought] (confidence: X.X)\n\n"
                    "When a step draws on the retrieved context, append the source labels "
                    "at the end of that step's line:\n"
                    "Step N: [thought] (confidence: X.X) [sources: Doc 1, Doc 3]\n\n"
                    "The final step MUST contain your complete direct answer to the "
                    "question — not a formatting header like '**Answer:**'. "
                    "Put the answer text itself in the last step."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {query}{context_section}",
            },
        ]

    def _parse_reasoning_steps(
        self,
        content: str,
        documents: list[str] | None = None,
    ) -> list[ReasoningStep]:
        """Parse LLM output into structured reasoning steps.

        Extracts confidence scores and, when the LLM includes ``[sources: Doc N]``
        annotations, resolves them to the corresponding document text and populates
        ``ReasoningStep.evidence``.

        Args:
            content: Raw LLM response text.
            documents: Retrieved documents used to resolve ``Doc N`` citations.

        Returns:
            List of parsed ReasoningStep objects.
        """
        docs = documents or []
        steps: list[ReasoningStep] = []
        lines = content.strip().split("\n")
        step_num = 0

        for idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # Extract source citations before other processing
            evidence: list[str] = []
            src_match = _SOURCE_PAT.search(line)
            if src_match and docs:
                for ref in src_match.group(1).split(","):
                    doc_m = _DOC_LABEL_PAT.match(ref.strip())
                    if doc_m:
                        doc_idx = int(doc_m.group(1)) - 1  # "Doc 1" → index 0
                        if 0 <= doc_idx < len(docs):
                            evidence.append(docs[doc_idx])
                # Strip the [sources: ...] annotation from the displayed thought
                line = _SOURCE_PAT.sub("", line).strip()

            # Try to extract confidence from the line
            confidence = 0.5
            thought = line

            if "(confidence:" in line.lower():
                parts = line.rsplit("(confidence:", 1)
                thought = parts[0].strip()
                try:
                    conf_str = parts[1].replace(")", "").strip()
                    confidence = float(conf_str)
                    confidence = max(0.0, min(1.0, confidence))
                except (ValueError, IndexError):
                    confidence = 0.5

            # Remove "Step N:" prefix if present
            if thought.lower().startswith("step"):
                colon_idx = thought.find(":")
                if colon_idx != -1:
                    thought = thought[colon_idx + 1 :].strip()

            if thought:
                step_num += 1
                steps.append(
                    ReasoningStep(
                        step_number=step_num,
                        thought=thought,
                        confidence=confidence,
                        evidence=evidence if evidence else None,
                    )
                )

            if step_num >= self.max_steps:
                # If the last step is a formatting header (e.g. "**Answer:**"),
                # the real answer text follows on the next lines. Replace the
                # header thought with that content so it isn't lost.
                if steps and self._is_header_only(steps[-1].thought):
                    for next_line in lines[idx + 1 :]:
                        next_line = next_line.strip()
                        if next_line and not next_line.lower().startswith("step"):
                            steps[-1] = ReasoningStep(
                                step_number=steps[-1].step_number,
                                thought=next_line,
                                confidence=steps[-1].confidence,
                                evidence=steps[-1].evidence if steps[-1].evidence else None,
                            )
                            break
                break

        return steps

    @staticmethod
    def _is_header_only(thought: str) -> bool:
        """Return True if a thought is just a markdown/formatting header.

        Catches artifacts like "**Answer:**" or "Final Answer:" that LLMs
        sometimes emit as a transition line instead of putting the answer
        directly in the step text.

        Args:
            thought: The step thought string to check.

        Returns:
            True if the thought has two or fewer content words after stripping
            markdown characters.
        """
        stripped = thought.strip("*:#[]() \t")
        return len(stripped.split()) <= 2

    def _fallback_reasoning(
        self,
        query: str,
        documents: list[str],
    ) -> list[ReasoningStep]:
        """Generate basic reasoning without an LLM call.

        Used when litellm is unavailable or the LLM call fails.

        Args:
            query: The user query.
            documents: Retrieved documents.

        Returns:
            Minimal reasoning chain.
        """
        steps = [
            ReasoningStep(
                step_number=1,
                thought=f"Analyzing query: {query}",
                confidence=0.6,
            ),
        ]

        if documents:
            steps.append(
                ReasoningStep(
                    step_number=2,
                    thought=f"Reviewing {len(documents)} retrieved documents for evidence",
                    confidence=0.5,
                    evidence=documents[:3],
                )
            )

        steps.append(
            ReasoningStep(
                step_number=len(steps) + 1,
                thought="Formulating response based on available evidence",
                confidence=0.4,
            )
        )
        return steps

    def _build_critique_prompt(
        self,
        query: str,
        steps: list["ReasoningStep"],
    ) -> list[dict[str, str]]:
        """Build a prompt asking the LLM to critique a draft reasoning chain.

        Args:
            query: The original user query.
            steps: Draft reasoning steps to critique.

        Returns:
            List of message dicts for the LLM API.
        """
        chain_text = "\n".join(
            f"Step {s.step_number}: {s.thought} (confidence: {s.confidence:.1f})"
            for s in steps
        )
        return [
            {
                "role": "system",
                "content": (
                    "You are a critical evaluator of reasoning chains. "
                    "Identify up to three specific weaknesses, unsupported assumptions, "
                    "or logical gaps in the reasoning. Be concise. "
                    "List each critique on its own line starting with '- '."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {query}\n\nReasoning chain:\n{chain_text}",
            },
        ]

    def _build_refine_prompt(
        self,
        query: str,
        steps: list["ReasoningStep"],
        critique: str,
        documents: list[str],
    ) -> list[dict[str, str]]:
        """Build a prompt asking the LLM to produce a refined reasoning chain.

        The prompt instructs the LLM to address the critiques and end with a
        standalone ``Conclusion:`` paragraph separate from the numbered steps.

        Args:
            query: The original user query.
            steps: Draft reasoning steps that were critiqued.
            critique: The critique text from the second LLM call.
            documents: Retrieved documents for context (re-injected for grounding).

        Returns:
            List of message dicts for the LLM API.
        """
        chain_text = "\n".join(
            f"Step {s.step_number}: {s.thought} (confidence: {s.confidence:.1f})"
            for s in steps
        )
        docs_section = ""
        if documents:
            docs_section = "\n\nRelevant context:\n" + "\n\n".join(
                f"[Doc {i + 1}]: {d}" for i, d in enumerate(documents)
            )
        return [
            {
                "role": "system",
                "content": (
                    "You are a precise reasoning agent. Given a draft reasoning chain "
                    "and critiques of it, produce an improved chain that addresses the "
                    "critiques. Format each step as:\n"
                    "Step N: [thought] (confidence: X.X)\n\n"
                    "When a step draws on the retrieved context, append source labels:\n"
                    "Step N: [thought] (confidence: X.X) [sources: Doc 1, Doc 3]\n\n"
                    "After all steps, write a single paragraph starting with 'Conclusion:' "
                    "that directly and completely answers the question in plain language."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {query}\n\n"
                    f"Draft reasoning:\n{chain_text}\n\n"
                    f"Critiques to address:\n{critique}"
                    f"{docs_section}"
                ),
            },
        ]

    def _parse_refined_output(
        self,
        content: str,
        documents: list[str],
    ) -> tuple[list["ReasoningStep"], str]:
        """Parse the refine-call output into steps and a synthesised conclusion.

        Splits the ``Conclusion:`` paragraph out before parsing steps so the
        conclusion text is stored separately from the numbered chain.

        Args:
            content: Raw LLM output from the refine call.
            documents: Retrieved documents for resolving ``[sources: ...]`` labels.

        Returns:
            Tuple of (refined_steps, conclusion_text). ``conclusion_text`` is an
            empty string when no ``Conclusion:`` marker is found.
        """
        conclusion = ""
        marker = re.search(r"^conclusion:\s*", content, re.IGNORECASE | re.MULTILINE)
        if marker:
            conclusion = content[marker.end() :].strip()
            # Take only the first paragraph of the conclusion
            conclusion = conclusion.split("\n\n")[0].strip()
            content = content[: marker.start()]

        steps = self._parse_reasoning_steps(content, documents)
        return steps, conclusion

    async def reason_with_refinement(
        self,
        query: str,
        context: dict[str, Any],
    ) -> tuple[list["ReasoningStep"], str]:
        """Execute a draft → critique → refine reasoning cycle.

        Makes three sequential LLM calls:
        1. **Draft** — initial chain-of-thought (same as ``reason()``).
        2. **Critique** — identify weaknesses in the draft.
        3. **Refine** — improved chain addressing critiques, plus a standalone
           ``Conclusion:`` paragraph separate from the numbered steps.

        Falls back to ``_fallback_reasoning()`` if litellm is unavailable or
        any call fails.

        Args:
            query: The question to reason about.
            context: Orchestration context from prior agents.

        Returns:
            Tuple of (refined_steps, synthesised_conclusion).
        """
        documents = self._extract_documents(context)

        try:
            import litellm

            # --- Call 1: Draft ---
            draft_messages = self._build_cot_prompt(query, documents)
            draft_response = await litellm.acompletion(
                model=self.model,
                messages=draft_messages,
                timeout=self.timeout_sec,
            )
            draft_content = draft_response.choices[0].message.content or ""
            draft_steps = self._parse_reasoning_steps(draft_content, documents)

            if not draft_steps:
                steps = self._fallback_reasoning(query, documents)
                conclusion = steps[-1].thought if steps else ""
                return steps, conclusion

            # --- Call 2: Critique ---
            critique_messages = self._build_critique_prompt(query, draft_steps)
            critique_response = await litellm.acompletion(
                model=self.model,
                messages=critique_messages,
                timeout=self.timeout_sec,
            )
            critique = critique_response.choices[0].message.content or ""

            # --- Call 3: Refine ---
            refine_messages = self._build_refine_prompt(query, draft_steps, critique, documents)
            refine_response = await litellm.acompletion(
                model=self.model,
                messages=refine_messages,
                timeout=self.timeout_sec,
            )
            refined_content = refine_response.choices[0].message.content or ""

            refined_steps, synthesised_conclusion = self._parse_refined_output(
                refined_content, documents
            )

            if not refined_steps:
                refined_steps = draft_steps
            if not synthesised_conclusion:
                content_steps = [
                    s for s in refined_steps if not self._is_header_only(s.thought)
                ]
                synthesised_conclusion = (
                    content_steps[-1].thought if content_steps else refined_steps[-1].thought
                )

            return refined_steps, synthesised_conclusion

        except ImportError:
            log.warning("reasoning_agent.litellm_not_available")
        except Exception as exc:
            log.error(
                "reasoning_agent.refinement_failed",
                error_type=type(exc).__name__,
                detail=str(exc)[:200],
            )

        steps = self._fallback_reasoning(query, documents)
        conclusion = steps[-1].thought if steps else ""
        return steps, conclusion

    @staticmethod
    def _extract_documents(context: dict[str, Any]) -> list[str]:
        """Extract retrieved documents from context.

        Args:
            context: Orchestration context that may contain retrieval output.

        Returns:
            List of document strings.
        """
        last_output = context.get("last_output", {})
        if isinstance(last_output, dict):
            docs = last_output.get("documents", [])
            return [d.get("document", str(d)) if isinstance(d, dict) else str(d) for d in docs]
        return []


class TreeOfThoughtsReasoner:
    """Explores multiple reasoning paths and selects the best.

    Generates branching thoughts and evaluates each path to find
    the most coherent reasoning chain.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.max_branches: int = config.get("max_branches", 3)
        self.max_depth: int = config.get("max_depth", 3)
        self.cot_reasoner = ChainOfThoughtReasoner(config)

    async def reason(
        self,
        query: str,
        context: dict[str, Any],
    ) -> tuple[list[ThoughtBranch], ThoughtBranch | None]:
        """Explore multiple reasoning paths.

        Args:
            query: The question or task to reason about.
            context: Accumulated context from prior agents.

        Returns:
            Tuple of (all branches, best branch or None).
        """
        branches: list[ThoughtBranch] = []

        for i in range(self.max_branches):
            branch = ThoughtBranch(branch_id=f"branch-{i}")
            steps = await self.cot_reasoner.reason(query, context)
            for step in steps:
                branch.add_step(step)
            branches.append(branch)

        # Select best branch by score
        best = max(branches, key=lambda b: b.score) if branches else None

        return branches, best


class UncertaintyQuantifier:
    """Quantifies reasoning uncertainty from step confidences and branch divergence."""

    @staticmethod
    def quantify_from_steps(steps: list[ReasoningStep]) -> dict[str, float]:
        """Calculate uncertainty metrics from reasoning steps.

        Args:
            steps: List of reasoning steps.

        Returns:
            Dict with uncertainty metrics.
        """
        if not steps:
            return {"overall_uncertainty": 1.0, "min_confidence": 0.0, "avg_confidence": 0.0}

        confidences = [s.confidence for s in steps]
        avg = sum(confidences) / len(confidences)
        min_conf = min(confidences)

        return {
            "overall_uncertainty": 1.0 - avg,
            "min_confidence": min_conf,
            "avg_confidence": avg,
            "confidence_spread": max(confidences) - min_conf,
            "num_steps": len(steps),
        }

    @staticmethod
    def quantify_from_branches(branches: list[ThoughtBranch]) -> dict[str, float]:
        """Calculate uncertainty from branch divergence.

        Args:
            branches: List of explored thought branches.

        Returns:
            Dict with branch-level uncertainty metrics.
        """
        if not branches:
            return {"branch_agreement": 0.0, "best_branch_score": 0.0}

        scores = [b.score for b in branches]
        best_score = max(scores)
        avg_score = sum(scores) / len(scores)

        # Agreement: how close are branches in score (high = agree, low = diverge)
        agreement = avg_score / best_score if best_score > 0 else 0.0

        return {
            "branch_agreement": agreement,
            "best_branch_score": best_score,
            "avg_branch_score": avg_score,
            "num_branches": len(branches),
        }


class ReasoningAgent(BaseAgent):
    """Structured reasoning agent with chain-of-thought and tree-of-thoughts.

    Generates reasoning traces with uncertainty quantification.
    Never raises from execute() (ADR-006).
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__("reasoner", config)
        self.cot_reasoner = ChainOfThoughtReasoner(config)
        self.tot_reasoner = TreeOfThoughtsReasoner(config)
        self.uncertainty = UncertaintyQuantifier()
        self.reasoning_mode: str = config.get("reasoning_mode", "chain_of_thought")
        self.use_refinement: bool = bool(config.get("use_refinement", False))

    async def execute(self, request: AgentRequest) -> AgentResponse:
        """Execute reasoning on the request payload.

        Args:
            request: The agent request containing query and context.

        Returns:
            AgentResponse with reasoning trace and uncertainty metrics.
        """
        try:
            query = request.payload.get("query", "")
            if not query:
                return AgentResponse(
                    agent_id=self.agent_id,
                    status=AgentStatus.FAILED,
                    error="No query provided in payload",
                    confidence_score=0.0,
                )

            context = request.context

            if self.reasoning_mode == "tree_of_thoughts":
                return await self._execute_tot(query, context, request.task_id)
            if self.use_refinement:
                return await self._execute_cot_refined(query, context, request.task_id)
            return await self._execute_cot(query, context, request.task_id)

        except Exception as exc:
            log.error(
                "reasoning_agent.unexpected_error",
                agent_id=self.agent_id,
                task_id=request.task_id,
                error_type=type(exc).__name__,
            )
            return AgentResponse(
                agent_id=self.agent_id,
                status=AgentStatus.FAILED,
                error=f"Unexpected reasoning error: {type(exc).__name__}",
                confidence_score=0.0,
            )

    async def _execute_cot(
        self,
        query: str,
        context: dict[str, Any],
        task_id: str,
    ) -> AgentResponse:
        """Execute chain-of-thought reasoning.

        Args:
            query: The query to reason about.
            context: Orchestration context.
            task_id: The task identifier.

        Returns:
            AgentResponse with reasoning chain.
        """
        steps = await self.cot_reasoner.reason(query, context)

        if not steps:
            return AgentResponse(
                agent_id=self.agent_id,
                status=AgentStatus.FAILED,
                error="Reasoning produced no steps",
                confidence_score=0.0,
                requires_human_review=True,
            )

        uncertainty = self.uncertainty.quantify_from_steps(steps)
        confidence = 1.0 - uncertainty["overall_uncertainty"]

        # Derive conclusion from the last step that contains real content.
        # Filters out header-only artifacts ("**Answer:**") the LLM may emit.
        content_steps = [s for s in steps if not self.cot_reasoner._is_header_only(s.thought)]
        conclusion = content_steps[-1].thought if content_steps else steps[-1].thought

        return AgentResponse(
            agent_id=self.agent_id,
            status=AgentStatus.COMPLETED,
            output={
                "conclusion": conclusion,
                "reasoning_chain": [s.to_dict() for s in steps],
                "reasoning_mode": "chain_of_thought",
            },
            confidence_score=confidence,
            metadata={
                "uncertainty": uncertainty,
                "num_steps": len(steps),
                "reasoning_mode": "chain_of_thought",
            },
        )

    async def _execute_cot_refined(
        self,
        query: str,
        context: dict[str, Any],
        task_id: str,
    ) -> AgentResponse:
        """Execute chain-of-thought reasoning with a draft → critique → refine cycle.

        Makes three sequential LLM calls to produce a self-corrected reasoning
        chain and a synthesised conclusion that is distinct from the last step.

        Args:
            query: The query to reason about.
            context: Orchestration context.
            task_id: The task identifier.

        Returns:
            AgentResponse with refined reasoning chain and synthesised conclusion.
        """
        steps, conclusion = await self.cot_reasoner.reason_with_refinement(query, context)

        if not steps:
            return AgentResponse(
                agent_id=self.agent_id,
                status=AgentStatus.FAILED,
                error="Reasoning produced no steps",
                confidence_score=0.0,
                requires_human_review=True,
            )

        uncertainty = self.uncertainty.quantify_from_steps(steps)
        confidence = 1.0 - uncertainty["overall_uncertainty"]

        return AgentResponse(
            agent_id=self.agent_id,
            status=AgentStatus.COMPLETED,
            output={
                "conclusion": conclusion,
                "reasoning_chain": [s.to_dict() for s in steps],
                "reasoning_mode": "chain_of_thought_refined",
            },
            confidence_score=confidence,
            metadata={
                "uncertainty": uncertainty,
                "num_steps": len(steps),
                "reasoning_mode": "chain_of_thought_refined",
            },
        )

    async def _execute_tot(
        self,
        query: str,
        context: dict[str, Any],
        task_id: str,
    ) -> AgentResponse:
        """Execute tree-of-thoughts reasoning.

        Args:
            query: The query to reason about.
            context: Orchestration context.
            task_id: The task identifier.

        Returns:
            AgentResponse with best branch and all branches explored.
        """
        branches, best_branch = await self.tot_reasoner.reason(query, context)

        if not best_branch or not best_branch.steps:
            return AgentResponse(
                agent_id=self.agent_id,
                status=AgentStatus.FAILED,
                error="Tree-of-thoughts produced no viable branches",
                confidence_score=0.0,
                requires_human_review=True,
            )

        step_uncertainty = self.uncertainty.quantify_from_steps(best_branch.steps)
        branch_uncertainty = self.uncertainty.quantify_from_branches(branches)

        content_steps = [
            s for s in best_branch.steps if not self.cot_reasoner._is_header_only(s.thought)
        ]
        conclusion = content_steps[-1].thought if content_steps else best_branch.steps[-1].thought
        confidence = best_branch.score

        return AgentResponse(
            agent_id=self.agent_id,
            status=AgentStatus.COMPLETED,
            output={
                "conclusion": conclusion,
                "best_branch": best_branch.to_dict(),
                "all_branches": [b.to_dict() for b in branches],
                "reasoning_mode": "tree_of_thoughts",
            },
            confidence_score=confidence,
            metadata={
                "step_uncertainty": step_uncertainty,
                "branch_uncertainty": branch_uncertainty,
                "num_branches": len(branches),
                "reasoning_mode": "tree_of_thoughts",
            },
        )

    def can_handle(self, request: AgentRequest) -> bool:
        """ReasoningAgent handles requests with a query field."""
        return bool(request.payload.get("query"))

    async def health_probe(self, timeout_sec: int = 2) -> dict[str, Any]:
        """Return health status for the reasoning agent."""
        return {
            "healthy": True,
            "agent_id": self.agent_id,
            "reasoning_mode": self.reasoning_mode,
            "use_refinement": self.use_refinement,
            "capabilities": [
                "chain_of_thought",
                "chain_of_thought_refined",
                "tree_of_thoughts",
                "uncertainty_quantification",
            ],
        }
