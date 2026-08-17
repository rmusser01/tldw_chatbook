"""
Reranking: reorder search results after retrieval.

Three strategies (pointwise/pairwise/listwise) evaluate and reorder search
results with a language model; ``cross_encoder`` (TASK-16965) does it with a
local sentence-transformers cross-encoder, requiring no provider, no
credential and no network.

**No strategy here is known to improve retrieval, and the one that has been
measured made it worse on average.** ``cross_encoder`` is the only strategy
the gated eval instrument can see (the other three are remote, priced and
non-reproducible, so TASK-3502 left their value unmeasured). Measured under
a rule fixed before the run, ``cross_encoder`` came out net HARMFUL on the
averaged row and strongly BIMODAL by query category -- see
``CrossEncoderReranker`` for the numbers and
``Docs/superpowers/qa/2026-08-17-cross-encoder/report.md`` for the run. It
therefore ships selectable but is the default of nothing and the
recommendation of nothing.
"""

import asyncio
import functools
import threading
from typing import List, Optional, Union, Literal, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
from loguru import logger

# Optional numpy import
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

from ..Chat.Chat_Functions import chat_api_call
from ..Metrics.metrics_logger import log_counter, log_histogram, timeit
from tldw_chatbook.Internal_Prompts import get_internal_prompt, safe_substitute
from .simplified.vector_store import SearchResult, SearchResultWithCitations


@dataclass
class RerankingResult:
    """Result from reranking operation."""

    original_rank: int
    new_rank: int
    original_score: float
    rerank_score: float
    reasoning: Optional[str] = None
    #: Did a model actually score this row? ``False`` when the scoring call
    #: failed (or its response could not be parsed) and ``rerank_score`` is
    #: merely the original retrieval score carried forward unchanged.
    #: ``_apply_scores`` keys the ``rerank_score`` metadata stamp off this --
    #: that stamp is the production "this score is no longer a similarity"
    #: marker (``Library/library_rag_score_kinds.py``), so stamping it on a
    #: row whose scoring call failed rendered " | reranked" over a row no
    #: model ever looked at: a 14/15-failed rerank claimed fourteen rows it
    #: never rescored (TASK-3502 note-b).
    scored: bool = True

    @property
    def rank_change(self) -> int:
        """Calculate rank change (negative means improved)."""
        return self.new_rank - self.original_rank


@dataclass(frozen=True)
class RerankOutcome:
    """What ONE ``rerank()`` call did: the reordered results, plus how many of
    that call's scoring attempts failed (per-result for pointwise/listwise,
    per-comparison for pairwise).

    The counts are RETURNED rather than recorded on the reranker because the
    RAG service holds a reranker as a singleton across concurrent
    ``search()`` calls. They used to live in
    ``BaseReranker.last_rerank_failures``/``last_rerank_total``, which the
    disclosure site read AFTER ``rerank()`` had returned -- a window in which
    a concurrent search could overwrite them, so one search's degradation
    tag could describe another search's failures. Scoping the counts to the
    call REMOVES that shared state rather than guarding a diagnostic path
    with a lock (TASK-3502 AC#4).

    ``failed`` is the only way a caller can tell "rerank() returned normally
    but silently scored nothing" -- e.g. every provider call exhausting
    retries under a missing credential -- from "nothing needed reranking":
    both look identical, an unchanged ordering with no exception raised
    (task-3170 P0 review finding).
    """

    results: List[Union[SearchResult, SearchResultWithCitations]]
    failed: int
    total: int

    @property
    def degraded(self) -> bool:
        """True when at least one scoring attempt in this call failed."""
        return self.failed > 0


@dataclass
class RerankingConfig:
    """Configuration for reranking operations."""

    # Model settings
    model_provider: str = "openai"
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.0  # Use deterministic scoring
    max_tokens: int = 100

    # Reranking settings.
    #
    # ``cross_encoder`` is implemented and selectable (TASK-16965) but is
    # NOT a recommendation: measured on the gated instrument it is net
    # harmful on the averaged row and bimodal by category. It is the
    # default of no profile and no config template; pick it only with
    # Docs/superpowers/qa/2026-08-17-cross-encoder/report.md in hand.
    strategy: Literal["pairwise", "listwise", "pointwise", "cross_encoder"] = (
        "pointwise"
    )
    top_k_to_rerank: int = 20  # Only rerank top K results
    batch_size: int = 5  # Number of results to evaluate at once
    include_reasoning: bool = False  # Whether to generate explanations

    # Scoring settings
    score_scale: Tuple[float, float] = (0.0, 1.0)  # Min and max scores
    combine_original_score: bool = True  # Combine with original retrieval score
    original_score_weight: float = 0.3  # Weight for original score (0-1)

    # Prompts
    system_prompt: Optional[str] = None
    scoring_prompt_template: Optional[str] = None

    # Performance settings
    cache_results: bool = True
    timeout_seconds: float = 30.0
    retry_on_failure: bool = True
    max_retries: int = 2


class BaseReranker(ABC):
    """Base class for reranking strategies."""

    def __init__(self, config: RerankingConfig):
        self.config = config
        self._cache = {} if config.cache_results else None
        # NOTE (TASK-3502 AC#4): a reranker carries NO per-call state. How
        # many scoring attempts failed is part of `rerank()`'s RETURN value
        # (`RerankOutcome`), because the service holds this object as a
        # singleton across concurrent searches -- see `RerankOutcome` for the
        # misattribution this closes.

    @abstractmethod
    async def rerank(
        self,
        query: str,
        results: List[Union[SearchResult, SearchResultWithCitations]],
        **kwargs,
    ) -> RerankOutcome:
        """Rerank search results, returning them with this call's outcome."""
        pass

    def _get_cache_key(self, query: str, result_ids: List[str]) -> str:
        """Generate cache key for reranking operation."""
        import hashlib

        key_str = f"{query}|{'|'.join(sorted(result_ids))}"
        return hashlib.md5(key_str.encode()).hexdigest()

    # NOTE (TASK-16965): these two live on the BASE class, not on
    # PointwiseReranker where they were written. They are the shared
    # score-application contract -- the `scored`-flag stamp rule in
    # particular (TASK-3502 note-b) -- and `CrossEncoderReranker` is the
    # second strategy that REPLACES a row's score, so duplicating them
    # would mean two copies of that honesty rule free to drift apart.
    def _apply_scores(
        self,
        results: List[Union[SearchResult, SearchResultWithCitations]],
        reranking_results: List[RerankingResult],
    ) -> List[Union[SearchResult, SearchResultWithCitations]]:
        """Apply reranking scores and reorder results."""
        # Calculate final scores
        scored_results = []
        for result, rerank_result in zip(results, reranking_results):
            if not rerank_result.scored:
                # This row's scoring call failed: it keeps the score it
                # arrived with, and -- crucially -- is NOT stamped with
                # `rerank_score`. That stamp is what the Library reads as
                # "this number is no longer a similarity, render
                # ' | reranked'"; claiming it here made a partly-failed run
                # over-claim on every row it never rescored (TASK-3502
                # note-b). Conservative in both directions: no fabricated
                # score, and no fabricated provenance.
                final_score = result.score
                metadata = {**result.metadata}
            else:
                if self.config.combine_original_score:
                    # Weighted combination
                    final_score = (
                        self.config.original_score_weight * rerank_result.original_score
                        + (1 - self.config.original_score_weight)
                        * rerank_result.rerank_score
                    )
                else:
                    final_score = rerank_result.rerank_score
                metadata = {
                    **result.metadata,
                    "rerank_score": rerank_result.rerank_score,
                }

            # Create a copy of the result with new score
            result_copy = type(result)(
                id=result.id,
                score=final_score,
                document=result.document,
                metadata=metadata,
            )

            # Preserve citations if present
            if hasattr(result, "citations"):
                result_copy.citations = result.citations

            scored_results.append((final_score, result_copy, rerank_result))

        # Sort by final score (descending)
        scored_results.sort(key=lambda x: x[0], reverse=True)

        # Update ranks in reranking results
        for new_rank, (_, _, rerank_result) in enumerate(scored_results):
            rerank_result.new_rank = new_rank

        # Return reordered results
        return [result for _, result, _ in scored_results]

    def _log_reranking_metrics(self, reranking_results: List[RerankingResult]):
        """Log metrics about reranking performance."""
        if not reranking_results:
            return

        # Calculate rank changes
        rank_changes = [r.rank_change for r in reranking_results]
        avg_rank_change = sum(rank_changes) / len(rank_changes)

        # Calculate score changes
        score_changes = [r.rerank_score - r.original_score for r in reranking_results]
        avg_score_change = sum(score_changes) / len(score_changes)

        # Log metrics
        log_histogram("reranker_avg_rank_change", avg_rank_change)
        log_histogram("reranker_avg_score_change", avg_score_change)
        log_counter("reranker_results_processed", value=len(reranking_results))

        # Log significant reorderings
        significant_changes = sum(
            1 for r in reranking_results if abs(r.rank_change) >= 3
        )
        log_counter("reranker_significant_changes", value=significant_changes)

    async def _call_llm(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Call LLM with retry logic."""
        retries = 0
        while retries <= self.config.max_retries:
            try:
                response = await asyncio.wait_for(
                    self._call_llm_impl(prompt, system_prompt),
                    timeout=self.config.timeout_seconds,
                )
                return response
            except asyncio.TimeoutError:
                logger.warning(
                    f"LLM call timed out after {self.config.timeout_seconds}s"
                )
                if (
                    not self.config.retry_on_failure
                    or retries >= self.config.max_retries
                ):
                    raise
                retries += 1
                await asyncio.sleep(1 * retries)  # Exponential backoff
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                if (
                    not self.config.retry_on_failure
                    or retries >= self.config.max_retries
                ):
                    raise
                retries += 1
                await asyncio.sleep(1 * retries)

    async def _call_llm_impl(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> str:
        """Implementation of LLM call.

        The reranker resolves NO credential of its own (TASK-17065). Each
        ``chat_with_<provider>`` handler behind ``chat_api_call`` resolves
        its own key when the caller passes none -- through the normalised
        config path, with the precedence CLAUDE.md documents and
        ``resolve_provider_api_key``'s validity check applied -- and the
        keyless local providers (``ollama``/``llama_cpp``/``vllm``/
        ``koboldcpp``/``mlx_lm``/...) need none at all. This module used to
        hand-roll an ``if/elif`` over ``self._settings["API"]``, a table
        ``load_settings()`` never builds, and then reject every provider it
        did not name; that is how reranking reached 0 of the 29 providers
        the picker offers. Credential resolution is not this module's job.
        """
        # Prepare messages
        messages_payload = []
        if system_prompt or self.config.system_prompt:
            messages_payload.append(
                {
                    "role": "system",
                    # The enclosing `if` guarantees one of these is truthy, so
                    # no literal fallback is needed (each reranker's __init__
                    # already populates config.system_prompt from the registry).
                    "content": system_prompt or self.config.system_prompt,
                }
            )
        messages_payload.append({"role": "user", "content": prompt})

        # Call using chat_api_call
        try:
            # Run in executor since chat_api_call is sync. KEYWORDS, through
            # a partial: `run_in_executor` forwards only positionals, and a
            # positional list at this signature is exactly what used to route
            # the credential into `api_endpoint` -- the "Unsupported API
            # endpoint: <key>" failure, and the key-in-a-log-line disclosure
            # TASK-17165 had to redact downstream.
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                functools.partial(
                    chat_api_call,
                    api_endpoint=self.config.model_provider,
                    messages_payload=messages_payload,
                    model=self.config.model_name,
                    temp=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    # STATED, not inherited. Every handler currently
                    # DECLARES `streaming=False`, so the config is never
                    # consulted -- but the shipped `CONFIG_TOML_CONTENT`
                    # sets `streaming = true` for 18 of the 29 providers
                    # (every keyless local among them), and one handler
                    # declaring `streaming: bool | None = None` would hand
                    # this function a generator. Nothing here would raise:
                    # `str(<generator>)` parses as no JSON, every row comes
                    # back `scored=False`, and the search is billed in full
                    # for an entirely unscored rerank. Pinned by
                    # Tests/RAG_Search/test_reranker_degraded_paths.py.
                    streaming=False,
                ),
            )

            # Extract the text response
            if isinstance(response, dict):
                # Handle standard OpenAI-style response
                if "choices" in response and len(response["choices"]) > 0:
                    return response["choices"][0]["message"]["content"]
                # Handle other response formats
                elif "content" in response:
                    return response["content"]
                elif "text" in response:
                    return response["text"]
                else:
                    logger.warning(f"Unexpected response format: {response}")
                    return str(response)
            else:
                return str(response)

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise


class PointwiseReranker(BaseReranker):
    """Reranks each result independently with a relevance score."""

    def __init__(self, config: RerankingConfig):
        super().__init__(config)

        # Registry defaults only when the caller supplied none (caller wins).
        if not config.system_prompt:
            self.config.system_prompt = get_internal_prompt(
                "rag_reranker.pointwise_system"
            )
        if not config.scoring_prompt_template:
            self.config.scoring_prompt_template = get_internal_prompt(
                "rag_reranker.pointwise_template"
            )

    @timeit("reranker_pointwise")
    async def rerank(
        self,
        query: str,
        results: List[Union[SearchResult, SearchResultWithCitations]],
        **kwargs,
    ) -> RerankOutcome:
        """Rerank results using pointwise scoring."""
        if not results:
            return RerankOutcome(results=results, failed=0, total=0)

        # Limit to top K results
        results_to_rerank = results[: self.config.top_k_to_rerank]
        remaining_results = results[self.config.top_k_to_rerank :]

        # Check cache if enabled
        cache_key = None
        if self._cache is not None:
            cache_key = self._get_cache_key(query, [r.id for r in results_to_rerank])
            if cache_key in self._cache:
                log_counter("reranker_cache_hit", labels={"strategy": "pointwise"})
                cached_scores = self._cache[cache_key]
                # The cached `RerankingResult`s carry each row's `scored`
                # flag, so a cache HIT re-applies the same partial-failure
                # honesty the original call produced.
                return RerankOutcome(
                    results=self._apply_scores(results_to_rerank, cached_scores)
                    + remaining_results,
                    failed=sum(1 for r in cached_scores if not r.scored),
                    total=len(results_to_rerank),
                )

        log_counter("reranker_cache_miss", labels={"strategy": "pointwise"})

        # Score each result
        scoring_tasks = []
        for i, result in enumerate(results_to_rerank):
            task = self._score_result(query, result, i)
            scoring_tasks.append(task)

        # Process in batches
        all_scores = []
        for i in range(0, len(scoring_tasks), self.config.batch_size):
            batch = scoring_tasks[i : i + self.config.batch_size]
            batch_scores = await asyncio.gather(*batch, return_exceptions=True)
            all_scores.extend(batch_scores)

        # Handle errors and compile results
        reranking_results = []
        failed_count = 0
        for i, score_result in enumerate(all_scores):
            if isinstance(score_result, Exception):
                failed_count += 1
                logger.error(f"Failed to score result {i}: {score_result}")
                # Keep original score on failure -- and, because no model
                # scored this row, do NOT let it claim the reranked kind
                # (`scored=False`; TASK-3502 note-b).
                reranking_results.append(
                    RerankingResult(
                        original_rank=i,
                        new_rank=i,
                        original_score=results_to_rerank[i].score,
                        rerank_score=results_to_rerank[i].score,
                        scored=False,
                    )
                )
            else:
                reranking_results.append(score_result)

        # Cache results if enabled
        if self._cache is not None and cache_key:
            self._cache[cache_key] = reranking_results

        # Apply scores and reorder
        reranked = self._apply_scores(results_to_rerank, reranking_results)

        # Log metrics
        self._log_reranking_metrics(reranking_results)

        # `failed_count` is the ONLY thing distinguishing "returned normally
        # but silently scored nothing" from "returned normally because
        # everything genuinely scored" -- both look identical (an
        # unchanged/near-unchanged ordering) without it.
        return RerankOutcome(
            results=reranked + remaining_results,
            failed=failed_count,
            total=len(results_to_rerank),
        )

    async def _score_result(
        self,
        query: str,
        result: Union[SearchResult, SearchResultWithCitations],
        original_rank: int,
    ) -> RerankingResult:
        """Score a single result."""
        # Prepare content
        title = result.metadata.get("doc_title", "Untitled")
        content = result.document[:500]  # Limit content length

        # Format prompt
        reasoning_part = (
            ', "reasoning": "explanation"' if self.config.include_reasoning else ""
        )
        prompt = safe_substitute(
            self.config.scoring_prompt_template,
            query=query,
            title=title,
            content=content,
            reasoning=reasoning_part,
        )

        try:
            # Get LLM response
            response = await self._call_llm(prompt)

            # Parse JSON response
            result_json = json.loads(response)
            score = float(result_json.get("score", 0.5))
            reasoning = (
                result_json.get("reasoning") if self.config.include_reasoning else None
            )

            # Clamp score to configured range
            score = max(
                self.config.score_scale[0], min(score, self.config.score_scale[1])
            )

            return RerankingResult(
                original_rank=original_rank,
                new_rank=original_rank,  # Will be updated later
                original_score=result.score,
                rerank_score=score,
                reasoning=reasoning,
            )

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(
                f"Failed to parse LLM response: {e}, Response: {response[:200]}"
            )
            # Return original score on parse error, unstamped: nothing
            # usable came back, so this row was not rescored (note-b).
            return RerankingResult(
                original_rank=original_rank,
                new_rank=original_rank,
                original_score=result.score,
                rerank_score=result.score,
                scored=False,
            )


@dataclass
class _ComparisonTally:
    """Pairwise comparison counters for ONE ``rerank()`` call.

    Passed down the merge-sort recursion rather than kept on the reranker:
    the instance is shared across concurrent searches, so instance counters
    let one search's comparisons land in another's disclosure (TASK-3502
    AC#4). A "failure" is a comparison whose LLM call raised and fell back
    to comparing the original retrieval scores.
    """

    failed: int = 0
    total: int = 0


class PairwiseReranker(BaseReranker):
    """Reranks by comparing pairs of results."""

    def __init__(self, config: RerankingConfig):
        super().__init__(config)

        # Registry defaults only when the caller supplied none (caller wins).
        if not config.system_prompt:
            self.config.system_prompt = get_internal_prompt(
                "rag_reranker.pairwise_system"
            )
        if not config.scoring_prompt_template:
            self.config.scoring_prompt_template = get_internal_prompt(
                "rag_reranker.pairwise_template"
            )

    @timeit("reranker_pairwise")
    async def rerank(
        self,
        query: str,
        results: List[Union[SearchResult, SearchResultWithCitations]],
        **kwargs,
    ) -> RerankOutcome:
        """Rerank using pairwise comparisons with tournament-style ranking."""
        if len(results) <= 1:
            return RerankOutcome(results=results, failed=0, total=0)

        # Limit to top K
        results_to_rerank = results[: self.config.top_k_to_rerank]
        remaining_results = results[self.config.top_k_to_rerank :]

        tally = _ComparisonTally()

        # Perform tournament-style comparisons
        reranked = await self._tournament_rank(query, results_to_rerank, tally)

        log_counter(
            "reranker_pairwise_complete", labels={"results": len(results_to_rerank)}
        )

        return RerankOutcome(
            results=reranked + remaining_results,
            failed=tally.failed,
            total=tally.total,
        )

    async def _tournament_rank(
        self,
        query: str,
        results: List[Union[SearchResult, SearchResultWithCitations]],
        tally: _ComparisonTally,
    ) -> List[Union[SearchResult, SearchResultWithCitations]]:
        """Use tournament-style ranking with pairwise comparisons."""
        # Implementation of merge sort with async comparisons
        if len(results) <= 1:
            return results

        mid = len(results) // 2
        left_half = results[:mid]
        right_half = results[mid:]

        # Recursively sort both halves
        left_sorted = await self._tournament_rank(query, left_half, tally)
        right_sorted = await self._tournament_rank(query, right_half, tally)

        # Merge with pairwise comparisons
        return await self._merge_with_comparisons(
            query, left_sorted, right_sorted, tally
        )

    async def _merge_with_comparisons(
        self, query: str, left: List, right: List, tally: _ComparisonTally
    ) -> List:
        """Merge two sorted lists using pairwise comparisons."""
        result = []
        i = j = 0

        while i < len(left) and j < len(right):
            # Compare current elements
            is_left_better = await self._compare_pair(query, left[i], right[j], tally)

            if is_left_better:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1

        # Add remaining elements
        result.extend(left[i:])
        result.extend(right[j:])

        return result

    async def _compare_pair(
        self,
        query: str,
        result1: Union[SearchResult, SearchResultWithCitations],
        result2: Union[SearchResult, SearchResultWithCitations],
        tally: _ComparisonTally,
    ) -> bool:
        """Compare two results and return True if result1 is better."""
        # Format prompt
        reasoning_part = (
            ', "reasoning": "explanation"' if self.config.include_reasoning else ""
        )
        prompt = safe_substitute(
            self.config.scoring_prompt_template,
            query=query,
            title1=result1.metadata.get("doc_title", "Untitled"),
            content1=result1.document[:300],
            title2=result2.metadata.get("doc_title", "Untitled"),
            content2=result2.document[:300],
            reasoning=reasoning_part,
        )

        tally.total += 1
        try:
            response = await self._call_llm(prompt)
            result_json = json.loads(response)
            choice = int(result_json.get("choice", 1))

            log_counter("reranker_pairwise_comparison")

            return choice == 1

        except Exception as e:
            logger.error(f"Pairwise comparison failed: {e}")
            tally.failed += 1
            # Fall back to original scores
            return result1.score > result2.score


class ListwiseReranker(BaseReranker):
    """Reranks all results together in a single prompt."""

    def __init__(self, config: RerankingConfig):
        super().__init__(config)

        # Registry defaults only when the caller supplied none (caller wins).
        if not config.system_prompt:
            self.config.system_prompt = get_internal_prompt(
                "rag_reranker.listwise_system"
            )
        if not config.scoring_prompt_template:
            self.config.scoring_prompt_template = get_internal_prompt(
                "rag_reranker.listwise_template"
            )

    @timeit("reranker_listwise")
    async def rerank(
        self,
        query: str,
        results: List[Union[SearchResult, SearchResultWithCitations]],
        **kwargs,
    ) -> RerankOutcome:
        """Rerank all results together."""
        if len(results) <= 1:
            return RerankOutcome(results=results, failed=0, total=0)

        # Limit to top K
        results_to_rerank = results[
            : min(self.config.top_k_to_rerank, 10)
        ]  # Limit for prompt size
        remaining_results = results[len(results_to_rerank) :]

        # Format results for prompt
        results_text = []
        for i, result in enumerate(results_to_rerank):
            title = result.metadata.get("doc_title", "Untitled")
            content = result.document[:200]
            results_text.append(f"{i}. Title: {title}\n   Content: {content}...")

        results_list = "\n\n".join(results_text)

        # Format prompt
        reasoning_part = (
            ', "reasoning": "explanation"' if self.config.include_reasoning else ""
        )
        prompt = safe_substitute(
            self.config.scoring_prompt_template,
            query=query,
            results_list=results_list,
            reasoning=reasoning_part,
        )

        try:
            response = await self._call_llm(prompt)
            result_json = json.loads(response)
            ranking = result_json.get("ranking", list(range(len(results_to_rerank))))

            # Validate ranking
            if not self._validate_ranking(ranking, len(results_to_rerank)):
                logger.warning("Invalid ranking returned, using original order")
                # No individual result was scored (the whole ranking is
                # discarded), so this counts as a total failure -- same
                # "returned normally but scored nothing" shape as the except
                # branch below.
                return RerankOutcome(
                    results=results,
                    failed=len(results_to_rerank),
                    total=len(results_to_rerank),
                )

            # Reorder results
            reranked = [results_to_rerank[i] for i in ranking]

            log_counter("reranker_listwise_complete")

            return RerankOutcome(
                results=reranked + remaining_results,
                failed=0,
                total=len(results_to_rerank),
            )

        except Exception as e:
            logger.error(f"Listwise reranking failed: {e}")
            return RerankOutcome(
                results=results,
                failed=len(results_to_rerank),
                total=len(results_to_rerank),
            )

    def _validate_ranking(self, ranking: List[int], expected_length: int) -> bool:
        """Validate that ranking contains all indices exactly once."""
        if len(ranking) != expected_length:
            return False
        return set(ranking) == set(range(expected_length))


#: The cross-encoder loaded when the config does not name one. Measured in
#: this environment before the strategy was written (TASK-16965): it loads
#: from the local HF cache offline and separates a relevant pair (+8.719)
#: from an irrelevant one (-11.14). ``mixedbread-ai/mxbai-rerank-large-v2``
#: is NOT usable here -- the cached copy is a 20 MB partial with no weights
#: file and raises ``OSError`` offline.
DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

#: Substrings that mark a model id as a reranker/cross-encoder artifact.
#: Checked FIRST, so a vendor namespace that publishes both chat models and
#: rerankers (``Qwen/Qwen3-Reranker-0.6B``) is not redirected to the default.
_RERANKER_NAME_MARKERS = ("rerank", "cross-encoder", "cross_encoder", "colbert")

#: Repo-id namespaces that publish CHAT models. A profile that names one of
#: these has configured an LLM strategy's model, not a cross-encoder.
_LLM_REPO_NAMESPACES = frozenset(
    {
        "openai",
        "anthropic",
        "google",
        "meta-llama",
        "mistralai",
        "deepseek-ai",
        "x-ai",
        "cohere",
        "perplexity",
        "openrouter",
        "groq",
        "moonshotai",
    }
)

#: Loaded cross-encoders, keyed by model name. Module level ON PURPOSE: the
#: model costs ~18 s to load, the RAG service builds a reranker per profile
#: switch, and a measurement run reranks 60 queries x 2 modes -- reloading
#: per instance would dominate the wall clock and measure the loader.
_CROSS_ENCODER_MODELS: dict = {}
_CROSS_ENCODER_MODELS_LOCK = threading.Lock()


def _import_cross_encoder_class():
    """Import ``sentence_transformers.CrossEncoder`` at FIRST USE.

    Deliberately not a module-level import: ``sentence-transformers`` ships
    in the ``embeddings_rag`` extra, and importing torch at
    ``reranker.py`` import time would tax every caller of the three
    LLM strategies (and every test that imports this module) with it.
    Isolated in its own function so a unit test can refuse it outright and
    prove no model is ever downloaded.
    """
    from sentence_transformers import CrossEncoder

    return CrossEncoder


def _load_cross_encoder(model_name: str):
    """Return the cached cross-encoder for ``model_name``, loading it once."""
    model = _CROSS_ENCODER_MODELS.get(model_name)
    if model is not None:
        return model

    with _CROSS_ENCODER_MODELS_LOCK:
        # Re-check inside the lock: two searches can race the first load.
        model = _CROSS_ENCODER_MODELS.get(model_name)
        if model is None:
            logger.info(f"Loading cross-encoder model '{model_name}' (first use)")
            model = _import_cross_encoder_class()(model_name)
            _CROSS_ENCODER_MODELS[model_name] = model
    return model


def _resolve_cross_encoder_model_name(model_name: Optional[str]) -> str:
    """Pick the cross-encoder artifact to load from a config's ``model_name``.

    ``RerankingConfig.model_name`` defaults to ``"gpt-3.5-turbo"`` and every
    shipped profile sets a chat model there, because the three strategies
    that existed before this one are LLM-driven. Handing such a name to
    ``CrossEncoder`` would fail at load and degrade every rerank, so a name
    that is not a plausible model artifact falls back to the measured
    default. A caller that genuinely wants a different cross-encoder passes
    its full repo id or a local path, which is used verbatim.
    """
    name = (model_name or "").strip()
    if not name:
        return DEFAULT_CROSS_ENCODER_MODEL

    lowered = name.lower()
    if any(marker in lowered for marker in _RERANKER_NAME_MARKERS):
        return name

    namespace, separator, _remainder = name.partition("/")
    if not separator:
        # No namespace: a bare chat model name ("gpt-4o-mini"), not a repo id.
        return DEFAULT_CROSS_ENCODER_MODEL
    if namespace.lower() in _LLM_REPO_NAMESPACES:
        return DEFAULT_CROSS_ENCODER_MODEL
    return name


class CrossEncoderReranker(BaseReranker):
    """Reranks with a LOCAL cross-encoder: no provider, no credential, no spend.

    The other three strategies ask an LLM to score, compare or reorder
    results through ``chat_api_call``. This one runs a sentence-transformers
    cross-encoder over ``(query, document)`` pairs in-process, which is what
    makes reranking measurable on the gated eval instrument at all
    (TASK-16965): deterministic, unpriced, offline.

    Two ``RerankingConfig`` fields are PROVIDER concepts and are therefore
    no-ops here, stated rather than implied: ``max_retries`` (there is no
    remote call to retry -- a failed ``predict`` is a local failure that a
    second identical call would repeat) and ``include_reasoning`` (there is
    no model being asked to explain itself; a cross-encoder emits one
    number). ``model_provider`` is likewise unused: the model IS the
    provider.

    Scores are min-max normalised into ``config.score_scale`` over the
    window. Cross-encoder outputs are unbounded logits (roughly -11..+11 for
    the ms-marco default), so CLAMPING them to the (0.0, 1.0) scale the way
    the pointwise strategy clamps an LLM's 0-1 score would collapse every
    strongly-relevant row to 1.0 -- ties, in the one place ordering is the
    entire product. Normalising is monotonic, so it preserves the model's
    ordering exactly while putting the numbers on the scale
    ``combine_original_score`` blends against.

    **What it measured, so nobody has to re-run the probe to find out.**
    TASK-16965 ran it over the 60-query golden set on the gated instrument,
    against a rule fixed in the plan BEFORE the strategy was written, in
    two pre-declared arms (rerank the k=10 window; and retrieve 20, rerank,
    score the first 10 -- the second arm exists because permuting a <=k
    list cannot move P@k/recall/F1 at all). **Verdict: HARMED.** On the
    averaged overall row at k=10, arm B: semantic MRR 0.808 -> 0.762 and
    NDCG 0.804 -> 0.776; hybrid MRR 0.812 -> 0.787 and NDCG 0.817 -> 0.805;
    recall@10 +0.022 on both. The strategy is NOT inert -- 3,621 rows
    scored, 0 failed, 1,950 rows moved -- and its effect is strongly
    BIMODAL by query category:

    * large gains where retrieval was weak: hybrid ``scoped`` MRR
      0.163 -> 0.929 (NDCG 0.348 -> 0.947), hybrid ``prompt`` MRR
      0.022 -> 0.200, semantic ``negation`` NDCG 0.000 -> 0.105;
    * losses where retrieval was already perfect: ``paraphrase`` (13
      queries) and ``vocabulary_mismatch`` (9) both sat at MRR 1.000, so
      the only movement available was down -- four queries lost rank 1,
      taking those categories to 0.87-0.94. Those are the cells that trip
      the rule's regression clause.

    Read on the overall row ALONE nothing moves beyond the 0.05 band, i.e.
    a NULL; the rule's regression clause is written at category level, so
    the reported verdict is HARMED. Either reading says the same thing in
    practice: **do not enable this expecting better search.** Enable it
    only if your corpus looks like the weak-retrieval half of that split,
    and measure it yourself. Full run, per-category tables and the census:
    ``Docs/superpowers/qa/2026-08-17-cross-encoder/report.md``.

    **Why this class still exists, as TWO facts and not one.** (1) The
    pre-registered rule was applied verbatim and said RETIRE the name --
    HARMED, on the category clause, in both arms. (2) The owner ruled
    otherwise on 2026-08-17, asked explicitly with the trade-offs shown:
    *"KEEP THE CODE, RETIRE THE PROMISE."* The deciding dimension was the
    bimodal split above -- the other three strategies bill a remote
    provider per call, which the local deterministic gate cannot run, so
    this is the ONLY reranking path anyone here can measure at all
    (TASK-3502 scoped the question out for exactly that reason), and
    deleting it for producing an unwelcome number would delete the
    instrument along with the result. The price of keeping it is stated
    rather than hidden: it ships selectable, the default of nothing, the
    recommendation of nothing, with its measured harm attached at every
    site that names it.
    """

    def __init__(self, config: RerankingConfig):
        super().__init__(config)
        self.model_name = _resolve_cross_encoder_model_name(config.model_name)

    @timeit("reranker_cross_encoder")
    async def rerank(
        self,
        query: str,
        results: List[Union[SearchResult, SearchResultWithCitations]],
        **kwargs,
    ) -> RerankOutcome:
        """Rerank the top-k window with the cross-encoder."""
        if not results:
            return RerankOutcome(results=results, failed=0, total=0)

        results_to_rerank = results[: self.config.top_k_to_rerank]
        remaining_results = results[self.config.top_k_to_rerank :]

        cache_key = None
        if self._cache is not None:
            cache_key = self._get_cache_key(query, [r.id for r in results_to_rerank])
            if cache_key in self._cache:
                log_counter("reranker_cache_hit", labels={"strategy": "cross_encoder"})
                cached_scores = self._cache[cache_key]
                return RerankOutcome(
                    results=self._apply_scores(results_to_rerank, cached_scores)
                    + remaining_results,
                    failed=sum(1 for r in cached_scores if not r.scored),
                    total=len(results_to_rerank),
                )

        log_counter("reranker_cache_miss", labels={"strategy": "cross_encoder"})

        raw_scores = await self._predict_scores(query, results_to_rerank)
        if raw_scores is None:
            # Same shape as ListwiseReranker's failure: nothing was scored,
            # so the caller's own ordering comes back untouched and unstamped
            # (TASK-3502 note-b -- no row may claim a rerank that never
            # happened), with the counts that let the service disclose it as
            # degraded rather than as a successful no-op. Not cached: a model
            # failure is transient in a way a parsed score is not.
            log_counter("reranker_cross_encoder_failed")
            return RerankOutcome(
                results=results,
                failed=len(results_to_rerank),
                total=len(results_to_rerank),
            )

        reranking_results = self._normalize_scores(results_to_rerank, raw_scores)

        if self._cache is not None and cache_key:
            self._cache[cache_key] = reranking_results

        reranked = self._apply_scores(results_to_rerank, reranking_results)
        self._log_reranking_metrics(reranking_results)

        return RerankOutcome(
            results=reranked + remaining_results,
            failed=0,
            total=len(results_to_rerank),
        )

    async def _predict_scores(
        self,
        query: str,
        rows: List[Union[SearchResult, SearchResultWithCitations]],
    ) -> Optional[List[float]]:
        """Score every row in one batched ``predict``; ``None`` on failure."""
        try:
            loop = asyncio.get_running_loop()
            scores = await loop.run_in_executor(
                None, functools.partial(self._predict_scores_sync, query, rows)
            )
        except Exception as exc:
            # Reranking must NEVER fail a search: a missing model file, a
            # missing sentence-transformers install and an OOM all land here
            # and degrade to the unreranked ordering.
            logger.error(f"Cross-encoder reranking failed ({self.model_name}): {exc}")
            return None

        if len(scores) != len(rows):
            logger.error(
                f"Cross-encoder returned {len(scores)} scores for {len(rows)} rows; "
                "discarding them"
            )
            return None
        return scores

    def _predict_scores_sync(
        self,
        query: str,
        rows: List[Union[SearchResult, SearchResultWithCitations]],
    ) -> List[float]:
        """Blocking model call, run in an executor.

        The document is passed WHOLE: the model truncates to its own
        ``max_length``, so a hand-picked character cap here would only be a
        second, wronger truncation.
        """
        model = _load_cross_encoder(self.model_name)
        pairs = [(query, row.document or "") for row in rows]
        return [float(score) for score in model.predict(pairs)]

    def _normalize_scores(
        self,
        rows: List[Union[SearchResult, SearchResultWithCitations]],
        raw_scores: List[float],
    ) -> List[RerankingResult]:
        """Min-max the window's logits into ``score_scale`` (order-preserving)."""
        low, high = self.config.score_scale
        lowest = min(raw_scores)
        span = max(raw_scores) - lowest

        reranking_results = []
        for rank, (row, raw_score) in enumerate(zip(rows, raw_scores)):
            if span > 0:
                normalized = low + (high - low) * (raw_score - lowest) / span
            else:
                # Every row scored identically: the model expressed no
                # preference, so give them all the same value and let the
                # original retrieval score break the tie.
                normalized = (low + high) / 2
            reranking_results.append(
                RerankingResult(
                    original_rank=rank,
                    new_rank=rank,  # Updated by _apply_scores.
                    original_score=row.score,
                    rerank_score=normalized,
                )
            )
        return reranking_results


def create_reranker(strategy: str = "pointwise", **kwargs) -> BaseReranker:
    """Factory function to create a reranker with the specified strategy."""
    config = RerankingConfig(strategy=strategy, **kwargs)
    return create_reranker_from_config(config)


def create_reranker_from_config(config: RerankingConfig) -> BaseReranker:
    """Build a reranker directly from an already-constructed ``RerankingConfig``.

    This is the seam every caller that already HAS a ``RerankingConfig``
    (profiles, experiment configs) must use. ``create_reranker(strategy=X,
    **config.__dict__)`` -- the previous call pattern everywhere in
    ``enhanced_rag_service_v2.py`` -- passes ``strategy`` twice (once
    positionally/by-keyword, once again inside ``**config.__dict__``, since
    ``strategy`` is itself a field of ``RerankingConfig``), which raised
    ``TypeError: RerankingConfig() got multiple values for keyword argument
    'strategy'`` on every reranking-enabled profile. This function takes the
    config object as-is and dispatches on ``config.strategy`` without
    reconstructing it.
    """
    strategy = config.strategy

    if strategy == "pointwise":
        return PointwiseReranker(config)
    elif strategy == "pairwise":
        return PairwiseReranker(config)
    elif strategy == "listwise":
        return ListwiseReranker(config)
    elif strategy == "cross_encoder":
        return CrossEncoderReranker(config)
    else:
        raise ValueError(f"Unknown reranking strategy: {strategy}")


# Convenience function for one-shot reranking
async def rerank_results(
    query: str,
    results: List[Union[SearchResult, SearchResultWithCitations]],
    strategy: str = "pointwise",
    **kwargs,
) -> RerankOutcome:
    """Rerank without creating a reranker instance.

    Returns the full ``RerankOutcome`` (not just the results) so a one-shot
    caller cannot silently drop the degradation counts -- the same reason
    ``rerank()`` itself returns one.
    """
    reranker = create_reranker(strategy, **kwargs)
    return await reranker.rerank(query, results, **kwargs)
