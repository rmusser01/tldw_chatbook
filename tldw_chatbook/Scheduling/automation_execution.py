"""Execution core for local `recurring_question` automation definitions.

Composes two already-hardened Library seams -- `run_library_rag_search`
(retrieval) and `generate_library_rag_answer` (contained, citation-grounded
generation) -- with `recurring_question_scope`'s normalizer, and classifies
the pair of results into an `ExecutionOutcome` mirroring the server's
`classify_rag_response` vocabulary (`finding`/`no_match`/`degraded` x
`synthesized`/`evidence_only`/`none`). No new retrieval or generation
machinery lives here.

This module is imported lazily, inside the spawned run coroutine (Task 4's
`AutomationDefinitionHandler`), never at that handler module's top level --
the two Library seams it imports are heavier than the boot-census ratchet
(ADR-097) allows a handler-module import to carry.

Score-filter decision (`finding_policy` preset `high_confidence_only`):
`LibraryRagResultRow.score` exists, but it is not a single comparable
"confidence" number on its own -- it sits on different scales depending on
`row.score_kind` (a plain vector cosine similarity, an RRF-fused rank score,
or an unbounded reranker score; see `Library/library_rag_score_kinds.py`,
which exists specifically to stop code from banding a fused ~0.02 the way a
cosine similarity of 0.5 would band). The post-filter therefore resolves
each row's comparable similarity via `library_rag_similarity_input(...)` and
keeps it when that similarity is `>= LIBRARY_RAG_MATCH_STRONG_THRESHOLD` --
the same threshold the evidence panel bands "match: strong" on. A row with
no comparable similarity at all (a reranked row, or an FTS-leg-only hybrid
row) resolves to `None` and is KEPT rather than dropped: dropping it would
assert a confidence judgement about a number the row does not carry, which
is exactly the invented claim the score-kind machinery exists to prevent.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..config import coerce_int_setting, get_cli_setting
from .recurring_question_scope import engine_source_types, normalize_recurring_question_scope
from ..Library.library_rag_answer_service import (
    ANSWER_STATUS_READY,
    LibraryRagAnswer,
    generate_library_rag_answer,
    resolve_library_rag_answer_provider,
)
from ..Library.library_rag_score_kinds import library_rag_similarity_input
from ..Library.library_rag_service import (
    LibraryRagSearchRequest,
    run_library_rag_search,
)
from ..Library.library_rag_state import LIBRARY_RAG_MATCH_STRONG_THRESHOLD

#: Every summary this module writes is bounded to this length (server parity).
RESULT_SUMMARY_MAX_CHARS = 1000
_SUMMARY_ELLIPSIS = "…"

#: `resolve_execution_target` bounds.
_MAX_TOKENS_CAP = 4000
_DEFAULT_MAX_TOKENS = 1000

#: `finding_policy` -> retrieval `top_k`.
_FINDING_POLICY_TOP_K_DEFAULT = 10
_FINDING_POLICY_TOP_K_MIN = 1
_FINDING_POLICY_TOP_K_MAX = 100
_HIGH_CONFIDENCE_PRESET = "high_confidence_only"

_VALID_GENERATION_MODES = {"disabled", "optional", "required"}
_RETRIEVAL_DEGRADED_STATUSES = frozenset({"blocked", "failed"})


@dataclass(frozen=True)
class ExecutionOutcome:
    """One `recurring_question` run's result, in the server's vocabulary."""

    outcome: str  # finding | no_match | degraded
    title: str
    summary: str
    answer: Any | None
    answer_mode: str  # synthesized | evidence_only | none
    confidence: dict
    source_refs: list[dict]
    evidence_summary: dict  # {result_count, answer_present, retrieval_status, generation_status}
    failure_reason: dict | None


def _bounded(text: str | None) -> str:
    """Cap `text` at `RESULT_SUMMARY_MAX_CHARS`, ellipsis included."""
    value = str(text or "")
    if len(value) <= RESULT_SUMMARY_MAX_CHARS:
        return value
    return value[: RESULT_SUMMARY_MAX_CHARS - len(_SUMMARY_ELLIPSIS)] + _SUMMARY_ELLIPSIS


def _sanitize_text(value: Any) -> str | None:
    """A non-blank stripped string, or `None` for blank/junk input."""
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _sanitize_positive_int(value: Any) -> int | None:
    """A positive int, or `None` for blank/junk/non-positive input."""
    if value is None or isinstance(value, bool):
        return None
    try:
        as_int = int(value)
    except (TypeError, ValueError):
        return None
    return as_int if as_int > 0 else None


def resolve_execution_target(definition_row: dict) -> dict:
    """Resolve `{provider, model, max_tokens}` for one execution.

    Precedence, sanitized independently at each layer before it is
    consulted (server review-#5 discipline -- a blank or junk value at a
    layer falls through to the next one, it never wins by being present):
    definition `input.provider`/`input.model`/`input.max_tokens`, then
    `[scheduling] executor_provider`/`executor_model`/`executor_max_tokens`,
    then `resolve_library_rag_answer_provider()` for provider/model (there
    is no config-default fallback for `max_tokens`). `max_tokens` defaults
    to `_DEFAULT_MAX_TOKENS` when nothing resolves one, and is always capped
    at `_MAX_TOKENS_CAP`.
    """
    definition_input = definition_row.get("input") if isinstance(definition_row, Mapping) else None
    if not isinstance(definition_input, Mapping):
        definition_input = {}

    provider = _sanitize_text(definition_input.get("provider"))
    model = _sanitize_text(definition_input.get("model"))
    max_tokens = _sanitize_positive_int(definition_input.get("max_tokens"))

    if provider is None:
        provider = _sanitize_text(get_cli_setting("scheduling", "executor_provider", None))
    if model is None:
        model = _sanitize_text(get_cli_setting("scheduling", "executor_model", None))
    if max_tokens is None:
        max_tokens = _sanitize_positive_int(
            get_cli_setting("scheduling", "executor_max_tokens", None)
        )

    if provider is None or model is None:
        fallback_provider, fallback_model = resolve_library_rag_answer_provider()
        if provider is None:
            provider = fallback_provider
        if model is None:
            model = fallback_model

    if max_tokens is None:
        max_tokens = _DEFAULT_MAX_TOKENS
    max_tokens = min(max_tokens, _MAX_TOKENS_CAP)

    return {"provider": provider, "model": model, "max_tokens": max_tokens}


def _resolve_finding_policy(finding_policy: Any) -> tuple[int, bool]:
    """`(top_k, high_confidence_only)` from a `finding_policy` dict."""
    policy = finding_policy if isinstance(finding_policy, Mapping) else {}
    preset = str(policy.get("preset") or "balanced_findings")
    top_k = _FINDING_POLICY_TOP_K_DEFAULT
    if "top_k" in policy:
        top_k = coerce_int_setting(
            policy.get("top_k"),
            top_k,
            minimum=_FINDING_POLICY_TOP_K_MIN,
            maximum=_FINDING_POLICY_TOP_K_MAX,
        )
    return top_k, preset == _HIGH_CONFIDENCE_PRESET


def _filter_high_confidence(results: tuple) -> tuple:
    """Keep rows whose comparable similarity is strong, or has none at all."""
    kept = []
    for row in results:
        similarity = library_rag_similarity_input(
            row.score, score_kind=row.score_kind, vector_score=row.vector_score
        )
        if similarity is None or similarity >= LIBRARY_RAG_MATCH_STRONG_THRESHOLD:
            kept.append(row)
    return tuple(kept)


def _source_refs(results: tuple) -> list[dict]:
    """One `{source, id, title}` dict per row, from fields the row carries."""
    refs = []
    for row in results:
        provenance = row.provenance if isinstance(row.provenance, Mapping) else {}
        refs.append(
            {
                "source": provenance.get("source_type", ""),
                "id": row.result_id,
                "title": row.title,
            }
        )
    return refs


def _evidence_summary(
    *, result_count: int, answer_present: bool, retrieval_status: str, generation_status: str
) -> dict:
    return {
        "result_count": result_count,
        "answer_present": answer_present,
        "retrieval_status": retrieval_status,
        "generation_status": generation_status,
    }


def _degraded(*, failure_code: str, evidence_summary: dict) -> ExecutionOutcome:
    return ExecutionOutcome(
        outcome="degraded",
        title="",
        summary="",
        answer=None,
        answer_mode="none",
        confidence={},
        source_refs=[],
        evidence_summary=evidence_summary,
        failure_reason={"code": failure_code},
    )


def _classify(
    *,
    retrieval_status: str,
    results: tuple,
    generation_mode: str,
    answer: LibraryRagAnswer | None,
) -> ExecutionOutcome:
    """The six-row classification ladder (server `classify_rag_response`,
    adapted to the `run_library_rag_search`/`generate_library_rag_answer`
    seams)."""
    if retrieval_status in _RETRIEVAL_DEGRADED_STATUSES:
        return _degraded(
            failure_code=f"retrieval_{retrieval_status}",
            evidence_summary=_evidence_summary(
                result_count=0,
                answer_present=False,
                retrieval_status=retrieval_status,
                generation_status="",
            ),
        )

    if not results:
        return ExecutionOutcome(
            outcome="no_match",
            title="No matching sources found",
            summary="",
            answer=None,
            answer_mode="none",
            confidence={},
            source_refs=[],
            evidence_summary=_evidence_summary(
                result_count=0,
                answer_present=False,
                retrieval_status=retrieval_status,
                generation_status="",
            ),
            failure_reason=None,
        )

    result_count = len(results)
    source_refs = _source_refs(results)

    if generation_mode == "disabled":
        return ExecutionOutcome(
            outcome="finding",
            title="Relevant evidence found",
            summary="",
            answer=None,
            answer_mode="evidence_only",
            confidence={},
            source_refs=source_refs,
            evidence_summary=_evidence_summary(
                result_count=result_count,
                answer_present=False,
                retrieval_status=retrieval_status,
                generation_status="",
            ),
            failure_reason=None,
        )

    # generation_mode is "optional" or "required": generation was attempted.
    if answer is None:
        # Defensive: `execute_recurring_question` only reaches this branch
        # when generation was actually attempted (its own guard mirrors
        # these same three preconditions), so this should be unreachable --
        # but an internal invariant break must degrade honestly rather than
        # crash the spawned run task (server parity: `classify_rag_response`
        # never asserts either).
        return _degraded(
            failure_code="generation_unavailable",
            evidence_summary=_evidence_summary(
                result_count=result_count,
                answer_present=False,
                retrieval_status=retrieval_status,
                generation_status="",
            ),
        )
    confidence = {"citation_status": answer.citation_status} if answer.citation_status else {}

    if answer.status == ANSWER_STATUS_READY:
        return ExecutionOutcome(
            outcome="finding",
            title="Possible answer found",
            summary=_bounded(answer.text),
            answer=answer.text,
            answer_mode="synthesized",
            confidence=confidence,
            source_refs=source_refs,
            evidence_summary=_evidence_summary(
                result_count=result_count,
                answer_present=True,
                retrieval_status=retrieval_status,
                generation_status=answer.status,
            ),
            failure_reason=None,
        )

    if generation_mode == "required":
        return _degraded(
            failure_code="generation_required_unavailable",
            evidence_summary=_evidence_summary(
                result_count=result_count,
                answer_present=False,
                retrieval_status=retrieval_status,
                generation_status=answer.status,
            ),
        )

    # optional, and the answer abstained / had no evidence / failed: keep the
    # evidence, drop the (unusable) answer text.
    return ExecutionOutcome(
        outcome="finding",
        title="Relevant evidence found",
        summary="",
        answer=None,
        answer_mode="evidence_only",
        confidence=confidence,
        source_refs=source_refs,
        evidence_summary=_evidence_summary(
            result_count=result_count,
            answer_present=False,
            retrieval_status=retrieval_status,
            generation_status=answer.status,
        ),
        failure_reason=None,
    )


async def execute_recurring_question(app: Any, definition_row: dict) -> ExecutionOutcome:
    """Run one `recurring_question` definition: scope -> retrieval -> (maybe)
    generation -> classification.

    Composes `run_library_rag_search` and `generate_library_rag_answer`,
    both hardened not to raise for their own internal failures -- but this
    function does not itself add a blanket try/except around that
    composition. Containment for anything that still escapes (this
    function's own bug, an unhardened seam change) is the caller's job:
    `AutomationDefinitionHandler._run`'s spawned-task wrapper is the actual
    "never raises" boundary for a scheduled run.
    """
    row = definition_row if isinstance(definition_row, Mapping) else {}

    definition_input = row.get("input")
    if not isinstance(definition_input, Mapping):
        definition_input = {}
    question = _sanitize_text(definition_input.get("question"))
    if question is None:
        return _degraded(
            failure_code="question_empty",
            evidence_summary=_evidence_summary(
                result_count=0,
                answer_present=False,
                retrieval_status="",
                generation_status="",
            ),
        )

    config = row.get("config")
    if not isinstance(config, Mapping):
        config = {}
    generation_mode = config.get("generation_mode")
    if generation_mode not in _VALID_GENERATION_MODES:
        generation_mode = "optional"

    normalized_scope, _errors, _warnings = normalize_recurring_question_scope(config.get("scope"))
    source_types = engine_source_types(normalized_scope)

    top_k, high_confidence_only = _resolve_finding_policy(row.get("finding_policy"))

    retrieval_outcome = await run_library_rag_search(
        app,
        LibraryRagSearchRequest(
            query=question,
            source_types=source_types,
            mode="rag",
            top_k=top_k,
            include_citations=True,
        ),
    )
    results = retrieval_outcome.results
    if high_confidence_only:
        results = _filter_high_confidence(results)

    answer: LibraryRagAnswer | None = None
    if (
        retrieval_outcome.status not in _RETRIEVAL_DEGRADED_STATUSES
        and results
        and generation_mode != "disabled"
    ):
        target = resolve_execution_target(row)
        # target["max_tokens"] is resolved for future executors (agent_task)
        # but not passed here: `generate_library_rag_answer` computes its own
        # budget via `_effective_max_tokens` (reasoning-aware, model-keyed).
        answer = await generate_library_rag_answer(
            query=question,
            results=results,
            coverage_note="",
            provider=target["provider"],
            model=target["model"],
            chat=None,
        )

    return _classify(
        retrieval_status=retrieval_outcome.status,
        results=results,
        generation_mode=generation_mode,
        answer=answer,
    )
