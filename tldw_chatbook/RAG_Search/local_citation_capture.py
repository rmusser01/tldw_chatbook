"""Strict normalization and exact formatting for local RAG prompt evidence."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from collections.abc import Mapping, Sequence
from typing import Any
import unicodedata

from ..Chat.citation_source_locators import CanonicalSourceKind
from ..Chat.citation_trace_builder import (
    LocalPromptEvidenceCapture,
    LocalRetrievalCandidateCapture,
)
from ..Chat.citation_trace_models import (
    EVIDENCE_ENTRIES_PER_PROMPT_MAX,
    RetrievalScoreKind,
    RetrievalScoreScale,
    SNAPSHOT_TEXT_UTF8_BYTES_MAX,
)

FINAL_SCORE_KIND_KEY = "_final_score_kind"
FINAL_SCORE_KIND_RERANKER = "reranker"
SEMANTIC_SCORE_KIND_KEY = "_semantic_score_kind"
SEMANTIC_SCORE_KIND_VECTOR_SIMILARITY = "vector_similarity"

_SOURCE_ALIASES = {
    "media": CanonicalSourceKind.MEDIA_DB,
    "media_db": CanonicalSourceKind.MEDIA_DB,
    "media-db": CanonicalSourceKind.MEDIA_DB,
    "note": CanonicalSourceKind.NOTES,
    "notes": CanonicalSourceKind.NOTES,
    "conversation": CanonicalSourceKind.CHAT_HISTORY,
    "conversations": CanonicalSourceKind.CHAT_HISTORY,
    "chat": CanonicalSourceKind.CHAT_HISTORY,
    "chat_history": CanonicalSourceKind.CHAT_HISTORY,
    "chat-history": CanonicalSourceKind.CHAT_HISTORY,
}
_METADATA_SOURCE_KEYS = ("source_type", "source_kind")
_DIRECT_LINEAGE_KEYS = ("chunk_id", "chunk_index")
_OFFSET_ALIASES = {
    "start_char": "chunk_start",
    "end_char": "chunk_end",
}
_TITLE_UTF8_BYTES_MAX = 4 * 1024
_SEPARATOR = "\n---\n"
_SOURCE_LABELS = {
    CanonicalSourceKind.MEDIA_DB: "MEDIA",
    CanonicalSourceKind.NOTES: "NOTES",
    CanonicalSourceKind.CHAT_HISTORY: "CHAT HISTORY",
}


class LocalResultRejectionCode(str, Enum):
    """Stable non-content reason codes for rejected local candidates."""

    INVALID_RESULT = "invalid_local_result"


class LocalResultNormalizationError(ValueError):
    """A local candidate was unsafe or incompatible with canonical capture."""

    def __init__(self, reason_code: LocalResultRejectionCode) -> None:
        self.reason_code = reason_code
        super().__init__(reason_code.value)


@dataclass(frozen=True)
class NormalizedLocalResult:
    """Allowlisted local result ready for canonical builder capture."""

    source_kind: CanonicalSourceKind
    source_id: str
    title: str
    content: str
    score_kind: RetrievalScoreKind
    score_scale: RetrievalScoreScale
    score: float
    chunk_id: str | None = None
    chunk_index: int | None = None
    start_char: int | None = None
    end_char: int | None = None
    candidate_rank: int | None = None

    def to_candidate_capture(
        self, *, candidate_rank: int | None = None
    ) -> LocalRetrievalCandidateCapture:
        """Convert to the canonical retrieval-candidate builder contract.

        Args:
            candidate_rank: Optional 1-based rank override.

        Returns:
            The canonical builder candidate capture.
        """

        rank = candidate_rank if candidate_rank is not None else self.candidate_rank
        if rank is None:
            raise ValueError("candidate_rank is required")
        return LocalRetrievalCandidateCapture(
            candidate_rank=rank,
            source_kind=self.source_kind,
            source_id=self.source_id,
            title=self.title,
            score_kind=self.score_kind,
            score_scale=self.score_scale,
            score=self.score,
            chunk_id=self.chunk_id,
            chunk_index=self.chunk_index,
            start_char=self.start_char,
            end_char=self.end_char,
        )


@dataclass(frozen=True)
class LocalEvidenceContext:
    """Exact context and canonical per-entry prompt capture blocks."""

    context: str
    entries: tuple[LocalPromptEvidenceCapture, ...]
    omitted_candidate_ranks: tuple[int, ...]


def _reject() -> None:
    raise LocalResultNormalizationError(LocalResultRejectionCode.INVALID_RESULT)


def _canonical_source(value: Any) -> CanonicalSourceKind | None:
    if not isinstance(value, str):
        return None
    return _SOURCE_ALIASES.get(value.strip().lower())


def _result_value(result: Any, field: str) -> Any:
    if isinstance(result, Mapping):
        return result.get(field)
    return getattr(result, field, None)


def _resolve_source(result: Any, metadata: Mapping[str, Any]) -> CanonicalSourceKind:
    source = _result_value(result, "source")
    if not isinstance(source, str):
        _reject()
    resolved: list[CanonicalSourceKind] = []
    canonical = _canonical_source(source)
    if canonical is not None:
        resolved.append(canonical)
    elif source.strip().lower() not in {"", "unknown"}:
        _reject()
    for key in _METADATA_SOURCE_KEYS:
        if key not in metadata:
            continue
        canonical = _canonical_source(metadata[key])
        if canonical is None:
            _reject()
        resolved.append(canonical)
    if not resolved or any(value is not resolved[0] for value in resolved[1:]):
        _reject()
    return resolved[0]


def _resolve_lineage(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve allowlisted lineage and strict semantic offset aliases."""

    lineage = {key: metadata.get(key) for key in _DIRECT_LINEAGE_KEYS}
    for canonical_key, producer_key in _OFFSET_ALIASES.items():
        canonical_present = canonical_key in metadata
        producer_present = producer_key in metadata
        if (
            canonical_present
            and producer_present
            and metadata[canonical_key] != metadata[producer_key]
        ):
            _reject()
        if canonical_present:
            lineage[canonical_key] = metadata[canonical_key]
        elif producer_present:
            lineage[canonical_key] = metadata[producer_key]
        else:
            lineage[canonical_key] = None
    return lineage


def _reliable_rrf(metadata: Mapping[str, Any], score: float) -> bool:
    fusion = metadata.get("hybrid_fusion")
    if not isinstance(fusion, Mapping):
        return False
    required = {
        "fts_rank",
        "vector_rank",
        "fts_rrf",
        "vector_rrf",
        "alpha",
        "rrf_k",
    }
    if set(fusion) != required:
        return False
    alpha = fusion["alpha"]
    rrf_k = fusion["rrf_k"]
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not math.isfinite(float(alpha))
        or not 0 <= float(alpha) <= 1
        or isinstance(rrf_k, bool)
        or not isinstance(rrf_k, int)
        or rrf_k < 0
    ):
        return False

    expected_contributions: list[float] = []
    for rank_key, contribution_key in (
        ("fts_rank", "fts_rrf"),
        ("vector_rank", "vector_rrf"),
    ):
        rank = fusion[rank_key]
        contribution = fusion[contribution_key]
        if (
            isinstance(contribution, bool)
            or not isinstance(contribution, (int, float))
            or not math.isfinite(float(contribution))
        ):
            return False
        if rank is None:
            expected = 0.0
        elif isinstance(rank, int) and not isinstance(rank, bool) and rank >= 1:
            expected = 1.0 / (rrf_k + rank)
        else:
            return False
        if not math.isclose(
            float(contribution), expected, rel_tol=1e-12, abs_tol=1e-15
        ):
            return False
        expected_contributions.append(expected)
    if not any(fusion[key] is not None for key in ("fts_rank", "vector_rank")):
        return False
    expected_score = (1 - float(alpha)) * expected_contributions[0] + float(
        alpha
    ) * expected_contributions[1]
    return math.isclose(score, expected_score, rel_tol=1e-12, abs_tol=1e-15)


def _score_semantics(
    metadata: Mapping[str, Any], score: float
) -> tuple[RetrievalScoreKind, RetrievalScoreScale]:
    if FINAL_SCORE_KIND_KEY in metadata:
        if metadata[FINAL_SCORE_KIND_KEY] == FINAL_SCORE_KIND_RERANKER:
            return RetrievalScoreKind.RERANKER, RetrievalScoreScale.UNBOUNDED
        return RetrievalScoreKind.LEGACY, RetrievalScoreScale.UNBOUNDED
    if "hybrid_fusion" in metadata:
        if _reliable_rrf(metadata, score):
            return RetrievalScoreKind.RRF, RetrievalScoreScale.NON_NEGATIVE
        return RetrievalScoreKind.LEGACY, RetrievalScoreScale.UNBOUNDED
    if metadata.get(SEMANTIC_SCORE_KIND_KEY) == SEMANTIC_SCORE_KIND_VECTOR_SIMILARITY:
        return RetrievalScoreKind.VECTOR_SIMILARITY, RetrievalScoreScale.UNBOUNDED
    return RetrievalScoreKind.LEGACY, RetrievalScoreScale.UNBOUNDED


def normalize_local_result(
    result: Any, *, candidate_rank: int | None = None
) -> NormalizedLocalResult:
    """Normalize one producer result into the strict local citation allowlist.

    Args:
        result: Search-result object or mapping from a local retrieval producer.
        candidate_rank: Optional 1-based rank in the retrieval run.

    Returns:
        Strict canonical identity, exact content, score semantics, and lineage.

    Raises:
        LocalResultNormalizationError: If any governed field is invalid.
    """

    metadata = _result_value(result, "metadata")
    if not isinstance(metadata, Mapping):
        _reject()
    source_kind = _resolve_source(result, metadata)

    producer_result_id = _result_value(result, "id")
    source_id = metadata.get("source_id", producer_result_id)
    title = _result_value(result, "title")
    content = _result_value(result, "content")
    score = _result_value(result, "score")
    if (
        not isinstance(producer_result_id, str)
        or not producer_result_id
        or not isinstance(source_id, str)
        or not source_id
        or not isinstance(title, str)
        or not title
        or not isinstance(content, str)
        or isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
        or len(title.encode("utf-8")) > _TITLE_UTF8_BYTES_MAX
        or any(unicodedata.category(character).startswith("C") for character in title)
    ):
        _reject()

    lineage = _resolve_lineage(metadata)
    if (
        "source_id" in metadata
        and lineage["chunk_id"] is None
        and producer_result_id != source_id
    ):
        lineage["chunk_id"] = producer_result_id
    score_kind, score_scale = _score_semantics(metadata, float(score))
    try:
        candidate = LocalRetrievalCandidateCapture(
            candidate_rank=candidate_rank if candidate_rank is not None else 1,
            source_kind=source_kind,
            source_id=source_id,
            title=title,
            score_kind=score_kind,
            score_scale=score_scale,
            score=float(score),
            **lineage,
        )
    except (TypeError, ValueError):
        _reject()

    return NormalizedLocalResult(
        source_kind=candidate.source_kind,
        source_id=candidate.source_id,
        title=candidate.title,
        content=content,
        score_kind=candidate.score_kind or RetrievalScoreKind.LEGACY,
        score_scale=candidate.score_scale or RetrievalScoreScale.UNBOUNDED,
        score=candidate.score if candidate.score is not None else float(score),
        chunk_id=candidate.chunk_id,
        chunk_index=candidate.chunk_index,
        start_char=candidate.start_char,
        end_char=candidate.end_char,
        candidate_rank=candidate_rank,
    )


def _content_prefix_within_budgets(
    content: str, *, character_budget: int, utf8_byte_budget: int
) -> str:
    """Return the longest whole-codepoint prefix within both budgets."""

    prefix: list[str] = []
    byte_count = 0
    for character in content:
        if len(prefix) >= character_budget:
            break
        character_bytes = len(character.encode("utf-8"))
        if byte_count + character_bytes > utf8_byte_budget:
            break
        prefix.append(character)
        byte_count += character_bytes
    return "".join(prefix)


def format_local_evidence_context(
    normalized_results: Sequence[NormalizedLocalResult], max_length: int = 90
) -> LocalEvidenceContext:
    """Format canonical marked evidence without reparsing aggregate context.

    Args:
        normalized_results: Authorized normalized results in retrieval order.
        max_length: Maximum Unicode codepoints in the aggregate context.

    Returns:
        Exact aggregate context, per-entry builder captures, and omitted ranks.
    """
    if isinstance(max_length, bool) or not isinstance(max_length, int):
        raise TypeError("max_length must be an integer")
    if max_length < 0:
        raise ValueError("max_length must be non-negative")

    blocks: list[str] = []
    entries: list[LocalPromptEvidenceCapture] = []
    omitted: list[int] = []
    context_length = 0
    for position, result in enumerate(normalized_results, start=1):
        if not isinstance(result, NormalizedLocalResult):
            raise TypeError("normalized_results must contain NormalizedLocalResult")
        candidate_rank = result.candidate_rank or position
        if len(entries) >= EVIDENCE_ENTRIES_PER_PROMPT_MAX:
            omitted.append(candidate_rank)
            continue
        ordinal = len(entries) + 1
        label = _SOURCE_LABELS[result.source_kind]
        header = f"[S{ordinal}] {label} — {result.title}\n"
        separator_size = len(_SEPARATOR) if blocks else 0
        available_characters = max_length - context_length - separator_size
        full_block = f"{header}{result.content}"
        transformations: tuple[str, ...] = ()
        if (
            len(full_block) <= available_characters
            and len(full_block.encode("utf-8")) <= SNAPSHOT_TEXT_UTF8_BYTES_MAX
        ):
            block = full_block
        else:
            ellipsis = "…"
            content_character_budget = (
                available_characters - len(header) - len(ellipsis)
            )
            content_byte_budget = (
                SNAPSHOT_TEXT_UTF8_BYTES_MAX
                - len(header.encode("utf-8"))
                - len(ellipsis.encode("utf-8"))
            )
            if content_character_budget < 0 or content_byte_budget < 0:
                omitted.append(candidate_rank)
                continue
            content = _content_prefix_within_budgets(
                result.content,
                character_budget=content_character_budget,
                utf8_byte_budget=content_byte_budget,
            )
            block = f"{header}{content}{ellipsis}"
            transformations = ("content_truncated",)
        blocks.append(block)
        context_length += separator_size + len(block)
        entries.append(
            LocalPromptEvidenceCapture(
                candidate_rank=candidate_rank,
                snapshot_text=block,
                transformations=transformations,
            )
        )

    context = _SEPARATOR.join(blocks)
    return LocalEvidenceContext(
        context=context,
        entries=tuple(entries),
        omitted_candidate_ranks=tuple(omitted),
    )


__all__ = [
    "FINAL_SCORE_KIND_KEY",
    "FINAL_SCORE_KIND_RERANKER",
    "LocalEvidenceContext",
    "LocalResultNormalizationError",
    "LocalResultRejectionCode",
    "NormalizedLocalResult",
    "SEMANTIC_SCORE_KIND_KEY",
    "SEMANTIC_SCORE_KIND_VECTOR_SIMILARITY",
    "format_local_evidence_context",
    "normalize_local_result",
]
