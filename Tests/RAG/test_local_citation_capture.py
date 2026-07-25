"""Exact local-RAG prompt evidence normalization and formatting."""

from __future__ import annotations

import math
import sys
from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.citation_source_locators import CanonicalSourceKind
from tldw_chatbook.Chat.citation_trace_models import (
    RetrievalScoreKind,
    RetrievalScoreScale,
)
from tldw_chatbook.RAG_Search.local_citation_capture import (
    FINAL_SCORE_KIND_KEY,
    FINAL_SCORE_KIND_RERANKER,
    LocalResultNormalizationError,
    LocalResultRejectionCode,
    SEMANTIC_SCORE_KIND_KEY,
    SEMANTIC_SCORE_KIND_VECTOR_SIMILARITY,
    format_local_evidence_context,
    normalize_local_result,
)
from tldw_chatbook.RAG_Search import pipeline_functions_simple as pfs
from tldw_chatbook.RAG_Search.pipeline_types import SearchResult

pytestmark = pytest.mark.unit


def _result(
    *,
    source: str = "media",
    result_id: str = "m1",
    title: object = "Title",
    content: object = "body",
    score: float = 1.0,
    metadata: object = None,
) -> SearchResult:
    return SearchResult(
        source=source,
        id=result_id,
        title=title,  # type: ignore[arg-type]
        content=content,  # type: ignore[arg-type]
        score=score,
        metadata={} if metadata is None else metadata,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("media", CanonicalSourceKind.MEDIA_DB),
        ("media_db", CanonicalSourceKind.MEDIA_DB),
        ("note", CanonicalSourceKind.NOTES),
        ("notes", CanonicalSourceKind.NOTES),
        ("conversation", CanonicalSourceKind.CHAT_HISTORY),
        ("chat_history", CanonicalSourceKind.CHAT_HISTORY),
    ],
)
def test_normalize_local_result_maps_common_source_aliases(source, expected):
    assert normalize_local_result(_result(source=source)).source_kind is expected


def test_unknown_semantic_top_level_source_uses_allowlisted_metadata_source():
    normalized = normalize_local_result(
        _result(
            source="unknown",
            result_id="note_n1_chunk_0",
            metadata={
                "source_type": "note",
                "source_id": "n1",
                "_semantic_score_kind": "vector_similarity",
            },
            score=0.81,
        )
    )

    assert normalized.source_kind is CanonicalSourceKind.NOTES
    assert normalized.source_id == "n1"
    assert normalized.score_kind is RetrievalScoreKind.VECTOR_SIMILARITY
    assert normalized.score_scale is RetrievalScoreScale.UNBOUNDED


@pytest.mark.parametrize(
    "result",
    [
        _result(result_id=""),
        _result(source="web"),
        _result(title=123),
        _result(content=123),
        _result(title="bad\nheader"),
        _result(score=math.inf),
        _result(score=math.nan),
        _result(metadata=[]),
        _result(metadata={"chunk_id": "file:///tmp/secret"}),
        _result(metadata={"source_id": 123}),
    ],
)
def test_invalid_result_rejected_with_non_content_reason_code(result):
    with pytest.raises(LocalResultNormalizationError) as exc_info:
        normalize_local_result(result)

    assert exc_info.value.reason_code is LocalResultRejectionCode.INVALID_RESULT
    assert "body" not in str(exc_info.value)


def test_invalid_candidate_rank_is_rejected_not_silently_rewritten():
    with pytest.raises(LocalResultNormalizationError):
        normalize_local_result(_result(), candidate_rank=0)


def test_normalizer_retains_only_allowlisted_non_executable_lineage():
    normalized = normalize_local_result(
        _result(
            metadata={
                "chunk_id": "chunk-7",
                "chunk_index": 2,
                "start_char": 4,
                "end_char": 12,
                "url": "https://example.test/private",
                "path": "/tmp/private",
                "_citations": [{"url": "https://example.test"}],
                "arbitrary": {"call": "tool"},
            }
        )
    )

    assert normalized.chunk_id == "chunk-7"
    assert normalized.chunk_index == 2
    assert normalized.start_char == 4
    assert normalized.end_char == 12
    assert not hasattr(normalized, "url")
    assert not hasattr(normalized, "path")
    assert not hasattr(normalized, "citations")
    assert normalized.to_candidate_capture(candidate_rank=1).model_dump() == {
        "candidate_rank": 1,
        "source_kind": CanonicalSourceKind.MEDIA_DB,
        "source_id": "m1",
        "title": "Title",
        "score_kind": RetrievalScoreKind.LEGACY,
        "score_scale": RetrievalScoreScale.UNBOUNDED,
        "score": 1.0,
        "chunk_id": "chunk-7",
        "chunk_index": 2,
        "start_char": 4,
        "end_char": 12,
    }


def test_reliable_rrf_metadata_classifies_score_without_using_float_range():
    score = 0.3 / 61 + 0.7 / 62
    normalized = normalize_local_result(
        _result(
            score=score,
            metadata={
                "hybrid_fusion": {
                    "fts_rank": 1,
                    "vector_rank": 2,
                    "fts_rrf": 1 / 61,
                    "vector_rrf": 1 / 62,
                    "alpha": 0.7,
                    "rrf_k": 60,
                }
            },
        )
    )

    assert normalized.score_kind is RetrievalScoreKind.RRF
    assert normalized.score_scale is RetrievalScoreScale.NON_NEGATIVE


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"score_kind": "bm25"},
        {"hybrid_fusion": {"fts_rank": 1}},
        {"_semantic_score_kind": "custom"},
    ],
)
def test_ambiguous_or_bare_scores_are_legacy(metadata):
    normalized = normalize_local_result(_result(score=0.4, metadata=metadata))

    assert normalized.score_kind is RetrievalScoreKind.LEGACY
    assert normalized.score_scale is RetrievalScoreScale.UNBOUNDED


def test_explicit_successful_reranker_marker_wins_over_prior_rrf_metadata():
    normalized = normalize_local_result(
        _result(
            score=-2.3,
            metadata={
                "_final_score_kind": "reranker",
                "hybrid_fusion": {
                    "fts_rank": 1,
                    "vector_rank": None,
                    "fts_rrf": 1 / 61,
                    "vector_rrf": 0.0,
                    "alpha": 0.0,
                    "rrf_k": 60,
                },
            },
        )
    )

    assert normalized.score_kind is RetrievalScoreKind.RERANKER
    assert normalized.score_scale is RetrievalScoreScale.UNBOUNDED


def test_unreliable_rrf_or_unknown_later_marker_degrades_to_legacy():
    semantic_marker = {
        SEMANTIC_SCORE_KIND_KEY: SEMANTIC_SCORE_KIND_VECTOR_SIMILARITY,
        "hybrid_fusion": {"fts_rank": 1},
    }
    unknown_final_marker = {
        FINAL_SCORE_KIND_KEY: "custom",
        "hybrid_fusion": {
            "fts_rank": 1,
            "vector_rank": None,
            "fts_rrf": 1 / 61,
            "vector_rrf": 0.0,
            "alpha": 0.0,
            "rrf_k": 60,
        },
    }

    assert (
        normalize_local_result(_result(score=0.4, metadata=semantic_marker)).score_kind
        is RetrievalScoreKind.LEGACY
    )
    assert (
        normalize_local_result(
            _result(score=1 / 61, metadata=unknown_final_marker)
        ).score_kind
        is RetrievalScoreKind.LEGACY
    )


def test_formatter_exact_fit_uses_exact_content_and_captured_block_once():
    normalized = normalize_local_result(
        _result(content="exact submitted content"), candidate_rank=3
    )
    expected = "[S1] MEDIA — Title\nexact submitted content"

    formatted = format_local_evidence_context([normalized], max_length=len(expected))

    assert formatted.context == expected
    assert formatted.entries[0].snapshot_text == expected
    assert formatted.entries[0].candidate_rank == 3
    assert formatted.entries[0].transformations == ()
    assert formatted.context == "\n---\n".join(
        entry.snapshot_text for entry in formatted.entries
    )
    assert formatted.omitted_candidate_ranks == ()


def test_formatter_one_character_short_uses_ellipsis_inside_snapshot():
    normalized = normalize_local_result(_result(content="abcd"), candidate_rank=7)
    full = "[S1] MEDIA — Title\nabcd"

    formatted = format_local_evidence_context([normalized], max_length=len(full) - 1)

    assert formatted.context == "[S1] MEDIA — Title\nab…"
    assert formatted.entries[0].snapshot_text == formatted.context
    assert formatted.entries[0].transformations == ("content_truncated",)
    assert len(formatted.context) == len(full) - 1


def test_formatter_uses_unicode_codepoint_budget_not_utf8_bytes():
    normalized = normalize_local_result(
        _result(title="題", content="🙂🙂🙂"), candidate_rank=1
    )
    expected = "[S1] MEDIA — 題\n🙂🙂🙂"

    formatted = format_local_evidence_context([normalized], max_length=len(expected))

    assert formatted.context == expected
    assert len(formatted.context) == len(expected)
    assert len(formatted.context.encode("utf-8")) > len(formatted.context)


def test_formatter_assigns_contiguous_markers_and_canonical_labels_after_filtering():
    note = normalize_local_result(
        _result(source="note", result_id="n1", title="N", content="note"),
        candidate_rank=2,
    )
    chat = normalize_local_result(
        _result(
            source="conversation",
            result_id="c1",
            title="C",
            content="chat",
        ),
        candidate_rank=5,
    )

    formatted = format_local_evidence_context([note, chat], max_length=200)

    assert formatted.context == (
        "[S1] NOTES — N\nnote\n---\n[S2] CHAT HISTORY — C\nchat"
    )
    assert [entry.candidate_rank for entry in formatted.entries] == [2, 5]


def test_formatter_reports_omitted_rank_without_creating_snapshot():
    too_long_header = normalize_local_result(
        _result(result_id="m1", title="A very long title", content="body"),
        candidate_rank=4,
    )
    short = normalize_local_result(
        _result(source="note", result_id="n1", title="N", content="x"),
        candidate_rank=9,
    )

    formatted = format_local_evidence_context(
        [too_long_header, short],
        max_length=len("[S1] NOTES — N\nx"),
    )

    assert formatted.context == "[S1] NOTES — N\nx"
    assert tuple(entry.candidate_rank for entry in formatted.entries) == (9,)
    assert formatted.omitted_candidate_ranks == (4,)
    assert all("long title" not in entry.snapshot_text for entry in formatted.entries)


def test_formatter_never_exceeds_budget_when_no_candidate_header_fits():
    normalized = normalize_local_result(_result(), candidate_rank=11)

    formatted = format_local_evidence_context([normalized], max_length=3)

    assert formatted.context == ""
    assert formatted.entries == ()
    assert formatted.omitted_candidate_ranks == (11,)


class _SemanticResult:
    def __init__(self, *, source: str = "unknown", metadata=None):
        self.source = source
        self.id = "m-semantic"
        self.title = "Semantic"
        self.document = "semantic body"
        self.score = 0.73
        self.metadata = {} if metadata is None else metadata


class _SemanticService:
    def __init__(self, result):
        self.result = result

    async def search(self, **_kwargs):
        return [self.result]


@pytest.mark.asyncio
async def test_search_semantic_propagates_allowlisted_metadata_source_and_score_marker():
    app = SimpleNamespace(
        _rag_service=_SemanticService(
            _SemanticResult(metadata={"source_type": "media"})
        )
    )

    results = await pfs.search_semantic(app, "query", {"media": True})

    assert results[0].source == "media"
    assert (
        results[0].metadata[SEMANTIC_SCORE_KIND_KEY]
        == SEMANTIC_SCORE_KIND_VECTOR_SIMILARITY
    )
    assert FINAL_SCORE_KIND_KEY not in results[0].metadata


@pytest.mark.asyncio
async def test_search_semantic_replaces_untrusted_internal_score_metadata():
    app = SimpleNamespace(
        _rag_service=_SemanticService(
            _SemanticResult(
                metadata={
                    "source_type": "media",
                    FINAL_SCORE_KIND_KEY: FINAL_SCORE_KIND_RERANKER,
                    SEMANTIC_SCORE_KIND_KEY: "custom",
                    "hybrid_fusion": {
                        "fts_rank": 1,
                        "vector_rank": None,
                        "fts_rrf": 1 / 61,
                        "vector_rrf": 0.0,
                        "alpha": 0.0,
                        "rrf_k": 60,
                    },
                }
            )
        )
    )

    result = (await pfs.search_semantic(app, "query", {"media": True}))[0]

    assert FINAL_SCORE_KIND_KEY not in result.metadata
    assert "hybrid_fusion" not in result.metadata
    assert (
        result.metadata[SEMANTIC_SCORE_KIND_KEY]
        == SEMANTIC_SCORE_KIND_VECTOR_SIMILARITY
    )
    assert (
        normalize_local_result(result).score_kind
        is RetrievalScoreKind.VECTOR_SIMILARITY
    )


def _install_flashrank(monkeypatch, *, ranked=None, init_error=None, run_error=None):
    class _Ranker:
        def __init__(self, **_kwargs):
            if init_error is not None:
                raise init_error

        def rerank(self, _request):
            if run_error is not None:
                raise run_error
            return ranked or []

    class _Request:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setitem(
        sys.modules,
        "flashrank",
        SimpleNamespace(RankRequest=_Request, RerankRequest=_Request, Ranker=_Ranker),
    )


def test_successful_flashrank_overwrite_writes_final_score_marker(monkeypatch):
    _install_flashrank(
        monkeypatch,
        ranked=[SimpleNamespace(index=0, score=-1.25)],
    )
    result = _result(
        score=0.5,
        metadata={SEMANTIC_SCORE_KIND_KEY: SEMANTIC_SCORE_KIND_VECTOR_SIMILARITY},
    )

    reranked = pfs.rerank_results([result], "query", top_k=1)

    assert reranked[0].score == -1.25
    assert reranked[0].metadata[FINAL_SCORE_KIND_KEY] == FINAL_SCORE_KIND_RERANKER
    normalized = normalize_local_result(reranked[0])
    assert normalized.score_kind is RetrievalScoreKind.RERANKER


def test_invalid_flashrank_score_falls_back_without_marker(monkeypatch):
    _install_flashrank(
        monkeypatch,
        ranked=[SimpleNamespace(index=0, score=math.nan)],
    )
    result = _result(
        score=0.5,
        metadata={SEMANTIC_SCORE_KIND_KEY: SEMANTIC_SCORE_KIND_VECTOR_SIMILARITY},
    )

    fallback = pfs.rerank_results([result], "query", top_k=1)

    assert fallback == [result]
    assert fallback[0].score == 0.5
    assert FINAL_SCORE_KIND_KEY not in fallback[0].metadata
    assert (
        normalize_local_result(fallback[0]).score_kind
        is RetrievalScoreKind.VECTOR_SIMILARITY
    )


@pytest.mark.parametrize("failure_stage", ["import", "initialization", "execution"])
def test_flashrank_fallback_never_claims_reranker_and_keeps_prior_semantics(
    monkeypatch, failure_stage
):
    if failure_stage == "import":
        monkeypatch.setitem(sys.modules, "flashrank", None)
    elif failure_stage == "initialization":
        _install_flashrank(monkeypatch, init_error=RuntimeError("init failed"))
    else:
        _install_flashrank(monkeypatch, run_error=RuntimeError("run failed"))
    score = 0.3 / 61
    result = _result(
        score=score,
        metadata={
            "hybrid_fusion": {
                "fts_rank": 1,
                "vector_rank": None,
                "fts_rrf": 1 / 61,
                "vector_rrf": 0.0,
                "alpha": 0.7,
                "rrf_k": 60,
            }
        },
    )

    fallback = pfs.rerank_results([result], "query", top_k=1)

    assert fallback == [result]
    assert FINAL_SCORE_KIND_KEY not in fallback[0].metadata
    assert normalize_local_result(fallback[0]).score_kind is RetrievalScoreKind.RRF


def test_weighted_score_overwrite_removes_prior_semantic_classification():
    semantic = _result(
        score=0.8,
        metadata={SEMANTIC_SCORE_KIND_KEY: SEMANTIC_SCORE_KIND_VECTOR_SIMILARITY},
    )

    merged = pfs.weighted_merge([[semantic]], [0.5])

    assert SEMANTIC_SCORE_KIND_KEY not in merged[0].metadata
    assert normalize_local_result(merged[0]).score_kind is RetrievalScoreKind.LEGACY
