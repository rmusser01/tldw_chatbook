"""Fusion must preserve original per-leg scores for score-kind-aware display.

RRF-fused scores max out at ~1/(rrf_k+1) — ~0.17 at the shipped
``rrf_k = 5`` (TASK-4110) and ~0.016 at the previous 60. Either way the
ceiling sits below the UI's similarity band thresholds and moves whenever k
is retuned, so the vector leg's original similarity is the only honest
banding input for hybrid rows (spec Workstream A item 3).
"""
import pytest
from tldw_chatbook.RAG_Search.simplified.rag_service import RAGService
from tldw_chatbook.RAG_Search.simplified.vector_store import SearchResult


def _result(rid: str, score: float) -> SearchResult:
    return SearchResult(id=rid, score=score, document=f"doc {rid}", metadata={})


def test_fused_rows_preserve_original_leg_scores():
    keyword = [_result("m1", 0.001), _result("m2", 0.001)]
    semantic = [_result("m2", 0.83), _result("m3", 0.41)]
    fused = RAGService._fuse_hybrid_results(
        keyword_results=keyword, semantic_results=semantic,
        top_k=10, alpha=0.7, include_citations=False,
    )
    by_id = {r.id: r for r in fused}
    both = by_id["m2"].metadata["hybrid_fusion"]
    assert both["vector_score"] == pytest.approx(0.83)
    assert both["fts_score"] == pytest.approx(0.001)
    fts_only = by_id["m1"].metadata["hybrid_fusion"]
    assert fts_only["vector_score"] is None
    vec_only = by_id["m3"].metadata["hybrid_fusion"]
    assert vec_only["vector_score"] == pytest.approx(0.41)
    assert vec_only["fts_score"] is None


def test_real_fused_metadata_still_classifies_as_rrf_downstream():
    """The producer's block and its one strict consumer must not drift apart.

    `local_citation_capture._reliable_rrf` certifies a fused score as
    `RetrievalScoreKind.RRF` / `NON_NEGATIVE` only after re-deriving it from
    the fusion block. It gated on an EXACT six-key set, so adding
    `fts_score`/`vector_score` above silently downgraded every real hybrid
    row to `LEGACY` / `UNBOUNDED` in the Console citation trace -- invisible
    to that module's own tests, which hand-build their fusion metadata
    instead of piping the producer's output through it.

    This pins the two ends together: the metadata block and the score here
    come verbatim from `_fuse_hybrid_results`, never from a literal.
    """
    from tldw_chatbook.Chat.citation_trace_models import (
        RetrievalScoreKind,
        RetrievalScoreScale,
    )
    from tldw_chatbook.RAG_Search.local_citation_capture import normalize_local_result
    from tldw_chatbook.RAG_Search.pipeline_types import (
        SearchResult as CaptureSearchResult,
    )

    fused = RAGService._fuse_hybrid_results(
        keyword_results=[_result("m1", 0.001), _result("m2", 0.001)],
        semantic_results=[_result("m2", 0.83), _result("m3", 0.41)],
        top_k=10,
        alpha=0.7,
        include_citations=False,
    )
    row = {r.id: r for r in fused}["m2"]

    normalized = normalize_local_result(
        CaptureSearchResult(
            source="media",
            id=row.id,
            title="Title",
            content=row.document,
            score=row.score,
            metadata=row.metadata,
        )
    )

    assert normalized.score_kind is RetrievalScoreKind.RRF
    assert normalized.score_scale is RetrievalScoreScale.NON_NEGATIVE
