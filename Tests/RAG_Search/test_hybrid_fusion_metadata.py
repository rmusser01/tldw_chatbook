"""Fusion must preserve original per-leg scores for score-kind-aware display.

RRF-fused scores max out at ~1/(rrf_k+1) ~= 0.016 — far below the UI's
similarity band thresholds — so the vector leg's original similarity is the
only honest banding input for hybrid rows (spec Workstream A item 3).
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
