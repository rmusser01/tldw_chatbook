"""Config knobs for hybrid fusion: rrf_k and hybrid_pool_multiplier (TASK-4110 T3).

Two previously-hard-coded fusion parameters become config knobs so Task 4's
strategy sweep can flip them per pass:

* ``config.search.rrf_k`` -- the RRF constant. Was ``DEFAULT_RRF_K`` baked
  into ``RAGService._fuse_hybrid_results``'s call to
  ``reciprocal_rank_fusion`` and into the ``hybrid_fusion`` metadata block,
  even though ``fusion.resolve_rrf_k`` already existed to validate a
  config-sourced value (mirroring ``resolve_hybrid_alpha``'s use-time
  pattern) -- it was simply never called from the service.
* ``config.search.hybrid_pool_multiplier`` -- how many candidates each
  hybrid leg (semantic + keyword) over-fetches before fusion narrows back to
  ``top_k``. Was the module-level ``SEARCH_RESULT_MULTIPLIER`` constant,
  shared with ``_semantic_search``'s OWN internal over-fetch multiplier for
  its raw vector-store call. This widens the hybrid legs only --
  ``_semantic_search``'s internal multiplier (used on both the hybrid and
  the direct semantic-mode paths) is untouched.

THE METADATA-HONESTY PIN (test 2): a fused row's ``hybrid_fusion`` metadata
must record the ACTUAL ``rrf_k`` used, not a literal -- because
``local_citation_capture._reliable_rrf`` re-derives the fused score from
those recorded values (``1 / (rrf_k + rank)``) to certify a hybrid row as
``RetrievalScoreKind.RRF`` rather than degrading it to ``LEGACY``. A wrong
recorded k breaks that re-derivation silently: the row is arithmetically
"fine" but no longer provably RRF to that consumer. Exercised at a
NON-DEFAULT k (10) so a hard-coded 60 anywhere on the path cannot pass.
"""
import asyncio

import pytest
from loguru import logger

from tldw_chatbook.RAG_Search.fusion import DEFAULT_RRF_K
from tldw_chatbook.RAG_Search.simplified.config import RAGConfig
from tldw_chatbook.RAG_Search.simplified.rag_service import (
    RAGService,
    SEARCH_RESULT_MULTIPLIER,
)
from tldw_chatbook.RAG_Search.simplified.vector_store import SearchResult

pytestmark = pytest.mark.unit


def _result(rid: str, score: float) -> SearchResult:
    return SearchResult(id=rid, score=score, document=f"doc {rid}", metadata={})


def _make_service(**search_overrides) -> RAGService:
    """A RAGService with the in-memory vector store and mock embeddings.

    Mirrors ``Tests/RAG_Search/test_keyword_leg_pushdown.py``'s
    ``_make_service`` -- no real DB, no real embedding model, safe for a
    unit-speed test.
    """
    cfg = RAGConfig()
    cfg.embedding.model = "mock"
    cfg.embedding.device = "cpu"
    cfg.vector_store.type = "memory"
    cfg.vector_store.persist_directory = None
    cfg.search.enable_cache = False
    for key, value in search_overrides.items():
        setattr(cfg.search, key, value)
    return RAGService(cfg)


@pytest.fixture
def warnings_captured():
    """Collect loguru WARNING+ records (pytest's capsys never sees loguru)."""
    messages = []
    sink_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        yield messages
    finally:
        logger.remove(sink_id)


# --- Test 1: config.search.rrf_k reaches the real fusion call --------------


def test_rrf_k_config_reaches_the_fusion_call(monkeypatch):
    """The config knob, not a hand-passed kwarg, drives the real wiring.

    Goes through the actual ``_hybrid_search`` seam (not a direct
    ``_fuse_hybrid_results`` call) so this proves ``self.config.search.rrf_k``
    is read there, not merely that the static method accepts a new
    parameter.
    """
    service = _make_service(rrf_k=10)

    keyword = [_result("m1", 0.5), _result("m2", 0.4)]  # m2 fts_rank=2
    semantic = [_result("m2", 0.9), _result("m3", 0.3)]  # m2 vector_rank=1

    async def fake_semantic_search(*args, **kwargs):
        return semantic

    async def fake_keyword_search(*args, **kwargs):
        return keyword

    monkeypatch.setattr(service, "_semantic_search", fake_semantic_search)
    monkeypatch.setattr(service, "_keyword_search", fake_keyword_search)

    fused = asyncio.run(service._hybrid_search("q", top_k=10, include_citations=False))
    row = {r.id: r for r in fused}["m2"]
    fusion_meta = row.metadata["hybrid_fusion"]

    assert fusion_meta["rrf_k"] == 10
    expected_fts_rrf = 1.0 / (10 + 2)
    expected_vector_rrf = 1.0 / (10 + 1)
    assert fusion_meta["fts_rrf"] == pytest.approx(expected_fts_rrf)
    assert fusion_meta["vector_rrf"] == pytest.approx(expected_vector_rrf)
    # Not the 1/(60+rank) terms a hard-coded DEFAULT_RRF_K would produce.
    assert fusion_meta["fts_rrf"] != pytest.approx(1.0 / (DEFAULT_RRF_K + 2))
    alpha = fusion_meta["alpha"]
    expected_score = (1 - alpha) * expected_fts_rrf + alpha * expected_vector_rrf
    assert row.score == pytest.approx(expected_score)


# --- Test 2: THE METADATA-HONESTY PIN ---------------------------------------


def test_metadata_records_actual_values_and_rederivation_certifies():
    """A non-default rrf_k must still certify as RRF downstream.

    Mirrors ``Tests/RAG_Search/test_hybrid_fusion_metadata.py``'s
    ``test_real_fused_metadata_still_classifies_as_rrf_downstream`` but at
    rrf_k=10 instead of the default 60, so a recorded-but-wrong k (e.g. the
    metadata block still writing the module ``DEFAULT_RRF_K`` literal while
    the math actually used 10) fails the re-derivation and is caught here,
    not just at the one k value that happens to equal the default.
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
        rrf_k=10,
        include_citations=False,
    )
    row = {r.id: r for r in fused}["m2"]
    assert row.metadata["hybrid_fusion"]["rrf_k"] == 10

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


# --- Test 3: hybrid_pool_multiplier widens the HYBRID legs only ------------


def test_pool_multiplier_widens_hybrid_legs_only(monkeypatch):
    """The multiplier widens ``_hybrid_search``'s two leg fetches only.

    Direct semantic-mode search (``search_type="semantic"``, never touching
    ``_hybrid_search``) must keep using the module ``SEARCH_RESULT_MULTIPLIER``
    for its own internal vector-store over-fetch, unaffected by a
    hybrid-only config knob.
    """
    service = _make_service(hybrid_pool_multiplier=5)

    leg_calls = {}

    async def fake_semantic_search(query, top_k, *args, **kwargs):
        leg_calls["semantic"] = top_k
        return []

    async def fake_keyword_search(query, top_k, *args, **kwargs):
        leg_calls["keyword"] = top_k
        return []

    monkeypatch.setattr(service, "_semantic_search", fake_semantic_search)
    monkeypatch.setattr(service, "_keyword_search", fake_keyword_search)

    asyncio.run(service._hybrid_search("q", top_k=4, include_citations=False))

    assert leg_calls == {"semantic": 20, "keyword": 20}, (
        f"hybrid legs were not widened by the configured multiplier: {leg_calls}"
    )

    # Now the direct semantic path: _semantic_search's OWN multiplier
    # (module SEARCH_RESULT_MULTIPLIER, still 2) must be untouched by the
    # hybrid_pool_multiplier=5 set above. A FRESH service/monkeypatch is used
    # so the `_semantic_search` fake installed above (which returns `[]`
    # without ever reaching the vector store) cannot mask this half of the
    # test -- `service.search(search_type="semantic")` would otherwise call
    # straight into that fake and leave `store_calls` empty regardless of
    # whether the production code is correct.
    direct_service = _make_service(hybrid_pool_multiplier=5)
    store_calls = []

    def fake_search_with_citations(embedding, query, top_k, *args, **kwargs):
        store_calls.append(top_k)
        return []

    def fake_search(embedding, top_k, *args, **kwargs):
        store_calls.append(top_k)
        return []

    monkeypatch.setattr(
        direct_service.vector_store,
        "search_with_citations",
        fake_search_with_citations,
    )
    monkeypatch.setattr(direct_service.vector_store, "search", fake_search)

    asyncio.run(
        direct_service.search(
            "q", top_k=4, search_type="semantic", include_citations=True
        )
    )

    assert store_calls == [4 * SEARCH_RESULT_MULTIPLIER] == [8]


# --- Test 4: invalid rrf_k falls back to the default, with a warning -------


def test_invalid_rrf_k_falls_back_to_default_with_warning(warnings_captured):
    fused = RAGService._fuse_hybrid_results(
        keyword_results=[_result("m1", 0.5)],
        semantic_results=[],
        top_k=10,
        alpha=0.7,
        rrf_k=-5,
        include_citations=False,
    )
    row = fused[0]
    assert row.metadata["hybrid_fusion"]["rrf_k"] == DEFAULT_RRF_K

    assert any("rrf_k" in message for message in warnings_captured), (
        "an invalid rrf_k must leave a warning trace"
    )


# --- Test 5: defaults unchanged (protected-oracle insurance) ---------------


def test_defaults_unchanged(monkeypatch):
    """A fresh config's fused-row arithmetic is byte-identical to pre-branch.

    Both new fields must default to the values that were previously
    hard-coded (``rrf_k=60`` == the old ``DEFAULT_RRF_K`` literal;
    ``hybrid_pool_multiplier=2`` == the old shared ``SEARCH_RESULT_MULTIPLIER``),
    so a caller that never touches either knob sees no behavior change at
    all -- not in the fused score, not in how many candidates each leg
    fetches.
    """
    cfg = RAGConfig()
    assert cfg.search.rrf_k == 60 == DEFAULT_RRF_K
    assert cfg.search.hybrid_pool_multiplier == 2 == SEARCH_RESULT_MULTIPLIER

    # Fusion arithmetic: identical to the pre-branch hard-coded DEFAULT_RRF_K.
    fused = RAGService._fuse_hybrid_results(
        keyword_results=[_result("m1", 0.001), _result("m2", 0.001)],
        semantic_results=[_result("m2", 0.83), _result("m3", 0.41)],
        top_k=10,
        alpha=0.7,
        include_citations=False,
    )
    row = {r.id: r for r in fused}["m2"]
    assert row.metadata["hybrid_fusion"]["rrf_k"] == 60
    expected_fts_rrf = 1.0 / (60 + 2)
    expected_vector_rrf = 1.0 / (60 + 1)
    assert row.metadata["hybrid_fusion"]["fts_rrf"] == pytest.approx(expected_fts_rrf)
    assert row.metadata["hybrid_fusion"]["vector_rrf"] == pytest.approx(
        expected_vector_rrf
    )
    expected_score = 0.3 * expected_fts_rrf + 0.7 * expected_vector_rrf
    assert row.score == pytest.approx(expected_score)

    # Leg over-fetch: an untouched-config hybrid search asks for exactly what
    # SEARCH_RESULT_MULTIPLIER always asked for.
    service = _make_service()
    leg_calls = {}

    async def fake_semantic_search(query, top_k, *args, **kwargs):
        leg_calls["semantic"] = top_k
        return []

    async def fake_keyword_search(query, top_k, *args, **kwargs):
        leg_calls["keyword"] = top_k
        return []

    monkeypatch.setattr(service, "_semantic_search", fake_semantic_search)
    monkeypatch.setattr(service, "_keyword_search", fake_keyword_search)

    asyncio.run(service._hybrid_search("q", top_k=7, include_citations=False))

    assert leg_calls == {
        "semantic": 7 * SEARCH_RESULT_MULTIPLIER,
        "keyword": 7 * SEARCH_RESULT_MULTIPLIER,
    }
