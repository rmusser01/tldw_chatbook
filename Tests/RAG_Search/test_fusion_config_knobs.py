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
from tldw_chatbook.RAG_Search.simplified.config import DEFAULT_HYBRID_RRF_K, RAGConfig
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
    """ORACLE UPDATE (Task 5, review round 2), disclosed: the fallback moves
    ``DEFAULT_RRF_K`` (60) -> ``DEFAULT_HYBRID_RRF_K`` (5).

    ``_fuse_hybrid_results`` sanitizes through ``resolve_rrf_k``, and every
    fallback in that APP-CONFIG resolver is now the shipped default -- a
    ``config.search.rrf_k`` polluted to a negative number must degrade to the
    weighting the app ships, not to the server-parity constant TASK-4110
    measured away from. What this test is really for is unchanged: an invalid
    k must not reach the fusion math (a negative k divides by zero at rank 1)
    and must leave a trace.
    """
    fused = RAGService._fuse_hybrid_results(
        keyword_results=[_result("m1", 0.5)],
        semantic_results=[],
        top_k=10,
        alpha=0.7,
        rrf_k=-5,
        include_citations=False,
    )
    row = fused[0]
    assert row.metadata["hybrid_fusion"]["rrf_k"] == DEFAULT_HYBRID_RRF_K

    assert any("rrf_k" in message for message in warnings_captured), (
        "an invalid rrf_k must leave a warning trace"
    )


# --- Test 5: the SHIPPED defaults (oracle updated in Task 5) ---------------


def test_shipped_defaults(monkeypatch):
    """What a fresh config actually ships (TASK-4110 Task 5 changed one).

    ORACLE UPDATE, disclosed: this test was ``test_defaults_unchanged`` and
    pinned ``rrf_k == 60 == DEFAULT_RRF_K`` -- the value T3 preserved so that
    config-threading alone changed no behavior. Task 5's measurement then
    moved the SHIPPED default to ``DEFAULT_HYBRID_RRF_K`` (5); at 60 an
    FTS-only row could never enter hybrid's fused top-k over a ~20-row
    candidate window (`Tests/RAG_Search/test_fusion_rescue_pin.py`). So the
    pin moves with it, deliberately, rather than being deleted.

    ``fusion.DEFAULT_RRF_K`` stays 60 and is still asserted here as the
    no-config fallback: a caller of ``_fuse_hybrid_results`` that predates
    the ``rrf_k`` parameter is unaffected by the shipped-profile change.
    ``hybrid_pool_multiplier`` is genuinely unchanged (2) -- the pool lever
    was measured and declined.
    """
    cfg = RAGConfig()
    assert cfg.search.rrf_k == 5 == DEFAULT_HYBRID_RRF_K
    assert DEFAULT_RRF_K == 60, "the server-parity fallback must not move"
    assert cfg.search.hybrid_pool_multiplier == 2 == SEARCH_RESULT_MULTIPLIER

    # The service reads the config knob, so a default-config hybrid search
    # fuses at 5.
    service_default_k = _make_service().config.search.rrf_k
    assert service_default_k == DEFAULT_HYBRID_RRF_K

    # The static method's own signature default is the no-config fallback and
    # is still DEFAULT_RRF_K -- unchanged arithmetic for a caller that passes
    # no k at all.
    fused = RAGService._fuse_hybrid_results(
        keyword_results=[_result("m1", 0.001), _result("m2", 0.001)],
        semantic_results=[_result("m2", 0.83), _result("m3", 0.41)],
        top_k=10,
        alpha=0.7,
        include_citations=False,
    )
    row = {r.id: r for r in fused}["m2"]
    assert row.metadata["hybrid_fusion"]["rrf_k"] == DEFAULT_RRF_K
    expected_fts_rrf = 1.0 / (DEFAULT_RRF_K + 2)
    expected_vector_rrf = 1.0 / (DEFAULT_RRF_K + 1)
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


# =============================================================================
# TASK-4110 review (coordinator round): the search cache blinds the sweep
# =============================================================================
#
# IMPORTANT 1: `SimpleRAGCache._make_key` had no fusion parameters, so a
# hybrid search at rrf_k=10 was served BYTE-IDENTICAL to a request at
# rrf_k=1000 (the reviewer's live probe). Task 4's runner mutates
# `config.search` in place on ONE service with `enable_cache=True` -- every
# k in the sweep would have reported identical metrics ("+0.000, k doesn't
# matter"), corrupting the measurement this whole arc exists to make.
#
# Fixed by threading the RESOLVED `(alpha, rrf_k, pool_multiplier)` into the
# cache key for HYBRID searches only -- semantic/keyword never depend on
# these three, so their key is untouched. RESOLVED (not raw config) values
# so two configs that happen to resolve to the same effective number
# correctly SHARE an entry rather than needlessly splitting one.


def test_make_key_hybrid_fusion_part_changes_the_key():
    """Direct `_make_key` pin: each of the three resolved values, alone,
    must produce a different key. (`SimpleRAGCache` is imported fresh here,
    matching `test_keyword_leg_pushdown.py`'s `_cache()` helper pattern.)
    """
    from tldw_chatbook.RAG_Search.simplified.simple_cache import SimpleRAGCache

    cache = SimpleRAGCache(enabled=True)
    base = cache._make_key("q", "hybrid", 10, None, None, None, (0.7, 60, 2))
    changed_alpha = cache._make_key("q", "hybrid", 10, None, None, None, (0.5, 60, 2))
    changed_k = cache._make_key("q", "hybrid", 10, None, None, None, (0.7, 10, 2))
    changed_multiplier = cache._make_key(
        "q", "hybrid", 10, None, None, None, (0.7, 60, 5)
    )

    assert len({base, changed_alpha, changed_k, changed_multiplier}) == 4, (
        "each fusion parameter must independently change the cache key"
    )


def test_make_key_no_hybrid_fusion_is_byte_identical_to_before_the_parameter_existed():
    """Backward-compat pin, same idiom as the keyword_source_types one in
    `test_keyword_leg_pushdown.py`: the pre-existing five/six-positional-arg
    call must still find entries written before `hybrid_fusion` existed.
    """
    from tldw_chatbook.RAG_Search.simplified.simple_cache import SimpleRAGCache

    cache = SimpleRAGCache(enabled=True)
    legacy = cache._make_key("q", "semantic", 10, None, None)
    explicit_none = cache._make_key("q", "semantic", 10, None, None, None, None)

    assert legacy == explicit_none


def test_cache_key_changes_when_rrf_k_changes(monkeypatch):
    """THE REVIEWER'S EXACT PROBE: same query, same service, flip
    `config.search.rrf_k` -> the second hybrid search MISSES the cache
    (not served the first's stale rows) and its returned metadata records
    the NEW k, proving the second call actually re-ran fusion rather than
    merely computing a different key that happened to still miss for some
    other reason.
    """
    service = _make_service(enable_cache=True, rrf_k=10)

    keyword = [_result("m1", 0.5), _result("m2", 0.4)]
    semantic = [_result("m2", 0.9), _result("m3", 0.3)]

    async def fake_semantic_search(*args, **kwargs):
        return semantic

    async def fake_keyword_search(*args, **kwargs):
        return keyword

    monkeypatch.setattr(service, "_semantic_search", fake_semantic_search)
    monkeypatch.setattr(service, "_keyword_search", fake_keyword_search)

    first = asyncio.run(service.search("quokka", top_k=10, search_type="hybrid"))
    assert service.cache._misses == 1
    assert {r.id: r for r in first}["m2"].metadata["hybrid_fusion"]["rrf_k"] == 10

    service.config.search.rrf_k = 1000

    second = asyncio.run(service.search("quokka", top_k=10, search_type="hybrid"))
    assert service.cache._misses == 2, (
        "flipping rrf_k must MISS the k=10 entry, not silently reuse it"
    )
    assert {r.id: r for r in second}["m2"].metadata["hybrid_fusion"]["rrf_k"] == 1000


def test_cache_key_changes_when_hybrid_alpha_changes(monkeypatch):
    """Same probe, for alpha."""
    service = _make_service(enable_cache=True, hybrid_alpha=0.7)

    keyword = [_result("m1", 0.5), _result("m2", 0.4)]
    semantic = [_result("m2", 0.9), _result("m3", 0.3)]

    async def fake_semantic_search(*args, **kwargs):
        return semantic

    async def fake_keyword_search(*args, **kwargs):
        return keyword

    monkeypatch.setattr(service, "_semantic_search", fake_semantic_search)
    monkeypatch.setattr(service, "_keyword_search", fake_keyword_search)

    first = asyncio.run(service.search("quokka", top_k=10, search_type="hybrid"))
    assert service.cache._misses == 1
    assert {r.id: r for r in first}["m2"].metadata["hybrid_fusion"]["alpha"] == 0.7

    service.config.search.hybrid_alpha = 0.1

    second = asyncio.run(service.search("quokka", top_k=10, search_type="hybrid"))
    assert service.cache._misses == 2, (
        "flipping hybrid_alpha must MISS the alpha=0.7 entry, not silently reuse it"
    )
    assert {r.id: r for r in second}["m2"].metadata["hybrid_fusion"]["alpha"] == 0.1


def test_cache_key_changes_when_hybrid_pool_multiplier_changes(monkeypatch):
    """Same probe, for the pool multiplier. It never lands in the returned
    metadata (only alpha/rrf_k do), so "records the new value" is pinned via
    the leg spy instead: the SECOND call must actually re-run the legs at
    the new multiplier's widened top_k, which is only observable if the
    second call was a genuine miss.
    """
    service = _make_service(enable_cache=True, hybrid_pool_multiplier=2)

    leg_calls = []

    async def fake_semantic_search(query, top_k, *args, **kwargs):
        leg_calls.append(top_k)
        return [_result("m1", 0.5)]

    async def fake_keyword_search(query, top_k, *args, **kwargs):
        return [_result("m1", 0.5)]

    monkeypatch.setattr(service, "_semantic_search", fake_semantic_search)
    monkeypatch.setattr(service, "_keyword_search", fake_keyword_search)

    asyncio.run(service.search("quokka", top_k=4, search_type="hybrid"))
    assert service.cache._misses == 1
    assert leg_calls == [8]  # 4 * 2

    service.config.search.hybrid_pool_multiplier = 5

    asyncio.run(service.search("quokka", top_k=4, search_type="hybrid"))
    assert service.cache._misses == 2, (
        "flipping hybrid_pool_multiplier must MISS the multiplier=2 entry"
    )
    assert leg_calls == [8, 20], (
        "the second call must have actually re-run the legs at the new "
        f"multiplier (4*5=20), not been served from cache: {leg_calls}"
    )


def test_semantic_mode_cache_key_is_unaffected_by_all_three_fusion_knobs(monkeypatch):
    """Semantic-mode search never depends on alpha/rrf_k/pool_multiplier, so
    flipping all three must NOT cause a semantic-mode cache miss.
    """
    service = _make_service(enable_cache=True)

    async def fake_semantic_search(*args, **kwargs):
        return [_result("m1", 0.5)]

    monkeypatch.setattr(service, "_semantic_search", fake_semantic_search)

    asyncio.run(service.search("quokka", top_k=5, search_type="semantic"))
    assert service.cache._misses == 1
    assert service.cache._hits == 0

    service.config.search.hybrid_alpha = 0.1
    service.config.search.rrf_k = 999
    service.config.search.hybrid_pool_multiplier = 7

    asyncio.run(service.search("quokka", top_k=5, search_type="semantic"))
    assert service.cache._misses == 1, (
        "semantic-mode cache key must be unaffected by the hybrid fusion knobs"
    )
    assert service.cache._hits == 1


def test_keyword_mode_cache_key_is_unaffected_by_all_three_fusion_knobs(monkeypatch):
    """Same pin, for keyword mode."""
    service = _make_service(enable_cache=True)

    async def fake_keyword_search(*args, **kwargs):
        return [_result("m1", 0.5)]

    monkeypatch.setattr(service, "_keyword_search", fake_keyword_search)

    asyncio.run(service.search("quokka", top_k=5, search_type="keyword"))
    assert service.cache._misses == 1

    service.config.search.hybrid_alpha = 0.1
    service.config.search.rrf_k = 999
    service.config.search.hybrid_pool_multiplier = 7

    asyncio.run(service.search("quokka", top_k=5, search_type="keyword"))
    assert service.cache._misses == 1, (
        "keyword-mode cache key must be unaffected by the hybrid fusion knobs"
    )
    assert service.cache._hits == 1


# =============================================================================
# TASK-4110 review (coordinator round): minors folded in
# =============================================================================
#
# Minor (a): the multiplier floor/cap guard had no DIRECT unit test (only
# exercised indirectly through _hybrid_search with a multiplier of 5) and
# was therefore deletable with a green suite. Minor (b): the invalid-value
# fallback must be the dataclass's OWN default (DEFAULT_HYBRID_POOL_
# MULTIPLIER, 2), not the unrelated module-level SEARCH_RESULT_MULTIPLIER
# constant -- the two happened to both be 2, which hid that they are
# different knobs.
#
# RELEASE NOTE (disclosure, per the coordinator): hybrid legs previously
# honored the undocumented `[rag.service] search_result_multiplier` TOML
# setting for their over-fetch (shared with `_semantic_search`'s own
# internal multiplier); they now honor `hybrid_pool_multiplier` instead. A
# user who had set `search_result_multiplier = 4` gets the hybrid legs back
# down to 2 until they explicitly set `hybrid_pool_multiplier`.
#
# Also for Task 4's sweep-range choice: the EFFECTIVE fetch compounds on the
# semantic leg only -- `_semantic_search` applies its own
# SEARCH_RESULT_MULTIPLIER on top of whatever top_k `_hybrid_search` hands
# it, so the semantic leg's raw vector-store fetch is
# `top_k * hybrid_pool_multiplier * SEARCH_RESULT_MULTIPLIER`, while the
# keyword leg's fetch is the simple `top_k * hybrid_pool_multiplier` (no
# second multiplier there).


@pytest.mark.parametrize(
    "raw,expected",
    [
        (0, 1),  # floored
        (-5, 1),  # floored
        (-1000, 1),  # floored, far below
        (1, 1),  # boundary: already valid, unchanged
        (100, 100),  # boundary: at the cap, unchanged
        (1000, 100),  # capped (MAX_HYBRID_POOL_MULTIPLIER)
        ("abc", 2),  # non-numeric -> DEFAULT_HYBRID_POOL_MULTIPLIER
        (None, 2),  # non-numeric -> DEFAULT_HYBRID_POOL_MULTIPLIER
        (object(), 2),  # non-numeric -> DEFAULT_HYBRID_POOL_MULTIPLIER
        # Qodo PR-1487: `int(value)` raises OverflowError -- not TypeError
        # or ValueError -- for an infinite float. TOML accepts a literal
        # `inf`/`-inf`, so `hybrid_pool_multiplier = inf` in a hand-edited
        # config reaches this resolver directly, before any hybrid leg
        # launches.
        (float("inf"), 2),  # overflow -> DEFAULT_HYBRID_POOL_MULTIPLIER
        (float("-inf"), 2),  # overflow -> DEFAULT_HYBRID_POOL_MULTIPLIER
    ],
)
def test_pool_multiplier_resolver_floors_caps_and_falls_back(raw, expected):
    from tldw_chatbook.RAG_Search.simplified.rag_service import (
        _resolve_hybrid_pool_multiplier,
    )

    assert _resolve_hybrid_pool_multiplier(raw) == expected


def test_overflow_range_pool_multiplier_falls_back_with_warning(warnings_captured):
    """Qodo PR-1487 RED pin: the overflow branch must fall back through the
    same warned path as every other invalid ``hybrid_pool_multiplier``, not
    raise ``OverflowError`` out of ``_hybrid_search`` before either leg
    launches.
    """
    from tldw_chatbook.RAG_Search.simplified.config import (
        DEFAULT_HYBRID_POOL_MULTIPLIER,
    )
    from tldw_chatbook.RAG_Search.simplified.rag_service import (
        _resolve_hybrid_pool_multiplier,
    )

    assert (
        _resolve_hybrid_pool_multiplier(float("inf"))
        == DEFAULT_HYBRID_POOL_MULTIPLIER
        == 2
    )
    assert any(
        "hybrid_pool_multiplier" in message for message in warnings_captured
    ), f"an overflow-range hybrid_pool_multiplier must leave a warning trace: {warnings_captured}"


def test_invalid_pool_multiplier_falls_back_to_its_own_default_not_search_result_multiplier(
    monkeypatch,
):
    """MINOR (b) PIN: the fallback source is DEFAULT_HYBRID_POOL_MULTIPLIER
    (this field's own default), never the unrelated module-level
    SEARCH_RESULT_MULTIPLIER -- proven by monkeypatching the latter to a
    DIFFERENT value and showing it does not leak into the fallback.
    """
    import tldw_chatbook.RAG_Search.simplified.rag_service as rag_service_module
    from tldw_chatbook.RAG_Search.simplified.config import (
        DEFAULT_HYBRID_POOL_MULTIPLIER,
    )

    monkeypatch.setattr(rag_service_module, "SEARCH_RESULT_MULTIPLIER", 4)

    assert (
        rag_service_module._resolve_hybrid_pool_multiplier("not-a-number")
        == DEFAULT_HYBRID_POOL_MULTIPLIER
        == 2
    )
