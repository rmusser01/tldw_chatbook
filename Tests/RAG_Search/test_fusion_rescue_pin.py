"""TASK-4110 AC#4: the shipped weighting must let an FTS-only row through.

The defect: with the server's `rrf_k = 60` over chatbook's ~20-row candidate
window, an FTS-only row at keyword rank 1 scores `(1-alpha)/(60+1)` = 0.00492
and is beaten by every vector-only row down to rank ~83 -- while
`_hybrid_search` only ever asks the vector leg for `top_k *
hybrid_pool_multiplier` (20) candidates. A document the vector leg missed
entirely was therefore *structurally* unable to enter hybrid's fused top-k,
which is the sense in which hybrid was a semantic-only mode for exactly the
documents keyword search exists to catch.

Task 4 measured the fix and Task 5 shipped it: `SearchConfig.rrf_k` defaults
to 5 (`config.DEFAULT_HYBRID_RRF_K`). These pins are the always-on guard, and
they are deliberately hand-built rather than corpus-driven -- the eval
harness that produced the measurement is env-gated and needs a warm model
cache, so it cannot be the thing that notices a defaults revert.

THREE PINS, because the boundary holds more than one thing and a single test
sitting on it would conflate them:

1. **The weighting pin** -- at vector rank 10 the keyword row wins on SCORE
   (0.0500 > 0.0467) with margin to spare. This is what reverting `rrf_k` to
   60 reds, and it depends on neither float rounding nor any sort convention.
   It is the conservative statement of the guarantee: *strictly outranks from
   vector rank 10*.
2. **The boundary pin** -- vector rank 9 is where the weighting stops
   *strictly* preferring the keyword row: in exact rational arithmetic the
   two scores are EQUAL there (3/10 x 1/6 == 7/10 x 1/14 == 1/20). The
   keyword row still ranks above it. Which mechanism delivers that depends on
   arithmetic, and the honest answer is "score, in the floats production
   actually computes": `(1.0 - 0.7) * (1.0/6)` rounds to exactly 0.05 while
   `0.7 * (1.0/14)` is 0.049999999999999996, so the comparison is decided one
   ULP apart and the sort never reaches the tie-break. In exact arithmetic it
   would be the tie-break instead. Both are true of different arithmetics;
   neither is a claim that the WEIGHTING prefers the keyword row at rank 9.
3. **The convention pin** -- `(-score, fts_rank, vector_rank)` with absent
   legs last, on a bit-identical pair so the comparison genuinely reaches the
   tie-break. This is the ONLY test here that pins that convention; pin 2
   does not exercise it under the numbers the product runs.

Tests 1 and 2 run through the real `_hybrid_search` seam on an untouched
`RAGConfig()`, so they fail if the shipped DEFAULT moves -- not merely if the
fusion arithmetic is wrong.
"""
import asyncio
from fractions import Fraction

import pytest

from tldw_chatbook.RAG_Search.fusion import reciprocal_rank_fusion
from tldw_chatbook.RAG_Search.simplified.config import RAGConfig
from tldw_chatbook.RAG_Search.simplified.rag_service import RAGService
from tldw_chatbook.RAG_Search.simplified.vector_store import SearchResult

pytestmark = pytest.mark.unit

#: The keyword-only document: found by the FTS leg at rank 1, absent from the
#: vector leg entirely (the case AC#1 calls structurally impossible at k=60).
KEYWORD_ONLY_ID = "note_777"


def _keyword_row(doc_id: str, source_type: str = "note") -> SearchResult:
    """An FTS-leg row with the metadata `_process_keyword_results_basic` stamps.

    Real metadata matters: `_fusion_doc_key` fuses on
    `(source_type, source_id or doc_id)`, so rows without it fall back to the
    row id and the "distinct documents" premise of this pin would be an
    accident of id spelling rather than of document identity.
    """
    return SearchResult(
        id=f"{source_type}_{doc_id}",
        score=0.9,
        document=f"keyword hit {doc_id}",
        metadata={
            "doc_id": doc_id,
            "source_id": doc_id,
            "source_type": source_type,
            "title": f"doc {doc_id}",
        },
    )


def _vector_row(doc_id: str, score: float, source_type: str = "media") -> SearchResult:
    """A vector-leg chunk row as `index_document` spreads document metadata."""
    return SearchResult(
        id=f"{source_type}_{doc_id}_chunk_0",
        score=score,
        document=f"semantic hit {doc_id}",
        metadata={
            "doc_id": f"{source_type}_{doc_id}",
            "source_id": doc_id,
            "source_type": source_type,
            "title": f"doc {doc_id}",
            "chunk_index": 0,
        },
    )


def _vector_leg(n: int) -> list[SearchResult]:
    """`n` DISTINCT documents, best first -- ranks 1..n after fusion."""
    return [_vector_row(str(i), score=1.0 - i / 100.0) for i in range(1, n + 1)]


def _default_service() -> RAGService:
    """A service on an UNTOUCHED `RAGConfig()` -- the shipped defaults.

    Only the embedding/vector-store plumbing is pinned to the in-memory
    fakes (mirroring `test_fusion_config_knobs._make_service`); nothing under
    `config.search` is touched, which is the whole point: these tests must
    read the shipped `rrf_k`/`hybrid_alpha`, never a value they set
    themselves.
    """
    cfg = RAGConfig()
    cfg.embedding.model = "mock"
    cfg.embedding.device = "cpu"
    cfg.vector_store.type = "memory"
    cfg.vector_store.persist_directory = None
    cfg.search.enable_cache = False
    return RAGService(cfg)


def _fuse_through_the_seam(
    service: RAGService,
    monkeypatch,
    keyword: list[SearchResult],
    semantic: list[SearchResult],
    top_k: int,
) -> list[str]:
    """Run `_hybrid_search` over hand-built legs; return fused doc ids in order."""

    async def fake_semantic_search(*args, **kwargs):
        return semantic

    async def fake_keyword_search(*args, **kwargs):
        return keyword

    monkeypatch.setattr(service, "_semantic_search", fake_semantic_search)
    monkeypatch.setattr(service, "_keyword_search", fake_keyword_search)

    fused = asyncio.run(
        service._hybrid_search("plant maintenance record", top_k=top_k, include_citations=False)
    )
    return [row.id for row in fused]


# --- Pin 1: THE WEIGHTING PIN ----------------------------------------------


def test_fts_only_row_outranks_a_vector_only_row_at_vector_rank_10(monkeypatch):
    """An FTS-only rank-1 row beats a vector-only rank-10 row ON SCORE.

    Under the shipped defaults (alpha 0.7, rrf_k 5) the keyword row scores
    0.3/6 = 0.0500 and the vector row at rank 10 scores 0.7/15 = 0.04667.
    Under the previous rrf_k = 60 the same comparison is 0.3/61 = 0.00492 vs
    0.7/70 = 0.01000 and the keyword row loses -- which is exactly the
    revert this test exists to catch.
    """
    service = _default_service()
    alpha = service.config.search.hybrid_alpha
    rrf_k = service.config.search.rrf_k

    keyword_score = (1 - alpha) / (rrf_k + 1)
    vector_rank_10_score = alpha / (rrf_k + 10)
    assert keyword_score > vector_rank_10_score, (
        f"the shipped weighting (alpha={alpha}, rrf_k={rrf_k}) does not let an "
        f"FTS-only rank-1 row ({keyword_score:.5f}) outrank a vector-only row at "
        f"rank 10 ({vector_rank_10_score:.5f}) -- AC#4's structural guarantee is gone"
    )

    # ... and the seam agrees, with both rows visible (top_k = 11 keeps the
    # rank-10 vector row in the output so the ORDER can be read).
    keyword = [_keyword_row(KEYWORD_ONLY_ID)]
    semantic = _vector_leg(10)
    order = _fuse_through_the_seam(service, monkeypatch, keyword, semantic, top_k=11)

    fts_only_id = keyword[0].id
    vector_rank_10_id = semantic[9].id
    assert fts_only_id in order, "the keyword-only document did not survive fusion"
    assert order.index(fts_only_id) < order.index(vector_rank_10_id), (
        f"fused order put the vector rank-10 row ahead of the FTS-only row: {order}"
    )

    # AC#1's product-level claim: with a vector leg of top_k DISTINCT
    # documents, the keyword-only row still makes the cut -- it displaces the
    # weakest vector row rather than being the row that is displaced.
    order_at_k = _fuse_through_the_seam(service, monkeypatch, keyword, semantic, top_k=10)
    assert fts_only_id in order_at_k, (
        "a keyword-only document must be able to enter hybrid's top-k when the "
        f"vector leg returns k distinct documents: {order_at_k}"
    )
    assert vector_rank_10_id not in order_at_k, (
        "the keyword row entered top-k without displacing anything, so this "
        "assertion is not testing what it claims"
    )


# --- Pin 2: where the boundary IS (and which arithmetic decides it) --------


def test_vector_rank_9_is_the_exact_boundary_and_the_keyword_row_still_ranks_above(
    monkeypatch,
):
    """Rank 9 is the exact equality point; pin 1's "from rank 10" is why.

    In exact rational arithmetic `3/10 x 1/6 == 7/10 x 1/14 == 1/20`, so the
    weighting does NOT strictly prefer the keyword row at vector rank 9 --
    which is precisely why the guarantee is stated conservatively as "from
    rank 10" and why this row is pinned separately.

    The keyword row nevertheless ranks above the rank-9 vector row, and this
    test asserts that outcome without claiming a mechanism it does not
    exercise: in the floats the code actually computes,
    `(1.0 - alpha) * (1.0/6)` rounds to exactly 0.05 while
    `alpha * (1.0/14)` is 0.049999999999999996, so the FTS row wins ON SCORE
    by one ULP and `reciprocal_rank_fusion`'s sort never reaches its
    tie-break. (In exact arithmetic the tie-break is what would decide it.
    That convention is pinned on its own, on a bit-identical pair, in the
    next test.) Both facts are asserted below so a future float change that
    flips the ULP fails HERE, loudly, rather than silently changing which
    mechanism the suite is really covering.
    """
    service = _default_service()
    alpha = service.config.search.hybrid_alpha
    rrf_k = service.config.search.rrf_k

    exact_alpha = Fraction(str(alpha))
    assert (1 - exact_alpha) / (rrf_k + 1) == exact_alpha / (rrf_k + 9), (
        "the shipped weighting no longer ties at vector rank 9 in exact "
        "arithmetic; the boundary moved, so this pin and pin 1 are describing "
        "the wrong ranks"
    )

    # The arithmetic the product runs, spelled exactly as fusion.py computes
    # it (`(1.0 - alpha) * fts_rrf`, never a 0.3 literal).
    keyword_score = (1.0 - alpha) * (1.0 / (rrf_k + 1))
    vector_rank_9_score = alpha * (1.0 / (rrf_k + 9))
    assert keyword_score > vector_rank_9_score, (
        "in float arithmetic the keyword row no longer edges the rank-9 vector "
        f"row on score ({keyword_score!r} vs {vector_rank_9_score!r}); the "
        "ordering below is now the tie-break's doing, not the score's, and "
        "this docstring must be corrected rather than the assertion relaxed"
    )

    keyword = [_keyword_row(KEYWORD_ONLY_ID)]
    semantic = _vector_leg(9)
    order = _fuse_through_the_seam(service, monkeypatch, keyword, semantic, top_k=10)

    assert order.index(keyword[0].id) < order.index(semantic[8].id), (
        f"the FTS-only row must still rank above the rank-9 vector row: {order}"
    )


# --- Pin 3: the ordering convention itself, without float luck --------------


def test_absent_legs_sort_last_on_an_exactly_equal_score():
    """`(-score, fts_rank, vector_rank)`: a present rank beats an absent one.

    Pin 2's rank-9 row is an exact tie in rational arithmetic, but the two
    sides round to *different* floats (0.05 vs 0.049999999999999996), so at
    the shipped numbers it never reaches this convention at all -- it is
    decided one ULP apart on score. Nothing else in the suite covers the
    convention, so this test builds a bit-identical pair (alpha 0.5 puts both
    legs on the same expression at the same rank) and the comparison genuinely
    does reach the tie-break.
    """
    fts_only = _keyword_row("111")
    vector_only = _vector_row("222", score=0.5)

    fused = reciprocal_rank_fusion(
        [fts_only],
        [vector_only],
        key=lambda r: r.id,
        alpha=0.5,
        rrf_k=5,
    )

    scores = [entry.score for entry in fused]
    assert scores[0] == scores[1], (
        f"this test needs an exact float tie to exercise the tie-break: {scores}"
    )
    assert fused[0].key == fts_only.id, (
        "on an exact tie the row with a present fts_rank must precede the row "
        "whose fts_rank is absent"
    )
