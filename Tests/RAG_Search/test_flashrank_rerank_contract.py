# TASK-34434: `rerank_results` must honor the REAL flashrank contract.
#
# flashrank 0.2.10's `Ranker.rerank()` (verified against the installed
# package source) mutates the input passage dicts in place -- it adds a
# ``"score"`` key to each -- sorts the passage list descending by score and
# returns that same list. Passages may carry an ``"id"`` which flashrank
# passes through untouched; ranked entries are plain dicts and have NO
# ``index`` attribute.
#
# The production code used to build passages without ``id`` and validate
# ``ranked.index`` (attribute access): against the real library that raises
# ``AttributeError`` inside the ``try``, which the broad ``except`` swallows,
# so every REAL flashrank rerank silently fell back to the original
# (unranked) order. The old tests faked the wrong contract
# (``SimpleNamespace(index=..., score=...)``) and so kept passing.
#
# These tests pin the real contract: passages carry ``id``, ranked dicts map
# back through it, output is descending by reranker score, and every
# fallback still preserves prior producer scores and provenance markers
# atomically.

import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_chatbook.RAG_Search import pipeline_functions_simple as pfs
from tldw_chatbook.RAG_Search.local_citation_capture import (
    FINAL_SCORE_KIND_KEY,
    FINAL_SCORE_KIND_RERANKER,
    SEMANTIC_SCORE_KIND_KEY,
    SEMANTIC_SCORE_KIND_VECTOR_SIMILARITY,
)
from tldw_chatbook.RAG_Search.pipeline_types import SearchResult

pytestmark = pytest.mark.unit


def _results() -> list:
    """Three results whose producer order is NOT the reranker's ranking.

    The fake ranker scores id 2 best and id 0 worst, so a working rerank
    must flip the order; a fallback (the historical bug) leaves it as-is.
    Prior scores/metadata mimic real producer output (RRF-ish floats and a
    semantic marker) so fallback-preservation is observable.
    """
    return [
        SearchResult(
            source="media",
            id="m0",
            title="t0",
            content="c0",
            score=0.30,
            metadata={SEMANTIC_SCORE_KIND_KEY: SEMANTIC_SCORE_KIND_VECTOR_SIMILARITY},
        ),
        SearchResult(
            source="media",
            id="m1",
            title="t1",
            content="c1",
            score=0.51,
            metadata={"producer": "opaque"},
        ),
        SearchResult(
            source="media",
            id="m2",
            title="t2",
            content="c2",
            score=0.72,
            metadata={SEMANTIC_SCORE_KIND_KEY: SEMANTIC_SCORE_KIND_VECTOR_SIMILARITY},
        ),
    ]


class _RealContractRanker:
    """Emulates flashrank 0.2.10's ``rerank`` exactly.

    Adds ``score`` to the INPUT passage dicts, sorts them descending and
    returns the same list -- no copies, no ``index`` attribute anywhere.
    """

    def __init__(self, **_kwargs):
        pass

    def rerank(self, request):
        score_by_id = {0: 0.10, 1: 0.50, 2: 0.90}  # id 2 best, id 0 worst
        for passage in request.passages:
            passage["score"] = score_by_id[passage["id"]]
        request.passages.sort(key=lambda p: p["score"], reverse=True)
        return request.passages


class _RaisingRanker:
    def __init__(self, **_kwargs):
        pass

    def rerank(self, _request):
        raise RuntimeError("model exploded")


class _GarbageRanker:
    """Returns contract-violating payloads (per-test ``payload``)."""

    def __init__(self, payload, **_kwargs):
        self._payload = payload

    def rerank(self, request):
        return self._payload(request)


@pytest.fixture()
def real_contract_ranker(monkeypatch):
    monkeypatch.setattr(pfs, "_RANKER_FACTORY", _RealContractRanker)
    pfs._reset_flashrank_ranker_for_tests()
    yield
    pfs._reset_flashrank_ranker_for_tests()


def test_real_flashrank_contract_reranks_by_passage_id(real_contract_ranker):
    # RED before the fix: attribute access (``ranked.index``) on the real
    # dict contract raises AttributeError, the broad except swallows it and
    # the fallback returns the ORIGINAL order -- rerank never re-ranked.
    reranked = pfs.rerank_results(_results(), "query", top_k=3)

    assert [r.id for r in reranked] == ["m2", "m1", "m0"]
    assert [r.score for r in reranked] == [0.90, 0.50, 0.10]
    assert all(
        r.metadata[FINAL_SCORE_KIND_KEY] == FINAL_SCORE_KIND_RERANKER for r in reranked
    )


def test_real_flashrank_contract_respects_top_k(real_contract_ranker):
    reranked = pfs.rerank_results(_results(), "query", top_k=2)

    # top_k selects by RERANKER score (id 2 > id 1), not producer order.
    assert [r.id for r in reranked] == ["m2", "m1"]


def test_real_contract_shuffled_return_is_sorted_descending(monkeypatch):
    # A ranker that returns the scored dicts in scrambled order (not the
    # pre-sorted list real flashrank returns) must still produce output
    # ordered by reranker score descending.
    class _ShuffledRanker(_RealContractRanker):
        def rerank(self, request):
            scored = super().rerank(request)
            return [scored[1], scored[2], scored[0]]

    monkeypatch.setattr(pfs, "_RANKER_FACTORY", _ShuffledRanker)
    pfs._reset_flashrank_ranker_for_tests()
    try:
        reranked = pfs.rerank_results(_results(), "query", top_k=3)
    finally:
        pfs._reset_flashrank_ranker_for_tests()

    assert [r.id for r in reranked] == ["m2", "m1", "m0"]


def test_passages_sent_to_ranker_carry_unique_ids(real_contract_ranker):
    # The fix's other half: passages must carry a unique id each (flashrank
    # passes them through; that is what maps ranked dicts back to results).
    seen_passages = []
    orig_rerank = _RealContractRanker.rerank

    def spy(self, request):
        seen_passages.append([dict(p) for p in request.passages])
        return orig_rerank(self, request)

    _RealContractRanker.rerank = spy
    try:
        pfs.rerank_results(_results(), "query", top_k=3)
    finally:
        _RealContractRanker.rerank = orig_rerank

    assert len(seen_passages) == 1
    ids = [p["id"] for p in seen_passages[0]]
    assert len(set(ids)) == len(ids), "passage ids must be unique"
    assert all("text" in p for p in seen_passages[0])


def test_attribute_style_ranker_response_still_tolerated(monkeypatch):
    # Defensive tolerance for the attribute-style shape (``.index``/``.score``)
    # older fakes used -- kept so alternate ranker backends keep working.
    class _AttrRanker:
        def __init__(self, **_kwargs):
            pass

        def rerank(self, _request):
            return [
                SimpleNamespace(index=2, score=0.9),
                SimpleNamespace(index=1, score=0.5),
                SimpleNamespace(index=0, score=0.1),
            ]

    monkeypatch.setattr(pfs, "_RANKER_FACTORY", _AttrRanker)
    pfs._reset_flashrank_ranker_for_tests()
    try:
        reranked = pfs.rerank_results(_results(), "query", top_k=3)
    finally:
        pfs._reset_flashrank_ranker_for_tests()

    assert [r.id for r in reranked] == ["m2", "m1", "m0"]


# ---------------------------------------------------------------------------
# Fallback semantics: garbage responses must preserve the original order and
# NEVER partially mutate producer scores or provenance markers.
# ---------------------------------------------------------------------------


def _garbage_case(monkeypatch, payload):
    def factory(**kwargs):
        return _GarbageRanker(payload, **kwargs)

    monkeypatch.setattr(pfs, "_RANKER_FACTORY", factory)
    pfs._reset_flashrank_ranker_for_tests()
    results = _results()
    snapshots = [(r.score, dict(r.metadata)) for r in results]

    fallback = pfs.rerank_results(results, "query", top_k=3)

    assert [r.id for r in fallback] == ["m0", "m1", "m2"], "original order kept"
    assert [(r.score, r.metadata) for r in results] == snapshots, "no partial mutation"
    assert all(FINAL_SCORE_KIND_KEY not in r.metadata for r in fallback)


def test_garbage_missing_id_falls_back(monkeypatch):
    # Scored dicts WITHOUT ids that are not the input objects cannot be
    # mapped back -- refuse rather than guess.
    _garbage_case(monkeypatch, lambda request: [{"text": "x", "score": 0.9}])


def test_garbage_missing_score_falls_back(monkeypatch):
    _garbage_case(monkeypatch, lambda request: [{"id": 0, "text": "x"}])


def test_garbage_fewer_items_than_passages_falls_back(monkeypatch):
    # flashrank always returns every passage; a short response is a broken
    # ranker -- returning a partial rerank would silently drop results.
    _garbage_case(
        monkeypatch,
        lambda request: [
            {"id": p["id"], "text": p["text"], "score": 0.5} for p in request.passages
        ][:1],
    )


def test_garbage_non_finite_score_falls_back(monkeypatch):
    _garbage_case(
        monkeypatch,
        lambda request: [
            {"id": p["id"], "text": p["text"], "score": math.nan}
            for p in request.passages
        ],
    )


def test_garbage_out_of_range_id_falls_back(monkeypatch):
    _garbage_case(
        monkeypatch,
        lambda request: [
            {"id": len(request.passages), "text": "x", "score": 0.5}
        ],
    )


def test_raising_ranker_falls_back(monkeypatch):
    monkeypatch.setattr(pfs, "_RANKER_FACTORY", _RaisingRanker)
    pfs._reset_flashrank_ranker_for_tests()
    results = _results()
    snapshots = [(r.score, dict(r.metadata)) for r in results]

    fallback = pfs.rerank_results(results, "query", top_k=3)

    assert [r.id for r in fallback] == ["m0", "m1", "m2"]
    assert [(r.score, r.metadata) for r in results] == snapshots
    assert all(FINAL_SCORE_KIND_KEY not in r.metadata for r in fallback)


# ---------------------------------------------------------------------------
# Real-library smoke (network-free when the model weights are already cached
# in the app data dir; skipped otherwise -- constructing the real Ranker
# would download ~35 MB).
# ---------------------------------------------------------------------------

#: The suite's conftest redirects HOME (bootstrap at import, per-test temp
#: dirs during tests), so the real weights are located through the passwd
#: database home -- immune to environment overrides. ``None`` on platforms
#: without ``pwd`` just skips the smoke.
_REAL_HOME: str | None = None
try:
    import pwd

    _REAL_HOME = pwd.getpwuid(os.getuid()).pw_dir
except (ImportError, KeyError):
    pass

#: ONNX weights file flashrank 0.2.10 downloads for ``_FLASHRANK_MODEL_NAME``.
_WEIGHTS_FILENAME = "flashrank-MiniLM-L-12-v2_Q.onnx"


def _cached_real_weights() -> Path | None:
    """Find previously-downloaded real flashrank weights, if any."""
    if not _REAL_HOME:
        return None
    roots = Path(_REAL_HOME) / ".local" / "share" / "tldw_cli"
    for candidate in roots.glob(f"*/cache/flashrank/*/{_WEIGHTS_FILENAME}"):
        return candidate
    return None


def test_real_flashrank_library_smoke(monkeypatch):
    flashrank = pytest.importorskip("flashrank", reason="flashrank not installed")
    weights = _cached_real_weights()
    if weights is None:
        pytest.skip(
            "flashrank model weights not cached anywhere under the real HOME; "
            "constructing the real Ranker would download them (network) -- "
            "skipping live smoke"
        )

    def real_ranker_factory(model_name, cache_dir):
        # Point the REAL Ranker at the real cached weights; the per-test
        # HOME isolation would otherwise resolve an empty cache dir and
        # trigger a download.
        real_cache_root = str(weights.parent.parent)
        return flashrank.Ranker(model_name=model_name, cache_dir=real_cache_root)

    monkeypatch.setattr(pfs, "_RANKER_FACTORY", real_ranker_factory)
    pfs._reset_flashrank_ranker_for_tests()
    try:
        results = [
            SearchResult(
                source="media",
                id="paris",
                title="Paris",
                content="Paris is the capital and largest city of France.",
                score=0.10,
                metadata={"producer": "smoke"},
            ),
            SearchResult(
                source="media",
                id="soup",
                title="Soup",
                content="Simmer the tomatoes with basil for twenty minutes.",
                score=0.90,
                metadata={"producer": "smoke"},
            ),
        ]

        reranked = pfs.rerank_results(
            results, "What is the capital of France?", top_k=2
        )
    finally:
        pfs._reset_flashrank_ranker_for_tests()

    assert len(reranked) == 2
    assert reranked[0].id == "paris", "the on-topic passage must actually rerank up"
    assert all(
        r.metadata[FINAL_SCORE_KIND_KEY] == FINAL_SCORE_KIND_RERANKER for r in reranked
    )
    assert all(math.isfinite(r.score) for r in reranked)
    assert reranked[0].score >= reranked[1].score
