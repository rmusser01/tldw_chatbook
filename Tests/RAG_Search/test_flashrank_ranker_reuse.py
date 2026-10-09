# T1 (F1): FlashRank ranker is constructed once and rerank work runs off
# the event loop.
#
# The default-enabled `rerank_results` process step used to do
# `ranker = Ranker(model_name=..., cache_dir="/tmp")` on EVERY query: a
# cold ORT/torch model load (~hundreds of ms) executed synchronously on the
# Textual event loop, freezing the UI on every RAG chat send (same
# cold-load pattern fixed in rag_service.py's vector-store query,
# TASK-32804.9). These tests pin the two halves of the fix:
#
#   1. the Ranker is a lazy, thread-safe, process-wide singleton
#      (`get_flashrank_ranker()`), constructed exactly once;
#   2. the pipeline's rerank step (and the async wrapper) dispatches the
#      model work to a worker thread, never the event-loop thread.
#
# No flashrank import is required: the `_RANKER_FACTORY` seam injects a
# fake ranker, keeping these tests dependency-free.

import threading
from types import SimpleNamespace
from typing import ClassVar

import pytest

from tldw_chatbook.RAG_Search import pipeline_functions_simple as pfs
from tldw_chatbook.RAG_Search.local_citation_capture import (
    FINAL_SCORE_KIND_KEY,
    FINAL_SCORE_KIND_RERANKER,
)
from tldw_chatbook.RAG_Search.pipeline_builder_simple import (
    _execute_process_step,
    execute_pipeline,
)
from tldw_chatbook.RAG_Search.pipeline_types import SearchResult

pytestmark = pytest.mark.unit


class _FakeRanker:
    """Stands in for flashrank.Ranker via the ``_RANKER_FACTORY`` seam."""

    instances = 0
    last_init_kwargs: ClassVar[dict] = {}

    def __init__(self, *args, **kwargs):
        type(self).instances += 1
        type(self).last_init_kwargs = dict(kwargs)

    def rerank(self, request):
        # Emulate flashrank's contract: ranked entries carry a passage
        # ``index`` and a relevance ``score``.
        return [
            SimpleNamespace(index=rank, score=1.0 - rank * 0.1)
            for rank in range(len(request.passages))
        ]


@pytest.fixture(autouse=True)
def _fake_flashrank(monkeypatch):
    _FakeRanker.instances = 0
    _FakeRanker.last_init_kwargs = {}
    monkeypatch.setattr(pfs, "_RANKER_FACTORY", _FakeRanker)
    pfs._reset_flashrank_ranker_for_tests()
    yield
    pfs._reset_flashrank_ranker_for_tests()


def _results(*, count: int = 2) -> list:
    return [
        SearchResult(
            source="media",
            id=f"m{i}",
            title=f"title-{i}",
            content=f"body-{i}",
            score=1.0 - i * 0.1,
            metadata={"producer": "test"},
        )
        for i in range(count)
    ]


def test_get_flashrank_ranker_returns_singleton():
    first = pfs.get_flashrank_ranker()
    second = pfs.get_flashrank_ranker()

    assert first is second
    assert _FakeRanker.instances == 1


def test_ranker_constructed_once_across_rerank_calls():
    # Two full rerank passes through the real function must share one Ranker
    # (previously each call constructed -- and re-loaded -- the model).
    pfs.rerank_results(_results(), "query", model="flashrank")
    pfs.rerank_results(_results(), "query", model="flashrank")

    assert _FakeRanker.instances == 1
    # And the rerank actually ran through the shared singleton (not a
    # fallback): reranked rows carry the reranker provenance marker.
    reranked = pfs.rerank_results(_results(), "query", model="flashrank")
    assert reranked[0].metadata[FINAL_SCORE_KIND_KEY] == FINAL_SCORE_KIND_RERANKER


def test_ranker_cache_dir_is_not_tmp_and_survives_reboot():
    # The old call hardcoded cache_dir="/tmp", so the ~100 MB model weights
    # were re-downloaded after every reboot (and leaked into /tmp).
    pfs.get_flashrank_ranker()

    cache_dir = _FakeRanker.last_init_kwargs.get("cache_dir")
    assert cache_dir, "factory must receive an explicit cache_dir"
    assert str(cache_dir) != "/tmp"
    assert str(cache_dir) == pfs._flashrank_cache_dir()


def test_reset_flashrank_ranker_for_tests_clears_the_singleton():
    first = pfs.get_flashrank_ranker()
    pfs._reset_flashrank_ranker_for_tests()

    assert pfs.get_flashrank_ranker() is not first
    assert _FakeRanker.instances == 2


async def test_rerank_results_async_runs_off_event_loop_thread():
    seen_threads: list[threading.Thread] = []
    orig_rerank = _FakeRanker.rerank

    def spy(self, request):
        seen_threads.append(threading.current_thread())
        return orig_rerank(self, request)

    _FakeRanker.rerank = spy
    try:
        reranked = await pfs.rerank_results_async(
            _results(), "query", model="flashrank"
        )
    finally:
        _FakeRanker.rerank = orig_rerank

    assert seen_threads, "the fake ranker's rerank() must have been called"
    assert seen_threads[0] is not threading.main_thread()
    assert reranked[0].metadata[FINAL_SCORE_KIND_KEY] == FINAL_SCORE_KIND_RERANKER


async def test_execute_process_step_runs_rerank_off_event_loop_thread():
    # The pipeline builder's process step -- the code path every RAG chat
    # send goes through -- must await rerank in a worker thread.
    seen_threads: list[threading.Thread] = []
    orig_rerank = _FakeRanker.rerank

    def spy(self, request):
        seen_threads.append(threading.current_thread())
        return orig_rerank(self, request)

    _FakeRanker.rerank = spy
    try:
        results = await _execute_process_step(
            {"type": "process", "function": "rerank_results"},
            {"results": _results(), "query": "query", "params": {}},
        )
    finally:
        _FakeRanker.rerank = orig_rerank

    assert seen_threads
    assert seen_threads[0] is not threading.main_thread()
    assert results[0].metadata[FINAL_SCORE_KIND_KEY] == FINAL_SCORE_KIND_RERANKER


async def test_execute_pipeline_rerank_step_still_reranks(monkeypatch):
    # End to end through execute_pipeline: the now-async process step is
    # awaited and its reranked results flow to the format step unchanged.
    formatted_marker: dict = {}

    async def fake_retrieve(app, query, sources, **kwargs):
        return _results()

    def fake_format(results, **kwargs):
        formatted_marker["count"] = len(results)
        formatted_marker["scores"] = [r.score for r in results]
        return "formatted"

    monkeypatch.setitem(pfs.RETRIEVAL_FUNCTIONS, "fake_retrieve", fake_retrieve)
    monkeypatch.setitem(pfs.FORMATTING_FUNCTIONS, "format_as_context", fake_format)

    results, formatted = await execute_pipeline(
        {
            "name": "test-pipeline",
            "steps": [
                {"type": "retrieve", "function": "fake_retrieve"},
                {"type": "process", "function": "rerank_results"},
                {"type": "format", "function": "format_as_context"},
            ],
        },
        app=None,
        query="query",
        sources={"media": True},
    )

    assert formatted == "formatted"
    assert _FakeRanker.instances == 1
    assert formatted_marker["count"] == len(results)
    # Fake scores are 1.0, 0.9 for indexes 0, 1 (descending by index).
    assert formatted_marker["scores"] == [1.0, 0.9]
