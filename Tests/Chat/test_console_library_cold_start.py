"""The first Automatic send waits out a cold Library runtime (TASK-33621.20 AC#4).

Live on dev 64579cce2c: the first Automatic Library send after every launch
paused with a retrieval timeout. Automatic preparation gives the search 5 s,
and the real ``LibraryLocalRagSearchService`` builds the shared RAG runtime
(embedding model, vector store) inside its first ``search`` call: about 4-5 s
with the model cached, 18 s when it is not. The build ate the whole budget.

These tests use the real controller and the real Library search service. Only
the shared-runtime factory is replaced, by one that takes longer than the
search budget to build, and the semantic route, so no embedding model loads.
"""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

from Tests.Chat.test_console_automatic_library_preparation import (
    _controller_for_preparation,
    _preparation,
)
from tldw_chatbook.Chat.console_turn_preparation import (
    ConsolePreparationPauseKind,
    ConsoleTurnPreparationState,
)
from tldw_chatbook.Library import library_local_rag_search_service as service_module
from tldw_chatbook.Library.library_local_rag_search_service import (
    LibraryLocalRagSearchService,
)

pytestmark = pytest.mark.bootstrap_profile

#: The search budget under test, and a runtime build that outlasts it.
SEARCH_BUDGET = 0.2
COLD_BUILD_SECONDS = 0.8


class _Runtime:
    """A built shared RAG runtime: only ``search`` is read before routing."""

    def search(self, *_args, **_kwargs):  # pragma: no cover - never reached
        raise AssertionError("the semantic route is replaced in these tests")


def _cold_library(monkeypatch, *, build_seconds: float, search_seconds: float = 0.0):
    """The real search service over a runtime that is slow to build."""
    builds: list[float] = []
    release = threading.Event()

    def build_shared_runtime():
        started = time.monotonic()
        release.wait(build_seconds)
        builds.append(time.monotonic() - started)
        return _Runtime()

    monkeypatch.setattr(service_module, "embeddings_rag_deps_installed", lambda: True)
    monkeypatch.setattr(service_module, "get_shared_rag_service", build_shared_runtime)
    app = SimpleNamespace()
    service = LibraryLocalRagSearchService(app)

    async def semantic(query, source_types, top_k, kwargs, *, scope, rag_service, **_):
        assert isinstance(rag_service, _Runtime)
        if search_seconds:
            await asyncio.sleep(search_seconds)
        return {"results": []}

    monkeypatch.setattr(service, "_search_semantic", semantic)
    return service, builds, release


@pytest.mark.asyncio
async def test_a_runtime_build_longer_than_the_budget_does_not_time_out_the_send(
    monkeypatch,
):
    service, builds, _release = _cold_library(
        monkeypatch, build_seconds=COLD_BUILD_SECONDS
    )
    controller, store = _controller_for_preparation(
        _preparation(), service, timeout=SEARCH_BUDGET
    )

    outcome = await controller.prepare_library_for_turn("preparation-1")

    assert outcome.error_code is None, outcome.error_code
    assert outcome.state is ConsoleTurnPreparationState.READY
    # The runtime really was cold: one build, and it outlasted the budget.
    assert len(builds) == 1 and builds[0] >= COLD_BUILD_SECONDS * 0.9
    assert store.preparation_for_session("session-1").pause_kind is None


@pytest.mark.asyncio
async def test_a_slow_search_on_a_warm_runtime_still_times_out(monkeypatch):
    """Negative control: only the runtime build is outside the budget."""
    service, builds, _release = _cold_library(
        monkeypatch, build_seconds=0.0, search_seconds=COLD_BUILD_SECONDS
    )
    controller, store = _controller_for_preparation(
        _preparation(), service, timeout=SEARCH_BUDGET
    )

    outcome = await controller.prepare_library_for_turn("preparation-1")

    assert outcome.state is ConsoleTurnPreparationState.PAUSED
    assert outcome.error_code == "library_retrieval_timeout"
    paused = store.preparation_for_session("session-1")
    assert paused.pause_kind is ConsolePreparationPauseKind.RETRIEVAL


@pytest.mark.asyncio
async def test_a_runtime_build_that_never_finishes_is_still_bounded(monkeypatch):
    """The build has its own, longer bound: a hung build cannot hang the send."""
    service, _builds, release = _cold_library(monkeypatch, build_seconds=30.0)
    controller, store = _controller_for_preparation(
        _preparation(), service, timeout=SEARCH_BUDGET
    )
    controller._library_initialization_timeout = 0.5
    try:
        outcome = await asyncio.wait_for(
            controller.prepare_library_for_turn("preparation-1"), timeout=10
        )
    finally:
        release.set()

    assert outcome.state is ConsoleTurnPreparationState.PAUSED
    assert outcome.error_code == "library_retrieval_timeout"
    assert (
        store.preparation_for_session("session-1").pause_kind
        is ConsolePreparationPauseKind.RETRIEVAL
    )
