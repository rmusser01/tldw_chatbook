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


async def _until(predicate, timeout: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        assert asyncio.get_running_loop().time() < deadline, "timed out"
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_stop_during_a_cold_build_pauses_the_send_at_once(monkeypatch):
    """A long first build is shown and stoppable; Stop pauses like a timeout."""
    service, _builds, release = _cold_library(monkeypatch, build_seconds=30.0)
    controller, store = _controller_for_preparation(
        _preparation(), service, timeout=SEARCH_BUDGET
    )
    store.switch_session("session-1")
    preparing = asyncio.ensure_future(
        controller.prepare_library_for_turn("preparation-1")
    )
    try:
        await _until(lambda: controller.is_stop_allowed)
        assert controller.run_state_for("session-1").visible_copy == (
            "Searching Library…"
        )
        assert controller.stop_active_run() is True
        outcome = await asyncio.wait_for(preparing, timeout=5)
    finally:
        release.set()

    assert outcome.state is ConsoleTurnPreparationState.PAUSED
    assert outcome.error_code == "library_retrieval_stopped"
    paused = store.preparation_for_session("session-1")
    assert paused.pause_kind is ConsolePreparationPauseKind.RETRIEVAL
    assert not controller.is_stop_allowed


@pytest.mark.asyncio
async def test_cancelling_the_send_itself_is_not_a_library_stop(monkeypatch):
    """Negative control: shutdown/close cancel the send; that is no pause."""
    service, _builds, release = _cold_library(monkeypatch, build_seconds=30.0)
    controller, store = _controller_for_preparation(
        _preparation(), service, timeout=SEARCH_BUDGET
    )
    preparing = asyncio.ensure_future(
        controller.prepare_library_for_turn("preparation-1")
    )
    try:
        await _until(lambda: "session-1" in controller._library_search_tasks)
        preparing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await preparing
    finally:
        release.set()

    assert controller._library_search_tasks == {}
    current = store.preparation_for_session("session-1")
    assert current.state is ConsoleTurnPreparationState.PREPARING


@pytest.mark.asyncio
async def test_retries_during_a_hung_build_join_it_instead_of_queueing_threads(
    monkeypatch,
):
    """Review MINOR 9: every Stop, timeout or Retry started another build thread.

    Each one blocked on the shared-build lock behind the hung build, so
    repeated retries could exhaust the default executor. They now join the one
    build in flight, which keeps running when a waiter gives up.
    """
    calls: list[int] = []
    release = threading.Event()

    def build_shared_runtime():
        calls.append(1)
        release.wait(30)
        return _Runtime()

    monkeypatch.setattr(service_module, "embeddings_rag_deps_installed", lambda: True)
    monkeypatch.setattr(service_module, "get_shared_rag_service", build_shared_runtime)
    service = LibraryLocalRagSearchService(SimpleNamespace())
    try:
        for _ in range(3):
            with pytest.raises(TimeoutError):
                async with asyncio.timeout(0.2):
                    await service.warm_up()
        assert len(calls) == 1
    finally:
        release.set()

    assert await asyncio.wait_for(service.warm_up(), timeout=10) is True
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_a_runtime_that_did_not_build_fails_at_once(monkeypatch):
    """Review MINOR 4: no second build attempt inside the search budget."""
    service, builds, release = _cold_library(monkeypatch, build_seconds=0.0)
    monkeypatch.setattr(service_module, "get_shared_rag_service", lambda: None)
    searched: list[str] = []

    async def search(*_args, **_kwargs):
        searched.append("search")
        return {"results": []}

    monkeypatch.setattr(service, "search", search)
    controller, store = _controller_for_preparation(
        _preparation(), service, timeout=SEARCH_BUDGET
    )
    release.set()

    outcome = await controller.prepare_library_for_turn("preparation-1")

    assert outcome.error_code == "library_retrieval_failed"
    assert searched == []
    assert (
        store.preparation_for_session("session-1").pause_kind
        is ConsolePreparationPauseKind.RETRIEVAL
    )


@pytest.mark.asyncio
async def test_a_retry_that_pauses_again_says_why_it_paused_this_time(monkeypatch):
    """Review MINOR 7: the chip kept the first pause's reason after a Retry."""
    from tldw_chatbook.Chat.console_chat_models import ConsoleRunState

    service, _builds, _release = _cold_library(
        monkeypatch, build_seconds=0.0, search_seconds=COLD_BUILD_SECONDS
    )
    controller, store = _controller_for_preparation(
        _preparation(
            state=ConsoleTurnPreparationState.PAUSED,
            pause_kind=ConsolePreparationPauseKind.RETRIEVAL,
        ),
        service,
        timeout=SEARCH_BUDGET,
    )
    controller._set_run_state(
        ConsoleRunState.blocked("Library search stopped"), session_id="session-1"
    )

    result = await controller.retry_library_preparation("preparation-1")

    assert result.accepted is False
    assert result.visible_copy == "Library search timed out"
    assert controller.run_state_for("session-1").visible_copy == (
        "Library search timed out"
    )
    assert (
        store.preparation_for_session("session-1").pause_kind
        is ConsolePreparationPauseKind.RETRIEVAL
    )
