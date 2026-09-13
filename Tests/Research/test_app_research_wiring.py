"""App-level research service wiring (task-16332).

The research wiring used to exist VERBATIM TWICE in app.py: once inside
`_wire_watchlists_and_notifications_services` (the broad parity bootstrap,
which runs first at startup) and once in `_wire_research_services` (which
then early-returned via its already-wired guard). task-16332 replaced the
embedded copy with a call to the method. These tests pin the contract that
made that replacement safe: the boot path wires the full research service
set exactly once, and a second `_wire_research_services()` call never
reconstructs them (the guard holds -- no torn/duplicate wiring).
"""

import asyncio
import threading
import time

import pytest

from Tests.UI.app_factory import _build_test_app


def test_boot_wires_full_research_service_set():
    app = _build_test_app()

    assert app.local_research_service is not None
    assert app.server_research_service is not None
    assert app.research_scope_service is not None
    assert app.local_research_search_service is not None
    assert app.server_research_search_service is not None
    assert app.research_search_scope_service is not None


def test_second_wire_research_services_call_reuses_existing_services():
    app = _build_test_app()

    scope_before = app.research_scope_service
    search_scope_before = app.research_search_scope_service
    local_before = app.local_research_service

    app._wire_research_services()

    # The guard must treat the already-wired state as done: same instances,
    # not fresh reconstructions (a duplicate wiring would silently detach
    # every consumer holding the originals).
    assert app.research_scope_service is scope_before
    assert app.research_search_scope_service is search_scope_before
    assert app.local_research_service is local_before


# ---------------------------------------------------------------------------
# TASK-21127: unmount releases the research store's held connections, and does
# it OFF the event loop.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_app_unmount_close_never_freezes_the_loop_or_breaks_a_run_write(
    tmp_path,
):
    """``close()`` waits for an operation still running on the backend thread.

    Called inline from the async ``on_unmount``, that wait would run ON the
    event loop -- freezing the UI for the whole settle timeout and starving the
    very operation it is waiting for, which then hits a closed connection and
    surfaces as ``Task exception was never retrieved`` (the TASK-21125 review's
    MAJOR-3 finding, reproduced here for the research store).
    """
    import types

    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.Research_Interop.local_research_service import (
        LocalResearchService,
    )

    service = LocalResearchService(tmp_path / "research.db")
    run = service.launch_run(query="unmount")

    entered = threading.Event()
    release = threading.Event()
    failures: list[BaseException] = []

    def _park_then_write():
        try:
            with service._transaction(immediate=True) as conn:
                entered.set()
                assert release.wait(10)
                conn.execute(
                    "UPDATE research_runs SET progress_message = ? WHERE id = ?",
                    ("in flight", run["id"]),
                )
        except BaseException as exc:  # pragma: no cover - failure path
            failures.append(exc)

    worker = threading.Thread(target=_park_then_write)
    worker.start()
    assert entered.wait(10), "the parked transaction never started"

    ticks = 0

    async def _ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.05)
            ticks += 1

    unhandled: list[dict] = []
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: unhandled.append(context))

    app = object.__new__(TldwCli)
    app.loguru_logger = types.SimpleNamespace(error=lambda *_a, **_k: None)
    app.local_research_service = service

    ticker = asyncio.create_task(_ticker())
    try:
        releaser = threading.Timer(0.3, release.set)
        releaser.start()
        started = time.perf_counter()
        await app._close_local_research_service()
        waited = time.perf_counter() - started
        releaser.join(10)
        worker.join(10)
        await asyncio.sleep(0.05)
    finally:
        ticker.cancel()
        loop.set_exception_handler(previous_handler)

    assert not failures, f"the in-flight operation was broken by close(): {failures}"
    assert waited >= 0.25, "close() did not wait for the in-flight operation"
    assert ticks >= 3, f"the event loop was frozen during close() ({ticks} ticks)"
    assert not unhandled, f"an exception escaped to the loop: {unhandled}"
    assert service.get_run(run["id"])["progress_message"] == "in flight"
    service.close()


@pytest.mark.asyncio
async def test_unmount_close_is_a_no_op_without_a_wired_research_service():
    """A service that was never wired must not be constructed just to close it."""
    import types

    from tldw_chatbook.app import TldwCli

    app = object.__new__(TldwCli)
    app.loguru_logger = types.SimpleNamespace(error=lambda *_a, **_k: None)
    app.local_research_service = None

    await app._close_local_research_service()  # must not raise
