"""A send's durable commit marks its store's commit window while it runs.

TASK-33620.15.4: the Console keeps its whole syncs quiet while a send's
durable turn commit runs on its worker thread (UI-loop work in that window
lengthened the commit and delayed the provider call, measured live
2026-10-23). The Console reads the window from the store
(``console_native_commit.durable_commit_started_at``); these tests pin that
the window is open exactly while a real, file-backed commit runs on its
thread, and closed after success, failure and a cancelled caller alike.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

import pytest

from Tests.Chat.test_console_first_send_atomicity import _controller
from Tests.private_profile import private_profile_test
from tldw_chatbook.Chat import console_native_commit

pytestmark = pytest.mark.bootstrap_profile


def _started_at(store):
    """The store's open commit window, if any (none exists on the base)."""
    read = getattr(console_native_commit, "durable_commit_started_at", None)
    return read(store) if read is not None else None


def _held_commit(monkeypatch, store, *, fail: bool = False):
    """Hold the store's commit on its worker thread until released."""
    entered, release = threading.Event(), threading.Event()
    real = store.commit_durable_turn

    def held(acceptance):
        entered.set()
        assert release.wait(timeout=10), "the test never released the commit"
        if fail:
            raise RuntimeError("injected commit failure")
        return real(acceptance)

    monkeypatch.setattr(store, "commit_durable_turn", held)
    return entered, release


async def _until(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, f"timed out waiting for {predicate}"
        await asyncio.sleep(0.01)


@pytest.mark.parametrize("outcome", ["committed", "failed", "caller_cancelled"])
@private_profile_test
async def test_the_commit_window_is_open_exactly_while_the_commit_runs(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    _db, store, controller, gateway = _controller(tmp_path)
    entered, release = _held_commit(monkeypatch, store, fail=outcome == "failed")
    assert _started_at(store) is None
    task = asyncio.create_task(
        controller.submit_draft("window probe", session_id="session-1")
    )
    try:
        await _until(entered.is_set)
        before = time.monotonic()
        started = _started_at(store)
        assert started is not None, (
            "no commit window was open while the durable commit ran on its thread"
        )
        assert started <= before
        if outcome == "caller_cancelled":
            task.cancel()
            await asyncio.sleep(0.05)
            assert _started_at(store) == started, (
                "a cancelled caller closed the window while its commit still ran"
            )
        release.set()
        result = await asyncio.gather(task, return_exceptions=True)
        assert _started_at(store) is None, "the commit window stayed open"
        if outcome == "committed":
            assert result[0].accepted is True
            assert gateway.calls == 1
        else:
            assert gateway.calls == 0
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
