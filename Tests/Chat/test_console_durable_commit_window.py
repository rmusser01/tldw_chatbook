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


@pytest.mark.parametrize("move", ["close", "replaced_session"])
@private_profile_test
async def test_a_chat_closed_during_its_native_commit_never_reads_as_an_owner_change(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    move: str,
) -> None:
    """TASK-39155: does a plain close reach the native-commit retain path
    ("Saved turn retained for recovery.", BLOCKED / OWNER_CHANGED)? No: the
    close cancels the turn at once, and deletes the chat only after the
    session's ordinary save has drained (``finalize_session_close`` refuses
    before), so the commit's ownership check still finds the chat and the
    turn ends stopped. A successor session under the same id is still an
    ownership change."""
    from loguru import logger

    from tldw_chatbook.Chat.console_chat_models import ConsoleRunStatus

    _db, store, controller, gateway = _controller(tmp_path)
    entered, release = _held_commit(monkeypatch, store)
    attention: list[str] = []
    sink = logger.add(
        lambda message: attention.append(str(message)),
        filter=lambda record: "requires attention" in record["message"],
    )
    task = asyncio.create_task(
        controller.submit_draft("close during commit", session_id="session-1")
    )
    try:
        await _until(entered.is_set)
        ticket = None
        if move == "close":
            # Production's two phases: fence and cancel now, delete once the
            # session's ordinary save has drained (finalize refuses before).
            impact = controller.lifecycle_impact(session_id="session-1")
            ticket = controller.begin_session_close(
                "session-1", expected_revision=impact.revision
            )
        else:
            original = store._sessions["session-1"]
            store._sessions["session-1"] = type(original)(
                id="session-1", title="Successor"
            )
        release.set()
        (result,) = await asyncio.gather(task, return_exceptions=True)
        if ticket is not None:
            controller.finalize_session_close(ticket)
            assert all(row.id != "session-1" for row in store.sessions())
        assert gateway.calls == 0
        if move == "close":
            assert result.visible_copy == "Response stopped.", result
            assert result.blocker is None and result.terminal_status is not (
                ConsoleRunStatus.BLOCKED
            ), result
            assert controller.run_state_for("session-1").status is ConsoleRunStatus.STOPPED
            assert attention == [], attention
        else:
            assert result.terminal_status is ConsoleRunStatus.BLOCKED, result
            assert result.blocker.value == "owner_changed", result
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        logger.remove(sink)


@pytest.mark.parametrize("setup", ["within_grace", "outlasts_grace"])
async def test_a_temporary_chat_closed_during_agent_setup_ends_the_turn_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    owned_console_databases,
    setup: str,
) -> None:
    """TASK-39155: a temporary chat's agent reply sets up (chain creation,
    "Connecting tools…") before the provider. A close whose drain outlasts
    that setup (the runtime's 2 s grace) deletes the chat first, and the
    setup's stop then read the closed chat as an ownership change: BLOCKED /
    OWNER_CHANGED and "Console submit requires attention". It ends as
    TASK-39138's closed session does."""
    from loguru import logger

    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
    from tldw_chatbook.Chat.console_chat_models import (
        ConsoleDispatchRecoveryKind,
        ConsoleRunStatus,
    )
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    db, store, controller, gateway = _controller(tmp_path)
    owned_console_databases(db, controller)
    session = store.create_session(session_id="temporary", title="Temporary", ephemeral=True)
    gateway.deferred_dispatch_boundary = True
    runs = AgentRunsDB(tmp_path / "temporary-runs.sqlite", client_id="temporary-close")
    controller._agent_bridge = ConsoleAgentBridge(
        agent_runs_db=runs, store=store, provider_gateway=gateway
    )
    controller._agent_runtime_enabled = False
    runtime = ConsoleRuntime(app=None)
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    entered, release = threading.Event(), threading.Event()
    original_chain = runs.automatic_work.create_chain

    def chain(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return original_chain(*args, **kwargs)

    monkeypatch.setattr(runs.automatic_work, "create_chain", chain)
    attention: list[str] = []
    sink = logger.add(
        lambda message: attention.append(str(message)),
        filter=lambda record: "requires attention" in record["message"],
    )
    task = asyncio.create_task(controller.submit_draft("temporary close", session_id=session.id))
    try:
        await _until(entered.is_set)
        recovery = store.dispatch_recovery_for_session(session.id)
        assert recovery.kind is ConsoleDispatchRecoveryKind.EPHEMERAL_ACCEPTED
        if setup == "within_grace":
            asyncio.get_running_loop().call_later(0.2, release.set)
        await runtime.close_session(
            session.id,
            expected_revision=controller.lifecycle_impact(session_id=session.id).revision,
            timeout_seconds=5.0 if setup == "within_grace" else 0.2,
        )
        assert all(row.id != session.id for row in store.sessions())
        release.set()
        (result,) = await asyncio.gather(task, return_exceptions=True)
        assert gateway.calls == 0
        assert getattr(result, "visible_copy", result) == "Session closed.", result
        assert result.terminal_status is not ConsoleRunStatus.BLOCKED, result
        assert controller.run_state_for(session.id).status is ConsoleRunStatus.STOPPED
        assert attention == [], attention
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        logger.remove(sink)
