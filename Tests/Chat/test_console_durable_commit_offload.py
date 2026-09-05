"""TASK-22205: the per-send durable commit must not block the event loop.

Three probes:

(a) event-loop-stall — a second connection holds ``BEGIN IMMEDIATE`` while a
    send is submitted; the event loop must stay responsive (the commit waits
    on a worker thread, not on the loop).
(b) ordering — provider dispatch must not begin before the durable commit is
    visible to an independent connection (committed, not dirty-read), and the
    checkpoint must already be in ``dispatch_started``.
(c) restore reconcile — a reconcile with nothing to write must not take the
    SQLite write lock (no ``BEGIN IMMEDIATE`` in the connection trace); a
    reconcile that DOES have a write to make still makes it.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from Tests.Chat.test_console_durable_turn_acceptance import _ready_store
from Tests.Chat.test_console_first_send_atomicity import (
    _CheckpointObservingGateway,
    _controller,
)
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleDispatchRecoveryKind,
    ConsoleRunStatus,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


@pytest.mark.parametrize("after_commit", [False, True])
@pytest.mark.parametrize("agent_path", [False, True])
async def test_stop_during_dispatch_cas_settles_before_returning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    after_commit: bool,
    agent_path: bool,
) -> None:
    """Stop must drain the actual SQLite handoff before terminal settlement."""
    db, store, controller, gateway = _controller(tmp_path)
    store.create_session(session_id="session-2", title="Background")
    background_entered = asyncio.Event()
    background_release = asyncio.Event()
    original_stream = gateway.stream_chat

    async def background_stream(*args, **kwargs):
        async for chunk in original_stream(*args, **kwargs):
            yield chunk
        background_entered.set()
        await background_release.wait()

    monkeypatch.setattr(gateway, "stream_chat", background_stream)
    background = asyncio.create_task(
        controller.submit_draft("keep running", session_id="session-2")
    )
    await asyncio.wait_for(background_entered.wait(), 5)
    background_owner = store.dispatch_recovery_for_session("session-2")
    background_cancel = controller._active_cancel_events["session-2"]
    store.switch_session("session-1")
    agent_run = Mock(side_effect=AssertionError("Stopped turn entered the agent"))
    if agent_path:
        controller._agent_bridge = SimpleNamespace(run_reply=agent_run)
    repository = store.persistence.console_dispatch_repository
    original_cas = repository.cas_state
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def gated_cas(transition):
        try:
            result = original_cas(transition) if after_commit else None
            entered.set()
            assert release.wait(timeout=10)
            return result if after_commit else original_cas(transition)
        finally:
            finished.set()

    monkeypatch.setattr(repository, "cas_state", gated_cas)
    submit = asyncio.create_task(
        controller.submit_draft("stop before first token", session_id="session-1")
    )
    try:
        try:
            async with asyncio.timeout(5):
                while not entered.is_set():
                    await asyncio.sleep(0.01)
            assert controller.run_state.status is ConsoleRunStatus.STREAMING
            assert controller.stop_active_run()
            await asyncio.sleep(0)
            # Repeated Stop cannot cancel the draining transaction's owner.
            controller.stop_active_run()
            await asyncio.sleep(0)
        finally:
            release.set()
        await asyncio.wait_for(submit, timeout=5)
        assert await asyncio.to_thread(finished.wait, 5)
        assert gateway.calls == 1  # Only the unrelated background session entered.
        agent_run.assert_not_called()
        assert controller.run_state.status is ConsoleRunStatus.STOPPED
        assert ConsoleRunStatus.BLOCKED not in controller.run_state_history
        assert store.dispatch_recovery_for_session("session-2") == background_owner
        assert not background_cancel.is_set()
        assert not background.done()
        assert [
            message.content for message in store.messages_for_session("session-1")
        ].count("Response stopped by user.") == 1
        assert not controller._pending_dispatch_transitions
        assert not controller._deferred_user_stop_markers
        with sqlite3.connect(tmp_path / "controller.sqlite") as fresh:
            assert fresh.execute(
                "SELECT assistant_generation_state, content FROM messages "
                "WHERE role = 'assistant' ORDER BY assistant_generation_state"
            ).fetchall() == [("dispatch_started", ""), ("stopped", "")]
            assert fresh.execute(
                "SELECT assistant_message_id FROM console_dispatch_checkpoints"
            ).fetchall() == [(background_owner.assistant_message_id,)]
        background_release.set()
        await asyncio.wait_for(background, 5)
        with sqlite3.connect(tmp_path / "controller.sqlite") as fresh:
            assert (
                fresh.execute(
                    "SELECT COUNT(*) FROM console_dispatch_checkpoints"
                ).fetchone()[0]
                == 0
            )

    finally:
        release.set()
        background_release.set()
        await asyncio.gather(submit, background, return_exceptions=True)
        db.close()


class _CommitVisibilityGateway(_CheckpointObservingGateway):
    """Record, at first stream call, what an INDEPENDENT connection can see."""

    def __init__(self, db: CharactersRAGDB, db_path: str) -> None:
        super().__init__(db)
        self.db_path = db_path
        self.visible_at_stream: list[dict[str, Any]] = []

    async def stream_chat(
        self, resolution: object, messages: list[dict[str, Any]], **kwargs: Any
    ):
        fresh = sqlite3.connect(self.db_path)
        try:
            fresh.row_factory = sqlite3.Row
            users = fresh.execute(
                "SELECT content FROM messages WHERE role = 'user'"
            ).fetchall()
            checkpoints = fresh.execute(
                "SELECT state FROM console_dispatch_checkpoints"
            ).fetchall()
        finally:
            fresh.close()
        self.visible_at_stream.append(
            {
                "user_contents": [row["content"] for row in users],
                "checkpoint_states": [row["state"] for row in checkpoints],
            }
        )
        async for chunk in super().stream_chat(resolution, messages, **kwargs):
            yield chunk


def _hold_write_lock(
    db_path: str,
    hold_seconds: float,
    acquired: threading.Event,
) -> None:
    connection = sqlite3.connect(db_path, timeout=5)
    try:
        connection.execute("BEGIN IMMEDIATE")
        acquired.set()
        time.sleep(hold_seconds)
        connection.commit()
    finally:
        connection.close()


async def test_send_does_not_stall_event_loop_while_write_lock_is_held(
    tmp_path: Path,
) -> None:
    """Probe (a): loop stays responsive while the durable commit waits."""

    db, store, controller, gateway = _controller(tmp_path)
    db_path = str(tmp_path / "controller.sqlite")
    hold_seconds = 2.0

    acquired = threading.Event()
    blocker = threading.Thread(
        target=_hold_write_lock,
        args=(db_path, hold_seconds, acquired),
        daemon=True,
    )
    blocker.start()
    assert acquired.wait(timeout=5)

    stalls: list[float] = []
    stop = asyncio.Event()

    async def heartbeat() -> None:
        last = time.monotonic()
        while not stop.is_set():
            await asyncio.sleep(0.01)
            now = time.monotonic()
            stalls.append(now - last)
            last = now

    monitor = asyncio.create_task(heartbeat())
    # Let the heartbeat actually start ticking BEFORE the send: a task that
    # has not reached its first ``await`` yet cannot observe a stall.
    await asyncio.sleep(0.05)
    assert stalls, "heartbeat must be running before the send starts"
    started = time.monotonic()
    result = await controller.submit_draft("stall probe", session_id="session-1")
    elapsed = time.monotonic() - started
    stop.set()
    await monitor
    blocker.join(timeout=5)

    max_stall = max(stalls) if stalls else 0.0
    # The send itself must have waited for the artificial lock holder --
    # otherwise the probe proved nothing about contention.
    assert elapsed >= hold_seconds * 0.5, (
        f"send finished in {elapsed:.3f}s; the 2s lock holder never contended"
    )
    assert result.accepted is True
    assert gateway.calls == 1
    # The event loop must never have been blocked for a contention-scale
    # interval: the commit waits on a worker thread, not on the loop.
    assert max_stall < 0.5, (
        f"event loop stalled {max_stall:.3f}s during the send "
        f"(send took {elapsed:.3f}s under a {hold_seconds}s write-lock holder)"
    )


async def test_dispatch_begins_only_after_commit_is_durably_visible(
    tmp_path: Path,
) -> None:
    """Probe (b): the provider stream starts only after the durable commit.

    Visibility is checked from an INDEPENDENT connection, so only committed
    data counts. This is the ordering barrier's regression guard: under the
    mutation "dispatch no longer waits for the commit" it must fail.
    """

    db = CharactersRAGDB(tmp_path / "controller.sqlite", client_id="task14-test")
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore

    store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    store.create_session(session_id="session-1", title="Chat 1")
    gateway = _CommitVisibilityGateway(db, str(tmp_path / "controller.sqlite"))
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        provider="llama_cpp",
        model="test-model",
    )

    result = await controller.submit_draft(
        "ordering probe draft", session_id="session-1"
    )

    assert result.accepted is True
    assert gateway.calls == 1
    assert gateway.visible_at_stream == [
        {
            "user_contents": ["ordering probe draft"],
            "checkpoint_states": ["dispatch_started"],
        }
    ]


async def test_cancel_during_offloaded_commit_leaves_consistent_state(
    tmp_path: Path,
) -> None:
    """Shutdown-path walk: cancelling a send mid-commit corrupts nothing.

    ``asyncio.to_thread`` survives task cancellation, so the commit thread
    runs its single transaction to completion. The DB must end atomically
    consistent (the whole turn, or nothing), the provider must never have
    been called, no unretrieved-exception noise may leak, and the restore
    reconcile must recognize the committed checkpoint — the same
    crash-window recovery the checkpoint machinery already owns.
    """

    import gc

    db, store, controller, gateway = _controller(tmp_path)
    db_path = str(tmp_path / "controller.sqlite")

    acquired = threading.Event()
    release = threading.Event()

    def hold() -> None:
        connection = sqlite3.connect(db_path, timeout=5)
        try:
            connection.execute("BEGIN IMMEDIATE")
            acquired.set()
            release.wait(timeout=10)
            connection.commit()
        finally:
            connection.close()

    blocker = threading.Thread(target=hold, daemon=True)
    blocker.start()
    assert acquired.wait(timeout=5)

    loop = asyncio.get_running_loop()
    unhandled: list[dict[str, Any]] = []
    loop.set_exception_handler(lambda _loop, context: unhandled.append(context))
    try:
        task = asyncio.create_task(
            controller.submit_draft("cancelled mid-commit", session_id="session-1")
        )
        # Wait until the durable commit is genuinely in flight (reservation
        # registered; the worker thread is blocked on the held write lock)
        # so the cancellation deterministically lands mid-commit.
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not store._durable_commit_in_flight:
            await asyncio.sleep(0.01)
        assert store._durable_commit_in_flight, (
            "the durable commit never started; the probe cancelled too early"
        )
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        blocker.join(timeout=5)
        # Give the surviving commit thread time to finish its transaction.
        fresh = sqlite3.connect(db_path, timeout=5)
        fresh.row_factory = sqlite3.Row
        try:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                checkpoints = fresh.execute(
                    "SELECT COUNT(*) FROM console_dispatch_checkpoints"
                ).fetchone()[0]
                if checkpoints:
                    break
                await asyncio.sleep(0.05)
            counts = {
                table: fresh.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in (
                    "conversations",
                    "messages",
                    "console_dispatch_checkpoints",
                )
            }
        finally:
            fresh.close()
        del task
        gc.collect()
        await asyncio.sleep(0.05)
    finally:
        loop.set_exception_handler(None)

    assert gateway.calls == 0, "dispatch must never precede a settled commit"
    assert unhandled == [], f"cancellation leaked loop exceptions: {unhandled!r}"
    # The surviving thread committed the whole turn atomically: the exact
    # crash-window state (commit durable, dispatch never started).
    assert counts["console_dispatch_checkpoints"] == 1
    assert counts["conversations"] == 1
    assert counts["messages"] == 2
    repository = store.persistence.console_dispatch_repository
    conversation_id = (
        db.get_connection().execute("SELECT id FROM conversations").fetchone()["id"]
    )
    state = repository.reconcile_for_session(conversation_id)
    assert state is not None, (
        "restore reconcile must surface the committed-but-undispatched "
        "turn as a recovery owner"
    )
    assert state.kind is not ConsoleDispatchRecoveryKind.QUARANTINED


def _traced_statements(db: CharactersRAGDB) -> list[str]:
    statements: list[str] = []
    db.get_connection().set_trace_callback(statements.append)
    return statements


def test_clean_restore_reconcile_takes_no_write_lock(tmp_path: Path) -> None:
    """Probe (c): a reconcile with nothing to write must use a read txn."""

    db, service, store, _preparation, acceptance = _ready_store(tmp_path)
    repository = service.console_dispatch_repository
    commit = store.commit_durable_turn(acceptance)
    conversation_id = commit.identity.conversation_id
    # Settle the turn out-of-band so the checkpoint is gone and the restore
    # is clean: terminal assistant + no checkpoint = nothing to reconcile.
    with db.transaction(immediate=True) as cursor:
        cursor.execute(
            "UPDATE messages SET assistant_generation_state = 'complete', "
            "content = 'done' WHERE id = ?",
            (commit.assistant_message_id,),
        )
        cursor.execute(
            "DELETE FROM console_dispatch_checkpoints WHERE assistant_message_id = ?",
            (commit.assistant_message_id,),
        )

    statements = _traced_statements(db)
    try:
        state = repository.reconcile_for_session(conversation_id)
    finally:
        db.get_connection().set_trace_callback(None)

    assert state is None
    begins = [s for s in statements if s.strip().upper().startswith("BEGIN")]
    assert begins, "reconcile must still read under a transaction"
    assert not any("IMMEDIATE" in s.upper() for s in begins), (
        f"clean restore reconcile took the write lock: {begins!r}"
    )


def test_valid_checkpoint_reconcile_takes_no_write_lock(tmp_path: Path) -> None:
    """Probe (c) variant: a valid recovery owner is read, never written."""

    db, service, store, _preparation, acceptance = _ready_store(tmp_path)
    repository = service.console_dispatch_repository
    commit = store.commit_durable_turn(acceptance)
    conversation_id = commit.identity.conversation_id

    statements = _traced_statements(db)
    try:
        state = repository.reconcile_for_session(conversation_id)
    finally:
        db.get_connection().set_trace_callback(None)

    assert state is not None
    assert state.kind is not ConsoleDispatchRecoveryKind.QUARANTINED
    assert state.assistant_message_id == commit.assistant_message_id
    begins = [s for s in statements if s.strip().upper().startswith("BEGIN")]
    assert begins, "reconcile must still read under a transaction"
    assert not any("IMMEDIATE" in s.upper() for s in begins), (
        f"read-only reconcile took the write lock: {begins!r}"
    )


def test_terminal_checkpoint_reconcile_still_deletes_under_write_lock(
    tmp_path: Path,
) -> None:
    """A reconcile that has a write to make still makes it (write txn)."""

    db, service, store, _preparation, acceptance = _ready_store(tmp_path)
    repository = service.console_dispatch_repository
    commit = store.commit_durable_turn(acceptance)
    conversation_id = commit.identity.conversation_id
    # Terminal assistant state with a lingering checkpoint row: reconcile
    # must delete the checkpoint -- a real write.
    with db.transaction(immediate=True) as cursor:
        cursor.execute(
            "UPDATE messages SET assistant_generation_state = 'complete', "
            "content = 'done' WHERE id = ?",
            (commit.assistant_message_id,),
        )

    statements = _traced_statements(db)
    try:
        state = repository.reconcile_for_session(conversation_id)
    finally:
        db.get_connection().set_trace_callback(None)

    assert state is None
    assert any("IMMEDIATE" in s.upper() for s in statements), (
        "the write pass must take the write lock up front (no deferred "
        "read-then-write upgrade)"
    )
    remaining = (
        db.get_connection()
        .execute("SELECT COUNT(*) FROM console_dispatch_checkpoints")
        .fetchone()[0]
    )
    assert remaining == 0
