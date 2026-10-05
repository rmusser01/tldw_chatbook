"""Quit anyway while a large Console Delete is saving (TASK-33628.5).

The Console's Delete saves its durable half off the event loop, and the
store's fork-source and voice-promotion fences are held from before the save
starts until its result is applied. Ctrl+Q over the receipt first says the
delete is still being saved; a second Ctrl+Q asks "Quit while still
working?", and Quit anyway goes on to the app's teardown while the save may
still be running.

The final gate on the branch that moved the save off the loop found what
that did: the held voice-promotion admission made
``ConsoleChatStore.end_app_runtime`` refuse ("Voice promotion state prevents
store replacement."), and ``ConsoleRuntime.dispose`` logged it and skipped
the whole store teardown -- the trace-settlement drain, the stream and
trace-settlement executor shutdowns and the teardown retries. Before that
branch the delete ran inline, so nothing could be in flight at quit.

Teardown now waits a bounded time for the save, which releases the
admission, and then runs the whole store teardown. If the save outlasts the
bound, it logs that without content and still runs every teardown step that
does not replace the store's state.

Both cases drive the real ``TldwCli`` (real Console runtime, controller and
store) over a real FILE-backed database -- a ``:memory:`` one is
thread-local, so the save would run inline -- press the real Ctrl+Q twice
and answer the real question. The durable subtree write is held at the
database call; on-disk state is read back with a fresh ``sqlite3``
connection after the app has gone.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from loguru import logger
from textual.widgets import Button

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_app_quit_in_flight_modals import (
    _click_when_shown,
    _still_working_notices,
)
from Tests.UI.test_app_quit_under_modal import (
    _dialogs_titled,
    _mounted_console,
    _until,
)
from Tests.UI.test_console_message_delete_off_loop import _REMOVED, _seed
from Tests.UI.test_console_message_delete_undo import _painted, _tree_nodes
from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from tldw_chatbook.Chat import console_runtime
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Console_Modules.message_delete import (
    handle_console_delete_action,
)
from tldw_chatbook.Widgets.Console import ConsoleTranscript

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.asyncio]

_QUESTION = "Quit while still working?"
#: How long the gated save keeps running once the Console's teardown has
#: begun, standing in for a slow disk. Well inside the teardown's bound.
_STILL_SAVING_SECONDS = 1.0
#: The teardown steps every app exit must run on the store, in order. The
#: first is the voice-fenced scope that replaces the store's volatile state.
_FULL_TEARDOWN = [
    "state-swap",
    "settle-trace-settlements",
    "drain-trace-settlements",
    "close-trace-executor",
    "retry-trace-settlements",
]


class _HeldSave:
    """Hold the durable subtree delete, off the loop, until told to go.

    ``until`` is polled from the save's own thread; the real write runs
    ``then`` seconds after it first holds. ``abort`` lets a failing test
    release the thread.
    """

    def __init__(
        self,
        db: CharactersRAGDB,
        loop_thread: int,
        until: Any,
        then: float = 0.0,
    ) -> None:
        self.entered = threading.Event()
        self.abort = threading.Event()
        self.thread: int | None = None
        real = db.soft_delete_message_subtree

        def held(*args: Any, **kwargs: Any) -> Any:
            self.thread = threading.get_ident()
            self.entered.set()
            assert self.thread != loop_thread, "the delete saved on the event loop"
            deadline = time.monotonic() + 120
            while not until() and not self.abort.is_set():
                assert time.monotonic() < deadline, "the held save was never released"
                time.sleep(0.01)
            time.sleep(then)
            return real(*args, **kwargs)

        db.soft_delete_message_subtree = held


def _record_teardown(store: Any) -> list[str]:
    """Record each store teardown step as it runs (it runs off the loop)."""
    steps: list[str] = []
    real_scope = store._voice_promotion_state_replacement_scope

    @contextmanager
    def state_swap():
        with real_scope():
            steps.append("state-swap")
            yield

    store._voice_promotion_state_replacement_scope = state_swap
    for name, step in (
        ("_settle_all_provider_trace_settlements", "settle-trace-settlements"),
        (
            "_drain_retained_provider_trace_settlements_on_teardown",
            "drain-trace-settlements",
        ),
        ("_close_provider_trace_settlement_executor", "close-trace-executor"),
        (
            "_retry_failed_provider_trace_settlements_on_teardown",
            "retry-trace-settlements",
        ),
    ):
        real = getattr(store, name)

        def recorded(*args: Any, _real: Any = real, _step: str = step, **kwargs: Any):
            steps.append(_step)
            return _real(*args, **kwargs)

        setattr(store, name, recorded)
    return steps


@contextmanager
def _warnings():
    """Collect loguru WARNING-and-above messages while the block runs."""
    messages: list[str] = []
    handler = logger.add(
        lambda m: messages.append(m.record["message"]), level="WARNING"
    )
    try:
        yield messages
    finally:
        logger.remove(handler)


def _saved_flags(path: Path, conversation_id: str) -> dict[str, int]:
    """Read each message's ``deleted`` flag straight from the database file."""
    with sqlite3.connect(path) as conn:
        rows = conn.execute(
            "SELECT id, deleted FROM messages WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchall()
    return {message_id: int(deleted) for message_id, deleted in rows}


async def _plain_until(predicate: Any, what: str, timeout: float = 30.0) -> None:
    """Wait with plain sleeps: once the app is exiting, Pilot's waits raise."""
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        await asyncio.sleep(0.02)


def _admitted(store: Any) -> bool:
    """Whether any store mutation still holds its promotion admission."""
    return bool(store._voice_promotion_mutation_admissions) or bool(
        store._fork_source_transitions
    )


async def _quit_anyway_mid_delete(
    app: Any, pilot: Any, db_path: Path, conversation_id: str, hold: Any
) -> dict[str, Any]:
    """Open the seeded chat, confirm its 3,002-message Delete, Quit anyway.

    Returns once Quit anyway has exited the app's message loop; leaving
    ``run_test`` then runs the app's shutdown, as ``App.run`` does after its
    loop ends. ``hold(db, loop_thread)`` installs the held save just before
    the confirm.
    """
    console = await _mounted_console(app, pilot)
    db = app.chachanotes_db
    store = console._ensure_console_chat_store()
    session = store.restore_persisted_session(
        title="Delete off loop",
        workspace_id=None,
        persisted_conversation_id=conversation_id,
        all_nodes=_tree_nodes(db, conversation_id),
        active_leaf_persisted_id="a2",
    )
    await console._sync_native_console_chat_ui()
    native = {
        node.persisted_message_id: node.id
        for node in store._nodes_by_session[session.id].values()
    }
    target = native["u2"]
    transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
    await _until(
        pilot,
        lambda: any(message.id == target for message in transcript._messages),
        "the transcript to ingest the opened conversation",
    )
    transcript.select_message(target)
    await _until(
        pilot, lambda: transcript.selected_message_id == target, "the row selected"
    )
    await handle_console_delete_action(console._message, "delete", target)
    confirm = f"#console-message-action-delete-confirm-{target}"
    await _until(pilot, lambda: bool(console.query(confirm)), "the armed confirm")
    assert f"Delete {_REMOVED} messages" in str(
        console.query_one(confirm, Button).label
    )

    save = hold(db, threading.get_ident())
    steps = _record_teardown(store)
    console.query_one(confirm, Button).press()
    await _until(pilot, save.entered.is_set, "the durable delete to start")
    await _until(
        pilot,
        lambda: f"Deleting {_REMOVED} messages" in _painted(app),
        "the receipt's in-progress state",
    )
    assert _admitted(store), "the save no longer holds the store's fences"

    # The first Ctrl+Q stays and says the delete is still being saved.
    await pilot.press("ctrl+q")
    await _until(
        pilot,
        lambda: (
            bool(_still_working_notices(app)) or bool(_dialogs_titled(app, _QUESTION))
        ),
        "the first Ctrl+Q to answer",
    )
    notices = _still_working_notices(app)
    assert len(notices) == 1 and "delete is still being saved" in notices[0], notices
    assert not _dialogs_titled(app, _QUESTION), "the first Ctrl+Q asked to quit"
    await _until(
        pilot, lambda: app._quit_in_progress is False, "the first Ctrl+Q to end"
    )
    assert app._shutting_down is False

    # The second asks; Quit anyway goes on while the save is still held.
    await pilot.press("ctrl+q")
    await _until(
        pilot, lambda: bool(_dialogs_titled(app, _QUESTION)), "the quit question"
    )
    assert not save.abort.is_set() and _admitted(store)
    assert set(_saved_flags(db_path, conversation_id).values()) == {0}
    assert await _click_when_shown(app, pilot, "#confirm-button")
    await _plain_until(lambda: app._exit, "Quit anyway to exit the app")
    assert app._console_runtime_shutdown_task is None, "teardown began early"
    return {"store": store, "steps": steps}


def _app_over(tmp_path: Path) -> tuple[Any, Path, str, list[str]]:
    """Build the real app over a seeded file-backed database."""
    db_path = tmp_path / "quit-mid-delete.db"
    db = CharactersRAGDB(db_path, "quit-mid-delete")
    conversation_id, persisted, _removed = _seed(db)
    app = _build_test_app(configured_default="chat")
    _configure_native_ready_console(app)
    app.chachanotes_db = db
    return app, db_path, conversation_id, persisted


def _assert_atomic(db_path: Path, conversation_id: str, persisted: list[str]) -> None:
    """The delete committed in full: every subtree row, and nothing else."""
    flags = _saved_flags(db_path, conversation_id)
    assert [flags[message_id] for message_id in persisted] == [0, 0] + [1] * _REMOVED


async def test_quit_anyway_mid_delete_waits_for_the_save_then_tears_down_fully(
    tmp_path,
):
    """Teardown waits for the save, then runs every store teardown step."""
    held: dict[str, _HeldSave] = {}

    def hold(db, loop_thread):
        # Held until the Console runtime's teardown has begun, then still
        # saving for a while: teardown reaches the store with it in flight.
        held["save"] = _HeldSave(
            db,
            loop_thread,
            until=lambda: app._console_runtime_shutdown_task is not None,
            then=_STILL_SAVING_SECONDS,
        )
        return held["save"]

    app, db_path, conversation_id, persisted = _app_over(tmp_path)
    with _warnings() as warnings:
        try:
            async with app.run_test(size=(160, 48)) as pilot:
                seen = await _quit_anyway_mid_delete(
                    app, pilot, db_path, conversation_id, hold
                )
        finally:
            if "save" in held:
                held["save"].abort.set()

    store = seen["store"]
    assert not [w for w in warnings if "shutdown failed at dispose" in w], warnings
    assert seen["steps"] == _FULL_TEARDOWN, seen["steps"]
    assert store._provider_trace_settlement_registration_closed is True
    assert store._stream_persistence_executor_closed is True
    assert store._stream_persistence_executor._shutdown is True
    # Teardown waited for the save: applied, fences released, on disk in full.
    assert not _admitted(store), "the app went before the delete was applied"
    _assert_atomic(db_path, conversation_id, persisted)
    assert not [w for w in warnings if "still saving" in w], warnings


async def test_a_save_past_the_teardown_bound_still_runs_every_other_step(
    tmp_path, monkeypatch
):
    """Past the bound: a content-free warning, and no teardown step skipped
    except the voice-fenced state swap; the save still lands atomically."""
    monkeypatch.setattr(console_runtime, "CONSOLE_DURABLE_WRITE_TEARDOWN_SECONDS", 0.3)
    release = threading.Event()
    held: dict[str, _HeldSave] = {}

    def hold(db, loop_thread):
        held["save"] = _HeldSave(db, loop_thread, until=release.is_set)
        return held["save"]

    app, db_path, conversation_id, persisted = _app_over(tmp_path)
    with _warnings() as warnings:
        try:
            async with app.run_test(size=(160, 48)) as pilot:
                seen = await _quit_anyway_mid_delete(
                    app, pilot, db_path, conversation_id, hold
                )
            store = seen["store"]
            # The app has gone with the save still held: nothing saved yet.
            assert set(_saved_flags(db_path, conversation_id).values()) == {0}
            assert _admitted(store)
            steps = list(seen["steps"])
        finally:
            release.set()
            if "save" in held:
                held["save"].abort.set()
        await _plain_until(lambda: not _admitted(store), "the save to be applied")

    _assert_atomic(db_path, conversation_id, persisted)
    assert steps == _FULL_TEARDOWN[1:], steps
    assert store._provider_trace_settlement_registration_closed is True
    assert store._stream_persistence_executor_closed is True
    assert store._stream_persistence_executor._shutdown is True
    still_saving = [w for w in warnings if "still saving" in w]
    assert len(still_saving) == 1, warnings
    for private in (conversation_id, "u2", "x0000", "text"):
        assert private not in still_saving[0], still_saving[0]
    assert not [w for w in warnings if "shutdown failed at dispose" in w], warnings
