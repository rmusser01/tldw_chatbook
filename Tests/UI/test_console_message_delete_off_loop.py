"""A large Console Delete and its Undo keep the event loop running (TASK-33628.5).

Before this task the confirmed Delete called ``store.delete_message`` inline
on the UI loop, and Undo called the durable undelete inline too. On a
file-backed database (dev 7d155170dc) the durable halves alone held the loop
for 0.45-1.1 s (delete) and 1.0-1.8 s (Undo) at 3,000 messages, with
nothing on screen saying anything was happening.

Each case drives the real ``ChatScreen`` with the real Console store over a
real FILE-backed ChaChaNotes database -- a ``:memory:`` database is
thread-local, so it would hide exactly the hand-off under test. The
selected message has a 3,000-message branch beneath it that is not on the
active path, so the transcript stays four rows long and what is measured is
the delete, not the transcript.

The durable write is held at a gate inside the database call. While it is
held, the event loop must keep running (a heartbeat counts its turns), the
receipt must say what is still in progress, and Escape must not close it.
The store is UI-thread state, so the move must keep the store's fences
(fork snapshots are refused, a voice promotion cannot claim the session)
until the result is applied, and the recovered-media hold -- per-thread --
must be opened in the thread that tombstones.
"""

from __future__ import annotations

import asyncio
import threading
import time
from contextlib import contextmanager
from typing import Any

import pytest
from textual.widgets import Button

from Tests.UI.test_console_message_delete_undo import (
    _deleted_flags,
    _painted,
    _tree_nodes,
    _tree_shape,
)
from Tests.UI.test_console_native_chat_flow import _wait_for_selector
from Tests.UI.test_destination_shells import _build_test_app
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Console_Modules.message_delete import (
    handle_console_delete_action,
)
from tldw_chatbook.Widgets.Console import ConsoleTranscript

# The real ChatScreen/store goes through config-participant admission, which
# the per-test sandbox refuses (RecoveryRequired); keep the collection-time
# profile.
pytestmark = pytest.mark.bootstrap_profile

#: Off-branch messages under the selected one.
_BRANCH = 3_000
#: What one confirmed Delete removes: the selected prompt, its reply on the
#: active path, and the whole off-branch chain.
_REMOVED = _BRANCH + 2


async def _until(predicate: Any, what: str, *, timeout: float = 60.0) -> None:
    """Poll on the event loop without Pilot's per-widget pause traffic."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f"timed out waiting for {what}")


class _Heartbeat:
    """Count event-loop turns; a blocked loop stops counting."""

    def __init__(self) -> None:
        self.ticks = 0
        self._stop = False
        self._task: asyncio.Task | None = None

    async def _run(self) -> None:
        while not self._stop:
            self.ticks += 1
            await asyncio.sleep(0.01)

    def start(self) -> None:
        self._task = asyncio.get_running_loop().create_task(self._run())

    async def stop(self) -> None:
        self._stop = True
        if self._task is not None:
            await self._task


class _Gate:
    """Hold one database method, off the loop only, until released.

    On the event-loop thread it never waits: that would deadlock the loop the
    test runs on, and the thread it ran on is the finding.
    """

    def __init__(self, db: CharactersRAGDB, name: str, loop_thread: int) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.thread: int | None = None
        real = getattr(db, name)

        def held(*args: Any, **kwargs: Any) -> Any:
            self.thread = threading.get_ident()
            self.entered.set()
            if self.thread != loop_thread:
                assert self.release.wait(60), f"{name} gate never released"
            return real(*args, **kwargs)

        setattr(db, name, held)


class _HoldSpy:
    """Record the recovered-media hold and every reference release.

    ``releases`` lists, per release attempt, its ids and whether a hold was
    open in the releasing thread (so the release was held back, not run).
    """

    def __init__(self, persistence: Any) -> None:
        self.threads: list[int] = []
        self.held: list[str] = []
        self.releases: list[tuple[tuple[str, ...], bool]] = []
        real_hold = persistence.hold_recovered_media_release
        real_release = persistence._release_recovered_messages

        @contextmanager
        def spied_hold():
            self.threads.append(threading.get_ident())
            with real_hold() as held:
                yield held
            self.held.extend(held)

        def spied_release(message_ids):
            diverted = (
                getattr(persistence._held_recovered_releases, "ids", None) is not None
            )
            self.releases.append((tuple(message_ids), diverted))
            return real_release(message_ids)

        persistence.hold_recovered_media_release = spied_hold
        persistence._release_recovered_messages = spied_release


def _fenced(store: Any, session_id: str, probe_message_id: str) -> bool:
    """Whether the store's fork-source and voice-promotion fences are held."""
    eligibility = store.fork_eligibility(probe_message_id)
    transition = bool(store._fork_source_transitions.get(session_id))
    admitted = bool(store._voice_promotion_mutation_admissions.get(session_id))
    assert transition == admitted, (transition, admitted)
    # A fork snapshot is refused as "source is changing" exactly while fenced
    # (it may still be refused for other reasons, such as an unloaded policy).
    refused_as_changing = (
        not eligibility.eligible and "changing" in eligibility.reason.lower()
    )
    assert refused_as_changing is transition, eligibility
    return transition


def _seed(db: CharactersRAGDB) -> tuple[str, list[str], list[str]]:
    """Save u1/a1/u2/a2 (active) plus a 3,000-row branch under u2."""
    conversation_id = ChatConversationService(db).create_conversation(
        id="delete-off-loop",
        title="Delete off loop",
        scope_type="global",
        state="in-progress",
    )
    rows: list[tuple[str, str, str | None]] = [
        ("u1", "user", None),
        ("a1", "assistant", "u1"),
        ("u2", "user", "a1"),
        ("a2", "assistant", "u2"),
    ]
    parent = "u2"
    for index in range(_BRANCH):
        message_id = f"x{index:04d}"
        rows.append((message_id, "user" if index % 2 == 0 else "assistant", parent))
        parent = message_id
    with db.transaction():
        for index, (message_id, role, parent_id) in enumerate(rows):
            db.add_message(
                {
                    "id": message_id,
                    "conversation_id": conversation_id,
                    "parent_message_id": parent_id,
                    "sender": role,
                    "role": role,
                    "content": f"{message_id} text",
                    "timestamp": (
                        f"2026-09-30T{index // 3600:02d}:{index // 60 % 60:02d}:"
                        f"{index % 60:02d}.000000+00:00"
                    ),
                }
            )
    assert db.set_conversation_active_cursor(
        conversation_id, active_leaf_message_id="a2", before_message_id=None
    )
    persisted = [message_id for message_id, _role, _parent in rows]
    return conversation_id, persisted, persisted[2:]


@pytest.mark.asyncio
async def test_a_3000_message_delete_and_undo_keep_the_event_loop_running(tmp_path):
    db = CharactersRAGDB(tmp_path / "delete-off-loop.db", "delete-off-loop")
    conversation_id, persisted, removed = _seed(db)
    assert len(removed) == _REMOVED
    app = _build_test_app()
    app.chachanotes_db = db
    notices: list[str] = []
    app.notify = lambda message, **kwargs: notices.append(str(message))
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        loop_thread = threading.get_ident()
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
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
        before = _tree_shape(store, session.id)
        target = native["u2"]
        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        await _until(
            lambda: any(message.id == target for message in transcript._messages),
            "the transcript to ingest the opened conversation",
        )

        # Arm through the controller the More menu dispatches to (the menu
        # path itself is pinned by test_console_message_delete_undo.py).
        transcript.select_message(target)
        await _until(
            lambda: transcript.selected_message_id == target, "the row to be selected"
        )
        await handle_console_delete_action(console._message, "delete", target)
        confirm = f"#console-message-action-delete-confirm-{target}"
        # Arming a 3,002-message Delete blocks the loop 0.2-0.8 s even on a
        # quiet machine (TASK-33628.5.2); the helper's 2 s default flaked
        # under load. This wait is not what the test measures.
        await _wait_for_selector(console, pilot, confirm, timeout=30.0)
        assert f"Delete {_REMOVED} messages" in str(
            console.query_one(confirm, Button).label
        )

        delete_gate = _Gate(db, "soft_delete_message_subtree", loop_thread)
        restore_gate = _Gate(db, "restore_message_subtree", loop_thread)
        hold = _HoldSpy(store.persistence)
        assert not _fenced(store, session.id, native["a1"])
        heartbeat = _Heartbeat()
        heartbeat.start()
        try:
            # --- Delete --------------------------------------------------
            console.query_one(confirm, Button).press()
            await _until(delete_gate.entered.is_set, "the durable delete to start")
            assert delete_gate.thread != loop_thread, (
                "the durable subtree delete ran on the event-loop thread"
            )
            ticks = heartbeat.ticks
            await _until(
                lambda: f"Deleting {_REMOVED} messages" in _painted(host),
                "the in-progress receipt to be painted",
            )
            await _until(
                lambda: heartbeat.ticks >= ticks + 20,
                "the event loop to keep turning while the delete is held",
            )
            # Nothing is final yet: no receipt, no Undo, the rows are live,
            # the store still fences the session, and Escape does not walk
            # away from the write.
            assert _fenced(store, session.id, native["a1"])
            assert hold.threads == [delete_gate.thread]
            assert not host.screen.query("#console-delete-receipt")
            assert set(_deleted_flags(db, conversation_id).values()) == {0}
            await pilot.press("escape")
            await _until(
                lambda: "can't be cancelled" in _painted(host),
                "Escape to be answered with why it can't close",
            )
            assert host.screen.query("#console-delete-receipt-progress")

            delete_gate.release.set()
            await _until(
                lambda: bool(host.screen.query("#console-delete-receipt")),
                "the receipt once the delete committed",
            )
            flags = _deleted_flags(db, conversation_id)
            assert [flags[message_id] for message_id in persisted] == [0, 0] + [
                1
            ] * _REMOVED
            assert set(store._nodes_by_session[session.id]) == {
                native["u1"],
                native["a1"],
            }
            # Undo is possible, so every committed tombstone's media
            # reference is held back, none released; the fences are released.
            assert sorted(hold.held) == sorted(removed)
            assert hold.releases and all(diverted for _ids, diverted in hold.releases)
            assert not _fenced(store, session.id, native["a1"])
            await _until(
                lambda: f"Deleted {_REMOVED} messages" in _painted(host),
                "the receipt to be painted",
            )

            # --- Undo ----------------------------------------------------
            host.screen.query_one("#console-delete-receipt-undo", Button).press()
            await _until(restore_gate.entered.is_set, "the durable undo to start")
            assert restore_gate.thread != loop_thread, (
                "the durable subtree undo ran on the event-loop thread"
            )
            ticks = heartbeat.ticks
            await _until(
                lambda: f"Restoring {_REMOVED} messages" in _painted(host),
                "the in-progress undo to be painted",
            )
            await _until(
                lambda: heartbeat.ticks >= ticks + 20,
                "the event loop to keep turning while the undo is held",
            )
            assert set(_deleted_flags(db, conversation_id).values()) == {0, 1}
            assert _fenced(store, session.id, native["a1"])

            restore_gate.release.set()
            await _until(
                lambda: set(_deleted_flags(db, conversation_id).values()) == {0},
                "every row to be live again",
            )
            await _until(
                lambda: not host.screen.query("#console-delete-receipt-box"),
                "the receipt to close after Undo",
            )
            assert _tree_shape(store, session.id) == before
            assert not _fenced(store, session.id, native["a1"])
            assert any(f"Restored {_REMOVED} messages" in n for n in notices), notices
        finally:
            delete_gate.release.set()
            restore_gate.release.set()
            await heartbeat.stop()
