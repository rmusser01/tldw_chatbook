"""The store's own locks must never be held across the adapter's transaction.

TASK-32801.4, from the core review. Two orders used to exist:

* lock-then-sqlite -- ``write_trajectory_rows`` took ``_trajectory_lock`` and
  ``_persist_exchanges_only`` took ``_capture_quiescence_lock``, then called
  through the persistence adapter, which opens its own ``BEGIN IMMEDIATE``.
* sqlite-then-lock -- ``_dispatch_branch_mutation`` holds the sqlite write
  lock and, inside it, reaches both of those methods (``_create_sibling`` ->
  ``_persist_new_message`` -> ``_write_trajectory_row_for_message``;
  ``_update_message_content`` -> ``_persist_existing_message`` ->
  ``_persist_exchanges_only``).

A second thread entering the first order while the UI thread sits in the
second is a classic ABBA: the UI thread waits on a Python lock the worker
will not release until sqlite grants it a write lock the UI thread holds.
Measured before the fix: neither call returned inside a 60 s join, eight
``OperationalError`` retries, and zero rows stored.

These tests are deliberately end-to-end on a real file-backed database --
the deadlock only exists when a real sqlite write lock is involved, so a
fake adapter cannot show it.
"""

from __future__ import annotations

import threading
import time

import pytest

from Tests.Chat.test_console_dispatch_recovery import (
    _acceptance,
    _database,
    _insert,
    _restored_store,
)
from tldw_chatbook.DB.ChaChaNotes_DB import TrajectoryRowWrite

# Generous next to the ~0.5 s the fixed path takes, and far below the
# adapter's own 15 s busy timeout -- a revived inversion blows through this
# rather than merely running slowly.
_JOIN_TIMEOUT = 8.0


@pytest.fixture()
def restored(tmp_path):
    db, conversation_id, repository = _database(tmp_path / "lock-order.sqlite")
    inserted = _insert(db, repository, _acceptance(conversation_id))
    with db.transaction(immediate=True) as cursor:
        cursor.execute(
            "DELETE FROM console_dispatch_checkpoints WHERE conversation_id = ?",
            (conversation_id,),
        )
    store, session_id = _restored_store(db, conversation_id)
    try:
        yield db, store, session_id, conversation_id, inserted.assistant_message_id
    finally:
        db.close_connection()


def test_a_worker_trajectory_write_does_not_deadlock_the_branch_mutation(restored):
    """The two orders must be able to run at once and both land."""
    db, store, session_id, conversation_id, message_id = restored

    def row(kind: str) -> TrajectoryRowWrite:
        return TrajectoryRowWrite(
            message_id=message_id,
            conversation_id=conversation_id,
            turn_id=message_id,
            seq=None,
            event_kind=kind,
            step_started_at=time.time(),
            payload_json="{}",
        )

    worker_result: dict[str, object] = {}
    ui_result: dict[str, object] = {}
    ui_inside = threading.Event()

    def worker() -> None:
        # Enter only once the UI thread holds the sqlite write lock, so the
        # orders genuinely cross rather than merely queueing.
        assert ui_inside.wait(_JOIN_TIMEOUT)
        worker_result["ok"] = store.write_trajectory_rows([row("worker")])

    def ui() -> None:
        with store._dispatch_branch_mutation(session_id):
            ui_inside.set()
            time.sleep(0.25)
            ui_result["ok"] = store.write_trajectory_rows([row("ui")])

    ui_thread = threading.Thread(target=ui, name="ui")
    worker_thread = threading.Thread(target=worker, name="worker")
    ui_thread.start()
    worker_thread.start()
    ui_thread.join(_JOIN_TIMEOUT)
    worker_thread.join(_JOIN_TIMEOUT)

    assert not ui_thread.is_alive(), (
        "the branch mutation never completed: a store lock is being held "
        "across the persistence adapter's transaction again"
    )
    assert not worker_thread.is_alive(), "the worker write never completed"
    assert ui_result == {"ok": True}
    assert worker_result == {"ok": True}
    stored = {r.event_kind for r in db.get_trajectory_rows(conversation_id)}
    assert {"ui", "worker"} <= stored, (
        f"a sidecar write was dropped by lock contention: {sorted(stored)}"
    )


def _store_locks_held_now(store) -> list[str]:
    """Names of the store's own locks currently held by anyone.

    ``RLock.acquire(blocking=False)`` succeeds for the thread that already
    owns it, so a probe built on ``acquire`` reports a re-entrant hold as
    "free" -- exactly the case under test. ``_is_owned``/``locked`` are the
    honest questions.
    """
    held = []
    for name in ("_trajectory_lock", "_capture_quiescence_lock"):
        lock = getattr(store, name, None)
        if lock is None:
            continue
        owned = getattr(lock, "_is_owned", None)
        locked = getattr(lock, "locked", None)
        if (owned is not None and owned()) or (locked is not None and locked()):
            held.append(name)
    return held


def test_no_store_lock_is_held_while_the_trajectory_adapter_writes(restored):
    """Structural twin of the deadlock test: name the invariant, not timing.

    A deadlock test can only fail by timing out, which is slow and reads as
    flake. This one fails at once if any store lock is held at the moment
    the adapter -- and therefore ``BEGIN IMMEDIATE`` -- is entered.
    """
    _db, store, _session_id, conversation_id, message_id = restored
    held: list[str] = []
    inner = store.persistence.write_trajectory_rows

    def spy(rows):
        held.extend(_store_locks_held_now(store))
        return inner(rows)

    store.persistence.write_trajectory_rows = spy
    try:
        assert store.write_trajectory_rows(
            [
                TrajectoryRowWrite(
                    message_id=message_id,
                    conversation_id=conversation_id,
                    turn_id=message_id,
                    seq=None,
                    event_kind="structural",
                    step_started_at=time.time(),
                    payload_json="{}",
                )
            ]
        )
    finally:
        store.persistence.write_trajectory_rows = inner
    assert held == [], f"store lock(s) held across the adapter write: {held}"


def test_no_store_lock_is_held_while_the_exchange_adapter_writes():
    """The quiescence lock's half of the same invariant.

    Driven through the purge suite's fixture because this check is about
    which locks are held, not about what sqlite does -- and that fixture
    already builds a message carrying captures.
    """
    from Tests.Chat.test_console_capture_purge import _store_with_captures

    store, session, message, persistence = _store_with_captures()
    stored_message = store._nodes_by_session[session.id][message.id]
    held: list[str] = []
    inner = persistence.append_message_exchanges

    def spy(*, message_id, rows):
        held.extend(_store_locks_held_now(store))
        return inner(message_id=message_id, rows=rows)

    persistence.append_message_exchanges = spy
    persistence.exchange_appends.clear()
    store._persist_exchanges_only(stored_message)

    assert persistence.exchange_appends, "the flush never reached the adapter"
    assert held == [], f"store lock(s) held across the exchange write: {held}"


def test_the_attach_path_also_flushes_outside_the_lock():
    """``attach_message_exchanges`` merges under the lock and flushes after."""
    from Tests.Chat.test_console_capture_purge import _capture, _store_with_captures
    from tldw_chatbook.Chat.console_exchange_capture import CaptureDetail

    store, session, message, persistence = _store_with_captures()
    held: list[str] = []
    inner = persistence.append_message_exchanges

    def spy(*, message_id, rows):
        held.extend(_store_locks_held_now(store))
        return inner(message_id=message_id, rows=rows)

    persistence.append_message_exchanges = spy
    persistence.exchange_appends.clear()
    store.attach_message_exchanges(
        message.id, [_capture("later", CaptureDetail.SAFE)]
    )

    assert persistence.exchange_appends, "the attach never flushed"
    assert held == [], f"store lock(s) held across the attach flush: {held}"
    assert "later" in {c.run_tag for c in store.get_message(message.id).exchanges}
