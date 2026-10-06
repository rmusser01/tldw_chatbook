"""TASK-34100.3 review G3V2-02: one continuation owner row, never two.

Continue (``--->``) in a saved Kimi chat sometimes stored the reply twice: a
truncated streaming copy under a database-assigned id, and the full reply with
its reasoning continuation under the store's native id, as siblings under the
same parent. After a relaunch the chat showed "Assistant (1/2)" with the
truncated copy selected.

Two writers raced for the same pending assistant message. The stream's
deferred flush (``_persist_pending_message_if_ready`` -> ``_persist_new_message``)
inserts it under ``_pending_row_insert_lock`` and skips when the row already
exists. ``persist_provider_continuation_event`` checked "not yet persisted"
without that lock, then created the row with its continuation, so a flush
landing between that check and the create produced a second row. These tests
put the flush in exactly that window, deterministically, on the real store and
a file-backed ChaChaNotes database.
"""

from __future__ import annotations

import threading

import pytest

from Tests.Chat.test_continuation_destination_identity import _kimi_final
from Tests.Chat.test_continuation_unwritten_store_failures import (
    _assistant_rows,
    _context,
    _local_store,
)
from tldw_chatbook.Agents.agent_models import FinalContinuation
from tldw_chatbook.Chat.provider_continuation import parse_provider_continuation_json

pytestmark = [pytest.mark.bootstrap_profile]

PARTIAL = "ans"
REPLY = "answer"


def _flush_inside_the_create_window(store, owner_id: str, monkeypatch) -> list[bool]:
    """Run the stream's deferred flush between the check and the create.

    ``_materialize_stream_buffer`` is what the deferred stream-persistence
    worker runs: fold the buffered chunks, then insert the pending row. The
    create path resolves the owner's parent (``_previous_persisted_message_id``)
    after its "not yet persisted" check and just before it creates the row, so
    the flush is made to land there.
    """
    flushed: list[bool] = []
    original = store._previous_persisted_message_id

    def flush_then_resolve_parent(message):
        if not flushed:
            owner = store._message_or_raise(owner_id)
            store._materialize_stream_buffer(owner)
            flushed.append(owner.persisted_message_id is not None)
        return original(message)

    monkeypatch.setattr(
        store, "_previous_persisted_message_id", flush_then_resolve_parent
    )
    return flushed


def test_a_flush_in_the_create_window_leaves_one_complete_row(
    tmp_path, monkeypatch
) -> None:
    database, store, session, owner = _local_store(tmp_path)
    try:
        store.append_stream_chunk(owner.id, PARTIAL)
        # Buffered only: reading through get_message() would flush it now.
        assert store._message_or_raise(owner.id).persisted_message_id is None
        flushed = _flush_inside_the_create_window(store, owner.id, monkeypatch)

        store.persist_provider_continuation_event(
            FinalContinuation(_context(owner.id), _kimi_final(), None, REPLY)
        )

        assert flushed == [True], "the flush must really have inserted a row"
        rows = _assistant_rows(database, session)
        assert [
            (row["content"], row["assistant_generation_state"]) for row in rows
        ] == [(REPLY, "complete")]
        stored = parse_provider_continuation_json(rows[0]["provider_continuation_json"])
        assert stored.model == "kimi-k3"
        message = store.get_message(owner.id)
        assert message.persisted_message_id == rows[0]["id"]
        assert message.provider_continuation_message_version == rows[0]["version"]

        # The ordinary completion that follows touches the same single row.
        store.mark_message_complete(owner.id)
        assert [row["content"] for row in _assistant_rows(database, session)] == [REPLY]
    finally:
        database.close_connection()


def test_a_worker_flush_during_the_create_waits_for_it(tmp_path, monkeypatch) -> None:
    """The live shape: the flush runs on the stream-persistence worker thread.

    The worker reaches the insert lock while the continuation is creating the
    row. It must wait, then find the row already there, instead of inserting
    a second one alongside it.
    """
    database, store, session, owner = _local_store(tmp_path)
    try:
        store.append_stream_chunk(owner.id, PARTIAL)
        real_lock = store._pending_row_insert_lock
        events: list[str] = []
        worker_at_lock = threading.Event()

        class _ObservedLock:
            def __enter__(self):
                on_worker = threading.current_thread() is not threading.main_thread()
                if on_worker:
                    worker_at_lock.set()
                real_lock.acquire()
                if on_worker:
                    events.append("flush took the insert lock")
                return self

            def __exit__(self, *exc_info):
                real_lock.release()
                return False

        monkeypatch.setattr(store, "_pending_row_insert_lock", _ObservedLock())
        worker = threading.Thread(
            target=store._materialize_stream_buffer,
            args=(store._message_or_raise(owner.id),),
        )
        real_create = database.create_assistant_with_continuation

        def create_while_the_worker_flushes(**kwargs):
            worker.start()
            assert worker_at_lock.wait(10), "the worker never reached the lock"
            worker.join(0.2)  # time enough to take the lock if nothing holds it
            created = real_create(**kwargs)
            events.append("continuation created the row")
            return created

        monkeypatch.setattr(
            database,
            "create_assistant_with_continuation",
            create_while_the_worker_flushes,
        )

        store.persist_provider_continuation_event(
            FinalContinuation(_context(owner.id), _kimi_final(), None, REPLY)
        )
        worker.join(10)

        assert not worker.is_alive()
        assert events == [
            "continuation created the row",
            "flush took the insert lock",
        ]
        rows = _assistant_rows(database, session)
        assert [(row["id"], row["content"]) for row in rows] == [(owner.id, REPLY)]
    finally:
        database.close_connection()


def test_without_a_flush_the_continuation_creates_the_row_itself(tmp_path) -> None:
    """Control: the ordinary create path still pins the store's native id."""
    database, store, session, owner = _local_store(tmp_path)
    try:
        store.append_stream_chunk(owner.id, PARTIAL)

        store.persist_provider_continuation_event(
            FinalContinuation(_context(owner.id), _kimi_final(), None, REPLY)
        )

        rows = _assistant_rows(database, session)
        assert [(row["id"], row["content"]) for row in rows] == [(owner.id, REPLY)]
        assert rows[0]["provider_continuation_json"] is not None
    finally:
        database.close_connection()
