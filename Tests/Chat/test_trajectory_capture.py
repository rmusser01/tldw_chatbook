"""Trajectory sidecar capture (schema v38, task 2).

Covers the Console persistence seam: every persisted Console message gets a
``user``/``assistant`` sidecar row, every TOOL marker gets ``tool_call`` +
``tool_result`` rows keyed to the parent assistant message (TOOL-marker
invariant: markers themselves are never persisted to ``messages``), and
streamed assistant rows carry step-start/first-token/completion timing.
"""

import json
import threading
import time

import pytest

from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, TrajectoryRowWrite


def _store_with_db(tmp_path):
    db = CharactersRAGDB(str(tmp_path / "chachanotes.sqlite"), "test_client")
    store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    return db, store


def test_persisted_user_message_produces_user_trajectory_row(tmp_path):
    db, store = _store_with_db(tmp_path)
    try:
        session = store.ensure_session(title="Trajectory")
        conversation_id = store.persist_session_if_needed(session.id)
        user = store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="hello",
            persist=True,
        )

        rows = db.get_trajectory_rows(conversation_id)
        assert [row.event_kind for row in rows] == ["user"]
        row = rows[0]
        assert row.message_id == user.persisted_message_id
        assert row.turn_id == store.get_message(user.id).turn_id
        assert row.payload_json is None
    finally:
        db.close()


def test_tool_marker_append_produces_tool_call_and_result_rows(tmp_path):
    db, store = _store_with_db(tmp_path)
    try:
        session = store.ensure_session(title="Trajectory")
        conversation_id = store.persist_session_if_needed(session.id)
        store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="list the files",
            persist=True,
        )
        assistant = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="working on it",
            persist=True,
        )

        marker = store.append_message(
            session.id,
            role=ConsoleMessageRole.TOOL,
            content="⚙ fs_list → (3 files)",
            tool_output_full="file-a\nfile-b\nfile-c",
        )
        assert marker.role is ConsoleMessageRole.TOOL
        # TOOL-marker invariant: the marker itself is never a persisted row.
        assert marker.persisted_message_id is None

        rows = db.get_trajectory_rows(conversation_id)
        tool_rows = [row for row in rows if row.event_kind.startswith("tool_")]
        assert sorted(row.event_kind for row in tool_rows) == [
            "tool_call",
            "tool_result",
        ]
        for row in tool_rows:
            assert row.message_id == assistant.persisted_message_id
            payload = json.loads(row.payload_json)
            assert payload["name"] == "fs_list"
            assert payload["result"] == "file-a\nfile-b\nfile-c"
            assert payload.get("truncated") is not True
    finally:
        db.close()


def test_tool_marker_before_assistant_persist_flushes_on_assistant_persist(tmp_path):
    """A marker appended while the assistant row is still streaming is buffered
    and flushed (remapped to the persisted id) when the assistant persists."""
    db, store = _store_with_db(tmp_path)
    try:
        session = store.ensure_session(title="Trajectory")
        conversation_id = store.persist_session_if_needed(session.id)
        store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="go",
            persist=True,
        )
        # Streaming assistant placeholder: NOT yet persisted.
        assistant = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="",
            persist=True,
        )
        assert assistant.persisted_message_id is None
        store.append_message(
            session.id,
            role=ConsoleMessageRole.TOOL,
            content="⚙ fs_read → preview",
            tool_output_full="full file contents",
        )
        # Nothing writable yet: the marker waits for the parent's durable id.
        assert db.get_trajectory_rows(conversation_id) == [] or all(
            row.event_kind == "user" for row in db.get_trajectory_rows(conversation_id)
        )

        store.append_stream_chunk(assistant.id, "done")
        completed = store.mark_message_complete(assistant.id)
        assert completed.persisted_message_id is not None

        tool_rows = [
            row
            for row in db.get_trajectory_rows(conversation_id)
            if row.event_kind.startswith("tool_")
        ]
        assert sorted(row.event_kind for row in tool_rows) == [
            "tool_call",
            "tool_result",
        ]
        for row in tool_rows:
            assert row.message_id == completed.persisted_message_id
    finally:
        db.close()


def test_streamed_assistant_row_carries_timing(tmp_path):
    db, store = _store_with_db(tmp_path)
    try:
        session = store.ensure_session(title="Trajectory")
        conversation_id = store.persist_session_if_needed(session.id)
        store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="hi",
            persist=True,
        )
        assistant = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="",
            persist=True,
        )

        step_started = time.time()
        store.record_trajectory_timing(
            assistant.id,
            step_started_at=step_started,
            model="test-model",
            provider="test-provider",
        )
        time.sleep(0.01)
        store.append_stream_chunk(assistant.id, "first")
        time.sleep(0.01)
        store.record_trajectory_timing(assistant.id, completed_at=time.time())

        completed = store.mark_message_complete(assistant.id)
        rows = [
            row
            for row in db.get_trajectory_rows(conversation_id)
            if row.event_kind == "assistant"
        ]
        assert len(rows) == 1
        row = rows[0]
        assert row.message_id == completed.persisted_message_id
        assert row.model == "test-model"
        assert row.provider == "test-provider"
        assert row.step_started_at == pytest.approx(step_started, abs=1.0)
        assert row.first_token_at is not None
        assert row.first_token_at - row.step_started_at > 0
        assert row.completed_at >= row.first_token_at
    finally:
        db.close()


def test_tool_result_capped_at_256kib_with_truncated_marker(tmp_path):
    db, store = _store_with_db(tmp_path)
    try:
        session = store.ensure_session(title="Trajectory")
        conversation_id = store.persist_session_if_needed(session.id)
        store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="go",
            persist=True,
        )
        store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="answer",
            persist=True,
        )
        huge = "x" * (300 * 1024)
        store.append_message(
            session.id,
            role=ConsoleMessageRole.TOOL,
            content="⚙ fs_read → preview",
            tool_output_full=huge,
        )

        tool_rows = [
            row
            for row in db.get_trajectory_rows(conversation_id)
            if row.event_kind.startswith("tool_")
        ]
        assert len(tool_rows) == 2
        for row in tool_rows:
            payload = json.loads(row.payload_json)
            assert payload["truncated"] is True
            assert len(payload["result"].encode("utf-8")) <= 256 * 1024
            assert payload["result"] != huge
    finally:
        db.close()


def test_tool_result_cap_is_byte_safe_for_multibyte_content(tmp_path):
    """The cap is BYTES, not characters: 4-byte emoji content truncated by a
    character slice could leave the stored result up to 4x over budget."""
    db, store = _store_with_db(tmp_path)
    try:
        session = store.ensure_session(title="Trajectory")
        conversation_id = store.persist_session_if_needed(session.id)
        store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="go",
            persist=True,
        )
        store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="answer",
            persist=True,
        )
        # U+1F600 encodes to 4 UTF-8 bytes per character: ~100k chars is
        # ~400 KiB, well over the 256 KiB byte cap.
        huge = "😀" * (100 * 1024)
        assert len(huge) < 256 * 1024  # characters under, bytes over
        store.append_message(
            session.id,
            role=ConsoleMessageRole.TOOL,
            content="⚙ fs_read → preview",
            tool_output_full=huge,
        )

        tool_rows = [
            row
            for row in db.get_trajectory_rows(conversation_id)
            if row.event_kind.startswith("tool_")
        ]
        assert len(tool_rows) == 2
        for row in tool_rows:
            payload = json.loads(row.payload_json)
            assert payload["truncated"] is True
            stored = payload["result"]
            assert len(stored.encode("utf-8")) <= 256 * 1024
            # The split codepoint at the byte boundary was dropped cleanly.
            assert stored.endswith("😀") or stored == ""
            assert stored != huge
    finally:
        db.close()


def test_trajectory_write_failure_never_fails_the_turn(tmp_path):
    db, store = _store_with_db(tmp_path)
    try:
        session = store.ensure_session(title="Trajectory")
        store.persist_session_if_needed(session.id)

        def exploding_writer(**kwargs):
            raise RuntimeError("sidecar unavailable")

        store.persistence.write_trajectory_rows = exploding_writer
        # The turn itself must still persist and complete normally.
        user = store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="hello",
            persist=True,
        )
        assert user.persisted_message_id is not None
        marker = store.append_message(
            session.id,
            role=ConsoleMessageRole.TOOL,
            content="⚙ fs_read → preview",
            tool_output_full="ok",
        )
        assert marker.persisted_message_id is None
    finally:
        db.close()


def test_concurrent_upserts_produce_unique_seqs(tmp_path):
    """Two threads writing trajectory rows for one conversation must produce
    unique, gap-free seqs. Exercises the Console's write seam
    (``ChatPersistenceService.write_trajectory_rows``), whose bounded retry
    absorbs the transient write-write lock contention of concurrent turns;
    every row lands exactly once with a distinct per-conversation seq."""
    db = CharactersRAGDB(str(tmp_path / "chachanotes.sqlite"), "test_client")
    try:
        service = ChatPersistenceService(db)
        conversation_id = db.add_conversation(
            {"chat_id": 1, "conversation_id": "traj-concurrency", "fragmentation": 0}
        )
        assert conversation_id is not None

        message_ids = [
            db.add_message(
                {
                    "conversation_id": conversation_id,
                    "sender": "assistant",
                    "content": f"message {index}",
                }
            )
            for index in range(50)
        ]

        def write_batch(prefix: str, ids: list) -> None:
            for index in range(25):
                assert service.write_trajectory_rows(
                    [
                        TrajectoryRowWrite(
                            message_id=ids.pop(),
                            conversation_id=conversation_id,
                            turn_id=f"{prefix}-turn",
                            seq=None,
                            event_kind="assistant",
                        )
                    ]
                )

        batch_a = message_ids[:25]
        batch_b = message_ids[25:]

        threads = [
            threading.Thread(target=write_batch, args=("t0", batch_a)),
            threading.Thread(target=write_batch, args=("t1", batch_b)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        rows = db.get_trajectory_rows(conversation_id)
        assert len(rows) == 50
        seqs = [row.seq for row in rows]
        assert sorted(seqs) == list(range(1, 51))
        assert len(set(seqs)) == 50
    finally:
        db.close()


def test_concurrent_direct_db_upserts_produce_unique_seqs(tmp_path):
    """The DB-layer upsert itself must be safe under cross-thread concurrency
    (BEGIN IMMEDIATE write lock): no thread's batch may roll back with the
    deferred-upgrade "database is locked" deadlock, and seqs stay unique."""
    db = CharactersRAGDB(str(tmp_path / "chachanotes.sqlite"), "test_client")
    try:
        conversation_id = db.add_conversation(
            {"chat_id": 1, "conversation_id": "traj-db-concurrency", "fragmentation": 0}
        )
        assert conversation_id is not None
        message_ids = [
            db.add_message(
                {
                    "conversation_id": conversation_id,
                    "sender": "assistant",
                    "content": f"message {index}",
                }
            )
            for index in range(50)
        ]

        def write_direct_batch(ids: list) -> None:
            for message_id in ids:
                db.upsert_trajectory_rows(
                    [
                        TrajectoryRowWrite(
                            message_id=message_id,
                            conversation_id=conversation_id,
                            turn_id="db-turn",
                            seq=None,
                            event_kind="assistant",
                        )
                    ]
                )

        threads = [
            threading.Thread(target=write_direct_batch, args=(message_ids[:25],)),
            threading.Thread(target=write_direct_batch, args=(message_ids[25:],)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        rows = db.get_trajectory_rows(conversation_id)
        assert len(rows) == 50
        seqs = [row.seq for row in rows]
        assert sorted(seqs) == list(range(1, 51))
        assert len(set(seqs)) == 50
    finally:
        db.close()


# --- selection feedback events (task-17169, phase 4) ---------------------------
#
# Console selection feedback (Request changes / LGTM / Comment) was ephemeral:
# composed into the next user message and forgotten. Decision AC#4 was Option A
# -- the ADR-066 trajectory sidecar, because feedback is a chronological run
# event and the sidecar is local-only (a synced annotations table would drag in
# sync-schema implications for what is really an audit record).


def test_selection_feedback_persists_as_a_user_feedback_row(tmp_path):
    db, store = _store_with_db(tmp_path)
    try:
        session = store.ensure_session(title="Feedback")
        conversation_id = store.persist_session_if_needed(session.id)
        store.append_message(
            session.id, role=ConsoleMessageRole.USER, content="do it", persist=True
        )
        assistant = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="here is the patch",
            persist=True,
        )

        assert store.record_feedback_event(
            session.id,
            anchor_message_id=assistant.id,
            action="request-changes",
            quote="here is the patch",
            comment="use a context manager",
        )

        rows = [
            row
            for row in db.get_trajectory_rows(conversation_id)
            if row.event_kind == "user_feedback"
        ]
        assert len(rows) == 1
        row = rows[0]
        assert row.message_id == assistant.persisted_message_id
        assert row.turn_id == store.get_message(assistant.id).turn_id
        payload = json.loads(row.payload_json)
        assert payload["action"] == "request-changes"
        assert payload["quote"] == "here is the patch"
        assert payload["comment"] == "use a context manager"
    finally:
        db.close()


def test_feedback_without_a_comment_records_no_comment_key(tmp_path):
    """LGTM and Request-changes carry no comment; the payload must not
    fabricate an empty one for the viewer to render."""
    db, store = _store_with_db(tmp_path)
    try:
        session = store.ensure_session(title="Feedback")
        conversation_id = store.persist_session_if_needed(session.id)
        assistant = store.append_message(
            session.id, role=ConsoleMessageRole.ASSISTANT, content="ok", persist=True
        )

        store.record_feedback_event(
            session.id,
            anchor_message_id=assistant.id,
            action="lgm",
            quote="ok",
            comment=None,
        )

        rows = [
            row
            for row in db.get_trajectory_rows(conversation_id)
            if row.event_kind == "user_feedback"
        ]
        assert json.loads(rows[0].payload_json) == {"action": "lgm", "quote": "ok"}
    finally:
        db.close()


def test_feedback_on_an_unpersisted_session_is_skipped_not_raised(tmp_path):
    """Ephemeral sessions have nothing to survive a restart -- the write is a
    silent no-op, and it must never take down the dispatch that triggered it."""
    db, store = _store_with_db(tmp_path)
    try:
        session = store.ensure_session(title="Ephemeral")
        assistant = store.append_message(
            session.id, role=ConsoleMessageRole.ASSISTANT, content="ok", persist=False
        )

        assert (
            store.record_feedback_event(
                session.id,
                anchor_message_id=assistant.id,
                action="lgm",
                quote="ok",
                comment=None,
            )
            is False
        )
    finally:
        db.close()


def test_feedback_for_an_unknown_anchor_never_raises(tmp_path):
    db, store = _store_with_db(tmp_path)
    try:
        session = store.ensure_session(title="Feedback")
        store.persist_session_if_needed(session.id)

        assert (
            store.record_feedback_event(
                session.id,
                anchor_message_id="no-such-message",
                action="comment",
                quote="q",
                comment="c",
            )
            is False
        )
    finally:
        db.close()


def test_feedback_survives_a_restart(tmp_path):
    """AC#1: the whole point is durability. Reopen the DB from disk with a
    fresh store and the feedback is still there."""
    db_path = str(tmp_path / "chachanotes.sqlite")
    db = CharactersRAGDB(db_path, "test_client")
    try:
        store = ConsoleChatStore(persistence=ChatPersistenceService(db))
        session = store.ensure_session(title="Feedback")
        conversation_id = store.persist_session_if_needed(session.id)
        assistant = store.append_message(
            session.id, role=ConsoleMessageRole.ASSISTANT, content="ok", persist=True
        )
        store.record_feedback_event(
            session.id,
            anchor_message_id=assistant.id,
            action="comment",
            quote="ok",
            comment="revisit this",
        )
    finally:
        db.close()

    reopened = CharactersRAGDB(db_path, "test_client")
    try:
        rows = [
            row
            for row in reopened.get_trajectory_rows(conversation_id)
            if row.event_kind == "user_feedback"
        ]
        assert [json.loads(row.payload_json)["comment"] for row in rows] == [
            "revisit this"
        ]
    finally:
        reopened.close()


def test_message_edit_and_branch_selection_append_payload_free_trace_events(tmp_path):
    db, store = _store_with_db(tmp_path)
    try:
        session = store.ensure_session(title="Mutations")
        conversation_id = store.persist_session_if_needed(session.id)
        first = store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="SECRET_ORIGINAL_CONTENT",
            persist=True,
        )
        second = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="second",
            persist=True,
        )

        store.update_message_content(second.id, "SECRET_EDITED_CONTENT")
        store.set_active_leaf(session.id, first.id)

        rows = db.get_trajectory_rows(conversation_id)
        by_kind = {
            row.event_kind: json.loads(row.payload_json)
            for row in rows
            if row.payload_json is not None
        }
        assert by_kind["message_edited"]["field_states"] == {"payload": "omitted"}
        assert by_kind["branch_selected"]["field_states"] == {"payload": "omitted"}
        assert "SECRET_ORIGINAL_CONTENT" not in repr(by_kind)
        assert "SECRET_EDITED_CONTENT" not in repr(by_kind)
    finally:
        db.close()


def test_trace_event_capture_failure_never_fails_the_user_mutation(
    tmp_path, monkeypatch
):
    db, store = _store_with_db(tmp_path)
    try:
        session = store.ensure_session(title="Capture failure")
        store.persist_session_if_needed(session.id)
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="before",
            persist=True,
        )
        monkeypatch.setattr(store, "write_trajectory_rows", lambda _rows: False)

        updated = store.update_message_content(message.id, "after")

        assert updated.content == "after"
    finally:
        db.close()
