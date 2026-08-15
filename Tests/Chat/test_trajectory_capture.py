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
            row.event_kind == "user"
            for row in db.get_trajectory_rows(conversation_id)
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
            assistant.id, step_started_at=step_started, model="test-model",
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
