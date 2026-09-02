"""Physical trace compaction, reopen, integrity, and shared-fork retention."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from tldw_chatbook.Chat.console_trace_maintenance import (
    PhysicalTraceCompactor,
    TraceCompactionPolicy,
    TraceGarbageCollector,
)
from tldw_chatbook.Chat.console_trace_models import (
    FrozenTracePolicy,
    SemanticRevisionRef,
    new_opaque_id,
)
from tldw_chatbook.Chat.console_trace_repository import ConsoleTraceRepository
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError


def _conversation(db: CharactersRAGDB, title: str) -> tuple[str, str]:
    conversation_id = db.add_conversation({"title": title})
    assert conversation_id is not None
    message_id = db.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "user",
            "role": "user",
            "content": title,
        }
    )
    assert message_id is not None
    return conversation_id, message_id


def _shared_fork_fixture(
    db: CharactersRAGDB,
) -> tuple[str, str, str, str]:
    repository = ConsoleTraceRepository()
    source_id, message_id = _conversation(db, "source")
    child_id, _child_message_id = _conversation(db, "child")
    with db.transaction(immediate=True) as cursor:
        segment = repository.create_segment(cursor)
        owner = repository.attach_owner(
            cursor,
            conversation_id=source_id,
            root_segment_id=segment.segment_id,
        )
        policy = repository.ensure_policy(
            cursor,
            FrozenTracePolicy(
                policy_id=new_opaque_id(),
                credential_filter_version="cred-v1",
                pii_redaction_enabled=False,
                pii_ruleset_revision_id=None,
            ),
        )
        revision = repository.ensure_semantic_revision(
            cursor,
            source_conversation_id=source_id,
            source_message_id=message_id,
            revision_sequence=0,
            normalized_role="user",
            content_kind="text",
            creation_reason="message_create",
            live_message_id=message_id,
        )
        node = repository.append_surface_node(
            cursor,
            segment_id=segment.segment_id,
            sequence=0,
            predecessor_node_id=None,
            component_kind="message",
            reference=SemanticRevisionRef(revision.revision_id),
        )
        repository.append_event(
            cursor,
            segment_id=segment.segment_id,
            sequence=0,
            event_type="surface_append",
            surface_node_id=node.node_id,
        )
        call = repository.reserve_call(
            cursor,
            owner_id=owner.owner_id,
            segment_id=segment.segment_id,
            turn_id="turn-1",
            run_id="run-turn-1",
            call_sequence=0,
            idempotency_key="compaction-shared-call",
            policy_id=policy.policy_id,
        )
        repository.append_event(
            cursor,
            segment_id=segment.segment_id,
            sequence=1,
            event_type="call_boundary",
            call_id=call.call_id,
        )
        boundary = repository.capture_fork_boundary(
            cursor,
            conversation_id=source_id,
            included_turn_ids=("turn-1",),
        )
        assert boundary is not None
        child_owner = repository.attach_fork_owner(
            cursor,
            conversation_id=child_id,
            boundary=boundary,
        )
        cursor.execute(
            "UPDATE console_trace_migration_state SET status = 'logical_complete' "
            "WHERE migration_name = 'legacy_exchange_normalization'"
        )
    return source_id, child_id, child_owner.root_segment_id, call.call_id


def _add_orphan_trace_payload(db: CharactersRAGDB, *, rows: int = 256) -> None:
    repository = ConsoleTraceRepository()
    with db.transaction(immediate=True) as cursor:
        for index in range(rows):
            block = index.to_bytes(4, "big") + bytes(range(256)) * 128
            repository.store_sanitized_artifact(
                cursor,
                sanitized_bytes=block,
                media_type="application/octet-stream",
                normalization_version="compaction-fixture-v1",
            )


def _permissive_policy() -> TraceCompactionPolicy:
    return TraceCompactionPolicy(
        min_database_bytes=1,
        min_freelist_bytes=1,
        min_freelist_ratio=0.0,
        min_idle_seconds=0.0,
        retry_initial_seconds=1.0,
        retry_max_seconds=10.0,
        quiesce_timeout_seconds=1.0,
        disk_safety_margin_bytes=0,
    )


def test_vacuum_progress_never_queries_the_active_vacuum_connection(
    tmp_path: Path,
) -> None:
    events: list[object] = []
    compactor = PhysicalTraceCompactor(
        type("Database", (), {"db_path_str": str(tmp_path / "fixture.sqlite")})(),
        progress=events.append,
    )
    compactor._PROGRESS_VM_STEPS = 1

    class Cursor:
        def __init__(self, value: int) -> None:
            self.value = value

        def fetchone(self) -> tuple[int]:
            return (self.value,)

    class Connection:
        callback = None
        in_vacuum = False

        def set_progress_handler(self, callback, _steps: int) -> None:
            self.callback = callback

        def execute(self, statement: str):
            if self.in_vacuum:
                raise AssertionError("progress handler queried active VACUUM connection")
            if statement == "PRAGMA page_size":
                return Cursor(4096)
            if statement == "PRAGMA page_count":
                return Cursor(1024)
            if statement == "PRAGMA freelist_count":
                return Cursor(512)
            assert statement == "VACUUM"
            self.in_vacuum = True
            try:
                assert self.callback is not None
                for _ in range(64):
                    assert self.callback() == 0
            finally:
                self.in_vacuum = False
            return Cursor(0)

    compactor._vacuum(Connection())  # type: ignore[arg-type]

    assert events


def test_vacuum_shrinks_file_and_preserves_shared_fork_after_reopen(
    tmp_path: Path,
) -> None:
    path = tmp_path / "trace-compaction.sqlite"
    database = CharactersRAGDB(path, "trace-compaction")
    source_id, child_id, child_segment_id, call_id = _shared_fork_fixture(database)
    _add_orphan_trace_payload(database)
    gc_result = TraceGarbageCollector(database).collect(
        request_id="gc-compaction-success"
    )
    assert gc_result.deleted_rows["console_trace_artifacts"] == 256
    assert gc_result.freelist_bytes_after > 0
    before_size = path.stat().st_size
    progress: list[tuple[str, int]] = []

    started = time.monotonic()
    outcome = PhysicalTraceCompactor(
        database,
        policy=_permissive_policy(),
        progress=lambda event: progress.append(
            (event.stage, event.progress_basis_points)
        ),
    ).run_after_gc(gc_result)
    elapsed = time.monotonic() - started

    assert outcome.completed is True
    assert outcome.reason_code == "complete"
    assert outcome.allocated_bytes_after < outcome.allocated_bytes_before
    assert path.stat().st_size < before_size
    assert elapsed <= 5.0
    assert progress and progress[0][0] == "vacuum"
    assert progress[-1] == ("complete", 10000)
    database.close_connection()

    reopened = CharactersRAGDB(path, "trace-compaction-reopen")
    try:
        repository = ConsoleTraceRepository()
        with reopened.transaction() as cursor:
            assert cursor.execute("PRAGMA quick_check(1)").fetchone()[0] == "ok"
            assert cursor.execute(
                "SELECT COUNT(*) FROM console_trace_owners "
                "WHERE attached = 1 AND conversation_id IN (?, ?)",
                (source_id, child_id),
            ).fetchone()[0] == 2
            child_segment = repository.get_segment(cursor, child_segment_id)
            assert child_segment is not None
            assert child_segment.parent_segment_id is not None
            assert [
                call.call_id
                for call in repository.read_conversation_call_lineage(cursor, child_id)
            ] == [call_id]
            state = cursor.execute(
                "SELECT status, reason_code, progress_basis_points, retry_count "
                "FROM console_trace_compaction_state WHERE singleton_id = 1"
            ).fetchone()
            assert tuple(state) == ("complete", "complete", 10000, 0)
    finally:
        reopened.close_connection()


def test_integrity_verification_failure_keeps_retry_state_and_readability(
    tmp_path: Path,
) -> None:
    path = tmp_path / "trace-compaction-integrity.sqlite"
    database = CharactersRAGDB(path, "trace-compaction-integrity")
    _shared_fork_fixture(database)
    _add_orphan_trace_payload(database, rows=8)
    gc_result = TraceGarbageCollector(database).collect(
        request_id="gc-compaction-integrity"
    )
    compactor = PhysicalTraceCompactor(database, policy=_permissive_policy())
    original_open = compactor._open_maintenance_connection
    open_count = 0

    class FailedIntegrityCursor:
        @staticmethod
        def fetchone() -> tuple[str]:
            return ("injected-integrity-failure",)

    class FailedIntegrityConnection:
        def __init__(self, connection: sqlite3.Connection) -> None:
            self.connection = connection

        def execute(self, sql: str, parameters=()):
            if sql == "PRAGMA quick_check(1)":
                return FailedIntegrityCursor()
            return self.connection.execute(sql, parameters)

        def close(self) -> None:
            self.connection.close()

    def open_with_failed_second_check():
        nonlocal open_count
        open_count += 1
        connection = original_open()
        if open_count == 2:
            return FailedIntegrityConnection(connection)
        return connection

    compactor._open_maintenance_connection = (  # type: ignore[method-assign]
        open_with_failed_second_check
    )

    outcome = compactor.run_after_gc(gc_result)

    assert outcome.reason_code == "integrity_check_failed"
    assert outcome.completed is False
    assert database.get_connection().execute(
        "PRAGMA quick_check(1)"
    ).fetchone()[0] == "ok"
    state = database.get_console_trace_compaction_status()
    assert state["status"] == "pending"
    assert state["reason_code"] == "integrity_check_failed"
    assert state["retry_pending"] is True
    database.close_connection()


def test_cancelled_vacuum_is_readable_and_retryable(tmp_path: Path) -> None:
    path = tmp_path / "trace-compaction-cancel.sqlite"
    database = CharactersRAGDB(path, "trace-compaction-cancel")
    _shared_fork_fixture(database)
    _add_orphan_trace_payload(database, rows=32)
    gc_result = TraceGarbageCollector(database).collect(
        request_id="gc-compaction-cancel"
    )
    compactor = PhysicalTraceCompactor(
        database,
        policy=_permissive_policy(),
        cancel_requested=lambda: True,
    )
    compactor._PROGRESS_VM_STEPS = 1

    outcome = compactor.run_after_gc(gc_result)

    assert outcome.reason_code == "cancelled"
    assert outcome.completed is False
    assert database.get_connection().execute(
        "PRAGMA quick_check(1)"
    ).fetchone()[0] == "ok"
    state = database.get_console_trace_compaction_status()
    assert state["status"] == "pending"
    assert state["reason_code"] == "cancelled"
    assert state["retry_pending"] is True
    database.close_connection()


def test_vacuum_failure_leaves_database_readable_and_retry_pending(
    tmp_path: Path,
) -> None:
    path = tmp_path / "trace-compaction-failure.sqlite"
    database = CharactersRAGDB(path, "trace-compaction-failure")
    source_id, child_id, _child_segment_id, _call_id = _shared_fork_fixture(database)
    _add_orphan_trace_payload(database, rows=8)
    gc_result = TraceGarbageCollector(database).collect(
        request_id="gc-compaction-failure"
    )
    compactor = PhysicalTraceCompactor(database, policy=_permissive_policy())
    original_store_pending = compactor._store_pending_best_effort
    failure_recorded_while_quiesced = False

    def fail_vacuum(_connection: sqlite3.Connection) -> None:
        raise sqlite3.OperationalError("injected")

    def store_pending_while_quiesced(**kwargs: object) -> None:
        nonlocal failure_recorded_while_quiesced
        with pytest.raises(
            CharactersRAGDBError, match="database_maintenance_in_progress"
        ):
            database.get_connection()
        failure_recorded_while_quiesced = True
        original_store_pending(**kwargs)  # type: ignore[arg-type]

    compactor._vacuum = fail_vacuum  # type: ignore[method-assign]
    compactor._store_pending_best_effort = (  # type: ignore[method-assign]
        store_pending_while_quiesced
    )
    outcome = compactor.run_after_gc(gc_result)

    assert outcome.completed is False
    assert outcome.reason_code == "sqlite_failure"
    assert failure_recorded_while_quiesced is True
    connection = database.get_connection()
    assert connection.execute("PRAGMA quick_check(1)").fetchone()[0] == "ok"
    assert connection.execute(
        "SELECT COUNT(*) FROM console_trace_owners "
        "WHERE conversation_id IN (?, ?)",
        (source_id, child_id),
    ).fetchone()[0] == 2
    state = connection.execute(
        "SELECT status, reason_code, retry_count, next_retry_at "
        "FROM console_trace_compaction_state WHERE singleton_id = 1"
    ).fetchone()
    assert state[0] == "pending"
    assert state[1] == "sqlite_failure"
    assert state[2] == 1
    assert state[3] is not None
    database.close_connection()
