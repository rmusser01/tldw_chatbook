"""Physical trace compaction, reopen, integrity, and shared-fork retention."""

from __future__ import annotations

import hashlib
import sqlite3
import time
from pathlib import Path
from uuid import uuid4

import pytest

import tldw_chatbook.Chat.console_trace_maintenance as trace_maintenance
from tldw_chatbook.Canvas.repository import CanvasRepository
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
    path = tmp_path / "progress.sqlite"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE payload(value BLOB NOT NULL)")
    connection.executemany(
        "INSERT INTO payload(value) VALUES (?)",
        ((b"x" * 4096,) for _ in range(512)),
    )
    connection.execute("DELETE FROM payload")
    connection.commit()
    events: list[object] = []
    compactor = PhysicalTraceCompactor(
        type("Database", (), {"db_path_str": str(path)})(),
        progress=events.append,
    )
    compactor._PROGRESS_VM_STEPS = 1
    try:
        compactor._vacuum(connection)
        assert events
        assert connection.execute("PRAGMA quick_check(1)").fetchone()[0] == "ok"
    finally:
        connection.close()


def test_vacuum_shrinks_file_and_preserves_shared_fork_after_reopen(
    tmp_path: Path,
) -> None:
    path = tmp_path / "trace-compaction.sqlite"
    database = CharactersRAGDB(path, "trace-compaction")
    source_id, child_id, child_segment_id, call_id = _shared_fork_fixture(database)
    message_id = database.add_message(
        {
            "conversation_id": source_id,
            "sender": "assistant",
            "role": "assistant",
            "content": "Canvas compaction origin",
        }
    )
    assert message_id is not None
    canvas_source = "<main>compaction λ root</main>"
    canvas_child_source = "<main>compaction λ child</main>"
    canvas_repository = CanvasRepository(database)
    canvas = canvas_repository.create_canvas(
        source_id,
        title="Compaction root",
        source=canvas_source,
        runtime_profile="canvas-v1",
        actor_kind="assistant",
        origin_message_id=message_id,
        origin_turn_id="turn-canvas-root",
    )
    canvas_child = canvas_repository.append_revision(
        source_id,
        canvas.revision.canvas_id,
        parent_revision_id=canvas.revision.revision_id,
        title="Compaction child",
        source=canvas_child_source,
        runtime_profile="canvas-v1",
        actor_kind="assistant",
        origin_message_id=message_id,
        origin_turn_id="turn-canvas-child",
    )
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
            canvas_rows = cursor.execute(
                "SELECT id, parent_revision_id, sequence, html, content_sha256, "
                "html_bytes, origin_message_id, origin_turn_id "
                "FROM canvas_revisions WHERE canvas_id = ? ORDER BY sequence",
                (canvas.revision.canvas_id,),
            ).fetchall()
            assert [tuple(row) for row in canvas_rows] == [
                (
                    canvas.revision.revision_id,
                    None,
                    1,
                    canvas_source,
                    hashlib.sha256(canvas_source.encode("utf-8")).hexdigest(),
                    len(canvas_source.encode("utf-8")),
                    message_id,
                    "turn-canvas-root",
                ),
                (
                    canvas_child.revision_id,
                    canvas.revision.revision_id,
                    2,
                    canvas_child_source,
                    hashlib.sha256(canvas_child_source.encode("utf-8")).hexdigest(),
                    len(canvas_child_source.encode("utf-8")),
                    message_id,
                    "turn-canvas-child",
                ),
            ]
            state = cursor.execute(
                "SELECT status, reason_code, progress_basis_points, retry_count "
                "FROM console_trace_compaction_state WHERE singleton_id = 1"
            ).fetchone()
            assert tuple(state) == ("complete", "complete", 10000, 0)
    finally:
        reopened.close_connection()


def _canvas_fixture(
    database: CharactersRAGDB,
) -> tuple[str, str, str, str]:
    conversation_id, message_id = _conversation(database, "canvas maintenance")
    created = CanvasRepository(database).create_canvas(
        conversation_id,
        title="Maintenance root",
        source="<main>root</main>",
        runtime_profile="canvas-v1",
        actor_kind="assistant",
        origin_message_id=message_id,
        origin_turn_id="turn-root",
    )
    return (
        conversation_id,
        message_id,
        created.revision.canvas_id,
        created.revision.revision_id,
    )


@pytest.mark.parametrize("invalid_kind", ["utf8", "digest", "size"])
def test_maintenance_connection_rejects_invalid_canvas_payloads(
    invalid_kind: str,
    tmp_path: Path,
) -> None:
    path = tmp_path / f"maintenance-invalid-{invalid_kind}.sqlite"
    database = CharactersRAGDB(path, f"maintenance-invalid-{invalid_kind}")
    _conversation_id, message_id, canvas_id, parent_revision_id = _canvas_fixture(
        database
    )
    source_bytes = b"\x80" if invalid_kind == "utf8" else b"<main>child</main>"
    source: object = (
        sqlite3.Binary(source_bytes)
        if invalid_kind == "utf8"
        else source_bytes.decode("utf-8")
    )
    digest = hashlib.sha256(source_bytes).hexdigest()
    declared_bytes = len(source_bytes)
    if invalid_kind == "digest":
        digest = "0" * 64
    elif invalid_kind == "size":
        declared_bytes += 1
    statement = (
        "INSERT INTO canvas_revisions "
        "(id, canvas_id, parent_revision_id, sequence, title, runtime_profile, "
        "html, content_sha256, html_bytes, actor_kind, origin_message_id, "
        "origin_turn_id, created_at, deleted_at) "
        "VALUES (?, ?, ?, 2, 'invalid', 'canvas-v1', "
        + ("CAST(? AS TEXT)" if invalid_kind == "utf8" else "?")
        + ", ?, ?, 'assistant', ?, 'turn-invalid', "
        "'2026-09-08T00:00:00.000Z', NULL)"
    )

    try:
        with database.quiesce_connections(timeout_seconds=1.0):
            connection = PhysicalTraceCompactor(
                database, policy=_permissive_policy()
            )._open_maintenance_connection()
            try:
                with pytest.raises(sqlite3.IntegrityError):
                    connection.execute(
                        statement,
                        (
                            str(uuid4()),
                            canvas_id,
                            parent_revision_id,
                            source,
                            digest,
                            declared_bytes,
                            message_id,
                        ),
                    )
                assert (
                    connection.execute(
                        "SELECT COUNT(*) FROM canvas_revisions WHERE canvas_id = ?",
                        (canvas_id,),
                    ).fetchone()[0]
                    == 1
                )
            finally:
                connection.close()
    finally:
        database.close_connection()


def test_maintenance_connection_has_validator_without_mutation_authority(
    tmp_path: Path,
) -> None:
    path = tmp_path / "maintenance-authority.sqlite"
    database = CharactersRAGDB(path, "maintenance-authority")
    _conversation_id, _message_id, canvas_id, revision_id = _canvas_fixture(database)

    try:
        with database.quiesce_connections(timeout_seconds=1.0):
            connection = PhysicalTraceCompactor(
                database, policy=_permissive_policy()
            )._open_maintenance_connection()
            try:
                functions = {
                    str(row[0]) for row in connection.execute("PRAGMA function_list")
                }
                assert "canvas_revision_payload_valid" in functions
                assert "canvas_revision_delete_authorized" not in functions
                assert "console_semantic_mutation_authorized" not in functions
                assert "console_trace_gc_delete_authorized" not in functions
                with pytest.raises(sqlite3.IntegrityError, match="immutable"):
                    connection.execute(
                        "UPDATE canvas_revisions SET title = 'changed' WHERE id = ?",
                        (revision_id,),
                    )
                with pytest.raises(sqlite3.OperationalError, match="no such function"):
                    connection.execute(
                        "DELETE FROM canvas_revisions WHERE id = ?", (revision_id,)
                    )
                retained = connection.execute(
                    "SELECT id, title FROM canvas_revisions WHERE canvas_id = ?",
                    (canvas_id,),
                ).fetchall()
                assert [tuple(row) for row in retained] == [
                    (revision_id, "Maintenance root")
                ]
            finally:
                connection.close()
    finally:
        database.close_connection()


def test_maintenance_setup_failure_closes_handle_and_releases_exclusion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "maintenance-setup-failure.sqlite"
    database = CharactersRAGDB(path, "maintenance-setup-failure")
    try:
        opened: list[sqlite3.Connection] = []
        real_connect = trace_maintenance.connect_private_sqlite

        def recording_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
            connection = real_connect(*args, **kwargs)  # type: ignore[arg-type]
            opened.append(connection)
            return connection

        def fail_registration(_connection: sqlite3.Connection) -> None:
            raise RuntimeError("injected_registration_failure")

        monkeypatch.setattr(
            trace_maintenance, "connect_private_sqlite", recording_connect
        )
        monkeypatch.setattr(
            trace_maintenance,
            "_install_canvas_revision_payload_validator",
            fail_registration,
        )

        with (
            database.quiesce_connections(timeout_seconds=1.0),
            pytest.raises(RuntimeError, match="injected_registration_failure"),
        ):
            PhysicalTraceCompactor(
                database, policy=_permissive_policy()
            )._open_maintenance_connection()

        assert len(opened) == 1
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            opened[0].execute("SELECT 1")
        resumed = database.get_connection()
        assert resumed.execute("PRAGMA quick_check(1)").fetchone()[0] == "ok"
        assert database.registered_connection_count() == 1
        state = resumed.execute(
            "SELECT state, lease_id, lease_owner FROM console_trace_maintenance_state "
            "WHERE singleton_id = 1"
        ).fetchone()
        assert tuple(state) == ("idle", None, None)
    finally:
        database.close_connection()


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
