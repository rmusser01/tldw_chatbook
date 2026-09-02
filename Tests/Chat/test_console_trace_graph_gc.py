"""Epoch-safe garbage collection for the Console semantic trace graph."""

from __future__ import annotations

from collections.abc import Iterator
import sqlite3

import pytest

from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_trace_maintenance import TraceGarbageCollector
from tldw_chatbook.Chat.console_trace_models import (
    FrozenTracePolicy,
    SemanticRevisionRef,
    TraceCallState,
    TraceContentRef,
    new_opaque_id,
)
from tldw_chatbook.Chat.console_trace_repository import ConsoleTraceRepository
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture
def db() -> Iterator[CharactersRAGDB]:
    database = CharactersRAGDB(":memory:", "console-trace-gc-test")
    yield database
    database.close_connection()


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


def _root_graph(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> tuple[str, str, str, str, str]:
    conversation_id, message_id = _conversation(db, "source")
    with db.transaction(immediate=True) as cursor:
        segment = repository.create_segment(cursor)
        owner = repository.attach_owner(
            cursor,
            conversation_id=conversation_id,
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
            source_conversation_id=conversation_id,
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
    return (
        conversation_id,
        owner.owner_id,
        segment.segment_id,
        policy.policy_id,
        node.node_id,
    )


def _terminal_call(
    cursor: sqlite3.Cursor,
    repository: ConsoleTraceRepository,
    *,
    owner_id: str,
    segment_id: str,
    policy_id: str,
    turn_id: str,
    event_sequence: int,
) -> str:
    call = repository.reserve_call(
        cursor,
        owner_id=owner_id,
        segment_id=segment_id,
        turn_id=turn_id,
        run_id=f"run-{turn_id}",
        call_sequence=0,
        idempotency_key=f"gc-{owner_id}-{turn_id}",
        policy_id=policy_id,
    )
    repository.append_event(
        cursor,
        segment_id=segment_id,
        sequence=event_sequence,
        event_type="call_boundary",
        call_id=call.call_id,
    )
    repository.advance_call_state(
        cursor,
        call_id=call.call_id,
        target=TraceCallState.NOT_DISPATCHED,
        occurred_at=f"2026-08-31T12:00:{event_sequence:02d}Z",
    )
    return call.call_id


def _finish_legacy_migration(db: CharactersRAGDB) -> None:
    with db.transaction(immediate=True) as cursor:
        cursor.execute(
            "UPDATE console_trace_migration_state "
            "SET status = 'logical_complete', updated_at = CURRENT_TIMESTAMP "
            "WHERE migration_name = 'legacy_exchange_normalization'"
        )


def test_terminal_lifecycle_transition_advances_reachability_epoch_once(
    db: CharactersRAGDB,
) -> None:
    repository = ConsoleTraceRepository()
    _conversation_id, owner_id, segment_id, policy_id, _node_id = _root_graph(
        db, repository
    )
    with db.transaction(immediate=True) as cursor:
        call = repository.reserve_call(
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            turn_id="turn-open",
            run_id="run-open",
            call_sequence=0,
            idempotency_key="gc-open-call",
            policy_id=policy_id,
        )
        before = repository.get_graph_epoch(cursor)
        repository.advance_call_state(
            cursor,
            call_id=call.call_id,
            target=TraceCallState.NOT_DISPATCHED,
            occurred_at="2026-08-31T12:00:00Z",
        )
        assert repository.get_graph_epoch(cursor) == before + 1


def test_direct_trace_deletion_fails_closed_without_sweep_grant(
    db: CharactersRAGDB,
) -> None:
    repository = ConsoleTraceRepository()
    _conversation_id, _owner_id, _segment_id, _policy_id, node_id = _root_graph(
        db, repository
    )
    with pytest.raises(sqlite3.DatabaseError, match="trace GC deletion authorization"):
        with db.transaction(immediate=True) as cursor:
            cursor.execute(
                "DELETE FROM console_trace_surface_nodes WHERE node_id = ?", (node_id,)
            )


def test_raw_sqlite_connection_cannot_bypass_trace_delete_guard(
    tmp_path,
) -> None:
    path = tmp_path / "trace-gc-raw-connection.sqlite"
    database = CharactersRAGDB(path, "trace-gc-raw-connection")
    repository = ConsoleTraceRepository()
    _conversation_id, _owner_id, _segment_id, _policy_id, node_id = _root_graph(
        database, repository
    )
    database.close_connection()

    raw = sqlite3.connect(path)
    try:
        with pytest.raises(sqlite3.OperationalError, match="no such function"):
            raw.execute(
                "DELETE FROM console_trace_surface_nodes WHERE node_id = ?", (node_id,)
            )
    finally:
        raw.close()


def test_stale_mark_never_sweeps_after_concurrent_reachability_change(
    db: CharactersRAGDB,
) -> None:
    repository = ConsoleTraceRepository()
    source_id, _owner_id, segment_id, _policy_id, _node_id = _root_graph(
        db, repository
    )
    _finish_legacy_migration(db)
    collector = TraceGarbageCollector(db)
    marked = collector.mark(request_id="gc-stale")

    child_id, _message_id = _conversation(db, "new owner")
    with db.transaction(immediate=True) as cursor:
        repository.attach_owner(
            cursor,
            conversation_id=child_id,
            root_segment_id=repository.create_segment(cursor).segment_id,
        )

    result = collector.sweep(request_id="gc-stale")

    assert result.status == "stale_epoch"
    assert result.marked_epoch == marked.marked_epoch
    assert result.swept_epoch is None
    with db.transaction() as cursor:
        assert repository.read_conversation_call_lineage(cursor, source_id) == ()
    assert db.get_connection().execute(
        "SELECT 1 FROM console_trace_segments WHERE segment_id = ?", (segment_id,)
    ).fetchone() is not None


def test_zero_event_terminal_call_in_fork_prefix_survives_source_purge(
    db: CharactersRAGDB,
) -> None:
    repository = ConsoleTraceRepository()
    source_id, source_owner_id, source_segment_id, policy_id, _node_id = (
        _root_graph(db, repository)
    )
    child_id, _child_message_id = _conversation(db, "child")
    with db.transaction(immediate=True) as cursor:
        call = repository.reserve_call(
            cursor,
            owner_id=source_owner_id,
            segment_id=source_segment_id,
            turn_id="shared",
            run_id="run-shared",
            call_sequence=0,
            idempotency_key="gc-zero-event-shared-call",
            policy_id=policy_id,
        )
        repository.advance_call_state(
            cursor,
            call_id=call.call_id,
            target=TraceCallState.NOT_DISPATCHED,
            occurred_at="2026-08-31T12:00:00Z",
        )
        repository.append_event(
            cursor,
            segment_id=source_segment_id,
            sequence=1,
            event_type="turn_boundary",
            turn_id="shared",
        )
        boundary = repository.capture_fork_boundary(
            cursor,
            conversation_id=source_id,
            included_turn_ids=("shared",),
        )
        assert boundary is not None
        repository.attach_fork_owner(
            cursor,
            conversation_id=child_id,
            boundary=boundary,
        )
        repository.detach_owner(
            cursor,
            owner_id=source_owner_id,
            detached_at="2026-08-31T12:01:00Z",
        )
    _finish_legacy_migration(db)

    result = TraceGarbageCollector(db).collect(request_id="gc-zero-event-prefix")

    assert result.status == "completed"
    assert db.get_connection().execute(
        "SELECT 1 FROM console_trace_calls WHERE call_id = ?", (call.call_id,)
    ).fetchone() is not None


def test_detached_source_suffix_is_reclaimed_but_shared_fork_prefix_survives(
    db: CharactersRAGDB,
) -> None:
    repository = ConsoleTraceRepository()
    source_id, source_owner_id, source_segment_id, policy_id, _node_id = _root_graph(
        db, repository
    )
    child_id, _child_message_id = _conversation(db, "child")
    with db.transaction(immediate=True) as cursor:
        shared_call_id = _terminal_call(
            cursor,
            repository,
            owner_id=source_owner_id,
            segment_id=source_segment_id,
            policy_id=policy_id,
            turn_id="shared",
            event_sequence=1,
        )
        boundary = repository.capture_fork_boundary(
            cursor,
            conversation_id=source_id,
            included_turn_ids=("shared",),
        )
        assert boundary is not None
        repository.attach_fork_owner(
            cursor,
            conversation_id=child_id,
            boundary=boundary,
        )
        dead_suffix_call_id = _terminal_call(
            cursor,
            repository,
            owner_id=source_owner_id,
            segment_id=source_segment_id,
            policy_id=policy_id,
            turn_id="source-suffix",
            event_sequence=2,
        )
        repository.detach_owner(
            cursor,
            owner_id=source_owner_id,
            detached_at="2026-08-31T12:01:00Z",
        )
    _finish_legacy_migration(db)

    result = TraceGarbageCollector(db).collect(request_id="gc-shared-prefix")

    assert result.status == "completed"
    assert result.deleted_rows["console_trace_calls"] == 1
    connection = db.get_connection()
    assert connection.execute(
        "SELECT 1 FROM console_trace_calls WHERE call_id = ?", (shared_call_id,)
    ).fetchone() is not None
    assert connection.execute(
        "SELECT 1 FROM console_trace_calls WHERE call_id = ?", (dead_suffix_call_id,)
    ).fetchone() is None
    assert [
        call.call_id
        for call in repository.read_conversation_call_lineage(
            connection.cursor(), child_id
        )
    ] == [shared_call_id]


def test_completed_request_is_idempotent_and_reports_logical_reclamation(
    db: CharactersRAGDB,
) -> None:
    repository = ConsoleTraceRepository()
    conversation_id, owner_id, segment_id, policy_id, _node_id = _root_graph(
        db, repository
    )
    with db.transaction(immediate=True) as cursor:
        _terminal_call(
            cursor,
            repository,
            owner_id=owner_id,
            segment_id=segment_id,
            policy_id=policy_id,
            turn_id="purged",
            event_sequence=1,
        )
        repository.detach_owner(
            cursor,
            owner_id=owner_id,
            detached_at="2026-08-31T12:01:00Z",
        )
    _finish_legacy_migration(db)
    collector = TraceGarbageCollector(db)

    first = collector.collect(request_id="gc-idempotent")
    second = collector.collect(request_id="gc-idempotent")

    assert first == second
    assert first.status == "completed"
    assert first.logical_rows > 0
    assert first.logical_bytes >= 0
    assert first.logical_bytes == first.reclaimed_bytes
    assert first.logical_live_bytes >= 0
    assert first.reclaimed_pages >= 0
    assert first.freelist_bytes_before == (
        first.freelist_pages_before * first.page_size_bytes
    )
    assert first.freelist_bytes_after == (
        first.freelist_pages_after * first.page_size_bytes
    )
    assert first.allocated_bytes_before == (
        first.allocated_pages_before * first.page_size_bytes
    )
    assert first.allocated_bytes_after == (
        first.allocated_pages_after * first.page_size_bytes
    )
    assert first.freelist_pages_after >= first.freelist_pages_before
    assert first.wal_bytes >= 0
    assert repository.read_conversation_call_lineage(
        db.get_connection().cursor(), conversation_id
    ) == ()


def test_soft_deleted_conversation_remains_an_attached_root(
    db: CharactersRAGDB,
) -> None:
    repository = ConsoleTraceRepository()
    conversation_id, owner_id, segment_id, _policy_id, _node_id = _root_graph(
        db, repository
    )
    with db.transaction(immediate=True) as cursor:
        cursor.execute(
            "UPDATE conversations SET deleted = 1 WHERE id = ?", (conversation_id,)
        )
    _finish_legacy_migration(db)

    result = TraceGarbageCollector(db).collect(request_id="gc-soft-delete")

    assert result.status == "completed"
    connection = db.get_connection()
    assert connection.execute(
        "SELECT 1 FROM console_trace_owners WHERE owner_id = ?", (owner_id,)
    ).fetchone() is not None
    assert connection.execute(
        "SELECT 1 FROM console_trace_segments WHERE segment_id = ?", (segment_id,)
    ).fetchone() is not None


def test_explicit_retention_keeps_a_detached_owner_graph(
    db: CharactersRAGDB,
) -> None:
    repository = ConsoleTraceRepository()
    conversation_id, owner_id, segment_id, _policy_id, _node_id = _root_graph(
        db, repository
    )
    collector = TraceGarbageCollector(db)
    retention_id = collector.retain_conversation(
        conversation_id=conversation_id,
        retain_until="2999-01-01T00:00:00Z",
        reason_code="legal_hold",
    )
    with db.transaction(immediate=True) as cursor:
        repository.detach_owner(
            cursor,
            owner_id=owner_id,
            detached_at="2026-08-31T12:01:00Z",
        )
    _finish_legacy_migration(db)

    result = collector.collect(request_id="gc-retained-owner")

    assert result.status == "completed"
    connection = db.get_connection()
    assert connection.execute(
        "SELECT 1 FROM console_trace_retention_roots WHERE retention_id = ?",
        (retention_id,),
    ).fetchone() is not None
    assert connection.execute(
        "SELECT 1 FROM console_trace_segments WHERE segment_id = ?", (segment_id,)
    ).fetchone() is not None


def test_expired_retention_is_removed_and_no_longer_roots_graph(
    db: CharactersRAGDB,
) -> None:
    repository = ConsoleTraceRepository()
    conversation_id, owner_id, segment_id, _policy_id, _node_id = _root_graph(
        db, repository
    )
    collector = TraceGarbageCollector(db)
    retention_id = collector.retain_conversation(
        conversation_id=conversation_id,
        retain_until="2000-01-01T00:00:00Z",
        reason_code="expired_hold",
    )
    with db.transaction(immediate=True) as cursor:
        repository.detach_owner(
            cursor,
            owner_id=owner_id,
            detached_at="2026-08-31T12:01:00Z",
        )
    _finish_legacy_migration(db)

    result = collector.collect(request_id="gc-expired-retention")

    assert result.status == "completed"
    connection = db.get_connection()
    assert connection.execute(
        "SELECT 1 FROM console_trace_retention_roots WHERE retention_id = ?",
        (retention_id,),
    ).fetchone() is None
    assert connection.execute(
        "SELECT 1 FROM console_trace_segments WHERE segment_id = ?", (segment_id,)
    ).fetchone() is None


def test_retention_deadline_requires_explicit_utc_offset(
    db: CharactersRAGDB,
) -> None:
    repository = ConsoleTraceRepository()
    conversation_id, _owner_id, _segment_id, _policy_id, _node_id = _root_graph(
        db, repository
    )

    with pytest.raises(ValueError, match="retain_until"):
        TraceGarbageCollector(db).retain_conversation(
            conversation_id=conversation_id,
            retain_until="2999-01-01 00:00:00",
            reason_code="ambiguous_clock",
        )


@pytest.mark.parametrize(
    "detached_at",
    ("not-a-timestamp", "2026-08-31T12:00:00"),
)
def test_purge_rejects_malformed_or_non_utc_detach_timestamp(
    db: CharactersRAGDB,
    detached_at: str,
) -> None:
    repository = ConsoleTraceRepository()
    conversation_id, owner_id, _segment_id, _policy_id, _node_id = _root_graph(
        db, repository
    )

    with pytest.raises(ValueError, match="detached_at"):
        TraceGarbageCollector(db).purge_conversation(
            conversation_id=conversation_id,
            request_id="gc-invalid-detached-at",
            detached_at=detached_at,
        )

    assert db.get_connection().execute(
        "SELECT attached FROM console_trace_owners WHERE owner_id = ?", (owner_id,)
    ).fetchone()[0] == 1


def test_pending_migration_is_a_conservative_all_graph_root(
    db: CharactersRAGDB,
) -> None:
    repository = ConsoleTraceRepository()
    _conversation_id, owner_id, segment_id, _policy_id, _node_id = _root_graph(
        db, repository
    )
    with db.transaction(immediate=True) as cursor:
        repository.detach_owner(
            cursor,
            owner_id=owner_id,
            detached_at="2026-08-31T12:01:00Z",
        )
    collector = TraceGarbageCollector(db)

    pending = collector.collect(request_id="gc-migration-pending")
    assert pending.logical_rows == 0
    assert db.get_connection().execute(
        "SELECT 1 FROM console_trace_segments WHERE segment_id = ?", (segment_id,)
    ).fetchone() is not None

    before_completion = repository.get_graph_epoch(db.get_connection().cursor())
    _finish_legacy_migration(db)
    assert (
        repository.get_graph_epoch(db.get_connection().cursor())
        == before_completion + 1
    )
    completed = collector.collect(request_id="gc-migration-complete")
    assert completed.logical_rows > 0
    assert db.get_connection().execute(
        "SELECT 1 FROM console_trace_segments WHERE segment_id = ?", (segment_id,)
    ).fetchone() is None


def test_open_call_roots_detached_graph_until_terminal_settlement(
    db: CharactersRAGDB,
) -> None:
    repository = ConsoleTraceRepository()
    _conversation_id, owner_id, segment_id, policy_id, _node_id = _root_graph(
        db, repository
    )
    with db.transaction(immediate=True) as cursor:
        call = repository.reserve_call(
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            turn_id="open-call",
            run_id="run-open-call",
            call_sequence=0,
            idempotency_key="gc-open-call-root",
            policy_id=policy_id,
        )
        repository.detach_owner(
            cursor,
            owner_id=owner_id,
            detached_at="2026-08-31T12:01:00Z",
        )
    _finish_legacy_migration(db)
    collector = TraceGarbageCollector(db)

    open_result = collector.collect(request_id="gc-open-root")
    assert open_result.status == "completed"
    assert repository.get_call(db.get_connection().cursor(), call.call_id) is not None

    with db.transaction(immediate=True) as cursor:
        repository.advance_call_state(
            cursor,
            call_id=call.call_id,
            target=TraceCallState.NOT_DISPATCHED,
            occurred_at="2026-08-31T12:02:00Z",
        )
    terminal_result = collector.collect(request_id="gc-terminal-root")
    assert terminal_result.deleted_rows["console_trace_calls"] == 1
    assert repository.get_call(db.get_connection().cursor(), call.call_id) is None


def test_purge_conversation_is_retry_safe_and_owner_scoped(
    db: CharactersRAGDB,
) -> None:
    repository = ConsoleTraceRepository()
    first_id, _first_owner, _first_segment, _policy_id, _node_id = _root_graph(
        db, repository
    )
    second_id, _second_message = _conversation(db, "unrelated owner")
    with db.transaction(immediate=True) as cursor:
        second_segment = repository.create_segment(cursor)
        second_owner = repository.attach_owner(
            cursor,
            conversation_id=second_id,
            root_segment_id=second_segment.segment_id,
        )
    _finish_legacy_migration(db)
    service = ChatPersistenceService(db)

    first = service.purge_console_trace(
        conversation_id=first_id,
        request_id="gc-conversation-purge",
        detached_at="2026-08-31T12:01:00Z",
    )
    retry = service.purge_console_trace(
        conversation_id=first_id,
        request_id="gc-conversation-purge",
        detached_at="2026-08-31T12:01:00Z",
    )

    assert first == retry
    remaining = repository.get_owner(db.get_connection().cursor(), second_owner.owner_id)
    assert remaining is not None and remaining.attached
    assert remaining.conversation_id == second_id


def test_purge_reports_attached_fork_owners_that_retain_shared_history(
    db: CharactersRAGDB,
) -> None:
    repository = ConsoleTraceRepository()
    source_id, _source_owner, source_segment, _policy_id, source_head = _root_graph(
        db, repository
    )
    child_id, _child_message = _conversation(db, "fork owner")
    with db.transaction(immediate=True) as cursor:
        child_segment = repository.create_segment(
            cursor,
            parent_segment_id=source_segment,
            inherited_through_sequence=0,
            inherited_surface_head_id=source_head,
        )
        repository.attach_owner(
            cursor,
            conversation_id=child_id,
            root_segment_id=child_segment.segment_id,
        )
    _finish_legacy_migration(db)

    result = TraceGarbageCollector(db).purge_conversation(
        conversation_id=source_id,
        request_id="gc-purge-shared-report",
        detached_at="2026-08-31T12:01:00Z",
    )

    assert result.remaining_owner_conversation_ids == (child_id,)
    assert db.get_connection().execute(
        "SELECT 1 FROM console_trace_segments WHERE segment_id = ?",
        (source_segment,),
    ).fetchone() is not None


def test_interrupted_sweep_rolls_back_and_same_request_retries(
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = ConsoleTraceRepository()
    _conversation_id, owner_id, segment_id, _policy_id, _node_id = _root_graph(
        db, repository
    )
    with db.transaction(immediate=True) as cursor:
        repository.detach_owner(
            cursor,
            owner_id=owner_id,
            detached_at="2026-08-31T12:01:00Z",
        )
    _finish_legacy_migration(db)
    collector = TraceGarbageCollector(db)
    collector.mark(request_id="gc-interrupted-sweep")
    original = TraceGarbageCollector._sweep_unmarked

    def fail_after_deletion(cursor: sqlite3.Cursor, request_id: str) -> dict[str, int]:
        deleted = original(cursor, request_id)
        assert sum(deleted.values()) > 0
        raise RuntimeError("injected_sweep_failure")

    monkeypatch.setattr(
        TraceGarbageCollector,
        "_sweep_unmarked",
        staticmethod(fail_after_deletion),
    )
    with pytest.raises(RuntimeError, match="injected_sweep_failure"):
        collector.sweep(request_id="gc-interrupted-sweep")

    connection = db.get_connection()
    assert connection.execute(
        "SELECT state FROM console_trace_maintenance_state WHERE singleton_id = 1"
    ).fetchone()[0] == "marking"
    assert connection.execute(
        "SELECT 1 FROM console_trace_segments WHERE segment_id = ?", (segment_id,)
    ).fetchone() is not None

    monkeypatch.setattr(
        TraceGarbageCollector,
        "_sweep_unmarked",
        staticmethod(original),
    )
    result = collector.sweep(request_id="gc-interrupted-sweep")
    assert result.status == "completed"
    assert connection.execute(
        "SELECT 1 FROM console_trace_segments WHERE segment_id = ?", (segment_id,)
    ).fetchone() is None


def test_expired_gc_lease_can_be_taken_over_without_using_stale_marks(
    db: CharactersRAGDB,
) -> None:
    _root_graph(db, ConsoleTraceRepository())
    _finish_legacy_migration(db)
    collector = TraceGarbageCollector(db)
    collector.mark(request_id="gc-expired-owner")
    with db.transaction(immediate=True) as cursor:
        cursor.execute(
            "UPDATE console_trace_maintenance_state "
            "SET lease_expires_at = '2000-01-01 00:00:00' WHERE singleton_id = 1"
        )

    replacement = collector.mark(request_id="gc-takeover")

    connection = db.get_connection()
    assert replacement.request_id == "gc-takeover"
    assert connection.execute(
        "SELECT COUNT(*) FROM console_trace_gc_marks WHERE request_id = ?",
        ("gc-expired-owner",),
    ).fetchone()[0] == 0
    assert tuple(
        connection.execute(
            "SELECT state, lease_id FROM console_trace_maintenance_state "
            "WHERE singleton_id = 1"
        ).fetchone()
    ) == ("marking", "gc-takeover")


def test_compaction_heavy_stale_mark_preserves_every_row_until_remark(
    db: CharactersRAGDB,
) -> None:
    repository = ConsoleTraceRepository()
    _conversation_id, _owner_id, live_segment_id, _policy_id, _node_id = _root_graph(
        db, repository
    )
    _finish_legacy_migration(db)
    with db.transaction(immediate=True) as cursor:
        for index in range(256):
            repository.store_sanitized_artifact(
                cursor,
                sanitized_bytes=(f"orphan-{index}" * 64).encode(),
                media_type="text/plain",
                normalization_version="gc-fixture-v1",
            )
    collector = TraceGarbageCollector(db)
    collector.mark(request_id="gc-heavy-stale")

    child_id, _message_id = _conversation(db, "concurrent owner")
    with db.transaction(immediate=True) as cursor:
        child_segment = repository.create_segment(cursor)
        repository.attach_owner(
            cursor,
            conversation_id=child_id,
            root_segment_id=child_segment.segment_id,
        )

    stale = collector.sweep(request_id="gc-heavy-stale")
    connection = db.get_connection()
    assert stale.status == "stale_epoch"
    assert connection.execute(
        "SELECT COUNT(*) FROM console_trace_artifacts"
    ).fetchone()[0] == 256
    assert connection.execute(
        "SELECT 1 FROM console_trace_segments WHERE segment_id = ?", (live_segment_id,)
    ).fetchone() is not None
    assert connection.execute(
        "SELECT 1 FROM console_trace_segments WHERE segment_id = ?",
        (child_segment.segment_id,),
    ).fetchone() is not None

    swept = collector.collect(request_id="gc-heavy-remark")
    assert swept.deleted_rows["console_trace_artifacts"] == 256
    assert connection.execute(
        "SELECT 1 FROM console_trace_segments WHERE segment_id = ?", (live_segment_id,)
    ).fetchone() is not None
    assert connection.execute(
        "SELECT 1 FROM console_trace_segments WHERE segment_id = ?",
        (child_segment.segment_id,),
    ).fetchone() is not None


def test_reopen_has_no_trace_payload_in_any_durable_owner(tmp_path) -> None:
    path = tmp_path / "trace-gc-privacy.sqlite"
    database = CharactersRAGDB(path, "trace-gc-privacy")
    repository = ConsoleTraceRepository()
    conversation_id, owner_id, segment_id, _policy_id, node_id = _root_graph(
        database, repository
    )
    secret = b"person@example.com sk-live-private-value"
    with database.transaction(immediate=True) as cursor:
        artifact = repository.store_sanitized_artifact(
            cursor,
            sanitized_bytes=secret,
            media_type="text/plain",
            normalization_version="gc-privacy-v1",
        )
        artifact_node = repository.append_surface_node(
            cursor,
            segment_id=segment_id,
            sequence=1,
            predecessor_node_id=node_id,
            component_kind="tool_result",
            reference=TraceContentRef(artifact.artifact_id, "tool_result"),
        )
        repository.append_event(
            cursor,
            segment_id=segment_id,
            sequence=1,
            event_type="surface_append",
            surface_node_id=artifact_node.node_id,
        )
        repository.detach_owner(
            cursor,
            owner_id=owner_id,
            detached_at="2026-08-31T12:01:00Z",
        )
    _finish_legacy_migration(database)
    result = TraceGarbageCollector(database).collect(request_id="gc-privacy")
    assert result.logical_bytes >= len(secret)
    database.close_connection()

    reopened = CharactersRAGDB(path, "trace-gc-privacy-reopen")
    try:
        connection = reopened.get_connection()
        payload_tables = (
            "console_trace_artifacts",
            "console_trace_calls",
            "console_trace_events",
            "console_trace_header_components",
            "console_trace_owners",
            "console_trace_policies",
            "console_trace_redaction_spans",
            "console_trace_request_headers",
            "console_trace_response_links",
            "console_trace_revision_bindings",
            "console_trace_segments",
            "console_trace_semantic_revisions",
            "console_trace_surface_nodes",
            "console_trace_surface_replacements",
        )
        assert all(
            connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
            for table in payload_tables
        )
        result_blob = connection.execute(
            "SELECT result_json FROM console_trace_gc_runs WHERE request_id = 'gc-privacy'"
        ).fetchone()[0]
        assert "person@example.com" not in result_blob
        assert "sk-live-private-value" not in result_blob
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        reopened.close_connection()
