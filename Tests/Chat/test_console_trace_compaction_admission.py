"""Admission gates for automatic physical Console trace compaction."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
import threading

import pytest

from tldw_chatbook.Chat.console_trace_maintenance import (
    PhysicalTraceCompactor,
    TraceCompactionPolicy,
    TraceGarbageCollector,
)
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.config import resolve_trace_compaction_policy


@pytest.fixture
def db(tmp_path: Path) -> Iterator[CharactersRAGDB]:
    database = CharactersRAGDB(tmp_path / "admission.sqlite", "admission")
    with database.transaction(immediate=True) as cursor:
        cursor.execute(
            "UPDATE console_trace_migration_state SET status = 'logical_complete' "
            "WHERE migration_name = 'legacy_exchange_normalization'"
        )
    yield database
    database.close_connection()


def _completed_gc(db: CharactersRAGDB, request_id: str = "gc-admission"):
    return TraceGarbageCollector(db).collect(request_id=request_id)


def _policy(**overrides: object) -> TraceCompactionPolicy:
    values: dict[str, object] = {
        "min_database_bytes": 1,
        "min_freelist_bytes": 0,
        "min_freelist_ratio": 0.0,
        "min_idle_seconds": 0.0,
        "retry_initial_seconds": 60.0,
        "retry_max_seconds": 600.0,
        "quiesce_timeout_seconds": 0.5,
        "disk_safety_margin_bytes": 0,
    }
    values.update(overrides)
    return TraceCompactionPolicy(**values)


def test_compaction_policy_resolves_explicit_bounded_config() -> None:
    policy = resolve_trace_compaction_policy(
        {
            "trace_compaction_min_database_bytes": "1024",
            "trace_compaction_min_freelist_bytes": 512,
            "trace_compaction_min_freelist_ratio": "0.25",
            "trace_compaction_min_idle_seconds": 45,
            "trace_compaction_retry_initial_seconds": 10,
            "trace_compaction_retry_max_seconds": 100,
            "trace_compaction_quiesce_timeout_seconds": 2,
            "trace_compaction_disk_safety_margin_bytes": 4096,
        }
    )

    assert policy == TraceCompactionPolicy(
        min_database_bytes=1024,
        min_freelist_bytes=512,
        min_freelist_ratio=0.25,
        min_idle_seconds=45.0,
        retry_initial_seconds=10.0,
        retry_max_seconds=100.0,
        quiesce_timeout_seconds=2.0,
        disk_safety_margin_bytes=4096,
    )


@pytest.mark.asyncio
async def test_provider_dispatch_gate_yields_until_maintenance_resumes() -> None:
    controller = object.__new__(ConsoleChatController)
    controller._trace_maintenance_dispatch_paused = threading.Event()
    controller.pause_trace_maintenance_dispatch()

    waiter = asyncio.create_task(controller._wait_for_trace_maintenance_dispatch())
    await asyncio.sleep(0)
    assert waiter.done() is False

    controller.resume_trace_maintenance_dispatch()
    await asyncio.wait_for(waiter, timeout=0.2)


def test_admission_requires_a_successful_durable_logical_gc(
    db: CharactersRAGDB,
) -> None:
    gc_result = _completed_gc(db)
    altered = replace(gc_result, status="stale_epoch")

    outcome = PhysicalTraceCompactor(db, policy=_policy()).run_after_gc(altered)

    assert outcome.admitted is False
    assert outcome.reason_code == "logical_gc_incomplete"


def test_admission_defers_for_provider_activity_and_always_resumes_dispatch(
    db: CharactersRAGDB,
) -> None:
    events: list[str] = []
    compactor = PhysicalTraceCompactor(
        db,
        policy=_policy(),
        provider_active=lambda: True,
        pause_dispatch=lambda: events.append("pause"),
        resume_dispatch=lambda: events.append("resume"),
    )

    outcome = compactor.run_after_gc(_completed_gc(db))

    assert outcome.admitted is False
    assert outcome.reason_code == "provider_active"
    assert events == ["pause", "resume"]
    assert db.get_connection().execute("SELECT 1").fetchone()[0] == 1


def test_admission_defers_for_activity_and_size_thresholds(
    db: CharactersRAGDB,
) -> None:
    gc_result = _completed_gc(db)
    active = PhysicalTraceCompactor(
        db,
        policy=_policy(min_idle_seconds=30.0),
        idle_seconds=lambda: 29.9,
    ).run_after_gc(gc_result)
    small = PhysicalTraceCompactor(
        db,
        policy=_policy(min_database_bytes=gc_result.allocated_bytes_after + 1),
    ).run_after_gc(gc_result)
    low_freelist = PhysicalTraceCompactor(
        db,
        policy=_policy(min_freelist_bytes=gc_result.freelist_bytes_after + 1),
    ).run_after_gc(gc_result)

    assert active.reason_code == "activity_threshold"
    assert small.reason_code == "database_threshold"
    assert low_freelist.reason_code == "freelist_threshold"
    assert not active.admitted and not small.admitted and not low_freelist.admitted
    state = db.get_console_trace_compaction_status()
    assert state["status"] == "pending"
    assert state["reason_code"] == "freelist_threshold"


def test_admission_rechecks_live_storage_thresholds_for_the_same_gc_result(
    db: CharactersRAGDB,
) -> None:
    gc_result = _completed_gc(db)
    compactor = PhysicalTraceCompactor(
        db,
        policy=_policy(
            min_database_bytes=gc_result.allocated_bytes_after + 512 * 1024,
            min_freelist_bytes=256 * 1024,
        ),
    )

    first = compactor.run_after_gc(gc_result)
    assert first.reason_code == "database_threshold"

    with db.transaction(immediate=True) as cursor:
        cursor.execute("CREATE TABLE threshold_payload(value BLOB NOT NULL)")
        cursor.execute(
            "INSERT INTO threshold_payload(value) VALUES (zeroblob(?))",
            (2 * 1024 * 1024,),
        )
        cursor.execute("DELETE FROM threshold_payload")

    second = compactor.run_after_gc(gc_result)

    assert second.completed is True
    assert second.reason_code == "complete"
    assert second.allocated_bytes_before >= gc_result.allocated_bytes_after + 512 * 1024
    assert second.freelist_bytes_before >= 256 * 1024


def test_admission_defers_when_another_maintenance_owner_holds_the_lease(
    db: CharactersRAGDB,
) -> None:
    gc_result = _completed_gc(db)
    with db.transaction(immediate=True) as cursor:
        cursor.execute(
            "UPDATE console_trace_maintenance_state SET state = 'marking', "
            "lease_id = 'other', lease_owner = 'trace_gc', "
            "lease_expires_at = datetime('now', '+5 minutes'), marked_epoch = 0 "
            "WHERE singleton_id = 1"
        )

    outcome = PhysicalTraceCompactor(db, policy=_policy()).run_after_gc(gc_result)

    assert outcome.admitted is False
    assert outcome.reason_code == "maintenance_busy"


def test_wal_disk_and_lease_failures_remain_pending_and_retryable(
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gc_result = _completed_gc(db)
    dispatch_events: list[str] = []
    no_disk = PhysicalTraceCompactor(
        db,
        policy=_policy(),
        disk_free_bytes=lambda _path: 0,
        pause_dispatch=lambda: dispatch_events.append("pause"),
        resume_dispatch=lambda: dispatch_events.append("resume"),
    ).run_after_gc(gc_result)
    assert no_disk.reason_code == "insufficient_disk"
    assert dispatch_events == ["pause", "resume"]
    db.get_connection().execute(
        "UPDATE console_trace_compaction_state SET next_retry_at = NULL"
    )

    checkpoint = PhysicalTraceCompactor(db, policy=_policy())
    monkeypatch.setattr(
        checkpoint,
        "_checkpoint_wal",
        lambda _connection: (_ for _ in ()).throw(RuntimeError("wal")),
    )
    wal = checkpoint.run_after_gc(gc_result)
    assert wal.reason_code == "wal_checkpoint_failed"
    db.get_connection().execute(
        "UPDATE console_trace_compaction_state SET next_retry_at = NULL"
    )

    lease = PhysicalTraceCompactor(db, policy=_policy())
    monkeypatch.setattr(lease, "_lease_current", lambda _connection, _attempt: False)
    lost = lease.run_after_gc(gc_result)
    assert lost.reason_code == "lease_lost"

    row = db.get_connection().execute(
        "SELECT status, reason_code, retry_count, next_retry_at "
        "FROM console_trace_compaction_state WHERE singleton_id = 1"
    ).fetchone()
    assert row[0] == "pending"
    assert row[1] == "lease_lost"
    assert 1 <= row[2] <= 32
    assert row[3] is not None


def test_retry_backoff_prevents_an_immediate_second_attempt(
    db: CharactersRAGDB,
) -> None:
    gc_result = _completed_gc(db)
    compactor = PhysicalTraceCompactor(
        db,
        policy=_policy(),
        disk_free_bytes=lambda _path: 0,
    )
    first = compactor.run_after_gc(gc_result)
    second = compactor.run_after_gc(gc_result)

    assert first.reason_code == "insufficient_disk"
    assert second.reason_code == "retry_backoff"
    assert second.admitted is False


def test_expired_compaction_lease_recovers_and_can_retry(
    db: CharactersRAGDB,
) -> None:
    with db.transaction(immediate=True) as cursor:
        cursor.execute(
            "UPDATE console_trace_maintenance_state SET state = 'compacting', "
            "lease_id = 'expired-attempt', lease_owner = 'trace_compaction', "
            "lease_expires_at = datetime('now', '-1 minute'), marked_epoch = NULL "
            "WHERE singleton_id = 1"
        )
        cursor.execute(
            "UPDATE console_trace_compaction_state SET status = 'running', "
            "reason_code = 'running', attempt_id = 'expired-attempt' "
            "WHERE singleton_id = 1"
        )

    gc_result = _completed_gc(db, "gc-after-interruption")
    recovered = db.get_connection().execute(
        "SELECT status, reason_code, retry_count, next_retry_at "
        "FROM console_trace_compaction_state WHERE singleton_id = 1"
    ).fetchone()
    outcome = PhysicalTraceCompactor(db, policy=_policy()).run_after_gc(gc_result)

    assert tuple(recovered) == ("pending", "interrupted", 1, None)
    assert outcome.completed is True
