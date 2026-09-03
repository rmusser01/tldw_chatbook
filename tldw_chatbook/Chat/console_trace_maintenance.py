"""Idle, bounded maintenance for legacy Console trace normalization."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil
import sqlite3
import time
from types import MappingProxyType

from tldw_chatbook.Chat.console_trace_legacy import (
    LegacyDecodedByteLimitError,
    LegacyTraceNormalizer,
)
from tldw_chatbook.Chat.console_trace_models import new_opaque_id
from tldw_chatbook.Chat.console_trace_repository import ConsoleTraceRepository
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.private_sqlite import connect_private_sqlite


LEGACY_MIGRATION_NAME = "legacy_exchange_normalization"
MAX_LEGACY_BATCH_ROWS = 100
MAX_LEGACY_BATCH_BYTES = 4 * 1024 * 1024
MAX_LEGACY_BATCH_SECONDS = 0.100
_GC_ENTITY_TABLES = (
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


@dataclass(frozen=True, slots=True)
class LegacyMaintenanceBatch:
    """Content-free outcome of one bounded maintenance attempt."""

    admitted: bool
    processed_rows: int
    processed_bytes: int
    logical_complete: bool


@dataclass(frozen=True, slots=True)
class TraceGCMark:
    """Content-free identity of one durable reachability snapshot."""

    request_id: str
    marked_epoch: int
    marked_entities: int


@dataclass(frozen=True, slots=True)
class TraceGCResult:
    """Logical reclamation and physical-file observations for one GC request."""

    request_id: str
    status: str
    marked_epoch: int
    swept_epoch: int | None
    deleted_rows: Mapping[str, int]
    logical_rows: int
    logical_bytes: int
    logical_live_bytes: int
    reclaimed_bytes: int
    reclaimed_pages: int
    page_size_bytes: int
    freelist_pages_before: int
    freelist_pages_after: int
    freelist_bytes_before: int
    freelist_bytes_after: int
    allocated_pages_before: int
    allocated_pages_after: int
    allocated_bytes_before: int
    allocated_bytes_after: int
    wal_bytes: int
    remaining_owner_conversation_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TraceCompactionPolicy:
    """Explicit automatic-compaction thresholds and retry bounds."""

    min_database_bytes: int = 64 * 1024 * 1024
    min_freelist_bytes: int = 16 * 1024 * 1024
    min_freelist_ratio: float = 0.20
    min_idle_seconds: float = 30.0
    retry_initial_seconds: float = 300.0
    retry_max_seconds: float = 3600.0
    quiesce_timeout_seconds: float = 5.0
    disk_safety_margin_bytes: int = 64 * 1024 * 1024

    def __post_init__(self) -> None:
        integer_fields = (
            "min_database_bytes",
            "min_freelist_bytes",
            "disk_safety_margin_bytes",
        )
        if any(
            type(getattr(self, field)) is not int or getattr(self, field) < 0
            for field in integer_fields
        ):
            raise ValueError("trace_compaction_byte_threshold")
        numeric_fields = (
            "min_idle_seconds",
            "retry_initial_seconds",
            "retry_max_seconds",
            "quiesce_timeout_seconds",
        )
        if any(
            type(getattr(self, field)) not in {int, float}
            or float(getattr(self, field)) < 0
            for field in numeric_fields
        ):
            raise ValueError("trace_compaction_time_threshold")
        if (
            type(self.min_freelist_ratio) not in {int, float}
            or not 0.0 <= float(self.min_freelist_ratio) <= 1.0
        ):
            raise ValueError("min_freelist_ratio")
        if self.retry_max_seconds < self.retry_initial_seconds:
            raise ValueError("retry_max_seconds")


@dataclass(frozen=True, slots=True)
class TraceCompactionOutcome:
    """Content-free result of one physical-compaction attempt."""

    admitted: bool
    completed: bool
    reason_code: str
    attempt_id: str | None
    allocated_bytes_before: int
    allocated_bytes_after: int
    freelist_bytes_before: int
    freelist_bytes_after: int
    wal_bytes_before: int
    wal_bytes_after: int
    logical_live_bytes: int


@dataclass(frozen=True, slots=True)
class TraceCompactionProgress:
    """One bounded, content-free worker progress event."""

    stage: str
    progress_basis_points: int
    allocated_bytes: int
    freelist_bytes: int


class _TraceCompactionFailure(RuntimeError):
    """Internal stage failure carrying one bounded operational reason code."""


@contextmanager
def _maintenance_transaction(
    connection: sqlite3.Connection,
) -> Iterator[sqlite3.Connection]:
    """Own one transaction on the private connection used while quiesced."""

    if connection.in_transaction:
        raise RuntimeError("maintenance_transaction_already_active")
    connection.execute("BEGIN IMMEDIATE")
    with connection:
        yield connection


class PhysicalTraceCompactor:
    """Run an admitted same-file VACUUM under the durable maintenance lease."""

    _LEASE_SECONDS = 3600
    _PROGRESS_VM_STEPS = 1000

    def __init__(
        self,
        db: CharactersRAGDB,
        *,
        policy: TraceCompactionPolicy | None = None,
        provider_active: Callable[[], bool] | None = None,
        idle_seconds: Callable[[], float] | None = None,
        pause_dispatch: Callable[[], None] | None = None,
        resume_dispatch: Callable[[], None] | None = None,
        disk_free_bytes: Callable[[Path], int] | None = None,
        cancel_requested: Callable[[], bool] | None = None,
        progress: Callable[[TraceCompactionProgress], None] | None = None,
    ) -> None:
        self.db = db
        self.policy = policy or TraceCompactionPolicy()
        self.provider_active = provider_active or (lambda: False)
        self.idle_seconds = idle_seconds or (lambda: float("inf"))
        self.pause_dispatch = pause_dispatch or (lambda: None)
        self.resume_dispatch = resume_dispatch or (lambda: None)
        self.disk_free_bytes = disk_free_bytes or (
            lambda path: int(shutil.disk_usage(path).free)
        )
        self.cancel_requested = cancel_requested or (lambda: False)
        self.progress = progress or (lambda _event: None)

    def run_after_gc(self, gc_result: TraceGCResult) -> TraceCompactionOutcome:
        """Compact only after the exact durable successful logical GC result.

        Args:
            gc_result: Completed durable collection result authorizing physical
                reclamation of pages that are no longer logically reachable.

        Returns:
            A content-free admission, completion, and byte-metric outcome.
        """

        if not isinstance(gc_result, TraceGCResult) or gc_result.status != "completed":
            return self._deferred(gc_result, "logical_gc_incomplete")
        if self.db.is_memory_db:
            return self._deferred(gc_result, "memory_database")
        gc_result = self._refresh_storage_observation(gc_result)
        eligibility = self._eligibility_reason(gc_result)
        if eligibility is not None:
            return self._deferred(gc_result, eligibility)

        paused = False
        try:
            self.pause_dispatch()
            paused = True
            if self.provider_active():
                return self._deferred(gc_result, "provider_active")
            if float(self.idle_seconds()) < self.policy.min_idle_seconds:
                return self._deferred(gc_result, "activity_threshold")
            attempt_id = new_opaque_id()
            if not self._acquire_lease(gc_result, attempt_id):
                return self._deferred(gc_result, "maintenance_busy")
            return self._run_admitted(gc_result, attempt_id)
        finally:
            if paused:
                self.resume_dispatch()

    def _eligibility_reason(self, result: TraceGCResult) -> str | None:
        if result.allocated_bytes_after < self.policy.min_database_bytes:
            return "database_threshold"
        if result.freelist_bytes_after < self.policy.min_freelist_bytes:
            return "freelist_threshold"
        ratio = (
            result.freelist_bytes_after / result.allocated_bytes_after
            if result.allocated_bytes_after
            else 0.0
        )
        if ratio < self.policy.min_freelist_ratio:
            return "freelist_ratio_threshold"
        with self.db.transaction() as cursor:
            durable = cursor.execute(
                "SELECT status FROM console_trace_gc_runs WHERE request_id = ?",
                (result.request_id,),
            ).fetchone()
            retry = cursor.execute(
                "SELECT next_retry_at IS NOT NULL AND "
                "julianday(next_retry_at) > julianday('now') "
                "FROM console_trace_compaction_state WHERE singleton_id = 1"
            ).fetchone()
        if durable is None or durable[0] != "completed":
            return "logical_gc_unavailable"
        if retry is not None and bool(retry[0]):
            return "retry_backoff"
        return None

    def _refresh_storage_observation(self, result: TraceGCResult) -> TraceGCResult:
        with self.db.transaction() as cursor:
            page_size = int(cursor.execute("PRAGMA page_size").fetchone()[0])
            allocated_pages = int(
                cursor.execute("PRAGMA page_count").fetchone()[0]
            )
            freelist_pages = int(
                cursor.execute("PRAGMA freelist_count").fetchone()[0]
            )
        wal_path = Path(f"{self.db.db_path_str}-wal")
        try:
            wal_bytes = max(0, wal_path.stat().st_size)
        except OSError:
            wal_bytes = 0
        return replace(
            result,
            page_size_bytes=page_size,
            allocated_pages_after=allocated_pages,
            allocated_bytes_after=allocated_pages * page_size,
            freelist_pages_after=freelist_pages,
            freelist_bytes_after=freelist_pages * page_size,
            wal_bytes=wal_bytes,
        )

    def _acquire_lease(self, result: TraceGCResult, attempt_id: str) -> bool:
        with self.db.transaction(immediate=True) as cursor:
            maintenance = cursor.execute(
                "SELECT state FROM console_trace_maintenance_state "
                "WHERE singleton_id = 1"
            ).fetchone()
            if maintenance is None or maintenance[0] != "idle":
                return False
            durable = cursor.execute(
                "SELECT status FROM console_trace_gc_runs WHERE request_id = ?",
                (result.request_id,),
            ).fetchone()
            if durable is None or durable[0] != "completed":
                return False
            cursor.execute(
                "UPDATE console_trace_maintenance_state SET state = 'compacting', "
                "lease_id = ?, lease_owner = 'trace_compaction', "
                "lease_expires_at = datetime('now', ?), marked_epoch = NULL, "
                "updated_at = CURRENT_TIMESTAMP WHERE singleton_id = 1 AND state = 'idle'",
                (attempt_id, f"+{self._LEASE_SECONDS} seconds"),
            )
            if cursor.rowcount != 1:
                return False
            cursor.execute(
                "UPDATE console_trace_compaction_state SET status = 'running', "
                "reason_code = 'running', last_gc_request_id = ?, attempt_id = ?, "
                "progress_basis_points = 0, allocated_bytes_before = ?, "
                "freelist_bytes_before = ?, wal_bytes_before = ?, logical_live_bytes = ?, "
                "started_at = CURRENT_TIMESTAMP, completed_at = NULL, "
                "updated_at = CURRENT_TIMESTAMP WHERE singleton_id = 1",
                (
                    result.request_id,
                    attempt_id,
                    result.allocated_bytes_after,
                    result.freelist_bytes_after,
                    result.wal_bytes,
                    result.logical_live_bytes,
                ),
            )
            return True

    def _run_admitted(
        self,
        result: TraceGCResult,
        attempt_id: str,
    ) -> TraceCompactionOutcome:
        reason: str | None = None
        failure_recorded = False
        after = (result.allocated_bytes_after, result.freelist_bytes_after, result.wal_bytes)
        try:
            with self.db.quiesce_connections(
                timeout_seconds=self.policy.quiesce_timeout_seconds
            ):
                try:
                    connection = self._open_maintenance_connection()
                    try:
                        if connection.in_transaction:
                            raise _TraceCompactionFailure("active_transaction")
                        try:
                            self._checkpoint_wal(connection)
                        except Exception as exc:
                            raise _TraceCompactionFailure(
                                "wal_checkpoint_failed"
                            ) from exc
                        if not self._lease_current(connection, attempt_id):
                            raise _TraceCompactionFailure("lease_lost")
                        required = (
                            self._allocated_metrics(connection)[0]
                            + self.policy.disk_safety_margin_bytes
                        )
                        if (
                            self.disk_free_bytes(Path(self.db.db_path_str).parent)
                            < required
                        ):
                            raise _TraceCompactionFailure("insufficient_disk")
                        self._emit_progress(connection, "vacuum", 500)
                        self._vacuum(connection)
                    finally:
                        connection.close()

                    verification = self._open_maintenance_connection()
                    try:
                        if not self._lease_current(verification, attempt_id):
                            raise _TraceCompactionFailure("lease_lost")
                        integrity = verification.execute(
                            "PRAGMA quick_check(1)"
                        ).fetchone()
                        if integrity is None or integrity[0] != "ok":
                            raise _TraceCompactionFailure("integrity_check_failed")
                        after = self._allocated_metrics(verification)
                        self._store_completed(
                            verification,
                            attempt_id=attempt_id,
                            after=after,
                        )
                        self._emit_progress(verification, "complete", 10000)
                    finally:
                        verification.close()
                except _TraceCompactionFailure as exc:
                    reason = str(exc)
                except sqlite3.Error:
                    reason = "sqlite_failure"
                except Exception:
                    reason = "compaction_failure"
                if reason is not None:
                    self._store_pending_best_effort(
                        result=result,
                        attempt_id=attempt_id,
                        reason_code=reason,
                    )
                    failure_recorded = True
        except TimeoutError:
            reason = "connections_busy"
        except sqlite3.Error:
            reason = "sqlite_failure"
        except Exception:
            reason = "compaction_failure"

        if reason is not None:
            if not failure_recorded:
                self._store_pending_best_effort(
                    result=result,
                    attempt_id=attempt_id,
                    reason_code=reason,
                )
            return self._outcome(
                result,
                admitted=True,
                completed=False,
                reason_code=reason,
                attempt_id=attempt_id,
                after=after,
            )
        return self._outcome(
            result,
            admitted=True,
            completed=True,
            reason_code="complete",
            attempt_id=attempt_id,
            after=after,
        )

    def _vacuum(self, connection: sqlite3.Connection) -> None:
        callbacks = 0
        allocated, freelist, _wal = self._allocated_metrics(connection)

        def on_progress() -> int:
            nonlocal callbacks
            callbacks += 1
            if self.cancel_requested():
                return 1
            if callbacks % 64 == 0:
                self._report_progress(
                    TraceCompactionProgress(
                        "vacuum",
                        min(9500, 500 + callbacks),
                        allocated,
                        freelist,
                    )
                )
            return 0

        connection.set_progress_handler(on_progress, self._PROGRESS_VM_STEPS)
        try:
            connection.execute("VACUUM")
        except sqlite3.OperationalError as exc:
            reason = "cancelled" if self.cancel_requested() else "vacuum_failed"
            raise _TraceCompactionFailure(reason) from exc
        finally:
            connection.set_progress_handler(None, 0)

    @staticmethod
    def _checkpoint_wal(connection: sqlite3.Connection) -> None:
        row = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if row is None or int(row[0]) != 0:
            raise RuntimeError("wal_checkpoint_busy")

    @staticmethod
    def _lease_current(connection: sqlite3.Connection, attempt_id: str) -> bool:
        row = connection.execute(
            "SELECT state = 'compacting' AND lease_id = ? "
            "AND lease_owner = 'trace_compaction' "
            "AND julianday(lease_expires_at) > julianday('now') "
            "FROM console_trace_maintenance_state WHERE singleton_id = 1",
            (attempt_id,),
        ).fetchone()
        return row is not None and bool(row[0])

    def _open_maintenance_connection(self) -> sqlite3.Connection:
        connection = connect_private_sqlite(
            "db.chachanotes.primary",
            self.db.db_path_str,
            must_exist=True,
            check_same_thread=False,
            timeout=15,
        )
        connection.row_factory = sqlite3.Row
        connection.isolation_level = None
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _allocated_metrics(
        self, connection: sqlite3.Connection
    ) -> tuple[int, int, int]:
        page_size = int(connection.execute("PRAGMA page_size").fetchone()[0])
        pages = int(connection.execute("PRAGMA page_count").fetchone()[0])
        freelist = int(connection.execute("PRAGMA freelist_count").fetchone()[0])
        wal_path = Path(f"{self.db.db_path_str}-wal")
        try:
            wal_bytes = max(0, wal_path.stat().st_size)
        except OSError:
            wal_bytes = 0
        return pages * page_size, freelist * page_size, wal_bytes

    def _emit_progress(
        self,
        connection: sqlite3.Connection,
        stage: str,
        basis_points: int,
    ) -> None:
        allocated, freelist, _wal = self._allocated_metrics(connection)
        self._report_progress(
            TraceCompactionProgress(stage, basis_points, allocated, freelist)
        )

    def _report_progress(self, event: TraceCompactionProgress) -> None:
        """Keep observational progress callbacks from changing maintenance safety."""

        try:
            self.progress(event)
        except Exception:
            return

    def _store_completed(
        self,
        connection: sqlite3.Connection,
        *,
        attempt_id: str,
        after: tuple[int, int, int],
    ) -> None:
        with _maintenance_transaction(connection):
            if not self._lease_current(connection, attempt_id):
                raise _TraceCompactionFailure("lease_lost")
            connection.execute(
                "UPDATE console_trace_compaction_state SET status = 'complete', "
                "reason_code = 'complete', progress_basis_points = 10000, "
                "allocated_bytes_after = ?, freelist_bytes_after = ?, "
                "wal_bytes_after = ?, retry_count = 0, next_retry_at = NULL, "
                "completed_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP "
                "WHERE singleton_id = 1 AND attempt_id = ?",
                (*after, attempt_id),
            )
            if connection.execute("SELECT changes()").fetchone()[0] != 1:
                raise _TraceCompactionFailure("lease_lost")
            connection.execute(
                "UPDATE console_trace_maintenance_state SET state = 'idle', "
                "lease_id = NULL, lease_owner = NULL, lease_expires_at = NULL, "
                "marked_epoch = NULL, updated_at = CURRENT_TIMESTAMP "
                "WHERE singleton_id = 1 AND state = 'compacting' AND lease_id = ?",
                (attempt_id,),
            )
            if connection.execute("SELECT changes()").fetchone()[0] != 1:
                raise _TraceCompactionFailure("lease_lost")

    def _store_pending_best_effort(
        self,
        *,
        result: TraceGCResult,
        attempt_id: str,
        reason_code: str,
    ) -> None:
        reason_code = reason_code[:64] or "compaction_failure"
        try:
            connection = self._open_maintenance_connection()
            try:
                with _maintenance_transaction(connection):
                    row = connection.execute(
                        "SELECT retry_count FROM console_trace_compaction_state "
                        "WHERE singleton_id = 1"
                    ).fetchone()
                    retry_count = min(32, (int(row[0]) if row else 0) + 1)
                    delay = min(
                        self.policy.retry_max_seconds,
                        self.policy.retry_initial_seconds
                        * (2 ** max(0, retry_count - 1)),
                    )
                    retry_at = datetime.now(timezone.utc) + timedelta(seconds=delay)
                    connection.execute(
                        "UPDATE console_trace_compaction_state SET status = 'pending', "
                        "reason_code = ?, last_gc_request_id = ?, retry_count = ?, "
                        "next_retry_at = ?, progress_basis_points = 0, "
                        "updated_at = CURRENT_TIMESTAMP WHERE singleton_id = 1 "
                        "AND attempt_id = ?",
                        (
                            reason_code,
                            result.request_id,
                            retry_count,
                            retry_at.isoformat(),
                            attempt_id,
                        ),
                    )
                    if connection.execute("SELECT changes()").fetchone()[0] != 1:
                        return
                    connection.execute(
                        "UPDATE console_trace_maintenance_state SET state = 'idle', "
                        "lease_id = NULL, lease_owner = NULL, "
                        "lease_expires_at = NULL, marked_epoch = NULL, "
                        "updated_at = CURRENT_TIMESTAMP WHERE singleton_id = 1 "
                        "AND state = 'compacting' AND lease_id = ?",
                        (attempt_id,),
                    )
            finally:
                connection.close()
        except Exception:
            # The database remains the source of truth. If even the bounded
            # diagnostic cannot be persisted, the unchanged/expired lease is
            # still safely retryable on the next application start.
            return

    def _deferred(
        self,
        result: object,
        reason_code: str,
    ) -> TraceCompactionOutcome:
        if isinstance(result, TraceGCResult) and reason_code in {
            "database_threshold",
            "freelist_threshold",
            "freelist_ratio_threshold",
            "activity_threshold",
            "provider_active",
            "logical_gc_unavailable",
        }:
            self._store_deferred_best_effort(result, reason_code)
        allocated = int(getattr(result, "allocated_bytes_after", 0))
        freelist = int(getattr(result, "freelist_bytes_after", 0))
        wal = int(getattr(result, "wal_bytes", 0))
        logical = int(getattr(result, "logical_live_bytes", 0))
        return TraceCompactionOutcome(
            False,
            False,
            reason_code,
            None,
            allocated,
            allocated,
            freelist,
            freelist,
            wal,
            wal,
            logical,
        )

    def _store_deferred_best_effort(
        self,
        result: TraceGCResult,
        reason_code: str,
    ) -> None:
        """Persist one safe pending admission reason without disturbing a lease."""

        try:
            with self.db.transaction(immediate=True) as cursor:
                maintenance = cursor.execute(
                    "SELECT state FROM console_trace_maintenance_state "
                    "WHERE singleton_id = 1"
                ).fetchone()
                if maintenance is None or maintenance[0] != "idle":
                    return
                cursor.execute(
                    "UPDATE console_trace_compaction_state SET status = 'pending', "
                    "reason_code = ?, last_gc_request_id = ?, attempt_id = NULL, "
                    "retry_count = 0, next_retry_at = NULL, "
                    "progress_basis_points = 0, updated_at = CURRENT_TIMESTAMP "
                    "WHERE singleton_id = 1 AND status <> 'running'",
                    (reason_code, result.request_id),
                )
        except Exception:
            return

    @staticmethod
    def _outcome(
        result: TraceGCResult,
        *,
        admitted: bool,
        completed: bool,
        reason_code: str,
        attempt_id: str,
        after: tuple[int, int, int],
    ) -> TraceCompactionOutcome:
        return TraceCompactionOutcome(
            admitted,
            completed,
            reason_code,
            attempt_id,
            result.allocated_bytes_after,
            after[0],
            result.freelist_bytes_after,
            after[1],
            result.wal_bytes,
            after[2],
            result.logical_live_bytes,
        )


class LegacyTraceMaintenance:
    """Normalize legacy rows only while provider and DB maintenance are idle."""

    def __init__(
        self,
        db: CharactersRAGDB,
        *,
        normalizer: LegacyTraceNormalizer | None = None,
        provider_active: Callable[[], bool] | None = None,
        max_rows: int = MAX_LEGACY_BATCH_ROWS,
        max_bytes: int = MAX_LEGACY_BATCH_BYTES,
        max_seconds: float = MAX_LEGACY_BATCH_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if type(max_rows) is not int or not 1 <= max_rows <= MAX_LEGACY_BATCH_ROWS:
            raise ValueError("max_rows")
        if type(max_bytes) is not int or not 1 <= max_bytes <= MAX_LEGACY_BATCH_BYTES:
            raise ValueError("max_bytes")
        if (
            type(max_seconds) not in {int, float}
            or not 0 < float(max_seconds) <= MAX_LEGACY_BATCH_SECONDS
        ):
            raise ValueError("max_seconds")
        self.db = db
        self.normalizer = normalizer or LegacyTraceNormalizer(db)
        self.provider_active = provider_active or (lambda: False)
        self.max_rows = max_rows
        self.max_bytes = max_bytes
        self.max_seconds = float(max_seconds)
        self.clock = clock

    def run_batch(self) -> LegacyMaintenanceBatch:
        """Run at most one bounded transaction and yield to the caller."""

        if self.provider_active():
            return LegacyMaintenanceBatch(False, 0, 0, False)
        started = self.clock()
        processed_rows = 0
        processed_bytes = 0
        with self.db.transaction(immediate=True) as cursor:
            lease = cursor.execute(
                "SELECT state FROM console_trace_maintenance_state WHERE singleton_id = 1"
            ).fetchone()
            if lease is None or lease[0] != "idle":
                return LegacyMaintenanceBatch(False, 0, 0, False)
            state = cursor.execute(
                """SELECT status, last_exchange_id, processed_rows, processed_bytes
                     FROM console_trace_migration_state
                    WHERE migration_name = ?""",
                (LEGACY_MIGRATION_NAME,),
            ).fetchone()
            if state is None:
                raise RuntimeError("legacy_migration_state_unavailable")
            last_exchange_id = -1 if state[1] is None else int(state[1])
            if state[0] == "logical_complete":
                pending = cursor.execute(
                    "SELECT 1 FROM message_exchanges WHERE id > ? LIMIT 1",
                    (last_exchange_id,),
                ).fetchone()
                if pending is None:
                    return LegacyMaintenanceBatch(True, 0, 0, True)
            cursor.execute(
                """UPDATE console_trace_migration_state
                      SET status = 'running', updated_at = CURRENT_TIMESTAMP
                    WHERE migration_name = ?""",
                (LEGACY_MIGRATION_NAME,),
            )
            rows = cursor.execute(
                """SELECT exchange.id, exchange.message_id, exchange.run_tag,
                          exchange.seq, exchange.status, exchange.abandoned,
                          exchange.capture_detail, exchange.capture_blob,
                          exchange.created_at
                     FROM message_exchanges AS exchange
                    WHERE exchange.id > ?
                    ORDER BY exchange.id
                    LIMIT ?""",
                (last_exchange_id, self.max_rows),
            ).fetchall()
            begin_batch = getattr(self.normalizer, "begin_batch", None)
            clear_matches = getattr(self.normalizer, "clear_ephemeral_matches", None)
            if callable(begin_batch):
                begin_batch(cursor)
            try:
                for raw in rows:
                    if self.clock() - started >= self.max_seconds:
                        break
                    remaining_bytes = self.max_bytes - processed_bytes
                    if remaining_bytes <= 0:
                        break
                    row = {
                        "id": int(raw[0]),
                        "message_id": str(raw[1]),
                        "run_tag": raw[2],
                        "seq": raw[3],
                        "status": raw[4],
                        "abandoned": bool(raw[5]),
                        "capture_detail": raw[6],
                        "capture_blob": bytes(raw[7]),
                        "created_at": raw[8],
                    }
                    try:
                        normalized = self.normalizer.normalize_exchange(
                            cursor,
                            row,
                            max_decoded_bytes=remaining_bytes,
                            oversized_policy=("omit" if not processed_rows else "defer"),
                        )
                    except LegacyDecodedByteLimitError:
                        break
                    if normalized.verification_status != "verified":
                        raise RuntimeError("legacy_equivalence_unverified")
                    if normalized.decoded_bytes > remaining_bytes:
                        raise RuntimeError("legacy_batch_decoded_byte_limit")
                    cursor.execute(
                        "DELETE FROM message_exchanges WHERE id = ?", (raw[0],)
                    )
                    if cursor.rowcount != 1:
                        raise RuntimeError("legacy_exchange_delete_conflict")
                    processed_rows += 1
                    processed_bytes += normalized.decoded_bytes
                    last_exchange_id = int(raw[0])
            finally:
                if callable(clear_matches):
                    clear_matches()

            remaining = cursor.execute(
                "SELECT 1 FROM message_exchanges WHERE id > ? LIMIT 1",
                (last_exchange_id,),
            ).fetchone()
            logical_complete = remaining is None
            cursor.execute(
                """UPDATE console_trace_migration_state
                      SET status = ?, last_exchange_id = ?,
                          processed_rows = processed_rows + ?,
                          processed_bytes = processed_bytes + ?,
                          updated_at = CURRENT_TIMESTAMP
                    WHERE migration_name = ?""",
                (
                    "logical_complete" if logical_complete else "running",
                    None if last_exchange_id < 0 else last_exchange_id,
                    processed_rows,
                    processed_bytes,
                    LEGACY_MIGRATION_NAME,
                ),
            )
        return LegacyMaintenanceBatch(
            True,
            processed_rows,
            processed_bytes,
            logical_complete,
        )


class TraceGarbageCollector:
    """Mark and sweep the semantic trace graph under one epoch-checked lease."""

    def __init__(self, db: CharactersRAGDB) -> None:
        self.db = db

    def current_graph_epoch(self) -> int:
        """Return the current durable graph epoch for automatic-run deduplication.

        Returns:
            The monotonically increasing trace-graph mutation epoch.
        """

        with self.db.transaction() as cursor:
            return self._graph_epoch(cursor)

    def mark(self, *, request_id: str) -> TraceGCMark:
        """Persist one schema-derived reachability snapshot for later sweep.

        Args:
            request_id: Stable idempotency identity for this collection run.

        Returns:
            The durable mark identity, epoch, and marked-entity count.

        Raises:
            ValueError: If ``request_id`` is invalid.
            RuntimeError: If maintenance state or the collector lease is unavailable.
        """

        self._validate_request_id(request_id)
        with self.db.transaction(immediate=True) as cursor:
            completed = self._completed_result(cursor, request_id)
            if completed is not None:
                return TraceGCMark(
                    request_id,
                    completed.marked_epoch,
                    self._mark_count(cursor, request_id),
                )
            cursor.execute(
                """INSERT OR IGNORE INTO console_trace_gc_runs(request_id, status)
                     VALUES (?, 'pending')""",
                (request_id,),
            )
            maintenance = cursor.execute(
                """SELECT state, lease_id, marked_epoch, lease_owner,
                          lease_expires_at
                     FROM console_trace_maintenance_state WHERE singleton_id = 1"""
            ).fetchone()
            if maintenance is None:
                raise RuntimeError("trace_gc_maintenance_state_unavailable")
            if maintenance[0] != "idle" and maintenance[1] != request_id:
                if not self._recover_expired_maintenance_lease(cursor, maintenance):
                    raise RuntimeError("trace_gc_maintenance_busy")
                maintenance = cursor.execute(
                    """SELECT state, lease_id, marked_epoch, lease_owner,
                              lease_expires_at
                         FROM console_trace_maintenance_state
                        WHERE singleton_id = 1"""
                ).fetchone()
                if maintenance is None:
                    raise RuntimeError("trace_gc_maintenance_state_unavailable")
            current_epoch = self._graph_epoch(cursor)
            run = cursor.execute(
                "SELECT status, marked_epoch FROM console_trace_gc_runs WHERE request_id = ?",
                (request_id,),
            ).fetchone()
            if (
                maintenance[0] == "marking"
                and maintenance[1] == request_id
                and maintenance[2] == current_epoch
                and run is not None
                and run[0] == "marked"
                and run[1] == current_epoch
                and self._lease_is_current(cursor, maintenance[4])
            ):
                return TraceGCMark(
                    request_id, current_epoch, self._mark_count(cursor, request_id)
                )
            cursor.execute(
                "DELETE FROM console_trace_gc_segment_scopes WHERE request_id = ?",
                (request_id,),
            )
            cursor.execute(
                "DELETE FROM console_trace_gc_marks WHERE request_id = ?",
                (request_id,),
            )
            cursor.execute(
                """UPDATE console_trace_maintenance_state
                      SET state = 'marking', lease_id = ?, lease_owner = 'trace_gc',
                          lease_expires_at = datetime('now', '+5 minutes'),
                          marked_epoch = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE singleton_id = 1""",
                (request_id, current_epoch),
            )
            migration_pending = cursor.execute(
                """SELECT 1 FROM console_trace_migration_state
                    WHERE status <> 'logical_complete' LIMIT 1"""
            ).fetchone()
            if migration_pending is not None:
                self._mark_every_trace_row(cursor, request_id, current_epoch)
            else:
                self._mark_reachable_rows(cursor, request_id, current_epoch)
            cursor.execute(
                """UPDATE console_trace_gc_runs
                      SET status = 'marked', marked_epoch = ?, swept_epoch = NULL,
                          result_json = NULL, updated_at = CURRENT_TIMESTAMP
                    WHERE request_id = ?""",
                (current_epoch, request_id),
            )
            return TraceGCMark(
                request_id, current_epoch, self._mark_count(cursor, request_id)
            )

    def sweep(self, *, request_id: str) -> TraceGCResult:
        """Sweep only the exact durable mark whose global epoch is unchanged.

        Args:
            request_id: Identity of the previously persisted mark.

        Returns:
            A content-free logical and physical reclamation result.

        Raises:
            ValueError: If ``request_id`` is invalid.
            RuntimeError: If the exact current mark and lease are unavailable.
        """

        self._validate_request_id(request_id)
        with self.db.transaction(immediate=True) as cursor:
            completed = self._completed_result(cursor, request_id)
            if completed is not None:
                return completed
            run = cursor.execute(
                """SELECT status, marked_epoch FROM console_trace_gc_runs
                    WHERE request_id = ?""",
                (request_id,),
            ).fetchone()
            maintenance = cursor.execute(
                """SELECT state, lease_id, marked_epoch, lease_expires_at
                     FROM console_trace_maintenance_state WHERE singleton_id = 1"""
            ).fetchone()
            if (
                run is None
                or run[0] != "marked"
                or type(run[1]) is not int
                or maintenance is None
                or maintenance[0] != "marking"
                or maintenance[1] != request_id
                or maintenance[2] != run[1]
                or not self._lease_is_current(cursor, maintenance[3])
            ):
                raise RuntimeError("trace_gc_mark_unavailable")
            marked_epoch = int(run[1])
            current_epoch = self._graph_epoch(cursor)
            if current_epoch != marked_epoch:
                result = self._empty_result(
                    cursor=cursor,
                    request_id=request_id,
                    status="stale_epoch",
                    marked_epoch=marked_epoch,
                )
                self._store_result(cursor, result)
                self._clear_lease(cursor, request_id)
                return result
            cursor.execute(
                """UPDATE console_trace_maintenance_state
                      SET state = 'sweeping', updated_at = CURRENT_TIMESTAMP
                    WHERE singleton_id = 1 AND state = 'marking'
                      AND lease_id = ? AND marked_epoch = ?""",
                (request_id, marked_epoch),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("trace_gc_lease_changed")
            pages_before = int(cursor.execute("PRAGMA page_count").fetchone()[0])
            free_before = int(cursor.execute("PRAGMA freelist_count").fetchone()[0])
            page_size = int(cursor.execute("PRAGMA page_size").fetchone()[0])
            reclaimed_bytes = int(
                cursor.execute(
                    """SELECT COALESCE(SUM(artifact.byte_length), 0)
                         FROM console_trace_artifacts AS artifact
                        WHERE NOT EXISTS (
                          SELECT 1 FROM console_trace_gc_marks AS mark
                           WHERE mark.request_id = ?
                             AND mark.entity_kind = 'console_trace_artifacts'
                             AND mark.entity_id = artifact.artifact_id
                        )""",
                    (request_id,),
                ).fetchone()[0]
            )
            remaining_owner_conversation_ids = self._remaining_owner_conversations(
                cursor, request_id
            )
            authorization = self.db._trace_gc_deletion_authorization_for_collector(
                cursor.connection
            )
            with authorization._authorize_trace_gc_deletion(
                cursor,
                lease_id=request_id,
                marked_epoch=marked_epoch,
            ):
                cursor.execute(
                    "DELETE FROM console_trace_gc_segment_scopes WHERE request_id = ?",
                    (request_id,),
                )
                cursor.execute(
                    """DELETE FROM console_trace_retention_roots
                        WHERE julianday(retain_until) <= julianday('now')"""
                )
                deleted_rows = self._sweep_unmarked(cursor, request_id)
            pages_after = int(cursor.execute("PRAGMA page_count").fetchone()[0])
            free_after = int(cursor.execute("PRAGMA freelist_count").fetchone()[0])
            logical_live_bytes = int(
                cursor.execute(
                    "SELECT COALESCE(SUM(byte_length), 0) FROM console_trace_artifacts"
                ).fetchone()[0]
            )
            logical_rows = sum(deleted_rows.values())
            result = TraceGCResult(
                request_id=request_id,
                status="completed",
                marked_epoch=marked_epoch,
                swept_epoch=marked_epoch,
                deleted_rows=MappingProxyType(dict(sorted(deleted_rows.items()))),
                logical_rows=logical_rows,
                logical_bytes=reclaimed_bytes,
                logical_live_bytes=logical_live_bytes,
                reclaimed_bytes=reclaimed_bytes,
                reclaimed_pages=max(0, free_after - free_before),
                page_size_bytes=page_size,
                freelist_pages_before=free_before,
                freelist_pages_after=free_after,
                freelist_bytes_before=free_before * page_size,
                freelist_bytes_after=free_after * page_size,
                allocated_pages_before=pages_before,
                allocated_pages_after=pages_after,
                allocated_bytes_before=pages_before * page_size,
                allocated_bytes_after=pages_after * page_size,
                wal_bytes=self._wal_bytes(),
                remaining_owner_conversation_ids=remaining_owner_conversation_ids,
            )
            self._store_result(cursor, result)
            self._clear_lease(cursor, request_id)
            return result

    def collect(self, *, request_id: str) -> TraceGCResult:
        """Mark then sweep, resuming a completed request without double work.

        Args:
            request_id: Stable idempotency identity for this collection run.

        Returns:
            The stored or newly completed reclamation result.

        Raises:
            ValueError: If ``request_id`` is invalid.
            RuntimeError: If maintenance state or the collector lease is unavailable.
        """

        self._validate_request_id(request_id)
        with self.db.transaction() as cursor:
            completed = self._completed_result(cursor, request_id)
        if completed is not None:
            return completed
        self.mark(request_id=request_id)
        return self.sweep(request_id=request_id)

    def retain_conversation(
        self,
        *,
        conversation_id: str,
        retain_until: str,
        reason_code: str,
    ) -> str:
        """Retain one conversation owner until an explicit UTC deadline.

        Args:
            conversation_id: Attached conversation whose trace graph is retained.
            retain_until: Timezone-aware UTC ISO 8601 retention deadline.
            reason_code: Stable non-sensitive reason for the retention root.

        Returns:
            The new or existing retention-root identity.

        Raises:
            ValueError: If an argument is invalid or conflicts with an existing root.
            KeyError: If the conversation has no attached trace owner.
        """

        if type(conversation_id) is not str or not conversation_id:
            raise ValueError("conversation_id")
        with self.db.transaction(immediate=True) as cursor:
            row = cursor.execute(
                """SELECT owner_id FROM console_trace_owners
                    WHERE conversation_id = ? AND attached = 1""",
                (conversation_id,),
            ).fetchone()
            if row is None:
                raise KeyError("trace_owner")
            return self._ensure_retention_root(
                cursor,
                entity_kind="owner",
                entity_id=str(row[0]),
                retain_until=retain_until,
                reason_code=reason_code,
            )

    def purge_conversation(
        self,
        *,
        conversation_id: str,
        request_id: str,
        detached_at: str,
    ) -> TraceGCResult:
        """Detach one owner and idempotently reclaim only its unreachable graph.

        Args:
            conversation_id: Attached conversation whose trace ownership is purged.
            request_id: Stable idempotency identity for the purge and collection.
            detached_at: Timezone-aware UTC ISO 8601 detachment timestamp.

        Returns:
            The stored or newly completed reclamation result.

        Raises:
            ValueError: If an argument is invalid or the request identity was reused
                for another purge scope.
            KeyError: If the conversation has no attached trace owner.
            RuntimeError: If maintenance state or the collector lease is unavailable.
        """

        self._validate_request_id(request_id)
        if type(conversation_id) is not str or not conversation_id:
            raise ValueError("conversation_id")
        self._validate_utc_timestamp(detached_at, "detached_at")
        with self.db.transaction(immediate=True) as cursor:
            run = cursor.execute(
                """SELECT operation_kind, target_conversation_id
                     FROM console_trace_gc_runs WHERE request_id = ?""",
                (request_id,),
            ).fetchone()
            if run is not None:
                if tuple(run) != ("purge_conversation", conversation_id):
                    raise ValueError("trace_gc_request_scope_conflict")
            else:
                row = cursor.execute(
                    """SELECT owner_id, root_segment_id
                         FROM console_trace_owners
                        WHERE conversation_id = ? AND attached = 1""",
                    (conversation_id,),
                ).fetchone()
                if row is None:
                    raise KeyError("trace_owner")
                cursor.execute(
                    """INSERT INTO console_trace_gc_runs(
                           request_id, status, operation_kind,
                           target_conversation_id, target_owner_id,
                           target_root_segment_id)
                         VALUES (?, 'pending', 'purge_conversation', ?, ?, ?)""",
                    (request_id, conversation_id, str(row[0]), str(row[1])),
                )
                ConsoleTraceRepository().detach_owner(
                    cursor,
                    owner_id=str(row[0]),
                    detached_at=detached_at,
                )
        return self.collect(request_id=request_id)

    @staticmethod
    def _ensure_retention_root(
        cursor: sqlite3.Cursor,
        *,
        entity_kind: str,
        entity_id: str,
        retain_until: str,
        reason_code: str,
    ) -> str:
        TraceGarbageCollector._validate_utc_timestamp(
            retain_until, "retain_until"
        )
        if type(reason_code) is not str or not 1 <= len(reason_code) <= 128:
            raise ValueError("reason_code")
        existing = cursor.execute(
            """SELECT retention_id, retain_until
                 FROM console_trace_retention_roots
                WHERE entity_kind = ? AND entity_id = ? AND reason_code = ?""",
            (entity_kind, entity_id, reason_code),
        ).fetchone()
        if existing is not None:
            if existing[1] != retain_until:
                raise ValueError("retention_root_conflict")
            return str(existing[0])
        retention_id = new_opaque_id()
        cursor.execute(
            """INSERT INTO console_trace_retention_roots(
                   retention_id, entity_kind, entity_id, retain_until, reason_code)
                 VALUES (?, ?, ?, ?, ?)""",
            (retention_id, entity_kind, entity_id, retain_until, reason_code),
        )
        return retention_id

    @staticmethod
    def _validate_request_id(request_id: str) -> None:
        if type(request_id) is not str or not 1 <= len(request_id) <= 128:
            raise ValueError("request_id")

    @staticmethod
    def _validate_utc_timestamp(value: str, field_name: str) -> None:
        """Require one parseable ISO 8601 timestamp with an explicit UTC offset."""

        if type(value) is not str or not value:
            raise ValueError(field_name)
        try:
            timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(field_name) from exc
        if timestamp.tzinfo is None or timestamp.utcoffset() != timedelta(0):
            raise ValueError(field_name)

    @staticmethod
    def _lease_is_current(cursor: sqlite3.Cursor, lease_expires_at: object) -> bool:
        """Return whether a parseable lease deadline remains in the future."""

        if type(lease_expires_at) is not str:
            return False
        return bool(
            cursor.execute(
                "SELECT julianday(?) > julianday('now')", (lease_expires_at,)
            ).fetchone()[0]
        )

    @classmethod
    def _recover_expired_maintenance_lease(
        cls,
        cursor: sqlite3.Cursor,
        maintenance: sqlite3.Row,
    ) -> bool:
        """Release an expired GC or compaction lease without touching a live owner."""

        state, lease_id, _epoch, lease_owner, lease_expires_at = tuple(maintenance)
        if (
            state == "idle"
            or type(lease_id) is not str
            or cls._lease_is_current(cursor, lease_expires_at)
        ):
            return False
        if lease_owner == "trace_compaction":
            cursor.execute(
                "UPDATE console_trace_compaction_state SET status = 'pending', "
                "reason_code = 'interrupted', retry_count = MIN(32, retry_count + 1), "
                "next_retry_at = NULL, progress_basis_points = 0, "
                "updated_at = CURRENT_TIMESTAMP WHERE singleton_id = 1 "
                "AND status = 'running' AND attempt_id = ?",
                (lease_id,),
            )
            cursor.execute(
                "UPDATE console_trace_maintenance_state SET state = 'idle', "
                "lease_id = NULL, lease_owner = NULL, lease_expires_at = NULL, "
                "marked_epoch = NULL, updated_at = CURRENT_TIMESTAMP "
                "WHERE singleton_id = 1 AND state = 'compacting' AND lease_id = ? "
                "AND lease_owner = 'trace_compaction'",
                (lease_id,),
            )
            return cursor.rowcount == 1
        if lease_owner != "trace_gc":
            return False
        cursor.execute(
            "DELETE FROM console_trace_gc_segment_scopes WHERE request_id = ?",
            (lease_id,),
        )
        cursor.execute(
            "DELETE FROM console_trace_gc_marks WHERE request_id = ?",
            (lease_id,),
        )
        cursor.execute(
            """UPDATE console_trace_gc_runs
                  SET status = 'pending', marked_epoch = NULL, swept_epoch = NULL,
                      result_json = NULL, updated_at = CURRENT_TIMESTAMP
                WHERE request_id = ? AND status <> 'completed'""",
            (lease_id,),
        )
        cursor.execute(
            """UPDATE console_trace_maintenance_state
                  SET state = 'idle', lease_id = NULL, lease_owner = NULL,
                      lease_expires_at = NULL, marked_epoch = NULL,
                      updated_at = CURRENT_TIMESTAMP
                WHERE singleton_id = 1 AND lease_id = ? AND lease_owner = 'trace_gc'""",
            (lease_id,),
        )
        return cursor.rowcount == 1

    @staticmethod
    def _graph_epoch(cursor: sqlite3.Cursor) -> int:
        row = cursor.execute(
            "SELECT epoch FROM console_trace_graph_epoch WHERE singleton_id = 1"
        ).fetchone()
        if row is None or type(row[0]) is not int:
            raise RuntimeError("graph_epoch_unavailable")
        return int(row[0])

    @staticmethod
    def _mark_count(cursor: object, request_id: str) -> int:
        return int(
            cursor.execute(
                "SELECT COUNT(*) FROM console_trace_gc_marks WHERE request_id = ?",
                (request_id,),
            ).fetchone()[0]
        )

    @staticmethod
    def _mark_sql(
        cursor: object,
        request_id: str,
        epoch: int,
        entity_kind: str,
        select_sql: str,
        params: tuple[object, ...] = (),
    ) -> None:
        cursor.execute(
            """INSERT OR IGNORE INTO console_trace_gc_marks(
                   request_id, entity_kind, entity_id, marked_epoch)
                 SELECT ?, ?, entity_id, ? FROM ("""
            + select_sql
            + ") WHERE entity_id IS NOT NULL",
            (request_id, entity_kind, epoch, *params),
        )

    def _mark_every_trace_row(
        self, cursor: object, request_id: str, epoch: int
    ) -> None:
        identities = {
            "console_trace_artifacts": "artifact_id",
            "console_trace_calls": "call_id",
            "console_trace_events": "event_id",
            "console_trace_header_components": (
                "header_id || char(31) || component_kind || char(31) || ordinal"
            ),
            "console_trace_owners": "owner_id",
            "console_trace_policies": "policy_id",
            "console_trace_redaction_spans": "span_id",
            "console_trace_request_headers": "header_id",
            "console_trace_response_links": "response_link_id",
            "console_trace_revision_bindings": (
                "revision_id || char(31) || policy_id"
            ),
            "console_trace_segments": "segment_id",
            "console_trace_semantic_revisions": "revision_id",
            "console_trace_surface_nodes": "node_id",
            "console_trace_surface_replacements": "replacement_id",
        }
        for table, identity in identities.items():
            self._mark_sql(
                cursor,
                request_id,
                epoch,
                table,
                f"SELECT {identity} AS entity_id FROM {table}",
            )

    def _mark_reachable_rows(
        self, cursor: object, request_id: str, epoch: int
    ) -> None:
        cursor.execute(
            """WITH RECURSIVE roots(segment_id, through_sequence) AS (
                   SELECT owner.root_segment_id, NULL
                     FROM console_trace_owners AS owner
                    WHERE owner.attached = 1
                   UNION ALL
                   SELECT owner.root_segment_id, NULL
                     FROM console_trace_retention_roots AS retention
                     JOIN console_trace_owners AS owner
                       ON retention.entity_kind = 'owner'
                      AND retention.entity_id = owner.owner_id
                    WHERE julianday(retention.retain_until) > julianday('now')
                   UNION ALL
                   SELECT call.segment_id, NULL
                     FROM console_trace_calls AS call
                    WHERE call.state IN ('reserved', 'dispatch_started', 'response_started')
                   UNION ALL
                   SELECT call.segment_id, NULL
                     FROM console_trace_retention_roots AS retention
                     JOIN console_trace_calls AS call
                       ON retention.entity_kind = 'call'
                      AND retention.entity_id = call.call_id
                    WHERE julianday(retention.retain_until) > julianday('now')
                 ), lineage(segment_id, through_sequence) AS (
                   SELECT segment_id, through_sequence FROM roots
                   UNION ALL
                   SELECT segment.parent_segment_id,
                          segment.inherited_through_sequence
                     FROM lineage AS child
                     JOIN console_trace_segments AS segment
                       ON segment.segment_id = child.segment_id
                    WHERE segment.parent_segment_id IS NOT NULL
                 )
                 INSERT INTO console_trace_gc_segment_scopes(
                   request_id, segment_id, through_sequence)
                 SELECT ?, segment_id,
                        CASE WHEN SUM(through_sequence IS NULL) > 0 THEN NULL
                             ELSE MAX(through_sequence) END
                   FROM lineage GROUP BY segment_id""",
            (request_id,),
        )
        self._mark_sql(
            cursor,
            request_id,
            epoch,
            "console_trace_segments",
            "SELECT segment_id AS entity_id FROM console_trace_gc_segment_scopes "
            "WHERE request_id = ?",
            (request_id,),
        )
        self._mark_sql(
            cursor,
            request_id,
            epoch,
            "console_trace_events",
            """SELECT event.event_id AS entity_id
                 FROM console_trace_events AS event
                 JOIN console_trace_gc_segment_scopes AS scope
                   ON scope.segment_id = event.segment_id
                WHERE scope.request_id = ?
                  AND (scope.through_sequence IS NULL
                       OR event.sequence <= scope.through_sequence)""",
            (request_id,),
        )
        self._mark_sql(
            cursor,
            request_id,
            epoch,
            "console_trace_calls",
            """SELECT call.call_id AS entity_id
                 FROM console_trace_calls AS call
                 JOIN console_trace_gc_segment_scopes AS scope
                   ON scope.segment_id = call.segment_id
                WHERE scope.request_id = ? AND (
                    scope.through_sequence IS NULL
                    OR EXISTS (
                        SELECT 1 FROM console_trace_events AS call_event
                         WHERE call_event.call_id = call.call_id
                           AND call_event.sequence <= scope.through_sequence
                    )
                    OR (
                        NOT EXISTS (
                            SELECT 1 FROM console_trace_events AS call_event
                             WHERE call_event.call_id = call.call_id
                        )
                        AND EXISTS (
                            SELECT 1 FROM console_trace_events AS turn_event
                             WHERE turn_event.segment_id = call.segment_id
                               AND turn_event.turn_id = call.turn_id
                               AND turn_event.sequence <= scope.through_sequence
                        )
                    )
                )
                UNION SELECT event.call_id FROM console_trace_events AS event
                 JOIN console_trace_gc_marks AS mark ON mark.entity_id = event.event_id
                WHERE mark.request_id = ? AND mark.entity_kind = 'console_trace_events'
                UNION SELECT entity_id FROM console_trace_retention_roots
                 WHERE entity_kind = 'call'
                   AND julianday(retain_until) > julianday('now')""",
            (request_id, request_id),
        )
        self._mark_sql(
            cursor,
            request_id,
            epoch,
            "console_trace_owners",
            """SELECT owner.owner_id AS entity_id
                 FROM console_trace_owners AS owner
                 JOIN console_trace_gc_segment_scopes AS scope
                   ON scope.segment_id = owner.root_segment_id
                WHERE scope.request_id = ?
                UNION SELECT call.owner_id FROM console_trace_calls AS call
                 JOIN console_trace_gc_marks AS mark ON mark.entity_id = call.call_id
                WHERE mark.request_id = ? AND mark.entity_kind = 'console_trace_calls'""",
            (request_id, request_id),
        )
        self._mark_sql(
            cursor,
            request_id,
            epoch,
            "console_trace_surface_replacements",
            """SELECT event.surface_replacement_id AS entity_id
                 FROM console_trace_events AS event
                 JOIN console_trace_gc_marks AS mark ON mark.entity_id = event.event_id
                WHERE mark.request_id = ? AND mark.entity_kind = 'console_trace_events'""",
            (request_id,),
        )
        self._mark_surface_nodes(cursor, request_id, epoch)
        self._mark_sql(
            cursor,
            request_id,
            epoch,
            "console_trace_request_headers",
            """SELECT call.request_header_id AS entity_id
                 FROM console_trace_calls AS call
                 JOIN console_trace_gc_marks AS mark ON mark.entity_id = call.call_id
                WHERE mark.request_id = ? AND mark.entity_kind = 'console_trace_calls'
                UNION SELECT event.request_header_id FROM console_trace_events AS event
                 JOIN console_trace_gc_marks AS mark ON mark.entity_id = event.event_id
                WHERE mark.request_id = ? AND mark.entity_kind = 'console_trace_events'""",
            (request_id, request_id),
        )
        self._mark_sql(
            cursor,
            request_id,
            epoch,
            "console_trace_header_components",
            """SELECT component.header_id || char(31) || component.component_kind
                       || char(31) || component.ordinal AS entity_id
                 FROM console_trace_header_components AS component
                 JOIN console_trace_gc_marks AS mark ON mark.entity_id = component.header_id
                WHERE mark.request_id = ?
                  AND mark.entity_kind = 'console_trace_request_headers'""",
            (request_id,),
        )
        self._mark_sql(
            cursor,
            request_id,
            epoch,
            "console_trace_response_links",
            """SELECT link.response_link_id AS entity_id
                 FROM console_trace_response_links AS link
                 JOIN console_trace_gc_marks AS mark ON mark.entity_id = link.call_id
                WHERE mark.request_id = ? AND mark.entity_kind = 'console_trace_calls'""",
            (request_id,),
        )
        self._mark_revisions_policies_artifacts(cursor, request_id, epoch)

    def _mark_surface_nodes(
        self, cursor: object, request_id: str, epoch: int
    ) -> None:
        self._mark_sql(
            cursor,
            request_id,
            epoch,
            "console_trace_surface_nodes",
            """WITH RECURSIVE seeds(node_id) AS (
                   SELECT event.surface_node_id FROM console_trace_events AS event
                    JOIN console_trace_gc_marks AS mark ON mark.entity_id = event.event_id
                   WHERE mark.request_id = ? AND mark.entity_kind = 'console_trace_events'
                   UNION SELECT segment.inherited_surface_head_id
                     FROM console_trace_segments AS segment
                     JOIN console_trace_gc_marks AS mark ON mark.entity_id = segment.segment_id
                    WHERE mark.request_id = ? AND mark.entity_kind = 'console_trace_segments'
                   UNION SELECT call.surface_node_id FROM console_trace_calls AS call
                     JOIN console_trace_gc_marks AS mark ON mark.entity_id = call.call_id
                    WHERE mark.request_id = ? AND mark.entity_kind = 'console_trace_calls'
                   UNION SELECT replacement.predecessor_head_id
                     FROM console_trace_surface_replacements AS replacement
                     JOIN console_trace_gc_marks AS mark ON mark.entity_id = replacement.replacement_id
                    WHERE mark.request_id = ? AND mark.entity_kind = 'console_trace_surface_replacements'
                   UNION SELECT replacement.start_node_id
                     FROM console_trace_surface_replacements AS replacement
                     JOIN console_trace_gc_marks AS mark ON mark.entity_id = replacement.replacement_id
                    WHERE mark.request_id = ? AND mark.entity_kind = 'console_trace_surface_replacements'
                   UNION SELECT replacement.end_node_id
                     FROM console_trace_surface_replacements AS replacement
                     JOIN console_trace_gc_marks AS mark ON mark.entity_id = replacement.replacement_id
                    WHERE mark.request_id = ? AND mark.entity_kind = 'console_trace_surface_replacements'
                   UNION SELECT replacement.replacement_node_id
                     FROM console_trace_surface_replacements AS replacement
                     JOIN console_trace_gc_marks AS mark ON mark.entity_id = replacement.replacement_id
                    WHERE mark.request_id = ? AND mark.entity_kind = 'console_trace_surface_replacements'
                 ), ancestry(node_id) AS (
                   SELECT node_id FROM seeds WHERE node_id IS NOT NULL
                   UNION
                   SELECT node.predecessor_node_id
                     FROM console_trace_surface_nodes AS node
                     JOIN ancestry ON ancestry.node_id = node.node_id
                    WHERE node.predecessor_node_id IS NOT NULL
                 ) SELECT node_id AS entity_id FROM ancestry""",
            (request_id,) * 7,
        )

    def _mark_revisions_policies_artifacts(
        self, cursor: object, request_id: str, epoch: int
    ) -> None:
        self._mark_sql(
            cursor,
            request_id,
            epoch,
            "console_trace_policies",
            """SELECT call.policy_id AS entity_id FROM console_trace_calls AS call
                 JOIN console_trace_gc_marks AS mark ON mark.entity_id = call.call_id
                WHERE mark.request_id = ? AND mark.entity_kind = 'console_trace_calls'""",
            (request_id,),
        )
        self._mark_sql(
            cursor,
            request_id,
            epoch,
            "console_trace_semantic_revisions",
            """WITH RECURSIVE seeds(revision_id) AS (
                   SELECT node.semantic_revision_id FROM console_trace_surface_nodes AS node
                    JOIN console_trace_gc_marks AS mark ON mark.entity_id = node.node_id
                   WHERE mark.request_id = ? AND mark.entity_kind = 'console_trace_surface_nodes'
                   UNION SELECT event.semantic_revision_id FROM console_trace_events AS event
                    JOIN console_trace_gc_marks AS mark ON mark.entity_id = event.event_id
                   WHERE mark.request_id = ? AND mark.entity_kind = 'console_trace_events'
                   UNION SELECT link.semantic_revision_id FROM console_trace_response_links AS link
                    JOIN console_trace_gc_marks AS mark ON mark.entity_id = link.response_link_id
                   WHERE mark.request_id = ? AND mark.entity_kind = 'console_trace_response_links'
                   UNION SELECT entity_id FROM console_trace_retention_roots
                    WHERE entity_kind = 'revision'
                      AND julianday(retain_until) > julianday('now')
                 ), ancestry(revision_id) AS (
                   SELECT revision_id FROM seeds WHERE revision_id IS NOT NULL
                   UNION
                   SELECT revision.predecessor_revision_id
                     FROM console_trace_semantic_revisions AS revision
                     JOIN ancestry ON ancestry.revision_id = revision.revision_id
                    WHERE revision.predecessor_revision_id IS NOT NULL
                 ) SELECT revision_id AS entity_id FROM ancestry""",
            (request_id, request_id, request_id),
        )
        self._mark_sql(
            cursor,
            request_id,
            epoch,
            "console_trace_revision_bindings",
            """SELECT binding.revision_id || char(31) || binding.policy_id AS entity_id
                 FROM console_trace_revision_bindings AS binding
                 JOIN console_trace_gc_marks AS revision
                   ON revision.entity_id = binding.revision_id
                  AND revision.entity_kind = 'console_trace_semantic_revisions'
                 JOIN console_trace_gc_marks AS policy
                   ON policy.entity_id = binding.policy_id
                  AND policy.entity_kind = 'console_trace_policies'
                WHERE revision.request_id = ? AND policy.request_id = ?""",
            (request_id, request_id),
        )
        self._mark_sql(
            cursor,
            request_id,
            epoch,
            "console_trace_redaction_spans",
            """SELECT span.span_id AS entity_id FROM console_trace_redaction_spans AS span
                 JOIN console_trace_gc_marks AS policy ON policy.entity_id = span.policy_id
                WHERE policy.request_id = ? AND policy.entity_kind = 'console_trace_policies'
                  AND ((span.semantic_revision_id IS NOT NULL AND EXISTS (
                         SELECT 1 FROM console_trace_gc_marks AS revision
                          WHERE revision.request_id = ?
                            AND revision.entity_kind = 'console_trace_semantic_revisions'
                            AND revision.entity_id = span.semantic_revision_id))
                    OR (span.artifact_id IS NOT NULL AND EXISTS (
                         SELECT 1 FROM console_trace_gc_marks AS artifact
                          WHERE artifact.request_id = ?
                            AND artifact.entity_kind = 'console_trace_artifacts'
                            AND artifact.entity_id = span.artifact_id)))""",
            (request_id, request_id, request_id),
        )
        self._mark_sql(
            cursor,
            request_id,
            epoch,
            "console_trace_artifacts",
            """SELECT node.artifact_id AS entity_id FROM console_trace_surface_nodes AS node
                 JOIN console_trace_gc_marks AS mark ON mark.entity_id = node.node_id
                WHERE mark.request_id = ? AND mark.entity_kind = 'console_trace_surface_nodes'
                UNION SELECT event.artifact_id FROM console_trace_events AS event
                 JOIN console_trace_gc_marks AS mark ON mark.entity_id = event.event_id
                WHERE mark.request_id = ? AND mark.entity_kind = 'console_trace_events'
                UNION SELECT link.artifact_id FROM console_trace_response_links AS link
                 JOIN console_trace_gc_marks AS mark ON mark.entity_id = link.response_link_id
                WHERE mark.request_id = ? AND mark.entity_kind = 'console_trace_response_links'
                UNION SELECT component.artifact_id FROM console_trace_header_components AS component
                 JOIN console_trace_gc_marks AS mark
                   ON mark.entity_id = component.header_id || char(31)
                        || component.component_kind || char(31) || component.ordinal
                WHERE mark.request_id = ? AND mark.entity_kind = 'console_trace_header_components'
                UNION SELECT binding.artifact_id FROM console_trace_revision_bindings AS binding
                 JOIN console_trace_gc_marks AS mark
                   ON mark.entity_id = binding.revision_id || char(31) || binding.policy_id
                WHERE mark.request_id = ? AND mark.entity_kind = 'console_trace_revision_bindings'
                UNION SELECT entity_id FROM console_trace_retention_roots
                 WHERE entity_kind = 'artifact'
                   AND julianday(retain_until) > julianday('now')""",
            (request_id,) * 5,
        )
        # Artifact-backed redaction spans become reachable only after artifacts
        # are known, so run the span mark once more after artifact closure.
        self._mark_sql(
            cursor,
            request_id,
            epoch,
            "console_trace_redaction_spans",
            """SELECT span.span_id AS entity_id FROM console_trace_redaction_spans AS span
                 JOIN console_trace_gc_marks AS artifact ON artifact.entity_id = span.artifact_id
                 JOIN console_trace_gc_marks AS policy ON policy.entity_id = span.policy_id
                WHERE artifact.request_id = ? AND policy.request_id = ?
                  AND artifact.entity_kind = 'console_trace_artifacts'
                  AND policy.entity_kind = 'console_trace_policies'""",
            (request_id, request_id),
        )

    @staticmethod
    def _remaining_owner_conversations(
        cursor: sqlite3.Cursor,
        request_id: str,
    ) -> tuple[str, ...]:
        """Return attached fork owners that still inherit the purged root."""

        row = cursor.execute(
            """SELECT operation_kind, target_root_segment_id
                 FROM console_trace_gc_runs WHERE request_id = ?""",
            (request_id,),
        ).fetchone()
        if row is None or row[0] != "purge_conversation" or row[1] is None:
            return ()
        rows = cursor.execute(
            """WITH RECURSIVE owner_lineage(
                   conversation_id, segment_id, depth
                 ) AS (
                   SELECT conversation_id, root_segment_id, 0
                     FROM console_trace_owners
                    WHERE attached = 1 AND conversation_id IS NOT NULL
                   UNION ALL
                   SELECT lineage.conversation_id,
                          segment.parent_segment_id,
                          lineage.depth + 1
                     FROM owner_lineage AS lineage
                     JOIN console_trace_segments AS segment
                       ON segment.segment_id = lineage.segment_id
                    WHERE segment.parent_segment_id IS NOT NULL
                      AND lineage.depth < 10000
                 )
                 SELECT DISTINCT conversation_id
                   FROM owner_lineage
                  WHERE segment_id = ?
                  ORDER BY conversation_id""",
            (str(row[1]),),
        ).fetchall()
        return tuple(str(item[0]) for item in rows)

    @staticmethod
    def _sweep_unmarked(
        cursor: sqlite3.Cursor, request_id: str
    ) -> dict[str, int]:
        statements = (
            ("console_trace_redaction_spans", "span_id"),
            ("console_trace_response_links", "response_link_id"),
            ("console_trace_events", "event_id"),
            ("console_trace_surface_replacements", "replacement_id"),
            (
                "console_trace_revision_bindings",
                "revision_id || char(31) || policy_id",
            ),
            (
                "console_trace_header_components",
                "header_id || char(31) || component_kind || char(31) || ordinal",
            ),
            ("console_trace_calls", "call_id"),
            ("console_trace_surface_nodes", "node_id"),
            ("console_trace_request_headers", "header_id"),
            ("console_trace_semantic_revisions", "revision_id"),
            ("console_trace_artifacts", "artifact_id"),
            ("console_trace_policies", "policy_id"),
            ("console_trace_owners", "owner_id"),
            ("console_trace_segments", "segment_id"),
        )
        deleted: dict[str, int] = {}
        for table, identity in statements:
            cursor.execute(
                f"""DELETE FROM {table} WHERE NOT EXISTS (
                       SELECT 1 FROM console_trace_gc_marks AS mark
                        WHERE mark.request_id = ? AND mark.entity_kind = ?
                          AND mark.entity_id = {table}.{identity})""",
                (request_id, table),
            )
            deleted[table] = max(0, int(cursor.rowcount))
        return deleted

    def _completed_result(
        self, cursor: object, request_id: str
    ) -> TraceGCResult | None:
        row = cursor.execute(
            "SELECT status, result_json FROM console_trace_gc_runs WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        if row is None or row[0] != "completed" or row[1] is None:
            return None
        return self._decode_result(str(row[1]))

    @staticmethod
    def _store_result(cursor: object, result: TraceGCResult) -> None:
        payload = {
            "request_id": result.request_id,
            "status": result.status,
            "marked_epoch": result.marked_epoch,
            "swept_epoch": result.swept_epoch,
            "deleted_rows": dict(result.deleted_rows),
            "logical_rows": result.logical_rows,
            "logical_bytes": result.logical_bytes,
            "logical_live_bytes": result.logical_live_bytes,
            "reclaimed_bytes": result.reclaimed_bytes,
            "reclaimed_pages": result.reclaimed_pages,
            "page_size_bytes": result.page_size_bytes,
            "freelist_pages_before": result.freelist_pages_before,
            "freelist_pages_after": result.freelist_pages_after,
            "freelist_bytes_before": result.freelist_bytes_before,
            "freelist_bytes_after": result.freelist_bytes_after,
            "allocated_pages_before": result.allocated_pages_before,
            "allocated_pages_after": result.allocated_pages_after,
            "allocated_bytes_before": result.allocated_bytes_before,
            "allocated_bytes_after": result.allocated_bytes_after,
            "wal_bytes": result.wal_bytes,
            "remaining_owner_conversation_ids": list(
                result.remaining_owner_conversation_ids
            ),
        }
        cursor.execute(
            """UPDATE console_trace_gc_runs
                  SET status = ?, swept_epoch = ?, result_json = ?,
                      updated_at = CURRENT_TIMESTAMP
                WHERE request_id = ?""",
            (
                result.status,
                result.swept_epoch,
                json.dumps(payload, separators=(",", ":"), sort_keys=True),
                result.request_id,
            ),
        )

    @staticmethod
    def _decode_result(value: str) -> TraceGCResult:
        payload = json.loads(value)
        deleted = {str(key): int(count) for key, count in payload["deleted_rows"].items()}
        page_size = int(payload.get("page_size_bytes", 0))
        free_before = int(payload["freelist_pages_before"])
        free_after = int(payload["freelist_pages_after"])
        pages_before = int(payload["allocated_pages_before"])
        pages_after = int(payload["allocated_pages_after"])
        logical_bytes = int(payload["logical_bytes"])
        return TraceGCResult(
            request_id=str(payload["request_id"]),
            status=str(payload["status"]),
            marked_epoch=int(payload["marked_epoch"]),
            swept_epoch=(
                None if payload["swept_epoch"] is None else int(payload["swept_epoch"])
            ),
            deleted_rows=MappingProxyType(dict(sorted(deleted.items()))),
            logical_rows=int(payload["logical_rows"]),
            logical_bytes=logical_bytes,
            logical_live_bytes=int(payload.get("logical_live_bytes", 0)),
            reclaimed_bytes=int(payload.get("reclaimed_bytes", logical_bytes)),
            reclaimed_pages=int(
                payload.get("reclaimed_pages", max(0, free_after - free_before))
            ),
            page_size_bytes=page_size,
            freelist_pages_before=free_before,
            freelist_pages_after=free_after,
            freelist_bytes_before=int(
                payload.get("freelist_bytes_before", free_before * page_size)
            ),
            freelist_bytes_after=int(
                payload.get("freelist_bytes_after", free_after * page_size)
            ),
            allocated_pages_before=pages_before,
            allocated_pages_after=pages_after,
            allocated_bytes_before=int(
                payload.get("allocated_bytes_before", pages_before * page_size)
            ),
            allocated_bytes_after=int(
                payload.get("allocated_bytes_after", pages_after * page_size)
            ),
            wal_bytes=int(payload["wal_bytes"]),
            remaining_owner_conversation_ids=tuple(
                str(item)
                for item in payload.get("remaining_owner_conversation_ids", ())
            ),
        )

    def _empty_result(
        self,
        *,
        cursor: sqlite3.Cursor,
        request_id: str,
        status: str,
        marked_epoch: int,
    ) -> TraceGCResult:
        pages = int(cursor.execute("PRAGMA page_count").fetchone()[0])
        free = int(cursor.execute("PRAGMA freelist_count").fetchone()[0])
        page_size = int(cursor.execute("PRAGMA page_size").fetchone()[0])
        live_bytes = int(
            cursor.execute(
                "SELECT COALESCE(SUM(byte_length), 0) FROM console_trace_artifacts"
            ).fetchone()[0]
        )
        return TraceGCResult(
            request_id=request_id,
            status=status,
            marked_epoch=marked_epoch,
            swept_epoch=None,
            deleted_rows=MappingProxyType({table: 0 for table in _GC_ENTITY_TABLES}),
            logical_rows=0,
            logical_bytes=0,
            logical_live_bytes=live_bytes,
            reclaimed_bytes=0,
            reclaimed_pages=0,
            page_size_bytes=page_size,
            freelist_pages_before=free,
            freelist_pages_after=free,
            freelist_bytes_before=free * page_size,
            freelist_bytes_after=free * page_size,
            allocated_pages_before=pages,
            allocated_pages_after=pages,
            allocated_bytes_before=pages * page_size,
            allocated_bytes_after=pages * page_size,
            wal_bytes=self._wal_bytes(),
            remaining_owner_conversation_ids=self._remaining_owner_conversations(
                cursor, request_id
            ),
        )

    @staticmethod
    def _clear_lease(cursor: object, request_id: str) -> None:
        cursor.execute(
            "DELETE FROM console_trace_gc_segment_scopes WHERE request_id = ?",
            (request_id,),
        )
        cursor.execute(
            "DELETE FROM console_trace_gc_marks WHERE request_id = ?",
            (request_id,),
        )
        cursor.execute(
            """UPDATE console_trace_maintenance_state
                  SET state = 'idle', lease_id = NULL, lease_owner = NULL,
                      lease_expires_at = NULL, marked_epoch = NULL,
                      updated_at = CURRENT_TIMESTAMP
                WHERE singleton_id = 1 AND lease_id = ?""",
            (request_id,),
        )

    def _wal_bytes(self) -> int:
        if self.db.is_memory_db:
            return 0
        wal_path = Path(f"{self.db.db_path_str}-wal")
        try:
            return max(0, wal_path.stat().st_size)
        except OSError:
            return 0


__all__ = [
    "LEGACY_MIGRATION_NAME",
    "LegacyMaintenanceBatch",
    "LegacyTraceMaintenance",
    "MAX_LEGACY_BATCH_BYTES",
    "MAX_LEGACY_BATCH_ROWS",
    "MAX_LEGACY_BATCH_SECONDS",
    "PhysicalTraceCompactor",
    "TraceCompactionOutcome",
    "TraceCompactionPolicy",
    "TraceCompactionProgress",
    "TraceGarbageCollector",
    "TraceGCMark",
    "TraceGCResult",
]
