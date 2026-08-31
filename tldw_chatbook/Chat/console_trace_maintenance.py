"""Idle, bounded maintenance for legacy Console trace normalization."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import time

from tldw_chatbook.Chat.console_trace_legacy import LegacyTraceNormalizer
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


LEGACY_MIGRATION_NAME = "legacy_exchange_normalization"
MAX_LEGACY_BATCH_ROWS = 100
MAX_LEGACY_BATCH_BYTES = 4 * 1024 * 1024
MAX_LEGACY_BATCH_SECONDS = 0.100


@dataclass(frozen=True, slots=True)
class LegacyMaintenanceBatch:
    """Content-free outcome of one bounded maintenance attempt."""

    admitted: bool
    processed_rows: int
    processed_bytes: int
    logical_complete: bool


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
                    if processed_rows and self.clock() - started >= self.max_seconds:
                        break
                    blob_bytes = len(bytes(raw[7]))
                    if processed_rows and processed_bytes + blob_bytes > self.max_bytes:
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
                    normalized = self.normalizer.normalize_exchange(cursor, row)
                    if normalized.verification_status != "verified":
                        raise RuntimeError("legacy_equivalence_unverified")
                    if (
                        processed_rows
                        and processed_bytes + normalized.decoded_bytes > self.max_bytes
                    ):
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


__all__ = [
    "LEGACY_MIGRATION_NAME",
    "LegacyMaintenanceBatch",
    "LegacyTraceMaintenance",
    "MAX_LEGACY_BATCH_BYTES",
    "MAX_LEGACY_BATCH_ROWS",
    "MAX_LEGACY_BATCH_SECONDS",
]
