"""Regenerate the pinned physical Console trace-compaction fixture."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sqlite3

from tldw_chatbook.Chat.console_trace_maintenance import TraceGarbageCollector
from tldw_chatbook.Chat.console_trace_repository import ConsoleTraceRepository
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


FIXTURE_DIR = Path(__file__).with_name("fixtures")
FIXTURE_PATH = FIXTURE_DIR / "console_trace_compaction.sqlite3"
CHECKSUM_PATH = FIXTURE_DIR / "console_trace_compaction.sha256"


def generate() -> None:
    """Build and checksum a content-free database with reclaimable free pages."""

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.unlink(missing_ok=True)
    database = CharactersRAGDB(FIXTURE_PATH, "trace-compaction-fixture")
    repository = ConsoleTraceRepository()
    try:
        with database.transaction(immediate=True) as cursor:
            cursor.execute(
                "UPDATE console_trace_migration_state SET status = 'logical_complete' "
                "WHERE migration_name = 'legacy_exchange_normalization'"
            )
            for index in range(96):
                block = index.to_bytes(4, "big") + bytes(range(256)) * 128
                repository.store_sanitized_artifact(
                    cursor,
                    sanitized_bytes=block,
                    media_type="application/octet-stream",
                    normalization_version="compaction-benchmark-v1",
                )
        result = TraceGarbageCollector(database).collect(
            request_id="gc-compaction-fixture"
        )
        assert result.deleted_rows["console_trace_artifacts"] == 96
        assert result.freelist_bytes_after >= 2 * 1024 * 1024
        database.get_connection().execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        database.close_connection()

    connection = sqlite3.connect(FIXTURE_PATH)
    try:
        assert connection.execute("PRAGMA quick_check(1)").fetchone()[0] == "ok"
        assert connection.execute("PRAGMA freelist_count").fetchone()[0] > 0
    finally:
        connection.close()
    digest = hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest()
    CHECKSUM_PATH.write_text(f"{digest}  {FIXTURE_PATH.name}\n", encoding="ascii")


if __name__ == "__main__":
    generate()
