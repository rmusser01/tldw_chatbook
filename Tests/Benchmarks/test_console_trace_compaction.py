"""Pinned physical compaction performance and size-reclamation gate."""

from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import time

from tldw_chatbook.Chat.console_trace_maintenance import (
    PhysicalTraceCompactor,
    TraceCompactionPolicy,
    TraceGarbageCollector,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


_FIXTURE_DIR = Path(__file__).with_name("fixtures")
_FIXTURE = _FIXTURE_DIR / "console_trace_compaction.sqlite3"
_CHECKSUM = _FIXTURE_DIR / "console_trace_compaction.sha256"


def test_pinned_fixture_compacts_within_five_seconds_and_shrinks(
    tmp_path: Path,
) -> None:
    expected = _CHECKSUM.read_text(encoding="ascii").split()[0]
    assert hashlib.sha256(_FIXTURE.read_bytes()).hexdigest() == expected
    working = tmp_path / _FIXTURE.name
    shutil.copyfile(_FIXTURE, working)
    database = CharactersRAGDB(working, "trace-compaction-benchmark")
    try:
        gc_result = TraceGarbageCollector(database).collect(
            request_id="gc-compaction-benchmark"
        )
        assert gc_result.freelist_bytes_after >= 2 * 1024 * 1024
        before = working.stat().st_size
        policy = TraceCompactionPolicy(
            min_database_bytes=2 * 1024 * 1024,
            min_freelist_bytes=2 * 1024 * 1024,
            min_freelist_ratio=0.50,
            min_idle_seconds=0.0,
            retry_initial_seconds=1.0,
            retry_max_seconds=10.0,
            quiesce_timeout_seconds=1.0,
            disk_safety_margin_bytes=0,
        )

        started = time.perf_counter()
        outcome = PhysicalTraceCompactor(database, policy=policy).run_after_gc(
            gc_result
        )
        elapsed = time.perf_counter() - started

        assert outcome.completed is True
        assert outcome.allocated_bytes_after < outcome.allocated_bytes_before
        assert working.stat().st_size < before
        assert elapsed <= 5.0, f"trace compaction took {elapsed:.3f}s"
        assert database.get_connection().execute(
            "PRAGMA quick_check(1)"
        ).fetchone()[0] == "ok"
    finally:
        database.close_connection()
