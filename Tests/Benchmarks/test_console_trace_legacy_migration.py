"""Pinned historical performance gate for legacy trace normalization."""

from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import time

from tldw_chatbook.Chat.console_trace_maintenance import LegacyTraceMaintenance
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


_FIXTURE_DIR = Path(__file__).with_name("fixtures")
_FIXTURE = _FIXTURE_DIR / "console_trace_legacy_200_turn_v53.sqlite3"
_CHECKSUM = _FIXTURE_DIR / "console_trace_legacy_200_turn_v53.sha256"


def test_pinned_v53_200_turn_fixture_normalizes_within_five_seconds(
    tmp_path: Path,
) -> None:
    expected = _CHECKSUM.read_text(encoding="ascii").split()[0]
    assert hashlib.sha256(_FIXTURE.read_bytes()).hexdigest() == expected
    working = tmp_path / _FIXTURE.name
    shutil.copyfile(_FIXTURE, working)
    db = CharactersRAGDB(str(working), "legacy-trace-benchmark")
    try:
        maintenance = LegacyTraceMaintenance(db)
        started = time.perf_counter()
        batch_count = 0
        while True:
            batch = maintenance.run_batch()
            batch_count += 1
            assert batch.admitted is True
            if batch.logical_complete:
                break
            assert batch.processed_rows > 0
            assert batch_count < 1_000
        elapsed = time.perf_counter() - started
        with db.transaction() as cursor:
            assert (
                cursor.execute("SELECT COUNT(*) FROM message_exchanges").fetchone()[0]
                == 0
            )
            assert (
                cursor.execute(
                    """SELECT COUNT(*) FROM console_trace_calls
                    WHERE route_identity = 'legacy_snapshot'"""
                ).fetchone()[0]
                == 200
            )
            assert (
                cursor.execute(
                    """SELECT COUNT(*) FROM console_trace_surface_nodes
                    WHERE component_kind = 'legacy_snapshot_message'"""
                ).fetchone()[0]
                == 399
            )
        assert elapsed < 5.0, f"legacy normalization took {elapsed:.3f}s"
    finally:
        db.close_connection()
