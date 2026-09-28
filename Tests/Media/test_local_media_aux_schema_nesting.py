"""`_ensure_local_reading_aux_schema` must not commit its caller's transaction.

Tier-2 S12 P3. `MediaDatabase.transaction()` is nesting-aware: a nested call
joins the already-open outer transaction and leaves commit/rollback to the
outermost block (`Client_Media_DB_v2.transaction`). `Cursor.executescript`
implicitly COMMITs whatever is open before it runs, so a nested
`_ensure_local_reading_aux_schema` silently committed half of its caller's
work and defeated the rollback the outer block relies on. Thirty-two call
sites reach this bootstrap.

The harness is a real `sqlite3` connection behind the same nesting-aware
context manager the production `transaction()` implements, because the trap
is pure sqlite3 semantics and `MediaDatabase` itself is unreachable under the
ADR-126 admission gate in a clean worktree.
"""

from __future__ import annotations

from contextlib import contextmanager
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sqlite3

import pytest


_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "tldw_chatbook"
    / "Media"
    / "local_media_reading_service.py"
)
_SPEC = spec_from_file_location("local_media_aux_schema_test_module", _MODULE_PATH)
_MODULE = module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)
LocalMediaReadingService = _MODULE.LocalMediaReadingService


class _NestingAwareDatabase:
    """The nesting contract of `Client_Media_DB_v2.MediaDatabase.transaction`."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    @contextmanager
    def transaction(self, immediate: bool = False):
        conn = self._connection
        in_outer = conn.in_transaction
        try:
            if not in_outer:
                conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            yield conn
            if not in_outer:
                conn.commit()
        except BaseException:
            if not in_outer:
                conn.rollback()
            raise


@pytest.fixture()
def database():
    connection = sqlite3.connect(":memory:", isolation_level=None)
    connection.execute("CREATE TABLE caller_work (id INTEGER PRIMARY KEY)")
    try:
        yield _NestingAwareDatabase(connection)
    finally:
        connection.close()


@pytest.mark.unit
def test_nested_bootstrap_leaves_the_outer_rollback_intact(database):
    with pytest.raises(RuntimeError):
        with database.transaction() as conn:
            conn.execute("INSERT INTO caller_work (id) VALUES (1)")
            LocalMediaReadingService._ensure_local_reading_aux_schema(database)
            raise RuntimeError("the outer block must roll back")

    with database.transaction() as conn:
        survived = conn.execute("SELECT COUNT(*) FROM caller_work").fetchone()[0]
    assert survived == 0


@pytest.mark.unit
def test_bootstrap_still_creates_every_table_and_index(database):
    LocalMediaReadingService._ensure_local_reading_aux_schema(database)

    with database.transaction() as conn:
        names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE name LIKE 'local_%'"
                " OR name LIKE 'idx_local_%'"
            ).fetchall()
        }
    assert {
        "local_reading_saved_searches",
        "local_reading_note_links",
        "local_reading_archives",
        "local_reading_highlights",
        "local_document_annotations",
        "local_reading_digest_schedules",
        "local_reading_digest_outputs",
        "local_file_artifacts",
        "idx_local_reading_highlights_item_id",
        "idx_local_document_annotations_media_id",
        "idx_local_reading_digest_outputs_schedule_id",
        "idx_local_file_artifacts_type_deleted",
    } <= names, sorted(names)


@pytest.mark.unit
def test_bootstrap_is_idempotent(database):
    LocalMediaReadingService._ensure_local_reading_aux_schema(database)
    LocalMediaReadingService._ensure_local_reading_aux_schema(database)
