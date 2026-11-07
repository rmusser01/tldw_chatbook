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


def test_schema_alias_preserves_the_standalone_loader_and_live_old_patch(
    database, monkeypatch
):
    from importlib.util import find_spec

    name = "tldw_chatbook.Media.local_media_reading_schema"
    assert find_spec(name) is not None, "local schema data leaf is missing"
    from tldw_chatbook.Media import local_media_reading_schema as schema

    assert _MODULE._LOCAL_READING_AUX_SCHEMA is schema._LOCAL_READING_AUX_SCHEMA
    monkeypatch.setattr(
        _MODULE,
        "_LOCAL_READING_AUX_SCHEMA",
        ("CREATE TABLE old_service_patch (value TEXT)",),
    )
    LocalMediaReadingService._ensure_local_reading_aux_schema(database)
    with database.transaction() as conn:
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE name='old_service_patch'"
        ).fetchone() == ("old_service_patch",)
    assert _MODULE._LOCAL_READING_AUX_SCHEMA is not schema._LOCAL_READING_AUX_SCHEMA


@pytest.mark.parametrize("schema_first", [True, False])
def test_media_clean_import_orders_preserve_collector_shapes(schema_first):
    import subprocess
    import sys

    script = """
import importlib
import sys
from types import FunctionType
schema_name = "tldw_chatbook.Media.local_media_reading_schema"
service_name = "tldw_chatbook.Media.local_media_reading_service"
order = (schema_name, service_name) if sys.argv[1] == "1" else (service_name, schema_name)
for name in order:
    importlib.import_module(name)
schema, service = (sys.modules[schema_name], sys.modules[service_name])
assert service._LOCAL_READING_AUX_SCHEMA is schema._LOCAL_READING_AUX_SCHEMA
cls = service.LocalMediaReadingService
assert service._NATIVE_ORDINARY_LIBRARY_CLASS is cls
methods = tuple((name, value) for name, value in vars(cls).items()
                if isinstance(value, (FunctionType, staticmethod, classmethod, property)))
assert service._NATIVE_ORDINARY_LIBRARY_METHODS == methods
functions = tuple(value for value in tuple(vars(service).values())
                  if type(value) is FunctionType and value.__module__ == service_name)
functions += tuple(value.__func__ if isinstance(value, (staticmethod, classmethod)) else value
                   for _, value in methods if not isinstance(value, property))
records = service._NATIVE_ORDINARY_LIBRARY_FUNCTIONS
assert len(records) == len(functions)
for record, fn in zip(records, functions, strict=True):
    assert len(record) == 7 and record[0] is fn
    assert record[1] is fn.__code__ and record[2] is fn.__defaults__
    assert record[3] is fn.__kwdefaults__ and record[4] == tuple((fn.__kwdefaults__ or {}).items())
    assert record[5] is fn.__closure__
    assert record[6] == tuple((cell, cell.cell_contents) for cell in fn.__closure__ or ())
    assert fn.__globals__ is vars(service)
for _, prop in methods:
    if isinstance(prop, property) and prop.fget is not None:
        assert all(record[0] is not prop.fget for record in records)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, "1" if schema_first else "0"],
        capture_output=True,
        check=False,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
