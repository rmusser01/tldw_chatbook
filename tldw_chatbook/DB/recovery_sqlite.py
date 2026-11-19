"""Shared checks for literal, installed recovery SQLite call sites.

These helpers neither choose an authority nor open/copy a database. Each owner
retains its statically qualified private-SQLite calls and exact schema policy.
"""

import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import Event

from tldw_chatbook.Backup_Recovery.models import StorageItem

_STATISTICS = {
    ("table", name, name, f"CREATE TABLE {name}({','.join(columns)})"): (
        name,
        tuple((i, col, "", 0, None, 0, 0) for i, col in enumerate(columns)),
        (),
    )
    for name, columns in (
        ("sqlite_stat1", ("tbl", "idx", "stat")),
        ("sqlite_stat4", ("tbl", "idx", "neq", "nlt", "ndlt", "sample")),
    )
}


def _split_statistics(rows):
    """Separate only SQLite's exact engine-produced statistics table forms."""
    stats = tuple(row for row in rows if row in _STATISTICS)
    if len(stats) == 1 and stats[0][1] == "sqlite_stat4":
        raise ValueError("unsupported_schema")
    return tuple(row for row in rows if row not in _STATISTICS), stats


def _statistics_metadata(metadata, stats):
    if not stats:
        return metadata
    return tuple(
        sorted(
            (*metadata, *(_STATISTICS[row] for row in stats)),
            key=lambda item: (len(item), item[0]),
        )
    )


def _validate_sqlite(
    connection: sqlite3.Connection,
    versions: tuple[int, ...],
    schemas: tuple[tuple[int, tuple[str, ...]], ...],
    *,
    version_query: str = "PRAGMA user_version",
    allow_statistics: bool = False,
) -> tuple[str, ...]:
    from tldw_chatbook.Backup_Recovery.sqlite_validation import (
        _canvas_payload_issues,
        _canvas_schema_access,
        _catalog,
        _current_restrictions,
        _restrict_connection,
    )

    restrictions = _current_restrictions(connection)
    if any(
        "canvas_revision_payload_valid" in sql
        for _, schema in schemas
        for sql in schema
    ):
        if restrictions is None:
            restrictions = _restrict_connection(connection)
    else:
        connection.execute("PRAGMA trusted_schema=OFF")
    catalog = _catalog(connection)
    if allow_statistics:
        catalog, _stats = _split_statistics(catalog)
    actual = tuple(row[3] for row in catalog if row[3] is not None)
    matched = tuple(known for known, schema in schemas if actual == schema)
    if not matched:
        return ("unsupported_schema",)
    version = connection.execute(version_query).fetchone()[0]
    if version not in versions or version not in matched:
        return ("unsupported_schema_version",)
    with _canvas_schema_access(connection, actual, restrictions):
        if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
            return ("invalid_domain_reference",)
        payload_issues = _canvas_payload_issues(connection, restrictions)
        if payload_issues:
            return payload_issues
        if connection.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
            return ("invalid_sqlite_integrity",)
        return ()


@contextmanager
def _checked_capture(
    owner_id: str,
    item: StorageItem,
    destination: Path,
    cancel: Event,
    validate: Callable[[Path], tuple[str, ...]],
) -> Iterator[Callable[[], None]]:
    from tldw_chatbook.Backup_Recovery.admission import _local

    if item.owner != owner_id or item.path is None or item.status != "included":
        raise ValueError("invalid_capture_item")
    scope = getattr(_local, "capture_scope", None)
    if scope is None:
        raise ValueError("capture_requires_maintenance")
    scope.check()
    if destination.exists():
        raise FileExistsError("capture_destination_exists")

    def guard():
        if cancel.is_set():
            raise InterruptedError("cancelled")

    guard()
    yield guard
    guard()
    issues = validate(destination)
    if issues:
        raise ValueError(issues[0])
