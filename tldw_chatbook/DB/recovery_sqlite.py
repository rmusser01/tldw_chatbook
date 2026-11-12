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


_CATALOG_LIMIT = 10_000
_CATALOG_BYTES = 8 * 1024**2


def _read_bounded_catalog(
    connection, *, row_limit=_CATALOG_LIMIT, byte_limit=_CATALOG_BYTES
):
    """Read the exact ordered SQLite catalog under fixed materialization bounds."""
    rows = []
    size = 0
    for row in connection.execute(
        "SELECT type,name,tbl_name,sql FROM sqlite_schema ORDER BY type,name"
    ):
        size += sum(len(value.encode("utf-8")) for value in row if value is not None)
        if len(rows) >= row_limit or size > byte_limit:
            raise ValueError("sqlite_resource_limit")
        rows.append(tuple(row))
    return tuple(rows)


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
        _validate_sqlite as validate,
    )

    return validate(
        connection,
        versions,
        schemas,
        version_query=version_query,
        allow_statistics=allow_statistics,
    )


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
