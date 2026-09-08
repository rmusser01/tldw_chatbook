"""Shared checks for literal, installed recovery SQLite call sites.

These helpers neither choose an authority nor open/copy a database. Each owner
retains its statically qualified private-SQLite calls and exact schema policy.
"""

from contextlib import contextmanager
from pathlib import Path
import sqlite3
from threading import Event
from typing import Callable, Iterator

from tldw_chatbook.Backup_Recovery.models import StorageItem


def _validate_sqlite(
    connection: sqlite3.Connection,
    versions: tuple[int, ...],
    schemas: tuple[tuple[int, tuple[str, ...]], ...],
    *,
    version_query: str = "PRAGMA user_version",
) -> tuple[str, ...]:
    connection.execute("PRAGMA trusted_schema=OFF")
    version = connection.execute(version_query).fetchone()[0]
    if version not in versions:
        return ("unsupported_schema_version",)
    actual = tuple(
        row[0]
        for row in connection.execute(
            "SELECT sql FROM sqlite_schema WHERE sql IS NOT NULL ORDER BY type,name"
        )
    )
    if not any(
        known_version == version and actual == schema
        for known_version, schema in schemas
    ):
        return ("unsupported_schema",)
    if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
        return ("invalid_domain_reference",)
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
