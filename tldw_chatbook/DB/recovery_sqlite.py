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
    from tldw_chatbook.Backup_Recovery.sqlite_validation import (
        _validate_sqlite as validate,
    )

    return validate(connection, versions, schemas, version_query=version_query)


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
