"""Private SQLite value types and source-free target/result codecs.

Native admission, connection issuance and lifetime ownership stay in private_sqlite.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PureWindowsPath
from urllib.parse import quote

from tldw_chatbook.DB.private_sqlite_process import (
    HelperProtocolError,
    HelperTimeoutError,
    HelperUnavailableError,
)
from tldw_chatbook.DB.private_sqlite_protocol import PrepareResult
from tldw_chatbook.Utils.platform_files import os
from tldw_chatbook.Utils.private_paths import (
    PrivatePathError,
    PrivatePathResult,
    PrivatePathStatus,
)


class SQLiteTargetKind(StrEnum):
    """Supported SQLite storage target classifications."""

    PRIVATE_FILE = "private_file"
    MEMORY = "memory"
    READ_ONLY_URI = "read_only_uri"


class SQLitePrivacyUnverifiedWarning(RuntimeWarning):
    """Warn that a successful SQLite file open lacks verified ACL privacy."""


class SQLiteRestoreBusyError(sqlite3.OperationalError):
    """Report that a live database cannot be restored safely."""


class SQLiteRestoreIndeterminateError(sqlite3.OperationalError):
    """Report that a committed restore could not be rolled back reliably."""

    def __init__(
        self,
        destination_path: str | os.PathLike[str],
        pre_restore_path: str | os.PathLike[str],
    ) -> None:
        self.destination_path = Path(destination_path)
        self.pre_restore_path = Path(pre_restore_path)
        super().__init__(
            "The live database may already contain restored data because "
            "automatic recovery failed. Inspect "
            f"{self.destination_path} and the pre-restore snapshot at "
            f"{self.pre_restore_path}. Do not retry automatically."
        )


@dataclass(frozen=True, slots=True)
class SQLiteOwnerPolicy:
    """Immutable storage policy for one registered production owner."""

    production_module: str
    allowed_target_kinds: frozenset[SQLiteTargetKind]
    reason: str
    centralized_backup_allowed: bool = False
    preserve_read_only_source_mode: bool = False
    foreign_read_only_source: bool = False
    recovery_capture_allowed: bool = False


_PRIVATE_FILE = frozenset({SQLiteTargetKind.PRIVATE_FILE})


_MEMORY = frozenset({SQLiteTargetKind.MEMORY})


_PRIVATE_OR_MEMORY = frozenset({SQLiteTargetKind.PRIVATE_FILE, SQLiteTargetKind.MEMORY})


_READ_ONLY_URI = frozenset({SQLiteTargetKind.READ_ONLY_URI})


_PRIVATE_AND_READ_ONLY = frozenset(
    {SQLiteTargetKind.PRIVATE_FILE, SQLiteTargetKind.READ_ONLY_URI}
)


_PRIVATE_MEMORY_AND_READ_ONLY = frozenset(
    {
        SQLiteTargetKind.PRIVATE_FILE,
        SQLiteTargetKind.MEMORY,
        SQLiteTargetKind.READ_ONLY_URI,
    }
)


class SQLitePrivateDestinationError(OSError):
    """Bounded failure for an opaque private SQLite destination."""

    def __init__(self, code: str = "destination_invalid") -> None:
        self.code = code
        super().__init__(f"SQLite private destination failed: {code}")

    def __repr__(self) -> str:
        return f"SQLitePrivateDestinationError({self.code!r})"


def _failure(
    selected: Path,
    status: PrivatePathStatus,
    reason: str,
) -> PrivatePathError:
    return PrivatePathError(PrivatePathResult(selected, status, reason=reason))


def _build_read_only_uri(
    database: str | os.PathLike[str],
    *,
    windows: bool | None = None,
    immutable: bool = False,
) -> str:
    raw = os.fspath(database)
    if not isinstance(raw, str):
        raise TypeError("SQLite paths must be text paths")
    if "\x00" in raw:
        raise ValueError("Path must not contain NUL")
    windows = os.name == "nt" if windows is None else windows
    if windows:
        path = PureWindowsPath(raw)
        posix_path = path.as_posix()
        if posix_path.startswith("//"):
            authority_and_path = posix_path[2:].split("/", 1)
            authority = authority_and_path[0]
            remainder = authority_and_path[1] if len(authority_and_path) == 2 else ""
            encoded = quote(remainder, safe="/")
            uri = f"file://{authority}/{encoded}?mode=ro"
            return f"{uri}&immutable=1" if immutable else uri
        encoded = quote(posix_path, safe="/:")
        uri = f"file:///{encoded}?mode=ro"
        return f"{uri}&immutable=1" if immutable else uri

    uri = f"{Path(raw).as_uri()}?mode=ro"
    return f"{uri}&immutable=1" if immutable else uri


def _build_existing_writable_uri(
    database: str | os.PathLike[str],
    *,
    windows: bool | None = None,
) -> str:
    """Build a seam-owned ``mode=rw`` URI that cannot create a database."""

    return _build_read_only_uri(database, windows=windows).replace(
        "?mode=ro",
        "?mode=rw",
        1,
    )


def _classify_target(
    database: str | os.PathLike[str],
    *,
    read_only: bool,
) -> tuple[str, SQLiteTargetKind]:
    raw = os.fspath(database)
    if not isinstance(raw, str):
        raise TypeError("SQLite paths must be text paths")
    if "\x00" in raw:
        raise ValueError("Path must not contain NUL")
    if raw.startswith("file:"):
        raise ValueError("Caller-supplied file: SQLite URIs are not supported")
    if raw == ":memory:":
        if read_only:
            raise ValueError("A read-only memory database is not supported")
        return raw, SQLiteTargetKind.MEMORY
    return raw, (
        SQLiteTargetKind.READ_ONLY_URI if read_only else SQLiteTargetKind.PRIVATE_FILE
    )


def _helper_result(selected: Path, response: dict[str, object]) -> PrepareResult:
    """Translate validated, source-free helper refusals at the parent boundary."""
    status = response["status"]
    if status == "private_path_error":
        raise _failure(
            selected,
            PrivatePathStatus(response["privacy_status"]),
            str(response["reason"]),
        )
    if status == "timeout":
        raise HelperTimeoutError()
    if status == "protocol_error":
        raise HelperProtocolError()
    if status != "ok":
        raise HelperUnavailableError()
    return PrepareResult.from_payload(response["result"])
