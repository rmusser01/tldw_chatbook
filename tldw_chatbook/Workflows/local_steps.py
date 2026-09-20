"""Blocking local effects for a retained workflow worker.

Source paths and their containing directories must remain stable during reads.
Known database identities are checked by metadata only, never raw DB probes.
"""

from __future__ import annotations

import asyncio
import os
import stat
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Notes.Notes_Library import NotesInteropService
from tldw_chatbook.Notes.notes_scope_service import NotesScopeService, ScopeType
from tldw_chatbook.Utils.path_validation import validate_path_simple
from tldw_chatbook.Utils.private_paths import (
    PrivatePathStatus,
    lexical_path,
    verify_trusted_directory,
)
from tldw_chatbook.Workflows.expressions import json_copy

_SOURCE_BYTES = 1024 * 1024
_RESULT_BYTES = 1024 * 1024
_DB_SUFFIXES = ("", "-journal", "-wal", "-shm")


def _require_source_file(entry: os.stat_result) -> None:
    if (
        not stat.S_ISREG(entry.st_mode)
        or entry.st_nlink != 1
        or entry.st_uid != os.geteuid()
    ):
        raise ValueError("source_unsafe")


def read_local_text(
    source: Path,
    *,
    protected_paths: tuple[Path, ...],
    before_read: Callable[[], None],
) -> str:
    """Read one stable UTF-8 text file without changing its permissions.

    Args:
        source: Captured local .txt selection.
        protected_paths: Exact known workflow/Notes database paths; their fixed
            sidecar names are also checked. No database discovery is performed.
        before_read: Exact-effect authority recheck immediately before opening.

    Returns:
        Exact text that fits both the raw-source and serialized-result budgets.

    Raises:
        ValueError: Unsafe source, invalid UTF-8 or exceeded byte budget.
        OSError: The selected file or trusted parent is unavailable.
    """
    if (
        os.name != "posix"
        or not getattr(os, "O_NOFOLLOW", 0)
        or not getattr(os, "O_NONBLOCK", 0)
        or not hasattr(os, "geteuid")
    ):
        raise ValueError("source_platform_unverified")
    selected = lexical_path(validate_path_simple(source, probe_existing=False))
    if selected.suffix.lower() != ".txt":
        raise ValueError("source_type")
    posture = verify_trusted_directory(selected.parent, allow_shared_sticky=False)
    if posture.status != PrivatePathStatus.TRUSTED_DIRECTORY:
        raise ValueError("source_platform_unverified")
    entry = selected.lstat()
    _require_source_file(entry)
    for database in protected_paths:
        for suffix in _DB_SUFFIXES:
            try:
                protected = Path(str(database) + suffix).stat()
            except FileNotFoundError:
                continue
            if (entry.st_dev, entry.st_ino) == (protected.st_dev, protected.st_ino):
                raise ValueError("source_database")
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    before_read()
    descriptor = os.open(selected, flags)
    try:
        opened = os.fstat(descriptor)
        _require_source_file(opened)
        if (opened.st_dev, opened.st_ino) != (entry.st_dev, entry.st_ino):
            raise ValueError("source_changed")
        stream = os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise
    with stream:
        raw = stream.read(_SOURCE_BYTES + 1)
    if len(raw) > _SOURCE_BYTES:
        raise ValueError("source_limit")
    text = raw.decode("utf-8", errors="strict")
    json_copy({"text": text}, byte_limit=_RESULT_BYTES)
    return text


@dataclass(frozen=True, repr=False)
class LocalNoteDestination:
    """Captured existing Notes route; never a portable authority grant."""

    scope: NotesScopeService
    owner: NotesInteropService
    db: CharactersRAGDB
    user_id: str
    db_path: str
    client_id: str


class LocalNoteCleanupError(RuntimeError):
    """An existing Notes close call raised; successful readback cannot clear it."""

    def __init__(self) -> None:
        super().__init__("note_cleanup_failed")


def _close_notes_connection(db: CharactersRAGDB) -> None:
    try:
        db.close_connection()
    except Exception:  # noqa: BLE001 - classify escaping cleanup errors without private exception text
        raise LocalNoteCleanupError() from None


def _require_worker() -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    raise RuntimeError("Local Notes operations require a blocking worker")


def capture_local_note_destination(
    scope: NotesScopeService, *, user_id: str
) -> LocalNoteDestination:
    """Capture the actual Notes route off-loop and close this thread's handle.

    Args:
        scope: Existing Notes scope whose local owner selects the database.
        user_id: Nonblank Notes user ID; surrounding whitespace is stripped.

    Returns:
        Frozen destination retaining the scope, owner, database, and user
        identity. The owner's cached database remains available for later use.

    Raises:
        RuntimeError: Called on a thread with a running event loop.
        ValueError: The user ID is invalid or the selected route changes.
        TypeError: The local owner is not a NotesInteropService.
        LocalNoteCleanupError: Closing this thread's database connection fails.
        Exception: An existing Notes owner or database operation fails.
    """
    _require_worker()
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("note_user")
    user_id = user_id.strip()
    owner = scope.local_notes_service
    if not isinstance(owner, NotesInteropService):
        raise TypeError("note_destination")
    db = owner.notes_db(user_id)
    try:
        with owner.bound_notes_db(user_id, db):
            if scope.local_notes_service is not owner or db.client_id != user_id:
                raise ValueError("note_destination_changed")
            return LocalNoteDestination(
                scope, owner, db, user_id, db.db_path_str, db.client_id
            )
    finally:
        _close_notes_connection(db)


@contextmanager
def _bound_destination(destination: LocalNoteDestination) -> Iterator[None]:
    _require_worker()
    try:
        with destination.owner.bound_notes_db(destination.user_id, destination.db):
            if (
                destination.scope.local_notes_service is not destination.owner
                or destination.db.db_path_str != destination.db_path
                or destination.db.client_id != destination.client_id
                or destination.user_id != destination.client_id
            ):
                raise ValueError("note_destination_changed")
            yield
    finally:
        # Never re-resolve the route or evict the owner's cached DB identity.
        _close_notes_connection(destination.db)


def create_local_note(
    destination: LocalNoteDestination,
    *,
    create_note_id: str,
    title: str,
    content: str,
    before_write: Callable[[], None],
) -> str:
    """Create once through existing local policy/transactions; errors may follow commit.

    The caller retains the attempted ID for authorized same-destination readback.
    No update, retry, organization change or Sync v2 profile is requested.

    Args:
        destination: Captured Notes route, rechecked while holding its owner lock.
        create_note_id: Caller-retained attempt ID, nonblank with no surrounding
            whitespace; reused only for authorized readback after uncertainty.
        title: Nonblank note title; surrounding whitespace is stripped.
        content: Exact note content to save, including any whitespace.
        before_write: Authority recheck called under the route lock immediately
            before invoking the existing Notes save operation.

    Returns:
        The created note ID, verified to equal create_note_id.

    Raises:
        RuntimeError: Called on a thread with a running event loop.
        ValueError: Inputs are invalid, the captured route changes, or the
            returned note ID differs from the attempted ID.
        LocalNoteCleanupError: Closing this thread's database connection fails.
        Exception: The authority callback or existing Notes policy or storage
            operation fails. An error does not prove the note was not committed.
    """
    with _bound_destination(destination):
        if (
            not isinstance(create_note_id, str)
            or not create_note_id.strip()
            or create_note_id != create_note_id.strip()
            or not isinstance(title, str)
            or not title.strip()
            or not isinstance(content, str)
        ):
            raise ValueError("note_input")
        before_write()
        note_id = asyncio.run(
            destination.scope.save_note(
                scope=ScopeType.LOCAL_NOTE,
                user_id=destination.user_id,
                create_note_id=create_note_id,
                title=title.strip(),
                content=content,
                sync_v2_profile=None,
            )
        )
        if note_id != create_note_id:
            raise ValueError("note_result_mismatch")
        return note_id


def read_local_note(
    destination: LocalNoteDestination, *, note_id: str
) -> dict[str, Any] | None:
    """Read the active attempted row through the captured route and Notes policy.

    The coordinator must compare normalized title and exact accepted content;
    an existing ID alone cannot confirm an uncertain write.

    Args:
        destination: Captured Notes route, rechecked while holding its owner lock.
        note_id: Exact attempted note ID to read through the captured route.

    Returns:
        The active note row with matching note and client IDs, or None when the
        existing Notes service returns no row. Content is not compared here.

    Raises:
        RuntimeError: Called on a thread with a running event loop.
        ValueError: The captured route changes or the returned row has an
            unexpected shape, identity, or deletion state.
        LocalNoteCleanupError: Closing this thread's database connection fails.
        Exception: An existing Notes policy or database read operation fails.
    """
    with _bound_destination(destination):
        row = asyncio.run(
            destination.scope.get_note_detail(
                scope=ScopeType.LOCAL_NOTE,
                user_id=destination.user_id,
                note_id=note_id,
            )
        )
        if row is None:
            return None
        if (
            not isinstance(row, dict)
            or row.get("id") != note_id
            or row.get("client_id") != destination.client_id
            or row.get("deleted") != 0
        ):
            raise ValueError("note_readback_mismatch")
        return row
