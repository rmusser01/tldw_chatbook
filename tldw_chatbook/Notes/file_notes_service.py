"""Disk-authoritative filesystem operations for a single File Notes root."""

from __future__ import annotations

import errno
import hashlib
import os
import stat
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import wraps
from pathlib import Path
from threading import RLock
from typing import Literal, TypeVar, cast

from loguru import logger

from tldw_chatbook.Notes.file_notes_replica import FileNotesReplica, ReplicaFileInfo
from tldw_chatbook.Notes.file_notes_session_owner import (
    FileNotesSessionOwner,
    SessionBinding,
    SessionChange,
)
from tldw_chatbook.Utils.path_validation import get_safe_relative_path

MAX_FILE_BYTES = 8_000_000
MAX_FILE_CHARS = 2_000_000
SUPPORTED_EXTENSIONS = frozenset({".md", ".markdown", ".txt", ".text"})
UTF8_BOM = b"\xef\xbb\xbf"

OperationStatus = Literal[
    "ok",
    "offline",
    "unsafe",
    "unsupported",
    "missing",
    "exists",
    "readonly",
    "conflict",
    "replica-error",
    "error",
]
_ServiceMethod = TypeVar("_ServiceMethod", bound=Callable[..., object])


def _serialized(method: _ServiceMethod) -> _ServiceMethod:
    @wraps(method)
    def wrapper(
        service: FileNotesService,
        *args: object,
        **kwargs: object,
    ) -> object:
        with service._operation_lock:
            return method(service, *args, **kwargs)

    return cast(_ServiceMethod, wrapper)


@dataclass(frozen=True)
class FileNoteEntry:
    """One supported regular file discovered below the selected root."""

    relative_path: str
    size: int
    mtime_ns: int
    content_hash: str
    editable: bool
    read_only_reason: str | None = None


@dataclass(frozen=True)
class OpenedFileNote:
    """Exact disk baseline and editable body for one opened note."""

    root: str
    relative_path: str
    body: str
    preserved_prefix: bytes
    content_hash: str
    newline: Literal["\n", "\r\n"]
    has_final_newline: bool
    size: int
    mtime_ns: int
    editable: bool
    read_only_reason: str | None
    protected: bool
    raw_bytes: bytes
    replica_warning: str | None = None


@dataclass(frozen=True)
class OperationResult:
    """Typed result for an expected filesystem or recovery outcome."""

    status: OperationStatus
    relative_path: str
    destination_path: str | None = None
    content_hash: str | None = None
    replica_warning: str | None = None
    message: str | None = None

    @property
    def succeeded(self) -> bool:
        """Return whether the disk mutation completed."""
        return self.status == "ok"


@dataclass(frozen=True)
class ScanResult:
    """Supported files currently visible under the selected root."""

    status: Literal["ok", "offline"]
    entries: tuple[FileNoteEntry, ...] = ()
    offline: bool = False
    replica_warning: str | None = None


@dataclass(frozen=True)
class ReconcileResult:
    """Disk scan plus root-scoped replica projection changes."""

    status: Literal["ok", "offline"]
    entries: tuple[FileNoteEntry, ...] = ()
    created: tuple[str, ...] = ()
    modified: tuple[str, ...] = ()
    deleted: tuple[str, ...] = ()
    offline: bool = False
    replica_warning: str | None = None


@dataclass(frozen=True)
class _ObservedFile:
    relative_path: str
    size: int
    mtime_ns: int


class FileNotesService:
    """Operate on one canonical notes root while disk remains authoritative."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        replica: FileNotesReplica | None,
        *,
        operation_lock: RLock | None = None,
        session_owner: FileNotesSessionOwner | None = None,
        session_binding: SessionBinding | None = None,
    ) -> None:
        """Bind the service to one canonical root and optional SQLite replica.

        Args:
            root: Filesystem directory kept authoritative by this service.
            replica: Optional SQLite search and recovery replica.
            operation_lock: Optional lock shared by services using one replica.
            session_owner: Optional process owner for session-change publication.
            session_binding: Root generation used to reject late publications.

        Raises:
            ValueError: If only one session dependency is supplied or the
                binding belongs to another root.
        """
        self.root = Path(root).expanduser().resolve(strict=False)
        self.root_key = str(self.root)
        self._replica = replica
        self._operation_lock = operation_lock or RLock()
        if (session_owner is None) != (session_binding is None):
            raise ValueError(
                "session_owner and session_binding must be supplied together"
            )
        if session_owner is None:
            session_owner = FileNotesSessionOwner()
            session_binding = session_owner.select_root(self.root)
        assert session_binding is not None
        if session_binding.root_key != self.root_key:
            raise ValueError("session binding belongs to another File Notes root")
        self._session_owner = session_owner
        self._session_binding = session_binding
        self._entry_cache: dict[str, FileNoteEntry] = {}
        self._pending_replica_moves: dict[str, str] = {}

    @property
    @_serialized
    def session_changes(self) -> tuple[SessionChange, ...]:
        """Return owner-wide changes for the current root session.

        Returns:
            An immutable snapshot of the owner's current-root session history.
        """
        snapshot = self._session_owner.snapshot(self._session_binding)
        return tuple(item.change for item in snapshot.changes)

    @_serialized
    def close(self) -> None:
        """Close the bound replica after any active service operation."""
        replica = self._replica
        self._replica = None
        if replica is not None:
            replica.close()

    @_serialized
    def scan(self) -> ScanResult:
        """Scan supported regular files without following symlinks.

        Returns:
            Current file entries and any replica warning.
        """
        if not self._root_is_online():
            return ScanResult(status="offline", offline=True)

        entries: list[FileNoteEntry] = []
        warning: str | None = None
        observed, uncertain_paths, _ = self._walk_candidates()
        for relative_path, observed_file in observed.items():
            try:
                opened = self._load_file(relative_path)
            except (OSError, ValueError):
                entries.append(_unreadable_entry(observed_file))
                continue
            entries.append(_entry_from_opened(opened))
            replica_warning = self._upsert_opened(opened)
            warning = _merge_warnings(warning, replica_warning)
        entries.extend(
            FileNoteEntry(
                relative_path=relative_path,
                size=0,
                mtime_ns=0,
                content_hash="",
                editable=False,
                read_only_reason="unreadable",
            )
            for relative_path in uncertain_paths
        )
        entries.sort(key=lambda entry: entry.relative_path)
        self._entry_cache = {
            entry.relative_path: entry
            for entry in entries
        }
        return ScanResult(
            status="ok",
            entries=tuple(entries),
            replica_warning=warning,
        )

    @_serialized
    def open_file(self, relative_path: str) -> OpenedFileNote:
        """Read a supported file from disk and record its exact-byte replica.

        Args:
            relative_path: File path relative to the configured notes root.

        Returns:
            The decoded note body and exact on-disk metadata.

        Raises:
            FileNotFoundError: If the notes root or file is unavailable.
            ValueError: If the path or file content is unsafe or unsupported.
            OSError: If the file cannot be read.
        """
        if not self._root_is_online():
            raise FileNotFoundError(f"File Notes root is offline: {self.root}")
        opened = self._load_file(relative_path)
        protected = False
        warning: str | None = None
        if self._replica is not None:
            try:
                protected = self._replica.is_protected(
                    self.root_key,
                    opened.relative_path,
                )
            except Exception as error:
                warning = _replica_warning(error)
        warning = _merge_warnings(warning, self._upsert_opened(opened))
        return _replace_opened(
            opened,
            protected=protected,
            replica_warning=warning,
        )

    @_serialized
    def save_file(
        self,
        opened: OpenedFileNote,
        body: str,
        *,
        session_key: str,
    ) -> OperationResult:
        """Atomically save an editable body after exact hash checks.

        Args:
            opened: Snapshot returned by :meth:`open_file`.
            body: Replacement note body.
            session_key: Editing-session identifier for protected checkpoints.

        Returns:
            Save status, resulting hash, and any replica warning.
        """
        relative_path = opened.relative_path
        if opened.root != self.root_key:
            return _result("unsafe", relative_path, "Opened note belongs to another root")
        if not opened.editable:
            return _result("readonly", relative_path, opened.read_only_reason)
        if not self._root_is_online():
            return _result("offline", relative_path)
        try:
            path = self._safe_path(relative_path)
            current_bytes, current_stat = _read_regular_file(path)
        except ValueError as error:
            return _result("unsafe", relative_path, str(error))
        except FileNotFoundError:
            return _result("missing", relative_path)
        except OSError as error:
            return _result("error", relative_path, str(error))

        current_hash = _digest(current_bytes)
        if current_hash != opened.content_hash:
            return _result("conflict", relative_path, "Disk bytes changed")

        new_bytes = _serialize_body(opened, body)
        if len(body) > MAX_FILE_CHARS or len(new_bytes) > MAX_FILE_BYTES:
            return _result("readonly", relative_path, "Edited content exceeds limits")

        warning: str | None = None
        protected = opened.protected
        if self._replica is not None:
            try:
                protected = self._replica.is_protected(
                    self.root_key,
                    relative_path,
                )
            except Exception as error:
                warning = _replica_warning(error)
                if opened.protected:
                    return _result(
                        "replica-error",
                        relative_path,
                        "Protected path status is unavailable",
                        warning=warning,
                    )
        if protected:
            if self._replica is None:
                return _result(
                    "replica-error",
                    relative_path,
                    "Protected saves require the replica",
                )
            if not session_key:
                return _result(
                    "replica-error",
                    relative_path,
                    "Protected saves require an editing session key",
                )
            try:
                self._replica.checkpoint(
                    self.root_key,
                    relative_path,
                    current_bytes,
                    content_hash=current_hash,
                    session_key=session_key,
                )
            except Exception as error:
                return _result(
                    "replica-error",
                    relative_path,
                    "Could not commit the protected pre-edit checkpoint",
                    warning=_replica_warning(error),
                )

        temporary_path: str | None = None
        try:
            descriptor, temporary_path = tempfile.mkstemp(
                prefix=f"{path.name}.",
                suffix=".tmp",
                dir=path.parent,
            )
            with os.fdopen(descriptor, "wb") as temporary:
                temporary.write(new_bytes)
                temporary.flush()
                file_mode = stat.S_IMODE(current_stat.st_mode)
                fchmod = getattr(os, "fchmod", None)
                if fchmod is not None:
                    fchmod(temporary.fileno(), file_mode)
            if fchmod is None:
                os.chmod(temporary_path, file_mode)

            rechecked_bytes, _ = _read_regular_file(path)
            if _digest(rechecked_bytes) != opened.content_hash:
                return _result("conflict", relative_path, "Disk bytes changed")
            os.replace(temporary_path, path)
            temporary_path = None
        except FileNotFoundError:
            return _result("missing", relative_path)
        except OSError as error:
            return _result("error", relative_path, str(error))
        finally:
            if temporary_path is not None:
                try:
                    os.unlink(temporary_path)
                except OSError as error:
                    logger.warning(
                        "Could not remove File Notes temporary file '{}': {}",
                        temporary_path,
                        error,
                    )

        new_hash = _digest(new_bytes)
        return self._finish_published_file(
            "modified",
            relative_path,
            path,
            new_bytes,
            content_hash=new_hash,
            warning=warning,
        )

    @_serialized
    def save_copy(
        self,
        opened: OpenedFileNote,
        body: str,
        destination_path: str,
    ) -> OperationResult:
        """Save an opened note's exact format to a new path without clobbering.

        Args:
            opened: Snapshot whose encoding and frontmatter are preserved.
            body: Note body to write.
            destination_path: New path relative to the notes root.

        Returns:
            Copy status, resulting hash, and any replica warning.
        """
        if opened.root != self.root_key:
            return _result(
                "unsafe",
                destination_path,
                "Opened note belongs to another root",
            )
        if not opened.editable:
            return _result("readonly", destination_path, opened.read_only_reason)
        if not self._root_is_online():
            return _result("offline", destination_path)
        try:
            path = self._safe_path(destination_path)
        except ValueError as error:
            return _result("unsafe", destination_path, str(error))
        if not self._is_supported(path):
            return _result("unsupported", destination_path)

        raw_bytes = _serialize_body(opened, body)
        if len(body) > MAX_FILE_CHARS or len(raw_bytes) > MAX_FILE_BYTES:
            return _result("readonly", destination_path, "Edited content exceeds limits")

        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags, 0o666)
        except FileExistsError:
            return _result("exists", destination_path)
        except OSError as error:
            if error.errno in {errno.ELOOP, errno.ENOTDIR}:
                return _result("unsafe", destination_path, str(error))
            return _result("error", destination_path, str(error))
        try:
            with os.fdopen(descriptor, "wb") as destination:
                destination.write(raw_bytes)
        except OSError as error:
            try:
                os.unlink(path)
            except OSError:
                pass
            return _result("error", destination_path, str(error))

        content_hash = _digest(raw_bytes)
        return self._finish_published_file(
            "created",
            destination_path,
            path,
            raw_bytes,
            content_hash=content_hash,
        )

    @_serialized
    def create_file(self, relative_path: str, body: str = "") -> OperationResult:
        """Create one supported UTF-8 file with an exclusive filesystem open.

        Args:
            relative_path: New file path relative to the notes root.
            body: Initial UTF-8 note content.

        Returns:
            Creation status, resulting hash, and any replica warning.
        """
        if not self._root_is_online():
            return _result("offline", relative_path)
        try:
            path = self._safe_path(relative_path)
        except ValueError as error:
            return _result("unsafe", relative_path, str(error))
        if not self._is_supported(path):
            return _result("unsupported", relative_path)
        raw_bytes = body.encode("utf-8")
        if len(body) > MAX_FILE_CHARS or len(raw_bytes) > MAX_FILE_BYTES:
            return _result("readonly", relative_path, "Content exceeds limits")

        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags, 0o666)
        except FileExistsError:
            return _result("exists", relative_path)
        except OSError as error:
            if error.errno in {errno.ELOOP, errno.ENOTDIR}:
                return _result("unsafe", relative_path, str(error))
            return _result("error", relative_path, str(error))
        try:
            with os.fdopen(descriptor, "wb") as destination:
                destination.write(raw_bytes)
        except OSError as error:
            try:
                os.unlink(path)
            except OSError:
                pass
            return _result("error", relative_path, str(error))

        content_hash = _digest(raw_bytes)
        return self._finish_published_file(
            "created",
            relative_path,
            path,
            raw_bytes,
            content_hash=content_hash,
        )

    @_serialized
    def move_file(
        self,
        relative_path: str,
        destination_path: str,
    ) -> OperationResult:
        """Move a file without clobbering by linking then unlinking the source.

        Args:
            relative_path: Existing file path relative to the notes root.
            destination_path: New path relative to the notes root.

        Returns:
            Move status and any replica warning.
        """
        if not self._root_is_online():
            return _result("offline", relative_path, destination=destination_path)
        try:
            source = self._safe_path(relative_path)
            destination = self._safe_path(destination_path)
            if not self._is_supported(source) or not self._is_supported(destination):
                return _result(
                    "unsupported",
                    relative_path,
                    destination=destination_path,
                )
            source_stat = os.lstat(source)
            if not stat.S_ISREG(source_stat.st_mode):
                return _result(
                    "unsafe",
                    relative_path,
                    "Source is not a regular file",
                    destination=destination_path,
                )
        except ValueError as error:
            return _result(
                "unsafe",
                relative_path,
                str(error),
                destination=destination_path,
            )
        except FileNotFoundError:
            return _result("missing", relative_path, destination=destination_path)
        except OSError as error:
            return _result(
                "error",
                relative_path,
                str(error),
                destination=destination_path,
            )
        try:
            os.link(source, destination, follow_symlinks=False)
        except FileExistsError:
            return _result("exists", relative_path, destination=destination_path)
        except OSError as error:
            return _result(
                "error",
                relative_path,
                str(error),
                destination=destination_path,
            )
        try:
            os.unlink(source)
        except OSError as error:
            rollback_error: OSError | None = None
            try:
                os.unlink(destination)
            except OSError as caught:
                rollback_error = caught
            message = str(error)
            if rollback_error is not None:
                message = f"{message}; destination rollback failed: {rollback_error}"
            return _result(
                "error",
                relative_path,
                message,
                destination=destination_path,
            )

        warning: str | None = None
        try:
            moved = self._load_file(destination_path)
        except (OSError, ValueError) as error:
            warning = f"Replica refresh failed: {error}"
        else:
            warning = self._move_opened(relative_path, moved)
        self._pending_replica_moves.pop(destination_path, None)
        for pending_source, pending_destination in tuple(
            self._pending_replica_moves.items()
        ):
            if pending_destination == relative_path:
                self._pending_replica_moves[pending_source] = destination_path
        if warning is None:
            self._pending_replica_moves.pop(relative_path, None)
        else:
            self._pending_replica_moves[relative_path] = destination_path
        self._record_session_change(
            SessionChange("moved", relative_path, destination_path)
        )
        return OperationResult(
            status="ok",
            relative_path=relative_path,
            destination_path=destination_path,
            replica_warning=warning,
        )

    @_serialized
    def delete_file(
        self,
        relative_path: str,
        *,
        expected_hash: str | None = None,
    ) -> OperationResult:
        """Commit a recovery tombstone, recheck bytes, and only then unlink.

        Args:
            relative_path: File path relative to the notes root.
            expected_hash: Optional digest required to match current disk bytes.

        Returns:
            Deletion status, deleted-content hash, and any replica warning.
        """
        if not self._root_is_online():
            return _result("offline", relative_path)
        try:
            path = self._safe_path(relative_path)
            if not self._is_supported(path):
                return _result("unsupported", relative_path)
            observed_stat = os.lstat(path)
            if not stat.S_ISREG(observed_stat.st_mode):
                return _result("unsafe", relative_path, "Not a regular file")
            if self._replica is None:
                return _result(
                    "replica-error",
                    relative_path,
                    "Delete requires the recovery replica",
                )
            raw_bytes, file_stat = _read_regular_file(path)
        except ValueError as error:
            return _result("unsafe", relative_path, str(error))
        except FileNotFoundError:
            return _result("missing", relative_path)
        except OSError as error:
            return _result("error", relative_path, str(error))

        content_hash = _digest(raw_bytes)
        if expected_hash is not None and content_hash != expected_hash:
            return _result("conflict", relative_path, "Disk bytes changed")
        decoded_text = _decode_for_replica(raw_bytes)
        try:
            self._replica.upsert_file(
                self.root_key,
                relative_path,
                raw_bytes,
                content_hash=content_hash,
                decoded_text=decoded_text,
                size=len(raw_bytes),
                mtime_ns=file_stat.st_mtime_ns,
            )
            self._replica.prepare_deletion(
                self.root_key,
                relative_path,
                raw_bytes,
                content_hash=content_hash,
                decoded_text=decoded_text,
            )
        except Exception as error:
            return _result(
                "replica-error",
                relative_path,
                "Could not commit deletion recovery data",
                warning=_replica_warning(error),
            )

        try:
            rechecked_bytes, _ = _read_regular_file(path)
        except (OSError, ValueError) as error:
            warning = self._clear_tombstone(relative_path)
            return _result(
                "error",
                relative_path,
                str(error),
                warning=warning,
            )
        if _digest(rechecked_bytes) != content_hash:
            warning = self._clear_tombstone(relative_path)
            return _result(
                "conflict",
                relative_path,
                "Disk bytes changed",
                warning=warning,
            )
        try:
            os.unlink(path)
        except OSError as error:
            warning = self._clear_tombstone(relative_path)
            return _result(
                "error",
                relative_path,
                str(error),
                warning=warning,
            )

        self._record_session_change(SessionChange("deleted", relative_path))
        return OperationResult(
            status="ok",
            relative_path=relative_path,
            content_hash=content_hash,
        )

    @_serialized
    def restore_file(self, relative_path: str) -> OperationResult:
        """Restore exact tombstoned bytes with an exclusive filesystem create.

        Args:
            relative_path: Tombstoned file path relative to the notes root.

        Returns:
            Restore status, resulting hash, and any replica warning.
        """
        if not self._root_is_online():
            return _result("offline", relative_path)
        if self._replica is None:
            return _result(
                "replica-error",
                relative_path,
                "Restore requires the recovery replica",
            )
        try:
            path = self._safe_path(relative_path)
        except ValueError as error:
            return _result("unsafe", relative_path, str(error))
        if not self._is_supported(path):
            return _result("unsupported", relative_path)
        try:
            raw_bytes = self._replica.get_restore_bytes(
                self.root_key,
                relative_path,
            )
        except Exception as error:
            return _result(
                "replica-error",
                relative_path,
                "Could not load restore bytes",
                warning=_replica_warning(error),
            )
        if raw_bytes is None:
            return _result("missing", relative_path, "No tombstone exists")

        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags, 0o666)
        except FileExistsError:
            return _result("exists", relative_path)
        except OSError as error:
            if error.errno in {errno.ELOOP, errno.ENOTDIR}:
                return _result("unsafe", relative_path, str(error))
            return _result("error", relative_path, str(error))
        try:
            with os.fdopen(descriptor, "wb") as destination:
                destination.write(raw_bytes)
        except OSError as error:
            try:
                os.unlink(path)
            except OSError:
                pass
            return _result("error", relative_path, str(error))

        content_hash = _digest(raw_bytes)
        return self._finish_published_file(
            "restored",
            relative_path,
            path,
            raw_bytes,
            content_hash=content_hash,
        )

    @_serialized
    def reconcile(self) -> ReconcileResult:
        """Project external create/modify/delete changes into the replica.

        Returns:
            Reconciled entries, external change sets, and any replica warning.
        """
        if not self._root_is_online():
            return ReconcileResult(status="offline", offline=True)

        old_files: dict[str, ReplicaFileInfo] | None = None
        warning: str | None = None
        if self._replica is not None:
            try:
                old_files = {
                    item.relative_path: item
                    for item in self._replica.list_active_files(self.root_key)
                }
            except Exception as error:
                warning = _replica_warning(error)
        else:
            warning = "Replica unavailable"

        observed, uncertain_paths, had_walk_error = self._walk_candidates()
        pending_move_entries: dict[str, OpenedFileNote] = {}
        pending_move_sources: set[str] = set()
        if old_files is not None and self._replica is not None:
            for source_path, destination_path in tuple(
                self._pending_replica_moves.items()
            ):
                if source_path not in old_files:
                    self._pending_replica_moves.pop(source_path, None)
                    continue
                source_info = old_files[source_path]
                if source_path in observed:
                    moved_info, move_warning = self._materialize_pending_move(
                        source_path,
                        fallback_mtime_ns=source_info.mtime_ns,
                    )
                    warning = _merge_warnings(warning, move_warning)
                    if moved_info is not None:
                        old_files[moved_info.relative_path] = moved_info
                    continue
                if destination_path not in observed:
                    if had_walk_error:
                        continue
                    moved_info, move_warning = self._materialize_pending_move(
                        source_path,
                        fallback_mtime_ns=source_info.mtime_ns,
                    )
                    warning = _merge_warnings(warning, move_warning)
                    if moved_info is None:
                        pending_move_sources.add(source_path)
                        continue
                    old_files.pop(source_path, None)
                    old_files[moved_info.relative_path] = moved_info
                    continue
                try:
                    moved = self._load_file(destination_path)
                except (OSError, ValueError):
                    pending_move_sources.add(source_path)
                    continue
                move_warning = self._move_opened(source_path, moved)
                if move_warning is not None:
                    warning = _merge_warnings(warning, move_warning)
                    pending_move_sources.add(source_path)
                    pending_move_entries[destination_path] = moved
                    continue
                self._pending_replica_moves.pop(source_path, None)
                old_files.pop(source_path, None)
                old_files[destination_path] = ReplicaFileInfo(
                    relative_path=destination_path,
                    content_hash=moved.content_hash,
                    size=moved.size,
                    mtime_ns=moved.mtime_ns,
                )

        entries: list[FileNoteEntry] = []
        created: list[str] = []
        modified: list[str] = []
        for relative_path, observed_file in observed.items():
            pending_move = pending_move_entries.get(relative_path)
            if pending_move is not None:
                entries.append(_entry_from_opened(pending_move))
                continue
            previous = None if old_files is None else old_files.get(relative_path)
            cached = self._entry_cache.get(relative_path)
            replica_matches = (
                previous is not None
                and previous.size == observed_file.size
                and previous.mtime_ns == observed_file.mtime_ns
            )
            cache_matches = (
                cached is not None
                and cached.size == observed_file.size
                and cached.mtime_ns == observed_file.mtime_ns
            )
            unchanged = replica_matches or (old_files is None and cache_matches)
            if unchanged:
                if cache_matches:
                    assert cached is not None
                    entries.append(cached)
                else:
                    assert previous is not None
                    oversized = observed_file.size > MAX_FILE_BYTES
                    entries.append(
                        FileNoteEntry(
                            relative_path=relative_path,
                            size=observed_file.size,
                            mtime_ns=observed_file.mtime_ns,
                            content_hash=previous.content_hash,
                            editable=not oversized,
                            read_only_reason=(
                                "too-many-bytes" if oversized else None
                            ),
                        )
                    )
                continue

            try:
                opened = self._load_file(relative_path)
            except (OSError, ValueError):
                entries.append(_unreadable_entry(observed_file))
                uncertain_paths.add(relative_path)
                continue
            entry = _entry_from_opened(opened)
            entries.append(entry)
            warning = _merge_warnings(warning, self._upsert_opened(opened))
            if previous is None:
                if old_files is not None:
                    created.append(relative_path)
            elif opened.content_hash != previous.content_hash:
                modified.append(relative_path)

        entries.extend(
            FileNoteEntry(
                relative_path=relative_path,
                size=0,
                mtime_ns=0,
                content_hash="",
                editable=False,
                read_only_reason="unreadable",
            )
            for relative_path in uncertain_paths
            if relative_path not in observed
        )
        deleted: list[str] = []
        if old_files is not None and not had_walk_error:
            deleted = sorted(
                set(old_files)
                - set(observed)
                - uncertain_paths
                - pending_move_sources
            )
            for relative_path in deleted:
                try:
                    assert self._replica is not None
                    self._replica.mark_deleted(self.root_key, relative_path)
                except Exception as error:
                    warning = _merge_warnings(warning, _replica_warning(error))
        entries.sort(key=lambda entry: entry.relative_path)
        self._entry_cache = {
            entry.relative_path: entry
            for entry in entries
        }
        return ReconcileResult(
            status="ok",
            entries=tuple(entries),
            created=tuple(sorted(created)),
            modified=tuple(sorted(modified)),
            deleted=tuple(deleted),
            replica_warning=warning,
        )

    @_serialized
    def protect_path(
        self,
        relative_path: str,
        *,
        is_prefix: bool = False,
    ) -> OperationResult:
        """Enable future pre-edit checkpoints for a file or folder prefix.

        Args:
            relative_path: Exact file path or folder prefix.
            is_prefix: Whether ``relative_path`` identifies a folder prefix.

        Returns:
            Protection status and any replica warning.
        """
        if self._replica is None:
            return _result("replica-error", relative_path)
        try:
            if relative_path or not is_prefix:
                self._safe_path(relative_path)
            self._replica.protect(
                self.root_key,
                relative_path,
                is_prefix=is_prefix,
            )
        except ValueError as error:
            return _result("unsafe", relative_path, str(error))
        except Exception as error:
            return _result(
                "replica-error",
                relative_path,
                warning=_replica_warning(error),
            )
        return OperationResult(status="ok", relative_path=relative_path)

    @_serialized
    def unprotect_path(
        self,
        relative_path: str,
        *,
        is_prefix: bool = False,
    ) -> OperationResult:
        """Stop future checkpoints without removing existing revisions.

        Args:
            relative_path: Exact file path or folder prefix.
            is_prefix: Whether ``relative_path`` identifies a folder prefix.

        Returns:
            Unprotection status and any replica warning.
        """
        if self._replica is None:
            return _result("replica-error", relative_path)
        try:
            if relative_path or not is_prefix:
                self._safe_path(relative_path)
            self._replica.unprotect(
                self.root_key,
                relative_path,
                is_prefix=is_prefix,
            )
        except ValueError as error:
            return _result("unsafe", relative_path, str(error))
        except Exception as error:
            return _result(
                "replica-error",
                relative_path,
                warning=_replica_warning(error),
            )
        return OperationResult(status="ok", relative_path=relative_path)

    def _walk_candidates(
        self,
    ) -> tuple[dict[str, _ObservedFile], set[str], bool]:
        observed: dict[str, _ObservedFile] = {}
        uncertain_paths: set[str] = set()
        had_walk_error = False

        def collect_error(error: OSError) -> None:
            nonlocal had_walk_error
            had_walk_error = True

        for current, directory_names, file_names in os.walk(
            self.root,
            followlinks=False,
            onerror=collect_error,
        ):
            current_path = Path(current)
            directory_names[:] = sorted(
                name
                for name in directory_names
                if name != ".git" and not _is_symlink(current_path / name)
            )
            for name in sorted(file_names):
                path = current_path / name
                if not self._is_supported(path) or _is_symlink(path):
                    continue
                relative_path = path.relative_to(self.root).as_posix()
                try:
                    file_stat = os.lstat(path)
                except OSError:
                    uncertain_paths.add(relative_path)
                    continue
                if not stat.S_ISREG(file_stat.st_mode):
                    continue
                observed[relative_path] = _ObservedFile(
                    relative_path=relative_path,
                    size=file_stat.st_size,
                    mtime_ns=file_stat.st_mtime_ns,
                )
        return dict(sorted(observed.items())), uncertain_paths, had_walk_error

    def _load_file(self, relative_path: str) -> OpenedFileNote:
        path = self._safe_path(relative_path)
        if not self._is_supported(path):
            raise ValueError(f"Unsupported File Notes extension: {relative_path}")
        file_stat = os.lstat(path)
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError(f"unsafe non-regular file: {relative_path}")
        raw_bytes, file_stat = _read_regular_file(path)
        return _parse_opened(
            self.root_key,
            relative_path,
            raw_bytes,
            file_stat,
        )

    def _safe_path(self, relative_path: str) -> Path:
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError("unsafe empty relative path")
        if "\x00" in relative_path:
            raise ValueError("unsafe relative path")
        candidate_path = Path(relative_path)
        if candidate_path.as_posix() != relative_path:
            raise ValueError(f"unsafe non-canonical path: {relative_path}")
        if candidate_path.is_absolute():
            raise ValueError("unsafe absolute path")
        parts = candidate_path.parts
        if not parts or any(part in {"", ".", ".."} for part in parts):
            raise ValueError("unsafe path traversal")

        current = self.root
        for part in parts:
            if part == ".git":
                raise ValueError("unsafe .git path")
            current = current / part
            try:
                file_stat = os.lstat(current)
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(file_stat.st_mode):
                raise ValueError(f"symlink traversal is not allowed: {relative_path}")

        resolved = current.resolve(strict=False)
        if get_safe_relative_path(current, self.root) is None:
            raise ValueError(f"unsafe path outside root: {relative_path}")
        try:
            resolved.relative_to(self.root)
        except ValueError as error:
            raise ValueError(f"unsafe path outside root: {relative_path}") from error
        return current

    def _root_is_online(self) -> bool:
        try:
            return (
                self.root.is_dir()
                and not self.root.is_symlink()
                and os.access(self.root, os.R_OK | os.X_OK)
            )
        except OSError:
            return False

    @staticmethod
    def _is_supported(path: Path) -> bool:
        return path.suffix.lower() in SUPPORTED_EXTENSIONS

    def _finish_published_file(
        self,
        action: Literal["created", "modified", "restored"],
        relative_path: str,
        path: Path,
        raw_bytes: bytes,
        *,
        content_hash: str,
        warning: str | None = None,
    ) -> OperationResult:
        self._record_session_change(SessionChange(action, relative_path))
        try:
            file_stat = path.stat()
        except OSError as error:
            warning = _merge_warnings(
                warning,
                f"Replica update deferred: {error}",
            )
        else:
            warning = _merge_warnings(
                warning,
                self._upsert_bytes(
                    relative_path,
                    raw_bytes,
                    content_hash=content_hash,
                    file_stat=file_stat,
                ),
            )
        return OperationResult(
            status="ok",
            relative_path=relative_path,
            content_hash=content_hash,
            replica_warning=warning,
        )

    def _record_session_change(self, change: SessionChange) -> None:
        self._session_owner.record_change(self._session_binding, change)

    def _bind_session_owner(
        self,
        owner: FileNotesSessionOwner,
        binding: SessionBinding,
    ) -> None:
        """Bind a scanned, unpublished service to the process session owner."""
        if binding.root_key != self.root_key:
            raise ValueError("session binding belongs to another File Notes root")
        self._session_owner = owner
        self._session_binding = binding

    def _upsert_opened(self, opened: OpenedFileNote) -> str | None:
        return self._upsert_bytes(
            opened.relative_path,
            opened.raw_bytes,
            content_hash=opened.content_hash,
            mtime_ns=opened.mtime_ns,
        )

    def _move_opened(
        self,
        source_path: str,
        opened: OpenedFileNote,
    ) -> str | None:
        if self._replica is None:
            return "Replica unavailable"
        try:
            self._replica.move_file(
                self.root_key,
                source_path,
                opened.relative_path,
                opened.raw_bytes,
                content_hash=opened.content_hash,
                decoded_text=_decode_for_replica(opened.raw_bytes),
                size=opened.size,
                mtime_ns=opened.mtime_ns,
            )
        except Exception as error:
            return _replica_warning(error)
        return None

    def _materialize_pending_move(
        self,
        source_path: str,
        *,
        fallback_mtime_ns: int,
    ) -> tuple[ReplicaFileInfo | None, str | None]:
        destination_path = self._pending_replica_moves.get(source_path)
        if destination_path is None:
            return None, None
        if self._replica is None:
            return None, "Replica unavailable"

        try:
            moved = self._load_file(destination_path)
        except FileNotFoundError:
            try:
                retained_bytes = self._replica.get_bytes(
                    self.root_key,
                    source_path,
                )
                if retained_bytes is None:
                    return (
                        None,
                        "Replica update deferred: retained move source is unavailable",
                    )
                retained_hash = _digest(retained_bytes)
                self._replica.move_file(
                    self.root_key,
                    source_path,
                    destination_path,
                    retained_bytes,
                    content_hash=retained_hash,
                    decoded_text=_decode_for_replica(retained_bytes),
                    size=len(retained_bytes),
                    mtime_ns=fallback_mtime_ns,
                )
            except Exception as error:
                return None, _replica_warning(error)
            moved_info = ReplicaFileInfo(
                relative_path=destination_path,
                content_hash=retained_hash,
                size=len(retained_bytes),
                mtime_ns=fallback_mtime_ns,
            )
        except (OSError, ValueError) as error:
            return None, f"Replica update deferred: {error}"
        else:
            warning = self._move_opened(source_path, moved)
            if warning is not None:
                return None, warning
            moved_info = ReplicaFileInfo(
                relative_path=destination_path,
                content_hash=moved.content_hash,
                size=moved.size,
                mtime_ns=moved.mtime_ns,
            )

        self._pending_replica_moves.pop(source_path, None)
        return moved_info, None

    def _upsert_bytes(
        self,
        relative_path: str,
        raw_bytes: bytes,
        *,
        content_hash: str,
        file_stat: os.stat_result | None = None,
        mtime_ns: int | None = None,
    ) -> str | None:
        if self._replica is None:
            return "Replica unavailable"
        observed_mtime_ns = (
            file_stat.st_mtime_ns if file_stat is not None else mtime_ns
        )
        if observed_mtime_ns is None:
            raise ValueError("mtime_ns is required for replica upsert")
        _, pending_warning = self._materialize_pending_move(
            relative_path,
            fallback_mtime_ns=observed_mtime_ns,
        )
        if pending_warning is not None:
            return pending_warning
        try:
            self._replica.upsert_file(
                self.root_key,
                relative_path,
                raw_bytes,
                content_hash=content_hash,
                decoded_text=_decode_for_replica(raw_bytes),
                size=len(raw_bytes),
                mtime_ns=observed_mtime_ns,
            )
        except Exception as error:
            return _replica_warning(error)
        return None

    def _clear_tombstone(self, relative_path: str) -> str | None:
        try:
            assert self._replica is not None
            self._replica.clear_tombstone(self.root_key, relative_path)
        except Exception as error:
            return _replica_warning(error)
        return None


def _unreadable_entry(observed: _ObservedFile) -> FileNoteEntry:
    return FileNoteEntry(
        relative_path=observed.relative_path,
        size=observed.size,
        mtime_ns=observed.mtime_ns,
        content_hash="",
        editable=False,
        read_only_reason="unreadable",
    )


def _parse_opened(
    root: str,
    relative_path: str,
    raw_bytes: bytes,
    file_stat: os.stat_result,
) -> OpenedFileNote:
    prefix = UTF8_BOM if raw_bytes.startswith(UTF8_BOM) else b""
    content = raw_bytes[len(prefix) :]
    frontmatter_end = _frontmatter_end(content)
    if frontmatter_end is not None:
        prefix += content[:frontmatter_end]
        body_bytes = content[frontmatter_end:]
    else:
        body_bytes = content
    reason: str | None = None
    body = ""
    newline: Literal["\n", "\r\n"] = "\n"
    has_final_newline = False
    try:
        decoded_content = content.decode("utf-8")
        body_text = body_bytes.decode("utf-8")
    except UnicodeDecodeError:
        decoded_content = ""
        body_text = ""
        reason = "undecodable-utf8"

    if reason is None:
        newline, mixed = _body_newline(body_text, prefix)
        body = body_text.replace("\r\n", "\n")
        has_final_newline = body.endswith("\n")
        if mixed:
            reason = "mixed-newlines"
        elif len(decoded_content) > MAX_FILE_CHARS:
            reason = "too-many-chars"
        elif len(raw_bytes) > MAX_FILE_BYTES:
            reason = "too-many-bytes"
    elif len(raw_bytes) > MAX_FILE_BYTES:
        reason = "too-many-bytes"

    if len(raw_bytes) > MAX_FILE_BYTES:
        reason = "too-many-bytes"
        body = ""
    elif reason == "too-many-chars":
        body = ""

    return OpenedFileNote(
        root=root,
        relative_path=relative_path,
        body=body,
        preserved_prefix=prefix,
        content_hash=_digest(raw_bytes),
        newline=newline,
        has_final_newline=has_final_newline,
        size=len(raw_bytes),
        mtime_ns=file_stat.st_mtime_ns,
        editable=reason is None,
        read_only_reason=reason,
        protected=False,
        raw_bytes=raw_bytes,
    )


def _frontmatter_end(content: bytes) -> int | None:
    lines = content.splitlines(keepends=True)
    if not lines or _line_without_ending(lines[0]) != b"---":
        return None
    offset = len(lines[0])
    for line in lines[1:]:
        offset += len(line)
        if _line_without_ending(line) in {b"---", b"..."}:
            return offset
    return None


def _line_without_ending(line: bytes) -> bytes:
    if line.endswith(b"\r\n"):
        return line[:-2]
    if line.endswith(b"\n") or line.endswith(b"\r"):
        return line[:-1]
    return line


def _body_newline(
    body: str,
    prefix: bytes,
) -> tuple[Literal["\n", "\r\n"], bool]:
    crlf_count = body.count("\r\n")
    without_crlf = body.replace("\r\n", "")
    has_bare_lf = "\n" in without_crlf
    has_bare_cr = "\r" in without_crlf
    mixed = has_bare_cr or (crlf_count > 0 and has_bare_lf)
    if crlf_count:
        return "\r\n", mixed
    if has_bare_lf:
        return "\n", mixed
    return ("\r\n" if b"\r\n" in prefix else "\n"), mixed


def _serialize_body(opened: OpenedFileNote, body: str) -> bytes:
    normalized = body.replace("\r\n", "\n").replace("\r", "\n")
    if opened.has_final_newline and not normalized.endswith("\n"):
        normalized += "\n"
    elif not opened.has_final_newline and normalized.endswith("\n"):
        normalized = normalized.rstrip("\n")
    if opened.newline == "\r\n":
        normalized = normalized.replace("\n", "\r\n")
    return opened.preserved_prefix + normalized.encode("utf-8")


def _read_regular_file(path: Path) -> tuple[bytes, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError(f"unsafe non-regular file: {path}")
        with os.fdopen(descriptor, "rb") as source:
            descriptor = -1
            return source.read(), file_stat
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _entry_from_opened(opened: OpenedFileNote) -> FileNoteEntry:
    return FileNoteEntry(
        relative_path=opened.relative_path,
        size=opened.size,
        mtime_ns=opened.mtime_ns,
        content_hash=opened.content_hash,
        editable=opened.editable,
        read_only_reason=opened.read_only_reason,
    )


def _replace_opened(
    opened: OpenedFileNote,
    *,
    protected: bool,
    replica_warning: str | None,
) -> OpenedFileNote:
    return replace(
        opened,
        protected=protected,
        replica_warning=replica_warning,
    )


def _decode_for_replica(raw_bytes: bytes) -> str | None:
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _digest(raw_bytes: bytes) -> str:
    return hashlib.sha256(raw_bytes).hexdigest()


def _is_symlink(path: Path) -> bool:
    try:
        return path.is_symlink()
    except OSError:
        return True


def _replica_warning(error: Exception) -> str:
    return f"Replica unavailable: {error}"


def _merge_warnings(*warnings: str | None) -> str | None:
    unique = tuple(dict.fromkeys(warning for warning in warnings if warning))
    return "; ".join(unique) if unique else None


def _result(
    status: OperationStatus,
    relative_path: str,
    message: str | None = None,
    *,
    destination: str | None = None,
    warning: str | None = None,
) -> OperationResult:
    return OperationResult(
        status=status,
        relative_path=relative_path,
        destination_path=destination,
        replica_warning=warning,
        message=message,
    )
