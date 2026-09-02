"""Crash-safe private files for Local Collections capture copies."""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from tldw_chatbook.Utils.private_paths import (
    PrivateFileWritePrecondition,
    PrivatePathError,
    atomic_private_write_bytes,
    lexical_path,
    open_private_binary,
    secure_private_directory,
    unlink_private_file,
)

from .collections_capture_models import (
    CaptureActionResult,
    CaptureIdentity,
    CaptureOfflineCopy,
    CollectionsCaptureError,
)
from .collections_capture_repository import CollectionsCaptureRepository


DEFAULT_MAX_COPY_BYTES = 50 * 1024 * 1024
DEFAULT_MAX_AUTHORITY_BYTES = 1024 * 1024 * 1024
MAX_RECONCILE_BATCH = 100
_FINGERPRINT_PATTERN = re.compile(r"[A-Za-z0-9_-]{8,128}")
_FILE_ID_PATTERN = re.compile(r"offline-[0-9a-f]{32}")


@dataclass(frozen=True)
class OfflineReconcileResult:
    """One bounded pass through the durable recovery cursor."""

    processed: int
    has_more: bool
    cursor_kind: str
    cursor_value: str


class CollectionsOfflineStore:
    """Coordinate capture file publication with repository metadata."""

    def __init__(
        self,
        repository: CollectionsCaptureRepository,
        *,
        data_root: Path,
        authority_fingerprint: str,
        max_copy_bytes: int = DEFAULT_MAX_COPY_BYTES,
        max_authority_bytes: int = DEFAULT_MAX_AUTHORITY_BYTES,
    ) -> None:
        if not isinstance(repository, CollectionsCaptureRepository):
            raise CollectionsCaptureError("invalid_capture_repository")
        if not isinstance(data_root, Path):
            raise CollectionsCaptureError("invalid_offline_data_root")
        if not isinstance(authority_fingerprint, str) or not _FINGERPRINT_PATTERN.fullmatch(
            authority_fingerprint
        ):
            raise CollectionsCaptureError("invalid_authority_fingerprint")
        self._require_limit(max_copy_bytes, "invalid_offline_copy_limit")
        self._require_limit(max_authority_bytes, "invalid_offline_authority_limit")
        if max_copy_bytes > max_authority_bytes:
            raise CollectionsCaptureError("invalid_offline_quota")

        self.repository = repository
        self.authority_fingerprint = authority_fingerprint
        self.max_copy_bytes = max_copy_bytes
        self.max_authority_bytes = max_authority_bytes
        self.authority_root = (
            lexical_path(data_root) / "collections_archives" / authority_fingerprint
        )
        try:
            secure_private_directory(
                self.authority_root,
                create=True,
                application_owned=True,
            )
        except (OSError, ValueError) as exc:
            raise CollectionsCaptureError("offline_store_unavailable") from exc
        self._lifecycle_lock_path = self.authority_root / ".lifecycle.lock"
        self._ensure_lifecycle_lock()
        self._initialize_cursor()

    def save_copy(
        self,
        identity: CaptureIdentity,
        payload: bytes,
        media_type: str,
    ) -> CaptureOfflineCopy:
        """Reserve quota, atomically publish bytes, and mark the copy ready."""
        with self._lifecycle_lock(blocking=True) as acquired:
            if not acquired:
                raise CollectionsCaptureError("offline_store_unavailable")
            return self._save_copy_locked(identity, payload, media_type)

    def _save_copy_locked(
        self,
        identity: CaptureIdentity,
        payload: bytes,
        media_type: str,
    ) -> CaptureOfflineCopy:
        if not isinstance(payload, bytes):
            raise CollectionsCaptureError("invalid_offline_payload")
        if len(payload) > self.max_copy_bytes:
            raise CollectionsCaptureError("offline_copy_too_large")
        digest = self._digest(payload)
        reservation = self.repository.reserve_offline_copy(
            identity,
            reserved_size=len(payload),
            media_type=media_type,
            content_hash=digest,
            max_copy_bytes=self.max_copy_bytes,
            max_authority_bytes=self.max_authority_bytes,
        )
        target = self._path_for_file(reservation.file_id)
        try:
            self._before_publish()
            secure_private_directory(
                target.parent,
                create=True,
                application_owned=True,
            )
            atomic_private_write_bytes(
                target,
                payload,
                application_owned_directory=target.parent,
                target_precondition=PrivateFileWritePrecondition.missing(),
            )
        except Exception as exc:
            self._record_write_failure(reservation)
            self._best_effort_unlink(target)
            self._remove_temporaries(reservation.file_id)
            raise CollectionsCaptureError("offline_store_unavailable") from exc

        # This seam deliberately sits outside the handled write block. A crash
        # here leaves a staged row plus a complete atomic file for reconciliation.
        self._after_publish()
        return self.repository.complete_offline_copy(
            identity,
            file_id=reservation.file_id,
            expected_revision=reservation.revision,
            content_hash=digest,
            actual_size=len(payload),
            media_type=media_type,
        )

    def open_copy(self, identity: CaptureIdentity) -> bytes:
        """Read and verify the ready private copy for one active capture."""
        with self._lifecycle_lock(blocking=True) as acquired:
            if not acquired:
                raise CollectionsCaptureError("offline_store_unavailable")
            return self._open_copy_locked(identity)

    def _open_copy_locked(self, identity: CaptureIdentity) -> bytes:
        detail = self.repository.get_detail(identity)
        if detail is None or detail.offline_copy is None:
            raise CollectionsCaptureError("offline_copy_not_found")
        copy = detail.offline_copy
        if copy.state != "ready":
            raise CollectionsCaptureError("offline_copy_unavailable")
        try:
            target = self._path_for_copy(copy)
        except CollectionsCaptureError:
            self._best_effort_fail(copy, "offline_integrity_failed")
            raise CollectionsCaptureError("offline_copy_unavailable") from None
        try:
            payload = self._read_private_file(target)
        except FileNotFoundError:
            self._best_effort_fail(copy, "offline_missing")
            raise CollectionsCaptureError("offline_copy_unavailable") from None
        except (CollectionsCaptureError, PrivatePathError):
            self._best_effort_fail(copy, "offline_integrity_failed")
            self._best_effort_unlink(target)
            raise CollectionsCaptureError("offline_copy_unavailable") from None
        if copy.size != len(payload) or copy.content_hash != self._digest(payload):
            self._best_effort_fail(copy, "offline_integrity_failed")
            self._best_effort_unlink(target)
            raise CollectionsCaptureError("offline_copy_unavailable")
        return payload

    def delete_copy(self, identity: CaptureIdentity) -> CaptureActionResult:
        """Tombstone, unlink, then remove one active capture's file metadata."""
        with self._lifecycle_lock(blocking=True) as acquired:
            if not acquired:
                raise CollectionsCaptureError("offline_store_unavailable")
            return self._delete_copy_locked(identity)

    def hard_delete(
        self,
        identity: CaptureIdentity,
        *,
        expected_revision: int,
    ) -> CaptureActionResult:
        """Tombstone a capture while excluding file publication/recovery."""
        with self._lifecycle_lock(blocking=True) as acquired:
            if not acquired:
                raise CollectionsCaptureError("offline_store_unavailable")
            return self._hard_delete_locked(
                identity,
                expected_revision=expected_revision,
            )

    def _hard_delete_locked(
        self,
        identity: CaptureIdentity,
        *,
        expected_revision: int,
    ) -> CaptureActionResult:
        result = self.repository.hard_delete(
            identity,
            expected_revision=expected_revision,
        )
        rows = self._capture_file_rows(identity.capture_id, MAX_RECONCILE_BATCH)
        for row in rows:
            self._reconcile_file_row(row)
        self._finish_capture_purge(identity.capture_id)
        return result

    def _delete_copy_locked(self, identity: CaptureIdentity) -> CaptureActionResult:
        detail = self.repository.get_detail(identity)
        if detail is None or detail.offline_copy is None:
            raise CollectionsCaptureError("offline_copy_not_found")
        copy = detail.offline_copy
        target = self._path_for_copy(copy)
        begun = self.repository.begin_offline_copy_purge(
            identity,
            file_id=copy.file_id,
            expected_revision=copy.revision,
        )
        try:
            unlink_private_file(
                target,
                application_owned_directory=self._file_root(copy.file_id),
            )
        except (OSError, ValueError) as exc:
            raise CollectionsCaptureError("offline_store_unavailable") from exc
        return self.repository.finish_offline_copy_purge(
            identity,
            file_id=copy.file_id,
            expected_revision=begun.revision or copy.revision + 1,
        )

    def reconcile_batch(self, *, limit: int) -> OfflineReconcileResult:
        """Process at most ``limit`` durable recovery records."""
        self._require_limit(limit, "invalid_reconcile_limit")
        if limit > MAX_RECONCILE_BATCH:
            raise CollectionsCaptureError("invalid_reconcile_limit")
        with self._lifecycle_lock(blocking=False) as acquired:
            if not acquired:
                cursor_kind, cursor_value = self._load_cursor()
                return OfflineReconcileResult(
                    0,
                    True,
                    cursor_kind,
                    cursor_value,
                )
            return self._reconcile_batch_locked(limit)

    def _reconcile_batch_locked(self, limit: int) -> OfflineReconcileResult:
        cursor_kind, cursor_value = self._load_cursor()
        if cursor_kind == "files":
            rows = self._file_rows_after(cursor_value, limit)
            if not rows:
                self._save_cursor("purges", "")
                return OfflineReconcileResult(0, True, "purges", "")
            for row in rows:
                self._reconcile_file_row(row)
                cursor_value = str(row["file_id"])
                self._save_cursor("files", cursor_value)
            return OfflineReconcileResult(
                len(rows),
                True,
                "files",
                cursor_value,
            )

        rows = self._purge_rows_after(cursor_value, limit)
        if not rows:
            self._save_cursor("files", "")
            return OfflineReconcileResult(0, False, "files", "")
        for row in rows:
            self._finish_capture_purge(str(row["capture_id"]))
            cursor_value = str(row["capture_id"])
            self._save_cursor("purges", cursor_value)
        return OfflineReconcileResult(
            len(rows),
            True,
            "purges",
            cursor_value,
        )

    def _before_publish(self) -> None:
        """Test seam immediately before the atomic write."""

    def _after_publish(self) -> None:
        """Test seam after the atomic write and before metadata promotion."""

    def _ensure_lifecycle_lock(self) -> None:
        try:
            with open_private_binary(self._lifecycle_lock_path):
                return
        except FileNotFoundError:
            pass
        except (FileNotFoundError, PrivatePathError) as exc:
            raise CollectionsCaptureError("offline_store_unavailable") from exc
        try:
            atomic_private_write_bytes(
                self._lifecycle_lock_path,
                b"\0",
                application_owned_directory=self.authority_root,
                target_precondition=PrivateFileWritePrecondition.missing(),
            )
        except PrivatePathError:
            # Another process may have won the missing-target race. Opening the
            # final private file distinguishes that case from an unsafe target.
            try:
                with open_private_binary(self._lifecycle_lock_path):
                    return
            except (FileNotFoundError, PrivatePathError) as exc:
                raise CollectionsCaptureError("offline_store_unavailable") from exc

    @contextmanager
    def _lifecycle_lock(self, *, blocking: bool) -> Iterator[bool]:
        try:
            with open_private_binary(self._lifecycle_lock_path) as opened:
                acquired = self._acquire_file_lock(
                    opened.stream.fileno(),
                    blocking=blocking,
                )
                try:
                    yield acquired
                finally:
                    if acquired:
                        self._release_file_lock(opened.stream.fileno())
        except PrivatePathError as exc:
            raise CollectionsCaptureError("offline_store_unavailable") from exc

    @staticmethod
    def _acquire_file_lock(file_descriptor: int, *, blocking: bool) -> bool:
        if os.name == "posix":
            import fcntl  # noqa: PLC0415 - unavailable on Windows

            flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
            try:
                fcntl.flock(file_descriptor, flags)
            except BlockingIOError:
                return False
            return True
        if os.name == "nt":
            import msvcrt  # noqa: PLC0415 - unavailable on POSIX

            mode = msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK
            try:
                os.lseek(file_descriptor, 0, os.SEEK_SET)
                msvcrt.locking(file_descriptor, mode, 1)
            except OSError:
                if not blocking:
                    return False
                raise
            return True
        raise CollectionsCaptureError("offline_store_unavailable")

    @staticmethod
    def _release_file_lock(file_descriptor: int) -> None:
        if os.name == "posix":
            import fcntl  # noqa: PLC0415 - unavailable on Windows

            fcntl.flock(file_descriptor, fcntl.LOCK_UN)
            return
        if os.name == "nt":
            import msvcrt  # noqa: PLC0415 - unavailable on POSIX

            os.lseek(file_descriptor, 0, os.SEEK_SET)
            msvcrt.locking(file_descriptor, msvcrt.LK_UNLCK, 1)
            return
        raise CollectionsCaptureError("offline_store_unavailable")

    def _initialize_cursor(self) -> None:
        now = self.repository._clock()
        with self.repository.db.transaction() as connection:
            row = connection.execute(
                "SELECT authority_fingerprint FROM collection_capture_scavenge_state "
                "WHERE authority_key = ?",
                (self.repository.authority_key,),
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO collection_capture_scavenge_state ("
                    "authority_key, authority_fingerprint, cursor_kind, cursor_value, "
                    "updated_at) VALUES (?, ?, 'files', '', ?)",
                    (
                        self.repository.authority_key,
                        self.authority_fingerprint,
                        now,
                    ),
                )
            elif row["authority_fingerprint"] is None:
                connection.execute(
                    "UPDATE collection_capture_scavenge_state "
                    "SET authority_fingerprint = ?, cursor_kind = 'files', "
                    "cursor_value = '', updated_at = ? WHERE authority_key = ? "
                    "AND authority_fingerprint IS NULL",
                    (
                        self.authority_fingerprint,
                        now,
                        self.repository.authority_key,
                    ),
                )
            elif row["authority_fingerprint"] != self.authority_fingerprint:
                raise CollectionsCaptureError("offline_authority_mismatch")

    def _load_cursor(self) -> tuple[str, str]:
        with self.repository.db.connection() as connection:
            row = connection.execute(
                "SELECT cursor_kind, cursor_value, authority_fingerprint "
                "FROM collection_capture_scavenge_state WHERE authority_key = ?",
                (self.repository.authority_key,),
            ).fetchone()
        if row is None or row["authority_fingerprint"] != self.authority_fingerprint:
            raise CollectionsCaptureError("offline_authority_mismatch")
        kind = str(row["cursor_kind"] or "files")
        if kind not in {"files", "purges"}:
            kind = "files"
        return kind, str(row["cursor_value"] or "")

    def _save_cursor(self, kind: str, value: str) -> None:
        with self.repository.db.transaction() as connection:
            cursor = connection.execute(
                "UPDATE collection_capture_scavenge_state SET cursor_kind = ?, "
                "cursor_value = ?, updated_at = ? WHERE authority_key = ? "
                "AND authority_fingerprint = ?",
                (
                    kind,
                    value,
                    self.repository._clock(),
                    self.repository.authority_key,
                    self.authority_fingerprint,
                ),
            )
            if cursor.rowcount != 1:
                raise CollectionsCaptureError("offline_authority_mismatch")

    def _file_rows_after(self, cursor: str, limit: int) -> list[sqlite3.Row]:
        with self.repository.db.connection() as connection:
            return list(
                connection.execute(
                    "SELECT offline.*, item.purge_state FROM "
                    "collection_capture_offline_files AS offline "
                    "JOIN collection_capture_items AS item "
                    "ON item.authority_key = offline.authority_key "
                    "AND item.capture_id = offline.capture_id "
                    "WHERE offline.authority_key = ? AND offline.file_id > ? "
                    "ORDER BY offline.file_id LIMIT ?",
                    (self.repository.authority_key, cursor, limit),
                ).fetchall()
            )

    def _purge_rows_after(self, cursor: str, limit: int) -> list[sqlite3.Row]:
        with self.repository.db.connection() as connection:
            return list(
                connection.execute(
                    "SELECT capture_id FROM collection_capture_items "
                    "WHERE authority_key = ? AND purge_state = 'pending' "
                    "AND capture_id > ? ORDER BY capture_id LIMIT ?",
                    (self.repository.authority_key, cursor, limit),
                ).fetchall()
            )

    def _capture_file_rows(
        self,
        capture_id: str,
        limit: int,
    ) -> list[sqlite3.Row]:
        with self.repository.db.connection() as connection:
            return list(
                connection.execute(
                    "SELECT offline.*, item.purge_state FROM "
                    "collection_capture_offline_files AS offline "
                    "JOIN collection_capture_items AS item "
                    "ON item.authority_key = offline.authority_key "
                    "AND item.capture_id = offline.capture_id "
                    "WHERE offline.authority_key = ? AND offline.capture_id = ? "
                    "ORDER BY offline.file_id LIMIT ?",
                    (self.repository.authority_key, capture_id, limit),
                ).fetchall()
            )

    def _reconcile_file_row(self, row: sqlite3.Row) -> None:
        try:
            identity = CaptureIdentity(
                self.repository.authority_key,
                str(row["capture_id"]),
            )
            file_id = str(row["file_id"])
            safe_target = self._path_for_file(file_id)
        except CollectionsCaptureError:
            return
        try:
            target = self._path_for_relative(
                file_id,
                str(row["relative_path"]),
            )
        except CollectionsCaptureError:
            if row["purge_state"] == "pending":
                target = safe_target
            elif str(row["state"]) in {
                "staging",
                "ready",
            }:
                self._best_effort_fail_row(row, "offline_integrity_failed")
                return
            else:
                return
        revision = int(row["revision"])

        if row["purge_state"] == "pending":
            try:
                unlink_private_file(
                    target,
                    application_owned_directory=self._file_root(file_id),
                )
            except (OSError, ValueError):
                return
            self._remove_temporaries(file_id)
            with self.repository.db.transaction() as connection:
                connection.execute(
                    "DELETE FROM collection_capture_offline_files "
                    "WHERE authority_key = ? AND file_id = ? AND revision = ? "
                    "AND EXISTS(SELECT 1 FROM collection_capture_items AS item "
                    "WHERE item.authority_key = collection_capture_offline_files.authority_key "
                    "AND item.capture_id = collection_capture_offline_files.capture_id "
                    "AND item.purge_state = 'pending')",
                    (self.repository.authority_key, file_id, revision),
                )
            self._remove_empty_file_root(file_id)
            return

        state = str(row["state"])
        if state == "purging":
            try:
                unlink_private_file(
                    target,
                    application_owned_directory=self._file_root(file_id),
                )
                self.repository.finish_offline_copy_purge(
                    identity,
                    file_id=file_id,
                    expected_revision=revision,
                )
            except CollectionsCaptureError:
                pass
            except (OSError, ValueError):
                pass
            self._remove_temporaries(file_id)
            self._remove_empty_file_root(file_id)
            return

        if state == "failed":
            self._best_effort_unlink(target)
            self._remove_temporaries(file_id)
            self._remove_empty_file_root(file_id)
            return

        try:
            payload = self._read_private_file(target)
        except FileNotFoundError:
            self._best_effort_fail_row(row, "offline_missing")
            self._remove_temporaries(file_id)
            self._remove_empty_file_root(file_id)
            return
        except (CollectionsCaptureError, PrivatePathError):
            self._best_effort_fail_row(row, "offline_integrity_failed")
            self._best_effort_unlink(target)
            return

        digest = self._digest(payload)
        if state == "staging":
            if (
                len(payload) != int(row["reserved_size"])
                or digest != row["content_hash"]
            ):
                self._best_effort_fail_row(row, "offline_integrity_failed")
                self._best_effort_unlink(target)
            else:
                try:
                    self.repository.complete_offline_copy(
                        identity,
                        file_id=file_id,
                        expected_revision=revision,
                        content_hash=digest,
                        actual_size=len(payload),
                        media_type=str(row["media_type"] or "application/octet-stream"),
                    )
                except CollectionsCaptureError:
                    pass
        elif (
            row["actual_size"] != len(payload)
            or row["content_hash"] != digest
        ):
            self._best_effort_fail_row(row, "offline_integrity_failed")
            self._best_effort_unlink(target)
        self._remove_temporaries(file_id)

    def _finish_capture_purge(self, capture_id: str) -> None:
        with self.repository.db.transaction() as connection:
            remaining = connection.execute(
                "SELECT 1 FROM collection_capture_offline_files "
                "WHERE authority_key = ? AND capture_id = ? LIMIT 1",
                (self.repository.authority_key, capture_id),
            ).fetchone()
            if remaining is not None:
                return
            connection.execute(
                "DELETE FROM collection_capture_items WHERE authority_key = ? "
                "AND capture_id = ? AND purge_state = 'pending'",
                (self.repository.authority_key, capture_id),
            )

    def _read_private_file(self, target: Path) -> bytes:
        with open_private_binary(target) as opened:
            payload = opened.stream.read(self.max_copy_bytes + 1)
        if len(payload) > self.max_copy_bytes:
            raise CollectionsCaptureError("offline_copy_too_large")
        return payload

    def _record_write_failure(self, copy: CaptureOfflineCopy) -> None:
        try:
            self.repository.fail_offline_copy(
                copy.identity,
                file_id=copy.file_id,
                expected_revision=copy.revision,
                reason="offline_write_failed",
            )
        except CollectionsCaptureError:
            pass

    def _best_effort_fail(self, copy: CaptureOfflineCopy, reason: str) -> None:
        try:
            self.repository.fail_offline_copy(
                copy.identity,
                file_id=copy.file_id,
                expected_revision=copy.revision,
                reason=reason,
            )
        except CollectionsCaptureError:
            pass

    def _best_effort_fail_row(self, row: sqlite3.Row, reason: str) -> None:
        self._best_effort_fail(
            CaptureOfflineCopy(
                CaptureIdentity(
                    self.repository.authority_key,
                    str(row["capture_id"]),
                ),
                str(row["file_id"]),
                str(row["state"]),
                content_hash=row["content_hash"],
                size=row["actual_size"],
                media_type=row["media_type"],
                failure_reason=row["failure_reason"],
                revision=int(row["revision"]),
            ),
            reason,
        )

    def _best_effort_unlink(self, target: Path) -> None:
        try:
            unlink_private_file(
                target,
                application_owned_directory=target.parent,
            )
        except (OSError, ValueError):
            pass

    def _remove_temporaries(self, file_id: str) -> None:
        file_root = self._file_root(file_id)
        prefix = f".{file_id}."
        try:
            secure_private_directory(
                file_root,
                create=False,
                application_owned=True,
            )
            # The atomic primitive can leave at most one sibling per staged
            # publication. The fixed cap also keeps tampered directories from
            # turning startup reconciliation into an unbounded scan.
            inspected = 0
            for entry in os.scandir(file_root):
                inspected += 1
                if entry.name.startswith(prefix) and entry.name.endswith(".tmp"):
                    self._best_effort_unlink(file_root / entry.name)
                if inspected >= 4:
                    return
        except OSError:
            return

    def _remove_empty_file_root(self, file_id: str) -> None:
        file_root = self._file_root(file_id)
        try:
            secure_private_directory(
                file_root,
                create=False,
                application_owned=True,
            )
            file_root.rmdir()
        except OSError:
            pass

    def _file_root(self, file_id: str) -> Path:
        if not _FILE_ID_PATTERN.fullmatch(file_id):
            raise CollectionsCaptureError("invalid_file_id")
        return self.authority_root / file_id

    def _path_for_copy(self, copy: CaptureOfflineCopy) -> Path:
        with self.repository.db.connection() as connection:
            row = connection.execute(
                "SELECT relative_path FROM collection_capture_offline_files "
                "WHERE authority_key = ? AND file_id = ? AND capture_id = ?",
                (
                    self.repository.authority_key,
                    copy.file_id,
                    copy.identity.capture_id,
                ),
            ).fetchone()
        if row is None:
            raise CollectionsCaptureError("offline_copy_not_found")
        return self._path_for_relative(copy.file_id, str(row["relative_path"]))

    def _path_for_relative(self, file_id: str, relative_path: str) -> Path:
        expected = f"{file_id}/{file_id}.bin"
        if relative_path != expected:
            raise CollectionsCaptureError("invalid_offline_relative_path")
        return self._path_for_file(file_id)

    def _path_for_file(self, file_id: str) -> Path:
        file_root = self._file_root(file_id)
        return file_root / f"{file_id}.bin"

    @staticmethod
    def _digest(payload: bytes) -> str:
        return f"sha256:{hashlib.sha256(payload).hexdigest()}"

    @staticmethod
    def _require_limit(value: Any, reason: str) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise CollectionsCaptureError(reason)
