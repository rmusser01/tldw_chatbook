"""Child-owned original-inode proof for one metadata-only exact TTS store.

No live repository opens this owner locally. Its immutable SQLite evidence view
closes before raw pins; only the exec child retains original file descriptors.
"""

from __future__ import annotations

import os
import sqlite3
import stat
import time
from collections.abc import Callable
from pathlib import Path

from tldw_chatbook.DB.private_sqlite_files import _open_artifact_fd
from tldw_chatbook.DB.private_sqlite_protocol import FileIdentity, TTSRestoreAuthority
from tldw_chatbook.TTS.profile_migration_journal import (
    MAX_PROFILE_MIGRATION_ARTIFACT_BYTES,
)
from tldw_chatbook.TTS.profile_validation import (
    CURRENT_PROFILE_SCHEMA_VERSION,
    _configure_connection,
    _run_with_deadline_progress,
    _stream_exact_store_metadata_evidence,
    _validate_schema,
    validate_profile_store_rows,
)
from tldw_chatbook.Utils import private_paths


class TTSProofError(RuntimeError):
    """Fixed semantic refusal, without source or exception details."""

    def __init__(self, reason: str = "operation_failed") -> None:
        self.reason = reason
        super().__init__("private_tts_proof_refused")


class TTSProofTimeout(BaseException):
    """Internal deadline signal that survives existing validator normalization."""


class TTSProof:
    """Own one original parent/main/cohort for the lifetime of an exec child."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.parent_fd = self.file_fd = -1
        self.sidecars: dict[str, int] = {}
        self.identities: dict[str, FileIdentity] = {}
        self.evidence: sqlite3.Connection | None = None
        self.initialized = False
        self.started = False

    def initialize(
        self, *, check_deadline: Callable[[], None] | None = None
    ) -> dict[str, object]:
        """Validate one immutable original descriptor; retain identity pins only."""
        if self.started:
            raise TTSProofError()
        self.started = True
        if (
            os.name != "posix"
            or not self.path.is_absolute()
            or not hasattr(os, "O_NOFOLLOW")
        ):
            raise TTSProofError()
        expires = time.monotonic() + 30.0

        def check() -> None:
            if time.monotonic() >= expires:
                raise TTSProofTimeout()
            if check_deadline is not None:
                check_deadline()

        check()
        self.parent_fd, leaf = private_paths._open_verified_parent(
            self.path, missing_leaf_allowed=False
        )
        self.identities["parent"] = FileIdentity.from_stat(os.fstat(self.parent_fd))
        self._namespace("exact_not_current")
        self.file_fd = _open_artifact_fd(
            self.parent_fd, leaf, writable=False, create=False
        )
        self.identities["main"] = self._file_identity(
            self.file_fd, leaf, "exact_not_current"
        )
        self._capture_sidecars(require=False)
        self.evidence = sqlite3.connect(
            f"file:/dev/fd/{self.file_fd}?mode=ro&immutable=1",
            uri=True,
            isolation_level=None,
        )
        try:
            _configure_connection(self.evidence)
            if (
                self.evidence.execute("PRAGMA user_version").fetchone()[0]
                != CURRENT_PROFILE_SCHEMA_VERSION
            ):
                raise TTSProofError("exact_not_current")
            _validate_schema(self.evidence, check_deadline=check)
            validate_profile_store_rows(self.evidence, check_deadline=check)
            _run_with_deadline_progress(
                self.evidence,
                check,
                lambda: _stream_exact_store_metadata_evidence(self.evidence),
            )
        except BaseException as primary:
            try:
                self._close_evidence()
            except BaseException:
                if not isinstance(primary, Exception):
                    primary.add_note("private_tts_evidence_cleanup_failed")
                    raise primary from None
                raise
            raise
        else:
            self._close_evidence()
        check()
        self.initialized = True
        return self.recheck()

    def _namespace(self, reason: str = "operation_failed") -> None:
        # Preserve the existing exact-opener preflight distinction: statically
        # unsafe sidecars require exclusive initialization; a capture race or
        # a changed retained cohort is an operation/authority failure.
        for suffix in ("wal", "shm"):
            try:
                observed = os.stat(
                    f"{self.path.name}-{suffix}",
                    dir_fd=self.parent_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                continue
            if (
                private_paths._classify_private_file_stat(
                    observed, expected_uid=os.geteuid()
                )
                is not None
                or stat.S_IMODE(observed.st_mode) != 0o600
            ):
                raise TTSProofError(reason)
        publication = f".{self.path.name}.migration-publication.json"
        for leaf in (
            self.path.name + "-journal",
            *(publication + suffix for suffix in ("", "-wal", "-shm", "-journal")),
        ):
            try:
                os.stat(leaf, dir_fd=self.parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            raise TTSProofError(reason)

    def _file_identity(
        self, fd: int, leaf: str, reason: str = "operation_failed"
    ) -> FileIdentity:
        opened = os.fstat(fd)
        named = os.stat(leaf, dir_fd=self.parent_fd, follow_symlinks=False)
        for observed in (opened, named):
            if (
                private_paths._classify_private_file_stat(
                    observed, expected_uid=os.geteuid()
                )
                is not None
                or stat.S_IMODE(observed.st_mode) != 0o600
                or observed.st_size > MAX_PROFILE_MIGRATION_ARTIFACT_BYTES
            ):
                raise TTSProofError(reason)
        identity = FileIdentity.from_stat(opened)
        if not identity.same_inode(FileIdentity.from_stat(named)):
            raise TTSProofError()
        return identity

    def _capture_sidecars(self, *, require: bool) -> None:
        if self.sidecars:
            self.recheck()
            return
        for suffix in ("wal", "shm"):
            leaf = f"{self.path.name}-{suffix}"
            try:
                fd = _open_artifact_fd(
                    self.parent_fd, leaf, writable=False, create=False
                )
            except FileNotFoundError:
                continue
            self.sidecars[suffix] = fd
            self.identities[suffix] = self._file_identity(fd, leaf)
        if len(self.sidecars) == 1 or (require and not self.sidecars):
            raise TTSProofError("operation_failed" if require else "exact_not_current")

    def recheck(self) -> dict[str, object]:
        """Compare original pins, private namespace, and exact parent authority."""
        if not self.initialized:
            raise TTSProofError()
        other_parent = -1
        try:
            other_parent, leaf = private_paths._open_verified_parent(
                self.path, missing_leaf_allowed=False
            )
            for fd in (self.parent_fd, other_parent):
                observed = FileIdentity.from_stat(os.fstat(fd))
                original = self.identities["parent"]
                if not observed.same_inode(original) or (
                    observed.mode,
                    observed.uid,
                    observed.gid,
                ) != (original.mode, original.uid, original.gid):
                    raise TTSProofError()
            if leaf != self.path.name:
                raise TTSProofError()
            main = self._file_identity(self.file_fd, leaf)
            if not main.same_inode(self.identities["main"]):
                raise TTSProofError()
            current_sidecars: dict[str, FileIdentity] = {}
            if self.sidecars:
                if set(self.sidecars) != {"wal", "shm"}:
                    raise TTSProofError()
                for suffix, fd in self.sidecars.items():
                    observed = self._file_identity(fd, f"{leaf}-{suffix}")
                    if not observed.same_inode(self.identities[suffix]):
                        raise TTSProofError()
                    current_sidecars[suffix] = observed
            else:
                for suffix in ("wal", "shm"):
                    try:
                        os.stat(
                            f"{leaf}-{suffix}",
                            dir_fd=self.parent_fd,
                            follow_symlinks=False,
                        )
                    except FileNotFoundError:
                        continue
                    raise TTSProofError()
            self._namespace()
            return {
                "parent": self.identities["parent"].to_payload(),
                "main": main.to_payload(),
                "wal": current_sidecars["wal"].to_payload() if self.sidecars else None,
                "shm": current_sidecars["shm"].to_payload() if self.sidecars else None,
            }
        except OSError:
            raise TTSProofError() from None
        finally:
            if other_parent >= 0:
                os.close(other_parent)

    def pin_sidecars(self) -> dict[str, object]:
        """Bind the first complete original WAL/SHM cohort, never a replacement."""
        if not self.initialized:
            raise TTSProofError()
        self._capture_sidecars(require=True)
        return self.recheck()

    def export_restore_authority(self) -> TTSRestoreAuthority:
        """Revalidate complete exact authority; generation is attached later."""
        identity = self.recheck()
        if not self.sidecars:
            raise TTSProofError()
        return TTSRestoreAuthority.from_payload(identity)

    def close(self) -> None:
        """Release SQL before its original raw pins; failure retains ownership."""
        self._close_evidence()
        for suffix, fd in list(self.sidecars.items()):
            os.close(fd)
            del self.sidecars[suffix]
        if self.file_fd >= 0:
            os.close(self.file_fd)
            self.file_fd = -1
        if self.parent_fd >= 0:
            os.close(self.parent_fd)
            self.parent_fd = -1
        self.initialized = False

    def _close_evidence(self) -> None:
        if self.evidence is not None:
            # A failed close leaves both this handle and all raw pins owned.
            self.evidence.close()
            self.evidence = None
