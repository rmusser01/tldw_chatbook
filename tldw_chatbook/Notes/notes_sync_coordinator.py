"""Cross-process ownership for lasting Database Notes roots."""

from __future__ import annotations

import hashlib
import inspect
import os
import stat
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from threading import RLock
from typing import IO, Literal

import portalocker

from tldw_chatbook.Notes.notes_sync_filesystem import PosixNotesSyncFilesystem
from tldw_chatbook.Utils.private_paths import (
    PrivatePathError,
    open_private_text_append_stream,
    secure_private_directory,
)
from tldw_chatbook.Utils.sensitive_paths import find_root_binding_conflict


_REPARSE_ATTRIBUTE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_AUTHORITY_OPERATIONS = frozenset({"watch", "plan", "write"})


class RootAdmissionState(StrEnum):
    """One bounded ownership result for a lasting root."""

    OWNER = "owner"
    PASSIVE = "passive"
    OFFLINE = "offline"
    REJECTED = "rejected"


class RootAdmissionError(OSError):
    """A privacy-safe candidate-root refusal."""

    def __init__(self, reason_code: str):
        self.reason_code = reason_code
        super().__init__(reason_code)


class RootAuthorityError(RuntimeError):
    """A process without an open OS lease attempted privileged work."""

    def __init__(self, reason_code: str):
        self.reason_code = reason_code
        super().__init__(reason_code)


class RootCoordinatorError(RuntimeError):
    """A privacy-safe lock or lifecycle failure."""

    def __init__(self, reason_code: str):
        self.reason_code = reason_code
        super().__init__(reason_code)


def _is_reparse(metadata: os.stat_result) -> bool:
    return bool(getattr(metadata, "st_file_attributes", 0) & _REPARSE_ATTRIBUTE)


def _overlaps(left: Path, right: Path) -> bool:
    try:
        if left.samefile(right):
            return True
        return any(parent.samefile(right) for parent in left.parents) or any(
            parent.samefile(left) for parent in right.parents
        )
    except OSError:
        raise RootAdmissionError("comparison_root_unavailable") from None


def _canonical_comparison_root(path: Path | str) -> Path:
    try:
        resolved = Path(path).expanduser().resolve(strict=True)
    except OSError:
        raise RootAdmissionError("comparison_root_unavailable") from None
    if not resolved.is_dir():
        raise RootAdmissionError("comparison_root_unavailable")
    return resolved


def validate_candidate_root(
    candidate: Path | str,
    *,
    lasting_roots: Iterable[Path | str] = (),
    file_notes_binding: object | None = None,
    write_supported: bool | None = None,
    private_conflict: Callable[[Path], Path | None] = find_root_binding_conflict,
) -> Path:
    """Return a canonical safe root or raise one bounded refusal."""

    selected = Path(candidate).expanduser()
    try:
        lexical = selected.lstat()
    except FileNotFoundError:
        raise RootAdmissionError("root_offline") from None
    except OSError:
        raise RootAdmissionError("root_unavailable") from None
    if stat.S_ISLNK(lexical.st_mode) or _is_reparse(lexical):
        raise RootAdmissionError("root_link_or_reparse")
    try:
        canonical = selected.resolve(strict=True)
    except OSError:
        raise RootAdmissionError("root_unavailable") from None
    if not canonical.is_dir():
        raise RootAdmissionError("root_not_directory")

    capability = (
        PosixNotesSyncFilesystem.supports_writes()
        if write_supported is None
        else write_supported
    )
    if type(capability) is not bool:
        raise TypeError("write_supported must be a boolean or None.")
    if not capability:
        raise RootAdmissionError("writable_filesystem_unsupported")

    for existing in lasting_roots:
        if _overlaps(canonical, _canonical_comparison_root(existing)):
            raise RootAdmissionError("lasting_root_overlap")

    if file_notes_binding is not None:
        root_key = getattr(file_notes_binding, "root_key", None)
        if type(root_key) is not str or not root_key:
            raise TypeError("file_notes_binding must expose a non-empty root_key.")
        if _overlaps(canonical, _canonical_comparison_root(root_key)):
            raise RootAdmissionError("file_notes_overlap")

    try:
        conflict = private_conflict(canonical)
    except Exception:
        raise RootAdmissionError("private_path_check_failed") from None
    if conflict is not None:
        raise RootAdmissionError("private_path_overlap")
    return canonical


class RootLease:
    """Private open handle whose OS lock grants root authority."""

    __slots__ = (
        "root_digest",
        "_admission_open",
        "_handle",
        "_lifecycle_lock",
        "_owner_token",
    )

    def __init__(self, root_digest: str, handle: IO[str], owner_token: object):
        self.root_digest = root_digest
        self._handle: IO[str] | None = handle
        self._owner_token = owner_token
        self._admission_open = True
        self._lifecycle_lock = RLock()

    def __repr__(self) -> str:
        return "RootLease(<private>)"

    @property
    def authoritative(self) -> bool:
        """Whether this lease can admit new privileged work."""

        with self._lifecycle_lock:
            return self._handle is not None and self._admission_open

    def _close_admission(self) -> None:
        with self._lifecycle_lock:
            self._admission_open = False

    def _take_handle(self) -> IO[str] | None:
        with self._lifecycle_lock:
            handle = self._handle
            self._handle = None
            self._admission_open = False
            return handle


@dataclass(frozen=True, slots=True, repr=False)
class RootAdmission:
    """Path-free projection of one coordinator admission attempt."""

    state: RootAdmissionState
    root_digest: str
    reason_code: str | None = None
    lease: RootLease | None = None

    def __post_init__(self) -> None:
        if type(self.state) is not RootAdmissionState:
            raise TypeError("state must be a RootAdmissionState.")
        if (
            type(self.root_digest) is not str
            or len(self.root_digest) != 64
            or any(
                character not in "0123456789abcdef" for character in self.root_digest
            )
        ):
            raise ValueError("root_digest must be a lowercase SHA-256 digest.")
        if self.state is RootAdmissionState.OWNER:
            if type(self.lease) is not RootLease or self.reason_code is not None:
                raise ValueError("owner admission requires only an acquired lease.")
        elif self.lease is not None:
            raise ValueError("non-owner admission cannot carry a lease.")

    def __repr__(self) -> str:
        return f"RootAdmission(state={self.state.value!r})"

    @property
    def label(self) -> str:
        return {
            RootAdmissionState.OWNER: "Active in this process",
            RootAdmissionState.PASSIVE: "Passive in this process",
            RootAdmissionState.OFFLINE: "Offline",
            RootAdmissionState.REJECTED: "Unsupported",
        }[self.state]

    @property
    def can_watch(self) -> bool:
        return self._has_authority()

    @property
    def can_plan(self) -> bool:
        return self._has_authority()

    @property
    def can_write(self) -> bool:
        return self._has_authority()

    def _has_authority(self) -> bool:
        return (
            self.state is RootAdmissionState.OWNER
            and self.lease is not None
            and self.lease.authoritative
        )

    def require_authority(
        self,
        operation: Literal["watch", "plan", "write"],
    ) -> RootLease:
        """Return the authoritative lease or refuse this operation."""

        if operation not in _AUTHORITY_OPERATIONS:
            raise ValueError("operation must be watch, plan, or write.")
        if self._has_authority():
            assert self.lease is not None
            return self.lease
        reason = (
            "passive_process"
            if self.state is RootAdmissionState.PASSIVE
            else "admission_closed"
        )
        raise RootAuthorityError(reason)


class NotesSyncRootCoordinator:
    """Own nonblocking exclusive OS locks for lasting roots."""

    def __init__(self, lock_directory: Path | str):
        try:
            result = secure_private_directory(
                lock_directory,
                create=True,
                application_owned=True,
            )
        except PrivatePathError:
            # A second process may win the same mkdir race. Re-verify the
            # resulting directory instead of treating safe concurrency as loss.
            try:
                result = secure_private_directory(
                    lock_directory,
                    create=False,
                    application_owned=True,
                )
            except (OSError, ValueError):
                raise RootCoordinatorError("lock_directory_unavailable") from None
        except (OSError, ValueError):
            raise RootCoordinatorError("lock_directory_unavailable") from None
        if not result.usable:
            raise RootCoordinatorError("lock_directory_unavailable")
        self._lock_directory = result.lexical_path
        self._owner_token = object()
        self._leases: dict[str, RootLease] = {}
        self._lifecycle_lock = RLock()

    @staticmethod
    def _digest(canonical_root: Path) -> str:
        try:
            identity = canonical_root.stat()
        except OSError:
            raise RootAdmissionError("root_unavailable") from None
        payload = f"{identity.st_dev}\0{identity.st_ino}".encode("ascii")
        return hashlib.sha256(payload).hexdigest()

    def try_acquire(
        self,
        candidate: Path | str,
        **validation: object,
    ) -> RootAdmission:
        """Try once for root authority; contention becomes passive state."""

        try:
            canonical = validate_candidate_root(candidate, **validation)
        except RootAdmissionError as exc:
            selected = os.fsencode(Path(candidate).expanduser())
            digest = hashlib.sha256(selected).hexdigest()
            state = (
                RootAdmissionState.OFFLINE
                if exc.reason_code == "root_offline"
                else RootAdmissionState.REJECTED
            )
            return RootAdmission(state, digest, exc.reason_code)

        try:
            digest = self._digest(canonical)
        except RootAdmissionError as exc:
            digest = hashlib.sha256(os.fsencode(canonical)).hexdigest()
            return RootAdmission(
                RootAdmissionState.OFFLINE,
                digest,
                exc.reason_code,
            )
        with self._lifecycle_lock:
            existing = self._leases.get(digest)
            if existing is not None and existing.authoritative:
                return RootAdmission(RootAdmissionState.OWNER, digest, lease=existing)
            lock_path = self._lock_directory / f"{digest}.lock"
            try:
                handle = open_private_text_append_stream(
                    lock_path,
                    application_owned_directory=self._lock_directory,
                )
            except PrivatePathError:
                # The other contender may have created the same lock file
                # between our no-follow stat and open. Re-run all verification.
                try:
                    handle = open_private_text_append_stream(
                        lock_path,
                        application_owned_directory=self._lock_directory,
                    )
                except (OSError, ValueError, PrivatePathError):
                    return RootAdmission(
                        RootAdmissionState.REJECTED,
                        digest,
                        "lock_unavailable",
                    )
            except (OSError, ValueError):
                return RootAdmission(
                    RootAdmissionState.REJECTED,
                    digest,
                    "lock_unavailable",
                )
            flags = portalocker.LockFlags.EXCLUSIVE | portalocker.LockFlags.NON_BLOCKING
            try:
                portalocker.lock(handle, flags)
            except portalocker.exceptions.AlreadyLocked:
                handle.close()
                return RootAdmission(
                    RootAdmissionState.PASSIVE,
                    digest,
                    "passive_process",
                )
            except Exception:
                handle.close()
                return RootAdmission(
                    RootAdmissionState.REJECTED,
                    digest,
                    "lock_unavailable",
                )
            lease = RootLease(digest, handle, self._owner_token)
            self._leases[digest] = lease
            return RootAdmission(RootAdmissionState.OWNER, digest, lease=lease)

    def _validate_lease(self, lease: RootLease | None) -> RootLease:
        if type(lease) is not RootLease or lease._owner_token is not self._owner_token:
            raise RootCoordinatorError("foreign_lease")
        return lease

    def release(self, lease: RootLease | None) -> None:
        """Release one owned OS lock idempotently without unlinking its file."""

        selected = self._validate_lease(lease)
        with self._lifecycle_lock:
            handle = selected._take_handle()
            if handle is None:
                return
            try:
                portalocker.unlock(handle)
            except Exception:
                selected._handle = handle
                raise RootCoordinatorError("lock_release_failed") from None
            self._leases.pop(selected.root_digest, None)
            try:
                handle.close()
            except OSError:
                raise RootCoordinatorError("lock_close_failed") from None

    def close_admission(
        self,
        lease: RootLease | None,
        settle: Callable[[], object],
    ) -> None:
        """Close new work, settle admitted work, then release OS authority."""

        selected = self._validate_lease(lease)
        if not callable(settle):
            raise TypeError("settle must be callable.")
        selected._close_admission()
        try:
            settlement = settle()
        except BaseException:
            raise RootCoordinatorError("settlement_failed") from None
        if inspect.isawaitable(settlement):
            close = getattr(settlement, "close", None)
            if callable(close):
                close()
            raise RootCoordinatorError("settlement_not_completed")
        self.release(selected)


__all__ = [
    "NotesSyncRootCoordinator",
    "RootAdmission",
    "RootAdmissionError",
    "RootAdmissionState",
    "RootAuthorityError",
    "RootCoordinatorError",
    "RootLease",
    "validate_candidate_root",
]
