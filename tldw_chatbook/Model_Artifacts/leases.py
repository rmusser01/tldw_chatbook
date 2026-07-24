"""Cross-platform operation leases for immutable model artifacts."""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import BinaryIO

import portalocker


_IDENTITY_SEPARATOR = "\x1f"


def _validate_timing(
    timeout_seconds: float,
    check_interval_seconds: float,
) -> None:
    if not math.isfinite(timeout_seconds) or timeout_seconds < 0:
        raise ValueError("timeout_seconds must be finite and nonnegative")
    if not math.isfinite(check_interval_seconds) or check_interval_seconds <= 0:
        raise ValueError("check_interval_seconds must be finite and positive")


def _release_for_context(
    release: Callable[[], None],
    body_error: BaseException | None,
) -> None:
    try:
        release()
    except BaseException as cleanup_error:
        if body_error is None:
            raise
        body_error.add_note(f"lease context cleanup failed: {cleanup_error!r}")
        for note in getattr(cleanup_error, "__notes__", ()):
            body_error.add_note(note)


def _close_handle(
    handle: BinaryIO,
    *,
    primary_error: BaseException | None,
    failure_message: str,
) -> None:
    try:
        handle.close()
    except Exception as close_error:
        if primary_error is None:
            raise ArtifactLeaseError(failure_message) from close_error
        primary_error.add_note(f"{failure_message}: {close_error!r}")


class ArtifactLeaseError(RuntimeError):
    """Base error for model-artifact lease operations."""


class ArtifactLeaseTimeoutError(ArtifactLeaseError):
    """Raised when an incompatible lease remains held past the deadline."""


class ArtifactLeaseCancelledError(ArtifactLeaseError):
    """Raised when the caller cancels while waiting for a lease."""


class LeaseMode(str, Enum):
    """Supported interprocess lease modes."""

    SHARED = "shared"
    EXCLUSIVE = "exclusive"


def _validate_mode(mode: LeaseMode) -> None:
    if not isinstance(mode, LeaseMode):
        raise TypeError("mode must be a LeaseMode")


@dataclass(frozen=True, order=True)
class ArtifactLeaseKey:
    """Opaque identity of one immutable artifact version."""

    artifact_id: str
    revision: str
    variant: str

    def __post_init__(self) -> None:
        for field_name, value in (
            ("artifact_id", self.artifact_id),
            ("revision", self.revision),
            ("variant", self.variant),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
            if _IDENTITY_SEPARATOR in value:
                raise ValueError(f"{field_name} contains the reserved separator")

    @property
    def canonical_identity(self) -> str:
        """Return the unambiguous identity used only as hash input."""

        return _IDENTITY_SEPARATOR.join((self.artifact_id, self.revision, self.variant))


class ArtifactOperationLease:
    """Own one shared or exclusive OS-backed artifact lock."""

    def __init__(
        self,
        lock_root: Path,
        key: ArtifactLeaseKey,
        mode: LeaseMode,
        *,
        timeout_seconds: float = 5.0,
        check_interval_seconds: float = 0.05,
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        _validate_mode(mode)
        _validate_timing(timeout_seconds, check_interval_seconds)
        self._lock_root = Path(lock_root)
        self.key = key
        self.mode = mode
        self.timeout_seconds = timeout_seconds
        self.check_interval_seconds = check_interval_seconds
        self._cancelled = cancelled
        self._handle: BinaryIO | None = None

    @property
    def lock_path(self) -> Path:
        """Return the contained lock-file path for this opaque identity."""

        digest = hashlib.sha256(self.key.canonical_identity.encode("utf-8")).hexdigest()
        return self._lock_root / f"{digest}.lock"

    @property
    def acquired(self) -> bool:
        """Return whether this object currently owns its OS-backed lock."""

        return self._handle is not None

    def acquire(self) -> ArtifactOperationLease:
        """Acquire the lease or raise a stable timeout/cancellation error."""

        return self._acquire_until(
            time.monotonic() + self.timeout_seconds,
            allow_initial_attempt=True,
        )

    def _acquire_until(
        self,
        deadline: float,
        *,
        allow_initial_attempt: bool,
    ) -> ArtifactOperationLease:
        if self._handle is not None:
            raise ArtifactLeaseError("lease is already acquired")

        try:
            self._lock_root.mkdir(parents=True, exist_ok=True)
            handle = self.lock_path.open("a+b")
        except OSError as error:
            raise ArtifactLeaseError(
                f"failed preparing {self.mode.value} lease for {self.key.artifact_id}"
            ) from error
        flags = portalocker.LockFlags.NON_BLOCKING
        flags |= (
            portalocker.LockFlags.SHARED
            if self.mode is LeaseMode.SHARED
            else portalocker.LockFlags.EXCLUSIVE
        )
        attempted = False
        last_contention: BaseException | None = None

        try:
            while True:
                if self._cancelled is not None and self._cancelled():
                    raise ArtifactLeaseCancelledError(
                        f"lease acquisition cancelled for {self.key.artifact_id}"
                    )
                if (attempted or not allow_initial_attempt) and (
                    time.monotonic() >= deadline
                ):
                    timeout_error = ArtifactLeaseTimeoutError(
                        f"timed out acquiring {self.mode.value} lease "
                        f"for {self.key.artifact_id}"
                    )
                    if last_contention is None:
                        raise timeout_error
                    raise timeout_error from last_contention
                try:
                    portalocker.lock(handle, flags)
                except portalocker.exceptions.AlreadyLocked as exc:
                    attempted = True
                    last_contention = exc
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise ArtifactLeaseTimeoutError(
                            f"timed out acquiring {self.mode.value} lease "
                            f"for {self.key.artifact_id}"
                        ) from exc
                    time.sleep(min(self.check_interval_seconds, remaining))
                    continue
                except portalocker.exceptions.LockException as exc:
                    raise ArtifactLeaseError(
                        f"failed acquiring {self.mode.value} lease "
                        f"for {self.key.artifact_id}"
                    ) from exc
                except Exception as exc:
                    raise ArtifactLeaseError(
                        f"failed acquiring {self.mode.value} lease "
                        f"for {self.key.artifact_id}"
                    ) from exc
                self._handle = handle
                return self
        except BaseException as acquisition_error:
            _close_handle(
                handle,
                primary_error=acquisition_error,
                failure_message=(
                    f"failed closing {self.mode.value} lease for {self.key.artifact_id}"
                ),
            )
            raise

    def release(self) -> None:
        """Release the OS lock and close its owning handle idempotently."""

        handle = self._handle
        self._handle = None
        if handle is None:
            return
        try:
            portalocker.unlock(handle)
        except Exception as error:
            release_error = ArtifactLeaseError(
                f"failed releasing {self.mode.value} lease for {self.key.artifact_id}"
            )
            _close_handle(
                handle,
                primary_error=release_error,
                failure_message=(
                    f"failed closing {self.mode.value} lease for {self.key.artifact_id}"
                ),
            )
            raise release_error from error
        except BaseException as error:
            _close_handle(
                handle,
                primary_error=error,
                failure_message=(
                    f"failed closing {self.mode.value} lease for {self.key.artifact_id}"
                ),
            )
            raise
        _close_handle(
            handle,
            primary_error=None,
            failure_message=(
                f"failed closing {self.mode.value} lease for {self.key.artifact_id}"
            ),
        )

    def __enter__(self) -> ArtifactOperationLease:
        return self.acquire()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        _release_for_context(self.release, exc)


class ArtifactOperationLeaseSet:
    """Acquire one mode over a canonical immutable artifact closure."""

    def __init__(
        self,
        lock_root: Path,
        keys: list[ArtifactLeaseKey] | tuple[ArtifactLeaseKey, ...],
        mode: LeaseMode,
        *,
        timeout_seconds: float = 5.0,
        check_interval_seconds: float = 0.05,
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        _validate_mode(mode)
        ordered_keys = tuple(sorted(set(keys)))
        if not ordered_keys:
            raise ValueError("lease set requires at least one artifact key")
        _validate_timing(timeout_seconds, check_interval_seconds)
        self._lock_root = Path(lock_root)
        self.keys = ordered_keys
        self.mode = mode
        self.timeout_seconds = timeout_seconds
        self.check_interval_seconds = check_interval_seconds
        self._cancelled = cancelled
        self._leases: list[ArtifactOperationLease] = []

    @property
    def acquired(self) -> bool:
        """Return whether the complete key set is currently held."""

        return len(self._leases) == len(self.keys)

    def acquire(self) -> ArtifactOperationLeaseSet:
        """Acquire every key in order under one total timeout budget."""

        if self._leases:
            raise ArtifactLeaseError("lease set is already acquired")
        deadline = time.monotonic() + self.timeout_seconds
        try:
            for index, key in enumerate(self.keys):
                if index > 0 and time.monotonic() >= deadline:
                    raise ArtifactLeaseTimeoutError(
                        f"timed out acquiring {self.mode.value} lease "
                        f"for {key.artifact_id}"
                    )
                lease = ArtifactOperationLease(
                    self._lock_root,
                    key,
                    self.mode,
                    timeout_seconds=self.timeout_seconds,
                    check_interval_seconds=self.check_interval_seconds,
                    cancelled=self._cancelled,
                )
                lease._acquire_until(
                    deadline,
                    allow_initial_attempt=index == 0,
                )
                self._leases.append(lease)
        except BaseException as acquisition_error:
            try:
                self.release()
            except BaseException as cleanup_error:
                acquisition_error.add_note(
                    f"lease rollback cleanup failed: {cleanup_error!r}"
                )
                for note in getattr(cleanup_error, "__notes__", ()):
                    acquisition_error.add_note(note)
            raise
        return self

    def release(self) -> None:
        """Release all acquired keys in reverse order."""

        first_error: BaseException | None = None
        while self._leases:
            try:
                self._leases.pop().release()
            except BaseException as error:
                if first_error is None:
                    first_error = error
                else:
                    first_error.add_note(f"additional lease release failure: {error!r}")
        if first_error is not None:
            raise first_error

    def __enter__(self) -> ArtifactOperationLeaseSet:
        return self.acquire()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        _release_for_context(self.release, exc)
