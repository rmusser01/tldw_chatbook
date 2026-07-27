"""Cross-process shared and exclusive locking for the TTS profile store."""

from __future__ import annotations

import math
import time
from enum import Enum
from pathlib import Path
from typing import BinaryIO, cast

import portalocker

from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError


_CLEANUP_FAILURE_NOTE = "TTS profile store lease cleanup failed"
_PATH_TYPE = type(Path())


class ProfileStoreLockMode(str, Enum):
    """Supported profile-store lock modes."""

    SHARED = "shared"
    EXCLUSIVE = "exclusive"


def _normalize_timing(value: object) -> float | None:
    if type(value) not in (int, float):
        return None
    timing = cast(int | float, value)
    conversion_failed = False
    normalized = 0.0
    try:
        normalized = float(timing)
    except Exception:
        conversion_failed = True
    if conversion_failed or not math.isfinite(normalized) or normalized <= 0:
        return None
    return normalized


def _unlock_and_close(
    handle: BinaryIO,
    *,
    may_be_locked: bool,
) -> BaseException | None:
    """Best-effort clean one handle and return its first cleanup failure."""

    first_error: BaseException | None = None
    try:
        if may_be_locked:
            portalocker.unlock(handle)
    except BaseException as error:
        first_error = error
    finally:
        try:
            handle.close()
        except BaseException as error:
            # Control-flow cleanup errors outrank ordinary cleanup failures.
            # If both are control-flow exceptions, the first one remains primary.
            if first_error is None or (
                isinstance(first_error, Exception) and not isinstance(error, Exception)
            ):
                first_error = error
    return first_error


class ProfileStoreLease:
    """Own a shared or exclusive OS lock for one profile database.

    Construction only validates and canonicalizes the database path. The
    adjacent persistent lock file is opened when :meth:`acquire` runs.
    """

    def __init__(
        self,
        database_path: Path,
        mode: ProfileStoreLockMode,
        *,
        timeout_seconds: int | float = 5.0,
        check_interval_seconds: int | float = 0.05,
    ) -> None:
        """Initialize a synchronous profile-store lease.

        Args:
            database_path: Exact path of the profile database.
            mode: Shared or exclusive lock mode.
            timeout_seconds: Bounded acquisition timeout.
            check_interval_seconds: Maximum delay between lock attempts.

        Raises:
            ProfileRepositoryError: If any constructor input is invalid or the
                path cannot be canonicalized.
        """

        normalized_timeout = _normalize_timing(timeout_seconds)
        normalized_check_interval = _normalize_timing(check_interval_seconds)
        if (
            type(database_path) is not _PATH_TYPE
            or type(mode) is not ProfileStoreLockMode
            or normalized_timeout is None
            or normalized_check_interval is None
        ):
            raise ProfileRepositoryError("operation_failed")

        resolution_failed = False
        resolved_path: Path | None = None
        try:
            resolved_path = database_path.resolve()
        except Exception:
            resolution_failed = True
        if resolution_failed or resolved_path is None:
            raise ProfileRepositoryError("operation_failed")

        self._database_path = resolved_path
        self.mode = mode
        self.timeout_seconds = normalized_timeout
        self.check_interval_seconds = normalized_check_interval
        self._handle: BinaryIO | None = None

    @property
    def lock_path(self) -> Path:
        """Return the stable persistent lock path adjacent to the database."""

        return self._database_path.with_name(f"{self._database_path.name}.lock")

    @property
    def acquired(self) -> bool:
        """Return whether this lease currently owns the OS lock."""

        return self._handle is not None

    def acquire(self) -> ProfileStoreLease:
        """Synchronously acquire this lease and return it.

        Returns:
            This acquired lease.

        Raises:
            ProfileRepositoryError: If the lease is already acquired, times
                out, or cannot use the locking backend.
        """

        if self._handle is not None:
            raise ProfileRepositoryError("invalid_state")

        timing_failed = False
        deadline = 0.0
        try:
            deadline = time.monotonic() + self.timeout_seconds
        except Exception:
            timing_failed = True
        if timing_failed:
            raise ProfileRepositoryError("operation_failed")

        handle: BinaryIO | None = None
        may_be_locked = False
        transferred = False
        primary_error: BaseException | None = None
        try:
            open_failed = False
            try:
                handle = cast(BinaryIO, self.lock_path.open("a+b"))
            except Exception:
                open_failed = True
            if open_failed or handle is None:
                raise ProfileRepositoryError("operation_failed")

            flags = portalocker.LockFlags.NON_BLOCKING
            flags |= (
                portalocker.LockFlags.SHARED
                if self.mode is ProfileStoreLockMode.SHARED
                else portalocker.LockFlags.EXCLUSIVE
            )
            attempted = False
            while True:
                if attempted:
                    deadline_failed = False
                    deadline_reached = False
                    try:
                        deadline_reached = time.monotonic() >= deadline
                    except Exception:
                        deadline_failed = True
                    if deadline_failed:
                        raise ProfileRepositoryError("operation_failed")
                    if deadline_reached:
                        raise ProfileRepositoryError("lock_timeout")

                contended = False
                backend_failed = False
                may_be_locked = True
                try:
                    portalocker.lock(handle, flags)
                except portalocker.exceptions.AlreadyLocked:
                    may_be_locked = False
                    contended = True
                except Exception:
                    backend_failed = True
                if backend_failed:
                    raise ProfileRepositoryError("operation_failed")
                if not contended:
                    self._handle = handle
                    transferred = True
                    return self

                attempted = True
                clock_failed = False
                remaining = 0.0
                try:
                    remaining = deadline - time.monotonic()
                except Exception:
                    clock_failed = True
                if clock_failed:
                    raise ProfileRepositoryError("operation_failed")
                if remaining <= 0:
                    raise ProfileRepositoryError("lock_timeout")

                sleep_failed = False
                try:
                    time.sleep(min(self.check_interval_seconds, remaining))
                except Exception:
                    sleep_failed = True
                if sleep_failed:
                    raise ProfileRepositoryError("operation_failed")
        except BaseException as error:
            primary_error = error

        cleanup_error: BaseException | None = None
        state_error: BaseException | None = None
        if handle is not None and not transferred:
            cleanup_error = _unlock_and_close(
                handle,
                may_be_locked=may_be_locked,
            )
            if self._handle is handle:
                state_error = self._clear_handle_state()

        if primary_error is not None and not isinstance(primary_error, Exception):
            raise primary_error
        for candidate_error in (cleanup_error, state_error):
            if candidate_error is not None and not isinstance(
                candidate_error,
                Exception,
            ):
                raise candidate_error
        if isinstance(primary_error, ProfileRepositoryError):
            raise primary_error
        raise ProfileRepositoryError("operation_failed")

    def _clear_handle_state(self) -> BaseException | None:
        """Clear acquired state even if a subclass interrupts assignment."""

        try:
            self._handle = None
        except BaseException as error:
            object.__setattr__(self, "_handle", None)
            return error
        return None

    def release(self) -> None:
        """Synchronously unlock and close this lease idempotently.

        Raises:
            ProfileRepositoryError: If ordinary unlock or close cleanup fails.
            BaseException: A control-flow exception raised by cleanup is
                preserved after the remaining cleanup is attempted.
        """

        handle: BinaryIO | None = None
        primary_error: BaseException | None = None
        cleanup_error: BaseException | None = None
        cleanup_completed = False
        try:
            handle = self._handle
            if handle is None:
                return
            cleanup_error = _unlock_and_close(handle, may_be_locked=True)
            cleanup_completed = True
        except BaseException as error:
            primary_error = error

        retry_cleanup_error: BaseException | None = None
        if handle is not None and not cleanup_completed:
            retry_cleanup_error = _unlock_and_close(
                handle,
                may_be_locked=True,
            )
        state_error = self._clear_handle_state()

        errors = (
            primary_error,
            cleanup_error,
            retry_cleanup_error,
            state_error,
        )
        for candidate_error in errors:
            if candidate_error is not None and not isinstance(
                candidate_error,
                Exception,
            ):
                raise candidate_error
        if any(candidate_error is not None for candidate_error in errors):
            raise ProfileRepositoryError("operation_failed")

    def __enter__(self) -> ProfileStoreLease:
        """Acquire and return this lease for a context manager."""

        return self.acquire()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        """Release the lease while preserving a context-body exception."""

        try:
            self.release()
        except BaseException:
            if exc is None:
                raise
            try:
                BaseException.add_note(exc, _CLEANUP_FAILURE_NOTE)
            except BaseException:
                pass
