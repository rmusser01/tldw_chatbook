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


def _close_after_failed_acquire(handle: BinaryIO) -> BaseException | None:
    """Close an unowned handle, returning only a control-flow exception."""

    try:
        handle.close()
    except Exception:
        return None
    except BaseException as error:
        return error
    return None


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

        open_failed = False
        handle: BinaryIO | None = None
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
            deadline_failed = False
            deadline_reached = False
            deadline_control_error: BaseException | None = None
            if attempted:
                try:
                    deadline_reached = time.monotonic() >= deadline
                except Exception:
                    deadline_failed = True
                except BaseException as error:
                    deadline_control_error = error
            if deadline_control_error is not None:
                _close_after_failed_acquire(handle)
                raise deadline_control_error
            if deadline_failed:
                self._fail_acquire(handle, "operation_failed")
            if deadline_reached:
                self._fail_acquire(handle, "lock_timeout")

            contended = False
            backend_failed = False
            control_error: BaseException | None = None
            try:
                portalocker.lock(handle, flags)
            except portalocker.exceptions.AlreadyLocked:
                contended = True
            except Exception:
                backend_failed = True
            except BaseException as error:
                control_error = error

            if control_error is not None:
                _close_after_failed_acquire(handle)
                raise control_error
            if backend_failed:
                self._fail_acquire(handle, "operation_failed")
            if not contended:
                self._handle = handle
                return self

            attempted = True
            clock_failed = False
            clock_control_error: BaseException | None = None
            remaining = 0.0
            try:
                remaining = deadline - time.monotonic()
            except Exception:
                clock_failed = True
            except BaseException as error:
                clock_control_error = error
            if clock_control_error is not None:
                _close_after_failed_acquire(handle)
                raise clock_control_error
            if clock_failed:
                self._fail_acquire(handle, "operation_failed")
            if remaining <= 0:
                self._fail_acquire(handle, "lock_timeout")

            sleep_failed = False
            sleep_control_error: BaseException | None = None
            try:
                time.sleep(min(self.check_interval_seconds, remaining))
            except Exception:
                sleep_failed = True
            except BaseException as error:
                sleep_control_error = error
            if sleep_control_error is not None:
                _close_after_failed_acquire(handle)
                raise sleep_control_error
            if sleep_failed:
                self._fail_acquire(handle, "operation_failed")

    @staticmethod
    def _fail_acquire(handle: BinaryIO, code: str) -> None:
        control_error = _close_after_failed_acquire(handle)
        if control_error is not None:
            raise control_error
        raise ProfileRepositoryError(code)

    def release(self) -> None:
        """Synchronously unlock and close this lease idempotently.

        Raises:
            ProfileRepositoryError: If ordinary unlock or close cleanup fails.
            BaseException: A control-flow exception raised by cleanup is
                preserved after the remaining cleanup is attempted.
        """

        handle = self._handle
        self._handle = None
        if handle is None:
            return

        unlock_failed = False
        unlock_control_error: BaseException | None = None
        try:
            portalocker.unlock(handle)
        except Exception:
            unlock_failed = True
        except BaseException as error:
            unlock_control_error = error

        close_failed = False
        close_control_error: BaseException | None = None
        try:
            handle.close()
        except Exception:
            close_failed = True
        except BaseException as error:
            close_control_error = error

        if unlock_control_error is not None:
            raise unlock_control_error
        if close_control_error is not None:
            raise close_control_error
        if unlock_failed or close_failed:
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
            exc.add_note(_CLEANUP_FAILURE_NOTE)
