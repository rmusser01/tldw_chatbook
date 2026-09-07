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


def _residual_target_is_replaceable(
    current_handle: BinaryIO | None,
    residual_handle: BinaryIO,
) -> bool:
    """Return whether a live residual may replace the current handle state."""

    if current_handle is None or current_handle is residual_handle:
        return True
    return current_handle.closed


def _handle_requires_cleanup(handle: BinaryIO | None) -> bool:
    """Conservatively report whether a represented handle may still be live."""

    if handle is None:
        return False
    try:
        return not handle.closed
    except BaseException:
        return True


def _transition_handle_requires_cleanup(handle: BinaryIO) -> bool:
    """Inspect transition liveness while preserving control-flow exceptions."""

    inspection_failed = False
    handle_closed = False
    try:
        handle_closed = handle.closed
    except Exception:
        inspection_failed = True
    if inspection_failed:
        raise ProfileRepositoryError("operation_failed")
    return not handle_closed


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


def _raise_recovery_failure(
    primary_error: BaseException | None,
    *cleanup_errors: BaseException | None,
) -> None:
    """Apply stable error precedence after best-effort recovery."""

    if primary_error is not None and not isinstance(primary_error, Exception):
        raise primary_error
    for cleanup_error in cleanup_errors:
        if cleanup_error is not None and not isinstance(cleanup_error, Exception):
            raise cleanup_error
    if isinstance(primary_error, ProfileRepositoryError):
        raise primary_error
    if primary_error is not None or any(
        cleanup_error is not None for cleanup_error in cleanup_errors
    ):
        raise ProfileRepositoryError("operation_failed")


class ProfileStoreLease:
    """Own a shared or exclusive OS lock for one profile database.

    Construction only validates and canonicalizes the database path. The
    adjacent persistent lock file is opened when :meth:`acquire` runs.

    Lease instances are synchronous and intentionally not thread-safe. One
    repository worker must own each instance and serialize its method calls.
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
        self._residual_handle: BinaryIO | None = None

    @property
    def lock_path(self) -> Path:
        """Return the stable persistent lock path adjacent to the database."""

        return self._database_path.with_name(f"{self._database_path.name}.lock")

    @property
    def acquired(self) -> bool:
        """Return whether any represented handle may still require cleanup."""

        handle = object.__getattribute__(self, "_handle")
        residual_handle = object.__getattribute__(self, "_residual_handle")
        return _handle_requires_cleanup(handle) or _handle_requires_cleanup(
            residual_handle
        )

    def acquire(self) -> ProfileStoreLease:
        """Synchronously acquire this lease and return it.

        Returns:
            This acquired lease.

        Raises:
            ProfileRepositoryError: If the lease is already acquired, times
                out, or cannot use the locking backend.
        """

        existing_handles = (
            object.__getattribute__(self, "_handle"),
            object.__getattribute__(self, "_residual_handle"),
        )
        normalized_ids: set[int] = set()
        for existing_handle in existing_handles:
            if existing_handle is None or id(existing_handle) in normalized_ids:
                continue
            normalized_ids.add(id(existing_handle))
            if _transition_handle_requires_cleanup(existing_handle):
                raise ProfileRepositoryError("invalid_state")
            normalization_error = self._clear_handle_state(existing_handle)
            _raise_recovery_failure(None, normalization_error)

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
        primary_error: BaseException | None = None
        recovery_errors: list[BaseException] = []
        try:
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

            if handle is not None:
                self._recover_acquisition_with_replay(
                    handle,
                    may_be_locked=may_be_locked,
                    errors=recovery_errors,
                )
        except BaseException as error:
            recovery_errors.append(error)
            if handle is not None:
                self._recover_acquisition_handle(
                    handle,
                    may_be_locked=may_be_locked,
                    errors=recovery_errors,
                )

        _raise_recovery_failure(primary_error, *recovery_errors)
        raise ProfileRepositoryError("operation_failed")

    def _recover_acquisition_with_replay(
        self,
        handle: BinaryIO,
        *,
        may_be_locked: bool,
        errors: list[BaseException],
    ) -> None:
        """Run complete acquisition recovery, replaying one aborted control path."""

        try:
            self._recover_acquisition_handle(
                handle,
                may_be_locked=may_be_locked,
                errors=errors,
            )
        except BaseException as error:
            errors.append(error)
            if not isinstance(error, Exception):
                # A complete recovery operation is replayed only after its one
                # promised control-flow interruption has been consumed.
                self._recover_acquisition_handle(
                    handle,
                    may_be_locked=may_be_locked,
                    errors=errors,
                )

    def _recover_acquisition_handle(
        self,
        handle: BinaryIO,
        *,
        may_be_locked: bool,
        errors: list[BaseException],
    ) -> None:
        """Clean and reconcile one acquisition handle as a complete operation."""

        cleanup_error = _unlock_and_close(handle, may_be_locked=may_be_locked)
        if cleanup_error is not None:
            errors.append(cleanup_error)

        state_error: BaseException | None = None
        try:
            if handle.closed:
                state_error = self._clear_handle_state(handle)
            else:
                state_error = self._retain_residual_handle(handle)
        except Exception as error:
            state_error = error
        if state_error is not None:
            errors.append(state_error)

        forced_state_error = self._force_recovery_state(handle)
        if forced_state_error is not None:
            errors.append(forced_state_error)

    def _clear_handle_state(self, expected_handle: BinaryIO) -> BaseException | None:
        """Identity-normalize a matching closed handle."""

        try:
            if not expected_handle.closed:
                return None
            if self._handle is expected_handle:
                self._handle = None
            if object.__getattribute__(self, "_residual_handle") is expected_handle:
                object.__setattr__(self, "_residual_handle", None)
        except BaseException as error:
            return error
        return None

    def _retain_residual_handle(self, handle: BinaryIO) -> BaseException | None:
        """Retain a residual handle without overwriting another live handle."""

        try:
            current_handle = self._handle
            if _residual_target_is_replaceable(current_handle, handle):
                self._handle = handle
        except BaseException as error:
            return error
        return None

    def _force_recovery_state(self, handle: BinaryIO) -> BaseException | None:
        """Force identity-safe state after one interrupted recovery attempt."""

        inspection_error: BaseException | None = None
        try:
            current_handle = object.__getattribute__(self, "_handle")
            residual_handle = object.__getattribute__(self, "_residual_handle")
            try:
                handle_closed = handle.closed
            except Exception as error:
                handle_closed = False
                inspection_error = error

            if handle_closed:
                self._force_clear_represented_handle(handle)
                return inspection_error

            try:
                replace_current = _residual_target_is_replaceable(
                    current_handle,
                    handle,
                )
            except Exception as error:
                replace_current = False
                inspection_error = error

            if replace_current:
                object.__setattr__(self, "_handle", handle)
                if residual_handle is handle:
                    object.__setattr__(self, "_residual_handle", None)
                return inspection_error

            try:
                replace_residual = _residual_target_is_replaceable(
                    residual_handle,
                    handle,
                )
            except Exception as error:
                replace_residual = False
                if inspection_error is None:
                    inspection_error = error
            if replace_residual:
                object.__setattr__(self, "_residual_handle", handle)
        except Exception as error:
            return error
        return inspection_error

    def _force_clear_represented_handle(self, handle: BinaryIO) -> None:
        """Clear every internal state slot matching one closed handle."""

        if object.__getattribute__(self, "_handle") is handle:
            object.__setattr__(self, "_handle", None)
        if object.__getattribute__(self, "_residual_handle") is handle:
            object.__setattr__(self, "_residual_handle", None)

    def release(self) -> None:
        """Synchronously unlock and close this lease idempotently.

        Raises:
            ProfileRepositoryError: If ordinary unlock or close cleanup fails.
            BaseException: A control-flow exception raised by cleanup is
                preserved after the remaining cleanup is attempted.
        """

        handle = object.__getattribute__(self, "_handle")
        residual_handle = object.__getattribute__(self, "_residual_handle")
        represented_handles: list[BinaryIO] = []
        for represented_handle in (handle, residual_handle):
            if represented_handle is not None and all(
                represented_handle is not existing for existing in represented_handles
            ):
                represented_handles.append(represented_handle)

        errors: list[BaseException] = []
        for represented_handle in represented_handles:
            try:
                self._release_represented_handle(represented_handle, errors)
            except BaseException as error:
                errors.append(error)
                if not isinstance(error, Exception):
                    self._release_represented_handle(represented_handle, errors)

        _raise_recovery_failure(None, *errors)

    def _release_represented_handle(
        self,
        handle: BinaryIO,
        errors: list[BaseException],
    ) -> None:
        """Best-effort clean and clear one represented handle."""

        try:
            handle_closed = handle.closed
        except Exception as error:
            errors.append(error)
            return
        if handle_closed:
            state_error = self._clear_handle_state(handle)
            if state_error is not None:
                errors.append(state_error)
            self._force_clear_represented_handle(handle)
            return

        cleanup_error = _unlock_and_close(handle, may_be_locked=True)
        if cleanup_error is not None:
            errors.append(cleanup_error)

        try:
            handle_closed = handle.closed
        except Exception as error:
            errors.append(error)
            return
        if handle_closed:
            state_error = self._clear_handle_state(handle)
            if state_error is not None:
                errors.append(state_error)
            self._force_clear_represented_handle(handle)

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
