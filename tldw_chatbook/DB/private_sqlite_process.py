"""Process-owned admission and bounded private SQLite helper lifetimes.

Only private pipes cross this boundary; original database file descriptors never
enter the parent. Reservations own children until reaping is actually confirmed.
"""

from __future__ import annotations

import copy
import math
import os
import selectors
import struct
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from tldw_chatbook.DB.private_sqlite_protocol import (
    MAX_BODY_BYTES,
    PrepareRequest,
    PrepareResult,
    ProtocolError,
    decode_frame,
    encode_frame,
)

_LEASE_KINDS = {
    "prepare": "transient",
    "pin_source": "transient",
    "tts_exact_current": "retained",
}


class HelperUnavailableError(RuntimeError):
    """A helper or its ownership envelope is unavailable."""

    def __init__(self) -> None:
        super().__init__("private_sqlite_helper_unavailable")


class HelperTimeoutError(RuntimeError):
    """The operation budget expired; cleanup has a separate bound."""

    def __init__(self) -> None:
        super().__init__("private_sqlite_helper_timeout")


class HelperProtocolError(RuntimeError):
    """The private channel failed its closed protocol."""

    def __init__(self) -> None:
        super().__init__("private_sqlite_helper_protocol_error")


class HelperCleanupError(RuntimeError):
    """Captured resources remain owned after bounded cleanup."""

    def __init__(self) -> None:
        super().__init__("private_sqlite_helper_cleanup_failed")


@dataclass(frozen=True)
class OperationDeadline:
    """Absolute monotonic operation budget, shared by admission and transport."""

    expires_at: float | None

    def __post_init__(self) -> None:
        if self.expires_at is not None and (
            type(self.expires_at) not in (int, float)
            or not math.isfinite(self.expires_at)
        ):
            raise HelperUnavailableError()

    def remaining(self, cap: float) -> float:
        """Refuse before further work once the absolute budget is exhausted."""
        remaining = (
            cap
            if self.expires_at is None
            else min(cap, self.expires_at - time.monotonic())
        )
        if remaining <= 0:
            raise HelperTimeoutError()
        return remaining

    def bounded(self, cap: float) -> OperationDeadline:
        """Anchor a phase cap once, never once per partial pipe event."""
        return OperationDeadline(time.monotonic() + self.remaining(cap))


class HelperAdmission:
    """One process's fixed four-transient/four-retained atomic accounting."""

    def __init__(self) -> None:
        self._pid = os.getpid()
        self._condition = threading.Condition()
        self._used = {"transient": 0, "retained": 0}
        # Strong ownership survives a failed start/cleanup and caller unwinding.
        self._owners: set[HelperReservation] = set()
        self._local = threading.local()

    def _check_process(self) -> None:
        # Check before touching an inherited lock, which may be held at fork.
        if os.getpid() != self._pid:
            raise HelperUnavailableError()

    def reserve(
        self, *, transient: int, retained: int, deadline: OperationDeadline
    ) -> HelperReservation:
        """Reserve the complete pair or wait without holding either half."""
        self._check_process()
        if getattr(self._local, "operation", None) is not None:
            raise HelperUnavailableError()
        if (
            type(transient) is not int
            or type(retained) is not int
            or not 0 <= transient <= 4
            or not 0 <= retained <= 4
            or transient + retained == 0
        ):
            raise HelperUnavailableError()
        budget = deadline.bounded(5.0)
        with self._condition:
            while True:
                wait_seconds = budget.remaining(5.0)
                if (
                    self._used["transient"] + transient <= 4
                    and self._used["retained"] + retained <= 4
                ):
                    owner = HelperReservation(self, transient, retained, deadline)
                    self._owners.add(owner)
                    self._used["transient"] += transient
                    self._used["retained"] += retained
                    return owner
                self._condition.wait(wait_seconds)


class HelperReservation:
    """Explicit whole-operation owner; nested callers borrow this same object."""

    def __init__(
        self,
        admission: HelperAdmission,
        transient: int,
        retained: int,
        deadline: OperationDeadline,
    ) -> None:
        self._admission = admission
        self._capacity = {"transient": transient, "retained": retained}
        self._children: dict[HelperLease, str] = {}
        self._handoffs: set[HelperLease] = set()
        self._closed = False
        self._terminal = False
        self._deadline = deadline
        self._thread = threading.get_ident()

    def _check_owner(self, *, allow_closed: bool = False) -> None:
        self._admission._check_process()
        if threading.get_ident() != self._thread or (self._closed and not allow_closed):
            raise HelperUnavailableError()

    def __enter__(self) -> Self:
        self._check_owner()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._check_owner(allow_closed=True)
        failure = None

        def settle(leases):
            nonlocal failure
            for lease in leases:
                try:
                    lease.close()
                except BaseException as error:  # noqa: BLE001 - settle every owner before propagating control flow
                    if failure is None or (
                        isinstance(failure, Exception)
                        and not isinstance(error, Exception)
                    ):
                        failure = error

        try:
            settle(
                tuple(lease for lease in self._children if lease not in self._handoffs)
            )
            # A handoff becomes effective only after the entire context settles
            # successfully, including cleanup of its transient preparation child.
            if exc is not None or failure is not None:
                settle(tuple(self._handoffs))
        finally:
            with self._admission._condition:
                self._closed = True
                self._release_unused()
        if failure is not None:
            if exc is not None and (
                not isinstance(exc, Exception) or isinstance(failure, Exception)
            ):
                exc.add_note("private_sqlite_helper_cleanup_failed")
            elif not isinstance(failure, Exception):
                if exc is not None:
                    failure.add_note("private_sqlite_operation_failed")
                raise failure from None
            else:
                raise HelperCleanupError() from None

    def handoff_retained(self, lease: HelperLease) -> None:
        """On successful context exit, transfer a retained lease to its caller.

        Unused transient capacity is released; the same reservation still owns
        the live retained charge. Exceptional exit cleans up pending handoffs.
        The caller must retain and close the lease on its owning worker thread.
        """
        self._check_owner()
        if self._children.get(lease) != "retained":
            raise HelperUnavailableError()
        self._handoffs.add(lease)

    @contextmanager
    def operation_scope(self) -> Iterator[HelperReservation]:
        """Guard the whole public operation, including reentrant SQL callbacks.

        Explicit nested borrowing may enter the same scope. Public reentry must
        acquire no new reservation while this scope owns an operation envelope.
        """
        self._check_owner()
        local = self._admission._local
        previous = getattr(local, "operation", None)
        if previous is not None and previous is not self:
            raise HelperUnavailableError()
        local.operation = self
        try:
            yield self
        finally:
            local.operation = previous

    def retain_terminal_owner(self) -> None:
        """Latch this already-reserved retained owner permit until process exit.

        Future live-proof consumers call this before relinquishing partial/live
        ownership. This neither grants capacity nor implements repository policy.
        """
        self._check_owner()
        if self._capacity["retained"] < 1:
            raise HelperUnavailableError()
        self._terminal = True

    def _claim(self, lease: HelperLease, kind: str) -> None:
        self._check_owner()
        with self._admission._condition:
            if (
                sum(value == kind for value in self._children.values())
                >= self._capacity[kind]
            ):
                raise HelperUnavailableError()
            self._children[lease] = kind

    def _reaped(self, lease: HelperLease) -> None:
        with self._admission._condition:
            self._children.pop(lease, None)
            self._handoffs.discard(lease)
            if self._closed:
                self._release_unused()

    def _release_unused(self) -> None:
        for kind in self._capacity:
            needed = sum(value == kind for value in self._children.values())
            if kind == "retained" and self._terminal:
                needed = max(1, needed)
            self._admission._used[kind] -= self._capacity[kind] - needed
            self._capacity[kind] = needed
        if not any(self._capacity.values()):
            self._admission._owners.discard(self)
        self._admission._condition.notify_all()


class HelperLease:
    """One captured exec child and at most one outstanding bounded request."""

    def __init__(self, reservation: HelperReservation, operation: str) -> None:
        self._reservation = reservation
        self._operation = operation
        self._child: subprocess.Popen | None = None
        self._response: dict[str, object] | None = None
        self._busy = False
        self._failed = False
        self._reaped_child = False

    @classmethod
    def start(
        cls,
        request: PrepareRequest,
        *,
        operation: str,
        reservation: HelperReservation,
        deadline: OperationDeadline,
    ) -> HelperLease:
        """Launch fixed installed code and retain its validated initial reply."""
        reservation._check_owner()
        if (
            type(operation) is not str
            or operation not in _LEASE_KINDS
            or type(request) is not PrepareRequest
        ):
            raise HelperProtocolError()
        try:
            frame = encode_frame(
                {
                    "version": 1,
                    "operation": operation,
                    "path": request.path,
                    "writable": request.writable,
                    "create_if_missing": request.create_if_missing,
                    "preserve_source_mode": request.preserve_source_mode,
                }
            )
        except ProtocolError:
            raise HelperProtocolError() from None
        phase_cap = 30.0 if operation == "tts_exact_current" else 5.0
        budget = OperationDeadline(
            time.monotonic()
            + min(
                deadline.remaining(phase_cap),
                reservation._deadline.remaining(phase_cap),
            )
        )
        lease = cls(reservation, operation)
        reservation._claim(lease, _LEASE_KINDS[operation])
        try:
            budget.remaining(5.0)
            # Trusted launch-only identity, never supplied by a request or an
            # ambient override. Capture before exec, including delayed startup.
            environment = os.environ.copy()
            environment["_TLDW_PRIVATE_SQLITE_PARENT_PID"] = str(os.getpid())
            lease._child = subprocess.Popen(
                [
                    sys.executable,
                    "-I",
                    "-S",
                    str(
                        Path(__file__)
                        .with_name("private_sqlite_helper_entry.py")
                        .resolve()
                    ),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                close_fds=True,
                bufsize=0,
                env=environment,
            )
            os.set_blocking(lease._child.stdin.fileno(), False)
            os.set_blocking(lease._child.stdout.fileno(), False)
            lease._response = lease._exchange(frame, operation, budget)
            return lease
        except BaseException as error:
            lease._failed = True
            try:
                lease.close()
            except BaseException as cleanup_error:
                if isinstance(error, Exception) and not isinstance(
                    cleanup_error, Exception
                ):
                    cleanup_error.add_note("private_sqlite_operation_failed")
                    raise cleanup_error from None
                error.add_note("private_sqlite_helper_cleanup_failed")
            if isinstance(error, OSError):
                normalized = HelperUnavailableError()
                for note in getattr(error, "__notes__", ()):
                    normalized.add_note(note)
                raise normalized from None
            raise

    @property
    def initial_response(self) -> dict[str, object]:
        """Defensive closed metadata projection, including private refusals."""
        self._reservation._check_owner(allow_closed=True)
        if self._response is None:
            raise HelperUnavailableError()
        return copy.deepcopy(self._response)

    @property
    def initial_result(self) -> PrepareResult:
        """Typed success identity; refuse rather than inventing absent proof."""
        response = self.initial_response
        if response["status"] != "ok":
            raise HelperUnavailableError()
        return PrepareResult.from_payload(response["result"])

    @property
    def cleanup_state(self) -> str:
        """Distinguish safe reaping, still-owned child, and retained owner permit."""
        self._reservation._check_owner(allow_closed=True)
        if self._reservation._terminal:
            return "terminal_retained"
        return "reaped" if self._reaped_child else "still_owned"

    def request(
        self, operation: str, *, deadline: OperationDeadline
    ) -> dict[str, object]:
        """Recheck this pin on its owning worker, without replacing its target."""
        self._reservation._check_owner(allow_closed=self in self._reservation._handoffs)
        allowed = (
            {"tts_pin_sidecars", "tts_recheck", "tts_export_restore_authority"}
            if self._operation == "tts_exact_current"
            else {"recheck_source"}
            if self._operation == "pin_source"
            else set()
        )
        if type(operation) is not str or operation not in allowed:
            raise HelperProtocolError()
        if self._busy or self._failed or self._reaped_child:
            raise HelperUnavailableError()
        remaining = deadline.remaining(5.0)
        # Marking a handoff does not settle the enclosing initialization budget.
        # Only a retained lease surviving owner exit starts its separate lifetime.
        if not self._reservation._closed or self not in self._reservation._handoffs:
            remaining = min(remaining, self._reservation._deadline.remaining(5.0))
        budget = OperationDeadline(time.monotonic() + remaining)
        return self._exchange(
            encode_frame({"version": 1, "operation": operation}), operation, budget
        )

    def retain_terminal_owner(self) -> None:
        """Keep this live retained owner charge even after independent reaping.

        A handed-off wrapper uses this before cleanup when it loses live proof;
        partial initialization can instead mark its still-open reservation.
        """
        owner = self._reservation
        owner._check_owner(allow_closed=self in owner._handoffs)
        if owner._children.get(self) != "retained":
            raise HelperUnavailableError()
        owner._terminal = True

    def _exchange(
        self, frame: bytes, operation: str, deadline: OperationDeadline
    ) -> dict[str, object]:
        phase_cap = 30.0 if operation == "tts_exact_current" else 5.0
        self._busy = True
        try:
            with selectors.DefaultSelector() as selector:
                incoming = self._child.stdout.fileno()
                outgoing = self._child.stdin.fileno()
                if self._response is not None:
                    self._reject_extra_reply(incoming)
                selector.register(outgoing, selectors.EVENT_WRITE)
                offset = 0
                while offset < len(frame):
                    if not selector.select(deadline.remaining(phase_cap)):
                        raise HelperTimeoutError()
                    deadline.remaining(phase_cap)
                    try:
                        offset += os.write(outgoing, frame[offset:])
                    except BlockingIOError:
                        continue
                selector.unregister(outgoing)
                selector.register(incoming, selectors.EVENT_READ)
                data = bytearray()
                expected = 4
                while len(data) < expected:
                    if not selector.select(deadline.remaining(phase_cap)):
                        raise HelperTimeoutError()
                    deadline.remaining(phase_cap)
                    try:
                        part = os.read(incoming, expected - len(data))
                    except BlockingIOError:
                        continue
                    if not part:
                        if data:
                            raise HelperProtocolError()
                        raise HelperUnavailableError()
                    data.extend(part)
                    if len(data) == 4 and expected == 4:
                        length = struct.unpack("!I", data)[0]
                        if not 0 < length <= MAX_BODY_BYTES:
                            raise HelperProtocolError()
                        expected += length
                response = decode_frame(bytes(data))
                self._reject_extra_reply(incoming)
                if response["operation"] != operation or "status" not in response:
                    raise HelperProtocolError()
                if response["status"] == "protocol_error":
                    raise HelperProtocolError()
                if response["status"] == "helper_unavailable":
                    raise HelperUnavailableError()
                if response["status"] == "timeout":
                    raise HelperTimeoutError()
                if response["status"] != "ok":
                    self._failed = True
                return response
        except (OSError, ProtocolError):
            self._failed = True
            raise HelperProtocolError() from None
        except BaseException:
            self._failed = True
            raise
        finally:
            self._busy = False

    @staticmethod
    def _reject_extra_reply(incoming: int) -> None:
        try:
            if os.read(incoming, 1):
                raise HelperProtocolError()
        except BlockingIOError:
            pass

    def close(self) -> None:
        """One-second orderly close, then two seconds for captured-child reap.

        An unreaped child stays strongly owned and charged. A later explicit
        close may retry; no unrelated process or descendant is ever signaled.
        """
        self._reservation._check_owner(allow_closed=True)
        if self._reaped_child:
            return
        child = self._child
        if child is None:
            self._reaped_child = True
            self._reservation._reaped(self)
            return
        primary = None
        normal = OperationDeadline(time.monotonic() + 1.0)
        try:
            if child.poll() is None:
                if not self._failed:
                    try:
                        self._exchange(
                            encode_frame({"version": 1, "operation": "close"}),
                            "close",
                            normal,
                        )
                    except (
                        HelperUnavailableError,
                        HelperProtocolError,
                        HelperTimeoutError,
                    ):
                        pass
                try:
                    child.wait(timeout=normal.remaining(1.0))
                except (subprocess.TimeoutExpired, HelperTimeoutError):
                    pass
        except BaseException as error:  # noqa: BLE001 - preserve control flow through owned cleanup
            primary = error
        cleanup = OperationDeadline(time.monotonic() + 2.0)
        try:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=cleanup.remaining(1.0))
                except (subprocess.TimeoutExpired, HelperTimeoutError):
                    child.kill()
                    child.wait(timeout=cleanup.remaining(2.0))
        except (OSError, subprocess.TimeoutExpired, HelperTimeoutError):
            pass
        except BaseException as error:  # noqa: BLE001 - re-raise after captured resources settle
            if primary is None:
                primary = error
        reaped = child.poll() is not None
        self._failed = True
        for stream in (child.stdin, child.stdout):
            try:
                stream.close()
            except OSError:
                reaped = False
        if reaped:
            self._reaped_child = True
            self._reservation._reaped(self)
        if primary is not None:
            if not reaped:
                primary.add_note("private_sqlite_helper_cleanup_failed")
            raise primary
        if not reaped:
            raise HelperCleanupError()


# Inherited production accounting is refused after fork; exec gets a fresh owner.
HELPER_ADMISSION = HelperAdmission()
