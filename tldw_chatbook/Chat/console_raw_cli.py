"""Process-lifetime arming and cancellation owner for raw CLI execution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
import threading
import time
from typing import Any, Literal, TypeAlias

from tldw_chatbook.STT.executor_process_tree import ExecutorProcessTree
from tldw_chatbook.Tools.raw_cli_executor import (
    RawCliRequest,
    RawCliResult,
    RawCliStreamEvent,
    RawShellExecutor,
    validate_raw_cli_request,
)

RAW_CLI_SHUTDOWN_TIMEOUT_SECONDS = 5.0

RawCliArmReason: TypeAlias = Literal["armed", "locked", "shutdown"]
RawCliEventSink: TypeAlias = Callable[[RawCliStreamEvent], None]


@dataclass(frozen=True, slots=True)
class RawCliArmResult:
    """Outcome of one immediate arm request."""

    armed: bool
    reason: RawCliArmReason


@dataclass(frozen=True, slots=True)
class RawCliShutdownResult:
    """Bounded snapshot returned by the first runtime shutdown."""

    cancelled_invocation_ids: tuple[str, ...]
    unfinished_invocation_ids: tuple[str, ...]


@dataclass(slots=True)
class _ActiveInvocation:
    cancel_event: threading.Event
    done_event: threading.Event


class RawCliRuntime:
    """Own one launch-local arm bit and all active raw CLI invocations."""

    def __init__(
        self,
        read_permitted: Callable[[], object],
        *,
        executor: Any | None = None,
        shutdown_timeout_seconds: float = RAW_CLI_SHUTDOWN_TIMEOUT_SECONDS,
    ) -> None:
        if not callable(read_permitted):
            raise TypeError("read_permitted must be callable")
        if (
            isinstance(shutdown_timeout_seconds, bool)
            or not isinstance(shutdown_timeout_seconds, (int, float))
            or not math.isfinite(shutdown_timeout_seconds)
            or shutdown_timeout_seconds < 0
        ):
            raise ValueError("shutdown timeout must be a finite nonnegative number")
        self._read_permitted = read_permitted
        self._executor = executor if executor is not None else RawShellExecutor()
        self._shutdown_timeout_seconds = float(shutdown_timeout_seconds)
        self._lock = threading.RLock()
        self._shutdown_call_lock = threading.Lock()
        self._armed = False
        self._shutdown_started = False
        self._shutdown_result: RawCliShutdownResult | None = None
        self._active_invocations: dict[str, _ActiveInvocation] = {}

    @property
    def permitted(self) -> bool:
        """Return the latest strict persisted unlock value."""
        with self._lock:
            return self._latest_permitted_locked()

    @property
    def armed(self) -> bool:
        """Return the process-memory-only arm bit."""
        with self._lock:
            return self._armed

    def arm(self) -> RawCliArmResult:
        """Arm this process only when the latest persisted unlock is true."""
        with self._lock:
            if self._shutdown_started:
                return RawCliArmResult(armed=False, reason="shutdown")
            if not self._latest_permitted_locked():
                return RawCliArmResult(armed=False, reason="locked")
            self._armed = True
            return RawCliArmResult(armed=True, reason="armed")

    def disarm(self) -> tuple[str, ...]:
        """Close future admission and signal every currently active invocation."""
        with self._lock:
            self._armed = False
            self._clear_model_session_grants_locked()
            active = tuple(sorted(self._active_invocations.items()))
        for _invocation_id, invocation in active:
            invocation.cancel_event.set()
        return tuple(invocation_id for invocation_id, _invocation in active)

    def execute(
        self,
        request: RawCliRequest,
        on_event: RawCliEventSink,
    ) -> RawCliResult:
        """Synchronously execute one request through the guarded admission seam."""
        validate_raw_cli_request(request)
        if not callable(on_event):
            raise TypeError("on_event must be callable")

        active = _ActiveInvocation(
            cancel_event=threading.Event(),
            done_event=threading.Event(),
        )
        with self._lock:
            if (
                self._shutdown_started
                or not self._latest_permitted_locked()
                or not self._armed
            ):
                return self._refused_result(request)
            if request.invocation_id in self._active_invocations:
                raise ValueError("raw CLI invocation id is already active")
            self._active_invocations[request.invocation_id] = active

        def admit_worker(
            tree: ExecutorProcessTree,
            commit_launch: Callable[[], None],
        ) -> bool:
            with self._lock:
                if (
                    self._shutdown_started
                    or self._active_invocations.get(request.invocation_id) is not active
                    or not self._latest_permitted_locked()
                    or not self._armed
                ):
                    return False
                tree.admit()
                commit_launch()
                return True

        try:
            return self._executor.execute(
                request,
                cancel_event=active.cancel_event,
                on_event=on_event,
                admit_worker=admit_worker,
            )
        finally:
            with self._lock:
                if self._active_invocations.get(request.invocation_id) is active:
                    del self._active_invocations[request.invocation_id]
                active.done_event.set()

    def cancel(self, invocation_id: str) -> bool:
        """Signal one invocation only while it remains active."""
        with self._lock:
            active = self._active_invocations.get(invocation_id)
            if active is None:
                return False
            active.cancel_event.set()
            return True

    def shutdown(self) -> RawCliShutdownResult:
        """Disarm, cancel active work, and wait only for the configured bound."""
        with self._shutdown_call_lock:
            with self._lock:
                if self._shutdown_result is not None:
                    return self._shutdown_result
                self._shutdown_started = True
                self._armed = False
                self._clear_model_session_grants_locked()
                active = tuple(sorted(self._active_invocations.items()))

            for _invocation_id, invocation in active:
                invocation.cancel_event.set()

            deadline = time.monotonic() + self._shutdown_timeout_seconds
            for _invocation_id, invocation in active:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                invocation.done_event.wait(remaining)

            with self._lock:
                unfinished = tuple(
                    invocation_id
                    for invocation_id, invocation in active
                    if self._active_invocations.get(invocation_id) is invocation
                )
                result = RawCliShutdownResult(
                    cancelled_invocation_ids=tuple(
                        invocation_id for invocation_id, _invocation in active
                    ),
                    unfinished_invocation_ids=unfinished,
                )
                self._shutdown_result = result
                return result

    def _latest_permitted_locked(self) -> bool:
        try:
            return self._read_permitted() is True
        except Exception:
            return False

    def _clear_model_session_grants_locked(self) -> None:
        """Task 3 hook; model session grants are introduced by a later task."""

    @staticmethod
    def _refused_result(request: RawCliRequest) -> RawCliResult:
        return RawCliResult(
            invocation_id=request.invocation_id,
            caller=request.caller,
            resolved_shell=request.shell,
            initial_directory=request.initial_directory,
            elapsed_seconds=0.0,
            stdout_preview="",
            stderr_preview="",
            record_output="",
            exit_code=None,
            terminal_state="refused",
            truncated=False,
            cleanup_proven=True,
        )


__all__ = [
    "RAW_CLI_SHUTDOWN_TIMEOUT_SECONDS",
    "RawCliArmReason",
    "RawCliArmResult",
    "RawCliEventSink",
    "RawCliRuntime",
    "RawCliShutdownResult",
]
