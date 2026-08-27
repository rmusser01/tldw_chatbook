"""Process-lifetime arming and cancellation owner for raw CLI execution."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
import threading
import time
from typing import Any, Literal, TypeAlias

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleActivityPresentation,
    ConsoleChatMessage,
    ConsoleMessageRole,
    RawCliLifecycleState,
    RawCliPresentation,
)
from tldw_chatbook.STT.executor_process_tree import ExecutorProcessTree
from tldw_chatbook.Tools.raw_cli_executor import (
    MAX_RAW_COMMAND_BYTES,
    MAX_RAW_PREVIEW_BYTES,
    RawCliRequest,
    RawCliResult,
    RawCliStreamEvent,
    RawShellExecutor,
    validate_raw_cli_request,
)

RAW_CLI_SHUTDOWN_TIMEOUT_SECONDS = 5.0
LOCAL_COMMAND_AGENT_KIND = "local_command"
LOCAL_COMMAND_TASK = "Local command"
LOCAL_COMMAND_TOOL_NAME = "raw_cli"
LOCAL_COMMAND_RUN_LOG_DIR = "local-command-runs"

_RAW_CLI_COMPACT_OUTPUT_BYTES = 4 * 1024

RawCliArmReason: TypeAlias = Literal["armed", "locked", "shutdown"]
RawCliEventSink: TypeAlias = Callable[[RawCliStreamEvent], None]
RawCliRegisteredSink: TypeAlias = Callable[[], None]
RawCliStartedSink: TypeAlias = Callable[[float], None]


def _literal_terminal_text(value: str) -> str:
    """Make terminal controls visible while leaving ordinary text literal."""
    return "".join(
        character
        if character in "\n\t"
        or not (ord(character) < 0x20 or 0x7F <= ord(character) <= 0x9F)
        else f"\\x{ord(character):02x}"
        for character in value
    )


def _utf8_prefix(value: str, byte_limit: int) -> tuple[str, bool]:
    encoded = value.encode("utf-8")
    if len(encoded) <= byte_limit:
        return value, False
    return encoded[:byte_limit].decode("utf-8", errors="ignore"), True


def _raw_cli_output(stdout: str, stderr: str) -> str:
    safe_stdout = _literal_terminal_text(stdout) or "(no output)"
    safe_stderr = _literal_terminal_text(stderr) or "(no output)"
    return f"stdout:\n{safe_stdout}\n\nstderr:\n{safe_stderr}"


def format_raw_cli_content(
    presentation: RawCliPresentation,
    stdout: str,
    stderr: str,
) -> tuple[str, str]:
    """Return the bounded compact marker and its bounded full-output body."""
    full_output = _raw_cli_output(stdout, stderr)
    compact_output, clipped = _utf8_prefix(
        full_output,
        _RAW_CLI_COMPACT_OUTPUT_BYTES,
    )
    if clipped:
        compact_output += "\n… output preview clipped; use Full output"
    exit_code = (
        "Pending" if presentation.exit_code is None else str(presentation.exit_code)
    )
    cleanup = {None: "Pending", True: "Proven", False: "Unproven"}[
        presentation.cleanup_proven
    ]
    content = (
        f"Command:\n{_literal_terminal_text(presentation.command)}\n\n"
        f"Caller: {presentation.caller.title()}\n"
        f"Shell: {_literal_terminal_text(presentation.shell)}\n"
        f"CWD: {_literal_terminal_text(presentation.cwd)}\n"
        f"Elapsed: {presentation.elapsed_seconds:.1f}s\n"
        f"Exit code: {exit_code}\n"
        f"Truncated: {'Yes' if presentation.truncated else 'No'}\n"
        f"Cleanup: {cleanup}\n\n"
        f"{compact_output}"
    )
    return content, full_output


def raw_cli_terminal_lifecycle(result: RawCliResult) -> RawCliLifecycleState:
    """Map executor settlement onto the display lifecycle vocabulary."""
    if result.terminal_state in {
        "exited",
        "timed_out",
        "cancelled",
        "cleanup_unproven",
    }:
        return result.terminal_state
    return "failed"


def raw_cli_activity_presentation(
    lifecycle_state: RawCliLifecycleState,
    exit_code: int | None,
) -> ConsoleActivityPresentation:
    """Build the shared live/resume activity header."""
    if lifecycle_state == "exited" and exit_code == 0:
        status = "success"
    elif lifecycle_state in {"starting", "running", "stopping"}:
        status = "done"
    elif lifecycle_state in {"timed_out", "cancelled"}:
        status = "blocked"
    else:
        status = "failed"
    return ConsoleActivityPresentation("tool", "Raw CLI", status)


def local_command_run_status(result: RawCliResult) -> str:
    """Map executor settlement onto the durable run terminal vocabulary."""
    if result.terminal_state == "exited":
        return "done"
    if result.terminal_state == "cancelled":
        return "cancelled"
    return "error"


def _resume_utf8_text(
    value: object,
    *,
    max_bytes: int,
    nonblank: bool = False,
    single_line: bool = False,
    nul_free: bool = False,
) -> str:
    """Return strictly typed, bounded UTF-8 text or reject the record."""
    if type(value) is not str:
        raise TypeError("persisted local-command text must be a string")
    encoded = value.encode("utf-8")
    if len(encoded) > max_bytes:
        raise ValueError("persisted local-command text exceeds its live limit")
    if nonblank and not value.strip():
        raise ValueError("persisted local-command text must not be blank")
    if single_line and any(character in value for character in "\r\n"):
        raise ValueError("persisted local-command text must be one line")
    if nul_free and "\x00" in value:
        raise ValueError("persisted local-command text must not contain NUL")
    return value


def local_command_resume_marker(record: Mapping[str, Any]) -> ConsoleChatMessage | None:
    """Rebuild one terminal display marker from a local-command run record."""
    try:
        if not isinstance(record, Mapping):
            return None
        run_id = _resume_utf8_text(
            record.get("id"),
            max_bytes=128,
            nonblank=True,
            single_line=True,
            nul_free=True,
        )
        if record.get("agent_kind") != LOCAL_COMMAND_AGENT_KIND:
            return None
        steps = record.get("steps")
        if (
            not isinstance(steps, Sequence)
            or isinstance(steps, (str, bytes, bytearray))
            or len(steps) != 2
            or not all(isinstance(step, Mapping) for step in steps)
        ):
            return None
        call, result = steps
        if (
            call.get("kind") != "tool_call"
            or call.get("tool_name") != LOCAL_COMMAND_TOOL_NAME
            or type(call.get("index")) is not int
            or call.get("index") != 0
            or result.get("kind") != "tool_result"
            or result.get("tool_name") != LOCAL_COMMAND_TOOL_NAME
            or type(result.get("index")) is not int
            or result.get("index") != 1
        ):
            return None
        call_args = call.get("args")
        result_args = result.get("args")
        if not isinstance(call_args, Mapping) or not isinstance(result_args, Mapping):
            return None

        invocation_id = _resume_utf8_text(
            call_args.get("invocation_id"),
            max_bytes=128,
            nonblank=True,
            single_line=True,
            nul_free=True,
        )
        result_invocation_id = _resume_utf8_text(
            result_args.get("invocation_id"),
            max_bytes=128,
            nonblank=True,
            single_line=True,
            nul_free=True,
        )
        if result_invocation_id != invocation_id:
            return None
        command = _resume_utf8_text(
            call_args.get("command"),
            max_bytes=MAX_RAW_COMMAND_BYTES,
            nonblank=True,
            nul_free=True,
        )
        _resume_utf8_text(
            call_args.get("shell"),
            max_bytes=4096,
            nonblank=True,
            single_line=True,
            nul_free=True,
        )
        call_cwd = _resume_utf8_text(
            call_args.get("cwd"),
            max_bytes=4096,
            nonblank=True,
            single_line=True,
            nul_free=True,
        )
        shell = _resume_utf8_text(
            result_args.get("shell"),
            max_bytes=4096,
            nonblank=True,
            single_line=True,
            nul_free=True,
        )
        cwd = _resume_utf8_text(
            result_args.get("cwd"),
            max_bytes=4096,
            nonblank=True,
            single_line=True,
            nul_free=True,
        )
        if cwd != call_cwd:
            return None
        stdout = _resume_utf8_text(
            result_args.get("stdout_preview"),
            max_bytes=MAX_RAW_PREVIEW_BYTES,
        )
        stderr = _resume_utf8_text(
            result_args.get("stderr_preview"),
            max_bytes=MAX_RAW_PREVIEW_BYTES,
        )
        if len(stdout.encode("utf-8")) + len(stderr.encode("utf-8")) > (
            MAX_RAW_PREVIEW_BYTES
        ):
            return None

        elapsed_seconds = result_args.get("elapsed_seconds")
        if (
            type(elapsed_seconds) not in {int, float}
            or not math.isfinite(elapsed_seconds)
            or elapsed_seconds < 0
        ):
            return None
        exit_code = result_args.get("exit_code")
        if exit_code is not None and type(exit_code) is not int:
            return None
        truncated = result_args.get("truncated")
        if type(truncated) is not bool:
            return None
        cleanup_proven = result_args.get("cleanup_proven")
        if type(cleanup_proven) is not bool:
            return None

        terminal_state = result_args.get("terminal_state")
        terminal_status = {
            "refused": "error",
            "shell_unavailable": "error",
            "spawn_failed": "error",
            "containment_unavailable": "error",
            "exited": "done",
            "timed_out": "error",
            "cancelled": "cancelled",
            "cleanup_unproven": "error",
        }.get(terminal_state)
        if (
            terminal_status is None
            or type(record.get("status")) is not str
            or record.get("status") != terminal_status
            or type(result.get("status")) is not str
            or result.get("status") != terminal_status
        ):
            return None
        lifecycle: RawCliLifecycleState = (
            terminal_state
            if terminal_state
            in {"exited", "timed_out", "cancelled", "cleanup_unproven"}
            else "failed"
        )
        presentation = RawCliPresentation(
            invocation_id=invocation_id,
            caller="user",
            lifecycle_state=lifecycle,
            command=command,
            shell=shell,
            cwd=cwd,
            started_at_monotonic=None,
            elapsed_seconds=elapsed_seconds,
            exit_code=exit_code,
            truncated=truncated,
            cleanup_proven=cleanup_proven,
        )
        content, full_output = format_raw_cli_content(presentation, stdout, stderr)
    except (
        AttributeError,
        KeyError,
        OverflowError,
        TypeError,
        ValueError,
        UnicodeError,
    ):
        return None
    return ConsoleChatMessage(
        id=f"raw-cli-run-{run_id}",
        role=ConsoleMessageRole.TOOL,
        content=content,
        status="complete",
        tool_output_full=full_output,
        activity_presentation=raw_cli_activity_presentation(
            lifecycle,
            exit_code,
        ),
        raw_cli_presentation=presentation,
    )


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
    console_session_id: str
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
        self._admission_lock = threading.Lock()
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
        *,
        on_registered: RawCliRegisteredSink | None = None,
        on_started: RawCliStartedSink | None = None,
    ) -> RawCliResult:
        """Synchronously execute one request through the guarded admission seam."""
        validate_raw_cli_request(request)
        if not callable(on_event):
            raise TypeError("on_event must be callable")
        if on_registered is not None and not callable(on_registered):
            raise TypeError("on_registered must be callable or None")
        if on_started is not None and not callable(on_started):
            raise TypeError("on_started must be callable or None")

        active = _ActiveInvocation(
            console_session_id=request.console_session_id,
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
            commit_launch: Callable[[], float | None],
        ) -> bool:
            def authority_allows_launch_locked() -> bool:
                return not (
                    self._shutdown_started
                    or self._active_invocations.get(request.invocation_id) is not active
                    or not self._latest_permitted_locked()
                    or not self._armed
                    or active.cancel_event.is_set()
                )

            with self._admission_lock:
                with self._lock:
                    if not authority_allows_launch_locked():
                        return False
                tree.admit()
                with self._lock:
                    # The second check and commit are the atomic launch boundary.
                    if not authority_allows_launch_locked():
                        return False
                    started_at = commit_launch()
                if started_at is None:
                    return False
                if on_started is not None:
                    try:
                        on_started(started_at)
                    except Exception:
                        pass
                return True

        try:
            if on_registered is not None:
                on_registered()
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

    def cancel_session(self, session_id: str) -> tuple[str, ...]:
        """Signal every active invocation owned by one Console session."""
        with self._lock:
            active = tuple(
                sorted(
                    (
                        invocation_id,
                        invocation,
                    )
                    for invocation_id, invocation in self._active_invocations.items()
                    if invocation.console_session_id == session_id
                )
            )
            for _invocation_id, invocation in active:
                invocation.cancel_event.set()
        return tuple(invocation_id for invocation_id, _invocation in active)

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
    "LOCAL_COMMAND_AGENT_KIND",
    "LOCAL_COMMAND_RUN_LOG_DIR",
    "LOCAL_COMMAND_TASK",
    "LOCAL_COMMAND_TOOL_NAME",
    "RAW_CLI_SHUTDOWN_TIMEOUT_SECONDS",
    "RawCliArmReason",
    "RawCliArmResult",
    "RawCliEventSink",
    "RawCliRuntime",
    "RawCliStartedSink",
    "RawCliShutdownResult",
    "format_raw_cli_content",
    "local_command_resume_marker",
    "local_command_run_status",
    "raw_cli_activity_presentation",
    "raw_cli_terminal_lifecycle",
]
