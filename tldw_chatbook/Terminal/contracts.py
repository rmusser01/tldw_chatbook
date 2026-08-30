"""Minimal value contracts for the persistent terminal boundary."""

from dataclasses import dataclass, replace
from enum import Enum


MAX_SESSION_RECORDS = 4
MIN_COLUMNS, MAX_COLUMNS = 5, 300
MIN_ROWS, MAX_ROWS = 2, 120
MAX_SCROLLBACK_LINES = 5_000
MAX_SCROLLBACK_BYTES = 4 * 1024 * 1024
MAX_PENDING_INPUT_BYTES = 512 * 1024
MAX_PENDING_OUTPUT_BYTES = 512 * 1024
MAX_PASTE_BYTES = 256 * 1024
MAX_IO_CHUNK_BYTES = 64 * 1024
MAX_PARSER_TURN_BYTES = 256 * 1024
MAX_PARSER_TURN_SECONDS = 0.008


class TerminalLifecycle(str, Enum):
    RESERVED = "reserved"
    CREATING = "creating"
    ADMITTING = "admitting"
    RUNNING = "running"
    DRAINING = "draining"
    EXITED = "exited"
    CLOSING = "closing"
    CLOSED = "closed"
    CLEANUP_UNPROVEN = "cleanup_unproven"


class TerminalReason(str, Enum):
    LOCKED = "locked"
    UNARMED = "unarmed"
    SESSION_LIMIT = "session_limit"
    INVALID_NAME = "invalid_name"
    INVALID_START_DIRECTORY = "invalid_start_directory"
    SHELL_UNAVAILABLE = "shell_unavailable"
    BACKEND_UNAVAILABLE = "backend_unavailable"
    ADMISSION_FAILED = "admission_failed"
    SPAWN_FAILED = "spawn_failed"
    INPUT_BACKPRESSURE = "input_backpressure"
    TERMINAL_PROTOCOL_FAILED = "terminal_protocol_failed"
    IO_FAILED = "io_failed"
    WORKER_FAILED = "worker_failed"
    OUTPUT_INCOMPLETE = "output_incomplete"
    CLEANUP_UNPROVEN = "cleanup_unproven"


@dataclass(frozen=True, slots=True)
class CleanupSchedule:
    deadline_seconds: float = 5.0
    hangup_no_later_than: float = 0.75
    terminate_no_later_than: float = 2.25
    force_kill_no_later_than: float = 3.75
    proof_reserve_seconds: float = 1.25


@dataclass(frozen=True, slots=True)
class TerminalLaunchRequest:
    name: str = ""
    shell: str = ""
    start_directory: str = ""
    columns: int = 0
    rows: int = 0


@dataclass(frozen=True, slots=True)
class AdmissionGate:
    admitted: bool = False
    token: str = ""


@dataclass(frozen=True, slots=True)
class BackendIdentity:
    session_id: str = ""


@dataclass(frozen=True, slots=True)
class CleanupAttempt:
    t0: float = 0.0


@dataclass(frozen=True, slots=True)
class CleanupProof:
    process_dead: bool = False
    stream_closed: bool = False
    output_complete: bool = False


@dataclass(frozen=True, slots=True)
class TerminalEvent:
    kind: str
    exit_code: int | None = None
    reason: TerminalReason | None = None


@dataclass(frozen=True, slots=True)
class TerminalProjection:
    session_id: str = ""
    name: str = ""
    lifecycle: TerminalLifecycle = TerminalLifecycle.RESERVED
    reason: TerminalReason | None = None
    exit_code: int | None = None
    stream_closed: bool = False
    output_complete: bool = False


@dataclass(frozen=True, slots=True)
class TerminalReceipt:
    attempt: CleanupAttempt
    action: str = ""


def validate_transition(current: TerminalLifecycle, target: TerminalLifecycle) -> bool:
    """Return whether a lifecycle transition is permitted."""
    allowed = {
        TerminalLifecycle.RESERVED: {
            TerminalLifecycle.CREATING,
            TerminalLifecycle.CLOSED,
        },
        TerminalLifecycle.CREATING: {
            TerminalLifecycle.ADMITTING,
            TerminalLifecycle.CLOSED,
        },
        TerminalLifecycle.ADMITTING: {
            TerminalLifecycle.RUNNING,
            TerminalLifecycle.CLOSED,
        },
        TerminalLifecycle.RUNNING: {
            TerminalLifecycle.DRAINING,
            TerminalLifecycle.CLOSING,
        },
        TerminalLifecycle.DRAINING: {
            TerminalLifecycle.EXITED,
            TerminalLifecycle.CLOSING,
        },
        TerminalLifecycle.EXITED: {TerminalLifecycle.CLOSING},
        TerminalLifecycle.CLOSING: {
            TerminalLifecycle.CLOSED,
            TerminalLifecycle.CLEANUP_UNPROVEN,
        },
        TerminalLifecycle.CLEANUP_UNPROVEN: {TerminalLifecycle.CLOSING},
        TerminalLifecycle.CLOSED: set(),
    }
    return target in allowed[current]


def apply_event(
    projection: TerminalProjection, event: TerminalEvent
) -> TerminalProjection:
    """Apply a lifecycle event to an immutable projection."""
    lifecycle = projection.lifecycle
    exit_code = projection.exit_code
    reason = projection.reason
    stream_closed = projection.stream_closed
    output_complete = projection.output_complete

    if event.kind == "shell_exit" and validate_transition(
        lifecycle, TerminalLifecycle.DRAINING
    ):
        lifecycle = TerminalLifecycle.DRAINING
        exit_code = event.exit_code
    elif event.kind == "admission_failure" and validate_transition(
        lifecycle, TerminalLifecycle.CLOSED
    ):
        lifecycle = TerminalLifecycle.CLOSED
        reason = TerminalReason.ADMISSION_FAILED
    elif event.kind == "parser_failure":
        reason = TerminalReason.TERMINAL_PROTOCOL_FAILED
        output_complete = False
        if lifecycle is not TerminalLifecycle.CLEANUP_UNPROVEN and validate_transition(
            lifecycle, TerminalLifecycle.CLOSING
        ):
            lifecycle = TerminalLifecycle.CLOSING
    elif event.kind == "cleanup_proven" and validate_transition(
        lifecycle, TerminalLifecycle.CLOSED
    ):
        lifecycle = TerminalLifecycle.CLOSED
    elif event.kind == "cleanup_failed" and validate_transition(
        lifecycle, TerminalLifecycle.CLEANUP_UNPROVEN
    ):
        lifecycle = TerminalLifecycle.CLEANUP_UNPROVEN
        reason = TerminalReason.CLEANUP_UNPROVEN
    elif (
        event.kind == "close"
        and lifecycle is not TerminalLifecycle.CLEANUP_UNPROVEN
        and validate_transition(lifecycle, TerminalLifecycle.CLOSING)
    ):
        lifecycle = TerminalLifecycle.CLOSING
    elif event.kind == "stream_closed":
        stream_closed = True
    elif event.kind == "output_complete":
        output_complete = True

    return TerminalProjection(
        session_id=projection.session_id,
        name=projection.name,
        lifecycle=lifecycle,
        reason=reason,
        exit_code=exit_code,
        stream_closed=stream_closed,
        output_complete=output_complete,
    )


def retry_cleanup(
    projection: TerminalProjection, t0: float
) -> tuple[TerminalProjection, TerminalReceipt]:
    """Start an explicit cleanup retry and create its fresh receipt."""
    if projection.lifecycle is not TerminalLifecycle.CLEANUP_UNPROVEN:
        raise ValueError("cleanup retry requires cleanup_unproven lifecycle")
    return (
        replace(projection, lifecycle=TerminalLifecycle.CLOSING),
        TerminalReceipt(CleanupAttempt(t0), "retry"),
    )


def join_cleanup(receipt: TerminalReceipt, t0: float) -> TerminalReceipt:
    """Join an existing cleanup attempt without resetting its T0."""
    return receipt


def slot_held(lifecycle: TerminalLifecycle) -> bool:
    """Return whether a lifecycle retains its session reservation."""
    return lifecycle is not TerminalLifecycle.CLOSED
