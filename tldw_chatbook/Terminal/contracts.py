"""Minimal value contracts for the persistent terminal boundary."""

from dataclasses import dataclass, replace
from enum import Enum
from typing import Literal, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)
from pydantic import ValidationError as PydanticValidationError


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

TERMINAL_DISCLOSURE_LINES = (
    "The Terminal shell and every program run with the same OS permissions as "
    "Chatbook.",
    "Programs may read, modify, or delete any accessible data, use the network, "
    "invoke credentialed clients, or exhaust machine resources.",
    "Chatbook starts Terminal from a scrubbed environment, but normal shell profile "
    "and startup files run and may restore secrets, credentials, agents, proxies, "
    "aliases, environment variables, or arbitrary commands.",
    "Shells and programs may save history, files, logs, caches, and other side "
    "effects outside Chatbook.",
    "Your active workspace—or your home directory when no workspace is selected—is "
    "only the starting directory. Terminal is not sandboxed or confined there.",
    "Closing, disarming, or quitting Chatbook attempts bounded cleanup, but "
    "deliberately detached processes may survive and cleanup may remain unproven.",
    "Terminal content is user-only and is not sent to a model.",
)


class TerminalLifecycle(str, Enum):
    """Lifecycle states retained by the terminal session manager."""

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
    """Content-free terminal failure categories."""

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


_SETTLEMENT_EVENT_SOURCES = frozenset(
    {
        TerminalLifecycle.RUNNING,
        TerminalLifecycle.DRAINING,
        TerminalLifecycle.EXITED,
        TerminalLifecycle.CLOSING,
        TerminalLifecycle.CLEANUP_UNPROVEN,
    }
)

_TERMINAL_EVENT_KINDS = frozenset(
    {
        "shell_exit",
        "admission_failure",
        "parser_failure",
        "cleanup_proven",
        "cleanup_failed",
        "close",
        "stream_closed",
        "output_complete",
    }
)


@dataclass(frozen=True, slots=True)
class CleanupSchedule:
    """Absolute cleanup-stage offsets from one monotonic attempt start.

    Attributes:
        deadline_seconds: Total cleanup deadline.
        hangup_no_later_than: End of the initial hangup window.
        terminate_no_later_than: Latest start for termination.
        force_kill_no_later_than: Latest start for force-kill handling.
        proof_reserve_seconds: Time reserved for settlement and death proof.
    """

    deadline_seconds: float = 5.0
    hangup_no_later_than: float = 0.75
    terminate_no_later_than: float = 2.25
    force_kill_no_later_than: float = 3.75
    proof_reserve_seconds: float = 1.25


@dataclass(frozen=True, slots=True)
class TerminalLaunchRequest:
    """Platform-neutral values requested for terminal startup.

    Attributes:
        name: User-visible session name.
        shell: Code-owned shell identity.
        start_directory: Validated absolute starting directory.
        columns: Initial terminal width.
        rows: Initial terminal height.
    """

    name: str = ""
    shell: str = ""
    start_directory: str = ""
    columns: int = 0
    rows: int = 0


@dataclass(frozen=True, slots=True)
class AdmissionGate:
    """Admission decision supplied before interactive shell startup.

    Attributes:
        admitted: Whether startup may cross the admission boundary.
        token: Opaque admission identity.
    """

    admitted: bool = False
    token: str = ""


@dataclass(frozen=True, slots=True)
class BackendIdentity:
    """Content-free identity returned by an admitted backend.

    Attributes:
        session_id: Opaque identity for the owned terminal session.
    """

    session_id: str = ""


@dataclass(frozen=True, slots=True)
class CleanupAttempt:
    """One cleanup attempt's monotonic start time.

    Attributes:
        t0: Monotonic start shared by all absolute cleanup offsets.
    """

    t0: float = 0.0


@dataclass(frozen=True, slots=True)
class CleanupProof:
    """Platform-neutral cleanup evidence returned by a backend.

    Attributes:
        process_dead: Whether backend-owned processes are proven dead.
        stream_closed: Whether terminal EOF is proven.
        output_complete: Whether all admitted bytes used the healthy parser path.

    Raises:
        TypeError: If any evidence field is not a boolean.
    """

    process_dead: bool = False
    stream_closed: bool = False
    output_complete: bool = False

    def __post_init__(self) -> None:
        for field_name in ("process_dead", "stream_closed", "output_complete"):
            if type(getattr(self, field_name)) is not bool:
                raise TypeError(f"{field_name} must be bool")


@dataclass(frozen=True, slots=True)
class TerminalEvent:
    """Validated, content-free event consumed by the projection reducer.

    Attributes:
        kind: Allowlisted terminal event name.
        exit_code: Authoritative shell exit status when available.
        reason: Structured source reason when applicable.
        cleanup_proof: Backend evidence required for proven closure.

    Raises:
        ValueError: If ``kind`` is not an allowlisted terminal event.
        TypeError: If ``cleanup_proof`` is not a ``CleanupProof`` instance.
    """

    kind: str
    exit_code: int | None = None
    reason: TerminalReason | None = None
    cleanup_proof: CleanupProof | None = None

    def __post_init__(self) -> None:
        if self.kind not in _TERMINAL_EVENT_KINDS:
            raise ValueError(f"unknown terminal event kind: {self.kind!r}")
        if self.cleanup_proof is not None and not isinstance(
            self.cleanup_proof, CleanupProof
        ):
            raise TypeError("cleanup_proof must be CleanupProof or None")


@dataclass(frozen=True, slots=True)
class TerminalProjection:
    """Immutable user-interface-safe terminal session projection.

    Attributes:
        session_id: Opaque session identity.
        name: User-visible session name.
        lifecycle: Current session lifecycle.
        reason: Content-free failure category, if any.
        exit_code: Authoritative shell exit status, if observed.
        stream_closed: Whether terminal EOF has been observed.
        output_complete: Whether all admitted bytes were parsed successfully.
        parser_failed: Whether a parser failure has made output incomplete.
    """

    session_id: str = ""
    name: str = ""
    lifecycle: TerminalLifecycle = TerminalLifecycle.RESERVED
    reason: TerminalReason | None = None
    exit_code: int | None = None
    stream_closed: bool = False
    output_complete: bool = False
    parser_failed: bool = False


@dataclass(frozen=True, slots=True)
class TerminalReceipt:
    """Retained cleanup authority and action metadata.

    Attributes:
        attempt: Cleanup attempt whose deadline governs the action.
        action: Content-free cleanup action label.
    """

    attempt: CleanupAttempt
    action: str = ""


def validate_transition(current: TerminalLifecycle, target: TerminalLifecycle) -> bool:
    """Return whether a lifecycle transition is permitted.

    Args:
        current: Existing lifecycle state.
        target: Proposed lifecycle state.

    Returns:
        ``True`` when the approved lifecycle graph permits the transition.
    """
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
    """Apply an authorized lifecycle event to an immutable projection.

    Args:
        projection: Existing terminal projection.
        event: Validated content-free event.

    Returns:
        A new projection, or the unchanged value when the event is not
        authorized for the current lifecycle.
    """
    lifecycle = projection.lifecycle
    exit_code = projection.exit_code
    reason = projection.reason
    stream_closed = projection.stream_closed
    output_complete = projection.output_complete
    parser_failed = projection.parser_failed

    if event.kind == "shell_exit" and lifecycle in {
        TerminalLifecycle.RUNNING,
        TerminalLifecycle.CLOSING,
        TerminalLifecycle.CLEANUP_UNPROVEN,
    }:
        if lifecycle is TerminalLifecycle.RUNNING:
            lifecycle = TerminalLifecycle.DRAINING
        exit_code = event.exit_code
    elif event.kind == "admission_failure" and lifecycle is TerminalLifecycle.ADMITTING:
        lifecycle = TerminalLifecycle.CLOSED
        reason = TerminalReason.ADMISSION_FAILED
    elif event.kind == "parser_failure" and lifecycle in {
        TerminalLifecycle.RUNNING,
        TerminalLifecycle.DRAINING,
        TerminalLifecycle.EXITED,
        TerminalLifecycle.CLOSING,
        TerminalLifecycle.CLEANUP_UNPROVEN,
    }:
        output_complete = False
        parser_failed = True
        if lifecycle is not TerminalLifecycle.CLEANUP_UNPROVEN:
            reason = TerminalReason.TERMINAL_PROTOCOL_FAILED
            if validate_transition(lifecycle, TerminalLifecycle.CLOSING):
                lifecycle = TerminalLifecycle.CLOSING
    elif (
        event.kind == "cleanup_proven"
        and lifecycle is TerminalLifecycle.CLOSING
        and event.cleanup_proof is not None
        and event.cleanup_proof.process_dead is True
        and event.cleanup_proof.stream_closed is True
    ):
        lifecycle = TerminalLifecycle.CLOSED
        stream_closed = True
        output_complete = event.cleanup_proof.output_complete and not parser_failed
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
    elif event.kind == "stream_closed" and lifecycle in _SETTLEMENT_EVENT_SOURCES:
        stream_closed = True
    elif (
        event.kind == "output_complete"
        and lifecycle in _SETTLEMENT_EVENT_SOURCES
        and stream_closed
        and not parser_failed
    ):
        output_complete = True

    return TerminalProjection(
        session_id=projection.session_id,
        name=projection.name,
        lifecycle=lifecycle,
        reason=reason,
        exit_code=exit_code,
        stream_closed=stream_closed,
        output_complete=output_complete,
        parser_failed=parser_failed,
    )


def retry_cleanup(
    projection: TerminalProjection, t0: float
) -> tuple[TerminalProjection, TerminalReceipt]:
    """Start an explicit cleanup retry and create its fresh receipt.

    Args:
        projection: Retained cleanup-unproven projection.
        t0: Monotonic start for the fresh cleanup attempt.

    Returns:
        The closing projection and its new cleanup receipt.

    Raises:
        ValueError: If cleanup authority is not retained.
    """
    if projection.lifecycle is not TerminalLifecycle.CLEANUP_UNPROVEN:
        raise ValueError("cleanup retry requires cleanup_unproven lifecycle")
    return (
        replace(projection, lifecycle=TerminalLifecycle.CLOSING),
        TerminalReceipt(CleanupAttempt(t0), "retry"),
    )


def join_cleanup(receipt: TerminalReceipt, t0: float) -> TerminalReceipt:
    """Join cleanup without extending the earliest attempt deadline.

    Args:
        receipt: Existing cleanup receipt.
        t0: Start time of the shared cleanup attempt being joined.

    Returns:
        A receipt governed by the earlier monotonic start.
    """
    if t0 < receipt.attempt.t0:
        return replace(receipt, attempt=CleanupAttempt(t0))
    return receipt


def slot_held(lifecycle: TerminalLifecycle) -> bool:
    """Return whether a lifecycle retains its session reservation.

    Args:
        lifecycle: Session lifecycle to inspect.

    Returns:
        ``True`` until proven closure releases the reservation.
    """
    return lifecycle is not TerminalLifecycle.CLOSED


TerminalPasteViolation = Literal["too_large", "prohibited_control"]


class TerminalPasteInput(BaseModel):
    """Strict shared boundary for one terminal paste offer."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    text: str = Field(repr=False)
    bracketed: bool

    def classify(self) -> tuple[TerminalPasteViolation | None, bytes]:
        """Classify paste content and return its replacement-encoded payload.

        Returns:
            A content-free violation category and payload bytes. The payload is
            empty when a violation is present.
        """
        if len(self.text) > MAX_PASTE_BYTES:
            return "too_large", b""
        for character in self.text:
            codepoint = ord(character)
            if codepoint < 0x20 and character not in "\t\n\r":
                return "prohibited_control", b""
            if 0x7F <= codepoint <= 0x9F:
                return "prohibited_control", b""
        payload = self.text.encode("utf-8", "replace")
        if len(payload) > MAX_PASTE_BYTES:
            return "too_large", b""
        return None, payload


class TerminalKeyInput(BaseModel):
    """Strict shared boundary for encoded terminal key bytes."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    data: bytes = Field(repr=False)


class TerminalReplyInput(BaseModel):
    """Strict shared boundary for code-owned terminal reply bytes."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    data: bytes = Field(repr=False)


class TerminalOutputInput(BaseModel):
    """Strict shared boundary for terminal backend output bytes."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    data: bytes = Field(repr=False)


class TerminalResizeInput(BaseModel):
    """Strict shared boundary for terminal resize dimensions."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    columns: int
    rows: int

    @field_validator("columns")
    @classmethod
    def _validate_columns(cls, value: int) -> int:
        if not MIN_COLUMNS <= value <= MAX_COLUMNS:
            raise ValueError("terminal columns are outside contract")
        return value

    @field_validator("rows")
    @classmethod
    def _validate_rows(cls, value: int) -> int:
        if not MIN_ROWS <= value <= MAX_ROWS:
            raise ValueError("terminal rows are outside contract")
        return value


_TerminalBytesInput = TypeVar(
    "_TerminalBytesInput",
    TerminalKeyInput,
    TerminalReplyInput,
    TerminalOutputInput,
)


def _validate_terminal_bytes_input(
    model_type: type[_TerminalBytesInput],
    data: object,
    *,
    label: str,
) -> _TerminalBytesInput:
    """Return one strict bytes model with a content-free type error."""
    try:
        return model_type.model_validate({"data": data})
    except PydanticValidationError:
        raise TypeError(f"terminal {label} data must be bytes") from None


def validate_terminal_key_input(data: object) -> TerminalKeyInput:
    """Validate encoded terminal key bytes through the shared model.

    Args:
        data: Candidate encoded key bytes.

    Returns:
        Strict validated key model.

    Raises:
        TypeError: If ``data`` is not immutable bytes.
    """
    return _validate_terminal_bytes_input(TerminalKeyInput, data, label="key")


def validate_terminal_paste_input(
    text: object, bracketed: object
) -> TerminalPasteInput:
    """Validate a terminal paste offer through the shared model.

    Args:
        text: Candidate paste text.
        bracketed: Candidate bracketed-paste mode flag.

    Returns:
        Strict validated paste model.

    Raises:
        TypeError: If either value has the wrong type.
    """
    try:
        return TerminalPasteInput.model_validate({"text": text, "bracketed": bracketed})
    except PydanticValidationError:
        raise TypeError("terminal paste values have invalid types") from None


def validate_terminal_reply_input(data: object) -> TerminalReplyInput:
    """Validate terminal reply bytes through the shared model.

    Args:
        data: Candidate terminal reply bytes.

    Returns:
        Strict validated reply model.

    Raises:
        TypeError: If ``data`` is not immutable bytes.
    """
    return _validate_terminal_bytes_input(TerminalReplyInput, data, label="reply")


def validate_terminal_output_input(data: object) -> TerminalOutputInput:
    """Validate terminal backend output through the shared model.

    Args:
        data: Candidate backend output bytes.

    Returns:
        Strict validated output model.

    Raises:
        TypeError: If ``data`` is not immutable bytes.
    """
    return _validate_terminal_bytes_input(TerminalOutputInput, data, label="output")


def validate_terminal_resize_input(
    columns: object, rows: object
) -> TerminalResizeInput:
    """Validate terminal dimensions through the shared model.

    Args:
        columns: Candidate terminal width.
        rows: Candidate terminal height.

    Returns:
        Strict validated resize model.

    Raises:
        TypeError: If either dimension is not an integer.
        ValueError: If either dimension is outside the terminal contract.
    """
    if type(columns) is not int or type(rows) is not int:
        raise TypeError("terminal dimensions must be integers")
    try:
        return TerminalResizeInput.model_validate({"columns": columns, "rows": rows})
    except PydanticValidationError:
        raise ValueError("terminal dimensions are outside contract") from None
