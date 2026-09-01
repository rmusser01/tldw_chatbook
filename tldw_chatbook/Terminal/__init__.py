"""Platform-neutral contracts for persistent terminal sessions."""

from typing import TYPE_CHECKING, Any

from .contracts import (
    MAX_IO_CHUNK_BYTES,
    MAX_PARSER_TURN_BYTES,
    MAX_PARSER_TURN_SECONDS,
    MAX_PASTE_BYTES,
    MAX_PENDING_INPUT_BYTES,
    MAX_PENDING_OUTPUT_BYTES,
    MAX_SCROLLBACK_BYTES,
    MAX_SCROLLBACK_LINES,
    MAX_SESSION_RECORDS,
    MAX_COLUMNS,
    MAX_ROWS,
    MIN_COLUMNS,
    MIN_ROWS,
    AdmissionGate,
    BackendIdentity,
    CleanupAttempt,
    CleanupProof,
    CleanupSchedule,
    TerminalEvent,
    TerminalLaunchRequest,
    TerminalLifecycle,
    TerminalProjection,
    TerminalReason,
    TerminalReceipt,
    apply_event,
    join_cleanup,
    retry_cleanup,
    slot_held,
    validate_transition,
)

if TYPE_CHECKING:
    from .backend import TerminalBackend


def __getattr__(name: str) -> Any:
    """Load the backend protocol only for callers that request it.

    Args:
        name: Package attribute requested by the caller.

    Returns:
        The lazily imported :class:`TerminalBackend` protocol.

    Raises:
        AttributeError: If ``name`` is not a supported lazy package export.
    """

    if name == "TerminalBackend":
        from .backend import TerminalBackend

        return TerminalBackend
    raise AttributeError(name)

__all__ = [
    "AdmissionGate",
    "BackendIdentity",
    "CleanupAttempt",
    "CleanupProof",
    "CleanupSchedule",
    "MAX_COLUMNS",
    "MAX_IO_CHUNK_BYTES",
    "MAX_PARSER_TURN_BYTES",
    "MAX_PARSER_TURN_SECONDS",
    "MAX_PASTE_BYTES",
    "MAX_PENDING_INPUT_BYTES",
    "MAX_PENDING_OUTPUT_BYTES",
    "MAX_ROWS",
    "MAX_SCROLLBACK_BYTES",
    "MAX_SCROLLBACK_LINES",
    "MAX_SESSION_RECORDS",
    "MIN_COLUMNS",
    "MIN_ROWS",
    "TerminalBackend",
    "TerminalEvent",
    "TerminalLaunchRequest",
    "TerminalLifecycle",
    "TerminalProjection",
    "TerminalReason",
    "TerminalReceipt",
    "apply_event",
    "join_cleanup",
    "retry_cleanup",
    "slot_held",
    "validate_transition",
]
