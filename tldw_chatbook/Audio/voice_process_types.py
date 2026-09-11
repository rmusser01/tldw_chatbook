"""Dependency-light values exchanged with the app-owned voice authority.

Handles identify originals retained in the parent; they never serialize provider
requests, turn contexts, tool arguments or promotion commits.
"""

from __future__ import annotations

from collections.abc import AsyncIterable, Awaitable
from dataclasses import dataclass
from enum import Enum
import re


RESPONSE_EAGERNESS_MIN_MS = 500
RESPONSE_EAGERNESS_MAX_MS = 3000
RESPONSE_EAGERNESS_DEFAULT_MS = 700


class VoiceSpeculationDecision(str, Enum):
    """Whether a frozen voice turn may enter the provisional gateway."""

    PROVISIONAL = "provisional"
    WAIT_FOR_STABLE_TURN = "waiting for stable turn"


class ControlKind(str, Enum):
    """Explicit actions that discard, rather than extend, a provisional turn."""

    STOP = "stop"
    ESCAPE = "escape"
    MICROPHONE_DISABLED = "microphone_disabled"
    HANDS_FREE_EXIT = "hands_free_exit"
    NAVIGATION = "navigation"
    TEARDOWN = "teardown"


class AttemptCleanupOutcome(str, Enum):
    """Terminal outcomes of one obsolete attempt cleanup."""

    CLEAN = "clean"
    FORCE_CLOSED = "force_closed"
    DETACHED = "detached"


class VoiceTerminalDisposition(str, Enum):
    """Categorical parent promotion result with no persistence authority."""

    PROMOTED = "promoted"
    RECOVERY = "recovery"
    FAILED = "failed"


class VoiceTransportFailure(RuntimeError):
    """Content-free terminal failure emitted by the audio owner."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class VoiceTranscriptCapacityError(RuntimeError):
    """Terminal signal that retained native PCM reached its hard bound."""

    def __init__(self) -> None:
        self.code = "transcript_capacity_exceeded"
        super().__init__(self.code)


class NormalizedPcmError(RuntimeError):
    """Content-free normalized-stream failure category."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class NormalizedPcmStream:
    """Normalized 48 kHz mono PCM16 frames plus their independent cleanup."""

    frames: AsyncIterable[bytes]
    cleanup: Awaitable[None]


def validate_attempt_epoch(value: int) -> None:
    """Reject non-integer and negative attempt identities."""

    if type(value) is not int or value < 0:
        raise ValueError("attempt_epoch must be a non-negative integer")


def _validate_handle(value: str) -> None:
    if type(value) is not str:
        raise TypeError("voice handle must be a string")
    if re.fullmatch(r"[0-9a-f]{32}", value) is None:
        raise ValueError("voice handle must contain 32 lowercase hexadecimal digits")


@dataclass(frozen=True, slots=True)
class VoiceRequestHandle:
    """Opaque identity of one original prepared provider request."""

    value: str

    def __post_init__(self) -> None:
        _validate_handle(self.value)


@dataclass(frozen=True, slots=True)
class VoiceTurnContextHandle:
    """Separate original context identity, retained through an accepted handoff."""

    value: str

    def __post_init__(self) -> None:
        _validate_handle(self.value)


@dataclass(frozen=True, slots=True)
class AttemptDispatchPrepared:
    """Validated parent policy and opaque references for one active attempt."""

    attempt_epoch: int
    decision: VoiceSpeculationDecision
    request_handle: VoiceRequestHandle
    turn_context_handle: VoiceTurnContextHandle

    def __post_init__(self) -> None:
        validate_attempt_epoch(self.attempt_epoch)
        if type(self.decision) is not VoiceSpeculationDecision:
            raise TypeError("decision must be a VoiceSpeculationDecision")
        if type(self.request_handle) is not VoiceRequestHandle:
            raise TypeError("request_handle must be a VoiceRequestHandle")
        if type(self.turn_context_handle) is not VoiceTurnContextHandle:
            raise TypeError("turn_context_handle must be a VoiceTurnContextHandle")


@dataclass(frozen=True, slots=True)
class AttemptToolPending:
    """Inert tool observation; the original arguments stay with the parent."""

    attempt_epoch: int

    def __post_init__(self) -> None:
        validate_attempt_epoch(self.attempt_epoch)


@dataclass(frozen=True, slots=True)
class ProviderAttemptFailed:
    """Content-free ordinary provider terminal failure."""

    attempt_epoch: int
    error_class: str

    def __post_init__(self) -> None:
        validate_attempt_epoch(self.attempt_epoch)
        if type(self.error_class) is not str:
            raise TypeError("error_class must be a string")
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", self.error_class) is None:
            raise ValueError("error_class must be a bounded categorical identifier")
