"""Immutable logical records for the Console semantic trace ledger."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from uuid import UUID, uuid4


_IDENTIFIER_TOKEN_MAX = 64
_IDENTIFIER_TOKEN_PATTERN = re.compile(
    r"[a-z][a-z0-9]*(?:[_-][a-z0-9]+)*\Z",
    re.ASCII,
)
MAX_SURFACE_REPLACEMENT_SPAN = 256


def _validate_opaque_id(value: str, field_name: str) -> None:
    if type(value) is not str:
        raise ValueError(f"{field_name} must be a canonical UUIDv4 string")
    try:
        parsed = UUID(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a canonical UUIDv4 string") from exc
    if parsed.version != 4 or str(parsed) != value:
        raise ValueError(f"{field_name} must be a canonical UUIDv4 string")


def _validate_identifier_token(value: str, field_name: str) -> None:
    if (
        type(value) is not str
        or len(value) > _IDENTIFIER_TOKEN_MAX
        or _IDENTIFIER_TOKEN_PATTERN.fullmatch(value) is None
    ):
        raise ValueError(
            f"{field_name} must be a lowercase ASCII identifier token of at most "
            f"{_IDENTIFIER_TOKEN_MAX} characters"
        )


def _validate_sequence(value: int, field_name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


def new_opaque_id() -> str:
    """Return a content-independent opaque UUID identifier."""

    return str(uuid4())


@dataclass(frozen=True, slots=True)
class TraceContentRef:
    """Opaque reference to content owned outside a logical trace record."""

    content_id: str
    content_kind: str

    def __post_init__(self) -> None:
        _validate_opaque_id(self.content_id, "content_id")
        if not self.content_kind:
            raise ValueError("content_kind must not be empty")


@dataclass(frozen=True, slots=True)
class TraceOmission:
    """Content-free disclosure that one semantic component is unavailable."""

    component_kind: str
    reason_code: str
    omission_id: str = field(default_factory=new_opaque_id)

    def __post_init__(self) -> None:
        _validate_opaque_id(self.omission_id, "omission_id")
        _validate_identifier_token(self.component_kind, "component_kind")
        _validate_identifier_token(self.reason_code, "reason_code")


@dataclass(frozen=True, slots=True)
class SemanticRevisionRef:
    """Opaque reference to one immutable provider-visible message revision."""

    revision_id: str

    def __post_init__(self) -> None:
        _validate_opaque_id(self.revision_id, "revision_id")


@dataclass(frozen=True, slots=True)
class SurfaceBoundary:
    """One immutable segment-local model-surface boundary."""

    segment_id: str
    sequence: int
    surface_head_id: str

    def __post_init__(self) -> None:
        _validate_opaque_id(self.segment_id, "segment_id")
        _validate_opaque_id(self.surface_head_id, "surface_head_id")
        _validate_sequence(self.sequence, "sequence")


@dataclass(frozen=True, slots=True)
class SurfaceReplacement:
    """One bounded contiguous replacement against a predecessor surface head."""

    predecessor_head_id: str
    start_node_id: str
    end_node_id: str
    start_sequence: int
    end_sequence: int
    replacement_node_id: str

    def __post_init__(self) -> None:
        for field_name in (
            "predecessor_head_id",
            "start_node_id",
            "end_node_id",
            "replacement_node_id",
        ):
            _validate_opaque_id(getattr(self, field_name), field_name)
        _validate_sequence(self.start_sequence, "start_sequence")
        _validate_sequence(self.end_sequence, "end_sequence")
        if self.end_sequence < self.start_sequence:
            raise ValueError("replacement range must be ordered")
        span = self.end_sequence - self.start_sequence + 1
        if span > MAX_SURFACE_REPLACEMENT_SPAN:
            raise ValueError(
                "replacement range may contain at most "
                f"{MAX_SURFACE_REPLACEMENT_SPAN} nodes"
            )


@dataclass(frozen=True, slots=True)
class FrozenTracePolicy:
    """Capture-time credential and PII policy frozen for one provider run."""

    policy_id: str
    credential_filter_version: str
    pii_redaction_enabled: bool
    pii_ruleset_revision_id: str | None

    def __post_init__(self) -> None:
        _validate_opaque_id(self.policy_id, "policy_id")
        if not self.credential_filter_version:
            raise ValueError("credential_filter_version must not be empty")
        if self.pii_redaction_enabled and self.pii_ruleset_revision_id is None:
            raise ValueError(
                "pii_ruleset_revision_id is required when PII redaction is enabled"
            )
        if self.pii_ruleset_revision_id is not None:
            _validate_opaque_id(
                self.pii_ruleset_revision_id,
                "pii_ruleset_revision_id",
            )


class TraceCallState(str, Enum):
    """Monotonic lifecycle state for one captured provider call."""

    RESERVED = "reserved"
    NOT_DISPATCHED = "not_dispatched"
    DISPATCH_STARTED = "dispatch_started"
    DISPATCH_UNKNOWN = "dispatch_unknown"
    RESPONSE_STARTED = "response_started"
    COMPLETE = "complete"
    STOPPED = "stopped"
    ERROR = "error"
    INTERRUPTED = "interrupted"
    ABANDONED = "abandoned"


class TraceOutcome(str, Enum):
    """Terminal outcome recorded for a provider call."""

    COMPLETE = "complete"
    STOPPED = "stopped"
    ERROR = "error"
    INTERRUPTED = "interrupted"
    ABANDONED = "abandoned"


class InvalidTraceTransition(ValueError):
    """Raised when a call lifecycle transition violates monotonic ordering."""


_PERMITTED_CALL_TRANSITIONS: dict[TraceCallState, frozenset[TraceCallState]] = {
    TraceCallState.RESERVED: frozenset(
        {TraceCallState.NOT_DISPATCHED, TraceCallState.DISPATCH_STARTED}
    ),
    TraceCallState.DISPATCH_STARTED: frozenset(
        {
            TraceCallState.DISPATCH_UNKNOWN,
            TraceCallState.RESPONSE_STARTED,
            TraceCallState.ERROR,
        }
    ),
    TraceCallState.RESPONSE_STARTED: frozenset(
        {
            TraceCallState.COMPLETE,
            TraceCallState.STOPPED,
            TraceCallState.ERROR,
            TraceCallState.INTERRUPTED,
        }
    ),
}
_TERMINAL_CALL_STATES = frozenset(
    {
        TraceCallState.NOT_DISPATCHED,
        TraceCallState.DISPATCH_UNKNOWN,
        TraceCallState.COMPLETE,
        TraceCallState.STOPPED,
        TraceCallState.ERROR,
        TraceCallState.INTERRUPTED,
        TraceCallState.ABANDONED,
    }
)


def is_terminal_call_state(state: TraceCallState) -> bool:
    """Return whether a call state rejects every later transition."""

    return state in _TERMINAL_CALL_STATES


def validate_call_transition(
    current: TraceCallState,
    target: TraceCallState,
    *,
    provider_operation_inactive: bool = False,
) -> TraceCallState:
    """Validate and return one permitted monotonic call transition.

    Args:
        current: Durable state before the transition.
        target: Proposed next durable state.
        provider_operation_inactive: Whether durable evidence proves that no provider
            operation remains live. Used only to authorize abandonment.

    Returns:
        The validated target state.

    Raises:
        InvalidTraceTransition: If the transition is backward, terminal, or invalid.
    """

    abandoned_with_evidence = (
        target is TraceCallState.ABANDONED
        and provider_operation_inactive
        and current is TraceCallState.DISPATCH_STARTED
    )
    if (
        target not in _PERMITTED_CALL_TRANSITIONS.get(current, frozenset())
        and not abandoned_with_evidence
    ):
        raise InvalidTraceTransition(
            f"invalid semantic trace transition: {current.value} -> {target.value}"
        )
    return target
