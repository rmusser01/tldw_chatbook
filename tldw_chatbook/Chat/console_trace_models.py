"""Immutable logical records for the Console semantic trace ledger."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Generic, TypeVar
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


def _validate_stable_opaque_id(value: str, field_name: str) -> None:
    """Accept random ledger IDs and repository-owned deterministic IDs."""

    if type(value) is not str:
        raise ValueError(f"{field_name} must be a canonical UUID string")
    try:
        parsed = UUID(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a canonical UUID string") from exc
    if parsed.version not in {4, 5} or str(parsed) != value:
        raise ValueError(f"{field_name} must be a canonical UUID string")


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
        _validate_stable_opaque_id(self.segment_id, "segment_id")
        _validate_stable_opaque_id(self.surface_head_id, "surface_head_id")
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
            _validate_stable_opaque_id(getattr(self, field_name), field_name)
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


class TraceReservationProvenance(str, Enum):
    """Why a durable call row exists before or after provider dispatch."""

    CRASH_DURABLE_RESERVED = "crash_durable_reserved"
    POST_DISPATCH_PROMOTED = "post_dispatch_promoted"


PROMOTED_VOICE_IMPORT_REASON = "provisional_voice_promoted"


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


# Pure sequence values and projection indexing; no repository ownership.


SurfaceReferenceKey = tuple[str, str, str]


class _SequenceNode:
    """One mutable node in an ephemeral implicit sequence tree."""

    __slots__ = (
        "sequence",
        "key",
        "priority",
        "left",
        "right",
        "parent",
        "size",
        "max_sequence",
    )

    def __init__(self, sequence: int, key: SurfaceReferenceKey) -> None:
        self.sequence = sequence
        self.key = key
        self.priority = _sequence_priority(sequence)
        self.left: _SequenceNode | None = None
        self.right: _SequenceNode | None = None
        self.parent: _SequenceNode | None = None
        self.size = 1
        self.max_sequence = sequence


def _sequence_priority(sequence: int) -> int:
    """Return a deterministic well-distributed treap priority."""

    mask = (1 << 64) - 1
    value = (sequence + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return value ^ (value >> 31)


def _sequence_size(node: _SequenceNode | None) -> int:
    return 0 if node is None else node.size


def _sequence_pull(node: _SequenceNode) -> None:
    node.size = 1 + _sequence_size(node.left) + _sequence_size(node.right)
    node.max_sequence = max(
        node.sequence,
        -1 if node.left is None else node.left.max_sequence,
        -1 if node.right is None else node.right.max_sequence,
    )
    if node.left is not None:
        node.left.parent = node
    if node.right is not None:
        node.right.parent = node


def _sequence_merge(
    left: _SequenceNode | None,
    right: _SequenceNode | None,
) -> _SequenceNode | None:
    if left is None:
        if right is not None:
            right.parent = None
        return right
    if right is None:
        left.parent = None
        return left
    if left.priority < right.priority:
        left.right = _sequence_merge(left.right, right)
        _sequence_pull(left)
        left.parent = None
        return left
    right.left = _sequence_merge(left, right.left)
    _sequence_pull(right)
    right.parent = None
    return right


def _sequence_split(
    root: _SequenceNode | None,
    count: int,
) -> tuple[_SequenceNode | None, _SequenceNode | None]:
    if root is None:
        return None, None
    if _sequence_size(root.left) >= count:
        left, root.left = _sequence_split(root.left, count)
        _sequence_pull(root)
        root.parent = None
        if left is not None:
            left.parent = None
        return left, root
    root.right, right = _sequence_split(
        root.right,
        count - _sequence_size(root.left) - 1,
    )
    _sequence_pull(root)
    root.parent = None
    if right is not None:
        right.parent = None
    return root, right


def _sequence_rank(node: _SequenceNode) -> int:
    rank = _sequence_size(node.left)
    current = node
    while current.parent is not None:
        parent = current.parent
        if current is parent.right:
            rank += _sequence_size(parent.left) + 1
        current = parent
    return rank


def _sequence_delete(
    root: _SequenceNode | None,
    node: _SequenceNode,
) -> _SequenceNode | None:
    position = _sequence_rank(node)
    left, remainder = _sequence_split(root, position)
    removed, right = _sequence_split(remainder, 1)
    if removed is not node:
        raise ValueError("surface_replacement_order")
    node.left = node.right = node.parent = None
    return _sequence_merge(left, right)


def _sequence_insert(
    root: _SequenceNode | None,
    position: int,
    node: _SequenceNode,
) -> _SequenceNode | None:
    left, right = _sequence_split(root, position)
    return _sequence_merge(_sequence_merge(left, node), right)


def _sequence_entries(
    root: _SequenceNode | None,
) -> tuple[tuple[int, SurfaceReferenceKey], ...]:
    result: list[tuple[int, SurfaceReferenceKey]] = []
    stack: list[_SequenceNode] = []
    current = root
    while current is not None or stack:
        while current is not None:
            stack.append(current)
            current = current.left
        current = stack.pop()
        result.append((current.sequence, current.key))
        current = current.right
    return tuple(result)


_SequenceValue = TypeVar("_SequenceValue")


@dataclass(frozen=True, slots=True)
class _PersistentSequenceNode(Generic[_SequenceValue]):
    """One path-copying implicit-treap node for live surface projections."""

    sequence: int
    value: _SequenceValue
    domain: str | None = None
    left: _PersistentSequenceNode[_SequenceValue] | None = None
    right: _PersistentSequenceNode[_SequenceValue] | None = None
    priority: int = field(init=False)
    size: int = field(init=False)
    min_sequence: int = field(init=False)
    max_sequence: int = field(init=False)
    message_count: int = field(init=False)
    continuation_count: int = field(init=False)

    def __post_init__(self) -> None:
        left = self.left
        right = self.right
        object.__setattr__(self, "priority", _sequence_priority(self.sequence))
        object.__setattr__(
            self,
            "size",
            1
            + (0 if left is None else left.size)
            + (0 if right is None else right.size),
        )
        object.__setattr__(
            self,
            "min_sequence",
            min(
                self.sequence,
                self.sequence if left is None else left.min_sequence,
                self.sequence if right is None else right.min_sequence,
            ),
        )
        object.__setattr__(
            self,
            "max_sequence",
            max(
                self.sequence,
                self.sequence if left is None else left.max_sequence,
                self.sequence if right is None else right.max_sequence,
            ),
        )
        object.__setattr__(
            self,
            "message_count",
            int(self.domain == "messages_payload")
            + (0 if left is None else left.message_count)
            + (0 if right is None else right.message_count),
        )
        object.__setattr__(
            self,
            "continuation_count",
            int(self.domain == "provider_continuations")
            + (0 if left is None else left.continuation_count)
            + (0 if right is None else right.continuation_count),
        )


def _persistent_size(
    root: _PersistentSequenceNode[_SequenceValue] | None,
) -> int:
    return 0 if root is None else root.size


def _persistent_merge(
    left: _PersistentSequenceNode[_SequenceValue] | None,
    right: _PersistentSequenceNode[_SequenceValue] | None,
) -> _PersistentSequenceNode[_SequenceValue] | None:
    if left is None:
        return right
    if right is None:
        return left
    if left.priority < right.priority:
        return _PersistentSequenceNode(
            left.sequence,
            left.value,
            left.domain,
            left.left,
            _persistent_merge(left.right, right),
        )
    return _PersistentSequenceNode(
        right.sequence,
        right.value,
        right.domain,
        _persistent_merge(left, right.left),
        right.right,
    )


def _persistent_split(
    root: _PersistentSequenceNode[_SequenceValue] | None,
    count: int,
) -> tuple[
    _PersistentSequenceNode[_SequenceValue] | None,
    _PersistentSequenceNode[_SequenceValue] | None,
]:
    if root is None:
        return None, None
    left_size = 0 if root.left is None else root.left.size
    if left_size >= count:
        left, remainder = _persistent_split(root.left, count)
        return left, _PersistentSequenceNode(
            root.sequence,
            root.value,
            root.domain,
            remainder,
            root.right,
        )
    remainder, right = _persistent_split(root.right, count - left_size - 1)
    return (
        _PersistentSequenceNode(
            root.sequence,
            root.value,
            root.domain,
            root.left,
            remainder,
        ),
        right,
    )


def _persistent_first_position_in_range(
    root: _PersistentSequenceNode[_SequenceValue] | None,
    start: int,
    end: int,
) -> int | None:
    if root is None or root.max_sequence < start or root.min_sequence > end:
        return None
    left = root.left
    left_position = _persistent_first_position_in_range(left, start, end)
    if left_position is not None:
        return left_position
    left_size = _persistent_size(left)
    if start <= root.sequence <= end:
        return left_size
    right_position = _persistent_first_position_in_range(root.right, start, end)
    if right_position is None:
        return None
    return left_size + 1 + right_position


def _persistent_without_sequence_range(
    root: _PersistentSequenceNode[_SequenceValue] | None,
    start: int,
    end: int,
) -> _PersistentSequenceNode[_SequenceValue] | None:
    if root is None or root.max_sequence < start or root.min_sequence > end:
        return root
    if start <= root.min_sequence and root.max_sequence <= end:
        return None
    left = _persistent_without_sequence_range(root.left, start, end)
    right = _persistent_without_sequence_range(root.right, start, end)
    if start <= root.sequence <= end:
        return _persistent_merge(left, right)
    if left is root.left and right is root.right:
        return root
    return _PersistentSequenceNode(
        root.sequence,
        root.value,
        root.domain,
        left,
        right,
    )


def _persistent_insert_entries(
    root: _PersistentSequenceNode[_SequenceValue] | None,
    position: int,
    entries: Iterable[tuple[int, _SequenceValue, str | None]],
) -> _PersistentSequenceNode[_SequenceValue] | None:
    inserted: _PersistentSequenceNode[_SequenceValue] | None = None
    for sequence, value, domain in entries:
        inserted = _persistent_merge(
            inserted,
            _PersistentSequenceNode(sequence, value, domain),
        )
    left, right = _persistent_split(root, position)
    return _persistent_merge(_persistent_merge(left, inserted), right)


def _persistent_iter(
    root: _PersistentSequenceNode[_SequenceValue] | None,
) -> Iterator[tuple[int, _SequenceValue, str | None]]:
    stack: list[_PersistentSequenceNode[_SequenceValue]] = []
    current = root
    while current is not None or stack:
        while current is not None:
            stack.append(current)
            current = current.left
        current = stack.pop()
        yield current.sequence, current.value, current.domain
        current = current.right


def _persistent_domain_count_before(
    root: _PersistentSequenceNode[_SequenceValue] | None,
    position: int,
    domain: str,
) -> int:
    count = 0
    current = root
    remaining = position
    while current is not None and remaining > 0:
        left_size = _persistent_size(current.left)
        if remaining <= left_size:
            current = current.left
            continue
        if current.left is not None:
            count += (
                current.left.message_count
                if domain == "messages_payload"
                else current.left.continuation_count
            )
        count += int(current.domain == domain)
        remaining -= left_size + 1
        current = current.right
    return count
