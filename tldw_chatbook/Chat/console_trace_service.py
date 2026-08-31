"""Persist bounded model surfaces and complete changed-only request headers."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Generic, Literal, TypeAlias, TypeVar, cast, overload
from weakref import ReferenceType, ref

from tldw_chatbook.Chat.console_prepared_request import freeze_json
from tldw_chatbook.Chat.console_semantic_revision import (
    project_semantic_revision_provider_continuations,
    project_semantic_revision_provider_message,
    project_semantic_revision_provider_messages,
)
from tldw_chatbook.Chat.console_trace_final_values import (
    _SURFACE_VERIFICATION_ISSUER,
    FinalValueBinding,
    ProviderCredentialSource,
    ProviderOverlayProvenance,
    ProviderRequestShadowBundle,
    SurfaceDeltaAdmission,
    VerifiedSurfaceDelta,
    VerifiedSurfaceDeltaItem,
    VerifiedSurfaceReplacement,
    VerifiedSurfaceReplacementRange,
    build_verified_surface_delta,
)
from tldw_chatbook.Chat.console_trace_models import (
    MAX_SURFACE_REPLACEMENT_SPAN,
    FrozenTracePolicy,
    SemanticRevisionRef,
    SurfaceReplacement,
    TraceCallState,
    TraceContentRef,
    TraceOmission,
)
from tldw_chatbook.Chat.console_trace_provenance import (
    DerivedTraceProvenance,
    OmittedTraceProvenance,
    ProviderArtifactTraceProvenance,
    ProviderRequestProvenance,
    RequestRouteTraceProvenance,
    SavedRevisionTraceProvenance,
    TraceOmissionReason,
    TraceProvenance,
    TraceProvenanceSource,
    TraceTransformKind,
    _project_verified_provider_request_provenance,
)
from tldw_chatbook.Chat.console_trace_redaction import (
    PII_DETECTOR_UNAVAILABLE,
    PIIRedactionSpan,
    redact_pii_value,
)
from tldw_chatbook.Chat.console_trace_repository import (
    ConsoleTraceRepository,
    HeaderComponentRef,
    RequestHeaderRecord,
    SurfaceNodeRecord,
    SurfaceReplacementRecord,
    TraceCallRecord,
    TraceEventType,
)
from tldw_chatbook.DB.transaction_observer import (
    current_managed_transaction,
    register_transaction_completion,
)

TRACE_VALUE_NORMALIZATION_VERSION = "canonical-json-v1"
TRACE_VALUE_MEDIA_TYPE = "application/json"
TRACE_CRITICAL_WRITE_WAL_AUTOCHECKPOINT_PAGES = 0


@contextmanager
def trace_critical_write_checkpoint_policy(database: object) -> Iterator[None]:
    """Keep checkpoint I/O out of a latency-critical trace write boundary.

    ChaChaNotes connections are thread-local, so the policy cannot retune
    ordinary writes on another thread.  The caller's connection setting is
    restored after the trace commit, reconciliation, or failure.
    """

    get_connection = getattr(database, "get_connection", None)
    if not callable(get_connection):
        yield
        return
    connection = get_connection()
    row = connection.execute("PRAGMA wal_autocheckpoint").fetchone()
    if row is None or type(row[0]) is not int or row[0] < 0:
        raise RuntimeError("wal_autocheckpoint is unavailable")
    previous_pages = row[0]
    connection.execute(
        f"PRAGMA wal_autocheckpoint={TRACE_CRITICAL_WRITE_WAL_AUTOCHECKPOINT_PAGES}"
    )
    try:
        yield
    finally:
        connection.execute(f"PRAGMA wal_autocheckpoint={previous_pages}")


_REASONING_KEYS = frozenset(
    {
        "reasoning_effort",
        "reasoning_summary",
        "verbosity",
        "thinking_effort",
        "thinking_budget_tokens",
    }
)
_GENERATION_KEYS = frozenset(
    {
        "api_mode",
        "frequency_penalty",
        "max_tokens",
        "maxp",
        "minp",
        "presence_penalty",
        "prompt_caching",
        "request_retries",
        "request_retry_delay",
        "request_timeout",
        "seed",
        "streaming",
        "temp",
        "topk",
        "topp",
    }
)


@dataclass(frozen=True, slots=True)
class PersistedTraceRequest:
    """Durable boundary material produced before call reservation/binding."""

    surface_head_id: str
    header: RequestHeaderRecord
    appended_nodes: tuple[SurfaceNodeRecord, ...]
    replacement: SurfaceReplacementRecord | None = None
    checkpoint: object | None = None


@dataclass(frozen=True, slots=True)
class TraceCallIdentity:
    """Immutable durable identity for one provider-call reservation."""

    owner_id: str
    segment_id: str
    turn_id: str
    run_id: str
    call_sequence: int
    idempotency_key: str
    policy_id: str


TraceCallReservationStatus: TypeAlias = Literal[
    "not_established", "established", "unknown"
]


class TraceCallPersistenceError(RuntimeError):
    """A content-free pre-dispatch trace write failure."""

    def __init__(
        self,
        *,
        boundary: object | None = None,
        reservation_status: TraceCallReservationStatus | None = None,
    ) -> None:
        super().__init__("Provider call trace persistence failed.")
        self.boundary = boundary
        self.reservation_status = reservation_status


@dataclass(slots=True)
class ConsoleTraceCallBoundary:
    """Own one reservation through its committed pre-adapter transition."""

    service: ConsoleTraceService
    database: object = field(repr=False)
    identity: TraceCallIdentity
    admission: SurfaceDeltaAdmission
    occurred_at_factory: Callable[[], str] = field(repr=False)
    surface_boundary: object | None = field(default=None, repr=False)
    _reserved: TraceCallRecord | None = field(default=None, init=False, repr=False)
    _started: TraceCallRecord | None = field(default=None, init=False, repr=False)
    _unknown: TraceCallRecord | None = field(default=None, init=False, repr=False)
    _response_started_at: str | None = field(default=None, init=False, repr=False)
    _settled: TraceCallState | None = field(default=None, init=False, repr=False)
    _settlement_fingerprint: str | None = field(default=None, init=False, repr=False)
    _pending_settlement_fingerprint: str | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _reservation_status: TraceCallReservationStatus = field(
        default="unknown", init=False, repr=False
    )

    @property
    def reservation_status(self) -> TraceCallReservationStatus:
        """Return the content-free durable outcome of reservation reconciliation."""

        return self._reservation_status

    @property
    def dispatch_started(self) -> bool:
        """Return whether this call crossed its normalized dispatch boundary."""

        return self._started is not None

    @property
    def preparation_identity(self) -> str:
        """Return the verifier identity admitted for this call."""

        return self.admission.preparation_identity

    def reserve(self) -> TraceCallRecord:
        """Commit or reconcile the immutable call identity once."""

        if self._reserved is None:
            try:
                self._reserved = self.service.reserve_call(self.database, self.identity)
            except TraceCallPersistenceError as exc:
                self._reservation_status = exc.reservation_status or "unknown"
                raise TraceCallPersistenceError(
                    boundary=self,
                    reservation_status=self._reservation_status,
                ) from None
            self._reservation_status = "established"
        return self._reserved

    def mark_dispatch_started(
        self,
        bundle: ProviderRequestShadowBundle,
        provenance: ProviderRequestProvenance,
    ) -> TraceCallRecord:
        """Persist the verified boundary and commit dispatch-start atomically."""

        if self._reserved is None or self._started is not None:
            raise TraceCallPersistenceError()
        try:
            projected = getattr(self.surface_boundary, "provenance", None)
            if projected is not None:
                if not isinstance(projected, ProviderRequestProvenance):
                    raise TypeError("surface provenance")
                provenance = projected
            surface_delta = build_verified_surface_delta(
                provenance,
                bundle,
                admission=self.admission,
            )
            self._started = self.service.bind_and_mark_dispatch(
                self.database,
                call_id=self._reserved.call_id,
                owner_id=self.identity.owner_id,
                segment_id=self.identity.segment_id,
                provenance=provenance,
                bundle=bundle,
                surface_delta=surface_delta,
                occurred_at=self.occurred_at_factory(),
            )
        except TraceCallPersistenceError:
            raise
        except Exception:
            raise TraceCallPersistenceError() from None
        return self._started

    def mark_dispatch_unknown(self) -> TraceCallRecord:
        """Record uncertainty when the final caller-owned checkpoint fails."""

        if self._unknown is not None:
            return self._unknown
        if self._started is None:
            raise TraceCallPersistenceError()
        try:
            with self.database.transaction(immediate=True) as cursor:  # type: ignore[attr-defined]
                self._unknown = self.service.repository.advance_call_state(
                    cursor,
                    call_id=self._started.call_id,
                    target=TraceCallState.DISPATCH_UNKNOWN,
                    occurred_at=self.occurred_at_factory(),
                    integrity_state=self._started.integrity_state,
                    omission_reason_code=self._started.omission_reason_code,
                )
        except Exception:
            raise TraceCallPersistenceError() from None
        return self._unknown

    def mark_not_dispatched(self) -> TraceCallRecord:
        """Terminally record an explicit cancel before adapter entry."""

        if self._reserved is None or self._started is not None:
            raise TraceCallPersistenceError(boundary=self)
        if self._reserved.state is TraceCallState.NOT_DISPATCHED:
            return self._reserved
        try:
            with self.database.transaction(immediate=True) as cursor:  # type: ignore[attr-defined]
                self._reserved = self.service.repository.advance_call_state(
                    cursor,
                    call_id=self._reserved.call_id,
                    target=TraceCallState.NOT_DISPATCHED,
                    occurred_at=self.occurred_at_factory(),
                    integrity_state=self._reserved.integrity_state,
                    omission_reason_code=self._reserved.omission_reason_code,
                )
        except Exception:
            raise TraceCallPersistenceError(boundary=self) from None
        return self._reserved

    def mark_response_started(self) -> bool:
        """Best-effort first-response checkpoint that cannot break a send."""

        if self._started is None or self._unknown is not None or self._settled:
            return False
        occurred_at = self._response_started_at or self.occurred_at_factory()
        try:
            self.service.mark_response_started(
                self.database,
                call_id=self._started.call_id,
                occurred_at=occurred_at,
            )
        except Exception:
            return False
        self._response_started_at = occurred_at
        return True

    def settle_response(
        self,
        response_envelope: object | None,
        outcome: TraceCallState,
        usage: Mapping[str, object] | None = None,
        *,
        canonical_message_id: str | None = None,
    ) -> bool:
        """Best-effort terminal seal; failures enter the bounded retry queue."""

        if self._started is None or self._unknown is not None:
            return False
        response_started_at = self._response_started_at or self.occurred_at_factory()
        settled_at = self.occurred_at_factory()
        try:
            fingerprint = self.service.settlement_fingerprint(
                call_id=self._started.call_id,
                outcome=outcome,
                response_envelope=response_envelope,
                usage=usage,
                response_started_at=response_started_at,
                settled_at=settled_at,
                canonical_message_id=canonical_message_id,
                prior_integrity_state=self._started.integrity_state,
                prior_omission_reason_code=self._started.omission_reason_code,
            )
            if self._settled is not None:
                return fingerprint == self._settlement_fingerprint
            if (
                self._pending_settlement_fingerprint is not None
                and fingerprint != self._pending_settlement_fingerprint
            ):
                return False
            submitted = self.service.submit_settlement(
                self.database,
                call_id=self._started.call_id,
                outcome=outcome,
                response_envelope=response_envelope,
                usage=usage,
                response_started_at=response_started_at,
                settled_at=settled_at,
                canonical_message_id=canonical_message_id,
                prior_integrity_state=self._started.integrity_state,
                prior_omission_reason_code=self._started.omission_reason_code,
            )
        except Exception:
            return False
        self._response_started_at = response_started_at
        if not submitted:
            self._pending_settlement_fingerprint = fingerprint
            return False
        self._settled = outcome
        self._settlement_fingerprint = fingerprint
        self._pending_settlement_fingerprint = None
        return True

    def prepare_response_settlement(
        self,
        response_envelope: object | None,
        outcome: TraceCallState,
        usage: Mapping[str, object] | None = None,
    ) -> object | None:
        """Return one sanitized handoff for canonical assistant persistence."""

        if self._started is None or self._unknown is not None:
            return None
        response_started_at = self._response_started_at or self.occurred_at_factory()
        try:
            handoff = self.service.prepare_settlement_handoff(
                self.database,
                call_id=self._started.call_id,
                outcome=outcome,
                response_envelope=response_envelope,
                usage=usage,
                response_started_at=response_started_at,
                settled_at=self.occurred_at_factory(),
                prior_integrity_state=self._started.integrity_state,
                prior_omission_reason_code=self._started.omission_reason_code,
            )
        except Exception:
            return None
        self._response_started_at = response_started_at
        return handoff


@dataclass(frozen=True, slots=True)
class ReconstructedHeaderComponent:
    """One resolved header artifact value."""

    component_kind: str
    ordinal: int
    value: object


@dataclass(frozen=True, slots=True)
class ReconstructedRequestHeader:
    """Complete logical non-history request envelope."""

    header_id: str
    provider_name: str
    model_name: str
    route_identity: str
    endpoint_identity: str
    generation_parameters: Mapping[str, object]
    adapter_defaults: Mapping[str, object]
    response_format: Mapping[str, object]
    reasoning_controls: Mapping[str, object]
    components: tuple[ReconstructedHeaderComponent, ...]
    system_revision_ids: tuple[str, ...]
    system_composition: tuple[Mapping[str, object], ...]


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


@dataclass(frozen=True, slots=True)
class _ProjectionRoot:
    parent: _ProjectionRoot | None
    base: tuple[tuple[int, SurfaceReferenceKey], ...] = ()
    appended: tuple[tuple[int, SurfaceReferenceKey], ...] = ()
    replacement: tuple[int, int] | None = None
    size: int = field(init=False)
    _tree: _PersistentSequenceNode[SurfaceReferenceKey] | None = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        tree = self.parent._tree if self.parent is not None else None
        if self.parent is None:
            tree = _persistent_insert_entries(
                None,
                0,
                ((sequence, key, None) for sequence, key in self.base),
            )
        elif self.replacement is None:
            tree = _persistent_insert_entries(
                tree,
                _persistent_size(tree),
                ((sequence, key, None) for sequence, key in self.appended),
            )
        else:
            start, end = self.replacement
            position = _persistent_first_position_in_range(tree, start, end)
            if position is None:
                raise ValueError("surface_replacement_target_inactive")
            tree = _persistent_without_sequence_range(tree, start, end)
            tree = _persistent_insert_entries(
                tree,
                position,
                ((sequence, key, None) for sequence, key in self.appended),
            )
        object.__setattr__(self, "_tree", tree)
        object.__setattr__(self, "size", _persistent_size(tree))
        object.__setattr__(self, "parent", None)

    def materialize(self) -> tuple[tuple[int, SurfaceReferenceKey], ...]:
        return tuple(self.iter_entries())

    def iter_entries(self) -> Iterator[tuple[int, SurfaceReferenceKey]]:
        for sequence, key, _ in _persistent_iter(self._tree):
            yield sequence, key

    def first_position_in_range(self, start: int, end: int) -> int | None:
        return _persistent_first_position_in_range(self._tree, start, end)


@dataclass(frozen=True, slots=True)
class _DescriptorRoot:
    """Private persistent structural projection; it never carries provider values."""

    parent: _DescriptorRoot | None
    base: tuple[tuple[int, TraceProvenance], ...] = ()
    appended: tuple[tuple[int, TraceProvenance], ...] = ()
    replacement: tuple[int, int] | None = None
    base_domains: tuple[tuple[int, str], ...] = ()
    appended_domains: tuple[tuple[int, str], ...] = ()
    removed_domain_counts: tuple[tuple[str, int], ...] = ()
    size: int = field(init=False)
    message_count: int = field(init=False)
    continuation_count: int = field(init=False)
    _tree: _PersistentSequenceNode[TraceProvenance] | None = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        tree = self.parent._tree if self.parent is not None else None
        if self.parent is None:
            base_domains = dict(self.base_domains)
            tree = _persistent_insert_entries(
                None,
                0,
                (
                    (sequence, descriptor, base_domains.get(sequence))
                    for sequence, descriptor in self.base
                ),
            )
        elif self.replacement is None:
            appended_domains = dict(self.appended_domains)
            tree = _persistent_insert_entries(
                tree,
                _persistent_size(tree),
                (
                    (sequence, descriptor, appended_domains.get(sequence))
                    for sequence, descriptor in self.appended
                ),
            )
        else:
            start, end = self.replacement
            position = _persistent_first_position_in_range(tree, start, end)
            if position is None:
                raise ValueError("surface_replacement_target_inactive")
            tree = _persistent_without_sequence_range(tree, start, end)
            appended_domains = dict(self.appended_domains)
            tree = _persistent_insert_entries(
                tree,
                position,
                (
                    (sequence, descriptor, appended_domains.get(sequence))
                    for sequence, descriptor in self.appended
                ),
            )
        object.__setattr__(self, "_tree", tree)
        object.__setattr__(self, "size", _persistent_size(tree))
        object.__setattr__(
            self,
            "message_count",
            0 if tree is None else tree.message_count,
        )
        object.__setattr__(
            self,
            "continuation_count",
            0 if tree is None else tree.continuation_count,
        )
        object.__setattr__(self, "parent", None)

    def domain_count(self, domain: str) -> int:
        if domain == "messages_payload":
            return self.message_count
        if domain == "provider_continuations":
            return self.continuation_count
        return 0

    def materialize(self) -> tuple[tuple[int, TraceProvenance], ...]:
        return tuple(self.iter_entries())

    def iter_entries(self) -> Iterator[tuple[int, TraceProvenance]]:
        for sequence, descriptor, _ in _persistent_iter(self._tree):
            yield sequence, descriptor

    def materialize_domains(self) -> tuple[tuple[int, str], ...]:
        return tuple(self.iter_domains())

    def iter_domains(self) -> Iterator[tuple[int, str]]:
        for sequence, _, domain in _persistent_iter(self._tree):
            if domain is not None:
                yield sequence, domain

    def iter_domain(self, domain: str) -> Iterator[tuple[int, TraceProvenance]]:
        for sequence, descriptor, entry_domain in _persistent_iter(self._tree):
            if entry_domain == domain:
                yield sequence, descriptor

    def domain_count_before(self, domain: str, position: int) -> int:
        return _persistent_domain_count_before(self._tree, position, domain)


class _ProjectedDescriptors(Sequence[TraceProvenance]):
    """Lazy view of a private persistent descriptor projection."""

    __slots__ = (
        "_root",
        "_domain",
        "_console_trace_delta",
        "_console_trace_delta_ordinal",
    )
    _console_trace_projection = True

    def __init__(
        self,
        root: _DescriptorRoot,
        domain: str,
        delta: tuple[TraceProvenance, ...],
        delta_ordinal: int,
    ) -> None:
        self._root = root
        self._domain = domain
        self._console_trace_delta = delta
        self._console_trace_delta_ordinal = delta_ordinal

    def __iter__(self) -> Iterator[TraceProvenance]:
        return (descriptor for _, descriptor in self._root.iter_domain(self._domain))

    def __len__(self) -> int:
        return self._root.domain_count(self._domain)

    @overload
    def __getitem__(self, index: int) -> TraceProvenance: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[TraceProvenance]: ...

    def __getitem__(
        self, index: int | slice
    ) -> TraceProvenance | Sequence[TraceProvenance]:
        values = tuple(self)
        return values[index]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Sequence):
            return False
        return tuple(self) == tuple(other)


class _ProjectedProviderValues(Sequence[object]):
    """Ephemeral values assembled from durable refs plus one admitted mutation."""

    __slots__ = (
        "_service",
        "_cursor",
        "_parent",
        "_appended",
        "_replacement",
        "_descriptor_root",
        "_owner_id",
        "_domain",
        "_cache",
    )

    def __init__(
        self,
        service: ConsoleTraceService,
        cursor: sqlite3.Cursor,
        parent: _ProjectionRoot,
        appended: tuple[tuple[int, TraceProvenance, object], ...],
        replacement: tuple[int, int] | None,
        descriptor_root: _DescriptorRoot,
        owner_id: str,
        domain: str,
    ) -> None:
        self._service = service
        self._cursor = cursor
        self._parent = parent
        self._appended = appended
        self._replacement = replacement
        self._descriptor_root = descriptor_root
        self._owner_id = owner_id
        self._domain = domain
        self._cache: tuple[object, ...] | None = None

    def __repr__(self) -> str:
        return f"<{type(self).__name__}>"

    def __iter__(self) -> Iterator[object]:
        if self._cache is not None:
            return iter(self._cache)
        parent = list(self._parent.iter_entries())
        inserted = tuple(
            (
                sequence,
                self._service._expected_descriptor_value(
                    self._cursor, self._owner_id, descriptor, value
                ),
            )
            for sequence, descriptor, value in self._appended
        )
        if self._replacement is None:
            combined: list[tuple[int, SurfaceReferenceKey | None, object | None]] = [
                (sequence, key, None) for sequence, key in parent
            ] + [(sequence, None, value) for sequence, value in inserted]
        else:
            start, end = self._replacement
            insert_at = self._parent.first_position_in_range(start, end)
            if insert_at is None:
                raise ValueError("surface_replacement_target_inactive")
            parent = [entry for entry in parent if not start <= entry[0] <= end]
            combined = [(sequence, key, None) for sequence, key in parent]
            combined[insert_at:insert_at] = [
                (sequence, None, value) for sequence, value in inserted
            ]
        domains = dict(self._descriptor_root.iter_domains())
        selected = tuple(
            (sequence, key, value)
            for sequence, key, value in combined
            if domains.get(sequence) == self._domain
        )
        resolved = self._service._resolve_reference_values(
            self._cursor,
            tuple(key for _, key, _ in selected if key is not None),
            owner_id=self._owner_id,
        )
        values: list[object] = []
        for _, key, value in selected:
            if key is None:
                values.append(value)
            else:
                values.append(resolved[key])
        self._cache = tuple(values)
        return iter(self._cache)

    def __len__(self) -> int:
        return self._descriptor_root.domain_count(self._domain)

    @overload
    def __getitem__(self, index: int) -> object: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[object]: ...

    def __getitem__(self, index: int | slice) -> object | Sequence[object]:
        return tuple(self)[index]


class _PreparedSurfaceBoundary:
    """One-shot content-bearing preparation outside capability registries."""

    __slots__ = (
        "_service",
        "provenance",
        "messages_payload",
        "provider_continuations",
        "_delta_values",
        "_issued_values",
    )

    def __init__(
        self,
        service: ConsoleTraceService,
        provenance: ProviderRequestProvenance,
        messages_payload: _ProjectedProviderValues,
        provider_continuations: _ProjectedProviderValues,
        delta_values: Mapping[str, tuple[object, ...]],
    ) -> None:
        self._service = service
        self.provenance = provenance
        self.messages_payload = messages_payload
        self.provider_continuations = provider_continuations
        self._delta_values = delta_values
        self._issued_values = MappingProxyType(
            {
                "messages_payload": tuple(
                    freeze_json(value) for value in messages_payload
                ),
                "provider_continuations": tuple(
                    freeze_json(value) for value in provider_continuations
                ),
            }
        )

    def __repr__(self) -> str:
        return f"<{type(self).__name__}>"

    def _verify_surface_values(
        self,
        provenance: ProviderRequestProvenance,
        actual_values: object,
        expected_values: object,
        issuer: object,
    ) -> bool:
        return self._service._verify_prepared_boundary(
            self,
            provenance,
            actual_values,
            expected_values,
            issuer,
        )

    def _provider_surface_values(self) -> Mapping[str, object]:
        return self._delta_values

    def _provider_request_surface_values(self) -> Mapping[str, object]:
        """Return the exact immutable surface values issued for provider dispatch."""

        return self._issued_values

    def _verify_raw_surface_values(
        self,
        provenance: ProviderRequestProvenance,
        actual_values: object,
        issuer: object,
    ) -> bool:
        if issuer is not _SURFACE_VERIFICATION_ISSUER:
            return False
        prepared = self._service._prepared_capabilities.get(id(provenance))
        if (
            prepared is None
            or prepared.provenance is not provenance
            or prepared.boundary_identity != id(self)
            or not isinstance(actual_values, Mapping)
        ):
            return False
        return all(
            actual_values.get(name) is value
            for name, value in self._issued_values.items()
        )

    def _bind_verified_bundle(
        self,
        provenance: ProviderRequestProvenance,
        bundle: ProviderRequestShadowBundle,
        issuer: object,
    ) -> None:
        self._service._bind_verified_bundle(self, provenance, bundle, issuer)

    def _extend_surface_projection(
        self,
        *,
        admission: object,
        replacement: VerifiedSurfaceReplacement | None,
        preparation_identity: str,
        items: tuple[VerifiedSurfaceDeltaItem, ...],
        bundle: ProviderRequestShadowBundle,
        surface_boundary_identity: int,
        provenance: ProviderRequestProvenance,
    ) -> object:
        return self._service._extend_prepared_surface_boundary(
            self,
            admission=admission,
            replacement=replacement,
            preparation_identity=preparation_identity,
            items=items,
            bundle=bundle,
            surface_boundary_identity=surface_boundary_identity,
            provenance=provenance,
        )


@dataclass(frozen=True, slots=True)
class _SurfaceProjection:
    head_id: str | None
    root: _ProjectionRoot
    checkpoint: object | None = None
    lineage_domains: tuple[tuple[int, str], ...] = ()

    @property
    def entries(self) -> tuple[tuple[int, SurfaceReferenceKey], ...]:
        return self.root.materialize()


@dataclass(frozen=True, slots=True)
class _ParentState:
    capability: object
    owner_id: str
    segment_id: str
    head_id: str | None
    route: str
    root: _ProjectionRoot
    descriptors: _DescriptorRoot
    next_sequence: int
    tool_loop: tuple[int, ...]
    active_domains: dict[int, str] = field(repr=False)
    lineage_domains: dict[int, str] = field(repr=False)
    live: bool = True
    surface_structure: Mapping[str, object] = field(default_factory=dict, repr=False)
    surface_policies: tuple[FrozenTracePolicy, ...] = ()


@dataclass(frozen=True, slots=True)
class _PreparedState:
    provenance: ProviderRequestProvenance
    parent: object
    admission: object
    descriptors: _DescriptorRoot
    boundary_identity: int
    items: tuple[VerifiedSurfaceDeltaItem, ...]
    verified: bool = False
    surface_structure: Mapping[str, object] | None = None
    surface_policies: tuple[FrozenTracePolicy, ...] = ()
    verified_bundle: ReferenceType[ProviderRequestShadowBundle] | None = None


@dataclass(frozen=True, slots=True)
class _ChildState:
    binding: object
    parent: object
    owner_id: str
    segment_id: str
    head_id: str | None
    route: str
    preparation_identity: str
    bundle: ReferenceType[ProviderRequestShadowBundle]
    provenance: ProviderRequestProvenance
    item_signature: tuple[tuple[str, int, TraceProvenance], ...]
    replacement: VerifiedSurfaceReplacement | None
    descriptors: _DescriptorRoot
    surface_structure: Mapping[str, object]
    surface_policies: tuple[FrozenTracePolicy, ...]


class _ParentSurfaceCapability:
    __slots__ = ("__service",)

    def __init__(self, service: ConsoleTraceService) -> None:
        self.__service = service

    def _extend_surface_projection(
        self,
        *,
        admission: object,
        replacement: VerifiedSurfaceReplacement | None,
        preparation_identity: str,
        items: tuple[VerifiedSurfaceDeltaItem, ...],
        bundle: ProviderRequestShadowBundle,
        surface_boundary_identity: int,
        provenance: ProviderRequestProvenance,
    ) -> object:
        return self.__service._extend_surface_capability(
            self,
            admission=admission,
            replacement=replacement,
            preparation_identity=preparation_identity,
            items=items,
            bundle=bundle,
            surface_boundary_identity=surface_boundary_identity,
            provenance=provenance,
        )


class _ChildSurfaceCapability:
    __slots__ = ()


class ConsoleTraceService:
    """Translate verified provider values into reference-backed trace records."""

    __slots__ = (
        "repository",
        "_surface_ref_cache",
        "_parent_capabilities",
        "_prepared_capabilities",
        "_child_capabilities",
        "_pending_child_uses",
        "_settlement",
    )

    def __init__(self, repository: ConsoleTraceRepository | None = None) -> None:
        self.repository = repository or ConsoleTraceRepository()
        # ponytail: one current immutable projection per segment; add LRU only if
        # long-lived services are shown to retain an excessive segment count.
        self._surface_ref_cache: dict[str, _SurfaceProjection] = {}
        self._parent_capabilities: dict[int, _ParentState] = {}
        self._prepared_capabilities: dict[int, _PreparedState] = {}
        self._child_capabilities: dict[int, _ChildState] = {}
        self._pending_child_uses: dict[int, object] = {}
        from tldw_chatbook.Chat.console_trace_settlement import (
            ConsoleTraceSettlementCoordinator,
        )

        self._settlement = ConsoleTraceSettlementCoordinator(self.repository)

    def mark_response_started(
        self,
        database: object,
        *,
        call_id: str,
        occurred_at: str,
    ) -> TraceCallRecord:
        """Record provider-response evidence through the shared coordinator."""

        return self._settlement.mark_response_started(
            database,
            call_id=call_id,
            occurred_at=occurred_at,
        )

    def submit_settlement(
        self,
        database: object,
        *,
        call_id: str,
        outcome: TraceCallState,
        response_envelope: object | None,
        usage: Mapping[str, object] | None,
        response_started_at: str,
        settled_at: str,
        canonical_message_id: str | None = None,
        prior_integrity_state: str = "pending",
        prior_omission_reason_code: str | None = None,
    ) -> bool:
        """Submit a sanitized post-dispatch seal without masking the result."""

        from tldw_chatbook.Chat.console_trace_settlement import TraceSettlementRequest

        return self._settlement.submit(
            database,
            TraceSettlementRequest(
                call_id=call_id,
                outcome=outcome,
                response_envelope=response_envelope,
                usage=usage,
                response_started_at=response_started_at,
                settled_at=settled_at,
                canonical_message_id=canonical_message_id,
                prior_integrity_state=prior_integrity_state,
                prior_omission_reason_code=prior_omission_reason_code,
            ),
        )

    def settlement_fingerprint(
        self,
        *,
        call_id: str,
        outcome: TraceCallState,
        response_envelope: object | None,
        usage: Mapping[str, object] | None,
        response_started_at: str,
        settled_at: str,
        canonical_message_id: str | None = None,
        prior_integrity_state: str = "pending",
        prior_omission_reason_code: str | None = None,
    ) -> str:
        """Return a content-free identity for one sanitized settlement signal."""

        from tldw_chatbook.Chat.console_trace_settlement import TraceSettlementRequest

        return self._settlement.fingerprint(
            TraceSettlementRequest(
                call_id=call_id,
                outcome=outcome,
                response_envelope=response_envelope,
                usage=usage,
                response_started_at=response_started_at,
                settled_at=settled_at,
                canonical_message_id=canonical_message_id,
                prior_integrity_state=prior_integrity_state,
                prior_omission_reason_code=prior_omission_reason_code,
            )
        )

    def prepare_settlement_handoff(
        self,
        database: object,
        *,
        call_id: str,
        outcome: TraceCallState,
        response_envelope: object | None,
        usage: Mapping[str, object] | None,
        response_started_at: str,
        settled_at: str,
        prior_integrity_state: str = "pending",
        prior_omission_reason_code: str | None = None,
    ) -> object:
        """Sanitize a terminal result before canonical persistence handoff."""

        from tldw_chatbook.Chat.console_trace_settlement import TraceSettlementRequest

        return self._settlement.prepare_handoff(
            database,
            TraceSettlementRequest(
                call_id=call_id,
                outcome=outcome,
                response_envelope=response_envelope,
                usage=usage,
                response_started_at=response_started_at,
                settled_at=settled_at,
                prior_integrity_state=prior_integrity_state,
                prior_omission_reason_code=prior_omission_reason_code,
            ),
        )

    def retry_pending_settlements(self) -> int:
        """Retry one bounded snapshot of failed post-dispatch seals."""

        return self._settlement.retry_pending()

    def recover_open_calls(
        self,
        database: object,
        *,
        occurred_at: str,
    ) -> tuple[TraceCallRecord, ...]:
        """Monotonically recover normalized calls left open by process death."""

        return self._settlement.recover_open_calls(
            database,
            occurred_at=occurred_at,
        )

    def reserve_call(
        self,
        database: object,
        identity: TraceCallIdentity,
    ) -> TraceCallRecord:
        """Commit one content-free call reservation before request persistence."""

        if type(identity) is not TraceCallIdentity:
            raise TypeError("identity")
        with trace_critical_write_checkpoint_policy(database):
            try:
                with database.transaction(immediate=True) as cursor:  # type: ignore[attr-defined]
                    return self.repository.reserve_call(
                        cursor,
                        owner_id=identity.owner_id,
                        segment_id=identity.segment_id,
                        turn_id=identity.turn_id,
                        run_id=identity.run_id,
                        call_sequence=identity.call_sequence,
                        idempotency_key=identity.idempotency_key,
                        policy_id=identity.policy_id,
                    )
            except Exception:
                # A transaction exit may report failure after SQLite committed. Query
                # both immutable identities before any later allocation.
                try:
                    with database.transaction(immediate=True) as cursor:  # type: ignore[attr-defined]
                        by_idempotency_key = (
                            self.repository.get_call_by_idempotency_key(
                                cursor,
                                identity.idempotency_key,
                            )
                        )
                        by_logical_identity = (
                            self.repository.get_call_by_logical_identity(
                                cursor,
                                owner_id=identity.owner_id,
                                segment_id=identity.segment_id,
                                turn_id=identity.turn_id,
                                run_id=identity.run_id,
                                call_sequence=identity.call_sequence,
                            )
                        )
                        if by_idempotency_key is None and by_logical_identity is None:
                            raise TraceCallPersistenceError(
                                reservation_status="not_established"
                            )
                        return self.repository.reserve_call(
                            cursor,
                            owner_id=identity.owner_id,
                            segment_id=identity.segment_id,
                            turn_id=identity.turn_id,
                            run_id=identity.run_id,
                            call_sequence=identity.call_sequence,
                            idempotency_key=identity.idempotency_key,
                            policy_id=identity.policy_id,
                        )
                except TraceCallPersistenceError:
                    raise
                except Exception:
                    raise TraceCallPersistenceError(
                        reservation_status="unknown"
                    ) from None

    def bind_and_mark_dispatch(
        self,
        database: object,
        *,
        call_id: str,
        owner_id: str,
        segment_id: str,
        provenance: ProviderRequestProvenance,
        bundle: ProviderRequestShadowBundle,
        surface_delta: VerifiedSurfaceDelta,
        occurred_at: str,
    ) -> TraceCallRecord:
        """Persist, bind, and start one reserved call in one transaction."""

        with trace_critical_write_checkpoint_policy(database):
            try:
                with database.transaction(immediate=True) as cursor:  # type: ignore[attr-defined]
                    persisted = self.persist_request(
                        cursor,
                        owner_id=owner_id,
                        segment_id=segment_id,
                        provenance=provenance,
                        bundle=bundle,
                        surface_delta=surface_delta,
                    )
                    self.repository.bind_call(
                        cursor,
                        call_id=call_id,
                        surface_node_id=persisted.surface_head_id,
                        request_header_id=persisted.header.header_id,
                        provider_name=persisted.header.provider_name,
                        model_name=persisted.header.model_name,
                        route_identity=persisted.header.route_identity,
                    )
                    return self.repository.advance_call_state(
                        cursor,
                        call_id=call_id,
                        target=TraceCallState.DISPATCH_STARTED,
                        occurred_at=occurred_at,
                        integrity_state=(
                            "complete" if bundle.available else "incomplete"
                        ),
                        omission_reason_code=(
                            None
                            if bundle.available or bundle.omission_reason is None
                            else bundle.omission_reason.value
                        ),
                    )
            except Exception:
                raise TraceCallPersistenceError() from None

    def persist_request(
        self,
        cursor: sqlite3.Cursor,
        *,
        owner_id: str,
        segment_id: str,
        provenance: ProviderRequestProvenance,
        bundle: ProviderRequestShadowBundle,
        surface_delta: VerifiedSurfaceDelta,
    ) -> PersistedTraceRequest:
        """Persist one verified request boundary in the caller-owned transaction."""

        if type(provenance) is not ProviderRequestProvenance:
            raise TypeError("provenance")
        if type(bundle) is not ProviderRequestShadowBundle:
            raise TypeError("bundle")
        if not cursor.connection.in_transaction:
            raise RuntimeError("caller_transaction_required")
        if type(surface_delta) is not VerifiedSurfaceDelta:
            raise TypeError("surface_delta")
        _validate_bundle(bundle)
        transaction_token = current_managed_transaction(cursor.connection)
        if bundle.available and transaction_token is None:
            raise RuntimeError("managed_transaction_required")
        if (
            surface_delta.owner_id != owner_id
            or surface_delta.segment_id != segment_id
            or bundle.preparation_identity != surface_delta.preparation_identity
            or _route(provenance) != surface_delta.route_identity
        ):
            raise ValueError("surface_delta_identity")
        self._validate_owner(cursor, owner_id=owner_id, segment_id=segment_id)
        tail = self._effective_surface_tail(cursor, segment_id)
        expected_predecessor = None if tail is None else tail.node_id
        for key, child in tuple(self._child_capabilities.items()):
            if (
                child.segment_id == segment_id
                and child.head_id != expected_predecessor
                and key not in self._pending_child_uses
            ):
                self._child_capabilities.pop(key, None)
                self._pending_child_uses.pop(key, None)
                self._parent_capabilities.pop(id(child.parent), None)
        pending_parent_ids = {
            id(child.parent)
            for key, child in self._child_capabilities.items()
            if key in self._pending_child_uses
        }
        for key, parent in tuple(self._parent_capabilities.items()):
            if (
                parent.segment_id == segment_id
                and parent.head_id != expected_predecessor
                and key not in pending_parent_ids
            ):
                self._parent_capabilities.pop(key, None)
        for key, prepared in tuple(self._prepared_capabilities.items()):
            if id(prepared.parent) not in self._parent_capabilities:
                self._prepared_capabilities.pop(key, None)
        if surface_delta.predecessor_surface_head_id != expected_predecessor:
            raise ValueError("surface_predecessor_mismatch")
        projection = self._surface_projection(cursor, segment_id, tail)
        replacement = surface_delta.replacement
        child_state = self._validated_child_binding(
            cursor,
            surface_delta,
            owner_id=owner_id,
            segment_id=segment_id,
            head_id=expected_predecessor,
            bundle=bundle,
            provenance=provenance,
        )
        if bundle.available and child_state is None:
            raise ValueError("surface_child_binding")
        delta_items = (
            (replacement.item,) if replacement is not None else surface_delta.items
        )
        descriptors, values = (
            self._resolve_child_items(bundle, delta_items)
            if child_state is not None
            else self._resolve_delta_items(provenance, bundle, delta_items)
        )
        saved_delta_pairs = tuple(
            (reference_key, value)
            for descriptor, value in zip(descriptors, values, strict=True)
            if (reference_key := _saved_reference_key(descriptor)) is not None
        )
        if saved_delta_pairs and child_state is None:
            expected_saved_values = self._resolve_reference_values(
                cursor,
                tuple(key for key, _ in saved_delta_pairs),
                owner_id=owner_id,
            )
            if any(
                _artifact_bytes(expected_saved_values[key]) != _artifact_bytes(value)
                for key, value in saved_delta_pairs
            ):
                raise ValueError("semantic_revision_value_mismatch")
        checkpoint_verified = child_state is not None
        if replacement is None and bundle.available and not checkpoint_verified:
            bindings = {item.name: item for item in bundle.components}
            message_values = bindings["messages_payload"].value
            continuation = bindings.get("provider_continuations")
            continuation_values = () if continuation is None else continuation.value
            if not isinstance(message_values, tuple) or not isinstance(
                continuation_values, tuple
            ):
                raise ValueError("surface_values_unavailable")
            domain_values = {
                "messages_payload": (
                    tuple(provenance.messages_payload),
                    message_values,
                ),
                "provider_continuations": (
                    tuple(provenance.continuations),
                    continuation_values,
                ),
            }
            consumed = {"messages_payload": 0, "provider_continuations": 0}
            current_prefix: list[SurfaceReferenceKey | None] = []
            durable_revision_values = self._resolve_reference_values(
                cursor,
                tuple(
                    durable_key
                    for _, durable_key in projection.entries
                    if durable_key[1] == "revision"
                ),
                owner_id=owner_id,
            )
            for _, durable_key in projection.entries:
                domain = _surface_reference_domain(durable_key)
                descriptors_for_domain, values_for_domain = domain_values[domain]
                ordinal = consumed[domain]
                if ordinal >= len(descriptors_for_domain) or ordinal >= len(
                    values_for_domain
                ):
                    raise ValueError("surface_prefix_mismatch")
                current_prefix.append(
                    self._reference_key(
                        cursor,
                        descriptors_for_domain[ordinal],
                        values_for_domain[ordinal],
                    )
                )
                if durable_key[1] == "revision" and _artifact_bytes(
                    values_for_domain[ordinal]
                ) != _artifact_bytes(durable_revision_values[durable_key]):
                    raise ValueError("surface_prefix_mismatch")
                consumed[domain] += 1
            if tuple(current_prefix) != tuple(key for _, key in projection.entries):
                raise ValueError("surface_prefix_mismatch")
            delta_ordinals = {
                domain: tuple(
                    item.ordinal
                    for item in delta_items
                    if item.component_name == domain
                )
                for domain in domain_values
            }
            for domain, (
                domain_descriptors,
                domain_provider_values,
            ) in domain_values.items():
                expected_ordinals = tuple(
                    range(consumed[domain], len(domain_descriptors))
                )
                if (
                    len(domain_provider_values) != len(domain_descriptors)
                    or delta_ordinals[domain] != expected_ordinals
                ):
                    raise ValueError("surface_delta_alignment")
        if replacement is not None and bundle.available and not checkpoint_verified:
            self._validate_replacement_projection(
                cursor,
                projection.entries,
                replacement,
                provenance,
                bundle,
                descriptors[0],
                values[0],
            )
        if replacement is not None:
            self._validate_replacement_range(
                cursor,
                segment_id=segment_id,
                tail=tail,
                plan=replacement,
                descriptor_count=1,
            )
        saved_system = (
            _saved_descriptors(provenance.system_message)
            if bundle.available and provenance.system_message is not None
            else ()
        )
        if len(saved_system) > MAX_SURFACE_REPLACEMENT_SPAN:
            raise ValueError("system_revision_span")
        for saved in saved_system:
            self._validate_revision_owner(cursor, owner_id, saved.revision_id)
        if bundle.available and provenance.system_message is not None:
            system_binding = _binding(bundle, "system_message")
            provider_value_count = len(bundle.system_components)
            if (
                provider_value_count == 0
                and not saved_system
                and system_binding is not None
            ):
                provider_value_count = 1
            self._validate_system_composition_shape(
                provenance.system_message,
                provider_value_count,
            )
            self._validate_saved_system_values(
                cursor,
                owner_id=owner_id,
                descriptor=provenance.system_message,
                bundle=bundle,
            )
        full_surface_descriptors = (
            ()
            if child_state is not None
            else (
                self._full_surface_values(provenance, bundle)[0]
                if bundle.available
                else descriptors
            )
        )
        header_policies = _policies(provenance, full_surface_descriptors)
        policy_candidates = (
            *(child_state.surface_policies if child_state is not None else ()),
            *header_policies,
        )
        policies = {policy.policy_id: policy for policy in policy_candidates}
        if any(
            candidate != policies[candidate.policy_id]
            for candidate in policy_candidates
        ):
            raise ValueError("trace_policy_mismatch")
        ordered_policies = tuple(policies[key] for key in sorted(policies))
        if len(ordered_policies) > 1:
            raise ValueError("trace_policy_mismatch")
        for policy in ordered_policies:
            self.repository.ensure_policy(cursor, policy)

        appended: list[SurfaceNodeRecord] = []
        stored_replacement: SurfaceReplacementRecord | None = None
        if replacement is not None:
            span = replacement.end_sequence - replacement.start_sequence + 1
            if span > MAX_SURFACE_REPLACEMENT_SPAN:
                node = self._append_omission(
                    cursor,
                    segment_id=segment_id,
                    tail=tail,
                    component_kind="surface_replacement",
                    reason=TraceOmissionReason.UNSUPPORTED_REPLACEMENT_SPAN,
                )
                appended.append(node)
                tail = node
            else:
                tail, stored_replacement = self._replace_surface(
                    cursor,
                    owner_id=owner_id,
                    segment_id=segment_id,
                    tail=tail,
                    descriptors=descriptors,
                    values=values,
                    plan=replacement,
                )
                appended.append(tail)
        else:
            for descriptor, value in zip(descriptors, values, strict=True):
                tail = self._append_descriptor(
                    cursor,
                    owner_id=owner_id,
                    segment_id=segment_id,
                    tail=tail,
                    descriptor=descriptor,
                    value=value,
                )
                appended.append(tail)
                self._append_event(
                    cursor,
                    segment_id=segment_id,
                    event_type="surface_append",
                    surface_node_id=tail.node_id,
                )
                if tail.reference_kind == "omission":
                    assert tail.omission_reason_code is not None
                    self._append_event(
                        cursor,
                        segment_id=segment_id,
                        event_type="gap",
                        omission_reason_code=tail.omission_reason_code,
                    )

        if tail is None:
            reason = bundle.omission_reason or TraceOmissionReason.SOURCE_UNAVAILABLE
            tail = self._append_omission(
                cursor,
                segment_id=segment_id,
                tail=None,
                component_kind="provider_request",
                reason=reason,
            )
            appended.append(tail)

        header = self._persist_header(
            cursor,
            provenance=provenance,
            bundle=bundle,
            surface_structure=(
                child_state.surface_structure
                if child_state is not None
                else _structural_provenance(full_surface_descriptors, {})
            ),
            artifact_policy_id=(
                ordered_policies[0].policy_id if ordered_policies else None
            ),
        )
        if child_state is not None:
            root = _ProjectionRoot(
                projection.root,
                appended=tuple(
                    (node.sequence, self._node_reference_key(node)) for node in appended
                ),
                replacement=(
                    (replacement.start_sequence, replacement.end_sequence)
                    if stored_replacement is not None and replacement is not None
                    else None
                ),
            )
        else:
            updated_entries = self._updated_projection_entries(
                projection.entries,
                appended,
                replacement if stored_replacement is not None else None,
            )
            root = _ProjectionRoot(None, base=updated_entries)
        descriptor_root = (
            child_state.descriptors
            if child_state is not None
            and (replacement is None or stored_replacement is not None)
            else (
                _DescriptorRoot(
                    None,
                    base=self._ordered_descriptor_entries(
                        root.materialize(), provenance
                    ),
                    base_domains=tuple(
                        (sequence, _surface_reference_domain(key))
                        for sequence, key in root.materialize()
                    ),
                )
                if root.size == len(full_surface_descriptors)
                else None
            )
        )
        active_domains: dict[int, str] | None = None
        lineage_domains: dict[int, str] | None = None
        if child_state is not None:
            parent_state = self._parent_capabilities[id(child_state.parent)]
            active_domains = dict(parent_state.active_domains)
            lineage_domains = dict(parent_state.lineage_domains)
            if stored_replacement is not None and replacement is not None:
                for sequence in range(
                    replacement.start_sequence, replacement.end_sequence + 1
                ):
                    active_domains.pop(sequence, None)
            for node in appended:
                domain = _surface_reference_domain(self._node_reference_key(node))
                active_domains[node.sequence] = domain
                lineage_domains[node.sequence] = domain
        else:
            lineage_domains = dict(projection.lineage_domains)
            for node in appended:
                lineage_domains[node.sequence] = _surface_reference_domain(
                    self._node_reference_key(node)
                )
        checkpoint = self._promote_surface_capability(
            owner_id=owner_id,
            segment_id=segment_id,
            head_id=tail.node_id,
            route=_route(provenance),
            root=root,
            descriptors=descriptor_root,
            next_sequence=tail.sequence + 1,
            tool_loop=tuple(provenance.tool_loop),
            active_domains=active_domains,
            lineage_domains=lineage_domains,
            surface_structure=(
                child_state.surface_structure
                if child_state is not None
                else _structural_provenance(full_surface_descriptors, {})
            ),
            surface_policies=(
                child_state.surface_policies
                if child_state is not None
                else _surface_projection_metadata(full_surface_descriptors)[1]
            ),
        )
        self._surface_ref_cache[segment_id] = _SurfaceProjection(
            tail.node_id,
            root,
            checkpoint,
            tuple((lineage_domains or {}).items()),
        )
        if child_state is not None:
            assert transaction_token is not None
            self._mark_child_used(
                cursor.connection,
                transaction_token,
                child_state,
            )
        return PersistedTraceRequest(
            surface_head_id=tail.node_id,
            header=header,
            appended_nodes=tuple(appended),
            replacement=stored_replacement,
            checkpoint=checkpoint,
        )

    @staticmethod
    def _ordered_descriptor_entries(
        entries: tuple[tuple[int, SurfaceReferenceKey], ...],
        provenance: ProviderRequestProvenance,
    ) -> tuple[tuple[int, TraceProvenance], ...]:
        descriptors = {
            "messages_payload": tuple(provenance.messages_payload),
            "provider_continuations": tuple(provenance.continuations),
        }
        consumed = {"messages_payload": 0, "provider_continuations": 0}
        ordered: list[tuple[int, TraceProvenance]] = []
        for sequence, key in entries:
            domain = _surface_reference_domain(key)
            ordinal = consumed[domain]
            if ordinal >= len(descriptors[domain]):
                raise ValueError("surface_provenance_mismatch")
            ordered.append((sequence, descriptors[domain][ordinal]))
            consumed[domain] += 1
        if any(
            consumed[domain] != len(values) for domain, values in descriptors.items()
        ):
            raise ValueError("surface_provenance_mismatch")
        return tuple(ordered)

    def current_surface_checkpoint(
        self, segment_id: str, *, expected_head_id: str | None = None
    ) -> object | None:
        """Return the transient checkpoint for the service's current segment head."""

        projection = self._surface_ref_cache.get(segment_id)
        if projection is None or (
            expected_head_id is not None and projection.head_id != expected_head_id
        ):
            return None
        return projection.checkpoint

    def prepare_current_surface_delta(
        self,
        cursor: sqlite3.Cursor,
        *,
        owner_id: str,
        segment_id: str,
        route_identity: str,
        preparation_identity: str,
        provenance: ProviderRequestProvenance,
        values: tuple[object, ...],
    ) -> tuple[SurfaceDeltaAdmission, object]:
        """Plan an append, no-op, or one-item bounded surface replacement.

        The comparison resolves prior references inside the caller transaction;
        no transcript-sized value is copied into the admission or call row.
        """

        descriptors = tuple(provenance.messages_payload) + tuple(
            provenance.continuations
        )
        if len(descriptors) != len(values):
            raise ValueError("surface_provenance_mismatch")
        domains = ("messages_payload",) * len(provenance.messages_payload) + (
            "provider_continuations",
        ) * len(provenance.continuations)
        tail = self._effective_surface_tail(cursor, segment_id)
        projection = self._surface_projection(cursor, segment_id, tail)
        active = projection.entries
        durable_keys = tuple(
            key for _, key in active if key[1] in {"artifact", "revision"}
        )
        durable_values = self._resolve_reference_values(
            cursor,
            durable_keys,
            owner_id=owner_id,
        )

        def matches(active_index: int, incoming_index: int) -> bool:
            key = active[active_index][1]
            return (
                _surface_reference_domain(key) == domains[incoming_index]
                and self._durable_reference_matches(
                    cursor,
                    descriptors[incoming_index],
                    values[incoming_index],
                    key,
                    durable_values,
                )
            )

        prefix = 0
        while prefix < min(len(active), len(descriptors)) and matches(prefix, prefix):
            prefix += 1

        replacement_range: VerifiedSurfaceReplacementRange | None = None
        admitted_from = prefix
        admitted_to = len(descriptors)
        if prefix < len(active):
            suffix = 0
            while (
                suffix < len(active) - prefix
                and suffix < len(descriptors) - prefix
                and matches(len(active) - 1 - suffix, len(descriptors) - 1 - suffix)
            ):
                suffix += 1
            incoming_changed = len(descriptors) - prefix - suffix
            active_changed = len(active) - prefix - suffix
            if incoming_changed != 1 or not 1 <= active_changed <= MAX_SURFACE_REPLACEMENT_SPAN:
                raise ValueError("unsupported_surface_change")
            changed_entries = active[prefix : len(active) - suffix]
            changed_sequences = {sequence for sequence, _key in changed_entries}
            start_sequence = min(changed_sequences)
            end_sequence = max(changed_sequences)
            if any(
                start_sequence <= sequence <= end_sequence
                and sequence not in changed_sequences
                for sequence, _key in active
            ):
                raise ValueError("unsupported_surface_change")
            surface_nodes = self._read_segment_surface_nodes(cursor, segment_id)
            nodes_by_sequence = {node.sequence: node for node in surface_nodes}
            start = self.repository.get_surface_node(
                cursor,
                nodes_by_sequence[start_sequence].node_id,
            )
            end = self.repository.get_surface_node(
                cursor,
                nodes_by_sequence[end_sequence].node_id,
            )
            if start is None or end is None or tail is None:
                raise ValueError("surface_replacement_target_unavailable")
            replacement_range = VerifiedSurfaceReplacementRange(
                predecessor_head_id=tail.node_id,
                start_node_id=start.node_id,
                end_node_id=end.node_id,
                start_sequence=start_sequence,
                end_sequence=end_sequence,
                current_ordinal=prefix,
                component_name=domains[prefix],
                component_ordinal=sum(
                    domain == domains[prefix] for domain in domains[:prefix]
                ),
            )
            admitted_to = prefix + 1

        predecessor = None if tail is None else tail.node_id
        checkpoint = self.current_surface_checkpoint(
            segment_id,
            expected_head_id=predecessor,
        )
        bootstrap = checkpoint is None and predecessor is not None
        admitted = descriptors[admitted_from:admitted_to]
        admission = SurfaceDeltaAdmission(
            owner_id=owner_id,
            segment_id=segment_id,
            predecessor_surface_head_id=predecessor,
            route_identity=route_identity,
            preparation_identity=preparation_identity,
            descriptors=admitted,
            projection_checkpoint=checkpoint,
            replacement_range=replacement_range,
        )
        if replacement_range is not None and bootstrap:
            raise ValueError("surface_replacement_checkpoint_unavailable")
        if bootstrap:
            delta_provenance = provenance
            delta_values = values
        else:
            message_delta = tuple(
                descriptor
                for index, descriptor in enumerate(descriptors[admitted_from:admitted_to], admitted_from)
                if domains[index] == "messages_payload"
            )
            continuation_delta = tuple(
                descriptor
                for index, descriptor in enumerate(descriptors[admitted_from:admitted_to], admitted_from)
                if domains[index] == "provider_continuations"
            )
            delta_provenance = replace(
                provenance,
                messages=message_delta,
                messages_payload=message_delta,
                continuations=continuation_delta,
                tool_loop=tuple(
                    index - admitted_from
                    for index in provenance.tool_loop
                    if admitted_from <= index < admitted_to
                ),
            )
            delta_values = values[admitted_from:admitted_to]
        boundary = self.prepare_surface_provenance(
            cursor,
            checkpoint,
            provenance=delta_provenance,
            admission=admission,
            values=delta_values,
        )
        return admission, boundary

    def _read_segment_surface_nodes(
        self,
        cursor: sqlite3.Cursor,
        segment_id: str,
    ) -> tuple[SurfaceNodeRecord, ...]:
        """Read a segment's bounded pages for replacement anchor lookup."""

        nodes: list[SurfaceNodeRecord] = []
        continuation = None
        while True:
            page = self.repository.read_surface_nodes(
                cursor,
                segment_id,
                after=continuation,
            )
            nodes.extend(page)
            if page.next_cursor is None:
                return tuple(nodes)
            continuation = page.next_cursor

    def _bootstrap_surface_parent(
        self,
        cursor: sqlite3.Cursor,
        *,
        owner_id: str,
        segment_id: str,
        tail: SurfaceNodeRecord,
        provenance: ProviderRequestProvenance,
        admitted: tuple[TraceProvenance, ...],
        values: tuple[object, ...],
    ) -> tuple[object, ProviderRequestProvenance, tuple[object, ...]]:
        projection = self._surface_projection(cursor, segment_id, tail)
        message_descriptors = tuple(provenance.messages_payload)
        continuation_descriptors = tuple(provenance.continuations)
        if len(values) != len(message_descriptors) + len(continuation_descriptors):
            raise ValueError("surface_prefix_mismatch")
        domain_values = {
            "messages_payload": (
                message_descriptors,
                values[: len(message_descriptors)],
            ),
            "provider_continuations": (
                continuation_descriptors,
                values[len(message_descriptors) :],
            ),
        }
        durable_values = self._resolve_reference_values(
            cursor,
            tuple(
                key
                for _, key in projection.entries
                if key[1] in {"artifact", "revision"}
            ),
            owner_id=owner_id,
        )
        consumed = {"messages_payload": 0, "provider_continuations": 0}
        prefix_descriptors: list[tuple[int, TraceProvenance]] = []
        prefix_domains: list[tuple[int, str]] = []
        for sequence, key in projection.entries:
            domain = _surface_reference_domain(key)
            descriptors, provider_values = domain_values[domain]
            ordinal = consumed[domain]
            if ordinal >= len(descriptors) or not self._durable_reference_matches(
                cursor,
                descriptors[ordinal],
                provider_values[ordinal],
                key,
                durable_values,
            ):
                raise ValueError("surface_prefix_mismatch")
            prefix_descriptors.append((sequence, descriptors[ordinal]))
            prefix_domains.append((sequence, domain))
            consumed[domain] += 1
        message_delta = message_descriptors[consumed["messages_payload"] :]
        continuation_delta = continuation_descriptors[
            consumed["provider_continuations"] :
        ]
        if message_delta + continuation_delta != admitted:
            raise ValueError("surface_delta_alignment")
        delta_values = (
            domain_values["messages_payload"][1][consumed["messages_payload"] :]
            + domain_values["provider_continuations"][1][
                consumed["provider_continuations"] :
            ]
        )
        descriptor_root = _DescriptorRoot(
            None,
            base=tuple(prefix_descriptors),
            base_domains=tuple(prefix_domains),
        )
        structure, policies = _surface_projection_metadata(
            descriptor for _, descriptor in prefix_descriptors
        )
        capability = self._promote_surface_capability(
            owner_id=owner_id,
            segment_id=segment_id,
            head_id=tail.node_id,
            route=_route(provenance),
            root=projection.root,
            descriptors=descriptor_root,
            next_sequence=tail.sequence + 1,
            tool_loop=tuple(
                index
                for index in provenance.tool_loop
                if index < consumed["messages_payload"]
            ),
            active_domains=dict(prefix_domains),
            lineage_domains=dict(projection.lineage_domains),
            surface_structure=structure,
            surface_policies=policies,
        )
        assert capability is not None
        delta_provenance = replace(
            provenance,
            messages=(
                message_delta
                if provenance.messages == provenance.messages_payload
                else provenance.messages
            ),
            messages_payload=message_delta,
            continuations=continuation_delta,
            tool_loop=tuple(
                index - consumed["messages_payload"]
                for index in provenance.tool_loop
                if index >= consumed["messages_payload"]
            ),
        )
        return capability, delta_provenance, tuple(delta_values)

    def prepare_surface_provenance(
        self,
        cursor: sqlite3.Cursor,
        capability: object | None,
        *,
        provenance: ProviderRequestProvenance,
        admission: object,
        values: tuple[object, ...],
    ) -> _PreparedSurfaceBoundary:
        """Derive one full structural projection from an opaque parent and delta.

        ``provenance`` is deliberately delta-only for the surface fields.  The
        returned object is the only provenance object that can extend this
        capability; callers cannot bless an arbitrary full prefix.
        """

        if type(admission) is not SurfaceDeltaAdmission:
            raise TypeError("admission")
        initial_parent = capability is None
        if capability is None:
            owner_id = str(getattr(admission, "owner_id", ""))
            segment_id = str(getattr(admission, "segment_id", ""))
            if getattr(admission, "projection_checkpoint", None) is not None:
                raise ValueError("surface_checkpoint_identity")
            self._validate_owner(
                cursor,
                owner_id=owner_id,
                segment_id=segment_id,
            )
            tail = self._effective_surface_tail(cursor, segment_id)
            expected_head = None if tail is None else tail.node_id
            if getattr(admission, "predecessor_surface_head_id", None) != expected_head:
                raise ValueError("surface_predecessor_mismatch")
            admitted = tuple(getattr(admission, "descriptors", ()))
            if tail is None:
                capability = self._promote_surface_capability(
                    owner_id=owner_id,
                    segment_id=segment_id,
                    head_id=None,
                    route=_route(provenance),
                    root=_ProjectionRoot(None),
                    descriptors=_DescriptorRoot(None),
                    next_sequence=0,
                    tool_loop=(),
                )
            else:
                capability, provenance, values = self._bootstrap_surface_parent(
                    cursor,
                    owner_id=owner_id,
                    segment_id=segment_id,
                    tail=tail,
                    provenance=provenance,
                    admitted=admitted,
                    values=values,
                )
            assert capability is not None
        parent = self._parent_capabilities.get(id(capability))
        admitted = tuple(getattr(admission, "descriptors", ()))
        replacement_range = getattr(admission, "replacement_range", None)
        if (
            parent is None
            or parent.capability is not capability
            or not parent.live
            or (
                not initial_parent
                and getattr(admission, "projection_checkpoint", None) is not capability
            )
            or getattr(admission, "owner_id", None) != parent.owner_id
            or getattr(admission, "segment_id", None) != parent.segment_id
            or getattr(admission, "predecessor_surface_head_id", None) != parent.head_id
            or getattr(admission, "route_identity", None) != parent.route
            or tuple(provenance.messages_payload) + tuple(provenance.continuations)
            != admitted
            or len(values) != len(admitted)
        ):
            raise ValueError("surface_checkpoint_identity")
        replacement_component_ordinal: int | None = None
        if replacement_range is not None:
            replacement_position = parent.root.first_position_in_range(
                replacement_range.start_sequence,
                replacement_range.end_sequence,
            )
            if replacement_position is None:
                raise ValueError("surface_replacement_target_inactive")
            replacement_component_ordinal = parent.descriptors.domain_count_before(
                replacement_range.component_name,
                replacement_position,
            )
            if replacement_range.current_ordinal != replacement_position or (
                replacement_range.component_ordinal is not None
                and replacement_range.component_ordinal != replacement_component_ordinal
            ):
                raise ValueError("replacement_component_ordinal")
            replacement_item = VerifiedSurfaceDeltaItem(
                replacement_range.component_name,
                replacement_component_ordinal,
                admitted[0],
            )
            self._validate_replacement_range(
                cursor,
                segment_id=parent.segment_id,
                tail=self._effective_surface_tail(cursor, parent.segment_id),
                plan=VerifiedSurfaceReplacement(
                    predecessor_head_id=replacement_range.predecessor_head_id,
                    start_node_id=replacement_range.start_node_id,
                    end_node_id=replacement_range.end_node_id,
                    start_sequence=replacement_range.start_sequence,
                    end_sequence=replacement_range.end_sequence,
                    current_ordinal=replacement_range.current_ordinal,
                    item=replacement_item,
                ),
                descriptor_count=len(admitted),
            )
        for descriptor in admitted:
            for saved in _saved_descriptors(descriptor):
                self._validate_revision_owner(
                    cursor, parent.owner_id, saved.revision_id
                )
        admitted_domains = ("messages_payload",) * len(provenance.messages_payload) + (
            "provider_continuations",
        ) * len(provenance.continuations)
        replaced_domain_counts: Counter[str] = Counter()
        if replacement_range is not None:
            if admitted_domains[0] != replacement_range.component_name:
                raise ValueError("replacement_component_domain")
            for sequence in range(
                replacement_range.start_sequence,
                replacement_range.end_sequence + 1,
            ):
                if parent.lineage_domains.get(sequence) != admitted_domains[0]:
                    raise ValueError("replacement_component_domain")
                active_domain = parent.active_domains.get(sequence)
                if active_domain is not None:
                    replaced_domain_counts[active_domain] += 1
        parent_domain_counts = {
            domain: parent.descriptors.domain_count(domain)
            for domain in {"messages_payload", "provider_continuations"}
        }
        prepared_items = tuple(
            VerifiedSurfaceDeltaItem(
                domain,
                parent_domain_counts[domain]
                + sum(previous == domain for previous in admitted_domains[:offset]),
                descriptor,
            )
            for offset, (domain, descriptor) in enumerate(
                zip(admitted_domains, admitted, strict=True)
            )
        )
        if replacement_range is not None:
            prepared_items = (
                replace(
                    prepared_items[0],
                    ordinal=cast(int, replacement_component_ordinal),
                ),
            )
        if replacement_range is None:
            appended = tuple(
                (parent.next_sequence + offset, descriptor)
                for offset, descriptor in enumerate(admitted)
            )
            descriptor_root = _DescriptorRoot(
                parent.descriptors,
                appended=appended,
                appended_domains=tuple(
                    (parent.next_sequence + offset, domain)
                    for offset, domain in enumerate(admitted_domains)
                ),
            )
        else:
            if len(admitted) != 1:
                raise ValueError("surface_delta_shape")
            descriptor_root = _DescriptorRoot(
                parent.descriptors,
                appended=((parent.next_sequence, admitted[0]),),
                replacement=(
                    int(getattr(replacement_range, "start_sequence")),
                    int(getattr(replacement_range, "end_sequence")),
                ),
                appended_domains=((parent.next_sequence, admitted_domains[0]),),
                removed_domain_counts=tuple(replaced_domain_counts.items()),
            )
        message_delta = tuple(provenance.messages_payload)
        continuation_delta = tuple(provenance.continuations)
        message_delta_ordinal = next(
            (
                item.ordinal
                for item in prepared_items
                if item.component_name == "messages_payload"
            ),
            parent_domain_counts["messages_payload"],
        )
        continuation_delta_ordinal = next(
            (
                item.ordinal
                for item in prepared_items
                if item.component_name == "provider_continuations"
            ),
            parent_domain_counts["provider_continuations"],
        )
        full_messages = _ProjectedDescriptors(
            descriptor_root,
            "messages_payload",
            message_delta,
            message_delta_ordinal,
        )
        full_continuations = _ProjectedDescriptors(
            descriptor_root,
            "provider_continuations",
            continuation_delta,
            continuation_delta_ordinal,
        )
        if replacement_range is None:
            parent_message_count = len(full_messages) - len(message_delta)
            projected_tool_loop = parent.tool_loop + tuple(
                parent_message_count + index for index in provenance.tool_loop
            )
        else:
            if admitted_domains[0] == "provider_continuations":
                if provenance.tool_loop:
                    raise ValueError("replacement_component_domain")
                projected_tool_loop = parent.tool_loop
            else:
                component_ordinal = cast(int, replacement_component_ordinal)
                replaced_count = replaced_domain_counts[admitted_domains[0]]
                projected_tool_loop = tuple(
                    index if index < component_ordinal else index - replaced_count + 1
                    for index in parent.tool_loop
                    if not component_ordinal
                    <= index
                    < component_ordinal + replaced_count
                )
                if provenance.tool_loop:
                    projected_tool_loop = tuple(
                        sorted((*projected_tool_loop, component_ordinal))
                    )
        projected = _project_verified_provider_request_provenance(
            provenance,
            messages=(
                full_messages
                if provenance.messages == provenance.messages_payload
                else provenance.messages
            ),
            messages_payload=full_messages,
            continuations=full_continuations,
            tool_loop=projected_tool_loop,
        )
        projected_values = tuple(
            (parent.next_sequence + offset, descriptor, value)
            for offset, (descriptor, value) in enumerate(
                zip(admitted, values, strict=True)
            )
        )
        replacement_span = (
            None
            if replacement_range is None
            else (
                replacement_range.start_sequence,
                replacement_range.end_sequence,
            )
        )
        message_values = _ProjectedProviderValues(
            self,
            cursor,
            parent.root,
            projected_values,
            replacement_span,
            descriptor_root,
            parent.owner_id,
            "messages_payload",
        )
        continuation_values = _ProjectedProviderValues(
            self,
            cursor,
            parent.root,
            projected_values,
            replacement_span,
            descriptor_root,
            parent.owner_id,
            "provider_continuations",
        )
        boundary = _PreparedSurfaceBoundary(
            self,
            projected,
            message_values,
            continuation_values,
            {
                domain: tuple(
                    value
                    for value, value_domain in zip(
                        values, admitted_domains, strict=True
                    )
                    if value_domain == domain
                )
                for domain in {"messages_payload", "provider_continuations"}
            },
        )
        self._prepared_capabilities[id(projected)] = _PreparedState(
            projected,
            capability,
            admission,
            descriptor_root,
            id(boundary),
            prepared_items,
        )
        return boundary

    def _extend_prepared_surface_boundary(
        self,
        boundary: object,
        *,
        admission: object,
        replacement: VerifiedSurfaceReplacement | None,
        preparation_identity: str,
        items: tuple[VerifiedSurfaceDeltaItem, ...],
        bundle: ProviderRequestShadowBundle,
        surface_boundary_identity: int,
        provenance: ProviderRequestProvenance,
    ) -> object:
        prepared = self._prepared_capabilities.get(id(provenance))
        if (
            prepared is None
            or prepared.provenance is not provenance
            or prepared.boundary_identity != id(boundary)
            or surface_boundary_identity != id(boundary)
        ):
            raise ValueError("surface_verified_bundle")
        return self._extend_surface_capability(
            prepared.parent,
            admission=admission,
            replacement=replacement,
            preparation_identity=preparation_identity,
            items=items,
            bundle=bundle,
            surface_boundary_identity=surface_boundary_identity,
            provenance=provenance,
        )

    def _verify_prepared_boundary(
        self,
        boundary: object,
        provenance: ProviderRequestProvenance,
        actual_values: object,
        expected_values: object,
        issuer: object,
    ) -> bool:
        prepared = self._prepared_capabilities.get(id(provenance))
        if (
            prepared is None
            or issuer is not _SURFACE_VERIFICATION_ISSUER
            or prepared.provenance is not provenance
            or prepared.boundary_identity != id(boundary)
            or not isinstance(actual_values, Mapping)
            or not isinstance(expected_values, Mapping)
        ):
            return False
        offsets = {"messages_payload": 0, "provider_continuations": 0}
        matches = True
        for item in prepared.items:
            actual_domain = actual_values.get(item.component_name)
            expected_domain = expected_values.get(item.component_name)
            offset = offsets[item.component_name]
            if (
                not isinstance(actual_domain, (list, tuple))
                or not isinstance(expected_domain, (list, tuple))
                or item.ordinal >= len(actual_domain)
                or offset >= len(expected_domain)
                or _artifact_bytes(actual_domain[item.ordinal])
                != _artifact_bytes(expected_domain[offset])
            ):
                matches = False
                break
            offsets[item.component_name] += 1
        matches = matches and all(
            offsets[domain]
            == len(cast(Sequence[object], expected_values.get(domain, ())))
            for domain in offsets
        )
        if matches:
            parent = self._parent_capabilities[id(prepared.parent)]
            delta_descriptors = tuple(item.provenance for item in prepared.items)
            structure = _structural_provenance(
                delta_descriptors,
                parent.surface_structure,
            )
            _, delta_policies = _surface_projection_metadata(delta_descriptors)
            policies_by_id = {
                policy.policy_id: policy
                for policy in (*parent.surface_policies, *delta_policies)
            }
            policies = tuple(policies_by_id[key] for key in sorted(policies_by_id))
            self._prepared_capabilities[id(provenance)] = replace(
                prepared,
                verified=True,
                surface_structure=structure,
                surface_policies=policies,
            )
        else:
            self._prepared_capabilities.pop(id(provenance), None)
        return matches

    def _bind_verified_bundle(
        self,
        boundary: object,
        provenance: ProviderRequestProvenance,
        bundle: ProviderRequestShadowBundle,
        issuer: object,
    ) -> None:
        prepared = self._prepared_capabilities.get(id(provenance))
        if (
            prepared is None
            or issuer is not _SURFACE_VERIFICATION_ISSUER
            or prepared.provenance is not provenance
            or prepared.boundary_identity != id(boundary)
            or not prepared.verified
            or prepared.verified_bundle is not None
        ):
            raise ValueError("surface_verified_bundle")
        self._prepared_capabilities[id(provenance)] = replace(
            prepared,
            verified_bundle=ref(bundle),
        )

    def _extend_surface_capability(
        self,
        capability: object,
        *,
        admission: object,
        replacement: VerifiedSurfaceReplacement | None,
        preparation_identity: str,
        items: tuple[VerifiedSurfaceDeltaItem, ...],
        bundle: ProviderRequestShadowBundle,
        surface_boundary_identity: int,
        provenance: ProviderRequestProvenance,
    ) -> object:
        if type(admission) is not SurfaceDeltaAdmission:
            raise TypeError("admission")
        parent = self._parent_capabilities.get(id(capability))
        prepared = self._prepared_capabilities.get(id(provenance))
        if (
            parent is None
            or parent.capability is not capability
            or not parent.live
            or getattr(admission, "owner_id", None) != parent.owner_id
            or getattr(admission, "segment_id", None) != parent.segment_id
            or getattr(admission, "predecessor_surface_head_id", None) != parent.head_id
            or getattr(admission, "route_identity", None) != parent.route
            or getattr(admission, "preparation_identity", None) != preparation_identity
        ):
            raise ValueError("surface_checkpoint_identity")
        if (
            prepared is None
            or prepared.provenance is not provenance
            or prepared.parent is not capability
            or prepared.admission is not admission
            or not prepared.verified
            or prepared.boundary_identity != surface_boundary_identity
            or prepared.verified_bundle is None
            or prepared.verified_bundle() is not bundle
        ):
            raise ValueError("surface_verified_bundle")
        replacement_range = admission.replacement_range
        if replacement_range is None:
            expected_items = prepared.items
            expected_replacement = None
        else:
            expected_items = prepared.items
            expected_replacement = VerifiedSurfaceReplacement(
                predecessor_head_id=replacement_range.predecessor_head_id,
                start_node_id=replacement_range.start_node_id,
                end_node_id=replacement_range.end_node_id,
                start_sequence=replacement_range.start_sequence,
                end_sequence=replacement_range.end_sequence,
                current_ordinal=replacement_range.current_ordinal,
                item=expected_items[0],
            )
        if items != expected_items or replacement != expected_replacement:
            raise ValueError("surface_prefix_mismatch")
        binding = _ChildSurfaceCapability()
        state = _ChildState(
            binding,
            capability,
            parent.owner_id,
            parent.segment_id,
            parent.head_id,
            str(getattr(admission, "route_identity")),
            preparation_identity,
            ref(bundle),
            provenance,
            tuple(
                (item.component_name, item.ordinal, item.provenance) for item in items
            ),
            replacement,
            prepared.descriptors,
            prepared.surface_structure or _structural_provenance((), {}),
            prepared.surface_policies,
        )
        self._child_capabilities[id(binding)] = state
        self._prepared_capabilities.pop(id(provenance), None)
        return binding

    def _validated_child_binding(
        self,
        cursor: sqlite3.Cursor,
        delta: VerifiedSurfaceDelta,
        *,
        owner_id: str,
        segment_id: str,
        head_id: str | None,
        bundle: ProviderRequestShadowBundle,
        provenance: ProviderRequestProvenance,
    ) -> _ChildState | None:
        binding = delta.child_binding
        if binding is None:
            return None
        child = self._child_capabilities.get(id(binding))
        items = (
            (delta.replacement.item,) if delta.replacement is not None else delta.items
        )
        signature = tuple(
            (item.component_name, item.ordinal, item.provenance) for item in items
        )
        if (
            child is None
            or id(binding) in self._pending_child_uses
            or child.binding is not binding
            or child.owner_id != owner_id
            or child.segment_id != segment_id
            or child.head_id != head_id
            or child.route != delta.route_identity
            or child.preparation_identity != delta.preparation_identity
            or child.bundle() is not bundle
            or child.provenance is not provenance
            or child.item_signature != signature
            or child.replacement != delta.replacement
        ):
            raise ValueError("surface_child_binding")
        return child

    def _mark_child_used(
        self,
        connection: sqlite3.Connection,
        transaction_token: object,
        child: _ChildState,
    ) -> None:
        key = id(child.binding)
        if key in self._pending_child_uses:
            raise ValueError("surface_child_binding")
        self._pending_child_uses[key] = transaction_token

        def complete(committed: bool | None) -> None:
            if self._pending_child_uses.get(key) is not transaction_token:
                return
            self._pending_child_uses.pop(key, None)
            if committed is not False:
                self._child_capabilities.pop(key, None)
                self._prune_unreferenced_parents()

        try:
            register_transaction_completion(
                connection,
                transaction_token,
                complete,
            )
        except Exception:
            self._pending_child_uses.pop(key, None)
            raise

    def _prune_unreferenced_parents(self) -> None:
        retained = (
            {id(child.parent) for child in self._child_capabilities.values()}
            | {id(prepared.parent) for prepared in self._prepared_capabilities.values()}
            | {
                id(projection.checkpoint)
                for projection in self._surface_ref_cache.values()
                if projection.checkpoint is not None
            }
        )
        for key in tuple(self._parent_capabilities):
            if key not in retained:
                self._parent_capabilities.pop(key, None)

    def _promote_surface_capability(
        self,
        *,
        owner_id: str,
        segment_id: str,
        head_id: str | None,
        route: str,
        root: _ProjectionRoot,
        descriptors: _DescriptorRoot | None,
        next_sequence: int,
        tool_loop: tuple[int, ...],
        active_domains: dict[int, str] | None = None,
        lineage_domains: dict[int, str] | None = None,
        surface_structure: Mapping[str, object] | None = None,
        surface_policies: tuple[FrozenTracePolicy, ...] = (),
    ) -> object | None:
        if descriptors is None:
            return None
        capability = _ParentSurfaceCapability(self)
        current_domains = (
            dict(descriptors.materialize_domains())
            if active_domains is None
            else active_domains
        )
        self._parent_capabilities[id(capability)] = _ParentState(
            capability,
            owner_id,
            segment_id,
            head_id,
            route,
            root,
            descriptors,
            next_sequence,
            tool_loop,
            current_domains,
            dict(current_domains) if lineage_domains is None else lineage_domains,
            True,
            {} if surface_structure is None else surface_structure,
            surface_policies,
        )
        return capability

    def _effective_surface_tail(
        self,
        cursor: sqlite3.Cursor,
        segment_id: str,
    ) -> SurfaceNodeRecord | None:
        """Return the segment-local tail or its immutable inherited head."""

        tail = self.repository.get_surface_tail(cursor, segment_id)
        if tail is not None:
            return tail
        segment = self.repository.get_segment(cursor, segment_id)
        if segment is None or segment.inherited_surface_head_id is None:
            return None
        inherited = self.repository.get_surface_node(
            cursor,
            segment.inherited_surface_head_id,
        )
        if inherited is None:
            raise ValueError("inherited_surface_head_unavailable")
        return inherited

    def _surface_projection(
        self,
        cursor: sqlite3.Cursor,
        segment_id: str,
        tail: SurfaceNodeRecord | None,
    ) -> _SurfaceProjection:
        segment = self.repository.get_segment(cursor, segment_id)
        if segment is None:
            raise ValueError("surface_segment_unavailable")
        inherited_projection: _SurfaceProjection | None = None
        if (
            segment.parent_segment_id is not None
            and segment.inherited_surface_head_id is not None
        ):
            inherited_head = self.repository.get_surface_node(
                cursor,
                segment.inherited_surface_head_id,
            )
            if inherited_head is None:
                raise ValueError("inherited_surface_head_unavailable")
            inherited_projection = self._surface_projection(
                cursor,
                segment.parent_segment_id,
                inherited_head,
            )
            if tail is None:
                tail = inherited_head
        head_id = None if tail is None else tail.node_id
        cached = self._surface_ref_cache.get(segment_id)
        if cached is not None and cached.head_id == head_id:
            return cached
        nodes: list[SurfaceNodeRecord] = []
        continuation = None
        while True:
            page = self.repository.read_surface_nodes(
                cursor,
                segment_id,
                after=continuation,
            )
            nodes.extend(page)
            if page.next_cursor is None:
                break
            continuation = page.next_cursor
        if tail is not None:
            nodes = [node for node in nodes if node.sequence <= tail.sequence]
        local_node_ids = {node.node_id for node in nodes}
        replacements = tuple(
            record
            for record in self.repository.read_surface_replacements(
                cursor,
                segment_id,
            )
            if record.replacement.replacement_node_id in local_node_ids
        )
        replacement_ids = {
            item.replacement.replacement_node_id for item in replacements
        }
        root_node: _SequenceNode | None = None
        active: dict[int, _SequenceNode] = {}
        if inherited_projection is not None:
            for sequence, key in inherited_projection.entries:
                sequence_node = _SequenceNode(sequence, key)
                root_node = _sequence_merge(root_node, sequence_node)
                active[sequence] = sequence_node
        for node in nodes:
            if node.node_id in replacement_ids:
                continue
            sequence_node = _SequenceNode(
                node.sequence,
                self._node_reference_key(node),
            )
            root_node = _sequence_merge(root_node, sequence_node)
            active[node.sequence] = sequence_node
        by_id = {node.node_id: node for node in nodes}
        for record in replacements:
            replacement = record.replacement
            span = replacement.end_sequence - replacement.start_sequence + 1
            if span > MAX_SURFACE_REPLACEMENT_SPAN:
                raise ValueError("surface_replacement_span")
            node = by_id[replacement.replacement_node_id]
            anchors = tuple(
                active[sequence]
                for sequence in range(
                    replacement.start_sequence,
                    replacement.end_sequence + 1,
                )
                if sequence in active
            )
            if not anchors:
                raise ValueError("surface_replacement_target_inactive")
            insert_at = min(_sequence_rank(anchor) for anchor in anchors)
            for sequence in range(
                replacement.start_sequence,
                replacement.end_sequence + 1,
            ):
                removed = active.pop(sequence, None)
                if removed is not None:
                    root_node = _sequence_delete(root_node, removed)
            sequence_node = _SequenceNode(
                node.sequence,
                self._node_reference_key(node),
            )
            root_node = _sequence_insert(root_node, insert_at, sequence_node)
            active[node.sequence] = sequence_node
        entries = _sequence_entries(root_node)
        root = _ProjectionRoot(None, base=tuple(entries))
        inherited_lineage = (
            ()
            if inherited_projection is None
            else inherited_projection.lineage_domains
        )
        projection = _SurfaceProjection(
            head_id,
            root,
            lineage_domains=inherited_lineage
            + tuple(
                (
                    node.sequence,
                    _surface_reference_domain(self._node_reference_key(node)),
                )
                for node in nodes
            ),
        )
        self._surface_ref_cache[segment_id] = projection
        return projection

    @staticmethod
    def _updated_projection_entries(
        entries: tuple[tuple[int, SurfaceReferenceKey], ...],
        appended: list[SurfaceNodeRecord],
        replacement: VerifiedSurfaceReplacement | None,
    ) -> tuple[tuple[int, SurfaceReferenceKey], ...]:
        result = list(entries)
        for node in appended:
            key = ConsoleTraceService._node_reference_key(node)
            if replacement is None or node.node_id != appended[-1].node_id:
                result.append((node.sequence, key))
                continue
            insert_at = ConsoleTraceService._active_replacement_position(
                result,
                replacement,
            )
            result = [
                entry
                for entry in result
                if not replacement.start_sequence
                <= entry[0]
                <= replacement.end_sequence
            ]
            result.insert(insert_at, (node.sequence, key))
        return tuple(result)

    def _validate_replacement_projection(
        self,
        cursor: sqlite3.Cursor,
        entries: tuple[tuple[int, SurfaceReferenceKey], ...],
        replacement: VerifiedSurfaceReplacement,
        provenance: ProviderRequestProvenance,
        bundle: ProviderRequestShadowBundle,
        replacement_descriptor: TraceProvenance,
        replacement_value: object,
    ) -> None:
        bindings = {item.name: item for item in bundle.components}
        message_binding = bindings.get("messages_payload")
        continuation_binding = bindings.get("provider_continuations")
        if message_binding is None or not isinstance(message_binding.value, tuple):
            raise ValueError("surface_values_unavailable")
        continuation_values = (
            () if continuation_binding is None else continuation_binding.value
        )
        if not isinstance(continuation_values, tuple):
            raise ValueError("surface_values_unavailable")
        domain_values = {
            "messages_payload": (
                tuple(provenance.messages_payload),
                message_binding.value,
            ),
            "provider_continuations": (
                tuple(provenance.continuations),
                continuation_values,
            ),
        }
        expected: list[tuple[int, SurfaceReferenceKey | None]] = list(entries)
        insert_at = self._active_replacement_position(
            expected,
            replacement,
        )
        expected = [
            entry
            for entry in expected
            if not replacement.start_sequence <= entry[0] <= replacement.end_sequence
        ]
        expected.insert(insert_at, (-1, None))
        if (
            len(expected)
            != sum(len(descriptors) for descriptors, _ in domain_values.values())
            or replacement.current_ordinal != insert_at
        ):
            raise ValueError("surface_prefix_mismatch")
        consumed = {"messages_payload": 0, "provider_continuations": 0}
        for index, (_, key) in enumerate(expected):
            domain = (
                replacement.item.component_name
                if index == insert_at
                else _surface_reference_domain(cast(SurfaceReferenceKey, key))
            )
            descriptors, values = domain_values[domain]
            ordinal = consumed[domain]
            if ordinal >= len(descriptors) or ordinal >= len(values):
                raise ValueError("surface_prefix_mismatch")
            descriptor = descriptors[ordinal]
            value = values[ordinal]
            if index == insert_at:
                if (
                    replacement.item.ordinal != ordinal
                    or descriptor != replacement_descriptor
                    or _artifact_bytes(value) != _artifact_bytes(replacement_value)
                ):
                    raise ValueError("surface_prefix_mismatch")
            elif self._reference_key(cursor, descriptor, value) != key:
                raise ValueError("surface_prefix_mismatch")
            consumed[domain] += 1
        if any(
            consumed[domain] != len(descriptors) or len(descriptors) != len(values)
            for domain, (descriptors, values) in domain_values.items()
        ):
            raise ValueError("surface_prefix_mismatch")

    @staticmethod
    def _active_replacement_position(
        entries: Sequence[tuple[int, object]],
        replacement: VerifiedSurfaceReplacement,
    ) -> int:
        position = next(
            (
                index
                for index, (sequence, _) in enumerate(entries)
                if replacement.start_sequence <= sequence <= replacement.end_sequence
            ),
            None,
        )
        if position is None:
            raise ValueError("surface_replacement_target_inactive")
        return position

    @staticmethod
    def _node_reference_key(node: SurfaceNodeRecord) -> SurfaceReferenceKey:
        identity = (
            node.semantic_revision_id or node.artifact_id or node.omission_reason_code
        )
        if identity is None:
            raise ValueError("surface_reference")
        return node.component_kind, node.reference_kind, identity

    def reconstruct_header(
        self,
        cursor: sqlite3.Cursor,
        header_id: str,
    ) -> ReconstructedRequestHeader:
        """Read the complete immutable header selected for a request."""

        header = self.repository.get_request_header(cursor, header_id)
        if header is None:
            raise ValueError("request_header_unavailable")
        components: list[ReconstructedHeaderComponent] = []
        for component in header.components:
            artifact = self.repository.get_artifact(cursor, component.artifact_id)
            if (
                artifact is None
                or artifact.media_type != TRACE_VALUE_MEDIA_TYPE
                or artifact.normalization_version != TRACE_VALUE_NORMALIZATION_VERSION
            ):
                raise ValueError("request_header_component_unavailable")
            try:
                value = json.loads(artifact.sanitized_bytes)
            except (TypeError, ValueError, UnicodeDecodeError) as exc:
                raise ValueError("request_header_component_invalid") from exc
            components.append(
                ReconstructedHeaderComponent(
                    component.component_kind,
                    component.ordinal,
                    value,
                )
            )
        composition = self._resolve_system_composition(header)
        return ReconstructedRequestHeader(
            header.header_id,
            header.provider_name,
            header.model_name,
            header.route_identity,
            header.endpoint_identity,
            header.generation_parameters,
            header.adapter_defaults,
            header.response_format,
            header.reasoning_controls,
            tuple(components),
            tuple(
                cast(str, item["revision_id"])
                for item in composition
                if item.get("kind") == "revision"
            ),
            composition,
        )

    @staticmethod
    def _resolve_system_composition(
        header: RequestHeaderRecord,
    ) -> tuple[Mapping[str, object], ...]:
        raw = header.adapter_defaults.get("system_composition")
        if raw is None:
            return ()
        if (
            not isinstance(raw, tuple)
            or len(raw) > MAX_SURFACE_REPLACEMENT_SPAN
            or any(not isinstance(item, Mapping) for item in raw)
        ):
            raise ValueError("system_composition_invalid")
        return cast(tuple[Mapping[str, object], ...], raw)

    reconstruct_logical_header = reconstruct_header

    def _validate_owner(
        self,
        cursor: sqlite3.Cursor,
        *,
        owner_id: str,
        segment_id: str,
    ) -> None:
        effective = self.repository.get_effective_owner(cursor, segment_id)
        if (
            effective is None
            or effective.owner_id != owner_id
            or not effective.attached
        ):
            raise ValueError("surface_owner_mismatch")

    def _expected_descriptor_value(
        self,
        cursor: sqlite3.Cursor,
        owner_id: str,
        descriptor: TraceProvenance,
        supplied: object,
    ) -> object:
        key = _saved_reference_key(descriptor)
        if key is None:
            return supplied
        return self._resolve_reference_value(
            cursor,
            key,
            owner_id=owner_id,
        )

    def _resolve_reference_value(
        self,
        cursor: sqlite3.Cursor,
        key: SurfaceReferenceKey,
        *,
        owner_id: str,
    ) -> object:
        _, reference_kind, identity = key
        if reference_kind == "artifact":
            artifact = self.repository.get_artifact(cursor, identity)
            if artifact is None:
                raise ValueError("surface_value_unavailable")
            try:
                return json.loads(artifact.sanitized_bytes)
            except (TypeError, ValueError, UnicodeDecodeError) as exc:
                raise ValueError("surface_value_unavailable") from exc
        if reference_kind != "revision":
            raise ValueError("surface_value_unavailable")
        self._validate_revision_owner(cursor, owner_id, identity)
        revision = self.repository.get_semantic_revision(cursor, identity)
        if revision is None:
            raise ValueError("revision_owner_mismatch")
        if key[0] == "continuation":
            return project_semantic_revision_provider_continuations(
                cursor,
                revision_ids=(identity,),
                expected_conversation_id=revision.source_conversation_id,
            )[identity]
        return project_semantic_revision_provider_message(
            cursor,
            revision_id=identity,
            expected_conversation_id=revision.source_conversation_id,
        )

    def _resolve_reference_values(
        self,
        cursor: sqlite3.Cursor,
        keys: tuple[SurfaceReferenceKey, ...],
        *,
        owner_id: str,
    ) -> dict[SurfaceReferenceKey, object]:
        """Resolve canonical provider values in bounded SQL batches.

        This is the provider-surface verification path, not a trace-disclosure
        reader. It deliberately returns live canonical values without applying
        trace-only PII masks, because masking must never alter bytes sent to a
        provider. Historical viewer/export code must instead resolve each call
        with its frozen policy through
        ``project_semantic_revision_trace_message``. Retired canonical locators
        remain unavailable here rather than substituting a masked trace artifact
        into a future provider request.
        """

        unique = tuple(dict.fromkeys(keys))
        artifact_keys = tuple(key for key in unique if key[1] == "artifact")
        revision_keys = tuple(key for key in unique if key[1] == "revision")
        if len(artifact_keys) + len(revision_keys) != len(unique):
            raise ValueError("surface_value_unavailable")
        resolved: dict[SurfaceReferenceKey, object] = {}
        for offset in range(0, len(artifact_keys), 256):
            chunk = artifact_keys[offset : offset + 256]
            placeholders = ",".join("?" for _ in chunk)
            rows = cursor.execute(
                f"""SELECT artifact_id, sanitized_bytes
                       FROM console_trace_artifacts
                      WHERE artifact_id IN ({placeholders})""",
                tuple(key[2] for key in chunk),
            ).fetchall()
            by_id = {str(row[0]): row[1] for row in rows}
            for key in chunk:
                raw = by_id.get(key[2])
                if raw is None:
                    raise ValueError("surface_value_unavailable")
                try:
                    resolved[key] = json.loads(raw)
                except (TypeError, ValueError, UnicodeDecodeError) as exc:
                    raise ValueError("surface_value_unavailable") from exc
        revision_keys_by_conversation: dict[str, list[SurfaceReferenceKey]] = {}
        for key in revision_keys:
            self._validate_revision_owner(cursor, owner_id, key[2])
            revision = self.repository.get_semantic_revision(cursor, key[2])
            if revision is None:
                raise ValueError("revision_owner_mismatch")
            revision_keys_by_conversation.setdefault(
                revision.source_conversation_id,
                [],
            ).append(key)
        for conversation_id, conversation_keys in revision_keys_by_conversation.items():
            for offset in range(0, len(conversation_keys), 256):
                chunk = conversation_keys[offset : offset + 256]
                message_chunk = tuple(key for key in chunk if key[0] != "continuation")
                continuation_chunk = tuple(
                    key for key in chunk if key[0] == "continuation"
                )
                if message_chunk:
                    projected = project_semantic_revision_provider_messages(
                        cursor,
                        revision_ids=tuple(key[2] for key in message_chunk),
                        expected_conversation_id=conversation_id,
                    )
                    resolved.update((key, projected[key[2]]) for key in message_chunk)
                if continuation_chunk:
                    projected_continuations = (
                        project_semantic_revision_provider_continuations(
                            cursor,
                            revision_ids=tuple(key[2] for key in continuation_chunk),
                            expected_conversation_id=conversation_id,
                        )
                    )
                    resolved.update(
                        (key, projected_continuations[key[2]])
                        for key in continuation_chunk
                    )
        return resolved

    @staticmethod
    def _resolve_child_items(
        bundle: ProviderRequestShadowBundle,
        items: tuple[VerifiedSurfaceDeltaItem, ...],
    ) -> tuple[tuple[TraceProvenance, ...], tuple[object, ...]]:
        """Resolve only service-bound delta ordinals from the verified bundle."""

        bindings = {item.name: item for item in bundle.components}
        descriptors: list[TraceProvenance] = []
        values: list[object] = []
        for item in items:
            binding = bindings.get(item.component_name)
            if (
                binding is None
                or not isinstance(binding.value, tuple)
                or item.ordinal >= len(binding.value)
            ):
                raise ValueError("surface_delta_alignment")
            descriptors.append(item.provenance)
            values.append(binding.value[item.ordinal])
        return tuple(descriptors), tuple(values)

    @staticmethod
    def _resolve_delta_items(
        provenance: ProviderRequestProvenance,
        bundle: ProviderRequestShadowBundle,
        items: tuple[VerifiedSurfaceDeltaItem, ...],
    ) -> tuple[tuple[TraceProvenance, ...], tuple[object, ...]]:
        if not bundle.available:
            if len(items) != 1:
                raise ValueError("surface_delta_alignment")
            item = items[0]
            if (
                getattr(item, "component_name", None) != "omission"
                or getattr(item, "ordinal", None) != 0
                or type(getattr(item, "provenance", None)) is not OmittedTraceProvenance
            ):
                raise ValueError("surface_delta_alignment")
            return (cast(TraceProvenance, item.provenance),), (None,)
        bindings = {item.name: item for item in bundle.components}
        resolved_descriptors: list[TraceProvenance] = []
        resolved_values: list[object] = []
        descriptor_slots = {
            "messages_payload": provenance.messages_payload,
            "provider_continuations": provenance.continuations,
        }
        for item in items:
            name = getattr(item, "component_name", None)
            ordinal = getattr(item, "ordinal", None)
            if name not in descriptor_slots or type(ordinal) is not int:
                raise ValueError("surface_delta_alignment")
            descriptors = descriptor_slots[name]
            binding = bindings.get(name)
            if (
                binding is None
                or not isinstance(binding.value, tuple)
                or ordinal < 0
                or ordinal >= len(descriptors)
                or ordinal >= len(binding.value)
                or descriptors[ordinal] != getattr(item, "provenance", None)
            ):
                raise ValueError("surface_delta_alignment")
            resolved_descriptors.append(descriptors[ordinal])
            resolved_values.append(binding.value[ordinal])
        return tuple(resolved_descriptors), tuple(resolved_values)

    @staticmethod
    def _full_surface_values(
        provenance: ProviderRequestProvenance,
        bundle: ProviderRequestShadowBundle,
    ) -> tuple[tuple[TraceProvenance, ...], tuple[object, ...]]:
        bindings = {item.name: item for item in bundle.components}
        messages = bindings.get("messages_payload")
        if messages is None or not isinstance(messages.value, tuple):
            raise ValueError("surface_values_unavailable")
        descriptors = tuple(provenance.messages_payload)
        values = tuple(messages.value)
        continuation = bindings.get("provider_continuations")
        if continuation is not None:
            if not isinstance(continuation.value, tuple):
                raise ValueError("surface_values_unavailable")
            descriptors += provenance.continuations
            values += tuple(continuation.value)
        if len(descriptors) != len(values):
            raise ValueError("surface_provenance_mismatch")
        return descriptors, values

    def _reference_key(
        self,
        cursor: sqlite3.Cursor,
        descriptor: TraceProvenance,
        value: object,
    ) -> SurfaceReferenceKey | None:
        if type(descriptor) is SavedRevisionTraceProvenance:
            return (
                "message",
                "revision",
                cast(SavedRevisionTraceProvenance, descriptor).revision_id,
            )
        if type(descriptor) is OmittedTraceProvenance:
            omitted = cast(OmittedTraceProvenance, descriptor)
            return omitted.source.value, "omission", omitted.reason.value
        if type(descriptor) is ProviderArtifactTraceProvenance:
            artifact = self.repository.find_sanitized_artifact(
                cursor,
                sanitized_bytes=_artifact_bytes(value),
                media_type=TRACE_VALUE_MEDIA_TYPE,
                normalization_version=TRACE_VALUE_NORMALIZATION_VERSION,
            )
            return (
                None
                if artifact is None
                else (
                    cast(ProviderArtifactTraceProvenance, descriptor).source.value,
                    "artifact",
                    artifact.artifact_id,
                )
            )
        if type(descriptor) is not DerivedTraceProvenance:
            return None
        derived = cast(DerivedTraceProvenance, descriptor)
        if derived.artifact is not None:
            artifact = self.repository.find_sanitized_artifact(
                cursor,
                sanitized_bytes=_artifact_bytes(value),
                media_type=TRACE_VALUE_MEDIA_TYPE,
                normalization_version=TRACE_VALUE_NORMALIZATION_VERSION,
            )
            return (
                None
                if artifact is None
                else (derived.artifact.source.value, "artifact", artifact.artifact_id)
            )
        saved_inputs = _saved_inputs(derived)
        if len(saved_inputs) == 1:
            return (
                _saved_revision_component_kind(derived),
                "revision",
                saved_inputs[0].revision_id,
            )
        omitted_inputs = _omitted_inputs(derived)
        if len(omitted_inputs) == 1:
            return (
                omitted_inputs[0].source.value,
                "omission",
                omitted_inputs[0].reason.value,
            )
        return (
            derived.transform.value,
            "omission",
            TraceOmissionReason.SOURCE_UNAVAILABLE.value,
        )

    def _durable_reference_matches(
        self,
        cursor: sqlite3.Cursor,
        descriptor: TraceProvenance,
        value: object,
        durable_key: SurfaceReferenceKey,
        durable_values: Mapping[SurfaceReferenceKey, object],
    ) -> bool:
        if type(descriptor) is ProviderArtifactTraceProvenance:
            source = cast(ProviderArtifactTraceProvenance, descriptor).source.value
            return (
                durable_key[:2] == (source, "artifact")
                and durable_key in durable_values
                and _artifact_bytes(value)
                == _artifact_bytes(durable_values[durable_key])
            )
        if type(descriptor) is DerivedTraceProvenance:
            derived = cast(DerivedTraceProvenance, descriptor)
            if derived.artifact is not None:
                return (
                    durable_key[:2] == (derived.artifact.source.value, "artifact")
                    and durable_key in durable_values
                    and _artifact_bytes(value)
                    == _artifact_bytes(durable_values[durable_key])
                )
        expected = self._reference_key(cursor, descriptor, value)
        if expected != durable_key:
            return False
        return durable_key[1] != "revision" or (
            durable_key in durable_values
            and _artifact_bytes(value) == _artifact_bytes(durable_values[durable_key])
        )

    def _append_descriptor(
        self,
        cursor: sqlite3.Cursor,
        *,
        owner_id: str,
        segment_id: str,
        tail: SurfaceNodeRecord | None,
        descriptor: TraceProvenance,
        value: object,
    ) -> SurfaceNodeRecord:
        component_kind, reference = self._reference(
            cursor,
            owner_id=owner_id,
            descriptor=descriptor,
            value=value,
        )
        return self.repository.append_surface_node(
            cursor,
            segment_id=segment_id,
            sequence=0 if tail is None else tail.sequence + 1,
            predecessor_node_id=None if tail is None else tail.node_id,
            component_kind=component_kind,
            reference=reference,
        )

    def _reference(
        self,
        cursor: sqlite3.Cursor,
        *,
        owner_id: str,
        descriptor: TraceProvenance,
        value: object,
    ) -> tuple[str, SemanticRevisionRef | TraceContentRef | TraceOmission]:
        if type(descriptor) is SavedRevisionTraceProvenance:
            saved = cast(SavedRevisionTraceProvenance, descriptor)
            self._validate_revision_owner(cursor, owner_id, saved.revision_id)
            return "message", SemanticRevisionRef(saved.revision_id)
        if type(descriptor) is OmittedTraceProvenance:
            omitted = cast(OmittedTraceProvenance, descriptor)
            return omitted.source.value, TraceOmission(
                omitted.source.value,
                omitted.reason.value,
            )
        if type(descriptor) is ProviderArtifactTraceProvenance:
            artifact = cast(ProviderArtifactTraceProvenance, descriptor)
            return artifact.source.value, self._store_artifact(
                cursor,
                value,
                policy=artifact.policy,
            )
        if type(descriptor) is DerivedTraceProvenance:
            derived = cast(DerivedTraceProvenance, descriptor)
            if derived.artifact is not None:
                return derived.artifact.source.value, self._store_artifact(
                    cursor,
                    value,
                    policy=derived.artifact.policy,
                )
            saved_inputs = _saved_inputs(derived)
            if len(saved_inputs) == 1:
                self._validate_revision_owner(
                    cursor, owner_id, saved_inputs[0].revision_id
                )
                return _saved_revision_component_kind(derived), SemanticRevisionRef(
                    saved_inputs[0].revision_id
                )
            omitted_inputs = _omitted_inputs(derived)
            if len(omitted_inputs) == 1:
                item = omitted_inputs[0]
                return item.source.value, TraceOmission(
                    item.source.value,
                    item.reason.value,
                )
            return derived.transform.value, TraceOmission(
                derived.transform.value,
                TraceOmissionReason.SOURCE_UNAVAILABLE.value,
            )
        raise ValueError("unsupported_surface_descriptor")

    def _surface_reference_matches(
        self,
        cursor: sqlite3.Cursor,
        node: SurfaceNodeRecord,
        descriptor: TraceProvenance,
        value: object,
    ) -> bool:
        if type(descriptor) is SavedRevisionTraceProvenance:
            saved = cast(SavedRevisionTraceProvenance, descriptor)
            return (
                node.component_kind == "message"
                and node.reference_kind == "revision"
                and node.semantic_revision_id == saved.revision_id
            )
        if type(descriptor) is OmittedTraceProvenance:
            omitted = cast(OmittedTraceProvenance, descriptor)
            return (
                node.component_kind == omitted.source.value
                and node.reference_kind == "omission"
                and node.omission_reason_code == omitted.reason.value
            )
        if type(descriptor) is ProviderArtifactTraceProvenance:
            artifact = cast(ProviderArtifactTraceProvenance, descriptor)
            return self._artifact_node_matches(
                cursor,
                node,
                component_kind=artifact.source.value,
                value=value,
                policy=artifact.policy,
            )
        if type(descriptor) is not DerivedTraceProvenance:
            return False
        derived = cast(DerivedTraceProvenance, descriptor)
        if derived.artifact is not None:
            return self._artifact_node_matches(
                cursor,
                node,
                component_kind=derived.artifact.source.value,
                value=value,
                policy=derived.artifact.policy,
            )
        saved_inputs = _saved_inputs(derived)
        if len(saved_inputs) == 1:
            return (
                node.component_kind == _saved_revision_component_kind(derived)
                and node.reference_kind == "revision"
                and node.semantic_revision_id == saved_inputs[0].revision_id
            )
        omitted_inputs = _omitted_inputs(derived)
        if len(omitted_inputs) == 1:
            return (
                node.component_kind == omitted_inputs[0].source.value
                and node.reference_kind == "omission"
                and node.omission_reason_code == omitted_inputs[0].reason.value
            )
        return (
            node.component_kind == derived.transform.value
            and node.reference_kind == "omission"
            and node.omission_reason_code
            == TraceOmissionReason.SOURCE_UNAVAILABLE.value
        )

    def _artifact_node_matches(
        self,
        cursor: sqlite3.Cursor,
        node: SurfaceNodeRecord,
        *,
        component_kind: str,
        value: object,
        policy: FrozenTracePolicy | None,
    ) -> bool:
        if (
            node.component_kind != component_kind
            or node.reference_kind != "artifact"
            or node.artifact_id is None
        ):
            return False
        projected = self._pii_projected_value(value, policy=policy)
        artifact = self.repository.find_sanitized_artifact(
            cursor,
            sanitized_bytes=_artifact_bytes(projected),
            media_type=TRACE_VALUE_MEDIA_TYPE,
            normalization_version=TRACE_VALUE_NORMALIZATION_VERSION,
        )
        return artifact is not None and artifact.artifact_id == node.artifact_id

    def _validate_revision_owner(
        self,
        cursor: sqlite3.Cursor,
        owner_id: str,
        revision_id: str,
    ) -> None:
        owner = self.repository.get_owner(cursor, owner_id)
        revision = self.repository.get_semantic_revision(cursor, revision_id)
        if owner is None or owner.conversation_id is None or revision is None:
            raise ValueError("revision_owner_mismatch")
        if revision.source_conversation_id == owner.conversation_id:
            return
        segment = self.repository.get_segment(cursor, owner.root_segment_id)
        inherited_head = (
            None if segment is None else segment.inherited_surface_head_id
        )
        head = (
            None
            if inherited_head is None
            else self.repository.get_surface_node(cursor, inherited_head)
        )
        if head is None:
            raise ValueError("revision_owner_mismatch")
        assert segment is not None and segment.parent_segment_id is not None
        inherited = self._surface_projection(
            cursor,
            segment.parent_segment_id,
            head,
        )
        if not any(
            key[1] == "revision" and key[2] == revision_id
            for _sequence, key in inherited.entries
        ):
            raise ValueError("revision_owner_mismatch")

    def _store_artifact(
        self,
        cursor: sqlite3.Cursor,
        value: object,
        *,
        policy: FrozenTracePolicy | None = None,
    ) -> TraceContentRef:
        projected = self._pii_projected_value(value, policy=policy)
        body = _artifact_bytes(projected)
        artifact = self.repository.store_sanitized_artifact(
            cursor,
            sanitized_bytes=body,
            media_type=TRACE_VALUE_MEDIA_TYPE,
            normalization_version=TRACE_VALUE_NORMALIZATION_VERSION,
        )
        if policy is not None and policy.pii_redaction_enabled:
            redaction = redact_pii_value(value)
            if redaction.available:
                by_path: dict[str, list[PIIRedactionSpan]] = {}
                for item in redaction.field_redactions:
                    by_path.setdefault(item.field_path, []).append(item.span)
                for field_path, spans in sorted(by_path.items()):
                    self.repository.ensure_redaction_spans(
                        cursor,
                        policy_id=policy.policy_id,
                        semantic_revision_id=None,
                        artifact_id=artifact.artifact_id,
                        field_path=field_path,
                        spans=spans,
                    )
        return TraceContentRef(artifact.artifact_id, "trace_artifact")

    @staticmethod
    def _pii_projected_value(
        value: object,
        *,
        policy: FrozenTracePolicy | None,
    ) -> object:
        if policy is None or not policy.pii_redaction_enabled:
            return value
        result = redact_pii_value(value)
        if not result.available:
            return {"omitted": PII_DETECTOR_UNAVAILABLE}
        return result.value

    def _append_omission(
        self,
        cursor: sqlite3.Cursor,
        *,
        segment_id: str,
        tail: SurfaceNodeRecord | None,
        component_kind: str,
        reason: TraceOmissionReason,
    ) -> SurfaceNodeRecord:
        node = self.repository.append_surface_node(
            cursor,
            segment_id=segment_id,
            sequence=0 if tail is None else tail.sequence + 1,
            predecessor_node_id=None if tail is None else tail.node_id,
            component_kind=component_kind,
            reference=TraceOmission(component_kind, reason.value),
        )
        self._append_event(
            cursor,
            segment_id=segment_id,
            event_type="surface_append",
            surface_node_id=node.node_id,
        )
        self._append_event(
            cursor,
            segment_id=segment_id,
            event_type="gap",
            omission_reason_code=reason.value,
        )
        return node

    def _replace_surface(
        self,
        cursor: sqlite3.Cursor,
        *,
        owner_id: str,
        segment_id: str,
        tail: SurfaceNodeRecord | None,
        descriptors: tuple[TraceProvenance, ...],
        values: tuple[object, ...],
        plan: VerifiedSurfaceReplacement,
    ) -> tuple[SurfaceNodeRecord, SurfaceReplacementRecord]:
        self._validate_replacement_range(
            cursor,
            segment_id=segment_id,
            tail=tail,
            plan=plan,
            descriptor_count=len(descriptors),
        )
        assert tail is not None
        replacement_node = self._append_descriptor(
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            tail=tail,
            descriptor=descriptors[0],
            value=values[0],
        )
        replacement = self.repository.append_surface_replacement(
            cursor,
            segment_id=segment_id,
            replacement=SurfaceReplacement(
                predecessor_head_id=plan.predecessor_head_id,
                start_node_id=plan.start_node_id,
                end_node_id=plan.end_node_id,
                start_sequence=plan.start_sequence,
                end_sequence=plan.end_sequence,
                replacement_node_id=replacement_node.node_id,
            ),
        )
        self._append_event(
            cursor,
            segment_id=segment_id,
            event_type="surface_replace",
            surface_replacement_id=replacement.replacement_id,
        )
        return replacement_node, replacement

    def _validate_replacement_range(
        self,
        cursor: sqlite3.Cursor,
        *,
        segment_id: str,
        tail: SurfaceNodeRecord | None,
        plan: VerifiedSurfaceReplacement,
        descriptor_count: int,
    ) -> None:
        """Validate the complete bounded range before any trace write."""

        if tail is None or tail.node_id != plan.predecessor_head_id:
            raise ValueError("replacement_predecessor_mismatch")
        if plan.end_sequence < plan.start_sequence:
            raise ValueError("replacement_range_order")
        if descriptor_count != 1:
            raise ValueError("replacement_value")
        start = self.repository.get_surface_node(cursor, plan.start_node_id)
        end = self.repository.get_surface_node(cursor, plan.end_node_id)
        if (
            start is None
            or end is None
            or start.segment_id != segment_id
            or end.segment_id != segment_id
            or start.sequence != plan.start_sequence
            or end.sequence != plan.end_sequence
        ):
            raise ValueError("replacement_range_mismatch")
        count = cursor.execute(
            """SELECT COUNT(*) FROM console_trace_surface_nodes
                 WHERE segment_id = ? AND sequence BETWEEN ? AND ?""",
            (segment_id, plan.start_sequence, plan.end_sequence),
        ).fetchone()[0]
        if count != plan.end_sequence - plan.start_sequence + 1:
            raise ValueError("replacement_range_noncontiguous")

    def _persist_header(
        self,
        cursor: sqlite3.Cursor,
        *,
        provenance: ProviderRequestProvenance,
        bundle: ProviderRequestShadowBundle,
        surface_structure: Mapping[str, object],
        artifact_policy_id: str | None,
    ) -> RequestHeaderRecord:
        route = _route(provenance)
        components: list[HeaderComponentRef] = []
        omissions: dict[str, str] = {}
        adapter_system_composition: tuple[Mapping[str, object], ...] = ()
        if bundle.available:
            bindings = {item.name: item for item in bundle.components}
            provider = _required_string(bindings["api_endpoint"].value, "provider_name")
            model = _required_string(bindings["model"].value, "model_name")
            endpoint = _required_string(bundle.endpoint_identity, "endpoint_identity")
            system = _binding(bundle, "system_message")
            if system is not None and provenance.system_message is not None:
                provider_values = bundle.system_components
                if not provider_values and not _saved_descriptors(
                    provenance.system_message
                ):
                    provider_values = (system.value,)
                adapter_system_composition = self._persist_system_composition(
                    cursor,
                    provenance.system_message,
                    provider_values,
                    components,
                )
            else:
                adapter_system_composition = ()
            tools = _binding(bundle, "tools")
            if tools is not None:
                if not isinstance(tools.value, tuple):
                    raise ValueError("tool_values_unavailable")
                for ordinal, (descriptor, value) in enumerate(
                    zip(provenance.tools, tools.value, strict=True)
                ):
                    self._header_component(
                        cursor,
                        components,
                        omissions,
                        kind="tool_schema",
                        ordinal=ordinal,
                        descriptor=descriptor,
                        value=value,
                    )
            generation = {
                key: binding.value
                for key, binding in bindings.items()
                if key in _GENERATION_KEYS
            }
            response_format = _object_field(
                bindings["response_format"].value
                if "response_format" in bindings
                else None
            )
            reasoning = {
                key: bindings[key].value
                for key in sorted(_REASONING_KEYS)
                if key in bindings
            }
        else:
            provider = model = endpoint = "unavailable"
            generation = {}
            response_format = {}
            reasoning = {}
            omissions["provider_request"] = (
                bundle.omission_reason or TraceOmissionReason.SOURCE_UNAVAILABLE
            ).value

        adapter_defaults: dict[str, object] = {
            "normalization_version": TRACE_VALUE_NORMALIZATION_VERSION,
            "credential_source": bundle.credential_source.value,
            "parameter_sources": {
                item.kind.split(":", 1)[1]: item.source
                for item in (bundle.overlays if bundle.available else ())
                if item.kind.startswith("parameter:")
            },
            "provider_overlays": [
                {"kind": item.kind, "source": item.source}
                for item in (bundle.overlays if bundle.available else ())
                if not item.kind.startswith("parameter:")
            ],
            "handler_projection": [
                {"name": item.name, "redacted": item.redacted}
                for item in (bundle.handler_components if bundle.available else ())
            ],
            **surface_structure,
            **_header_structural_provenance(provenance),
        }
        if adapter_system_composition:
            adapter_defaults["system_composition"] = list(adapter_system_composition)
        adapter_defaults["artifact_policy_id"] = artifact_policy_id
        literal = bundle.literal_payload_value if bundle.available else None
        if literal is not None:
            if not isinstance(literal, Mapping):
                omissions["provider_literal_envelope"] = (
                    TraceOmissionReason.SOURCE_UNAVAILABLE.value
                )
            else:
                literal_envelope = {
                    str(key): value
                    for key, value in literal.items()
                    if key != "messages"
                }
                if "messages" in literal:
                    adapter_defaults["literal_surface_field"] = "messages"
                if literal_envelope:
                    artifact = self._store_artifact(
                        cursor,
                        literal_envelope,
                        policy=(
                            None
                            if artifact_policy_id is None
                            else self.repository.get_policy(cursor, artifact_policy_id)
                        ),
                    )
                    components.append(
                        HeaderComponentRef(
                            "provider_literal_envelope",
                            0,
                            artifact.content_id,
                        )
                    )
        if omissions:
            adapter_defaults["header_omissions"] = omissions
        return self.repository.create_or_reuse_request_header(
            cursor,
            provider_name=provider,
            model_name=model,
            route_identity=route,
            endpoint_identity=endpoint,
            generation_parameters=generation,
            adapter_defaults=adapter_defaults,
            response_format=response_format,
            reasoning_controls=reasoning,
            components=components,
        )

    def _header_component(
        self,
        cursor: sqlite3.Cursor,
        components: list[HeaderComponentRef],
        omissions: dict[str, str],
        *,
        kind: str,
        ordinal: int,
        descriptor: TraceProvenance,
        value: object,
    ) -> None:
        if type(descriptor) is OmittedTraceProvenance:
            omissions[f"{kind}:{ordinal}"] = cast(
                OmittedTraceProvenance, descriptor
            ).reason.value
            return
        artifact = self._store_artifact(
            cursor,
            value,
            policy=_artifact_policy(descriptor),
        )
        components.append(HeaderComponentRef(kind, ordinal, artifact.content_id))

    def _persist_system_composition(
        self,
        cursor: sqlite3.Cursor,
        descriptor: TraceProvenance,
        provider_values: tuple[object, ...],
        components: list[HeaderComponentRef],
    ) -> tuple[Mapping[str, object], ...]:
        tokens: list[Mapping[str, object]] = []
        provider_ordinal = 0

        def visit(item: TraceProvenance) -> None:
            nonlocal provider_ordinal
            if type(item) is SavedRevisionTraceProvenance:
                tokens.append(
                    {
                        "kind": "revision",
                        "revision_id": cast(
                            SavedRevisionTraceProvenance, item
                        ).revision_id,
                    }
                )
                return
            if type(item) is OmittedTraceProvenance:
                omitted = cast(OmittedTraceProvenance, item)
                tokens.append(
                    {
                        "kind": "omission",
                        "source": omitted.source.value,
                        "reason": omitted.reason.value,
                    }
                )
                return
            if type(item) is ProviderArtifactTraceProvenance:
                if provider_ordinal >= len(provider_values):
                    raise ValueError("system_component_alignment")
                artifact = self._store_artifact(
                    cursor,
                    provider_values[provider_ordinal],
                    policy=cast(ProviderArtifactTraceProvenance, item).policy,
                )
                components.append(
                    HeaderComponentRef(
                        "rendered_system_part",
                        provider_ordinal,
                        artifact.content_id,
                    )
                )
                tokens.append(
                    {"kind": "artifact", "component_ordinal": provider_ordinal}
                )
                provider_ordinal += 1
                return
            if type(item) is not DerivedTraceProvenance:
                raise ValueError("system_component_alignment")
            derived = cast(DerivedTraceProvenance, item)
            tokens.append(
                {"kind": "transform_start", "transform": derived.transform.value}
            )
            for nested in derived.inputs:
                visit(nested)
            if derived.artifact is not None:
                visit(derived.artifact)
            tokens.append(
                {"kind": "transform_end", "transform": derived.transform.value}
            )

        visit(descriptor)
        if provider_ordinal != len(provider_values):
            raise ValueError("system_component_alignment")
        if len(tokens) > MAX_SURFACE_REPLACEMENT_SPAN:
            raise ValueError("system_composition_span")
        return tuple(tokens)

    @staticmethod
    def _validate_system_composition_shape(
        descriptor: TraceProvenance,
        provider_value_count: int,
    ) -> None:
        token_count = 0
        provider_count = 0

        def visit(item: TraceProvenance) -> None:
            nonlocal provider_count, token_count
            if type(item) in {SavedRevisionTraceProvenance, OmittedTraceProvenance}:
                token_count += 1
                return
            if type(item) is ProviderArtifactTraceProvenance:
                token_count += 1
                provider_count += 1
                return
            if type(item) is not DerivedTraceProvenance:
                raise ValueError("system_component_alignment")
            derived = cast(DerivedTraceProvenance, item)
            token_count += 2
            for nested in derived.inputs:
                visit(nested)
            if derived.artifact is not None:
                visit(derived.artifact)

        visit(descriptor)
        if token_count > MAX_SURFACE_REPLACEMENT_SPAN:
            raise ValueError("system_composition_span")
        if provider_count != provider_value_count:
            raise ValueError("system_component_alignment")

    def _validate_saved_system_values(
        self,
        cursor: sqlite3.Cursor,
        *,
        owner_id: str,
        descriptor: TraceProvenance,
        bundle: ProviderRequestShadowBundle,
    ) -> None:
        leaves: list[TraceProvenance] = []

        def visit(item: TraceProvenance) -> None:
            if type(item) in {
                SavedRevisionTraceProvenance,
                ProviderArtifactTraceProvenance,
            }:
                leaves.append(item)
                return
            if type(item) is OmittedTraceProvenance:
                return
            if type(item) is not DerivedTraceProvenance:
                raise ValueError("system_component_alignment")
            derived = cast(DerivedTraceProvenance, item)
            for nested in derived.inputs:
                visit(nested)
            if derived.artifact is not None:
                visit(derived.artifact)

        visit(descriptor)
        saved = tuple(
            cast(SavedRevisionTraceProvenance, item)
            for item in leaves
            if type(item) is SavedRevisionTraceProvenance
        )
        if not saved:
            return
        parts = bundle.system_leaf_components
        if not parts:
            system = _binding(bundle, "system_message")
            if len(leaves) != 1 or system is None:
                raise ValueError("system_revision_value_mismatch")
            parts = (system.value,)
        if len(parts) != len(leaves):
            raise ValueError("system_revision_value_mismatch")
        keys = tuple(("message", "revision", item.revision_id) for item in saved)
        canonical = self._resolve_reference_values(cursor, keys, owner_id=owner_id)
        saved_index = 0
        for leaf, supplied in zip(leaves, parts, strict=True):
            if type(leaf) is not SavedRevisionTraceProvenance:
                continue
            value = canonical[keys[saved_index]]
            saved_index += 1
            if isinstance(value, Mapping) and "content" in value:
                value = value["content"]
            if _artifact_bytes(value) != _artifact_bytes(supplied):
                raise ValueError("system_revision_value_mismatch")

    def _append_event(
        self,
        cursor: sqlite3.Cursor,
        *,
        segment_id: str,
        event_type: TraceEventType,
        **references: str,
    ) -> None:
        tail = self.repository.get_event_tail(cursor, segment_id)
        self.repository.append_event(
            cursor,
            segment_id=segment_id,
            sequence=0 if tail is None else tail.sequence + 1,
            event_type=event_type,
            **references,
        )


def _binding(
    bundle: ProviderRequestShadowBundle,
    name: str,
) -> FinalValueBinding | None:
    return next((item for item in bundle.components if item.name == name), None)


def _validate_bundle(bundle: ProviderRequestShadowBundle) -> None:
    if type(bundle.credential_source) is not ProviderCredentialSource:
        raise ValueError("credential_source")
    if not bundle.available:
        if bundle.omission_reason is None:
            raise ValueError("unavailable_bundle_reason")
        if any(
            (
                bundle.components,
                bundle.handler_components,
                bundle.literal_payload is not None,
                bundle.system_components,
                bundle.system_leaf_components,
                bundle.endpoint_identity is not None,
                bundle.overlays,
            )
        ):
            raise ValueError("unavailable_bundle_content")
        return
    if bundle.omission_reason is not None:
        raise ValueError("available_bundle_omission")
    if type(bundle.endpoint_identity) is not str or not bundle.endpoint_identity:
        raise ValueError("endpoint_identity")
    if any(type(item) is not FinalValueBinding for item in bundle.components):
        raise TypeError("components")
    if any(type(item) is not FinalValueBinding for item in bundle.handler_components):
        raise TypeError("handler_components")
    if type(bundle.system_components) is not tuple:
        raise TypeError("system_components")
    if type(bundle.system_leaf_components) is not tuple:
        raise TypeError("system_leaf_components")
    if any(type(item) is not ProviderOverlayProvenance for item in bundle.overlays):
        raise TypeError("overlays")
    component_names = [item.name for item in bundle.components]
    handler_names = [item.name for item in bundle.handler_components]
    if len(component_names) != len(set(component_names)):
        raise ValueError("duplicate_components")
    if len(handler_names) != len(set(handler_names)):
        raise ValueError("duplicate_handler_components")
    if not {"api_endpoint", "messages_payload", "model"}.issubset(component_names):
        raise ValueError("required_components")


def _required_string(value: object, field_name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(field_name)
    return cast(str, value)


def _object_field(value: object) -> Mapping[str, object]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return cast(Mapping[str, object], value)
    return {"value": value}


def _route(provenance: ProviderRequestProvenance) -> str:
    routes = tuple(
        item
        for item in provenance.metadata
        if type(item) is RequestRouteTraceProvenance
    )
    if len(routes) != 1:
        raise ValueError("request_route_unavailable")
    return routes[0].route.value


def _surface_reference_domain(key: SurfaceReferenceKey) -> str:
    return (
        "provider_continuations"
        if key[0] in {"continuation", TraceTransformKind.CONTINUATION_ATTACHMENT.value}
        else "messages_payload"
    )


def _saved_reference_key(
    descriptor: TraceProvenance,
) -> SurfaceReferenceKey | None:
    if type(descriptor) is SavedRevisionTraceProvenance:
        return (
            "message",
            "revision",
            cast(SavedRevisionTraceProvenance, descriptor).revision_id,
        )
    if type(descriptor) is not DerivedTraceProvenance:
        return None
    derived = cast(DerivedTraceProvenance, descriptor)
    if derived.artifact is not None:
        return None
    saved = _saved_inputs(derived)
    if len(saved) != 1:
        return None
    return _saved_revision_component_kind(derived), "revision", saved[0].revision_id


def _saved_inputs(
    descriptor: DerivedTraceProvenance,
) -> tuple[SavedRevisionTraceProvenance, ...]:
    result: list[SavedRevisionTraceProvenance] = []
    for item in descriptor.inputs:
        if type(item) is SavedRevisionTraceProvenance:
            result.append(cast(SavedRevisionTraceProvenance, item))
        elif type(item) is DerivedTraceProvenance:
            result.extend(_saved_inputs(cast(DerivedTraceProvenance, item)))
    return tuple(result)


def _saved_revision_component_kind(descriptor: DerivedTraceProvenance) -> str:
    if descriptor.transform is TraceTransformKind.CONTINUATION_ATTACHMENT:
        return "continuation"
    return "message"


def _saved_descriptors(
    descriptor: TraceProvenance,
) -> tuple[SavedRevisionTraceProvenance, ...]:
    if type(descriptor) is SavedRevisionTraceProvenance:
        return (cast(SavedRevisionTraceProvenance, descriptor),)
    if type(descriptor) is DerivedTraceProvenance:
        return _saved_inputs(cast(DerivedTraceProvenance, descriptor))
    return ()


def _revision_only(descriptor: TraceProvenance) -> bool:
    if type(descriptor) is SavedRevisionTraceProvenance:
        return True
    if type(descriptor) is not DerivedTraceProvenance:
        return False
    derived = cast(DerivedTraceProvenance, descriptor)
    return derived.artifact is None and all(
        _revision_only(item) for item in derived.inputs
    )


def _omitted_inputs(
    descriptor: DerivedTraceProvenance,
) -> tuple[OmittedTraceProvenance, ...]:
    result: list[OmittedTraceProvenance] = []
    for item in descriptor.inputs:
        if type(item) is OmittedTraceProvenance:
            result.append(cast(OmittedTraceProvenance, item))
        elif type(item) is DerivedTraceProvenance:
            result.extend(_omitted_inputs(cast(DerivedTraceProvenance, item)))
    return tuple(result)


def _structural_provenance(
    descriptors: tuple[TraceProvenance, ...],
    previous_defaults: Mapping[str, object],
) -> dict[str, object]:
    sources: Counter[str] = Counter()
    transforms: Counter[str] = Counter()
    omissions: Counter[str] = Counter()

    for key, counter in (
        ("surface_sources", sources),
        ("surface_transforms", transforms),
        ("surface_omissions", omissions),
    ):
        previous = previous_defaults.get(key)
        if isinstance(previous, Mapping):
            counter.update({str(item): 1 for item in previous})

    def visit(item: TraceProvenance) -> None:
        if type(item) is ProviderArtifactTraceProvenance:
            artifact = cast(ProviderArtifactTraceProvenance, item)
            sources[artifact.source.value] = 1
        elif type(item) is OmittedTraceProvenance:
            omitted = cast(OmittedTraceProvenance, item)
            sources[omitted.source.value] = 1
            omissions[f"{omitted.source.value}:{omitted.reason.value}"] = 1
        elif type(item) is DerivedTraceProvenance:
            derived = cast(DerivedTraceProvenance, item)
            transforms[derived.transform.value] = 1
            if derived.artifact is not None:
                sources[derived.artifact.source.value] = 1
            for nested in derived.inputs:
                visit(nested)

    for descriptor in descriptors:
        visit(descriptor)
    return {
        "surface_sources": dict(sorted(sources.items())),
        "surface_transforms": dict(sorted(transforms.items())),
        "surface_omissions": dict(sorted(omissions.items())),
    }


def _surface_projection_metadata(
    descriptors: Iterable[TraceProvenance],
) -> tuple[Mapping[str, object], tuple[FrozenTracePolicy, ...]]:
    """Fold one verified structural projection during provider verification."""

    sources: Counter[str] = Counter()
    transforms: Counter[str] = Counter()
    omissions: Counter[str] = Counter()
    policies: dict[str, FrozenTracePolicy] = {}

    def add_policy(policy: FrozenTracePolicy) -> None:
        if policies.get(policy.policy_id, policy) != policy:
            raise ValueError("trace_policy_mismatch")
        policies[policy.policy_id] = policy

    def visit(item: TraceProvenance) -> None:
        if type(item) is ProviderArtifactTraceProvenance:
            artifact = cast(ProviderArtifactTraceProvenance, item)
            sources[artifact.source.value] = 1
            add_policy(artifact.policy)
        elif type(item) is OmittedTraceProvenance:
            omitted = cast(OmittedTraceProvenance, item)
            sources[omitted.source.value] = 1
            omissions[f"{omitted.source.value}:{omitted.reason.value}"] = 1
        elif type(item) is DerivedTraceProvenance:
            derived = cast(DerivedTraceProvenance, item)
            transforms[derived.transform.value] = 1
            if derived.artifact is not None:
                sources[derived.artifact.source.value] = 1
                add_policy(derived.artifact.policy)
            for nested in derived.inputs:
                visit(nested)

    for descriptor in descriptors:
        visit(descriptor)
    return (
        {
            "surface_sources": dict(sorted(sources.items())),
            "surface_transforms": dict(sorted(transforms.items())),
            "surface_omissions": dict(sorted(omissions.items())),
        },
        tuple(policies[key] for key in sorted(policies)),
    )


def _header_structural_provenance(
    provenance: ProviderRequestProvenance,
) -> dict[str, object]:
    sources: Counter[str] = Counter()
    transforms: Counter[str] = Counter()
    omissions: Counter[str] = Counter()

    def visit(item: TraceProvenance) -> None:
        if type(item) is ProviderArtifactTraceProvenance:
            sources[cast(ProviderArtifactTraceProvenance, item).source.value] += 1
        elif type(item) is OmittedTraceProvenance:
            omitted = cast(OmittedTraceProvenance, item)
            sources[omitted.source.value] += 1
            omissions[f"{omitted.source.value}:{omitted.reason.value}"] += max(
                1, omitted.omitted_count
            )
        elif type(item) is DerivedTraceProvenance:
            derived = cast(DerivedTraceProvenance, item)
            transforms[derived.transform.value] += 1
            if derived.artifact is not None:
                sources[derived.artifact.source.value] += 1
            for nested in derived.inputs:
                visit(nested)

    if provenance.system_message is not None:
        visit(provenance.system_message)
    for descriptor in (
        *provenance.tools,
        *provenance.thinking,
        *provenance.metadata,
    ):
        if type(descriptor) is not RequestRouteTraceProvenance:
            visit(descriptor)
    return {
        "header_sources": dict(sorted(sources.items())),
        "header_transforms": dict(sorted(transforms.items())),
        "header_provenance_omissions": dict(sorted(omissions.items())),
    }


def _omission_source(
    provenance: ProviderRequestProvenance,
) -> TraceProvenanceSource:
    for descriptor in provenance.messages_payload:
        if type(descriptor) is OmittedTraceProvenance:
            return cast(OmittedTraceProvenance, descriptor).source
        if type(descriptor) is ProviderArtifactTraceProvenance:
            return cast(ProviderArtifactTraceProvenance, descriptor).source
    return TraceProvenanceSource.PROVIDER_OVERLAY


def _policies(
    provenance: ProviderRequestProvenance,
    surface_descriptors: tuple[TraceProvenance, ...],
) -> tuple[FrozenTracePolicy, ...]:
    policies: dict[str, FrozenTracePolicy] = {}

    def visit(item: TraceProvenance) -> None:
        if type(item) is ProviderArtifactTraceProvenance:
            policy = cast(ProviderArtifactTraceProvenance, item).policy
            if policies.get(policy.policy_id, policy) != policy:
                raise ValueError("trace_policy_mismatch")
            policies[policy.policy_id] = policy
        elif type(item) is DerivedTraceProvenance:
            derived = cast(DerivedTraceProvenance, item)
            if derived.artifact is not None:
                policy = derived.artifact.policy
                if policies.get(policy.policy_id, policy) != policy:
                    raise ValueError("trace_policy_mismatch")
                policies[policy.policy_id] = policy
            for nested in derived.inputs:
                visit(nested)

    if provenance.system_message is not None:
        visit(provenance.system_message)
    for descriptor in (
        *provenance.tools,
        *provenance.thinking,
        *provenance.metadata,
        *surface_descriptors,
    ):
        visit(descriptor)
    return tuple(policies[key] for key in sorted(policies))


def _artifact_policy(descriptor: TraceProvenance) -> FrozenTracePolicy | None:
    """Return the frozen policy that owns one provider-only artifact."""

    if type(descriptor) is ProviderArtifactTraceProvenance:
        return cast(ProviderArtifactTraceProvenance, descriptor).policy
    if type(descriptor) is DerivedTraceProvenance:
        artifact = cast(DerivedTraceProvenance, descriptor).artifact
        return None if artifact is None else artifact.policy
    return None


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _artifact_bytes(value: object) -> bytes:
    return json.dumps(
        _thaw(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
