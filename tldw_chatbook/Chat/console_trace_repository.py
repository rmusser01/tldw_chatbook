"""Transaction-participating persistence for the Console semantic trace ledger.

All public operations accept a caller-owned cursor. This module never opens or
completes a transaction, so trace writes compose with the caller's Chat mutation.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import re
import sqlite3
from types import MappingProxyType
from typing import Literal, TypeAlias, cast

from tldw_chatbook.Chat.console_trace_models import (
    FrozenTracePolicy,
    SemanticRevisionRef,
    SurfaceReplacement,
    TraceCallState,
    TraceContentRef,
    TraceOmission,
    is_terminal_call_state,
    new_opaque_id,
    validate_call_transition,
)

_TOKEN = re.compile(r"[a-z][a-z0-9]*(?:[_-][a-z0-9]+)*\Z", re.ASCII)
_TOKEN_MAX = 64
TraceEventType: TypeAlias = Literal[
    "turn_boundary",
    "call_boundary",
    "surface_append",
    "surface_replace",
    "tool_call",
    "tool_result",
    "request_header_selection",
    "provider_route_selection",
    "response_selection",
    "call_outcome",
    "usage",
    "gap",
]
IntegrityState: TypeAlias = Literal["pending", "complete", "incomplete"]
SurfaceNodeCursor: TypeAlias = tuple[int, str]
DEFAULT_SURFACE_NODE_PAGE_SIZE = 128
MAX_SURFACE_NODE_PAGE_SIZE = 256


class TraceIdentityConflict(ValueError):
    """An immutable key already names a different logical record."""


@dataclass(frozen=True, slots=True)
class TraceSegmentRecord:
    segment_id: str
    parent_segment_id: str | None
    inherited_through_sequence: int | None
    inherited_surface_head_id: str | None


@dataclass(frozen=True, slots=True)
class TraceOwnerRecord:
    owner_id: str
    conversation_id: str | None
    root_segment_id: str
    attached: bool
    detached_at: str | None


@dataclass(frozen=True, slots=True)
class TraceForkBoundary:
    """Immutable source prefix inherited by one conversation fork."""

    source_conversation_id: str
    source_owner_id: str
    parent_segment_id: str
    inherited_through_sequence: int
    inherited_surface_head_id: str


@dataclass(frozen=True, slots=True)
class SemanticRevisionRecord:
    revision_id: str
    source_conversation_id: str
    source_message_id: str
    revision_sequence: int
    normalized_role: str
    content_kind: str
    creation_reason: str
    predecessor_revision_id: str | None
    live_message_id: str | None
    live_locator_retired_at: str | None


@dataclass(frozen=True, slots=True)
class TraceArtifactRecord:
    artifact_id: str
    identity_digest: str
    media_type: str
    normalization_version: str
    sanitized_bytes: bytes


@dataclass(frozen=True, slots=True)
class RevisionPolicyBindingRecord:
    revision_id: str
    policy_id: str
    binding_outcome: Literal["artifact", "omission"]
    artifact_id: str | None
    omission_reason_code: str | None


@dataclass(frozen=True, slots=True)
class SurfaceNodeRecord:
    node_id: str
    segment_id: str
    sequence: int
    predecessor_node_id: str | None
    component_kind: str
    reference_kind: Literal["revision", "artifact", "omission"]
    semantic_revision_id: str | None
    artifact_id: str | None
    omission_reason_code: str | None


@dataclass(frozen=True, slots=True)
class SurfaceNodePage(Sequence[SurfaceNodeRecord]):
    """One bounded seek page of segment-local surface nodes."""

    items: tuple[SurfaceNodeRecord, ...]
    next_cursor: SurfaceNodeCursor | None

    def __iter__(self) -> Iterator[SurfaceNodeRecord]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(
        self, index: int | slice
    ) -> SurfaceNodeRecord | tuple[SurfaceNodeRecord, ...]:
        return self.items[index]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SurfaceNodePage):
            return self.items == other.items and self.next_cursor == other.next_cursor
        if isinstance(other, Sequence):
            return self.items == tuple(other)
        return False


@dataclass(frozen=True, slots=True)
class SurfaceReplacementRecord:
    replacement_id: str
    segment_id: str
    replacement: SurfaceReplacement


@dataclass(frozen=True, slots=True)
class HeaderComponentRef:
    component_kind: str
    ordinal: int
    artifact_id: str

    def __post_init__(self) -> None:
        _validate_token(self.component_kind, "component_kind")
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("ordinal")
        SemanticRevisionRef(self.artifact_id)


@dataclass(frozen=True, slots=True)
class RequestHeaderRecord:
    header_id: str
    provider_name: str
    model_name: str
    route_identity: str
    endpoint_identity: str
    generation_parameters: Mapping[str, object]
    adapter_defaults: Mapping[str, object]
    response_format: Mapping[str, object]
    reasoning_controls: Mapping[str, object]
    components: tuple[HeaderComponentRef, ...]


@dataclass(frozen=True, slots=True)
class TraceCallRecord:
    call_id: str
    owner_id: str
    segment_id: str
    turn_id: str
    run_id: str
    call_sequence: int
    idempotency_key: str
    policy_id: str
    state: TraceCallState
    surface_node_id: str | None
    request_header_id: str | None
    provider_name: str | None
    model_name: str | None
    route_identity: str | None
    dispatch_started_at: str | None
    response_started_at: str | None
    settled_at: str | None
    provider_inactive_at: str | None
    outcome: str | None
    usage: Mapping[str, object] | None
    integrity_state: IntegrityState
    omission_reason_code: str | None


@dataclass(frozen=True, slots=True)
class TraceEventRecord:
    event_id: str
    segment_id: str
    sequence: int
    event_type: TraceEventType
    turn_id: str | None
    call_id: str | None
    surface_node_id: str | None
    surface_replacement_id: str | None
    request_header_id: str | None
    semantic_revision_id: str | None
    artifact_id: str | None
    omission_reason_code: str | None


@dataclass(frozen=True, slots=True)
class TraceResponseLinkRecord:
    response_link_id: str
    call_id: str
    link_kind: Literal["revision", "artifact"]
    semantic_revision_id: str | None
    artifact_id: str | None
    verification_outcome: Literal["verified_equal", "sanitized_artifact"]


def _validate_token(value: str, field_name: str) -> None:
    if (
        type(value) is not str
        or len(value) > _TOKEN_MAX
        or _TOKEN.fullmatch(value) is None
    ):
        raise ValueError(field_name)


def _nonempty(value: str, field_name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(field_name)
    return value


def _json_object(
    value: Mapping[str, object],
    field_name: str,
    *,
    allow_frozen: bool = False,
) -> str:
    if not isinstance(value, Mapping):
        raise ValueError(field_name)
    try:
        canonical = _canonical_json_value(
            value,
            field_name,
            allow_frozen=allow_frozen,
        )
        encoded = json.dumps(
            canonical, allow_nan=False, separators=(",", ":"), sort_keys=True
        )
        if not isinstance(json.loads(encoded), dict):
            raise ValueError(field_name)
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError(field_name) from exc
    return encoded


def _canonical_json_value(
    value: object,
    field_name: str,
    *,
    allow_frozen: bool,
) -> object:
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(field_name)
        return value
    if type(value) is list:
        return [
            _canonical_json_value(item, field_name, allow_frozen=allow_frozen)
            for item in value
        ]
    if allow_frozen and type(value) is tuple:
        return [
            _canonical_json_value(item, field_name, allow_frozen=True) for item in value
        ]
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise ValueError(field_name)
        return {
            key: _canonical_json_value(
                item,
                field_name,
                allow_frozen=allow_frozen,
            )
            for key, item in value.items()
        }
    raise ValueError(field_name)


def _freeze_json_value(value: object) -> object:
    if type(value) is dict:
        return MappingProxyType(
            {key: _freeze_json_value(item) for key, item in value.items()}
        )
    if type(value) is list:
        return tuple(_freeze_json_value(item) for item in value)
    return value


def _decode_object(value: str | None) -> Mapping[str, object] | None:
    if value is None:
        return None
    decoded = json.loads(value)
    if not isinstance(decoded, dict):
        raise ValueError("stored_json_object")
    return cast(Mapping[str, object], _freeze_json_value(decoded))


class ConsoleTraceRepository:
    """Typed semantic trace storage over caller-owned SQLite transactions.

    Lookup-before-write mutators require an active caller transaction and claim
    SQLite write intent before their first lookup. Callers should prefer
    ``transaction(immediate=True)`` to acquire that lock at the outer
    transaction boundary.
    """

    def get_graph_epoch(self, cursor: sqlite3.Cursor) -> int:
        row = cursor.execute(
            "SELECT epoch FROM console_trace_graph_epoch WHERE singleton_id = 1"
        ).fetchone()
        if row is None or type(row[0]) is not int:
            raise RuntimeError("graph_epoch_unavailable")
        return int(row[0])

    def create_segment(
        self,
        cursor: sqlite3.Cursor,
        *,
        parent_segment_id: str | None = None,
        inherited_through_sequence: int | None = None,
        inherited_surface_head_id: str | None = None,
    ) -> TraceSegmentRecord:
        segment_id = new_opaque_id()
        cursor.execute(
            """INSERT INTO console_trace_segments(
                   segment_id, parent_segment_id, inherited_through_sequence,
                   inherited_surface_head_id) VALUES (?, ?, ?, ?)""",
            (
                segment_id,
                parent_segment_id,
                inherited_through_sequence,
                inherited_surface_head_id,
            ),
        )
        if parent_segment_id is not None:
            self._advance_graph_epoch(cursor)
        record = self.get_segment(cursor, segment_id)
        assert record is not None
        return record

    def get_segment(
        self, cursor: sqlite3.Cursor, segment_id: str
    ) -> TraceSegmentRecord | None:
        row = cursor.execute(
            """SELECT segment_id, parent_segment_id, inherited_through_sequence,
                      inherited_surface_head_id FROM console_trace_segments
                 WHERE segment_id = ?""",
            (segment_id,),
        ).fetchone()
        return None if row is None else TraceSegmentRecord(*row)

    def attach_owner(
        self,
        cursor: sqlite3.Cursor,
        *,
        conversation_id: str,
        root_segment_id: str,
    ) -> TraceOwnerRecord:
        owner_id = new_opaque_id()
        cursor.execute(
            """INSERT INTO console_trace_owners(
                   owner_id, conversation_id, root_segment_id, attached)
                 VALUES (?, ?, ?, 1)""",
            (owner_id, conversation_id, root_segment_id),
        )
        self._advance_graph_epoch(cursor)
        record = self.get_owner(cursor, owner_id)
        assert record is not None
        return record

    def capture_fork_boundary(
        self,
        cursor: sqlite3.Cursor,
        *,
        conversation_id: str,
        included_turn_ids: Sequence[str],
    ) -> TraceForkBoundary | None:
        """Capture the newest trace event belonging to a forked message prefix.

        Call events carry their turn identity on the immutable call row. Turn
        boundary events carry it directly. Restricting the boundary to the
        supplied active-lineage turns excludes later calls and excluded
        regeneration branches even when they already exist in the source
        segment.

        Args:
            cursor: Caller-owned transaction cursor.
            conversation_id: Durable source conversation identity.
            included_turn_ids: Ordered unique trace-turn identities retained by
                the forked message prefix.

        Returns:
            The newest matching immutable boundary, or None when no attached
            trace owner or matching event exists.

        Raises:
            ValueError: If a conversation or turn identity is empty, duplicated,
                or otherwise invalid.
            TraceIdentityConflict: If the selected event has no reconstructable
                surface head.
        """

        _nonempty(conversation_id, "conversation_id")
        turn_ids = tuple(included_turn_ids)
        if not turn_ids:
            return None
        if any(type(turn_id) is not str or not turn_id for turn_id in turn_ids):
            raise ValueError("included_turn_ids")
        if len(turn_ids) != len(set(turn_ids)):
            raise ValueError("included_turn_ids")
        owner_row = cursor.execute(
            """SELECT owner_id, conversation_id, root_segment_id, attached,
                      detached_at FROM console_trace_owners
                 WHERE conversation_id = ? AND attached = 1""",
            (conversation_id,),
        ).fetchone()
        if owner_row is None:
            return None
        owner = self._owner(owner_row)
        placeholders = ",".join("?" for _ in turn_ids)
        boundary_segment_id: str | None = None
        boundary_sequence: int | None = None
        for segment_id, through_sequence in self._segment_lineage(
            cursor,
            owner.root_segment_id,
        ):
            params: list[object] = [segment_id, *turn_ids, *turn_ids]
            upper_clause = ""
            if through_sequence is not None:
                upper_clause = " AND event.sequence <= ?"
                params.append(through_sequence)
            boundary_row = cursor.execute(
                f"""SELECT MAX(event.sequence)
                       FROM console_trace_events AS event
                  LEFT JOIN console_trace_calls AS call
                         ON call.call_id = event.call_id
                      WHERE event.segment_id = ?
                        AND (event.turn_id IN ({placeholders})
                             OR call.turn_id IN ({placeholders}))
                        {upper_clause}""",
                tuple(params),
            ).fetchone()
            candidate = boundary_row[0] if boundary_row is not None else None
            if type(candidate) is int and candidate >= 0:
                boundary_segment_id = segment_id
                boundary_sequence = candidate
        if boundary_segment_id is None or boundary_sequence is None:
            return None
        surface_head_id = self._surface_head_at_event_boundary(
            cursor,
            segment_id=boundary_segment_id,
            through_sequence=boundary_sequence,
        )
        if surface_head_id is None:
            raise TraceIdentityConflict("fork_boundary_surface")
        return TraceForkBoundary(
            source_conversation_id=conversation_id,
            source_owner_id=owner.owner_id,
            parent_segment_id=boundary_segment_id,
            inherited_through_sequence=boundary_sequence,
            inherited_surface_head_id=surface_head_id,
        )

    def attach_fork_owner(
        self,
        cursor: sqlite3.Cursor,
        *,
        conversation_id: str,
        boundary: TraceForkBoundary,
    ) -> TraceOwnerRecord:
        """Attach a child owner to an exact immutable source boundary.

        Args:
            cursor: Caller-owned transaction cursor.
            conversation_id: Durable child conversation identity.
            boundary: Validated source prefix boundary captured by a fork fence.

        Returns:
            The newly attached child trace owner.

        Raises:
            TypeError: If boundary is not a TraceForkBoundary.
            TraceIdentityConflict: If source ownership, reachability, event
                identity, or surface state no longer matches the boundary.
        """

        if not isinstance(boundary, TraceForkBoundary):
            raise TypeError("boundary")
        source_owner = self.get_owner(cursor, boundary.source_owner_id)
        if (
            source_owner is None
            or not source_owner.attached
            or source_owner.conversation_id != boundary.source_conversation_id
        ):
            raise TraceIdentityConflict("fork_boundary_owner")
        reachable_boundaries = dict(
            self._segment_lineage(cursor, source_owner.root_segment_id)
        )
        allowed_sequence = reachable_boundaries.get(boundary.parent_segment_id)
        if (
            boundary.parent_segment_id not in reachable_boundaries
            or (
                allowed_sequence is not None
                and boundary.inherited_through_sequence > allowed_sequence
            )
        ):
            raise TraceIdentityConflict("fork_boundary_owner")
        current_head = self._surface_head_at_event_boundary(
            cursor,
            segment_id=boundary.parent_segment_id,
            through_sequence=boundary.inherited_through_sequence,
        )
        event_exists = cursor.execute(
            """SELECT 1 FROM console_trace_events
                 WHERE segment_id = ? AND sequence = ?""",
            (
                boundary.parent_segment_id,
                boundary.inherited_through_sequence,
            ),
        ).fetchone()
        if (
            event_exists is None
            or current_head != boundary.inherited_surface_head_id
        ):
            raise TraceIdentityConflict("fork_boundary_state")
        child = self.create_segment(
            cursor,
            parent_segment_id=boundary.parent_segment_id,
            inherited_through_sequence=boundary.inherited_through_sequence,
            inherited_surface_head_id=boundary.inherited_surface_head_id,
        )
        return self.attach_owner(
            cursor,
            conversation_id=conversation_id,
            root_segment_id=child.segment_id,
        )

    def fork_owner_matches_boundary(
        self,
        cursor: sqlite3.Cursor,
        *,
        conversation_id: str,
        boundary: TraceForkBoundary,
    ) -> bool:
        """Return whether an attached child owner exactly matches a fork boundary.

        Args:
            cursor: Caller-owned transaction cursor.
            conversation_id: Durable child conversation identity.
            boundary: Expected immutable parent prefix boundary.

        Returns:
            True only when the attached child root records the exact boundary.

        Raises:
            TypeError: If boundary is not a TraceForkBoundary.
        """

        if not isinstance(boundary, TraceForkBoundary):
            raise TypeError("boundary")
        row = cursor.execute(
            """SELECT segment.parent_segment_id,
                      segment.inherited_through_sequence,
                      segment.inherited_surface_head_id
                 FROM console_trace_owners AS owner
                 JOIN console_trace_segments AS segment
                   ON segment.segment_id = owner.root_segment_id
                WHERE owner.conversation_id = ? AND owner.attached = 1""",
            (conversation_id,),
        ).fetchone()
        return row is not None and tuple(row) == (
            boundary.parent_segment_id,
            boundary.inherited_through_sequence,
            boundary.inherited_surface_head_id,
        )

    def read_conversation_call_lineage(
        self,
        cursor: sqlite3.Cursor,
        conversation_id: str,
    ) -> tuple[TraceCallRecord, ...]:
        """Read a conversation's shared prefix and private suffix in order.

        Args:
            cursor: Caller-owned transaction cursor.
            conversation_id: Durable conversation whose attached lineage is read.

        Returns:
            Trace calls ordered root-to-leaf and by event sequence within each
            segment, or an empty tuple when no trace owner is attached.

        Raises:
            RuntimeError: If a referenced lineage call cannot be reconstructed.
        """

        owner_row = cursor.execute(
            """SELECT owner_id, conversation_id, root_segment_id, attached,
                      detached_at FROM console_trace_owners
                 WHERE conversation_id = ? AND attached = 1""",
            (conversation_id,),
        ).fetchone()
        if owner_row is None:
            return ()
        owner = self._owner(owner_row)
        lineage = self._segment_lineage(cursor, owner.root_segment_id)

        calls: list[TraceCallRecord] = []
        for segment_id, through_sequence in lineage:
            params: list[object] = [segment_id]
            boundary_clause = ""
            if through_sequence is not None:
                boundary_clause = " HAVING MIN(event.sequence) <= ?"
                params.append(through_sequence)
            rows = cursor.execute(
                """SELECT call.call_id, MIN(event.sequence)
                     FROM console_trace_calls AS call
                     JOIN console_trace_events AS event
                       ON event.call_id = call.call_id
                    WHERE call.segment_id = ?
                 GROUP BY call.call_id"""
                + boundary_clause
                + " ORDER BY MIN(event.sequence), call.call_id",
                tuple(params),
            ).fetchall()
            for row in rows:
                call = self.get_call(cursor, row[0])
                if call is None:
                    raise RuntimeError("trace_lineage_call_unavailable")
                calls.append(call)
        return tuple(calls)

    def _segment_lineage(
        self,
        cursor: sqlite3.Cursor,
        root_segment_id: str,
    ) -> list[tuple[str, int | None]]:
        """Return root-to-leaf segment IDs with each inherited upper bound."""

        lineage: list[tuple[str, int | None]] = []
        segment = self.get_segment(cursor, root_segment_id)
        upper_bound: int | None = None
        while segment is not None:
            lineage.append((segment.segment_id, upper_bound))
            upper_bound = segment.inherited_through_sequence
            if segment.parent_segment_id is None:
                break
            segment = self.get_segment(cursor, segment.parent_segment_id)
        lineage.reverse()
        return lineage

    def _surface_head_at_event_boundary(
        self,
        cursor: sqlite3.Cursor,
        *,
        segment_id: str,
        through_sequence: int,
    ) -> str | None:
        row = cursor.execute(
            """SELECT CASE event.event_type
                       WHEN 'surface_append' THEN event.surface_node_id
                       ELSE replacement.replacement_node_id
                     END
                   FROM console_trace_events AS event
              LEFT JOIN console_trace_surface_replacements AS replacement
                     ON replacement.replacement_id = event.surface_replacement_id
                  WHERE event.segment_id = ? AND event.sequence <= ?
                    AND event.event_type IN ('surface_append', 'surface_replace')
               ORDER BY event.sequence DESC LIMIT 1""",
            (segment_id, through_sequence),
        ).fetchone()
        if row is not None:
            return cast(str | None, row[0])
        segment = self.get_segment(cursor, segment_id)
        return None if segment is None else segment.inherited_surface_head_id

    def get_owner(
        self, cursor: sqlite3.Cursor, owner_id: str
    ) -> TraceOwnerRecord | None:
        row = cursor.execute(
            """SELECT owner_id, conversation_id, root_segment_id, attached, detached_at
                 FROM console_trace_owners WHERE owner_id = ?""",
            (owner_id,),
        ).fetchone()
        return None if row is None else self._owner(row)

    def get_effective_owner(
        self,
        cursor: sqlite3.Cursor,
        segment_id: str,
    ) -> TraceOwnerRecord | None:
        row = cursor.execute(
            """WITH RECURSIVE segment_ancestry(segment_id, depth) AS (
                   SELECT ?, 0 UNION ALL
                   SELECT segment.parent_segment_id, child.depth + 1
                     FROM console_trace_segments AS segment
                     JOIN segment_ancestry AS child ON child.segment_id = segment.segment_id
                    WHERE segment.parent_segment_id IS NOT NULL)
                 SELECT owner.owner_id, owner.conversation_id, owner.root_segment_id,
                        owner.attached, owner.detached_at
                   FROM segment_ancestry AS owned
                   JOIN console_trace_owners AS owner
                     ON owner.root_segment_id = owned.segment_id
                  ORDER BY owned.depth LIMIT 1""",
            (segment_id,),
        ).fetchone()
        return None if row is None else self._owner(row)

    def detach_owner(
        self,
        cursor: sqlite3.Cursor,
        *,
        owner_id: str,
        detached_at: str,
    ) -> TraceOwnerRecord:
        _nonempty(detached_at, "detached_at")
        cursor.execute(
            """UPDATE console_trace_owners
                  SET conversation_id = NULL, attached = 0, detached_at = ?
                WHERE owner_id = ? AND attached = 1""",
            (detached_at, owner_id),
        )
        if cursor.rowcount != 1:
            raise TraceIdentityConflict("owner_detach")
        self._advance_graph_epoch(cursor)
        record = self.get_owner(cursor, owner_id)
        assert record is not None
        return record

    def ensure_policy(
        self,
        cursor: sqlite3.Cursor,
        policy: FrozenTracePolicy,
    ) -> FrozenTracePolicy:
        if not isinstance(policy, FrozenTracePolicy):
            raise TypeError("policy")
        self._claim_write_intent(cursor)
        row = cursor.execute(
            """SELECT policy_id, credential_filter_version, pii_redaction_enabled,
                      pii_ruleset_revision_id FROM console_trace_policies
                 WHERE policy_id = ?""",
            (policy.policy_id,),
        ).fetchone()
        if row is not None:
            stored = FrozenTracePolicy(row[0], row[1], bool(row[2]), row[3])
            if stored != policy:
                raise TraceIdentityConflict("policy_id")
            return stored
        cursor.execute(
            """INSERT INTO console_trace_policies(
                   policy_id, credential_filter_version, pii_redaction_enabled,
                   pii_ruleset_revision_id) VALUES (?, ?, ?, ?)""",
            (
                policy.policy_id,
                policy.credential_filter_version,
                int(policy.pii_redaction_enabled),
                policy.pii_ruleset_revision_id,
            ),
        )
        if policy.pii_ruleset_revision_id is not None:
            self._advance_graph_epoch(cursor)
        return policy

    def get_policy(
        self,
        cursor: sqlite3.Cursor,
        policy_id: str,
    ) -> FrozenTracePolicy | None:
        row = cursor.execute(
            """SELECT policy_id, credential_filter_version, pii_redaction_enabled,
                      pii_ruleset_revision_id FROM console_trace_policies
                 WHERE policy_id = ?""",
            (policy_id,),
        ).fetchone()
        return (
            None
            if row is None
            else FrozenTracePolicy(row[0], row[1], bool(row[2]), row[3])
        )

    def ensure_semantic_revision(
        self,
        cursor: sqlite3.Cursor,
        *,
        source_conversation_id: str,
        source_message_id: str,
        revision_sequence: int,
        normalized_role: str,
        content_kind: str,
        creation_reason: str,
        predecessor_revision_id: str | None = None,
        live_message_id: str | None = None,
        live_locator_retired_at: str | None = None,
    ) -> SemanticRevisionRecord:
        _validate_token(normalized_role, "normalized_role")
        _validate_token(content_kind, "content_kind")
        _validate_token(creation_reason, "creation_reason")
        self._claim_write_intent(cursor)
        expected = (
            source_conversation_id,
            source_message_id,
            revision_sequence,
            normalized_role,
            content_kind,
            creation_reason,
            predecessor_revision_id,
            live_message_id,
            live_locator_retired_at,
        )
        row = cursor.execute(
            """SELECT revision_id, source_conversation_id, source_message_id,
                      revision_sequence, normalized_role, content_kind, creation_reason,
                      predecessor_revision_id, live_message_id, live_locator_retired_at
                 FROM console_trace_semantic_revisions
                WHERE source_message_id = ? AND revision_sequence = ?""",
            (source_message_id, revision_sequence),
        ).fetchone()
        if row is not None:
            existing_record = SemanticRevisionRecord(*row)
            if tuple(row[1:]) != expected:
                raise TraceIdentityConflict("semantic_revision")
            return existing_record
        revision_id = new_opaque_id()
        cursor.execute(
            """INSERT INTO console_trace_semantic_revisions(
                   revision_id, source_conversation_id, source_message_id,
                   revision_sequence, normalized_role, content_kind, creation_reason,
                   predecessor_revision_id, live_message_id, live_locator_retired_at)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (revision_id, *expected),
        )
        if predecessor_revision_id is not None or live_message_id is not None:
            self._advance_graph_epoch(cursor)
        created_record = self.get_semantic_revision(cursor, revision_id)
        assert created_record is not None
        return created_record

    def get_semantic_revision(
        self,
        cursor: sqlite3.Cursor,
        revision_id: str,
    ) -> SemanticRevisionRecord | None:
        row = cursor.execute(
            """SELECT revision_id, source_conversation_id, source_message_id,
                      revision_sequence, normalized_role, content_kind, creation_reason,
                      predecessor_revision_id, live_message_id, live_locator_retired_at
                 FROM console_trace_semantic_revisions WHERE revision_id = ?""",
            (revision_id,),
        ).fetchone()
        return None if row is None else SemanticRevisionRecord(*row)

    def store_sanitized_artifact(
        self,
        cursor: sqlite3.Cursor,
        *,
        sanitized_bytes: bytes,
        media_type: str,
        normalization_version: str,
    ) -> TraceArtifactRecord:
        if type(sanitized_bytes) is not bytes:
            raise TypeError("sanitized_bytes")
        _nonempty(media_type, "media_type")
        _nonempty(normalization_version, "normalization_version")
        self._claim_write_intent(cursor)
        identity_digest = hashlib.sha256(sanitized_bytes).hexdigest()
        existing = self.find_sanitized_artifact(
            cursor,
            sanitized_bytes=sanitized_bytes,
            media_type=media_type,
            normalization_version=normalization_version,
        )
        if existing is not None:
            return existing
        artifact_id = new_opaque_id()
        cursor.execute(
            """INSERT INTO console_trace_artifacts(
                   artifact_id, identity_digest, media_type, normalization_version,
                   sanitized_bytes, byte_length) VALUES (?, ?, ?, ?, ?, ?)""",
            (
                artifact_id,
                identity_digest,
                media_type,
                normalization_version,
                sqlite3.Binary(sanitized_bytes),
                len(sanitized_bytes),
            ),
        )
        created_record = self.get_artifact(cursor, artifact_id)
        assert created_record is not None
        return created_record

    def find_sanitized_artifact(
        self,
        cursor: sqlite3.Cursor,
        *,
        sanitized_bytes: bytes,
        media_type: str,
        normalization_version: str,
    ) -> TraceArtifactRecord | None:
        """Return an exact existing artifact without claiming write intent."""

        identity_digest = hashlib.sha256(sanitized_bytes).hexdigest()
        rows = cursor.execute(
            """SELECT artifact_id, identity_digest, media_type, normalization_version,
                      sanitized_bytes FROM console_trace_artifacts
                 WHERE identity_digest = ? AND media_type = ?
                   AND normalization_version = ? ORDER BY artifact_id""",
            (identity_digest, media_type, normalization_version),
        ).fetchall()
        for row in rows:
            existing_record = self._artifact(row)
            if existing_record.sanitized_bytes == sanitized_bytes:
                return existing_record
        return None

    def get_artifact(
        self,
        cursor: sqlite3.Cursor,
        artifact_id: str,
    ) -> TraceArtifactRecord | None:
        row = cursor.execute(
            """SELECT artifact_id, identity_digest, media_type, normalization_version,
                      sanitized_bytes FROM console_trace_artifacts WHERE artifact_id = ?""",
            (artifact_id,),
        ).fetchone()
        return None if row is None else self._artifact(row)

    def bind_revision_policy(
        self,
        cursor: sqlite3.Cursor,
        *,
        revision_id: str,
        policy_id: str,
        artifact_id: str | None = None,
        omission_reason_code: str | None = None,
    ) -> RevisionPolicyBindingRecord:
        if (artifact_id is None) == (omission_reason_code is None):
            raise ValueError("revision_policy_binding_shape")
        if omission_reason_code is not None:
            _validate_token(omission_reason_code, "omission_reason_code")
        self._claim_write_intent(cursor)
        outcome: Literal["artifact", "omission"] = (
            "artifact" if artifact_id is not None else "omission"
        )
        expected = RevisionPolicyBindingRecord(
            revision_id, policy_id, outcome, artifact_id, omission_reason_code
        )
        row = cursor.execute(
            """SELECT revision_id, policy_id, binding_outcome, artifact_id,
                      omission_reason_code FROM console_trace_revision_bindings
                 WHERE revision_id = ? AND policy_id = ?""",
            (revision_id, policy_id),
        ).fetchone()
        if row is not None:
            stored = RevisionPolicyBindingRecord(*row)
            if stored != expected:
                raise TraceIdentityConflict("revision_policy_binding")
            return stored
        cursor.execute(
            """INSERT INTO console_trace_revision_bindings(
                   revision_id, policy_id, binding_outcome, artifact_id,
                   omission_reason_code) VALUES (?, ?, ?, ?, ?)""",
            (revision_id, policy_id, outcome, artifact_id, omission_reason_code),
        )
        self._advance_graph_epoch(cursor)
        return expected

    def get_revision_policy_binding(
        self,
        cursor: sqlite3.Cursor,
        *,
        revision_id: str,
        policy_id: str,
    ) -> RevisionPolicyBindingRecord | None:
        row = cursor.execute(
            """SELECT revision_id, policy_id, binding_outcome, artifact_id,
                      omission_reason_code FROM console_trace_revision_bindings
                 WHERE revision_id = ? AND policy_id = ?""",
            (revision_id, policy_id),
        ).fetchone()
        return None if row is None else RevisionPolicyBindingRecord(*row)

    def append_surface_node(
        self,
        cursor: sqlite3.Cursor,
        *,
        segment_id: str,
        sequence: int,
        predecessor_node_id: str | None,
        component_kind: str,
        reference: SemanticRevisionRef | TraceContentRef | TraceOmission,
    ) -> SurfaceNodeRecord:
        _validate_token(component_kind, "component_kind")
        semantic_revision_id: str | None = None
        artifact_id: str | None = None
        omission_reason_code: str | None = None
        if isinstance(reference, SemanticRevisionRef):
            reference_kind: Literal["revision", "artifact", "omission"] = "revision"
            semantic_revision_id = reference.revision_id
        elif isinstance(reference, TraceContentRef):
            reference_kind = "artifact"
            artifact_id = reference.content_id
        elif isinstance(reference, TraceOmission):
            reference_kind = "omission"
            if reference.component_kind != component_kind:
                raise ValueError("component_kind")
            omission_reason_code = reference.reason_code
        else:
            raise TypeError("reference")
        node_id = new_opaque_id()
        cursor.execute(
            """INSERT INTO console_trace_surface_nodes(
                   node_id, segment_id, sequence, predecessor_node_id, component_kind,
                   reference_kind, semantic_revision_id, artifact_id,
                   omission_reason_code) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                node_id,
                segment_id,
                sequence,
                predecessor_node_id,
                component_kind,
                reference_kind,
                semantic_revision_id,
                artifact_id,
                omission_reason_code,
            ),
        )
        self._advance_graph_epoch(cursor)
        record = self.get_surface_node(cursor, node_id)
        assert record is not None
        return record

    def get_surface_node(
        self,
        cursor: sqlite3.Cursor,
        node_id: str,
    ) -> SurfaceNodeRecord | None:
        row = cursor.execute(
            """SELECT node_id, segment_id, sequence, predecessor_node_id,
                      component_kind, reference_kind, semantic_revision_id,
                      artifact_id, omission_reason_code
                 FROM console_trace_surface_nodes WHERE node_id = ?""",
            (node_id,),
        ).fetchone()
        return None if row is None else SurfaceNodeRecord(*row)

    def get_surface_tail(
        self,
        cursor: sqlite3.Cursor,
        segment_id: str,
    ) -> SurfaceNodeRecord | None:
        """Return the newest segment-local surface node without scanning history."""

        row = cursor.execute(
            """SELECT node_id, segment_id, sequence, predecessor_node_id,
                      component_kind, reference_kind, semantic_revision_id,
                      artifact_id, omission_reason_code
                 FROM console_trace_surface_nodes WHERE segment_id = ?
                ORDER BY sequence DESC, node_id DESC LIMIT 1""",
            (segment_id,),
        ).fetchone()
        return None if row is None else SurfaceNodeRecord(*row)

    def read_surface_nodes(
        self,
        cursor: sqlite3.Cursor,
        segment_id: str,
        *,
        page_size: int = DEFAULT_SURFACE_NODE_PAGE_SIZE,
        after: SurfaceNodeCursor | None = None,
    ) -> SurfaceNodePage:
        """Read one bounded seek page of segment-local surface nodes.

        Args:
            cursor: Caller-owned SQLite cursor.
            segment_id: Segment whose nodes should be read.
            page_size: Requested page size, capped at
                ``MAX_SURFACE_NODE_PAGE_SIZE``.
            after: Exclusive ``(sequence, node_id)`` continuation cursor.

        Returns:
            A bounded page plus a continuation cursor when more rows exist.

        Raises:
            ValueError: If ``page_size`` or ``after`` is malformed.
        """

        if (
            type(page_size) is not int
            or not 1 <= page_size <= MAX_SURFACE_NODE_PAGE_SIZE
        ):
            raise ValueError("page_size")
        if after is not None and (
            type(after) is not tuple
            or len(after) != 2
            or type(after[0]) is not int
            or after[0] < 0
            or type(after[1]) is not str
            or not after[1]
        ):
            raise ValueError("after")
        continuation_clause = ""
        parameters: tuple[object, ...] = (segment_id, page_size + 1)
        if after is not None:
            continuation_clause = (
                " AND (sequence > ? OR (sequence = ? AND node_id > ?))"
            )
            parameters = (segment_id, after[0], after[0], after[1], page_size + 1)
        rows = cursor.execute(
            f"""SELECT node_id, segment_id, sequence, predecessor_node_id,
                       component_kind, reference_kind, semantic_revision_id,
                       artifact_id, omission_reason_code
                  FROM console_trace_surface_nodes WHERE segment_id = ?
                       {continuation_clause}
                 ORDER BY sequence, node_id LIMIT ?""",
            parameters,
        ).fetchall()
        records = tuple(SurfaceNodeRecord(*row) for row in rows[:page_size])
        next_cursor = (
            (records[-1].sequence, records[-1].node_id)
            if len(rows) > page_size
            else None
        )
        return SurfaceNodePage(records, next_cursor)

    def append_surface_replacement(
        self,
        cursor: sqlite3.Cursor,
        *,
        segment_id: str,
        replacement: SurfaceReplacement,
    ) -> SurfaceReplacementRecord:
        if not isinstance(replacement, SurfaceReplacement):
            raise TypeError("replacement")
        replacement_id = new_opaque_id()
        cursor.execute(
            """INSERT INTO console_trace_surface_replacements(
                   replacement_id, segment_id, predecessor_head_id, start_node_id,
                   start_sequence, end_node_id, end_sequence, replacement_node_id)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                replacement_id,
                segment_id,
                replacement.predecessor_head_id,
                replacement.start_node_id,
                replacement.start_sequence,
                replacement.end_node_id,
                replacement.end_sequence,
                replacement.replacement_node_id,
            ),
        )
        self._advance_graph_epoch(cursor)
        return SurfaceReplacementRecord(replacement_id, segment_id, replacement)

    def read_surface_replacements(
        self,
        cursor: sqlite3.Cursor,
        segment_id: str,
    ) -> tuple[SurfaceReplacementRecord, ...]:
        rows = cursor.execute(
            """SELECT r.replacement_id, r.segment_id, r.predecessor_head_id,
                      r.start_node_id, r.start_sequence, r.end_node_id,
                      r.end_sequence, r.replacement_node_id
                 FROM console_trace_surface_replacements AS r
                 JOIN console_trace_surface_nodes AS n
                   ON n.node_id = r.replacement_node_id
                WHERE r.segment_id = ?
                ORDER BY n.sequence, r.start_sequence, r.end_sequence,
                         r.replacement_id""",
            (segment_id,),
        ).fetchall()
        return tuple(
            SurfaceReplacementRecord(
                row[0],
                row[1],
                SurfaceReplacement(
                    predecessor_head_id=row[2],
                    start_node_id=row[3],
                    start_sequence=row[4],
                    end_node_id=row[5],
                    end_sequence=row[6],
                    replacement_node_id=row[7],
                ),
            )
            for row in rows
        )

    def create_or_reuse_request_header(
        self,
        cursor: sqlite3.Cursor,
        *,
        provider_name: str,
        model_name: str,
        route_identity: str,
        endpoint_identity: str,
        generation_parameters: Mapping[str, object],
        adapter_defaults: Mapping[str, object],
        response_format: Mapping[str, object],
        reasoning_controls: Mapping[str, object],
        components: Sequence[HeaderComponentRef],
        previous_header_id: str | None = None,
    ) -> RequestHeaderRecord:
        scalars = (
            _nonempty(provider_name, "provider_name"),
            _nonempty(model_name, "model_name"),
            _nonempty(route_identity, "route_identity"),
            _nonempty(endpoint_identity, "endpoint_identity"),
            _json_object(generation_parameters, "generation_parameters"),
            _json_object(adapter_defaults, "adapter_defaults"),
            _json_object(response_format, "response_format"),
            _json_object(reasoning_controls, "reasoning_controls"),
        )
        normalized = tuple(components)
        if any(not isinstance(item, HeaderComponentRef) for item in normalized):
            raise TypeError("components")
        keys = [(item.component_kind, item.ordinal) for item in normalized]
        if len(set(keys)) != len(keys):
            raise ValueError("components")
        normalized = tuple(
            sorted(normalized, key=lambda item: (item.component_kind, item.ordinal))
        )
        self._claim_write_intent(cursor)
        if previous_header_id is not None:
            previous = self._get_request_header(cursor, previous_header_id)
            if previous is None:
                raise ValueError("previous_header_id")
            previous_scalars = (
                previous.provider_name,
                previous.model_name,
                previous.route_identity,
                previous.endpoint_identity,
                _json_object(
                    previous.generation_parameters,
                    "generation_parameters",
                    allow_frozen=True,
                ),
                _json_object(
                    previous.adapter_defaults,
                    "adapter_defaults",
                    allow_frozen=True,
                ),
                _json_object(
                    previous.response_format,
                    "response_format",
                    allow_frozen=True,
                ),
                _json_object(
                    previous.reasoning_controls,
                    "reasoning_controls",
                    allow_frozen=True,
                ),
            )
            if previous_scalars == scalars and previous.components == normalized:
                return previous
            candidates: Sequence[sqlite3.Row] = ()
        else:
            candidates = cursor.execute(
                """SELECT header_id FROM console_trace_request_headers
                    WHERE provider_name = ? AND model_name = ? AND route_identity = ?
                      AND endpoint_identity = ? AND generation_parameters_json = ?
                      AND adapter_defaults_json = ? AND response_format_json = ?
                      AND reasoning_controls_json = ? ORDER BY header_id""",
                scalars,
            ).fetchall()
        for row in candidates:
            header_id = str(row[0])
            if self._header_components(cursor, header_id) == normalized:
                record = self._get_request_header(cursor, header_id)
                assert record is not None
                return record
        header_id = new_opaque_id()
        cursor.execute(
            """INSERT INTO console_trace_request_headers(
                   header_id, provider_name, model_name, route_identity,
                   endpoint_identity, generation_parameters_json,
                   adapter_defaults_json, response_format_json,
                   reasoning_controls_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (header_id, *scalars),
        )
        cursor.executemany(
            """INSERT INTO console_trace_header_components(
                   header_id, component_kind, ordinal, artifact_id)
                 VALUES (?, ?, ?, ?)""",
            [
                (header_id, item.component_kind, item.ordinal, item.artifact_id)
                for item in normalized
            ],
        )
        if normalized:
            self._advance_graph_epoch(cursor)
        record = self._get_request_header(cursor, header_id)
        assert record is not None
        return record

    def get_request_header(
        self,
        cursor: sqlite3.Cursor,
        header_id: str,
    ) -> RequestHeaderRecord | None:
        return self._get_request_header(cursor, header_id)

    def reserve_call(
        self,
        cursor: sqlite3.Cursor,
        *,
        owner_id: str,
        segment_id: str,
        turn_id: str,
        run_id: str,
        call_sequence: int,
        idempotency_key: str,
        policy_id: str,
    ) -> TraceCallRecord:
        expected = (
            owner_id,
            segment_id,
            turn_id,
            run_id,
            call_sequence,
            idempotency_key,
            policy_id,
        )
        self._claim_write_intent(cursor)
        existing = self._reconcile_call_reservation(
            expected,
            self.get_call_by_idempotency_key(cursor, idempotency_key),
            self.get_call_by_logical_identity(
                cursor,
                owner_id=owner_id,
                segment_id=segment_id,
                turn_id=turn_id,
                run_id=run_id,
                call_sequence=call_sequence,
            ),
        )
        if existing is not None:
            return existing
        effective_owner = self.get_effective_owner(cursor, segment_id)
        if (
            effective_owner is None
            or effective_owner.owner_id != owner_id
            or not effective_owner.attached
        ):
            raise TraceIdentityConflict("call_owner")
        call_id = new_opaque_id()
        try:
            cursor.execute(
                """INSERT INTO console_trace_calls(
                       call_id, owner_id, segment_id, turn_id, run_id, call_sequence,
                       idempotency_key, policy_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (call_id, *expected),
            )
        except sqlite3.IntegrityError:
            existing = self._reconcile_call_reservation(
                expected,
                self.get_call_by_idempotency_key(cursor, idempotency_key),
                self.get_call_by_logical_identity(
                    cursor,
                    owner_id=owner_id,
                    segment_id=segment_id,
                    turn_id=turn_id,
                    run_id=run_id,
                    call_sequence=call_sequence,
                ),
            )
            if existing is not None:
                return existing
            raise
        self._advance_graph_epoch(cursor)
        record = self.get_call(cursor, call_id)
        assert record is not None
        return record

    def get_call(self, cursor: sqlite3.Cursor, call_id: str) -> TraceCallRecord | None:
        row = cursor.execute(
            self._call_select() + " WHERE call_id = ?", (call_id,)
        ).fetchone()
        return None if row is None else self._call(row)

    def get_call_by_idempotency_key(
        self,
        cursor: sqlite3.Cursor,
        idempotency_key: str,
    ) -> TraceCallRecord | None:
        row = cursor.execute(
            self._call_select() + " WHERE idempotency_key = ?",
            (idempotency_key,),
        ).fetchone()
        return None if row is None else self._call(row)

    def get_call_by_logical_identity(
        self,
        cursor: sqlite3.Cursor,
        *,
        owner_id: str,
        segment_id: str,
        turn_id: str,
        run_id: str,
        call_sequence: int,
    ) -> TraceCallRecord | None:
        """Return the reservation occupying one immutable logical call slot."""

        row = cursor.execute(
            self._call_select()
            + """ WHERE owner_id = ? AND segment_id = ? AND turn_id = ?
                    AND run_id = ? AND call_sequence = ?""",
            (owner_id, segment_id, turn_id, run_id, call_sequence),
        ).fetchone()
        return None if row is None else self._call(row)

    @staticmethod
    def _reconcile_call_reservation(
        expected: tuple[str, str, str, str, int, str, str],
        by_idempotency_key: TraceCallRecord | None,
        by_logical_identity: TraceCallRecord | None,
    ) -> TraceCallRecord | None:
        if by_idempotency_key is None and by_logical_identity is None:
            return None
        if (
            by_idempotency_key is not None
            and by_logical_identity is not None
            and by_idempotency_key.call_id != by_logical_identity.call_id
        ):
            raise TraceIdentityConflict("call_reservation")
        existing = by_idempotency_key or by_logical_identity
        assert existing is not None
        actual = (
            existing.owner_id,
            existing.segment_id,
            existing.turn_id,
            existing.run_id,
            existing.call_sequence,
            existing.idempotency_key,
            existing.policy_id,
        )
        if actual != expected:
            raise TraceIdentityConflict("call_reservation")
        return existing

    def read_calls(
        self,
        cursor: sqlite3.Cursor,
        owner_id: str,
    ) -> tuple[TraceCallRecord, ...]:
        rows = cursor.execute(
            self._call_select()
            + " WHERE owner_id = ? ORDER BY turn_id, run_id, call_sequence, call_id",
            (owner_id,),
        ).fetchall()
        return tuple(self._call(row) for row in rows)

    def bind_call(
        self,
        cursor: sqlite3.Cursor,
        *,
        call_id: str,
        surface_node_id: str,
        request_header_id: str,
        provider_name: str,
        model_name: str,
        route_identity: str,
    ) -> TraceCallRecord:
        self._claim_write_intent(cursor)
        existing = self.get_call(cursor, call_id)
        if existing is None:
            raise KeyError("call_id")
        requested = (
            surface_node_id,
            request_header_id,
            provider_name,
            model_name,
            route_identity,
        )
        current = (
            existing.surface_node_id,
            existing.request_header_id,
            existing.provider_name,
            existing.model_name,
            existing.route_identity,
        )
        if existing.surface_node_id is not None:
            if current != requested:
                raise TraceIdentityConflict("call_binding")
            return existing
        cursor.execute(
            """UPDATE console_trace_calls
                  SET surface_node_id = ?, request_header_id = ?, provider_name = ?,
                      model_name = ?, route_identity = ?
                WHERE call_id = ? AND surface_node_id IS NULL""",
            (*requested, call_id),
        )
        if cursor.rowcount != 1:
            raise TraceIdentityConflict("call_binding")
        self._advance_graph_epoch(cursor)
        record = self.get_call(cursor, call_id)
        assert record is not None
        return record

    def advance_call_state(
        self,
        cursor: sqlite3.Cursor,
        *,
        call_id: str,
        target: TraceCallState,
        occurred_at: str,
        provider_operation_inactive: bool = False,
        usage: Mapping[str, object] | None = None,
        integrity_state: IntegrityState | None = None,
        omission_reason_code: str | None = None,
    ) -> TraceCallRecord:
        self._claim_write_intent(cursor)
        existing = self.get_call(cursor, call_id)
        if existing is None:
            raise KeyError("call_id")
        if not isinstance(target, TraceCallState):
            raise TypeError("target")
        _nonempty(occurred_at, "occurred_at")
        if integrity_state is not None and integrity_state not in {
            "pending",
            "complete",
            "incomplete",
        }:
            raise ValueError("integrity_state")
        if omission_reason_code is not None:
            _validate_token(omission_reason_code, "omission_reason_code")
        if target is existing.state and target is not TraceCallState.RESERVED:
            if self._lifecycle_retry_matches(
                existing,
                target=target,
                occurred_at=occurred_at,
                provider_operation_inactive=provider_operation_inactive,
                usage=usage,
                integrity_state=integrity_state,
                omission_reason_code=omission_reason_code,
            ):
                return existing
            raise TraceIdentityConflict("call_lifecycle_retry")
        validate_call_transition(
            existing.state,
            target,
            provider_operation_inactive=provider_operation_inactive,
        )
        dispatch_at = existing.dispatch_started_at
        response_at = existing.response_started_at
        settled_at = existing.settled_at
        inactive_at = existing.provider_inactive_at
        if target is TraceCallState.DISPATCH_STARTED:
            dispatch_at = occurred_at
        elif target is TraceCallState.RESPONSE_STARTED:
            response_at = occurred_at
        elif target not in {
            TraceCallState.RESERVED,
            TraceCallState.DISPATCH_STARTED,
            TraceCallState.RESPONSE_STARTED,
        }:
            settled_at = occurred_at
        if target is TraceCallState.ABANDONED:
            inactive_at = occurred_at
        outcome = (
            target.value
            if is_terminal_call_state(target)
            and target
            not in {TraceCallState.NOT_DISPATCHED, TraceCallState.DISPATCH_UNKNOWN}
            else None
        )
        cursor.execute(
            """UPDATE console_trace_calls
                  SET state = ?, dispatch_started_at = ?, response_started_at = ?,
                      settled_at = ?, provider_inactive_at = ?, outcome = ?,
                      usage_json = ?, integrity_state = ?, omission_reason_code = ?
                WHERE call_id = ? AND state = ?""",
            (
                target.value,
                dispatch_at,
                response_at,
                settled_at,
                inactive_at,
                outcome,
                None if usage is None else _json_object(usage, "usage"),
                integrity_state or existing.integrity_state,
                omission_reason_code or existing.omission_reason_code,
                call_id,
                existing.state.value,
            ),
        )
        if cursor.rowcount != 1:
            raise TraceIdentityConflict("call_lifecycle")
        record = self.get_call(cursor, call_id)
        assert record is not None
        return record

    @staticmethod
    def _lifecycle_retry_matches(
        existing: TraceCallRecord,
        *,
        target: TraceCallState,
        occurred_at: str,
        provider_operation_inactive: bool,
        usage: Mapping[str, object] | None,
        integrity_state: IntegrityState | None,
        omission_reason_code: str | None,
    ) -> bool:
        if target is TraceCallState.ABANDONED and not provider_operation_inactive:
            return False
        if target is TraceCallState.DISPATCH_STARTED:
            durable_time = existing.dispatch_started_at
        elif target is TraceCallState.RESPONSE_STARTED:
            durable_time = existing.response_started_at
        else:
            durable_time = existing.settled_at
        if durable_time != occurred_at:
            return False
        if (
            target is TraceCallState.ABANDONED
            and existing.provider_inactive_at != occurred_at
        ):
            return False
        if usage is not None:
            if existing.usage is None or _json_object(
                usage,
                "usage",
            ) != _json_object(existing.usage, "usage", allow_frozen=True):
                return False
        if integrity_state is not None and integrity_state != existing.integrity_state:
            return False
        return (
            omission_reason_code is None
            or omission_reason_code == existing.omission_reason_code
        )

    def append_event(
        self,
        cursor: sqlite3.Cursor,
        *,
        segment_id: str,
        sequence: int,
        event_type: TraceEventType,
        turn_id: str | None = None,
        call_id: str | None = None,
        surface_node_id: str | None = None,
        surface_replacement_id: str | None = None,
        request_header_id: str | None = None,
        semantic_revision_id: str | None = None,
        artifact_id: str | None = None,
        omission_reason_code: str | None = None,
    ) -> TraceEventRecord:
        if omission_reason_code is not None:
            _validate_token(omission_reason_code, "omission_reason_code")
        event_id = new_opaque_id()
        cursor.execute(
            """INSERT INTO console_trace_events(
                   event_id, segment_id, sequence, event_type, turn_id, call_id,
                   surface_node_id, surface_replacement_id, request_header_id,
                   semantic_revision_id, artifact_id, omission_reason_code)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event_id,
                segment_id,
                sequence,
                event_type,
                turn_id,
                call_id,
                surface_node_id,
                surface_replacement_id,
                request_header_id,
                semantic_revision_id,
                artifact_id,
                omission_reason_code,
            ),
        )
        self._advance_graph_epoch(cursor)
        record = self._get_event(cursor, event_id)
        assert record is not None
        return record

    def read_events(
        self,
        cursor: sqlite3.Cursor,
        segment_id: str,
    ) -> tuple[TraceEventRecord, ...]:
        rows = cursor.execute(
            self._event_select() + " WHERE segment_id = ? ORDER BY sequence, event_id",
            (segment_id,),
        ).fetchall()
        return tuple(TraceEventRecord(*row) for row in rows)

    def get_event_tail(
        self,
        cursor: sqlite3.Cursor,
        segment_id: str,
    ) -> TraceEventRecord | None:
        """Return the newest segment-local event without scanning history."""

        row = cursor.execute(
            self._event_select()
            + " WHERE segment_id = ? ORDER BY sequence DESC, event_id DESC LIMIT 1",
            (segment_id,),
        ).fetchone()
        return None if row is None else TraceEventRecord(*row)

    def store_response_link(
        self,
        cursor: sqlite3.Cursor,
        *,
        call_id: str,
        response: SemanticRevisionRef | TraceContentRef,
    ) -> TraceResponseLinkRecord:
        if isinstance(response, SemanticRevisionRef):
            shape = ("revision", response.revision_id, None, "verified_equal")
        elif isinstance(response, TraceContentRef):
            shape = ("artifact", None, response.content_id, "sanitized_artifact")
        else:
            raise TypeError("response")
        self._claim_write_intent(cursor)
        existing = self._get_response_link(cursor, call_id)
        if existing is not None:
            current = (
                existing.link_kind,
                existing.semantic_revision_id,
                existing.artifact_id,
                existing.verification_outcome,
            )
            if current != shape:
                raise TraceIdentityConflict("response_link")
            return existing
        response_link_id = new_opaque_id()
        cursor.execute(
            """INSERT INTO console_trace_response_links(
                   response_link_id, call_id, link_kind, semantic_revision_id,
                   artifact_id, verification_outcome) VALUES (?, ?, ?, ?, ?, ?)""",
            (response_link_id, call_id, *shape),
        )
        self._advance_graph_epoch(cursor)
        record = self._get_response_link(cursor, call_id)
        assert record is not None
        return record

    def get_response_link(
        self,
        cursor: sqlite3.Cursor,
        call_id: str,
    ) -> TraceResponseLinkRecord | None:
        return self._get_response_link(cursor, call_id)

    @staticmethod
    def _claim_write_intent(cursor: sqlite3.Cursor) -> None:
        if not cursor.connection.in_transaction:
            raise RuntimeError("caller_transaction_required")
        cursor.execute(
            """INSERT OR IGNORE INTO console_trace_graph_epoch(singleton_id, epoch)
                 VALUES (1, 0)"""
        )

    @staticmethod
    def _advance_graph_epoch(cursor: sqlite3.Cursor) -> None:
        cursor.execute(
            """UPDATE console_trace_graph_epoch
                  SET epoch = epoch + 1, updated_at = CURRENT_TIMESTAMP
                WHERE singleton_id = 1"""
        )
        if cursor.rowcount != 1:
            raise RuntimeError("graph_epoch_unavailable")

    @staticmethod
    def _owner(row: sqlite3.Row) -> TraceOwnerRecord:
        return TraceOwnerRecord(row[0], row[1], row[2], bool(row[3]), row[4])

    @staticmethod
    def _artifact(row: sqlite3.Row) -> TraceArtifactRecord:
        return TraceArtifactRecord(row[0], row[1], row[2], row[3], bytes(row[4]))

    @staticmethod
    def _call_select() -> str:
        return """SELECT call_id, owner_id, segment_id, turn_id, run_id,
                         call_sequence, idempotency_key, policy_id, state,
                         surface_node_id, request_header_id, provider_name,
                         model_name, route_identity, dispatch_started_at,
                         response_started_at, settled_at, provider_inactive_at,
                         outcome, usage_json, integrity_state, omission_reason_code
                    FROM console_trace_calls"""

    @staticmethod
    def _call(row: sqlite3.Row) -> TraceCallRecord:
        return TraceCallRecord(
            call_id=row[0],
            owner_id=row[1],
            segment_id=row[2],
            turn_id=row[3],
            run_id=row[4],
            call_sequence=row[5],
            idempotency_key=row[6],
            policy_id=row[7],
            state=TraceCallState(row[8]),
            surface_node_id=row[9],
            request_header_id=row[10],
            provider_name=row[11],
            model_name=row[12],
            route_identity=row[13],
            dispatch_started_at=row[14],
            response_started_at=row[15],
            settled_at=row[16],
            provider_inactive_at=row[17],
            outcome=row[18],
            usage=_decode_object(row[19]),
            integrity_state=row[20],
            omission_reason_code=row[21],
        )

    @staticmethod
    def _event_select() -> str:
        return """SELECT event_id, segment_id, sequence, event_type, turn_id,
                         call_id, surface_node_id, surface_replacement_id,
                         request_header_id, semantic_revision_id, artifact_id,
                         omission_reason_code FROM console_trace_events"""

    def _get_event(
        self,
        cursor: sqlite3.Cursor,
        event_id: str,
    ) -> TraceEventRecord | None:
        row = cursor.execute(
            self._event_select() + " WHERE event_id = ?",
            (event_id,),
        ).fetchone()
        return None if row is None else TraceEventRecord(*row)

    @staticmethod
    def _header_components(
        cursor: sqlite3.Cursor,
        header_id: str,
    ) -> tuple[HeaderComponentRef, ...]:
        rows = cursor.execute(
            """SELECT component_kind, ordinal, artifact_id
                 FROM console_trace_header_components WHERE header_id = ?
                ORDER BY component_kind, ordinal""",
            (header_id,),
        ).fetchall()
        return tuple(HeaderComponentRef(*row) for row in rows)

    def _get_request_header(
        self,
        cursor: sqlite3.Cursor,
        header_id: str,
    ) -> RequestHeaderRecord | None:
        row = cursor.execute(
            """SELECT header_id, provider_name, model_name, route_identity,
                      endpoint_identity, generation_parameters_json,
                      adapter_defaults_json, response_format_json,
                      reasoning_controls_json FROM console_trace_request_headers
                 WHERE header_id = ?""",
            (header_id,),
        ).fetchone()
        if row is None:
            return None
        objects = tuple(_decode_object(value) for value in row[5:9])
        if any(value is None for value in objects):
            raise ValueError("stored_header_json")
        return RequestHeaderRecord(
            row[0],
            row[1],
            row[2],
            row[3],
            row[4],
            cast(Mapping[str, object], objects[0]),
            cast(Mapping[str, object], objects[1]),
            cast(Mapping[str, object], objects[2]),
            cast(Mapping[str, object], objects[3]),
            self._header_components(cursor, header_id),
        )

    def _get_response_link(
        self,
        cursor: sqlite3.Cursor,
        call_id: str,
    ) -> TraceResponseLinkRecord | None:
        row = cursor.execute(
            """SELECT response_link_id, call_id, link_kind, semantic_revision_id,
                      artifact_id, verification_outcome
                 FROM console_trace_response_links WHERE call_id = ?""",
            (call_id,),
        ).fetchone()
        return None if row is None else TraceResponseLinkRecord(*row)
