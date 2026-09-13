"""Transaction-participating persistence for the Console semantic trace ledger.

All public operations accept a caller-owned cursor. This module never opens or
completes a transaction, so trace writes compose with the caller's Chat mutation.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, TypeAlias, cast

from tldw_chatbook.Chat.console_trace_models import (
    FrozenTracePolicy,
    SemanticRevisionRef,
    SurfaceReplacement,
    MAX_SURFACE_REPLACEMENT_SPAN,
    TraceCallState,
    TraceContentRef,
    TraceOmission,
    TraceReservationProvenance,
    PROMOTED_VOICE_IMPORT_REASON,
    is_terminal_call_state,
    new_opaque_id,
    validate_call_transition,
    _validate_stable_opaque_id,
)
from tldw_chatbook.Chat.console_trace_redaction import (
    CREDENTIAL_SANITIZER_UNAVAILABLE,
    CredentialSanitizer,
    PIIRedactionSpan,
    merge_pii_spans,
)

if TYPE_CHECKING:
    from tldw_chatbook.Chat.console_voice_trace_promotion import (
        PostDispatchTraceArtifact,
        PostDispatchTraceCall,
        PostDispatchTraceImport,
        PostDispatchTraceImportResult,
        PostDispatchTraceSurfaceComponent,
    )

if TYPE_CHECKING:
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

_TOKEN = re.compile(r"[a-z][a-z0-9]*(?:[_-][a-z0-9]+)*\Z", re.ASCII)
_TOKEN_MAX = 64
MESSAGE_CALL_LINEAGE_BATCH_SIZE = 128
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
_CANONICAL_TRACE_JSON = "canonical-json-v1"
_CONTENT_FREE_CREDENTIAL_OMISSION = {
    "omitted": CREDENTIAL_SANITIZER_UNAVAILABLE,
}


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
        _validate_stable_opaque_id(self.artifact_id, "artifact_id")


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
    reservation_provenance: TraceReservationProvenance
    import_reason_code: str | None


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


@dataclass(frozen=True, slots=True)
class TraceRedactionSpanRecord:
    """One immutable content-free mask bound to a source and policy."""

    span_id: str
    policy_id: str
    source_kind: Literal["revision", "artifact"]
    semantic_revision_id: str | None
    artifact_id: str | None
    field_path: str
    start_codepoint: int
    end_codepoint: int
    category: str
    rule_id: str
    detector_version: str
    outcome: Literal["applied", "omitted", "unavailable"]


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


def _sanitize_trace_scalar(value: str, field_name: str) -> str:
    """Return a mandatory credential-safe durable scalar."""

    value = _nonempty(value, field_name)
    result = CredentialSanitizer().sanitize(value)
    if not result.available or type(result.value) is not str or not result.value:
        return "[credential omitted]"
    return result.value


def _sanitize_trace_artifact_bytes(
    value: bytes,
    *,
    media_type: str,
    normalization_version: str,
) -> bytes:
    """Sanitize canonical JSON before its durable identity is calculated."""

    if (
        normalization_version != _CANONICAL_TRACE_JSON
        or media_type.split(";", 1)[0].strip().lower() != "application/json"
    ):
        return value
    try:
        decoded = json.loads(value.decode("utf-8"))
        result = CredentialSanitizer().sanitize(decoded)
        sanitized = (
            result.value
            if result.available
            else _CONTENT_FREE_CREDENTIAL_OMISSION
        )
        return json.dumps(
            sanitized,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError):
        return json.dumps(
            _CONTENT_FREE_CREDENTIAL_OMISSION,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")


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
        result = CredentialSanitizer().sanitize(canonical)
        sanitized = (
            result.value
            if result.available
            else _CONTENT_FREE_CREDENTIAL_OMISSION
        )
        encoded = json.dumps(
            sanitized, allow_nan=False, separators=(",", ":"), sort_keys=True
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

    def import_post_dispatch_trace(
        self,
        db: "CharactersRAGDB",
        request: PostDispatchTraceImport,
    ) -> PostDispatchTraceImportResult:
        """Atomically import one complete already-observed winning call set.

        This is the sole repository API that may insert terminal calls directly.
        It owns an immediate transaction and a connection-local exact-call grant;
        ordinary reservation APIs retain their pre-dispatch lifecycle.
        """
        from tldw_chatbook.Chat.console_voice_trace_promotion import (
            ConfirmedPreCommitTraceImportError,
            PostDispatchTraceImport,
        )

        if type(request) is not PostDispatchTraceImport:
            raise TypeError("request must be a PostDispatchTraceImport")
        connection = db.get_connection()
        managed_depth = getattr(db._local, "transaction_depth", 0)
        if managed_depth or connection.in_transaction:
            raise TraceIdentityConflict("post_dispatch_transaction_owner")
        self._validate_post_dispatch_request(request)
        transaction = db.transaction(immediate=True)
        transaction_body_error: Exception | None = None
        try:
            with transaction as cursor:
                try:
                    self._validate_post_dispatch_message_lineage(cursor, request)
                    existing = self._reconcile_post_dispatch_trace(cursor, request)
                    if existing is not None:
                        return existing
                    result = self._write_post_dispatch_trace(db, cursor, request)
                except Exception as exc:
                    transaction_body_error = exc
                    raise
        except Exception as exc:
            if (
                exc is transaction_body_error
                and not isinstance(exc, TraceIdentityConflict)
                and transaction.is_outermost_transaction
                and transaction.conn is not None
                and not transaction.conn.in_transaction
            ):
                raise ConfirmedPreCommitTraceImportError() from None
            raise
        self._after_post_dispatch_trace_commit(request)
        return result

    @staticmethod
    def _after_post_dispatch_trace_commit(request: PostDispatchTraceImport) -> None:
        """Private test seam for exception-after-commit reconciliation."""

        del request

    @staticmethod
    def _validate_post_dispatch_request(request: PostDispatchTraceImport) -> None:
        from tldw_chatbook.Chat.console_voice_trace_promotion import (
            MAX_PROMOTED_TRACE_BYTES,
            MAX_PROMOTED_TRACE_CALLS,
            derive_post_dispatch_trace_ids,
            derive_post_dispatch_trace_node_id,
        )
        calls = request.calls
        if not 1 <= len(calls) <= MAX_PROMOTED_TRACE_CALLS:
            raise ValueError("promoted calls must be complete and bounded")
        if len(calls) != request.expected_call_count:
            raise ValueError("promoted call aggregate is incomplete")
        if tuple(call.call_sequence for call in calls) != tuple(range(len(calls))):
            raise ValueError("promoted call sequence must be contiguous and ordered")
        if len({call.call_id for call in calls}) != len(calls) or len(
            {call.idempotency_key for call in calls}
        ) != len(calls):
            raise ValueError("promoted call identities must be unique")
        identities = derive_post_dispatch_trace_ids(
            request.import_id,
            call_count=len(calls),
        )
        if tuple(call.call_id for call in calls) != identities.call_ids:
            raise TraceIdentityConflict("promoted_call_ids")
        previous_surface: tuple[PostDispatchTraceSurfaceComponent, ...] = ()
        seen_node_ids: set[str] = set()
        aggregate_header_bytes = 0
        aggregate_artifacts: dict[str, PostDispatchTraceArtifact] = {}
        previous_settled_at: str | None = None
        for call in calls:
            for name in ("provider_name", "model_name", "route_identity", "endpoint_identity"):
                value = getattr(call, name)
                if _sanitize_trace_scalar(value, name) != value:
                    raise TraceIdentityConflict("post_dispatch_credential_projection")
            for name in ("generation_parameters_json", "adapter_defaults_json",
                         "response_format_json", "reasoning_controls_json", "usage_json"):
                value = getattr(call, name)
                if value is not None:
                    decoded = json.loads(value)
                    if json.loads(_json_object(decoded, name)) != decoded:
                        raise TraceIdentityConflict("post_dispatch_credential_projection")
            if (
                call.call_sequence < len(calls) - 1
                and call.response.kind == "committed_revision"
            ):
                raise TraceIdentityConflict("promoted_nonfinal_response")
            common_prefix = 0
            for previous, current in zip(
                previous_surface, call.request_surface, strict=False
            ):
                if previous != current:
                    break
                common_prefix += 1
            if previous_settled_at is not None and datetime.fromisoformat(
                call.dispatch_started_at[:-1] + "+00:00"
            ) < datetime.fromisoformat(previous_settled_at[:-1] + "+00:00"):
                raise ValueError("promoted call chronology must be sequential")
            for ordinal, component in enumerate(call.request_surface):
                if ordinal >= common_prefix:
                    expected_node_id = derive_post_dispatch_trace_node_id(
                        request.import_id,
                        call.call_sequence,
                        ordinal,
                    )
                    if component.node_id != expected_node_id:
                        raise TraceIdentityConflict("promoted_surface_node_ids")
                    if component.node_id in seen_node_ids:
                        raise TraceIdentityConflict("promoted_surface_node_ids")
                    seen_node_ids.add(component.node_id)
            inline_payload_bytes = sum(
                len(value.encode("utf-8"))
                for value in (
                    call.generation_parameters_json,
                    call.adapter_defaults_json,
                    call.response_format_json,
                    call.reasoning_controls_json,
                    *((call.usage_json,) if call.usage_json is not None else ()),
                )
            )
            call_artifacts = ConsoleTraceRepository._post_dispatch_call_artifacts(call)
            call_payload_bytes = inline_payload_bytes + sum(
                artifact.retained_bytes for artifact in call_artifacts
            )
            if call_payload_bytes != call.sealed_payload_bytes:
                raise ValueError("promoted call byte accounting is not exact")
            if (
                call.response.kind == "committed_revision"
                and call.response.committed_revision_id != request.assistant_revision_id
            ):
                raise TraceIdentityConflict("assistant_revision")
            aggregate_header_bytes += inline_payload_bytes
            for artifact in call_artifacts:
                if (
                    artifact.media_type.split(";", 1)[0].strip().lower()
                    != "application/json"
                    or artifact.normalization_version
                    not in (_CANONICAL_TRACE_JSON, "json-v1")
                ):
                    raise TraceIdentityConflict("post_dispatch_artifact_envelope")
                # Both supported JSON envelopes must satisfy the same mandatory
                # projection without changing sealed bytes or their identity.
                if _sanitize_trace_artifact_bytes(
                    artifact.sanitized_bytes, media_type="application/json",
                    normalization_version=_CANONICAL_TRACE_JSON,
                ) != artifact.sanitized_bytes:
                    raise TraceIdentityConflict("post_dispatch_artifact_projection")
                existing = aggregate_artifacts.get(artifact.artifact_id)
                if existing is not None and existing != artifact:
                    raise TraceIdentityConflict("post_dispatch_artifact_identity")
                aggregate_artifacts[artifact.artifact_id] = artifact
            previous_surface = call.request_surface
            previous_settled_at = call.settled_at
        aggregate_payload_bytes = aggregate_header_bytes + sum(
            artifact.retained_bytes for artifact in aggregate_artifacts.values()
        )
        if aggregate_payload_bytes > MAX_PROMOTED_TRACE_BYTES:
            raise ValueError("promoted aggregate exceeds the byte bound")
        if aggregate_payload_bytes != request.aggregate_payload_bytes:
            raise ValueError("promoted aggregate byte accounting is not exact")
        final_call = request.calls[-1]
        if final_call.response.kind != "committed_revision":
            raise TraceIdentityConflict("promoted_final_response")
        if final_call.terminal_state is not TraceCallState.COMPLETE:
            raise TraceIdentityConflict("promoted_final_state")
        if not any(
            component.reference_kind == "revision"
            and component.revision_id == request.user_revision_id
            for component in request.calls[-1].request_surface
        ):
            raise TraceIdentityConflict("promoted_final_user_surface")

    @staticmethod
    def _post_dispatch_call_artifacts(
        call: PostDispatchTraceCall,
    ) -> tuple[PostDispatchTraceArtifact, ...]:
        """Return retained call artifacts once per exact artifact identity."""

        artifacts: dict[str, PostDispatchTraceArtifact] = {}

        def retain(artifact: PostDispatchTraceArtifact | None) -> None:
            if artifact is None:
                return
            existing = artifacts.get(artifact.artifact_id)
            if existing is not None and existing != artifact:
                raise TraceIdentityConflict("post_dispatch_artifact_identity")
            artifacts[artifact.artifact_id] = artifact

        for component in call.request_surface:
            retain(component.artifact_value)
        for component in call.header_components:
            retain(component.artifact_value)
        retain(call.response.artifact_value)
        return tuple(artifacts.values())

    @staticmethod
    def _validate_post_dispatch_message_lineage(
        cursor: sqlite3.Cursor,
        request: PostDispatchTraceImport,
    ) -> None:
        conversation = cursor.execute(
            """SELECT id, deleted FROM conversations WHERE id = ?""",
            (request.conversation_id,),
        ).fetchone()
        if conversation is None or conversation[1]:
            raise TraceIdentityConflict("conversation")
        messages = cursor.execute(
            """SELECT id, conversation_id, parent_message_id, sender, role,
                      deleted, assistant_generation_state
                 FROM messages WHERE id IN (?, ?)""",
            (request.user_message_id, request.assistant_message_id),
        ).fetchall()
        by_id = {str(row[0]): row for row in messages}
        user = by_id.get(request.user_message_id)
        assistant = by_id.get(request.assistant_message_id)
        if (
            user is None
            or user[1] != request.conversation_id
            or user[3] != "user"
            or user[4] != "user"
            or user[5]
        ):
            raise TraceIdentityConflict("user_message")
        if request.turn_id != request.user_message_id:
            raise TraceIdentityConflict("turn_lineage")
        if (
            assistant is None
            or assistant[1] != request.conversation_id
            or assistant[2] != request.user_message_id
            or assistant[3] != "assistant"
            or assistant[4] != "assistant"
            or assistant[5]
            or assistant[6] != "complete"
        ):
            raise TraceIdentityConflict("assistant_message")
        surface_revision_ids = {
            component.revision_id
            for call in request.calls
            for component in call.request_surface
            if component.revision_id is not None
        }
        system_revision_ids = {
            component.revision_id
            for call in request.calls
            for component in call.system_composition
            if component.revision_id is not None
        }
        required_revision_ids = {
            request.user_revision_id,
            request.assistant_revision_id,
            *surface_revision_ids,
            *system_revision_ids,
        }
        placeholders = ",".join("?" for _ in required_revision_ids)
        revisions = cursor.execute(
            """SELECT revision_id, source_conversation_id, source_message_id,
                      revision_sequence, normalized_role, content_kind,
                      predecessor_revision_id, live_message_id,
                      live_locator_retired_at
                 FROM console_trace_semantic_revisions
                WHERE revision_id IN ("""
            + placeholders
            + ")",
            tuple(required_revision_ids),
        ).fetchall()
        revisions_by_id = {str(row[0]): row for row in revisions}
        if any(
            revision_id not in revisions_by_id
            or revisions_by_id[revision_id][1] != request.conversation_id
            for revision_id in required_revision_ids
        ):
            raise TraceIdentityConflict("surface_revision")
        for label, revision_id, message_id, role in (
            (
                "user_revision",
                request.user_revision_id,
                request.user_message_id,
                "user",
            ),
            (
                "assistant_revision",
                request.assistant_revision_id,
                request.assistant_message_id,
                "assistant",
            ),
        ):
            row = revisions_by_id.get(revision_id)
            if (
                row is None
                or row[1] != request.conversation_id
                or row[2] != message_id
                or row[3] != 0
                or row[4] != role
                or row[5] != "text"
                or row[6] is not None
                or row[7] != message_id
                or row[8] is not None
            ):
                raise TraceIdentityConflict(label)

    def _post_dispatch_initial_replacement(
        self,
        cursor: sqlite3.Cursor,
        *,
        segment_id: str,
        predecessor: SurfaceNodeRecord | None,
        replacement_node_id: str,
    ) -> SurfaceReplacement | None:
        if predecessor is None:
            return None
        # Reuse the ordinary reader's inherited/replaced projection algebra.
        # This short-lived service only reads; it reserves or settles nothing.
        from tldw_chatbook.Chat.console_trace_service import ConsoleTraceService

        projection = ConsoleTraceService(repository=self)._surface_projection(
            cursor, segment_id, predecessor
        )
        if not projection.entries:
            raise TraceIdentityConflict("post_dispatch_surface_predecessor")
        sequences = tuple(sequence for sequence, _key in projection.entries)
        start_sequence, end_sequence = min(sequences), max(sequences)
        if end_sequence - start_sequence + 1 > MAX_SURFACE_REPLACEMENT_SPAN:
            raise TraceIdentityConflict("post_dispatch_surface_span")
        nodes = self.read_lineage_surface_nodes(
            cursor, segment_id=segment_id,
            start_sequence=start_sequence, end_sequence=end_sequence,
        )
        return SurfaceReplacement(
            predecessor_head_id=predecessor.node_id,
            start_node_id=nodes[0].node_id, start_sequence=start_sequence,
            end_node_id=nodes[-1].node_id, end_sequence=end_sequence,
            replacement_node_id=replacement_node_id,
        )

    def _write_post_dispatch_trace(
        self,
        db: "CharactersRAGDB",
        cursor: sqlite3.Cursor,
        request: PostDispatchTraceImport,
    ) -> PostDispatchTraceImportResult:
        from tldw_chatbook.Chat.console_voice_trace_promotion import (
            PostDispatchTraceImportResult,
            derive_post_dispatch_trace_ids,
            derive_post_dispatch_trace_replacement_id,
        )
        identities = derive_post_dispatch_trace_ids(
            request.import_id,
            call_count=len(request.calls),
        )
        self._reject_post_dispatch_identity_residue(cursor, request)
        policy = self.ensure_policy(cursor, request.policy)
        self._admit_post_dispatch_revision_privacy(cursor, request)
        owner = self.get_attached_owner_by_conversation(
            cursor, request.conversation_id
        )
        if owner is None:
            cursor.execute(
                "INSERT INTO console_trace_segments(segment_id) VALUES (?)",
                (identities.root_segment_id,),
            )
            cursor.execute(
                """INSERT INTO console_trace_owners(
                       owner_id, conversation_id, root_segment_id, attached)
                     VALUES (?, ?, ?, 1)""",
                (
                    identities.owner_id,
                    request.conversation_id,
                    identities.root_segment_id,
                ),
            )
            self._advance_graph_epoch(cursor)
            owner = self.get_owner(cursor, identities.owner_id)
            assert owner is not None
        segment_id = owner.root_segment_id
        from tldw_chatbook.Chat.console_trace_service import ConsoleTraceService

        surface_tail = ConsoleTraceService(repository=self)._effective_surface_tail(
            cursor, segment_id
        )
        surface_sequence = 0 if surface_tail is None else surface_tail.sequence + 1
        predecessor_node_id = None if surface_tail is None else surface_tail.node_id
        previous_surface: tuple[PostDispatchTraceSurfaceComponent, ...] = ()
        surface_heads: dict[str, str] = {}
        new_surface_nodes: dict[str, tuple[str, ...]] = {}
        surface_replacement_ids: dict[str, str] = {}
        surface_replacement_targets: dict[str, str] = {}
        initial_replacement = self._post_dispatch_initial_replacement(
            cursor, segment_id=segment_id, predecessor=surface_tail,
            replacement_node_id=request.calls[0].request_surface[0].node_id,
        )
        for call in request.calls:
            common_prefix = 0
            for previous, current in zip(
                previous_surface, call.request_surface, strict=False
            ):
                if previous != current:
                    break
                common_prefix += 1
            appended_node_ids: list[str] = []
            for component in call.request_surface[common_prefix:]:
                artifact_id: str | None = None
                if component.artifact_value is not None:
                    artifact_id = self._ensure_post_dispatch_artifact(
                        cursor, component.artifact_value, policy
                    )
                cursor.execute(
                    """INSERT INTO console_trace_surface_nodes(
                           node_id, segment_id, sequence, predecessor_node_id,
                           component_kind, reference_kind, semantic_revision_id,
                           artifact_id, omission_reason_code)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        component.node_id,
                        segment_id,
                        surface_sequence,
                        predecessor_node_id,
                        component.component_kind,
                        component.reference_kind,
                        component.revision_id,
                        artifact_id,
                        component.omission_reason_code,
                    ),
                )
                self._advance_graph_epoch(cursor)
                predecessor_node_id = component.node_id
                surface_sequence += 1
                appended_node_ids.append(component.node_id)
            replacement = initial_replacement if call.call_sequence == 0 else None
            if previous_surface and common_prefix < len(previous_surface):
                start = self.get_surface_node(
                    cursor, previous_surface[common_prefix].node_id
                )
                end = self.get_surface_node(cursor, previous_surface[-1].node_id)
                assert start is not None and end is not None
                replacement = SurfaceReplacement(
                    predecessor_head_id=previous_surface[-1].node_id,
                    start_node_id=start.node_id, start_sequence=start.sequence,
                    end_node_id=end.node_id, end_sequence=end.sequence,
                    replacement_node_id=call.request_surface[common_prefix].node_id,
                )
            if replacement is not None:
                replacement_id = derive_post_dispatch_trace_replacement_id(
                    request.import_id, call.call_sequence
                )
                cursor.execute(
                    """INSERT INTO console_trace_surface_replacements(
                           replacement_id, segment_id, predecessor_head_id,
                           start_node_id, start_sequence, end_node_id, end_sequence,
                           replacement_node_id)
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
                surface_replacement_ids[call.call_id] = replacement_id
                surface_replacement_targets[call.call_id] = replacement.replacement_node_id
            surface_heads[call.call_id] = call.request_surface[-1].node_id
            new_surface_nodes[call.call_id] = tuple(appended_node_ids)
            previous_surface = call.request_surface
        for sequence, call in enumerate(request.calls):
            header_id = identities.header_ids[sequence]
            header_component_rows = []
            for component in call.header_components:
                artifact_id = self._ensure_post_dispatch_artifact(
                    cursor, component.artifact_value, policy
                )
                header_component_rows.append(
                    (
                        header_id,
                        component.component_kind,
                        component.ordinal,
                        artifact_id,
                    )
                )
            cursor.execute(
                """INSERT INTO console_trace_request_headers(
                       header_id, provider_name, model_name, route_identity,
                       endpoint_identity, generation_parameters_json,
                       adapter_defaults_json, response_format_json,
                       reasoning_controls_json)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    header_id,
                    call.provider_name,
                    call.model_name,
                    call.route_identity,
                    call.endpoint_identity,
                    call.generation_parameters_json,
                    call.adapter_defaults_json,
                    call.response_format_json,
                    call.reasoning_controls_json,
                ),
            )
            cursor.executemany(
                """INSERT INTO console_trace_header_components(
                       header_id, component_kind, ordinal, artifact_id)
                     VALUES (?, ?, ?, ?)""",
                header_component_rows,
            )
            if header_component_rows:
                self._advance_graph_epoch(cursor)
        authorization = db._voice_trace_import_authorization_for_repository(
            cursor.connection
        )
        with authorization._authorize(tuple(call.call_id for call in request.calls)):
            for sequence, call in enumerate(request.calls):
                cursor.execute(
                    """INSERT INTO console_trace_calls(
                           call_id, owner_id, segment_id, turn_id, run_id,
                           call_sequence, idempotency_key, policy_id, state,
                           surface_node_id, request_header_id, provider_name,
                           model_name, route_identity, dispatch_started_at,
                           response_started_at, settled_at, outcome, usage_json,
                           integrity_state, reservation_provenance,
                           import_reason_code)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                                 ?, ?, ?, ?, ?, 'complete',
                                 'post_dispatch_promoted',
                                 'provisional_voice_promoted')""",
                    (
                        call.call_id,
                        owner.owner_id,
                        segment_id,
                        request.turn_id,
                        request.run_id,
                        call.call_sequence,
                        call.idempotency_key,
                        policy.policy_id,
                        call.terminal_state.value,
                        surface_heads[call.call_id],
                        identities.header_ids[sequence],
                        call.provider_name,
                        call.model_name,
                        call.route_identity,
                        call.dispatch_started_at,
                        call.response_started_at,
                        call.settled_at,
                        call.terminal_state.value,
                        call.usage_json,
                    ),
                )
        for sequence, call in enumerate(request.calls):
            response = call.response
            if response.kind == "committed_revision":
                cursor.execute(
                    """INSERT INTO console_trace_response_links(
                           response_link_id, call_id, link_kind,
                           semantic_revision_id, verification_outcome)
                         VALUES (?, ?, 'revision', ?, 'verified_equal')""",
                    (
                        identities.response_link_ids[sequence],
                        call.call_id,
                        response.committed_revision_id,
                    ),
                )
            elif response.kind == "artifact":
                assert response.artifact_value is not None
                artifact_id = self._ensure_post_dispatch_artifact(
                    cursor, response.artifact_value, policy
                )
                cursor.execute(
                    """INSERT INTO console_trace_response_links(
                           response_link_id, call_id, link_kind, artifact_id,
                           verification_outcome)
                         VALUES (?, ?, 'artifact', ?, 'sanitized_artifact')""",
                    (
                        identities.response_link_ids[sequence],
                        call.call_id,
                        artifact_id,
                    ),
                )
        event_tail = self.get_event_tail(cursor, segment_id)
        event_sequence = 0 if event_tail is None else event_tail.sequence + 1

        def event(event_type: TraceEventType, **values: object) -> None:
            nonlocal event_sequence
            self.append_event(
                cursor,
                segment_id=segment_id,
                sequence=event_sequence,
                event_type=event_type,
                **values,  # type: ignore[arg-type]
            )
            event_sequence += 1

        event("turn_boundary", turn_id=request.turn_id)
        for sequence, call in enumerate(request.calls):
            for node_id in new_surface_nodes[call.call_id]:
                event("surface_append", surface_node_id=node_id)
                if node_id == surface_replacement_targets.get(call.call_id):
                    event("surface_replace", surface_replacement_id=surface_replacement_ids[call.call_id])
            header_id = identities.header_ids[sequence]
            event("call_boundary", call_id=call.call_id)
            event(
                "request_header_selection",
                call_id=call.call_id,
                request_header_id=header_id,
            )
            event(
                "provider_route_selection",
                call_id=call.call_id,
                request_header_id=header_id,
            )
            if call.response.kind == "committed_revision":
                event(
                    "response_selection",
                    call_id=call.call_id,
                    semantic_revision_id=call.response.committed_revision_id,
                )
            elif call.response.kind == "artifact":
                assert call.response.artifact_value is not None
                event(
                    "response_selection",
                    call_id=call.call_id,
                    artifact_id=call.response.artifact_value.artifact_id,
                )
            else:
                event(
                    "gap",
                    omission_reason_code=call.response.omission_reason_code,
                )
            event("call_outcome", call_id=call.call_id)
            if call.usage_json is not None:
                event("usage", call_id=call.call_id)
        return PostDispatchTraceImportResult(
            conversation_id=request.conversation_id,
            owner_id=owner.owner_id,
            segment_id=segment_id,
            call_ids=tuple(call.call_id for call in request.calls),
            already_imported=False,
        )

    def _admit_post_dispatch_revision_privacy(
        self,
        cursor: sqlite3.Cursor,
        request: PostDispatchTraceImport,
    ) -> None:
        # This fresh-import step shares the call transaction. Reconciliation
        # never invokes detectors again, including for revisions with no spans.
        from tldw_chatbook.Chat.console_semantic_revision import (
            project_semantic_revision_provider_message,
        )
        from tldw_chatbook.Chat.console_trace_custom_pii import (
            CUSTOM_PII_RULESET_UNAVAILABLE,
            redact_pii_value_for_ruleset_revision,
        )

        policy = request.policy
        if not policy.pii_redaction_enabled:
            return
        revision_ids = {request.user_revision_id, request.assistant_revision_id}
        for call in request.calls:
            revision_ids.update(
                component.revision_id for component in call.request_surface
                if component.revision_id is not None
            )
            if call.response.committed_revision_id is not None:
                revision_ids.add(call.response.committed_revision_id)
            revision_ids.update(
                component.revision_id for component in call.system_composition
                if component.revision_id is not None
            )
        for revision_id in sorted(revision_ids):
            omission = None
            try:
                value = project_semantic_revision_provider_message(
                    cursor, revision_id=revision_id,
                    expected_conversation_id=request.conversation_id,
                )
            except (ValueError, LookupError):
                omission = "trace_source_unavailable"
            if omission is None:
                credential = CredentialSanitizer().sanitize(value)
                if not credential.available:
                    omission = CREDENTIAL_SANITIZER_UNAVAILABLE
                elif policy.pii_ruleset_revision_id is None:
                    omission = CUSTOM_PII_RULESET_UNAVAILABLE
                else:
                    redaction = redact_pii_value_for_ruleset_revision(
                        credential.value, policy.pii_ruleset_revision_id,
                    )
                    if not redaction.available:
                        omission = redaction.omission_reason_code or CUSTOM_PII_RULESET_UNAVAILABLE
                    else:
                        by_path: dict[str, list[PIIRedactionSpan]] = {}
                        for item in redaction.field_redactions:
                            by_path.setdefault(item.field_path, []).append(item.span)
                        for field_path, spans in sorted(by_path.items()):
                            self.ensure_redaction_spans(
                                cursor, policy_id=policy.policy_id,
                                semantic_revision_id=revision_id, artifact_id=None,
                                field_path=field_path, spans=spans,
                            )
            if omission is not None:
                self.bind_revision_policy(
                    cursor, revision_id=revision_id, policy_id=policy.policy_id,
                    omission_reason_code=omission,
                )

    def _ensure_post_dispatch_artifact(
        self,
        cursor: sqlite3.Cursor,
        artifact: PostDispatchTraceArtifact,
        policy: FrozenTracePolicy,
    ) -> str:
        existing = cursor.execute(
            """SELECT identity_digest, media_type, normalization_version,
                      sanitized_bytes
                 FROM console_trace_artifacts WHERE artifact_id = ?""",
            (artifact.artifact_id,),
        ).fetchone()
        expected = (
            artifact.identity_digest,
            artifact.media_type,
            artifact.normalization_version,
            artifact.sanitized_bytes,
        )
        if existing is not None:
            stored = (existing[0], existing[1], existing[2], bytes(existing[3]))
            if stored != expected:
                raise TraceIdentityConflict("post_dispatch_artifact")
            self._ensure_post_dispatch_artifact_masks(cursor, artifact, policy)
            return artifact.artifact_id
        cursor.execute(
            """INSERT INTO console_trace_artifacts(
                   artifact_id, identity_digest, media_type, normalization_version,
                   sanitized_bytes, byte_length) VALUES (?, ?, ?, ?, ?, ?)""",
            (
                artifact.artifact_id,
                artifact.identity_digest,
                artifact.media_type,
                artifact.normalization_version,
                sqlite3.Binary(artifact.sanitized_bytes),
                len(artifact.sanitized_bytes),
            ),
        )
        self._ensure_post_dispatch_artifact_masks(cursor, artifact, policy)
        return artifact.artifact_id

    def _ensure_post_dispatch_artifact_masks(
        self, cursor: sqlite3.Cursor, artifact: PostDispatchTraceArtifact,
        policy: FrozenTracePolicy,
    ) -> None:
        by_path: dict[str, list[PIIRedactionSpan]] = {}
        for item in artifact.field_redactions:
            by_path.setdefault(item.field_path, []).append(item.span)
        for field_path, spans in sorted(by_path.items()):
            self.ensure_redaction_spans(
                cursor, policy_id=policy.policy_id, semantic_revision_id=None,
                artifact_id=artifact.artifact_id, field_path=field_path, spans=spans,
            )

    @staticmethod
    def _reject_post_dispatch_identity_residue(
        cursor: sqlite3.Cursor,
        request: PostDispatchTraceImport,
    ) -> None:
        from tldw_chatbook.Chat.console_voice_trace_promotion import (
            derive_post_dispatch_trace_ids,
            derive_post_dispatch_trace_replacement_id,
        )
        identities = derive_post_dispatch_trace_ids(
            request.import_id,
            call_count=len(request.calls),
        )
        surface_node_ids = tuple(
            dict.fromkeys(
                component.node_id
                for call in request.calls
                for component in call.request_surface
            )
        )
        replacement_ids: list[str] = [
            derive_post_dispatch_trace_replacement_id(request.import_id, 0)
        ]
        previous_surface: tuple[PostDispatchTraceSurfaceComponent, ...] = ()
        for call in request.calls:
            common_prefix = 0
            for previous, current in zip(
                previous_surface, call.request_surface, strict=False
            ):
                if previous != current:
                    break
                common_prefix += 1
            if previous_surface and common_prefix < len(previous_surface):
                replacement_ids.append(
                    derive_post_dispatch_trace_replacement_id(
                        request.import_id, call.call_sequence
                    )
                )
            previous_surface = call.request_surface
        checks = (
            ("console_trace_calls", "call_id", identities.call_ids),
            (
                "console_trace_request_headers",
                "header_id",
                identities.header_ids,
            ),
            (
                "console_trace_response_links",
                "response_link_id",
                identities.response_link_ids,
            ),
            (
                "console_trace_surface_nodes",
                "node_id",
                surface_node_ids,
            ),
            (
                "console_trace_surface_replacements",
                "replacement_id",
                tuple(replacement_ids),
            ),
        )
        for table, column, values in checks:
            if not values:
                continue
            placeholders = ",".join("?" for _ in values)
            if (
                cursor.execute(
                    f'SELECT 1 FROM "{table}" WHERE "{column}" IN ({placeholders}) LIMIT 1',
                    values,
                ).fetchone()
                is not None
            ):
                raise TraceIdentityConflict("post_dispatch_residue")

    def _reconcile_post_dispatch_trace(
        self,
        cursor: sqlite3.Cursor,
        request: PostDispatchTraceImport,
    ) -> PostDispatchTraceImportResult | None:
        from tldw_chatbook.Chat.console_voice_trace_promotion import (
            PostDispatchTraceImportResult,
            derive_post_dispatch_trace_ids,
            derive_post_dispatch_trace_replacement_id,
        )
        calls = tuple(self.get_call(cursor, call.call_id) for call in request.calls)
        present = tuple(call for call in calls if call is not None)
        if not present:
            return None
        if len(present) != len(request.calls):
            raise TraceIdentityConflict("post_dispatch_partial_calls")
        assert all(call is not None for call in calls)
        stored_calls = cast(tuple[TraceCallRecord, ...], calls)
        owner_id = stored_calls[0].owner_id
        segment_id = stored_calls[0].segment_id
        owner = self.get_owner(cursor, owner_id)
        policy = self.get_policy(cursor, request.policy.policy_id)
        if (
            owner is None
            or not owner.attached
            or owner.conversation_id != request.conversation_id
            or policy != request.policy
        ):
            raise TraceIdentityConflict("post_dispatch_owner_policy")
        identities = derive_post_dispatch_trace_ids(
            request.import_id,
            call_count=len(request.calls),
        )
        expected_components: list[PostDispatchTraceSurfaceComponent] = []
        seen_node_ids: set[str] = set()
        for call in request.calls:
            for component in call.request_surface:
                if component.node_id not in seen_node_ids:
                    expected_components.append(component)
                    seen_node_ids.add(component.node_id)
        stored_surfaces = []
        for component in expected_components:
            surface = self.get_surface_node(cursor, component.node_id)
            artifact_id = (
                None
                if component.artifact_value is None
                else component.artifact_value.artifact_id
            )
            if (
                surface is None
                or surface.segment_id != segment_id
                or surface.component_kind != component.component_kind
                or surface.reference_kind != component.reference_kind
                or surface.semantic_revision_id != component.revision_id
                or surface.artifact_id != artifact_id
                or surface.omission_reason_code != component.omission_reason_code
                or (
                    component.artifact_value is not None
                    and not self._post_dispatch_artifact_matches(
                        cursor, component.artifact_value, request.policy
                    )
                )
            ):
                raise TraceIdentityConflict("post_dispatch_surface")
            stored_surfaces.append(surface)
        for ordinal, surface in enumerate(stored_surfaces):
            if ordinal:
                predecessor = stored_surfaces[ordinal - 1]
                if (
                    surface.sequence != predecessor.sequence + 1
                    or surface.predecessor_node_id != predecessor.node_id
                ):
                    raise TraceIdentityConflict("post_dispatch_surface_lineage")
            elif surface.predecessor_node_id is None:
                if surface.sequence != 0:
                    raise TraceIdentityConflict("post_dispatch_surface_lineage")
            else:
                predecessor = self.get_surface_node(cursor, surface.predecessor_node_id)
                segment = self.get_segment(cursor, segment_id)
                if (
                    predecessor is None
                    or (
                        predecessor.segment_id != segment_id
                        and (segment is None or predecessor.node_id != segment.inherited_surface_head_id)
                    )
                    or predecessor.sequence + 1 != surface.sequence
                ):
                    raise TraceIdentityConflict("post_dispatch_surface_lineage")
        surfaces_by_id = {surface.node_id: surface for surface in stored_surfaces}
        replacements_by_id = {
            record.replacement_id: record.replacement
            for record in self.read_surface_replacements(cursor, segment_id)
        }
        expected_replacement_ids: list[str] = []
        initial_predecessor_id = stored_surfaces[0].predecessor_node_id
        initial_replacement = self._post_dispatch_initial_replacement(
            cursor, segment_id=segment_id,
            predecessor=(None if initial_predecessor_id is None else self.get_surface_node(cursor, initial_predecessor_id)),
            replacement_node_id=request.calls[0].request_surface[0].node_id,
        )
        previous_surface = ()
        for call in request.calls:
            common_prefix = 0
            for previous, current in zip(
                previous_surface, call.request_surface, strict=False
            ):
                if previous != current:
                    break
                common_prefix += 1
            expected_replacement = initial_replacement if call.call_sequence == 0 else None
            if previous_surface and common_prefix < len(previous_surface):
                replacement_id = derive_post_dispatch_trace_replacement_id(
                    request.import_id, call.call_sequence
                )
                start = surfaces_by_id[previous_surface[common_prefix].node_id]
                end = surfaces_by_id[previous_surface[-1].node_id]
                expected_replacement = SurfaceReplacement(
                    predecessor_head_id=previous_surface[-1].node_id,
                    start_node_id=start.node_id,
                    start_sequence=start.sequence,
                    end_node_id=end.node_id,
                    end_sequence=end.sequence,
                    replacement_node_id=call.request_surface[common_prefix].node_id,
                )
            replacement_id = derive_post_dispatch_trace_replacement_id(request.import_id, call.call_sequence)
            if expected_replacement is not None:
                expected_replacement_ids.append(replacement_id)
                if replacements_by_id.get(replacement_id) != expected_replacement:
                    raise TraceIdentityConflict("post_dispatch_surface_replacement")
            elif replacement_id in replacements_by_id:
                raise TraceIdentityConflict("post_dispatch_surface_replacement")
            previous_surface = call.request_surface
        for sequence, (expected, stored) in enumerate(
            zip(request.calls, stored_calls, strict=True)
        ):
            header = self.get_request_header(cursor, identities.header_ids[sequence])
            response = self.get_response_link(cursor, expected.call_id)
            if header is None:
                raise TraceIdentityConflict("post_dispatch_call_links")
            expected_header_components = tuple(
                HeaderComponentRef(
                    component.component_kind,
                    component.ordinal,
                    component.artifact_value.artifact_id,
                )
                for component in expected.header_components
            )
            if (
                header.provider_name != expected.provider_name
                or header.model_name != expected.model_name
                or header.route_identity != expected.route_identity
                or header.endpoint_identity != expected.endpoint_identity
                or _json_object(
                    header.generation_parameters,
                    "generation_parameters",
                    allow_frozen=True,
                )
                != expected.generation_parameters_json
                or _json_object(
                    header.adapter_defaults,
                    "adapter_defaults",
                    allow_frozen=True,
                )
                != expected.adapter_defaults_json
                or _json_object(
                    header.response_format,
                    "response_format",
                    allow_frozen=True,
                )
                != expected.response_format_json
                or _json_object(
                    header.reasoning_controls,
                    "reasoning_controls",
                    allow_frozen=True,
                )
                != expected.reasoning_controls_json
                or header.components != expected_header_components
                or any(
                    not self._post_dispatch_artifact_matches(
                        cursor, component.artifact_value, request.policy
                    )
                    for component in expected.header_components
                )
            ):
                raise TraceIdentityConflict("post_dispatch_header")
            expected_usage = (
                None
                if expected.usage_json is None
                else _decode_object(expected.usage_json)
            )
            if (
                stored.owner_id != owner_id
                or stored.segment_id != segment_id
                or stored.turn_id != request.turn_id
                or stored.run_id != request.run_id
                or stored.call_sequence != expected.call_sequence
                or stored.idempotency_key != expected.idempotency_key
                or stored.policy_id != request.policy.policy_id
                or stored.state is not expected.terminal_state
                or stored.surface_node_id != expected.request_surface[-1].node_id
                or stored.request_header_id != identities.header_ids[sequence]
                or stored.provider_name != expected.provider_name
                or stored.model_name != expected.model_name
                or stored.route_identity != expected.route_identity
                or stored.dispatch_started_at != expected.dispatch_started_at
                or stored.response_started_at != expected.response_started_at
                or stored.settled_at != expected.settled_at
                or stored.outcome != expected.terminal_state.value
                or stored.usage != expected_usage
                or stored.integrity_state != "complete"
                or stored.omission_reason_code is not None
                or stored.reservation_provenance
                is not TraceReservationProvenance.POST_DISPATCH_PROMOTED
                or stored.import_reason_code != PROMOTED_VOICE_IMPORT_REASON
            ):
                raise TraceIdentityConflict("post_dispatch_call")
            expected_response = expected.response
            if expected_response.kind == "committed_revision":
                if (
                    response is None
                    or response.response_link_id
                    != identities.response_link_ids[sequence]
                    or response.link_kind != "revision"
                    or response.semantic_revision_id
                    != expected_response.committed_revision_id
                    or response.artifact_id is not None
                    or response.verification_outcome != "verified_equal"
                ):
                    raise TraceIdentityConflict("post_dispatch_response")
            elif expected_response.kind == "artifact":
                artifact = expected_response.artifact_value
                assert artifact is not None
                if (
                    response is None
                    or response.response_link_id
                    != identities.response_link_ids[sequence]
                    or response.link_kind != "artifact"
                    or response.semantic_revision_id is not None
                    or response.artifact_id != artifact.artifact_id
                    or response.verification_outcome != "sanitized_artifact"
                    or not self._post_dispatch_artifact_matches(cursor, artifact, request.policy)
                ):
                    raise TraceIdentityConflict("post_dispatch_response")
            elif (
                response is not None
                or cursor.execute(
                    """SELECT 1 FROM console_trace_response_links
                    WHERE response_link_id = ?""",
                    (identities.response_link_ids[sequence],),
                ).fetchone()
                is not None
            ):
                raise TraceIdentityConflict("post_dispatch_response")
        events = tuple(self.read_events(cursor, segment_id))
        if (
            not any(
                event.event_type == "turn_boundary" and event.turn_id == request.turn_id
                for event in events
            )
            or any(
                not any(
                    event.event_type == "surface_append"
                    and event.surface_node_id == component.node_id
                    for event in events
                )
                for component in expected_components
            )
            or any(
                not any(
                    event.event_type == "surface_replace"
                    and event.surface_replacement_id == replacement_id
                    for event in events
                )
                for replacement_id in expected_replacement_ids
            )
        ):
            raise TraceIdentityConflict("post_dispatch_events")
        for sequence, call in enumerate(request.calls):
            call_events = tuple(
                event for event in events if event.call_id == call.call_id
            )
            boundaries = tuple(event for event in call_events if event.event_type == "call_boundary")
            if len(boundaries) != 1 or self.surface_head_at_event_boundary(
                cursor, segment_id=segment_id, through_sequence=boundaries[0].sequence,
            ) != call.request_surface[-1].node_id:
                raise TraceIdentityConflict("post_dispatch_surface_event_head")
            # Gap events are segment-scoped by the existing schema. Their
            # position immediately after this call's route binds the omission.
            if call.response.kind == "no_response":
                route_events = tuple(
                    event for event in call_events
                    if event.event_type == "provider_route_selection"
                )
                if len(route_events) != 1:
                    raise TraceIdentityConflict("post_dispatch_events")
                call_events += tuple(
                    event for event in events
                    if event.sequence == route_events[0].sequence + 1
                    and event.event_type == "gap" and event.call_id is None
                )
            expected_event_types = [
                "call_boundary",
                "request_header_selection",
                "provider_route_selection",
                "response_selection" if call.response.kind != "no_response" else "gap",
                "call_outcome",
            ]
            if call.usage_json is not None:
                expected_event_types.append("usage")
            if sorted(event.event_type for event in call_events) != sorted(
                expected_event_types
            ):
                raise TraceIdentityConflict("post_dispatch_events")
            required_event_shapes = (
                ("call_boundary", None, None, None),
                (
                    "request_header_selection",
                    identities.header_ids[sequence],
                    None,
                    None,
                ),
                (
                    "provider_route_selection",
                    identities.header_ids[sequence],
                    None,
                    None,
                ),
                ("call_outcome", None, None, None),
            )
            if any(
                not any(
                    event.event_type == event_type
                    and event.request_header_id == header_id
                    and event.semantic_revision_id == revision_id
                    and event.artifact_id == artifact_id
                    for event in call_events
                )
                for event_type, header_id, revision_id, artifact_id in required_event_shapes
            ):
                raise TraceIdentityConflict("post_dispatch_events")
            response_events = tuple(
                event
                for event in call_events
                if event.event_type in {"response_selection", "gap"}
            )
            if call.response.kind == "committed_revision":
                response_events_match = len(response_events) == 1 and (
                    response_events[0].event_type == "response_selection"
                    and response_events[0].semantic_revision_id
                    == call.response.committed_revision_id
                    and response_events[0].artifact_id is None
                )
            elif call.response.kind == "artifact":
                assert call.response.artifact_value is not None
                response_events_match = len(response_events) == 1 and (
                    response_events[0].event_type == "response_selection"
                    and response_events[0].semantic_revision_id is None
                    and response_events[0].artifact_id
                    == call.response.artifact_value.artifact_id
                )
            else:
                response_events_match = len(response_events) == 1 and (
                    response_events[0].event_type == "gap"
                    and response_events[0].omission_reason_code
                    == call.response.omission_reason_code
                )
            usage_events = tuple(
                event for event in call_events if event.event_type == "usage"
            )
            if not response_events_match or len(usage_events) != int(
                call.usage_json is not None
            ):
                raise TraceIdentityConflict("post_dispatch_events")
        return PostDispatchTraceImportResult(
            conversation_id=request.conversation_id,
            owner_id=owner_id,
            segment_id=segment_id,
            call_ids=tuple(call.call_id for call in request.calls),
            already_imported=True,
        )

    @staticmethod
    def _post_dispatch_artifact_matches(
        cursor: sqlite3.Cursor,
        artifact: PostDispatchTraceArtifact,
        policy: FrozenTracePolicy,
    ) -> bool:
        by_path: dict[str, list[PIIRedactionSpan]] = {}
        for item in artifact.field_redactions:
            by_path.setdefault(item.field_path, []).append(item.span)
        expected_masks = tuple(
            (path, span.start_codepoint, span.end_codepoint, span.category,
             span.rule_id, span.detector_version, "applied")
            for path, spans in sorted(by_path.items()) for span in merge_pii_spans(spans)
        )
        stored_masks = tuple(tuple(row) for row in cursor.execute(
            """SELECT field_path, start_codepoint, end_codepoint, category,
                      rule_id, detector_version, outcome
                 FROM console_trace_redaction_spans
                WHERE policy_id = ? AND artifact_id = ?
                ORDER BY field_path, start_codepoint, end_codepoint, span_id""",
            (policy.policy_id, artifact.artifact_id),
        ))
        if stored_masks != expected_masks:
            return False
        row = cursor.execute(
            """SELECT identity_digest, media_type, normalization_version,
                      sanitized_bytes
                 FROM console_trace_artifacts WHERE artifact_id = ?""",
            (artifact.artifact_id,),
        ).fetchone()
        return row is not None and (
            row[0],
            row[1],
            row[2],
            bytes(row[3]),
        ) == (
            artifact.identity_digest,
            artifact.media_type,
            artifact.normalization_version,
            artifact.sanitized_bytes,
        )

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
        surface_head_id = self.surface_head_at_event_boundary(
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
        current_head = self.surface_head_at_event_boundary(
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

    def iter_message_call_lineage(
        self,
        cursor: sqlite3.Cursor,
        conversation_id: str,
        message_id: str,
        *,
        turn_id: str | None = None,
    ) -> Iterator[TraceCallRecord]:
        """Yield only calls associated with one message in bounded SQL pages.

        Args:
            cursor: Caller-owned transaction cursor.
            conversation_id: Durable conversation whose attached lineage is read.
            message_id: Selected request or response message identity.
            turn_id: Durable user-turn identity associated with the selected
                message, when it differs from ``message_id``.

        Yields:
            Matching trace calls in root-to-leaf event order.

        Raises:
            RuntimeError: If a referenced lineage call cannot be reconstructed.
        """

        associated_turn_id = message_id if turn_id is None else turn_id
        _nonempty(associated_turn_id, "turn_id")
        owner_row = cursor.execute(
            """SELECT owner_id, conversation_id, root_segment_id, attached,
                      detached_at FROM console_trace_owners
                 WHERE conversation_id = ? AND attached = 1""",
            (conversation_id,),
        ).fetchone()
        if owner_row is None:
            return
        owner = self._owner(owner_row)
        for segment_id, through_sequence in self._segment_lineage(
            cursor,
            owner.root_segment_id,
        ):
            last_sequence = -1
            last_call_id = ""
            while True:
                params: list[object] = [
                    segment_id,
                    message_id,
                    associated_turn_id,
                    message_id,
                    last_sequence,
                    last_sequence,
                    last_call_id,
                ]
                boundary_clause = ""
                if through_sequence is not None:
                    boundary_clause = " AND first_sequence <= ?"
                    params.append(through_sequence)
                params.append(MESSAGE_CALL_LINEAGE_BATCH_SIZE)
                rows = cursor.execute(
                    """WITH matching AS (
                           SELECT call.call_id,
                                  MIN(event.sequence) AS first_sequence
                             FROM console_trace_calls AS call
                             JOIN console_trace_events AS event
                               ON event.call_id = call.call_id
                        LEFT JOIN console_trace_response_links AS response
                               ON response.call_id = call.call_id
                        LEFT JOIN console_trace_semantic_revisions AS revision
                               ON revision.revision_id = response.semantic_revision_id
                            WHERE call.segment_id = ?
                              AND (call.turn_id = ?
                                   OR call.turn_id = ?
                                   OR revision.source_message_id = ?)
                         GROUP BY call.call_id
                       )
                       SELECT call_id, first_sequence
                         FROM matching
                        WHERE (first_sequence > ?
                               OR (first_sequence = ? AND call_id > ?))"""
                    + boundary_clause
                    + " ORDER BY first_sequence, call_id LIMIT ?",
                    tuple(params),
                ).fetchall()
                if not rows:
                    break
                for row in rows:
                    call = self.get_call(cursor, row[0])
                    if call is None:
                        raise RuntimeError("trace_lineage_call_unavailable")
                    yield call
                if len(rows) < MESSAGE_CALL_LINEAGE_BATCH_SIZE:
                    break
                last_call_id = str(rows[-1][0])
                last_sequence = int(rows[-1][1])

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

    def surface_head_at_event_boundary(
        self,
        cursor: sqlite3.Cursor,
        *,
        segment_id: str,
        through_sequence: int,
    ) -> str | None:
        """Return the effective surface head at one recorded event boundary.

        Args:
            cursor: Active trace transaction cursor.
            segment_id: Segment whose effective surface is requested.
            through_sequence: Inclusive non-negative event-sequence boundary.

        Returns:
            The latest effective surface-node ID at the boundary, the segment's
            inherited surface head when no local event qualifies, or ``None``.

        Raises:
            ValueError: If ``segment_id`` is empty or ``through_sequence`` is
                not a non-negative integer.
        """

        _nonempty(segment_id, "segment_id")
        if type(through_sequence) is not int or through_sequence < 0:
            raise ValueError("through_sequence")
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

    def get_attached_owner_by_conversation(
        self,
        cursor: sqlite3.Cursor,
        conversation_id: str,
    ) -> TraceOwnerRecord | None:
        """Return the conversation's currently attached trace owner, if any.

        Args:
            cursor: Cursor for the caller-owned transaction.
            conversation_id: Conversation whose attached owner is requested.

        Returns:
            The attached owner record, or None when the conversation has none.
        """

        _nonempty(conversation_id, "conversation_id")
        row = cursor.execute(
            """SELECT owner_id, conversation_id, root_segment_id, attached,
                      detached_at FROM console_trace_owners
                 WHERE conversation_id = ? AND attached = 1""",
            (conversation_id,),
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

    def ensure_redaction_spans(
        self,
        cursor: sqlite3.Cursor,
        *,
        policy_id: str,
        semantic_revision_id: str | None,
        artifact_id: str | None,
        field_path: str,
        spans: Sequence[PIIRedactionSpan],
        outcome: Literal["applied", "omitted", "unavailable"] = "applied",
    ) -> tuple[TraceRedactionSpanRecord, ...]:
        """Persist or reuse immutable content-free masks for one frozen policy.

        Args:
            cursor: Caller-owned write transaction.
            policy_id: Frozen trace policy governing these masks.
            semantic_revision_id: Canonical revision source, or None for an artifact.
            artifact_id: Provider artifact source, or None for a revision.
            field_path: Stable structured path within the source value.
            spans: Content-free merged or unmerged PII codepoint spans.
            outcome: Persisted detector outcome for the supplied ranges.

        Returns:
            The existing or newly appended immutable rows in codepoint order.

        Raises:
            ValueError: If the source, policy, path, spans, or outcome is invalid.
            TraceIdentityConflict: If the source-policy-path already has
                different immutable masks.
        """

        if (semantic_revision_id is None) == (artifact_id is None):
            raise ValueError("redaction_source")
        if type(field_path) is not str or not field_path or len(field_path) > 512:
            raise ValueError("field_path")
        if outcome not in {"applied", "omitted", "unavailable"}:
            raise ValueError("outcome")
        merged = merge_pii_spans(spans)
        policy = self.get_policy(cursor, policy_id)
        if policy is None or not policy.pii_redaction_enabled:
            raise ValueError("pii_policy")
        source_kind: Literal["revision", "artifact"] = (
            "revision" if semantic_revision_id is not None else "artifact"
        )
        source_column = (
            "semantic_revision_id" if semantic_revision_id is not None else "artifact_id"
        )
        source_id = semantic_revision_id or artifact_id
        assert source_id is not None
        rows = cursor.execute(
            f"""SELECT span_id, policy_id, source_kind, semantic_revision_id,
                       artifact_id, field_path, start_codepoint, end_codepoint,
                       category, rule_id, detector_version, outcome
                  FROM console_trace_redaction_spans
                 WHERE policy_id = ? AND {source_column} = ? AND field_path = ?
                 ORDER BY start_codepoint, end_codepoint, span_id""",
            (policy_id, source_id, field_path),
        ).fetchall()
        existing = tuple(TraceRedactionSpanRecord(*row) for row in rows)
        expected = tuple(
            (
                span.start_codepoint,
                span.end_codepoint,
                span.category,
                span.rule_id,
                span.detector_version,
                outcome,
            )
            for span in merged
        )
        if existing:
            stored = tuple(
                (
                    row.start_codepoint,
                    row.end_codepoint,
                    row.category,
                    row.rule_id,
                    row.detector_version,
                    row.outcome,
                )
                for row in existing
            )
            if stored != expected:
                raise TraceIdentityConflict("redaction_spans")
            return existing
        if not merged:
            return ()
        self._claim_write_intent(cursor)
        for span in merged:
            cursor.execute(
                """INSERT INTO console_trace_redaction_spans(
                       span_id, policy_id, source_kind, semantic_revision_id,
                       artifact_id, field_path, start_codepoint, end_codepoint,
                       category, rule_id, detector_version, outcome)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    new_opaque_id(),
                    policy_id,
                    source_kind,
                    semantic_revision_id,
                    artifact_id,
                    field_path,
                    span.start_codepoint,
                    span.end_codepoint,
                    span.category,
                    span.rule_id,
                    span.detector_version,
                    outcome,
                ),
            )
        self._advance_graph_epoch(cursor)
        rows = cursor.execute(
            f"""SELECT span_id, policy_id, source_kind, semantic_revision_id,
                       artifact_id, field_path, start_codepoint, end_codepoint,
                       category, rule_id, detector_version, outcome
                  FROM console_trace_redaction_spans
                 WHERE policy_id = ? AND {source_column} = ? AND field_path = ?
                 ORDER BY start_codepoint, end_codepoint, span_id""",
            (policy_id, source_id, field_path),
        ).fetchall()
        return tuple(TraceRedactionSpanRecord(*row) for row in rows)

    def read_redaction_spans(
        self,
        cursor: sqlite3.Cursor,
        *,
        policy_id: str,
        semantic_revision_id: str | None,
        artifact_id: str | None,
        field_path: str,
    ) -> tuple[TraceRedactionSpanRecord, ...]:
        """Read immutable masks for exactly one source, policy, and field path."""

        if (semantic_revision_id is None) == (artifact_id is None):
            raise ValueError("redaction_source")
        source_column = (
            "semantic_revision_id" if semantic_revision_id is not None else "artifact_id"
        )
        source_id = semantic_revision_id or artifact_id
        rows = cursor.execute(
            f"""SELECT span_id, policy_id, source_kind, semantic_revision_id,
                       artifact_id, field_path, start_codepoint, end_codepoint,
                       category, rule_id, detector_version, outcome
                  FROM console_trace_redaction_spans
                 WHERE policy_id = ? AND {source_column} = ? AND field_path = ?
                 ORDER BY start_codepoint, end_codepoint, span_id""",
            (policy_id, source_id, field_path),
        ).fetchall()
        return tuple(TraceRedactionSpanRecord(*row) for row in rows)

    def read_source_redaction_spans(
        self,
        cursor: sqlite3.Cursor,
        *,
        policy_id: str,
        semantic_revision_id: str | None,
        artifact_id: str | None,
    ) -> tuple[TraceRedactionSpanRecord, ...]:
        """Read every bounded field mask for one source under one policy."""

        if (semantic_revision_id is None) == (artifact_id is None):
            raise ValueError("redaction_source")
        source_column = (
            "semantic_revision_id" if semantic_revision_id is not None else "artifact_id"
        )
        source_id = semantic_revision_id or artifact_id
        rows = cursor.execute(
            f"""SELECT span_id, policy_id, source_kind, semantic_revision_id,
                       artifact_id, field_path, start_codepoint, end_codepoint,
                       category, rule_id, detector_version, outcome
                  FROM console_trace_redaction_spans
                 WHERE policy_id = ? AND {source_column} = ?
                 ORDER BY field_path, start_codepoint, end_codepoint, span_id
                 LIMIT 10001""",
            (policy_id, source_id),
        ).fetchall()
        if len(rows) > 10_000:
            raise ValueError("redaction_span_limit")
        return tuple(TraceRedactionSpanRecord(*row) for row in rows)

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
        sanitized_bytes = _sanitize_trace_artifact_bytes(
            sanitized_bytes,
            media_type=media_type,
            normalization_version=normalization_version,
        )
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

        if type(sanitized_bytes) is not bytes:
            raise TypeError("sanitized_bytes")
        _nonempty(media_type, "media_type")
        _nonempty(normalization_version, "normalization_version")
        sanitized_bytes = _sanitize_trace_artifact_bytes(
            sanitized_bytes,
            media_type=media_type,
            normalization_version=normalization_version,
        )
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

    def read_lineage_surface_nodes(
        self,
        cursor: sqlite3.Cursor,
        *,
        segment_id: str,
        start_sequence: int,
        end_sequence: int,
    ) -> tuple[SurfaceNodeRecord, ...]:
        """Read physical nodes in one bounded fork-lineage range.

        Args:
            cursor: Caller-owned SQLite cursor.
            segment_id: Leaf segment whose immutable ancestry is counted.
            start_sequence: Inclusive range start.
            end_sequence: Inclusive range end.

        Returns:
            Lineage-visible physical nodes ordered by sequence.

        Raises:
            ValueError: If the segment or range is malformed.
        """

        _nonempty(segment_id, "segment_id")
        if (
            type(start_sequence) is not int
            or type(end_sequence) is not int
            or start_sequence < 0
            or end_sequence < start_sequence
        ):
            raise ValueError("sequence_range")
        nodes: list[SurfaceNodeRecord] = []
        for lineage_segment_id, through_sequence in self._surface_segment_lineage(
            cursor, segment_id
        ):
            upper_bound = (
                end_sequence
                if through_sequence is None
                else min(end_sequence, through_sequence)
            )
            if upper_bound < start_sequence:
                continue
            rows = cursor.execute(
                """SELECT node_id, segment_id, sequence, predecessor_node_id,
                          component_kind, reference_kind, semantic_revision_id,
                          artifact_id, omission_reason_code
                     FROM console_trace_surface_nodes
                    WHERE segment_id = ? AND sequence BETWEEN ? AND ?
                    ORDER BY sequence""",
                (lineage_segment_id, start_sequence, upper_bound),
            ).fetchall()
            nodes.extend(SurfaceNodeRecord(*row) for row in rows)
        return tuple(sorted(nodes, key=lambda node: node.sequence))

    def _surface_segment_lineage(
        self,
        cursor: sqlite3.Cursor,
        root_segment_id: str,
    ) -> list[tuple[str, int | None]]:
        """Return root-to-leaf segments bounded by inherited surface heads."""

        lineage: list[tuple[str, int | None]] = []
        segment = self.get_segment(cursor, root_segment_id)
        upper_bound: int | None = None
        while segment is not None:
            lineage.append((segment.segment_id, upper_bound))
            if segment.parent_segment_id is None:
                break
            inherited_head_id = segment.inherited_surface_head_id
            if inherited_head_id is None:
                inherited_bound = -1
            else:
                inherited_head = self.get_surface_node(cursor, inherited_head_id)
                if inherited_head is None:
                    raise RuntimeError("inherited_surface_head_unavailable")
                inherited_bound = inherited_head.sequence
            upper_bound = (
                inherited_bound
                if upper_bound is None
                else min(upper_bound, inherited_bound)
            )
            segment = self.get_segment(cursor, segment.parent_segment_id)
        lineage.reverse()
        return lineage

    def read_next_call_sequence(self, cursor: sqlite3.Cursor, run_id: str) -> int:
        """Return the first unused durable sequence for one provider-call chain.

        Args:
            cursor: Cursor for the caller-owned transaction.
            run_id: Durable provider-call chain identity.

        Returns:
            Zero for a new chain, otherwise one past its greatest sequence.

        Raises:
            RuntimeError: If a stored call sequence is invalid.
        """

        _nonempty(run_id, "run_id")
        row = cursor.execute(
            "SELECT MAX(call_sequence) FROM console_trace_calls WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        current = None if row is None else row[0]
        if current is None:
            return 0
        if type(current) is not int or current < 0:
            raise RuntimeError("call_sequence_unavailable")
        return current + 1

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
            _sanitize_trace_scalar(provider_name, "provider_name"),
            _sanitize_trace_scalar(model_name, "model_name"),
            _sanitize_trace_scalar(route_identity, "route_identity"),
            _sanitize_trace_scalar(endpoint_identity, "endpoint_identity"),
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

    def get_run_origin(
        self, cursor: sqlite3.Cursor, run_id: str
    ) -> TraceCallRecord | None:
        """Read a chain's unique first reservation, refusing ambiguous ownership.

        Args:
            cursor: Cursor for the caller-owned transaction.
            run_id: Opaque provider-call chain identity.

        Returns:
            The sole sequence-zero call, or None if missing or ambiguous.

        Raises:
            ValueError: If run_id is not a non-empty string.
        """
        _nonempty(run_id, "run_id")
        rows = cursor.execute(
            self._call_select() + " WHERE run_id = ? AND call_sequence = 0 LIMIT 2",
            (run_id,),
        ).fetchall()
        return self._call(rows[0]) if len(rows) == 1 else None

    def has_unresolved_call_for_turn(
        self,
        cursor: sqlite3.Cursor,
        *,
        owner_id: str,
        turn_id: str,
    ) -> bool:
        """Check one owner's saved turn without materializing its call history.

        Args:
            cursor: Cursor for the caller-owned transaction.
            owner_id: Exact attached trace owner.
            turn_id: Saved user-turn identity whose replay is being considered.

        Returns:
            Whether any call for this owner and turn remains unresolved.
        """
        # The existing owner-order index starts with (owner_id, turn_id).
        return cursor.execute(
            """SELECT 1 FROM console_trace_calls
                WHERE owner_id = ? AND turn_id = ? AND state IN (?, ?, ?, ?)
                LIMIT 1""",
            (
                owner_id,
                turn_id,
                TraceCallState.RESERVED.value,
                TraceCallState.DISPATCH_STARTED.value,
                TraceCallState.DISPATCH_UNKNOWN.value,
                TraceCallState.RESPONSE_STARTED.value,
            ),
        ).fetchone() is not None

    def get_latest_call_boundary(
        self, cursor: sqlite3.Cursor, segment_id: str
    ) -> TraceEventRecord | None:
        """Read the latest reserved call in segment event order.

        Args:
            cursor: Cursor for the caller-owned transaction.
            segment_id: Local trace segment, excluding inherited fork history.

        Returns:
            The latest call boundary, or None for a segment without one.
        """
        row = cursor.execute(
            self._event_select()
            + " WHERE segment_id = ? AND event_type = 'call_boundary'"
            + " ORDER BY sequence DESC LIMIT 1",
            (segment_id,),
        ).fetchone()
        return None if row is None else TraceEventRecord(*row)

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
                         outcome, usage_json, integrity_state, omission_reason_code,
                         reservation_provenance, import_reason_code
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
            reservation_provenance=TraceReservationProvenance(row[22]),
            import_reason_code=row[23],
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
