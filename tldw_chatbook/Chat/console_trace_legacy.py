"""Bounded normalization of legacy exchange blobs into snapshot surfaces."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import hashlib
import json
import sqlite3
from typing import Literal, cast

from tldw_chatbook.Chat.console_exchange_capture import (
    CaptureUnavailableError,
    CaptureDetail,
    ExchangeCapture,
    capture_from_storage,
    history_elision_marker,
    sanitize_capture_value_with_omission,
)
from tldw_chatbook.Chat.console_semantic_revision import (
    SemanticRevisionCoordinator,
    project_semantic_revision_provider_messages,
)
from tldw_chatbook.Chat.console_trace_models import (
    FrozenTracePolicy,
    SemanticRevisionRef,
    TraceCallState,
    TraceContentRef,
    TraceOmission,
)
from tldw_chatbook.Chat.console_trace_projection import NormalizedTraceCall
from tldw_chatbook.Chat.console_trace_redaction import CREDENTIAL_FILTER_VERSION
from tldw_chatbook.Chat.console_trace_repository import (
    ConsoleTraceRepository,
    HeaderComponentRef,
    SurfaceNodeRecord,
    TraceOwnerRecord,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


_LEGACY_POLICY_ID = "00000000-0000-4000-8000-000000000097"
_LEGACY_JSON_VERSION = "canonical-json-v1"
_LEGACY_MEDIA_TYPE = "application/json"
_LEGACY_NORMALIZATION_VERSION = 1
_CREDENTIAL_FILTER_OMISSION = "legacy_credential_filter_unavailable"
_HISTORY_OMISSION = "legacy_history_omitted"
_IMPORT_BOUNDARY = "legacy_import_boundary"
_SANITIZER_OMISSION = {"omitted": True}


@dataclass(frozen=True, slots=True)
class LegacyNormalizationResult:
    """Durable identity and verification result for one imported exchange."""

    call_id: str
    surface_head_id: str
    verification_status: Literal["verified", "unverified"]
    uncertainty_codes: tuple[str, ...]
    decoded_bytes: int


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _json_from_artifact(value: bytes) -> object:
    return json.loads(value.decode("utf-8"))


def _sanitize_legacy_message(value: object) -> tuple[object, bool]:
    """Sanitize one row, avoiding repeated scans of known-safe field names.

    Exact ordinary transcript rows have only ``role`` and ``content`` keys.
    Sanitizing their values as a sequence is equivalent to sanitizing the
    mapping, while avoiding the generic credential-key detector for the same
    two constant keys across every accumulated legacy snapshot.
    """

    if isinstance(value, Mapping) and set(value) == {"role", "content"}:
        sanitized, _redacted = sanitize_capture_value_with_omission(
            [value["role"], value["content"]]
        )
        if isinstance(sanitized, list) and len(sanitized) == 2:
            return {
                "role": sanitized[0],
                "content": sanitized[1],
            }, False
        return dict(_SANITIZER_OMISSION), True
    sanitized, _redacted = sanitize_capture_value_with_omission(value)
    return sanitized, sanitized == _SANITIZER_OMISSION


class LegacyTraceNormalizer:
    """Normalize and reconstruct legacy calls without inventing lineage."""

    def __init__(
        self,
        db: CharactersRAGDB,
        *,
        repository: ConsoleTraceRepository | None = None,
    ) -> None:
        self.db = db
        self.repository = repository or ConsoleTraceRepository()
        self.revisions = SemanticRevisionCoordinator(
            db,
            repository=self.repository,
        )
        self._batch_cursor_id: int | None = None
        self._batch_match_indexes: dict[str, dict[bytes, list[tuple[str, bytes]]]] = {}
        self._batch_revision_ids: dict[str, str] = {}
        self._batch_projected_revisions: dict[tuple[str, str], dict[str, object]] = {}

    def begin_batch(self, cursor: sqlite3.Cursor) -> None:
        """Enable one transaction-scoped ephemeral canonical-match index."""

        self.clear_ephemeral_matches()
        self._batch_cursor_id = id(cursor)

    def clear_ephemeral_matches(self) -> None:
        """Discard every content fingerprint and comparison value before commit."""

        self._batch_cursor_id = None
        self._batch_match_indexes.clear()
        self._batch_revision_ids.clear()
        self._batch_projected_revisions.clear()

    def normalize_exchange(
        self,
        cursor: sqlite3.Cursor,
        row: Mapping[str, object],
    ) -> LegacyNormalizationResult:
        """Normalize one caller-validated exchange inside its transaction."""

        exchange_id = row.get("id")
        message_id = row.get("message_id")
        if type(exchange_id) is not int or exchange_id < 0:
            raise ValueError("exchange_id")
        if type(message_id) is not str or not message_id:
            raise ValueError("message_id")
        decode_uncertainty: tuple[str, ...] = ()
        try:
            capture = self._decode_row(row)
            decoded_bytes = len(_canonical_bytes(self._capture_payload(capture)))
        except CaptureUnavailableError:
            capture = self._unavailable_capture(row)
            decoded_bytes = len(bytes(cast(object, row["capture_blob"])))
            decode_uncertainty = ("legacy_capture_unavailable",)
        abandoned = row.get("abandoned")
        if type(abandoned) is not bool:
            raise ValueError("abandoned")
        conversation_row = cursor.execute(
            "SELECT conversation_id FROM messages WHERE id = ? AND deleted = 0",
            (message_id,),
        ).fetchone()
        if conversation_row is None or type(conversation_row[0]) is not str:
            raise ValueError("legacy_message_owner_unavailable")
        conversation_id = str(conversation_row[0])

        existing = self.repository.get_call_by_idempotency_key(
            cursor,
            f"legacy-exchange-{exchange_id}",
        )
        if existing is not None:
            if existing.surface_node_id is None:
                raise ValueError("legacy_call_incomplete")
            return LegacyNormalizationResult(
                call_id=existing.call_id,
                surface_head_id=existing.surface_node_id,
                verification_status=(
                    "verified"
                    if existing.integrity_state == "complete"
                    else "unverified"
                ),
                uncertainty_codes=("legacy_chronology_unknown",),
                decoded_bytes=decoded_bytes,
            )

        capture, sanitizer_uncertainty = self._sanitize_capture(capture)
        owner = self._ensure_owner(cursor, conversation_id)
        policy = self.repository.ensure_policy(
            cursor,
            FrozenTracePolicy(
                policy_id=_LEGACY_POLICY_ID,
                credential_filter_version=CREDENTIAL_FILTER_VERSION,
                pii_redaction_enabled=False,
                pii_ruleset_revision_id=None,
            ),
        )

        request = dict(capture.request)
        raw_messages = request.pop("messages_payload", [])
        messages = raw_messages if isinstance(raw_messages, list) else []
        request_artifact = self._store_json(cursor, request)
        metadata_artifact = self._store_json(
            cursor,
            {
                "version": _LEGACY_NORMALIZATION_VERSION,
                "run_tag": capture.run_tag,
                "seq": capture.seq,
                "created_at": capture.created_at,
                "provider": capture.provider,
                "model": capture.model,
                "endpoint": capture.endpoint,
                "status": capture.status,
                "usage_json": capture.usage_json,
                "omitted_keys": list(capture.omitted_keys),
                "capture_detail": capture.capture_detail.value,
                "abandoned": abandoned,
                "messages_payload_present": "messages_payload" in capture.request,
            },
        )
        response_artifact = self._store_json(cursor, capture.response)
        header = self.repository.create_or_reuse_request_header(
            cursor,
            provider_name=capture.provider or "legacy_unknown",
            model_name=capture.model or "legacy_unknown",
            route_identity="legacy_snapshot",
            endpoint_identity=capture.endpoint or "legacy_unknown",
            generation_parameters={},
            adapter_defaults={},
            response_format={},
            reasoning_controls={},
            components=(
                HeaderComponentRef(
                    "legacy_snapshot_metadata", 0, metadata_artifact.content_id
                ),
                HeaderComponentRef(
                    "legacy_request_context", 0, request_artifact.content_id
                ),
            ),
        )

        predecessor = self._ensure_import_boundary(
            cursor, owner.root_segment_id
        ).node_id
        uncertainty = [
            "legacy_chronology_unknown",
            *decode_uncertainty,
            *sanitizer_uncertainty,
        ]
        normalized_messages: list[object] = []
        for legacy_row in messages:
            component_kind = "legacy_snapshot_message"
            if history_elision_marker([legacy_row]) is not None:
                reference: SemanticRevisionRef | TraceContentRef | TraceOmission = (
                    TraceOmission(component_kind, _HISTORY_OMISSION)
                )
                uncertainty.append("legacy_history_omitted")
                normalized_messages.append(
                    {"kind": "legacy_omission", "reason": _HISTORY_OMISSION}
                )
            else:
                revision_id = self._unique_revision_match(
                    cursor,
                    conversation_id=conversation_id,
                    legacy_row=legacy_row,
                )
                if revision_id is not None:
                    reference = SemanticRevisionRef(revision_id)
                else:
                    artifact = self._store_json(cursor, legacy_row)
                    reference = TraceContentRef(
                        artifact.content_id,
                        "legacy_message",
                    )
                    uncertainty.append("legacy_message_source_unknown")
                normalized_messages.append(legacy_row)
            predecessor = self._append_or_reuse_prefix_node(
                cursor,
                predecessor_node_id=predecessor,
                component_kind=component_kind,
                reference=reference,
            ).node_id

        predecessor_node = self.repository.get_surface_node(cursor, predecessor)
        assert predecessor_node is not None
        call_segment = self.repository.create_segment(
            cursor,
            parent_segment_id=predecessor_node.segment_id,
            inherited_through_sequence=predecessor_node.sequence,
            inherited_surface_head_id=predecessor_node.node_id,
        )
        terminal = self.repository.append_surface_node(
            cursor,
            segment_id=call_segment.segment_id,
            sequence=predecessor_node.sequence + 1,
            predecessor_node_id=predecessor,
            component_kind="legacy_snapshot",
            reference=TraceContentRef(
                metadata_artifact.content_id,
                "legacy_snapshot_metadata",
            ),
        )
        call = self.repository.reserve_call(
            cursor,
            owner_id=owner.owner_id,
            segment_id=call_segment.segment_id,
            turn_id=message_id,
            run_id=capture.run_tag,
            call_sequence=capture.seq,
            idempotency_key=f"legacy-exchange-{exchange_id}",
            policy_id=policy.policy_id,
        )
        self.repository.bind_call(
            cursor,
            call_id=call.call_id,
            surface_node_id=terminal.node_id,
            request_header_id=header.header_id,
            provider_name=capture.provider or "legacy_unknown",
            model_name=capture.model or "legacy_unknown",
            route_identity="legacy_snapshot",
        )
        self.repository.store_response_link(
            cursor,
            call_id=call.call_id,
            response=TraceContentRef(response_artifact.content_id, "legacy_response"),
        )
        self.repository.advance_call_state(
            cursor,
            call_id=call.call_id,
            target=TraceCallState.DISPATCH_STARTED,
            occurred_at=capture.created_at,
        )
        if abandoned:
            terminal_state = TraceCallState.ABANDONED
            self.repository.advance_call_state(
                cursor,
                call_id=call.call_id,
                target=terminal_state,
                occurred_at=capture.created_at,
                provider_operation_inactive=True,
                integrity_state="complete",
            )
        else:
            if capture.status != "error":
                self.repository.advance_call_state(
                    cursor,
                    call_id=call.call_id,
                    target=TraceCallState.RESPONSE_STARTED,
                    occurred_at=capture.created_at,
                )
            terminal_state = TraceCallState(capture.status)
            self.repository.advance_call_state(
                cursor,
                call_id=call.call_id,
                target=terminal_state,
                occurred_at=capture.created_at,
                usage=(
                    None
                    if capture.usage_json is None
                    else cast(dict[str, object], json.loads(capture.usage_json))
                ),
                integrity_state="complete",
            )

        if "messages_payload" in capture.request:
            capture = replace(
                capture,
                request={**request, "messages_payload": normalized_messages},
            )
        reconstructed = self._read_call(cursor, call.call_id, conversation_id)
        if reconstructed is None or reconstructed.capture != capture:
            raise ValueError("legacy_structural_equivalence_failed")
        return LegacyNormalizationResult(
            call_id=call.call_id,
            surface_head_id=terminal.node_id,
            verification_status="verified",
            uncertainty_codes=tuple(dict.fromkeys(uncertainty)),
            decoded_bytes=decoded_bytes,
        )

    def read_calls(self, message_id: str) -> tuple[NormalizedTraceCall, ...]:
        """Read verified legacy snapshots for one assistant message."""

        if type(message_id) is not str or not message_id:
            return ()
        with self.db.transaction() as cursor:
            owner = cursor.execute(
                "SELECT conversation_id FROM messages WHERE id = ?",
                (message_id,),
            ).fetchone()
            if owner is None or type(owner[0]) is not str:
                return ()
            rows = cursor.execute(
                """SELECT call.call_id
                     FROM console_trace_calls AS call
                     JOIN console_trace_request_headers AS header
                       ON header.header_id = call.request_header_id
                    WHERE call.turn_id = ?
                      AND header.route_identity = 'legacy_snapshot'
                    ORDER BY call.run_id, call.call_sequence, call.call_id""",
                (message_id,),
            ).fetchall()
            return tuple(
                call
                for (call_id,) in rows
                if (call := self._read_call(cursor, str(call_id), str(owner[0])))
                is not None
            )

    def _read_call(
        self,
        cursor: sqlite3.Cursor,
        call_id: str,
        conversation_id: str,
    ) -> NormalizedTraceCall | None:
        call = self.repository.get_call(cursor, call_id)
        if (
            call is None
            or call.surface_node_id is None
            or call.request_header_id is None
        ):
            return None
        header = self.repository.get_request_header(cursor, call.request_header_id)
        link = self.repository.get_response_link(cursor, call_id)
        if header is None or link is None or link.artifact_id is None:
            return None
        components = {
            item.component_kind: item.artifact_id for item in header.components
        }
        metadata_id = components.get("legacy_snapshot_metadata")
        request_id = components.get("legacy_request_context")
        if metadata_id is None or request_id is None:
            return None
        metadata_artifact = self.repository.get_artifact(cursor, metadata_id)
        request_artifact = self.repository.get_artifact(cursor, request_id)
        response_artifact = self.repository.get_artifact(cursor, link.artifact_id)
        if (
            metadata_artifact is None
            or request_artifact is None
            or response_artifact is None
        ):
            return None
        metadata = _json_from_artifact(metadata_artifact.sanitized_bytes)
        request_value = _json_from_artifact(request_artifact.sanitized_bytes)
        response_value = _json_from_artifact(response_artifact.sanitized_bytes)
        if (
            not isinstance(metadata, dict)
            or not isinstance(request_value, dict)
            or not isinstance(response_value, dict)
        ):
            return None
        uncertainty = ["legacy_chronology_unknown"]
        omitted_keys = tuple(str(item) for item in metadata.get("omitted_keys", []))
        if "legacy_capture_unavailable" in omitted_keys:
            uncertainty.append("legacy_capture_unavailable")
        if (
            request_value.get("legacy_omission") == _CREDENTIAL_FILTER_OMISSION
            or response_value.get("legacy_omission") == _CREDENTIAL_FILTER_OMISSION
        ):
            uncertainty.append("legacy_credential_filter_unavailable")
        messages: list[object] = []
        node = self.repository.get_surface_node(cursor, call.surface_node_id)
        if node is None or node.component_kind != "legacy_snapshot":
            return None
        rows = cursor.execute(
            """WITH RECURSIVE legacy_chain(
                   node_id, predecessor_node_id, component_kind, reference_kind,
                   semantic_revision_id, artifact_id, omission_reason_code, sequence
                 ) AS (
                   SELECT node_id, predecessor_node_id, component_kind, reference_kind,
                          semantic_revision_id, artifact_id, omission_reason_code, sequence
                     FROM console_trace_surface_nodes WHERE node_id = ?
                   UNION ALL
                   SELECT parent.node_id, parent.predecessor_node_id,
                          parent.component_kind, parent.reference_kind,
                          parent.semantic_revision_id, parent.artifact_id,
                          parent.omission_reason_code, parent.sequence
                     FROM console_trace_surface_nodes AS parent
                     JOIN legacy_chain AS child
                       ON parent.node_id = child.predecessor_node_id
                 )
                 SELECT node_id, predecessor_node_id, component_kind, reference_kind,
                        semantic_revision_id, artifact_id, omission_reason_code, sequence
                   FROM legacy_chain
                  WHERE component_kind = 'legacy_snapshot_message'
                  ORDER BY sequence""",
            (call.surface_node_id,),
        ).fetchall()
        revision_ids = tuple(
            str(row[4]) for row in rows if row[3] == "revision" and row[4] is not None
        )
        projected_revisions: dict[str, dict[str, object]] = {}
        unique_revision_ids = tuple(dict.fromkeys(revision_ids))
        if self._batch_cursor_id == id(cursor):
            missing_revision_ids = tuple(
                revision_id
                for revision_id in unique_revision_ids
                if (conversation_id, revision_id) not in self._batch_projected_revisions
            )
        else:
            missing_revision_ids = unique_revision_ids
        for offset in range(0, len(missing_revision_ids), 256):
            projected = project_semantic_revision_provider_messages(
                cursor,
                revision_ids=missing_revision_ids[offset : offset + 256],
                expected_conversation_id=conversation_id,
            )
            for revision_id, candidate in projected.items():
                value, failed = _sanitize_legacy_message(candidate)
                if failed or not isinstance(value, dict):
                    return None
                projected_revisions[revision_id] = value
                if self._batch_cursor_id == id(cursor):
                    self._batch_projected_revisions[(conversation_id, revision_id)] = (
                        value
                    )
        if self._batch_cursor_id == id(cursor):
            projected_revisions.update(
                {
                    revision_id: self._batch_projected_revisions[
                        (conversation_id, revision_id)
                    ]
                    for revision_id in unique_revision_ids
                }
            )
        for row in rows:
            reference_kind = row[3]
            semantic_revision_id = row[4]
            artifact_id = row[5]
            omission_reason_code = row[6]
            if reference_kind == "revision" and semantic_revision_id is not None:
                value = projected_revisions[str(semantic_revision_id)]
            elif reference_kind == "artifact" and artifact_id is not None:
                artifact = self.repository.get_artifact(cursor, str(artifact_id))
                if artifact is None:
                    return None
                value = _json_from_artifact(artifact.sanitized_bytes)
                uncertainty.append("legacy_message_source_unknown")
            elif reference_kind == "omission":
                value = {"kind": "legacy_omission", "reason": omission_reason_code}
                if omission_reason_code == _HISTORY_OMISSION:
                    uncertainty.append("legacy_history_omitted")
            else:
                return None
            messages.append(value)
        request = dict(request_value)
        if metadata.get("messages_payload_present"):
            request["messages_payload"] = messages
        detail = CaptureDetail(str(metadata["capture_detail"]))
        capture = ExchangeCapture(
            run_tag=str(metadata["run_tag"]),
            seq=int(metadata["seq"]),
            created_at=str(metadata["created_at"]),
            provider=str(metadata["provider"]),
            model=str(metadata["model"]),
            endpoint=(
                None if metadata.get("endpoint") is None else str(metadata["endpoint"])
            ),
            request=request,
            response=response_value,
            status=str(metadata["status"]),
            usage_json=(
                None
                if metadata.get("usage_json") is None
                else str(metadata["usage_json"])
            ),
            omitted_keys=omitted_keys,
            capture_detail=detail,
        )
        return NormalizedTraceCall(
            call_id=call.call_id,
            capture=capture,
            abandoned=bool(metadata.get("abandoned")),
            verification_status=(
                "verified" if call.integrity_state == "complete" else "unverified"
            ),
            provenance="legacy_snapshot",
            chronology="recorded_call_only",
            uncertainty_codes=tuple(dict.fromkeys(uncertainty)),
        )

    def _unique_revision_match(
        self,
        cursor: sqlite3.Cursor,
        *,
        conversation_id: str,
        legacy_row: object,
    ) -> str | None:
        if not isinstance(legacy_row, Mapping) or set(legacy_row) != {
            "role",
            "content",
        }:
            return None
        sanitized, unavailable = _sanitize_legacy_message(dict(legacy_row))
        if unavailable or not isinstance(sanitized, Mapping):
            return None
        encoded = _canonical_bytes(sanitized)
        digest = hashlib.sha256(encoded).digest()
        if self._batch_cursor_id == id(cursor):
            index = self._batch_match_indexes.get(conversation_id)
            if index is None:
                index = self._build_match_index(cursor, conversation_id)
                self._batch_match_indexes[conversation_id] = index
            candidates = [
                message_id
                for message_id, candidate_bytes in index.get(digest, [])
                if candidate_bytes == encoded
            ]
        else:
            candidates = []
            for message_id, sender, role, content in cursor.execute(
                """SELECT id, sender, role, content FROM messages
                    WHERE conversation_id = ? AND deleted = 0
                    ORDER BY timestamp, id""",
                (conversation_id,),
            ):
                candidate, failed = _sanitize_legacy_message(
                    {"role": role or sender, "content": content}
                )
                if not failed and candidate == sanitized:
                    candidates.append(str(message_id))
        if len(candidates) != 1:
            return None
        cached_revision_id = self._batch_revision_ids.get(candidates[0])
        if self._batch_cursor_id == id(cursor) and cached_revision_id is not None:
            return cached_revision_id
        revision_id = self.revisions.ensure_current_revision(
            cursor,
            message_id=candidates[0],
            creation_reason="legacy_reference",
        ).revision_id
        if self._batch_cursor_id == id(cursor):
            self._batch_revision_ids[candidates[0]] = revision_id
        return revision_id

    @staticmethod
    def _build_match_index(
        cursor: sqlite3.Cursor,
        conversation_id: str,
    ) -> dict[bytes, list[tuple[str, bytes]]]:
        index: dict[bytes, list[tuple[str, bytes]]] = {}
        for message_id, sender, role, content in cursor.execute(
            """SELECT id, sender, role, content FROM messages
                WHERE conversation_id = ? AND deleted = 0
                ORDER BY timestamp, id""",
            (conversation_id,),
        ):
            candidate, failed = _sanitize_legacy_message(
                {"role": role or sender, "content": content}
            )
            if failed:
                continue
            encoded = _canonical_bytes(candidate)
            digest = hashlib.sha256(encoded).digest()
            index.setdefault(digest, []).append((str(message_id), encoded))
        return index

    def _ensure_owner(
        self,
        cursor: sqlite3.Cursor,
        conversation_id: str,
    ) -> TraceOwnerRecord:
        row = cursor.execute(
            """SELECT owner_id FROM console_trace_owners
                WHERE conversation_id = ? AND attached = 1""",
            (conversation_id,),
        ).fetchone()
        if row is not None:
            owner = self.repository.get_owner(cursor, str(row[0]))
            assert owner is not None
            return owner
        segment = self.repository.create_segment(cursor)
        return self.repository.attach_owner(
            cursor,
            conversation_id=conversation_id,
            root_segment_id=segment.segment_id,
        )

    def _store_json(self, cursor: sqlite3.Cursor, value: object) -> TraceContentRef:
        artifact = self.repository.store_sanitized_artifact(
            cursor,
            sanitized_bytes=_canonical_bytes(value),
            media_type=_LEGACY_MEDIA_TYPE,
            normalization_version=_LEGACY_JSON_VERSION,
        )
        return TraceContentRef(artifact.artifact_id, "legacy_json")

    def _append_or_reuse_prefix_node(
        self,
        cursor: sqlite3.Cursor,
        *,
        predecessor_node_id: str | None,
        component_kind: str,
        reference: SemanticRevisionRef | TraceContentRef | TraceOmission,
    ) -> SurfaceNodeRecord:
        reference_kind, reference_id, omission = self._reference_columns(reference)
        row = cursor.execute(
            """SELECT node_id FROM console_trace_surface_nodes
                WHERE predecessor_node_id IS ? AND component_kind = ?
                  AND reference_kind = ?
                  AND semantic_revision_id IS ? AND artifact_id IS ?
                  AND omission_reason_code IS ?
                ORDER BY node_id LIMIT 1""",
            (
                predecessor_node_id,
                component_kind,
                reference_kind,
                reference_id if reference_kind == "revision" else None,
                reference_id if reference_kind == "artifact" else None,
                omission,
            ),
        ).fetchone()
        if row is not None:
            existing = self.repository.get_surface_node(cursor, str(row[0]))
            assert existing is not None
            return existing
        if predecessor_node_id is None:
            raise ValueError("legacy_prefix_boundary_missing")
        predecessor = self.repository.get_surface_node(cursor, predecessor_node_id)
        if predecessor is None:
            raise ValueError("legacy_prefix_predecessor_missing")
        tail = self.repository.get_surface_tail(cursor, predecessor.segment_id)
        if tail is not None and tail.node_id == predecessor.node_id:
            target_segment_id = predecessor.segment_id
        else:
            target_segment_id = self.repository.create_segment(
                cursor,
                parent_segment_id=predecessor.segment_id,
                inherited_through_sequence=predecessor.sequence,
                inherited_surface_head_id=predecessor.node_id,
            ).segment_id
        node = self.repository.append_surface_node(
            cursor,
            segment_id=target_segment_id,
            sequence=predecessor.sequence + 1,
            predecessor_node_id=predecessor_node_id,
            component_kind=component_kind,
            reference=reference,
        )
        self.repository.append_event(
            cursor,
            segment_id=target_segment_id,
            sequence=node.sequence,
            event_type="surface_append",
            surface_node_id=node.node_id,
        )
        return node

    def _ensure_import_boundary(
        self,
        cursor: sqlite3.Cursor,
        segment_id: str,
    ) -> SurfaceNodeRecord:
        row = cursor.execute(
            """WITH RECURSIVE descendants(segment_id) AS (
                   SELECT ?
                   UNION ALL
                   SELECT child.segment_id
                     FROM console_trace_segments AS child
                     JOIN descendants AS parent
                       ON child.parent_segment_id = parent.segment_id
                    WHERE NOT EXISTS (
                      SELECT 1 FROM console_trace_owners AS nested_owner
                       WHERE nested_owner.root_segment_id = child.segment_id
                         AND nested_owner.attached = 1
                    )
                 )
                 SELECT node.node_id
                   FROM console_trace_surface_nodes AS node
                   JOIN descendants
                     ON descendants.segment_id = node.segment_id
                  WHERE node.component_kind = 'legacy_snapshot_root'
                  ORDER BY node.node_id LIMIT 1""",
            (segment_id,),
        ).fetchone()
        if row is not None:
            boundary = self.repository.get_surface_node(cursor, str(row[0]))
            assert boundary is not None
            return boundary
        tail = self.repository.get_surface_tail(cursor, segment_id)
        boundary_predecessor_id: str | None = None
        boundary_sequence = 0
        if tail is not None:
            event_tail = self.repository.get_event_tail(cursor, segment_id)
            if event_tail is None:
                raise ValueError("legacy_import_boundary_unavailable")
            inherited_head = self.repository.surface_head_at_event_boundary(
                cursor,
                segment_id=segment_id,
                through_sequence=event_tail.sequence,
            )
            if inherited_head is None:
                raise ValueError("legacy_import_boundary_unavailable")
            segment_id = self.repository.create_segment(
                cursor,
                parent_segment_id=segment_id,
                inherited_through_sequence=event_tail.sequence,
                inherited_surface_head_id=inherited_head,
            ).segment_id
            inherited_node = self.repository.get_surface_node(cursor, inherited_head)
            if inherited_node is None:
                raise ValueError("legacy_import_boundary_unavailable")
            boundary_predecessor_id = inherited_node.node_id
            boundary_sequence = inherited_node.sequence + 1
        boundary = self.repository.append_surface_node(
            cursor,
            segment_id=segment_id,
            sequence=boundary_sequence,
            predecessor_node_id=boundary_predecessor_id,
            component_kind="legacy_snapshot_root",
            reference=TraceOmission("legacy_snapshot_root", _IMPORT_BOUNDARY),
        )
        self.repository.append_event(
            cursor,
            segment_id=segment_id,
            sequence=boundary_sequence,
            event_type="surface_append",
            surface_node_id=boundary.node_id,
        )
        return boundary

    @staticmethod
    def _reference_columns(
        reference: SemanticRevisionRef | TraceContentRef | TraceOmission,
    ) -> tuple[str, str | None, str | None]:
        if isinstance(reference, SemanticRevisionRef):
            return "revision", reference.revision_id, None
        if isinstance(reference, TraceContentRef):
            return "artifact", reference.content_id, None
        return "omission", None, reference.reason_code

    @staticmethod
    def _decode_row(row: Mapping[str, object]) -> ExchangeCapture:
        blob = row.get("capture_blob")
        detail = row.get("capture_detail")
        if not isinstance(blob, (bytes, bytearray, memoryview)):
            raise TypeError("capture_blob")
        capture = capture_from_storage(bytes(blob), detail)
        if (
            capture.run_tag != row.get("run_tag")
            or capture.seq != row.get("seq")
            or capture.status != row.get("status")
            or capture.created_at != row.get("created_at")
        ):
            raise ValueError("capture_authority_mismatch")
        return capture

    @staticmethod
    def _unavailable_capture(row: Mapping[str, object]) -> ExchangeCapture:
        detail = row.get("capture_detail")
        if type(detail) is not str or detail not in {"safe", "full"}:
            detail = "safe"
        return ExchangeCapture(
            run_tag=str(row.get("run_tag") or "legacy-unavailable"),
            seq=(int(row["seq"]) if type(row.get("seq")) is int else 0),
            created_at=str(row.get("created_at") or "legacy-unavailable"),
            provider="",
            model="",
            endpoint=None,
            request={"legacy_omission": "legacy_capture_unavailable"},
            response={"legacy_omission": "legacy_capture_unavailable"},
            status=(
                str(row["status"])
                if row.get("status") in {"complete", "stopped", "error"}
                else "error"
            ),
            usage_json=None,
            omitted_keys=("legacy_capture_unavailable",),
            capture_detail=CaptureDetail(detail),
        )

    @staticmethod
    def _capture_payload(capture: ExchangeCapture) -> dict[str, object]:
        return {
            "run_tag": capture.run_tag,
            "seq": capture.seq,
            "created_at": capture.created_at,
            "provider": capture.provider,
            "model": capture.model,
            "endpoint": capture.endpoint,
            "request": capture.request,
            "response": capture.response,
            "status": capture.status,
            "usage_json": capture.usage_json,
            "omitted_keys": list(capture.omitted_keys),
            "capture_detail": capture.capture_detail.value,
        }

    @staticmethod
    def _sanitize_capture(
        capture: ExchangeCapture,
    ) -> tuple[ExchangeCapture, tuple[str, ...]]:
        request_source = dict(capture.request)
        messages_present = "messages_payload" in request_source
        raw_messages = request_source.pop("messages_payload", None)
        request, _request_redacted = sanitize_capture_value_with_omission(
            request_source
        )
        request_failed = not isinstance(request, dict) or request == _SANITIZER_OMISSION
        messages_failed = False
        if messages_present and isinstance(raw_messages, list):
            sanitized_messages: list[object] = []
            for message in raw_messages:
                sanitized_message, failed = _sanitize_legacy_message(message)
                messages_failed = messages_failed or failed
                sanitized_messages.append(sanitized_message)
            if isinstance(request, dict):
                request["messages_payload"] = sanitized_messages
        elif messages_present:
            sanitized_messages, _redacted = sanitize_capture_value_with_omission(
                raw_messages
            )
            messages_failed = sanitized_messages == _SANITIZER_OMISSION
            if isinstance(request, dict):
                request["messages_payload"] = sanitized_messages

        response, _response_redacted = sanitize_capture_value_with_omission(
            capture.response
        )
        response_failed = (
            not isinstance(response, dict) or response == _SANITIZER_OMISSION
        )
        request_failed = request_failed or messages_failed
        if request_failed or not isinstance(request, dict):
            request = {"legacy_omission": _CREDENTIAL_FILTER_OMISSION}
        if response_failed or not isinstance(response, dict):
            response = {"legacy_omission": _CREDENTIAL_FILTER_OMISSION}
        uncertainty = (
            ("legacy_credential_filter_unavailable",)
            if request_failed or response_failed
            else ()
        )
        return replace(capture, request=request, response=response), uncertainty


__all__ = ["LegacyNormalizationResult", "LegacyTraceNormalizer"]
