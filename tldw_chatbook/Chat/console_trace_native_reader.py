"""Reconstruct viewer captures from the native Console semantic-trace ledger."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from typing import Any

from tldw_chatbook.Chat.console_exchange_capture import CaptureDetail, ExchangeCapture
from tldw_chatbook.Chat.console_semantic_revision import (
    project_semantic_revision_provider_continuations,
    project_semantic_revision_trace_message,
)
from tldw_chatbook.Chat.console_trace_continuation import (
    CONTINUATION_DELTA_COMPONENT,
    resolve_continuation_node,
)
from tldw_chatbook.Chat.console_trace_models import TraceCallState
from tldw_chatbook.Chat.console_trace_projection import NormalizedTraceCall
from tldw_chatbook.Chat.console_trace_provenance import TraceTransformKind
from tldw_chatbook.Chat.console_trace_redaction import (
    CREDENTIAL_SANITIZER_UNAVAILABLE,
    PII_DETECTOR_UNAVAILABLE,
    CredentialSanitizer,
)
from tldw_chatbook.Chat.console_trace_repository import (
    ConsoleTraceRepository,
    TraceArtifactRecord,
    TraceCallRecord,
)
from tldw_chatbook.Chat.console_trace_service import (
    ConsoleTraceService,
    ReconstructedRequestHeader,
)
from tldw_chatbook.Chat.provider_usage import ProviderUsage

_PROVIDER_USAGE_FIELDS = frozenset(
    field.name for field in dataclasses.fields(ProviderUsage)
)
_CONTINUATION_COMPONENTS = frozenset(
    {
        "continuation",
        CONTINUATION_DELTA_COMPONENT,
        TraceTransformKind.CONTINUATION_ATTACHMENT.value,
    }
)
_STATUS = {
    TraceCallState.COMPLETE: "complete",
    TraceCallState.STOPPED: "stopped",
    TraceCallState.INTERRUPTED: "stopped",
    TraceCallState.ERROR: "error",
    TraceCallState.ABANDONED: "error",
    TraceCallState.NOT_DISPATCHED: "error",
    TraceCallState.DISPATCH_UNKNOWN: "error",
}


class ConsoleTraceNativeReader:
    """Read message-associated calls from the normalized production ledger.

    Args:
        database: Transaction-owning Chat database.
        repository: Optional shared normalized trace repository.
        service: Optional shared trace reconstruction service.
    """

    def __init__(
        self,
        database: object,
        *,
        repository: ConsoleTraceRepository | None = None,
        service: ConsoleTraceService | None = None,
    ) -> None:
        self.database = database
        self.repository = repository or ConsoleTraceRepository()
        self.service = service or ConsoleTraceService(self.repository)

    def read_report_tool_fields(
        self, conversation_id: str, events: tuple, *, max_source_bytes: int = 65_536,
        max_total_source_bytes: int = 262_144, max_fields: int = 500,
    ) -> tuple[Mapping[str, object], ...]:
        """Read selected immutable tool components; never reconstruct exchanges."""
        from tldw_chatbook.Chat.console_semantic_revision import project_semantic_revision_trace_tool_fields
        from tldw_chatbook.Utils.log_sanitizer import sanitize_string

        if not 1 <= max_fields <= 500 or not 1 <= max_source_bytes <= 65_536 or not 1 <= max_total_source_bytes <= 262_144:
            raise ValueError("report tool read budget")
        result = []
        total = 0
        output_size = 0
        with self.database.transaction() as cursor:
            owner = self.repository.get_attached_owner_by_conversation(cursor, conversation_id)
            if owner is None:
                return ()
            fences: dict[str, int | None] = {}
            segment_id = owner.root_segment_id
            upper = None
            while segment_id and len(fences) < 64:
                if segment_id in fences:
                    raise ValueError("report lineage cycle")
                fences[segment_id] = upper
                segment = self.repository.get_segment(cursor, segment_id)
                if segment is None:
                    break
                segment_id, upper = segment.parent_segment_id, segment.inherited_through_sequence
            allowed_conversations = {conversation_id}
            for owned_segment in fences:
                rows = cursor.execute("SELECT conversation_id FROM console_trace_owners WHERE root_segment_id = ?", (owned_segment,)).fetchall()
                allowed_conversations.update(str(row[0]) for row in rows)
            for event in reversed(events[:max_fields]):
                if event.event_type not in {"tool_call", "tool_result"}:
                    continue
                unavailable = {"event_key": event.event_id, "unavailable": "governed_tool_field_unavailable"}
                try:
                    if event.segment_id not in fences or (
                        fences[event.segment_id] is not None and event.sequence > fences[event.segment_id]
                    ):
                        raise ValueError("report tool owner mismatch")
                    current = self.repository._get_event(cursor, event.event_id)
                    if current != event:
                        raise ValueError("report tool reference changed")
                    call = self.repository.get_call(cursor, event.call_id or "")
                    if call is None or call.segment_id != event.segment_id:
                        raise ValueError("report tool policy unavailable")
                    policy = self.repository.get_policy(cursor, call.policy_id)
                    if policy is None:
                        raise ValueError("report tool policy unavailable")
                    if event.semantic_revision_id is not None:
                        revision = self.repository.get_semantic_revision(cursor, event.semantic_revision_id)
                        if revision is None or revision.source_conversation_id not in allowed_conversations or event.event_type != "tool_result":
                            raise ValueError("report tool revision unavailable")
                        value = project_semantic_revision_trace_tool_fields(
                            cursor, revision_id=revision.revision_id,
                            expected_conversation_id=revision.source_conversation_id,
                            policy_id=call.policy_id, max_source_bytes=min(max_source_bytes, max_total_source_bytes - total),
                        )
                        total += max_source_bytes  # conservative bound for a revision read
                    elif event.artifact_id is not None:
                        row = cursor.execute(
                            """SELECT length(sanitized_bytes),
                                      CASE WHEN length(sanitized_bytes) <= ? THEN sanitized_bytes ELSE NULL END
                                 FROM console_trace_artifacts WHERE artifact_id = ?""",
                            (min(max_source_bytes, max_total_source_bytes - total), event.artifact_id),
                        ).fetchone()
                        if row is None or row[1] is None:
                            raise ValueError("report tool artifact budget")
                        total += int(row[0])
                        # These immutable provider-only artifacts have already
                        # crossed the capture-time credential/PII boundary.
                        value = json.loads(row[1])
                        checked = CredentialSanitizer().sanitize(value)
                        if not checked.available or not isinstance(checked.value, Mapping):
                            raise ValueError("report tool sanitizer unavailable")
                        governed = checked.value
                        if event.event_type == "tool_result":
                            value = {key: governed[key] for key in ("name", "tool_name", "result", "output", "content") if key in governed}
                        else:
                            value = {key: governed[key] for key in ("name", "tool_name", "args", "arguments", "tool_calls") if key in governed}
                        if not value:
                            raise ValueError("report tool shape unavailable")
                    else:
                        raise ValueError("report tool source omitted")
                    if total > max_total_source_bytes:
                        raise ValueError("report aggregate tool budget")
                    text = sanitize_string(json.dumps(value, ensure_ascii=False, sort_keys=True))
                    if output_size + len(text.encode("utf-8")) > 8_000:
                        result.append({**unavailable, "unavailable": "tool_output_budget"})
                        break
                    output_size += len(text.encode("utf-8"))
                    result.append({"event_key": event.event_id, "text": text})
                except (KeyError, TypeError, ValueError, UnicodeError):
                    result.append(unavailable)
                if total >= max_total_source_bytes:
                    break
        return tuple(reversed(result))

    def read_calls(self, message_id: str) -> tuple[NormalizedTraceCall, ...]:
        """Reconstruct native calls attached to one conversation message.

        Args:
            message_id: Durable message identity selected in the viewer.

        Returns:
            Native calls in durable conversation-lineage order. Legacy snapshot
            routes are excluded because the legacy normalizer owns them.
        """

        if type(message_id) is not str or not message_id:
            return ()
        with self.database.transaction() as cursor:
            row = cursor.execute(
                """SELECT conversation_id, parent_message_id,
                          COALESCE(role, sender)
                     FROM messages WHERE id = ?""",
                (message_id,),
            ).fetchone()
            if row is None or not row[0]:
                return ()
            conversation_id = str(row[0])
            turn_id = (
                str(row[1])
                if row[1] and str(row[2] or "").casefold() == "assistant"
                else message_id
            )
            result: list[NormalizedTraceCall] = []
            for call in self.repository.iter_message_call_lineage(
                cursor,
                conversation_id,
                message_id,
                turn_id=turn_id,
            ):
                if (
                    call.route_identity == "legacy_snapshot"
                    or call.state not in _STATUS
                ):
                    continue
                result.append(self._reconstruct_call(cursor, call))
            return tuple(result)

    def _reconstruct_call(
        self,
        cursor: Any,
        call: TraceCallRecord,
    ) -> NormalizedTraceCall:
        uncertainty: list[str] = []
        omitted: list[str] = []
        request: dict[str, object]
        try:
            request, request_omissions = self._reconstruct_request(cursor, call)
            omitted.extend(request_omissions)
        except (KeyError, TypeError, ValueError, UnicodeDecodeError):
            request = {"omitted": "native_request_reconstruction_unavailable"}
            omitted.append("request")
            uncertainty.append("native_request_reconstruction_unavailable")
        try:
            response = self._reconstruct_response(cursor, call)
        except (KeyError, TypeError, ValueError, UnicodeDecodeError):
            response = {"omitted": "native_response_reconstruction_unavailable"}
            omitted.append("response")
            uncertainty.append("native_response_reconstruction_unavailable")

        created_at = (
            call.dispatch_started_at
            or call.response_started_at
            or call.settled_at
            or "recorded-before-dispatch"
        )
        # TASK-33621.9: settled usage is the provider's raw nested payload,
        # frozen all the way down. ``json.dumps(dict(usage))`` raised on its
        # mappingproxy children, and the raw keys would still fail the
        # capture contract, which carries normalized ProviderUsage (as the
        # gateway's own capture does). Usage is supplementary: one that
        # cannot be normalized is omitted (and listed as such), never the
        # whole call; an empty payload reported nothing, so it stays None
        # rather than becoming an all-zero "reported" record.
        try:
            usage_json = (
                _normalized_usage_json(
                    call.usage,
                    provider=call.provider_name or "",
                    model=call.model_name or "",
                )
                if call.usage
                else None
            )
        except (TypeError, ValueError, RecursionError):
            usage_json = None
        if call.usage and usage_json is None:
            omitted.append("usage")
        capture = ExchangeCapture(
            run_tag=call.run_id,
            seq=call.call_sequence,
            created_at=created_at,
            provider=call.provider_name or "unavailable",
            model=call.model_name or "unavailable",
            endpoint=self._endpoint(cursor, call),
            request=request,
            response=response,
            status=_STATUS[call.state],
            usage_json=usage_json,
            omitted_keys=tuple(sorted(set(omitted))),
            capture_detail=CaptureDetail.FULL,
            trace_provenance="native",
            trace_chronology="known",
            trace_uncertainty=tuple(uncertainty),
        )
        verified = call.integrity_state == "complete" and not uncertainty
        return NormalizedTraceCall(
            call_id=call.call_id,
            capture=capture,
            abandoned=call.state is TraceCallState.ABANDONED,
            verification_status="verified" if verified else "unverified",
            uncertainty_codes=tuple(uncertainty),
        )

    def _endpoint(self, cursor: Any, call: TraceCallRecord) -> str | None:
        if call.request_header_id is None:
            return None
        header = self.repository.get_request_header(cursor, call.request_header_id)
        return None if header is None else header.endpoint_identity

    def _reconstruct_request(
        self,
        cursor: Any,
        call: TraceCallRecord,
    ) -> tuple[dict[str, object], tuple[str, ...]]:
        if call.surface_node_id is None or call.request_header_id is None:
            raise ValueError("native_request_boundary_unavailable")
        tail = self.repository.get_surface_node(cursor, call.surface_node_id)
        if tail is None:
            raise ValueError("native_surface_unavailable")
        projection = self.service._surface_projection(cursor, call.segment_id, tail)
        messages: list[object] = []
        continuations: list[object] = []
        omissions: list[str] = []
        for _, key in projection.entries:
            component_kind, reference_kind, identity = key
            destination = (
                continuations
                if component_kind in _CONTINUATION_COMPONENTS
                else messages
            )
            if reference_kind == "revision":
                revision = self.repository.get_semantic_revision(cursor, identity)
                if revision is None:
                    raise ValueError("semantic_revision_unavailable")
                if destination is continuations:
                    value = self._project_continuation(
                        cursor,
                        call=call,
                        revision_id=identity,
                        expected_conversation_id=revision.source_conversation_id,
                    )
                    if value == {"omitted": CREDENTIAL_SANITIZER_UNAVAILABLE}:
                        omissions.append("provider_continuations")
                else:
                    value = project_semantic_revision_trace_message(
                        cursor,
                        revision_id=identity,
                        expected_conversation_id=revision.source_conversation_id,
                        policy_id=call.policy_id,
                    )
                destination.append(value)
            elif component_kind == CONTINUATION_DELTA_COMPONENT:
                destination.append(
                    resolve_continuation_node(
                        self.repository,
                        cursor,
                        self.repository.get_surface_node(cursor, identity),
                        policy_id=call.policy_id,
                    )
                )
            elif reference_kind == "artifact":
                destination.append(
                    self._decode_artifact(
                        self.repository.get_artifact(cursor, identity)
                    )
                )
            elif reference_kind == "omission":
                destination.append({"omitted": identity})
                omissions.append(
                    "provider_continuations" if destination is continuations else "messages_payload"
                )
            else:
                raise ValueError("native_surface_reference_invalid")

        header = self.service.reconstruct_header(cursor, call.request_header_id)
        system_rows = [
            item
            for item in header.components
            if item.component_kind == "rendered_system_row"
        ]
        system_slot = header.adapter_defaults.get("rendered_system_slot")
        if system_rows or system_slot is not None:
            if (
                type(system_slot) is not int
                or system_slot != 0
                or len(system_rows) != 1
                or system_rows[0].ordinal != 0
                or not projection.entries
                or projection.entries[0][1][:2] != ("rendered_system", "artifact")
                or not messages
                or not isinstance(messages[0], Mapping)
                or messages[0].get("role") != "system"
            ):
                raise ValueError("rendered_system_slot_mismatch")
            row = system_rows[0].value
            if isinstance(row, Mapping) and row.get("role") == "system":
                messages[0] = row
            elif isinstance(row, Mapping) and set(row) == {"omitted"}:
                messages[0] = row
                omissions.append("rendered_system_row")
            else:
                raise ValueError("rendered_system_slot_unavailable")
        request = self._header_request(cursor, call, header)
        request["messages_payload"] = messages
        if continuations:
            request["provider_continuations"] = continuations
        raw_omissions = header.adapter_defaults.get("header_omissions", {})
        if isinstance(raw_omissions, Mapping):
            omissions.extend(str(key) for key in raw_omissions)
        return request, tuple(omissions)

    def _project_continuation(
        self,
        cursor: Any,
        *,
        call: TraceCallRecord,
        revision_id: str,
        expected_conversation_id: str,
    ) -> object:
        revision = self.repository.get_semantic_revision(cursor, revision_id)
        if (
            revision is None
            or revision.source_conversation_id != expected_conversation_id
        ):
            raise ValueError("revision_owner_mismatch")
        policy = self.repository.get_policy(cursor, call.policy_id)
        if policy is None:
            raise ValueError("trace_policy_unavailable")
        if policy.pii_redaction_enabled:
            binding = self.repository.get_revision_policy_binding(
                cursor,
                revision_id=revision_id,
                policy_id=policy.policy_id,
            )
            if binding is None:
                return {"omitted": PII_DETECTOR_UNAVAILABLE}
            if binding.omission_reason_code is not None:
                return {"omitted": binding.omission_reason_code}
            materialized = self._decode_artifact(
                self.repository.get_artifact(cursor, binding.artifact_id or "")
            )
            if not isinstance(materialized, Mapping):
                raise ValueError("semantic_continuation_unavailable")
            continuation = materialized.get("provider_continuation")
            if continuation is None:
                raise ValueError("semantic_continuation_unavailable")
            return continuation
        value = project_semantic_revision_provider_continuations(
            cursor,
            revision_ids=(revision_id,),
            expected_conversation_id=expected_conversation_id,
        )[revision_id]
        sanitized = CredentialSanitizer().sanitize(value)
        if not sanitized.available:
            return {"omitted": CREDENTIAL_SANITIZER_UNAVAILABLE}
        return sanitized.value

    def _header_request(
        self,
        cursor: Any,
        call: TraceCallRecord,
        header: ReconstructedRequestHeader,
    ) -> dict[str, object]:
        request: dict[str, object] = {}
        literal_components = [
            item.value
            for item in header.components
            if item.component_kind == "provider_literal_envelope"
        ]
        for value in literal_components:
            if isinstance(value, Mapping):
                request.update({str(key): item for key, item in value.items()})
        request.update(header.generation_parameters)
        request.update(
            {
                "api_endpoint": header.provider_name,
                "api_base_url": header.endpoint_identity,
                "model": header.model_name,
            }
        )
        if header.response_format:
            request["response_format"] = dict(header.response_format)
        if header.reasoning_controls:
            request.update(header.reasoning_controls)
        tools = [
            item.value
            for item in sorted(header.components, key=lambda item: item.ordinal)
            if item.component_kind == "tool_schema"
        ]
        if tools:
            request["tools"] = tools
        system = self._system_message(cursor, call, header)
        if system is not None:
            request["system_message"] = system
        return request

    def _system_message(
        self,
        cursor: Any,
        call: TraceCallRecord,
        header: ReconstructedRequestHeader,
    ) -> object | None:
        rendered = {
            item.ordinal: item.value
            for item in header.components
            if item.component_kind == "rendered_system_part"
        }
        parts: list[object] = []
        for token in header.system_composition:
            kind = token.get("kind")
            if kind == "revision":
                revision_id = token.get("revision_id")
                if type(revision_id) is not str:
                    raise ValueError("system_revision_unavailable")
                revision = self.repository.get_semantic_revision(cursor, revision_id)
                if revision is None:
                    raise ValueError("system_revision_unavailable")
                projected = project_semantic_revision_trace_message(
                    cursor,
                    revision_id=revision_id,
                    expected_conversation_id=revision.source_conversation_id,
                    policy_id=call.policy_id,
                )
                parts.append(projected.get("content", projected))
            elif kind == "artifact":
                ordinal = token.get("component_ordinal")
                if type(ordinal) is not int or ordinal not in rendered:
                    raise ValueError("system_artifact_unavailable")
                parts.append(rendered[ordinal])
            elif kind == "omission":
                parts.append(f"[{token.get('reason', 'omitted')}]")
        if not parts:
            parts = [rendered[key] for key in sorted(rendered)]
        if not parts:
            return None
        if all(isinstance(part, str) for part in parts):
            return "\n".join(str(part) for part in parts)
        return parts[0] if len(parts) == 1 else parts

    def _reconstruct_response(
        self,
        cursor: Any,
        call: TraceCallRecord,
    ) -> dict[str, object]:
        response = self.repository.get_response_link(cursor, call.call_id)
        if response is None:
            return {"omitted": call.omission_reason_code or "response_unavailable"}
        if response.semantic_revision_id is not None:
            revision = self.repository.get_semantic_revision(
                cursor, response.semantic_revision_id
            )
            if revision is None:
                raise ValueError("response_revision_unavailable")
            return project_semantic_revision_trace_message(
                cursor,
                revision_id=revision.revision_id,
                expected_conversation_id=revision.source_conversation_id,
                policy_id=call.policy_id,
                response_call=call,
            )
        value = self._decode_artifact(
            self.repository.get_artifact(cursor, response.artifact_id or "")
        )
        return dict(value) if isinstance(value, Mapping) else {"value": value}

    @staticmethod
    def _decode_artifact(artifact: TraceArtifactRecord | None) -> object:
        if artifact is None:
            raise ValueError("trace_artifact_unavailable")
        return json.loads(artifact.sanitized_bytes)


def _thaw_json(value: object) -> object:
    """Return plain JSON containers for a repository-frozen value.

    Args:
        value: A decoded column whose objects are ``MappingProxyType`` and
            whose arrays are tuples, at any depth.

    Returns:
        The same value built from ``dict`` and ``list``.
    """
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _normalized_usage_json(
    usage: Mapping[str, object], *, provider: str, model: str
) -> str | None:
    """Normalize one call's stored usage into ProviderUsage JSON.

    Args:
        usage: Stored usage: the provider's raw payload, or ProviderUsage
            fields for rows imported from legacy captures.
        provider: The call's provider, used when the payload names none.
        model: The call's model, used when the payload names none.

    Returns:
        ``ProviderUsage.to_json()``, or None for an unrecognized shape.
    """
    payload = _thaw_json(usage)
    if not isinstance(payload, dict):
        return None
    if set(payload) <= _PROVIDER_USAGE_FIELDS:
        normalized = ProviderUsage.from_json(json.dumps(payload))
        if normalized is not None:
            # A legacy row that never named them takes the call's own.
            normalized = dataclasses.replace(
                normalized,
                provider=normalized.provider or provider,
                model=normalized.model or model,
            )
    else:
        normalized = ProviderUsage.from_provider_payload(
            payload, provider=provider, model=model
        )
    return None if normalized is None else normalized.to_json()


__all__ = ["ConsoleTraceNativeReader"]
