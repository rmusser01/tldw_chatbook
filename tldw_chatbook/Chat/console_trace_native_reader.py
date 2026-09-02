"""Reconstruct viewer captures from the native Console semantic-trace ledger."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from tldw_chatbook.Chat.console_exchange_capture import CaptureDetail, ExchangeCapture
from tldw_chatbook.Chat.console_semantic_revision import (
    project_semantic_revision_provider_continuations,
    project_semantic_revision_trace_message,
)
from tldw_chatbook.Chat.console_trace_models import TraceCallState
from tldw_chatbook.Chat.console_trace_projection import NormalizedTraceCall
from tldw_chatbook.Chat.console_trace_provenance import TraceTransformKind
from tldw_chatbook.Chat.console_trace_redaction import (
    CREDENTIAL_SANITIZER_UNAVAILABLE,
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

_CONTINUATION_COMPONENTS = frozenset(
    {"continuation", TraceTransformKind.CONTINUATION_ATTACHMENT.value}
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
        usage_json = (
            json.dumps(dict(call.usage), separators=(",", ":"), sort_keys=True)
            if call.usage is not None
            else None
        )
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
        value = project_semantic_revision_provider_continuations(
            cursor,
            revision_ids=(revision_id,),
            expected_conversation_id=expected_conversation_id,
        )[revision_id]
        sanitized = CredentialSanitizer().sanitize(value)
        if not sanitized.available:
            return {"omitted": CREDENTIAL_SANITIZER_UNAVAILABLE}
        policy = self.repository.get_policy(cursor, call.policy_id)
        if policy is None:
            raise ValueError("trace_policy_unavailable")
        return self.service._pii_projected_value(sanitized.value, policy=policy)

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


__all__ = ["ConsoleTraceNativeReader"]
