"""Credential-safe verification of final Chatbook-owned provider values."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import cast
from tldw_chatbook.Chat.console_project_instructions import (
    canonical_provider_endpoint_identity,
)
from tldw_chatbook.Chat.console_prepared_request import freeze_json
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
)
from tldw_chatbook.Chat.console_trace_redaction import (
    CredentialSanitizationResult,
    CredentialSanitizer,
)
from tldw_chatbook.Chat.console_trace_models import SemanticRevisionRef, new_opaque_id
from tldw_chatbook.Chat.provider_continuation import (
    ProviderContinuationCheckpoint,
    dump_provider_continuation_json,
)


_SURFACE_VERIFICATION_ISSUER = object()


class FinalValueIntent(str, Enum):
    """Durable intent for one verified semantic value."""

    REVISION_REFERENCE = "revision_reference"
    PROVIDER_ARTIFACT = "provider_artifact"
    STRUCTURAL_VALUE = "structural_value"
    OMITTED = "omitted"


class ProviderCredentialSource(str, Enum):
    """Bounded credential-decision provenance; never a credential value."""

    RESOLVED_PRESENT = "resolved_present"
    EXPLICIT_KEYLESS = "explicit_keyless"
    NOT_SUPPLIED = "not_supplied"


@dataclass(frozen=True, slots=True)
class FinalValueBinding:
    """One final semantic component and its content ownership intent."""

    name: str
    value: object = field(repr=False)
    intents: tuple[FinalValueIntent, ...]
    provenance: tuple[TraceProvenance, ...] = field(default=(), repr=False)
    redacted: bool = False


@dataclass(frozen=True, slots=True)
class ProviderOverlayProvenance:
    """Bounded Chatbook-owned provider transform/default annotation."""

    kind: str
    source: str


@dataclass(frozen=True, slots=True, weakref_slot=True)
class ProviderRequestShadowBundle:
    """Verified sanitized shadow values, or one content-free omission."""

    available: bool
    components: tuple[FinalValueBinding, ...] = field(default=(), repr=False)
    handler_components: tuple[FinalValueBinding, ...] = field(default=(), repr=False)
    literal_payload: object | None = field(default=None, repr=False)
    system_components: tuple[object, ...] = field(default=(), repr=False)
    system_leaf_components: tuple[object, ...] = field(default=(), repr=False)
    endpoint_identity: str | None = field(default=None, repr=False)
    overlays: tuple[ProviderOverlayProvenance, ...] = ()
    credential_source: ProviderCredentialSource = ProviderCredentialSource.NOT_SUPPLIED
    omission_reason: TraceOmissionReason | None = None
    redacted: bool = False
    preparation_identity: str | None = field(default=None, repr=False)
    surface_boundary: object | None = field(default=None, repr=False)

    @property
    def boundary_kwargs(self) -> dict[str, object]:
        return {item.name: _thaw(item.value) for item in self.components}

    @property
    def handler_kwargs(self) -> dict[str, object]:
        return {item.name: _thaw(item.value) for item in self.handler_components}

    @property
    def literal_payload_value(self) -> object | None:
        """Return a detached mutable copy for a later persistence consumer."""

        return _thaw(self.literal_payload)

    def as_content_free_record(self) -> dict[str, object]:
        """Return the only safe durable shape for an unavailable bundle."""

        if self.available:
            raise ValueError("available provider shadow is content-bearing")
        return {
            "available": False,
            "omission_reason": (
                self.omission_reason.value if self.omission_reason is not None else None
            ),
        }


@dataclass(frozen=True, slots=True)
class VerifiedSurfaceDeltaItem:
    """One newly admitted structural slot; provider bytes stay in the bundle."""

    component_name: str
    ordinal: int
    provenance: TraceProvenance = field(repr=False)

    def __post_init__(self) -> None:
        if (
            self.component_name
            not in {
                "messages_payload",
                "provider_continuations",
                "omission",
            }
            or type(self.ordinal) is not int
            or self.ordinal < 0
        ):
            raise ValueError("surface_delta_item")


@dataclass(frozen=True, slots=True)
class VerifiedSurfaceReplacement:
    """One bounded replacement whose structural slot is carried by the delta."""

    predecessor_head_id: str
    start_node_id: str
    end_node_id: str
    start_sequence: int
    end_sequence: int
    current_ordinal: int
    item: VerifiedSurfaceDeltaItem = field(repr=False)


@dataclass(frozen=True, slots=True)
class VerifiedSurfaceReplacementRange:
    """Content-free range selected by preparation for one replacement."""

    predecessor_head_id: str
    start_node_id: str
    end_node_id: str
    start_sequence: int
    end_sequence: int
    current_ordinal: int = 0
    component_name: str = "messages_payload"
    component_ordinal: int | None = None

    def __post_init__(self) -> None:
        if (
            self.component_name not in {"messages_payload", "provider_continuations"}
            or type(self.current_ordinal) is not int
            or self.current_ordinal < 0
            or (
                self.component_ordinal is not None
                and (
                    type(self.component_ordinal) is not int
                    or self.component_ordinal < 0
                )
            )
        ):
            raise ValueError("surface_replacement_range")


@dataclass(frozen=True, slots=True)
class CompletedToolTurnWitness:
    """Content-free evidence rechecked against the durable call ledger."""

    origin_call_id: str
    terminal_call_id: str
    assistant_revision_id: str
    user_revision_id: str

    def __post_init__(self) -> None:
        for identity in (
            self.origin_call_id,
            self.terminal_call_id,
            self.assistant_revision_id,
            self.user_revision_id,
        ):
            SemanticRevisionRef(identity)


@dataclass(frozen=True, slots=True)
class SurfaceDeltaAdmission:
    """Preparation-owned declaration containing only newly admitted descriptors."""

    owner_id: str
    segment_id: str
    predecessor_surface_head_id: str | None
    route_identity: str
    preparation_identity: str = field(repr=False)
    descriptors: tuple[TraceProvenance, ...] = field(repr=False)
    projection_checkpoint: object | None = field(default=None, repr=False)
    replacement_range: VerifiedSurfaceReplacementRange | None = field(
        default=None, repr=False
    )
    completed_tool_turn: CompletedToolTurnWitness | None = field(
        default=None, kw_only=True
    )

    def __post_init__(self) -> None:
        if not all(
            (
                self.owner_id,
                self.segment_id,
                self.route_identity,
                self.preparation_identity,
            )
        ):
            raise ValueError("surface_delta_admission")
        if not self.descriptors and self.predecessor_surface_head_id is None:
            raise ValueError("surface_delta_admission")
        if self.completed_tool_turn is not None:
            witness = self.completed_tool_turn
            if (
                type(witness) is not CompletedToolTurnWitness
                or self.replacement_range is None
                or self.replacement_range.component_name != "messages_payload"
                or self.route_identity not in {"agent_first", "fresh"}
                or self.descriptors
                != (
                    SavedRevisionTraceProvenance(witness.assistant_revision_id),
                    SavedRevisionTraceProvenance(witness.user_revision_id),
                )
            ):
                raise ValueError("surface_delta_shape")
        elif self.replacement_range is not None and len(self.descriptors) != 1:
            raise ValueError("surface_delta_shape")


@dataclass(frozen=True, slots=True)
class VerifiedSurfaceDelta:
    """Preparation-bound new surface work; never a copy of prior history."""

    owner_id: str
    segment_id: str
    predecessor_surface_head_id: str | None
    route_identity: str
    preparation_identity: str = field(repr=False)
    child_binding: object | None = field(default=None, repr=False)
    items: tuple[VerifiedSurfaceDeltaItem, ...] = field(default=(), repr=False)
    replacement: VerifiedSurfaceReplacement | None = field(default=None, repr=False)
    completed_tool_turn: CompletedToolTurnWitness | None = field(
        default=None, kw_only=True
    )

    def __post_init__(self) -> None:
        if (
            not self.owner_id
            or not self.segment_id
            or not self.route_identity
            or not self.preparation_identity
        ):
            raise ValueError("surface_delta_identity")
        if self.completed_tool_turn is not None:
            witness = self.completed_tool_turn
            if (
                type(witness) is not CompletedToolTurnWitness
                or self.replacement is None
                or len(self.items) != 1
                or self.route_identity not in {"agent_first", "fresh"}
                or self.replacement.item.component_name != "messages_payload"
                or self.items[0].component_name != "messages_payload"
                or self.items[0].ordinal != self.replacement.item.ordinal + 1
                or self.replacement.item.provenance
                != SavedRevisionTraceProvenance(witness.assistant_revision_id)
                or self.items[0].provenance
                != SavedRevisionTraceProvenance(witness.user_revision_id)
            ):
                raise ValueError("surface_delta_shape")
        elif self.replacement is not None and self.items:
            raise ValueError("surface_delta_shape")
        if (
            self.replacement is None
            and not self.items
            and self.predecessor_surface_head_id is None
        ):
            raise ValueError("surface_delta_shape")


def build_verified_surface_delta(
    provenance: ProviderRequestProvenance,
    bundle: ProviderRequestShadowBundle,
    *,
    admission: SurfaceDeltaAdmission,
) -> VerifiedSurfaceDelta:
    """Build a delta from a preparation containing only newly admitted values."""

    if bundle.preparation_identity != admission.preparation_identity:
        raise ValueError("surface_delta_identity")
    routes = tuple(
        item
        for item in provenance.metadata
        if type(item) is RequestRouteTraceProvenance
    )
    if len(routes) != 1:
        raise ValueError("request_route_unavailable")
    if routes[0].route.value != admission.route_identity:
        raise ValueError("surface_delta_identity")
    if admission.projection_checkpoint is not None and not bundle.available:
        raise ValueError("surface_prefix_mismatch")
    items: tuple[VerifiedSurfaceDeltaItem, ...]
    if not bundle.available:
        reason = bundle.omission_reason or TraceOmissionReason.SOURCE_UNAVAILABLE
        items = (
            VerifiedSurfaceDeltaItem(
                "omission",
                0,
                OmittedTraceProvenance(
                    source=_surface_omission_source(provenance),
                    reason=reason,
                ),
            ),
        )
    else:
        bindings = {item.name: item for item in bundle.components}
        messages = bindings.get("messages_payload")
        if messages is None or not isinstance(messages.value, tuple):
            raise ValueError("surface_values_unavailable")
        message_projection_delta = getattr(
            provenance.messages_payload, "_console_trace_delta", None
        )
        continuation_projection_delta = getattr(
            provenance.continuations, "_console_trace_delta", None
        )
        projected = (
            admission.projection_checkpoint is not None
            or message_projection_delta is not None
            or continuation_projection_delta is not None
        )
        if (
            projected
            and message_projection_delta is None
            and continuation_projection_delta is None
        ):
            admitted = admission.descriptors
            full_messages = tuple(provenance.messages_payload)
            full_continuations = tuple(provenance.continuations)
            message_delta: tuple[TraceProvenance, ...]
            continuation_delta: tuple[TraceProvenance, ...]
            if admitted and full_messages[-len(admitted) :] == admitted:
                message_delta, continuation_delta = admitted, ()
            elif admitted and full_continuations[-len(admitted) :] == admitted:
                message_delta, continuation_delta = (), admitted
            elif admitted:
                raise ValueError("surface_delta_alignment")
            else:
                message_delta = continuation_delta = ()
        else:
            message_delta = tuple(
                provenance.messages_payload
                if message_projection_delta is None
                else message_projection_delta
            )
            continuation_delta = tuple(
                provenance.continuations
                if continuation_projection_delta is None
                else continuation_projection_delta
            )
        full_messages = provenance.messages_payload
        full_continuations = provenance.continuations
        if not projected:
            descriptors = full_messages + full_continuations
            values = tuple(messages.value)
            continuations = bindings.get("provider_continuations")
            if continuations is not None:
                if not isinstance(continuations.value, tuple):
                    raise ValueError("surface_values_unavailable")
                values += tuple(continuations.value)
            if len(descriptors) != len(values):
                raise ValueError("surface_provenance_mismatch")
            admitted = admission.descriptors
            if admitted == descriptors:
                message_delta, continuation_delta = full_messages, full_continuations
            elif admitted and full_messages[-len(admitted) :] == admitted:
                message_delta, continuation_delta = admitted, ()
            elif admitted and full_continuations[-len(admitted) :] == admitted:
                message_delta, continuation_delta = (), admitted
            elif admitted:
                raise ValueError("surface_delta_alignment")
            else:
                message_delta = continuation_delta = ()
        descriptors = message_delta + continuation_delta
        if descriptors != admission.descriptors:
            raise ValueError("surface_delta_alignment")
        replacement_range = admission.replacement_range
        delta_count = len(admission.descriptors)
        if replacement_range is not None:
            ordinal = replacement_range.current_ordinal
            if (
                delta_count != (2 if admission.completed_tool_turn is not None else 1)
                or ordinal < 0
                or ordinal >= len(full_messages) + len(full_continuations)
            ):
                raise ValueError("surface_delta_alignment")
        if delta_count == 0:
            items = ()
        else:
            message_delta_ordinal = int(
                getattr(
                    full_messages,
                    "_console_trace_delta_ordinal",
                    len(full_messages) - len(message_delta),
                )
            )
            continuation_delta_ordinal = int(
                getattr(
                    full_continuations,
                    "_console_trace_delta_ordinal",
                    len(full_continuations) - len(continuation_delta),
                )
            )
            items = tuple(
                VerifiedSurfaceDeltaItem(
                    "messages_payload",
                    (
                        message_delta_ordinal
                        if projected and replacement_range is not None
                        else len(full_messages) - len(message_delta)
                    )
                    + offset,
                    descriptor,
                )
                for offset, descriptor in enumerate(message_delta)
            ) + tuple(
                VerifiedSurfaceDeltaItem(
                    "provider_continuations",
                    (
                        continuation_delta_ordinal
                        if projected and replacement_range is not None
                        else len(full_continuations) - len(continuation_delta)
                    )
                    + offset,
                    descriptor,
                )
                for offset, descriptor in enumerate(continuation_delta)
            )
    replacement: VerifiedSurfaceReplacement | None = None
    if admission.replacement_range is not None:
        if len(items) != (2 if admission.completed_tool_turn is not None else 1):
            raise ValueError("surface_delta_shape")
        replacement_range = admission.replacement_range
        if items[0].component_name != replacement_range.component_name or (
            replacement_range.component_ordinal is not None
            and items[0].ordinal != replacement_range.component_ordinal
        ):
            raise ValueError("surface_delta_alignment")
        replacement = VerifiedSurfaceReplacement(
            predecessor_head_id=replacement_range.predecessor_head_id,
            start_node_id=replacement_range.start_node_id,
            end_node_id=replacement_range.end_node_id,
            start_sequence=replacement_range.start_sequence,
            end_sequence=replacement_range.end_sequence,
            current_ordinal=replacement_range.current_ordinal,
            item=items[0],
        )
        items = items[1:]
    child_binding = None
    if bundle.available:
        issuer = bundle.surface_boundary or admission.projection_checkpoint
        extend = getattr(issuer, "_extend_surface_projection", None)
        if not callable(extend):
            raise ValueError("surface_verified_bundle")
        child_binding = extend(
            admission=admission,
            replacement=replacement,
            preparation_identity=bundle.preparation_identity,
            items=((replacement.item,) + items) if replacement is not None else items,
            bundle=bundle,
            surface_boundary_identity=id(bundle.surface_boundary),
            provenance=provenance,
        )
    return VerifiedSurfaceDelta(
        owner_id=admission.owner_id,
        segment_id=admission.segment_id,
        predecessor_surface_head_id=admission.predecessor_surface_head_id,
        route_identity=routes[0].route.value,
        preparation_identity=bundle.preparation_identity,
        child_binding=child_binding,
        items=items,
        replacement=replacement,
        completed_tool_turn=admission.completed_tool_turn,
    )


def _surface_omission_source(
    provenance: ProviderRequestProvenance,
) -> TraceProvenanceSource:
    for descriptor in provenance.messages_payload:
        source = getattr(descriptor, "source", None)
        if source is not None:
            return source
    return TraceProvenanceSource.PROVIDER_OVERLAY


_DEFAULT_PROVENANCE_KEYS = (
    "streaming",
    "temp",
    "topp",
    "topk",
    "minp",
    "max_tokens",
    "seed",
    "presence_penalty",
    "frequency_penalty",
    "response_format",
    "reasoning_effort",
    "reasoning_summary",
    "verbosity",
    "thinking_effort",
    "thinking_budget_tokens",
    "prompt_caching",
)


def reconstruct_provider_gateway_kwargs(
    resolution: object,
    request: object,
) -> dict[str, object]:
    """Independently reconstruct the exact generic gateway boundary values."""

    execution_key = getattr(resolution, "execution_key")
    kwargs: dict[str, object | None] = {
        "api_endpoint": execution_key,
        "system_message": getattr(request, "system_message"),
        "messages_payload": _thaw(getattr(request, "messages_payload")),
        "api_key": getattr(resolution, "api_key"),
        "model": getattr(resolution, "model"),
        "streaming": getattr(resolution, "streaming"),
        "temp": getattr(resolution, "temperature"),
        "topp": getattr(resolution, "top_p"),
        "maxp": getattr(resolution, "top_p"),
        "topk": getattr(resolution, "top_k"),
        "minp": getattr(resolution, "min_p"),
        "max_tokens": getattr(resolution, "max_tokens"),
        "seed": getattr(resolution, "seed"),
        "presence_penalty": getattr(resolution, "presence_penalty"),
        "frequency_penalty": getattr(resolution, "frequency_penalty"),
        "reasoning_effort": getattr(resolution, "reasoning_effort"),
        "reasoning_summary": getattr(resolution, "reasoning_summary"),
        "verbosity": getattr(resolution, "verbosity"),
        "thinking_effort": getattr(resolution, "thinking_effort"),
        "thinking_budget_tokens": getattr(resolution, "thinking_budget_tokens"),
        "tools": _thaw(getattr(request, "tools")) or None,
        "response_format": _thaw(getattr(request, "response_format")),
        "prompt_caching": getattr(resolution, "prompt_caching"),
    }
    if execution_key == "qwencloud":
        kwargs["api_mode"] = getattr(resolution, "api_mode")
        kwargs["api_base_url"] = getattr(resolution, "base_url") or None
    elif execution_key in {"moonshot", "zai"}:
        kwargs.update(
            api_base_url=getattr(resolution, "base_url") or None,
            request_timeout=getattr(resolution, "request_timeout"),
            request_retries=getattr(resolution, "request_retries"),
            request_retry_delay=getattr(resolution, "request_retry_delay"),
        )
        continuations = getattr(request, "continuation_groups")
        if continuations:
            kwargs["provider_continuations"] = [
                group.checkpoint for group in continuations
            ]
    elif execution_key in {
        "anthropic",
        "custom-openai-api",
        "custom-openai-api-2",
        "local_vllm",
        "mistral",
        "mistralai",
        "vllm",
    }:
        kwargs["api_base_url"] = getattr(resolution, "base_url") or None
        if execution_key in {"custom-openai-api", "custom-openai-api-2"}:
            kwargs["api_key_resolved"] = True
    elif execution_key == "openai" and getattr(request, "response_format") is not None:
        kwargs["api_base_url"] = getattr(resolution, "base_url") or None
    return {name: value for name, value in kwargs.items() if value is not None}


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def verify_provider_request_shadow(
    *,
    actual_kwargs: Mapping[str, object],
    expected_kwargs: Mapping[str, object],
    provenance: ProviderRequestProvenance,
    project_handler_kwargs: Callable[[dict[str, object]], Mapping[str, object]],
    handler_source_names: Mapping[str, str] | None = None,
    known_credentials: tuple[str, ...] = (),
    literal_payload: object | None = None,
    endpoint_identity: str | None = None,
    extra_overlays: tuple[ProviderOverlayProvenance, ...] = (),
    preparation_identity: str | None = None,
    system_component_values: tuple[object, ...] = (),
    surface_boundary: object | None = None,
    omit_ephemeral_endpoint: bool = False,
) -> ProviderRequestShadowBundle:
    """Sanitize, independently compare, bind, and project final provider values."""

    effective_preparation_identity = preparation_identity or new_opaque_id()
    credential_source = _credential_source(actual_kwargs)
    expected_credential_source = _credential_source(expected_kwargs)
    if surface_boundary is not None:
        verify_raw_surface = getattr(
            surface_boundary, "_verify_raw_surface_values", None
        )
        if not callable(verify_raw_surface) or not verify_raw_surface(
            provenance,
            actual_kwargs,
            _SURFACE_VERIFICATION_ISSUER,
        ):
            return _unavailable(
                TraceOmissionReason.ALIGNMENT_MISMATCH,
                credential_source,
                effective_preparation_identity,
            )
    try:
        actual_values = _normalize_provider_continuations(actual_kwargs)
        expected_values = _normalize_provider_continuations(expected_kwargs)
    except Exception:  # noqa: BLE001 - checkpoint context may contain credentials
        return _unavailable(
            TraceOmissionReason.SANITIZER_FAILED,
            credential_source,
            effective_preparation_identity,
        )
    if credential_source is not expected_credential_source or not _constant_equal(
        actual_values,
        expected_values,
    ):
        return _unavailable(
            TraceOmissionReason.ALIGNMENT_MISMATCH,
            credential_source,
            effective_preparation_identity,
        )
    sanitizer = CredentialSanitizer(known_credentials=known_credentials)
    actual = sanitizer.sanitize(actual_values)
    expected = sanitizer.sanitize(expected_values)
    prepared_surface = None
    if surface_boundary is not None:
        surface_values = getattr(surface_boundary, "_provider_surface_values", None)
        verify_surface = getattr(surface_boundary, "_verify_surface_values", None)
        if not callable(surface_values) or not callable(verify_surface):
            return _unavailable(
                TraceOmissionReason.ALIGNMENT_MISMATCH,
                credential_source,
                effective_preparation_identity,
            )
        prepared_surface = sanitizer.sanitize(surface_values())
    literal = (
        sanitizer.sanitize(literal_payload) if literal_payload is not None else None
    )
    system_parts = sanitizer.sanitize(system_component_values)
    if omit_ephemeral_endpoint:
        from tldw_chatbook.Chat.console_endpoint_provenance import (
            EPHEMERAL_SESSION_ENDPOINT_OMITTED,
        )

        canonical_endpoint: str | None = EPHEMERAL_SESSION_ENDPOINT_OMITTED
    elif endpoint_identity is None:
        canonical_endpoint: str | None = None
    else:
        try:
            canonical_endpoint = canonical_provider_endpoint_identity(endpoint_identity)
        except ValueError:
            return _unavailable(
                TraceOmissionReason.SANITIZER_FAILED,
                credential_source,
                effective_preparation_identity,
            )
    endpoint = sanitizer.sanitize(canonical_endpoint)
    if (
        not actual.available
        or not expected.available
        or not endpoint.available
        or not system_parts.available
        or (prepared_surface is not None and not prepared_surface.available)
        or not isinstance(system_parts.value, (list, tuple))
        or (literal is not None and not literal.available)
    ):
        return _unavailable(
            TraceOmissionReason.SANITIZER_FAILED,
            credential_source,
            effective_preparation_identity,
        )
    if (
        actual.value != expected.value
        or actual.redacted != expected.redacted
        or not isinstance(actual.value, dict)
        or not _descriptors_align(actual.value, literal, provenance)
    ):
        return _unavailable(
            TraceOmissionReason.ALIGNMENT_MISMATCH,
            credential_source,
            effective_preparation_identity,
        )
    if prepared_surface is not None:
        actual_surface = {
            "messages_payload": actual.value.get("messages_payload"),
            "provider_continuations": actual.value.get("provider_continuations", []),
        }
        verify_surface = getattr(surface_boundary, "_verify_surface_values")
        if not verify_surface(
            provenance,
            actual_surface,
            prepared_surface.value,
            _SURFACE_VERIFICATION_ISSUER,
        ):
            return _unavailable(
                TraceOmissionReason.ALIGNMENT_MISMATCH,
                credential_source,
                effective_preparation_identity,
            )
    provider_system_parts = _verified_provider_system_parts(
        actual.value,
        provenance.system_message,
        tuple(cast(list[object] | tuple[object, ...], system_parts.value)),
        components_supplied=bool(system_component_values),
    )
    if provider_system_parts is None:
        return _unavailable(
            TraceOmissionReason.ALIGNMENT_MISMATCH,
            credential_source,
            effective_preparation_identity,
        )
    try:
        raw_projected = project_handler_kwargs(dict(actual.value))
        projected = sanitizer.sanitize(raw_projected)
    except Exception:  # noqa: BLE001 - exception context may contain credentials
        return _unavailable(
            TraceOmissionReason.SANITIZER_FAILED,
            credential_source,
            effective_preparation_identity,
        )
    if not projected.available or not isinstance(projected.value, dict):
        return _unavailable(
            TraceOmissionReason.SANITIZER_FAILED,
            credential_source,
            effective_preparation_identity,
        )
    component_redactions = _component_redactions(actual_values, sanitizer)
    handler_redactions = _component_redactions(raw_projected, sanitizer)
    if component_redactions is None or handler_redactions is None:
        return _unavailable(
            TraceOmissionReason.SANITIZER_FAILED,
            credential_source,
            effective_preparation_identity,
        )
    omitted_component_names = (
        frozenset({"api_base_url"}) if omit_ephemeral_endpoint else frozenset()
    )
    components = tuple(
        _bind_component(
            name,
            freeze_json(value),
            provenance,
            redacted=component_redactions.get(name, False),
        )
        for name, value in actual.value.items()
        if name not in omitted_component_names
    )
    handler_components = tuple(
        FinalValueBinding(
            name=name,
            value=freeze_json(value),
            intents=(FinalValueIntent.STRUCTURAL_VALUE,),
            redacted=(
                handler_redactions.get(name, False)
                or component_redactions.get(
                    (handler_source_names or {}).get(name, ""), False
                )
            ),
        )
        for name, value in projected.value.items()
        if (handler_source_names or {}).get(name, name) not in omitted_component_names
    )
    was_redacted = (
        actual.redacted
        or projected.redacted
        or endpoint.redacted
        or (literal.redacted if literal is not None else False)
        or system_parts.redacted
    )
    redaction_overlay = (
        (ProviderOverlayProvenance("credential_redaction", "mandatory_filter"),)
        if was_redacted
        else ()
    )
    bundle = ProviderRequestShadowBundle(
        available=True,
        components=components,
        handler_components=handler_components,
        literal_payload=(
            freeze_json(literal.value)
            if literal is not None and literal.value is not None
            else None
        ),
        system_components=tuple(freeze_json(value) for value in provider_system_parts),
        system_leaf_components=(
            tuple(
                freeze_json(value)
                for value in cast(Sequence[object], system_parts.value)
            )
            if system_component_values
            else ()
        ),
        endpoint_identity=(endpoint.value if isinstance(endpoint.value, str) else None),
        overlays=(
            *_provider_overlays(actual.value),
            *extra_overlays,
            *(
                (
                    ProviderOverlayProvenance(
                        "ephemeral_endpoint",
                        "session_policy",
                    ),
                )
                if omit_ephemeral_endpoint
                else ()
            ),
            *redaction_overlay,
        ),
        credential_source=credential_source,
        redacted=was_redacted,
        preparation_identity=effective_preparation_identity,
        surface_boundary=surface_boundary,
    )
    bind_bundle = getattr(surface_boundary, "_bind_verified_bundle", None)
    if callable(bind_bundle):
        bind_bundle(provenance, bundle, _SURFACE_VERIFICATION_ISSUER)
    return bundle


def _verified_provider_system_parts(
    values: Mapping[str, object],
    descriptor: TraceProvenance | None,
    parts: tuple[object, ...],
    *,
    components_supplied: bool,
) -> tuple[object, ...] | None:
    """Verify a SINGLE_PREAMBLE decomposition and retain provider-only leaves."""

    if not components_supplied:
        return ()
    system_value = values.get("system_message")
    if not isinstance(system_value, str) or any(
        not isinstance(part, str) for part in parts
    ):
        return None
    if "\n\n".join(cast(tuple[str, ...], parts)) != system_value:
        return None
    provider_mask: list[bool] = []

    def visit(item: TraceProvenance) -> None:
        if type(item) is SavedRevisionTraceProvenance:
            provider_mask.append(False)
        elif type(item) is ProviderArtifactTraceProvenance:
            provider_mask.append(True)
        elif type(item) is OmittedTraceProvenance:
            return
        elif type(item) is DerivedTraceProvenance:
            derived = cast(DerivedTraceProvenance, item)
            for nested in derived.inputs:
                visit(nested)
            if derived.artifact is not None:
                visit(derived.artifact)
        else:
            raise ValueError("system_component_alignment")

    if descriptor is None:
        return None
    visit(descriptor)
    if len(provider_mask) != len(parts):
        return None
    return tuple(
        part for part, provider_only in zip(parts, provider_mask) if provider_only
    )


def _normalize_provider_continuations(
    values: Mapping[str, object],
) -> Mapping[str, object]:
    continuations = values.get("provider_continuations")
    if type(continuations) not in {list, tuple} or not any(
        type(item) is ProviderContinuationCheckpoint
        for item in cast(list[object] | tuple[object, ...], continuations)
    ):
        return values
    continuation_values = cast(list[object] | tuple[object, ...], continuations)
    normalized_items = [
        _checkpoint_semantic_value(item)
        if type(item) is ProviderContinuationCheckpoint
        else item
        for item in continuation_values
    ]
    normalized = dict(values)
    normalized["provider_continuations"] = (
        normalized_items if type(continuations) is list else tuple(normalized_items)
    )
    return normalized


def _checkpoint_semantic_value(
    checkpoint: ProviderContinuationCheckpoint,
) -> object:
    serialized = dump_provider_continuation_json(checkpoint)
    if serialized is None:  # pragma: no cover - exact checkpoint is never absent
        raise ValueError("checkpoint unavailable")
    return json.loads(serialized)


def _constant_equal(
    left: object,
    right: object,
    *,
    active: set[tuple[int, int]] | None = None,
) -> bool:
    """Compare canonical values without hashes, repr, or retained diagnostics."""

    if type(left) is not type(right):
        return False
    pairs = active if active is not None else set()
    pair = (id(left), id(right))
    if pair in pairs:
        return True
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if tuple(left.keys()) != tuple(right.keys()):
            return False
        pairs.add(pair)
        try:
            return all(
                _constant_equal(left[key], right[key], active=pairs) for key in left
            )
        finally:
            pairs.remove(pair)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return False
        pairs.add(pair)
        try:
            return all(
                _constant_equal(left_item, right_item, active=pairs)
                for left_item, right_item in zip(left, right)
            )
        finally:
            pairs.remove(pair)
    try:
        return bool(left == right)
    except Exception:  # noqa: BLE001 - equality context may contain credentials
        return False


def _component_redactions(
    values: Mapping[str, object],
    sanitizer: CredentialSanitizer,
) -> dict[str, bool] | None:
    flags: dict[str, bool] = {}
    for name, value in values.items():
        item = sanitizer.sanitize({name: value})
        if not item.available or not isinstance(item.value, dict):
            return None
        for sanitized_name in item.value:
            flags[sanitized_name] = item.redacted
    return flags


def _unavailable(
    reason: TraceOmissionReason,
    credential_source: ProviderCredentialSource,
    preparation_identity: str,
) -> ProviderRequestShadowBundle:
    return ProviderRequestShadowBundle(
        available=False,
        credential_source=credential_source,
        omission_reason=reason,
        preparation_identity=preparation_identity,
    )


def _credential_source(values: Mapping[str, object]) -> ProviderCredentialSource:
    key = values.get("api_key")
    if isinstance(key, str) and key:
        return ProviderCredentialSource.RESOLVED_PRESENT
    if values.get("api_key_resolved") is True:
        return ProviderCredentialSource.EXPLICIT_KEYLESS
    return ProviderCredentialSource.NOT_SUPPLIED


def _descriptors_align(
    values: Mapping[str, object],
    literal: CredentialSanitizationResult | None,
    provenance: ProviderRequestProvenance,
) -> bool:
    messages = values.get("messages_payload")
    tools = values.get("tools", [])
    continuations = values.get("provider_continuations", [])
    if not isinstance(messages, (list, tuple)) or len(messages) != len(
        provenance.messages_payload
    ):
        return False
    if not isinstance(tools, list) or len(tools) != len(provenance.tools):
        return False
    if not isinstance(continuations, (list, tuple)) or len(continuations) != len(
        provenance.continuations
    ):
        return False
    has_system = "system_message" in values
    if has_system != (provenance.system_message is not None):
        return False
    if literal is not None and literal.available:
        literal_value = literal.value
        if isinstance(literal_value, Mapping) and "messages" in literal_value:
            literal_messages = literal_value["messages"]
            if not isinstance(literal_messages, list) or len(literal_messages) != len(
                provenance.messages
            ):
                return False
    return True


def _bind_component(
    name: str,
    value: object,
    provenance: ProviderRequestProvenance,
    *,
    redacted: bool,
) -> FinalValueBinding:
    descriptors: tuple[TraceProvenance, ...]
    if name == "system_message" and provenance.system_message is not None:
        descriptors = (provenance.system_message,)
    elif name == "messages_payload":
        projection_delta = getattr(
            provenance.messages_payload,
            "_console_trace_delta",
            None,
        )
        descriptors = (
            tuple(projection_delta)
            if projection_delta is not None
            else tuple(provenance.messages_payload)
        )
    elif name == "tools":
        descriptors = provenance.tools
    elif name == "provider_continuations":
        projection_delta = getattr(
            provenance.continuations,
            "_console_trace_delta",
            None,
        )
        descriptors = (
            tuple(projection_delta)
            if projection_delta is not None
            else tuple(provenance.continuations)
        )
    elif name in {"reasoning_effort", "thinking_effort", "thinking_budget_tokens"}:
        descriptors = provenance.thinking
    else:
        descriptors = ()
    intents = tuple(_intent(item) for item in descriptors) or (
        FinalValueIntent.STRUCTURAL_VALUE,
    )
    return FinalValueBinding(
        name=name,
        value=value,
        intents=intents,
        provenance=descriptors,
        redacted=redacted,
    )


def _intent(descriptor: TraceProvenance) -> FinalValueIntent:
    if type(descriptor) is SavedRevisionTraceProvenance:
        return FinalValueIntent.REVISION_REFERENCE
    if type(descriptor) is OmittedTraceProvenance:
        return FinalValueIntent.OMITTED
    if type(descriptor) is ProviderArtifactTraceProvenance:
        return FinalValueIntent.PROVIDER_ARTIFACT
    if type(descriptor) is DerivedTraceProvenance:
        return (
            FinalValueIntent.PROVIDER_ARTIFACT
            if descriptor.artifact is not None
            else FinalValueIntent.STRUCTURAL_VALUE
        )
    return FinalValueIntent.STRUCTURAL_VALUE


def _provider_overlays(
    values: Mapping[str, object],
) -> tuple[ProviderOverlayProvenance, ...]:
    endpoint = str(values.get("api_endpoint") or "")
    overlays = [
        ProviderOverlayProvenance(
            f"parameter:{name}", "explicit" if name in values else "adapter_default"
        )
        for name in _DEFAULT_PROVENANCE_KEYS
    ]
    if endpoint == "anthropic" and values.get("prompt_caching") is True:
        overlays.append(
            ProviderOverlayProvenance("anthropic_cache_overlay", "explicit")
        )
    if endpoint == "qwencloud":
        overlays.append(
            ProviderOverlayProvenance(
                "qwen_api_mode",
                "explicit" if "api_mode" in values else "adapter_default",
            )
        )
    if endpoint in {"moonshot", "zai"} and any(
        key in values
        for key in ("request_timeout", "request_retries", "request_retry_delay")
    ):
        overlays.append(ProviderOverlayProvenance("transport_retry_policy", "explicit"))
    if values.get("provider_continuations"):
        overlays.append(ProviderOverlayProvenance("provider_continuation", "explicit"))
    if values.get("api_key_resolved") is True:
        overlays.append(ProviderOverlayProvenance("credential_decision", "resolved"))
    return tuple(overlays)
