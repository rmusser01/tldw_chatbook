"""Gateway-owned bounded capabilities for post-dispatch voice trace promotion.

Semantic provider values never enter the attempt snapshot.  The gateway retains
them here and gives callers only identity-bearing, one-use opaque capabilities.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum, auto
import json
from threading import RLock
import time
from typing import TYPE_CHECKING, TypeVar, cast
from uuid import UUID, uuid4, uuid5

from tldw_chatbook.Chat.console_exchange_capture import (
    FrozenProvisionalCaptureEligibility,
)
from tldw_chatbook.Chat.console_trace_models import FrozenTracePolicy, TraceCallState
from tldw_chatbook.Chat.console_voice_trace_promotion import (
    MAX_PROMOTED_TRACE_BYTES,
    MAX_PROMOTED_TRACE_CALLS,
    PostDispatchTraceArtifact,
    PostDispatchTraceCall,
    PostDispatchTraceHeaderComponent,
    PostDispatchTraceImport,
    PostDispatchTraceResponse,
    PostDispatchTraceSurfaceComponent,
    PostDispatchTraceSystemComponent,
    derive_post_dispatch_trace_ids,
    derive_post_dispatch_trace_node_id,
)

if TYPE_CHECKING:
    from tldw_chatbook.Chat.console_trace_final_values import (
        ProviderRequestShadowBundle,
    )
    from tldw_chatbook.Chat.console_trace_provenance import (
        ProviderRequestProvenance,
    )


MAX_PROVISIONAL_TRACE_APP_BYTES = 128 * 1024 * 1024
PROVISIONAL_TRACE_TTL_SECONDS = 10 * 60.0
_MAX_INELIGIBLE_TOMBSTONES = 4_096
_VOICE_TRACE_ARTIFACT_NAMESPACE = UUID("c1881f60-7944-4f2f-b278-5302c9d57a46")
_VOICE_USER_REVISION_PENDING = "voice_user_revision_pending"
_VOICE_ASSISTANT_REVISION_PENDING = "voice_assistant_revision_pending"
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
_REASONING_KEYS = frozenset(
    {
        "reasoning_effort",
        "reasoning_summary",
        "thinking_budget_tokens",
        "thinking_effort",
        "verbosity",
    }
)


class ProvisionalTraceUnavailable(RuntimeError):
    """A content-free failure to issue or redeem a trace capability."""

    def __init__(self) -> None:
        super().__init__("Provisional exchange capture is unavailable.")


class _ClaimReleaseDisposition(Enum):
    """Content-free result of settling a confirmed pre-commit failure."""

    RELEASED_RETRYABLE = auto()
    EXPIRED_DESTROYED = auto()


def _uuid(value: str, name: str) -> None:
    try:
        parsed = UUID(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a canonical UUID") from exc
    if parsed.version not in {4, 5} or str(parsed) != value:
        raise ValueError(f"{name} must be a canonical UUID")


@dataclass(frozen=True, slots=True)
class VoiceTraceImportContext:
    """Content-free committed-pair lineage supplied only after pair success."""

    import_id: str
    conversation_id: str
    user_message_id: str
    user_revision_id: str
    assistant_message_id: str
    assistant_revision_id: str
    turn_id: str
    run_id: str
    policy: FrozenTracePolicy

    def __post_init__(self) -> None:
        for name in (
            "import_id",
            "conversation_id",
            "user_message_id",
            "user_revision_id",
            "assistant_message_id",
            "assistant_revision_id",
            "turn_id",
        ):
            _uuid(getattr(self, name), name)
        if type(self.run_id) is not str or not self.run_id or len(self.run_id) > 512:
            raise ValueError("run_id must be bounded and non-empty")
        if type(self.policy) is not FrozenTracePolicy:
            raise TypeError("policy must be a FrozenTracePolicy")


@dataclass(frozen=True, slots=True)
class ProvisionalTraceObservation:
    """Actual content-free call chronology observed by the gateway."""

    call_id: str
    call_sequence: int
    dispatch_started_at: str
    response_started_at: str | None
    settled_at: str


class ProvisionalTraceAttempt:
    """Private-construction gateway authority for one dispatch-time attempt."""

    __slots__ = ("_handle", "_issuer", "promotion_id", "attempt_id")

    def __new__(cls) -> ProvisionalTraceAttempt:
        raise TypeError("ProvisionalTraceAttempt is gateway-issued")

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("gateway capabilities are immutable")

    @classmethod
    def _issue(
        cls,
        *,
        issuer: object,
        handle: str,
        promotion_id: str,
        attempt_id: str,
    ) -> ProvisionalTraceAttempt:
        value = object.__new__(cls)
        object.__setattr__(value, "_issuer", issuer)
        object.__setattr__(value, "_handle", handle)
        object.__setattr__(value, "promotion_id", promotion_id)
        object.__setattr__(value, "attempt_id", attempt_id)
        return value

    def __repr__(self) -> str:
        return (
            "ProvisionalTraceAttempt("
            f"promotion_id={self.promotion_id!r}, attempt_id={self.attempt_id!r})"
        )


class ProvisionalTraceEnvelope:
    """Private-construction opaque identity for one retained provider call."""

    __slots__ = (
        "_issuer",
        "_attempt_handle",
        "envelope_id",
        "promotion_id",
        "attempt_id",
        "call_id",
        "call_sequence",
    )

    def __new__(cls) -> ProvisionalTraceEnvelope:
        raise TypeError("ProvisionalTraceEnvelope is gateway-issued")

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("gateway capabilities are immutable")

    @classmethod
    def _issue(
        cls,
        *,
        issuer: object,
        attempt_handle: str,
        promotion_id: str,
        attempt_id: str,
        call_id: str,
        call_sequence: int,
    ) -> ProvisionalTraceEnvelope:
        value = object.__new__(cls)
        object.__setattr__(value, "_issuer", issuer)
        object.__setattr__(value, "_attempt_handle", attempt_handle)
        object.__setattr__(value, "envelope_id", str(uuid4()))
        object.__setattr__(value, "promotion_id", promotion_id)
        object.__setattr__(value, "attempt_id", attempt_id)
        object.__setattr__(value, "call_id", call_id)
        object.__setattr__(value, "call_sequence", call_sequence)
        return value

    def __repr__(self) -> str:
        return (
            "ProvisionalTraceEnvelope("
            f"envelope_id={self.envelope_id!r}, call_sequence={self.call_sequence})"
        )


class ProvisionalTraceManifest:
    """Private-construction sealed completeness proof for one attempt."""

    __slots__ = (
        "_issuer",
        "_attempt_handle",
        "manifest_id",
        "promotion_id",
        "attempt_id",
        "envelope_ids",
        "expected_call_count",
        "aggregate_payload_bytes",
        "observed_chronology",
    )

    def __new__(cls) -> ProvisionalTraceManifest:
        raise TypeError("ProvisionalTraceManifest is gateway-issued")

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("gateway capabilities are immutable")

    @classmethod
    def _issue(
        cls,
        *,
        issuer: object,
        attempt_handle: str,
        promotion_id: str,
        attempt_id: str,
        envelopes: tuple[ProvisionalTraceEnvelope, ...],
        aggregate_payload_bytes: int,
        chronology: tuple[ProvisionalTraceObservation, ...],
    ) -> ProvisionalTraceManifest:
        value = object.__new__(cls)
        object.__setattr__(value, "_issuer", issuer)
        object.__setattr__(value, "_attempt_handle", attempt_handle)
        object.__setattr__(value, "manifest_id", str(uuid4()))
        object.__setattr__(value, "promotion_id", promotion_id)
        object.__setattr__(value, "attempt_id", attempt_id)
        object.__setattr__(
            value, "envelope_ids", tuple(item.envelope_id for item in envelopes)
        )
        object.__setattr__(value, "expected_call_count", len(envelopes))
        object.__setattr__(value, "aggregate_payload_bytes", aggregate_payload_bytes)
        object.__setattr__(value, "observed_chronology", chronology)
        return value

    def __repr__(self) -> str:
        return (
            "ProvisionalTraceManifest("
            f"manifest_id={self.manifest_id!r}, "
            f"expected_call_count={self.expected_call_count}, "
            f"aggregate_payload_bytes={self.aggregate_payload_bytes})"
        )


class _ProvisionalTraceClaim:
    __slots__ = ("_issuer", "_attempt_handle", "_claim_id")

    def __new__(cls) -> _ProvisionalTraceClaim:
        raise TypeError("claims are registry-issued")

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("gateway capabilities are immutable")

    @classmethod
    def _issue(
        cls,
        *,
        issuer: object,
        attempt_handle: str,
    ) -> _ProvisionalTraceClaim:
        value = object.__new__(cls)
        object.__setattr__(value, "_issuer", issuer)
        object.__setattr__(value, "_attempt_handle", attempt_handle)
        object.__setattr__(value, "_claim_id", str(uuid4()))
        return value

    def __repr__(self) -> str:
        return "ProvisionalTraceClaim()"


@dataclass(slots=True)
class _AttemptState:
    capability: ProvisionalTraceAttempt
    promotion_id: str
    attempt_id: str
    created_at: float
    policy: FrozenTracePolicy
    calls: list[PostDispatchTraceCall] = field(default_factory=list, repr=False)
    envelopes: list[ProvisionalTraceEnvelope] = field(default_factory=list)
    envelope_ids: list[str] = field(default_factory=list)
    artifacts: dict[str, PostDispatchTraceArtifact] = field(
        default_factory=dict, repr=False
    )
    retained_bytes: int = 0
    inline_payload_bytes: int = 0
    manifest: ProvisionalTraceManifest | None = None
    manifest_id: str | None = None
    sealed_at: float | None = None
    claim: _ProvisionalTraceClaim | None = None
    reserved_call_count: int = 0


@dataclass(frozen=True, slots=True)
class _ProvisionalCallIdentity:
    call_id: str
    call_sequence: int
    idempotency_key: str


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(
        _thaw(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _binding(
    bundle: object,
    name: str,
) -> object | None:
    return next(
        (item for item in getattr(bundle, "components", ()) if item.name == name),
        None,
    )


def _artifact(
    *,
    promotion_id: str,
    call_sequence: int,
    label: str,
    value: object,
    policy: FrozenTracePolicy | None = None,
) -> PostDispatchTraceArtifact:
    from tldw_chatbook.Chat.console_trace_custom_pii import (
        CUSTOM_PII_RULESET_UNAVAILABLE,
        redact_pii_value_for_ruleset_revision,
    )
    from tldw_chatbook.Chat.console_trace_repository import (
        _sanitize_trace_artifact_bytes,
    )

    body = _sanitize_trace_artifact_bytes(
        _canonical_json(value).encode("utf-8"),
        media_type="application/json",
        normalization_version="canonical-json-v1",
    )
    redactions = ()
    if policy is not None and policy.pii_redaction_enabled:
        result = redact_pii_value_for_ruleset_revision(
            json.loads(body), policy.pii_ruleset_revision_id
        )
        value = (
            result.value
            if result.available
            else {
                "omitted": result.omission_reason_code or CUSTOM_PII_RULESET_UNAVAILABLE
            }
        )
        body = _sanitize_trace_artifact_bytes(
            _canonical_json(value).encode("utf-8"),
            media_type="application/json",
            normalization_version="canonical-json-v1",
        )
        redactions = result.field_redactions if result.available else ()
    return PostDispatchTraceArtifact(
        artifact_id=str(
            uuid5(
                _VOICE_TRACE_ARTIFACT_NAMESPACE,
                f"{promotion_id}:{call_sequence}:{label}",
            )
        ),
        media_type="application/json",
        normalization_version="canonical-json-v1",
        sanitized_bytes=body,
        field_redactions=redactions,
    )


class ProvisionalVoiceTraceCallBoundary:
    """In-memory Capture-On boundary whose terminal value stays gateway-owned."""

    __slots__ = (
        "_attempt",
        "_bound_bundle",
        "_bound_provenance",
        "_verified_bundle",
        "_verified_provenance",
        "_policy",
        "_dispatch_started_at",
        "_retain_call",
        "_reserved",
        "_response_started_at",
        "identity",
        "preparation_identity",
    )

    def __init__(
        self,
        attempt: ProvisionalTraceAttempt,
        call_sequence: int,
        retain_call: Callable[
            [ProvisionalTraceAttempt, PostDispatchTraceCall],
            ProvisionalTraceEnvelope | None,
        ],
        policy: FrozenTracePolicy,
    ) -> None:
        self._retain_call = retain_call
        self._policy = policy
        self._attempt = attempt
        self.identity = _ProvisionalCallIdentity(
            call_id=derive_post_dispatch_trace_ids(
                attempt.promotion_id,
                call_count=call_sequence + 1,
            ).call_ids[call_sequence],
            call_sequence=call_sequence,
            idempotency_key=f"voice:{attempt.attempt_id}:{call_sequence}",
        )
        self.preparation_identity = str(uuid4())
        self._reserved = False
        self._bound_bundle: ProviderRequestShadowBundle | None = None
        self._bound_provenance: ProviderRequestProvenance | None = None
        self._dispatch_started_at: str | None = None
        self._response_started_at: str | None = None
        self._verified_bundle: ProviderRequestShadowBundle | None = None
        self._verified_provenance: ProviderRequestProvenance | None = None

    def _bind_verified_bundle(self, provenance, bundle, issuer: object) -> None:
        """Receive the exact immutable result from dev's existing verifier."""
        from tldw_chatbook.Chat.console_trace_final_values import (
            _SURFACE_VERIFICATION_ISSUER,
            ProviderRequestShadowBundle,
        )
        from tldw_chatbook.Chat.console_trace_provenance import (
            ProviderRequestProvenance,
        )

        if (
            issuer is not _SURFACE_VERIFICATION_ISSUER
            or type(bundle) is not ProviderRequestShadowBundle
            or type(provenance) is not ProviderRequestProvenance
            or not self._reserved
            or self._verified_bundle is not None
            or self._bound_bundle is not None
            or bundle.preparation_identity != self.preparation_identity
        ):
            raise ProvisionalTraceUnavailable()
        self._verified_bundle = bundle
        self._verified_provenance = provenance

    @property
    def trace_call_id(self) -> str:
        return self.identity.call_id

    def reserve(self) -> _ProvisionalCallIdentity:
        if self._reserved:
            raise ProvisionalTraceUnavailable()
        self._reserved = True
        return self.identity

    def bind_request(
        self,
        bundle: object,
        provenance: object,
    ) -> None:
        if (
            not self._reserved
            or self._bound_bundle is not None
            or bundle is not self._verified_bundle
            or provenance is not self._verified_provenance
            or bundle is None
        ):
            raise ProvisionalTraceUnavailable()
        self._bound_bundle, self._verified_bundle = self._verified_bundle, None
        self._bound_provenance, self._verified_provenance = (
            self._verified_provenance,
            None,
        )

    def mark_dispatch_started(
        self,
        bundle: object | None = None,
        provenance: object | None = None,
    ) -> _ProvisionalCallIdentity:
        if bundle is not None:
            self.bind_request(bundle, provenance)
        if self._bound_bundle is None or self._dispatch_started_at is not None:
            raise ProvisionalTraceUnavailable()
        self._dispatch_started_at = _utc_now()
        return self.identity

    def mark_response_started(self) -> _ProvisionalCallIdentity:
        if self._dispatch_started_at is None:
            raise ProvisionalTraceUnavailable()
        if self._response_started_at is None:
            self._response_started_at = _utc_now()
        return self.identity

    def settle_response(
        self,
        response_envelope: object | None = None,
        outcome: TraceCallState = TraceCallState.COMPLETE,
        usage: object | None = None,
        *,
        provider_response: object | None = None,
        assistant_message_id: str | None = None,
        response_incomplete_reason: str | None = None,
        tool_events: tuple[tuple[str, object], ...] = (),
        known_credentials: tuple[str, ...] = (),
    ) -> ProvisionalTraceEnvelope | None:
        del (
            response_envelope,
            assistant_message_id,
            known_credentials,
            provider_response,
            response_incomplete_reason,
            tool_events,
        )
        if self._dispatch_started_at is None:
            raise ProvisionalTraceUnavailable()
        if self._bound_bundle is None or not self._bound_bundle.available:
            # An authenticated unavailable verification admits the spoken reply,
            # never content-bearing trace. The incomplete attempt cannot seal.
            return None
        call = self._build_call(outcome=outcome, usage=usage, settled_at=_utc_now())
        return self._retain_call(self._attempt, call)

    def _build_call(
        self,
        *,
        outcome: TraceCallState,
        usage: object | None,
        settled_at: str,
    ) -> PostDispatchTraceCall:
        from tldw_chatbook.Chat.console_trace_repository import _json_object
        from tldw_chatbook.Chat.console_trace_provenance import (
            RequestRouteTraceProvenance,
        )

        bundle = self._bound_bundle
        provenance = self._bound_provenance
        if bundle is None or provenance is None:
            raise ProvisionalTraceUnavailable()
        components = {item.name: item for item in bundle.components}
        provider = (
            _thaw(components.get("api_endpoint").value)
            if components.get("api_endpoint")
            else None
        )
        model = (
            _thaw(components.get("model").value) if components.get("model") else None
        )
        if isinstance(usage, Mapping):
            from tldw_chatbook.Chat.provider_usage import ProviderUsage

            usage = ProviderUsage.from_provider_payload(
                usage, provider=str(provider or ""), model=str(model or "")
            )
        routes = tuple(
            item
            for item in provenance.metadata
            if type(item) is RequestRouteTraceProvenance
        )
        route_identity = routes[0].route.value if len(routes) == 1 else "unavailable"
        endpoint = bundle.endpoint_identity or "unavailable"
        surface = self._request_surface(bundle, provenance)
        header_components, system_composition = self._header_components(
            bundle,
            provenance,
        )
        generation = {
            key: _thaw(item.value)
            for key, item in components.items()
            if key in _GENERATION_KEYS
        }
        reasoning = {
            key: _thaw(components[key].value)
            for key in sorted(_REASONING_KEYS)
            if key in components
        }
        response_format = (
            _thaw(components["response_format"].value)
            if "response_format" in components
            else {}
        )
        if not isinstance(response_format, Mapping):
            response_format = {"value": response_format}
        adapter_defaults: dict[str, object] = {
            "credential_source": bundle.credential_source.value,
            "handler_projection": [
                {"name": item.name, "redacted": item.redacted}
                for item in bundle.handler_components
            ],
            "normalization_version": "canonical-json-v1",
            "parameter_sources": {
                item.kind.split(":", 1)[1]: item.source
                for item in bundle.overlays
                if item.kind.startswith("parameter:")
            },
            "provider_overlays": [
                {"kind": item.kind, "source": item.source}
                for item in bundle.overlays
                if not item.kind.startswith("parameter:")
            ],
        }
        if system_composition:
            adapter_defaults["system_composition"] = [
                item.as_json_object() for item in system_composition
            ]
        to_json = getattr(usage, "to_json", None)
        usage_json = (
            None
            if not callable(to_json)
            else _json_object(json.loads(to_json()), "usage_json")
        )
        response = PostDispatchTraceResponse.no_response(
            _VOICE_ASSISTANT_REVISION_PENDING
            if outcome is TraceCallState.COMPLETE
            else "provider_error_no_response"
        )
        json_values = (
            _json_object(generation, "generation_parameters"),
            _json_object(adapter_defaults, "adapter_defaults"),
            _json_object(response_format, "response_format"),
            _json_object(reasoning, "reasoning_controls"),
        )
        artifacts = {
            item.artifact_value.artifact_id: item.artifact_value
            for item in (*surface, *header_components)
            if item.artifact_value is not None
        }
        sealed_payload_bytes = (
            sum(len(value.encode("utf-8")) for value in json_values)
            + (0 if usage_json is None else len(usage_json.encode("utf-8")))
            + sum(item.retained_bytes for item in artifacts.values())
        )
        return PostDispatchTraceCall(
            call_id=self.identity.call_id,
            idempotency_key=self.identity.idempotency_key,
            call_sequence=self.identity.call_sequence,
            provider_name=provider
            if isinstance(provider, str) and provider
            else "unavailable",
            model_name=model if isinstance(model, str) and model else "unavailable",
            route_identity=route_identity,
            endpoint_identity=endpoint,
            generation_parameters_json=json_values[0],
            adapter_defaults_json=json_values[1],
            response_format_json=json_values[2],
            reasoning_controls_json=json_values[3],
            dispatch_started_at=cast(str, self._dispatch_started_at),
            response_started_at=self._response_started_at,
            settled_at=settled_at,
            usage_json=usage_json,
            header_components=header_components,
            system_composition=system_composition,
            request_surface=surface,
            response=response,
            sealed_payload_bytes=sealed_payload_bytes,
            terminal_state=outcome,
        )

    def _request_surface(
        self,
        bundle: object,
        provenance: object,
    ) -> tuple[PostDispatchTraceSurfaceComponent, ...]:
        from tldw_chatbook.Chat.console_trace_provenance import (
            OmittedTraceProvenance,
            SavedRevisionTraceProvenance,
        )

        messages = _binding(bundle, "messages_payload")
        values = (
            tuple(messages.value)
            if messages is not None and isinstance(messages.value, tuple)
            else ()
        )
        if len(values) != len(provenance.messages_payload) or not values:
            raise ProvisionalTraceUnavailable()
        result: list[PostDispatchTraceSurfaceComponent] = []
        final_ordinal = len(values) - 1
        for ordinal, (descriptor, value) in enumerate(
            zip(provenance.messages_payload, values, strict=True)
        ):
            node_id = derive_post_dispatch_trace_node_id(
                self._attempt.promotion_id,
                self.identity.call_sequence,
                ordinal,
            )
            if ordinal == final_ordinal:
                component = PostDispatchTraceSurfaceComponent.omission(
                    node_id=node_id,
                    component_kind="message",
                    reason_code=_VOICE_USER_REVISION_PENDING,
                )
            elif type(descriptor) is SavedRevisionTraceProvenance:
                component = PostDispatchTraceSurfaceComponent.revision(
                    node_id=node_id,
                    component_kind="message",
                    revision_id=cast(
                        SavedRevisionTraceProvenance, descriptor
                    ).revision_id,
                )
            elif type(descriptor) is OmittedTraceProvenance:
                component = PostDispatchTraceSurfaceComponent.omission(
                    node_id=node_id,
                    component_kind="provider_message",
                    reason_code=cast(OmittedTraceProvenance, descriptor).reason.value,
                )
            else:
                component = PostDispatchTraceSurfaceComponent.artifact(
                    node_id=node_id,
                    component_kind="provider_message",
                    artifact=_artifact(
                        promotion_id=self._attempt.promotion_id,
                        call_sequence=self.identity.call_sequence,
                        label=f"message:{ordinal}",
                        value=value,
                        policy=self._policy,
                    ),
                )
            result.append(component)
        return tuple(result)

    def _header_components(
        self,
        bundle: object,
        provenance: object,
    ) -> tuple[
        tuple[PostDispatchTraceHeaderComponent, ...],
        tuple[PostDispatchTraceSystemComponent, ...],
    ]:
        components: list[PostDispatchTraceHeaderComponent] = []
        composition: list[PostDispatchTraceSystemComponent] = []
        system = _binding(bundle, "system_message")
        if system is not None and provenance.system_message is not None:
            system_values = bundle.system_components or (system.value,)
            for ordinal, value in enumerate(system_values):
                artifact = _artifact(
                    promotion_id=self._attempt.promotion_id,
                    call_sequence=self.identity.call_sequence,
                    label=f"system:{ordinal}",
                    value=value,
                    policy=self._policy,
                )
                components.append(
                    PostDispatchTraceHeaderComponent(
                        "rendered_system_part", ordinal, artifact
                    )
                )
                composition.append(PostDispatchTraceSystemComponent.artifact(ordinal))
        tools = _binding(bundle, "tools")
        tool_values = (
            tuple(tools.value)
            if tools is not None and isinstance(tools.value, tuple)
            else ()
        )
        for ordinal, value in enumerate(tool_values):
            components.append(
                PostDispatchTraceHeaderComponent(
                    "tool_schema",
                    ordinal,
                    _artifact(
                        promotion_id=self._attempt.promotion_id,
                        call_sequence=self.identity.call_sequence,
                        label=f"tool:{ordinal}",
                        value=value,
                        policy=self._policy,
                    ),
                )
            )
        return tuple(
            sorted(components, key=lambda item: (item.component_kind, item.ordinal))
        ), tuple(composition)


def _call_artifacts(
    call: PostDispatchTraceCall,
) -> dict[str, PostDispatchTraceArtifact]:
    artifacts: dict[str, PostDispatchTraceArtifact] = {}
    response = call.response
    candidates = (
        *(component.artifact_value for component in call.request_surface),
        *(component.artifact_value for component in call.header_components),
        None if response is None else response.artifact_value,
    )
    for artifact in candidates:
        if artifact is None:
            continue
        existing = artifacts.get(artifact.artifact_id)
        if existing is not None and existing != artifact:
            raise ProvisionalTraceUnavailable()
        artifacts[artifact.artifact_id] = artifact
    return artifacts


def _inline_payload_bytes(call: PostDispatchTraceCall) -> int:
    return sum(
        len(value.encode("utf-8"))
        for value in (
            call.generation_parameters_json,
            call.adapter_defaults_json,
            call.response_format_json,
            call.reasoning_controls_json,
            *((call.usage_json,) if call.usage_json is not None else ()),
        )
    )


def _canonicalize_call(
    call: PostDispatchTraceCall,
    artifacts: dict[str, PostDispatchTraceArtifact],
) -> PostDispatchTraceCall:
    request_surface = tuple(
        replace(
            component,
            artifact_value=artifacts[component.artifact_value.artifact_id],
        )
        if component.artifact_value is not None
        else component
        for component in call.request_surface
    )
    header_components = tuple(
        replace(
            component,
            artifact_value=artifacts[component.artifact_value.artifact_id],
        )
        for component in call.header_components
    )
    response = call.response
    if response is not None and response.artifact_value is not None:
        response = replace(
            response,
            artifact_value=artifacts[response.artifact_value.artifact_id],
        )
    return replace(
        call,
        request_surface=request_surface,
        header_components=header_components,
        response=response,
    )


def _materialize_committed_pair(
    calls: tuple[PostDispatchTraceCall, ...],
    context: VoiceTraceImportContext,
) -> tuple[PostDispatchTraceCall, ...]:
    """Resolve only the gateway-sealed winning-pair placeholders."""

    pending_users = tuple(
        tuple(
            index
            for index, component in enumerate(call.request_surface)
            if component.reference_kind == "omission"
            and component.omission_reason_code == _VOICE_USER_REVISION_PENDING
        )
        for call in calls
    )
    pending_assistants = tuple(
        index
        for index, call in enumerate(calls)
        if call.response is not None
        and call.response.kind == "no_response"
        and call.response.omission_reason_code == _VOICE_ASSISTANT_REVISION_PENDING
    )
    if not any(pending_users) and not pending_assistants:
        return calls
    final_index = len(calls) - 1
    if (
        any(len(pending) != 1 for pending in pending_users)
        or pending_assistants != (final_index,)
        or calls[final_index].terminal_state is not TraceCallState.COMPLETE
    ):
        raise ProvisionalTraceUnavailable()

    materialized = []
    for call_index, (call, pending) in enumerate(zip(calls, pending_users)):
        surface = list(call.request_surface)
        placeholder = surface[pending[0]]
        surface[pending[0]] = PostDispatchTraceSurfaceComponent.revision(
            node_id=placeholder.node_id,
            component_kind=placeholder.component_kind,
            revision_id=context.user_revision_id,
        )
        materialized.append(
            replace(
                call,
                request_surface=tuple(surface),
                response=(
                    PostDispatchTraceResponse.committed_revision(
                        context.assistant_revision_id
                    )
                    if call_index == final_index
                    else call.response
                ),
            )
        )
    return tuple(materialized)


_ResultT = TypeVar("_ResultT")


class ProvisionalTraceRegistry:
    """App-lifetime authority for bounded semantic payload retention."""

    __slots__ = (
        "_app_byte_limit",
        "_artifact_refcounts",
        "_artifacts",
        "_attempt_byte_limit",
        "_clock",
        "_ineligible",
        "_ineligible_overflow_until",
        "_issuer",
        "_lock",
        "_retained_bytes",
        "_states",
    )

    def __init__(
        self,
        *,
        attempt_byte_limit: int = MAX_PROMOTED_TRACE_BYTES,
        app_byte_limit: int = MAX_PROVISIONAL_TRACE_APP_BYTES,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if (
            type(attempt_byte_limit) is not int
            or type(app_byte_limit) is not int
            or not 1 <= attempt_byte_limit <= MAX_PROMOTED_TRACE_BYTES
            or not attempt_byte_limit
            <= app_byte_limit
            <= MAX_PROVISIONAL_TRACE_APP_BYTES
        ):
            raise ValueError("provisional trace budgets are invalid")
        self._attempt_byte_limit = attempt_byte_limit
        self._app_byte_limit = app_byte_limit
        self._clock = clock
        self._issuer = object()
        self._lock = RLock()
        self._states: dict[str, _AttemptState] = {}
        self._ineligible: dict[tuple[str, str], float] = {}
        self._ineligible_overflow_until: float | None = None
        self._artifacts: dict[str, PostDispatchTraceArtifact] = {}
        self._artifact_refcounts: dict[str, int] = {}
        self._retained_bytes = 0

    def __repr__(self) -> str:
        return (
            "ProvisionalTraceRegistry("
            f"retained_bytes={self.retained_bytes}, attempts={self.attempt_count})"
        )

    @property
    def retained_bytes(self) -> int:
        with self._lock:
            return self._retained_bytes

    @property
    def attempt_count(self) -> int:
        with self._lock:
            return len(self._states)

    def begin_attempt(
        self,
        *,
        promotion_id: str,
        attempt_id: str,
        eligibility: FrozenProvisionalCaptureEligibility,
        policy: FrozenTracePolicy,
    ) -> ProvisionalTraceAttempt | None:
        """Freeze eligibility and issue an authority only for a saved session."""

        _uuid(promotion_id, "promotion_id")
        _uuid(attempt_id, "attempt_id")
        if type(eligibility) is not FrozenProvisionalCaptureEligibility:
            raise TypeError("eligibility must be frozen at dispatch")
        if type(policy) is not FrozenTracePolicy:
            raise TypeError("policy must be frozen at dispatch")
        identity = (promotion_id, attempt_id)
        with self._lock:
            self._reap_expired_locked()
            now = self._clock()
            if self._ineligible_overflow_until is not None:
                if not eligibility.eligible:
                    self._ineligible_overflow_until = max(
                        self._ineligible_overflow_until,
                        now + PROVISIONAL_TRACE_TTL_SECONDS,
                    )
                return None
            if identity in self._ineligible:
                return None
            if not eligibility.eligible:
                if len(self._ineligible) >= _MAX_INELIGIBLE_TOMBSTONES:
                    self._ineligible_overflow_until = (
                        now + PROVISIONAL_TRACE_TTL_SECONDS
                    )
                else:
                    self._ineligible[identity] = now
                return None
            if any(
                state.capability.promotion_id == promotion_id
                and state.capability.attempt_id == attempt_id
                for state in self._states.values()
            ):
                raise ProvisionalTraceUnavailable()
            handle = str(uuid4())
            capability = ProvisionalTraceAttempt._issue(
                issuer=self._issuer,
                handle=handle,
                promotion_id=promotion_id,
                attempt_id=attempt_id,
            )
            self._states[handle] = _AttemptState(
                capability=capability,
                promotion_id=promotion_id,
                attempt_id=attempt_id,
                created_at=self._clock(),
                policy=policy,
            )
            return capability

    def _retain_gateway_call(
        self,
        attempt: ProvisionalTraceAttempt,
        call: PostDispatchTraceCall,
    ) -> ProvisionalTraceEnvelope | None:
        """Retain one exact sequential call and return only its opaque identity."""

        if type(call) is not PostDispatchTraceCall:
            raise TypeError("call must be a PostDispatchTraceCall")
        with self._lock:
            state = self._attempt_state_locked(attempt)
            if state.manifest is not None or state.claim is not None:
                self._destroy_locked(attempt._handle)
                raise ProvisionalTraceUnavailable()
            sequence = len(state.calls)
            if sequence >= MAX_PROMOTED_TRACE_CALLS:
                self._destroy_locked(attempt._handle)
                raise ProvisionalTraceUnavailable()
            expected_call_id = derive_post_dispatch_trace_ids(
                state.promotion_id,
                call_count=sequence + 1,
            ).call_ids[sequence]
            if call.call_sequence != sequence or call.call_id != expected_call_id:
                self._destroy_locked(attempt._handle)
                raise ProvisionalTraceUnavailable()
            try:
                artifacts = _call_artifacts(call)
            except ProvisionalTraceUnavailable:
                self._destroy_locked(attempt._handle)
                raise
            inline_bytes = _inline_payload_bytes(call)
            call_bytes = inline_bytes + sum(
                artifact.retained_bytes for artifact in artifacts.values()
            )
            if call.sealed_payload_bytes != call_bytes:
                self._destroy_locked(attempt._handle)
                raise ProvisionalTraceUnavailable()
            attempt_artifact_bytes = 0
            app_artifact_bytes = 0
            canonical_artifacts: dict[str, PostDispatchTraceArtifact] = {}
            for artifact_id, artifact in artifacts.items():
                state_artifact = state.artifacts.get(artifact_id)
                app_artifact = self._artifacts.get(artifact_id)
                if (
                    state_artifact is not None
                    and state_artifact != artifact
                    or app_artifact is not None
                    and app_artifact != artifact
                ):
                    self._destroy_locked(attempt._handle)
                    raise ProvisionalTraceUnavailable()
                canonical = state_artifact or app_artifact or artifact
                canonical_artifacts[artifact_id] = canonical
                if state_artifact is None:
                    attempt_artifact_bytes += canonical.retained_bytes
                    if app_artifact is None:
                        app_artifact_bytes += canonical.retained_bytes
            attempt_incremental = inline_bytes + attempt_artifact_bytes
            app_incremental = inline_bytes + app_artifact_bytes
            if (
                state.retained_bytes + attempt_incremental > self._attempt_byte_limit
                or self._retained_bytes + app_incremental > self._app_byte_limit
            ):
                self._destroy_locked(attempt._handle)
                raise ProvisionalTraceUnavailable()
            envelope = ProvisionalTraceEnvelope._issue(
                issuer=self._issuer,
                attempt_handle=attempt._handle,
                promotion_id=state.promotion_id,
                attempt_id=state.attempt_id,
                call_id=call.call_id,
                call_sequence=sequence,
            )
            canonical_call = _canonicalize_call(call, canonical_artifacts)
            state.calls.append(canonical_call)
            state.envelopes.append(envelope)
            state.envelope_ids.append(envelope.envelope_id)
            for artifact_id, artifact in canonical_artifacts.items():
                if artifact_id in state.artifacts:
                    continue
                if artifact_id not in self._artifacts:
                    self._artifacts[artifact_id] = artifact
                    self._artifact_refcounts[artifact_id] = 0
                state.artifacts[artifact_id] = self._artifacts[artifact_id]
                self._artifact_refcounts[artifact_id] += 1
            state.inline_payload_bytes += inline_bytes
            state.retained_bytes += attempt_incremental
            self._retained_bytes += app_incremental
            return envelope

    def _begin_gateway_call(
        self,
        attempt: ProvisionalTraceAttempt,
        retain_call: Callable[
            [ProvisionalTraceAttempt, PostDispatchTraceCall],
            ProvisionalTraceEnvelope | None,
        ],
    ) -> ProvisionalVoiceTraceCallBoundary:
        """Reserve one in-memory provider-call boundary without persistence."""

        with self._lock:
            state = self._attempt_state_locked(attempt)
            if state.manifest is not None or state.claim is not None:
                self._destroy_locked(attempt._handle)
                raise ProvisionalTraceUnavailable()
            sequence = state.reserved_call_count
            if sequence >= MAX_PROMOTED_TRACE_CALLS:
                self._destroy_locked(attempt._handle)
                raise ProvisionalTraceUnavailable()
            state.reserved_call_count += 1
            return ProvisionalVoiceTraceCallBoundary(
                attempt,
                sequence,
                retain_call,
                state.policy,
            )

    def seal_attempt(
        self,
        attempt: ProvisionalTraceAttempt,
        *,
        expected_call_count: int,
    ) -> ProvisionalTraceManifest:
        """Seal the exact complete contiguous call set once."""

        with self._lock:
            state = self._attempt_state_locked(attempt)
            if (
                type(expected_call_count) is not int
                or not 1 <= expected_call_count <= MAX_PROMOTED_TRACE_CALLS
                or len(state.calls) != expected_call_count
                or (
                    state.reserved_call_count
                    and len(state.calls) != state.reserved_call_count
                )
                or state.manifest is not None
            ):
                self._destroy_locked(attempt._handle)
                raise ProvisionalTraceUnavailable()
            chronology = tuple(
                ProvisionalTraceObservation(
                    call_id=call.call_id,
                    call_sequence=call.call_sequence,
                    dispatch_started_at=call.dispatch_started_at,
                    response_started_at=call.response_started_at,
                    settled_at=call.settled_at,
                )
                for call in state.calls
            )
            manifest = ProvisionalTraceManifest._issue(
                issuer=self._issuer,
                attempt_handle=attempt._handle,
                promotion_id=state.promotion_id,
                attempt_id=state.attempt_id,
                envelopes=tuple(state.envelopes),
                aggregate_payload_bytes=state.retained_bytes,
                chronology=chronology,
            )
            state.manifest = manifest
            state.manifest_id = manifest.manifest_id
            state.sealed_at = self._clock()
            return manifest

    def claim(
        self,
        manifest: ProvisionalTraceManifest,
        envelopes: tuple[ProvisionalTraceEnvelope, ...],
    ) -> _ProvisionalTraceClaim:
        """Claim one exact manifest/envelope set for a single redemption."""

        if type(envelopes) is not tuple:
            raise TypeError("envelopes must be a typed tuple")
        with self._lock:
            self._reap_expired_locked()
            state = self._manifest_state_locked(manifest)
            supplied_valid = len(envelopes) == len(state.envelopes)
            if supplied_valid:
                try:
                    supplied_valid = all(
                        supplied is issued
                        and supplied._issuer is self._issuer
                        and supplied._attempt_handle == manifest._attempt_handle
                        and supplied.envelope_id == state.envelope_ids[sequence]
                        and supplied.promotion_id == state.promotion_id
                        and supplied.attempt_id == state.attempt_id
                        and supplied.call_id == state.calls[sequence].call_id
                        and supplied.call_sequence == sequence
                        for sequence, (supplied, issued) in enumerate(
                            zip(envelopes, state.envelopes, strict=True)
                        )
                    )
                except (AttributeError, TypeError):
                    supplied_valid = False
            if state.claim is not None:
                raise ProvisionalTraceUnavailable()
            if not supplied_valid:
                self._destroy_locked(manifest._attempt_handle)
                raise ProvisionalTraceUnavailable()
            claim = _ProvisionalTraceClaim._issue(
                issuer=self._issuer,
                attempt_handle=manifest._attempt_handle,
            )
            state.claim = claim
            return claim

    def abandon_attempt(self, attempt: ProvisionalTraceAttempt) -> None:
        """Destroy a cancelled or losing attempt, sealed or unsealed."""

        with self._lock:
            state = self._attempt_state_locked(attempt)
            self._destroy_locked(state.capability._handle)

    def abandon_manifest(self, manifest: ProvisionalTraceManifest) -> None:
        """Destroy a sealed capability after explicit recovery abandonment."""

        with self._lock:
            state = self._manifest_state_locked(manifest)
            self._destroy_locked(state.capability._handle)

    def reap_expired(self) -> int:
        """Destroy every manifest whose ten-minute redemption window elapsed."""

        with self._lock:
            before = len(self._states)
            self._reap_expired_locked()
            return before - len(self._states)

    def _import_claim(
        self,
        claim: _ProvisionalTraceClaim,
        context: VoiceTraceImportContext,
        importer: Callable[[PostDispatchTraceImport], _ResultT],
    ) -> _ResultT:
        with self._lock:
            state = self._claim_state_locked(claim)
            if (
                context.import_id != state.promotion_id
                or context.policy != state.policy
            ):
                self._destroy_locked(claim._attempt_handle, claim=claim)
                raise ProvisionalTraceUnavailable()
            calls = _materialize_committed_pair(tuple(state.calls), context)
            request = PostDispatchTraceImport(
                import_id=context.import_id,
                conversation_id=context.conversation_id,
                user_message_id=context.user_message_id,
                user_revision_id=context.user_revision_id,
                assistant_message_id=context.assistant_message_id,
                assistant_revision_id=context.assistant_revision_id,
                turn_id=context.turn_id,
                run_id=context.run_id,
                policy=context.policy,
                expected_call_count=len(state.calls),
                calls=calls,
                aggregate_payload_bytes=state.retained_bytes,
            )
        return importer(request)

    def _release_claim(
        self,
        claim: _ProvisionalTraceClaim,
    ) -> _ClaimReleaseDisposition:
        with self._lock:
            state = self._claim_state_locked(claim)
            state.claim = None
            if self._state_expired_locked(state, self._clock()):
                self._destroy_locked(claim._attempt_handle)
                return _ClaimReleaseDisposition.EXPIRED_DESTROYED
            return _ClaimReleaseDisposition.RELEASED_RETRYABLE

    def _consume_claim(self, claim: _ProvisionalTraceClaim) -> None:
        with self._lock:
            state = self._claim_state_locked(claim)
            self._destroy_locked(state.capability._handle, claim=claim)

    def _abandon_claim(self, claim: _ProvisionalTraceClaim) -> None:
        self._consume_claim(claim)

    def _attempt_state_locked(
        self,
        attempt: ProvisionalTraceAttempt,
    ) -> _AttemptState:
        if (
            type(attempt) is not ProvisionalTraceAttempt
            or getattr(attempt, "_issuer", None) is not self._issuer
        ):
            raise ProvisionalTraceUnavailable()
        state = self._states.get(attempt._handle)
        if state is None or state.capability is not attempt:
            raise ProvisionalTraceUnavailable()
        if state.claim is None and self._state_expired_locked(state, self._clock()):
            self._destroy_locked(attempt._handle)
            raise ProvisionalTraceUnavailable()
        if (
            attempt.promotion_id != state.promotion_id
            or attempt.attempt_id != state.attempt_id
        ):
            self._destroy_locked(attempt._handle)
            raise ProvisionalTraceUnavailable()
        return state

    def _manifest_state_locked(
        self,
        manifest: ProvisionalTraceManifest,
    ) -> _AttemptState:
        if (
            type(manifest) is not ProvisionalTraceManifest
            or getattr(manifest, "_issuer", None) is not self._issuer
        ):
            raise ProvisionalTraceUnavailable()
        state = self._states.get(manifest._attempt_handle)
        if state is None or state.manifest is not manifest:
            raise ProvisionalTraceUnavailable()
        if state.claim is None and self._state_expired_locked(state, self._clock()):
            self._destroy_locked(manifest._attempt_handle)
            raise ProvisionalTraceUnavailable()
        chronology = tuple(
            ProvisionalTraceObservation(
                call_id=call.call_id,
                call_sequence=call.call_sequence,
                dispatch_started_at=call.dispatch_started_at,
                response_started_at=call.response_started_at,
                settled_at=call.settled_at,
            )
            for call in state.calls
        )
        if (
            manifest.manifest_id != state.manifest_id
            or manifest.promotion_id != state.promotion_id
            or manifest.attempt_id != state.attempt_id
            or manifest.envelope_ids != tuple(state.envelope_ids)
            or manifest.expected_call_count != len(state.calls)
            or manifest.aggregate_payload_bytes != state.retained_bytes
            or manifest.observed_chronology != chronology
        ):
            self._destroy_locked(manifest._attempt_handle)
            raise ProvisionalTraceUnavailable()
        return state

    def _claim_state_locked(self, claim: _ProvisionalTraceClaim) -> _AttemptState:
        if (
            type(claim) is not _ProvisionalTraceClaim
            or getattr(claim, "_issuer", None) is not self._issuer
        ):
            raise ProvisionalTraceUnavailable()
        state = self._states.get(claim._attempt_handle)
        if state is None or state.claim is not claim:
            raise ProvisionalTraceUnavailable()
        return state

    def _destroy_locked(
        self,
        handle: str,
        *,
        claim: _ProvisionalTraceClaim | None = None,
    ) -> None:
        state = self._states.get(handle)
        if state is None:
            return
        if state.claim is not None and state.claim is not claim:
            raise ProvisionalTraceUnavailable()
        self._states.pop(handle)
        self._retained_bytes -= state.inline_payload_bytes
        for artifact_id, artifact in state.artifacts.items():
            remaining = self._artifact_refcounts[artifact_id] - 1
            if remaining:
                self._artifact_refcounts[artifact_id] = remaining
                continue
            self._artifact_refcounts.pop(artifact_id)
            self._artifacts.pop(artifact_id)
            self._retained_bytes -= artifact.retained_bytes
        state.calls.clear()
        state.envelopes.clear()
        state.envelope_ids.clear()
        state.artifacts.clear()
        state.retained_bytes = 0
        state.inline_payload_bytes = 0
        state.manifest = None
        state.claim = None
        state.reserved_call_count = 0

    def _reap_expired_locked(self) -> None:
        now = self._clock()
        if (
            self._ineligible_overflow_until is not None
            and now > self._ineligible_overflow_until
        ):
            self._ineligible_overflow_until = None
        expired_ineligible = tuple(
            identity
            for identity, created_at in self._ineligible.items()
            if now - created_at > PROVISIONAL_TRACE_TTL_SECONDS
        )
        for identity in expired_ineligible:
            self._ineligible.pop(identity, None)
        expired = tuple(
            handle
            for handle, state in self._states.items()
            if state.claim is None and self._state_expired_locked(state, now)
        )
        for handle in expired:
            self._destroy_locked(handle)

    @staticmethod
    def _state_expired_locked(state: _AttemptState, now: float) -> bool:
        lifetime_started_at = (
            state.created_at if state.sealed_at is None else state.sealed_at
        )
        return now - lifetime_started_at > PROVISIONAL_TRACE_TTL_SECONDS


__all__ = [
    "MAX_PROVISIONAL_TRACE_APP_BYTES",
    "PROVISIONAL_TRACE_TTL_SECONDS",
    "ProvisionalTraceEnvelope",
    "ProvisionalTraceManifest",
    "ProvisionalTraceRegistry",
    "ProvisionalTraceUnavailable",
    "VoiceTraceImportContext",
]
