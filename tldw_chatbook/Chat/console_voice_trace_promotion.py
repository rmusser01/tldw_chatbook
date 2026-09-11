"""Immutable values for exceptional post-dispatch voice trace imports.

The provider gateway owns semantic payload retention and capability redemption.
This module defines only the sealed, already-observed values accepted by the
repository after a completed conversation pair has committed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
import re
from typing import Literal
from uuid import UUID, uuid5
from tldw_chatbook.Chat.console_trace_redaction import (
    PIIFieldRedaction,
    PIIRedactionSpan,
)

from tldw_chatbook.Chat.console_trace_models import (
    MAX_SURFACE_REPLACEMENT_SPAN,
    FrozenTracePolicy,
    TraceCallState,
)


_POST_DISPATCH_TRACE_NAMESPACE = UUID("31be366d-c917-48f1-b9a2-b9e347cfbe1d")
MAX_PROMOTED_TRACE_CALLS = 8
_MAX_JSON_LENGTH = 1_000_000
MAX_PROMOTED_TRACE_BYTES = 64 * 1024 * 1024
_TOKEN = re.compile(r"[a-z][a-z0-9]*(?:[_-][a-z0-9]+)*\Z", re.ASCII)


class ConfirmedPreCommitTraceImportError(RuntimeError):
    """Repository proof that a promoted trace transaction rolled back."""

    def __init__(self) -> None:
        super().__init__("Promoted trace import did not commit.")


def _stable_id(value: object, name: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{name} must be a canonical UUID string")
    try:
        parsed = UUID(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a canonical UUID string") from exc
    if parsed.version not in {4, 5} or str(parsed) != value:
        raise ValueError(f"{name} must be a canonical UUID string")
    return value


def _required(value: object, name: str, *, maximum: int = 512) -> str:
    if type(value) is not str or not value or len(value) > maximum:
        raise ValueError(f"{name} must be a bounded non-empty string")
    return value


def _canonical_json_object(value: object, name: str) -> str:
    encoded = _required(value, name, maximum=_MAX_JSON_LENGTH)
    try:
        decoded = json.loads(encoded)
        canonical = json.dumps(
            decoded,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError(f"{name} must be canonical JSON") from exc
    if type(decoded) is not dict or canonical != encoded:
        raise ValueError(f"{name} must be a canonical JSON object")
    return encoded


def _observed_timestamp(value: object, name: str) -> datetime:
    encoded = _required(value, name, maximum=64)
    if not encoded.endswith("Z"):
        raise ValueError(f"{name} must be an observed UTC timestamp")
    try:
        parsed = datetime.fromisoformat(encoded[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{name} must be an observed UTC timestamp") from exc
    return parsed


def _token(value: object, name: str) -> str:
    encoded = _required(value, name, maximum=64)
    if _TOKEN.fullmatch(encoded) is None:
        raise ValueError(f"{name} must be a lowercase identifier token")
    return encoded


@dataclass(frozen=True, slots=True)
class PostDispatchTraceArtifact:
    """One exact sanitized provider-only artifact retained by the gateway."""

    artifact_id: str
    media_type: str
    normalization_version: str
    sanitized_bytes: bytes = field(repr=False)
    field_redactions: tuple[PIIFieldRedaction, ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        _stable_id(self.artifact_id, "artifact_id")
        _required(self.media_type, "media_type")
        _required(self.normalization_version, "normalization_version")
        if type(self.sanitized_bytes) is not bytes:
            raise TypeError("sanitized_bytes must be bytes")
        if len(self.sanitized_bytes) > MAX_PROMOTED_TRACE_BYTES:
            raise ValueError("sanitized artifact exceeds the attempt byte bound")
        if (
            type(self.field_redactions) is not tuple
            or len(self.field_redactions) > 4096
        ):
            raise ValueError(
                "artifact redaction metadata must be bounded and immutable"
            )
        for item in self.field_redactions:
            if (
                type(item) is not PIIFieldRedaction
                or type(item.span) is not PIIRedactionSpan
                or type(item.field_path) is not str
                or not 0 < len(item.field_path) <= 512
            ):
                raise ValueError("artifact redaction metadata")
        if self.retained_bytes > MAX_PROMOTED_TRACE_BYTES:
            raise ValueError("artifact and masks exceed the attempt byte bound")

    @property
    def mask_metadata_bytes(self) -> bytes:
        """Canonical content-free metadata included in capability accounting."""
        return (
            json.dumps(
                [
                    [
                        item.field_path,
                        item.span.start_codepoint,
                        item.span.end_codepoint,
                        item.span.category,
                        item.span.rule_id,
                        item.span.detector_version,
                    ]
                    for item in self.field_redactions
                ],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            if self.field_redactions
            else b""
        )

    @property
    def retained_bytes(self) -> int:
        return len(self.sanitized_bytes) + len(self.mask_metadata_bytes)

    @property
    def identity_digest(self) -> str:
        """Return the content-addressed identity used by the normalized ledger."""

        return hashlib.sha256(self.sanitized_bytes).hexdigest()


@dataclass(frozen=True, slots=True)
class PostDispatchTraceHeaderComponent:
    """One artifact-backed row in ``console_trace_header_components``."""

    component_kind: str
    ordinal: int
    artifact_value: PostDispatchTraceArtifact = field(repr=False)

    def __post_init__(self) -> None:
        _token(self.component_kind, "component_kind")
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("header component ordinal must be non-negative")
        if type(self.artifact_value) is not PostDispatchTraceArtifact:
            raise TypeError("header component artifact must be typed")


@dataclass(frozen=True, slots=True)
class PostDispatchTraceHeaderOmission:
    """One exact omission retained in a request header's structural JSON."""

    component_kind: str
    ordinal: int | None
    reason_code: str

    def __post_init__(self) -> None:
        _token(self.component_kind, "component_kind")
        if self.ordinal is not None and (
            type(self.ordinal) is not int or self.ordinal < 0
        ):
            raise ValueError("header omission ordinal must be non-negative")
        _token(self.reason_code, "reason_code")

    @property
    def json_key(self) -> str:
        """Return the canonical ``header_omissions`` object key."""

        if self.ordinal is None:
            return self.component_kind
        return f"{self.component_kind}:{self.ordinal}"


@dataclass(frozen=True, slots=True)
class PostDispatchTraceSystemComponent:
    """One ordered token from ADR-097's system-composition header model."""

    reference_kind: Literal[
        "transform_start", "transform_end", "revision", "artifact", "omission"
    ]
    transform_name: str | None = None
    revision_id: str | None = None
    artifact_component_ordinal: int | None = None
    omission_source: str | None = None
    omission_reason_code: str | None = None

    def __post_init__(self) -> None:
        populated = (
            self.transform_name,
            self.revision_id,
            self.artifact_component_ordinal,
            self.omission_source,
            self.omission_reason_code,
        )
        if self.reference_kind in {"transform_start", "transform_end"}:
            if self.transform_name is None or any(
                value is not None for value in populated[1:]
            ):
                raise ValueError("system transform component shape is invalid")
            _token(self.transform_name, "transform_name")
        elif self.reference_kind == "revision":
            if self.revision_id is None or any(
                value is not None for value in (populated[0], *populated[2:])
            ):
                raise ValueError("system revision component shape is invalid")
            _stable_id(self.revision_id, "revision_id")
        elif self.reference_kind == "artifact":
            if (
                type(self.artifact_component_ordinal) is not int
                or self.artifact_component_ordinal < 0
                or any(
                    value is not None
                    for value in (
                        self.transform_name,
                        self.revision_id,
                        self.omission_source,
                        self.omission_reason_code,
                    )
                )
            ):
                raise ValueError("system artifact component shape is invalid")
        elif self.reference_kind == "omission":
            if (
                self.omission_source is None
                or self.omission_reason_code is None
                or any(value is not None for value in populated[:3])
            ):
                raise ValueError("system omission component shape is invalid")
            _token(self.omission_source, "omission_source")
            _token(self.omission_reason_code, "omission_reason_code")
        else:
            raise ValueError("system component reference kind is invalid")

    @classmethod
    def transform_start(cls, transform: str) -> "PostDispatchTraceSystemComponent":
        return cls("transform_start", transform_name=transform)

    @classmethod
    def transform_end(cls, transform: str) -> "PostDispatchTraceSystemComponent":
        return cls("transform_end", transform_name=transform)

    @classmethod
    def revision(cls, revision_id: str) -> "PostDispatchTraceSystemComponent":
        return cls("revision", revision_id=revision_id)

    @classmethod
    def artifact(cls, component_ordinal: int) -> "PostDispatchTraceSystemComponent":
        return cls("artifact", artifact_component_ordinal=component_ordinal)

    @classmethod
    def omission(
        cls,
        *,
        source: str,
        reason_code: str,
    ) -> "PostDispatchTraceSystemComponent":
        return cls(
            "omission",
            omission_source=source,
            omission_reason_code=reason_code,
        )

    def as_json_object(self) -> dict[str, object]:
        """Return the exact existing ADR-097 JSON token shape."""

        if self.reference_kind in {"transform_start", "transform_end"}:
            return {"kind": self.reference_kind, "transform": self.transform_name}
        if self.reference_kind == "revision":
            return {"kind": "revision", "revision_id": self.revision_id}
        if self.reference_kind == "artifact":
            return {
                "kind": "artifact",
                "component_ordinal": self.artifact_component_ordinal,
            }
        return {
            "kind": "omission",
            "source": self.omission_source,
            "reason": self.omission_reason_code,
        }


@dataclass(frozen=True, slots=True)
class PostDispatchTraceSurfaceComponent:
    """One exact ordered request-surface component for a sealed call."""

    node_id: str
    component_kind: str
    reference_kind: Literal["revision", "artifact", "omission"]
    revision_id: str | None = None
    artifact_value: PostDispatchTraceArtifact | None = field(default=None, repr=False)
    omission_reason_code: str | None = None

    def __post_init__(self) -> None:
        _stable_id(self.node_id, "node_id")
        _token(self.component_kind, "component_kind")
        if self.reference_kind == "revision":
            if (
                self.revision_id is None
                or self.artifact_value is not None
                or self.omission_reason_code is not None
            ):
                raise ValueError("revision surface component shape is invalid")
            _stable_id(self.revision_id, "revision_id")
        elif self.reference_kind == "artifact":
            if (
                self.revision_id is not None
                or type(self.artifact_value) is not PostDispatchTraceArtifact
                or self.omission_reason_code is not None
            ):
                raise ValueError("artifact surface component shape is invalid")
        elif self.reference_kind == "omission":
            if (
                self.revision_id is not None
                or self.artifact_value is not None
                or self.omission_reason_code is None
            ):
                raise ValueError("omission surface component shape is invalid")
            _token(self.omission_reason_code, "omission_reason_code")
        else:
            raise ValueError("reference_kind is invalid")

    @classmethod
    def revision(
        cls, *, node_id: str, component_kind: str, revision_id: str
    ) -> "PostDispatchTraceSurfaceComponent":
        return cls(node_id, component_kind, "revision", revision_id=revision_id)

    @classmethod
    def artifact(
        cls,
        *,
        node_id: str,
        component_kind: str,
        artifact: PostDispatchTraceArtifact,
    ) -> "PostDispatchTraceSurfaceComponent":
        return cls(
            node_id,
            component_kind,
            "artifact",
            artifact_value=artifact,
        )

    @classmethod
    def omission(
        cls,
        *,
        node_id: str,
        component_kind: str,
        reason_code: str,
    ) -> "PostDispatchTraceSurfaceComponent":
        return cls(
            node_id,
            component_kind,
            "omission",
            omission_reason_code=reason_code,
        )


@dataclass(frozen=True, slots=True)
class PostDispatchTraceResponse:
    """Exact response disposition for one terminal promoted call."""

    kind: Literal["committed_revision", "artifact", "no_response"]
    committed_revision_id: str | None = None
    artifact_value: PostDispatchTraceArtifact | None = field(default=None, repr=False)
    omission_reason_code: str | None = None

    def __post_init__(self) -> None:
        if self.kind == "committed_revision":
            if (
                self.committed_revision_id is None
                or self.artifact_value is not None
                or self.omission_reason_code is not None
            ):
                raise ValueError("committed response revision shape is invalid")
            _stable_id(self.committed_revision_id, "committed_revision_id")
        elif self.kind == "artifact":
            if (
                self.committed_revision_id is not None
                or type(self.artifact_value) is not PostDispatchTraceArtifact
                or self.omission_reason_code is not None
            ):
                raise ValueError("artifact response shape is invalid")
        elif self.kind == "no_response":
            if (
                self.committed_revision_id is not None
                or self.artifact_value is not None
                or self.omission_reason_code is None
            ):
                raise ValueError("no-response disposition shape is invalid")
            _token(self.omission_reason_code, "omission_reason_code")
        else:
            raise ValueError("response disposition is invalid")

    @classmethod
    def committed_revision(cls, revision_id: str) -> "PostDispatchTraceResponse":
        return cls("committed_revision", committed_revision_id=revision_id)

    @classmethod
    def artifact(
        cls, artifact: PostDispatchTraceArtifact
    ) -> "PostDispatchTraceResponse":
        return cls("artifact", artifact_value=artifact)

    @classmethod
    def no_response(cls, reason_code: str) -> "PostDispatchTraceResponse":
        return cls("no_response", omission_reason_code=reason_code)


@dataclass(frozen=True, slots=True)
class PostDispatchTraceCall:
    """One terminal provider call observed and sealed by the gateway."""

    call_id: str
    idempotency_key: str
    call_sequence: int
    provider_name: str
    model_name: str
    route_identity: str
    endpoint_identity: str
    generation_parameters_json: str = field(repr=False)
    adapter_defaults_json: str = field(repr=False)
    response_format_json: str = field(repr=False)
    reasoning_controls_json: str = field(repr=False)
    dispatch_started_at: str
    response_started_at: str | None
    settled_at: str
    usage_json: str | None = field(default=None, repr=False)
    header_components: tuple[PostDispatchTraceHeaderComponent, ...] = field(
        default=(), repr=False
    )
    system_composition: tuple[PostDispatchTraceSystemComponent, ...] = field(
        default=(), repr=False
    )
    header_omissions: tuple[PostDispatchTraceHeaderOmission, ...] = field(
        default=(), repr=False
    )
    request_surface: tuple[PostDispatchTraceSurfaceComponent, ...] = field(
        default=(), repr=False
    )
    response: PostDispatchTraceResponse | None = field(default=None, repr=False)
    sealed_payload_bytes: int = 0
    terminal_state: TraceCallState = TraceCallState.COMPLETE

    def __post_init__(self) -> None:
        _stable_id(self.call_id, "call_id")
        _required(self.idempotency_key, "idempotency_key")
        if type(self.call_sequence) is not int or self.call_sequence < 0:
            raise ValueError("call_sequence must be a non-negative integer")
        for name in (
            "provider_name",
            "model_name",
            "route_identity",
            "endpoint_identity",
        ):
            _required(getattr(self, name), name)
        for name in (
            "generation_parameters_json",
            "adapter_defaults_json",
            "response_format_json",
            "reasoning_controls_json",
        ):
            _canonical_json_object(getattr(self, name), name)
        if type(self.header_components) is not tuple or any(
            type(component) is not PostDispatchTraceHeaderComponent
            for component in self.header_components
        ):
            raise TypeError("header_components must be a typed tuple")
        header_component_keys = tuple(
            (component.component_kind, component.ordinal)
            for component in self.header_components
        )
        if header_component_keys != tuple(sorted(header_component_keys)) or len(
            set(header_component_keys)
        ) != len(header_component_keys):
            raise ValueError("header_components must be unique and canonically ordered")
        if type(self.system_composition) is not tuple or any(
            type(component) is not PostDispatchTraceSystemComponent
            for component in self.system_composition
        ):
            raise TypeError("system_composition must be a typed tuple")
        if len(self.system_composition) > MAX_SURFACE_REPLACEMENT_SPAN:
            raise ValueError("system_composition exceeds the bounded span")
        transforms: list[str] = []
        for component in self.system_composition:
            if component.reference_kind == "transform_start":
                assert component.transform_name is not None
                transforms.append(component.transform_name)
            elif component.reference_kind == "transform_end":
                if not transforms or transforms.pop() != component.transform_name:
                    raise ValueError("system_composition transforms are unbalanced")
        if transforms:
            raise ValueError("system_composition transforms are unbalanced")
        if type(self.header_omissions) is not tuple or any(
            type(omission) is not PostDispatchTraceHeaderOmission
            for omission in self.header_omissions
        ):
            raise TypeError("header_omissions must be a typed tuple")
        omission_keys = tuple(omission.json_key for omission in self.header_omissions)
        if omission_keys != tuple(sorted(omission_keys)) or len(
            set(omission_keys)
        ) != len(omission_keys):
            raise ValueError("header_omissions must be unique and canonically ordered")
        if set(header_component_keys) & {
            (omission.component_kind, omission.ordinal)
            for omission in self.header_omissions
            if omission.ordinal is not None
        }:
            raise ValueError("header component and omission keys collide")
        adapter_defaults = json.loads(self.adapter_defaults_json)
        expected_system_composition = [
            component.as_json_object() for component in self.system_composition
        ]
        if adapter_defaults.get("system_composition") != (
            expected_system_composition or None
        ) or (
            not expected_system_composition and "system_composition" in adapter_defaults
        ):
            raise ValueError("typed system composition does not match header JSON")
        expected_omissions = {
            omission.json_key: omission.reason_code
            for omission in self.header_omissions
        }
        if adapter_defaults.get("header_omissions") != (expected_omissions or None) or (
            not expected_omissions and "header_omissions" in adapter_defaults
        ):
            raise ValueError("typed header omissions do not match header JSON")
        system_artifact_ordinals = tuple(
            component.artifact_component_ordinal
            for component in self.system_composition
            if component.reference_kind == "artifact"
        )
        rendered_system_ordinals = tuple(
            component.ordinal
            for component in self.header_components
            if component.component_kind == "rendered_system_part"
        )
        if (
            system_artifact_ordinals != rendered_system_ordinals
            or system_artifact_ordinals != tuple(range(len(system_artifact_ordinals)))
        ):
            raise ValueError("typed system artifacts do not match header components")
        dispatch = _observed_timestamp(self.dispatch_started_at, "dispatch_started_at")
        response = (
            None
            if self.response_started_at is None
            else _observed_timestamp(self.response_started_at, "response_started_at")
        )
        settled = _observed_timestamp(self.settled_at, "settled_at")
        if not dispatch <= settled or (
            response is not None and not dispatch <= response <= settled
        ):
            raise ValueError("observed call chronology must be ordered")
        if self.usage_json is not None:
            _canonical_json_object(self.usage_json, "usage_json")
        if (
            type(self.request_surface) is not tuple
            or not self.request_surface
            or len(self.request_surface) > MAX_SURFACE_REPLACEMENT_SPAN
            or any(
                type(item) is not PostDispatchTraceSurfaceComponent
                for item in self.request_surface
            )
        ):
            raise ValueError("request_surface must be a bounded non-empty typed tuple")
        if type(self.response) is not PostDispatchTraceResponse:
            raise TypeError("response must be a PostDispatchTraceResponse")
        if (
            type(self.sealed_payload_bytes) is not int
            or not 0 <= self.sealed_payload_bytes <= MAX_PROMOTED_TRACE_BYTES
        ):
            raise ValueError("sealed_payload_bytes is invalid")
        if self.terminal_state not in {
            TraceCallState.COMPLETE,
            TraceCallState.ERROR,
            TraceCallState.STOPPED,
            TraceCallState.INTERRUPTED,
        }:
            raise ValueError("terminal_state is not importable")
        if (
            self.terminal_state is TraceCallState.COMPLETE
            and self.response.kind == "no_response"
            and self.response.omission_reason_code != "voice_assistant_revision_pending"
        ):
            raise ValueError("a complete call requires an observed response")
        if (
            self.terminal_state
            in {
                TraceCallState.COMPLETE,
                TraceCallState.STOPPED,
                TraceCallState.INTERRUPTED,
            }
            and response is None
        ):
            raise ValueError("this terminal outcome requires response chronology")
        if self.response.kind != "no_response" and response is None:
            raise ValueError("an observed response requires response_started_at")


@dataclass(frozen=True, slots=True)
class PostDispatchTraceImport:
    """Complete sealed input to one atomic repository import."""

    import_id: str
    conversation_id: str
    user_message_id: str
    user_revision_id: str
    assistant_message_id: str
    assistant_revision_id: str
    turn_id: str
    run_id: str
    policy: FrozenTracePolicy
    expected_call_count: int
    calls: tuple[PostDispatchTraceCall, ...] = field(repr=False)
    aggregate_payload_bytes: int = 0

    def __post_init__(self) -> None:
        _stable_id(self.import_id, "import_id")
        for name in (
            "conversation_id",
            "user_message_id",
            "user_revision_id",
            "assistant_message_id",
            "assistant_revision_id",
        ):
            _stable_id(getattr(self, name), name)
        _stable_id(self.turn_id, "turn_id")
        _required(self.run_id, "run_id")
        if type(self.policy) is not FrozenTracePolicy:
            raise TypeError("policy must be a FrozenTracePolicy")
        if (
            type(self.expected_call_count) is not int
            or not 1 <= self.expected_call_count <= MAX_PROMOTED_TRACE_CALLS
        ):
            raise ValueError("expected_call_count must be between one and eight")
        if type(self.calls) is not tuple or any(
            type(call) is not PostDispatchTraceCall for call in self.calls
        ):
            raise TypeError("calls must be a tuple of PostDispatchTraceCall values")
        if (
            type(self.aggregate_payload_bytes) is not int
            or not 0 <= self.aggregate_payload_bytes <= MAX_PROMOTED_TRACE_BYTES
        ):
            raise ValueError("aggregate_payload_bytes is invalid")


@dataclass(frozen=True, slots=True)
class PostDispatchTraceIdentitySet:
    """Domain-separated identities used only when an import creates lineage."""

    owner_id: str
    root_segment_id: str
    call_ids: tuple[str, ...]
    header_ids: tuple[str, ...]
    response_link_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PostDispatchTraceImportResult:
    """Content-free identities returned after an exact complete import."""

    conversation_id: str
    owner_id: str
    segment_id: str
    call_ids: tuple[str, ...]
    already_imported: bool


def derive_post_dispatch_trace_ids(
    import_id: str,
    *,
    call_count: int,
) -> PostDispatchTraceIdentitySet:
    """Derive stable IDs solely from an opaque import identity."""

    _stable_id(import_id, "import_id")
    if type(call_count) is not int or not 1 <= call_count <= MAX_PROMOTED_TRACE_CALLS:
        raise ValueError("call_count must be between one and eight")

    def derived(label: str) -> str:
        return str(uuid5(_POST_DISPATCH_TRACE_NAMESPACE, f"{label}:{import_id}"))

    return PostDispatchTraceIdentitySet(
        owner_id=derived("owner"),
        root_segment_id=derived("root-segment"),
        call_ids=tuple(derived(f"call:{index}") for index in range(call_count)),
        header_ids=tuple(derived(f"header:{index}") for index in range(call_count)),
        response_link_ids=tuple(
            derived(f"response-link:{index}") for index in range(call_count)
        ),
    )


def derive_post_dispatch_trace_node_id(
    import_id: str,
    call_sequence: int,
    component_ordinal: int,
) -> str:
    """Derive one stable request-surface node identity."""

    _stable_id(import_id, "import_id")
    if type(call_sequence) is not int or call_sequence < 0:
        raise ValueError("call_sequence")
    if type(component_ordinal) is not int or component_ordinal < 0:
        raise ValueError("component_ordinal")
    return str(
        uuid5(
            _POST_DISPATCH_TRACE_NAMESPACE,
            f"surface:{call_sequence}:{component_ordinal}:{import_id}",
        )
    )


def derive_post_dispatch_trace_replacement_id(
    import_id: str,
    call_sequence: int,
) -> str:
    """Derive one stable changed-surface replacement identity."""

    _stable_id(import_id, "import_id")
    if type(call_sequence) is not int or call_sequence < 0:
        raise ValueError("call_sequence")
    return str(
        uuid5(
            _POST_DISPATCH_TRACE_NAMESPACE,
            f"surface-replacement:{call_sequence}:{import_id}",
        )
    )


__all__ = [
    "ConfirmedPreCommitTraceImportError",
    "MAX_PROMOTED_TRACE_CALLS",
    "MAX_PROMOTED_TRACE_BYTES",
    "PostDispatchTraceArtifact",
    "PostDispatchTraceCall",
    "PostDispatchTraceHeaderComponent",
    "PostDispatchTraceHeaderOmission",
    "PostDispatchTraceIdentitySet",
    "PostDispatchTraceImport",
    "PostDispatchTraceImportResult",
    "PostDispatchTraceResponse",
    "PostDispatchTraceSurfaceComponent",
    "PostDispatchTraceSystemComponent",
    "derive_post_dispatch_trace_ids",
    "derive_post_dispatch_trace_node_id",
    "derive_post_dispatch_trace_replacement_id",
]
