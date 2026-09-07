"""Content-free capture provenance for provider-neutral Console requests.

These values travel beside semantic request values.  They are never provider
payload, authority, permission, or a durable copy/fingerprint of message text.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast

from tldw_chatbook.Chat.console_trace_models import (
    FrozenTracePolicy,
    SemanticRevisionRef,
)

if TYPE_CHECKING:
    from tldw_chatbook.Chat.console_semantic_revision import (
        SemanticRevisionCoordinator,
    )
else:
    SemanticRevisionCoordinator = Any


MAX_PROVENANCE_TRANSFORM_INPUTS = 256


class TraceProvenanceAlignmentError(ValueError):
    """Semantic values and their capture-only descriptors are not aligned."""


class TraceProvenancePersistenceError(RuntimeError):
    """Content-free failure raised when saved-revision admission cannot commit."""

    def __init__(self) -> None:
        super().__init__("trace_provenance_persistence_failed")


class TraceProvenanceSource(str, Enum):
    """Closed source kinds for semantic request components."""

    RENDERED_SYSTEM = "rendered_system"
    CONVERSATION_MEMORY = "conversation_memory"
    MANDATORY_CONTEXT = "mandatory_context"
    COMPACTABLE_HISTORY = "compactable_history"
    ACTIVE_REQUEST = "active_request"
    PROJECT_INSTRUCTION = "project_instruction"
    RAG_CONTEXT = "rag_context"
    TOOL_DEFINITION = "tool_definition"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    PROVIDER_OVERLAY = "provider_overlay"
    CONTINUATION = "continuation"
    THINKING = "thinking"
    VISUAL_TRANSCRIPT = "visual_transcript"
    CONTEXT_SUMMARY = "context_summary"
    SKILL_RENDER = "skill_render"
    WORLD_INFO = "world_info"
    DICTIONARY_TRANSFORM = "dictionary_transform"
    CITATION_EVIDENCE = "citation_evidence"
    PREFILL = "prefill"


class TraceOmissionReason(str, Enum):
    """Closed content-free reasons a semantic component is unavailable."""

    FRESH_RAG_NOT_SELECTED = "fresh_rag_not_selected"
    RETRY_RAG_NOT_REPLAYED = "retry_rag_not_replayed"
    AGENT_WAKE_RAG_SKIPPED = "agent_wake_rag_skipped"
    WINDOWED = "windowed"
    SOURCE_UNAVAILABLE = "source_unavailable"
    SANITIZER_FAILED = "sanitizer_failed"
    PERSISTENCE_FAILED = "persistence_failed"
    ALIGNMENT_MISMATCH = "alignment_mismatch"
    UNSUPPORTED_REPLACEMENT_SPAN = "unsupported_replacement_span"


class TraceTransformKind(str, Enum):
    """Closed structural transforms that may merge or split semantic values."""

    SINGLE_PREAMBLE = "single_preamble"
    DISTINCT_ROLES = "distinct_roles"
    REPLACEMENT = "replacement"
    THINKING_ATTACHMENT = "thinking_attachment"
    CONTINUATION_ATTACHMENT = "continuation_attachment"
    VISUAL_COMPACTION = "visual_compaction"
    TEXT_COMPACTION = "text_compaction"
    HYBRID_COMPACTION = "hybrid_compaction"
    SYSTEM_FRAMING = "system_framing"
    PROVIDER_OVERLAY = "provider_overlay"
    MESSAGE_REWRITE = "message_rewrite"
    WINDOWING = "windowing"


class ConsoleTraceCaptureMode(str, Enum):
    """Explicit capture admission selected for one preparation attempt."""

    CAPTURE_ON = "capture_on"
    CAPTURE_OFF = "capture_off"


class ConsoleRequestRoute(str, Enum):
    """Closed logical routes that can prepare a Console-owned provider call."""

    FRESH = "fresh"
    RETRY = "retry"
    CONTINUE = "continue"
    REGENERATE = "regenerate"
    EDIT = "edit"
    DIRECT_PREFILL = "direct_prefill"
    AGENT_FIRST = "agent_first"
    TOOL_LOOP = "tool_loop"
    CITATION_REPAIR = "citation_repair"
    MANUAL_SUMMARY = "manual_summary"
    IMPERSONATE = "impersonate"
    AUTO_COMPACTION = "auto_compaction"
    LLAMA_FALLBACK = "llama_fallback"


class RouteRagDisposition(str, Enum):
    """Required RAG provenance state for one logical route."""

    SOURCE_OR_FRESH_OMISSION = "source_or_fresh_omission"
    RETRY_OMISSION = "retry_omission"
    AGENT_WAKE_OMISSION = "agent_wake_omission"
    NOT_APPLICABLE = "not_applicable"


class ConsoleRouteCaptureDisposition(str, Enum):
    """Whether one request route belongs to the conversation trace ledger."""

    CONVERSATION_TRACE = "console_conversation"
    CAPTURE_OFF = "capture_off"
    EXCLUDED = "excluded"


@dataclass(frozen=True, slots=True)
class ConsoleRequestRouteRecord:
    """Capture ownership requirements for one logical request route."""

    route: ConsoleRequestRoute
    gateway: Literal["stream_chat", "complete_auxiliary"]
    rag: RouteRagDisposition
    source_module: str
    source_function: str
    source_line: int
    source_marker: str
    predicate: str
    capture: ConsoleRouteCaptureDisposition
    actor_chain_required: bool = False


CONSOLE_REQUEST_ROUTE_CENSUS = (
    ConsoleRequestRouteRecord(
        ConsoleRequestRoute.FRESH,
        "stream_chat",
        RouteRagDisposition.SOURCE_OR_FRESH_OMISSION,
        "tldw_chatbook/Chat/console_chat_controller.py",
        "_submit_draft_inner",
        6750,
        "route=ConsoleRequestRoute.FRESH,",
        "fresh_submit",
        ConsoleRouteCaptureDisposition.CONVERSATION_TRACE,
    ),
    ConsoleRequestRouteRecord(
        ConsoleRequestRoute.RETRY,
        "stream_chat",
        RouteRagDisposition.RETRY_OMISSION,
        "tldw_chatbook/Chat/console_chat_controller.py",
        "retry_message",
        11814,
        "route=ConsoleRequestRoute.RETRY,",
        "retry_requested",
        ConsoleRouteCaptureDisposition.CONVERSATION_TRACE,
    ),
    ConsoleRequestRouteRecord(
        ConsoleRequestRoute.CONTINUE,
        "stream_chat",
        RouteRagDisposition.RETRY_OMISSION,
        "tldw_chatbook/Chat/console_chat_controller.py",
        "continue_from_message",
        11968,
        "route=ConsoleRequestRoute.CONTINUE,",
        "continue_requested",
        ConsoleRouteCaptureDisposition.CONVERSATION_TRACE,
    ),
    ConsoleRequestRouteRecord(
        ConsoleRequestRoute.REGENERATE,
        "stream_chat",
        RouteRagDisposition.RETRY_OMISSION,
        "tldw_chatbook/Chat/console_chat_controller.py",
        "regenerate_message",
        12102,
        "route=ConsoleRequestRoute.REGENERATE,",
        "regenerate_requested",
        ConsoleRouteCaptureDisposition.CONVERSATION_TRACE,
    ),
    ConsoleRequestRouteRecord(
        ConsoleRequestRoute.EDIT,
        "stream_chat",
        RouteRagDisposition.RETRY_OMISSION,
        "tldw_chatbook/Chat/console_chat_controller.py",
        "edit_and_resend_message",
        12761,
        "route=ConsoleRequestRoute.EDIT,",
        "edit_and_resend_requested",
        ConsoleRouteCaptureDisposition.CONVERSATION_TRACE,
    ),
    ConsoleRequestRouteRecord(
        ConsoleRequestRoute.DIRECT_PREFILL,
        "stream_chat",
        RouteRagDisposition.NOT_APPLICABLE,
        "tldw_chatbook/Chat/console_chat_controller.py",
        "_stream_assistant_response_inner",
        16055,
        "route=(ConsoleRequestRoute.DIRECT_PREFILL if prefill else route),",
        "direct_prefill_present",
        ConsoleRouteCaptureDisposition.CONVERSATION_TRACE,
    ),
    ConsoleRequestRouteRecord(
        ConsoleRequestRoute.AGENT_FIRST,
        "stream_chat",
        RouteRagDisposition.AGENT_WAKE_OMISSION,
        "tldw_chatbook/Chat/console_agent_bridge.py",
        "chat_call",
        2619,
        "ConsoleRequestRoute.AGENT_FIRST",
        "agent_first_wake",
        ConsoleRouteCaptureDisposition.CONVERSATION_TRACE,
        actor_chain_required=True,
    ),
    ConsoleRequestRouteRecord(
        ConsoleRequestRoute.TOOL_LOOP,
        "stream_chat",
        RouteRagDisposition.AGENT_WAKE_OMISSION,
        "tldw_chatbook/Chat/console_agent_bridge.py",
        "chat_call",
        2621,
        "else ConsoleRequestRoute.TOOL_LOOP",
        "agent_tool_loop_iteration",
        ConsoleRouteCaptureDisposition.CONVERSATION_TRACE,
        actor_chain_required=True,
    ),
    ConsoleRequestRouteRecord(
        ConsoleRequestRoute.CITATION_REPAIR,
        "stream_chat",
        RouteRagDisposition.NOT_APPLICABLE,
        "tldw_chatbook/Chat/console_chat_controller.py",
        "_select_post_generation_body",
        17344,
        "route=ConsoleRequestRoute.CITATION_REPAIR,",
        "citation_repair_required",
        ConsoleRouteCaptureDisposition.CONVERSATION_TRACE,
    ),
    ConsoleRequestRouteRecord(
        ConsoleRequestRoute.MANUAL_SUMMARY,
        "stream_chat",
        RouteRagDisposition.NOT_APPLICABLE,
        "tldw_chatbook/Chat/console_chat_controller.py",
        "summarize_up_to",
        12277,
        "route=ConsoleRequestRoute.MANUAL_SUMMARY,",
        "manual_summary_requested",
        ConsoleRouteCaptureDisposition.CONVERSATION_TRACE,
    ),
    ConsoleRequestRouteRecord(
        ConsoleRequestRoute.IMPERSONATE,
        "stream_chat",
        RouteRagDisposition.NOT_APPLICABLE,
        "tldw_chatbook/Chat/console_chat_controller.py",
        "impersonate_user_reply",
        12467,
        "route=ConsoleRequestRoute.IMPERSONATE,",
        "impersonate_requested",
        ConsoleRouteCaptureDisposition.CONVERSATION_TRACE,
    ),
    ConsoleRequestRouteRecord(
        ConsoleRequestRoute.AUTO_COMPACTION,
        "complete_auxiliary",
        RouteRagDisposition.NOT_APPLICABLE,
        "tldw_chatbook/Chat/console_context_compaction.py",
        "compact",
        483,
        "route=ConsoleRequestRoute.AUTO_COMPACTION,",
        "automatic_conversation_compaction",
        ConsoleRouteCaptureDisposition.CAPTURE_OFF,
    ),
    ConsoleRequestRouteRecord(
        ConsoleRequestRoute.LLAMA_FALLBACK,
        "stream_chat",
        RouteRagDisposition.NOT_APPLICABLE,
        "tldw_chatbook/Chat/console_provider_gateway.py",
        "stream_chat",
        4063,
        "ConsoleRequestRoute.LLAMA_FALLBACK",
        "llama_fallback_selected",
        ConsoleRouteCaptureDisposition.CONVERSATION_TRACE,
    ),
)


@dataclass(frozen=True, slots=True)
class GatewayCallsiteRecord:
    """One AST-discoverable gateway callsite and its preparation ownership."""

    module: str
    function: str
    gateway: Literal["stream_chat", "complete_auxiliary"]
    source_line: int
    owner: ConsoleRouteCaptureDisposition
    routes: tuple[ConsoleRequestRoute, ...]
    route_binding: str = ""


CONSOLE_GATEWAY_CALLSITE_CENSUS = (
    GatewayCallsiteRecord(
        "tldw_chatbook/Chat/console_chat_controller.py",
        "_collect_summary_completion",
        "stream_chat",
        12521,
        ConsoleRouteCaptureDisposition.CONVERSATION_TRACE,
        (ConsoleRequestRoute.MANUAL_SUMMARY, ConsoleRequestRoute.IMPERSONATE),
        "route",
    ),
    GatewayCallsiteRecord(
        "tldw_chatbook/Chat/console_chat_controller.py",
        "_run_direct_provider_reply",
        "stream_chat",
        16804,
        ConsoleRouteCaptureDisposition.CONVERSATION_TRACE,
        (
            ConsoleRequestRoute.FRESH,
            ConsoleRequestRoute.RETRY,
            ConsoleRequestRoute.CONTINUE,
            ConsoleRequestRoute.REGENERATE,
            ConsoleRequestRoute.EDIT,
            ConsoleRequestRoute.DIRECT_PREFILL,
            ConsoleRequestRoute.LLAMA_FALLBACK,
        ),
        "route",
    ),
    GatewayCallsiteRecord(
        "tldw_chatbook/Chat/console_chat_controller.py",
        "_select_post_generation_body",
        "stream_chat",
        17340,
        ConsoleRouteCaptureDisposition.CONVERSATION_TRACE,
        (ConsoleRequestRoute.CITATION_REPAIR,),
        "ConsoleRequestRoute.CITATION_REPAIR",
    ),
    GatewayCallsiteRecord(
        "tldw_chatbook/Chat/console_agent_bridge.py",
        "_consume",
        "stream_chat",
        2757,
        ConsoleRouteCaptureDisposition.CONVERSATION_TRACE,
        (ConsoleRequestRoute.AGENT_FIRST, ConsoleRequestRoute.TOOL_LOOP),
        "route",
    ),
    GatewayCallsiteRecord(
        "tldw_chatbook/Chat/console_context_compaction.py",
        "compact",
        "complete_auxiliary",
        476,
        ConsoleRouteCaptureDisposition.CAPTURE_OFF,
        (ConsoleRequestRoute.AUTO_COMPACTION,),
        "ConsoleRequestRoute.AUTO_COMPACTION",
    ),
    GatewayCallsiteRecord(
        "tldw_chatbook/Chat/console_side_chat.py",
        "run",
        "stream_chat",
        151,
        ConsoleRouteCaptureDisposition.EXCLUDED,
        (),
        "None",
    ),
    GatewayCallsiteRecord(
        "tldw_chatbook/Chat/console_visual_evaluation.py",
        "_evaluate_representation",
        "stream_chat",
        833,
        ConsoleRouteCaptureDisposition.EXCLUDED,
        (),
        "None",
    ),
    GatewayCallsiteRecord(
        "tldw_chatbook/Prompt_Management/prompt_improvement_service.py",
        "improve",
        "complete_auxiliary",
        447,
        ConsoleRouteCaptureDisposition.EXCLUDED,
        (),
        "None",
    ),
    GatewayCallsiteRecord(
        "tldw_chatbook/UI/Persona_Modules/personas_preview_controller.py",
        "_consume",
        "stream_chat",
        775,
        ConsoleRouteCaptureDisposition.EXCLUDED,
        (),
        "None",
    ),
    GatewayCallsiteRecord(
        "tldw_chatbook/Character_Chat/character_generation_controller.py",
        "_run",
        "stream_chat",
        97,
        ConsoleRouteCaptureDisposition.EXCLUDED,
        (),
        "None",
    ),
)


def rag_provenance_for_route(
    route: ConsoleRequestRoute,
    selected: ProviderArtifactTraceProvenance | None,
) -> TraceProvenance:
    """Resolve a RAG source or an explicit route-specific absence descriptor."""

    if selected is not None:
        if selected.source is not TraceProvenanceSource.RAG_CONTEXT:
            raise TraceProvenanceAlignmentError(
                "selected RAG provenance has the wrong source kind"
            )
        return selected
    record = next(item for item in CONSOLE_REQUEST_ROUTE_CENSUS if item.route is route)
    reason = (
        TraceOmissionReason.FRESH_RAG_NOT_SELECTED
        if record.rag is RouteRagDisposition.SOURCE_OR_FRESH_OMISSION
        else TraceOmissionReason.RETRY_RAG_NOT_REPLAYED
        if record.rag is RouteRagDisposition.RETRY_OMISSION
        else TraceOmissionReason.AGENT_WAKE_RAG_SKIPPED
        if record.rag is RouteRagDisposition.AGENT_WAKE_OMISSION
        else TraceOmissionReason.SOURCE_UNAVAILABLE
    )
    return OmittedTraceProvenance(TraceProvenanceSource.RAG_CONTEXT, reason)


@dataclass(frozen=True, slots=True)
class SavedRevisionTraceProvenance:
    """Reference to ordinary saved semantics; never a copy or text digest."""

    revision_id: str

    def __post_init__(self) -> None:
        SemanticRevisionRef(self.revision_id)


@dataclass(frozen=True, slots=True)
class RequestRouteTraceProvenance:
    """Content-free logical route annotation outside value-parallel slots."""

    route: ConsoleRequestRoute
    predicate: str = ""
    actor_id: str | None = field(default=None, repr=False)
    chain_id: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if type(self.route) is not ConsoleRequestRoute:
            raise ValueError("route must be a ConsoleRequestRoute")
        record = next(
            item for item in CONSOLE_REQUEST_ROUTE_CENSUS if item.route is self.route
        )
        if not self.predicate:
            object.__setattr__(self, "predicate", record.predicate)
        if self.predicate != record.predicate:
            raise TraceProvenanceAlignmentError("route predicate does not match route")
        actor_values = (self.actor_id, self.chain_id)
        if record.actor_chain_required:
            try:
                for value in actor_values:
                    SemanticRevisionRef(cast(str, value))
            except (TypeError, ValueError) as exc:
                raise TraceProvenanceAlignmentError(
                    "opaque actor and chain identity are required for this route"
                ) from exc
        if not record.actor_chain_required and any(
            value is not None for value in actor_values
        ):
            raise TraceProvenanceAlignmentError(
                "actor and chain identity are not valid for this route"
            )


def request_route_provenance(
    route: ConsoleRequestRoute,
    *,
    actor_id: str | None = None,
    chain_id: str | None = None,
) -> RequestRouteTraceProvenance:
    """Build the structural route descriptor from the closed route census."""

    if type(route) is not ConsoleRequestRoute:
        raise TypeError("route must be ConsoleRequestRoute")
    record = next(item for item in CONSOLE_REQUEST_ROUTE_CENSUS if item.route is route)
    return RequestRouteTraceProvenance(
        route,
        record.predicate,
        actor_id=actor_id,
        chain_id=chain_id,
    )


@dataclass(frozen=True, slots=True)
class ProviderArtifactTraceProvenance:
    """Intent to store one provider-only value under a frozen trace policy."""

    source: TraceProvenanceSource
    policy: FrozenTracePolicy = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.source) is not TraceProvenanceSource:
            raise ValueError("source must be a TraceProvenanceSource")
        if type(self.policy) is not FrozenTracePolicy:
            raise TypeError("policy must be a FrozenTracePolicy")


@dataclass(frozen=True, slots=True)
class OmittedTraceProvenance:
    """Content-free disclosure that a source was deliberately unavailable."""

    source: TraceProvenanceSource
    reason: TraceOmissionReason
    omitted_count: int = 0

    def __post_init__(self) -> None:
        if type(self.source) is not TraceProvenanceSource:
            raise ValueError("source must be a TraceProvenanceSource")
        if type(self.reason) is not TraceOmissionReason:
            raise ValueError("reason must be a TraceOmissionReason")
        if type(self.omitted_count) is not int or self.omitted_count < 0:
            raise ValueError("omitted_count must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class DerivedTraceProvenance:
    """Bounded structural transform over one or more content-free descriptors."""

    transform: TraceTransformKind
    inputs: tuple["TraceProvenance", ...] = field(repr=False)
    artifact: ProviderArtifactTraceProvenance | None = field(
        default=None,
        repr=False,
    )

    def __post_init__(self) -> None:
        if type(self.transform) is not TraceTransformKind:
            raise ValueError("transform must be a TraceTransformKind")
        values = tuple(self.inputs)
        if not values:
            raise TraceProvenanceAlignmentError(
                "a derived provenance transform requires at least one input"
            )
        if len(values) > MAX_PROVENANCE_TRANSFORM_INPUTS:
            raise TraceProvenanceAlignmentError(
                "a provenance transform may contain at most 256 inputs"
            )
        if any(not _is_descriptor(value) for value in values):
            raise TypeError("transform inputs must be trace provenance descriptors")
        if (
            self.artifact is not None
            and type(self.artifact) is not ProviderArtifactTraceProvenance
        ):
            raise TypeError("artifact must be provider-only trace provenance")
        object.__setattr__(self, "inputs", values)
        _validate_derived_shape(self)


TraceProvenance: TypeAlias = (
    SavedRevisionTraceProvenance
    | RequestRouteTraceProvenance
    | ProviderArtifactTraceProvenance
    | OmittedTraceProvenance
    | DerivedTraceProvenance
)


def _is_descriptor(value: object) -> bool:
    return type(value) in {
        SavedRevisionTraceProvenance,
        RequestRouteTraceProvenance,
        ProviderArtifactTraceProvenance,
        OmittedTraceProvenance,
        DerivedTraceProvenance,
    }


def _validate_derived_shape(descriptor: DerivedTraceProvenance) -> None:
    """Reject contradictory transform, input, and artifact combinations."""

    transform = descriptor.transform
    artifact = descriptor.artifact
    if transform in {
        TraceTransformKind.THINKING_ATTACHMENT,
        TraceTransformKind.CONTINUATION_ATTACHMENT,
    }:
        if len(descriptor.inputs) != 1:
            raise TraceProvenanceAlignmentError(
                "sidecar attachment requires exactly one owner"
            )
        source = (
            TraceProvenanceSource.THINKING
            if transform is TraceTransformKind.THINKING_ATTACHMENT
            else TraceProvenanceSource.CONTINUATION
        )
        owner_is_saved = type(descriptor.inputs[0]) is SavedRevisionTraceProvenance
        artifact_required = (
            transform is TraceTransformKind.CONTINUATION_ATTACHMENT
            or not owner_is_saved
        )
        if artifact_required == (artifact is None) or (
            artifact is not None and artifact.source is not source
        ):
            raise TraceProvenanceAlignmentError(
                "sidecar attachment artifact contradicts its owner"
            )
        return
    required_artifacts = {
        TraceTransformKind.PROVIDER_OVERLAY: frozenset(
            {TraceProvenanceSource.PROVIDER_OVERLAY}
        ),
        TraceTransformKind.TEXT_COMPACTION: frozenset(
            {TraceProvenanceSource.CONTEXT_SUMMARY}
        ),
        TraceTransformKind.VISUAL_COMPACTION: frozenset(
            {TraceProvenanceSource.VISUAL_TRANSCRIPT}
        ),
        TraceTransformKind.HYBRID_COMPACTION: frozenset(
            {
                TraceProvenanceSource.CONTEXT_SUMMARY,
                TraceProvenanceSource.VISUAL_TRANSCRIPT,
            }
        ),
    }
    allowed_sources = required_artifacts.get(transform)
    if allowed_sources is not None:
        if artifact is None or artifact.source not in allowed_sources:
            raise TraceProvenanceAlignmentError(
                "derived transform requires its exact provider artifact"
            )
    elif artifact is not None:
        raise TraceProvenanceAlignmentError(
            "derived transform does not permit a provider artifact"
        )
    if transform is TraceTransformKind.MESSAGE_REWRITE and any(
        type(item) is not DerivedTraceProvenance
        or item.transform
        not in {
            TraceTransformKind.THINKING_ATTACHMENT,
            TraceTransformKind.CONTINUATION_ATTACHMENT,
        }
        for item in descriptor.inputs[1:]
    ):
        raise TraceProvenanceAlignmentError(
            "message rewrite accepts only exact sidecar attachments"
        )
    if transform is TraceTransformKind.MESSAGE_REWRITE and any(
        item.inputs[0] != descriptor.inputs[0]
        for item in descriptor.inputs[1:]
        if type(item) is DerivedTraceProvenance
    ):
        raise TraceProvenanceAlignmentError(
            "message rewrite sidecar does not match its exact owner"
        )


def _descriptor_matches_category(
    descriptor: TraceProvenance,
    *,
    allow_saved: bool,
    artifact_sources: frozenset[TraceProvenanceSource],
) -> bool:
    if (
        type(descriptor) is DerivedTraceProvenance
        and descriptor.transform is TraceTransformKind.MESSAGE_REWRITE
    ):
        if not descriptor.inputs or not _descriptor_matches_category(
            descriptor.inputs[0],
            allow_saved=allow_saved,
            artifact_sources=artifact_sources,
        ):
            return False
        return all(
            type(item) is DerivedTraceProvenance
            and item.transform
            in {
                TraceTransformKind.THINKING_ATTACHMENT,
                TraceTransformKind.CONTINUATION_ATTACHMENT,
            }
            and _descriptor_matches_category(
                item,
                allow_saved=True,
                artifact_sources=frozenset(
                    {
                        TraceProvenanceSource.THINKING
                        if item.transform is TraceTransformKind.THINKING_ATTACHMENT
                        else TraceProvenanceSource.CONTINUATION,
                    }
                ),
            )
            for item in descriptor.inputs[1:]
        )
    if type(descriptor) is DerivedTraceProvenance and descriptor.artifact is not None:
        return descriptor.artifact.source in artifact_sources
    if type(descriptor) is SavedRevisionTraceProvenance:
        return allow_saved
    if type(descriptor) is ProviderArtifactTraceProvenance:
        return (
            cast(ProviderArtifactTraceProvenance, descriptor).source in artifact_sources
        )
    if type(descriptor) is OmittedTraceProvenance:
        return cast(OmittedTraceProvenance, descriptor).source in artifact_sources
    if type(descriptor) is not DerivedTraceProvenance:
        return False
    return all(
        _descriptor_matches_category(
            item,
            allow_saved=allow_saved,
            artifact_sources=artifact_sources,
        )
        for item in descriptor.inputs
    )


def _validate_category(
    field_name: str,
    descriptors: tuple[TraceProvenance, ...],
    *,
    allow_saved: bool,
    artifact_sources: frozenset[TraceProvenanceSource],
) -> None:
    for descriptor in descriptors:
        if _descriptor_matches_category(
            descriptor,
            allow_saved=allow_saved,
            artifact_sources=artifact_sources,
        ):
            continue
        raise TraceProvenanceAlignmentError(
            f"trace provenance category mismatch: {field_name}"
        )


def _bounded_transform_inputs(
    values: tuple[TraceProvenance, ...],
    *,
    source: TraceProvenanceSource,
    reason: TraceOmissionReason,
) -> tuple[TraceProvenance, ...]:
    if len(values) <= MAX_PROVENANCE_TRANSFORM_INPUTS:
        return values
    retained = values[: MAX_PROVENANCE_TRANSFORM_INPUTS - 1]
    return retained + (
        OmittedTraceProvenance(
            source,
            reason,
            omitted_count=len(values) - len(retained),
        ),
    )


def frozen_policy_from_provenance(
    provenance: ConsoleRequestProvenance,
) -> FrozenTracePolicy:
    """Return the unique frozen policy carried by provider-only descriptors."""

    return provenance.capture_policy


def compaction_transform_provenance(
    provenance: ConsoleRequestProvenance,
    *,
    selected_units: int,
    transform: TraceTransformKind,
    source: TraceProvenanceSource,
    include_memory: bool = True,
) -> DerivedTraceProvenance:
    """Describe one bounded replacement without retaining its semantic values."""

    if (
        type(selected_units) is not int
        or selected_units < 0
        or selected_units > len(provenance.compactable)
    ):
        raise TraceProvenanceAlignmentError("selected_units is out of range")
    replaced = (provenance.memory if include_memory else ()) + tuple(
        item
        for unit in provenance.compactable[:selected_units]
        for item in unit.messages
    )
    return DerivedTraceProvenance(
        transform,
        _bounded_transform_inputs(
            replaced,
            source=source,
            reason=TraceOmissionReason.UNSUPPORTED_REPLACEMENT_SPAN,
        ),
        artifact=ProviderArtifactTraceProvenance(
            source,
            frozen_policy_from_provenance(provenance),
        ),
    )


def _freeze_descriptors(
    values: tuple[TraceProvenance, ...], field_name: str
) -> tuple[TraceProvenance, ...]:
    frozen = tuple(values)
    if any(not _is_descriptor(value) for value in frozen):
        raise TypeError(
            f"{field_name} must contain exact closed trace provenance descriptors"
        )
    return frozen


@dataclass(frozen=True, slots=True)
class ConsoleUnitProvenance:
    """Descriptors parallel to one complete compactable conversation unit."""

    messages: tuple[TraceProvenance, ...] = field(repr=False)
    tool_loop: tuple[TraceProvenance, ...] = field(default=(), repr=False)
    thinking: tuple[TraceProvenance, ...] = field(default=(), repr=False)
    continuations: tuple[TraceProvenance, ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        for name in ("messages", "tool_loop", "thinking", "continuations"):
            object.__setattr__(
                self,
                name,
                _freeze_descriptors(tuple(getattr(self, name)), name),
            )


@dataclass(frozen=True, slots=True)
class ConsoleRequestProvenance:
    """Capture-only aggregate parallel to every provider-neutral category."""

    system: tuple[TraceProvenance, ...] = field(repr=False)
    memory: tuple[TraceProvenance, ...] = field(repr=False)
    mandatory: tuple[TraceProvenance, ...] = field(repr=False)
    compactable: tuple[ConsoleUnitProvenance, ...] = field(repr=False)
    active_request: tuple[TraceProvenance, ...] = field(repr=False)
    active_thinking: tuple[TraceProvenance, ...] = field(repr=False)
    active_continuations: tuple[TraceProvenance, ...] = field(repr=False)
    tools: tuple[TraceProvenance, ...] = field(repr=False)
    capture_policy: FrozenTracePolicy = field(repr=False)
    tool_loop: tuple[TraceProvenance, ...] = field(default=(), repr=False)
    metadata: tuple[TraceProvenance, ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        if type(self.capture_policy) is not FrozenTracePolicy:
            raise TypeError("capture_policy must be FrozenTracePolicy")
        for name in (
            "system",
            "memory",
            "mandatory",
            "active_request",
            "tool_loop",
            "active_thinking",
            "active_continuations",
            "tools",
            "metadata",
        ):
            object.__setattr__(
                self,
                name,
                _freeze_descriptors(tuple(getattr(self, name)), name),
            )
        units = tuple(self.compactable)
        if any(type(unit) is not ConsoleUnitProvenance for unit in units):
            raise TypeError("compactable must contain ConsoleUnitProvenance values")
        object.__setattr__(self, "compactable", units)
        _validate_category(
            "system",
            self.system,
            allow_saved=True,
            artifact_sources=frozenset({TraceProvenanceSource.RENDERED_SYSTEM}),
        )
        _validate_category(
            "memory",
            self.memory,
            allow_saved=False,
            artifact_sources=frozenset(
                {
                    TraceProvenanceSource.CONVERSATION_MEMORY,
                    TraceProvenanceSource.CONTEXT_SUMMARY,
                    TraceProvenanceSource.VISUAL_TRANSCRIPT,
                }
            ),
        )
        _validate_category(
            "mandatory",
            self.mandatory,
            allow_saved=False,
            artifact_sources=frozenset(
                {
                    TraceProvenanceSource.MANDATORY_CONTEXT,
                    TraceProvenanceSource.PROJECT_INSTRUCTION,
                    TraceProvenanceSource.RAG_CONTEXT,
                    TraceProvenanceSource.SKILL_RENDER,
                    TraceProvenanceSource.WORLD_INFO,
                    TraceProvenanceSource.DICTIONARY_TRANSFORM,
                    TraceProvenanceSource.CITATION_EVIDENCE,
                }
            ),
        )
        for unit in self.compactable:
            _validate_category(
                "compactable.messages",
                unit.messages,
                allow_saved=True,
                artifact_sources=frozenset(
                    {
                        TraceProvenanceSource.ACTIVE_REQUEST,
                        TraceProvenanceSource.PREFILL,
                        TraceProvenanceSource.TOOL_CALL,
                        TraceProvenanceSource.TOOL_RESULT,
                    }
                ),
            )
            _validate_category(
                "compactable.tool_loop",
                unit.tool_loop,
                allow_saved=True,
                artifact_sources=frozenset(
                    {
                        TraceProvenanceSource.TOOL_CALL,
                        TraceProvenanceSource.TOOL_RESULT,
                    }
                ),
            )
            _validate_category(
                "compactable.thinking",
                unit.thinking,
                allow_saved=True,
                artifact_sources=frozenset({TraceProvenanceSource.THINKING}),
            )
            _validate_category(
                "compactable.continuations",
                unit.continuations,
                allow_saved=True,
                artifact_sources=frozenset({TraceProvenanceSource.CONTINUATION}),
            )
        _validate_category(
            "active_request",
            self.active_request,
            allow_saved=True,
            artifact_sources=frozenset(
                {
                    TraceProvenanceSource.ACTIVE_REQUEST,
                    TraceProvenanceSource.PREFILL,
                    TraceProvenanceSource.TOOL_CALL,
                    TraceProvenanceSource.TOOL_RESULT,
                }
            ),
        )
        _validate_category(
            "tool_loop",
            self.tool_loop,
            allow_saved=True,
            artifact_sources=frozenset(
                {
                    TraceProvenanceSource.TOOL_CALL,
                    TraceProvenanceSource.TOOL_RESULT,
                }
            ),
        )
        _validate_category(
            "active_thinking",
            self.active_thinking,
            allow_saved=True,
            artifact_sources=frozenset({TraceProvenanceSource.THINKING}),
        )
        _validate_category(
            "active_continuations",
            self.active_continuations,
            allow_saved=True,
            artifact_sources=frozenset({TraceProvenanceSource.CONTINUATION}),
        )
        _validate_category(
            "tools",
            self.tools,
            allow_saved=False,
            artifact_sources=frozenset({TraceProvenanceSource.TOOL_DEFINITION}),
        )
        if any(
            type(item)
            not in {
                RequestRouteTraceProvenance,
                OmittedTraceProvenance,
                DerivedTraceProvenance,
                ProviderArtifactTraceProvenance,
            }
            for item in self.metadata
        ):
            raise TraceProvenanceAlignmentError(
                "trace provenance category mismatch: metadata"
            )
        policies: set[FrozenTracePolicy] = set()

        def collect(descriptor: TraceProvenance) -> None:
            if type(descriptor) is ProviderArtifactTraceProvenance:
                policies.add(descriptor.policy)
            elif type(descriptor) is DerivedTraceProvenance:
                if descriptor.artifact is not None:
                    policies.add(descriptor.artifact.policy)
                for item in descriptor.inputs:
                    collect(item)

        for descriptor in (
            self.flattened_messages()
            + tuple(
                item
                for unit in self.compactable
                for item in (*unit.thinking, *unit.continuations)
            )
            + self.active_thinking
            + self.active_continuations
            + self.tools
            + self.metadata
        ):
            collect(descriptor)
        if any(policy != self.capture_policy for policy in policies):
            raise TraceProvenanceAlignmentError(
                "provider artifact policy does not match the frozen run policy"
            )

    def flattened_messages(self) -> tuple[TraceProvenance, ...]:
        """Return message descriptors in the semantic request's exact order."""

        return (
            self.system
            + self.memory
            + self.mandatory
            + tuple(item for unit in self.compactable for item in unit.messages)
            + self.active_request
        )

    def without_oldest_units(self, count: int) -> "ConsoleRequestProvenance":
        """Return the aggregate aligned after deterministic unit windowing."""

        if type(count) is not int or count < 0 or count > len(self.compactable):
            raise TraceProvenanceAlignmentError("window unit count is out of range")
        dropped = tuple(
            item
            for unit in self.compactable[:count]
            for item in (
                *unit.messages,
                *unit.thinking,
                *unit.continuations,
            )
        )
        metadata = self.metadata
        if dropped:
            metadata += (
                DerivedTraceProvenance(
                    TraceTransformKind.WINDOWING,
                    _bounded_transform_inputs(
                        dropped,
                        source=TraceProvenanceSource.COMPACTABLE_HISTORY,
                        reason=TraceOmissionReason.WINDOWED,
                    ),
                ),
            )
            window_metadata = tuple(
                item
                for item in metadata
                if (
                    type(item) is DerivedTraceProvenance
                    and item.transform is TraceTransformKind.WINDOWING
                )
                or (
                    type(item) is OmittedTraceProvenance
                    and item.source is TraceProvenanceSource.COMPACTABLE_HISTORY
                    and item.reason is TraceOmissionReason.WINDOWED
                )
            )
            if len(window_metadata) > MAX_PROVENANCE_TRANSFORM_INPUTS:
                removed = window_metadata[: -MAX_PROVENANCE_TRANSFORM_INPUTS + 1]
                retained = window_metadata[-MAX_PROVENANCE_TRANSFORM_INPUTS + 1 :]
                omitted_count = 0
                for item in removed:
                    if type(item) is OmittedTraceProvenance:
                        omitted_count += cast(
                            OmittedTraceProvenance, item
                        ).omitted_count
                        continue
                    derived = cast(DerivedTraceProvenance, item)
                    omitted_count += sum(
                        cast(OmittedTraceProvenance, nested).omitted_count
                        if type(nested) is OmittedTraceProvenance
                        else 1
                        for nested in derived.inputs
                    )
                metadata = tuple(
                    item for item in metadata if item not in window_metadata
                ) + (
                    OmittedTraceProvenance(
                        TraceProvenanceSource.COMPACTABLE_HISTORY,
                        TraceOmissionReason.WINDOWED,
                        omitted_count=omitted_count,
                    ),
                    *retained,
                )

        return ConsoleRequestProvenance(
            system=self.system,
            memory=self.memory,
            mandatory=self.mandatory,
            compactable=self.compactable[count:],
            active_request=self.active_request,
            tool_loop=self.tool_loop,
            active_thinking=self.active_thinking,
            active_continuations=self.active_continuations,
            tools=self.tools,
            capture_policy=self.capture_policy,
            metadata=metadata,
        )

    def validate_alignment(
        self,
        *,
        system: int,
        memory: int,
        mandatory: int,
        compactable: tuple[tuple[int, int, int, int], ...],
        active_request: int,
        tool_loop: int = 0,
        active_thinking: int,
        active_continuations: int,
        tools: int,
    ) -> None:
        """Fail with a typed, content-free error on any one-for-one mismatch."""

        expected = {
            "system": system,
            "memory": memory,
            "mandatory": mandatory,
            "active_request": active_request,
            "tool_loop": tool_loop,
            "active_thinking": active_thinking,
            "active_continuations": active_continuations,
            "tools": tools,
        }
        for name, count in expected.items():
            if len(getattr(self, name)) != count:
                raise TraceProvenanceAlignmentError(
                    f"trace provenance alignment mismatch: {name}"
                )
        actual_units = tuple(
            (
                len(unit.messages),
                len(unit.tool_loop),
                len(unit.thinking),
                len(unit.continuations),
            )
            for unit in self.compactable
        )
        if actual_units != compactable:
            raise TraceProvenanceAlignmentError(
                "trace provenance alignment mismatch: compactable"
            )


@dataclass(frozen=True, slots=True)
class ProviderRequestProvenance:
    """Descriptors aligned to the final provider-prepared semantic shapes."""

    system_message: TraceProvenance | None = field(default=None, repr=False)
    messages: tuple[TraceProvenance, ...] = field(default=(), repr=False)
    messages_payload: tuple[TraceProvenance, ...] = field(default=(), repr=False)
    tools: tuple[TraceProvenance, ...] = field(default=(), repr=False)
    tool_loop: tuple[int, ...] = field(default=(), repr=False)
    thinking: tuple[TraceProvenance, ...] = field(default=(), repr=False)
    continuations: tuple[TraceProvenance, ...] = field(default=(), repr=False)
    metadata: tuple[TraceProvenance, ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        if self.system_message is not None and not _is_descriptor(self.system_message):
            raise TypeError("system_message must be a trace provenance descriptor")
        for name in (
            "messages",
            "messages_payload",
            "tools",
            "thinking",
            "continuations",
            "metadata",
        ):
            object.__setattr__(
                self,
                name,
                _freeze_descriptors(tuple(getattr(self, name)), name),
            )
        tool_loop = tuple(self.tool_loop)
        if (
            len(tool_loop) > MAX_PROVENANCE_TRANSFORM_INPUTS
            or any(type(index) is not int for index in tool_loop)
            or any(
                index < 0 or index >= len(self.messages_payload) for index in tool_loop
            )
            or any(left >= right for left, right in zip(tool_loop, tool_loop[1:]))
        ):
            raise TraceProvenanceAlignmentError(
                "tool_loop must be a bounded ordered provider-message overlay"
            )
        object.__setattr__(self, "tool_loop", tool_loop)
        message_sources = frozenset(
            source
            for source in TraceProvenanceSource
            if source
            not in {
                TraceProvenanceSource.TOOL_DEFINITION,
                TraceProvenanceSource.THINKING,
                TraceProvenanceSource.CONTINUATION,
            }
        )
        if self.system_message is not None:
            _validate_category(
                "system_message",
                (self.system_message,),
                allow_saved=True,
                artifact_sources=message_sources,
            )
        for name in ("messages", "messages_payload"):
            _validate_category(
                name,
                getattr(self, name),
                allow_saved=True,
                artifact_sources=message_sources,
            )
        _validate_category(
            "tools",
            self.tools,
            allow_saved=False,
            artifact_sources=frozenset({TraceProvenanceSource.TOOL_DEFINITION}),
        )
        _validate_category(
            "thinking",
            self.thinking,
            allow_saved=True,
            artifact_sources=frozenset({TraceProvenanceSource.THINKING}),
        )
        _validate_category(
            "continuations",
            self.continuations,
            allow_saved=True,
            artifact_sources=frozenset({TraceProvenanceSource.CONTINUATION}),
        )
        if any(
            type(item)
            not in {
                RequestRouteTraceProvenance,
                OmittedTraceProvenance,
                DerivedTraceProvenance,
                ProviderArtifactTraceProvenance,
            }
            for item in self.metadata
        ):
            raise TraceProvenanceAlignmentError(
                "trace provenance category mismatch: metadata"
            )


def _project_verified_provider_request_provenance(
    template: ProviderRequestProvenance,
    *,
    messages: Sequence[TraceProvenance],
    messages_payload: Sequence[TraceProvenance],
    continuations: Sequence[TraceProvenance],
    tool_loop: tuple[int, ...],
) -> ProviderRequestProvenance:
    """Build the private lazy projection after its delta passed normal validation."""

    if (
        type(template) is not ProviderRequestProvenance
        or not getattr(messages_payload, "_console_trace_projection", False)
        or not getattr(continuations, "_console_trace_projection", False)
        or len(tool_loop) > MAX_PROVENANCE_TRANSFORM_INPUTS
        or any(type(index) is not int for index in tool_loop)
        or any(index < 0 or index >= len(messages_payload) for index in tool_loop)
        or any(left >= right for left, right in zip(tool_loop, tool_loop[1:]))
    ):
        raise TraceProvenanceAlignmentError("invalid verified provider projection")
    projected = object.__new__(ProviderRequestProvenance)
    object.__setattr__(projected, "system_message", template.system_message)
    object.__setattr__(projected, "messages", messages)
    object.__setattr__(projected, "messages_payload", messages_payload)
    object.__setattr__(projected, "tools", template.tools)
    object.__setattr__(projected, "tool_loop", tool_loop)
    object.__setattr__(projected, "thinking", template.thinking)
    object.__setattr__(projected, "continuations", continuations)
    object.__setattr__(projected, "metadata", template.metadata)
    return projected


def admit_message_provenance(
    cursor: sqlite3.Cursor,
    *,
    coordinator: SemanticRevisionCoordinator,
    message_ids: tuple[str, ...],
) -> tuple[SavedRevisionTraceProvenance, ...]:
    """Ensure/reuse revisions for selected saved rows in the caller transaction.

    Any failure is converted to one typed content-free error.  The exception is
    intentionally raised inside the caller transaction so its context manager
    rolls back every revision created earlier in the batch.
    """

    admitted: list[SavedRevisionTraceProvenance] = []
    failure: TraceProvenancePersistenceError | None = None
    try:
        for message_id in message_ids:
            revision = coordinator.ensure_current_revision(
                cursor,
                message_id=message_id,
                creation_reason="request_capture",
            )
            admitted.append(
                SavedRevisionTraceProvenance(revision_id=revision.revision_id)
            )
    except Exception:
        failure = TraceProvenancePersistenceError()
    if failure is not None:
        raise failure
    return tuple(admitted)


@contextmanager
def trace_provenance_admission_transaction(
    database: object,
) -> Iterator[sqlite3.Cursor]:
    """Own the full revision-admission commit boundary and sanitize failures."""

    transaction = getattr(database, "transaction", None)
    if not callable(transaction):
        raise TypeError("database must provide a transaction context")
    local = getattr(database, "_local", None)
    if local is not None:
        active = getattr(local, "transaction_depth", 0) > 0
        connection = getattr(local, "conn", None)
        active = active or bool(
            connection is not None and getattr(connection, "in_transaction", False)
        )
        if active:
            raise TraceProvenancePersistenceError()
    failure: TraceProvenancePersistenceError | None = None
    cursor: sqlite3.Cursor | None = None
    try:
        with transaction(immediate=True) as active_cursor:
            cursor = active_cursor
            yield active_cursor
    except Exception as error:
        failure = (
            error
            if type(error) is TraceProvenancePersistenceError
            else TraceProvenancePersistenceError()
        )
        failure.__cause__ = None
        failure.__context__ = None
        failure.__traceback__ = None
        failure.__dict__.clear()
        TraceProvenancePersistenceError.__init__(failure)
    if failure is not None:
        raise failure
    if cursor is None:  # pragma: no cover - context managers must yield once
        raise TraceProvenancePersistenceError()
