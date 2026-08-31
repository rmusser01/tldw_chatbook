"""Capture-only provenance contracts for prepared Console requests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from contextlib import contextmanager
import hashlib

import pytest

from Tests.DB.fixtures.chachanotes_v54 import genuine_v54_database
from tldw_chatbook.Chat import console_trace_provenance as provenance_module
from tldw_chatbook.Chat.console_semantic_revision import SemanticRevisionCoordinator
from tldw_chatbook.Chat.console_prepared_request import (
    CONTINUATION_OWNER_KEY,
    THINKING_OWNER_KEY,
    attach_thinking_history,
    build_console_request,
    prepare_provider_request,
    resolve_request_capacity,
    tagged_memory_message,
    tagged_visual_memory_message,
)
from tldw_chatbook.Chat.console_thinking_history import (
    ResolvedThinkingBlock,
    ThinkingOwnerGroup,
)
from tldw_chatbook.Chat.provider_continuation import (
    continuation_owner_group,
    parse_provider_continuation_json,
)
from tldw_chatbook.Chat.console_trace_models import FrozenTracePolicy, new_opaque_id
from tldw_chatbook.Chat.console_trace_provenance import (
    MAX_PROVENANCE_TRANSFORM_INPUTS,
    ConsoleRequestProvenance,
    ConsoleUnitProvenance,
    DerivedTraceProvenance,
    OmittedTraceProvenance,
    ProviderArtifactTraceProvenance,
    ProviderRequestProvenance,
    SavedRevisionTraceProvenance,
    TraceProvenanceAlignmentError,
    TraceProvenance,
    TraceOmissionReason,
    TraceProvenancePersistenceError,
    TraceProvenanceSource,
    TraceTransformKind,
    admit_message_provenance,
    compaction_transform_provenance,
    trace_provenance_admission_transaction,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


def _policy() -> FrozenTracePolicy:
    return FrozenTracePolicy(
        policy_id=new_opaque_id(),
        credential_filter_version="credentials-v1",
        pii_redaction_enabled=False,
        pii_ruleset_revision_id=None,
    )


def _artifact(
    source: TraceProvenanceSource,
    policy: FrozenTracePolicy | None = None,
) -> ProviderArtifactTraceProvenance:
    return ProviderArtifactTraceProvenance(source=source, policy=policy or _policy())


def _word_count(messages: list[dict], _model: str) -> int:
    return sum(
        len(str(message.get("content") or "").split()) + 1 for message in messages
    )


def test_descriptors_are_immutable_closed_and_content_free() -> None:
    secret = "sk-live-raw-provider-value"
    saved = SavedRevisionTraceProvenance(revision_id=new_opaque_id())
    artifact = ProviderArtifactTraceProvenance(
        source=TraceProvenanceSource.RAG_CONTEXT,
        policy=_policy(),
    )
    omission = OmittedTraceProvenance(
        source=TraceProvenanceSource.RAG_CONTEXT,
        reason=TraceOmissionReason.FRESH_RAG_NOT_SELECTED,
    )
    derived = DerivedTraceProvenance(
        transform=TraceTransformKind.SINGLE_PREAMBLE,
        inputs=(saved, artifact, omission),
    )
    overlay = DerivedTraceProvenance(
        transform=TraceTransformKind.PROVIDER_OVERLAY,
        inputs=(saved,),
        artifact=ProviderArtifactTraceProvenance(
            TraceProvenanceSource.PROVIDER_OVERLAY,
            _policy(),
        ),
    )

    for descriptor in (saved, artifact, omission, derived, overlay):
        rendered = repr(descriptor)
        assert secret not in rendered
        assert "content" not in rendered.lower()
        assert "payload" not in rendered.lower()
        assert "authority" not in rendered.lower()
        assert "permission" not in rendered.lower()
        with pytest.raises((FrozenInstanceError, TypeError)):
            setattr(descriptor, "kind", "mutated")

    with pytest.raises(ValueError, match="source"):
        ProviderArtifactTraceProvenance(source="rag", policy=_policy())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="transform"):
        DerivedTraceProvenance(transform="merge", inputs=(saved,))  # type: ignore[arg-type]


def test_derived_descriptor_bounds_replacement_inputs() -> None:
    leaf = OmittedTraceProvenance(
        source=TraceProvenanceSource.COMPACTABLE_HISTORY,
        reason=TraceOmissionReason.WINDOWED,
    )

    with pytest.raises(TraceProvenanceAlignmentError, match="256"):
        DerivedTraceProvenance(
            transform=TraceTransformKind.REPLACEMENT,
            inputs=(leaf,) * (MAX_PROVENANCE_TRANSFORM_INPUTS + 1),
        )


def test_request_aggregate_rejects_partial_or_misaligned_provenance() -> None:
    saved = SavedRevisionTraceProvenance(revision_id=new_opaque_id())
    aggregate = ConsoleRequestProvenance(
        system=(),
        memory=(),
        mandatory=(),
        compactable=(ConsoleUnitProvenance(messages=(saved,)),),
        active_request=(saved,),
        active_thinking=(),
        active_continuations=(),
        tools=(),
        capture_policy=_policy(),
    )

    aggregate.validate_alignment(
        system=0,
        memory=0,
        mandatory=0,
        compactable=((1, 0, 0, 0),),
        active_request=1,
        active_thinking=0,
        active_continuations=0,
        tools=0,
    )
    with pytest.raises(TraceProvenanceAlignmentError, match="compactable"):
        aggregate.validate_alignment(
            system=0,
            memory=0,
            mandatory=0,
            compactable=((2, 0, 0, 0),),
            active_request=1,
            active_thinking=0,
            active_continuations=0,
            tools=0,
        )


def test_build_classifies_provenance_parallel_to_every_semantic_category() -> None:
    policy = _policy()
    messages = [
        {"role": "system", "content": "framing"},
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "active"},
    ]
    descriptors: tuple[TraceProvenance, ...] = (
        _artifact(TraceProvenanceSource.RENDERED_SYSTEM, policy),
        SavedRevisionTraceProvenance(new_opaque_id()),
        SavedRevisionTraceProvenance(new_opaque_id()),
        SavedRevisionTraceProvenance(new_opaque_id()),
    )
    memory = ({"role": "system", "content": "memory"},)
    mandatory = ({"role": "system", "content": "project"},)
    tools = ({"type": "function", "function": {"name": "search"}},)

    semantic = build_console_request(
        messages,
        memory=memory,
        mandatory=mandatory,
        tools=tools,
        message_provenance=descriptors,
        memory_provenance=(
            _artifact(TraceProvenanceSource.CONVERSATION_MEMORY, policy),
        ),
        mandatory_provenance=(
            _artifact(TraceProvenanceSource.PROJECT_INSTRUCTION, policy),
        ),
        tool_provenance=(_artifact(TraceProvenanceSource.TOOL_DEFINITION, policy),),
        capture_policy=policy,
    )

    assert semantic.provenance is not None
    assert semantic.provenance.system == descriptors[:1]
    assert semantic.provenance.compactable[0].messages == descriptors[1:3]
    assert semantic.provenance.active_request == descriptors[3:]
    assert len(semantic.provenance.memory) == 1
    assert len(semantic.provenance.mandatory) == 1
    assert len(semantic.provenance.tools) == 1
    assert semantic.flattened_messages() == tuple(
        messages[:1] + list(memory) + list(mandatory) + messages[1:]
    )


def test_native_tool_loop_provenance_has_its_own_ordered_semantic_category() -> None:
    policy = _policy()
    active = SavedRevisionTraceProvenance(new_opaque_id())
    tool_call = _artifact(TraceProvenanceSource.TOOL_CALL, policy)
    tool_result = _artifact(TraceProvenanceSource.TOOL_RESULT, policy)
    semantic = build_console_request(
        [
            {"role": "user", "content": "Find it"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call-1", "function": {"name": "lookup"}}],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "found"},
        ],
        message_provenance=(active, tool_call, tool_result),
        memory_provenance=(),
        mandatory_provenance=(),
        tool_provenance=(),
        capture_policy=policy,
    )

    assert semantic.provenance is not None
    assert semantic.provenance.active_request == (active, tool_call, tool_result)
    assert semantic.provenance.tool_loop == (tool_call, tool_result)
    assert semantic.provenance.flattened_messages() == (
        active,
        tool_call,
        tool_result,
    )
    prepared = prepare_provider_request(
        semantic,
        wire_style="distinct_roles",
        model="m",
        capacity=resolve_request_capacity(context_window_tokens=None),
        count_fn=_word_count,
    )
    assert prepared.provenance is not None
    assert prepared.provenance.tool_loop == (1, 2)
    assert prepared.provenance.messages == (active, tool_call, tool_result)


@pytest.mark.parametrize(
    "rows",
    (
        (
            {"role": "user", "content": "Find it"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call-1", "function": {"name": "lookup"}}],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "found"},
        ),
        (
            {"role": "user", "content": "Find it"},
            {
                "role": "assistant",
                "content": '```tool_call\n{"name":"lookup","arguments":{}}\n```',
            },
            {"role": "user", "content": "Tool result for lookup: found"},
        ),
        (
            {"role": "user", "content": "Find it"},
            {
                "role": "assistant",
                "content": (
                    "I will use lookup.\n```tool_call\n"
                    '{"name":"lookup","arguments":{}}\n```'
                ),
            },
            {"role": "user", "content": "Tool result for lookup: found"},
        ),
    ),
)
def test_persisted_tool_loop_revisions_remain_saved_message_provenance(
    rows: tuple[dict, ...],
) -> None:
    policy = _policy()
    revisions = tuple(SavedRevisionTraceProvenance(new_opaque_id()) for _ in range(3))
    semantic = build_console_request(
        rows,
        message_provenance=revisions,
        memory_provenance=(),
        mandatory_provenance=(),
        tool_provenance=(),
        capture_policy=policy,
    )

    prepared = prepare_provider_request(
        semantic,
        wire_style="distinct_roles",
        model="m",
        capacity=resolve_request_capacity(context_window_tokens=None),
        count_fn=_word_count,
    )

    assert semantic.provenance is not None
    assert semantic.provenance.tool_loop == revisions[1:]
    assert prepared.provenance is not None
    assert prepared.provenance.messages == revisions
    assert prepared.provenance.tool_loop == (1, 2)


def test_single_preamble_tool_overlay_uses_payload_ordinals() -> None:
    policy = _policy()
    revisions: tuple[TraceProvenance, ...] = (
        _artifact(TraceProvenanceSource.RENDERED_SYSTEM, policy),
        *(SavedRevisionTraceProvenance(new_opaque_id()) for _ in range(3)),
    )
    semantic = build_console_request(
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "Find it"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c"}]},
            {"role": "tool", "tool_call_id": "c", "content": "found"},
        ],
        message_provenance=revisions,
        memory_provenance=(),
        mandatory_provenance=(),
        tool_provenance=(),
        capture_policy=policy,
    )

    prepared = prepare_provider_request(
        semantic,
        wire_style="single_preamble",
        model="m",
        capacity=resolve_request_capacity(context_window_tokens=None),
        count_fn=_word_count,
    )

    assert prepared.provenance is not None
    assert prepared.provenance.messages_payload == revisions[1:]
    assert prepared.provenance.tool_loop == (1, 2)


def test_fenced_tool_overlay_survives_thinking_wire_rewrite() -> None:
    owner_id = "tool-owner"
    policy = _policy()
    revisions = tuple(SavedRevisionTraceProvenance(new_opaque_id()) for _ in range(3))
    semantic = build_console_request(
        [
            {"role": "user", "content": "Find it"},
            {
                "role": "assistant",
                "content": '```tool_call\n{"name":"lookup","arguments":{}}\n```',
                THINKING_OWNER_KEY: owner_id,
            },
            {"role": "user", "content": "Tool result for lookup: found"},
        ],
        thinking_groups=(
            ThinkingOwnerGroup(
                owner_id,
                (
                    ResolvedThinkingBlock(
                        owner_message_id=owner_id,
                        source_format="start_anchored_think",
                        text="private provider thinking",
                    ),
                ),
            ),
        ),
        message_provenance=revisions,
        memory_provenance=(),
        mandatory_provenance=(),
        tool_provenance=(),
        capture_policy=policy,
    )

    prepared = prepare_provider_request(
        semantic,
        wire_style="distinct_roles",
        model="m",
        capacity=resolve_request_capacity(context_window_tokens=None),
        count_fn=_word_count,
    )

    assert prepared.provenance is not None
    assert prepared.provenance.tool_loop == (1, 2)


def test_provider_tool_loop_is_an_exact_ordered_message_overlay() -> None:
    policy = _policy()
    descriptors: tuple[TraceProvenance, ...] = (
        SavedRevisionTraceProvenance(new_opaque_id()),
        _artifact(TraceProvenanceSource.TOOL_CALL, policy),
        _artifact(TraceProvenanceSource.TOOL_RESULT, policy),
    )
    semantic = build_console_request(
        [
            {"role": "user", "content": "Find it"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c"}]},
            {"role": "tool", "tool_call_id": "c", "content": "found"},
        ],
        message_provenance=descriptors,
        memory_provenance=(),
        mandatory_provenance=(),
        tool_provenance=(),
        capture_policy=policy,
    )
    prepared = prepare_provider_request(
        semantic,
        wire_style="distinct_roles",
        model="m",
        capacity=resolve_request_capacity(context_window_tokens=None),
        count_fn=_word_count,
    )
    assert prepared.provenance is not None

    with pytest.raises(TraceProvenanceAlignmentError, match="provider provenance"):
        replace(
            prepared,
            provenance=replace(prepared.provenance, tool_loop=(0, 1)),
        )
    with pytest.raises(TraceProvenanceAlignmentError, match="tool_loop"):
        replace(prepared.provenance, tool_loop=(2, 1))
    for field in ("messages", "messages_payload"):
        swapped = list(getattr(prepared.provenance, field))
        swapped[:2] = reversed(swapped[:2])
        with pytest.raises(TraceProvenanceAlignmentError, match="provider provenance"):
            replace(
                prepared,
                provenance=replace(
                    prepared.provenance,
                    **{field: tuple(swapped)},
                ),
            )
    for field in ("messages", "messages_payload"):
        with pytest.raises(TraceProvenanceAlignmentError, match="provider wire"):
            replace(prepared, **{field: ()})
    with pytest.raises(TraceProvenanceAlignmentError, match="provider wire"):
        replace(
            prepared,
            tools=({"type": "function", "function": {"name": "injected"}},),
        )


def test_capture_off_has_no_partial_provenance_and_wire_is_unchanged() -> None:
    policy = _policy()
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "active"},
    ]
    capacity = resolve_request_capacity(context_window_tokens=None)
    off = build_console_request(messages)
    on = build_console_request(
        messages,
        message_provenance=(
            _artifact(TraceProvenanceSource.RENDERED_SYSTEM, policy),
            SavedRevisionTraceProvenance(new_opaque_id()),
            SavedRevisionTraceProvenance(new_opaque_id()),
            SavedRevisionTraceProvenance(new_opaque_id()),
        ),
        memory_provenance=(),
        mandatory_provenance=(),
        tool_provenance=(),
        capture_policy=policy,
    )

    prepared_off = prepare_provider_request(
        off,
        wire_style="single_preamble",
        model="m",
        capacity=capacity,
        count_fn=_word_count,
    )
    prepared_on = prepare_provider_request(
        on,
        wire_style="single_preamble",
        model="m",
        capacity=capacity,
        count_fn=_word_count,
    )

    assert off.provenance is None
    assert prepared_off.provenance is None
    assert prepared_on.provenance is not None
    assert prepared_off.system_message == prepared_on.system_message
    assert prepared_off.messages == prepared_on.messages
    assert prepared_off.messages_payload == prepared_on.messages_payload
    assert prepared_off.tools == prepared_on.tools
    assert prepared_off.accounting == prepared_on.accounting

    with pytest.raises(TraceProvenanceAlignmentError, match="all or none"):
        build_console_request(
            messages,
            message_provenance=(
                _artifact(TraceProvenanceSource.RENDERED_SYSTEM),
                SavedRevisionTraceProvenance(new_opaque_id()),
                SavedRevisionTraceProvenance(new_opaque_id()),
                SavedRevisionTraceProvenance(new_opaque_id()),
            ),
        )


def test_capture_metadata_survives_without_parallel_rag_values_and_reaches_provider() -> (
    None
):
    route_type = getattr(provenance_module, "RequestRouteTraceProvenance", None)
    assert route_type is not None
    policy = _policy()
    active = SavedRevisionTraceProvenance(new_opaque_id())
    omission = provenance_module.rag_provenance_for_route(
        provenance_module.ConsoleRequestRoute.FRESH,
        None,
    )
    route = route_type(provenance_module.ConsoleRequestRoute.FRESH)
    overlay = DerivedTraceProvenance(
        TraceTransformKind.PROVIDER_OVERLAY,
        (active,),
        artifact=_artifact(TraceProvenanceSource.PROVIDER_OVERLAY, policy),
    )

    semantic = build_console_request(
        [{"role": "user", "content": "active"}],
        message_provenance=(active,),
        memory_provenance=(),
        mandatory_provenance=(),
        tool_provenance=(),
        metadata_provenance=(omission, route, overlay),
        capture_policy=policy,
    )
    prepared = prepare_provider_request(
        semantic,
        wire_style="distinct_roles",
        model="m",
        capacity=resolve_request_capacity(context_window_tokens=None),
        count_fn=_word_count,
    )

    assert semantic.mandatory == ()
    assert semantic.provenance is not None
    assert semantic.provenance.metadata == (omission, route, overlay)
    assert prepared.provenance is not None
    assert prepared.provenance.metadata == (omission, route, overlay)


def test_capture_off_mode_rejects_capture_policy_and_descriptors() -> None:
    capture_mode = getattr(provenance_module, "ConsoleTraceCaptureMode", None)
    assert capture_mode is not None

    with pytest.raises(TraceProvenanceAlignmentError, match="Capture Off"):
        build_console_request(
            [{"role": "user", "content": "active"}],
            message_provenance=(SavedRevisionTraceProvenance(new_opaque_id()),),
            memory_provenance=(),
            mandatory_provenance=(),
            tool_provenance=(),
            capture_policy=_policy(),
            capture_mode=capture_mode.CAPTURE_OFF,
        )


def test_descriptor_categories_are_closed_and_source_compatible() -> None:
    class SavedSubclass(SavedRevisionTraceProvenance):
        pass

    revision = SavedRevisionTraceProvenance(new_opaque_id())
    policy = _policy()
    common = dict(
        system=(),
        mandatory=(),
        compactable=(),
        active_thinking=(),
        active_continuations=(),
        capture_policy=policy,
    )
    with pytest.raises(TypeError, match="closed"):
        ConsoleRequestProvenance(
            **common,
            memory=(),
            active_request=(SavedSubclass(new_opaque_id()),),
            tools=(),
        )
    with pytest.raises(TraceProvenanceAlignmentError, match="memory"):
        ConsoleRequestProvenance(
            **common,
            memory=(revision,),
            active_request=(revision,),
            tools=(),
        )
    with pytest.raises(TraceProvenanceAlignmentError, match="tools"):
        ConsoleRequestProvenance(
            **common,
            memory=(),
            active_request=(revision,),
            tools=(revision,),
        )
    with pytest.raises(TraceProvenanceAlignmentError, match="active_request"):
        ConsoleRequestProvenance(
            **common,
            memory=(),
            active_request=(_artifact(TraceProvenanceSource.RAG_CONTEXT, policy),),
            tools=(),
        )
    with pytest.raises(TraceProvenanceAlignmentError, match="active_request"):
        ConsoleRequestProvenance(
            **common,
            memory=(),
            active_request=(
                DerivedTraceProvenance(
                    TraceTransformKind.PROVIDER_OVERLAY,
                    (revision,),
                    artifact=_artifact(TraceProvenanceSource.PROVIDER_OVERLAY, policy),
                ),
            ),
            tools=(),
        )
    with pytest.raises(TraceProvenanceAlignmentError, match="active_request"):
        ConsoleRequestProvenance(
            **common,
            memory=(),
            active_request=(
                DerivedTraceProvenance(
                    TraceTransformKind.WINDOWING,
                    (
                        revision,
                        _artifact(TraceProvenanceSource.RAG_CONTEXT, policy),
                    ),
                ),
            ),
            tools=(),
        )
    with pytest.raises(TraceProvenanceAlignmentError, match="tools"):
        ProviderRequestProvenance(tools=(revision,))


@pytest.mark.parametrize(
    ("transform", "wrong_source"),
    (
        (
            TraceTransformKind.THINKING_ATTACHMENT,
            TraceProvenanceSource.CONTINUATION,
        ),
        (
            TraceTransformKind.CONTINUATION_ATTACHMENT,
            TraceProvenanceSource.THINKING,
        ),
    ),
)
def test_message_rewrite_rejects_cross_paired_sidecar_artifacts(
    transform: TraceTransformKind,
    wrong_source: TraceProvenanceSource,
) -> None:
    revision = SavedRevisionTraceProvenance(new_opaque_id())
    with pytest.raises(TraceProvenanceAlignmentError, match="contradicts"):
        DerivedTraceProvenance(
            transform,
            (revision,),
            artifact=_artifact(wrong_source),
        )


@pytest.mark.parametrize(
    ("transform", "artifact_source"),
    (
        (TraceTransformKind.THINKING_ATTACHMENT, TraceProvenanceSource.THINKING),
        (
            TraceTransformKind.CONTINUATION_ATTACHMENT,
            TraceProvenanceSource.CONTINUATION,
        ),
    ),
)
def test_saved_sidecar_attachment_rejects_provider_artifact(
    transform: TraceTransformKind,
    artifact_source: TraceProvenanceSource,
) -> None:
    revision = SavedRevisionTraceProvenance(new_opaque_id())

    with pytest.raises(TraceProvenanceAlignmentError, match="attachment"):
        DerivedTraceProvenance(
            transform,
            (revision,),
            artifact=_artifact(artifact_source),
        )


def test_message_rewrite_rejects_cross_owner_attachment_at_public_boundary() -> None:
    owner = SavedRevisionTraceProvenance(new_opaque_id())
    wrong_owner = SavedRevisionTraceProvenance(new_opaque_id())
    attachment = DerivedTraceProvenance(
        TraceTransformKind.THINKING_ATTACHMENT,
        (wrong_owner,),
    )

    with pytest.raises(TraceProvenanceAlignmentError, match="exact owner"):
        DerivedTraceProvenance(
            TraceTransformKind.MESSAGE_REWRITE,
            (owner, attachment),
        )


@pytest.mark.parametrize(
    ("transform", "owner_source"),
    (
        (TraceTransformKind.THINKING_ATTACHMENT, TraceProvenanceSource.ACTIVE_REQUEST),
        (
            TraceTransformKind.CONTINUATION_ATTACHMENT,
            TraceProvenanceSource.ACTIVE_REQUEST,
        ),
    ),
)
def test_synthetic_sidecar_attachment_requires_provider_artifact(
    transform: TraceTransformKind,
    owner_source: TraceProvenanceSource,
) -> None:
    with pytest.raises(TraceProvenanceAlignmentError, match="attachment"):
        DerivedTraceProvenance(transform, (_artifact(owner_source),))


@pytest.mark.parametrize(
    "descriptor",
    (
        lambda policy, revision: DerivedTraceProvenance(
            TraceTransformKind.PROVIDER_OVERLAY,
            (revision,),
            artifact=ProviderArtifactTraceProvenance(
                TraceProvenanceSource.RAG_CONTEXT,
                policy,
            ),
        ),
        lambda policy, revision: DerivedTraceProvenance(
            TraceTransformKind.MESSAGE_REWRITE,
            (revision,),
            artifact=ProviderArtifactTraceProvenance(
                TraceProvenanceSource.PROVIDER_OVERLAY,
                policy,
            ),
        ),
        lambda policy, revision: DerivedTraceProvenance(
            TraceTransformKind.WINDOWING,
            (revision,),
            artifact=ProviderArtifactTraceProvenance(
                TraceProvenanceSource.COMPACTABLE_HISTORY,
                policy,
            ),
        ),
        lambda policy, revision: DerivedTraceProvenance(
            TraceTransformKind.TEXT_COMPACTION,
            (revision,),
            artifact=ProviderArtifactTraceProvenance(
                TraceProvenanceSource.VISUAL_TRANSCRIPT,
                policy,
            ),
        ),
    ),
)
def test_derived_transform_matrix_rejects_contradictory_artifact_shapes(
    descriptor,
) -> None:
    with pytest.raises(TraceProvenanceAlignmentError, match="transform"):
        descriptor(_policy(), SavedRevisionTraceProvenance(new_opaque_id()))


def test_compaction_and_window_metadata_are_bounded_and_semantically_exact() -> None:
    policy = _policy()
    memory = _artifact(TraceProvenanceSource.CONVERSATION_MEMORY, policy)
    message = SavedRevisionTraceProvenance(new_opaque_id())
    thinking = _artifact(TraceProvenanceSource.THINKING, policy)
    continuation = _artifact(TraceProvenanceSource.CONTINUATION, policy)
    provenance = ConsoleRequestProvenance(
        system=(),
        memory=(memory,),
        mandatory=(),
        compactable=(
            ConsoleUnitProvenance(
                messages=(message,),
                thinking=(thinking,),
                continuations=(continuation,),
            ),
        ),
        active_request=(SavedRevisionTraceProvenance(new_opaque_id()),),
        active_thinking=(),
        active_continuations=(),
        tools=(),
        capture_policy=policy,
    )

    compacted = compaction_transform_provenance(
        provenance,
        selected_units=1,
        transform=TraceTransformKind.TEXT_COMPACTION,
        source=TraceProvenanceSource.CONTEXT_SUMMARY,
    )
    assert compacted.inputs == (memory, message)

    large = replace(
        provenance,
        memory=(),
        compactable=(
            ConsoleUnitProvenance(
                messages=tuple(
                    SavedRevisionTraceProvenance(new_opaque_id())
                    for _ in range(MAX_PROVENANCE_TRANSFORM_INPUTS + 9)
                )
            ),
        ),
    )
    bounded_compaction = compaction_transform_provenance(
        large,
        selected_units=1,
        transform=TraceTransformKind.TEXT_COMPACTION,
        source=TraceProvenanceSource.CONTEXT_SUMMARY,
    )
    assert len(bounded_compaction.inputs) == MAX_PROVENANCE_TRANSFORM_INPUTS
    compaction_omission = bounded_compaction.inputs[-1]
    assert isinstance(compaction_omission, OmittedTraceProvenance)
    assert (
        compaction_omission.reason is TraceOmissionReason.UNSUPPORTED_REPLACEMENT_SPAN
    )
    assert compaction_omission.omitted_count == 10

    windowed = large.without_oldest_units(1)
    assert windowed.compactable == ()
    window = windowed.metadata[-1]
    assert isinstance(window, DerivedTraceProvenance)
    assert window.transform is TraceTransformKind.WINDOWING
    assert len(window.inputs) == MAX_PROVENANCE_TRANSFORM_INPUTS
    window_omission = window.inputs[-1]
    assert isinstance(window_omission, OmittedTraceProvenance)
    assert window_omission.reason is TraceOmissionReason.WINDOWED
    assert window_omission.omitted_count == 10

    with pytest.raises(TraceProvenanceAlignmentError, match="selected_units"):
        compaction_transform_provenance(
            provenance,
            selected_units=-1,
            transform=TraceTransformKind.TEXT_COMPACTION,
            source=TraceProvenanceSource.CONTEXT_SUMMARY,
        )

    repeatedly_windowed = replace(
        provenance,
        memory=(),
        compactable=tuple(
            ConsoleUnitProvenance(
                messages=(SavedRevisionTraceProvenance(new_opaque_id()),)
            )
            for _ in range(MAX_PROVENANCE_TRANSFORM_INPUTS + 1)
        ),
    )
    for _ in range(MAX_PROVENANCE_TRANSFORM_INPUTS + 1):
        repeatedly_windowed = repeatedly_windowed.without_oldest_units(1)
    assert len(repeatedly_windowed.metadata) == MAX_PROVENANCE_TRANSFORM_INPUTS
    accumulated_omission = repeatedly_windowed.metadata[0]
    assert isinstance(accumulated_omission, OmittedTraceProvenance)
    assert accumulated_omission.source is TraceProvenanceSource.COMPACTABLE_HISTORY
    assert accumulated_omission.reason is TraceOmissionReason.WINDOWED
    assert accumulated_omission.omitted_count == 2

    multi_message_windowed = replace(
        provenance,
        memory=(),
        compactable=tuple(
            ConsoleUnitProvenance(
                messages=(
                    SavedRevisionTraceProvenance(new_opaque_id()),
                    SavedRevisionTraceProvenance(new_opaque_id()),
                )
            )
            for _ in range(MAX_PROVENANCE_TRANSFORM_INPUTS + 1)
        ),
    )
    for _ in range(MAX_PROVENANCE_TRANSFORM_INPUTS + 1):
        multi_message_windowed = multi_message_windowed.without_oldest_units(1)
    multi_omission = multi_message_windowed.metadata[0]
    assert isinstance(multi_omission, OmittedTraceProvenance)
    assert multi_omission.omitted_count == 4


def test_windowing_and_single_preamble_keep_final_descriptors_aligned() -> None:
    policy = _policy()
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "old one two three four"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "new"},
        {"role": "assistant", "content": "new answer"},
        {"role": "user", "content": "active"},
    ]
    descriptors: tuple[TraceProvenance, ...] = (
        _artifact(TraceProvenanceSource.RENDERED_SYSTEM, policy),
        *(SavedRevisionTraceProvenance(new_opaque_id()) for _ in range(5)),
    )
    semantic = build_console_request(
        messages,
        message_provenance=descriptors,
        memory_provenance=(),
        mandatory_provenance=(),
        tool_provenance=(),
        capture_policy=policy,
    )
    capacity = resolve_request_capacity(
        context_window_tokens=520,
        requested_response_tokens=1,
    )

    prepared = prepare_provider_request(
        semantic,
        wire_style="single_preamble",
        model="m",
        capacity=capacity,
        count_fn=_word_count,
    )

    assert prepared.dropped_units == 2
    assert prepared.semantic.provenance is not None
    assert prepared.semantic.provenance.compactable == ()
    assert prepared.provenance is not None
    assert len(prepared.provenance.messages_payload) == len(prepared.messages_payload)
    assert len(prepared.provenance.messages) == len(prepared.messages)
    assert isinstance(
        prepared.provenance.messages[0],
        DerivedTraceProvenance,
    )
    assert (
        prepared.provenance.messages[0].transform is TraceTransformKind.SINGLE_PREAMBLE
    )


def test_visual_memory_role_rewrite_keeps_payload_provenance_aligned() -> None:
    page = b"deterministic-png-fixture"
    policy = _policy()
    visual = tagged_visual_memory_message(
        (page,),
        page_hashes=(hashlib.sha256(page).hexdigest(),),
    )
    semantic = build_console_request(
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "active"},
        ],
        memory=(visual,),
        message_provenance=(
            ProviderArtifactTraceProvenance(
                TraceProvenanceSource.RENDERED_SYSTEM,
                policy,
            ),
            SavedRevisionTraceProvenance(new_opaque_id()),
        ),
        memory_provenance=(
            ProviderArtifactTraceProvenance(
                TraceProvenanceSource.VISUAL_TRANSCRIPT,
                policy,
            ),
        ),
        mandatory_provenance=(),
        tool_provenance=(),
        capture_policy=policy,
    )

    prepared = prepare_provider_request(
        semantic,
        wire_style="single_preamble",
        model="m",
        capacity=resolve_request_capacity(context_window_tokens=None),
        count_fn=_word_count,
    )

    assert prepared.messages_payload[0]["role"] == "user"
    assert prepared.provenance is not None
    assert len(prepared.provenance.messages_payload) == len(prepared.messages_payload)


def test_compaction_replacement_constructor_keeps_capture_on_alignment() -> None:
    policy = _policy()
    semantic = build_console_request(
        [
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "active"},
        ],
        message_provenance=(
            SavedRevisionTraceProvenance(new_opaque_id()),
            SavedRevisionTraceProvenance(new_opaque_id()),
            SavedRevisionTraceProvenance(new_opaque_id()),
        ),
        memory_provenance=(),
        mandatory_provenance=(),
        tool_provenance=(),
        capture_policy=policy,
    )
    assert semantic.provenance is not None
    summary = compaction_transform_provenance(
        semantic.provenance,
        selected_units=1,
        transform=TraceTransformKind.TEXT_COMPACTION,
        source=TraceProvenanceSource.CONTEXT_SUMMARY,
    )
    remaining = semantic.without_oldest_units(1)
    assert remaining.provenance is not None
    after = replace(
        remaining,
        memory=(tagged_memory_message("summary"),),
        provenance=replace(remaining.provenance, memory=(summary,)),
    )

    prepared = prepare_provider_request(
        after,
        wire_style="single_preamble",
        model="m",
        capacity=resolve_request_capacity(context_window_tokens=None),
        count_fn=_word_count,
    )

    assert prepared.provenance is not None
    assert prepared.semantic.provenance is not None
    assert prepared.semantic.provenance.memory == (summary,)
    assert summary.artifact is not None
    assert summary.artifact.source is TraceProvenanceSource.CONTEXT_SUMMARY
    assert summary.artifact.policy is policy


def test_explicit_prepared_request_constructor_rejects_missing_on_provenance() -> None:
    policy = _policy()
    semantic = build_console_request(
        [{"role": "user", "content": "active"}],
        message_provenance=(SavedRevisionTraceProvenance(new_opaque_id()),),
        memory_provenance=(),
        mandatory_provenance=(),
        tool_provenance=(),
        capture_policy=policy,
    )
    prepared = prepare_provider_request(
        semantic,
        wire_style="distinct_roles",
        model="m",
        capacity=resolve_request_capacity(context_window_tokens=None),
        count_fn=_word_count,
    )

    values = {
        field: getattr(prepared, field)
        for field in prepared.__dataclass_fields__
        if field != "provenance"
    }
    with pytest.raises(TraceProvenanceAlignmentError, match="missing"):
        type(prepared)(**values)


def test_saved_thinking_and_continuation_sidecars_share_owner_revision() -> None:
    owner_id = "assistant-owner"
    thinking = ThinkingOwnerGroup(
        owner_id,
        (
            ResolvedThinkingBlock(
                owner_message_id=owner_id,
                source_format="start_anchored_think",
                text="private provider thinking",
            ),
        ),
    )
    checkpoint = parse_provider_continuation_json(
        {
            "schema_version": 1,
            "checkpoint_revision": 1,
            "provider": "deepseek",
            "protocol": "responses",
            "model": "deepseek-v4-flash",
            "api_base_url": "https://api.deepseek.com/v1",
            "state": "active",
            "rounds": [
                {
                    "assistant_content": "",
                    "reasoning_blocks": ["private provider reasoning"],
                    "calls": [
                        {
                            "call_id": "call_1",
                            "name": "calculator",
                            "arguments": '{"expression":"2+2"}',
                            "state": "pending",
                        }
                    ],
                }
            ],
        }
    )
    continuation = continuation_owner_group(
        {"id": owner_id, "role": "assistant", "content": ""},
        checkpoint,
    )
    policy = _policy()
    owner_revision = SavedRevisionTraceProvenance(new_opaque_id())
    semantic = build_console_request(
        [
            {
                "role": "assistant",
                "content": "answer",
                THINKING_OWNER_KEY: owner_id,
                CONTINUATION_OWNER_KEY: owner_id,
            },
            {"role": "user", "content": "active"},
        ],
        thinking_groups=(thinking,),
        continuation_groups=(continuation,),
        message_provenance=(
            owner_revision,
            SavedRevisionTraceProvenance(new_opaque_id()),
        ),
        memory_provenance=(),
        mandatory_provenance=(),
        tool_provenance=(),
        capture_policy=policy,
    )

    assert semantic.provenance is not None
    unit = semantic.provenance.compactable[0]
    thinking_intent = unit.thinking[0]
    continuation_intent = unit.continuations[0]
    assert isinstance(thinking_intent, DerivedTraceProvenance)
    assert thinking_intent.inputs == (owner_revision,)
    assert thinking_intent.artifact is None
    assert isinstance(continuation_intent, DerivedTraceProvenance)
    assert continuation_intent.inputs == (owner_revision,)
    assert continuation_intent.artifact is None

    prepared = prepare_provider_request(
        semantic,
        wire_style="distinct_roles",
        model="m",
        capacity=resolve_request_capacity(context_window_tokens=None),
        count_fn=_word_count,
    )
    assert prepared.provenance is not None
    rewritten = prepared.provenance.messages[0]
    assert isinstance(rewritten, DerivedTraceProvenance)
    assert rewritten.transform is TraceTransformKind.MESSAGE_REWRITE
    assert rewritten.inputs == (
        owner_revision,
        thinking_intent,
        continuation_intent,
    )
    assert prepared.provenance.thinking == (thinking_intent,)
    assert prepared.provenance.continuations == (continuation_intent,)
    with pytest.raises(TraceProvenanceAlignmentError, match="provider wire"):
        replace(prepared, continuation_groups=())
    with pytest.raises(TraceProvenanceAlignmentError, match="provider wire"):
        replace(prepared, thinking_groups=())


@pytest.mark.parametrize(
    ("sidecar_field", "transform"),
    (
        ("thinking", TraceTransformKind.THINKING_ATTACHMENT),
        ("continuations", TraceTransformKind.CONTINUATION_ATTACHMENT),
    ),
)
def test_provider_serialization_rejects_correct_source_for_wrong_sidecar_owner(
    sidecar_field: str,
    transform: TraceTransformKind,
) -> None:
    owner_id = "assistant-owner"
    thinking = ThinkingOwnerGroup(
        owner_id,
        (
            ResolvedThinkingBlock(
                owner_message_id=owner_id,
                source_format="start_anchored_think",
                text="private provider thinking",
            ),
        ),
    )
    checkpoint = parse_provider_continuation_json(
        {
            "schema_version": 1,
            "checkpoint_revision": 1,
            "provider": "deepseek",
            "protocol": "responses",
            "model": "deepseek-v4-flash",
            "api_base_url": "https://api.deepseek.com/v1",
            "state": "active",
            "rounds": [
                {
                    "assistant_content": "",
                    "reasoning_blocks": [],
                    "calls": [
                        {
                            "call_id": "call_1",
                            "name": "calculator",
                            "arguments": '{"expression":"2+2"}',
                            "state": "pending",
                        }
                    ],
                }
            ],
        }
    )
    continuation = continuation_owner_group(
        {"id": owner_id, "role": "assistant", "content": ""},
        checkpoint,
    )
    owner_revision = SavedRevisionTraceProvenance(new_opaque_id())
    wrong_owner_revision = SavedRevisionTraceProvenance(new_opaque_id())
    semantic = build_console_request(
        [
            {
                "role": "assistant",
                "content": "answer",
                THINKING_OWNER_KEY: owner_id,
                CONTINUATION_OWNER_KEY: owner_id,
            },
            {"role": "user", "content": "active"},
        ],
        thinking_groups=(thinking,),
        continuation_groups=(continuation,),
        message_provenance=(owner_revision, wrong_owner_revision),
        memory_provenance=(),
        mandatory_provenance=(),
        tool_provenance=(),
        capture_policy=_policy(),
    )
    assert semantic.provenance is not None
    unit = semantic.provenance.compactable[0]
    wrong_attachment = DerivedTraceProvenance(transform, (wrong_owner_revision,))
    corrupted_unit = replace(unit, **{sidecar_field: (wrong_attachment,)})
    corrupted = replace(
        semantic,
        provenance=replace(semantic.provenance, compactable=(corrupted_unit,)),
    )

    with pytest.raises(TraceProvenanceAlignmentError, match="owner"):
        prepare_provider_request(
            corrupted,
            wire_style="distinct_roles",
            model="m",
            capacity=resolve_request_capacity(context_window_tokens=None),
            count_fn=_word_count,
        )


def test_synthetic_thinking_sidecar_keeps_provider_artifact_intent() -> None:
    owner_id = "synthetic-owner"
    policy = _policy()
    owner = _artifact(TraceProvenanceSource.ACTIVE_REQUEST, policy)
    thinking = ThinkingOwnerGroup(
        owner_id,
        (
            ResolvedThinkingBlock(
                owner_message_id=owner_id,
                source_format="start_anchored_think",
                text="synthetic provider thinking",
            ),
        ),
    )

    semantic = build_console_request(
        [
            {
                "role": "assistant",
                "content": "synthetic",
                THINKING_OWNER_KEY: owner_id,
            },
            {"role": "user", "content": "active"},
        ],
        thinking_groups=(thinking,),
        message_provenance=(owner, SavedRevisionTraceProvenance(new_opaque_id())),
        memory_provenance=(),
        mandatory_provenance=(),
        tool_provenance=(),
        capture_policy=policy,
    )

    assert semantic.provenance is not None
    intent = semantic.provenance.compactable[0].thinking[0]
    assert isinstance(intent, DerivedTraceProvenance)
    assert intent.inputs == (owner,)
    assert intent.artifact is not None
    assert intent.artifact.source is TraceProvenanceSource.THINKING

    prepared = prepare_provider_request(
        semantic,
        wire_style="distinct_roles",
        model="m",
        capacity=resolve_request_capacity(context_window_tokens=None),
        count_fn=_word_count,
    )
    assert prepared.provenance is not None
    rewritten = prepared.provenance.messages[0]
    assert isinstance(rewritten, DerivedTraceProvenance)
    assert rewritten.transform is TraceTransformKind.MESSAGE_REWRITE
    assert rewritten.inputs == (owner, intent)


def test_saved_input_inside_provider_composite_does_not_own_sidecar() -> None:
    owner_id = "mixed-owner"
    policy = _policy()
    owner = DerivedTraceProvenance(
        TraceTransformKind.WINDOWING,
        (SavedRevisionTraceProvenance(new_opaque_id()),),
    )
    thinking = ThinkingOwnerGroup(
        owner_id,
        (
            ResolvedThinkingBlock(
                owner_message_id=owner_id,
                source_format="start_anchored_think",
                text="provider-owned thinking",
            ),
        ),
    )

    semantic = build_console_request(
        [
            {
                "role": "assistant",
                "content": "mixed",
                THINKING_OWNER_KEY: owner_id,
            },
            {"role": "user", "content": "active"},
        ],
        thinking_groups=(thinking,),
        message_provenance=(owner, SavedRevisionTraceProvenance(new_opaque_id())),
        memory_provenance=(),
        mandatory_provenance=(),
        tool_provenance=(),
        capture_policy=policy,
    )

    assert semantic.provenance is not None
    intent = semantic.provenance.compactable[0].thinking[0]
    assert isinstance(intent, DerivedTraceProvenance)
    assert intent.artifact is not None
    assert intent.artifact.source is TraceProvenanceSource.THINKING


def test_post_build_thinking_attachment_updates_parallel_provenance() -> None:
    owner_id = "assistant-post-build"
    owner_key = "_temporary_owner"
    policy = _policy()
    semantic = build_console_request(
        [
            {
                "role": "assistant",
                "content": "answer",
                owner_key: owner_id,
            },
            {"role": "user", "content": "active"},
        ],
        message_provenance=(
            SavedRevisionTraceProvenance(new_opaque_id()),
            SavedRevisionTraceProvenance(new_opaque_id()),
        ),
        memory_provenance=(),
        mandatory_provenance=(),
        tool_provenance=(),
        capture_policy=policy,
    )
    group = ThinkingOwnerGroup(
        owner_id,
        (
            ResolvedThinkingBlock(
                owner_message_id=owner_id,
                source_format="start_anchored_think",
                text="private provider thinking",
            ),
        ),
    )

    attached = attach_thinking_history(
        semantic,
        groups=(group,),
        owner_key=owner_key,
        thinking_policy="include",
        effective_thinking_policy="include",
    )

    assert attached.provenance is not None
    intent = attached.provenance.compactable[0].thinking[0]
    assert isinstance(intent, DerivedTraceProvenance)
    assert intent.artifact is None
    assert intent.inputs == (semantic.provenance.compactable[0].messages[0],)


def test_repeated_attachment_reorders_semantics_and_descriptors_by_message_owner() -> (
    None
):
    owner_key = "_temporary_owner"
    owner_ids = ("assistant-a", "assistant-b")
    groups = tuple(
        ThinkingOwnerGroup(
            owner_id,
            (
                ResolvedThinkingBlock(
                    owner_message_id=owner_id,
                    source_format="start_anchored_think",
                    text=f"thinking-{owner_id}",
                ),
            ),
        )
        for owner_id in owner_ids
    )
    checkpoint = parse_provider_continuation_json(
        {
            "schema_version": 1,
            "checkpoint_revision": 1,
            "provider": "deepseek",
            "protocol": "responses",
            "model": "deepseek-v4-flash",
            "api_base_url": "https://api.deepseek.com/v1",
            "state": "active",
            "rounds": [
                {
                    "assistant_content": "",
                    "reasoning_blocks": [],
                    "calls": [
                        {
                            "call_id": "call_1",
                            "name": "calculator",
                            "arguments": '{"expression":"2+2"}',
                            "state": "pending",
                        }
                    ],
                }
            ],
        }
    )
    continuations = tuple(
        continuation_owner_group(
            {"id": owner_id, "role": "assistant", "content": ""},
            checkpoint,
        )
        for owner_id in owner_ids
    )
    revisions = tuple(SavedRevisionTraceProvenance(new_opaque_id()) for _ in range(3))
    semantic = build_console_request(
        [
            {
                "role": "assistant",
                "content": "answer-a",
                owner_key: owner_ids[0],
                THINKING_OWNER_KEY: owner_ids[0],
                CONTINUATION_OWNER_KEY: owner_ids[0],
            },
            {
                "role": "assistant",
                "content": "answer-b",
                owner_key: owner_ids[1],
                THINKING_OWNER_KEY: owner_ids[1],
                CONTINUATION_OWNER_KEY: owner_ids[1],
            },
            {"role": "user", "content": "active"},
        ],
        thinking_groups=groups,
        continuation_groups=continuations,
        message_provenance=revisions,
        memory_provenance=(),
        mandatory_provenance=(),
        tool_provenance=(),
        capture_policy=_policy(),
    )
    assert semantic.provenance is not None
    original_unit = semantic.compactable[0]
    original_provenance = semantic.provenance.compactable[0]
    reversed_request = replace(
        semantic,
        compactable=(
            replace(
                original_unit,
                thinking_groups=tuple(reversed(original_unit.thinking_groups)),
                continuation_groups=tuple(reversed(original_unit.continuation_groups)),
            ),
        ),
        provenance=replace(
            semantic.provenance,
            compactable=(
                replace(
                    original_provenance,
                    thinking=tuple(reversed(original_provenance.thinking)),
                    continuations=tuple(reversed(original_provenance.continuations)),
                ),
            ),
        ),
    )

    attached = attach_thinking_history(
        reversed_request,
        groups=groups,
        owner_key=owner_key,
        thinking_policy="include",
        effective_thinking_policy="include",
    )

    assert (
        tuple(
            group.owner_message_id for group in attached.compactable[0].thinking_groups
        )
        == owner_ids
    )
    assert (
        tuple(
            group.owner_message_id
            for group in attached.compactable[0].continuation_groups
        )
        == owner_ids
    )
    prepared = prepare_provider_request(
        attached,
        wire_style="distinct_roles",
        model="m",
        capacity=resolve_request_capacity(context_window_tokens=None),
        count_fn=_word_count,
    )
    assert prepared.provenance is not None
    for rewritten, revision in zip(prepared.provenance.messages[:2], revisions[:2]):
        assert isinstance(rewritten, DerivedTraceProvenance)
        assert rewritten.inputs[0] is revision
        assert all(
            isinstance(attachment, DerivedTraceProvenance)
            and attachment.inputs == (revision,)
            for attachment in rewritten.inputs[1:]
        )


def test_genuine_v54_saved_rows_are_lazily_admitted_once(tmp_path) -> None:
    path = tmp_path / "genuine-v54-provenance.sqlite"
    with genuine_v54_database(path) as historical:
        conversation_id = historical.add_conversation({"title": "legacy"})
        assert conversation_id is not None
        message_id = historical.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "user",
                "role": "user",
                "content": "ordinary saved bytes",
            }
        )
        assert message_id is not None

    db = CharactersRAGDB(path, client_id="provenance-admission")
    coordinator = SemanticRevisionCoordinator(db)
    try:
        with trace_provenance_admission_transaction(db) as cursor:
            first = admit_message_provenance(
                cursor,
                coordinator=coordinator,
                message_ids=(message_id,),
            )
            second = admit_message_provenance(
                cursor,
                coordinator=coordinator,
                message_ids=(message_id,),
            )
        assert first == second
        assert isinstance(first[0], SavedRevisionTraceProvenance)
        assert first[0].revision_id == second[0].revision_id
        with db.transaction() as cursor:
            rows = cursor.execute(
                """SELECT revision_id, live_message_id
                     FROM console_trace_semantic_revisions
                    WHERE source_message_id = ?""",
                (message_id,),
            ).fetchall()
        assert [tuple(row) for row in rows] == [(first[0].revision_id, message_id)]
    finally:
        db.close_connection()


def test_genuine_v54_nested_admission_returns_no_descriptor_before_outer_commit(
    tmp_path,
) -> None:
    path = tmp_path / "genuine-v54-nested-provenance.sqlite"
    with genuine_v54_database(path) as historical:
        message_id = historical.execute_query(
            "SELECT id FROM messages ORDER BY rowid LIMIT 1"
        ).fetchone()[0]

    db = CharactersRAGDB(path, client_id="provenance-nested-admission")
    coordinator = SemanticRevisionCoordinator(db)
    descriptor = None
    try:
        with pytest.raises(RuntimeError, match="force outer rollback"):
            with db.transaction(immediate=True):
                with pytest.raises(TraceProvenancePersistenceError) as raised:
                    with trace_provenance_admission_transaction(db) as cursor:
                        descriptor = admit_message_provenance(
                            cursor,
                            coordinator=coordinator,
                            message_ids=(message_id,),
                        )
                assert str(raised.value) == "trace_provenance_persistence_failed"
                assert raised.value.__context__ is None
                assert descriptor is None
                raise RuntimeError("force outer rollback")

        with db.transaction() as cursor:
            count = cursor.execute(
                "SELECT COUNT(*) FROM console_trace_semantic_revisions"
            ).fetchone()[0]
        assert count == 0
    finally:
        db.close_connection()


def test_admission_failure_is_typed_content_free_and_rolls_back(tmp_path) -> None:
    db = CharactersRAGDB(tmp_path / "rollback.sqlite", "provenance-rollback")
    conversation_id = db.add_conversation({"title": "rollback"})
    assert conversation_id is not None
    message_id = db._generate_uuid()
    now = db._get_current_utc_timestamp_iso()
    with db.transaction(immediate=True) as cursor:
        cursor.execute(
            """INSERT INTO messages(
                   id, conversation_id, sender, content, timestamp,
                   last_modified, client_id, version, deleted, role)
                 VALUES (?, ?, 'user', ?, ?, ?, ?, 1, 0, 'user')""",
            (
                message_id,
                conversation_id,
                "never leak this failure fixture",
                now,
                now,
                db.client_id,
            ),
        )

    class FailingCoordinator(SemanticRevisionCoordinator):
        def ensure_current_revision(
            self, cursor, *, message_id, creation_reason="legacy_reference"
        ):
            super().ensure_current_revision(
                cursor,
                message_id=message_id,
                creation_reason=creation_reason,
            )
            raise RuntimeError("never leak this failure fixture")

    with pytest.raises(TraceProvenancePersistenceError) as raised:
        with trace_provenance_admission_transaction(db) as cursor:
            admit_message_provenance(
                cursor,
                coordinator=FailingCoordinator(db),
                message_ids=(message_id,),
            )
    assert str(raised.value) == "trace_provenance_persistence_failed"
    assert "never leak" not in repr(raised.value)
    assert raised.value.__context__ is None
    with db.transaction() as cursor:
        assert (
            cursor.execute(
                "SELECT COUNT(*) FROM console_trace_semantic_revisions"
            ).fetchone()[0]
            == 0
        )
    db.close_connection()


def test_admission_commit_failure_is_sanitized_outside_exception_context() -> None:
    canary = "PRIVATE-COMMIT-FAILURE-CANARY"

    class CommitFailingDatabase:
        @contextmanager
        def transaction(self, *, immediate: bool = False):
            assert immediate is True
            yield object()
            raise RuntimeError(canary)

    with pytest.raises(TraceProvenancePersistenceError) as raised:
        with trace_provenance_admission_transaction(CommitFailingDatabase()):
            pass

    assert str(raised.value) == "trace_provenance_persistence_failed"
    assert canary not in repr(raised.value)
    assert raised.value.__context__ is None


def test_typed_admission_failure_discards_private_cause_and_context() -> None:
    canary = "PRIVATE-TYPED-FAILURE-CANARY"

    class TypedFailingDatabase:
        @contextmanager
        def transaction(self, *, immediate: bool = False):
            assert immediate is True
            try:
                raise RuntimeError(canary)
            except RuntimeError as error:
                failure = TraceProvenancePersistenceError()
                failure.private = "PRIVATE-ATTR-CANARY"
                failure.add_note("PRIVATE-NOTE-CANARY")
                raise failure from error
            yield object()  # pragma: no cover

    with pytest.raises(TraceProvenancePersistenceError) as raised:
        with trace_provenance_admission_transaction(TypedFailingDatabase()):
            pass

    assert str(raised.value) == "trace_provenance_persistence_failed"
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert getattr(raised.value, "private", None) is None
    assert getattr(raised.value, "__notes__", None) is None
