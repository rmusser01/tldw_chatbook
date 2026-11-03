"""Literal, exact-owner mix-in composition."""

from copy import deepcopy
from dataclasses import replace

from tldw_chatbook.Chat.console_prompt_mixins import (
    MixinAttemptSnapshot,
    MixinSnapshot,
    MixinSourceRef,
)
from tldw_chatbook.Prompt_Management.prompt_artifact_models import (
    LiteralMixinDefinition,
)

REF = MixinSourceRef("profile-test", "local", "00000000-0000-4000-8000-000000000001")


def snapshot(body="  {style}\n", placement="before"):
    return MixinAttemptSnapshot(
        (MixinSnapshot(REF, 1, "Before", LiteralMixinDefinition(body, placement)),)
    )


def test_before_user_after_preserves_braces_and_whitespace():
    from tldw_chatbook.Chat.console_prompt_mixins import compose_mixin_content

    before = snapshot()
    assert compose_mixin_content("message", before) == "  {style}\n\n\nmessage"
    after = MixinSnapshot(
        replace(REF, record_uuid="00000000-0000-4000-8000-000000000002"),
        1,
        "After",
        LiteralMixinDefinition(" /command {{variable}} ", "after"),
    )
    assert (
        compose_mixin_content("message", MixinAttemptSnapshot((*before.items, after)))
        == "  {style}\n\n\nmessage\n\n /command {{variable}} "
    )
    assert compose_mixin_content("message", MixinAttemptSnapshot()) == "message"


def test_multipart_preserves_original_parts_and_does_not_mutate():
    from tldw_chatbook.Chat.console_prompt_mixins import compose_mixin_content

    parts = [
        {"type": "text", "text": "message"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
        {"type": "image", "data": b"png"},
    ]
    original = deepcopy(parts)
    after = replace(
        snapshot("after", "after").items[0],
        source=replace(REF, record_uuid="00000000-0000-4000-8000-000000000002"),
    )
    assert compose_mixin_content(
        parts, MixinAttemptSnapshot((*snapshot("before").items, after))
    ) == [
        {"type": "text", "text": "before"},
        *original,
        {"type": "text", "text": "after"},
    ]
    assert parts == original
    assert compose_mixin_content(parts, MixinAttemptSnapshot()) is parts


def test_exact_anchor_is_active_and_repreparation_does_not_wrap_again():
    from tldw_chatbook.Chat.console_prepared_request import (
        MIXIN_ATTEMPT_KEY,
        build_console_request,
        compose_current_mixin_messages,
        prepare_provider_request,
        resolve_request_capacity,
        thaw_json,
    )

    messages = [
        {"role": "user", "content": "old", "_native_message_id": "old"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "current", "_native_message_id": "current"},
        {"role": "user", "content": "tool or hook context"},
    ]
    composed = compose_current_mixin_messages(
        messages, snapshot("literal"), user_message_id="current", attempt_id="attempt"
    )
    assert (
        compose_current_mixin_messages(
            composed,
            snapshot("literal"),
            user_message_id="current",
            attempt_id="attempt",
        )
        is composed
    )
    assert messages[2]["content"] == "current"
    request = build_console_request(composed)
    assert request.active_request[0]["content"] == "literal\n\ncurrent"
    assert request.active_request[1]["content"] == "tool or hook context"
    capacity = resolve_request_capacity(
        context_window_tokens=2000,
        context_window_verified=True,
        requested_response_tokens=100,
    )
    prepared = prepare_provider_request(
        request, wire_style="distinct_roles", model="test", capacity=capacity
    )
    again = prepare_provider_request(
        prepared.semantic, wire_style="distinct_roles", model="test", capacity=capacity
    )
    assert prepared.messages_payload == again.messages_payload
    assert all(
        MIXIN_ATTEMPT_KEY not in row and "_native_message_id" not in row
        for row in again.messages_payload
    )
    assert thaw_json(again.messages_payload)[2]["content"].count("literal") == 1
    assert again.accounting.active_request_tokens > 0


def test_unknown_anchor_refuses_instead_of_using_last_user():
    import pytest

    from tldw_chatbook.Chat.console_prepared_request import (
        compose_current_mixin_messages,
    )
    from tldw_chatbook.Chat.console_prompt_mixins import MixinSnapshotUnavailableError

    with pytest.raises(MixinSnapshotUnavailableError, match="mixin_anchor_unavailable"):
        compose_current_mixin_messages(
            [{"role": "user", "content": "hook"}],
            snapshot(),
            user_message_id="missing",
            attempt_id="attempt",
        )


def test_agent_tool_loop_preserves_mixin_owner_and_counts_body_once():
    from uuid import uuid4

    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentTraceRequestFactory
    from tldw_chatbook.Chat.console_prepared_request import (
        build_console_request,
        compose_current_mixin_messages,
        prepare_provider_request,
        resolve_request_capacity,
        thaw_json,
    )
    from tldw_chatbook.Chat.console_trace_provenance import (
        ConsoleRequestRoute,
        ConsoleTraceCaptureMode,
        DerivedTraceProvenance,
        FrozenTracePolicy,
        ProviderArtifactTraceProvenance,
        SavedRevisionTraceProvenance,
        TraceProvenanceSource,
        TraceTransformKind,
    )

    policy = FrozenTracePolicy(str(uuid4()), "credentials-v1", False, None)
    rows = compose_current_mixin_messages(
        [{"role": "user", "content": "input", "_native_message_id": "user"}],
        snapshot("literal"),
        user_message_id="user",
        attempt_id="attempt",
    )
    descriptor = DerivedTraceProvenance(
        TraceTransformKind.PROMPT_MIXINS,
        (SavedRevisionTraceProvenance(str(uuid4())),),
        ProviderArtifactTraceProvenance(TraceProvenanceSource.PROMPT_MIXINS, policy),
    )
    admitted = build_console_request(
        rows,
        message_provenance=(descriptor,),
        memory_provenance=(),
        mandatory_provenance=(),
        tool_provenance=(),
        capture_policy=policy,
        capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
    )
    factory = ConsoleAgentTraceRequestFactory(admitted)
    tool_rows = [
        *thaw_json(admitted.flattened_messages()),
        {
            "role": "assistant",
            "content": '```tool_calls\n[{"name":"calculator","arguments":{"expression":"6*7"}}]\n```',
        },
        {"role": "tool", "content": "42", "tool_call_id": "call"},
    ]
    request = factory.build(
        tool_rows,
        tools=(),
        route=ConsoleRequestRoute.TOOL_LOOP,
        actor_id=str(uuid4()),
        chain_id=str(uuid4()),
    )
    assert request.provenance.active_request[0] == descriptor
    capacity = resolve_request_capacity(
        context_window_tokens=1000,
        context_window_verified=True,
        requested_response_tokens=50,
    )
    first = prepare_provider_request(
        request, wire_style="distinct_roles", model="test", capacity=capacity
    )
    again = prepare_provider_request(
        first.semantic, wire_style="distinct_roles", model="test", capacity=capacity
    )
    assert first.messages_payload == again.messages_payload
    assert first.accounting.total_input_tokens == again.accounting.total_input_tokens
    assert again.messages_payload[0]["content"] == "literal\n\ninput"
    assert all(
        not any(key.startswith("_tldw") or key == "_native_message_id" for key in row)
        for row in again.messages_payload
    )


def test_mixin_overflow_is_mandatory_and_clean_history_remains_compactable():
    from tldw_chatbook.Chat.console_prepared_request import (
        build_console_request,
        compose_current_mixin_messages,
        prepare_provider_request,
        resolve_request_capacity,
    )

    rows = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "new", "_native_message_id": "new"},
    ]
    composed = compose_current_mixin_messages(
        rows, snapshot("literal " * 100), user_message_id="new", attempt_id="attempt"
    )
    request = build_console_request(composed)
    assert request.compactable[0].messages[0]["content"] == "old"
    capacity = resolve_request_capacity(
        context_window_tokens=20,
        context_window_verified=True,
        requested_response_tokens=10,
    )
    prepared = prepare_provider_request(
        request,
        wire_style="distinct_roles",
        model="test",
        capacity=capacity,
        count_fn=lambda text, _model: len(str(text).split()),
    )
    assert prepared.known_overflow
    assert prepared.semantic.active_request[0]["content"] == composed[-1]["content"]
