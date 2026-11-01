"""Completed native history stays atomic while its exact wire replay is counted."""

import json

import pytest

from Tests.Chat.test_console_prepared_request import _capacity, _word_count
from tldw_chatbook.Chat.console_prepared_request import (
    CONTINUATION_OWNER_KEY,
    build_console_request,
    prepare_provider_request,
    thaw_json,
)
from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway
from tldw_chatbook.Chat.provider_continuation import (
    continuation_owner_group,
    parse_provider_continuation_json,
)


def completed_group():
    checkpoint = parse_provider_continuation_json(
        {
            "schema_version": 1,
            "checkpoint_revision": 3,
            "provider": "custom-hosted",
            "protocol": "chat_completions",
            "model": "synthetic",
            "api_base_url": "http://127.0.0.1:8010/v1",
            "state": "complete",
            "rounds": [
                {
                    "assistant_content": "Checking",
                    "reasoning_blocks": ["PRIVATE reasoning"],
                    "calls": [
                        {
                            "call_id": "call_1",
                            "name": "lookup",
                            "arguments": '{ "name": "x" }',
                            "state": "completed",
                            "result": "result " * 40,
                        }
                    ],
                },
            ],
        }
    )
    return continuation_owner_group(
        {"id": "owner", "role": "assistant", "content": "Final answer"}, checkpoint
    )


def prepare(rows, group, ceiling=None):
    return prepare_provider_request(
        build_console_request(rows, continuation_groups=(group,)),
        wire_style="single_preamble",
        model="synthetic",
        capacity=_capacity(ceiling),
        count_fn=_word_count,
    )


def test_completed_replay_keeps_semantic_owner_and_exact_final_wire():
    group = completed_group()
    rows = [
        {"role": "user", "content": "old"},
        {
            "role": "assistant",
            "content": "Final answer",
            CONTINUATION_OWNER_KEY: "owner",
        },
        {"role": "user", "content": "wake"},
    ]
    prepared = prepare(rows, group)
    canonical = ConsoleProviderGateway.expand_provider_continuation(group.checkpoint)
    assert thaw_json(prepared.messages_payload) == [
        rows[0],
        *canonical,
        {"role": "assistant", "content": "Final answer"},
        rows[-1],
    ]
    assert len(prepared.semantic.compactable[0].messages) == 2
    assert "PRIVATE reasoning" not in json.dumps(thaw_json(prepared.messages_payload))
    assert (
        sum(row.get("content") == "Final answer" for row in prepared.messages_payload)
        == 1
    )
    without = prepared.semantic.without_oldest_units(1)
    assert not without.compactable and not without.active_continuation_groups


@pytest.mark.parametrize("change", ["final", "partial", "duplicate"])
def test_completed_replay_refuses_changed_or_partial_owner(change):
    group = completed_group()
    rows = [{"role": "user", "content": "old"}]
    if change != "final":
        native = ConsoleProviderGateway.expand_provider_continuation(group.checkpoint)
        rows.extend(native[:1] if change == "partial" else native * 2)
    rows.extend(
        [
            {
                "role": "assistant",
                "content": [] if change == "final" else "Final answer",
                CONTINUATION_OWNER_KEY: "owner",
            },
            {"role": "user", "content": "wake"},
        ]
    )
    with pytest.raises(ValueError, match="completed continuation"):
        prepare(rows, group)


def test_completed_replay_counts_native_bytes_once_and_evicts_whole_owner():
    group = completed_group()
    owner = {
        "role": "assistant",
        "content": "Final answer",
        CONTINUATION_OWNER_KEY: "owner",
    }
    rows = [
        {"role": "user", "content": "old"},
        owner,
        {"role": "user", "content": "wake"},
    ]
    projected = prepare(rows, group)
    expected = _word_count(thaw_json(projected.messages), "synthetic") + _word_count(
        [{"role": "assistant", "content": "PRIVATE reasoning"}], "synthetic"
    )
    assert projected.accounting.total_input_tokens == expected
    evicted = prepare(rows, group, expected - 1)
    assert evicted.dropped_units == 1 and not evicted.continuation_groups
    assert thaw_json(evicted.messages_payload) == [rows[-1]]
    already = [
        rows[0],
        *ConsoleProviderGateway.expand_provider_continuation(group.checkpoint),
        owner,
        rows[-1],
    ]
    reused = prepare(already, group)
    assert reused.messages_payload == projected.messages_payload
    assert reused.accounting.total_input_tokens == expected
    assert len(reused.semantic.compactable[0].messages) == 2
    evicted_reused = prepare(already, group, expected - 1)
    assert evicted_reused.dropped_units == 1 and not evicted_reused.continuation_groups
    assert thaw_json(evicted_reused.messages_payload) == [rows[-1]]


@pytest.mark.parametrize("change", ["arguments", "result", "order", "inserted"])
def test_completed_replay_refuses_altered_recorded_sequence(change):
    native = ConsoleProviderGateway.expand_provider_continuation(
        completed_group().checkpoint
    )
    if change == "arguments":
        native[0]["tool_calls"][0]["function"]["arguments"] = '{"name":"x"}'
    elif change == "result":
        native[1]["content"] = "changed result"
    elif change == "order":
        native.reverse()
    else:
        native.insert(1, {"role": "assistant", "content": "unrelated"})
    with pytest.raises(ValueError, match="completed continuation replay"):
        prepare(
            [
                {"role": "user", "content": "old"},
                *native,
                {
                    "role": "assistant",
                    "content": "Final answer",
                    CONTINUATION_OWNER_KEY: "owner",
                },
                {"role": "user", "content": "wake"},
            ],
            completed_group(),
        )


def test_completed_trace_comparison_uses_frozen_credential_projection():
    from tldw_chatbook.Chat.console_trace_models import FrozenTracePolicy, new_opaque_id
    from tldw_chatbook.Chat.console_trace_service import ConsoleTraceService
    from tldw_chatbook.Chat.provider_continuation import dump_provider_continuation_json

    value = json.loads(dump_provider_continuation_json(completed_group().checkpoint))
    canary = "sk-proj-" + "x" * 48
    value["rounds"][0]["reasoning_blocks"] = ["Bearer " + canary]
    value["rounds"][0]["calls"][0]["result"] = "Bearer " + canary
    policy = FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None)
    projected = ConsoleTraceService._completed_replay_trace_checkpoint(value, policy)
    assert canary not in dump_provider_continuation_json(projected)
    assert value["rounds"][0]["calls"][0]["result"] == "Bearer " + canary
    assert (
        projected.rounds[0].calls[0].arguments
        == completed_group().rounds[0].calls[0].arguments
    )
    assert projected.rounds[0].calls[0].call_id == "call_1"


def test_completed_trace_comparison_preserves_opaque_known_credential_mask():
    from tldw_chatbook.Chat.console_trace_models import FrozenTracePolicy, new_opaque_id
    from tldw_chatbook.Chat.console_trace_redaction import CredentialSanitizer
    from tldw_chatbook.Chat.console_trace_service import ConsoleTraceService
    from tldw_chatbook.Chat.provider_continuation import (
        dump_provider_continuation_json,
        parse_provider_continuation_json,
    )

    secret = "syntheticOpaqueCredentialValue42"
    value = json.loads(dump_provider_continuation_json(completed_group().checkpoint))
    value["rounds"][0]["calls"][0]["result"] = "Read file value: " + secret
    policy = FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None)
    prior = CredentialSanitizer(known_credentials=(secret,)).sanitize(value)
    projected = ConsoleTraceService._completed_replay_trace_checkpoint(
        value, policy, known_credentials=(secret,)
    )
    assert (
        prior.available
        and projected.rounds == parse_provider_continuation_json(prior.value).rounds
    )
    assert secret not in dump_provider_continuation_json(projected)
    assert secret in value["rounds"][0]["calls"][0]["result"]
