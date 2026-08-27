"""Optional displayable-thinking replay policy and exact wire projection."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tldw_chatbook.Chat.console_prepared_request import (
    THINKING_OWNER_KEY,
    build_console_request,
    prepare_provider_request,
    resolve_request_capacity,
    thaw_json,
)
from tldw_chatbook.Chat.console_history_budget import ProviderContinuationSidecar
from tldw_chatbook.Chat.console_provider_gateway import (
    ConsoleProviderGateway,
    ConsoleProviderResolution,
)
from tldw_chatbook.Chat.console_thinking_history import (
    ProviderThinkingSidecar,
    ThinkingHistorySerializationError,
    ThinkingReplayTarget,
    resolve_thinking_history,
)
from tldw_chatbook.Chat.provider_continuation import (
    ContinuationRestoreTarget,
    parse_provider_continuation_json,
)
from tldw_chatbook.Chat.thinking_blocks import (
    DisplayableThinkingBlock,
    ProprietaryThinkingBlock,
    ThinkingEnvelope,
)


def _displayable(
    text: str = "DISPLAYABLE-THINKING-CANARY",
    *,
    status: str = "complete",
    provider: str = "llama_cpp",
    protocol: str = "chat_completions",
    source_format: str = "start_anchored_think",
) -> DisplayableThinkingBlock:
    return DisplayableThinkingBlock(
        block_id=f"block-{text[:8]}-{status}",
        round_ordinal=0,
        provider=provider,
        model="source-model",
        protocol=protocol,
        source_format=source_format,
        status=status,  # type: ignore[arg-type]
        text=text,
    )


def _proprietary() -> ProprietaryThinkingBlock:
    return ProprietaryThinkingBlock(
        block_id="proprietary",
        round_ordinal=0,
        provider="moonshot",
        model="kimi",
        protocol="chat_completions",
        source_format="reasoning_content",
        status="complete",
    )


def _sidecar(*blocks) -> ProviderThinkingSidecar:
    return ProviderThinkingSidecar("assistant-1", ThinkingEnvelope(tuple(blocks)))


def _target(**changes) -> ThinkingReplayTarget:
    target = ThinkingReplayTarget(
        provider="llama_cpp",
        model="target-model",
        protocol="chat_completions",
        disposition="displayable",
        round_trip_version=1,
    )
    return replace(target, **changes)


@pytest.mark.parametrize("policy", [None, "", "auto", "include"])
def test_auto_legacy_and_include_replay_compatible_complete_blocks(policy) -> None:
    resolved = resolve_thinking_history(
        target=_target(),
        policy=policy,
        sidecars=(_sidecar(_displayable()),),
    )

    assert resolved.saved_policy == ("include" if policy == "include" else "auto")
    assert resolved.effective_policy == resolved.saved_policy
    assert [block.text for block in resolved.groups[0].blocks] == [
        "DISPLAYABLE-THINKING-CANARY"
    ]


def test_required_overlay_does_not_overwrite_saved_optional_preference() -> None:
    resolved = resolve_thinking_history(
        target=_target(),
        policy="exclude",
        sidecars=(_sidecar(_displayable()),),
        continuation_required=True,
    )

    assert resolved.saved_policy == "exclude"
    assert resolved.effective_policy == "required"
    assert resolved.groups == ()


@pytest.mark.parametrize(
    "target",
    [
        _target(provider="openai", disposition="ignored", round_trip_version=None),
        _target(protocol="responses"),
        _target(disposition="ignored", round_trip_version=None),
    ],
)
def test_incompatible_target_never_receives_generic_thinking_translation(
    target: ThinkingReplayTarget,
) -> None:
    resolved = resolve_thinking_history(
        target=target,
        policy="include",
        sidecars=(_sidecar(_displayable()),),
    )

    assert resolved.groups == ()


@pytest.mark.parametrize("status", ["stopped", "failed"])
def test_noncomplete_displayable_blocks_are_not_replayed(status: str) -> None:
    resolved = resolve_thinking_history(
        target=_target(),
        policy="include",
        sidecars=(_sidecar(_displayable(status=status)),),
    )

    assert resolved.groups == ()


def test_proprietary_blocks_are_structurally_excluded() -> None:
    resolved = resolve_thinking_history(
        target=_target(),
        policy="include",
        sidecars=(_sidecar(_proprietary()),),
    )

    assert resolved.groups == ()
    assert "Proprietary thinking obfuscated" not in repr(resolved)


def test_exclude_omits_compatible_optional_thinking() -> None:
    resolved = resolve_thinking_history(
        target=_target(),
        policy="exclude",
        sidecars=(_sidecar(_displayable()),),
    )

    assert resolved.groups == ()


def test_strict_include_rejects_claimed_compatible_unknown_source_content_free() -> (
    None
):
    canary = "UNSAFE-SOURCE-CANARY"
    with pytest.raises(
        ThinkingHistorySerializationError,
        match="Thinking history could not be serialized safely",
    ) as error:
        resolve_thinking_history(
            target=_target(),
            policy="include",
            sidecars=(
                _sidecar(_displayable(canary, source_format="future-local-format")),
            ),
        )

    assert canary not in str(error.value)


def test_local_serializer_reconstructs_exact_source_once_without_mutating_semantic() -> (
    None
):
    canary = "EXACT-SOURCE-CANARY"
    resolved = resolve_thinking_history(
        target=_target(),
        policy="include",
        sidecars=(_sidecar(_displayable(canary)),),
    )
    semantic = build_console_request(
        [
            {"role": "user", "content": "old"},
            {
                "role": "assistant",
                "content": "visible answer",
                THINKING_OWNER_KEY: "assistant-1",
            },
            {"role": "user", "content": "current"},
        ],
        thinking_groups=resolved.groups,
        thinking_policy=resolved.saved_policy,
        effective_thinking_policy=resolved.effective_policy,
    )
    counted: list[list[dict]] = []

    def count_spy(messages: list[dict], _model: str) -> int:
        counted.append(messages)
        return len(str(messages))

    prepared = prepare_provider_request(
        semantic,
        wire_style="distinct_roles",
        provider="llama_cpp",
        model="target-model",
        capacity=resolve_request_capacity(context_window_tokens=None),
        count_fn=count_spy,
    )

    wire = [thaw_json(row) for row in prepared.messages]
    assistant_wire = next(row for row in wire if row["role"] == "assistant")
    semantic_answer = prepared.semantic.compactable[0].messages[1]["content"]
    assert assistant_wire["content"] == f"<think>{canary}</think>\nvisible answer"
    assert semantic_answer == "visible answer"
    assert str(wire).count(canary) == 1
    assert counted[-1] == wire
    assert prepared.thinking_groups == resolved.groups
    assert THINKING_OWNER_KEY not in assistant_wire


def test_evicting_oldest_unit_drops_its_visible_owner_and_thinking_together() -> None:
    resolved = resolve_thinking_history(
        target=_target(),
        policy="include",
        sidecars=(_sidecar(_displayable("EVICTED-THINKING-CANARY")),),
    )
    semantic = build_console_request(
        [
            {"role": "user", "content": "old"},
            {
                "role": "assistant",
                "content": "old answer",
                THINKING_OWNER_KEY: "assistant-1",
            },
            {"role": "user", "content": "current"},
        ],
        thinking_groups=resolved.groups,
        thinking_policy=resolved.saved_policy,
        effective_thinking_policy=resolved.effective_policy,
    )

    evicted = semantic.without_oldest_units(1)
    prepared = prepare_provider_request(
        evicted,
        wire_style="distinct_roles",
        provider="llama_cpp",
        model="target-model",
        capacity=resolve_request_capacity(context_window_tokens=None),
    )

    assert [row["content"] for row in prepared.messages_payload] == ["current"]
    assert prepared.thinking_groups == ()
    assert "EVICTED-THINKING-CANARY" not in repr(prepared.messages_payload)


def test_gateway_include_refuses_before_provider_contact() -> None:
    called = False

    def provider_spy(**_kwargs):
        nonlocal called
        called = True
        return {"choices": [{"message": {"content": "not reached"}}]}

    gateway = ConsoleProviderGateway(chat_api_call_fn=provider_spy)
    resolution = ConsoleProviderResolution(
        provider="llama_cpp",
        base_url="http://127.0.0.1:9099",
        model="target-model",
        ready=True,
        execution_key="llama_cpp",
        continuation_protocol="chat_completions",
        thinking_stream_disposition="displayable",
        thinking_round_trip_version=1,
    )

    with pytest.raises(ThinkingHistorySerializationError):
        gateway.prepare_chat_request(
            resolution,
            [{"role": "assistant", "content": "answer", "_owner": "assistant-1"}],
            thinking_sidecar=(
                _sidecar(_displayable(source_format="future-local-format")),
            ),
            thinking_policy="include",
            thinking_owner_key="_owner",
        )

    assert called is False


@pytest.mark.asyncio
async def test_gateway_shared_raw_owner_marker_attaches_continuation_and_thinking_once() -> (
    None
):
    checkpoint = parse_provider_continuation_json(
        {
            "schema_version": 1,
            "checkpoint_revision": 1,
            "provider": "deepseek",
            "protocol": "chat_completions",
            "model": "target-model",
            "api_base_url": "http://127.0.0.1:9099",
            "state": "complete",
            "rounds": [
                {
                    "assistant_content": "visible answer",
                    "reasoning_blocks": [],
                    "calls": [
                        {
                            "call_id": "call_shared",
                            "name": "lookup",
                            "arguments": "{}",
                            "state": "completed",
                            "result": "done",
                        }
                    ],
                }
            ],
        }
    )
    dispatched: list[dict] = []

    def provider_spy(**kwargs):
        dispatched.append(kwargs)
        return {"choices": [{"message": {"content": "ok"}}]}

    gateway = ConsoleProviderGateway(chat_api_call_fn=provider_spy)
    resolution = ConsoleProviderResolution(
        provider="deepseek",
        base_url="http://127.0.0.1:9099",
        model="target-model",
        ready=True,
        execution_key="llama_cpp",
        continuation_protocol="chat_completions",
        thinking_stream_disposition="displayable",
        thinking_round_trip_version=1,
    )

    prepared = gateway.prepare_chat_request(
        resolution,
        [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "visible answer", "_owner": "a1"},
            {"role": "user", "content": "current"},
        ],
        continuation_sidecar=(ProviderContinuationSidecar("a1", checkpoint),),
        continuation_target=ContinuationRestoreTarget(
            provider="deepseek",
            protocol="chat_completions",
            model="target-model",
            api_base_url="http://127.0.0.1:9099",
        ),
        continuation_owner_key="_owner",
        thinking_sidecar=(
            ProviderThinkingSidecar(
                "a1", ThinkingEnvelope((_displayable("SHARED-OWNER-CANARY"),))
            ),
        ),
        thinking_policy="include",
        thinking_owner_key="_owner",
    )

    assert [group.owner_message_id for group in prepared.continuation_groups] == ["a1"]
    assert [group.owner_message_id for group in prepared.thinking_groups] == ["a1"]
    assert repr(prepared.messages_payload).count("SHARED-OWNER-CANARY") == 1
    assert [item async for item in gateway.stream_chat(resolution, prepared)] == ["ok"]
    assert repr(dispatched[0]["messages_payload"]).count("SHARED-OWNER-CANARY") == 1
