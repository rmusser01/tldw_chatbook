"""Hosted reasoning evidence must preserve the saved response and next send."""

import json
from dataclasses import replace

import pytest

from Tests.Chat.test_console_trace_sidecar_replay import (
    replay_harness as replay_harness,  # noqa: PLC0414 - pytest fixture re-export
)
from tldw_chatbook.Chat.console_trace_native_reader import ConsoleTraceNativeReader
from tldw_chatbook.Chat.console_trace_runtime import ConsoleTraceBoundaryFactory
from tldw_chatbook.Chat.provider_continuation import ContinuationRound
from tldw_chatbook.Chat.thinking_blocks import ProprietaryThinkingBlock
from tldw_chatbook.LLM_Calls.hosted_chat import HostedChatTurn
from tldw_chatbook.LLM_Calls.moonshot import MoonshotResponse


@pytest.mark.asyncio
@pytest.mark.parametrize("replay_harness", [False, True], indirect=True)
@pytest.mark.parametrize("with_checkpoint", [False, True])
async def test_proprietary_response_preserves_transformed_cold_successor(
    replay_harness, with_checkpoint, monkeypatch
):
    harness = replay_harness
    gateway = harness.controller.provider_gateway
    resolve = gateway.resolve_for_send
    canary = "NEW_PROPRIETARY_REASONING_CANARY"
    checkpoint = (
        replace(
            harness.checkpoint,
            rounds=(ContinuationRound("answer", (canary,), ()),),
        )
        if with_checkpoint
        else None
    )

    async def proprietary_resolution(selection):
        return replace(
            await resolve(selection),
            thinking_stream_disposition="proprietary",
            thinking_round_trip_version=1,
        )

    def adapter(**kwargs):
        harness.entries.append(kwargs)
        return MoonshotResponse(
            {"choices": [{"message": {"content": "answer"}}]},
            terminal_turn=HostedChatTurn(
                text="answer",
                tool_calls=(),
                assistant_message={"role": "assistant", "content": "answer"},
                finish_reason="stop",
                reasoning_content=canary,
            ),
            provider_continuation=checkpoint,
        )

    monkeypatch.setattr(gateway, "resolve_for_send", proprietary_resolution)
    monkeypatch.setattr(gateway, "_chat_api_call_fn", adapter)
    first = await harness.controller.submit_draft(
        "alias request", session_id="session-1"
    )
    assert first.accepted and first.provider_started, (first, harness.failures)
    assistant = harness.store.get_message(first.assistant_message_id)
    assert assistant.content == "answer"
    assert len(assistant.thinking.blocks) == 1
    block = assistant.thinking.blocks[0]
    assert isinstance(block, ProprietaryThinkingBlock)
    assert (block.provider, block.model, block.source_format, block.status) == (
        "moonshot",
        "kimi-k3",
        "reasoning_content",
        "complete",
    )
    row = harness.database.get_message_by_id(assistant.persisted_message_id)
    assert canary not in row["thinking_blocks_json"]
    saved_user = harness.store.get_message(first.user_message_id)
    reader = ConsoleTraceNativeReader(harness.database)
    old_trace = reader.read_calls(saved_user.persisted_message_id)
    old_heads = list(
        harness.database.get_connection().execute(
            "SELECT call_id, surface_node_id FROM console_trace_calls ORDER BY call_id"
        )
    )
    if harness.capture_on:
        assert len(old_trace) == 1
        assert canary not in json.dumps(old_trace[0].capture.response)
        assert old_trace[0].capture.response["proprietary_thinking_evidence"] == [
            {
                "provider": "moonshot",
                "model": "kimi-k3",
                "protocol": "chat_completions",
                "source_format": "reasoning_content",
            }
        ]

    harness.factory = ConsoleTraceBoundaryFactory(harness.database)
    second = await harness.controller.submit_draft(
        "ordinary successor", session_id="session-1"
    )
    assert second.accepted and second.provider_started, (second, harness.failures)
    assert len(harness.entries) == 2
    assert tuple(harness.entries[-1]["provider_continuations"]) == (harness.checkpoint,)
    if harness.capture_on:
        assert reader.read_calls(saved_user.persisted_message_id) == old_trace
        connection = harness.database.get_connection()
        for call_id, head in old_heads:
            assert (
                connection.execute(
                    "SELECT surface_node_id FROM console_trace_calls WHERE call_id = ?",
                    (call_id,),
                ).fetchone()[0]
                == head
            )
        assert tuple(
            connection.execute(
                "SELECT link_kind, verification_outcome FROM console_trace_response_links "
                "WHERE call_id = ?",
                (old_heads[0][0],),
            ).fetchone()
        ) == ("revision", "verified_equal")
