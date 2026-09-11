"""The durable controller trace path honors local structured replay capability."""

from dataclasses import replace

import pytest

from Tests.Chat.test_console_trace_sidecar_replay import (
    replay_harness as replay_harness,  # noqa: PLC0414
)
from tldw_chatbook.Chat.console_prepared_request import thaw_json
from tldw_chatbook.Chat.console_trace_native_reader import ConsoleTraceNativeReader
from tldw_chatbook.Chat.local_reasoning import ReasoningReplayPolicy


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "replay_harness", [(False, "thinking"), (True, "thinking")], indirect=True
)
@pytest.mark.parametrize(
    "mode,expected", [("all", True), ("off", False), ("current", False)]
)
async def test_controller_structured_alias_replay_matches_capture_mode(
    replay_harness, monkeypatch, mode, expected
):
    harness = replay_harness
    canonical = harness.store.get_message(harness.prior.id).thinking
    canonical = replace(
        canonical,
        blocks=(replace(canonical.blocks[0], source_format="reasoning_content"),),
    )
    harness.store.replace_message_thinking(harness.prior.id, canonical)
    assert harness.store.persist_selected_generation(harness.prior.id)
    gateway = harness.controller.provider_gateway
    original_resolve = gateway.resolve_for_send

    async def resolve(selection):
        resolution = await original_resolve(selection)
        return replace(
            resolution,
            thinking_stream_disposition="ignored",
            thinking_round_trip_version=None,
            local_structured_thinking=True,
            reasoning_replay=ReasoningReplayPolicy(mode, "test"),
        )

    monkeypatch.setattr(gateway, "resolve_for_send", resolve)
    result = await harness.controller.submit_draft(
        "ordinary successor", session_id="session-1"
    )
    assert result.accepted and result.provider_started, (
        result,
        harness.failures,
        harness.build_errors,
    )
    messages = thaw_json(harness.entries[-1]["messages_payload"])
    prior = next(row for row in messages if row["content"] == "prior answer")
    assert prior.get("reasoning_content") == (
        "REPLAY_THINKING_CANARY" if expected else None
    )
    if harness.capture_on:
        user = harness.store.get_message(result.user_message_id)
        traces = ConsoleTraceNativeReader(harness.database).read_calls(
            user.persisted_message_id
        )
        assert traces[0].capture.request["messages_payload"] == messages
