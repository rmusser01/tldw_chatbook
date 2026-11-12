"""The durable controller trace path honors local structured replay capability."""

import json
from dataclasses import replace

import httpx
import pytest

from Tests.Chat.test_console_trace_sidecar_replay import (
    replay_harness as replay_harness,  # noqa: PLC0414
)
from Tests.console_provider_doubles import persisted_console_store
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_prepared_request import THINKING_OWNER_KEY, thaw_json
from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Chat.console_trace_native_reader import ConsoleTraceNativeReader
from tldw_chatbook.Chat.local_reasoning import ReasoningReplayPolicy
from tldw_chatbook.Chat.thinking_blocks import (
    DisplayableThinkingBlock,
    dump_thinking_blocks_json,
)


@pytest.mark.bootstrap_profile
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


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
async def test_local_reasoning_history_on_off_on_preserves_durable_thinking(tmp_path):
    endpoint = "http://127.0.0.1:9099"
    provider, model = "llama_cpp", "Qwen3.8-27B"
    config = {"console": {"show_thinking": False, "reasoning_history": "all"}}
    exchanges = [
        ("first answer", "FIRST-LOCAL-THINKING"),
        ("second answer", "SECOND-LOCAL-THINKING"),
        ("third answer", "THIRD-LOCAL-THINKING"),
    ]
    sent = []

    def respond(request):
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.method == "GET":
            return httpx.Response(404)
        assert request.url.path == "/v1/chat/completions"
        answer, thinking = exchanges[len(sent)]
        sent.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": answer, "reasoning_content": thinking},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    store = persisted_console_store(db_path=tmp_path / "thinking.sqlite")
    database = store.persistence.db
    settings = ConsoleSessionSettings(
        provider=provider,
        model=model,
        base_url=endpoint,
        reasoning_effort="low",
        streaming=False,
    )
    session = store.create_session(
        settings=settings,
        canonical_settings_baseline=settings,
        thinking_history_policy="auto",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        gateway = ConsoleProviderGateway(
            http_client=client, config_provider=lambda: config, environ={}
        )
        controller = ConsoleChatController(
            store=store, provider_gateway=gateway, provider_config=lambda: config
        )
        owners = set()
        try:
            for index, (mode, (answer, thinking)) in enumerate(
                zip(("all", "off", "all"), exchanges, strict=True), start=1
            ):
                config["console"]["reasoning_history"] = mode
                result = await controller.submit_draft(
                    f"question {index}", session_id=session.id
                )
                assert result.accepted and result.provider_started, result
                row = store.get_message(result.assistant_message_id)
                assert row.content == answer and row.status == "complete"
                assert row.thinking is not None
                [block] = row.thinking.blocks
                assert isinstance(block, DisplayableThinkingBlock)
                assert (block.provider, block.model, block.protocol) == (
                    provider,
                    model,
                    "chat_completions",
                )
                assert (block.source_format, block.status, block.text) == (
                    "reasoning_content",
                    "complete",
                    thinking,
                )
                assert row.persisted_message_id is not None
                owners.add(row.persisted_message_id)
                durable = database.get_message_by_id(row.persisted_message_id)
                assert durable["content"] == answer
                assert durable["thinking_blocks_json"] == dump_thinking_blocks_json(
                    row.thinking
                )
            assert store.session_thinking_history_policy(session.id) == "auto"
        finally:
            try:
                await controller.shutdown()
            finally:
                try:
                    await gateway.aclose()
                finally:
                    with database.quiesce_connections(timeout_seconds=2.0):
                        pass

    assert len(owners) == len(sent) == 3
    assert all("reasoning_content" not in row for row in sent[1]["messages"])
    assistant_history = [
        (row["content"], row.get("reasoning_content"))
        for row in sent[2]["messages"]
        if row["role"] == "assistant"
    ]
    assert assistant_history == exchanges[:2]
    assert all(
        THINKING_OWNER_KEY not in row for payload in sent for row in payload["messages"]
    )
