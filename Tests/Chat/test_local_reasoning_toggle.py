"""Replay is an independent default-on preference for local reasoning transports."""

from dataclasses import replace
import json
import pytest
import httpx
from tldw_chatbook.Chat.local_reasoning import (
    LocalReasoningReplay,
    LOCAL_REASONING_KEY,
    project_local_reasoning,
    supports_local_reasoning,
)
from tldw_chatbook.Chat.console_provider_gateway import (
    ConsoleProviderGateway,
    ConsoleProviderResolution,
)


@pytest.mark.parametrize(
    "provider", ["llama_cpp", "local_llamafile", "local_vllm", "local_ollama"]
)
@pytest.mark.parametrize(
    "model", ["gemma-4-26B-A4B", "my-model-alias", "DeepSeek-R1", "Qwen3.8"]
)
def test_local_replay_is_transport_based(provider, model):
    assert supports_local_reasoning(provider, model)


def test_disabled_replay_drops_local_fields_without_losing_record():
    replay = LocalReasoningReplay.capture(
        model="gemma-4", content="answer", reasoning="thought"
    )
    row = {
        "role": "assistant",
        "content": "answer",
        LOCAL_REASONING_KEY: replay.to_dict(),
    }
    assert project_local_reasoning(
        row, provider="llama_cpp", model="gemma-4", enabled=False
    ) == {"role": "assistant", "content": "answer"}
    assert (
        project_local_reasoning(row, provider="llama_cpp", model="gemma-4")[
            "reasoning_content"
        ]
        == "thought"
    )
    assert project_local_reasoning(
        {**row, "reasoning": "thought"},
        provider="ollama",
        model="gemma-4",
        enabled=False,
    ) == {"role": "assistant", "content": "answer"}
    hosted = {
        "role": "assistant",
        "content": "answer",
        "reasoning_content": "required hosted data",
    }
    assert (
        project_local_reasoning(
            hosted, provider="moonshot", model="kimi", enabled=False
        )
        == hosted
    )


def test_disabled_replay_removes_reasoning_from_prepared_budget():
    from tldw_chatbook.Chat.console_prepared_request import thaw_json

    model = "gemma-4"
    replay = LocalReasoningReplay.capture(
        model=model, content="answer", reasoning="thought words " * 500
    )
    rows = [
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "content": "answer",
            LOCAL_REASONING_KEY: replay.to_dict(),
        },
        {"role": "user", "content": "next"},
    ]
    gateway = ConsoleProviderGateway()
    resolution = ConsoleProviderResolution(
        ready=True,
        provider="llama_cpp",
        model=model,
        base_url="http://localhost",
        replay_thinking=False,
    )
    off = gateway.prepare_chat_request(resolution, rows, apply_safety_window=False)
    on = gateway.prepare_chat_request(
        replace(resolution, replay_thinking=True), rows, apply_safety_window=False
    )
    assert "reasoning_content" not in thaw_json(off.messages[1])
    assert on.accounting.total_input_tokens > off.accounting.total_input_tokens + 100
    from tldw_chatbook.Agents.agent_service import _count_model_messages

    assert (
        _count_model_messages(rows, model, "llama_cpp", replay_thinking=True)
        > _count_model_messages(rows, model, "llama_cpp", replay_thinking=False) + 100
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["llama_cpp", "local_vllm", "local_ollama"])
async def test_resolution_snapshots_replay_preference_without_restart(provider):
    from tldw_chatbook.Chat.console_chat_models import ConsoleProviderSelection

    config = {
        "console": {},
        "api_settings": {
            key: {"api_url": "http://localhost"}
            for key in ("vllm", "ollama", "local_vllm", "local_ollama")
        },
    }
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda req: httpx.Response(200, json={"data": [{"id": "gemma-4"}]})
        )
    ) as client:
        gateway = ConsoleProviderGateway(
            http_client=client, config_provider=lambda: config
        )
        selection = ConsoleProviderSelection(
            provider=provider, base_url="http://localhost", explicit_model="gemma-4"
        )
        first = await gateway.resolve_for_send(selection)
        assert first.ready and first.replay_thinking is True, first.visible_copy
        config["console"]["replay_thinking"] = False
        second = await gateway.resolve_for_send(selection)
        assert second.ready and second.replay_thinking is False, second.visible_copy
        assert first.replay_thinking is True


@pytest.mark.asyncio
async def test_console_on_off_on_preserves_thoughts_and_replays_while_enabled():
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore

    config = {"enabled": True}
    payloads = []

    def respond(request):
        payloads.append(json.loads(request.content))
        return httpx.Response(
            200,
            text="data: "
            + json.dumps(
                {
                    "choices": [
                        {
                            "delta": {
                                "reasoning_content": "Thought",
                                "content": "Answer",
                            },
                            "finish_reason": "stop",
                        }
                    ]
                }
            )
            + "\n\ndata: [DONE]\n\n",
            headers={"content-type": "text/event-stream"},
        )

    class Gateway(ConsoleProviderGateway):
        async def resolve_for_send(self, selection):
            return ConsoleProviderResolution(
                ready=True,
                provider="llama_cpp",
                model="gemma-4",
                base_url="http://localhost",
                replay_thinking=config["enabled"],
            )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        store = ConsoleChatStore()
        controller = ConsoleChatController(
            store=store,
            provider_gateway=Gateway(http_client=client),
            provider_config=lambda: {"console": {"show_thinking": False}},
        )
        for enabled in (True, False, True):
            config["enabled"] = enabled
            await controller.submit_draft("question")
            row = store.messages_for_session(store.active_session_id)[-1]
            assert row.content == "Answer"
            assert row.thinking_text == "Thought"
            assert row.local_reasoning_replay is not None
    assert len(payloads) == 3
    assert all("reasoning_content" not in row for row in payloads[1]["messages"])
    history = [row for row in payloads[2]["messages"] if row["role"] == "assistant"]
    assert len(history) == 2
    assert all(row["reasoning_content"] == "Thought" for row in history)
    assert all("chat_template_kwargs" not in payload for payload in payloads)
