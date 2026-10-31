"""Local Qwen reasoning must survive the actual request boundaries."""

import pytest

from tldw_chatbook.Chat.local_reasoning import (
    LocalReasoningReplay,
    LOCAL_REASONING_KEY,
    project_local_reasoning,
    supports_local_reasoning,
)


@pytest.mark.parametrize(
    "provider",
    ["llama_cpp", "local_llamacpp", "vllm", "local_vllm", "ollama", "local_ollama"],
)
@pytest.mark.parametrize(
    "model",
    [
        "Qwen3-8B",
        "Qwen/Qwen3.5-27B",
        "unsloth/Qwen3.6-27B-Q4.gguf",
        "qwen3.8:27b",
        "Qwen3-Coder-30B",
    ],
)
def test_local_qwen_families(provider, model):
    assert supports_local_reasoning(provider, model)


@pytest.mark.parametrize(
    "provider,model",
    [
        ("openai", "Qwen3.8-27B"),
        ("openai", "Llama-3"),
        ("anthropic", "Qwen2.5"),
        ("custom", "qwen30"),
        ("ollama", ""),
    ],
)
def test_unrelated_targets_are_not_opted_in(provider, model):
    assert not supports_local_reasoning(provider, model)


@pytest.mark.parametrize(
    "provider,key",
    [
        ("llama_cpp", "reasoning_content"),
        ("vllm", "reasoning_content"),
        ("ollama", "reasoning"),
    ],
)
def test_projection_pairs_reasoning_with_unchanged_answer(provider, key):
    replay = LocalReasoningReplay.capture(
        model="Qwen3.8-27B", content="answer", reasoning="thought"
    )
    row = {
        "role": "assistant",
        "content": "answer",
        LOCAL_REASONING_KEY: replay.to_dict(),
    }
    assert project_local_reasoning(row, provider=provider, model=replay.model) == {
        "role": "assistant",
        "content": "answer",
        key: "thought",
    }
    assert LOCAL_REASONING_KEY in row
    for target, model, content in [
        ("openai", replay.model, "answer"),
        (provider, "Qwen3.5-27B", "answer"),
        (provider, replay.model, "edited"),
    ]:
        projected = project_local_reasoning(
            {**row, "content": content}, provider=target, model=model
        )
        assert projected == {"role": "assistant", "content": content}


def test_record_is_strict_and_its_repr_excludes_content():
    replay = LocalReasoningReplay.capture(
        model="qwen3.8", content="answer", reasoning="THOUGHT"
    )
    assert "THOUGHT" not in repr(replay)
    assert LocalReasoningReplay.from_dict(replay.to_dict()) == replay
    assert (
        LocalReasoningReplay.from_dict({**replay.to_dict(), "reasoning": ["bad"]})
        is None
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider,key",
    [
        ("llama_cpp", "reasoning_content"),
        ("local_vllm", "reasoning_content"),
        ("local_llamafile", "reasoning_content"),
        ("local_ollama", "reasoning"),
    ],
)
@pytest.mark.parametrize("streaming", [True, False])
async def test_two_console_turns_replay_through_real_local_adapters(
    monkeypatch, provider, key, streaming
):
    import json
    import httpx
    from unittest.mock import Mock
    from tldw_chatbook.Chat.console_provider_gateway import (
        ConsoleProviderGateway,
        ConsoleProviderResolution,
    )
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore

    payloads = []
    response_key = "reasoning" if provider == "local_ollama" else "reasoning_content"
    body = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Answer",
                    response_key: "Exact call thought",
                },
                "finish_reason": "stop",
            }
        ]
    }
    sse = (
        "\n\n".join(
            "data: " + json.dumps(row)
            for row in [
                {"choices": [{"delta": {response_key: "Exact call thought"}}]},
                {
                    "choices": [
                        {"delta": {"content": "Answer"}, "finish_reason": "stop"}
                    ]
                },
            ]
        )
        + "\n\ndata: [DONE]\n\n"
    )

    def direct(request):
        payloads.append(json.loads(request.content))
        return (
            httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})
            if streaming
            else httpx.Response(200, json=body)
        )

    def compatible(_session, url, **kwargs):
        payloads.append(kwargs["json"])
        response = Mock()
        response.status_code = 200
        response.iter_lines.return_value = iter(sse.splitlines())
        response.json.return_value = body
        return response

    monkeypatch.setattr("requests.Session.post", compatible)
    resolution = ConsoleProviderResolution(
        ready=True,
        provider=provider,
        model="Qwen3.6-27B",
        base_url="http://localhost:9099",
        streaming=streaming,
        execution_key=provider,
    )

    class Gateway(ConsoleProviderGateway):
        async def resolve_for_send(self, selection):
            return resolution

    async with httpx.AsyncClient(transport=httpx.MockTransport(direct)) as client:
        gateway = Gateway(http_client=client)
        store = ConsoleChatStore()
        controller = ConsoleChatController(
            store=store,
            provider_gateway=gateway,
            provider_config=lambda: {"console": {"show_thinking": False}},
        )
        await controller.submit_draft("first question")
        first = store.messages_for_session(store.active_session_id)[-1]
        assert first.content == "Answer"
        assert first.thinking_text == "Exact call thought"
        assert first.local_reasoning_replay is not None
        await controller.submit_draft("next question")
    assert len(payloads) == 2
    history = next(row for row in payloads[1]["messages"] if row["role"] == "assistant")
    assert history[key] == "Exact call thought"
    assert history["content"] == "Answer"
    assert LOCAL_REASONING_KEY not in history
    if provider != "local_ollama":
        assert payloads[1]["chat_template_kwargs"]["preserve_thinking"] is True


@pytest.mark.asyncio
async def test_child_and_failed_calls_cannot_publish_parent_replay():
    from tldw_chatbook.Chat.console_provider_gateway import (
        ConsoleProviderGateway,
        ConsoleProviderResolution,
        ConsoleProviderStreamSignals,
    )

    class Gateway(ConsoleProviderGateway):
        async def _stream_chat(self, resolution, messages, tools=None, signals=None):
            signals.thinking_callback(resolution.model)
            yield "answer"
            if messages == "fail":
                raise RuntimeError("stream failed")

    gateway = Gateway()
    signals = ConsoleProviderStreamSignals()
    resolution = ConsoleProviderResolution(
        ready=True,
        provider="local_vllm",
        model="Qwen3.8-27B",
        base_url="http://localhost",
    )
    try:
        assert [
            x async for x in gateway.stream_chat(resolution, [], signals=signals)
        ] == ["answer"]
        parent = signals.local_reasoning_replay
        child = signals.new_usage_call(include_thinking=False)
        assert [
            x async for x in gateway.stream_chat(resolution, [], signals=child)
        ] == ["answer"]
        assert child.local_reasoning_replay is not None
        assert signals.local_reasoning_replay is parent
        with pytest.raises(RuntimeError):
            _ = [
                x
                async for x in gateway.stream_chat(resolution, "fail", signals=signals)
            ]
        assert signals.local_reasoning_replay is None
    finally:
        await gateway.aclose()


def test_real_agent_fence_tool_loop_preserves_each_call_separately(tmp_path):
    from Tests.Chat.test_console_agent_bridge import _bridge_with_gateway, _run, _fence
    from tldw_chatbook.Chat.console_provider_gateway import (
        ConsoleProviderGateway,
        ConsoleProviderResolution,
        ConsoleProviderStreamSignals,
    )
    from tldw_chatbook.Chat.console_prepared_request import thaw_json
    from tldw_chatbook.Chat.local_reasoning import LOCAL_TOOL_RESULT_KEY

    class Gateway(ConsoleProviderGateway):
        def __init__(self):
            super().__init__()
            self.sent = []

        async def _stream_chat(self, resolution, messages, tools=None, signals=None):
            prepared = self.prepare_chat_request(resolution, messages, tools=tools)
            self.sent.append([thaw_json(row) for row in prepared.messages_payload])
            if len(self.sent) == 1:
                signals.thinking_callback("Need the date")
                yield _fence("get_current_datetime", {})
            else:
                signals.thinking_callback("Use the date")
                yield "Final answer"

    gateway = Gateway()
    bridge, db, store, session, aid = _bridge_with_gateway(
        tmp_path, gateway, native_tools_enabled=lambda: False
    )
    signals = ConsoleProviderStreamSignals(
        thinking_callback=lambda text: store.append_thinking_chunk(aid, text)
    )
    resolution = ConsoleProviderResolution(
        ready=True,
        provider="local_vllm",
        execution_key="local_vllm",
        model="Qwen3.5-27B",
        base_url="http://localhost",
    )
    outcome = _run(
        bridge,
        store,
        session,
        aid,
        resolution=resolution,
        model=resolution.model,
        provider_stream_signals=signals,
    )
    assert outcome.status == "done"
    assert len(gateway.sent) == 2
    assistant = next(row for row in gateway.sent[1] if row["role"] == "assistant")
    assert assistant["reasoning_content"] == "Need the date"
    tool_result = gateway.sent[1][-1]
    assert tool_result["role"] == "user"
    assert tool_result["content"].startswith("<tool_response>\n")
    assert tool_result["content"].endswith("\n</tool_response>")
    assert LOCAL_TOOL_RESULT_KEY not in tool_result
    assert signals.local_reasoning_replay.reasoning == "Use the date"
    assert store.get_message(aid).thinking_text == "Need the dateUse the date"


def test_local_replay_survives_database_variants_and_screen_state(tmp_path):
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.Chat.message_metadata import MessageMetadata
    from tldw_chatbook.Chat.console_conversation_hydration import (
        console_messages_from_conversation_tree,
    )
    from tldw_chatbook.UI.Console_Modules.message import ConsoleMessageController

    db = CharactersRAGDB(tmp_path / "replay.db", "replay-test")
    try:
        store = ConsoleChatStore(persistence=ChatPersistenceService(db))
        session = store.create_session()
        store.persist_session_if_needed(session.id)
        row = store.append_message(
            session.id, role=ConsoleMessageRole.ASSISTANT, content="", persist=True
        )
        store.append_stream_chunk(row.id, "original")
        replay = LocalReasoningReplay.capture(
            model="qwen3.8", content="original", reasoning="original thought"
        )
        store.set_local_reasoning_replay(row.id, replay)
        store.mark_message_complete(row.id)
        store.begin_variant_stream(row.id)
        store.append_stream_chunk(row.id, "variant")
        new = LocalReasoningReplay.capture(
            model="qwen3.8", content="variant", reasoning="variant thought"
        )
        store.set_local_reasoning_replay(row.id, new)
        store.finalize_variant_stream(row.id)
        assert store.select_variant(row.id, 0).local_reasoning_replay == replay
        selected = store.select_variant(row.id, 1)
        screen_state = ConsoleMessageController._serialize_console_message(selected)
        assert (
            ConsoleMessageController._restore_console_message(
                screen_state
            ).local_reasoning_replay
            == new
        )
        assert (
            ConsoleMessageController._restore_console_message(screen_state)
            .variants.variants[0]
            .local_reasoning_replay
            == replay
        )
        durable = db.get_message_by_id(selected.persisted_message_id)
        assert (
            MessageMetadata.from_json(durable["metadata_json"]).local_reasoning_replay
            == new
        )
        hydrated = console_messages_from_conversation_tree({"root_threads": [durable]})
        assert hydrated[0].local_reasoning_replay == new
        store.add_variant(row.id, "no reasoning")
        durable = db.get_message_by_id(selected.persisted_message_id)
        assert (
            MessageMetadata.from_json(durable["metadata_json"]).local_reasoning_replay
            is None
        )
    finally:
        db.close_connection()


def test_reasoning_is_counted_and_evicted_with_its_turn():
    from tldw_chatbook.Chat.console_provider_gateway import (
        ConsoleProviderGateway,
        ConsoleProviderResolution,
    )
    from tldw_chatbook.Chat.console_history_budget import count_console_messages_tokens
    from tldw_chatbook.Chat.console_prepared_request import thaw_json

    resolution = ConsoleProviderResolution(
        ready=True,
        provider="llama_cpp",
        model="Qwen3.8-27B",
        base_url="http://localhost",
        max_tokens=512,
    )
    gateway = ConsoleProviderGateway()
    replay = LocalReasoningReplay.capture(
        model=resolution.model, content="answer", reasoning="Reasoning words. " * 8000
    )
    messages = [
        {"role": "user", "content": "old question"},
        {
            "role": "assistant",
            "content": "answer",
            LOCAL_REASONING_KEY: replay.to_dict(),
        },
        {"role": "user", "content": "new question"},
    ]
    full = gateway.prepare_chat_request(resolution, messages, apply_safety_window=False)
    assert full.accounting.total_input_tokens > count_console_messages_tokens(
        [
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "new question"},
        ],
        resolution.model,
    )
    bounded = gateway.prepare_chat_request(
        resolution, messages, context_window_override_tokens=2048
    )
    assert [thaw_json(row) for row in bounded.messages] == [
        {"role": "user", "content": "new question"}
    ]
    assert not bounded.known_overflow


@pytest.mark.asyncio
async def test_direct_fallback_replays_only_successful_attempt_reasoning():
    import json
    import httpx
    from tldw_chatbook.Chat.console_provider_gateway import (
        ConsoleProviderGateway,
        ConsoleProviderResolution,
        ConsoleProviderStreamSignals,
    )

    payloads = []

    def respond(request):
        payload = json.loads(request.content)
        payloads.append(payload)
        if payload["stream"]:
            return httpx.Response(
                200,
                text="data: "
                + json.dumps(
                    {"choices": [{"delta": {"reasoning_content": "abandoned thought"}}]}
                )
                + "\n\ndata: [DONE]\n\n",
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "Answer",
                            "reasoning_content": "successful thought",
                        }
                    }
                ]
            },
        )

    transcript = []
    signals = ConsoleProviderStreamSignals(thinking_callback=transcript.append)
    resolution = ConsoleProviderResolution(
        ready=True,
        provider="llama_cpp",
        model="Qwen3.8-27B",
        base_url="http://localhost",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        gateway = ConsoleProviderGateway(http_client=client)
        assert [
            chunk
            async for chunk in gateway.stream_chat(
                resolution, [{"role": "user", "content": "question"}], signals=signals
            )
        ] == ["Answer"]
    assert len(payloads) == 2
    assert "".join(transcript) == "abandoned thoughtsuccessful thought"
    assert signals.local_reasoning_replay.reasoning == "successful thought"


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["local_vllm", "local_ollama"])
async def test_compatible_inline_tags_split_across_chunks(provider):
    from tldw_chatbook.Chat.console_provider_gateway import (
        ConsoleProviderGateway,
        ConsoleProviderResolution,
        ConsoleProviderStreamSignals,
    )

    class Gateway(ConsoleProviderGateway):
        async def _stream_chat(self, resolution, messages, tools=None, signals=None):
            for chunk in ["<thi", "nk>Exact ", "thought</thi", "nk>\nAnswer"]:
                yield chunk

    transcript = []
    signals = ConsoleProviderStreamSignals(thinking_callback=transcript.append)
    resolution = ConsoleProviderResolution(
        ready=True, provider=provider, model="Qwen3.8-27B", base_url="http://localhost"
    )
    gateway = Gateway()
    try:
        assert (
            "".join(
                [
                    chunk
                    async for chunk in gateway.stream_chat(
                        resolution, [], signals=signals
                    )
                ]
            )
            == "Answer"
        )
        assert "".join(transcript) == "Exact thought"
        assert signals.local_reasoning_replay.matches(
            model=resolution.model, content="Answer"
        )
        assert signals.local_reasoning_replay.reasoning == "Exact thought"
    finally:
        await gateway.aclose()


def test_delayed_fleet_usage_cannot_replace_selected_variant_reasoning():
    from types import SimpleNamespace
    from tldw_chatbook.Chat.console_provider_gateway import (
        ConsoleProviderGateway,
        ConsoleProviderResolution,
        ConsoleProviderStreamSignals,
    )
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole

    store = ConsoleChatStore()
    controller = ConsoleChatController(
        store=store, provider_gateway=ConsoleProviderGateway()
    )
    session = store.create_session()
    row = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="answer"
    )
    resolution = ConsoleProviderResolution(
        ready=True,
        provider="local_vllm",
        model="Qwen3.8-27B",
        base_url="http://localhost",
    )
    old = LocalReasoningReplay.capture(
        model=resolution.model, content="answer", reasoning="old thought"
    )
    new = LocalReasoningReplay.capture(
        model=resolution.model, content="answer", reasoning="new thought"
    )
    signals = ConsoleProviderStreamSignals(local_reasoning_replay=old)
    controller._attach_stream_usage(row.id, signals, resolution, partial=False)
    store.add_variant(row.id, "answer")
    store.set_local_reasoning_replay(row.id, new)
    controller._fleet_usage_reattach_sources[row.id] = (signals, resolution, False)
    controller._reattach_fleet_usage(
        SimpleNamespace(
            children=[
                SimpleNamespace(assistant_message_id=row.id, session_id=session.id)
            ]
        )
    )
    assert store.get_message(row.id).local_reasoning_replay == new


def test_namespaced_qwen_agent_spend_estimate_counts_replayed_reasoning(tmp_path):
    from Tests.Agents.test_agent_service import _service_with_chat
    from tldw_chatbook.Agents.agent_models import AgentConfig
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    db = CharactersRAGDB(tmp_path / "agent.db", "reasoning-budget")
    try:
        service = _service_with_chat(
            db, lambda **kwargs: {"choices": [{"message": {"content": "answer"}}]}
        )
        model = "Qwen/Qwen3.8-27B"
        call = service._make_call_model(
            AgentConfig(model=model, system_prompt="s", native_tools=False),
            "local_vllm",
            [],
        )
        messages = [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
        bare = call(messages, ()).tokens
        replay = LocalReasoningReplay.capture(
            model=model, content="answer", reasoning="Detailed thought. " * 1000
        )
        enriched = call(
            [messages[0], {**messages[1], LOCAL_REASONING_KEY: replay.to_dict()}], ()
        ).tokens
        assert enriched > bare + 1000
    finally:
        db.close_connection()
