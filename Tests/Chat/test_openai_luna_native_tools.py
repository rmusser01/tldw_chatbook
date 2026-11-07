"""Luna native functions through the real Console, dispatcher and HTTP owner."""

import asyncio
import io
import json
import threading

import pytest
import urllib3

from tldw_chatbook.Agents.agent_models import (
    AgentConfig,
    RunBudget,
    ToolCatalogEntry,
    ToolResult,
    ToolSchema,
)
from tldw_chatbook.Agents.agent_service import AgentService
from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry
from tldw_chatbook.Chat.console_agent_bridge import _StreamingModelAdapter
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_provider_gateway import (
    ConsoleProviderGateway,
    ConsoleProviderResolution,
)
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.Utils.token_counter import ContextWindowResolution


@pytest.mark.bootstrap_profile
@pytest.mark.parametrize("streaming", [True, False])
def test_luna_native_function_result_round_trip(tmp_path, monkeypatch, streaming):
    """A native call executes once, then its exact ID/result reaches the next POST."""
    wire = []
    invoked = []
    call = {
        "id": "call_native_1",
        "type": "function",
        "function": {"name": "probe_marker", "arguments": "{}"},
    }

    class Tools:
        def list_catalog(self):
            return [
                ToolCatalogEntry(
                    "probe:marker", "probe_marker", "Read a synthetic marker", "probe"
                )
            ]

        def load_schema(self, tool_id):
            return ToolSchema(
                id=tool_id,
                name="probe_marker",
                description="Read a synthetic marker",
                parameters={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            )

        def invoke(self, tool_id, args):
            assert args == {}
            invoked.append(tool_id)
            return ToolResult(ok=True, content="synthetic-marker")

    def fake_http(pool, connection, method, url, *args, **kwargs):
        request = json.loads(kwargs.get("body", args[0] if args else None))
        first = not wire
        wire.append({"path": url, "payload": request})
        assert url.endswith("/chat/completions")
        message = (
            {"role": "assistant", "content": None, "tool_calls": [call]}
            if first
            else {"role": "assistant", "content": "Synthetic done."}
        )
        finish = "tool_calls" if first else "stop"
        response = {
            "id": "chat_1",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-6-luna",
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        if streaming:
            events = [
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"tool_calls": [dict(call, index=0)]}
                            if first
                            else {"content": "Synthetic done."},
                            "finish_reason": None,
                        }
                    ]
                },
                {
                    "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                    "usage": response["usage"],
                },
            ]
            body = (
                "".join("data: " + json.dumps(event) + "\n\n" for event in events)
                + "data: [DONE]\n\n"
            ).encode()
        else:
            body = json.dumps(response).encode()
        return urllib3.response.HTTPResponse(
            body=io.BytesIO(body),
            status=200,
            preload_content=False,
            headers={
                "Content-Type": "text/event-stream" if streaming else "application/json"
            },
        )

    monkeypatch.setattr(
        urllib3.connectionpool.HTTPConnectionPool, "_make_request", fake_http
    )
    resolution = ConsoleProviderResolution(
        provider="openai",
        execution_key="openai",
        base_url="https://openai.example.invalid/v1",
        model="gpt-6-luna",
        ready=True,
        readiness_key="openai",
        api_key="synthetic-native-key",
        max_tokens=1024,
        streaming=streaming,
        reasoning_effort="none",
        context_window=ContextWindowResolution(32768, "synthetic metadata", True),
    )
    gateway = ConsoleProviderGateway(config_provider=dict, environ={})
    store = ConsoleChatStore()
    session = store.ensure_session()
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    adapter = _StreamingModelAdapter(
        store=store,
        provider_gateway=gateway,
        resolution=resolution,
        assistant_message_id=assistant.id,
        should_cancel=lambda: False,
        loop=loop,
        native_tools=True,
    )
    registry = ToolCatalogRegistry()
    registry.register_provider(Tools())
    db = AgentRunsDB(tmp_path / "runs.db", client_id="native-luna")
    service = AgentService(
        db=db, registry=registry, chat_call=adapter.chat_call, app_config={}
    )
    config = AgentConfig(
        model="gpt-6-luna",
        provider="openai",
        system_prompt="Synthetic native tool check.",
        allowed_tools=("probe_marker",),
        native_tools=True,
        response_reserve_tokens=1024,
        budget=RunBudget(
            max_steps=16,
            max_model_turns=2,
            max_subagents=0,
            max_model_retries=0,
            max_wall_seconds=30,
            max_tool_call_seconds=3,
            budget_warning_fraction=1.0,
        ),
    )
    try:
        _, outcome = service.run_turn(
            conversation_id=session.id,
            messages=[{"role": "user", "content": "Read the marker, then finish."}],
            config=config,
            api_endpoint="openai",
        )
        assert invoked == ["probe:marker"], (outcome.status, wire)
        assert outcome.status == "done"
        assert outcome.final_text == "Synthetic done."
        assert len(wire) == 2
        for request in wire:
            assert request["payload"]["reasoning_effort"] == "none"
            assert request["payload"]["max_completion_tokens"] == 1024
            assert "max_tokens" not in request["payload"]
        tools = wire[0]["payload"]["tools"]
        assert any(tool["function"]["name"] == "probe_marker" for tool in tools)
        history = wire[1]["payload"]["messages"]
        assert any(row.get("tool_calls") == [call] for row in history)
        assert (
            sum(
                row.get("role") == "tool"
                and row.get("tool_call_id") == "call_native_1"
                and row.get("content") == "synthetic-marker"
                for row in history
            )
            == 1
        )
    finally:
        asyncio.run_coroutine_threadsafe(gateway.aclose(), loop).result(timeout=10)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(5)
        assert not thread.is_alive()
        loop.close()
        db.close()
