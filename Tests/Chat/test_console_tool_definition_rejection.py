"""A default Console send must reach the model (TASK-33621.1, finding G4-01).

Every default Console send to Anthropic and OpenAI came back HTTP 400: three
built-in local tools declared a top-level ``anyOf``/``oneOf``/``allOf``, which
both providers refuse before the model runs. The user was then told to
"Confirm the model is still available, or choose another model from the
model picker" -- advice that cannot help, because every model rejects the
same request.

These tests are joined end to end: the real ``ConsoleAgentBridge`` composes
the real ``LocalToolProvider`` (todo + Watchlists specs, as on a default
profile) and an MCP-bridged third-party tool, the real
``ConsoleProviderGateway`` dispatches through the real ``chat_api_call`` and
``chat_with_anthropic``, and the provider is a loopback HTTP server that
applies Anthropic's own documented top-level ``input_schema`` rule and answers
with Anthropic's real error body. Only the socket is local.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pytest
import requests
from loguru import logger

from tldw_chatbook.Agents.local_tool_provider import LocalToolProvider
from tldw_chatbook.Agents.mcp_tool_provider import MCPToolProvider
from tldw_chatbook.Agents.run_context import use_run_id
from tldw_chatbook.Agents.session_todo_store import SessionTodoStore
from tldw_chatbook.Chat.Chat_Deps import ChatBadRequestError
from tldw_chatbook.Chat.Chat_Functions import chat_api_call
from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_provider_gateway import (
    ConsoleProviderGateway,
    ConsoleProviderResolution,
)
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.MCP.hub_tool_catalog import HubTool
from tldw_chatbook.MCP.permission_store import EffectiveToolState

#: The adapters read provider settings through the guarded config loader;
#: under the per-test redirect that admission fails closed with
#: RecoveryRequired("raw_source_selection_changed"). bootstrap_profile keeps
#: the collection-time sandbox profile (still isolated from the user's).
pytestmark = pytest.mark.bootstrap_profile

_REPLY = "SCHEMA-CONFORMANCE-REPLY-SENTINEL"
_MODEL_ADVICE = (
    "Confirm the model is still available",
    "model picker",
)
#: Anthropic's documented top-level refusal (verified live 2026-09-29, see
#: qa/console-ux-review-2026-09-29 G4-01).
_ANTHROPIC_TOP_LEVEL = ("anyOf", "oneOf", "allOf")
ALLOW = EffectiveToolState(state="allow", origin="tool_override")


def _anthropic_tool_error(index: int) -> bytes:
    return json.dumps(
        {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": (
                    f"tools.{index}.custom.input_schema: input_schema does not "
                    "support oneOf, allOf, or anyOf at the top level"
                ),
            },
            "request_id": "req_schema_conformance",
        }
    ).encode()


def _anthropic_reply(stream: bool) -> tuple[str, bytes]:
    if not stream:
        return "application/json", json.dumps(
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-haiku-4-5",
                "content": [{"type": "text", "text": _REPLY}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 12, "output_tokens": 4},
            }
        ).encode()
    events = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-haiku-4-5",
                "content": [],
                "usage": {"input_tokens": 12, "output_tokens": 1},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": _REPLY},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 4},
        },
        {"type": "message_stop"},
    ]
    wire = b"".join(
        b"event: "
        + event["type"].encode()
        + b"\ndata: "
        + json.dumps(event).encode()
        + b"\n\n"
        for event in events
    )
    return "text/event-stream", wire


class _AnthropicServer(ThreadingHTTPServer):
    """Loopback Messages API that validates tools the way Anthropic does."""

    daemon_threads = True

    def __init__(self, *, reject_tool: str | None = None) -> None:
        super().__init__(("127.0.0.1", 0), _AnthropicHandler)
        self.reject_tool = reject_tool
        self.requests: list[dict[str, Any]] = []
        self.rejected: list[str] = []

    @property
    def base_url(self) -> str:
        host, port = self.server_address[:2]
        return f"http://{host}:{port}/v1"


class _AnthropicHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:  # noqa: N802 - http.server contract
        server = self.server
        assert isinstance(server, _AnthropicServer)
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        server.requests.append(payload)
        tools = payload.get("tools") or []
        for index, tool in enumerate(tools):
            schema = tool.get("input_schema")
            refused = (
                not isinstance(schema, dict)
                or schema.get("type") != "object"
                or any(key in schema for key in _ANTHROPIC_TOP_LEVEL)
                or tool.get("name") == server.reject_tool
            )
            if refused:
                server.rejected.append(tool.get("name"))
                self._send(400, "application/json", _anthropic_tool_error(index))
                return
        self._send(200, *_anthropic_reply(bool(payload.get("stream"))))

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()
        self.close_connection = True

    def log_message(self, _format: str, *_args: object) -> None:
        return


@contextmanager
def _anthropic_server(**kwargs: Any) -> Iterator[_AnthropicServer]:
    server = _AnthropicServer(**kwargs)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class _ThirdPartyMCPHub:
    """The hub seams ``MCPToolProvider.compose_catalog`` reads (real names).

    Serves one third-party tool whose schema has a top-level ``anyOf`` -- the
    shape that must not be able to break every send.
    """

    def __init__(self) -> None:
        self.local_service = SimpleNamespace(get_inventory=lambda: {"tools": []})

    def get_kill_switch(self) -> bool:
        return False

    async def local_external_catalog(self) -> list[dict]:
        return [
            {
                "profile_id": "thirdparty",
                "is_connected": True,
                "discovery_snapshot": {
                    "tools": [
                        {
                            "name": "lookup",
                            "description": "Look something up by id or by name.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "name": {"type": "string"},
                                },
                                "anyOf": [
                                    {"required": ["id"]},
                                    {"required": ["name"]},
                                ],
                            },
                        }
                    ]
                },
            }
        ]

    def effective_tool_states(self, tools: list[HubTool]):
        return {
            (tool.server_key, tool.name): EffectiveToolState(
                state="ask", origin="global_default"
            )
            for tool in tools
        }


class _NoopExecutor:
    def execute(self, operation: str, arguments: dict, *, intent: str) -> str:
        raise AssertionError(f"a plain send dispatched {operation}")


def _default_profile_providers(
    tmp_path: Path,
) -> tuple[LocalToolProvider, MCPToolProvider]:
    """The tool providers a default Console profile composes for one run."""
    local = LocalToolProvider(
        workspace_root=tmp_path,
        resolve_state=lambda _hub: ALLOW,
        todo_store=SessionTodoStore(),
        workspace_executor=_NoopExecutor(),
    )
    loop = asyncio.new_event_loop()
    mcp = MCPToolProvider(service=_ThirdPartyMCPHub(), main_loop=loop)
    with use_run_id("compose"):
        asyncio.run(mcp.compose_catalog())
    loop.close()
    return local, mcp


def _run_console_reply(
    tmp_path: Path, resolution: ConsoleProviderResolution
) -> tuple[Any, ConsoleChatStore]:
    local, mcp = _default_profile_providers(tmp_path)
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hello")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    bridge = ConsoleAgentBridge(
        agent_runs_db=AgentRunsDB(tmp_path / "runs.db", client_id="schema-conformance"),
        store=store,
        provider_gateway=ConsoleProviderGateway(),
    )
    _run_id, outcome = bridge.run_reply(
        conversation_id="schema-conformance",
        session_id=session.id,
        resolution=resolution,
        assistant_message_id=assistant.id,
        model=resolution.model,
        session_system_prompt="",
        agent_messages=[{"role": "user", "content": "hello"}],
        should_cancel=lambda: False,
        local_provider=local,
        mcp_provider=mcp,
    )
    return outcome, store


def _anthropic_resolution(
    server: _AnthropicServer, *, streaming: bool
) -> ConsoleProviderResolution:
    return ConsoleProviderResolution(
        provider="anthropic",
        base_url=server.base_url,
        model="claude-haiku-4-5",
        ready=True,
        readiness_key="anthropic",
        execution_key="anthropic",
        api_key="test-anthropic-key-not-real",
        streaming=streaming,
    )


# ---------------------------------------------------------------------------
# AC#1 (automated half; the live half is in the task's evidence)
# ---------------------------------------------------------------------------


@pytest.mark.loopback_network
@pytest.mark.parametrize("streaming", [True, False], ids=["streaming", "non-streaming"])
def test_default_console_send_to_anthropic_gets_a_reply(tmp_path, streaming):
    with _anthropic_server() as server:
        outcome, store = _run_console_reply(
            tmp_path, _anthropic_resolution(server, streaming=streaming)
        )

    assert server.rejected == []
    assert outcome.status == "done", ConsoleChatController._agent_failure_visible_copy(
        outcome
    )
    assert outcome.final_text == _REPLY
    # The request really carried the default profile's tool set -- including
    # the three built-ins that used to break it and the third-party MCP tool
    # -- so a green run is not a run that simply sent no tools.
    names = {tool["name"] for tool in server.requests[0]["tools"]}
    assert {
        "watchlists_update_collection_sources",
        "watchlists_check_sources",
        "todo_update",
    } <= names
    assert "mcp__thirdparty__lookup" in names
    # The MCP tool's stripped "id or name" rule reaches the model in words.
    [lookup] = [
        tool
        for tool in server.requests[0]["tools"]
        if tool["name"] == "mcp__thirdparty__lookup"
    ]
    assert lookup["description"] == (
        "Look something up by id or by name. "
        "Argument rule: provide at least one of id or name."
    )
    for tool in server.requests[0]["tools"]:
        schema = tool["input_schema"]
        assert schema["type"] == "object", tool["name"]
        assert not {"anyOf", "oneOf", "allOf", "enum", "const", "not"} & set(schema), (
            tool["name"]
        )
    assistant_rows = [
        message
        for message in store.messages_for_session(store.sessions()[0].id)
        if message.role is ConsoleMessageRole.ASSISTANT
    ]
    assert assistant_rows[-1].content == _REPLY


# ---------------------------------------------------------------------------
# AC#4: a tool-definition rejection names the tool, not the model
# ---------------------------------------------------------------------------


@pytest.mark.loopback_network
@pytest.mark.parametrize(
    ("reject_tool", "advice"),
    [
        ("todo_update", "one of Chatbook's own tools, so please report it"),
        (
            "mcp__thirdparty__lookup",
            "Turn off the MCP server that provides it on the MCP screen, "
            "then send again.",
        ),
    ],
    ids=["chatbook-tool", "mcp-tool"],
)
def test_anthropic_tool_definition_rejection_names_the_tool_not_the_model(
    tmp_path, reject_tool, advice
):
    with _anthropic_server(reject_tool=reject_tool) as server:
        outcome, _store = _run_console_reply(
            tmp_path, _anthropic_resolution(server, streaming=True)
        )

    assert outcome.status != "done"
    assert server.rejected == [reject_tool]
    copy = ConsoleChatController._agent_failure_visible_copy(outcome)
    assert "HTTP 400" in copy
    assert (
        f"The provider rejected the tool definition for {reject_tool} before "
        "the model ran, so choosing another model will not help."
    ) in copy, copy
    assert advice in copy, copy
    for model_advice in _MODEL_ADVICE:
        assert model_advice not in copy, copy


def _openai_tool_error(name: str, index: int) -> Mock:
    body = json.dumps(
        {
            "error": {
                "message": (
                    f"Invalid schema for function '{name}': schema must have type "
                    "'object' and not have 'oneOf'/'anyOf'/'allOf'/'enum'/'const'/"
                    "'not' at the top level."
                ),
                "type": "invalid_request_error",
                "param": f"tools[{index}].function.parameters",
                "code": "invalid_function_parameters",
            }
        },
        indent=2,
    )
    response = Mock()
    response.status_code = 400
    response.text = body
    response.headers = {}
    response.raise_for_status.side_effect = requests.exceptions.HTTPError(
        response=response
    )
    return response


def test_openai_tool_definition_rejection_names_the_tool_not_the_model(tmp_path):
    rejected: list[str] = []

    def provider(_self, url, **kwargs):
        tools = kwargs["json"].get("tools") or []
        names = [
            tool.get("name") or (tool.get("function") or {}).get("name")
            for tool in tools
        ]
        index = names.index("watchlists_check_sources")
        rejected.append(names[index])
        return _openai_tool_error(names[index], index)

    resolution = ConsoleProviderResolution(
        provider="openai",
        base_url="https://api.openai.com/v1",
        model="gpt-4.1-mini",
        ready=True,
        readiness_key="openai",
        execution_key="openai",
        api_key="sk-test-not-real",
        streaming=False,
    )
    with patch("requests.Session.post", autospec=True, side_effect=provider):
        outcome, _store = _run_console_reply(tmp_path, resolution)

    assert rejected == ["watchlists_check_sources"]
    assert outcome.status != "done"
    copy = ConsoleChatController._agent_failure_visible_copy(outcome)
    assert "HTTP 400" in copy
    assert "tool definition for watchlists_check_sources" in copy, copy
    for advice in _MODEL_ADVICE:
        assert advice not in copy, copy


# ---------------------------------------------------------------------------
# AC#5: the Anthropic 400 path logs the provider's own message, redacted
# ---------------------------------------------------------------------------


def _logged_anthropic_bad_request(api_key: str, message: str) -> str:
    """Send one Anthropic request that fails 400 with ``message``; return the
    ERROR-level log text it produced."""
    body = json.dumps(
        {
            "type": "error",
            "error": {"type": "invalid_request_error", "message": message},
            "request_id": "req_logging",
        }
    )
    response = Mock()
    response.status_code = 400
    response.text = body
    response.raise_for_status.side_effect = requests.exceptions.HTTPError(
        response=response
    )
    captured: list[str] = []
    sink = logger.add(lambda record: captured.append(str(record)), level="ERROR")
    try:
        with patch("requests.Session.post", return_value=response):
            with pytest.raises(ChatBadRequestError):
                chat_api_call(
                    "anthropic",
                    messages_payload=[{"role": "user", "content": "hi"}],
                    api_key=api_key,
                    model="claude-haiku-4-5",
                    streaming=False,
                )
    finally:
        logger.remove(sink)
    return "\n".join(captured)


def test_anthropic_bad_request_logs_the_redacted_provider_message():
    api_key = "anthropic-canary-credential-7f3e9a1c2b"
    pattern_key = "sk-ant-api03-" + "Q" * 40

    log = _logged_anthropic_bad_request(
        api_key,
        "tools.34.custom.input_schema: input_schema does not support "
        f"oneOf, allOf, or anyOf at the top level (key {api_key}; "
        f"also {pattern_key})",
    )

    assert "Anthropic request failed; status=400" in log
    assert "input_schema does not support oneOf, allOf, or anyOf" in log
    assert "req_logging" in log
    assert api_key not in log
    assert pattern_key not in log


def test_anthropic_bad_request_log_masks_a_short_echoed_key():
    """Qodo #2931: the request's own key is masked however short it is.

    ``chat_with_anthropic`` sends any non-empty key, and a key this short
    matches none of ``redact_log_line``'s credential shapes, so only the
    literal mask stands between an echoing endpoint and the log.
    """
    api_key = "Zq7xKw"

    log = _logged_anthropic_bad_request(
        api_key, f"invalid x-api-key {api_key} for this workspace"
    )

    assert "Anthropic request failed; status=400" in log
    assert "invalid x-api-key" in log
    assert api_key not in log
