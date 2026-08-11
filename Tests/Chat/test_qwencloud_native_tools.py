"""Joined QwenCloud native-tool tests through the real Console path."""

from __future__ import annotations

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from typing import Any, Iterator

import pytest

from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_provider_gateway import (
    ConsoleProviderGateway,
    ConsoleProviderResolution,
)
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


_FINAL_SENTINEL = "QWENCLOUD-NATIVE-FINAL-SENTINEL"


def _sse(events: list[dict[str, Any]]) -> bytes:
    return (
        b"".join(
            b"data: " + json.dumps(event, separators=(",", ":")).encode() + b"\n\n"
            for event in events
        )
        + b"data: [DONE]\n\n"
    )


def _responses_tool_turn(
    calls: list[tuple[str, str, str]] | None = None,
    *,
    usage: dict[str, int] | None = None,
) -> bytes:
    calls = calls or [
        ("fc_A", "call_A", '{"expression":"6*7"}'),
        ("fc_B", "call_B", '{"expression":"8*8"}'),
    ]
    events: list[dict[str, Any]] = []
    sequence = 0
    output: list[dict[str, Any]] = []
    for output_index, (item_id, call_id, arguments) in enumerate(calls):
        item = {
            "id": item_id,
            "type": "function_call",
            "status": "completed",
            "call_id": call_id,
            "name": "calculator",
            "arguments": arguments,
        }
        events.extend(
            [
                {
                    "type": "response.output_item.added",
                    "sequence_number": sequence,
                    "output_index": output_index,
                    "item": {**item, "status": "in_progress", "arguments": ""},
                },
                {
                    "type": "response.function_call_arguments.delta",
                    "sequence_number": sequence + 1,
                    "output_index": output_index,
                    "item_id": item_id,
                    "delta": arguments,
                },
                {
                    "type": "response.function_call_arguments.done",
                    "sequence_number": sequence + 2,
                    "output_index": output_index,
                    "item_id": item_id,
                    "arguments": arguments,
                },
            ]
        )
        sequence += 3
        output.append(item)
    events.append(
        {
            "type": "response.completed",
            "sequence_number": sequence,
            "response": {
                "id": "resp_tools",
                "object": "response",
                "status": "completed",
                "output": output,
                "usage": usage
                or {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30},
            },
        }
    )
    return _sse(events)


def _responses_final_turn(text: str = _FINAL_SENTINEL) -> bytes:
    item = {
        "id": "msg_final",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }
    return _sse(
        [
            {
                "type": "response.output_item.added",
                "sequence_number": 0,
                "output_index": 0,
                "item": {**item, "status": "in_progress", "content": []},
            },
            {
                "type": "response.content_part.added",
                "sequence_number": 1,
                "output_index": 0,
                "item_id": "msg_final",
                "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []},
            },
            {
                "type": "response.output_text.delta",
                "sequence_number": 2,
                "output_index": 0,
                "item_id": "msg_final",
                "content_index": 0,
                "delta": text,
                "logprobs": [],
            },
            {
                "type": "response.output_text.done",
                "sequence_number": 3,
                "output_index": 0,
                "item_id": "msg_final",
                "content_index": 0,
                "text": text,
                "logprobs": [],
            },
            {
                "type": "response.completed",
                "sequence_number": 4,
                "response": {
                    "id": "resp_final",
                    "object": "response",
                    "status": "completed",
                    "output": [item],
                    "usage": {
                        "input_tokens": 30,
                        "output_tokens": 5,
                        "total_tokens": 35,
                    },
                },
            },
        ]
    )


def _chat_tool_turn(
    calls: list[tuple[str, str]] | None = None,
    *,
    usage: dict[str, int] | None = None,
) -> bytes:
    calls = calls or [
        ("call_A", '{"expression":"6*7"}'),
        ("call_B", '{"expression":"8*8"}'),
    ]
    return _sse(
        [
            {
                "id": "chatcmpl_tools",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "index": index,
                                    "id": call_id,
                                    "type": "function",
                                    "function": {
                                        "name": "calculator",
                                        "arguments": arguments,
                                    },
                                }
                                for index, (call_id, arguments) in enumerate(calls)
                            ],
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl_tools",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
            },
            {
                "id": "chatcmpl_tools",
                "choices": [],
                "usage": usage
                or {
                    "prompt_tokens": 20,
                    "completion_tokens": 10,
                    "total_tokens": 30,
                },
            },
        ]
    )


def _chat_final_turn(text: str = _FINAL_SENTINEL) -> bytes:
    return _sse(
        [
            {
                "id": "chatcmpl_final",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
            },
            {
                "id": "chatcmpl_final",
                "choices": [],
                "usage": {
                    "prompt_tokens": 30,
                    "completion_tokens": 5,
                    "total_tokens": 35,
                },
            },
        ]
    )


class _JoinedQwenServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        bodies: list[bytes],
        *,
        on_request: Any = None,
    ) -> None:
        super().__init__(("127.0.0.1", 0), _JoinedQwenHandler)
        self.bodies = list(bodies)
        self.requests: list[dict[str, Any]] = []
        self.on_request = on_request
        self._lock = threading.Lock()

    def handle_request_payload(self, path: str, payload: dict[str, Any]) -> bytes:
        with self._lock:
            self.requests.append({"path": path, "payload": payload})
            if self.on_request is not None:
                self.on_request(len(self.requests))
            assert self.bodies, "provider script exhausted"
            return self.bodies.pop(0)


class _JoinedQwenHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        server = self.server
        assert isinstance(server, _JoinedQwenServer)
        body = server.handle_request_payload(self.path, payload)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()
        self.close_connection = True

    def log_message(self, _format: str, *_args: object) -> None:
        return


@contextmanager
def _joined_qwen_server(
    bodies: list[bytes],
    *,
    on_request: Any = None,
) -> Iterator[_JoinedQwenServer]:
    server = _JoinedQwenServer(bodies, on_request=on_request)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _resolution(server: _JoinedQwenServer, api_mode: str) -> ConsoleProviderResolution:
    address = server.server_address
    assert isinstance(address, tuple)
    host, port = address[0], address[1]
    assert isinstance(host, str)
    assert isinstance(port, int)
    return ConsoleProviderResolution(
        provider="QwenCloud",
        base_url=f"http://{host}:{port}/compatible-mode/v1",
        model="qwen3.8-max",
        ready=True,
        readiness_key="qwencloud",
        execution_key="qwencloud",
        api_key="test-qwen-key",
        streaming=True,
        api_mode=api_mode,
    )


def _run_joined_reply(
    tmp_path: Any,
    server: _JoinedQwenServer,
    api_mode: str,
    *,
    should_cancel: Any = lambda: False,
) -> tuple[Any, ConsoleChatStore]:
    db = AgentRunsDB(tmp_path / f"{api_mode}.db", client_id="qwen-native")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="Calculate two expressions.",
    )
    assistant = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
    )
    bridge = ConsoleAgentBridge(
        agent_runs_db=db,
        store=store,
        provider_gateway=ConsoleProviderGateway(),
    )
    _run_id, outcome = bridge.run_reply(
        conversation_id="qwen-conversation",
        session_id=session.id,
        resolution=_resolution(server, api_mode),
        assistant_message_id=assistant.id,
        model="qwen3.8-max",
        session_system_prompt="",
        agent_messages=[{"role": "user", "content": "Calculate two expressions."}],
        should_cancel=should_cancel,
    )
    return outcome, store


@pytest.mark.parametrize("api_mode", ["responses", "chat_completions"])
@pytest.mark.allow_network
def test_console_agent_bridge_runs_qwencloud_two_call_continuation(
    tmp_path: Any,
    api_mode: str,
) -> None:
    bodies = (
        [_responses_tool_turn(), _responses_final_turn()]
        if api_mode == "responses"
        else [_chat_tool_turn(), _chat_final_turn()]
    )
    with _joined_qwen_server(bodies) as server:
        outcome, store = _run_joined_reply(tmp_path, server, api_mode)

    assert outcome.status == "done"
    assert outcome.final_text == _FINAL_SENTINEL
    assistant_rows = [
        message
        for session in store.sessions()
        for message in store.messages_for_session(session.id)
        if message.role is ConsoleMessageRole.ASSISTANT
    ]
    assert assistant_rows[-1].content == _FINAL_SENTINEL
    assert len(server.requests) == 2

    first = server.requests[0]
    second = server.requests[1]
    assert first["path"] == (
        "/compatible-mode/v1/responses"
        if api_mode == "responses"
        else "/compatible-mode/v1/chat/completions"
    )
    assert len(first["payload"]["tools"]) >= 2

    if api_mode == "responses":
        continuation = second["payload"]["input"]
        assert continuation[-4:] == [
            {
                "type": "function_call",
                "call_id": "call_A",
                "name": "calculator",
                "arguments": '{"expression":"6*7"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_A",
                "output": '{"expression": "6*7", "result": 42, "result_type": "int"}',
            },
            {
                "type": "function_call",
                "call_id": "call_B",
                "name": "calculator",
                "arguments": '{"expression":"8*8"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_B",
                "output": '{"expression": "8*8", "result": 64, "result_type": "int"}',
            },
        ]
        assert [item for item in continuation if item.get("role") == "user"] == [
            {"role": "user", "content": "Calculate two expressions."}
        ]
        assert second["payload"]["store"] is False
    else:
        continuation = second["payload"]["messages"]
        assistant = next(row for row in continuation if row.get("role") == "assistant")
        assert [call["id"] for call in assistant["tool_calls"]] == [
            "call_A",
            "call_B",
        ]
        tool_rows = [row for row in continuation if row.get("role") == "tool"]
        assert [row["tool_call_id"] for row in tool_rows] == ["call_A", "call_B"]
        assert [row for row in continuation if row.get("role") == "user"] == [
            {"role": "user", "content": "Calculate two expressions."}
        ]
        assert second["payload"]["preserve_thinking"] is False


@pytest.mark.parametrize("api_mode", ["responses", "chat_completions"])
@pytest.mark.allow_network
def test_qwencloud_tool_error_continues_structurally(
    tmp_path: Any,
    api_mode: str,
) -> None:
    invalid_arguments = '{"expression":"1/0"}'
    first = (
        _responses_tool_turn([("fc_error", "call_error", invalid_arguments)])
        if api_mode == "responses"
        else _chat_tool_turn([("call_error", invalid_arguments)])
    )
    with _joined_qwen_server(
        [first, _responses_final_turn("ERROR-RECOVERED")]
        if api_mode == "responses"
        else [first, _chat_final_turn("ERROR-RECOVERED")]
    ) as server:
        outcome, _store = _run_joined_reply(tmp_path, server, api_mode)

    assert outcome.status == "done"
    assert outcome.final_text == "ERROR-RECOVERED"
    assert len(server.requests) == 2
    second = server.requests[1]["payload"]
    if api_mode == "responses":
        output = second["input"][-1]
        assert output["type"] == "function_call_output"
        assert output["call_id"] == "call_error"
        assert output["output"].startswith("ERROR: ")
        assert "zero" in output["output"].lower()
        assert [row for row in second["input"] if row.get("role") == "user"] == [
            {"role": "user", "content": "Calculate two expressions."}
        ]
    else:
        output = second["messages"][-1]
        assert output["role"] == "tool"
        assert output["tool_call_id"] == "call_error"
        assert output["content"].startswith("ERROR: ")
        assert "zero" in output["content"].lower()
        assert second["preserve_thinking"] is False


def _responses_partial_call_then_text() -> bytes:
    return _sse(
        [
            {
                "type": "response.output_item.added",
                "sequence_number": 0,
                "output_index": 0,
                "item": {
                    "id": "fc_partial",
                    "type": "function_call",
                    "status": "in_progress",
                    "call_id": "call_partial",
                    "name": "calculator",
                    "arguments": "",
                },
            },
            {
                "type": "response.function_call_arguments.delta",
                "sequence_number": 1,
                "output_index": 0,
                "item_id": "fc_partial",
                "delta": '{"expression":',
            },
            {
                "type": "response.output_item.added",
                "sequence_number": 2,
                "output_index": 1,
                "item": {
                    "id": "msg_cancel",
                    "type": "message",
                    "role": "assistant",
                    "status": "in_progress",
                    "content": [],
                },
            },
            {
                "type": "response.content_part.added",
                "sequence_number": 3,
                "output_index": 1,
                "item_id": "msg_cancel",
                "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []},
            },
            {
                "type": "response.output_text.delta",
                "sequence_number": 4,
                "output_index": 1,
                "item_id": "msg_cancel",
                "content_index": 0,
                "delta": "cancel-checkpoint",
                "logprobs": [],
            },
        ]
    )


def _chat_partial_call_then_text() -> bytes:
    return _sse(
        [
            {
                "id": "chatcmpl_partial",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": "cancel-checkpoint",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_partial",
                                    "type": "function",
                                    "function": {
                                        "name": "calculator",
                                        "arguments": '{"expression":',
                                    },
                                }
                            ],
                        },
                        "finish_reason": None,
                    }
                ],
            }
        ]
    )


@pytest.mark.parametrize("api_mode", ["responses", "chat_completions"])
@pytest.mark.allow_network
def test_qwencloud_partial_call_cancellation_never_executes(
    tmp_path: Any,
    api_mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_chatbook.Tools.tool_executor import CalculatorTool

    executions: list[str] = []
    real_execute = CalculatorTool.execute

    async def recording_execute(self: CalculatorTool, expression: str) -> dict:
        executions.append(expression)
        return await real_execute(self, expression)

    monkeypatch.setattr(CalculatorTool, "execute", recording_execute)
    cancelled = threading.Event()
    body = (
        _responses_partial_call_then_text()
        if api_mode == "responses"
        else _chat_partial_call_then_text()
    )

    with _joined_qwen_server(
        [body],
        on_request=lambda request_count: cancelled.set(),
    ) as server:
        outcome, _store = _run_joined_reply(
            tmp_path,
            server,
            api_mode,
            should_cancel=cancelled.is_set,
        )

    assert outcome.status == "cancelled"
    assert len(server.requests) == 1
    assert executions == []
    assert not any(step.kind == "tool_result" for step in outcome.steps)


@pytest.mark.allow_network
def test_qwencloud_responses_usage_enforces_agent_budget(
    tmp_path: Any,
) -> None:
    usage = {
        "input_tokens": 1_000_000,
        "output_tokens": 1,
        "total_tokens": 1_000_001,
    }
    body = _responses_tool_turn(
        [("fc_budget", "call_budget", '{"expression":"2+2"}')],
        usage=usage,
    )
    with _joined_qwen_server([body]) as server:
        outcome, _store = _run_joined_reply(tmp_path, server, "responses")

    assert outcome.status == "stuck"
    assert outcome.total_tokens == 1_000_001
    assert len(server.requests) == 1
    assert outcome.steps[-1].kind == "error"
    assert outcome.steps[-1].summary == "token budget exhausted"
