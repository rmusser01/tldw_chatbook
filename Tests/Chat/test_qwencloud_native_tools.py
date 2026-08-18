"""Joined QwenCloud native-tool tests through the real Console path."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from typing import Any, Callable, Iterator

import pytest
import requests

from tldw_chatbook.Chat.console_agent_bridge import (
    CONSOLE_MAX_TOTAL_TOKENS,
    ConsoleAgentBridge,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_provider_gateway import (
    ConsoleProviderGateway,
    ConsoleProviderResolution,
)
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


_FINAL_SENTINEL = "QWENCLOUD-NATIVE-FINAL-SENTINEL"
_CANCEL_CHECKPOINT = "cancel-checkpoint"
# StreamGate keeps a fence-sized suffix; extra visible text lets the complete
# checkpoint reach the real store before the still-open response is cancelled.
_CANCEL_STREAM_TEXT = f"{_CANCEL_CHECKPOINT}-visible-stream-padding"
# Exceed requests' 8 KiB read size with valid SSE comments while withholding
# the terminal HTTP chunk, ensuring parser consumption precedes client closure.
_STREAM_HOLD_PADDING = b": hold-open\n\n" * 1024


def _sse(events: list[dict[str, Any]], *, done: bool = True) -> bytes:
    wire = b"".join(
        b"data: " + json.dumps(event, separators=(",", ":")).encode() + b"\n\n"
        for event in events
    )
    return wire + (b"data: [DONE]\n\n" if done else b"")


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


@dataclass(frozen=True)
class _StalledSSE:
    """One non-terminal chunk that remains live until the client closes."""

    prefix: bytes
    client_close_started: threading.Event


@dataclass(frozen=True)
class _ValidationFailure:
    message: str


_RequestValidator = Callable[[str, dict[str, Any]], None]
_ScriptedBody = bytes | _StalledSSE


class _JoinedQwenServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        bodies: Sequence[_ScriptedBody],
        *,
        validators: list[_RequestValidator] | None = None,
    ) -> None:
        super().__init__(("127.0.0.1", 0), _JoinedQwenHandler)
        self.bodies = list(bodies)
        self.validators = list(validators or [])
        self.requests: list[dict[str, Any]] = []
        self.validation_errors: list[str] = []
        self.stall_started = threading.Event()
        self.stall_timed_out = threading.Event()
        self._lock = threading.Lock()

    def handle_request_payload(
        self, path: str, payload: dict[str, Any]
    ) -> _ScriptedBody | _ValidationFailure:
        with self._lock:
            self.requests.append({"path": path, "payload": payload})
            assert self.bodies, "provider script exhausted"
            if self.validators:
                try:
                    self.validators[0](path, payload)
                except AssertionError as exc:
                    message = str(exc) or "scripted request validation failed"
                    self.validation_errors.append(message)
                    return _ValidationFailure(message)
                self.validators.pop(0)
            return self.bodies.pop(0)


class _JoinedQwenHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        server = self.server
        assert isinstance(server, _JoinedQwenServer)
        action = server.handle_request_payload(self.path, payload)
        if isinstance(action, _ValidationFailure):
            body = json.dumps({"error": {"message": action.message}}).encode()
            self.send_response(422)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()
            self.close_connection = True
            return
        if isinstance(action, _StalledSSE):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Connection", "close")
            self.end_headers()
            chunk = action.prefix
            self.wfile.write(f"{len(chunk):X}\r\n".encode())
            self.wfile.write(chunk)
            self.wfile.write(b"\r\n")
            self.wfile.flush()
            server.stall_started.set()
            if not action.client_close_started.wait(timeout=2):
                server.stall_timed_out.set()
            try:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            self.close_connection = True
            return
        body = action
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
    bodies: Sequence[_ScriptedBody],
    *,
    validators: list[_RequestValidator] | None = None,
) -> Iterator[_JoinedQwenServer]:
    server = _JoinedQwenServer(
        bodies,
        validators=validators,
    )
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
    agent_messages: list[dict[str, Any]] | None = None,
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
        agent_messages=(
            agent_messages
            if agent_messages is not None
            else [{"role": "user", "content": "Calculate two expressions."}]
        ),
        should_cancel=should_cancel,
    )
    return outcome, store


def _canonical_tool_calls() -> list[dict[str, Any]]:
    return [
        {
            "id": "call_A",
            "type": "function",
            "function": {
                "name": "calculator",
                "arguments": '{"expression":"6*7"}',
            },
        },
        {
            "id": "call_B",
            "type": "function",
            "function": {
                "name": "calculator",
                "arguments": '{"expression":"8*8"}',
            },
        },
    ]


def _assert_common_request(api_mode: str, path: str, payload: dict[str, Any]) -> None:
    expected_path = (
        "/compatible-mode/v1/responses"
        if api_mode == "responses"
        else "/compatible-mode/v1/chat/completions"
    )
    assert path == expected_path
    assert payload["stream"] is True
    assert any(
        tool.get("name") == "calculator"
        or tool.get("function", {}).get("name") == "calculator"
        for tool in payload["tools"]
    )


def _initial_request_validator(api_mode: str) -> _RequestValidator:
    def validate(path: str, payload: dict[str, Any]) -> None:
        _assert_common_request(api_mode, path, payload)
        if api_mode == "responses":
            assert payload["input"] == [
                {"role": "user", "content": "Calculate two expressions."}
            ]
            assert payload["store"] is False
        else:
            assert payload["messages"][0]["role"] == "system"
            assert payload["messages"][1:] == [
                {"role": "user", "content": "Calculate two expressions."}
            ]
            assert payload["preserve_thinking"] is False

    return validate


def _continuation_request_validator(
    api_mode: str,
    *,
    result_a: str,
    result_b: str,
) -> _RequestValidator:
    calls = _canonical_tool_calls()

    def validate(path: str, payload: dict[str, Any]) -> None:
        _assert_common_request(api_mode, path, payload)
        if api_mode == "responses":
            assert payload["input"] == [
                {"role": "user", "content": "Calculate two expressions."},
                {
                    "type": "function_call",
                    "call_id": "call_A",
                    "name": "calculator",
                    "arguments": '{"expression":"6*7"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_A",
                    "output": result_a,
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
                    "output": result_b,
                },
            ]
            return
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][1:] == [
            {"role": "user", "content": "Calculate two expressions."},
            {"role": "assistant", "content": "", "tool_calls": calls},
            {
                "role": "tool",
                "tool_call_id": "call_A",
                "content": result_a,
            },
            {
                "role": "tool",
                "tool_call_id": "call_B",
                "content": result_b,
            },
        ]
        assert payload["preserve_thinking"] is False

    return validate


@pytest.mark.parametrize("api_mode", ["responses", "chat_completions"])
@pytest.mark.allow_network
def test_console_agent_bridge_runs_qwencloud_two_call_continuation(
    tmp_path: Any,
    api_mode: str,
) -> None:
    result_a = '{"expression": "6*7", "result": 42, "result_type": "int"}'
    result_b = '{"expression": "8*8", "result": 64, "result_type": "int"}'
    bodies = (
        [_responses_tool_turn(), _responses_final_turn()]
        if api_mode == "responses"
        else [_chat_tool_turn(), _chat_final_turn()]
    )
    with _joined_qwen_server(
        bodies,
        validators=[
            _initial_request_validator(api_mode),
            _continuation_request_validator(
                api_mode,
                result_a=result_a,
                result_b=result_b,
            ),
        ],
    ) as server:
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
    assert server.validation_errors == []
    assert server.validators == []

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
                "output": result_a,
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
                "output": result_b,
            },
        ]
        assert [item for item in continuation if item.get("role") == "user"] == [
            {"role": "user", "content": "Calculate two expressions."}
        ]
        assert second["payload"]["store"] is False
    else:
        continuation = second["payload"]["messages"]
        assert continuation[0]["role"] == "system"
        assert continuation[1:] == [
            {"role": "user", "content": "Calculate two expressions."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": _canonical_tool_calls(),
            },
            {"role": "tool", "tool_call_id": "call_A", "content": result_a},
            {"role": "tool", "tool_call_id": "call_B", "content": result_b},
        ]
        assert second["payload"]["preserve_thinking"] is False


@pytest.mark.allow_network
def test_qwencloud_responses_joined_runtime_history_pairs_out_of_order_results(
    tmp_path: Any,
) -> None:
    """Runtime-shaped B/A result rows become adjacent A/A then B/B on wire."""
    result_a = "runtime result A"
    result_b = "runtime result B"
    runtime_history: list[dict[str, Any]] = [
        {"role": "user", "content": "Calculate two expressions."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": _canonical_tool_calls(),
        },
        {"role": "tool", "tool_call_id": "call_B", "content": result_b},
        {"role": "tool", "tool_call_id": "call_A", "content": result_a},
    ]
    with _joined_qwen_server(
        [_responses_final_turn("REORDERED-HISTORY-ACCEPTED")],
        validators=[
            _continuation_request_validator(
                "responses",
                result_a=result_a,
                result_b=result_b,
            )
        ],
    ) as server:
        outcome, _store = _run_joined_reply(
            tmp_path,
            server,
            "responses",
            agent_messages=runtime_history,
        )

    assert outcome.status == "done"
    assert outcome.final_text == "REORDERED-HISTORY-ACCEPTED"
    assert len(server.requests) == 1
    assert server.validation_errors == []
    assert server.validators == []


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
                "delta": _CANCEL_STREAM_TEXT,
                "logprobs": [],
            },
        ],
        done=False,
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
                            "content": _CANCEL_STREAM_TEXT,
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
        ],
        done=False,
    )


def _assert_partial_call_precedes_checkpoint(api_mode: str, body: bytes) -> str:
    """Validate the cancellation fixture cannot degrade to ordinary text."""
    events = [
        json.loads(record.removeprefix(b"data: "))
        for record in body.split(b"\n\n")
        if record.startswith(b"data: {")
    ]
    if api_mode == "responses":
        partial_item_indexes = [
            index
            for index, event in enumerate(events)
            if event.get("type") == "response.output_item.added"
            and event.get("item", {}).get("type") == "function_call"
            and event.get("item", {}).get("status") == "in_progress"
        ]
        partial_delta_indexes = [
            index
            for index, event in enumerate(events)
            if event.get("type") == "response.function_call_arguments.delta"
            and event.get("delta") == '{"expression":'
        ]
        checkpoint_indexes = [
            index
            for index, event in enumerate(events)
            if event.get("type") == "response.output_text.delta"
            and event.get("delta") == _CANCEL_STREAM_TEXT
        ]
        assert partial_item_indexes
        assert partial_delta_indexes
        assert checkpoint_indexes
        assert partial_item_indexes[0] < checkpoint_indexes[0]
        assert partial_delta_indexes[0] < checkpoint_indexes[0]
        return "response.function_call_arguments.delta"

    checkpoint_events = []
    for event in events:
        choices = event.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        if choices[0].get("delta", {}).get("content") == _CANCEL_STREAM_TEXT:
            checkpoint_events.append(event)
    assert checkpoint_events
    tool_calls = checkpoint_events[0]["choices"][0]["delta"].get("tool_calls")
    assert isinstance(tool_calls, list) and len(tool_calls) == 1
    assert tool_calls[0]["function"] == {
        "name": "calculator",
        "arguments": '{"expression":',
    }
    return "chat.delta.tool_calls"


@pytest.mark.parametrize("api_mode", ["responses", "chat_completions"])
def test_partial_call_cancellation_fixture_rejects_text_only_mutation(
    api_mode: str,
) -> None:
    text_only = (
        _responses_final_turn(_CANCEL_STREAM_TEXT)
        if api_mode == "responses"
        else _chat_final_turn(_CANCEL_STREAM_TEXT)
    )
    with pytest.raises(AssertionError):
        _assert_partial_call_precedes_checkpoint(api_mode, text_only)


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
    checkpoint_observed = threading.Event()
    body = (
        _responses_partial_call_then_text()
        if api_mode == "responses"
        else _chat_partial_call_then_text()
    )
    partial_event = _assert_partial_call_precedes_checkpoint(api_mode, body)
    close_calls: list[int] = []
    client_close_started = threading.Event()
    real_response_close = requests.Response.close
    real_append_stream_chunk = ConsoleChatStore.append_stream_chunk

    def recording_stream_chunk(
        store: ConsoleChatStore, message_id: str, chunk: str
    ) -> Any:
        message = real_append_stream_chunk(store, message_id, chunk)
        if _CANCEL_CHECKPOINT in chunk:
            checkpoint_observed.set()
            cancelled.set()
        return message

    monkeypatch.setattr(
        ConsoleChatStore,
        "append_stream_chunk",
        recording_stream_chunk,
    )
    assert not cancelled.is_set()

    with _joined_qwen_server(
        [_StalledSSE(body + _STREAM_HOLD_PADDING, client_close_started)]
    ) as server:
        address = server.server_address
        assert isinstance(address, tuple)
        host, port = address[0], address[1]
        assert isinstance(host, str)
        assert isinstance(port, int)
        request_prefix = f"http://{host}:{port}/compatible-mode/v1/"

        def recording_response_close(response: requests.Response) -> None:
            if str(response.url).startswith(request_prefix):
                close_calls.append(id(response))
                client_close_started.set()
            real_response_close(response)

        monkeypatch.setattr(requests.Response, "close", recording_response_close)
        outcome, _store = _run_joined_reply(
            tmp_path,
            server,
            api_mode,
            should_cancel=cancelled.is_set,
        )

    assert outcome.status == "cancelled"
    assert checkpoint_observed.is_set()
    assert partial_event == (
        "response.function_call_arguments.delta"
        if api_mode == "responses"
        else "chat.delta.tool_calls"
    )
    assert len(server.requests) == 1
    assert server.stall_started.is_set()
    assert not server.stall_timed_out.is_set()
    assert len(close_calls) == 1
    assert executions == []
    assert not any(step.kind in {"tool_call", "tool_result"} for step in outcome.steps)
    request_payload = server.requests[0]["payload"]
    if api_mode == "responses":
        assert not any(
            item.get("type") == "function_call_output"
            for item in request_payload["input"]
        )
    else:
        assert not any(
            item.get("role") == "tool" for item in request_payload["messages"]
        )


@pytest.mark.allow_network
def test_qwencloud_responses_usage_enforces_agent_budget(
    tmp_path: Any,
) -> None:
    # Derived from the shipped budget so this keeps testing "one turn that
    # exhausts the budget" rather than "one turn of exactly 1,000,001
    # tokens", and does not need editing again the next time the default
    # moves. (On this branch the default is still CONSOLE_MAX_TOTAL_TOKENS;
    # PR #1824's DEFAULT_CONSOLE_RUN_BUDGET supersedes it there.)
    budget = CONSOLE_MAX_TOTAL_TOKENS
    usage = {
        "input_tokens": budget,
        "output_tokens": 1,
        "total_tokens": budget + 1,
    }
    body = _responses_tool_turn(
        [("fc_budget", "call_budget", '{"expression":"2+2"}')],
        usage=usage,
    )
    with _joined_qwen_server([body]) as server:
        outcome, _store = _run_joined_reply(tmp_path, server, "responses")

    assert outcome.status == "stuck"
    assert outcome.total_tokens == budget + 1
    assert len(server.requests) == 1
    assert outcome.steps[-1].kind == "error"
    assert outcome.steps[-1].summary == "token budget exhausted"
