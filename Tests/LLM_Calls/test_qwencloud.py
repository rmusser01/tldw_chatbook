"""Pure request-translation contracts for the QwenCloud adapter."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import socket
import threading
from types import SimpleNamespace
from typing import Any, Iterator, Never

import pytest
import requests
from loguru import logger
from urllib3.exceptions import ReadTimeoutError
from urllib3.util import Retry

from tldw_chatbook.Chat.Chat_Deps import (
    ChatAuthenticationError,
    ChatBadRequestError,
    ChatConfigurationError,
    ChatProviderError,
    ChatRateLimitError,
)
import tldw_chatbook.LLM_Calls.qwencloud as qwencloud
from tldw_chatbook.LLM_Calls.qwencloud import (
    build_qwencloud_payload,
    chat_with_qwencloud,
    normalize_qwencloud_api_mode,
    normalize_qwencloud_base_url,
    resolve_qwencloud_api_key,
)
from tldw_chatbook.Utils.sensitive_llm_logging import sensitive_llm_request


class _TransportResponse:
    def __init__(
        self,
        payload: dict[str, Any],
        *,
        status_code: int = 200,
        text: str = "",
        headers: dict[str, str] | None = None,
        chunks: list[bytes] | None = None,
        close_error: Exception | None = None,
    ) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}
        self.chunks = chunks or []
        self.close_error = close_error
        self.closed = False
        self.close_calls = 0

    def json(self) -> dict[str, Any]:
        return deepcopy(self._payload)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(response=self)  # type: ignore[arg-type]

    def close(self) -> None:
        self.close_calls += 1
        self.closed = True
        if self.close_error is not None:
            raise self.close_error

    def iter_content(self, chunk_size: int) -> Iterator[bytes]:
        assert chunk_size > 0
        yield from self.chunks


class _RecordingSession:
    def __init__(
        self,
        response: _TransportResponse,
        *,
        error: requests.exceptions.RequestException | None = None,
        close_error: Exception | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.close_error = close_error
        self.mounts: list[tuple[str, object]] = []
        self.posts: list[dict[str, Any]] = []
        self.closed = False
        self.close_calls = 0

    def __enter__(self) -> _RecordingSession:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def mount(self, prefix: str, adapter: object) -> None:
        self.mounts.append((prefix, adapter))

    def post(self, url: str, **kwargs: Any) -> _TransportResponse:
        self.posts.append({"url": url, **deepcopy(kwargs)})
        if self.error is not None:
            raise self.error
        return self.response

    def close(self) -> None:
        self.close_calls += 1
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


_SCRIPTED_SUCCESS_BODY = (
    b'{"choices":[{"message":{"role":"assistant","content":"ok"},'
    b'"finish_reason":"stop"}]}'
)
_STALLED_ERROR_CANARY = b"RAW-STALLED-400-CANARY"
_TRUNCATED_BODY_CANARY = b'RAW-TRUNCATED-BODY-CANARY{"choices":['
_INVALID_JSON_CANARY = b"RAW-INVALID-JSON-CANARY"
_INVALID_GZIP_CANARY = b"RAW-CONTENT-DECODING-CANARY"
_ScriptedAction = str | tuple[int, dict[str, str]]


class _ScriptedQwenHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length:
            self.rfile.read(content_length)

        server = self.server
        assert isinstance(server, _ScriptedQwenServer)
        action = server.next_action()
        if action == "truncated":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(_TRUNCATED_BODY_CANARY) + 100))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(_TRUNCATED_BODY_CANARY)
            self.wfile.flush()
            self.close_connection = True
            return
        if action == "invalid_json":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(_INVALID_JSON_CANARY)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(_INVALID_JSON_CANARY)
            self.wfile.flush()
            self.close_connection = True
            return
        if action == "invalid_gzip":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(_INVALID_GZIP_CANARY)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(_INVALID_GZIP_CANARY)
            self.wfile.flush()
            self.close_connection = True
            return
        if action == "stall_400":
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(_STALLED_ERROR_CANARY) + 100))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(_STALLED_ERROR_CANARY)
            self.wfile.flush()
            server.release_stalls.wait(timeout=1)
            self.close_connection = True
            return
        if action == "stall":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(_SCRIPTED_SUCCESS_BODY)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.flush()
            server.release_stalls.wait(timeout=1)
            self.close_connection = True
            return

        if action == "success":
            status_code = 200
            headers: dict[str, str] = {}
            body = _SCRIPTED_SUCCESS_BODY
        else:
            status_code, headers = action
            body = b'{"error":{"message":"scripted provider failure"}}'

        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        for name, value in headers.items():
            self.send_header(name, value)
        self.end_headers()
        try:
            self.wfile.write(body)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        self.close_connection = True

    def log_message(self, _format: str, *_args: object) -> None:
        return


class _ScriptedQwenServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, actions: list[_ScriptedAction]) -> None:
        super().__init__(("127.0.0.1", 0), _ScriptedQwenHandler)
        self.actions = actions
        self.attempts: list[_ScriptedAction] = []
        self.release_stalls = threading.Event()
        self._attempt_lock = threading.Lock()

    def next_action(self) -> _ScriptedAction:
        with self._attempt_lock:
            attempt_index = len(self.attempts)
            action = (
                self.actions[attempt_index]
                if attempt_index < len(self.actions)
                else (599, {})
            )
            self.attempts.append(action)
            return action


@contextmanager
def _scripted_qwen_server(
    actions: list[_ScriptedAction],
) -> Iterator[tuple[str, _ScriptedQwenServer]]:
    server = _ScriptedQwenServer(actions)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}/compatible-mode/v1", server
    finally:
        server.release_stalls.set()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)


def _configure_qwencloud_transport(
    monkeypatch: pytest.MonkeyPatch,
    *,
    retries: int,
    retry_delay: float = 0,
    timeout: float = 0.05,
) -> None:
    monkeypatch.setattr(
        qwencloud,
        "get_runtime_config_snapshot",
        lambda: SimpleNamespace(
            values={
                "api_settings": {
                    "qwencloud": {
                        "timeout": timeout,
                        "retries": retries,
                        "retry_delay": retry_delay,
                    }
                }
            }
        ),
    )


def _track_real_transport_resources(
    monkeypatch: pytest.MonkeyPatch,
    *,
    connect_timeout: float | None = None,
) -> tuple[list[str], list[requests.Response], set[int]]:
    post_urls: list[str] = []
    returned_responses: list[requests.Response] = []
    closed_response_ids: set[int] = set()
    real_post = requests.Session.post
    real_close = requests.Response.close

    def recording_post(
        session: requests.Session, url: str, **kwargs: Any
    ) -> requests.Response:
        post_urls.append(url)
        if connect_timeout is not None:
            read_timeout = kwargs.get("timeout")
            assert isinstance(read_timeout, (int, float)) and not isinstance(
                read_timeout, bool
            )
            kwargs["timeout"] = (connect_timeout, float(read_timeout))
        response = real_post(session, url, **kwargs)
        returned_responses.append(response)
        return response

    def recording_close(response: requests.Response) -> None:
        closed_response_ids.add(id(response))
        real_close(response)

    monkeypatch.setattr(requests.Session, "post", recording_post)
    monkeypatch.setattr(requests.Response, "close", recording_close)
    return post_urls, returned_responses, closed_response_ids


def _track_real_session_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> set[int]:
    closed_session_ids: set[int] = set()
    real_close = requests.Session.close

    def recording_close(session: requests.Session) -> None:
        closed_session_ids.add(id(session))
        real_close(session)

    monkeypatch.setattr(requests.Session, "close", recording_close)
    return closed_session_ids


def _call_scripted_qwencloud(api_base_url: str) -> dict[str, Any]:
    result = chat_with_qwencloud(
        input_data=[{"role": "user", "content": "hello"}],
        model="qwen3.8-max",
        api_key="key",
        streaming=False,
        api_base_url=api_base_url,
        api_mode="chat_completions",
    )
    assert isinstance(result, dict)
    return result


@contextmanager
def _captured_qwencloud_logs() -> Iterator[list[str]]:
    records: list[str] = []
    sink_id = logger.add(lambda message: records.append(str(message)), level="DEBUG")
    try:
        yield records
    finally:
        logger.remove(sink_id)


def test_api_mode_config_then_default_and_exact_values() -> None:
    assert normalize_qwencloud_api_mode(None) == "responses"
    assert (
        normalize_qwencloud_api_mode(
            None, provider_settings={"api_mode": " CHAT_COMPLETIONS "}
        )
        == "chat_completions"
    )
    assert (
        normalize_qwencloud_api_mode(
            " Responses ", provider_settings={"api_mode": "chat_completions"}
        )
        == "responses"
    )
    assert (
        normalize_qwencloud_api_mode(
            "responses",
            provider_settings=7,  # type: ignore[arg-type]
        )
        == "responses"
    )

    for rejected in ("response", "chat", "chat-completions", "unknown", ""):
        with pytest.raises(ChatConfigurationError) as exc_info:
            normalize_qwencloud_api_mode(rejected)
        assert exc_info.value.provider == "qwencloud"


def test_base_url_normalizes_base_and_pasted_endpoints() -> None:
    expected = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    assert normalize_qwencloud_base_url(None) == expected
    assert normalize_qwencloud_base_url(f"  {expected}///  ") == expected
    assert normalize_qwencloud_base_url(f"{expected}/responses") == expected
    assert normalize_qwencloud_base_url(f"{expected}/chat/completions/") == expected
    assert (
        normalize_qwencloud_base_url("http://gateway.internal:8080/team/qwen/v1/")
        == "http://gateway.internal:8080/team/qwen/v1"
    )


def test_base_url_rejects_unsafe_or_malformed_values() -> None:
    rejected = (
        "dashscope.example/v1",
        "ftp://dashscope.example/v1",
        "https:///v1",
        "https://user:secret@dashscope.example/v1",
        "https://dashscope.example/v1?tenant=a",
        "https://dashscope.example/v1#fragment",
        "https://dashscope.example/v1?",
        "https://dashscope.example/v1#",
        "https://dashscope.example/v1/models",
        "https://dashscope.example/v1/responses/responses",
        "https://dashscope.example/v1/chat/completions/chat/completions",
        "https://dashscope.example/v1/responses/extra",
        "https://dashscope.example/v1/chat/completions/extra",
        "https://dashscope.example//compatible-mode/v1",
        "https://bad host.example/v1",
        "https://dashscope.example:/v1",
        "https://dashscope.example\n.evil/v1",
        "https://dashscope.example/%zz",
        "   ",
    )
    for value in rejected:
        with pytest.raises(ChatConfigurationError) as exc_info:
            normalize_qwencloud_base_url(value)
        assert exc_info.value.provider == "qwencloud"
        assert "secret" not in str(exc_info.value)


def test_base_url_rejects_malformed_authorities() -> None:
    malformed_authorities = (
        "https://good.example\\evil/v1",
        "https://%zz/v1",
        "https://good.example|evil/v1",
        "https://good.example^evil/v1",
        "https://good.example\x00evil/v1",
    )
    for value in malformed_authorities:
        with pytest.raises(ChatConfigurationError) as exc_info:
            normalize_qwencloud_base_url(value)
        assert exc_info.value.provider == "qwencloud"


@pytest.mark.parametrize(
    "value",
    [
        "https://dashscope.example/api%2fv2",
        "https://dashscope.example/api%252Fv2",
        "https://dashscope.example/api%5Cv2",
        "https://dashscope.example/api/v2/%2e/responses",
        "https://dashscope.example/api/v2/%2E%2e/responses",
        "https://dashscope.example/api/v2/%252e%252e/responses",
        "https://dashscope.example/api/v2/res%70onses",
        "https://dashscope.example/api/v2/RES%70ONSES",
        "https://dashscope.example/api/v2/chat/%63ompletions",
        "https://dashscope.example/api/v2/mod%65ls",
        "https://dashscope.example/api/v2/res%2570onses",
        "https://dashscope.example/api/v2/chat/%2563ompletions",
        "https://dashscope.example/api/v2/mod%2565ls",
        "https://dashscope.example/api/v2/res%252570onses",
    ],
)
def test_base_url_rejects_encoded_endpoint_structure(value: str) -> None:
    with pytest.raises(ChatConfigurationError) as exc_info:
        normalize_qwencloud_base_url(value)

    assert exc_info.value.provider == "qwencloud"
    assert value not in str(exc_info.value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("https://dashscope.example", "https://dashscope.example"),
        (
            "https://dashscope.example/api/RESPONSES",
            "https://dashscope.example/api/RESPONSES",
        ),
        (
            "https://dashscope.example/tenant-responses/api/v2",
            "https://dashscope.example/tenant-responses/api/v2",
        ),
        (
            "https://dashscope.example/completion-gateway/api/v2",
            "https://dashscope.example/completion-gateway/api/v2",
        ),
        (
            "https://dashscope.example/api/v2/responses-extra",
            "https://dashscope.example/api/v2/responses-extra",
        ),
        (
            "https://dashscope.example/api/v2/chat/completions-extra",
            "https://dashscope.example/api/v2/chat/completions-extra",
        ),
        (
            "https://dashscope.example/api/v2/myresponses",
            "https://dashscope.example/api/v2/myresponses",
        ),
        (
            "https://dashscope.example/tenant%20alpha/api/v2",
            "https://dashscope.example/tenant%20alpha/api/v2",
        ),
        (
            "https://dashscope.example/tenant%2520alpha/api/v2",
            "https://dashscope.example/tenant%2520alpha/api/v2",
        ),
        (
            "https://dashscope.example/mod%65ls/api/v2",
            "https://dashscope.example/mod%65ls/api/v2",
        ),
    ],
)
def test_base_url_preserves_valid_arbitrary_prefix_contract(
    value: str,
    expected: str,
) -> None:
    assert normalize_qwencloud_base_url(value) == expected


def test_api_key_precedence_is_provider_isolated() -> None:
    environ = {
        "DASHSCOPE_API_KEY": "default-env-key",
        "QWEN_KEY": "selected-env-key",
        "OPENAI_API_KEY": "other-provider-key",
    }
    settings = {
        "api_key": "modern-key",
        "api_key_env_var": "QWEN_KEY",
        "openai_api_key": "other-provider-setting",
    }

    assert (
        resolve_qwencloud_api_key(
            "trusted-key", provider_settings=settings, environ=environ
        )
        == "trusted-key"
    )
    assert (
        resolve_qwencloud_api_key(
            "trusted-key",
            provider_settings=7,  # type: ignore[arg-type]
            environ=7,  # type: ignore[arg-type]
        )
        == "trusted-key"
    )
    assert (
        resolve_qwencloud_api_key(None, provider_settings=settings, environ=environ)
        == "modern-key"
    )
    assert (
        resolve_qwencloud_api_key(
            None,
            provider_settings={"api_key": "modern-key"},
            environ=7,  # type: ignore[arg-type]
        )
        == "modern-key"
    )
    assert (
        resolve_qwencloud_api_key(
            None,
            provider_settings={"api_key_env_var": "QWEN_KEY"},
            environ=environ,
        )
        == "selected-env-key"
    )
    assert resolve_qwencloud_api_key(None, environ=environ) == "default-env-key"

    with pytest.raises(ChatConfigurationError) as exc_info:
        resolve_qwencloud_api_key(
            None,
            provider_settings={"openai_api_key": "do-not-use"},
            environ={"OPENAI_API_KEY": "do-not-use"},
        )
    assert exc_info.value.provider == "qwencloud"
    assert "do-not-use" not in str(exc_info.value)


def test_api_key_resolution_strips_and_skips_repository_placeholders() -> None:
    assert resolve_qwencloud_api_key("  explicit-key  ", environ={}) == "explicit-key"
    assert (
        resolve_qwencloud_api_key(
            " YOUR_KEY ",
            provider_settings={"api_key": "  modern-key  "},
            environ={"DASHSCOPE_API_KEY": "env-key"},
        )
        == "modern-key"
    )
    assert (
        resolve_qwencloud_api_key(
            "<API_KEY_HERE>",
            provider_settings={
                "api_key": " your-api-key ",
                "api_key_env_var": "QWEN_KEY",
            },
            environ={"QWEN_KEY": "  env-key  "},
        )
        == "env-key"
    )

    with pytest.raises(ChatConfigurationError) as exc_info:
        resolve_qwencloud_api_key(
            "YOUR_KEY",
            provider_settings={"api_key": " <API_KEY_HERE> "},
            environ={"DASHSCOPE_API_KEY": " your_key "},
        )
    assert exc_info.value.provider == "qwencloud"


def test_resolution_helpers_reject_invalid_mapping_shapes() -> None:
    with pytest.raises(ChatConfigurationError) as exc_info:
        normalize_qwencloud_api_mode(
            None,
            provider_settings=7,  # type: ignore[arg-type]
        )
    assert exc_info.value.provider == "qwencloud"

    with pytest.raises(ChatConfigurationError) as exc_info:
        resolve_qwencloud_api_key(
            None,
            provider_settings=7,
            environ={},  # type: ignore[arg-type]
        )
    assert exc_info.value.provider == "qwencloud"

    with pytest.raises(ChatConfigurationError) as exc_info:
        resolve_qwencloud_api_key(
            None,
            provider_settings={},
            environ=7,  # type: ignore[arg-type]
        )
    assert exc_info.value.provider == "qwencloud"


def test_responses_payload_has_exact_allowlist_and_stateless_invariants() -> None:
    payload = build_qwencloud_payload(
        api_mode="responses",
        model="qwen3.8-max",
        system_message="Be concise.",
        messages_payload=[{"role": "user", "content": "Hello"}],
        streaming=True,
        temp=0.2,
        topp=0.8,
        topk=20,
        max_tokens=128,
        seed=7,
        presence_penalty=0.3,
        stop=["END"],
        response_format={"type": "json_object"},
        n=2,
        logprobs=True,
        top_logprobs=3,
    )

    assert payload == {
        "model": "qwen3.8-max",
        "input": [{"role": "user", "content": "Hello"}],
        "instructions": "Be concise.",
        "stream": True,
        "store": False,
        "temperature": 0.2,
        "top_p": 0.8,
        "max_output_tokens": 128,
    }
    assert "previous_response_id" not in payload
    assert "conversation" not in payload

    with pytest.raises(ChatBadRequestError) as exc_info:
        build_qwencloud_payload(
            api_mode="responses",
            model="qwen3.8-max",
            system_message=None,
            messages_payload=[{"role": "user", "content": "Hello"}],
            streaming=False,
            max_tokens=15,
        )
    assert exc_info.value.provider == "qwencloud"

    with pytest.raises(ChatBadRequestError) as exc_info:
        build_qwencloud_payload(
            api_mode="responses",
            model="qwen3.8-max",
            system_message=None,
            messages_payload=[{"role": "user", "content": "Hello"}],
            streaming=False,
            max_tokens="128",  # type: ignore[arg-type]
        )
    assert exc_info.value.provider == "qwencloud"


def test_responses_system_message_maps_to_instructions() -> None:
    kwargs = {
        "api_mode": "responses",
        "model": "qwen3.8-max",
        "streaming": False,
    }

    from_leading_row = build_qwencloud_payload(
        **kwargs,
        system_message=None,
        messages_payload=[
            {"role": "system", "content": "Be precise."},
            {"role": "user", "content": "Hello"},
        ],
    )
    assert from_leading_row["instructions"] == "Be precise."
    assert from_leading_row["input"] == [{"role": "user", "content": "Hello"}]

    duplicate = build_qwencloud_payload(
        **kwargs,
        system_message="Be precise.",
        messages_payload=[
            {"role": "system", "content": "Be precise."},
            {"role": "user", "content": "Hello"},
        ],
    )
    assert duplicate["instructions"] == "Be precise."
    assert duplicate["input"] == [{"role": "user", "content": "Hello"}]

    with pytest.raises(ChatBadRequestError):
        build_qwencloud_payload(
            **kwargs,
            system_message="Be concise.",
            messages_payload=[
                {"role": "system", "content": "Be expansive."},
                {"role": "user", "content": "Hello"},
            ],
        )
    with pytest.raises(ChatBadRequestError):
        build_qwencloud_payload(
            **kwargs,
            system_message=None,
            messages_payload=[
                {"role": "user", "content": "Hello"},
                {"role": "system", "content": "Too late."},
            ],
        )


def test_leading_system_row_with_tool_calls_is_rejected() -> None:
    messages = [
        {
            "role": "system",
            "content": "Never execute tools.",
            "tool_calls": [
                {
                    "id": "call_system",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        },
        {"role": "user", "content": "Hello"},
    ]
    for mode in ("responses", "chat_completions"):
        with pytest.raises(ChatBadRequestError) as exc_info:
            build_qwencloud_payload(
                api_mode=mode,
                model="qwen3.8-max",
                system_message=None,
                messages_payload=messages,
                streaming=False,
            )
        assert exc_info.value.provider == "qwencloud"


def test_responses_reasoning_effort_enum_is_exact() -> None:
    base = {
        "api_mode": "responses",
        "model": "qwen3.8-max",
        "system_message": None,
        "messages_payload": [{"role": "user", "content": "Hello"}],
        "streaming": False,
    }
    for effort in ("none", "minimal", "low", "medium", "high", "xhigh", "max"):
        payload = build_qwencloud_payload(**base, reasoning_effort=effort)
        assert payload["reasoning"] == {"effort": effort}

    for rejected in ("", "LOW", "ultra", "maximum"):
        with pytest.raises(ChatBadRequestError) as exc_info:
            build_qwencloud_payload(**base, reasoning_effort=rejected)
        assert exc_info.value.provider == "qwencloud"
    with pytest.raises(ChatBadRequestError) as exc_info:
        build_qwencloud_payload(**base, reasoning_effort=[])  # type: ignore[arg-type]
    assert exc_info.value.provider == "qwencloud"


def test_chat_payload_has_exact_allowlist_and_thinking_invariant() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look something up.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    payload = build_qwencloud_payload(
        api_mode="chat_completions",
        model="qwen3.8-max",
        system_message="Be precise.",
        messages_payload=[{"role": "user", "content": "Hello"}],
        streaming=True,
        tools=tools,
        tool_choice="auto",
        temp=0.2,
        topp=0.8,
        topk=20,
        max_tokens=128,
        seed=7,
        presence_penalty=0.3,
        stop=["END"],
        response_format={"type": "json_object"},
        n=1,
        logprobs=True,
        top_logprobs=3,
        reasoning_effort="high",
    )
    assert payload == {
        "model": "qwen3.8-max",
        "messages": [
            {"role": "system", "content": "Be precise."},
            {"role": "user", "content": "Hello"},
        ],
        "stream": True,
        "temperature": 0.2,
        "top_p": 0.8,
        "top_k": 20,
        "max_completion_tokens": 128,
        "seed": 7,
        "presence_penalty": 0.3,
        "stop": ["END"],
        "response_format": {"type": "json_object"},
        "n": 1,
        "logprobs": True,
        "top_logprobs": 3,
        "tools": tools,
        "tool_choice": "auto",
        "reasoning_effort": "high",
        "preserve_thinking": False,
        "stream_options": {"include_usage": True},
    }

    nonstream = build_qwencloud_payload(
        api_mode="chat_completions",
        model="qwen3.8-max",
        system_message=None,
        messages_payload=[{"role": "user", "content": "Hello"}],
        streaming=False,
        response_format={"type": "text"},
        n=2,
    )
    assert nonstream["n"] == 2
    assert nonstream["preserve_thinking"] is False
    assert "stream_options" not in nonstream

    with pytest.raises(ChatBadRequestError):
        build_qwencloud_payload(
            api_mode="chat_completions",
            model="qwen3.8-max",
            system_message=None,
            messages_payload=[{"role": "user", "content": "Hello"}],
            streaming=False,
            tools=tools,
            n=2,
        )
    for rejected_format in (
        {"type": "json_schema"},
        {"type": "text", "extra": "not-allowed"},
    ):
        with pytest.raises(ChatBadRequestError):
            build_qwencloud_payload(
                api_mode="chat_completions",
                model="qwen3.8-max",
                system_message=None,
                messages_payload=[{"role": "user", "content": "Hello"}],
                streaming=False,
                response_format=rejected_format,
            )


@pytest.mark.parametrize(
    "api_mode",
    ("responses", "chat_completions"),
    ids=("responses", "chat-completions"),
)
@pytest.mark.parametrize(
    ("field", "invalid_value"),
    (
        pytest.param("model", "", id="model-empty"),
        pytest.param("model", "   ", id="model-blank"),
        pytest.param("model", 7, id="model-non-string"),
        pytest.param("streaming", "no", id="streaming-non-bool"),
        pytest.param("temp", "hot", id="temperature-non-numeric"),
        pytest.param("temp", True, id="temperature-bool"),
        pytest.param("temp", float("nan"), id="temperature-nan"),
        pytest.param("topp", float("inf"), id="top-p-infinity"),
        pytest.param("topk", 1.5, id="top-k-non-int"),
        pytest.param("max_tokens", False, id="max-tokens-bool"),
        pytest.param("seed", "seven", id="seed-non-int"),
        pytest.param("presence_penalty", "high", id="presence-penalty-non-numeric"),
        pytest.param(
            "presence_penalty",
            float("-inf"),
            id="presence-penalty-negative-infinity",
        ),
        pytest.param("stop", 7, id="stop-non-sequence"),
        pytest.param("stop", ["END", 7], id="stop-non-string-member"),
        pytest.param("n", True, id="n-bool"),
        pytest.param("logprobs", "yes", id="logprobs-non-bool"),
        pytest.param("top_logprobs", False, id="top-logprobs-bool"),
        pytest.param("reasoning_effort", [], id="reasoning-effort-non-string"),
    ),
)
def test_scalar_boundaries_reject_invalid_shapes_in_both_modes(
    api_mode: str,
    field: str,
    invalid_value: object,
) -> None:
    kwargs: dict[str, object] = {
        "api_mode": api_mode,
        "model": "qwen3.8-max",
        "system_message": None,
        "messages_payload": [{"role": "user", "content": "Hello"}],
        "streaming": False,
    }
    kwargs[field] = invalid_value

    with pytest.raises(ChatBadRequestError) as exc_info:
        build_qwencloud_payload(**kwargs)  # type: ignore[arg-type]
    assert exc_info.value.provider == "qwencloud"


def test_chat_stop_sequence_is_deep_copied() -> None:
    stop = ["END", "DONE"]
    payload = build_qwencloud_payload(
        api_mode="chat_completions",
        model="qwen3.8-max",
        system_message=None,
        messages_payload=[{"role": "user", "content": "Hello"}],
        streaming=False,
        stop=stop,
    )

    assert payload["stop"] == ["END", "DONE"]
    assert payload["stop"] is not stop
    stop.append("MUTATED")
    assert payload["stop"] == ["END", "DONE"]


def test_function_tools_translate_by_mode() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look something up.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }
    ]
    original = deepcopy(tools)
    common = {
        "model": "qwen3.8-max",
        "system_message": None,
        "messages_payload": [{"role": "user", "content": "Hello"}],
        "streaming": False,
        "tools": tools,
        "tool_choice": "auto",
    }

    chat_payload = build_qwencloud_payload(api_mode="chat_completions", **common)
    assert chat_payload["tools"] == tools
    assert chat_payload["tool_choice"] == "auto"
    assert chat_payload["n"] == 1

    responses_payload = build_qwencloud_payload(api_mode="responses", **common)
    assert responses_payload["tools"] == [
        {
            "type": "function",
            "name": "lookup",
            "description": "Look something up.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        }
    ]
    assert responses_payload["tool_choice"] == "auto"
    assert tools == original
    assert chat_payload["tools"] is not tools
    assert (
        responses_payload["tools"][0]["parameters"]
        is not tools[0]["function"]["parameters"]
    )

    for mode in ("responses", "chat_completions"):
        for accepted_choice in (None, "auto", "none"):
            payload = build_qwencloud_payload(
                api_mode=mode, **{**common, "tool_choice": accepted_choice}
            )
            if accepted_choice is None:
                assert "tool_choice" not in payload
            else:
                assert payload["tool_choice"] == accepted_choice


@pytest.mark.parametrize(
    "api_mode",
    ("responses", "chat_completions"),
    ids=("responses", "chat-completions"),
)
def test_function_tools_reject_private_top_level_metadata_without_disclosure(
    api_mode: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    canary = "SECRET-QWENCLOUD-TOOL-CANARY"
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {"type": "object", "properties": {}},
            },
            "private_metadata": {"token": canary},
        }
    ]

    with pytest.raises(ChatBadRequestError) as exc_info:
        build_qwencloud_payload(
            api_mode=api_mode,  # type: ignore[arg-type]
            model="qwen3.8-max",
            system_message=None,
            messages_payload=[{"role": "user", "content": "Hello"}],
            streaming=False,
            tools=tools,
        )
    assert exc_info.value.provider == "qwencloud"
    assert canary not in str(exc_info.value)
    captured = capsys.readouterr()
    assert canary not in captured.out
    assert canary not in captured.err


@pytest.mark.parametrize(
    "api_mode",
    ("responses", "chat_completions"),
    ids=("responses", "chat-completions"),
)
@pytest.mark.parametrize(
    "rejected_tools",
    (
        [{"type": "web_search"}],
        [{"type": "function", "function": {"name": "", "parameters": {}}}],
        [{"type": "function", "function": {"name": "   ", "parameters": {}}}],
        [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {"type": "object"},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {"type": "object"},
                },
            },
        ],
        [
            {
                "type": "function",
                "function": {"name": "lookup", "parameters": []},
            }
        ],
        [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {"type": "array"},
                },
            }
        ],
        [
            {
                "type": "function",
                "function": {
                    "type": "web_search",
                    "name": "lookup",
                    "parameters": {"type": "object"},
                },
            }
        ],
    ),
    ids=(
        "builtin-web-search",
        "empty-function-name",
        "blank-function-name",
        "duplicate-function-name",
        "non-object-parameters",
        "array-parameters-schema",
        "nested-builtin-tool-type",
    ),
)
def test_invalid_or_builtin_tools_fail_before_network(
    api_mode: str,
    rejected_tools: list[dict[str, object]],
) -> None:
    with pytest.raises(ChatBadRequestError) as exc_info:
        build_qwencloud_payload(
            api_mode=api_mode,  # type: ignore[arg-type]
            model="qwen3.8-max",
            system_message=None,
            messages_payload=[{"role": "user", "content": "Hello"}],
            streaming=False,
            tools=rejected_tools,  # type: ignore[arg-type]
        )
    assert exc_info.value.provider == "qwencloud"


@pytest.mark.parametrize(
    "api_mode",
    ("responses", "chat_completions"),
    ids=("responses", "chat-completions"),
)
@pytest.mark.parametrize(
    "rejected_choice",
    ("required", "lookup", {"type": "function"}),
    ids=("required", "forced-name", "mapping-choice"),
)
def test_forced_function_tool_choices_fail_before_network(
    api_mode: str,
    rejected_choice: object,
) -> None:
    with pytest.raises(ChatBadRequestError):
        build_qwencloud_payload(
            api_mode=api_mode,  # type: ignore[arg-type]
            model="qwen3.8-max",
            system_message=None,
            messages_payload=[{"role": "user", "content": "Hello"}],
            streaming=False,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            tool_choice=rejected_choice,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "invalid_override",
    (
        {"messages_payload": None},
        {"tools": 7},
        {"response_format": 7},
    ),
    ids=("messages-none", "tools-int", "response-format-int"),
)
def test_invalid_public_build_shapes_raise_typed_error(
    invalid_override: dict[str, object],
) -> None:
    kwargs: dict[str, object] = {
        "api_mode": "chat_completions",
        "model": "qwen3.8-max",
        "system_message": None,
        "messages_payload": [{"role": "user", "content": "Hello"}],
        "streaming": False,
    }
    kwargs.update(invalid_override)

    with pytest.raises(ChatBadRequestError) as exc_info:
        build_qwencloud_payload(**kwargs)  # type: ignore[arg-type]
    assert exc_info.value.provider == "qwencloud"


def test_message_content_translation_is_role_safe_and_immutable() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello "},
                {"type": "text", "text": "world"},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is shown?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,AAAA",
                        "detail": "auto",
                    },
                },
            ],
        },
    ]
    original = deepcopy(messages)
    common = {
        "model": "qwen3.8-max",
        "system_message": None,
        "messages_payload": messages,
        "streaming": False,
    }

    chat = build_qwencloud_payload(api_mode="chat_completions", **common)
    assert chat["messages"] == [
        {"role": "user", "content": "Hello world"},
        original[1],
    ]
    responses = build_qwencloud_payload(api_mode="responses", **common)
    assert responses["input"] == [
        {"role": "user", "content": "Hello world"},
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "What is shown?"},
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,AAAA",
                },
            ],
        },
    ]
    assert messages == original
    assert chat["messages"][1]["content"] is not messages[1]["content"]

    empty_assistant_batch = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_empty",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_empty", "content": "ok"},
    ]
    empty_chat = build_qwencloud_payload(
        api_mode="chat_completions",
        model="qwen3.8-max",
        system_message=None,
        messages_payload=empty_assistant_batch,
        streaming=False,
    )
    assert empty_chat["messages"][0]["content"] == ""
    empty_responses = build_qwencloud_payload(
        api_mode="responses",
        model="qwen3.8-max",
        system_message=None,
        messages_payload=empty_assistant_batch,
        streaming=False,
    )
    assert empty_responses["input"] == [
        {
            "type": "function_call",
            "call_id": "call_empty",
            "name": "lookup",
            "arguments": "{}",
        },
        {
            "type": "function_call_output",
            "call_id": "call_empty",
            "output": "ok",
        },
    ]

    rejected_messages = (
        [{"role": "assistant", "content": [original[1]["content"][1]]}],
        [{"role": "user", "content": [{"type": "audio", "audio": "x"}]}],
        [{"role": "user", "content": [{"type": "video", "video": "x"}]}],
        [{"role": "user", "content": [{"type": "file", "file": "x"}]}],
        [{"role": "user", "content": [{"type": "unknown", "value": "x"}]}],
        [{"role": "critic", "content": "No"}],
        [{"role": 42, "content": "No"}],
        [{"role": "user", "content": 42}],
        [{"role": "user", "content": [{"type": "text", "text": 42}]}],
        [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}],
        [
            {"role": "user", "content": "Hello"},
            {"role": "system", "content": "Too late"},
        ],
    )
    for mode in ("responses", "chat_completions"):
        for rejected in rejected_messages:
            with pytest.raises(ChatBadRequestError) as exc_info:
                build_qwencloud_payload(
                    api_mode=mode,
                    model="qwen3.8-max",
                    system_message=None,
                    messages_payload=rejected,  # type: ignore[arg-type]
                    streaming=False,
                )
            assert exc_info.value.provider == "qwencloud"


def test_responses_assistant_text_is_id_free_easy_input_message() -> None:
    payload = build_qwencloud_payload(
        api_mode="responses",
        model="qwen3.8-max",
        system_message=None,
        messages_payload=[
            {"role": "user", "content": "Question"},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Prior answer"}],
            },
        ],
        streaming=False,
    )
    assistant_item = payload["input"][1]
    assert assistant_item == {
        "role": "assistant",
        "content": [{"type": "output_text", "text": "Prior answer"}],
    }
    assert set(assistant_item) == {"role", "content"}
    assert "id" not in assistant_item
    assert "status" not in assistant_item
    assert "type" not in assistant_item


def test_responses_pairs_out_of_order_results_by_call_id() -> None:
    messages = [
        {"role": "user", "content": "Compare both."},
        {
            "role": "assistant",
            "content": "I'll check.",
            "tool_calls": [
                {
                    "id": "call_A",
                    "type": "function",
                    "function": {
                        "name": "first_tool",
                        "arguments": '{"value": 1}',
                    },
                },
                {
                    "id": "call_B",
                    "type": "function",
                    "function": {
                        "name": "second_tool",
                        "arguments": '{"value": 2}',
                    },
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call_B", "content": "result B"},
        {"role": "tool", "tool_call_id": "call_A", "content": "result A"},
    ]
    original = deepcopy(messages)

    payload = build_qwencloud_payload(
        api_mode="responses",
        model="qwen3.8-max",
        system_message=None,
        messages_payload=messages,
        streaming=False,
    )
    assert payload["input"] == [
        {"role": "user", "content": "Compare both."},
        {
            "role": "assistant",
            "content": [{"type": "output_text", "text": "I'll check."}],
        },
        {
            "type": "function_call",
            "call_id": "call_A",
            "name": "first_tool",
            "arguments": '{"value": 1}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_A",
            "output": "result A",
        },
        {
            "type": "function_call",
            "call_id": "call_B",
            "name": "second_tool",
            "arguments": '{"value": 2}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_B",
            "output": "result B",
        },
    ]
    assert messages == original

    chat = build_qwencloud_payload(
        api_mode="chat_completions",
        model="qwen3.8-max",
        system_message=None,
        messages_payload=messages,
        streaming=False,
    )
    assert chat["messages"] == original


def test_tool_call_arguments_reject_non_finite_json_constants() -> None:
    for arguments in ('{"x":NaN}', '{"x":Infinity}', '{"x":-Infinity}'):
        history = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_strict_json",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": arguments},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_strict_json",
                "content": "ok",
            },
        ]
        with pytest.raises(ChatBadRequestError) as exc_info:
            build_qwencloud_payload(
                api_mode="responses",
                model="qwen3.8-max",
                system_message=None,
                messages_payload=history,
                streaming=False,
            )
        assert exc_info.value.provider == "qwencloud"


def test_responses_rejects_unpairable_tool_batches_before_network() -> None:
    def call(
        call_id: object = "call_A", name: object = "lookup", arguments: object = "{}"
    ) -> dict:
        return {
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        }

    def assistant(*calls: dict, content: object = "") -> dict:
        return {"role": "assistant", "content": content, "tool_calls": list(calls)}

    def result(call_id: object = "call_A", content: object = "ok") -> dict:
        return {"role": "tool", "tool_call_id": call_id, "content": content}

    rejected_histories = (
        [assistant(call())],
        [assistant(call()), result(), result()],
        [result()],
        [assistant(call()), result("call_extra")],
        [assistant(call(), call()), result()],
        [assistant(call("")), result("")],
        [assistant(call(name="")), result()],
        [assistant(call(arguments="{")), result()],
        [assistant(call(arguments=42)), result()],
        [assistant(call(arguments="[]")), result()],
        [assistant(call()), result(content={"not": "a string"})],
        [assistant(call()), {"role": "tool", "content": "missing id"}],
        [
            assistant(call("call_A")),
            result("call_A"),
            assistant(call("call_A")),
            result("call_A"),
        ],
        [
            assistant(call("call_A")),
            assistant(call("call_B")),
            result("call_A"),
            result("call_B"),
        ],
    )
    for history in rejected_histories:
        with pytest.raises(ChatBadRequestError) as exc_info:
            build_qwencloud_payload(
                api_mode="responses",
                model="qwen3.8-max",
                system_message=None,
                messages_payload=history,
                streaming=False,
            )
        assert exc_info.value.provider == "qwencloud"


def test_nonstream_normalizes_text_tools_finish_and_usage() -> None:
    responses_mixed = qwencloud.normalize_qwencloud_response(
        {
            "id": "resp_123",
            "object": "response",
            "status": "completed",
            "output": [
                {
                    "id": "msg_123",
                    "type": "message",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "I will check both."}],
                },
                {
                    "id": "fc_transport_A",
                    "type": "function_call",
                    "status": "completed",
                    "call_id": "call_A",
                    "name": "first_tool",
                    "arguments": '{"value":1}',
                },
                {
                    "id": "fc_transport_B",
                    "type": "function_call",
                    "status": "completed",
                    "call_id": "call_B",
                    "name": "second_tool",
                    "arguments": '{"value":2}',
                },
            ],
            "usage": {
                "input_tokens": 11,
                "output_tokens": 7,
                "total_tokens": 18,
                "input_tokens_details": {"cached_tokens": 3},
            },
        },
        api_mode="responses",
    )
    assert responses_mixed == {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "I will check both.",
                    "tool_calls": [
                        {
                            "id": "call_A",
                            "type": "function",
                            "function": {
                                "name": "first_tool",
                                "arguments": '{"value":1}',
                            },
                        },
                        {
                            "id": "call_B",
                            "type": "function",
                            "function": {
                                "name": "second_tool",
                                "arguments": '{"value":2}',
                            },
                        },
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "input_tokens": 11,
            "output_tokens": 7,
            "total_tokens": 18,
            "input_tokens_details": {"cached_tokens": 3},
        },
    }

    responses_text = qwencloud.normalize_qwencloud_response(
        {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": "Hello"},
                        {"type": "output_text", "text": " world"},
                    ],
                }
            ],
        },
        api_mode="responses",
    )
    assert responses_text["choices"][0] == {
        "message": {"role": "assistant", "content": "Hello world"},
        "finish_reason": "stop",
    }
    assert responses_text["usage"] == {}

    responses_tool_only = qwencloud.normalize_qwencloud_response(
        {
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "status": "completed",
                    "call_id": "call_only",
                    "name": "lookup",
                    "arguments": "{}",
                }
            ],
        },
        api_mode="responses",
    )
    assert responses_tool_only["choices"][0]["message"]["content"] is None
    assert responses_tool_only["choices"][0]["finish_reason"] == "tool_calls"

    responses_partial = qwencloud.normalize_qwencloud_response(
        {
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [
                {
                    "type": "message",
                    "status": "incomplete",
                    "content": [{"type": "output_text", "text": "Partial answer"}],
                }
            ],
        },
        api_mode="responses",
    )
    assert responses_partial["choices"][0] == {
        "message": {"role": "assistant", "content": "Partial answer"},
        "finish_reason": "length",
    }

    chat_mixed = qwencloud.normalize_qwencloud_response(
        {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Checking now.",
                        "tool_calls": [
                            {
                                "id": "call_chat_A",
                                "type": "function",
                                "function": {
                                    "name": "first_tool",
                                    "arguments": "{}",
                                },
                            },
                            {
                                "id": "call_chat_B",
                                "type": "function",
                                "function": {
                                    "name": "second_tool",
                                    "arguments": '{"x":2}',
                                },
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {
                "prompt_tokens": 9,
                "completion_tokens": 5,
                "total_tokens": 14,
            },
        },
        api_mode="chat_completions",
    )
    assert chat_mixed["choices"][0]["message"] == {
        "role": "assistant",
        "content": "Checking now.",
        "tool_calls": [
            {
                "id": "call_chat_A",
                "type": "function",
                "function": {"name": "first_tool", "arguments": "{}"},
            },
            {
                "id": "call_chat_B",
                "type": "function",
                "function": {"name": "second_tool", "arguments": '{"x":2}'},
            },
        ],
    }
    assert chat_mixed["choices"][0]["finish_reason"] == "tool_calls"
    assert chat_mixed["usage"] == {
        "prompt_tokens": 9,
        "completion_tokens": 5,
        "total_tokens": 14,
    }


def test_nonstream_rejects_empty_success_and_malformed_shapes() -> None:
    malformed_cases = (
        ("responses", {"status": "completed", "output": []}),
        ("responses", {"status": "completed", "output": {}}),
        ("responses", {"status": "failed", "output": []}),
        ("responses", {"status": "cancelled", "output": []}),
        (
            "responses",
            {
                "status": "incomplete",
                "incomplete_details": {"reason": "content_filter"},
                "output": [
                    {
                        "type": "message",
                        "status": "incomplete",
                        "content": [{"type": "output_text", "text": "private"}],
                    }
                ],
            },
        ),
        (
            "responses",
            {
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "output": [
                    {
                        "type": "function_call",
                        "status": "in_progress",
                        "call_id": "call_partial",
                        "name": "lookup",
                        "arguments": "{",
                    }
                ],
            },
        ),
        ("chat_completions", {"choices": []}),
        ("chat_completions", {"choices": {}}),
        ("chat_completions", {"choices": [{"message": "bad"}]}),
        (
            "chat_completions",
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": None},
                        "finish_reason": "stop",
                    }
                ]
            },
        ),
        (
            "chat_completions",
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_bad",
                                    "type": "function",
                                    "function": {
                                        "name": "lookup",
                                        "arguments": 7,
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
        ),
    )
    for api_mode, payload in malformed_cases:
        with pytest.raises(ChatProviderError) as exc_info:
            qwencloud.normalize_qwencloud_response(
                payload,  # type: ignore[arg-type]
                api_mode=api_mode,  # type: ignore[arg-type]
            )
        assert exc_info.value.provider == "qwencloud"
        assert "private" not in str(exc_info.value)


@pytest.mark.parametrize(
    ("content", "tool_calls", "raw_finish_reason", "expected_finish_reason"),
    (
        pytest.param("answer", None, "stop", "stop", id="text-stop"),
        pytest.param("partial", None, "length", "length", id="text-length"),
        pytest.param(
            "I will check.",
            [
                {
                    "id": "call_chat",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
            "tool_calls",
            "tool_calls",
            id="mixed-tool-calls",
        ),
    ),
)
def test_chat_finish_reason_accepts_only_consistent_terminal_states(
    content: str,
    tool_calls: list[dict[str, Any]] | None,
    raw_finish_reason: str,
    expected_finish_reason: str,
) -> None:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls

    normalized = qwencloud.normalize_qwencloud_response(
        {"choices": [{"message": message, "finish_reason": raw_finish_reason}]},
        api_mode="chat_completions",
    )

    assert normalized["choices"][0]["finish_reason"] == expected_finish_reason


@pytest.mark.parametrize(
    ("message", "choice_fields"),
    (
        pytest.param(
            {"role": "assistant", "content": "private"},
            {"finish_reason": "content_filter"},
            id="content-filter",
        ),
        pytest.param(
            {"role": "assistant", "content": "private"},
            {"finish_reason": "future-value"},
            id="unknown",
        ),
        pytest.param(
            {"role": "assistant", "content": "private"},
            {},
            id="missing",
        ),
        pytest.param(
            {"role": "assistant", "content": "private"},
            {"finish_reason": ""},
            id="empty",
        ),
        pytest.param(
            {"role": "assistant", "content": "private"},
            {"finish_reason": "   "},
            id="blank",
        ),
        pytest.param(
            {"role": "assistant", "content": "private"},
            {"finish_reason": "tool_calls"},
            id="tool-reason-without-calls",
        ),
        pytest.param(
            {
                "role": "assistant",
                "content": "private",
                "tool_calls": [
                    {
                        "id": "call_chat",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            },
            {"finish_reason": "stop"},
            id="calls-with-stop",
        ),
        pytest.param(
            {
                "role": "assistant",
                "content": "private",
                "tool_calls": [
                    {
                        "id": "call_chat",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            },
            {"finish_reason": "length"},
            id="calls-with-length",
        ),
    ),
)
def test_chat_finish_reason_rejects_unknown_or_contradictory_states(
    message: dict[str, Any],
    choice_fields: dict[str, Any],
) -> None:
    with pytest.raises(ChatProviderError) as exc_info:
        qwencloud.normalize_qwencloud_response(
            {"choices": [{"message": message, **choice_fields}]},
            api_mode="chat_completions",
        )

    assert exc_info.value.provider == "qwencloud"
    assert "private" not in str(exc_info.value)


def test_responses_requires_call_id_not_transport_id() -> None:
    for call_id_fields in ({}, {"call_id": "  "}):
        with pytest.raises(ChatProviderError) as exc_info:
            qwencloud.normalize_qwencloud_response(
                {
                    "status": "completed",
                    "output": [
                        {
                            "id": "fc_transport_only",
                            "type": "function_call",
                            "status": "completed",
                            "name": "lookup",
                            "arguments": "{}",
                            **call_id_fields,
                        }
                    ],
                },
                api_mode="responses",
            )
        assert exc_info.value.provider == "qwencloud"
        assert "incomplete function call" in str(exc_info.value).lower()


@pytest.mark.parametrize(
    ("api_mode", "suffix", "response_payload"),
    (
        (
            "responses",
            "/responses",
            {
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "ok"}],
                    }
                ],
            },
        ),
        (
            "chat_completions",
            "/chat/completions",
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ]
            },
        ),
    ),
)
def test_nonstream_transport_uses_exact_mode_url_headers_and_timeout(
    monkeypatch: pytest.MonkeyPatch,
    api_mode: str,
    suffix: str,
    response_payload: dict[str, Any],
) -> None:
    response = _TransportResponse(response_payload)
    session = _RecordingSession(response)
    monkeypatch.setattr(
        qwencloud, "create_default_session", lambda: session, raising=False
    )
    monkeypatch.setattr(
        qwencloud,
        "get_runtime_config_snapshot",
        lambda: SimpleNamespace(
            values={
                "api_settings": {
                    "qwencloud": {"timeout": 37, "retries": 0, "retry_delay": 0}
                }
            }
        ),
        raising=False,
    )

    chat_with_qwencloud(
        input_data=[{"role": "user", "content": "hello"}],
        model="qwen3.8-max",
        api_key="qwen-secret",
        streaming=False,
        api_base_url="https://qwen.example/compatible-mode/v1/responses",
        api_mode=api_mode,
    )

    assert len(session.posts) == 1
    request = session.posts[0]
    assert request["url"] == f"https://qwen.example/compatible-mode/v1{suffix}"
    assert request["headers"] == {
        "Authorization": "Bearer qwen-secret",
        "Content-Type": "application/json",
    }
    assert request["timeout"] == 37.0
    assert request["json"]["model"] == "qwen3.8-max"
    assert request["json"]["stream"] is False
    assert session.closed is True
    assert response.closed is True


@pytest.mark.parametrize("path", ("success", "provider-error"))
def test_nonstream_cleanup_failures_never_mask_result_or_provider_error(
    monkeypatch: pytest.MonkeyPatch,
    path: str,
) -> None:
    payload = (
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ]
        }
        if path == "success"
        else {"choices": []}
    )
    response = _TransportResponse(
        payload,
        close_error=RuntimeError("RAW-RESPONSE-CLOSE-CANARY"),
    )
    session = _RecordingSession(
        response,
        close_error=RuntimeError("RAW-SESSION-CLOSE-CANARY"),
    )
    monkeypatch.setattr(qwencloud, "create_default_session", lambda: session)
    monkeypatch.setattr(
        qwencloud,
        "get_runtime_config_snapshot",
        lambda: SimpleNamespace(
            values={
                "api_settings": {
                    "qwencloud": {"timeout": 3, "retries": 0, "retry_delay": 0}
                }
            }
        ),
    )

    with _captured_qwencloud_logs() as logs:
        if path == "success":
            result = chat_with_qwencloud(
                input_data=[{"role": "user", "content": "hello"}],
                model="qwen3.8-max",
                api_key="key",
                streaming=False,
                api_base_url="https://qwen.example/v1",
                api_mode="chat_completions",
            )
            assert result["choices"][0]["message"]["content"] == "ok"
            disclosure = "".join(logs)
        else:
            with pytest.raises(ChatProviderError) as exc_info:
                chat_with_qwencloud(
                    input_data=[{"role": "user", "content": "hello"}],
                    model="qwen3.8-max",
                    api_key="key",
                    streaming=False,
                    api_base_url="https://qwen.example/v1",
                    api_mode="chat_completions",
                )
            assert exc_info.value.provider == "qwencloud"
            assert exc_info.value.__cause__ is None
            assert exc_info.value.__context__ is None
            disclosure = str(exc_info.value) + "".join(logs)

    assert "RAW-RESPONSE-CLOSE-CANARY" not in disclosure
    assert "RAW-SESSION-CLOSE-CANARY" not in disclosure
    assert response.close_calls == 1
    assert session.close_calls == 1


def test_streaming_transport_transfers_response_and_session_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _TransportResponse(
        {},
        chunks=[
            b'data: {"choices":[{"index":0,"delta":{"content":"owned"},'
            b'"finish_reason":"stop"}]}\n\n',
            b"data: [DONE]\n\n",
        ],
    )
    session = _RecordingSession(response)
    monkeypatch.setattr(
        qwencloud,
        "create_default_session",
        lambda: session,
    )
    monkeypatch.setattr(
        qwencloud,
        "get_runtime_config_snapshot",
        lambda: SimpleNamespace(
            values={
                "api_settings": {
                    "qwencloud": {"timeout": 9, "retries": 0, "retry_delay": 0}
                }
            }
        ),
    )

    stream = chat_with_qwencloud(
        input_data=[{"role": "user", "content": "hello"}],
        model="qwen3.8-max",
        api_key="qwen-secret",
        streaming=True,
        api_base_url="https://qwen.example/compatible-mode/v1",
        api_mode="chat_completions",
    )

    assert not isinstance(stream, dict)
    assert session.closed is False
    assert response.closed is False
    assert next(stream)["choices"][0]["delta"]["content"] == "owned"
    assert list(stream) == []
    assert session.closed is True
    assert response.close_calls == 1


def test_direct_adapter_loads_only_qwencloud_config_when_arguments_are_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _TransportResponse(
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ]
        }
    )
    session = _RecordingSession(response)
    monkeypatch.setattr(
        qwencloud, "create_default_session", lambda: session, raising=False
    )
    monkeypatch.setenv("DASHSCOPE_API_KEY", "qwen-env-lower-priority")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-env-canary")
    monkeypatch.setattr(
        qwencloud,
        "get_runtime_config_snapshot",
        lambda: SimpleNamespace(
            values={
                "api_settings": {
                    "qwencloud": {
                        "api_mode": "chat_completions",
                        "api_base_url": "https://qwen-only.example/compatible-mode/v1",
                        "api_key": "qwen-modern-key",
                        "model": "qwen-config-model",
                        "timeout": 41,
                        "retries": 0,
                        "retry_delay": 0,
                    },
                    "openai": {
                        "api_base_url": "https://openai-canary.example/v1",
                        "api_key": "openai-config-canary",
                        "model": "openai-model-canary",
                    },
                    "deepseek": {
                        "api_base_url": "https://deepseek-canary.example/v1",
                        "api_key": "deepseek-config-canary",
                    },
                }
            }
        ),
        raising=False,
    )

    chat_with_qwencloud(
        input_data=[{"role": "user", "content": "hello"}],
        model=None,
        api_key=None,
        streaming=False,
        api_base_url=None,
        api_mode=None,
    )

    assert len(session.posts) == 1
    request = session.posts[0]
    assert request["url"] == (
        "https://qwen-only.example/compatible-mode/v1/chat/completions"
    )
    assert request["headers"]["Authorization"] == "Bearer qwen-modern-key"
    assert request["json"]["model"] == "qwen-config-model"
    assert request["timeout"] == 41.0
    serialized = repr(request)
    for canary in (
        "openai-env-canary",
        "openai-config-canary",
        "openai-model-canary",
        "openai-canary.example",
        "deepseek-config-canary",
        "deepseek-canary.example",
    ):
        assert canary not in serialized


def test_direct_adapter_loads_alias_only_qwencloud_config_without_mutation_or_leakage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _TransportResponse(
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ]
        }
    )
    session = _RecordingSession(response)
    monkeypatch.setattr(
        qwencloud,
        "create_default_session",
        lambda: session,
    )
    monkeypatch.setenv("QWEN_ALIAS_KEY", "alias-key-canary")
    source = {
        "api_settings": {
            "QwenCloud": {
                "api_mode": "chat_completions",
                "api_base_url": "https://alias.example/compatible-mode/v1",
                "api_key_env_var": "QWEN_ALIAS_KEY",
                "model": "alias-model",
                "timeout": 19,
                "retries": 0,
                "retry_delay": 0.25,
            }
        }
    }
    original = deepcopy(source)
    retry_configuration: list[tuple[int, float]] = []
    real_build_retry_policy = qwencloud._build_retry_policy

    def record_retry_policy(*, retries: int, retry_delay: float):
        retry_configuration.append((retries, retry_delay))
        return real_build_retry_policy(retries=retries, retry_delay=retry_delay)

    monkeypatch.setattr(qwencloud, "_build_retry_policy", record_retry_policy)
    monkeypatch.setattr(
        qwencloud,
        "get_runtime_config_snapshot",
        lambda: SimpleNamespace(values=source),
    )

    with _captured_qwencloud_logs() as logs:
        chat_with_qwencloud(
            input_data=[{"role": "user", "content": "hello"}],
            model=None,
            api_key=None,
            streaming=False,
            api_base_url=None,
            api_mode=None,
        )

    assert source == original
    assert retry_configuration == [(0, 0.25)]
    assert len(session.posts) == 1
    request = session.posts[0]
    assert request["url"] == (
        "https://alias.example/compatible-mode/v1/chat/completions"
    )
    assert request["headers"]["Authorization"] == "Bearer alias-key-canary"
    assert request["json"]["model"] == "alias-model"
    assert request["timeout"] == 19.0
    assert "alias-key-canary" not in "".join(logs)


@pytest.mark.parametrize("canonical_first", [False, True])
def test_direct_adapter_canonical_qwencloud_config_overrides_alias_in_any_order(
    monkeypatch: pytest.MonkeyPatch,
    canonical_first: bool,
) -> None:
    response = _TransportResponse(
        {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "ok"}],
                }
            ],
        }
    )
    session = _RecordingSession(response)
    monkeypatch.setattr(
        qwencloud,
        "create_default_session",
        lambda: session,
    )
    monkeypatch.setenv("QWEN_ALIAS_KEY", "alias-key-canary")
    monkeypatch.setenv("QWEN_CANONICAL_KEY", "canonical-key-canary")
    alias = {
        "api_mode": "chat_completions",
        "api_base_url": "https://alias.example/compatible-mode/v1",
        "api_key_env_var": "QWEN_ALIAS_KEY",
        "model": "alias-model",
        "timeout": 17,
        "retries": 4,
        "retry_delay": 0.75,
    }
    canonical = {
        "api_mode": "responses",
        "api_base_url": "https://canonical.example/compatible-mode/v1",
        "api_key_env_var": "QWEN_CANONICAL_KEY",
        "model": "canonical-model",
        "timeout": 23,
        "retries": 0,
        "retry_delay": 0.5,
    }
    entries = (
        [("qwencloud", canonical), ("QwenCloud", alias)]
        if canonical_first
        else [("QwenCloud", alias), ("qwencloud", canonical)]
    )
    source = {"api_settings": dict(entries)}
    original = deepcopy(source)
    retry_configuration: list[tuple[int, float]] = []
    real_build_retry_policy = qwencloud._build_retry_policy

    def record_retry_policy(*, retries: int, retry_delay: float):
        retry_configuration.append((retries, retry_delay))
        return real_build_retry_policy(retries=retries, retry_delay=retry_delay)

    monkeypatch.setattr(qwencloud, "_build_retry_policy", record_retry_policy)
    monkeypatch.setattr(
        qwencloud,
        "get_runtime_config_snapshot",
        lambda: SimpleNamespace(values=source),
    )

    with _captured_qwencloud_logs() as logs:
        chat_with_qwencloud(
            input_data=[{"role": "user", "content": "hello"}],
            model=None,
            api_key=None,
            streaming=False,
            api_base_url=None,
            api_mode=None,
        )

    assert source == original
    assert retry_configuration == [(0, 0.5)]
    assert len(session.posts) == 1
    request = session.posts[0]
    assert request["url"] == ("https://canonical.example/compatible-mode/v1/responses")
    assert request["headers"]["Authorization"] == "Bearer canonical-key-canary"
    assert request["json"]["model"] == "canonical-model"
    assert request["timeout"] == 23.0
    disclosure = repr(request) + "".join(logs)
    assert "alias-key-canary" not in disclosure
    assert "canonical-key-canary" not in "".join(logs)


@pytest.mark.parametrize("canonical_first", [False, True])
def test_direct_adapter_rejects_malformed_canonical_table_without_alias_leakage(
    monkeypatch: pytest.MonkeyPatch,
    canonical_first: bool,
) -> None:
    alias = {
        "api_key": "ALIAS-SECRET-CANARY",
        "api_mode": "responses",
        "model": "ALIAS-MODEL-CANARY",
    }
    entries = (
        [("qwencloud", []), ("QwenCloud", alias)]
        if canonical_first
        else [("QwenCloud", alias), ("qwencloud", [])]
    )
    source = {"api_settings": dict(entries)}
    original = deepcopy(source)
    monkeypatch.setattr(
        qwencloud,
        "get_runtime_config_snapshot",
        lambda: SimpleNamespace(values=source),
    )
    monkeypatch.setattr(
        qwencloud,
        "create_default_session",
        lambda: pytest.fail("malformed Qwen config must fail before network"),
    )

    with _captured_qwencloud_logs() as logs:
        with pytest.raises(ChatConfigurationError):
            chat_with_qwencloud(
                input_data=[{"role": "user", "content": "hello"}],
                model=None,
                api_key=None,
                streaming=False,
                api_base_url=None,
                api_mode=None,
            )

    assert source == original
    assert "ALIAS-SECRET-CANARY" not in "".join(logs)
    assert "ALIAS-MODEL-CANARY" not in "".join(logs)


def test_direct_adapter_rejects_alias_only_malformed_table_before_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = {"api_settings": {"QwenCloud": ["SECRET-CANARY"]}}
    original = deepcopy(source)
    monkeypatch.setattr(
        qwencloud,
        "get_runtime_config_snapshot",
        lambda: SimpleNamespace(values=source),
    )
    monkeypatch.setattr(
        qwencloud,
        "create_default_session",
        lambda: pytest.fail("malformed Qwen config must fail before network"),
    )

    with _captured_qwencloud_logs() as logs:
        with pytest.raises(ChatConfigurationError):
            chat_with_qwencloud(
                input_data=[{"role": "user", "content": "hello"}],
                model="explicit-model",
                api_key="explicit-key",
                streaming=False,
                api_base_url="https://explicit.example.test/v1",
                api_mode="responses",
            )

    assert source == original
    assert "SECRET-CANARY" not in "".join(logs)


@pytest.mark.parametrize("canonical_first", [False, True])
def test_direct_adapter_ignores_malformed_alias_when_canonical_is_valid(
    monkeypatch: pytest.MonkeyPatch,
    canonical_first: bool,
) -> None:
    response = _TransportResponse(
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ]
        }
    )
    session = _RecordingSession(response)
    monkeypatch.setattr(
        qwencloud,
        "create_default_session",
        lambda: session,
    )
    canonical = {
        "api_key": "canonical-key",
        "api_mode": "chat_completions",
        "api_base_url": "https://canonical.example.test/v1",
        "model": "canonical-model",
        "timeout": 17,
        "retries": 0,
    }
    entries = (
        [("qwencloud", canonical), ("QwenCloud", ["SECRET-CANARY"])]
        if canonical_first
        else [("QwenCloud", ["SECRET-CANARY"]), ("qwencloud", canonical)]
    )
    source = {"api_settings": dict(entries)}
    original = deepcopy(source)
    monkeypatch.setattr(
        qwencloud,
        "get_runtime_config_snapshot",
        lambda: SimpleNamespace(values=source),
    )

    with _captured_qwencloud_logs() as logs:
        chat_with_qwencloud(
            input_data=[{"role": "user", "content": "hello"}],
            model=None,
            api_key=None,
            streaming=False,
            api_base_url=None,
            api_mode=None,
        )

    assert source == original
    assert session.posts[0]["url"] == (
        "https://canonical.example.test/v1/chat/completions"
    )
    assert session.posts[0]["headers"]["Authorization"] == "Bearer canonical-key"
    assert session.posts[0]["json"]["model"] == "canonical-model"
    assert session.posts[0]["timeout"] == 17.0
    assert "SECRET-CANARY" not in "".join(logs)


@pytest.mark.parametrize(
    "api_settings",
    (
        {"qwencloud": []},
        {
            "qwencloud": {
                "api_key": "key",
                "api_base_url": 17,
                "retries": 0,
            }
        },
    ),
    ids=("non-mapping-provider-table", "non-string-configured-base"),
)
def test_direct_adapter_rejects_malformed_provider_config_before_network(
    monkeypatch: pytest.MonkeyPatch,
    api_settings: dict[str, Any],
) -> None:
    def unexpected_session() -> Never:
        raise AssertionError("network must not be initialized")

    monkeypatch.setattr(
        qwencloud,
        "create_default_session",
        unexpected_session,
    )
    monkeypatch.setattr(
        qwencloud,
        "get_runtime_config_snapshot",
        lambda: SimpleNamespace(values={"api_settings": api_settings}),
    )

    with pytest.raises(ChatConfigurationError) as exc_info:
        chat_with_qwencloud(
            input_data=[{"role": "user", "content": "hello"}],
            model="qwen3.8-max",
            api_key="key",
            streaming=False,
            api_base_url=None,
            api_mode="chat_completions",
        )
    assert exc_info.value.provider == "qwencloud"


def test_direct_adapter_uses_stripped_lower_key_and_explicit_base_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _TransportResponse(
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ]
        }
    )
    session = _RecordingSession(response)
    monkeypatch.setattr(
        qwencloud,
        "create_default_session",
        lambda: session,
    )
    monkeypatch.setenv("QWEN_KEY", "  env-fallback-key  ")
    monkeypatch.setattr(
        qwencloud,
        "get_runtime_config_snapshot",
        lambda: SimpleNamespace(
            values={
                "api_settings": {
                    "qwencloud": {
                        "api_key": " YOUR_KEY ",
                        "api_key_env_var": "QWEN_KEY",
                        "api_base_url": 17,
                        "timeout": 3,
                        "retries": 0,
                        "retry_delay": 0,
                    }
                }
            }
        ),
    )

    chat_with_qwencloud(
        input_data=[{"role": "user", "content": "hello"}],
        model="qwen3.8-max",
        api_key=" <API_KEY_HERE> ",
        streaming=False,
        api_base_url="https://explicit.example/v1",
        api_mode="chat_completions",
    )

    assert len(session.posts) == 1
    assert session.posts[0]["url"] == ("https://explicit.example/v1/chat/completions")
    assert session.posts[0]["headers"]["Authorization"] == ("Bearer env-fallback-key")


def test_direct_adapter_rejects_unresolved_placeholders_before_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_session() -> Never:
        raise AssertionError("network must not be initialized")

    monkeypatch.setattr(
        qwencloud,
        "create_default_session",
        unexpected_session,
    )
    monkeypatch.setenv("DASHSCOPE_API_KEY", " your_key ")
    monkeypatch.setattr(
        qwencloud,
        "get_runtime_config_snapshot",
        lambda: SimpleNamespace(
            values={
                "api_settings": {
                    "qwencloud": {
                        "api_key": " YOUR_KEY ",
                        "timeout": 3,
                        "retries": 0,
                        "retry_delay": 0,
                    }
                }
            }
        ),
    )

    with pytest.raises(ChatConfigurationError) as exc_info:
        chat_with_qwencloud(
            input_data=[{"role": "user", "content": "hello"}],
            model="qwen3.8-max",
            api_key=" <API_KEY_HERE> ",
            streaming=False,
            api_base_url="https://explicit.example/v1",
            api_mode="chat_completions",
        )
    assert exc_info.value.provider == "qwencloud"


@pytest.mark.allow_network
def test_retry_policy_counts_status_connection_and_timeout_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retry_errors: list[requests.exceptions.RequestException] = []
    real_advance_retry_policy = qwencloud._advance_retry_policy

    def record_retry_error(
        retry_policy: Retry,
        *,
        api_url: str,
        response: requests.Response | None = None,
        error: requests.exceptions.RequestException | None = None,
    ) -> tuple[Retry, float]:
        if error is not None:
            retry_errors.append(error)
        return real_advance_retry_policy(
            retry_policy,
            api_url=api_url,
            response=response,
            error=error,
        )

    monkeypatch.setattr(qwencloud, "_advance_retry_policy", record_retry_error)
    post_urls, returned_responses, closed_response_ids = (
        _track_real_transport_resources(monkeypatch, connect_timeout=1.0)
    )
    _configure_qwencloud_transport(monkeypatch, retries=2)

    post_start = len(post_urls)
    response_start = len(returned_responses)
    with _scripted_qwen_server(["stall", "stall", "success"]) as (
        api_base_url,
        timeout_server,
    ):
        result = _call_scripted_qwencloud(api_base_url)
    timeout_responses = returned_responses[response_start:]
    assert result["choices"][0]["message"]["content"] == "ok"
    assert len(timeout_server.attempts) == 3
    assert len(post_urls[post_start:]) == 3
    assert len(timeout_responses) == 3
    assert all(id(response) in closed_response_ids for response in timeout_responses)
    assert len(retry_errors) == 2
    assert all(
        any(isinstance(arg, ReadTimeoutError) for arg in error.args)
        for error in retry_errors
    )

    post_start = len(post_urls)
    response_start = len(returned_responses)
    with _scripted_qwen_server([(503, {}), (503, {}), "success"]) as (
        api_base_url,
        status_server,
    ):
        result = _call_scripted_qwencloud(api_base_url)
    status_responses = returned_responses[response_start:]
    assert result["choices"][0]["message"]["content"] == "ok"
    assert len(status_server.attempts) == 3
    assert len(post_urls[post_start:]) == 3
    assert len(status_responses) == 3
    assert all(id(response) in closed_response_ids for response in status_responses)

    guarded_connect = socket.socket.connect
    connection_attempts = 0
    post_start = len(post_urls)
    response_start = len(returned_responses)
    with _scripted_qwen_server(["success"]) as (api_base_url, connection_server):
        target_port = connection_server.server_address[1]

        def flaky_connect(
            client_socket: socket.socket, address: tuple[str, int]
        ) -> None:
            nonlocal connection_attempts
            if address[1] == target_port:
                connection_attempts += 1
                if connection_attempts < 3:
                    raise ConnectionRefusedError("scripted connection failure")
            guarded_connect(client_socket, address)

        monkeypatch.setattr(socket.socket, "connect", flaky_connect)
        result = _call_scripted_qwencloud(api_base_url)
    connection_responses = returned_responses[response_start:]
    assert result["choices"][0]["message"]["content"] == "ok"
    assert connection_attempts == 3
    assert len(connection_server.attempts) == 1
    assert len(post_urls[post_start:]) == 3
    assert len(connection_responses) == 1
    assert id(connection_responses[0]) in closed_response_ids

    _configure_qwencloud_transport(monkeypatch, retries=-9)
    post_start = len(post_urls)
    with _scripted_qwen_server([(503, {}), "success"]) as (
        api_base_url,
        negative_server,
    ):
        with pytest.raises(ChatProviderError):
            _call_scripted_qwencloud(api_base_url)
    assert len(negative_server.attempts) == 1
    assert len(post_urls[post_start:]) == 1


@pytest.mark.allow_network
def test_sensitive_request_forces_zero_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_qwencloud_transport(monkeypatch, retries=7)
    with (
        _scripted_qwen_server([(503, {}), "success"]) as (
            api_base_url,
            server,
        ),
        sensitive_llm_request(),
        pytest.raises(ChatProviderError),
    ):
        _call_scripted_qwencloud(api_base_url)
    assert len(server.attempts) == 1


@pytest.mark.allow_network
def test_retry_policy_honors_retry_after_and_exponential_delay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("time.sleep", lambda delay: sleeps.append(delay))

    _configure_qwencloud_transport(monkeypatch, retries=1)
    with _scripted_qwen_server([(429, {"Retry-After": "4"}), "success"]) as (
        api_base_url,
        integer_server,
    ):
        _call_scripted_qwencloud(api_base_url)
    assert len(integer_server.attempts) == 2
    assert sleeps == [4]

    sleeps.clear()
    date_header = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=5))
    with _scripted_qwen_server([(503, {"Retry-After": date_header}), "success"]) as (
        api_base_url,
        date_server,
    ):
        _call_scripted_qwencloud(api_base_url)
    assert len(date_server.attempts) == 2
    assert len(sleeps) == 1
    assert 3 <= sleeps[0] <= 6

    sleeps.clear()
    _configure_qwencloud_transport(monkeypatch, retries=2, retry_delay=0.25)
    with _scripted_qwen_server([(503, {}), (503, {}), "success"]) as (
        api_base_url,
        backoff_server,
    ):
        _call_scripted_qwencloud(api_base_url)
    assert len(backoff_server.attempts) == 3
    assert sleeps == [pytest.approx(0.5)]


@pytest.mark.allow_network
@pytest.mark.parametrize("status_code", (429, 503))
def test_invalid_retry_after_uses_exponential_fallback_without_disclosure(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("time.sleep", lambda delay: sleeps.append(delay))
    post_urls, returned_responses, closed_response_ids = (
        _track_real_transport_resources(monkeypatch)
    )
    closed_session_ids = _track_real_session_closes(monkeypatch)
    _configure_qwencloud_transport(monkeypatch, retries=2, retry_delay=0.25)

    retry_after_canary = "RAW-RETRY-AFTER-CANARY"
    with (
        _captured_qwencloud_logs() as logs,
        _scripted_qwen_server(
            [
                (503, {}),
                (status_code, {"Retry-After": retry_after_canary}),
                "success",
            ]
        ) as (api_base_url, server),
    ):
        result = _call_scripted_qwencloud(api_base_url)

    assert result["choices"][0]["message"]["content"] == "ok"
    assert len(server.attempts) == 3
    assert len(post_urls) == 3
    assert len(returned_responses) == 3
    assert all(id(response) in closed_response_ids for response in returned_responses)
    assert len(closed_session_ids) == 1
    assert sleeps == [pytest.approx(0.5)]
    rendered = "\n".join(logs)
    assert retry_after_canary not in rendered
    assert "InvalidHeader" not in rendered


@pytest.mark.allow_network
def test_retry_policy_uses_one_global_budget_across_mixed_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    post_urls, returned_responses, closed_response_ids = (
        _track_real_transport_resources(monkeypatch)
    )
    _configure_qwencloud_transport(monkeypatch, retries=2)

    with _scripted_qwen_server([(503, {}), "stall", (503, {}), "success"]) as (
        api_base_url,
        server,
    ):
        with pytest.raises(ChatProviderError):
            _call_scripted_qwencloud(api_base_url)

    assert len(server.attempts) == 3
    assert len(post_urls) == 3
    assert len(returned_responses) == 3
    assert all(id(response) in closed_response_ids for response in returned_responses)


@pytest.mark.allow_network
def test_truncated_body_retries_once_and_closes_each_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    post_urls, returned_responses, closed_response_ids = (
        _track_real_transport_resources(monkeypatch)
    )
    closed_session_ids = _track_real_session_closes(monkeypatch)
    _configure_qwencloud_transport(monkeypatch, retries=1)

    with (
        _captured_qwencloud_logs() as logs,
        _scripted_qwen_server(["truncated", "success"]) as (api_base_url, server),
    ):
        result = _call_scripted_qwencloud(api_base_url)

    assert result["choices"][0]["message"]["content"] == "ok"
    assert len(server.attempts) == 2
    assert len(post_urls) == 2
    assert len(returned_responses) == 2
    assert all(id(response) in closed_response_ids for response in returned_responses)
    assert len(closed_session_ids) == 1
    assert _TRUNCATED_BODY_CANARY.decode() not in "\n".join(logs)


@pytest.mark.allow_network
@pytest.mark.parametrize(
    ("action", "canary"),
    (
        pytest.param("invalid_json", _INVALID_JSON_CANARY, id="invalid-json"),
        pytest.param("invalid_gzip", _INVALID_GZIP_CANARY, id="invalid-content"),
    ),
)
def test_malformed_success_body_is_typed_redacted_and_not_retried(
    monkeypatch: pytest.MonkeyPatch,
    action: str,
    canary: bytes,
) -> None:
    post_urls, returned_responses, closed_response_ids = (
        _track_real_transport_resources(monkeypatch)
    )
    closed_session_ids = _track_real_session_closes(monkeypatch)
    _configure_qwencloud_transport(monkeypatch, retries=3)

    with (
        _captured_qwencloud_logs() as logs,
        _scripted_qwen_server([action, "success"]) as (api_base_url, server),
        pytest.raises(ChatProviderError) as exc_info,
    ):
        _call_scripted_qwencloud(api_base_url)

    assert len(server.attempts) == 1
    assert len(post_urls) == 1
    assert len(returned_responses) == 1
    assert id(returned_responses[0]) in closed_response_ids
    assert len(closed_session_ids) == 1
    assert exc_info.value.provider == "qwencloud"
    rendered = "\n".join(logs) + "\n" + str(exc_info.value)
    assert "malformed" in rendered.lower()
    assert "network request failed" not in rendered.lower()
    assert canary.decode() not in rendered


def test_nontransient_4xx_and_mode_model_mismatch_are_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = (
        _TransportResponse(
            {"error": {"message": "ordinary validation failure"}},
            status_code=422,
            text='{"error":{"message":"ordinary validation failure"}}',
        ),
        _TransportResponse(
            {
                "error": {
                    "message": (
                        "model qwen-canary is not supported by the Responses API "
                        "RAW-MISMATCH-CANARY"
                    )
                }
            },
            status_code=400,
            text=(
                "model qwen-canary is not supported by the Responses API "
                "RAW-MISMATCH-CANARY"
            ),
        ),
    )
    for index, response in enumerate(responses):
        session = _RecordingSession(response)
        monkeypatch.setattr(
            qwencloud,
            "create_default_session",
            lambda: session,
        )
        monkeypatch.setattr(
            qwencloud,
            "get_runtime_config_snapshot",
            lambda: SimpleNamespace(
                values={
                    "api_settings": {
                        "qwencloud": {
                            "timeout": 3,
                            "retries": 8,
                            "retry_delay": 0,
                        }
                    }
                }
            ),
        )

        with pytest.raises(ChatBadRequestError) as exc_info:
            chat_with_qwencloud(
                input_data=[{"role": "user", "content": "private"}],
                model="qwen3.8-max",
                api_key="key",
                streaming=False,
                api_base_url="https://qwen.example/v1",
                api_mode="responses",
            )
        assert len(session.posts) == 1
        assert exc_info.value.provider == "qwencloud"
        if index == 1:
            recovery = str(exc_info.value)
            assert "compatible model" in recovery
            assert "api_mode" in recovery
            assert "RAW-MISMATCH-CANARY" not in recovery


@pytest.mark.allow_network
def test_stalled_nonretryable_400_is_typed_redacted_and_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    post_urls, returned_responses, closed_response_ids = (
        _track_real_transport_resources(monkeypatch)
    )
    closed_session_ids = _track_real_session_closes(monkeypatch)
    _configure_qwencloud_transport(monkeypatch, retries=3)

    with (
        _captured_qwencloud_logs() as logs,
        _scripted_qwen_server(["stall_400", "success"]) as (api_base_url, server),
        pytest.raises(ChatBadRequestError) as exc_info,
    ):
        _call_scripted_qwencloud(api_base_url)

    assert len(server.attempts) == 1
    assert len(post_urls) == 1
    assert len(returned_responses) == 1
    assert id(returned_responses[0]) in closed_response_ids
    assert len(closed_session_ids) == 1
    assert exc_info.value.provider == "qwencloud"

    rendered = "\n".join(logs) + "\n" + str(exc_info.value)
    assert _STALLED_ERROR_CANARY.decode() not in rendered
    assert "ConnectionError" not in rendered
    assert "HTTPConnectionPool" not in rendered


def test_qwencloud_errors_and_logs_redact_private_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canaries = (
        "AUTHORIZATION-CANARY",
        "MESSAGE-CANARY",
        "TOOL-DESCRIPTION-CANARY",
        "TOOL-ARGUMENT-CANARY",
        "TOOL-CALL-ARGUMENT-CANARY",
        "TOOL-RESULT-CANARY",
        "RAW-BODY-CANARY",
    )
    response = _TransportResponse(
        {"error": {"message": "RAW-BODY-CANARY"}},
        status_code=500,
        text='{"error":{"message":"RAW-BODY-CANARY"}}',
    )
    session = _RecordingSession(response)
    monkeypatch.setattr(
        qwencloud,
        "create_default_session",
        lambda: session,
    )
    monkeypatch.setattr(
        qwencloud,
        "get_runtime_config_snapshot",
        lambda: SimpleNamespace(
            values={
                "api_settings": {
                    "qwencloud": {
                        "timeout": 3,
                        "retries": 0,
                        "retry_delay": 0,
                    }
                }
            }
        ),
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "TOOL-DESCRIPTION-CANARY",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "value": {
                            "type": "string",
                            "description": "TOOL-ARGUMENT-CANARY",
                        }
                    },
                },
            },
        }
    ]

    with (
        _captured_qwencloud_logs() as logs,
        pytest.raises(ChatProviderError) as exc_info,
    ):
        chat_with_qwencloud(
            input_data=[
                {"role": "user", "content": "MESSAGE-CANARY"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_private",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": ('{"secret":"TOOL-CALL-ARGUMENT-CANARY"}'),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_private",
                    "content": "TOOL-RESULT-CANARY",
                },
            ],
            model="qwen3.8-max",
            api_key="AUTHORIZATION-CANARY",
            streaming=False,
            tools=tools,
            api_base_url="https://qwen.example/v1",
            api_mode="chat_completions",
        )

    assert exc_info.value.provider == "qwencloud"
    captured = "\n".join(logs) + "\n" + str(exc_info.value)
    for canary in canaries:
        assert canary not in captured
    assert "qwencloud" in captured.lower()
    assert "status=500" in captured
    assert session.closed is True
    assert response.closed is True

    for status_code, expected_type in (
        (401, ChatAuthenticationError),
        (403, ChatAuthenticationError),
        (429, ChatRateLimitError),
    ):
        status_response = _TransportResponse(
            {"error": {"message": "RAW-BODY-CANARY"}},
            status_code=status_code,
            text="RAW-BODY-CANARY",
        )
        status_session = _RecordingSession(status_response)
        monkeypatch.setattr(
            qwencloud,
            "create_default_session",
            lambda: status_session,
        )
        with (
            _captured_qwencloud_logs() as status_logs,
            pytest.raises(expected_type) as status_exc,
        ):
            chat_with_qwencloud(
                input_data=[{"role": "user", "content": "MESSAGE-CANARY"}],
                model="qwen3.8-max",
                api_key="AUTHORIZATION-CANARY",
                streaming=False,
                api_base_url="https://qwen.example/v1",
                api_mode="chat_completions",
            )
        status_capture = "\n".join(status_logs) + "\n" + str(status_exc.value)
        for canary in canaries:
            assert canary not in status_capture
        assert f"status={status_code}" in status_capture
        assert status_session.closed is True
        assert status_response.closed is True

    for network_error in (
        requests.exceptions.ConnectionError("RAW-BODY-CANARY"),
        requests.exceptions.Timeout("RAW-BODY-CANARY"),
    ):
        network_session = _RecordingSession(_TransportResponse({}), error=network_error)
        monkeypatch.setattr(
            qwencloud,
            "create_default_session",
            lambda: network_session,
        )
        with (
            _captured_qwencloud_logs() as network_logs,
            pytest.raises(ChatProviderError) as network_exc,
        ):
            chat_with_qwencloud(
                input_data=[{"role": "user", "content": "MESSAGE-CANARY"}],
                model="qwen3.8-max",
                api_key="AUTHORIZATION-CANARY",
                streaming=False,
                api_base_url="https://qwen.example/v1",
                api_mode="chat_completions",
            )
        network_capture = "\n".join(network_logs) + "\n" + str(network_exc.value)
        for canary in canaries:
            assert canary not in network_capture
        assert network_session.closed is True
