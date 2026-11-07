"""SSE keep-alive comment frames must never reach the Console transcript.

A proxy or gateway in front of an OpenAI-compatible endpoint legally injects
Server-Sent Events comment frames (lines whose first character is ``:``, e.g.
``: keep-alive``) between records as a connection heartbeat. Two layers
streamed that transport noise into the chat as assistant content:

- the ``chat_with_openai`` and local OpenAI-compatible stream generators
  relayed every non-empty raw line, comments included;
- the Console gateway's content sink (``_content_from_provider_item``)
  returned any non-``data:`` string item verbatim, so a relayed comment
  frame reached the transcript and spammed "keep-alive" repeatedly.

These tests pin all three defences: both generator relays drop non-``data:``
lines, and the sink drops SSE comment frames even if a relay regresses.
"""

import json
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from tldw_chatbook.Chat.Chat_Functions import chat_api_call
from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway
from tldw_chatbook.LLM_Calls.LLM_API_Calls_Local import (
    _chat_with_openai_compatible_local_server,
)

_KEEPALIVE_FRAMES = [
    ": keep-alive",
    "",
    ": keep-alive",
    "event: ping",
    ": OPENROUTER PROCESSING",
]


def _openai_stream_lines():
    return [
        ": keep-alive",
        'data: {"choices": [{"index": 0, "delta": {"content": "Hi"}}]}',
        ": keep-alive",
        'data: {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}',
        ": keep-alive",
        (
            'data: {"choices": [], "usage": {"prompt_tokens": 11, '
            '"completion_tokens": 5, "total_tokens": 16}}'
        ),
        "data: [DONE]",
        ": keep-alive",
    ]


def _streaming_ok_response(lines):
    response = Mock()
    response.status_code = 200
    response.raise_for_status = Mock()
    response.iter_lines.return_value = iter(lines)
    response.close = Mock()
    return response


@pytest.fixture
def hermetic_provider_config():
    """Pin handler config/session reads away from developer-machine state.

    Same contract as ``test_huggingface_streaming_usage``: empty config plus
    plain ``requests.Session`` objects keep these tests on the stream-relay
    contract instead of backup-recovery admission.
    """
    import requests

    empty_snapshot = lambda *a, **k: SimpleNamespace(values={})  # noqa: E731
    patch_specs = [
        ("tldw_chatbook.LLM_Calls.LLM_API_Calls.load_settings", {"return_value": {}}),
        (
            "tldw_chatbook.LLM_Calls.LLM_API_Calls.create_owned_default_session",
            {"side_effect": lambda *a, **k: requests.Session()},
        ),
        (
            "tldw_chatbook.LLM_Calls.LLM_API_Calls.create_default_session",
            {"side_effect": lambda *a, **k: requests.Session()},
        ),
        (
            "tldw_chatbook.LLM_Calls.LLM_API_Calls.requests_verify",
            {"return_value": True},
        ),
        (
            "tldw_chatbook.LLM_Calls.LLM_API_Calls.get_runtime_config_snapshot",
            {"side_effect": empty_snapshot},
        ),
        (
            "tldw_chatbook.LLM_Calls.LLM_API_Calls_Local.create_pooled_default_session",
            {"side_effect": lambda *a, **k: requests.Session()},
        ),
    ]
    with ExitStack() as stack:
        for target, kwargs in patch_specs:
            stack.enter_context(patch(target, **kwargs))
        yield


def _decoded_content_chunks(items):
    chunks = []
    for item in items:
        # The local relay (dev) yields an empty string per non-data line --
        # control frames stay invisible while Stop keeps its close cue.
        if not item.strip():
            continue
        assert isinstance(item, str) and item.startswith("data: "), item
        payload = item[len("data: ") :].strip()
        if payload == "[DONE]":
            continue
        decoded = json.loads(payload)
        choices = decoded.get("choices") or [{}]
        delta = choices[0].get("delta") or {}
        if isinstance(delta.get("content"), str):
            chunks.append(delta["content"])
    return chunks


@patch("requests.Session.post")
@pytest.mark.usefixtures("hermetic_provider_config")
def test_openai_stream_relay_drops_keepalive_comment_frames(mock_post):
    mock_post.return_value = _streaming_ok_response(_openai_stream_lines())

    result = chat_api_call(
        "openai",
        messages_payload=[{"role": "user", "content": "hi"}],
        api_key="sk-test",
        model="gpt-4o",
        streaming=True,
    )
    items = list(result)

    assert not any("keep-alive" in item for item in items), items
    assert not any(item.lstrip().startswith(":") for item in items), items
    assert _decoded_content_chunks(items) == ["Hi"]
    assert items[-1].strip() == "data: [DONE]"


@pytest.mark.usefixtures("hermetic_provider_config")
def test_local_openai_compatible_stream_relay_drops_keepalive_comment_frames():
    """Every local provider (llama.cpp/vllm/ollama/... ) shares this relay."""
    with patch(
        "requests.Session.post",
        return_value=_streaming_ok_response(_openai_stream_lines()),
    ):
        result = _chat_with_openai_compatible_local_server(
            api_base_url="http://127.0.0.1:8080/v1",
            model_name="test-model",
            input_data=[{"role": "user", "content": "hi"}],
            streaming=True,
        )
        items = list(result)

    assert not any("keep-alive" in item for item in items), items
    assert not any(item.lstrip().startswith(":") for item in items), items
    assert _decoded_content_chunks(items) == ["Hi"]
    assert items[-1].strip() == "data: [DONE]"


def test_gateway_sink_drops_sse_comment_frames_from_any_stream():
    """Defence in depth: the sink drops comment frames a relay forwards."""
    stream = iter(
        [
            ": keep-alive\n",
            'data: {"choices": [{"delta": {"content": "Hi"}}]}\n',
            b": keep-alive",
            'data: {"choices": [{"delta": {"content": "!"}}]}\n',
            "data: [DONE]\n",
        ]
    )

    chunks = list(ConsoleProviderGateway.normalize_provider_response(stream))

    assert chunks == ["Hi", "!"], chunks


def test_gateway_sink_comment_only_stream_yields_fallback_copy_not_spam():
    stream = iter(_KEEPALIVE_FRAMES)

    chunks = list(ConsoleProviderGateway.normalize_provider_response(stream))

    assert chunks == ["Provider returned no assistant content."], chunks
