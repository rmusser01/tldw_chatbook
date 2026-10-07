"""TASK-2122: huggingface streaming usage + the all-provider stream guard.

HuggingFace was the last of the eleven first-party providers whose streaming
path reported no usage: the payload never asked for
``stream_options.include_usage`` and the generator yielded bare text strings,
so even a usage frame the provider sent could never reach the gateway (which
parses usage out of OpenAI-shaped SSE ``data:`` lines only).
"""

import json
from unittest.mock import Mock, patch

import pytest

from tldw_chatbook.Chat.Chat_Functions import chat_api_call
from tldw_chatbook.Chat.console_provider_gateway import (
    ConsoleProviderStreamSignals,
    _content_from_sse_data,
)

_HF_BASE_URL = "https://router.huggingface.co/v1"


@pytest.fixture
def hermetic_hf_config():
    """Pin the handler's config/session reads away from developer-machine state.

    The provider handlers read config through load_settings()/
    get_runtime_config_snapshot() and build transport sessions via
    create_default_session()/requests_verify(); on a developer machine with
    pending backup-recovery state those raise RecoveryRequired before the
    handler reaches the (mocked) transport. Empty config + a plain Session
    keeps these tests on the request/stream contract only.
    """
    import requests
    from contextlib import ExitStack
    from types import SimpleNamespace

    empty_snapshot = lambda *a, **k: SimpleNamespace(values={})  # noqa: E731
    patch_specs = [
        ("tldw_chatbook.LLM_Calls.LLM_API_Calls.load_settings", {"return_value": {}}),
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
    ]
    patch_specs.extend(
        (
            f"tldw_chatbook.LLM_Calls.{module}.get_runtime_config_snapshot",
            {"side_effect": empty_snapshot},
        )
        for module in (
            "deepseek",
            "groq",
            "mistral",
            "openrouter",
            "moonshot",
            "zai",
        )
    )
    with ExitStack() as stack:
        for target, kwargs in patch_specs:
            stack.enter_context(patch(target, **kwargs))
        yield


def _streaming_ok_response(lines):
    response = Mock()
    response.status_code = 200
    response.raise_for_status = Mock()
    response.iter_lines.return_value = iter(lines)
    response.close = Mock()
    return response


def _hf_stream_lines(include_done=True):
    lines = [
        'data: {"choices": [{"index": 0, "delta": {"role": "assistant"}}]}',
        'data: {"choices": [{"index": 0, "delta": {"content": "Hi"}}]}',
        (
            'data: {"choices": [{"index": 0, "delta": {}, '
            '"finish_reason": "stop"}]}'
        ),
        (
            'data: {"choices": [], "usage": {"prompt_tokens": 11, '
            '"completion_tokens": 5, "total_tokens": 16}}'
        ),
    ]
    if include_done:
        lines.append("data: [DONE]")
    return lines


def _call_huggingface_stream(mock_post, lines, **extra):
    mock_post.return_value = _streaming_ok_response(lines)
    result = chat_api_call(
        "huggingface",
        messages_payload=[{"role": "user", "content": "hi"}],
        api_key="hf-test",
        model="openai/gpt-oss-120b",
        streaming=True,
        api_base_url=_HF_BASE_URL,
        **extra,
    )
    return list(result)


def _decode_sse_items(items):
    """Split OpenAI-shaped SSE lines / dict events into decoded mappings."""

    decoded = []
    for item in items:
        if isinstance(item, dict):
            decoded.append(item)
            continue
        assert isinstance(item, str) and item.startswith("data: "), item
        payload = item[len("data: ") :].strip()
        if payload == "[DONE]":
            continue
        decoded.append(json.loads(payload))
    return decoded


@patch("requests.Session.post")
@pytest.mark.usefixtures("hermetic_hf_config")
def test_streaming_payload_includes_stream_options(mock_post):
    _call_huggingface_stream(mock_post, _hf_stream_lines())
    sent_payload = mock_post.call_args[1]["json"]
    assert sent_payload["stream_options"] == {"include_usage": True}


@patch("requests.Session.post")
@pytest.mark.usefixtures("hermetic_hf_config")
def test_non_streaming_payload_omits_stream_options(mock_post):
    mock_post.return_value = _streaming_ok_response([])
    mock_post.return_value.json.return_value = {
        "choices": [{"message": {"content": "ok"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }
    chat_api_call(
        "huggingface",
        messages_payload=[{"role": "user", "content": "hi"}],
        api_key="hf-test",
        model="openai/gpt-oss-120b",
        streaming=False,
        api_base_url=_HF_BASE_URL,
    )
    sent_payload = mock_post.call_args[1]["json"]
    assert "stream_options" not in sent_payload


@patch("requests.Session.post")
@pytest.mark.usefixtures("hermetic_hf_config")
def test_stream_forwards_trailing_usage_frame(mock_post):
    """A usage frame the provider sends must reach the consumer verbatim."""

    items = _call_huggingface_stream(mock_post, _hf_stream_lines())
    chunks = _decode_sse_items(items)

    usage = [c["usage"] for c in chunks if isinstance(c.get("usage"), dict)]
    assert usage, f"no usage forwarded on the stream: {chunks}"
    assert usage[-1] == {
        "prompt_tokens": 11,
        "completion_tokens": 5,
        "total_tokens": 16,
    }
    # The content delta still streams (the generator is a relay, not a
    # buffer), and exactly one trailing [DONE] sentinel ends the stream.
    assert "Hi" in [
        c["choices"][0]["delta"].get("content")
        for c in chunks
        if c.get("choices") and c["choices"][0].get("delta")
    ]
    assert items[-1].strip() == "data: [DONE]"
    assert sum(1 for item in items if item.strip() == "data: [DONE]") == 1


@patch("requests.Session.post")
@pytest.mark.usefixtures("hermetic_hf_config")
def test_gateway_records_usage_from_huggingface_stream(mock_post):
    """The gateway's SSE consumer records the forwarded usage payload."""

    items = _call_huggingface_stream(mock_post, _hf_stream_lines())
    signals = ConsoleProviderStreamSignals()
    for item in items:
        _content_from_sse_data(item, signals=signals)
    signals.close_usage_call()

    assert signals.completed_usage_payloads == [
        {"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16}
    ], signals.completed_usage_payloads


@patch("requests.Session.post")
@pytest.mark.usefixtures("hermetic_hf_config")
def test_400_naming_stream_options_retries_without_it(mock_post):
    """An OpenAI-compatible endpoint that rejects the field still streams."""

    rejected = Mock()
    rejected.status_code = 400
    rejected.text = '{"error": {"message": "Unknown parameter: stream_options"}}'
    rejected.close = Mock()
    degraded_lines = [
        line for line in _hf_stream_lines() if '"usage"' not in line
    ]
    mock_post.side_effect = [
        rejected,
        _streaming_ok_response(degraded_lines),
    ]

    items = _call_huggingface_stream(mock_post, degraded_lines)
    chunks = _decode_sse_items(items)
    assert not any(isinstance(c.get("usage"), dict) for c in chunks)
    assert any(
        c.get("choices") and c["choices"][0].get("delta", {}).get("content") == "Hi"
        for c in chunks
    )
    retried_payload = mock_post.call_args_list[1][1]["json"]
    assert "stream_options" not in retried_payload
    first_payload = mock_post.call_args_list[0][1]["json"]
    assert first_payload["stream_options"] == {"include_usage": True}


@patch("requests.Session.post")
@pytest.mark.usefixtures("hermetic_hf_config")
def test_stream_without_provider_done_still_ends_with_single_sentinel(mock_post):
    """No provider [DONE]: exactly one synthetic sentinel closes the stream."""

    items = _call_huggingface_stream(mock_post, _hf_stream_lines(include_done=False))
    assert items[-1].strip() == "data: [DONE]"
    assert sum(1 for item in items if item.strip() == "data: [DONE]") == 1


# ---------------------------------------------------------------------------
# TASK-2122 AC#5: the guard. Enumerate the eleven first-party streaming send
# paths and fail if one relays no usage, so a newly added provider cannot
# silently regress. Each family is driven through chat_api_call with its
# real transport seam mocked and a recorded stream that carries usage:
#   - requests-family (openai, anthropic, cohere, google, huggingface) via
#     requests.Session.post;
#   - hosted-engine family (deepseek, groq, mistral, openrouter, moonshot)
#     via hosted_chat.owned_json_post, the single definition site their
#     shared hosted_chat_request resolves at call time;
#   - zai via its own module binding of owned_json_post.
# ---------------------------------------------------------------------------


class _RecordIterator:
    """Canned SSE record stream with the engine stream's close contract."""

    def __init__(self, records):
        self._iter = iter(records)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iter)

    def close(self):
        return None


def _engine_records():
    from tldw_chatbook.LLM_Calls.hosted_chat_streaming import SSERecord

    return [
        SSERecord(event=None, data='{"choices": [{"index": 0, "delta": '
        '{"content": "Hi"}}]}'),
        SSERecord(
            event=None,
            data='{"choices": [{"index": 0, "delta": {}, '
            '"finish_reason": "stop"}]}',
        ),
        SSERecord(
            event=None,
            data='{"choices": [], "usage": {"prompt_tokens": 11, '
            '"completion_tokens": 5, "total_tokens": 16}}',
        ),
        SSERecord(event=None, data="[DONE]"),
    ]


def _fake_owned_json_post(captured):
    from tldw_chatbook.LLM_Calls.hosted_chat_streaming import SSERecord

    def _post(*, config, route, payload, streaming):
        captured.append(dict(payload))
        return _RecordIterator(_engine_records())

    return _post


_REQUESTS_FAMILY_STREAMS = {
    "openai": _hf_stream_lines(),
    "huggingface": _hf_stream_lines(),
    "anthropic": [
        (
            'data: {"type": "message_start", "message": {"usage": '
            '{"input_tokens": 11, "output_tokens": 1}}}'
        ),
        'data: {"type": "content_block_delta", "index": 0, "delta": '
        '{"type": "text_delta", "text": "Hi"}}',
        (
            'data: {"type": "message_delta", "delta": {"stop_reason": '
            '"end_turn"}, "usage": {"output_tokens": 5}}'
        ),
        'data: {"type": "message_stop"}',
    ],
    "cohere": [
        'data: {"type": "message-start"}',
        (
            'data: {"type": "content-delta", "delta": {"message": '
            '{"content": {"text": "Hi"}}}}'
        ),
        (
            'data: {"type": "message-end", "delta": {"finish_reason": '
            '"COMPLETE", "usage": {"tokens": {"input_tokens": 11, '
            '"output_tokens": 5}}}}'
        ),
    ],
    "google": [
        (
            'data: {"candidates": [{"content": {"parts": [{"text": "Hi"}], '
            '"role": "model"}, "index": 0}]}'
        ),
        (
            'data: {"candidates": [{"content": {"parts": [{"text": ""}], '
            '"role": "model"}, "finishReason": "STOP", "index": 0}], '
            '"usageMetadata": {"promptTokenCount": 11, '
            '"candidatesTokenCount": 5, "totalTokenCount": 16}}'
        ),
    ],
}

_ENGINE_FAMILY_KEYS = ("deepseek", "groq", "mistral", "openrouter", "moonshot")

#: OpenAI-semantics providers that must ask for streamed usage via
#: ``stream_options.include_usage`` (the native translators do not have the
#: flag; zai's GLM API sends usage on the final chunk without being asked).
_FLAG_ASKERS = frozenset({"openai", "huggingface", *_ENGINE_FAMILY_KEYS})


@pytest.mark.parametrize(
    "provider_key",
    [
        "openai",
        "anthropic",
        "cohere",
        "google",
        "huggingface",
        *_ENGINE_FAMILY_KEYS,
        "zai",
    ],
)
@pytest.mark.usefixtures("hermetic_hf_config")
def test_every_first_party_streaming_path_emits_usage(provider_key):
    model = {
        "openai": "gpt-4o",
        "anthropic": "claude-sonnet-4-5",
        "cohere": "command-a-03-2025",
        "google": "gemini-2.5-flash",
        "huggingface": "openai/gpt-oss-120b",
        "deepseek": "deepseek-chat",
        "groq": "llama-3.1-8b-instant",
        "mistral": "mistral-large-latest",
        "openrouter": "meta-llama/llama-3.1-8b-instruct",
        "moonshot": "kimi-k3",
        "zai": "glm-4.7",
    }[provider_key]

    if provider_key in _REQUESTS_FAMILY_STREAMS:
        lines = _REQUESTS_FAMILY_STREAMS[provider_key]
        if provider_key == "anthropic":
            # The Anthropic generator decodes raw bytes lines itself.
            lines = [line.encode() for line in lines]
        mock_response = _streaming_ok_response(lines)
        with patch("requests.Session.post", return_value=mock_response) as mock_post:
            result = chat_api_call(
                provider_key,
                messages_payload=[{"role": "user", "content": "hi"}],
                api_key="test-key",
                model=model,
                streaming=True,
                **(
                    {"api_base_url": _HF_BASE_URL}
                    if provider_key == "huggingface"
                    else {}
                ),
            )
            items = list(result)
            sent_payload = mock_post.call_args[1]["json"]
    else:
        captured: list[dict] = []
        targets = (
            ["tldw_chatbook.LLM_Calls.hosted_chat.owned_json_post"]
            if provider_key in _ENGINE_FAMILY_KEYS
            else ["tldw_chatbook.LLM_Calls.zai.owned_json_post"]
        )
        fake = _fake_owned_json_post(captured)
        with (
            patch(targets[0], side_effect=fake),
            patch(
                "tldw_chatbook.LLM_Calls.zai.owned_json_post", side_effect=fake
            ),
        ):
            result = chat_api_call(
                provider_key,
                messages_payload=[{"role": "user", "content": "hi"}],
                api_key="test-key",
                model=model,
                streaming=True,
            )
            items = list(result)
            sent_payload = captured[0]

    chunks = _decode_sse_items(items)
    usage_blocks = [c["usage"] for c in chunks if isinstance(c.get("usage"), dict)]
    assert usage_blocks, (
        f"{provider_key}: streaming output carries no usage block; "
        f"chunks={chunks[:4]}"
    )
    merged_usage: dict = {}
    for block in usage_blocks:
        merged_usage.update(block)
    # Bucket naming follows the gateway's own recorder semantics: OpenAI
    # names win, provider-native names (Anthropic input_/output_tokens,
    # Gemini token counts) are the accepted alternative.
    assert merged_usage.get("prompt_tokens") or merged_usage.get("input_tokens"), (
        f"{provider_key}: prompt-side usage bucket missing; merged={merged_usage}"
    )
    assert merged_usage.get("completion_tokens") or merged_usage.get(
        "output_tokens"
    ), f"{provider_key}: completion-side usage bucket missing; merged={merged_usage}"
    if provider_key in _FLAG_ASKERS:
        # Every OpenAI-semantics provider must ASK for streamed usage too.
        assert sent_payload.get("stream_options") == {"include_usage": True}, (
            f"{provider_key}: streaming payload did not request include_usage"
        )
