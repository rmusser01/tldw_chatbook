"""Session-usage tap tests: provider functions record into the ledger.

Mirrors the mocking pattern from Tests/Chat/test_chat_mocked_apis.py /
test_openai_streaming_usage.py: patch ``requests.Session.post``, drive the
real dispatcher via ``chat_api_call``, and inspect the session ledger.

The first three tests are the brief's verbatim tap tests (openai exact,
openai estimate fallback, anthropic exact). The remaining tests extend the
same idiom to every other tapped site (cohere, google, huggingface) and the
moonshot/zai compatibility wrappers, so a wrong variable name at any tap
cannot slip through unexercised.
"""

import importlib
import json
from unittest.mock import Mock, patch

import pytest

from tldw_chatbook.Chat.session_usage import reset_for_tests, session_usage

# These tests drive the real dispatcher, whose handler hot path reads the
# guarded config loader; under the per-test env redirect that admission
# fails closed with RecoveryRequired("raw_source_selection_changed") --
# the same signature test_chat_unit_mocked_APIs.py / test_hosted_chat.py
# carry in Tests/conftest.py's keep-list. Keep the bootstrap profile.
pytestmark = pytest.mark.bootstrap_profile


@pytest.fixture(autouse=True)
def _fresh_ledger():
    reset_for_tests()
    yield
    reset_for_tests()


def _mock_post(payload_dict):
    response = Mock()
    response.status_code = 200
    response.raise_for_status = Mock()
    response.json.return_value = payload_dict
    return response


def test_openai_nonstreaming_records_exact_usage():
    body = {
        "choices": [{"message": {"role": "assistant", "content": "hello there"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    with patch("requests.Session.post", return_value=_mock_post(body)):
        from tldw_chatbook.Chat.Chat_Functions import chat_api_call

        chat_api_call(
            "openai",
            messages_payload=[{"role": "user", "content": "hi"}],
            api_key="sk-test",
            model="gpt-4o",
            streaming=False,
        )
    snap = session_usage().snapshot()
    assert snap.exact_tokens == 15
    assert snap.estimated_tokens == 0
    assert snap.calls == 1


def test_openai_nonstreaming_without_usage_records_estimate():
    body = {
        "choices": [{"message": {"role": "assistant", "content": "hello there"}}],
        # no "usage" key at all
    }
    with patch("requests.Session.post", return_value=_mock_post(body)):
        from tldw_chatbook.Chat.Chat_Functions import chat_api_call

        chat_api_call(
            "openai",
            messages_payload=[{"role": "user", "content": "hi"}],
            api_key="sk-test",
            model="gpt-4o",
            streaming=False,
        )
    snap = session_usage().snapshot()
    # Exact arithmetic pins BOTH estimate sides: a broken prompt-side
    # extractor (e.g. a never-raising helper that silently returns "")
    # undercounts and fails this assertion.
    from tldw_chatbook.Chat.usage_recorder import estimate_tokens

    expected = estimate_tokens(
        json.dumps([{"role": "user", "content": "hi"}])
    ) + estimate_tokens("hello there")
    assert snap.estimated_tokens == expected
    assert snap.exact_tokens == 0
    assert snap.calls == 1


def test_anthropic_nonstreaming_records_exact_usage():
    body = {
        "content": [{"type": "text", "text": "hi back"}],
        "usage": {"input_tokens": 7, "output_tokens": 3},
    }
    with patch("requests.Session.post", return_value=_mock_post(body)):
        from tldw_chatbook.Chat.Chat_Functions import chat_api_call

        chat_api_call(
            "anthropic",
            messages_payload=[{"role": "user", "content": "hi"}],
            api_key="sk-test",
            model="claude-3-5-sonnet",
            streaming=False,
        )
    snap = session_usage().snapshot()
    assert snap.exact_tokens == 10
    assert snap.calls == 1


def test_cohere_nonstreaming_records_exact_usage():
    body = {
        "id": "chat-1",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": "hi back"}],
        },
        "finish_reason": "COMPLETE",
        "usage": {"billed_units": {"input_tokens": 8, "output_tokens": 4}},
    }
    with patch("requests.Session.post", return_value=_mock_post(body)):
        from tldw_chatbook.Chat.Chat_Functions import chat_api_call

        chat_api_call(
            "cohere",
            messages_payload=[{"role": "user", "content": "hi"}],
            api_key="sk-test",
            model="command-r-plus",
            streaming=False,
        )
    snap = session_usage().snapshot()
    assert snap.exact_tokens == 12
    assert snap.estimated_tokens == 0
    assert snap.calls == 1


def test_google_nonstreaming_records_exact_usage():
    body = {
        "candidates": [
            {
                "content": {"parts": [{"text": "hello"}], "role": "model"},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 6,
            "candidatesTokenCount": 4,
            "totalTokenCount": 10,
        },
    }
    with patch("requests.Session.post", return_value=_mock_post(body)):
        from tldw_chatbook.Chat.Chat_Functions import chat_api_call

        chat_api_call(
            "google",
            messages_payload=[{"role": "user", "content": "hi"}],
            api_key="test-key",
            model="gemini-2.0-flash",
            streaming=False,
        )
    snap = session_usage().snapshot()
    assert snap.exact_tokens == 10
    assert snap.estimated_tokens == 0
    assert snap.calls == 1


def test_huggingface_nonstreaming_records_exact_usage():
    body = {
        "choices": [{"message": {"role": "assistant", "content": "hello"}}],
        "usage": {"prompt_tokens": 9, "completion_tokens": 6, "total_tokens": 15},
    }
    with patch("requests.Session.post", return_value=_mock_post(body)):
        from tldw_chatbook.Chat.Chat_Functions import chat_api_call

        chat_api_call(
            "huggingface",
            messages_payload=[{"role": "user", "content": "hi"}],
            api_key="hf_test",
            model="meta-llama/Llama-3.1-8B-Instruct",
            streaming=False,
        )
    snap = session_usage().snapshot()
    assert snap.exact_tokens == 15
    assert snap.estimated_tokens == 0
    assert snap.calls == 1


def test_moonshot_nonstreaming_records_exact_usage():
    body = {
        "id": "chatcmpl-moon-1",
        "object": "chat.completion",
        "created": 1,
        "model": "kimi-k2-instruct",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hi back"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    with patch("requests.Session.post", return_value=_mock_post(body)):
        from tldw_chatbook.Chat.Chat_Functions import chat_api_call

        chat_api_call(
            "moonshot",
            messages_payload=[{"role": "user", "content": "hi"}],
            api_key="sk-test",
            model="kimi-k2-instruct",
            streaming=False,
        )
    snap = session_usage().snapshot()
    assert snap.exact_tokens == 15
    assert snap.estimated_tokens == 0
    assert snap.calls == 1


# ---------------------------------------------------------------------------
# Streaming + service boundary taps (Task 3): every completed response
# records into the ledger exactly once (`calls == 1` is the
# double-counting invariant).
# ---------------------------------------------------------------------------


def _sse(event: dict) -> bytes:
    return f"data: {json.dumps(event)}".encode("utf-8")


ANTHROPIC_STREAM_LINES = [
    _sse({"type": "message_start", "message": {"usage": {"input_tokens": 12}}}),
    _sse({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "he"}}),
    _sse(
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 4},
        }
    ),
    _sse({"type": "message_stop"}),
]


def test_anthropic_streaming_records_exact_once():
    response = Mock()
    response.status_code = 200
    response.raise_for_status = Mock()
    response.iter_lines.return_value = iter(ANTHROPIC_STREAM_LINES)
    response.close = Mock()
    with patch("requests.Session.post", return_value=response):
        from tldw_chatbook.Chat.Chat_Functions import chat_api_call

        generator = chat_api_call(
            "anthropic",
            messages_payload=[{"role": "user", "content": "hi"}],
            api_key="sk-test",
            model="claude-3-5-sonnet",
            streaming=True,
        )
        list(generator)
    snap = session_usage().snapshot()
    assert snap.exact_tokens == 16  # 12 input + 4 output
    assert snap.calls == 1  # exactly once per response


OPENAI_STREAM_LINES = [
    'data: {"id": "1", "choices": [{"index": 0, "delta": {"content": "he"}, "finish_reason": null}]}',
    'data: {"id": "1", "choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}}',
    "data: [DONE]",
]


def test_openai_chat_completions_stream_records_exact_once():
    response = Mock()
    response.status_code = 200
    response.raise_for_status = Mock()
    response.iter_lines.return_value = iter(OPENAI_STREAM_LINES)
    response.close = Mock()
    with patch("requests.Session.post", return_value=response):
        from tldw_chatbook.Chat.Chat_Functions import chat_api_call

        generator = chat_api_call(
            "openai",
            messages_payload=[{"role": "user", "content": "hi"}],
            api_key="sk-test",
            model="gpt-4o",
            streaming=True,
        )
        list(generator)
    snap = session_usage().snapshot()
    assert snap.exact_tokens == 10
    assert snap.calls == 1


def test_openai_stream_cumulative_usage_chunks_record_once():
    """OpenAI-compatible gateways sometimes emit CUMULATIVE usage per
    chunk; keep-last semantics must record only the final total once.
    """
    lines = [
        'data: {"id": "1", "choices": [{"index": 0, "delta": {"content": "he"}, "finish_reason": null}]}',
        'data: {"id": "1", "choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}}',
        'data: {"id": "1", "choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 4, "total_tokens": 11}}',
        "data: [DONE]",
    ]
    response = Mock()
    response.status_code = 200
    response.raise_for_status = Mock()
    response.iter_lines.return_value = iter(lines)
    response.close = Mock()
    with patch("requests.Session.post", return_value=response):
        from tldw_chatbook.Chat.Chat_Functions import chat_api_call

        generator = chat_api_call(
            "openai",
            messages_payload=[{"role": "user", "content": "hi"}],
            api_key="sk-test",
            model="gpt-4o",
            streaming=True,
        )
        list(generator)
    snap = session_usage().snapshot()
    assert snap.exact_tokens == 11  # the LAST cumulative chunk, once
    assert snap.calls == 1


def test_responses_stream_records_completed_usage_once():
    from tldw_chatbook.LLM_Calls.LLM_API_Calls import _responses_stream_to_chat_sse

    completed_event = {
        "type": "response.completed",
        "response": {
            "usage": {
                "input_tokens": 9,
                "output_tokens": 6,
                "total_tokens": 15,
            }
        },
    }
    response = Mock()
    response.iter_lines.return_value = iter([_sse(completed_event)])
    response.close = Mock()
    lines = list(_responses_stream_to_chat_sse(response, model="gpt-5.6"))
    assert any('"usage"' in line for line in lines)
    snap = session_usage().snapshot()
    assert snap.exact_tokens == 15
    assert snap.calls == 1


def test_gateway_sse_relay_does_not_double_record():
    """SSE lines the gateway RELAYS (which may come from provider generators
    that already recorded inside themselves) feed the console signal but
    NOT the session ledger; gateway-native HTTP records at its own sites.
    """
    from tldw_chatbook.Chat.console_provider_gateway import _content_from_sse_data

    class _Signals:
        def __init__(self):
            self.payloads = []

        def record_usage_payload(self, usage):
            self.payloads.append(usage)

    signals = _Signals()
    line = (
        "data: "
        + json.dumps(
            {
                "choices": [],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
            }
        )
        + "\n\n"
    )
    _content_from_sse_data(line, signals=signals)
    snap = session_usage().snapshot()
    assert snap.calls == 0  # no ledger double-record on relayed lines
    assert len(signals.payloads) == 1  # console signal unaffected


def test_legacy_line_stream_records_usage_on_exhaustion():
    """groq/deepseek/mistral/openrouter streaming returns LegacyLineStream
    BEFORE their `_log_usage_metrics` funnel; the shim's natural exhaustion
    is the only place those streams record (cancelled streams skip it).
    """
    from unittest.mock import MagicMock

    from tldw_chatbook.LLM_Calls.legacy_line_stream import LegacyLineStream

    stream = MagicMock()
    stream.__next__.side_effect = StopIteration
    stream.terminal_turn.usage = {
        "prompt_tokens": 8,
        "completion_tokens": 7,
        "total_tokens": 15,
    }
    lines = list(LegacyLineStream(stream))
    assert lines == ["data: [DONE]\n\n"]
    snap = session_usage().snapshot()
    assert snap.exact_tokens == 15
    assert snap.calls == 1


def test_full_response_body_is_not_parsed_as_usage():
    """Contract for gateway-native taps (final-review Critical fix):
    `record_provider_payload` takes the BARE usage dict --
    `from_provider_payload` does not unwrap a nested "usage" key, so a
    full response body records NOTHING. Boundary callers must extract
    `.get("usage")` first, mirroring `_maybe_record_usage`.
    """
    full_body = {
        "choices": [{"message": {"role": "assistant", "content": "hi"}}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
    }
    session_usage().record_provider_payload(full_body)
    assert session_usage().snapshot().calls == 0  # nested usage NOT parsed

    session_usage().record_provider_payload(full_body.get("usage"))
    snap = session_usage().snapshot()
    assert snap.exact_tokens == 6
    assert snap.calls == 1  # extraction records exactly once


def test_gateway_native_usage_helper_extracts_and_is_none_safe():
    """The gateway-native tap funnel: bare-dict extraction, no-op without
    usage, never raises (PR #3046 review -- the native inline sites had no
    coverage)."""
    from tldw_chatbook.Chat.console_provider_gateway import (
        _record_gateway_native_usage,
    )

    _record_gateway_native_usage(
        {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}}
    )
    _record_gateway_native_usage({"choices": []})  # no usage -> no-op
    _record_gateway_native_usage({"usage": None})  # null usage -> no-op
    snap = session_usage().snapshot()
    assert snap.exact_tokens == 8
    assert snap.calls == 1


@pytest.mark.parametrize("module_name", ["moonshot", "zai"])
def test_strict_adapter_streams_record_terminal_usage_on_exhaustion(module_name):
    """MoonshotStream/ZAIStream exhaustion tap (PR #3046 review): streaming
    returns the shim before any usage funnel, so natural exhaustion is the
    only place those streams record."""
    from unittest.mock import MagicMock

    module = importlib.import_module(f"tldw_chatbook.LLM_Calls.{module_name}")
    stream = MagicMock()
    stream.__next__.side_effect = StopIteration
    stream.terminal_turn.usage = {
        "prompt_tokens": 6,
        "completion_tokens": 4,
        "total_tokens": 10,
    }
    shim_class = module.MoonshotStream if module_name == "moonshot" else module.ZAIStream
    with pytest.raises(StopIteration):
        shim_class(stream).__next__()
    snap = session_usage().snapshot()
    assert snap.exact_tokens == 10
    assert snap.calls == 1


@pytest.mark.parametrize(
    "module_name",
    ["groq", "deepseek", "mistral", "openrouter"],
)
def test_extracted_provider_usage_helper_records(module_name):
    module = importlib.import_module(f"tldw_chatbook.LLM_Calls.{module_name}")
    module._log_usage_metrics(
        "model-x",
        {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    )
    snap = session_usage().snapshot()
    assert snap.exact_tokens == 5
    assert snap.calls == 1


def test_zai_nonstreaming_records_exact_usage():
    body = {
        "id": "chatcmpl-zai-1",
        "object": "chat.completion",
        "created": 1,
        "model": "glm-4.6",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hi back"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
    }
    with patch("requests.Session.post", return_value=_mock_post(body)):
        from tldw_chatbook.Chat.Chat_Functions import chat_api_call

        chat_api_call(
            "zai",
            messages_payload=[{"role": "user", "content": "hi"}],
            api_key="sk-test",
            model="glm-4.6",
            streaming=False,
        )
    snap = session_usage().snapshot()
    assert snap.exact_tokens == 15
    assert snap.estimated_tokens == 0
    assert snap.calls == 1


def test_gateway_native_helper_reads_anthropic_message_start_usage():
    """message_start nests usage under "message"; message_delta carries
    output at the top level. Both record, and the ledger sums them."""
    from tldw_chatbook.Chat.console_provider_gateway import (
        _record_gateway_native_usage,
    )

    _record_gateway_native_usage(
        {"type": "message_start", "message": {"usage": {"input_tokens": 9}}}
    )
    _record_gateway_native_usage(
        {"type": "message_delta", "delta": {}, "usage": {"output_tokens": 4}}
    )
    snap = session_usage().snapshot()
    assert snap.exact_tokens == 13
    assert snap.calls == 2


def test_estimate_prompt_text_strips_image_parts():
    from tldw_chatbook.LLM_Calls.LLM_API_Calls import _estimate_prompt_text
    from tldw_chatbook.Chat.usage_recorder import estimate_tokens

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe this"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 10000}},
            ],
        }
    ]
    text = _estimate_prompt_text(messages)
    assert "AAAA" not in text  # base64 gone
    assert estimate_tokens(text) < 100  # anchored to the text, not the image


def test_openai_embeddings_records_to_embeddings_bucket(monkeypatch):
    import tldw_chatbook.LLM_Calls.LLM_API_Calls as llm_calls

    monkeypatch.setattr(
        llm_calls, "load_settings", lambda: {"openai_api": {"api_key": "sk-test"}}
    )
    monkeypatch.setattr(llm_calls, "resolve_provider_api_key", lambda v: v)
    body = {
        "data": [{"embedding": [0.1, 0.2]}],
        "usage": {"prompt_tokens": 7, "total_tokens": 7},
    }
    with patch("requests.Session.post", return_value=_mock_post(body)):
        from tldw_chatbook.LLM_Calls.LLM_API_Calls import get_openai_embeddings

        get_openai_embeddings("hello world", "text-embedding-3-small")
    snap = session_usage().snapshot()
    assert snap.embeddings_tokens == 7
    assert snap.exact_tokens == 0
    assert snap.calls == 0
