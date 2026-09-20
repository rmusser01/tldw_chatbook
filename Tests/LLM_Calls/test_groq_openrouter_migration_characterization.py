"""TASK-32851: groq and openrouter chat through the hosted_chat engine.

These tests were characterization pins of the pre-migration handlers (commit
533c8ef84f) and are now the migration's contract: each one either pins a
consumer-visible shape the migration must PRESERVE (non-streaming legacy dict,
payload fields, attribution headers, key requirement) or a defect the
migration FIXES (clean Stop with exactly-once close, requested + forwarded
streamed usage, one [DONE] sentinel, provider-correct metric labels).

Defect history, pinned by the 2026-09-19 cascade review on the old handlers:
the DONE sentinel was yielded inside ``finally`` (Stop raised RuntimeError and
leaked the response, and every normal completion ended with a DUPLICATE
[DONE]); ``stream_options`` was never requested so streamed turns carried no
usage; groq logged ``openrouter_api_response_time`` (streaming) and
``mistral_api_response_time`` (non-streaming).
"""

from __future__ import annotations

import json
from unittest.mock import Mock, patch

import pytest

from tldw_chatbook.Chat.Chat_Deps import ChatConfigurationError
from tldw_chatbook.Chat.Chat_Functions import chat_api_call


_MESSAGES = [{"role": "user", "content": "hi"}]


def _nonstreaming_ok_response(payload):
    response = Mock()
    response.status_code = 200
    response.raise_for_status = Mock()
    response.json.return_value = payload
    response.close = Mock()
    return response


def _sse_stream_response(*body_parts: bytes):
    """A response whose body is consumed via iter_content, as the engine does.

    The pre-migration handlers relayed ``iter_lines``; the hosted_chat engine
    owns the response and decodes SSE from ``iter_content`` chunks, so the
    fakes follow the engine's seam.
    """
    response = Mock()
    response.status_code = 200
    response.raise_for_status = Mock()
    response.iter_content = Mock(return_value=iter(list(body_parts)))
    response.close = Mock()
    return response


def _groq_stream_body() -> bytes:
    """A minimal groq-shaped SSE body the engine accepts.

    Terminal choice event with finish_reason, trailing usage event (what
    ``stream_options.include_usage`` produces), then [DONE].
    """
    return (
        b'data: {"choices": [{"index": 0, "delta": {"content": "hi"}}]}\n\n'
        b'data: {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}\n\n'
        b'data: {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 1,'
        b' "total_tokens": 4}}\n\n'
        b"data: [DONE]\n\n"
    )


def _data_lines(yielded):
    parsed = []
    for line in yielded:
        assert isinstance(line, str), f"raw-line contract: got {type(line)}"
        assert line.startswith("data: ")
        parsed.append(line[len("data: ") :].rstrip("\n"))
    return parsed


def test_groq_nonstreaming_returns_legacy_dict_and_standard_payload():
    body = {
        "id": "chatcmpl-x",
        "object": "chat.completion",
        "created": 1,
        "model": "llama-3.1-8b-instant",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    with patch("requests.Session.post") as mock_post:
        mock_post.return_value = _nonstreaming_ok_response(body)
        result = chat_api_call(
            "groq",
            messages_payload=_MESSAGES,
            api_key="gsk-test",
            model="llama-3.1-8b-instant",
            streaming=False,
        )

    assert result["choices"][0]["message"]["content"] == "hi"
    assert result["usage"]["total_tokens"] == 2
    sent = mock_post.call_args
    assert sent.kwargs["headers"]["Authorization"] == "Bearer gsk-test"
    payload = sent.kwargs["json"]
    assert payload["model"] == "llama-3.1-8b-instant"
    assert payload["stream"] is False
    assert "stream_options" not in payload
    # 0.7 is the EFFECTIVE default: the code literal says 0.2
    # (groq_config.get("temperature", 0.2)) but the default config template
    # ships [api_settings.groq] temperature = 0.7. Two defaults for one knob;
    # the migration profile kept the existing resolution unchanged.
    assert payload["temperature"] == 0.7


def test_groq_requires_an_api_key():
    with pytest.raises(ChatConfigurationError, match="Groq API Key required"):
        chat_api_call(
            "groq", messages_payload=_MESSAGES, api_key="", streaming=False
        )


def test_groq_streaming_relays_line_events_with_single_done_sentinel():
    """The consumer contract stays newline-terminated data lines with exactly
    one final 'data: [DONE]\\n\\n' -- the old handler relayed the provider's
    own [DONE] and THEN yielded a synthetic one, so consumers saw two."""
    with patch("requests.Session.post") as mock_post:
        mock_post.return_value = _sse_stream_response(_groq_stream_body())
        generator = chat_api_call(
            "groq",
            messages_payload=_MESSAGES,
            api_key="gsk-test",
            streaming=True,
        )
        yielded = list(generator)

    payload = mock_post.call_args.kwargs["json"]
    assert payload["stream"] is True
    events = _data_lines(yielded)
    # exactly one sentinel, and it is last
    assert events.count("[DONE]") == 1
    assert events[-1] == "[DONE]"
    # the visible delta content round-trips as JSON data lines
    parsed = [json.loads(item) for item in events[:-1]]
    assert any(
        choice.get("delta", {}).get("content") == "hi"
        for event in parsed
        for choice in event.get("choices", [])
    )


def test_groq_streaming_requests_and_forwards_usage():
    """The fixed contract: stream_options.include_usage is requested, and the
    provider's trailing usage chunk reaches the consumer as a data line the
    gateway's usage ledger can parse."""
    with patch("requests.Session.post") as mock_post:
        mock_post.return_value = _sse_stream_response(_groq_stream_body())
        generator = chat_api_call(
            "groq",
            messages_payload=_MESSAGES,
            api_key="gsk-test",
            streaming=True,
        )
        yielded = list(generator)

    payload = mock_post.call_args.kwargs["json"]
    assert payload["stream_options"] == {"include_usage": True}
    parsed = [json.loads(item) for item in _data_lines(yielded)[:-1]]
    usage_events = [event for event in parsed if event.get("usage")]
    assert usage_events, "streamed usage chunk must be forwarded"
    assert usage_events[-1]["usage"]["total_tokens"] == 4


def test_groq_streaming_stop_is_clean_and_closes_exactly_once():
    """The fixed contract: Stop (generator close) is a clean method close with
    no yield-after-GeneratorExit, and the response closes exactly once."""
    response = _sse_stream_response(_groq_stream_body())
    with patch("requests.Session.post") as mock_post:
        mock_post.return_value = response
        generator = chat_api_call(
            "groq",
            messages_payload=_MESSAGES,
            api_key="gsk-test",
            streaming=True,
        )
        assert "hi" in next(generator)
        generator.close()

    response.close.assert_called_once_with()


def test_groq_metric_labels_name_groq():
    with (
        patch("requests.Session.post") as mock_post,
        patch(
            "tldw_chatbook.LLM_Calls.groq.log_histogram"
        ) as mock_histogram,
        patch("tldw_chatbook.LLM_Calls.groq.log_counter"),
    ):
        mock_post.return_value = _sse_stream_response(_groq_stream_body())
        list(
            chat_api_call(
                "groq",
                messages_payload=_MESSAGES,
                api_key="gsk-test",
                streaming=True,
            )
        )
        streaming_names = [c.args[0] for c in mock_histogram.call_args_list]
        assert "groq_api_response_time" in streaming_names
        assert "openrouter_api_response_time" not in streaming_names
        assert "mistral_api_response_time" not in streaming_names

    with (
        patch("requests.Session.post") as mock_post,
        patch(
            "tldw_chatbook.LLM_Calls.groq.log_histogram"
        ) as mock_histogram,
        patch("tldw_chatbook.LLM_Calls.groq.log_counter"),
    ):
        mock_post.return_value = _nonstreaming_ok_response(
            {
                "id": "x",
                "object": "chat.completion",
                "created": 1,
                "model": "llama-3.1-8b-instant",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        )
        chat_api_call(
            "groq",
            messages_payload=_MESSAGES,
            api_key="gsk-test",
            streaming=False,
        )
        nonstreaming_names = [c.args[0] for c in mock_histogram.call_args_list]
        assert "groq_api_response_time" in nonstreaming_names
        assert "groq_api_input_tokens" in nonstreaming_names
        assert "mistral_api_response_time" not in nonstreaming_names


def test_openrouter_sends_attribution_headers():
    """The engine-gap evidence, now the contract: openrouter's
    HTTP-Referer/X-Title ride on every request through the neutral
    extra_headers capability."""
    with patch("requests.Session.post") as mock_post:
        mock_post.return_value = _nonstreaming_ok_response(
            {
                "id": "x",
                "object": "chat.completion",
                "created": 1,
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        )
        chat_api_call(
            "openrouter",
            messages_payload=_MESSAGES,
            api_key="sk-or-test",
            streaming=False,
        )

    headers = mock_post.call_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer sk-or-test"
    assert headers["HTTP-Referer"] == "http://localhost"
    assert headers["X-Title"] == "TLDW-API"


def test_openrouter_streaming_stop_and_usage_are_clean():
    with patch("requests.Session.post") as mock_post:
        response = _sse_stream_response(_groq_stream_body())
        mock_post.return_value = response
        generator = chat_api_call(
            "openrouter",
            messages_payload=_MESSAGES,
            api_key="sk-or-test",
            streaming=True,
        )
        assert "hi" in next(generator)
        generator.close()

    response.close.assert_called_once_with()
    payload = mock_post.call_args.kwargs["json"]
    assert payload["stream_options"] == {"include_usage": True}
