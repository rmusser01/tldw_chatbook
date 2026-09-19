"""TASK-32851 characterization: chat_with_groq / chat_with_openrouter exactly
as they behave on dev today, ahead of their migration onto the hosted_chat
engine.

These tests deliberately pin CURRENT behavior -- including three known defects
documented by the 2026-09-19 cascade review (qa/cascade-review-2026-09-19) --
so the migration has a safety net that fails loudly if anything else moves.
Each defect pin says what TASK-32851 must CHANGE; flip (do not carry) those
assertions in the migration PR:

* Stop on a live stream: the ``data: [DONE]`` sentinel is yielded inside
  ``finally``, so closing the generator raises RuntimeError
  ("generator ignored GeneratorExit") and the response is never closed.
* Streamed usage is never requested (no ``stream_options``) and never
  forwarded, so the gateway usage ledger sees nothing for streamed turns.
* groq's streaming histogram is labelled ``openrouter_api_response_time`` and
  its non-streaming histogram ``mistral_api_response_time``.

The openrouter header pin (``HTTP-Referer``/``X-Title``) is the evidence for
the engine gap TASK-32851 closes first: ``owned_json_post`` hardcodes
Authorization/Content-Type, so openrouter needs an ``extra_headers`` hook
added as a neutral hosted_chat capability.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from tldw_chatbook.Chat.Chat_Deps import ChatConfigurationError
from tldw_chatbook.Chat.Chat_Functions import chat_api_call


def _streaming_ok_response(lines):
    response = Mock()
    response.status_code = 200
    response.raise_for_status = Mock()
    response.iter_lines.return_value = iter(lines)
    response.close = Mock()
    return response


def _nonstreaming_ok_response(payload):
    response = Mock()
    response.status_code = 200
    response.raise_for_status = Mock()
    response.json.return_value = payload
    response.close = Mock()
    return response


_MESSAGES = [{"role": "user", "content": "hi"}]


def test_groq_nonstreaming_returns_legacy_dict_and_standard_payload():
    body = {
        "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
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
    # 0.7 is the EFFECTIVE default: the code literal says 0.2
    # (groq_config.get("temperature", 0.2)) but the default config template
    # the sandbox materializes ships [api_settings.groq] temperature = 0.7,
    # so users get 0.7 unless they edit config. Two defaults for one knob --
    # the migration profile should name one.
    assert payload["temperature"] == 0.7


def test_groq_requires_an_api_key():
    with pytest.raises(ChatConfigurationError, match="Groq API Key required"):
        chat_api_call(
            "groq", messages_payload=_MESSAGES, api_key="", streaming=False
        )


def test_groq_streaming_relays_raw_sse_lines():
    """The current consumer contract: raw SSE text lines, newline-terminated,
    with the final sentinel as 'data: [DONE]\\n\\n' -- exactly the shapes the
    provider sent. TASK-32851 must keep this shape reachable (via a legacy
    shim) or migrate the consumers in the same PR."""
    lines = [
        'data: {"choices": [{"delta": {"content": "hi"}}]}',
        "data: [DONE]",
    ]
    with patch("requests.Session.post") as mock_post:
        mock_post.return_value = _streaming_ok_response(lines)
        generator = chat_api_call(
            "groq",
            messages_payload=_MESSAGES,
            api_key="gsk-test",
            streaming=True,
        )
        yielded = list(generator)

    assert yielded == [
        'data: {"choices": [{"delta": {"content": "hi"}}]}\n',
        # The provider's own [DONE] is relayed...
        "data: [DONE]\n",
        # ...and then the finally block ALSO yields a synthetic sentinel, so
        # every normally-completed groq stream ends with a DUPLICATE [DONE].
        # Consumers tolerate it today; TASK-32851's engine path emits exactly
        # one (this duplication is part of the finally-yield defect family).
        "data: [DONE]\n\n",
    ]


def test_groq_streaming_stop_currently_raises_and_leaks_response():
    """DEFECT PIN (TASK-32851 flips this): the DONE sentinel is yielded inside
    ``finally`` (LLM_API_Calls.py groq stream_generator), so a normal Stop
    close raises RuntimeError and ``response.close()`` after the yield never
    runs. The fixed shape is the OpenAI handler's (pinned in
    Tests/Chat/test_openai_streaming_usage.py); this documents today's."""
    response = _streaming_ok_response(
        [
            'data: {"choices": [{"delta": {"content": "first"}}]}',
            'data: {"choices": [{"delta": {"content": "second"}}]}',
        ]
    )
    with patch("requests.Session.post") as mock_post:
        mock_post.return_value = response
        generator = chat_api_call(
            "groq",
            messages_payload=_MESSAGES,
            api_key="gsk-test",
            streaming=True,
        )
        assert "first" in next(generator)
        with pytest.raises(RuntimeError, match="generator ignored GeneratorExit"):
            generator.close()

    response.close.assert_not_called()


def test_groq_streaming_never_requests_usage():
    """DEFECT PIN (TASK-32851 flips the payload half): no ``stream_options``
    is ever sent, so OpenAI-compatible servers that gate their usage chunk on
    ``include_usage`` never volunteer one and the gateway ledger sees nothing
    for streamed groq turns. A provider that volunteers usage anyway DOES get
    its line relayed verbatim -- whether that becomes a recorded usage chunk
    is the consumer's parsing problem, not this handler's."""
    with patch("requests.Session.post") as mock_post:
        mock_post.return_value = _streaming_ok_response(
            [
                'data: {"choices": [{"delta": {"content": "hi"}}]}',
                'data: {"choices": [], "usage": {"prompt_tokens": 3}}',
            ]
        )
        generator = chat_api_call(
            "groq",
            messages_payload=_MESSAGES,
            api_key="gsk-test",
            streaming=True,
        )
        yielded = list(generator)

    payload = mock_post.call_args.kwargs["json"]
    assert "stream_options" not in payload
    assert 'data: {"choices": [], "usage": {"prompt_tokens": 3}}\n' in yielded


def test_groq_metric_labels_currently_name_other_providers():
    """DEFECT PIN (TASK-32851 renames these): the groq handler logs its
    streaming histogram as ``openrouter_api_response_time`` and its
    non-streaming histogram as ``mistral_api_response_time``."""
    with (
        patch("requests.Session.post") as mock_post,
        patch(
            "tldw_chatbook.LLM_Calls.LLM_API_Calls.log_histogram"
        ) as mock_histogram,
        patch("tldw_chatbook.LLM_Calls.LLM_API_Calls.log_counter"),
    ):
        mock_post.return_value = _streaming_ok_response(["data: [DONE]"])
        list(
            chat_api_call(
                "groq",
                messages_payload=_MESSAGES,
                api_key="gsk-test",
                streaming=True,
            )
        )
        streaming_names = [c.args[0] for c in mock_histogram.call_args_list]
        assert "openrouter_api_response_time" in streaming_names

    with (
        patch("requests.Session.post") as mock_post,
        patch(
            "tldw_chatbook.LLM_Calls.LLM_API_Calls.log_histogram"
        ) as mock_histogram,
        patch("tldw_chatbook.LLM_Calls.LLM_API_Calls.log_counter"),
    ):
        mock_post.return_value = _nonstreaming_ok_response(
            {"choices": [{"message": {"content": "hi"}}]}
        )
        chat_api_call(
            "groq",
            messages_payload=_MESSAGES,
            api_key="gsk-test",
            streaming=False,
        )
        nonstreaming_names = [c.args[0] for c in mock_histogram.call_args_list]
        assert "mistral_api_response_time" in nonstreaming_names


def test_openrouter_sends_attribution_headers():
    """The engine-gap evidence: openrouter sends ``HTTP-Referer``/``X-Title``
    on every request. ``owned_json_post`` hardcodes Authorization/Content-Type
    today, so TASK-32851 adds a neutral ``extra_headers`` capability before
    this provider can migrate."""
    with patch("requests.Session.post") as mock_post:
        mock_post.return_value = _nonstreaming_ok_response(
            {"choices": [{"message": {"content": "hi"}}]}
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
