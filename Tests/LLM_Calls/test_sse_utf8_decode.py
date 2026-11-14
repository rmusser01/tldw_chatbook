"""TASK-38589: legacy SSE paths must decode streams as UTF-8 explicitly.

Four legacy streaming paths used ``requests``' ``iter_lines(decode_unicode=True)``,
which decodes with the response header's charset. Endpoints that declare
``text/event-stream`` WITHOUT ``charset=utf-8`` (common for llama.cpp/vLLM/
LiteLLM behind the ``openai`` provider and for local_* providers) get
ISO-8859-1 from ``requests.utils.get_encoding_from_headers`` -- silently
corrupting every non-ASCII character (the JSON still parses, so nothing
fails loudly). The fix iterates BYTES (``iter_lines()`` no arg) and decodes
each line as UTF-8 explicitly, mirroring the anthropic path and the hosted
SSERecordDecoder.

Each test feeds a fake response that reproduces the bug's mechanics
faithfully: ``iter_lines(decode_unicode=True)`` decodes with the
charset-less header's ISO-8859-1 (exactly what a real ``requests.Response``
does), while plain ``iter_lines()`` (no arg) yields raw bytes.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
import requests

from tldw_chatbook.Chat.Chat_Functions import chat_api_call
from tldw_chatbook.LLM_Calls.LLM_API_Calls import _responses_stream_to_chat_sse
from tldw_chatbook.LLM_Calls import LLM_API_Calls_Local
from tldw_chatbook.LLM_Calls.LLM_API_Calls_Local import (
    _chat_with_openai_compatible_local_server,
)

pytestmark = [pytest.mark.unit]

# ISO-8859-1 mis-decode signatures (UTF-8 bytes read as latin-1) that must
# never appear in output once the explicit UTF-8 decode lands.
_MOJIBAKE_SIGNATURES = ("Ã©", "Ã¯", "Ã¼", "â€”", "Ä‘")


class FakeSession:
    """Minimal session stand-in: ``post`` returns a fixed streaming response."""

    def __init__(self, response):
        self._response = response
        self.trust_env = True

    def mount(self, *args, **kwargs):
        pass

    def post(self, url, **kwargs):
        return self._response

    def close(self):
        pass


class CharsetlessSSEStreamResponse:
    """A requests-like streaming response with no charset in Content-Type.

    ``encoding`` is resolved exactly the way a real ``requests.Response``
    resolves it (``get_encoding_from_headers`` -> ISO-8859-1 for charset-less
    ``text/*``), so ``iter_lines(decode_unicode=True)`` -- the pre-fix call --
    mojibake-decodes the UTF-8 body, while ``iter_lines()`` (no arg) yields
    the raw bytes the fixed code decodes itself.
    """

    def __init__(self, lines: list[str]):
        # Real requests responses use a case-insensitive dict; the plain
        # lowercase key is what get_encoding_from_headers reads.
        self.headers = {"content-type": "text/event-stream"}
        self.status_code = 200
        self._byte_lines = [line.encode("utf-8") for line in lines]
        self.encoding = requests.utils.get_encoding_from_headers(self.headers)
        self.closed = False

    def raise_for_status(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True

    def iter_lines(self, decode_unicode: bool = False):
        if decode_unicode:
            return iter([line.decode(self.encoding) for line in self._byte_lines])
        return iter(list(self._byte_lines))


def _sse_data_payloads(stream_text: str) -> list:
    """Parse every non-[DONE] ``data:`` frame of an SSE stream."""
    payloads = []
    for frame in stream_text.split("\n"):
        if frame.startswith("data:"):
            raw = frame[len("data:") :].strip()
            if raw and raw != "[DONE]":
                payloads.append(json.loads(raw))
    return payloads


def test_fake_reproduces_the_header_charset_mechanics():
    response = CharsetlessSSEStreamResponse(["data: café"])
    assert response.encoding == "ISO-8859-1"
    # The pre-fix decode path mojibakes the UTF-8 bytes:
    assert list(response.iter_lines(decode_unicode=True)) == ["data: cafÃ©"]
    # The byte path yields the raw UTF-8 payload:
    assert list(response.iter_lines()) == ["data: café".encode("utf-8")]


@patch("requests.Session.post")
def test_openai_chat_stream_decodes_utf8_without_charset(mock_post, monkeypatch):
    """LLM_API_Calls.py openai chat streaming: a charset-less endpoint's
    non-ASCII content arrives intact, not ISO-8859-1 mojibake."""
    # Stub the settings/session seams so the guarded config loader never
    # runs (this test exercises SSE decoding, not config resolution).
    from contextlib import nullcontext

    from tldw_chatbook.LLM_Calls import LLM_API_Calls as llm_calls
    from tldw_chatbook.LLM_Calls import recovery_review

    response = CharsetlessSSEStreamResponse(
        [
            'data: {"choices": [{"delta": {"content": "Résumé — naïve café"}}]}',
            "data: [DONE]",
        ]
    )
    monkeypatch.setattr(
        recovery_review,
        "recovered_settings",
        lambda: {"openai_api": {}, "api_settings": {}},
    )
    monkeypatch.setattr(
        llm_calls,
        "create_owned_default_session",
        lambda: nullcontext(FakeSession(response)),
    )
    generator = llm_calls.chat_with_openai(
        input_data=[{"role": "user", "content": "hi"}],
        api_key="sk-test",
        model="gpt-4o",
        streaming=True,
    )
    output = "".join(generator)

    assert "Résumé — naïve café" in output
    assert not any(signature in output for signature in _MOJIBAKE_SIGNATURES)
    # Per-site line shape preserved: pass-through lines keep their data: form.
    assert "data: [DONE]" in output


def test_openai_responses_stream_translator_decodes_utf8():
    """LLM_API_Calls.py ``_responses_stream_to_chat_sse`` (Responses-API
    translator): deltas decode as UTF-8 despite the charset-less header.

    The translator re-serializes frames with ``json.dumps`` (ASCII-escaped),
    so the assertion parses the frames instead of matching raw substrings.
    """
    response = CharsetlessSSEStreamResponse(
        [
            'data: {"type": "response.output_text.delta", "delta": "Grüße aus München"}',
            'data: {"type": "response.completed", "response": {"usage": {"input_tokens": 1, "output_tokens": 1}}}',
        ]
    )
    chunks = list(_responses_stream_to_chat_sse(response, model="gpt-4o"))
    joined = "".join(chunks)

    payloads = _sse_data_payloads(joined)
    deltas = [
        payload["choices"][0]["delta"].get("content", "")
        for payload in payloads
        if payload.get("object") == "chat.completion.chunk"
    ]
    assert "Grüße aus München" in deltas
    assert not any(
        any(signature in delta for signature in _MOJIBAKE_SIGNATURES)
        for delta in deltas
    )
    # Per-site line shape preserved: chat.completion.chunk SSE frames.
    assert all(chunk.endswith("\n\n") for chunk in chunks)
    assert "chat.completion.chunk" in joined


@patch("requests.post")
@patch("requests.Session.post")
def test_gemini_stream_decodes_utf8_without_charset(mock_session_post, _, monkeypatch):
    """LLM_API_Calls.py Gemini stream translator: non-ASCII Gemini output
    survives a charset-less ``text/event-stream`` endpoint. The translator
    re-serializes to OpenAI chunk frames (ASCII-escaped), so parse them."""
    # Stub the settings/session seams so the guarded config loader never runs.
    from tldw_chatbook.LLM_Calls import LLM_API_Calls as llm_calls
    from tldw_chatbook.LLM_Calls.provider_sessions import close_all_for_current_thread
    from tldw_chatbook.config import RuntimeConfigSnapshot

    response = CharsetlessSSEStreamResponse(
        [
            'data: {"candidates": [{"content": {"parts": [{"text": "Grüße aus Köln"}], "role": "model"}, "index": 0}]}',
        ]
    )
    monkeypatch.setattr(
        llm_calls,
        "get_runtime_config_snapshot",
        lambda: RuntimeConfigSnapshot(generation=0, values={}),
    )
    # Registry factory seam: a unique-key cleanup keeps this test's cached
    # session from leaking into later same-thread tests.
    close_all_for_current_thread()
    monkeypatch.setattr(
        llm_calls,
        "create_pooled_default_session",
        lambda **kwargs: FakeSession(response),
    )
    generator = llm_calls.chat_with_google(
        input_data=[{"role": "user", "content": "hi"}],
        api_key="g-test",
        model="gemini-test",
        streaming=True,
    )
    output = "".join(generator)

    texts = []
    for payload in _sse_data_payloads(output):
        for choice in payload.get("choices", []):
            content = choice.get("delta", {}).get("content")
            if isinstance(content, str):
                texts.append(content)
    assert any("Grüße aus Köln" in text for text in texts)
    assert not any(
        any(signature in text for signature in _MOJIBAKE_SIGNATURES)
        for text in texts
    )


def test_local_openai_compatible_stream_decodes_utf8_without_charset(monkeypatch):
    """LLM_API_Calls_Local.py shared local engine (llama.cpp/vLLM/ollama/...):
    raw SSE pass-through decodes as UTF-8 despite the charset-less header."""
    response = CharsetlessSSEStreamResponse(
        [
            'data: {"choices": [{"delta": {"content": "Đây là văn bản tiếng Việt"}}]}',
            "data: [DONE]",
        ]
    )

    class FakeSession:
        def mount(self, *args, **kwargs):
            pass

        def post(self, url, **kwargs):
            return response

        def close(self):
            pass

    monkeypatch.setattr(
        LLM_API_Calls_Local,
        "create_pooled_default_session",
        lambda **kwargs: FakeSession(),
    )

    generator = _chat_with_openai_compatible_local_server(
        api_base_url="http://127.0.0.1:9123/v1",
        model_name="test-model",
        input_data=[{"role": "user", "content": "hi"}],
        streaming=True,
        provider_name="test-local",
    )
    output = list(generator)
    joined = "".join(output)

    assert "Đây là văn bản tiếng Việt" in joined
    assert not any(signature in joined for signature in _MOJIBAKE_SIGNATURES)
    # Per-site line shape preserved: raw pass-through lines + trailing \n\n,
    # plus the [DONE] sentinel emitted after the finally block.
    assert output[-1] == "data: [DONE]\n\n"
    assert all(
        line.endswith("\n\n") for line in output if line != "data: [DONE]\n\n"
    )
