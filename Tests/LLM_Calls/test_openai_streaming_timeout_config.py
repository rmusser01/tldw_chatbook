"""TASK-38609 item 5: chat_with_openai streaming must honor the configured
``api_timeout`` instead of a hardcoded 180s (the non-streaming path already
honors ``openai_config.get("api_timeout", 90.0)``).
"""

from __future__ import annotations

from contextlib import nullcontext

import pytest

from tldw_chatbook.LLM_Calls import LLM_API_Calls as llm_calls
from tldw_chatbook.LLM_Calls import recovery_review

pytestmark = [pytest.mark.unit]


class _TimeoutCapturingSession:
    def __init__(self, response):
        self._response = response
        self.post_kwargs: list[dict] = []

    def post(self, url, **kwargs):
        self.post_kwargs.append(kwargs)
        return self._response

    def close(self):
        pass


class _StreamResponse:
    status_code = 200
    headers = {"content-type": "text/event-stream; charset=utf-8"}
    text = ""

    def raise_for_status(self):
        return None

    def iter_lines(self, decode_unicode: bool = False):
        return iter([b"data: [DONE]"])

    def close(self):
        pass


def test_streaming_timeout_honors_configured_api_timeout(monkeypatch):
    response = _StreamResponse()
    session = _TimeoutCapturingSession(response)
    monkeypatch.setattr(
        recovery_review,
        "recovered_settings",
        lambda: {"openai_api": {"api_timeout": 37.5}, "api_settings": {}},
    )
    monkeypatch.setattr(
        llm_calls,
        "create_owned_default_session",
        lambda: nullcontext(session),
    )

    generator = llm_calls.chat_with_openai(
        input_data=[{"role": "user", "content": "hi"}],
        api_key="sk-test",
        model="gpt-4o",
        streaming=True,
    )
    list(generator)

    assert session.post_kwargs, "streaming post never fired"
    assert all(
        kwargs.get("timeout") == 37.5 for kwargs in session.post_kwargs
    ), f"expected configured api_timeout=37.5, got {[k.get('timeout') for k in session.post_kwargs]}"


def test_streaming_timeout_falls_back_to_90_default(monkeypatch):
    response = _StreamResponse()
    session = _TimeoutCapturingSession(response)
    monkeypatch.setattr(
        recovery_review,
        "recovered_settings",
        lambda: {"openai_api": {}, "api_settings": {}},
    )
    monkeypatch.setattr(
        llm_calls,
        "create_owned_default_session",
        lambda: nullcontext(session),
    )

    generator = llm_calls.chat_with_openai(
        input_data=[{"role": "user", "content": "hi"}],
        api_key="sk-test",
        model="gpt-4o",
        streaming=True,
    )
    list(generator)

    assert all(
        kwargs.get("timeout") == 90.0 for kwargs in session.post_kwargs
    ), f"expected the 90.0 default, got {[k.get('timeout') for k in session.post_kwargs]}"
