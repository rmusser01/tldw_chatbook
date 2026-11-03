"""TASK-38609 item 2: the shared local engine routes through the ADR-222
per-thread session registry instead of building a fresh pooled session per
call. Two calls with the same provider/endpoint/retry identity must reuse
one session object (registry identity); a different endpoint gets its own.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.LLM_Calls import LLM_API_Calls_Local
from tldw_chatbook.LLM_Calls.LLM_API_Calls_Local import (
    _chat_with_openai_compatible_local_server,
)
from tldw_chatbook.LLM_Calls.provider_sessions import close_all_for_current_thread

pytestmark = [pytest.mark.unit]

_OK_RESPONSE_BODY = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}


class _FakeResponse:
    status_code = 200
    text = '{"choices": [{"message": {"content": "ok"}}]}'

    def json(self):
        return _OK_RESPONSE_BODY

    def raise_for_status(self):
        return None


class _CountingSession:
    """Session stand-in that counts posts; identity is the test subject."""

    def __init__(self):
        self.posts = 0

    def mount(self, *args, **kwargs):
        pass

    def post(self, url, **kwargs):
        self.posts += 1
        return _FakeResponse()

    def close(self):
        pass


@pytest.fixture(autouse=True)
def _clean_registry():
    close_all_for_current_thread()
    yield
    close_all_for_current_thread()


def _install_factory(monkeypatch, created: list) -> None:
    def factory(**kwargs):
        session = _CountingSession()
        created.append(session)
        return session

    monkeypatch.setattr(
        LLM_API_Calls_Local, "create_pooled_default_session", factory
    )


def test_same_endpoint_reuses_one_registry_session(monkeypatch):
    created: list[_CountingSession] = []
    _install_factory(monkeypatch, created)

    for _ in range(2):
        _chat_with_openai_compatible_local_server(
            api_base_url="http://127.0.0.1:9311/v1",
            model_name="test-model",
            input_data=[{"role": "user", "content": "hi"}],
            streaming=False,
            provider_name="registry-identity-test",
        )

    assert len(created) == 1, "second call must reuse the registry session"
    assert created[0].posts == 2


def test_streaming_and_non_streaming_share_the_registry_session(monkeypatch):
    """The registry owns the session for BOTH paths (documented choice for
    the item-2 ambiguity): the streaming generator closes the per-call
    RESPONSE, never the session, so reuse across paths is safe."""
    created: list[_CountingSession] = []
    _install_factory(monkeypatch, created)

    class _StreamingResponse(_FakeResponse):
        closed = False

        def iter_lines(self, decode_unicode: bool = False):
            return iter([b"data: [DONE]"])

        def close(self):
            self.closed = True

    stream_response = _StreamingResponse()
    sessions: list[_CountingSession] = []

    class _StreamingSession(_CountingSession):
        def post(self, url, **kwargs):
            self.posts += 1
            return stream_response

    def factory(**kwargs):
        session = _StreamingSession()
        sessions.append(session)
        return session

    monkeypatch.setattr(
        LLM_API_Calls_Local, "create_pooled_default_session", factory
    )

    generator = _chat_with_openai_compatible_local_server(
        api_base_url="http://127.0.0.1:9312/v1",
        model_name="test-model",
        input_data=[{"role": "user", "content": "hi"}],
        streaming=True,
        provider_name="registry-identity-test",
    )
    list(generator)
    assert stream_response.closed, "streaming generator must close the response"

    _chat_with_openai_compatible_local_server(
        api_base_url="http://127.0.0.1:9312/v1",
        model_name="test-model",
        input_data=[{"role": "user", "content": "hi"}],
        streaming=False,
        provider_name="registry-identity-test",
    )

    assert len(sessions) == 1, "streaming and non-streaming share one session"
    assert sessions[0].posts == 2


def test_different_endpoint_gets_its_own_session(monkeypatch):
    created: list[_CountingSession] = []
    _install_factory(monkeypatch, created)

    for port in (9321, 9322):
        _chat_with_openai_compatible_local_server(
            api_base_url=f"http://127.0.0.1:{port}/v1",
            model_name="test-model",
            input_data=[{"role": "user", "content": "hi"}],
            streaming=False,
            provider_name="registry-identity-test",
        )

    assert len(created) == 2
