"""The dispatcher must not erase what the provider actually said.

TASK-25902 review C3c / TASK-25901 review I3. Every 4xx used to collapse into
`ChatBadRequestError` with a hardcoded 400, which erased the difference between
"our request is malformed" and "the account is out of money" -- so the
credit-terminal fallback trigger could never fire on real traffic. And nothing
ever read the Retry-After header, so the retry policy's header-honouring branch
was unreachable.

These drive the real `chat_api_call` mapping with a stubbed handler raising
`requests.exceptions.HTTPError`, the shape every raise_for_status produces.
"""

from __future__ import annotations

import pytest
import requests

import tldw_chatbook.Chat.Chat_Functions as chat_functions
from tldw_chatbook.Chat.Chat_Deps import (
    ChatBadRequestError,
    ChatRateLimitError,
)
from tldw_chatbook.Agents.fallback_chain import is_credit_terminal


def _http_error(status, headers=None):
    response = requests.Response()
    response.status_code = status
    response._content = b"detail"
    if headers:
        response.headers.update(headers)
    return requests.exceptions.HTTPError(response=response)


def _call_with(monkeypatch, exc):
    def handler(**kwargs):
        raise exc

    monkeypatch.setitem(chat_functions.API_CALL_HANDLERS, "openai", handler)
    return chat_functions.chat_api_call(
        api_endpoint="openai",
        messages_payload=[{"role": "user", "content": "hi"}],
    )


@pytest.mark.parametrize("status", [402, 403])
def test_credit_statuses_survive_the_mapping(monkeypatch, status):
    with pytest.raises(ChatBadRequestError) as caught:
        _call_with(monkeypatch, _http_error(status))

    assert caught.value.status_code == status
    assert is_credit_terminal(caught.value) is True


def test_a_plain_400_is_not_credit_terminal(monkeypatch):
    with pytest.raises(ChatBadRequestError) as caught:
        _call_with(monkeypatch, _http_error(400))

    assert caught.value.status_code == 400
    assert is_credit_terminal(caught.value) is False


def test_retry_after_header_reaches_the_exception(monkeypatch):
    with pytest.raises(ChatRateLimitError) as caught:
        _call_with(monkeypatch, _http_error(429, {"Retry-After": "17"}))

    assert caught.value.retry_after == 17.0


def test_a_date_shaped_retry_after_is_dropped_not_fatal(monkeypatch):
    with pytest.raises(ChatRateLimitError) as caught:
        _call_with(
            monkeypatch,
            _http_error(429, {"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}),
        )

    assert caught.value.retry_after is None


def test_a_429_without_the_header_carries_none(monkeypatch):
    with pytest.raises(ChatRateLimitError) as caught:
        _call_with(monkeypatch, _http_error(429))

    assert caught.value.retry_after is None
