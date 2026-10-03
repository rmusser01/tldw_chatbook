"""HTTP failures name the provider the way the user sees it (TASK-33002.14).

``owned_json_post`` turns an HTTP status into the error a user reads ("<name>
authentication failed. Check the API key."). Engine presets pass their
catalog display name, so the copy says "NVIDIA NIM", not the key "nvidia";
legacy adapters pass no display name and keep their copy unchanged. A 404
explains that the model or endpoint was not found, since Fireworks,
SambaNova, Nous and GMI answer an unknown model that way (probed 2026-09-30).
"""

from __future__ import annotations

from typing import Any

import pytest

from tldw_chatbook.Chat.Chat_Deps import (
    ChatAPIError,
    ChatAuthenticationError,
    ChatBadRequestError,
    ChatProviderError,
    ChatRateLimitError,
)
from tldw_chatbook.LLM_Calls import hosted_chat
from tldw_chatbook.LLM_Calls.hosted_chat import HostedHTTPTransportConfig, owned_json_post


class _Response:
    def __init__(self, status: int) -> None:
        self.status_code = status
        self.headers: dict[str, str] = {}

    def json(self) -> Any:
        """An empty HTTP error body has no JSON, as with requests.Response."""
        raise ValueError("The scripted HTTP response has no JSON body.")

    def close(self) -> None:
        """Nothing to release."""


class _Session:
    status = 401

    def mount(self, *_args: Any) -> None:
        """Adapters are irrelevant here."""

    def post(self, *_args: Any, **_kwargs: Any) -> _Response:
        return _Response(type(self).status)

    def close(self) -> None:
        """Nothing to release."""


def _post(monkeypatch: pytest.MonkeyPatch, status: int, display_name: str | None) -> Exception:
    _Session.status = status
    monkeypatch.setattr(hosted_chat, "create_default_session", _Session)
    config = HostedHTTPTransportConfig(
        provider="nvidia",
        base_url="https://integrate.api.nvidia.com/v1",
        api_key="secret",
        timeout=5.0,
        retries=0,
        retry_delay=0.0,
        display_name=display_name,
    )
    with pytest.raises(Exception) as caught:
        owned_json_post(config=config, route="chat/completions", payload={"model": "m"}, streaming=False)
    return caught.value


@pytest.mark.parametrize(
    ("status", "error_type", "copy"),
    [
        (401, ChatAuthenticationError, "NVIDIA NIM authentication failed. Check the API key."),
        (403, ChatAuthenticationError, "NVIDIA NIM authentication failed. Check the API key."),
        (429, ChatRateLimitError, "NVIDIA NIM rate limit exceeded. Retry later."),
        (404, ChatBadRequestError, "NVIDIA NIM could not find that model or endpoint (status 404)."),
        (400, ChatBadRequestError, "NVIDIA NIM rejected the request (status 400)."),
        (503, ChatProviderError, "NVIDIA NIM service failed (status 503)."),
    ],
)
def test_engine_errors_use_the_display_name_but_keep_the_key_identity(
    monkeypatch: pytest.MonkeyPatch, status: int, error_type: type[Exception], copy: str
) -> None:
    """The copy carries the display name; ``error.provider`` stays the key.

    Args:
        monkeypatch: Replaces the HTTP session.
        status: The provider's HTTP status.
        error_type: The error the status maps to.
        copy: The start of the message the user reads.
    """
    error = _post(monkeypatch, status, "NVIDIA NIM")
    assert isinstance(error, error_type)
    assert str(error).startswith(copy) or copy in str(error)
    assert error.provider == "nvidia"


def test_without_a_display_name_the_copy_keeps_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Legacy adapters pass no display name; their copy is unchanged.

    Args:
        monkeypatch: Replaces the HTTP session with one that answers 401.
    """
    error = _post(monkeypatch, 401, None)
    assert "nvidia authentication failed. Check the API key." in str(error)


def test_a_404_tells_the_user_what_to_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 404 names the likely cause instead of a bare status.

    Args:
        monkeypatch: Replaces the HTTP session with one that answers 404.
    """
    error = _post(monkeypatch, 404, "Fireworks")
    assert "Check the model name and that your key can use it." in str(error)


def test_the_engine_passes_its_record_display_name_to_the_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    """The engine hands every preset's catalog name to the transport.

    Args:
        monkeypatch: Replaces request resolution and the transport call.
    """
    from tldw_chatbook.LLM_Calls import hosted_provider_engine
    from tldw_chatbook.LLM_Calls.hosted_provider_engine import HostedProviderResolution
    from tldw_chatbook.provider_registry import RECORDS_BY_KEY

    record = RECORDS_BY_KEY["nvidia"]
    resolution = HostedProviderResolution(
        provider="nvidia", model="m", api_key="secret", base_url=record.default_base_url,
        timeout=5.0, retries=0, retry_delay=0.0, streaming=False,
    )
    monkeypatch.setattr(hosted_provider_engine, "resolve_hosted_request", lambda _r, **_k: resolution)
    captured: dict[str, Any] = {}

    def fake_post(**kwargs: Any) -> Any:
        captured.update(kwargs)
        raise ChatAuthenticationError(provider="nvidia", message="stop here")

    monkeypatch.setattr(hosted_provider_engine, "owned_json_post", fake_post)
    with pytest.raises(ChatAuthenticationError):
        hosted_provider_engine.build_hosted_chat_handler(record)(
            input_data=[{"role": "user", "content": "hi"}], api_key="secret", streaming=False,
        )
    assert captured["config"].display_name == "NVIDIA NIM"


@pytest.mark.parametrize(
    ("status", "category"),
    [(401, "authentication failed"), (404, "model or endpoint not found")],
)
def test_an_engine_failure_reaches_console_with_the_display_name(
    monkeypatch: pytest.MonkeyPatch, status: int, category: str
) -> None:
    """Engine -> real transport -> Console copy, with only the session faked.

    Args:
        monkeypatch: Replaces request resolution and the HTTP session.
        status: The provider's HTTP status.
        category: The failure Console names.
    """
    from tldw_chatbook.Chat.console_provider_gateway import (
        _provider_error_copy_with_model_recovery,
        safe_provider_error_copy,
    )
    from tldw_chatbook.LLM_Calls import hosted_provider_engine
    from tldw_chatbook.LLM_Calls.hosted_provider_engine import HostedProviderResolution
    from tldw_chatbook.provider_registry import RECORDS_BY_KEY

    record = RECORDS_BY_KEY["nvidia"]
    resolution = HostedProviderResolution(
        provider="nvidia", model="m", api_key="secret", base_url=record.default_base_url,
        timeout=5.0, retries=0, retry_delay=0.0, streaming=False,
    )
    monkeypatch.setattr(hosted_provider_engine, "resolve_hosted_request", lambda _r, **_k: resolution)
    _Session.status = status
    monkeypatch.setattr(hosted_chat, "create_default_session", _Session)

    with pytest.raises(ChatAPIError) as caught:
        hosted_provider_engine.build_hosted_chat_handler(record)(
            input_data=[{"role": "user", "content": "hi"}], api_key="secret", streaming=False,
        )
    error = caught.value
    assert error.provider == "nvidia"
    assert "NVIDIA NIM" in str(error)
    copy = safe_provider_error_copy("nvidia", error)
    assert copy.startswith(f"Provider error from NVIDIA NIM: {category}.")
    if status == 404:
        copy = _provider_error_copy_with_model_recovery(copy, model="m", status_code=404)
        assert "could not find this model or endpoint" in copy
