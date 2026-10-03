"""A request stopped by a local check is not reported as a provider error (TASK-32369).

The Anthropic and Cohere handlers refuse a request with no message to send
before anything leaves the client. They used to raise ``ChatBadRequestError``
(status 400), so Console said "provider returned HTTP 400" for a request the
provider never saw; streaming then rewrapped it as a provider 502. A
status-less ``ChatConfigurationError`` naming the field keeps the blame local,
as task-32342 did for the custom OpenAI-compatible path.
"""

from __future__ import annotations

from typing import Any

import pytest

from tldw_chatbook.Chat.Chat_Deps import ChatConfigurationError
from tldw_chatbook.Chat.console_provider_gateway import (
    ConsoleProviderGateway,
    ConsoleProviderSelection,
    safe_provider_error_copy,
)
from tldw_chatbook.Chat.provider_failures import describe_stream_failure
from tldw_chatbook.LLM_Calls import LLM_API_Calls
from tldw_chatbook.LLM_Calls.LLM_API_Calls import chat_with_anthropic, chat_with_cohere
from tldw_chatbook.Utils.sensitive_llm_logging import sensitive_llm_request

# The handlers read provider settings through the guarded config loader
# before their local check (same admission signature as test_hosted_chat.py
# in Tests/conftest.py), so keep the bootstrap profile.
pytestmark = pytest.mark.bootstrap_profile

_SYSTEM_ONLY = [{"role": "system", "content": "Be brief."}]


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test, rather than reach a real API, if a handler sends anything.

    Args:
        monkeypatch: Replaces the handlers' session factory.
    """

    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("the local check must stop the request before any HTTP")

    monkeypatch.setattr(LLM_API_Calls, "create_default_session", refuse)


def _assert_local(error: ChatConfigurationError, provider: str) -> None:
    assert error.status_code is None
    assert error.provider == provider
    assert error.field == "messages"


def test_anthropic_without_a_user_message_is_stopped_locally() -> None:
    """Anthropic needs a user message; the refusal carries no HTTP status."""
    with pytest.raises(ChatConfigurationError) as caught:
        chat_with_anthropic(
            input_data=_SYSTEM_ONLY, model="claude-sonnet-4-5", api_key="test-key"
        )
    _assert_local(caught.value, "anthropic")


def test_cohere_without_a_conversation_message_is_stopped_locally() -> None:
    """Cohere needs a non-system message; the refusal carries no HTTP status."""
    with pytest.raises(ChatConfigurationError) as caught:
        chat_with_cohere(input_data=_SYSTEM_ONLY, model="command-a-03-2025", api_key="test-key")
    _assert_local(caught.value, "cohere")


def test_console_copy_says_not_sent_and_names_the_field() -> None:
    """The copy names the field and never blames the provider or shows a status."""
    error = ChatConfigurationError(
        "raw detail SECRET-CANARY", provider="anthropic", status_code=None, field="messages"
    )
    copy = safe_provider_error_copy("anthropic", error)
    assert copy == "Request to Anthropic not sent: it failed a local check on messages."
    assert "SECRET-CANARY" not in copy
    # No field: possibly a reply that could not be read (task-32342), so the
    # request may have been sent -- the copy must not claim otherwise.
    unnamed = ChatConfigurationError("x", provider="anthropic", status_code=None)
    assert safe_provider_error_copy("anthropic", unnamed) == (
        "Provider error from Anthropic: configuration error."
    )
    # A configuration error the provider DID answer keeps the provider copy.
    answered = ChatConfigurationError("x", provider="anthropic", status_code=500, field="messages")
    assert safe_provider_error_copy("anthropic", answered).startswith(
        "Provider error from Anthropic: configuration error."
    )


@pytest.mark.asyncio
async def test_a_streamed_local_refusal_reaches_console_as_not_sent() -> None:
    """Through the Console stream path with the real handler: no invented 502."""

    def call_real_handler(**_kwargs: Any) -> Any:
        return chat_with_anthropic(
            input_data=_SYSTEM_ONLY, model="claude-sonnet-4-5", api_key="test-key"
        )

    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": {"anthropic": {"api_key": "test-key"}}},
        chat_api_call_fn=call_real_handler,
    )
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="anthropic", explicit_model="claude-sonnet-4-5")
    )
    with pytest.raises(ChatConfigurationError) as caught:
        _ = [chunk async for chunk in gateway.stream_chat(resolution, [{"role": "user", "content": "hi"}])]

    assert caught.value.status_code is None
    visible = describe_stream_failure(caught.value)
    assert "provider returned HTTP" not in visible
    assert visible.startswith("the app could not build the request")
    assert "Anthropic not sent: it failed a local check on messages" in visible


@pytest.mark.asyncio
async def test_a_sensitive_run_through_the_default_adapter_still_names_the_field() -> None:
    """Qodo #2974: automatic runs mark the request sensitive, and chat_api_call
    then rebuilds the error with redacted text. The field must survive that."""
    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": {"anthropic": {"api_key": "test-key"}}},
    )
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="anthropic", explicit_model="claude-sonnet-4-5")
    )
    # Console refuses an empty history itself, so reach the handler's check the
    # way a real turn can: Anthropic inlines only data-URL images, so a user
    # turn holding just a remote image leaves it no user message to send.
    remote_image_only = [
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "https://example.invalid/a.png"}}],
        }
    ]
    with sensitive_llm_request():
        with pytest.raises(ChatConfigurationError) as caught:
            _ = [chunk async for chunk in gateway.stream_chat(resolution, remote_image_only)]

    assert caught.value.status_code is None
    assert "Anthropic not sent: it failed a local check on messages" in describe_stream_failure(
        caught.value
    )

