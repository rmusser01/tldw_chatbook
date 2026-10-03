"""TASK-34100.5 AC#4 (gap-06): a failed first reply says what the provider said.

The bodies below are REAL provider responses, recorded 2026-10-03 with curl
(evidence: .worktrees/setup-wizard-ux-qa/evidence/g5-errors/0[1-4]-*.json):
an expired OpenRouter key, a retired Gemini model id, and an invalid OpenAI
and Anthropic key. Only ``error.message`` may reach the user, capped at 200
characters, scrubbed of key-shaped text and prefixed with the provider name;
each failure category names its own fix, and a stream stall is not an
"unexpected provider error".
"""

from __future__ import annotations

import json

import pytest
import requests

from tldw_chatbook.Chat.Chat_Deps import (
    ChatAuthenticationError,
    ChatBadRequestError,
    ChatProviderError,
)
from tldw_chatbook.Chat.console_provider_gateway import (
    _provider_error_copy_with_model_recovery,
    safe_provider_error_copy,
)
from tldw_chatbook.Chat.provider_failures import describe_stream_failure
from tldw_chatbook.Chat.stream_stall_watchdog import StreamStallError

OPENROUTER_EXPIRED_KEY_401 = (
    '{"error":{"message":"API key expired.","code":401,"metadata":{"headers":'
    '{"WWW-Authenticate":"Bearer error=\\"invalid_token\\", '
    'error_description=\\"API key expired\\""}}}}'
)
GEMINI_RETIRED_MODEL_404 = json.dumps(
    {
        "error": {
            "code": 404,
            "message": (
                "This model models/gemini-2.0-flash is no longer available. "
                "Please update your code to use models/gemini-3.8-flash for the "
                "latest features and improvements. We recommend you to use the "
                "Interactions API (https://ai.google.dev/gemini-api/docs/get-started)."
            ),
            "status": "NOT_FOUND",
        }
    }
)
OPENAI_INVALID_KEY_401 = json.dumps(
    {
        "error": {
            "message": (
                "Incorrect API key provided: sk-inval*********************0000. "
                "You can find your API key at "
                "https://platform.openai.com/account/api-keys."
            ),
            "type": "invalid_request_error",
            "code": "invalid_api_key",
            "param": None,
        },
        "status": 401,
    }
)
ANTHROPIC_INVALID_KEY_401 = (
    '{"type":"error","error":{"type":"authentication_error",'
    '"message":"invalid x-api-key"},"request_id":"req_011CffVwrLugJZ2nFTofxCRm"}'
)


def _http_error(status: int, body: str) -> requests.exceptions.HTTPError:
    response = requests.models.Response()
    response.status_code = status
    response._content = body.encode()
    return requests.exceptions.HTTPError(f"{status} Client Error", response=response)


def _raised_from_http(chat_error: Exception, status: int, body: str) -> Exception:
    """Shape of chat_api_call's mapping: raised inside ``except HTTPError``."""
    try:
        try:
            raise _http_error(status, body)
        except requests.exceptions.HTTPError:
            raise chat_error
    except Exception as exc:  # noqa: BLE001 - the mapped exception is the subject
        return exc


def test_expired_openrouter_key_shows_the_providers_reason_and_the_fix() -> None:
    from tldw_chatbook.LLM_Calls.hosted_chat import _raise_http_error

    response = requests.models.Response()
    response.status_code = 401
    response._content = OPENROUTER_EXPIRED_KEY_401.encode()
    with pytest.raises(ChatAuthenticationError) as caught:
        _raise_http_error("openrouter", 401, label="OpenRouter", response=response)

    copy = safe_provider_error_copy("openrouter", caught.value)

    assert copy.startswith("Provider error from OpenRouter: authentication failed.")
    assert "OpenRouter says: “API key expired.”" in copy
    assert "Update the API key" in copy
    # Nothing outside the allowlisted field leaks (echoed headers, codes).
    assert "WWW-Authenticate" not in copy and "invalid_token" not in copy


def test_retired_gemini_model_names_the_404_reason_and_switch_model() -> None:
    exc = _raised_from_http(
        ChatBadRequestError(
            provider="google",
            message="Bad request to google (Status 404).",
            status_code=404,
        ),
        404,
        GEMINI_RETIRED_MODEL_404,
    )

    raw = safe_provider_error_copy("google", exc)
    copy = _provider_error_copy_with_model_recovery(
        raw, model="gemini-2.0-flash", status_code=404
    )

    assert "model or endpoint not found" in copy
    assert "says: “This model models/gemini-2.0-flash is no longer available." in copy
    reason = copy.split("says: “", 1)[1].split("”", 1)[0]
    assert len(reason) <= 200 and reason.endswith("…")
    assert "Alt+M" in copy and "switch model" in copy.lower()


@pytest.mark.parametrize(
    ("provider", "body", "expected"),
    [
        ("openai", OPENAI_INVALID_KEY_401, "Incorrect API key provided: (key hidden)."),
        ("anthropic", ANTHROPIC_INVALID_KEY_401, "invalid x-api-key"),
    ],
)
def test_invalid_key_reason_is_scrubbed_and_only_error_message_is_read(
    provider: str, body: str, expected: str
) -> None:
    exc = _raised_from_http(
        ChatAuthenticationError(provider=provider, message="Auth failed."),
        401,
        body,
    )

    copy = safe_provider_error_copy(provider, exc)

    assert expected in copy
    assert "sk-inval" not in copy and "0000" not in copy
    assert "req_011" not in copy and "authentication_error" not in copy
    assert "Update the API key" in copy


def test_a_severed_chain_never_reads_the_body() -> None:
    """``raise ... from None`` (the sensitive-request policy) stays severed."""
    try:
        try:
            raise _http_error(401, OPENROUTER_EXPIRED_KEY_401)
        except requests.exceptions.HTTPError:
            raise ChatAuthenticationError(provider="openrouter", message="x") from None
    except ChatAuthenticationError as exc:
        copy = safe_provider_error_copy("openrouter", exc)
    assert "says:" not in copy


def test_a_stream_stall_has_its_own_category_and_offers_retry_or_wait() -> None:
    copy = describe_stream_failure(StreamStallError(90, provider="llama_cpp"))

    assert "unexpected provider error" not in copy
    assert "no reply" in copy.lower() or "stopped sending" in copy.lower()
    assert "Retry" in copy
    assert "stream_stall_timeout_seconds" in copy


def test_plain_provider_unavailable_copy_is_unchanged_without_a_body() -> None:
    assert (
        safe_provider_error_copy("openai", ChatProviderError(status_code=503))
        == "Provider error from OpenAI: provider unavailable. Status: 503."
    )
