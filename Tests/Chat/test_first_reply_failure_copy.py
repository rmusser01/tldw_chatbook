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


def test_a_404_reads_model_not_found_whatever_error_class_carries_it() -> None:
    """Live (fresh profile, Gemini, gemini-2.0-flash, 2026-10-03): the 404
    reached Console as a provider error and read 'provider unavailable.
    Status: 404.' -- an outage -- although Google said the model is retired."""
    exc = _raised_from_http(
        ChatProviderError(
            provider="google",
            message="Error from google (Status 404).",
            status_code=404,
        ),
        404,
        GEMINI_RETIRED_MODEL_404,
    )

    copy = safe_provider_error_copy("google", exc)

    assert "model or endpoint not found" in copy
    assert "provider unavailable" not in copy


def test_the_agent_failure_summary_keeps_the_whole_fix() -> None:
    """Live: the agent step summary was cut at 500 characters, so the 404
    copy ended '...choose another model from the.' and lost its Alt+M fix."""
    from tldw_chatbook.Chat.provider_failures import FAILURE_SUMMARY_MAX_CHARS

    exc = _raised_from_http(
        ChatProviderError(
            provider="google",
            message="Error from google (Status 404).",
            status_code=404,
        ),
        404,
        GEMINI_RETIRED_MODEL_404,
    )
    copy = _provider_error_copy_with_model_recovery(
        safe_provider_error_copy("google", exc),
        model="gemini-2.0-flash",
        status_code=404,
    )
    summary = describe_stream_failure(RuntimeError(copy))

    assert len(summary) <= FAILURE_SUMMARY_MAX_CHARS
    assert "(Alt+M: Switch model)." in summary[:FAILURE_SUMMARY_MAX_CHARS]
    assert len(summary) > 500, "the old cut would have dropped the fix"


def test_an_agent_failure_reason_that_ends_a_sentence_gets_no_second_period() -> None:
    """Live (slow llama.cpp, 2026-10-03): 'Agent run failed: no first token
    after 90 s -- ... or try a smaller model..' -- the wrapper added a period
    to a reason that already ended with one."""
    from types import SimpleNamespace

    from tldw_chatbook.Agents.agent_models import RUN_ERROR, STEP_ERROR
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController

    reason = describe_stream_failure(
        StreamStallError(90, provider="llama_cpp", first_token=True)
    )
    outcome = SimpleNamespace(
        status=RUN_ERROR,
        steps=[SimpleNamespace(kind=STEP_ERROR, summary=reason)],
        final_text="",
    )

    copy = ConsoleChatController._agent_failure_visible_copy(outcome)

    assert copy.endswith("try a smaller model."), copy
    assert not copy.endswith("..")


# --- Review round 1 -----------------------------------------------------------

#: DeepSeek's 401 body shape: the provider echoes a masked key fragment.
DEEPSEEK_MASKED_KEY_401 = json.dumps(
    {
        "error": {
            "message": "Authentication Fails, Your api key: ****abcd is invalid",
            "type": "authentication_error",
            "param": None,
            "code": "invalid_request_error",
        }
    }
)


def test_a_reason_the_sanitizer_rejects_drops_only_the_reason() -> None:
    """F2: the whole copy -- category, status, fix -- used to collapse to
    'Provider request failed.' whenever the provider's sentence tripped the
    credential sanitizer. Only the reason goes; the rest stays."""
    from tldw_chatbook.LLM_Calls.hosted_chat import _raise_http_error

    response = requests.models.Response()
    response.status_code = 401
    response._content = DEEPSEEK_MASKED_KEY_401.encode()
    with pytest.raises(ChatAuthenticationError) as caught:
        _raise_http_error("deepseek", 401, label="DeepSeek", response=response)

    copy = safe_provider_error_copy("deepseek", caught.value)

    assert copy.startswith("Provider error from DeepSeek: authentication failed.")
    assert "Status: 401." in copy
    assert "Update the API key" in copy
    assert "abcd" not in copy and "****" not in copy


@pytest.mark.parametrize(
    "secret",
    [
        "gsk_abcdefghijklmnopqrstuvwxyz0123456789ABCD",
        "hf_abcdefghijklmnopqrstuvwxyz0123456789",
        "xai-abcdefghijklmnopqrstuvwxyz0123456789",
    ],
)
def test_underscore_and_vendor_prefixed_keys_are_hidden(secret: str) -> None:
    """F3: ``gsk_`` (Groq) and ``hf_`` keys passed both scrubbers."""
    from tldw_chatbook.Chat.console_provider_gateway import (
        _sanitized_provider_diagnostic,
    )
    from tldw_chatbook.Chat.provider_error_reason import safe_provider_reason

    shown = _sanitized_provider_diagnostic(
        safe_provider_reason(f"Invalid API Key {secret}")
    )

    assert secret not in shown
    assert secret[6:20] not in shown


def test_the_exact_credential_is_hidden_even_when_it_has_no_key_shape() -> None:
    """F3: the gateway knows the key it sent; a provider echo of it is
    hidden whatever its shape, and the rest of the copy survives."""
    credential = "plainlocaltoken0123456789"
    body = json.dumps({"error": {"message": f"Token {credential} is not allowed"}})
    exc = _raised_from_http(
        ChatAuthenticationError(provider="openai", message="Auth failed."), 401, body
    )

    copy = safe_provider_error_copy("openai", exc, known_credentials=(credential,))

    assert credential not in copy
    assert copy.startswith("Provider error from OpenAI: authentication failed.")
    assert "Update the API key" in copy


def test_a_sensitive_request_attaches_no_provider_reason() -> None:
    """F3: every other error-detail path honours is_sensitive_llm_request()."""
    from tldw_chatbook.LLM_Calls.hosted_chat import _raise_http_error
    from tldw_chatbook.Utils.sensitive_llm_logging import sensitive_llm_request

    response = requests.models.Response()
    response.status_code = 401
    response._content = OPENROUTER_EXPIRED_KEY_401.encode()
    with sensitive_llm_request(), pytest.raises(ChatAuthenticationError) as caught:
        _raise_http_error("openrouter", 401, label="OpenRouter", response=response)

    assert getattr(caught.value, "provider_reason", None) is None
    assert "says:" not in safe_provider_error_copy("openrouter", caught.value)


def test_the_hosted_transport_carries_a_streamed_error_body_to_the_copy(
    monkeypatch,
) -> None:
    """C-F2: the production raise site. A real error response arrives
    streamed (``stream=True``), so the body is read through ``iter_content``
    -- nothing preloaded into ``_content``."""
    import io

    from tldw_chatbook.LLM_Calls import hosted_chat

    class _Session:
        def mount(self, *_args, **_kwargs):
            return None

        def post(self, *_args, **_kwargs):
            response = requests.models.Response()
            response.status_code = 401
            response.raw = io.BytesIO(OPENROUTER_EXPIRED_KEY_401.encode())
            assert response._content is False  # nothing preloaded
            return response

        def close(self):
            return None

    monkeypatch.setattr(hosted_chat, "create_default_session", lambda: _Session())
    config = hosted_chat.HostedHTTPTransportConfig(
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-or-v1-0000000000000000000000000000",
        timeout=12.0,
        retries=0,
        retry_delay=0.0,
        display_name="OpenRouter",
    )

    with pytest.raises(ChatAuthenticationError) as caught:
        hosted_chat.owned_json_post(
            config=config,
            route="chat/completions",
            payload={"model": "openai/gpt-4.1-nano", "messages": []},
            streaming=False,
        )

    copy = safe_provider_error_copy("openrouter", caught.value)
    assert "OpenRouter says: “API key expired.”" in copy


@pytest.mark.asyncio
async def test_the_stream_path_hides_the_sent_key_and_keeps_the_copy() -> None:
    """F3: the gateway threads the credential it resolved into the copy, so an
    echo of it is hidden in place -- the category and fix survive instead of
    the whole line collapsing to 'Provider request failed.'"""
    from tldw_chatbook.Chat.console_provider_gateway import (
        ConsoleProviderGateway,
        ConsoleProviderResolution,
    )

    credential = "plainlocaltoken0123456789"

    def failing_call(**_kwargs):
        error = ChatAuthenticationError(provider="openai", message="Auth failed.")
        error.provider_reason = f"Token {credential} is not allowed"
        raise error

    resolution = ConsoleProviderResolution(
        provider="openai",
        base_url="https://api.openai.com/v1",
        model="gpt-4.1",
        ready=True,
        execution_key="openai",
        api_key=credential,
        streaming=False,
    )
    gateway = ConsoleProviderGateway(chat_api_call_fn=failing_call)

    with pytest.raises(ChatProviderError) as caught:
        _ = [
            item
            async for item in gateway.stream_chat(
                resolution, [{"role": "user", "content": "q"}]
            )
        ]

    copy = str(caught.value)
    assert credential not in copy
    assert "authentication failed" in copy, copy
    assert "(key hidden)" in copy
