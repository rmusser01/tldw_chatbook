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
    # The wrapper's closing parenthesis ends the copy's last clause (V2-F6).
    assert summary.endswith("(Alt+M: Switch model))"), summary
    assert len(summary) > 500, "the old cut would have dropped the fix"



def test_agent_service_persists_the_whole_404_fix_in_its_error_step(
    tmp_path,
) -> None:
    """Review C-F3: run the real AgentService error path, not the slice by
    hand. Its model call raises the recorded Gemini 404 copy; the run's
    error step -- the summary Console renders as the failure copy -- must
    still end with the Alt+M fix (the old [:500] cut it). The run-log row
    keeps its own shorter, marked truncation; that is storage, not copy."""
    from tldw_chatbook.Agents.agent_models import AgentConfig, RUN_ERROR, STEP_ERROR
    from tldw_chatbook.Agents.agent_service import AgentService
    from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

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
    assert len(copy) > 500, "the recorded copy must be longer than the old cut"

    def failing_model_call(**_kwargs):
        raise RuntimeError(copy)

    db = AgentRunsDB(tmp_path / "runs.db", client_id="test")
    service = AgentService(
        db=db, registry=ToolCatalogRegistry(), chat_call=failing_model_call
    )
    run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "hi"}],
        config=AgentConfig(model="gemini-2.0-flash", system_prompt="s"),
        api_endpoint="google",
    )

    assert outcome.status == RUN_ERROR
    errors = [step for step in outcome.steps if step.kind == STEP_ERROR]
    assert errors, outcome.steps
    summary = errors[-1].summary
    assert summary.endswith("(Alt+M: Switch model))"), summary
    assert db.get_run(run_id)["status"] == RUN_ERROR


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


def test_the_gateways_own_copy_is_not_wrapped_in_a_second_status() -> None:
    """Review round 2 (V2-F6), live g5-v2-gem: the gateway raises its own
    finished copy with the status attached, and the agent step wrapped it
    again -- 'provider returned HTTP 404 (Provider error from Google Gemini
    ... Status: 404. ... (Alt+M: Switch model).).'"""
    from types import SimpleNamespace

    from tldw_chatbook.Agents.agent_models import RUN_ERROR, STEP_ERROR
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController

    raw = _raised_from_http(
        ChatProviderError(
            provider="google",
            message="Error from google (Status 404).",
            status_code=404,
        ),
        404,
        GEMINI_RETIRED_MODEL_404,
    )
    gateway_copy = _provider_error_copy_with_model_recovery(
        safe_provider_error_copy("google", raw),
        model="gemini-2.0-flash",
        status_code=404,
    )
    # The gateway's raise: its copy, the status, and no response.
    raised = ChatProviderError(gateway_copy, provider="google", status_code=404)
    outcome = SimpleNamespace(
        status=RUN_ERROR,
        steps=[SimpleNamespace(kind=STEP_ERROR, summary=describe_stream_failure(raised))],
        final_text="",
    )

    copy = ConsoleChatController._agent_failure_visible_copy(outcome)

    assert "HTTP 404" not in copy, copy
    assert copy.count("404") == 1, copy
    assert ".)" not in copy, copy
    assert copy.endswith("(Alt+M: Switch model)."), copy


def test_a_wrapped_reason_that_ends_a_sentence_keeps_one_period() -> None:
    """V2-F6: a status-carrying error whose own text ends a sentence read
    '... (Something broke.).' once the agent wrapper closed it."""
    from types import SimpleNamespace

    from tldw_chatbook.Agents.agent_models import RUN_ERROR, STEP_ERROR
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController

    reason = describe_stream_failure(
        ChatProviderError("Something broke.", provider="x", status_code=500)
    )
    outcome = SimpleNamespace(
        status=RUN_ERROR,
        steps=[SimpleNamespace(kind=STEP_ERROR, summary=reason)],
        final_text="",
    )

    copy = ConsoleChatController._agent_failure_visible_copy(outcome)

    assert copy == "Agent run failed: provider returned HTTP 500 (Something broke).", copy


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


# --- Review round 2 -----------------------------------------------------------

#: A REAL llama.cpp reply (llama-server 0.5.0 build 11146, Qwen2.5-0.5B,
#: ``-c 8192``), recorded 2026-10-04 with curl for both ``stream: true`` and
#: ``stream: false`` -- the two are byte-identical.
LLAMACPP_CONTEXT_OVERFLOW_400 = (
    '{"error":{"code":400,"message":"request (14030 tokens) exceeds the '
    'available context size (8192 tokens), try increasing it","type":'
    '"exceed_context_size_error","n_prompt_tokens":14030,"n_ctx":8192}}'
)


async def _llamacpp_stream_failure(body: str, status: int = 400):
    """Run the gateway's real llama.cpp stream path against ``body``."""
    import httpx

    from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway

    requests_seen: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests_seen.append(request)
        return httpx.Response(
            status, content=body.encode(), headers={"content-type": "application/json"}
        )

    gateway = ConsoleProviderGateway(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )

    async def admit():
        return gateway._capture_off_admission(None)

    async def admit_fallback(_endpoint, _payload):
        return gateway._capture_off_admission(None)

    with pytest.raises(Exception) as caught:
        _ = [
            item
            async for item in gateway.stream_llamacpp_chat(
                base_url="http://127.0.0.1:9099",
                model="qwen2.5-0.5b-instruct",
                messages=[{"role": "user", "content": "a very long message"}],
                before_adapter=admit,
                before_fallback_adapter=admit_fallback,
            )
        ]
    return caught.value, requests_seen


@pytest.mark.asyncio
async def test_a_llamacpp_context_overflow_is_sent_once() -> None:
    """V2-F3 (live g5-v2-local): the server refused a 6,202-token request for
    its 4,096-token context, and the stream path's non-streaming fallback sent
    the identical request again 0.8 s later. A context overflow is the same
    with or without streaming, so it is not retried."""
    _exc, sent = await _llamacpp_stream_failure(LLAMACPP_CONTEXT_OVERFLOW_400)

    assert len(sent) == 1


@pytest.mark.asyncio
async def test_a_llamacpp_context_overflow_names_the_server_and_the_fix() -> None:
    """V2-F3: the copy read 'Agent run failed: provider returned HTTP 400
    (request (6202 tokens) exceeds ...)' -- no provider, no category, no fix."""
    exc, _sent = await _llamacpp_stream_failure(LLAMACPP_CONTEXT_OVERFLOW_400)

    copy = describe_stream_failure(exc)

    assert copy.startswith("Provider error from llama.cpp:"), copy
    assert "doesn't fit" in copy
    assert (
        "llama.cpp says: “request (14030 tokens) exceeds the available context "
        "size (8192 tokens), try increasing it”"
    ) in copy
    assert "larger context" in copy and "-c" in copy
    assert "context_window" in copy
    assert "HTTP 400" not in copy


@pytest.mark.asyncio
async def test_a_llamacpp_400_that_is_not_an_overflow_still_falls_back() -> None:
    """Control: the fallback exists for servers that refuse streaming with a
    400; that path is unchanged."""
    _exc, sent = await _llamacpp_stream_failure('{"error": "streaming disabled"}')

    assert len(sent) == 2


@pytest.mark.parametrize(
    ("provider", "exc", "joined"),
    [
        (
            "anthropic",
            _raised_from_http(
                ChatAuthenticationError(provider="anthropic", message="Auth failed."),
                401,
                ANTHROPIC_INVALID_KEY_401,
            ),
            "“invalid x-api-key”. Update the API key",
        ),
        (
            "llama_cpp",
            _raised_from_http(
                ChatBadRequestError(
                    provider="llama_cpp", message="Bad request.", status_code=400
                ),
                400,
                LLAMACPP_CONTEXT_OVERFLOW_400,
            ),
            "try increasing it”. Start the server",
        ),
        (
            "openrouter",
            _raised_from_http(
                ChatAuthenticationError(provider="openrouter", message="Auth failed."),
                401,
                OPENROUTER_EXPIRED_KEY_401,
            ),
            "“API key expired.” Update the API key",
        ),
    ],
)
def test_a_reason_without_final_punctuation_still_ends_its_sentence(
    provider, exc, joined
) -> None:
    """Review round 2 live re-check (g5-r2b-local 05): llama.cpp's reason has no
    final period, so the fix ran on: '... try increasing it” Start the server
    ...'. A reason that ends without punctuation gets a period after its
    closing quote; one that already ends a sentence gets none."""
    copy = safe_provider_error_copy(provider, exc)

    assert joined in copy, copy
    assert ".”." not in copy
