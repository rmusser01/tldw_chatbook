"""TASK-34100.2 AC#18: a provider rate limit reads as a rate limit.

Gemini's free tier allows 5 requests a minute per model. Live, a 429 there
reached the Console as a 504 'Network error' or as a 'Trace capture blocked
-- The provider was not contacted' card:

* ``chat_with_google`` retried 429 inside urllib3 and, with
  ``raise_on_status`` left on, an exhausted retry raised ``RetryError`` -- a
  ``RequestException`` -- which the handler reported as 'Network error' 504.
  A per-minute quota cannot clear in urllib3's 1-4 s backoff, so 429 is no
  longer retried there, and an exhausted 5xx retry returns the provider's
  own answer instead of a transport error.
* The Agents loop then retried the failed model request (TASK-25901). Under
  trace capture (the default), that retry is a second call in the run whose
  first call bore no response, and the trace contract refuses to reserve it
  ('trace_tool_chain_unavailable'), which the Console shows as 'Trace capture
  blocked ... The provider was not contacted' -- after the provider WAS
  contacted. A captured run therefore never retries a failed model request;
  the provider's own error ends the run.

The 429 body below is the shape of a real Gemini reply recorded 2026-10-04.
"""

from __future__ import annotations

import json
from unittest.mock import Mock, patch

import pytest
import requests

from tldw_chatbook.Chat.Chat_Deps import ChatProviderError, ChatRateLimitError
from tldw_chatbook.Chat.console_provider_gateway import safe_provider_error_copy

# The handler reads its config table through the participant admission, which
# the per-test redirect refuses; these fake the transport, not the config.
pytestmark = pytest.mark.bootstrap_profile

_GOOGLE_429 = {
    "error": {
        "code": 429,
        "message": (
            "You exceeded your current quota, please check your plan and billing "
            "details. For more information on this error, head to: "
            "https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your "
            "current usage, head to: https://ai.dev/rate-limit. \n* Quota exceeded "
            "for metric: generativelanguage.googleapis.com/generate_content_free_"
            "tier_requests, limit: 5, model: gemini-3.8-flash\nPlease retry in "
            "59.316483662s."
        ),
        "status": "RESOURCE_EXHAUSTED",
        "details": [
            {
                "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                "violations": [
                    {
                        "quotaMetric": "generativelanguage.googleapis.com/"
                        "generate_content_free_tier_requests",
                        "quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier",
                        "quotaDimensions": {
                            "location": "global",
                            "model": "gemini-3.8-flash",
                        },
                        "quotaValue": "5",
                    }
                ],
            },
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "59s"},
        ],
    }
}


def _http_response(status: int, body: dict) -> Mock:
    response = Mock()
    response.status_code = status
    response.headers = {}
    response.text = json.dumps(body)
    response.json = Mock(return_value=body)
    response.close = Mock()

    def raise_for_status() -> None:
        raise requests.exceptions.HTTPError(f"{status} Client Error", response=response)

    response.raise_for_status = raise_for_status
    return response


class _Session:
    def __init__(self, response: Mock) -> None:
        self._response = response
        self.adapters: list[object] = []

    def __enter__(self) -> "_Session":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def mount(self, prefix: str, adapter: object) -> None:
        self.adapters.append(adapter)

    def post(self, *args: object, **kwargs: object) -> Mock:
        return self._response


def _pooled(session: _Session):
    """Dev's ADR-222 seam: the Anthropic, Cohere, Google and Hugging Face
    handlers build their per-thread session with
    ``create_pooled_default_session(mount_retry=..., mount_prefixes=...)``,
    which mounts the retry adapter at creation. The fake records that mount
    the way the real pool would make it (Tests/conftest.py resets the
    per-thread registry between tests)."""
    from requests.adapters import HTTPAdapter

    def factory(*_args: object, mount_retry=None, mount_prefixes=("https://",), **_kw):
        if mount_retry is not None:
            adapter = HTTPAdapter(max_retries=mount_retry)
            for prefix in mount_prefixes:
                session.mount(prefix, adapter)
        return session

    return factory


def _fake_transport(session: _Session):
    from contextlib import ExitStack

    from tldw_chatbook.LLM_Calls import LLM_API_Calls

    stack = ExitStack()
    stack.enter_context(
        patch.object(LLM_API_Calls, "create_default_session", return_value=session)
    )
    stack.enter_context(
        patch.object(LLM_API_Calls, "create_pooled_default_session", _pooled(session))
    )
    return stack


def _call_google(response: Mock) -> tuple[BaseException, _Session]:
    from tldw_chatbook.LLM_Calls import LLM_API_Calls

    session = _Session(response)
    with _fake_transport(session):
        with pytest.raises(Exception) as raised:
            LLM_API_Calls.chat_with_google(
                [{"role": "user", "content": "Say OK"}],
                model="gemini-flash-latest",
                api_key="AIza-test-key",
                streaming=False,
            )
    return raised.value, session


def test_a_google_429_is_a_rate_limit_with_googles_reason() -> None:
    error, _session = _call_google(_http_response(429, _GOOGLE_429))

    assert isinstance(error, ChatRateLimitError)
    reason = error.provider_reason
    assert reason.startswith("You exceeded your current quota")
    assert "5 requests per minute" in reason
    assert "gemini-3.8-flash" in reason
    assert reason.endswith("Retry in 59s.")
    assert "https://" not in reason  # the help links are not the reason
    assert "AIza-test-key" not in str(error)


def test_google_does_not_retry_a_429_inside_the_transport() -> None:
    _error, session = _call_google(_http_response(429, _GOOGLE_429))

    (adapter,) = session.adapters
    retry = adapter.max_retries
    assert 429 not in retry.status_forcelist
    # An exhausted 5xx retry returns the provider's answer, never RetryError
    # (which the handler reported as 'Network error' 504).
    assert retry.raise_on_status is False


def test_the_console_copy_names_the_rate_limit_and_the_providers_reason() -> None:
    error = ChatRateLimitError(
        "Rate limit.",
        provider="google",
        provider_reason="You exceeded your current quota. Retry in 59s.",
    )

    copy = safe_provider_error_copy("google", error)

    assert "rate limit exceeded" in copy
    assert "Status: 429." in copy
    # TASK-34100.5 (dev) quotes the provider's own sentence for every
    # failure; a rate limit's reason rides the same path.
    assert copy.endswith(
        "Google Gemini says: “You exceeded your current quota. Retry in 59s.”"
    )
    assert "Network error" not in copy


def test_a_reason_that_carries_a_credential_hides_the_key_not_the_copy() -> None:
    """TASK-34100.5 (dev): a key-shaped token in the reason is hidden in
    place; the category, the status and the rest of the reason stay."""
    error = ChatRateLimitError(
        "Rate limit.",
        provider="google",
        provider_reason="Quota exceeded for key sk-proj-abcdefghijklmnopqrstuvwxyz0123456789.",
    )

    copy = safe_provider_error_copy("google", error)

    assert copy.startswith(
        "Provider error from Google Gemini: rate limit exceeded. Status: 429."
    )
    assert "sk-proj-abcdef" not in copy
    assert "Quota exceeded for key (key hidden)" in copy


def test_a_captured_agent_run_never_retries_a_failed_model_request() -> None:
    from tldw_chatbook.Agents.agent_models import RunBudget
    from tldw_chatbook.Chat.console_agent_bridge import captured_run_budget
    from tldw_chatbook.Chat.console_trace_provenance import ConsoleTraceCaptureMode

    admitted = RunBudget(max_model_retries=2)

    captured = captured_run_budget(admitted, ConsoleTraceCaptureMode.CAPTURE_ON)
    uncaptured = captured_run_budget(admitted, ConsoleTraceCaptureMode.CAPTURE_OFF)

    assert captured is not None and captured.max_model_retries == 0
    assert uncaptured is admitted  # Capture Off keeps TASK-25901's retries
    assert captured_run_budget(None, ConsoleTraceCaptureMode.CAPTURE_OFF) is None
    from_live = captured_run_budget(None, ConsoleTraceCaptureMode.CAPTURE_ON)
    assert from_live is not None and from_live.max_model_retries == 0


def test_an_unclassified_provider_error_keeps_its_plain_copy() -> None:
    copy = safe_provider_error_copy(
        "google", ChatProviderError("boom", provider="google", status_code=503)
    )

    assert copy == "Provider error from Google Gemini: provider unavailable. Status: 503."


class _PlanReached(Exception):
    pass


@pytest.mark.parametrize(
    ("mode", "retries"), [("CAPTURE_ON", 0), ("CAPTURE_OFF", 2)]
)
def test_run_reply_plans_a_captured_run_without_model_retries(
    mode: str, retries: int
) -> None:
    """The wiring, not just the rule: run_reply hands the plan builder the
    captured budget (mutating the call site back to ``run_budget`` fails)."""
    from unittest.mock import MagicMock

    from tldw_chatbook.Agents.agent_models import RunBudget
    from tldw_chatbook.Chat import console_agent_bridge as bridge_module
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.Chat.console_trace_provenance import ConsoleTraceCaptureMode

    store = MagicMock()
    store.messages_for_session.return_value = []
    identity_store = ConsoleChatStore()
    identity_store.create_session(session_id="s1")
    store.progress_owner_scope = identity_store.progress_owner_scope
    store.register_progress_message_store = (
        identity_store.register_progress_message_store
    )
    bridge = bridge_module.ConsoleAgentBridge(
        agent_runs_db=MagicMock(), store=store, provider_gateway=MagicMock()
    )
    seen: list[RunBudget | None] = []

    def plan(**kwargs: object) -> None:
        seen.append(kwargs["run_budget"])  # type: ignore[arg-type]
        raise _PlanReached

    with patch.object(bridge_module, "build_console_first_request_plan", plan):
        with pytest.raises(_PlanReached):
            bridge.run_reply(
                conversation_id="c1",
                session_id="s1",
                resolution=None,
                assistant_message_id="a1",
                model="gpt-4",
                session_system_prompt="sys",
                agent_messages=[{"role": "user", "content": "hi"}],
                should_cancel=lambda: False,
                run_budget=RunBudget(max_model_retries=2),
                capture_mode=getattr(ConsoleTraceCaptureMode, mode),
            )

    (budget,) = seen
    assert budget is not None and budget.max_model_retries == retries


@pytest.mark.parametrize(
    ("mode", "chain"),
    [("CAPTURE_ON", ()), ("CAPTURE_OFF", ("anthropic", "groq"))],
)
def test_a_captured_run_never_falls_back_to_another_provider(
    mode: str, chain: tuple[str, ...]
) -> None:
    """Review round 1 (F7): with [console] agent_fallback_providers set, a
    429 or 5xx under capture went straight to the ADR-110 fallback call --
    the run's next call after one that bore no response -- and the trace
    contract refused it: the 'Trace capture blocked -- the provider was not
    contacted' card again, hiding the provider's own error. Until the trace
    contract can record such a call (TASK-34100.2.1), a captured run plans no
    fallback chain; Capture Off keeps the configured one."""
    from unittest.mock import MagicMock

    from tldw_chatbook.Agents.agent_models import RunBudget
    from tldw_chatbook.Chat import console_agent_bridge as bridge_module
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.Chat.console_trace_provenance import ConsoleTraceCaptureMode

    store = MagicMock()
    store.messages_for_session.return_value = []
    identity_store = ConsoleChatStore()
    identity_store.create_session(session_id="s1")
    store.progress_owner_scope = identity_store.progress_owner_scope
    store.register_progress_message_store = (
        identity_store.register_progress_message_store
    )
    bridge = bridge_module.ConsoleAgentBridge(
        agent_runs_db=MagicMock(), store=store, provider_gateway=MagicMock()
    )
    seen: list[object] = []

    def plan(**kwargs: object) -> None:
        fallback = kwargs.get("fallback_providers")
        seen.append(
            bridge_module.console_fallback_providers() if fallback is None else fallback
        )
        raise _PlanReached

    with (
        patch.object(bridge_module, "build_console_first_request_plan", plan),
        patch.object(
            bridge_module, "console_fallback_providers", lambda: ("anthropic", "groq")
        ),
    ):
        with pytest.raises(_PlanReached):
            bridge.run_reply(
                conversation_id="c1",
                session_id="s1",
                resolution=None,
                assistant_message_id="a1",
                model="gpt-4",
                session_system_prompt="sys",
                agent_messages=[{"role": "user", "content": "hi"}],
                should_cancel=lambda: False,
                run_budget=RunBudget(max_model_retries=2),
                capture_mode=getattr(ConsoleTraceCaptureMode, mode),
            )

    assert seen == [chain]


def test_an_anthropic_retired_model_keeps_its_404() -> None:
    """Live (g2b-ant/06): a retired Anthropic id answered 404 ('model:
    claude-3-haiku-20240307') but chat_with_anthropic raised every 4xx as a
    400, so readiness never recorded 'model not found' (AC#2: a 404 takes the
    model out of Ready) and the copy said 'bad request'. The real 4xx is
    carried (ChatBadRequestError accepts it, TASK-25902)."""
    from tldw_chatbook.Chat.Chat_Deps import ChatBadRequestError
    from tldw_chatbook.LLM_Calls import LLM_API_Calls

    body = {
        "type": "error",
        "error": {"type": "not_found_error", "message": "model: claude-3-haiku-20240307"},
    }
    session = _Session(_http_response(404, body))
    with _fake_transport(session):
        with pytest.raises(ChatBadRequestError) as raised:
            LLM_API_Calls.chat_with_anthropic(
                [{"role": "user", "content": "Say OK"}],
                model="claude-3-haiku-20240307",
                api_key="sk-ant-test-key",
                streaming=False,
            )

    assert raised.value.status_code == 404
    assert "model or endpoint not found" in safe_provider_error_copy(
        "anthropic", raised.value
    )


def test_an_anthropic_429_carries_anthropics_reason() -> None:
    from tldw_chatbook.LLM_Calls import LLM_API_Calls

    body = {
        "type": "error",
        "error": {
            "type": "rate_limit_error",
            "message": "This request would exceed your organization's rate limit "
            "of 50 requests per minute.",
        },
    }
    session = _Session(_http_response(429, body))
    with _fake_transport(session):
        with pytest.raises(ChatRateLimitError) as raised:
            LLM_API_Calls.chat_with_anthropic(
                [{"role": "user", "content": "Say OK"}],
                model="claude-sonnet-5",
                api_key="sk-ant-test-key",
                streaming=False,
            )

    assert raised.value.provider_reason == (
        "This request would exceed your organization's rate limit of 50 requests "
        "per minute."
    )


def test_a_dispatched_openai_429_carries_openais_reason() -> None:
    """OpenAI (and the other requests-based handlers) reach the shared
    dispatcher's HTTPError mapping, which now passes the reason on."""
    from tldw_chatbook.Chat.Chat_Functions import chat_api_call

    body = {
        "error": {
            "message": "Rate limit reached for gpt-4o on requests per min (RPM): "
            "Limit 3, Used 3, Requested 1. Please try again in 20s.",
            "type": "requests",
            "code": "rate_limit_exceeded",
        }
    }
    with patch("requests.Session.post", return_value=_http_response(429, body)):
        with pytest.raises(ChatRateLimitError) as raised:
            chat_api_call(
                "openai",
                messages_payload=[{"role": "user", "content": "Say OK"}],
                api_key="sk-proj-test-key",
                model="gpt-4o",
                streaming=False,
            )

    assert raised.value.provider_reason.startswith("Rate limit reached for gpt-4o")
    assert raised.value.provider_reason.endswith("Please try again in 20s.")


# --- The real transport (review round 1, F1 / TE-7) -------------------------
#
# The fakes above return the response straight from ``post``, so urllib3's
# Retry never runs. These go through a real ``requests`` session with each
# handler's own mounted HTTPAdapter against a stub on 127.0.0.1 that answers
# every request with one status. A 429 left in ``status_forcelist`` with
# ``raise_on_status`` on makes urllib3 raise ``RetryError`` (a
# ``RequestException``) whatever ``total`` is, which the dispatcher and
# Anthropic's handler reported as 'Network error' 504.

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_OPENAI_429 = {
    "error": {
        "message": "You exceeded your current quota, please check your plan and "
        "billing details.",
        "type": "insufficient_quota",
        "code": "insufficient_quota",
    }
}
_ANTHROPIC_429 = {
    "type": "error",
    "error": {
        "type": "rate_limit_error",
        "message": "This request would exceed your organization's rate limit of "
        "50 requests per minute.",
    },
}
_UNAVAILABLE_503 = {"error": {"message": "The server is overloaded."}}


class _StubServer:
    """Answer every POST with one status and JSON body; count the requests."""

    def __init__(self, status: int, body: dict) -> None:
        stub = self
        self.hits = 0

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - http.server API
                stub.hits += 1
                length = int(self.headers.get("Content-Length") or 0)
                self.rfile.read(length)
                payload = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                # Retry-After 0 keeps any transport retry instant.
                self.send_header("Retry-After", "0")
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args: object) -> None:
                return None

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.base_url = f"http://127.0.0.1:{self._server.server_address[1]}/v1"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> "_StubServer":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()


def _real_transport():
    """The handlers' own sessions and adapters, with two test-only changes:
    one transport retry (so a retry is instant and countable), and the
    adapter a handler mounts for ``https://`` also serves the stub's
    ``http://`` (Anthropic and Google mount only ``https://``). Dev's
    ADR-222 pooled sessions mount their retry through
    ``create_pooled_default_session``; its real pool is used, with
    ``http://`` added to the prefixes."""
    from contextlib import ExitStack

    from tldw_chatbook.LLM_Calls import LLM_API_Calls

    real_session = LLM_API_Calls.create_default_session
    real_pooled = LLM_API_Calls.create_pooled_default_session

    def _pooled_session(*args: object, mount_prefixes=("https://",), **kwargs):
        if "https://" in mount_prefixes and "http://" not in mount_prefixes:
            mount_prefixes = (*mount_prefixes, "http://")
        return real_pooled(*args, mount_prefixes=mount_prefixes, **kwargs)

    def _session(*args: object, **kwargs: object):
        session = real_session(*args, **kwargs)
        mount = session.mount

        def _mirror(prefix: str, adapter: object) -> None:
            mount(prefix, adapter)
            if prefix == "https://":
                mount("http://", adapter)

        session.mount = _mirror
        return session

    stack = ExitStack()
    stack.enter_context(
        patch.object(LLM_API_Calls, "create_default_session", side_effect=_session)
    )
    stack.enter_context(
        patch.object(
            LLM_API_Calls, "create_pooled_default_session", side_effect=_pooled_session
        )
    )
    stack.enter_context(patch.object(LLM_API_Calls, "llm_retry_count", lambda _n: 1))
    return stack


def _dispatch(provider: str, base_url: str, model: str) -> BaseException:
    from tldw_chatbook.Chat.Chat_Functions import chat_api_call

    with _real_transport(), pytest.raises(Exception) as raised:
        chat_api_call(
            provider,
            messages_payload=[{"role": "user", "content": "Say OK"}],
            api_key="sk-test-key-0123456789",
            model=model,
            streaming=False,
            api_base_url=base_url,
        )
    return raised.value


_COHERE_429 = {
    "id": "0d7c5f0e-0000-4000-8000-000000000000",
    "message": "You are using a Trial key, which is limited to 20 API calls / "
    "minute.",
}
_HUGGINGFACE_429 = {
    "error": "Rate limit reached. You reached free usage limit (reset hourly)."
}


@pytest.mark.loopback_network
@pytest.mark.parametrize(
    ("provider", "model", "body", "reason_start"),
    [
        ("openai", "gpt-5.6-terra", _OPENAI_429, "You exceeded your current quota"),
        ("anthropic", "claude-sonnet-5", _ANTHROPIC_429, "This request would exceed"),
        # Review round 2 (R3-4): Cohere's and Hugging Face's adapters and
        # their ``provider_reason`` were otherwise unpinned.
        ("cohere", "command-a-03-2025", _COHERE_429, "You are using a Trial key"),
        (
            "huggingface",
            "meta-llama/Llama-3.1-8B-Instruct",
            _HUGGINGFACE_429,
            "Rate limit reached.",
        ),
    ],
)
def test_a_real_429_through_the_mounted_transport_is_a_rate_limit(
    provider: str, model: str, body: dict, reason_start: str
) -> None:
    with _StubServer(429, body) as stub:
        error = _dispatch(provider, stub.base_url, model)

    assert isinstance(error, ChatRateLimitError), (type(error), str(error))
    assert error.provider_reason.startswith(reason_start)
    assert "Network error" not in str(error)
    # A per-minute or per-day quota cannot clear in urllib3's backoff.
    assert stub.hits == 1


@pytest.mark.loopback_network
@pytest.mark.parametrize(
    ("provider", "model"),
    [
        # OpenAI's non-streaming call mounts no transport retry (dev's
        # TASK-38600; pre-merge review I-1); its 5xx is pinned below.
        ("anthropic", "claude-sonnet-5"),
        ("cohere", "command-a-03-2025"),
        ("huggingface", "meta-llama/Llama-3.1-8B-Instruct"),
    ],
)
def test_an_exhausted_5xx_retry_reports_the_providers_status(
    provider: str, model: str
) -> None:
    with _StubServer(503, _UNAVAILABLE_503) as stub:
        error = _dispatch(provider, stub.base_url, model)

    assert isinstance(error, ChatProviderError), (type(error), str(error))
    assert error.status_code == 503
    assert "Network error" not in str(error)
    assert stub.hits == 2  # a 5xx is still retried once in the transport


@pytest.mark.loopback_network
def test_a_real_google_429_through_the_mounted_transport_is_a_rate_limit() -> None:
    from tldw_chatbook.LLM_Calls import LLM_API_Calls

    with _StubServer(429, _GOOGLE_429) as stub:
        with _real_transport(), pytest.raises(Exception) as raised:
            LLM_API_Calls.chat_with_google(
                [{"role": "user", "content": "Say OK"}],
                model="gemini-flash-latest",
                api_key="AIza-test-key",
                streaming=False,
                api_base_url=stub.base_url,
            )

    assert isinstance(raised.value, ChatRateLimitError), str(raised.value)
    assert raised.value.provider_reason.startswith("You exceeded your current quota")
    assert stub.hits == 1


def test_the_transport_retry_refuses_to_retry_a_429() -> None:
    """Review round 2 (R3-4): the guard is the policy's last line -- a
    caller cannot hand the shared retry a 429 to retry."""
    from tldw_chatbook.LLM_Calls.chat_transport_retry import chat_transport_retry

    with pytest.raises(ValueError, match="429"):
        chat_transport_retry(1, 0.0, statuses=(429,))
    with pytest.raises(ValueError, match="429"):
        chat_transport_retry(1, 0.0, statuses=(503, 429))
    retry = chat_transport_retry(2, 0.5)
    assert 429 not in retry.status_forcelist
    assert retry.raise_on_status is False
    assert retry.respect_retry_after_header is False


# --- OpenAI non-streaming: one POST per request (pre-merge review I-1) -------
#
# The rebase onto dev 4b90eb860b re-mounted ``chat_transport_retry`` on the
# private session OpenAI's non-streaming call posts through, with its
# ``read`` budget left to ``total``. A generation slower than ``api_timeout``
# (Library analysis, the summarizer, evals, the Console's non-streaming
# fallback, ``[chat_defaults] streaming = false``) was then POSTed again up
# to ``api_retries`` more times -- each one billed -- and the exhausted retry
# surfaced as ``requests.ConnectionError`` ("Max retries exceeded"), not as a
# timeout. Dev mounts no retry there (TASK-38600 deleted the dead block).
# These drive the real handler, its real session and the shipped retry count
# against a loopback server that answers only after the client gave up; no
# session, adapter or retry is patched -- only the timeout is shortened.

_OPENAI_SHORT_TIMEOUT_SETTINGS = {
    "openai_api": {"api_timeout": 0.3, "api_retries": 3, "api_retry_delay": 0.0},
    "api_settings": {},
}


class _SlowStubServer:
    """Hold every POST past the client's read timeout; count the POSTs."""

    def __init__(self) -> None:
        stub = self
        self.hits = 0
        self._lock = threading.Lock()
        self._release = threading.Event()

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - http.server API
                with stub._lock:
                    stub.hits += 1
                length = int(self.headers.get("Content-Length") or 0)
                self.rfile.read(length)
                # Released when the test ends; the client timed out long before.
                stub._release.wait(10.0)
                payload = json.dumps({"choices": []}).encode()
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                except OSError:
                    pass  # the client hung up first -- that is the point

            def log_message(self, *args: object) -> None:
                return None

        class _QuietServer(ThreadingHTTPServer):
            def handle_error(self, request: object, client_address: object) -> None:
                return None

        self._server = _QuietServer(("127.0.0.1", 0), _Handler)
        self.base_url = f"http://127.0.0.1:{self._server.server_address[1]}/v1"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> "_SlowStubServer":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._release.set()
        self._server.shutdown()
        self._server.server_close()


def _openai_short_timeout():
    from tldw_chatbook.LLM_Calls import LLM_API_Calls

    return patch.object(
        LLM_API_Calls, "load_settings", return_value=_OPENAI_SHORT_TIMEOUT_SETTINGS
    )


def _dispatch_openai(base_url: str) -> BaseException:
    from tldw_chatbook.Chat.Chat_Functions import chat_api_call

    with _openai_short_timeout(), pytest.raises(Exception) as raised:
        chat_api_call(
            "openai",
            messages_payload=[{"role": "user", "content": "Say OK"}],
            api_key="sk-test-key-0123456789",
            model="gpt-5.6-terra",
            streaming=False,
            api_base_url=base_url,
        )
    return raised.value


@pytest.mark.loopback_network
def test_an_openai_read_timeout_is_posted_once_and_raised_as_a_timeout() -> None:
    from tldw_chatbook.LLM_Calls import LLM_API_Calls

    with _SlowStubServer() as stub:
        with _openai_short_timeout(), pytest.raises(Exception) as raised:
            LLM_API_Calls.chat_with_openai(
                [{"role": "user", "content": "Say OK"}],
                api_key="sk-test-key-0123456789",
                model="gpt-5.6-terra",
                streaming=False,
                api_base_url=stub.base_url,
            )
        hits = stub.hits

    error = raised.value
    assert hits == 1, f"one read timeout was POSTed {hits} times"
    assert isinstance(error, requests.exceptions.ReadTimeout), (type(error), str(error))
    assert not isinstance(error, requests.exceptions.ConnectionError)


@pytest.mark.loopback_network
def test_a_dispatched_openai_read_timeout_is_posted_once_and_reads_as_a_timeout() -> (
    None
):
    with _SlowStubServer() as stub:
        error = _dispatch_openai(stub.base_url)
        hits = stub.hits

    assert hits == 1, f"one read timeout was POSTed {hits} times"
    assert isinstance(error, ChatProviderError), (type(error), str(error))
    assert isinstance(error.__context__, requests.exceptions.ReadTimeout), repr(
        error.__context__
    )
    assert "Read timed out" in str(error)
    assert "Max retries exceeded" not in str(error)
    # The Console row names a timeout (status 504's category).
    assert safe_provider_error_copy("openai", error) == (
        "Provider error from OpenAI: gateway timed out. Status: 504."
    )


@pytest.mark.loopback_network
def test_an_openai_429_with_the_shipped_retry_count_is_one_post_and_a_rate_limit() -> (
    None
):
    """Nothing in the transport is patched: the 429 reaches
    ``raise_for_status`` and the dispatcher reads OpenAI's reason."""
    with _StubServer(429, _OPENAI_429) as stub:
        error = _dispatch_openai(stub.base_url)

    assert isinstance(error, ChatRateLimitError), (type(error), str(error))
    assert error.provider_reason.startswith("You exceeded your current quota")
    assert "Network error" not in str(error)
    assert stub.hits == 1


@pytest.mark.loopback_network
def test_an_openai_5xx_is_posted_once_and_keeps_its_status() -> None:
    """Dev's behaviour: no transport retry on the owned session, so the 503
    reaches ``raise_for_status`` with its own status after one POST."""
    with _StubServer(503, _UNAVAILABLE_503) as stub:
        error = _dispatch_openai(stub.base_url)

    assert isinstance(error, ChatProviderError), (type(error), str(error))
    assert error.status_code == 503
    assert "Network error" not in str(error)
    assert stub.hits == 1


# --- The local OpenAI-compatible family and KoboldCpp (review round 2, R3-1) --
#
# Custom / Custom 2, llama.cpp, vLLM, Ollama, TabbyAPI, Aphrodite, MLX-LM and
# Oobabooga all post through ``_chat_with_openai_compatible_local_server``;
# KoboldCpp has its own native call. They mounted ``Retry(status_forcelist=
# [429, ...])`` with ``raise_on_status`` on, and their own ``except
# requests.RequestException`` turned the resulting RetryError into
# 'Network error making request to ...' 503 before the dispatcher saw it.
# A Custom endpoint pointed at a hosted OpenAI-compatible service is a common
# way to meet a real 429.


def _real_local_transport():
    """The local handlers' own sessions and adapters, with one transport
    retry so a retry is instant and countable."""
    from tldw_chatbook.LLM_Calls import LLM_API_Calls_Local

    return patch.object(LLM_API_Calls_Local, "llm_retry_count", lambda _n: 1)


def _dispatch_local(provider: str, base_url: str, model: str) -> BaseException:
    from tldw_chatbook.Chat.Chat_Functions import chat_api_call

    with _real_local_transport(), pytest.raises(Exception) as raised:
        chat_api_call(
            provider,
            messages_payload=[{"role": "user", "content": "Say OK"}],
            model=model,
            streaming=False,
            api_base_url=base_url,
        )
    return raised.value


_LOCAL_PROVIDERS = [
    ("custom-openai-api", "stub-model"),
    ("llama_cpp", "stub-model"),
    ("vllm", "stub-model"),
    ("koboldcpp", "stub-model"),
]


@pytest.mark.loopback_network
@pytest.mark.parametrize(("provider", "model"), _LOCAL_PROVIDERS)
def test_a_real_local_429_is_a_rate_limit_with_the_servers_reason(
    provider: str, model: str
) -> None:
    with _StubServer(429, _OPENAI_429) as stub:
        error = _dispatch_local(provider, stub.base_url, model)

    assert isinstance(error, ChatRateLimitError), (type(error), str(error))
    assert error.provider_reason.startswith("You exceeded your current quota")
    assert "Network error" not in str(error)
    assert stub.hits == 1  # a 429 is never retried in the transport


@pytest.mark.loopback_network
@pytest.mark.parametrize(("provider", "model"), _LOCAL_PROVIDERS)
def test_an_exhausted_local_5xx_retry_reports_the_servers_status(
    provider: str, model: str
) -> None:
    with _StubServer(503, _UNAVAILABLE_503) as stub:
        error = _dispatch_local(provider, stub.base_url, model)

    assert isinstance(error, ChatProviderError), (type(error), str(error))
    assert error.status_code == 503
    assert "Network error" not in str(error)
    assert stub.hits == 2  # a 5xx is still retried once in the transport


def test_the_local_transport_keeps_its_no_read_retry() -> None:
    """TASK-34100.5 review (B-F1): a read timeout means a self-hosted server
    is still working on the prompt; re-sending makes it start over. The
    shared retry keeps that rule for the local family."""
    from tldw_chatbook.LLM_Calls.chat_transport_retry import chat_transport_retry

    assert chat_transport_retry(3, 1.0, read=0).read == 0
    assert chat_transport_retry(3, 1.0).read is None


def test_a_retry_error_that_still_escapes_a_handler_reads_as_a_rate_limit() -> None:
    """Defence in depth: no shipped chat handler retries a 429 in urllib3
    any more (the tests above pin each family), but a RetryError for an
    exhausted 429 that still reaches the dispatcher -- a handler added later
    with its own ``Retry`` -- is a rate limit, never a 504 'Network error'.
    The handler here is a fake that raises one directly."""
    from tldw_chatbook.Chat import Chat_Functions

    exhausted = requests.exceptions.RetryError(
        "HTTPConnectionPool(host='127.0.0.1', port=9): Max retries exceeded with "
        "url: /v1/chat/completions (Caused by ResponseError('too many 429 error "
        "responses'))"
    )

    def _raise(*_args: object, **_kwargs: object) -> None:
        raise exhausted

    with patch.dict(Chat_Functions.API_CALL_HANDLERS, {"cohere": _raise}):
        with pytest.raises(ChatRateLimitError) as raised:
            Chat_Functions.chat_api_call(
                "cohere",
                messages_payload=[{"role": "user", "content": "Say OK"}],
                api_key="co-test-key",
                model="command-a-03-2025",
                streaming=False,
            )
    assert "Network error" not in str(raised.value)
