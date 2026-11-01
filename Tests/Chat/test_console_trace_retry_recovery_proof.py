"""TASK-33621.6: when a failed retry reservation may be recovered without a duplicate.

The agent retries a call the provider refused (HTTP 429/5xx) in place, as a
TOOL_LOOP call. If that retry's trace reservation fails, the retry never
reached the provider and the provider accepted nothing before it, so Send
without capture (and Capture-On Retry, when nothing was reserved) may run
the turn again. Any run in which the provider answered keeps the original
rule: only the exact live first call's failure is recoverable.
"""

import httpx
import pytest

from Tests.Chat.test_console_trace_runtime import (
    _saved_message,
    _semantic_request,
)
from Tests.Chat.test_console_trace_runtime import (
    make_database as _database_fixture,
)
from Tests.Chat.test_console_trace_runtime import (
    make_gateway as _gateway_fixture,
)
from tldw_chatbook.Chat.Chat_Deps import ChatAPIError, ChatProviderError
from tldw_chatbook.Chat.console_provider_gateway import (
    ConsoleProviderGateway,
    ConsoleProviderResolution,
    ConsoleProviderStreamSignals,
)
from tldw_chatbook.Chat.console_trace_call_recovery import TraceProviderRejection
from tldw_chatbook.Chat.console_trace_errors import (
    TraceCallPersistenceError,
    TraceReservationRefused,
    TraceReservationUnconfirmed,
)
from tldw_chatbook.Chat.console_trace_models import FrozenTracePolicy, new_opaque_id
from tldw_chatbook.Chat.console_trace_provenance import (
    ConsoleRequestRoute,
    ConsoleTraceCaptureMode,
)
from tldw_chatbook.Chat.console_trace_runtime import ConsoleTraceBoundaryFactory

REJECTION = TraceProviderRejection("OpenAI", 429, "rate limit exceeded")
make_database = _database_fixture
make_gateway = _gateway_fixture


@pytest.mark.asyncio
@pytest.mark.bootstrap_profile
@pytest.mark.loopback_network
@pytest.mark.parametrize("provider", ["vllm", "llama_cpp"])
@pytest.mark.parametrize("empty_stream", [True, False])
async def test_physical_fallback_outcomes_bound_retry_recovery(
    tmp_path, make_database, provider, empty_stream
):
    """Real HTTP adapters: an empty 200 is never proof of refused delivery."""
    from Tests.Chat.test_console_refused_stream_retry import _server

    database = make_database(tmp_path / "fallback.sqlite", "fallback")
    conversation = database.add_conversation({"title": "physical fallback"})
    _, revision = _saved_message(database, conversation, "hello")
    policy = FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None)
    real_factory = ConsoleTraceBoundaryFactory(database)
    reservations = []

    def factory(request, resolution, route):
        reservations.append(route)
        if len(reservations) > 2:
            raise ValueError("synthetic construction failure")
        return real_factory(request, resolution, route)

    with _server(
        empty_stream=empty_stream,
        stream_status=429 if empty_stream else 400,
        refuse_all=True,
    ) as server:
        async with httpx.AsyncClient() as client:
            gateway = ConsoleProviderGateway(
                http_client=client, trace_call_boundary_factory=factory
            )
            resolution = ConsoleProviderResolution(
                provider=provider,
                execution_key=provider,
                model="local-model",
                ready=True,
                base_url=server.base_url,
                streaming=True,
                request_retries=0,
            )
            prepared = gateway.prepare_chat_request(
                resolution,
                _semantic_request(
                    [{"role": "user", "content": "hello"}], [revision], policy
                ),
                route=ConsoleRequestRoute.FRESH,
                capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
            )
            owner = object()
            signals = ConsoleProviderStreamSignals()
            gateway._bind_trace_preparation(signals, owner)
            with pytest.raises((ChatAPIError, httpx.HTTPStatusError)):
                async for _ in gateway.stream_chat(
                    resolution,
                    prepared,
                    signals=signals,
                    route=ConsoleRequestRoute.FRESH,
                    capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
                ):
                    pass
            # vLLM may retry a 429 inside its nonstreaming adapter. Every
            # such request is refused; no new request may come from recovery.
            assert server.stream_flags[0] is True
            assert server.stream_flags[1:] and not any(server.stream_flags[1:])
            posts_before_recovery = list(server.stream_flags)
            assert reservations == [
                ConsoleRequestRoute.FRESH, ConsoleRequestRoute.LLAMA_FALLBACK
            ]
            with pytest.raises(TraceCallPersistenceError) as caught:
                gateway._reserve_trace_call(
                    prepared, resolution, ConsoleRequestRoute.TOOL_LOOP, signals=signals
                )
            if empty_stream:
                for mode in (
                    ConsoleTraceCaptureMode.CAPTURE_ON,
                    ConsoleTraceCaptureMode.CAPTURE_OFF,
                ):
                    with pytest.raises(TraceCallPersistenceError) as kept:
                        gateway._verify_trace_preparation_recovery(
                            owner, caught.value, signals, mode
                        )
                    assert type(kept.value) is TraceCallPersistenceError
            else:
                # Unknown reservation: only explicit Capture Off can recover,
                # and only for this owner, once.
                with pytest.raises(TraceCallPersistenceError):
                    gateway._verify_trace_preparation_recovery(
                        object(), caught.value, signals, ConsoleTraceCaptureMode.CAPTURE_OFF
                    )
                with pytest.raises(TraceReservationUnconfirmed):
                    gateway._verify_trace_preparation_recovery(
                        owner, caught.value, signals, ConsoleTraceCaptureMode.CAPTURE_ON
                    )
                gateway._verify_trace_preparation_recovery(
                    owner, caught.value, signals, ConsoleTraceCaptureMode.CAPTURE_OFF
                )
                with pytest.raises(TraceCallPersistenceError):
                    gateway._verify_trace_preparation_recovery(
                        owner, caught.value, signals, ConsoleTraceCaptureMode.CAPTURE_OFF
                    )
            scope = gateway._trace_preparation_scope(signals)
            assert scope.provider_dispatches == 2
            # Exhausted vLLM 429 retries have no response-owned final error;
            # they remain unproven, even though our server observed rejection.
            expected_rejections = (0 if provider == "vllm" else 1) if empty_stream else 2
            assert scope.provider_rejections == expected_rejections
            assert scope.provider_answered is empty_stream
            assert server.stream_flags == posts_before_recovery


def _record_call(gateway, signals, outcome) -> None:
    """One dispatched call and, unless ``outcome`` is ``"unrecorded"``, its result."""
    gateway._note_provider_dispatch(signals)
    if outcome != "unrecorded":
        gateway._note_provider_outcome(signals, outcome)


@pytest.mark.asyncio
@pytest.mark.bootstrap_profile
async def test_real_adapter_timeout_cannot_authorize_refusal_only_replay(
    tmp_path, make_database, make_gateway, monkeypatch
):
    """A synthetic server-error status is not an observed HTTP refusal."""
    import requests

    from tldw_chatbook.Chat.console_trace_call_recovery import provider_contact_for_pause

    requests_seen = []

    def timeout(_session, request, **_kwargs):
        requests_seen.append(request)
        raise requests.exceptions.ReadTimeout("response unknown", request=request)

    monkeypatch.setattr(requests.Session, "send", timeout)
    database = make_database(tmp_path / "timeout.sqlite", "timeout")
    conversation = database.add_conversation({"title": "unknown delivery"})
    _, revision = _saved_message(database, conversation, "hello")
    policy = FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None)
    real_factory = ConsoleTraceBoundaryFactory(database)
    reservations = []

    def factory(request, resolution, route):
        reservations.append(route)
        if len(reservations) > 1:
            raise ValueError("synthetic construction failure")
        return real_factory(request, resolution, route)

    gateway = make_gateway(trace_call_boundary_factory=factory)
    resolution = ConsoleProviderResolution(
        provider="vllm", execution_key="vllm", model="local-model",
        ready=True, base_url="http://127.0.0.1:1/v1", streaming=False,
    )
    prepared = gateway.prepare_chat_request(
        resolution,
        _semantic_request([{"role": "user", "content": "hello"}], [revision], policy),
        route=ConsoleRequestRoute.FRESH,
        capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
    )
    owner = object()
    signals = ConsoleProviderStreamSignals()
    gateway._bind_trace_preparation(signals, owner)
    with pytest.raises(ChatAPIError) as failed:
        async for _ in gateway.stream_chat(
            resolution, prepared, signals=signals, route=ConsoleRequestRoute.FRESH,
            capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
        ):
            pass
    assert requests_seen
    assert failed.value.status_code in {503, 504}
    with pytest.raises(TraceCallPersistenceError) as caught:
        gateway._reserve_trace_call(
            prepared, resolution, ConsoleRequestRoute.TOOL_LOOP, signals=signals
        )
    with pytest.raises(TraceCallPersistenceError) as kept:
        gateway._verify_trace_preparation_recovery(
            owner, caught.value, signals, ConsoleTraceCaptureMode.CAPTURE_OFF
        )
    assert type(kept.value) is TraceCallPersistenceError
    contact = provider_contact_for_pause(gateway, signals, dispatch_started=True)
    assert contact.contacted == "unknown"
    assert contact.rejection is None


@pytest.mark.parametrize("history", ["none", "refused", "timeout", "redirect", "missing"])
def test_final_http_refusal_cannot_hide_unknown_earlier_attempts(history):
    import requests
    from urllib3.response import HTTPResponse
    from urllib3.util.retry import RequestHistory, Retry

    response = requests.Response()
    response.status_code = 400
    attempts = {
        "none": (),
        "refused": (RequestHistory("POST", "/", None, 503, None),),
        "timeout": (RequestHistory("POST", "/", requests.ReadTimeout(), None, None),),
        "redirect": (RequestHistory("POST", "/", None, 307, "/elsewhere"),),
        "missing": (),
    }
    if history != "missing":
        response.raw = HTTPResponse(status=400, retries=Retry(history=attempts[history]))
    gateway = ConsoleProviderGateway()
    signals = ConsoleProviderStreamSignals()
    gateway._bind_trace_preparation(signals, object())
    gateway._note_provider_dispatch(signals)
    gateway._note_provider_failure(signals, "vllm", requests.HTTPError(response=response))
    scope = gateway._trace_preparation_scope(signals)
    assert scope.provider_rejections == (1 if history in {"none", "refused"} else 0)
    assert gateway.trace_provider_contact(signals)[0] is (history in {"none", "refused"})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("history", "failure", "mode", "expected"),
    [
        # The provider refused every call so far: recoverable.
        ("rejected", "unknown", "off", "recovered"),
        ("rejected", "refused", "off", "recovered"),
        ("rejected", "refused", "on", "recovered"),
        # Capture On cannot prove an unknown reservation does not exist.
        ("rejected", "unknown", "on", "unconfirmed"),
        # The provider answered a call: the original first-call-only rule.
        ("answered", "unknown", "off", "refused"),
        ("answered", "refused", "on", "refused"),
        ("rejected_then_answered", "refused", "off", "refused"),
        # A dispatched call with no recorded outcome may have been answered:
        # fail closed (the review's llama.cpp-direct case, which records none).
        ("rejected_then_unrecorded", "refused", "off", "refused"),
        ("rejected_then_unrecorded", "unknown", "off", "refused"),
        # Nothing reached the provider yet: a TOOL_LOOP call is never first.
        ("none", "refused", "off", "refused"),
    ],
)
async def test_a_retry_after_definitive_rejections_is_recoverable(
    tmp_path, make_database, make_gateway, history, failure, mode, expected
):
    database = make_database(tmp_path / "retry.sqlite", "retry")
    conversation = database.add_conversation({"title": "retry recovery"})
    _, revision = _saved_message(database, conversation, "hello")
    policy = FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None)

    def fail(_request, _resolution, _route):
        if failure == "unknown":
            raise ValueError("synthetic construction failure")
        raise TraceReservationRefused("trace_tool_chain_unavailable")

    gateway = make_gateway(trace_call_boundary_factory=fail)
    resolution = ConsoleProviderResolution(
        provider="openai",
        model="test-model",
        ready=True,
        base_url="https://api.openai.com/v1",
        execution_key="openai",
        streaming=False,
    )
    request = gateway.prepare_chat_request(
        resolution,
        _semantic_request([{"role": "user", "content": "hello"}], [revision], policy),
        # As the existing first-call tests do: the route the gateway reserves
        # under (below) is what decides eligibility, not the prepared one.
        route=ConsoleRequestRoute.FRESH,
        capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
    )
    owner = object()
    signals = ConsoleProviderStreamSignals()
    gateway._bind_trace_preparation(signals, owner)
    if history.startswith("rejected"):
        _record_call(gateway, signals, REJECTION)
    if history in {"answered", "rejected_then_answered"}:
        _record_call(gateway, signals, None)
    if history == "rejected_then_unrecorded":
        _record_call(gateway, signals, "unrecorded")
    with pytest.raises(TraceCallPersistenceError) as caught:
        gateway._reserve_trace_call(
            request, resolution, ConsoleRequestRoute.TOOL_LOOP, signals=signals
        )
    capture = (
        ConsoleTraceCaptureMode.CAPTURE_ON
        if mode == "on"
        else ConsoleTraceCaptureMode.CAPTURE_OFF
    )
    if expected == "recovered":
        assert (
            gateway._verify_trace_preparation_recovery(
                owner, caught.value, signals, capture
            )
            is None
        )
        # The proof is single-use.
        with pytest.raises(TraceCallPersistenceError):
            gateway._verify_trace_preparation_recovery(
                owner, caught.value, signals, capture
            )
    elif expected == "unconfirmed":
        with pytest.raises(TraceReservationUnconfirmed):
            gateway._verify_trace_preparation_recovery(
                owner, caught.value, signals, capture
            )
        # The proof survives for an explicit Capture Off send.
        assert (
            gateway._verify_trace_preparation_recovery(
                owner, caught.value, signals, ConsoleTraceCaptureMode.CAPTURE_OFF
            )
            is None
        )
    else:
        with pytest.raises(TraceCallPersistenceError) as refused:
            gateway._verify_trace_preparation_recovery(
                owner, caught.value, signals, capture
            )
        assert type(refused.value) is TraceCallPersistenceError
    # A later dispatch retires the refusal, including an unknown outcome:
    # the previous 429 is no longer the latest request's verdict.
    contact = gateway.trace_provider_contact(signals)
    assert contact == (
        history != "none",
        REJECTION if history == "rejected" else None,
    )


@pytest.mark.asyncio
async def test_a_llamacpp_direct_answer_after_a_refusal_keeps_the_first_call_rule(
    tmp_path, make_database
):
    """Review blocker: OpenAI 402, then a configured llama.cpp fallback answers.

    The llama.cpp direct path (no native tools) is a separate adapter route
    from the generic one. Its answer must count: the run's next TOOL_LOOP
    reservation failure then keeps the unknown-delivery rule (a replay would
    re-send the answered call and re-run its tools), and the card must not
    say the 402 was the outcome.
    """
    database = make_database(tmp_path / "llama.sqlite", "llama")
    conversation = database.add_conversation({"title": "fallback chain"})
    _, revision = _saved_message(database, conversation, "hello")
    policy = FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None)
    real_factory = ConsoleTraceBoundaryFactory(database)
    reservations: list[object] = []

    def factory(request, resolution, route):
        reservations.append(route)
        if len(reservations) > 2:  # the agent's next tool-loop call
            raise ValueError("synthetic construction failure")
        return real_factory(request, resolution, route)

    def refuse(**_kwargs):
        import requests
        from urllib3.response import HTTPResponse
        from urllib3.util.retry import Retry

        response = requests.Response()
        response.status_code = 402
        response.raw = HTTPResponse(status=402, retries=Retry(total=0))
        response.raise_for_status()

    llama_posts: list[httpx.Request] = []

    def llama(request: httpx.Request) -> httpx.Response:
        llama_posts.append(request)
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "fenced tool call"}}]}
        )

    openai = ConsoleProviderResolution(
        provider="openai",
        model="gpt-test",
        ready=True,
        base_url="https://api.openai.com/v1",
        execution_key="openai",
        api_key="secret",
        streaming=False,
    )
    local = ConsoleProviderResolution(
        provider="llama_cpp",
        model="local-model",
        ready=True,
        base_url="http://localhost:8080",
        execution_key="llama_cpp",
        streaming=False,
    )
    messages = [{"role": "user", "content": "hello"}]
    async with httpx.AsyncClient(transport=httpx.MockTransport(llama)) as client:
        gateway = ConsoleProviderGateway(
            http_client=client,
            chat_api_call_fn=refuse,
            trace_call_boundary_factory=factory,
        )

        def prepare(resolution):
            return gateway.prepare_chat_request(
                resolution,
                _semantic_request(messages, [revision], policy),
                route=ConsoleRequestRoute.FRESH,
                capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
            )

        owner = object()
        signals = ConsoleProviderStreamSignals()
        gateway._bind_trace_preparation(signals, owner)
        with pytest.raises(ChatProviderError):
            async for _ in gateway.stream_chat(
                openai,
                prepare(openai),
                signals=signals,
                route=ConsoleRequestRoute.FRESH,
                capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
            ):
                pass
        refused = gateway.trace_provider_contact(signals)
        assert refused is not None and refused[1] is not None
        assert refused[1].status_code == 402
        answered = [
            item
            async for item in gateway.stream_chat(
                local,
                prepare(local),
                signals=signals,
                route=ConsoleRequestRoute.FRESH,
                capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
            )
        ]
        assert answered == ["fenced tool call"]
        assert len(llama_posts) == 1
        with pytest.raises(TraceCallPersistenceError) as caught:
            gateway._reserve_trace_call(
                prepare(openai),
                openai,
                ConsoleRequestRoute.TOOL_LOOP,
                signals=signals,
            )
        assert caught.value.reservation_status == "unknown"
        for mode in (
            ConsoleTraceCaptureMode.CAPTURE_OFF,
            ConsoleTraceCaptureMode.CAPTURE_ON,
        ):
            with pytest.raises(TraceCallPersistenceError) as kept:
                gateway._verify_trace_preparation_recovery(
                    owner, caught.value, signals, mode
                )
            assert type(kept.value) is TraceCallPersistenceError
        # The llama.cpp answer retired the 402 as the run's last word.
        assert gateway.trace_provider_contact(signals) == (True, None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "refusal"), [(429, True), (504, True), (200, False), (None, False)]
)
async def test_only_an_http_error_status_counts_as_a_refusal(
    make_gateway, status, refusal
):
    """A stream error carried on a 200 means the provider answered (review)."""
    gateway = make_gateway()
    resolution = ConsoleProviderResolution(
        provider="openai",
        model="m",
        ready=True,
        base_url="https://api.openai.com/v1",
        execution_key="openai",
    )
    signals = ConsoleProviderStreamSignals()
    gateway._bind_trace_preparation(signals, object())
    gateway._note_provider_dispatch(signals)
    error = ChatProviderError("boom", provider="openai", status_code=status)
    if status is not None:
        error.response = httpx.Response(status, request=httpx.Request("POST", "http://localhost"))
    gateway._note_provider_failure(signals, resolution.provider, error, direct_http=True)
    contacted, rejection = gateway.trace_provider_contact(signals)
    assert contacted is (status is not None)
    assert (rejection is not None) is refusal
    if refusal:
        assert rejection.status_code == status
