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
    make_database as make_database,  # noqa: PLC0414 - pytest fixture re-export
)
from Tests.Chat.test_console_trace_runtime import (
    make_gateway as make_gateway,  # noqa: PLC0414 - pytest fixture re-export
)
from tldw_chatbook.Chat.Chat_Deps import ChatProviderError
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


def _record_call(gateway, signals, outcome) -> None:
    """One dispatched call and, unless ``outcome`` is ``"unrecorded"``, its result."""
    gateway._note_provider_dispatch(signals)
    if outcome != "unrecorded":
        gateway._note_provider_outcome(signals, outcome)


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
    # A later answer retires the refusal: the card must not report an HTTP
    # 429 as the outcome of a run whose retry was answered.
    contact = gateway.trace_provider_contact(signals)
    assert contact == (
        history != "none",
        REJECTION if history.startswith("rejected") and "answered" not in history else None,
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
        raise ChatProviderError(
            "Payment required", provider="openai", status_code=402
        )

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
    gateway._note_provider_failure(
        signals,
        resolution,
        ChatProviderError("boom", provider="openai", status_code=status),
        status,
    )
    contacted, rejection = gateway.trace_provider_contact(signals)
    assert contacted is True
    assert (rejection is not None) is refusal
    if refusal:
        assert rejection.status_code == status
