"""TASK-33621.6: when a failed retry reservation may be recovered without a duplicate.

The agent retries a call the provider refused (HTTP 429/5xx) in place, as a
TOOL_LOOP call. If that retry's trace reservation fails, the retry never
reached the provider and the provider accepted nothing before it, so Send
without capture (and Capture-On Retry, when nothing was reserved) may run
the turn again. Any run in which the provider answered keeps the original
rule: only the exact live first call's failure is recoverable.
"""

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
from tldw_chatbook.Chat.console_provider_gateway import (
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

REJECTION = TraceProviderRejection("OpenAI", 429, "rate limit exceeded")


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
    if history in {"rejected", "rejected_then_answered"}:
        gateway._note_provider_outcome(signals, REJECTION)
    if history in {"answered", "rejected_then_answered"}:
        gateway._note_provider_outcome(signals, None)
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
    assert contact == (history != "none", REJECTION if history == "rejected" else None)
