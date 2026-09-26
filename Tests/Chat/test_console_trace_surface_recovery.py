"""Only exact live first-call surface refusals authorize a capture retry."""

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
from tldw_chatbook.Chat.console_trace_errors import (
    TraceCallPersistenceError,
    TraceSurfaceChangeRefused,
)
from tldw_chatbook.Chat.console_trace_models import FrozenTracePolicy, new_opaque_id
from tldw_chatbook.Chat.console_trace_provenance import (
    ConsoleRequestRoute,
    ConsoleTraceCaptureMode,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario",
    [
        "first",
        "unknown",
        "foreign_owner",
        "foreign_gateway",
        "forged",
        "rebound",
        "later_call",
        "tool_loop",
    ],
)
async def test_surface_refusal_retry_requires_exact_live_first_call(
    tmp_path, make_database, make_gateway, scenario
):
    database = make_database(tmp_path / "surface.sqlite", "surface")
    conversation = database.add_conversation({"title": "surface recovery"})
    _, revision = _saved_message(database, conversation, "hello")
    policy = FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None)

    def refuse(_request, _resolution, _route):
        if scenario == "unknown":
            raise ValueError("unsupported_surface_change")
        raise TraceSurfaceChangeRefused()

    gateway = make_gateway(trace_call_boundary_factory=refuse)
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
        route=ConsoleRequestRoute.FRESH,
        capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
    )
    owner = object()
    signals = ConsoleProviderStreamSignals()
    gateway._bind_trace_preparation(signals, owner)
    route = (
        ConsoleRequestRoute.TOOL_LOOP
        if scenario == "tool_loop"
        else ConsoleRequestRoute.FRESH
    )
    with pytest.raises(TraceCallPersistenceError) as caught:
        gateway._reserve_trace_call(request, resolution, route, signals=signals)
    failure = caught.value
    assert failure.reservation_status == (
        "unknown" if scenario == "unknown" else "not_established"
    )
    if scenario == "foreign_owner":
        owner = object()
    elif scenario == "foreign_gateway":
        gateway = make_gateway(trace_call_boundary_factory=refuse)
    elif scenario == "forged":
        failure = TraceCallPersistenceError(reservation_status="not_established")
    elif scenario == "rebound":
        gateway._bind_trace_preparation(signals, owner)
    elif scenario == "later_call":
        with pytest.raises(TraceCallPersistenceError):
            gateway._reserve_trace_call(request, resolution, route, signals=signals)
    if scenario == "first":
        assert (
            gateway._verify_trace_preparation_recovery(owner, failure, signals) is None
        )
    with pytest.raises(TraceCallPersistenceError):
        gateway._verify_trace_preparation_recovery(owner, failure, signals)
