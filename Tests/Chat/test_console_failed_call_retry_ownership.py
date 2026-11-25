"""A failed call authorizes an unchanged retry, never arbitrary tool history."""

import functools
import json
from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest

from Tests.Chat.test_console_trace_runtime import (
    _saved_message,
    _semantic_request,
)
from Tests.Chat.test_console_trace_runtime import (
    make_database as _make_database_fixture,
)
from Tests.Chat.test_console_trace_runtime import (
    make_gateway as _make_gateway_fixture,
)
from tldw_chatbook.Chat.Chat_Deps import ChatRateLimitError
from tldw_chatbook.Chat.console_prepared_request import prepare_provider_request
from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderResolution
from tldw_chatbook.Chat.console_trace_errors import TraceCallPersistenceError
from tldw_chatbook.Chat.console_trace_models import FrozenTracePolicy, new_opaque_id
from tldw_chatbook.Chat.console_trace_provenance import (
    ConsoleRequestRoute,
    ConsoleTraceCaptureMode,
    ProviderArtifactTraceProvenance,
    TraceProvenanceSource,
)
from tldw_chatbook.Chat.console_trace_runtime import ConsoleTraceBoundaryFactory

# The gateway reads config: stream_chat -> prepare_chat_request ->
# get_model_capabilities reads [model_catalog] use_models_dev. In the per-test
# sandbox that read raises RecoveryRequired and models_dev_catalog._enabled()
# swallows it, so the file only looked config-free (TASK-42472 review,
# 2026-11-24; the TASK-32628 swallowed-admission trap).
pytestmark = pytest.mark.bootstrap_profile
make_database = _make_database_fixture
make_gateway = _make_gateway_fixture


@pytest.mark.parametrize("wire_style", ["single_preamble", "distinct_roles"])
@pytest.mark.parametrize(
    "scenario",
    [
        "valid",
        "cold",
        "system_valid",
        "updated_system",
        "changed_system",
        "changed_surface",
        "changed_provider",
        "changed_model",
        "changed_endpoint",
        "changed_temperature",
        "changed_response_format",
        "changed_reasoning",
        "changed_streaming",
        "changed_actor",
        "changed_chain",
        "changed_policy",
        "foreign_turn",
        "dispatch_unknown",
        "dispatch_started",
        "settlement_failed",
    ],
)
async def test_failed_call_retry_requires_exact_durable_chain(
    tmp_path, monkeypatch, make_database, make_gateway, scenario, wire_style
):
    database = make_database(tmp_path / "retry.sqlite", "retry")
    conversation = database.add_conversation({"title": "retry"})
    _, revision = _saved_message(database, conversation, "hello")
    policy = FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None)
    actor, chain = new_opaque_id(), new_opaque_id()
    factory = ConsoleTraceBoundaryFactory(database)
    unfinished = scenario in {
        "dispatch_unknown",
        "dispatch_started",
        "settlement_failed",
    }
    if unfinished:
        # Fail the settlement write before it commits. Terminal settlements
        # cannot be rewritten to manufacture an open call: SQLite enforces it.
        monkeypatch.setattr(
            type(factory.service),
            "prepare_settlement_handoff",
            lambda *args, **kwargs: SimpleNamespace(settle=lambda _owner: False),
        )
    calls = []
    errors = []

    def safe_error(_provider, exc):
        errors.append(exc)
        return "fixture provider failure"

    def provider(**kwargs):
        calls.append(kwargs)
        if len(calls) == (2 if scenario == "updated_system" else 1):
            raise ChatRateLimitError("fixture", provider="deepseek")
        return {"choices": [{"message": {"content": "recovered"}}]}

    gateway = make_gateway(
        chat_api_call_fn=provider,
        trace_call_boundary_factory=factory,
        safe_error_copy=safe_error,
    )
    prepare = gateway.prepare_chat_request

    def prepare_request(*args, **kwargs):
        prepared = prepare(*args, **kwargs)
        if wire_style == "distinct_roles":
            prepared = prepare_provider_request(
                prepared.semantic,
                wire_style=wire_style,
                provider=prepared.provider,
                model=prepared.model,
                capacity=prepared.capacity,
            )
        return replace(prepared, response_format=response_format)

    monkeypatch.setattr(gateway, "prepare_chat_request", prepare_request)
    resolution = ConsoleProviderResolution(
        ready=True,
        provider="deepseek",
        execution_key="deepseek",
        model="deepseek-chat",
        base_url="https://api.deepseek.com",
        api_key="fixture",
        streaming=False,
    )
    messages, descriptors = [{"role": "user", "content": "hello"}], [revision]
    if scenario in {"system_valid", "updated_system", "changed_system"}:
        messages.insert(0, {"role": "system", "content": "Original instructions"})
        descriptors.insert(
            0,
            ProviderArtifactTraceProvenance(
                TraceProvenanceSource.RENDERED_SYSTEM, policy
            ),
        )

    response_format = None

    async def send(route):
        request = _semantic_request(
            messages, descriptors, policy, route=route, actor_id=actor, chain_id=chain
        )
        return [
            item
            async for item in gateway.stream_chat(
                resolution,
                request,
                route=route,
                route_actor_id=actor,
                route_chain_id=chain,
                capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
            )
        ]

    if scenario == "updated_system":
        assert await send(ConsoleRequestRoute.AGENT_FIRST) == ["recovered"]
        messages[0] = {"role": "system", "content": "Updated after successful call"}
    with pytest.raises(ChatRateLimitError) as failure:
        await send(
            ConsoleRequestRoute.TOOL_LOOP
            if scenario == "updated_system"
            else ConsoleRequestRoute.AGENT_FIRST
        )
    assert type(failure.value) is ChatRateLimitError
    assert failure.value.provider == "deepseek"
    assert failure.value.status_code == 429, errors
    failed_attempts = 2 if scenario == "updated_system" else 1
    assert len(calls) == failed_attempts
    with database.transaction() as cursor:
        assert cursor.execute(
            "SELECT state FROM console_trace_calls ORDER BY call_sequence DESC LIMIT 1"
        ).fetchone()[0] == ("dispatch_started" if unfinished else "error")
    if scenario == "cold":
        gateway._trace_call_boundary_factory = ConsoleTraceBoundaryFactory(database)
    elif scenario == "changed_provider":
        resolution = replace(resolution, provider="groq", execution_key="groq")
    elif scenario == "changed_model":
        resolution = replace(resolution, model="deepseek-reasoner")
    elif scenario == "changed_endpoint":
        resolution = replace(resolution, base_url="https://other.example/v1")
    elif scenario == "changed_temperature":
        resolution = replace(resolution, temperature=0.25)
    elif scenario == "changed_reasoning":
        resolution = replace(resolution, reasoning_effort="high")
    elif scenario == "changed_streaming":
        resolution = replace(resolution, streaming=True)
    elif scenario == "changed_response_format":
        response_format = {"type": "json_object"}
    elif scenario == "changed_system":
        messages[0] = {"role": "system", "content": "Unobserved changed instructions"}
    elif scenario == "changed_surface":
        messages = messages + [
            {"role": "tool", "content": "unobserved addition", "tool_call_id": "forged"}
        ]
        descriptors = descriptors + [
            ProviderArtifactTraceProvenance(TraceProvenanceSource.TOOL_RESULT, policy)
        ]
    elif scenario == "foreign_turn":
        other = database.add_conversation({"title": "other"})
        _, revision = _saved_message(database, other, "hello")
        descriptors = [revision]
    elif scenario == "changed_actor":
        actor = new_opaque_id()
    elif scenario == "changed_chain":
        chain = new_opaque_id()
    elif scenario == "changed_policy":
        policy = FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None)
    elif scenario == "dispatch_unknown":
        with database.transaction() as cursor:
            cursor.execute(
                "UPDATE console_trace_calls SET state = 'dispatch_unknown', settled_at = dispatch_started_at"
            )
    if scenario in {"valid", "cold", "system_valid", "updated_system"}:
        assert await send(ConsoleRequestRoute.TOOL_LOOP) == ["recovered"]
        assert len(calls) == failed_attempts + 1
    else:
        with pytest.raises(TraceCallPersistenceError):
            await send(ConsoleRequestRoute.TOOL_LOOP)
        assert len(calls) == failed_attempts


@pytest.mark.parametrize(
    "scenario",
    [
        "valid",
        "second_round",
        "fallback_completed",
        "fallback_completed_tool_step",
        "other_chain_fallback",
        "other_chain_answered",
        "other_chain_failure",
        "changed_model",
        "changed_temperature",
        "changed_surface",
    ],
)
async def test_a_retry_after_a_failed_llama_stream_fallback_is_checked_against_the_stream(
    tmp_path, make_database, make_gateway, scenario
):
    """TASK-42472 AC#8: a failed llama.cpp stream is sent once more without
    streaming on its own LLAMA_FALLBACK call. When both fail, the agent's
    unchanged retry of the streamed call is admitted; the fallback never
    widens what a failed call authorizes.

    TASK-42530: when the fallback answered instead, the next call carries its
    reply and tool result, and is admitted; repeating the failed stream
    unchanged is still refused, and another chain's answer authorizes
    nothing here."""

    database = make_database(tmp_path / "llama-retry.sqlite", "llama-retry")
    conversation = database.add_conversation({"title": "retry"})
    _, revision = _saved_message(database, conversation, "hello")
    policy = FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None)
    actor, chain = new_opaque_id(), new_opaque_id()
    # One HTTP status per request, in order; the fallback is a second request.
    script = {
        "fallback_completed": [503, 200],
        "fallback_completed_tool_step": [503, 200],
        "other_chain_fallback": [503, 503, 503, 503],
        "other_chain_answered": [503, 503, 503, 200],
        "other_chain_failure": [503, 503],
        "second_round": [503, 503, 503, 503],
    }.get(scenario, [503, 503])
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        status = script[len(requests) - 1] if len(requests) <= len(script) else 200
        if status != 200:
            return httpx.Response(status, json={"error": {"message": "fixture"}})
        if json.loads(request.content).get("stream"):
            return httpx.Response(
                200,
                text='data: {"choices":[{"delta":{"content":"recovered"}}]}\n\n'
                "data: [DONE]\n\n",
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "recovered"}}]}
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    gateway = make_gateway(
        http_client=client,
        trace_call_boundary_factory=ConsoleTraceBoundaryFactory(database),
    )
    resolution = ConsoleProviderResolution(
        ready=True,
        provider="llama_cpp",
        execution_key="llama_cpp",
        model="qwen3.7-27b",
        base_url="http://localhost:8080",
        # A non-streaming call has no resend: the failed call directly after
        # this run's is another run's own call, never a fallback of it.
        streaming=scenario != "other_chain_failure",
    )
    messages, descriptors = [{"role": "user", "content": "hello"}], [revision]

    async def send(route, actor_id=actor, chain_id=chain):
        request = _semantic_request(
            messages,
            descriptors,
            policy,
            route=route,
            actor_id=actor_id,
            chain_id=chain_id,
        )
        items = [
            item
            async for item in gateway.stream_chat(
                resolution,
                request,
                route=route,
                route_actor_id=actor_id,
                route_chain_id=chain_id,
                capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
            )
        ]
        return "".join(item for item in items if isinstance(item, str))

    if scenario.startswith("fallback_completed"):
        assert await send(ConsoleRequestRoute.AGENT_FIRST) == "recovered"
    else:
        with pytest.raises(httpx.HTTPStatusError):
            await send(ConsoleRequestRoute.AGENT_FIRST)
    if scenario == "second_round":
        with pytest.raises(httpx.HTTPStatusError):
            await send(ConsoleRequestRoute.TOOL_LOOP)
    elif scenario in {"other_chain_fallback", "other_chain_failure"}:
        # Another run's failed call (and resend, when streaming) after this
        # run's.
        with pytest.raises(httpx.HTTPStatusError):
            await send(
                ConsoleRequestRoute.AGENT_FIRST, new_opaque_id(), new_opaque_id()
            )
    elif scenario == "other_chain_answered":
        # Another run's failed stream, whose resend answered.
        assert (
            await send(
                ConsoleRequestRoute.AGENT_FIRST, new_opaque_id(), new_opaque_id()
            )
            == "recovered"
        )
    with database.transaction() as cursor:
        routes = cursor.execute(
            "SELECT c.route_identity, c.state FROM console_trace_events e "
            "JOIN console_trace_calls c ON c.call_id = e.call_id "
            "WHERE e.event_type = 'call_boundary' ORDER BY e.sequence"
        ).fetchall()
    failed = [("agent_first", "error"), ("llama_fallback", "error")]
    answered = [("agent_first", "error"), ("llama_fallback", "complete")]
    assert [tuple(row) for row in routes] == {
        "fallback_completed": answered,
        "fallback_completed_tool_step": answered,
        "second_round": [*failed, ("tool_loop", "error"), ("llama_fallback", "error")],
        "other_chain_fallback": [*failed, *failed],
        "other_chain_answered": [*failed, *answered],
        "other_chain_failure": [("agent_first", "error")] * 2,
    }.get(scenario, failed)
    sent = len(requests)
    if scenario == "changed_model":
        resolution = replace(resolution, model="qwen3.7-8b")
    elif scenario == "changed_temperature":
        resolution = replace(resolution, temperature=0.25)
    elif scenario == "changed_surface":
        messages = messages + [
            {"role": "tool", "content": "unobserved addition", "tool_call_id": "forged"}
        ]
        descriptors = descriptors + [
            ProviderArtifactTraceProvenance(TraceProvenanceSource.TOOL_RESULT, policy)
        ]
    elif scenario in {"fallback_completed_tool_step", "other_chain_answered"}:
        # The resend's reply (a tool call) and the tool's result.
        messages = messages + [
            {"role": "assistant", "content": "recovered"},
            {"role": "user", "content": "Tool result: 42"},
        ]
        descriptors = descriptors + [
            ProviderArtifactTraceProvenance(TraceProvenanceSource.TOOL_CALL, policy),
            ProviderArtifactTraceProvenance(TraceProvenanceSource.TOOL_RESULT, policy),
        ]
    if scenario in {"valid", "second_round", "fallback_completed_tool_step"}:
        assert await send(ConsoleRequestRoute.TOOL_LOOP) == "recovered"
        assert len(requests) == sent + 1
    else:
        with pytest.raises(TraceCallPersistenceError):
            await send(ConsoleRequestRoute.TOOL_LOOP)
        assert len(requests) == sent


def _call_record(**changes):
    from tldw_chatbook.Chat.console_trace_models import (
        TraceCallState,
        TraceReservationProvenance,
    )
    from tldw_chatbook.Chat.console_trace_repository import TraceCallRecord

    values = dict(
        call_id="streamed",
        owner_id="owner",
        segment_id="segment",
        turn_id="turn",
        run_id="run",
        call_sequence=0,
        idempotency_key="key",
        policy_id="policy",
        state=TraceCallState.ERROR,
        surface_node_id="surface",
        request_header_id="header",
        provider_name="llama_cpp",
        model_name="qwen3.7-27b",
        route_identity="agent_first",
        dispatch_started_at="2026-11-24T00:00:00Z",
        response_started_at=None,
        settled_at="2026-11-24T00:00:01Z",
        provider_inactive_at=None,
        outcome="error",
        usage=None,
        integrity_state="complete",
        omission_reason_code=None,
        reservation_provenance=TraceReservationProvenance.CRASH_DURABLE_RESERVED,
        import_reason_code=None,
    )
    values.update(changes)
    return TraceCallRecord(**values)


@pytest.mark.parametrize(
    ("which", "changes"),
    [
        ("none", {}),
        ("fallback", {"route_identity": "tool_loop"}),
        ("streamed", {"state": "complete"}),
        ("fallback", {"state": "complete"}),
        ("fallback", {"outcome": "complete"}),
        ("fallback", {"settled_at": None}),
        ("fallback", {"owner_id": "other"}),
        ("fallback", {"segment_id": "other"}),
        ("fallback", {"turn_id": "other"}),
        ("fallback", {"policy_id": "other"}),
        ("fallback", {"surface_node_id": "other"}),
    ],
    ids=lambda value: value if isinstance(value, str) else "-".join(value) or "valid",
)
def test_only_an_errored_resend_of_an_errored_call_is_its_failed_stream_fallback(
    which, changes
):
    """TASK-42472: the shared predicate the factory's retry check and the
    completed-turn lineage check both use. One differing condition refuses."""
    from tldw_chatbook.Chat.console_trace_continuation import is_failed_stream_fallback
    from tldw_chatbook.Chat.console_trace_models import TraceCallState

    changes = {
        key: TraceCallState(value) if key == "state" else value
        for key, value in changes.items()
    }
    streamed = _call_record(**(changes if which == "streamed" else {}))
    fallback = _call_record(
        **{
            "call_id": "fallback",
            "run_id": "resend",
            "route_identity": "llama_fallback",
            **(changes if which == "fallback" else {}),
        }
    )

    assert is_failed_stream_fallback(streamed, fallback) is (which == "none")


_TOOL_CALL = (
    "```tool_call\n"
    + json.dumps({"name": "calculator", "arguments": {"expression": "6*7"}})
    + "\n```"
)


def _busy():
    from tldw_chatbook.Provider_Contract import ChatProviderError

    return ChatProviderError("fixture busy", provider="deepseek", status_code=503)


def _hosted_replies(app, monkeypatch, replies):
    """Send through deepseek's handler seam; each call takes the next reply.

    The stand-in keeps ``chat_api_call``'s signature (the TASK-41310 trap).
    An exception in ``replies`` is raised, a dict is the whole message; past
    the script, the call answers. Returns each call's ``messages_payload``.
    """
    import functools

    from tldw_chatbook.Chat import Chat_Functions
    from tldw_chatbook.Chat.console_dispatch_checkpoint import (
        ConsoleEgressClass,
        ConsoleResolvedDestination,
    )

    calls = []

    @functools.wraps(Chat_Functions.chat_api_call)
    def handler(*_args, **kwargs):
        reply = replies[len(calls)] if len(calls) < len(replies) else "Fixture reply"
        calls.append(kwargs["messages_payload"])
        if isinstance(reply, Exception):
            raise reply
        if not isinstance(reply, dict):
            reply = {"role": "assistant", "content": reply}
        return {"choices": [{"message": reply}]}

    async def resolve(_selection):
        return ConsoleProviderResolution(
            ready=True,
            provider="deepseek",
            execution_key="deepseek",
            model="deepseek-chat",
            base_url="https://api.deepseek.com",
            api_key="fixture",
            streaming=False,
            resolved_destination=ConsoleResolvedDestination(
                provider="deepseek",
                model="deepseek-chat",
                endpoint_identity="https://api.deepseek.com",
                egress_class=ConsoleEgressClass.PUBLIC_NETWORK,
            ),
        )

    monkeypatch.setattr(app.gateway, "resolve_for_send", resolve)
    monkeypatch.setattr(app.gateway, "_chat_api_call_fn", handler)
    return calls


# Each run: (harness options, hosted script or None, first run's status, lineage).
def _llama_script(*steps):
    """A `first_failure` that answers each llama.cpp request with its step.

    `tool` answers with a tool call (streamed or whole, as asked), `empty`
    with an empty stream, `busy` with a 503.
    """
    seen = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        step = steps[len(seen) - 1]
        if step == "busy":
            return httpx.Response(503, json={"error": {"message": "busy"}})
        if json.loads(request.content).get("stream"):
            text = "" if step == "empty" else _TOOL_CALL
            body = (
                "data: "
                + json.dumps({"choices": [{"delta": {"content": text}}]})
                + "\n\ndata: [DONE]\n\n"
                if text
                else ""
            )
            return httpx.Response(
                200, text=body, headers={"content-type": "text/event-stream"}
            )
        return httpx.Response(
            200, json={"choices": [{"message": {"content": _TOOL_CALL}}]}
        )

    return respond


_FAILED = ("tool_loop", "error")
_RESEND_FAILED = ("llama_fallback", "error")
_RUNS = {
    # Every call failed before any response: the in-place retries ran out.
    "hosted_exhausted": (
        {},
        [_busy(), _busy(), _busy()],
        "failed",
        [("agent_first", "error"), _FAILED, _FAILED],
    ),
    # The first call answered with a tool call; every call after it failed.
    "hosted_tool_then_exhausted": (
        {},
        [_TOOL_CALL, _busy(), _busy(), _busy()],
        "failed",
        [("agent_first", "complete"), _FAILED, _FAILED, _FAILED],
    ),
    # Each stream and its non-streaming resend failed, three rounds.
    "llama_stream_exhausted": (
        {"response_mode": "stream", "failures": 6},
        None,
        "failed",
        [("agent_first", "error"), ("llama_fallback", "error")]
        + [_FAILED, ("llama_fallback", "error")] * 2,
    ),
    # A streamed tool call; then each stream and its resend failed, three
    # rounds (TASK-42534 review): the terminal call is the last resend.
    "llama_stream_tool_then_exhausted": (
        {"response_mode": "stream", "script": ("tool",) + ("busy",) * 6},
        None,
        "failed",
        [("agent_first", "complete")] + [_FAILED, _RESEND_FAILED] * 3,
    ),
    # TASK-42530's shape: the first stream is empty and its resend answers
    # with the tool call; then every stream and resend failed.
    "llama_resend_tool_then_exhausted": (
        {"response_mode": "stream", "script": ("empty", "tool") + ("busy",) * 6},
        None,
        "failed",
        [("agent_first", "error"), ("llama_fallback", "complete")]
        + [_FAILED, _RESEND_FAILED] * 3,
    ),
    # Controls: a run that bore a response still closes its turn as before.
    "hosted_recovered": (
        {},
        [_busy()],
        "completed",
        [("agent_first", "error"), ("tool_loop", "complete")],
    ),
    "hosted_tool_completed": (
        {},
        [_TOOL_CALL],
        "completed",
        [("agent_first", "complete"), ("tool_loop", "complete")],
    ),
}


def _turn_calls(app, turn_id):
    with app.db.transaction() as cursor:
        calls = cursor.execute(
            "SELECT call_id, route_identity, state, outcome, response_started_at, "
            "settled_at FROM console_trace_calls WHERE turn_id = ? "
            "ORDER BY rowid",
            (turn_id,),
        ).fetchall()
        links = cursor.execute(
            "SELECT l.call_id FROM console_trace_response_links l "
            "JOIN console_trace_calls c ON c.call_id = l.call_id "
            "WHERE c.turn_id = ?",
            (turn_id,),
        ).fetchall()
    return [tuple(row) for row in calls], {row[0] for row in links}


async def _first_run(app, monkeypatch, run):
    options, script, status, lineage = _RUNS[run]
    calls = (
        _hosted_replies(app, monkeypatch, script)
        if script is not None
        else app.attempts
    )
    first = await app.controller.submit_draft(
        "Use calculator once" if _TOOL_CALL in (script or ()) else "Hello",
        session_id=app.session.id,
    )
    assert first.accepted
    assert app.controller.run_state.status.value == status, app.reservation_errors
    assert app.reservation_errors == []
    turn_id = app.store.get_message(first.user_message_id).persisted_message_id
    before, links = _turn_calls(app, turn_id)
    assert [(row[1], row[2]) for row in before] == lineage
    return calls, turn_id, before, links


@pytest.mark.usefixtures("shut_down_console_controllers")
@pytest.mark.parametrize("project", [True, False], ids=["project", "no_project"])
@pytest.mark.parametrize("run", list(_RUNS))
async def test_the_message_after_a_captured_run_that_failed_is_sent(
    tmp_path, monkeypatch, run, project
):
    """TASK-42534: a failed call never closes the chat to the next message.

    A Capture-On agent run whose calls failed before any response ends
    `failed`. The next message's trace proves the finished turn from its
    terminal call, and refused one that never started a response, so the
    send ended `blocked` on 'Trace capture could not start' and the provider
    was never asked. That needs a replaced request suffix: project context
    (`AGENTS.md`), or tool history earlier in the run. A failed call adds
    nothing to the run: the next message is sent, and the failed calls are
    never recorded as having borne a response.
    """
    from Tests.Chat.test_console_project_instruction_traces import (
        GUIDANCE,
        _llama_failure,
        _project_console,
    )

    options = dict(_RUNS[run][0])
    script = options.pop("script", None)
    if script is not None:
        options.update(first_failure=_llama_script(*script), failures=len(script))
    elif "failures" in options:
        options.update(first_failure=_llama_failure("http503"))
    async with _project_console(
        tmp_path,
        monkeypatch,
        source_name="AGENTS.md" if project else "NOTES.md",
        **options,
    ) as app:
        calls, turn_id, before, links = await _first_run(app, monkeypatch, run)
        failed = [row for row in before if row[2] == "error"]
        assert all(row[4] is None for row in failed)
        assert not links & {row[0] for row in failed}
        sent = len(calls)

        follow_up = await app.controller.submit_draft(
            "And now?", session_id=app.session.id
        )

        assert follow_up.accepted
        assert app.reservation_errors == []
        assert app.controller.run_state.status.value == "completed"
        assert len(calls) == sent + 1
        reply = app.store.get_message(follow_up.assistant_message_id)
        assert (reply.status, reply.content) == ("complete", "Fixture reply")
        wire = calls[-1] if _RUNS[run][1] else app.http_payloads[-1]["messages"]
        assert any(GUIDANCE in str(row.get("content")) for row in wire) is project
        # AC#2: the failed turn's calls are unchanged; none bore a response.
        assert _turn_calls(app, turn_id) == (before, links)


@pytest.mark.usefixtures("shut_down_console_controllers")
@pytest.mark.parametrize("run", ["hosted_tool_completed", "hosted_tool_then_exhausted"])
async def test_a_failed_call_never_authorizes_the_tool_history_a_turn_closes(
    tmp_path, monkeypatch, run
):
    """TASK-42534: the next message rechecks who answered with the tool call.

    Its trace may now close a turn on a call that failed before any response,
    so it must not take a run's tool history on trust. Here the call that
    answered with the tool call reads as failed: a failed call authorizes an
    unchanged retry, never tool history, so the next message is refused
    before the provider is asked, and the earlier trace is left alone.
    """
    from Tests.Chat.test_console_project_instruction_traces import _project_console
    from tldw_chatbook.Chat.console_trace_models import TraceCallState

    async with _project_console(tmp_path, monkeypatch) as app:
        calls, turn_id, before, links = await _first_run(app, monkeypatch, run)
        sent = len(calls)
        repository = app.factory.repository
        logical = repository.get_call_by_logical_identity

        def answered_by_a_failed_call(cursor, **identity):
            call = logical(cursor, **identity)
            if call is not None and (call.turn_id, call.call_sequence) == (turn_id, 0):
                return replace(call, state=TraceCallState.ERROR, outcome="error")
            return call

        monkeypatch.setattr(
            repository, "get_call_by_logical_identity", answered_by_a_failed_call
        )
        follow_up = await app.controller.submit_draft(
            "And now?", session_id=app.session.id
        )

        assert follow_up.accepted
        assert app.reservation_errors == ["completed_tool_turn_lineage"]
        assert app.controller.run_state.status.value == "blocked"
        assert len(calls) == sent
        assert _turn_calls(app, turn_id) == (before, links)


@pytest.mark.usefixtures("shut_down_console_controllers")
@pytest.mark.parametrize("project", [True, False], ids=["project", "no_project"])
async def test_a_tool_run_interrupted_by_a_restart_and_discarded_still_closes(
    tmp_path, monkeypatch, project
):
    """TASK-42534 review: startup recovery keeps a call's response.

    A tool run whose third call was refused is left with two calls that bore
    a response (`response_started`). After a restart, startup recovery
    settles them as `interrupted`, the user discards the reply, and sends
    again. Those calls bore the tool call that the removed history carries,
    so the tool-history check must accept them; it refused the send.
    """
    from datetime import datetime, timezone

    from Tests.Chat.test_console_project_instruction_traces import _project_console
    from Tests.Chat.test_console_trace_discarded_tool_run import _reload_console
    from tldw_chatbook.Agents import agent_service
    from tldw_chatbook.Chat.console_trace_settlement import (
        ConsoleTraceSettlementCoordinator,
    )

    monkeypatch.setattr(agent_service, "get_model_token_limit", lambda *_a: 100_000)
    monkeypatch.setattr(
        agent_service, "catalog_schema_tokens", lambda *_a, **_k: 10_001
    )
    async with _project_console(
        tmp_path, monkeypatch, tools=2, discover_tools=True
    ) as app:
        app.store.set_session_project_instruction_state(
            app.session.id,
            replace(
                app.session.project_instruction_state,
                project_instructions_enabled=project,
            ),
        )
        original = type(app.factory.service).prepare_current_surface_delta

        def reject_third(instance, *args, **kwargs):
            if len(app.http_payloads) == 2:
                raise ValueError("unsupported_surface_change")
            return original(instance, *args, **kwargs)

        with monkeypatch.context() as fault:
            fault.setattr(
                type(app.factory.service), "prepare_current_surface_delta", reject_third
            )
            first = await app.controller.submit_draft(
                "Find and load the calculator", session_id=app.session.id
            )
        assert first.accepted
        turn_id = app.store.get_message(first.user_message_id).persisted_message_id
        recovered = ConsoleTraceSettlementCoordinator(
            app.factory.repository
        ).recover_open_calls(
            app.db,
            occurred_at=datetime.now(timezone.utc).isoformat(),
            recovery_grace_seconds=0,
        )
        assert [call.state.value for call in recovered] == ["interrupted"] * 2
        before, links = _turn_calls(app, turn_id)
        assert [row[2] for row in before] == ["interrupted", "interrupted"]
        await _reload_console(app)
        assert (await app.controller.discard_dispatch_recovery(app.session.id)).accepted
        sent = len(app.http_payloads)

        follow_up = await app.controller.submit_draft(
            "And now?", session_id=app.session.id
        )

        assert follow_up.accepted
        assert app.reservation_errors == ["unsupported_surface_change"]
        assert app.controller.run_state.status.value == "completed"
        assert len(app.http_payloads) == sent + 1
        assert _turn_calls(app, turn_id) == (before, links)


def _empty_stream(_request):
    return httpx.Response(200, text="", headers={"content-type": "text/event-stream"})


@pytest.mark.usefixtures("shut_down_console_controllers")
@pytest.mark.parametrize("project", [True, False], ids=["project", "no_project"])
@pytest.mark.parametrize("empty", ["first", "every", "every_transformed"])
async def test_a_llamacpp_tool_call_on_the_stream_resend_runs_the_tool(
    tmp_path, monkeypatch, empty, project
):
    """TASK-42530: the resend's tool call runs, and the run and chat go on.

    A llama.cpp stream that returns no content is sent once more without
    streaming, on its own `llama_fallback` call. When that resend answered
    with a tool call, the agent's next call (carrying the tool result) was
    refused at trace reservation: the trace looked for the failed stream as
    the latest call. When every stream is empty, the next call's own resend
    was refused too (`trace_turn_unavailable`): a resend of a tool-loop call
    had no saved user message to own it. A chat-dictionary transform of the
    user's message (`every_transformed`) must not refuse that resend either
    (TASK-42534 review).
    """
    from Tests.Chat.test_console_project_instruction_traces import (
        GUIDANCE,
        _project_console,
    )

    transformed = empty == "every_transformed"
    empty = "every" if transformed else empty
    async with _project_console(
        tmp_path,
        monkeypatch,
        source_name="AGENTS.md" if project else "NOTES.md",
        **(
            {"response_mode": "fallback", "tools": 2}
            if empty == "every"
            else {
                "response_mode": "stream",
                "tools": 1,
                "first_failure": _empty_stream,
                "failures": 1,
            }
        ),
    ) as app:
        if transformed:
            assert app.store.persist_session_if_needed(app.session.id)
            app.controller._chat_dictionary_applier = (
                lambda _conversation, text, _frozen_inputs: text.replace(
                    "calculator", "calculator tool"
                )
            )
        result = await app.controller.submit_draft(
            "Use calculator once", session_id=app.session.id
        )
        assert result.accepted
        assert app.reservation_errors == []
        assert app.controller.run_state.status.value == "completed"
        if transformed:
            assert any(
                row.get("content") == "Use calculator tool once"
                for row in app.http_payloads[0]["messages"]
            )
        turn_id = app.store.get_message(result.user_message_id).persisted_message_id
        calls, _links = _turn_calls(app, turn_id)
        resend = ("llama_fallback", "complete")
        assert [(row[1], row[2]) for row in calls] == (
            [("agent_first", "error"), resend, ("tool_loop", "complete")]
            if empty == "first"
            else [("agent_first", "error"), resend, ("tool_loop", "error"), resend]
        )
        # The tool ran: its result (6*7, or 7*7 when the first stream's
        # payload was counted) went out on the run's last call.
        assert any(
            str(42 if empty == "first" else 49) in str(row.get("content"))
            for row in app.http_payloads[-1]["messages"][1:]
        )
        reply = app.store.get_message(result.assistant_message_id)
        assert (reply.status, reply.content) == ("complete", "Fixture reply")
        sent = len(app.attempts)

        follow_up = await app.controller.submit_draft(
            "And now?", session_id=app.session.id
        )

        assert follow_up.accepted
        assert app.reservation_errors == []
        assert app.controller.run_state.status.value == "completed"
        assert len(app.attempts) == sent + (1 if empty == "first" else 2)
        wire = app.http_payloads[-1]["messages"]
        assert any(GUIDANCE in str(row.get("content")) for row in wire) is project


@pytest.mark.parametrize(
    "latest",
    [
        "failed_tool_loop",
        "answered",
        "failed_first_call",
        "unchanged_after_failed_tool_loop",
        "appended_after_failed_tool_loop",
    ],
)
async def test_a_resend_of_a_tool_loop_call_repeats_its_failed_call(
    tmp_path, make_database, make_gateway, latest
):
    """TASK-42530: a llama.cpp resend of a tool-loop call owns no new history.

    It runs on its own `llama_fallback` call, and takes the tool loop's turn
    from the request, so it is admitted only directly after the failed
    tool-loop call it repeats, on that call's surface, with nothing
    appended (TASK-42534 review).
    """
    from tldw_chatbook.Chat.console_prepared_request import (
        resolve_request_capacity,
    )
    from tldw_chatbook.Chat.console_trace_errors import TraceReservationRefused

    database = make_database(tmp_path / "resend.sqlite", "resend")
    conversation = database.add_conversation({"title": "resend"})
    _, revision = _saved_message(database, conversation, "hello")
    policy = FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None)
    actor, chain = new_opaque_id(), new_opaque_id()
    script = {"failed_tool_loop": [200, 503, 200], "failed_first_call": [503]}.get(
        latest, [200, 503]
    )
    after_failed_tool_loop = latest.endswith("_after_failed_tool_loop")
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        status = script[len(requests) - 1] if len(requests) <= len(script) else 200
        if status != 200:
            return httpx.Response(status, json={"error": {"message": "fixture"}})
        if json.loads(request.content).get("stream"):
            return httpx.Response(
                200,
                text='data: {"choices":[{"delta":{"content":"recovered"}}]}\n\n'
                "data: [DONE]\n\n",
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "recovered"}}]}
        )

    factory = ConsoleTraceBoundaryFactory(database)
    gateway = make_gateway(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        trace_call_boundary_factory=factory,
    )
    resolution = ConsoleProviderResolution(
        ready=True,
        provider="llama_cpp",
        execution_key="llama_cpp",
        model="qwen3.7-27b",
        base_url="http://localhost:8080",
        # A non-streaming first call has no resend of its own.
        streaming=latest != "failed_first_call",
    )
    messages, descriptors = [{"role": "user", "content": "hello"}], [revision]

    async def send(route, resolution=resolution):
        request = _semantic_request(
            messages, descriptors, policy, route=route, actor_id=actor, chain_id=chain
        )
        items = [
            item
            async for item in gateway.stream_chat(
                resolution,
                request,
                route=route,
                route_actor_id=actor,
                route_chain_id=chain,
                capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
            )
        ]
        return "".join(item for item in items if isinstance(item, str))

    if latest == "failed_first_call":
        with pytest.raises(httpx.HTTPStatusError):
            await send(ConsoleRequestRoute.AGENT_FIRST)
    else:
        assert await send(ConsoleRequestRoute.AGENT_FIRST) == "recovered"
    if latest == "failed_tool_loop":
        # The gateway resends the failed streamed tool-loop call itself.
        assert await send(ConsoleRequestRoute.TOOL_LOOP) == "recovered"
        with database.transaction() as cursor:
            routes = cursor.execute(
                "SELECT c.route_identity, c.state FROM console_trace_events e "
                "JOIN console_trace_calls c ON c.call_id = e.call_id "
                "WHERE e.event_type = 'call_boundary' ORDER BY e.sequence"
            ).fetchall()
        assert [tuple(row) for row in routes] == [
            ("agent_first", "complete"),
            ("tool_loop", "error"),
            ("llama_fallback", "complete"),
        ]
        return
    if after_failed_tool_loop:
        # Not streamed, so the gateway sends no resend of its own: the
        # failed tool-loop call stays the segment's latest call.
        with pytest.raises(httpx.HTTPStatusError):
            await send(
                ConsoleRequestRoute.TOOL_LOOP, replace(resolution, streaming=False)
            )
    if latest == "appended_after_failed_tool_loop":
        messages = messages + [{"role": "user", "content": "Tool result: 42"}]
        descriptors = descriptors + [
            ProviderArtifactTraceProvenance(TraceProvenanceSource.TOOL_RESULT, policy)
        ]
    # A resend after another kind of latest call, or that appends history.
    prepared = prepare_provider_request(
        _semantic_request(
            messages,
            descriptors,
            policy,
            route=ConsoleRequestRoute.TOOL_LOOP,
            actor_id=actor,
            chain_id=chain,
        ),
        wire_style="distinct_roles",
        provider="llama_cpp",
        model="qwen3.7-27b",
        capacity=resolve_request_capacity(context_window_tokens=None),
    )
    if latest == "unchanged_after_failed_tool_loop":
        assert factory(prepared, resolution, ConsoleRequestRoute.LLAMA_FALLBACK)
        return
    with pytest.raises(TraceReservationRefused) as refused:
        factory(prepared, resolution, ConsoleRequestRoute.LLAMA_FALLBACK)
    assert refused.value.args[0] == "trace_tool_chain_unavailable"


@pytest.mark.parametrize(
    "forged",
    ["none", "model", "envelope", "missing_envelope", "unstreamed_envelope"],
)
async def test_a_llamacpp_resend_is_compared_with_the_call_it_repeats(
    tmp_path, make_database, make_gateway, monkeypatch, forged
):
    """TASK-42534 review: a resend repeats its failed stream, settings and all.

    A llama.cpp resend runs on its own run, so the retry comparison of its
    request header never ran: nothing proved it repeated the failed call.
    It is compared with the call whose boundary directly precedes its own;
    only the verified streaming flip (in the settings and in the literal
    envelope's `stream`) may differ. Here the stored stream is made to read
    as another model, or as an envelope with another key; or both envelopes
    read as missing; or the stream's envelope reads as not streamed, so the
    pair is no True -> False flip.
    """
    database = make_database(tmp_path / "resend-header.sqlite", "resend-header")
    conversation = database.add_conversation({"title": "resend"})
    _, revision = _saved_message(database, conversation, "hello")
    policy = FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None)
    actor, chain = new_opaque_id(), new_opaque_id()
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(503, json={"error": {"message": "fixture"}})
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "recovered"}}]}
        )

    factory = ConsoleTraceBoundaryFactory(database)
    repository = factory.repository
    header_of, artifact_of = repository.get_request_header, repository.get_artifact

    def forged_header(cursor, header_id):
        header = header_of(cursor, header_id)
        if forged == "model" and header.generation_parameters.get("streaming"):
            return replace(header, model_name="qwen3.7-8b")
        return header

    def forged_artifact(cursor, artifact_id):
        artifact = artifact_of(cursor, artifact_id)
        value = json.loads(artifact.sanitized_bytes)
        streamed = value == {"model": "qwen3.7-27b", "stream": True}
        if (
            forged == "missing_envelope"
            # Only once the resend's header is written: writes read it back.
            and written
            and isinstance(value, dict)
            and set(value) == {"model", "stream"}
        ):
            return None
        if streamed and forged in {"envelope", "unstreamed_envelope"}:
            changed = (
                {**value, "top_k": 5}
                if forged == "envelope"
                else {**value, "stream": False}
            )
            return replace(artifact, sanitized_bytes=json.dumps(changed).encode())
        return artifact

    written = []
    create_header = repository.create_or_reuse_request_header

    def note_written(cursor, **values):
        header = create_header(cursor, **values)
        if requests:  # the resend's header, after the stream was sent
            written.append(header)
        return header

    monkeypatch.setattr(repository, "get_request_header", forged_header)
    monkeypatch.setattr(repository, "get_artifact", forged_artifact)
    monkeypatch.setattr(repository, "create_or_reuse_request_header", note_written)
    gateway = make_gateway(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        trace_call_boundary_factory=factory,
    )
    resolution = ConsoleProviderResolution(
        ready=True,
        provider="llama_cpp",
        execution_key="llama_cpp",
        model="qwen3.7-27b",
        base_url="http://localhost:8080",
        streaming=True,
    )
    request = _semantic_request(
        [{"role": "user", "content": "hello"}],
        [revision],
        policy,
        route=ConsoleRequestRoute.AGENT_FIRST,
        actor_id=actor,
        chain_id=chain,
    )

    async def send():
        items = [
            item
            async for item in gateway.stream_chat(
                resolution,
                request,
                route=ConsoleRequestRoute.AGENT_FIRST,
                route_actor_id=actor,
                route_chain_id=chain,
                capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
            )
        ]
        return "".join(item for item in items if isinstance(item, str))

    if forged == "none":
        assert await send() == "recovered"
        assert len(requests) == 2
    else:
        with pytest.raises(TraceCallPersistenceError):
            await send()
        assert len(requests) == 1


_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
}


@pytest.mark.parametrize(
    "scenario",
    [
        "valid",
        "unchanged_resend",
        "changed_model",
        "changed_temperature",
        "changed_surface",
        "forged_stream_settings",
        "streamed_bore_response",
    ],
)
@pytest.mark.parametrize("path", ["self_hosted", "llama_tools"])
async def test_a_streamed_retry_after_a_failed_in_chain_resend_repeats_the_stream(
    tmp_path, make_database, make_gateway, monkeypatch, path, scenario
):
    """TASK-42532: the retry repeats the streamed call, not its failed resend.

    A self-hosted server's refused stream (TASK-37421), and a llama.cpp stream
    with tools in the request, are sent once more without streaming as the
    run's next TOOL_LOOP call. When that resend failed too, the agent's
    in-place retry streams again; it was compared with the resend
    (`streaming=False`) and refused. It is compared with the streamed call
    the resend repeated: only that True -> False flip is the resend's, and
    any other change of target, settings or surface is still refused.
    """
    import functools

    from tldw_chatbook.Chat import Chat_Functions
    from tldw_chatbook.Provider_Contract import ChatProviderError

    database = make_database(tmp_path / "resend-retry.sqlite", "resend-retry")
    conversation = database.add_conversation({"title": "retry"})
    _, revision = _saved_message(database, conversation, "hello")
    policy = FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None)
    actor, chain = new_opaque_id(), new_opaque_id()
    provider = "vllm" if path == "self_hosted" else "llama_cpp"
    calls = []

    @functools.wraps(Chat_Functions.chat_api_call)
    def handler(*_args, **kwargs):
        calls.append(kwargs.get("streaming"))
        if len(calls) == 1:
            # The server refuses to stream.
            raise ChatProviderError(
                "fixture refusal", provider=provider, status_code=400
            )
        if len(calls) == 2:
            raise ChatProviderError("fixture busy", provider=provider, status_code=503)
        return {"choices": [{"message": {"role": "assistant", "content": "recovered"}}]}

    factory = ConsoleTraceBoundaryFactory(database)
    gateway = make_gateway(
        chat_api_call_fn=handler,
        trace_call_boundary_factory=factory,
    )
    resolution = ConsoleProviderResolution(
        ready=True,
        provider=provider,
        execution_key=provider,
        model="qwen3.7-27b",
        base_url="http://localhost:8080",
        streaming=True,
    )
    tools = [_TOOL_SCHEMA] if path == "llama_tools" else []
    messages, descriptors = [{"role": "user", "content": "hello"}], [revision]

    async def send(route, resolution):
        from tldw_chatbook.Chat.console_prepared_request import build_console_request
        from tldw_chatbook.Chat.console_trace_provenance import (
            request_route_provenance,
        )

        request = build_console_request(
            messages,
            tools=tools,
            message_provenance=tuple(descriptors),
            memory_provenance=(),
            mandatory_provenance=(),
            tool_provenance=tuple(
                ProviderArtifactTraceProvenance(
                    TraceProvenanceSource.TOOL_DEFINITION, policy
                )
                for _ in tools
            ),
            metadata_provenance=(
                request_route_provenance(route, actor_id=actor, chain_id=chain),
            ),
            capture_policy=policy,
            capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
        )
        items = [
            item
            async for item in gateway.stream_chat(
                resolution,
                request,
                route=route,
                route_actor_id=actor,
                route_chain_id=chain,
                capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
            )
        ]
        return "".join(item for item in items if isinstance(item, str))

    with pytest.raises(ChatProviderError) as failure:
        await send(ConsoleRequestRoute.AGENT_FIRST, resolution)
    assert failure.value.status_code == 503
    assert calls == [True, False]
    with database.transaction() as cursor:
        routes = cursor.execute(
            "SELECT c.route_identity, c.state FROM console_trace_events e "
            "JOIN console_trace_calls c ON c.call_id = e.call_id "
            "WHERE e.event_type = 'call_boundary' ORDER BY e.sequence"
        ).fetchall()
    # The resend is the run's own next call, not a llama_fallback run.
    assert [tuple(row) for row in routes] == [
        ("agent_first", "error"),
        ("tool_loop", "error"),
    ]
    retry = resolution
    if scenario == "unchanged_resend":
        retry = replace(resolution, streaming=False)
    elif scenario == "changed_model":
        retry = replace(resolution, model="qwen3.7-8b")
    elif scenario == "changed_temperature":
        retry = replace(resolution, temperature=0.25)
    elif scenario == "forged_stream_settings":
        # The stored stream reads as if it had the retry's temperature: then
        # the failed resend is no longer its verified flip, and the retry
        # is compared with the resend, as before (TASK-42534 review).
        retry = replace(resolution, temperature=0.25)
        with database.transaction() as cursor:
            streamed_header_id = cursor.execute(
                "SELECT request_header_id FROM console_trace_calls "
                "WHERE call_sequence = 0"
            ).fetchone()[0]
        header_of = factory.repository.get_request_header

        def forged(cursor, header_id):
            header = header_of(cursor, header_id)
            if header_id != streamed_header_id:
                return header
            return replace(
                header,
                generation_parameters={**header.generation_parameters, "temp": 0.25},
            )

        monkeypatch.setattr(factory.repository, "get_request_header", forged)
    elif scenario == "streamed_bore_response":
        # The stored stream reads as if a response had started: then its
        # resend was not sent for want of one, and the retry is compared
        # with the resend, as before (TASK-42534 re-review).
        logical = factory.repository.get_call_by_logical_identity

        def bore_response(cursor, **identity):
            call = logical(cursor, **identity)
            if call is not None and call.call_sequence == 0:
                return replace(call, response_started_at=call.dispatch_started_at)
            return call

        monkeypatch.setattr(
            factory.repository, "get_call_by_logical_identity", bore_response
        )
    elif scenario == "changed_surface":
        messages = messages + [
            {"role": "tool", "content": "unobserved addition", "tool_call_id": "forged"}
        ]
        descriptors = descriptors + [
            ProviderArtifactTraceProvenance(TraceProvenanceSource.TOOL_RESULT, policy)
        ]
    if scenario in {"valid", "unchanged_resend"}:
        assert await send(ConsoleRequestRoute.TOOL_LOOP, retry) == "recovered"
        assert calls == [True, False, retry.streaming]
    else:
        with pytest.raises(TraceCallPersistenceError):
            await send(ConsoleRequestRoute.TOOL_LOOP, retry)
        assert calls == [True, False]


@pytest.mark.usefixtures("shut_down_console_controllers")
@pytest.mark.parametrize("failed_resends", [1, 99], ids=["recovered", "exhausted"])
async def test_a_captured_run_retries_a_refused_stream_whose_resend_failed(
    tmp_path, monkeypatch, failed_resends
):
    """TASK-42532: the run retries in place and ends on the provider's error.

    A self-hosted server refuses every stream (HTTP 400); each is sent once
    more without streaming, and that resend fails with a 503. The agent's
    in-place retry streams again. It was refused at trace reservation, so
    the run ended on the trace card instead of the provider's error.
    """
    import functools

    from Tests.Chat.test_console_project_instruction_traces import _project_console
    from tldw_chatbook.Chat import Chat_Functions
    from tldw_chatbook.Chat.console_dispatch_checkpoint import (
        ConsoleEgressClass,
        ConsoleResolvedDestination,
    )
    from tldw_chatbook.Provider_Contract import ChatProviderError

    async with _project_console(tmp_path, monkeypatch) as app:
        calls = []

        @functools.wraps(Chat_Functions.chat_api_call)
        def handler(*_args, **kwargs):
            calls.append(kwargs.get("streaming"))
            if kwargs.get("streaming"):
                raise ChatProviderError(
                    "fixture refusal", provider="vllm", status_code=400
                )
            if calls.count(False) <= failed_resends:
                raise ChatProviderError(
                    "fixture busy", provider="vllm", status_code=503
                )
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

        async def resolve(_selection):
            return ConsoleProviderResolution(
                ready=True,
                provider="vllm",
                execution_key="vllm",
                model="qwen3.7-27b",
                base_url="http://localhost:8000",
                api_key="fixture",
                streaming=True,
                resolved_destination=ConsoleResolvedDestination(
                    provider="vllm",
                    model="qwen3.7-27b",
                    endpoint_identity="http://localhost:8000",
                    egress_class=ConsoleEgressClass.ON_DEVICE,
                ),
            )

        monkeypatch.setattr(app.gateway, "resolve_for_send", resolve)
        monkeypatch.setattr(app.gateway, "_chat_api_call_fn", handler)
        result = await app.controller.submit_draft("Hello", session_id=app.session.id)

        assert result.accepted
        assert app.reservation_errors == []
        assert app.store.dispatch_recovery_for_session(app.session.id) is None
        turn_id = app.store.get_message(result.user_message_id).persisted_message_id
        lineage = [(row[1], row[2]) for row in _turn_calls(app, turn_id)[0]]
        failed_round = [_FAILED, _FAILED]
        if failed_resends == 1:
            assert app.controller.run_state.status.value == "completed"
            assert calls == [True, False, True, False]
            assert lineage == [
                ("agent_first", "error"),
                _FAILED,
                _FAILED,
                ("tool_loop", "complete"),
            ]
        else:
            assert app.controller.run_state.status.value == "failed"
            assert calls == [True, False] * 3
            assert lineage == [("agent_first", "error"), _FAILED, *failed_round * 2]
            copy = app.controller.run_state.visible_copy
            assert "503" in copy and "trace" not in copy.lower(), copy


# TASK-42525: a provider fallback under Capture On starts a new chain.
_GROQ = ConsoleProviderResolution(
    ready=True,
    provider="groq",
    execution_key="groq",
    model="llama-3.3-70b-versatile",
    base_url="https://api.groq.com/openai/v1",
    api_key="fixture",
    streaming=False,
)


def _credit():
    from tldw_chatbook.Provider_Contract import ChatProviderError

    return ChatProviderError(
        "fixture out of credit", provider="deepseek", status_code=402
    )


def _fallback_replies(app, monkeypatch, replies, *, configured=("groq",), chain=None):
    """deepseek falls back to groq; each call takes the next reply.

    As ``_hosted_replies``, but each call is resolved and answered for the
    provider it names. ``configured`` is `[console] agent_fallback_providers`.
    ``chain``, when given, is the provider chain the runtime walks instead:
    a runtime that strays from the configuration. Returns each call's
    ``(provider, messages_payload)``.
    """
    import functools

    from tldw_chatbook.Agents import fallback_chain
    from tldw_chatbook.Chat import console_agent_bridge, provider_readiness
    from tldw_chatbook.Chat.console_dispatch_checkpoint import (
        ConsoleEgressClass,
        ConsoleResolvedDestination,
    )

    payloads = _hosted_replies(app, monkeypatch, replies)
    primary, handler = app.gateway.resolve_for_send, app.gateway._chat_api_call_fn
    calls = []

    @functools.wraps(handler)
    def routed(*args, **kwargs):
        try:
            return handler(*args, **kwargs)
        finally:
            calls.append((kwargs["api_endpoint"], payloads[-1]))

    async def resolve(selection):
        if selection.provider not in {*configured, *(chain or ())}:
            return await primary(selection)
        return replace(
            _GROQ,
            provider=selection.provider,
            execution_key=selection.provider,
            resolved_destination=ConsoleResolvedDestination(
                provider=selection.provider,
                model=_GROQ.model,
                endpoint_identity=_GROQ.base_url,
                egress_class=ConsoleEgressClass.PUBLIC_NETWORK,
            ),
        )

    monkeypatch.setattr(app.gateway, "resolve_for_send", resolve)
    monkeypatch.setattr(app.gateway, "_chat_api_call_fn", routed)
    monkeypatch.setattr(
        console_agent_bridge, "console_fallback_providers", lambda: configured
    )
    monkeypatch.setattr(
        provider_readiness, "configured_provider_is_ready", lambda _provider: True
    )
    if chain is not None:
        resolve_chain = fallback_chain.resolve_fallback_chain
        monkeypatch.setattr(
            fallback_chain,
            "resolve_fallback_chain",
            lambda _configured, primary, ready: resolve_chain(chain, primary, ready),
        )
    return calls


def _chain_calls(app, turn_id):
    """The turn's calls as (route, state, provider, chain, sequence)."""
    with app.db.transaction() as cursor:
        rows = cursor.execute(
            "SELECT route_identity, state, provider_name, run_id, call_sequence "
            "FROM console_trace_calls WHERE turn_id = ? ORDER BY rowid",
            (turn_id,),
        ).fetchall()
        reserved = cursor.execute(
            "SELECT COUNT(*) FROM console_trace_calls WHERE state = 'reserved'"
        ).fetchone()[0]
    actors = {row[3].split(":")[0] for row in rows}
    chains = list(dict.fromkeys(row[3] for row in rows))
    assert len(actors) == 1, rows  # one run
    assert reserved == 0  # AC#4
    return [(row[0], row[1], row[2], chains.index(row[3]), row[4]) for row in rows]


_DS, _GQ = "deepseek", "groq"
_NATIVE_TOOL_CALL = {
    "role": "assistant",
    "content": "",
    "tool_calls": [
        {
            "id": "call_fixture",
            "type": "function",
            "function": {"name": "calculator", "arguments": '{"expression": "6*7"}'},
        }
    ],
}
_FALLBACK_RUNS = {
    # A 402 is credit-terminal: no in-place retry, straight to the fallback.
    "credit": (
        [_credit()],
        [("agent_first", "error", _DS, 0, 0), ("tool_loop", "complete", _GQ, 1, 0)],
    ),
    # The in-place retries ran out first.
    "exhausted": (
        [_busy(), _busy(), _busy()],
        [
            ("agent_first", "error", _DS, 0, 0),
            ("tool_loop", "error", _DS, 0, 1),
            ("tool_loop", "error", _DS, 0, 2),
            ("tool_loop", "complete", _GQ, 1, 0),
        ],
    ),
    # The fallback carries the tool history its run earned.
    "tool_then_credit": (
        [_NATIVE_TOOL_CALL, _credit()],
        [
            ("agent_first", "complete", _DS, 0, 0),
            ("tool_loop", "error", _DS, 0, 1),
            ("tool_loop", "complete", _GQ, 1, 0),
        ],
    ),
    # The fallback answers with a tool call and its chain goes on.
    "credit_then_tool": (
        [_credit(), _NATIVE_TOOL_CALL],
        [
            ("agent_first", "error", _DS, 0, 0),
            ("tool_loop", "complete", _GQ, 1, 0),
            ("tool_loop", "complete", _GQ, 1, 1),
        ],
    ),
}


async def _fallback_run(app, monkeypatch, run, **options):
    script, lineage = {**_FALLBACK_RUNS, **_FAILED_FALLBACK_RUNS}[run]
    calls = _fallback_replies(app, monkeypatch, script, **options)
    result = await app.controller.submit_draft(
        "Use calculator once" if _NATIVE_TOOL_CALL in script else "Hello",
        session_id=app.session.id,
    )
    assert result.accepted
    turn_id = app.store.get_message(result.user_message_id).persisted_message_id
    return calls, result, turn_id, lineage


@pytest.mark.usefixtures("shut_down_console_controllers")
@pytest.mark.parametrize("run", list(_FALLBACK_RUNS))
async def test_a_captured_agent_run_falls_back_to_its_configured_provider(
    tmp_path, monkeypatch, run
):
    """TASK-42525 AC#1, #2, #4 and #5: the fallback runs and is traced.

    With `[console] agent_fallback_providers = ["groq"]`, a run whose
    deepseek calls fail goes on to groq, with Capture Off. With Capture On
    the run planned no fallback (round 24), and before that the fallback was
    refused at dispatch (`trace_tool_chain_unavailable`) and left its call
    `reserved`. The fallback now starts a new chain of the same run.
    """
    from Tests.Chat.test_console_project_instruction_traces import _project_console
    from tldw_chatbook.Chat.console_trace_native_reader import ConsoleTraceNativeReader

    async with _project_console(tmp_path, monkeypatch) as app:
        calls, result, turn_id, lineage = await _fallback_run(app, monkeypatch, run)

        assert app.controller.run_state.status.value == "completed", (
            app.controller.run_state.visible_copy,
            app.reservation_errors,
        )
        assert app.reservation_errors == []
        reply = app.store.get_message(result.assistant_message_id)
        assert (reply.status, reply.content) == ("complete", "Fixture reply")
        assert [provider for provider, _ in calls] == [row[2] for row in lineage]
        # AC#2, #4: the failed calls, then the fallback's new chain, each
        # with the provider it went to; nothing is left reserved.
        assert _chain_calls(app, turn_id) == lineage
        traced = ConsoleTraceNativeReader(app.db).read_calls(turn_id)
        assert [(call.capture.provider, call.capture.status) for call in traced] == [
            (row[2], row[1]) for row in lineage
        ]


# A failed fallback run: the fallback fails too, and the list is spent.
_FAILED_FALLBACK_RUNS = {
    "credit_then_failed": (
        [_credit(), _credit()],
        [("agent_first", "error", _DS, 0, 0), ("tool_loop", "error", _GQ, 1, 0)],
    ),
    "tool_then_credit_then_failed": (
        [_NATIVE_TOOL_CALL, _credit(), _credit()],
        [
            ("agent_first", "complete", _DS, 0, 0),
            ("tool_loop", "error", _DS, 0, 1),
            ("tool_loop", "error", _GQ, 1, 0),
        ],
    ),
    "credit_then_tool_then_failed": (
        [_credit(), _NATIVE_TOOL_CALL, _credit()],
        [
            ("agent_first", "error", _DS, 0, 0),
            ("tool_loop", "complete", _GQ, 1, 0),
            ("tool_loop", "error", _GQ, 1, 1),
        ],
    ),
}


@pytest.mark.usefixtures("shut_down_console_controllers")
@pytest.mark.parametrize("project", [True, False], ids=["project", "no_project"])
@pytest.mark.parametrize(
    "run", ["credit", "tool_then_credit", "credit_then_tool", *_FAILED_FALLBACK_RUNS]
)
async def test_the_message_after_a_captured_fallback_run_is_sent(
    tmp_path, monkeypatch, run, project
):
    """TASK-42525: the next message proves the run across its chains.

    Its trace replaces the finished run's suffix (project context, tool
    history) only on the run's own evidence. A fallback run spans two chains:
    the history was earned in either, and the terminal call is the fallback
    chain's. Completed or failed (TASK-42534), the next message is sent and
    the run's calls are unchanged.
    """
    from Tests.Chat.test_console_project_instruction_traces import (
        GUIDANCE,
        _project_console,
    )

    async with _project_console(
        tmp_path, monkeypatch, source_name="AGENTS.md" if project else "NOTES.md"
    ) as app:
        calls, _result, turn_id, lineage = await _fallback_run(app, monkeypatch, run)
        status = "failed" if run in _FAILED_FALLBACK_RUNS else "completed"
        assert app.controller.run_state.status.value == status
        before = _chain_calls(app, turn_id)
        assert before == lineage

        follow_up = await app.controller.submit_draft(
            "And now?", session_id=app.session.id
        )

        assert follow_up.accepted
        assert app.reservation_errors == []
        assert app.controller.run_state.status.value == "completed"
        reply = app.store.get_message(follow_up.assistant_message_id)
        assert (reply.status, reply.content) == ("complete", "Fixture reply")
        provider, wire = calls[-1]
        assert provider == _DS  # a new run starts on the session's provider
        assert any(GUIDANCE in str(row.get("content")) for row in wire) is project
        assert _chain_calls(app, turn_id) == before


@pytest.mark.usefixtures("shut_down_console_controllers")
@pytest.mark.parametrize(
    ("run", "chain"), [("tool_then_credit", 0), ("credit_then_tool", 1)]
)
async def test_a_fallback_run_never_lends_its_tool_history_a_failed_call(
    tmp_path, monkeypatch, run, chain
):
    """TASK-42525 with TASK-42534's rule: in either chain of a fallback run,
    the call that answered with the tool call must have borne a response.
    Read as failed, the next message is refused before the provider is asked,
    and the trace is left alone."""
    from Tests.Chat.test_console_project_instruction_traces import _project_console
    from tldw_chatbook.Chat.console_trace_models import TraceCallState

    async with _project_console(tmp_path, monkeypatch) as app:
        calls, _result, turn_id, _lineage = await _fallback_run(app, monkeypatch, run)
        before = _chain_calls(app, turn_id)
        sent = len(calls)
        repository = app.factory.repository
        logical = repository.get_call_by_logical_identity
        answered = [row for row in before if row[3] == chain and row[1] == "complete"][
            0
        ]

        def answered_by_a_failed_call(cursor, **identity):
            call = logical(cursor, **identity)
            if (
                call is not None
                and call.turn_id == turn_id
                and call.provider_name == answered[2]
                and call.call_sequence == answered[4]
            ):
                return replace(call, state=TraceCallState.ERROR, outcome="error")
            return call

        monkeypatch.setattr(
            repository, "get_call_by_logical_identity", answered_by_a_failed_call
        )
        follow_up = await app.controller.submit_draft(
            "And now?", session_id=app.session.id
        )

        assert follow_up.accepted
        assert app.reservation_errors == ["completed_tool_turn_lineage"]
        assert app.controller.run_state.status.value == "blocked"
        assert len(calls) == sent
        assert _chain_calls(app, turn_id) == before


@pytest.mark.usefixtures("shut_down_console_controllers")
async def test_an_interrupted_fallback_run_that_was_discarded_still_closes(
    tmp_path, monkeypatch
):
    """TASK-42525 with TASK-42534's restart case: the fallback answered with a
    tool call and the run's next call was refused, so two calls (one per
    chain) bore a response and stayed `response_started`. Startup recovery
    settles them `interrupted`; the user discards the reply and sends again,
    and the next message closes the run across both chains."""
    from datetime import datetime, timezone

    from Tests.Chat.test_console_project_instruction_traces import _project_console
    from Tests.Chat.test_console_trace_discarded_tool_run import _reload_console
    from tldw_chatbook.Chat.console_trace_settlement import (
        ConsoleTraceSettlementCoordinator,
    )

    async with _project_console(tmp_path, monkeypatch) as app:
        calls = _fallback_replies(
            app, monkeypatch, [_NATIVE_TOOL_CALL, _credit(), _NATIVE_TOOL_CALL]
        )
        original = type(app.factory.service).prepare_current_surface_delta

        def reject_fourth(instance, *args, **kwargs):
            if len(calls) == 3:
                raise ValueError("unsupported_surface_change")
            return original(instance, *args, **kwargs)

        with monkeypatch.context() as fault:
            fault.setattr(
                type(app.factory.service),
                "prepare_current_surface_delta",
                reject_fourth,
            )
            first = await app.controller.submit_draft(
                "Use calculator once", session_id=app.session.id
            )
        assert first.accepted
        turn_id = app.store.get_message(first.user_message_id).persisted_message_id
        recovered = ConsoleTraceSettlementCoordinator(
            app.factory.repository
        ).recover_open_calls(
            app.db,
            occurred_at=datetime.now(timezone.utc).isoformat(),
            recovery_grace_seconds=0,
        )
        # Recovery orders by run id, and each chain's id is random.
        assert sorted(call.provider_name for call in recovered) == [_DS, _GQ]
        before = _chain_calls(app, turn_id)
        assert before == [
            ("agent_first", "interrupted", _DS, 0, 0),
            ("tool_loop", "error", _DS, 0, 1),
            ("tool_loop", "interrupted", _GQ, 1, 0),
        ]
        await _reload_console(app)
        assert (await app.controller.discard_dispatch_recovery(app.session.id)).accepted
        app.reservation_errors.clear()

        follow_up = await app.controller.submit_draft(
            "And now?", session_id=app.session.id
        )

        assert follow_up.accepted
        assert app.reservation_errors == []
        assert app.controller.run_state.status.value == "completed"
        assert len(calls) == 4
        assert _chain_calls(app, turn_id) == before


@pytest.mark.usefixtures("shut_down_console_controllers")
@pytest.mark.parametrize(
    ("configured", "chain"),
    [
        # A runtime that strays from the configuration.
        pytest.param(("groq",), ("openai",), id="unconfigured"),
        # The list is spent: a second fallback has nowhere to go.
        pytest.param(("groq",), ("groq", "openai"), id="beyond_the_list"),
        # The list is walked once, in order; it never goes back.
        pytest.param(("groq", "openai"), ("openai", "groq"), id="back_up_the_list"),
    ],
)
async def test_a_captured_run_falls_back_only_along_its_configured_list(
    tmp_path, monkeypatch, configured, chain
):
    """TASK-42525: only the run's configured fallback list starts a chain.

    The adapter mints a new chain for a call to another target only when the
    target is the next unused entry of the list the run was planned with.
    Any other switch is refused before the request is prepared, so no call is
    reserved and the provider is never asked.
    """
    from Tests.Chat.test_console_project_instruction_traces import _project_console

    async with _project_console(tmp_path, monkeypatch) as app:
        calls = _fallback_replies(
            app,
            monkeypatch,
            [_credit()] * len(chain),
            configured=configured,
            chain=chain,
        )
        result = await app.controller.submit_draft("Hello", session_id=app.session.id)

        assert result.accepted
        assert app.controller.run_state.status.value == "failed"
        assert app.reservation_errors == []
        allowed = len(chain) - 1
        assert [provider for provider, _ in calls] == [_DS, *chain[:allowed]]
        turn_id = app.store.get_message(result.user_message_id).persisted_message_id
        assert [(row[2], row[3]) for row in _chain_calls(app, turn_id)] == [
            (_DS, 0),
            *((provider, index + 1) for index, provider in enumerate(chain[:allowed])),
        ]


@pytest.mark.usefixtures("shut_down_console_controllers")
async def test_a_captured_run_passes_over_an_unready_fallback_entry(
    tmp_path, monkeypatch
):
    """TASK-42525 review finding M4: any later unused entry, in list order.

    The runtime skips a configured fallback that is not ready, so the call
    names a later entry. The adapter takes any entry after the last one used,
    not only the next, and the skipped one is spent.
    """
    from Tests.Chat.test_console_project_instruction_traces import _project_console
    from tldw_chatbook.Chat import provider_readiness

    async with _project_console(tmp_path, monkeypatch) as app:
        calls = _fallback_replies(
            app, monkeypatch, [_credit()], configured=("huggingface", _GQ)
        )
        monkeypatch.setattr(
            provider_readiness,
            "configured_provider_is_ready",
            lambda provider: provider != "huggingface",
        )
        result = await app.controller.submit_draft("Hello", session_id=app.session.id)

        assert app.controller.run_state.status.value == "completed"
        assert app.reservation_errors == []
        assert [provider for provider, _ in calls] == [_DS, _GQ]
        turn_id = app.store.get_message(result.user_message_id).persisted_message_id
        assert _chain_calls(app, turn_id) == [
            ("agent_first", "error", _DS, 0, 0),
            ("tool_loop", "complete", _GQ, 1, 0),
        ]


@pytest.mark.parametrize(
    "scenario",
    [
        "valid",
        "new_instructions",
        "same_target",
        "unearned_history",
        "after_response",
        "after_partial_response",
        "other_actor",
        "cross_turn",
        "cross_owner",
    ],
)
async def test_a_fallback_chain_follows_only_its_failed_chain(
    tmp_path, make_database, make_gateway, monkeypatch, scenario
):
    """TASK-42525: a new chain's first call is admitted only as a fallback.

    It must follow the failed call of another chain of its actor, in the
    same owner, turn and policy; that call bore no response; the new call
    goes to another target and appends nothing. Its own rendered system
    row may differ from the failed call's (`new_instructions`, with the
    system row in the message list). Each refusal comes before anything is
    reserved, and the provider is never asked.
    """
    database = make_database(tmp_path / "fallback.sqlite", "fallback")
    conversation = database.add_conversation({"title": "fallback"})
    _, revision = _saved_message(database, conversation, "hello")
    policy = FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None)
    actor = new_opaque_id()
    factory = ConsoleTraceBoundaryFactory(database)
    boundary, refusals = _recording(factory)
    calls = []

    def provider(**kwargs):
        calls.append(kwargs["api_endpoint"])
        if kwargs["api_endpoint"] == "deepseek" and scenario != "after_response":
            raise ChatRateLimitError("fixture", provider="deepseek")
        return {"choices": [{"message": {"content": "answered"}}]}

    gateway = make_gateway(
        chat_api_call_fn=provider,
        trace_call_boundary_factory=boundary,
        safe_error_copy=lambda _provider, _exc: "fixture provider failure",
    )
    deepseek = ConsoleProviderResolution(
        ready=True,
        provider="deepseek",
        execution_key="deepseek",
        model="deepseek-chat",
        base_url="https://api.deepseek.com",
        api_key="fixture",
        streaming=False,
    )
    messages, descriptors = [{"role": "user", "content": "hello"}], [revision]
    if scenario == "new_instructions":
        messages.insert(0, {"role": "system", "content": "Native tool protocol"})
        descriptors.insert(
            0,
            ProviderArtifactTraceProvenance(
                TraceProvenanceSource.RENDERED_SYSTEM, policy
            ),
        )
        prepare = gateway.prepare_chat_request

        def distinct_roles(*args, **kwargs):
            prepared = prepare(*args, **kwargs)
            return prepare_provider_request(
                prepared.semantic,
                wire_style="distinct_roles",
                provider=prepared.provider,
                model=prepared.model,
                capacity=prepared.capacity,
            )

        monkeypatch.setattr(gateway, "prepare_chat_request", distinct_roles)

    async def send(resolution, route, chain):
        request = _semantic_request(
            messages, descriptors, policy, route=route, actor_id=actor, chain_id=chain
        )
        return [
            item
            async for item in gateway.stream_chat(
                resolution,
                request,
                route=route,
                route_actor_id=actor,
                route_chain_id=chain,
                capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
            )
        ]

    if scenario == "after_response":
        assert await send(deepseek, ConsoleRequestRoute.AGENT_FIRST, new_opaque_id())
    else:
        with pytest.raises(ChatRateLimitError):
            await send(deepseek, ConsoleRequestRoute.AGENT_FIRST, new_opaque_id())
    target = _GROQ
    if scenario == "new_instructions":
        messages[0] = {"role": "system", "content": "Text tool protocol"}
    elif scenario == "same_target":
        target = deepseek
    elif scenario == "unearned_history":
        messages = messages + [
            {"role": "tool", "content": "unobserved", "tool_call_id": "forged"}
        ]
        descriptors = descriptors + [
            ProviderArtifactTraceProvenance(TraceProvenanceSource.TOOL_RESULT, policy)
        ]
    elif scenario == "after_partial_response":
        get_call = factory.repository.get_call

        def partly_answered(cursor, call_id):
            call = get_call(cursor, call_id)
            if call is not None and call.provider_name == "deepseek":
                return replace(call, response_started_at=call.dispatch_started_at)
            return call

        monkeypatch.setattr(factory.repository, "get_call", partly_answered)
    elif scenario == "other_actor":
        actor = new_opaque_id()
    elif scenario == "cross_turn":
        _, later = _saved_message(database, conversation, "again")
        messages = messages + [{"role": "user", "content": "again"}]
        descriptors = descriptors + [later]
    elif scenario == "cross_owner":
        other = database.add_conversation({"title": "other"})
        _, revision = _saved_message(database, other, "hello")
        descriptors = [revision]
    if scenario in {"valid", "new_instructions"}:
        assert await send(target, ConsoleRequestRoute.TOOL_LOOP, new_opaque_id()) == [
            "answered"
        ]
        assert calls == ["deepseek", "groq"]
    else:
        with pytest.raises(TraceCallPersistenceError):
            await send(target, ConsoleRequestRoute.TOOL_LOOP, new_opaque_id())
        assert calls == ["deepseek"]
        # History the new call adds is a rewrite the run did not earn; every
        # other scenario is not a fallback at all.
        assert refusals == [
            "trace_fallback_rewrite_unavailable"
            if scenario == "unearned_history"
            else "trace_tool_chain_unavailable"
        ]
    with database.transaction() as cursor:
        assert (
            cursor.execute(
                "SELECT COUNT(*) FROM console_trace_calls WHERE state = 'reserved'"
            ).fetchone()[0]
            == 0
        )


@pytest.mark.usefixtures("shut_down_console_controllers")
@pytest.mark.parametrize("reads_as", ["answered", "unsettled"])
async def test_the_next_message_rechecks_the_fallbacks_link(
    tmp_path, monkeypatch, reads_as
):
    """TASK-42525: the next message follows the run back to its first chain
    and rechecks each link. The call a fallback followed must be a failed
    call that bore no response; read as anything else, the earlier chain's
    tool history is not this run's, and the next message is refused before
    the provider is asked."""
    from Tests.Chat.test_console_project_instruction_traces import _project_console
    from tldw_chatbook.Chat.console_trace_models import TraceCallState

    async with _project_console(tmp_path, monkeypatch) as app:
        calls, _result, turn_id, _lineage = await _fallback_run(
            app, monkeypatch, "tool_then_credit"
        )
        before = _chain_calls(app, turn_id)
        sent = len(calls)
        repository = app.factory.repository
        get_call = repository.get_call

        def misread(cursor, call_id):
            call = get_call(cursor, call_id)
            if call is None or (
                call.turn_id,
                call.provider_name,
                call.call_sequence,
            ) != (
                turn_id,
                _DS,
                1,
            ):
                return call
            if reads_as == "answered":
                return replace(
                    call,
                    state=TraceCallState.COMPLETE,
                    outcome="complete",
                    response_started_at=call.dispatch_started_at,
                )
            return replace(call, state=TraceCallState.DISPATCH_UNKNOWN, outcome=None)

        monkeypatch.setattr(repository, "get_call", misread)
        follow_up = await app.controller.submit_draft(
            "And now?", session_id=app.session.id
        )

        assert follow_up.accepted
        assert app.reservation_errors == ["unsupported_surface_change"]
        assert app.controller.run_state.status.value == "blocked"
        assert len(calls) == sent
        assert _chain_calls(app, turn_id) == before


@pytest.mark.usefixtures("shut_down_console_controllers")
async def test_a_fallback_to_a_text_tool_provider_brings_its_own_instructions(
    tmp_path, monkeypatch
):
    """TASK-42525: a new chain's system prompt is its own.

    deepseek takes tool calls natively; huggingface reads them from text, so
    its calls carry no tool schemas and a system prompt with the text tool
    protocol. Before any tool call the history needs no rewrite. The new
    chain's system prompt and schemas are its own header's, not compared with
    the failed call's as a retry's would be.
    """
    from Tests.Chat.test_console_project_instruction_traces import _project_console

    async with _project_console(tmp_path, monkeypatch) as app:
        calls = _fallback_replies(
            app, monkeypatch, [_credit()], configured=("huggingface",)
        )
        handler, headers = app.gateway._chat_api_call_fn, []

        def header_of(*args, **kwargs):
            headers.append((kwargs["system_message"], "tools" in kwargs))
            return handler(*args, **kwargs)

        monkeypatch.setattr(app.gateway, "_chat_api_call_fn", header_of)
        result = await app.controller.submit_draft("Hello", session_id=app.session.id)

        assert app.controller.run_state.status.value == "completed", (
            app.reservation_errors
        )
        (_, primary), (_, fallback) = calls
        assert primary == fallback  # the same history
        (system, native), (text_system, text_native) = headers
        assert (native, text_native) == (True, False)
        assert "```tool_call" in text_system and system != text_system
        turn_id = app.store.get_message(result.user_message_id).persisted_message_id
        assert _chain_calls(app, turn_id) == [
            ("agent_first", "error", _DS, 0, 0),
            ("tool_loop", "complete", "huggingface", 1, 0),
        ]


def _roles(rows):
    return [row.get("role") for row in rows]


@pytest.mark.usefixtures("shut_down_console_controllers")
@pytest.mark.parametrize("project", [True, False], ids=["project", "no_project"])
async def test_a_fallback_rewrites_the_runs_tool_calls_for_its_provider(
    tmp_path, monkeypatch, project
):
    """TASK-42525, second owner decision: the re-encoded tool history.

    The model wrote its tool call as text; groq takes tool calls natively, so
    ADR-110 rewrites the run's tool rows for it. That rewrite replaced rows
    already on the trace surface and was refused. The fallback chain's first
    call may now record it: the run completes, the trace keeps deepseek's
    calls as sent and records groq's as sent, and the next message is sent.
    """
    from Tests.Chat.test_console_project_instruction_traces import _project_console
    from tldw_chatbook.Chat.console_trace_native_reader import (
        ConsoleTraceNativeReader,
    )

    async with _project_console(
        tmp_path, monkeypatch, source_name="AGENTS.md" if project else "NOTES.md"
    ) as app:
        calls = _fallback_replies(app, monkeypatch, [_TOOL_CALL, _credit()])
        result = await app.controller.submit_draft(
            "Use calculator once", session_id=app.session.id
        )

        assert app.controller.run_state.status.value == "completed"
        assert app.reservation_errors == []
        turn_id = app.store.get_message(result.user_message_id).persisted_message_id
        before = _chain_calls(app, turn_id)
        assert before == [
            ("agent_first", "complete", _DS, 0, 0),
            ("tool_loop", "error", _DS, 0, 1),
            ("tool_loop", "complete", _GQ, 1, 0),
        ]
        (_, _), (_, as_text), (_, native) = calls
        assert "tool" not in _roles(as_text) and "tool" in _roles(native)
        traced = ConsoleTraceNativeReader(app.db).read_calls(turn_id)
        assert [
            _roles(call.capture.request["messages_payload"]) for call in traced[1:]
        ] == [_roles(as_text), _roles(native)]

        follow_up = await app.controller.submit_draft(
            "And now?", session_id=app.session.id
        )

        assert follow_up.accepted
        assert app.reservation_errors == []
        assert app.controller.run_state.status.value == "completed"
        assert _chain_calls(app, turn_id) == before


def _misread_as_failed(monkeypatch, app, turn_id, provider, sequence):
    """Read one of the turn's calls as failed wherever a later call looks it up.

    Patched on the repository's class: a reloaded Console builds a new one.
    """
    from tldw_chatbook.Chat.console_trace_models import TraceCallState

    repository_type = type(app.factory.repository)
    logical = repository_type.get_call_by_logical_identity

    def misread(instance, cursor, **identity):
        call = logical(instance, cursor, **identity)
        if (
            call is not None
            and call.turn_id == turn_id
            and (call.provider_name, call.call_sequence) == (provider, sequence)
        ):
            return replace(call, state=TraceCallState.ERROR, outcome="error")
        return call

    monkeypatch.setattr(repository_type, "get_call_by_logical_identity", misread)


@pytest.mark.usefixtures("shut_down_console_controllers")
@pytest.mark.parametrize("misread", [False, True], ids=["plain", "misread"])
@pytest.mark.parametrize("ending", ["failed", "discarded", "failed_unextended"])
async def test_the_message_after_a_rewritten_run_that_did_not_complete_is_sent(
    tmp_path, monkeypatch, ending, misread
):
    """TASK-42525 review finding I2.

    Without project context, the next message's trace replaces the run's
    tool history from the row after the run's origin. A fallback rewrite left
    those rows dead, replaced further on, so the range began at the
    fallback's rows: a failed or discarded run's next message was refused
    (`surface_replacement_checkpoint_unavailable`). The range now reaches
    down over the run's dead rows and the witness rechecks them, so reading
    the call that answered with the tool call as failed still refuses it.
    `failed_unextended` keeps the old start (the start pin relaxed to admit
    it): the witness still rechecks the rows the rewrite replaced, wherever
    they are.
    """
    from datetime import datetime, timezone

    from Tests.Chat.test_console_project_instruction_traces import _project_console
    from Tests.Chat.test_console_trace_discarded_tool_run import _reload_console
    from tldw_chatbook.Chat import console_trace_fallback_rewrite as rewrite
    from tldw_chatbook.Chat.console_trace_final_values import (
        CompletedToolTurnWitness,
    )
    from tldw_chatbook.Chat.console_trace_settlement import (
        ConsoleTraceSettlementCoordinator,
    )

    async with _project_console(tmp_path, monkeypatch, source_name="NOTES.md") as app:
        if ending == "discarded":
            # The fallback answered with a tool call; the run's next call was
            # refused, a restart settled it, and the user discarded the reply.
            calls = _fallback_replies(
                app, monkeypatch, [_TOOL_CALL, _credit(), _NATIVE_TOOL_CALL]
            )
            original = type(app.factory.service).prepare_current_surface_delta

            def reject_fourth(instance, *args, **kwargs):
                if len(calls) == 3:
                    raise ValueError("unsupported_surface_change")
                return original(instance, *args, **kwargs)

            with monkeypatch.context() as fault:
                fault.setattr(
                    type(app.factory.service),
                    "prepare_current_surface_delta",
                    reject_fourth,
                )
                result = await app.controller.submit_draft(
                    "Use calculator once", session_id=app.session.id
                )
            ConsoleTraceSettlementCoordinator(app.factory.repository).recover_open_calls(
                app.db,
                occurred_at=datetime.now(timezone.utc).isoformat(),
                recovery_grace_seconds=0,
            )
            await _reload_console(app)
            assert (
                await app.controller.discard_dispatch_recovery(app.session.id)
            ).accepted
            app.reservation_errors.clear()
            lineage = [
                ("agent_first", "interrupted", _DS, 0, 0),
                ("tool_loop", "error", _DS, 0, 1),
                ("tool_loop", "interrupted", _GQ, 1, 0),
            ]
        else:
            calls = _fallback_replies(
                app, monkeypatch, [_TOOL_CALL, _credit(), _credit()]
            )
            result = await app.controller.submit_draft(
                "Use calculator once", session_id=app.session.id
            )
            assert app.controller.run_state.status.value == "failed"
            assert app.reservation_errors == []
            lineage = [
                ("agent_first", "complete", _DS, 0, 0),
                ("tool_loop", "error", _DS, 0, 1),
                ("tool_loop", "error", _GQ, 1, 0),
            ]
        turn_id = app.store.get_message(result.user_message_id).persisted_message_id
        before = _chain_calls(app, turn_id)
        assert before == lineage
        if ending == "failed_unextended":
            monkeypatch.setattr(
                rewrite, "extend_over_rewrites", lambda *_args, start, **_kw: start
            )
            monkeypatch.setattr(
                CompletedToolTurnWitness,
                "matches_start_sequence",
                lambda _self, start, origin: start >= origin + 1,
            )
        if misread:
            _misread_as_failed(monkeypatch, app, turn_id, _DS, 0)
        sent = len(calls)

        follow_up = await app.controller.submit_draft(
            "And now?", session_id=app.session.id
        )

        assert follow_up.accepted
        if misread:
            assert app.reservation_errors == ["completed_tool_turn_lineage"]
            assert app.controller.run_state.status.value == "blocked"
            assert len(calls) == sent
        else:
            assert app.reservation_errors == []
            assert app.controller.run_state.status.value == "completed"
            assert len(calls) == sent + 1
        assert _chain_calls(app, turn_id) == before


@asynccontextmanager
async def _llamacpp_falls_back(
    tmp_path, monkeypatch, *, tool_first, project, groq_replies=()
):
    """A captured llama.cpp run that falls back to groq (TASK-42472's path).

    llama.cpp answers with a text tool call first when ``tool_first``, then
    is busy until its in-place retries are spent. groq takes each of
    ``groq_replies`` in turn, then answers. Yields the app, each groq
    request's messages, and ``switch``: after it, the session's sends go to a
    hosted provider (groq's endpoint) instead of llama.cpp.
    """
    from Tests.Chat.test_console_project_instruction_traces import _project_console
    from tldw_chatbook.Chat import (
        Chat_Functions,
        console_agent_bridge,
        provider_readiness,
    )
    from tldw_chatbook.Chat.console_dispatch_checkpoint import (
        ConsoleEgressClass,
        ConsoleResolvedDestination,
    )

    steps = ("tool",) * tool_first + ("busy",) * 3
    async with _project_console(
        tmp_path,
        monkeypatch,
        source_name="AGENTS.md" if project else "NOTES.md",
        first_failure=_llama_script(*steps),
        failures=len(steps),
    ) as app:
        llama, groq, replies, switched = (
            app.gateway.resolve_for_send,
            [],
            list(groq_replies),
            [],
        )
        hosted = replace(
            _GROQ,
            resolved_destination=ConsoleResolvedDestination(
                provider=_GQ,
                model=_GROQ.model,
                endpoint_identity=_GROQ.base_url,
                egress_class=ConsoleEgressClass.PUBLIC_NETWORK,
            ),
        )

        async def resolve(selection):
            if selection.provider == _GQ:
                return _GROQ
            return hosted if switched else await llama(selection)

        @functools.wraps(Chat_Functions.chat_api_call)
        def handler(*_args, **kwargs):
            groq.append(kwargs["messages_payload"])
            reply = (
                replies.pop(0) if replies else {"role": "assistant", "content": "ok"}
            )
            return {"choices": [{"message": reply}]}

        monkeypatch.setattr(app.gateway, "resolve_for_send", resolve)
        monkeypatch.setattr(app.gateway, "_chat_api_call_fn", handler)
        monkeypatch.setattr(
            console_agent_bridge, "console_fallback_providers", lambda: (_GQ,)
        )
        monkeypatch.setattr(
            provider_readiness, "configured_provider_is_ready", lambda _p: True
        )
        yield app, groq, lambda: switched.append(True)


@pytest.mark.usefixtures("shut_down_console_controllers")
@pytest.mark.parametrize("misread", [False, True], ids=["plain", "misread"])
@pytest.mark.parametrize("tool_first", [False, True], ids=["no_tools", "tool_first"])
async def test_after_a_llamacpp_fallback_the_next_message_to_a_hosted_provider_is_sent(
    tmp_path, monkeypatch, tool_first, misread
):
    """TASK-42525 review finding I1.

    With project context, the next message's trace replaces the run's
    suffix from its project context on. The fallback's removal marker sits
    inside that range, behind the llama.cpp rows, and the witness read it as
    a missing row (`completed_tool_turn_range`). It now counts the run's
    marker as one of its dead rows. groq answered with a tool call, so its
    chain's rows are still rechecked: read as failed, groq's first call
    refuses the next message.
    """
    async with _llamacpp_falls_back(
        tmp_path,
        monkeypatch,
        tool_first=tool_first,
        project=True,
        groq_replies=[_NATIVE_TOOL_CALL],
    ) as (app, groq, switch):
        result = await app.controller.submit_draft(
            "Use calculator once", session_id=app.session.id
        )

        assert app.controller.run_state.status.value == "completed"
        assert app.reservation_errors == []
        turn_id = app.store.get_message(result.user_message_id).persisted_message_id
        before = _chain_calls(app, turn_id)
        assert before == [
            ("agent_first", "complete" if tool_first else "error", "llama_cpp", 0, 0),
            *(("tool_loop", "error", "llama_cpp", 0, n) for n in range(1, 3 + tool_first)),
            ("tool_loop", "complete", _GQ, 1, 0),
            ("tool_loop", "complete", _GQ, 1, 1),
        ]
        if misread:
            _misread_as_failed(monkeypatch, app, turn_id, _GQ, 0)
        switch()
        sent = len(groq)

        follow_up = await app.controller.submit_draft(
            "And now?", session_id=app.session.id
        )

        assert follow_up.accepted
        if misread:
            assert app.reservation_errors == ["completed_tool_turn_lineage"]
            assert app.controller.run_state.status.value == "blocked"
            assert len(groq) == sent
        else:
            assert app.reservation_errors == []
            assert app.controller.run_state.status.value == "completed"
            assert len(groq) == sent + 1
            assert groq[-1][0]["role"] != "system"
        assert _chain_calls(app, turn_id) == before


@pytest.mark.usefixtures("shut_down_console_controllers")
@pytest.mark.parametrize("project", [True, False], ids=["project", "no_project"])
@pytest.mark.parametrize("tool_first", [False, True], ids=["no_tools", "tool_first"])
async def test_a_captured_llamacpp_run_falls_back_to_a_hosted_provider(
    tmp_path, monkeypatch, tool_first, project
):
    """TASK-42525 on TASK-42472's path, second owner decision.

    llama.cpp HTTP failures are transient, so a llama.cpp run reaches its
    fallback once the in-place retries are spent. llama.cpp is sent the
    system prompt as the first message, groq is not, and after a tool call
    the run's text tool rows are rewritten for groq's native format. The
    fallback chain's first call removes the leading system slot and rewrites
    those rows: the run completes, and the trace keeps the llama.cpp calls
    with their system row and records groq's call without it.

    The next message goes back to llama.cpp and needs the slot back. Putting
    a leading system row back is the refusal any hosted -> llama.cpp switch
    meets under capture (`unsupported_surface_change`), which the owner's
    decision did not grant; it stays refused, and Send without capture
    sends it.
    """
    import functools

    from Tests.Chat.test_console_project_instruction_traces import _project_console
    from tldw_chatbook.Chat import (
        Chat_Functions,
        console_agent_bridge,
        provider_readiness,
    )
    from tldw_chatbook.Chat.console_trace_native_reader import (
        ConsoleTraceNativeReader,
    )

    steps = ("tool",) * tool_first + ("busy",) * 3
    async with _project_console(
        tmp_path,
        monkeypatch,
        source_name="AGENTS.md" if project else "NOTES.md",
        first_failure=_llama_script(*steps),
        failures=len(steps),
    ) as app:
        llama, groq = app.gateway.resolve_for_send, []

        async def resolve(selection):
            return _GROQ if selection.provider == _GQ else await llama(selection)

        @functools.wraps(Chat_Functions.chat_api_call)
        def handler(*_args, **kwargs):
            groq.append(kwargs["messages_payload"])
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

        monkeypatch.setattr(app.gateway, "resolve_for_send", resolve)
        monkeypatch.setattr(app.gateway, "_chat_api_call_fn", handler)
        monkeypatch.setattr(
            console_agent_bridge, "console_fallback_providers", lambda: (_GQ,)
        )
        monkeypatch.setattr(
            provider_readiness, "configured_provider_is_ready", lambda _p: True
        )
        result = await app.controller.submit_draft(
            "Use calculator once" if tool_first else "Hello",
            session_id=app.session.id,
        )

        assert app.controller.run_state.status.value == "completed"
        assert app.reservation_errors == []
        turn_id = app.store.get_message(result.user_message_id).persisted_message_id
        failed = [("tool_loop", "error", "llama_cpp", 0, n) for n in (1, 2, 3)]
        before = _chain_calls(app, turn_id)
        assert before == [
            ("agent_first", "complete" if tool_first else "error", "llama_cpp", 0, 0),
            *failed[: 2 + tool_first],
            ("tool_loop", "complete", _GQ, 1, 0),
        ]
        (sent,) = groq
        assert sent[0]["role"] != "system"
        assert ("tool" in _roles(sent)) is tool_first
        traced = ConsoleTraceNativeReader(app.db).read_calls(turn_id)
        requests = [call.capture.request["messages_payload"] for call in traced]
        assert all(rows[0]["role"] == "system" for rows in requests[:-1])
        assert _roles(requests[-1]) == _roles(sent)

        follow_up = await app.controller.submit_draft(
            "And now?", session_id=app.session.id
        )

        assert follow_up.accepted
        assert app.reservation_errors == ["unsupported_surface_change"]
        assert app.controller.run_state.status.value == "blocked"
        assert _chain_calls(app, turn_id) == before
        paused = app.store.preparation_for_session(app.session.id)
        resent = await app.controller.send_without_capture(paused.preparation_id)
        assert resent.accepted, resent.visible_copy
        assert app.controller.run_state.status.value == "completed"


def _recording(factory):
    """``factory`` as the gateway calls it; each refusal's code is kept."""
    refusals = []

    def boundary(*args, **kwargs):
        try:
            return factory(*args, **kwargs)
        except Exception as exc:
            refusals.append(str(exc))
            raise

    return boundary, refusals


def _tool_rows(policy, tag, count):
    """``count`` alternating tool-call and tool-result rows, as one format."""
    messages, descriptors = [], []
    for index in range(count):
        call = index % 2 == 0
        messages.append(
            {"role": "assistant", "content": f"{tag} call {index}"}
            if call
            else {"role": "tool", "content": f"{tag} result {index}"}
        )
        descriptors.append(
            ProviderArtifactTraceProvenance(
                TraceProvenanceSource.TOOL_CALL
                if call
                else TraceProvenanceSource.TOOL_RESULT,
                policy,
            )
        )
    return messages, descriptors


async def _failed_tool_run(
    tmp_path, make_database, make_gateway, monkeypatch, *, rows=2, system=False
):
    """A deepseek run whose first call answered and whose tool-loop call failed.

    Its tool rows are written as text (``"text"``). Returns a namespace with
    the database, factory, the history, ``send`` and the providers asked.
    """
    database = make_database(tmp_path / "rewrite.sqlite", "rewrite")
    conversation = database.add_conversation({"title": "rewrite"})
    _, earlier = _saved_message(database, conversation, "earlier")
    _, revision = _saved_message(database, conversation, "hello")
    policy = FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None)
    factory = ConsoleTraceBoundaryFactory(database)
    boundary, refusals = _recording(factory)
    asked, failing = [], {"deepseek"}

    def provider(**kwargs):
        asked.append(kwargs["api_endpoint"])
        if kwargs["api_endpoint"] in failing and len(asked) > 1:
            raise ChatRateLimitError("fixture", provider=kwargs["api_endpoint"])
        return {"choices": [{"message": {"content": "answered"}}]}

    gateway = make_gateway(
        chat_api_call_fn=provider,
        trace_call_boundary_factory=boundary,
        safe_error_copy=lambda _provider, _exc: "fixture provider failure",
    )
    if system:
        # llama.cpp's shape: the system prompt is the first message.
        prepare = gateway.prepare_chat_request

        def distinct_roles(*args, **kwargs):
            prepared = prepare(*args, **kwargs)
            return prepare_provider_request(
                prepared.semantic,
                wire_style="distinct_roles",
                provider=prepared.provider,
                model=prepared.model,
                capacity=prepared.capacity,
            )

        monkeypatch.setattr(gateway, "prepare_chat_request", distinct_roles)
    deepseek = ConsoleProviderResolution(
        ready=True,
        provider="deepseek",
        execution_key="deepseek",
        model="deepseek-chat",
        base_url="https://api.deepseek.com",
        api_key="fixture",
        streaming=False,
    )
    head = [
        {"role": "user", "content": "earlier"},
        {"role": "user", "content": "hello"},
    ]
    head_descriptors = [earlier, revision]
    if system:
        head.insert(0, {"role": "system", "content": "Text tool protocol"})
        head_descriptors.insert(
            0,
            ProviderArtifactTraceProvenance(
                TraceProvenanceSource.RENDERED_SYSTEM, policy
            ),
        )
    actor, chain = new_opaque_id(), new_opaque_id()

    async def send(resolution, route, chain_id, messages, descriptors):
        request = _semantic_request(
            messages,
            descriptors,
            policy,
            route=route,
            actor_id=actor,
            chain_id=chain_id,
        )
        return [
            item
            async for item in gateway.stream_chat(
                resolution,
                request,
                route=route,
                route_actor_id=actor,
                route_chain_id=chain_id,
                capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
            )
        ]

    assert await send(
        deepseek, ConsoleRequestRoute.AGENT_FIRST, chain, head, head_descriptors
    )
    text, text_descriptors = _tool_rows(policy, "text", rows)
    with pytest.raises(ChatRateLimitError):
        await send(
            deepseek,
            ConsoleRequestRoute.TOOL_LOOP,
            chain,
            head + text,
            head_descriptors + text_descriptors,
        )
    return SimpleNamespace(
        database=database,
        conversation=conversation,
        factory=factory,
        policy=policy,
        send=send,
        asked=asked,
        failing=failing,
        deepseek=deepseek,
        chain=chain,
        head=head,
        head_descriptors=head_descriptors,
        rows=rows,
        refusals=refusals,
    )


def _reserved(database):
    with database.transaction() as cursor:
        return cursor.execute(
            "SELECT COUNT(*) FROM console_trace_calls WHERE state = 'reserved'"
        ).fetchone()[0]


@pytest.mark.parametrize(
    "scenario",
    [
        "valid",
        "valid_slot",
        "valid_second_rewrite",
        "valid_second_rewrite_after_slot",
        "unearned_second_rewrite_after_slot",
        "retry_rewrite",
        "second_call",
        "user_row",
        "foreign_run",
        "slot_not_the_runs",
        "over_256",
    ],
)
async def test_a_fallback_rewrite_is_admitted_only_in_its_one_shape(
    tmp_path, make_database, make_gateway, monkeypatch, scenario
):
    """TASK-42525, second owner decision: the rewrite's bounds.

    A fallback chain's first call may remove the leading system slot and
    replace the run's earned tool rows one for one. Anything else is refused
    before a call is reserved, and the provider is never asked: a rewrite on
    any other call, a removed row that is not a tool row, rows the run did not
    earn, a slot the run's own call did not carry, and more than 256 rows.
    """
    from tldw_chatbook.Chat import console_trace_fallback_rewrite as rewrite

    rows = 257 if scenario == "over_256" else 2
    system = scenario in {"valid_slot", "slot_not_the_runs"} or scenario.endswith(
        "after_slot"
    )
    run = await _failed_tool_run(
        tmp_path, make_database, make_gateway, monkeypatch, rows=rows, system=system
    )
    native, native_descriptors = _tool_rows(run.policy, "native", rows)
    head, head_descriptors = run.head, run.head_descriptors
    if system:
        head, head_descriptors = head[1:], head_descriptors[1:]
    messages = head + native
    descriptors = head_descriptors + native_descriptors
    target, chain = _GROQ, new_opaque_id()
    if scenario == "retry_rewrite":
        # An in-place retry of the failed call that rewrites its tool rows.
        target, chain = run.deepseek, run.chain
    elif scenario == "second_call":
        text, text_descriptors = _tool_rows(run.policy, "text", rows)
        assert await run.send(
            _GROQ,
            ConsoleRequestRoute.TOOL_LOOP,
            chain,
            head + text,
            head_descriptors + text_descriptors,
        )
    elif scenario == "valid_second_rewrite":
        # The first fallback rewrote the rows and failed too; the second
        # rewrites the first one's rows again, which stand on the earned ones.
        run.failing.add("groq")
        with pytest.raises(ChatRateLimitError):
            await run.send(
                _GROQ, ConsoleRequestRoute.TOOL_LOOP, chain, messages, descriptors
            )
        again, again_descriptors = _tool_rows(run.policy, "text again", rows)
        messages, descriptors = head + again, head_descriptors + again_descriptors
        target = replace(_GROQ, provider="openai", execution_key="openai")
        chain = new_opaque_id()
    elif scenario.endswith("second_rewrite_after_slot"):
        # The first fallback removed the slot and rewrote the rows; its chain
        # added rows and failed. The second rewrites all of them, across the
        # first one's removal marker, which holds no row (review finding I1).
        assert await run.send(
            _GROQ, ConsoleRequestRoute.TOOL_LOOP, chain, messages, descriptors
        ) == ["answered"]
        more, more_descriptors = _tool_rows(run.policy, "native more", rows)
        run.failing.add("groq")
        with pytest.raises(ChatRateLimitError):
            await run.send(
                _GROQ,
                ConsoleRequestRoute.TOOL_LOOP,
                chain,
                messages + more,
                descriptors + more_descriptors,
            )
        again, again_descriptors = _tool_rows(run.policy, "text again", 2 * rows)
        messages, descriptors = head + again, head_descriptors + again_descriptors
        target = replace(_GROQ, provider="openai", execution_key="openai")
        chain = new_opaque_id()
        if scenario.startswith("unearned"):
            # Read groq's first call as failed: the rows its chain added
            # were not earned, marker or no marker.
            from tldw_chatbook.Chat.console_trace_models import TraceCallState

            logical = run.factory.repository.get_call_by_logical_identity

            def unanswered(cursor, **key):
                call = logical(cursor, **key)
                if call is not None and (call.provider_name, call.call_sequence) == (
                    "groq",
                    0,
                ):
                    return replace(call, state=TraceCallState.ERROR, outcome="error")
                return call

            monkeypatch.setattr(
                run.factory.repository, "get_call_by_logical_identity", unanswered
            )
    elif scenario == "user_row":
        _, changed = _saved_message(run.database, run.conversation, "changed")
        messages = [{"role": "user", "content": "changed"}] + messages[1:]
        descriptors = [changed] + descriptors[1:]
    elif scenario == "foreign_run":
        monkeypatch.setattr(
            rewrite,
            "run_chains",
            lambda *_args: (None, frozenset({f"{new_opaque_id()}:{new_opaque_id()}"})),
        )
    elif scenario == "slot_not_the_runs":
        get_header = run.factory.repository.get_request_header

        def without_slot(cursor, header_id):
            header = get_header(cursor, header_id)
            return replace(
                header,
                components=tuple(
                    item
                    for item in header.components
                    if item.component_kind != "rendered_system_row"
                ),
            )

        monkeypatch.setattr(run.factory.repository, "get_request_header", without_slot)
    asked = list(run.asked)
    if scenario.startswith("valid"):
        assert await run.send(
            target, ConsoleRequestRoute.TOOL_LOOP, chain, messages, descriptors
        ) == ["answered"]
        assert run.asked == asked + [target.execution_key]
    else:
        with pytest.raises(TraceCallPersistenceError):
            await run.send(
                target, ConsoleRequestRoute.TOOL_LOOP, chain, messages, descriptors
            )
        assert run.asked == asked
        # A rewrite on any call but a fallback chain's first is no rewrite:
        # the planner refuses the changed surface before it is considered.
        assert run.refusals == [
            "unsupported_surface_change"
            if scenario in {"retry_rewrite", "second_call"}
            else "trace_fallback_rewrite_unavailable"
        ]
    assert _reserved(run.database) == 0


@pytest.mark.parametrize(
    "changed", ["none", "owner", "turn", "policy", "run", "unanswered"]
)
async def test_only_the_runs_own_tool_rows_are_earned(
    tmp_path, make_database, make_gateway, monkeypatch, changed
):
    """TASK-42525: the rewrite's run-ownership check, row by row.

    A removed tool row is earned only when a tool-loop call of one of the
    run's chains appended it, in the run's owner, turn and policy, after a
    call of its chain that bore a response (``unanswered`` reads that call
    as failed).
    """
    from tldw_chatbook.Chat.console_trace_models import TraceCallState

    from tldw_chatbook.Chat.console_trace_continuation import run_chains
    from tldw_chatbook.Chat.console_trace_fallback_rewrite import earned_tool_rows

    run = await _failed_tool_run(tmp_path, make_database, make_gateway, monkeypatch)
    with run.database.transaction() as cursor:
        failed = cursor.execute(
            "SELECT call_id FROM console_trace_calls WHERE call_sequence = 1"
        ).fetchone()[0]
        failed = run.factory.repository.get_call(cursor, failed)
        _origin, runs = run_chains(run.factory.repository, cursor, failed.run_id)
        tail = run.factory.repository.get_surface_tail(cursor, failed.segment_id)
        identity = {
            "segment_id": failed.segment_id,
            "owner_id": failed.owner_id,
            "turn_id": failed.turn_id,
            "policy_id": failed.policy_id,
            "runs": runs,
        }
        if changed == "run":
            identity["runs"] = frozenset({f"{new_opaque_id()}:{new_opaque_id()}"})
        elif changed == "unanswered":
            logical = run.factory.repository.get_call_by_logical_identity

            def unanswered(cursor, **key):
                call = logical(cursor, **key)
                return replace(call, state=TraceCallState.ERROR, outcome="error")

            monkeypatch.setattr(
                run.factory.repository, "get_call_by_logical_identity", unanswered
            )
        elif changed != "none":
            identity[f"{changed}_id"] = new_opaque_id()
        earned = earned_tool_rows(
            run.factory.service,
            cursor,
            sequences=(tail.sequence - 1, tail.sequence),
            **identity,
        )
    assert earned is (changed == "none")


async def test_a_rewrites_bookkeeping_reads_only_its_own_rows(
    tmp_path, make_database, make_gateway, monkeypatch
):
    """TASK-42525 review findings M6 and I1, called directly.

    A rewrite whose new rows do not match the removed ones in number is
    refused with the rewrite's own code, not zip's ValueError. A run's dead
    rows are the rows its fallback rewrite replaced plus that rewrite's
    removal marker, and only its own: another run's chains name none.
    """
    from tldw_chatbook.Chat.console_trace_continuation import run_chains
    from tldw_chatbook.Chat.console_trace_fallback_rewrite import (
        extend_over_rewrites,
        rewritten_sequences,
        validate_fallback_rewrite,
    )
    from tldw_chatbook.Chat.console_trace_final_values import FallbackRewriteWitness

    run = await _failed_tool_run(
        tmp_path, make_database, make_gateway, monkeypatch, system=True
    )
    repository = run.factory.repository
    _calls, descriptors = _tool_rows(run.policy, "native", 2)
    with run.database.transaction() as cursor:
        (failed_id,) = cursor.execute(
            "SELECT call_id FROM console_trace_calls WHERE call_sequence = 1"
        ).fetchone()
        failed = repository.get_call(cursor, failed_id)
        tail = repository.get_surface_tail(cursor, failed.segment_id)
        plan = SimpleNamespace(
            start_sequence=tail.sequence - 1,
            end_sequence=tail.sequence,
            predecessor_head_id=tail.node_id,
            chain=(),
        )
        for count in (2, 1):
            witness = FallbackRewriteWitness(
                failed.call_id, None, tuple(descriptors[:count])
            )
            check = functools.partial(
                validate_fallback_rewrite,
                run.factory.service,
                cursor,
                witness=witness,
                owner_id=failed.owner_id,
                segment_id=failed.segment_id,
                plan=plan,
                descriptors=witness.descriptors,
            )
            if count == 2:
                check()  # the control: one for one
            else:
                with pytest.raises(ValueError) as refused:
                    check()
                assert refused.value.args == ("trace_fallback_rewrite_unavailable",)
    rows = (tail.sequence - 1, tail.sequence)
    native, native_descriptors = _tool_rows(run.policy, "native", 2)
    assert await run.send(
        _GROQ,
        ConsoleRequestRoute.TOOL_LOOP,
        new_opaque_id(),
        run.head[1:] + native,
        run.head_descriptors[1:] + native_descriptors,
    ) == ["answered"]
    with run.database.transaction() as cursor:
        (marker,) = cursor.execute(
            "SELECT sequence FROM console_trace_surface_nodes "
            "WHERE component_kind = 'surface_removal'"
        ).fetchone()
        groq = cursor.execute(
            "SELECT run_id FROM console_trace_calls WHERE provider_name = 'groq'"
        ).fetchone()[0]
        _origin, runs = run_chains(repository, cursor, groq)
        dead = functools.partial(
            rewritten_sequences, cursor, segment_id=failed.segment_id, start=0
        )
        assert dead(runs=runs, end=marker) == {*rows, marker}
        assert dead(runs=runs, end=marker - 1) == set(rows)
        foreign = frozenset({f"{new_opaque_id()}:{new_opaque_id()}"})
        assert dead(runs=foreign, end=marker) == set()
        # A completed-turn range reaches down from groq's rows over the rows
        # its rewrite replaced, to the row after the origin's head, and only
        # for the run that rewrote them.
        (groq_call,) = cursor.execute(
            "SELECT call_id FROM console_trace_calls WHERE provider_name = 'groq'"
        ).fetchone()
        (origin_head,) = cursor.execute(
            "SELECT n.sequence FROM console_trace_calls c JOIN "
            "console_trace_surface_nodes n ON n.node_id = c.surface_node_id "
            "WHERE c.route_identity = 'agent_first'"
        ).fetchone()
        extend = functools.partial(
            extend_over_rewrites,
            repository,
            cursor,
            segment_id=failed.segment_id,
            start=max(rows) + 1,
            floor=origin_head,
        )
        assert min(rows) == origin_head + 1
        assert extend(terminal_call_id=groq_call) == min(rows)
        assert extend(terminal_call_id=failed.call_id) == max(rows) + 1
