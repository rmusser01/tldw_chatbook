"""A failed call authorizes an unchanged retry, never arbitrary tool history."""

import json
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
    An exception in ``replies`` is raised; past the script, the call answers.
    Returns each call's ``messages_payload``, in order.
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
        return {"choices": [{"message": {"role": "assistant", "content": reply}}]}

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
_FAILED = ("tool_loop", "error")
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

    options = _RUNS[run][0]
    async with _project_console(
        tmp_path,
        monkeypatch,
        source_name="AGENTS.md" if project else "NOTES.md",
        first_failure=_llama_failure("http503") if "failures" in options else None,
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


def _empty_stream(_request):
    return httpx.Response(200, text="", headers={"content-type": "text/event-stream"})


@pytest.mark.usefixtures("shut_down_console_controllers")
@pytest.mark.parametrize("project", [True, False], ids=["project", "no_project"])
@pytest.mark.parametrize("empty", ["first", "every"])
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
    had no saved user message to own it.
    """
    from Tests.Chat.test_console_project_instruction_traces import (
        GUIDANCE,
        _project_console,
    )

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
        result = await app.controller.submit_draft(
            "Use calculator once", session_id=app.session.id
        )
        assert result.accepted
        assert app.reservation_errors == []
        assert app.controller.run_state.status.value == "completed"
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
    "latest", ["failed_tool_loop", "answered", "failed_first_call"]
)
async def test_a_resend_of_a_tool_loop_call_repeats_its_failed_call(
    tmp_path, make_database, make_gateway, latest
):
    """TASK-42530: a llama.cpp resend of a tool-loop call owns no new history.

    It runs on its own `llama_fallback` call, and takes the tool loop's turn
    from the request, so it is admitted only directly after the failed
    tool-loop call it repeats, on that call's surface.
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
        latest, [200]
    )
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

    async def send(route):
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
    # A resend whose latest call is not a failed tool-loop call.
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
    with pytest.raises(TraceReservationRefused) as refused:
        factory(prepared, resolution, ConsoleRequestRoute.LLAMA_FALLBACK)
    assert refused.value.args[0] == "trace_tool_chain_unavailable"


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
    ],
)
@pytest.mark.parametrize("path", ["self_hosted", "llama_tools"])
async def test_a_streamed_retry_after_a_failed_in_chain_resend_repeats_the_stream(
    tmp_path, make_database, make_gateway, path, scenario
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

    gateway = make_gateway(
        chat_api_call_fn=handler,
        trace_call_boundary_factory=ConsoleTraceBoundaryFactory(database),
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
