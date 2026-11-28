"""A new turn's first call drops the leading system slot (TASK-42550).

ADR-337 amendment 2026-11-26 (owner-approved), the mirror of the 2026-11-25
slot restore. llama.cpp sends the system prompt as the first message, so a
chat that started there has a leading rendered-system slot on its trace
surface; a hosted provider takes the system prompt outside the message list,
so its request has no row for it.
"""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from Tests.Chat.test_console_failed_call_retry_ownership import (
    _GQ,
    _GROQ,
    _NATIVE_TOOL_CALL,
    _assert_slot_first,
    _chain_calls,
    _llamacpp_falls_back,
    _recording,
    _reserved,
    _roles,
    _tool_rows,
)
from Tests.Chat.test_console_project_instruction_traces import GUIDANCE
from Tests.Chat.test_console_trace_current_turn_transforms import (  # noqa: F401
    trace_harness,
)
from Tests.Chat.test_console_trace_rewritten_history import (  # noqa: F401
    _LONG,
    SESSION,
    _assert_captured,
    _restart_trace_runtime,
    harness,
)
from Tests.Chat.test_console_trace_rewritten_history import _send as _history_send
from Tests.Chat.test_console_trace_runtime import _saved_message, _semantic_request
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

# The Console bridge reads the tool gates through config (TASK-41508), and
# the gateway reads [model_catalog] (TASK-42472's lesson).
pytestmark = pytest.mark.bootstrap_profile
make_database = _make_database_fixture
make_gateway = _make_gateway_fixture


@pytest.fixture(autouse=True)
def _close_idle_handles(monkeypatch, request):
    """Close the handles each test's databases opened on other threads.

    recovered-suites runs this file under ``--storage-lease-guard``
    (TASK-38986). The Console opens the chat database on an executor thread
    and the agent-runs database on its own, and closing them from the test
    leaves those handles live; as in ``test_console_rewind_summarize.py``.
    """
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.DB.store_kernel import SQLiteStoreKernel

    stores = []
    init, opened = CharactersRAGDB.__init__, SQLiteStoreKernel._on_connection_opened

    def created(self, *args, **kwargs):
        init(self, *args, **kwargs)
        stores.append(self)

    def connected(self, conn):
        stores.append(self)
        opened(self, conn)

    monkeypatch.setattr(CharactersRAGDB, "__init__", created)
    monkeypatch.setattr(SQLiteStoreKernel, "_on_connection_opened", connected)
    request.addfinalizer(lambda: [store.close_idle_connections() for store in stores])


def _traced(app, result):
    """The turn's traced calls and the last one's rebuilt messages."""
    from tldw_chatbook.Chat.console_trace_native_reader import (
        ConsoleTraceNativeReader,
    )

    turn = app.store.get_message(result.user_message_id).persisted_message_id
    calls = ConsoleTraceNativeReader(app.db).read_calls(turn)
    return calls, calls[-1].capture.request["messages_payload"]


def _surface_kinds(app, result):
    """The kinds of the surface's active rows after ``result``'s turn."""
    turn = app.store.get_message(result.user_message_id).persisted_message_id
    with app.db.transaction() as cursor:
        (segment,) = cursor.execute(
            "SELECT DISTINCT segment_id FROM console_trace_calls WHERE turn_id = ?",
            (turn,),
        ).fetchone()
        tail = app.factory.repository.get_surface_tail(cursor, segment)
        entries = app.factory.service._surface_projection(
            cursor, segment, tail
        ).entries
    return [key[0] for _, key in entries]


def _without_ids(row):
    row = {key: value for key, value in row.items() if key != "tool_call_id"}
    if "tool_calls" in row:
        row["tool_calls"] = [
            {key: value for key, value in call.items() if key != "id"}
            for call in row["tool_calls"]
        ]
    return row


def _assert_traced_as_sent(rows, wire):
    """The trace rebuilds the request, as the credential filter stores it.

    The filter rewrites the project file's example credential and the
    fixture's tool-call id; every other value is the one sent.
    """
    assert _roles(rows) == _roles(wire)
    for row, sent in zip(rows, wire, strict=True):
        if GUIDANCE in str(sent["content"]):
            assert GUIDANCE in str(row["content"])
        else:
            assert _without_ids(row) == _without_ids(sent)


async def _send(app, text):
    result = await app.controller.submit_draft(text, session_id=app.session.id)
    assert result.accepted
    assert app.reservation_errors == []
    assert app.controller.run_state.status.value == "completed"
    return result


@pytest.mark.usefixtures("shut_down_console_controllers")
@pytest.mark.parametrize("tools", [False, True], ids=["no_tools", "tools"])
@pytest.mark.parametrize("project", [True, False], ids=["project", "no_project"])
async def test_a_captured_switch_from_llamacpp_to_a_hosted_provider_is_sent(
    tmp_path, monkeypatch, project, tools
):
    """The probe's shape: llama.cpp first, then the session resolves to groq.

    With project instructions the hosted message was refused
    (`unsupported_surface_change`, the run `blocked`, groq never asked);
    without them it was sent. Now both are sent and traced without the
    system row, and so is the message after it. With ``tools`` groq answers
    the switched message with a tool call, so the next message proves that
    run's tool history, which follows the removal marker.
    """
    async with _llamacpp_falls_back(
        tmp_path,
        monkeypatch,
        tool_first=False,
        project=project,
        busy=0,
        groq_replies=[_NATIVE_TOOL_CALL] * tools,
        retire_agent_runs=True,
    ) as (app, groq, hosted):
        first = await _send(app, "Hello")
        assert app.http_payloads[-1]["messages"][0]["role"] == "system"
        assert _surface_kinds(app, first)[0] == "rendered_system"
        hosted.append(True)

        switched = await _send(app, "Use calculator once" if tools else "And now?")

        assert len(groq) == 1 + tools
        assert all(sent[0]["role"] != "system" for sent in groq)
        assert any(GUIDANCE in str(row["content"]) for row in groq[0]) is project
        assert ("tool" in _roles(groq[-1])) is tools
        calls, rows = _traced(app, switched)
        assert [call.capture.status for call in calls] == ["complete"] * (1 + tools)
        turn = app.store.get_message(switched.user_message_id).persisted_message_id
        assert _chain_calls(app, turn) == [
            ("agent_first", "complete", _GQ, 0, 0),
            *([("tool_loop", "complete", _GQ, 0, 1)] * tools),
        ]
        _assert_traced_as_sent(rows, groq[-1])
        assert "rendered_system" not in _surface_kinds(app, switched)
        assert [route for _, route in _markers(app.db)] == ["agent_first"]

        again = await _send(app, "Once more?")

        assert len(groq) == 2 + tools
        _calls, rows = _traced(app, again)
        _assert_traced_as_sent(rows, groq[-1])
        assert groq[-1][0]["role"] != "system"
        turn = app.store.get_message(again.user_message_id).persisted_message_id
        assert _chain_calls(app, turn) == [("agent_first", "complete", _GQ, 0, 0)]


@pytest.mark.usefixtures("shut_down_console_controllers")
@pytest.mark.parametrize("restart", [False, True], ids=["warm", "restart"])
@pytest.mark.parametrize("project", [True, False], ids=["project", "no_project"])
async def test_switching_back_to_llamacpp_after_a_drop_puts_the_slot_back(
    tmp_path, monkeypatch, project, restart
):
    """llama.cpp -> groq -> groq -> llama.cpp: the slot-restore rule takes it back.

    With ``restart`` every send starts on a cold trace factory, as after an
    app restart: the drop and the sends after it plan from the ledger alone.
    """
    async with _llamacpp_falls_back(
        tmp_path, monkeypatch, tool_first=False, project=project, busy=0,
        retire_agent_runs=True,
    ) as (app, groq, hosted):

        async def send(text):
            if restart:
                app.restart_factory()
            return await _send(app, text)

        await send("Hello")
        hosted.append(True)
        await send("And now?")
        await send("Once more?")
        hosted.clear()

        back = await send("Back on llama.cpp?")

        assert len(groq) == 2
        _assert_slot_first(app, back, 1)
        kinds = _surface_kinds(app, back)
        assert kinds[0] == "rendered_system"
        assert kinds.count("rendered_system") == 1
        wire = json.loads(app.attempts[-1].content)["messages"]
        assert wire[0]["role"] == "system"
        assert any(GUIDANCE in str(row["content"]) for row in wire) is project


@pytest.mark.usefixtures("shut_down_console_controllers")
@pytest.mark.parametrize("path", ["drop", "always_llama"])
@pytest.mark.parametrize("edit", ["first", "second"])
async def test_edit_and_resend_after_a_drop_keeps_the_next_send(
    tmp_path, monkeypatch, edit, path
):
    """Edit & resend of the message before the drop, or of the one that dropped.

    The Edit & resend's own call is traced (TASK-42560): it re-records its
    turn as a rewritten tail that starts before the removal marker, which
    then sits mid-history; the next message, and the one after it back on
    llama.cpp, are sent and traced. Control: a chat always on llama.cpp.
    Without project instructions here; `test_console_trace_action_calls.py`
    covers the action and the next message with them.
    """
    async with _llamacpp_falls_back(
        tmp_path, monkeypatch, tool_first=False, project=False, busy=0,
        retire_agent_runs=True,
    ) as (app, groq, hosted):
        first = await _send(app, "Hello")
        if path == "drop":
            hosted.append(True)
        second = await _send(app, "And now?")
        await _send(app, "Third")
        markers = ["agent_first"] if path == "drop" else []
        assert [route for _, route in _markers(app.db)] == markers
        target = first if edit == "first" else second

        resent = await app.controller.edit_and_resend_message(
            target.user_message_id, "Edited"
        )

        assert resent.accepted
        assert app.reservation_errors == []
        assert app.controller.run_state.status.value == "completed"
        after = await _send(app, "After edit?")
        _calls, rows = _traced(app, after)
        wire = (
            groq[-1] if hosted else json.loads(app.attempts[-1].content)["messages"]
        )
        _assert_traced_as_sent(rows, wire)
        assert [row["content"] for row in wire if row["role"] == "user"][
            0 if edit == "first" else 1
        ] == "Edited"
        kinds = _surface_kinds(app, after)
        assert ("rendered_system" in kinds) is (path == "always_llama")
        hosted.clear()
        back = await _send(app, "Back on llama.cpp?")
        _assert_slot_first(app, back, 1)
        assert len(groq) == (4 if path == "drop" else 0)
        assert [route for _, route in _markers(app.db)] == markers


def _fail_after_commit(monkeypatch, *, when=lambda _persisted: True, forge=None):
    """Raise once, after the commit, on the first dispatch bind ``when`` names.

    The gateway then reads the write back (`_reconcile_dispatch_write`).
    Returns the state (``armed`` is False once raised) and, per read-back,
    (outcome, replacement written, removal written). ``forge`` may rewrite
    the read-back's keyword arguments.
    """
    from contextlib import contextmanager

    from tldw_chatbook.Chat import console_trace_service as trace

    service = trace.ConsoleTraceService
    state = {"armed": True, "in_bind": False, "persisted": None}
    outcomes = []
    policy_of = trace.trace_critical_write_checkpoint_policy
    bind, persist, reconcile = (
        service.bind_and_mark_dispatch,
        service.persist_request,
        service._reconcile_dispatch_write,
    )

    @contextmanager
    def policy(database):
        with policy_of(database):
            yield
        persisted = state["persisted"]
        if state["armed"] and state["in_bind"] and persisted and when(persisted):
            state["armed"] = False
            raise RuntimeError("post-commit failure")

    def binding(self, *args, **kwargs):
        state["in_bind"], state["persisted"] = True, None
        try:
            return bind(self, *args, **kwargs)
        finally:
            state["in_bind"] = False

    def persisting(self, *args, **kwargs):
        state["persisted"] = persist(self, *args, **kwargs)
        return state["persisted"]

    def reconciled(self, *args, **kwargs):
        if forge is not None:
            forge(kwargs)
        outcome = reconcile(self, *args, **kwargs)
        persisted = kwargs["persisted"]
        outcomes.append(
            (
                outcome if isinstance(outcome, str) else "durable",
                persisted.replacement is not None,
                persisted.removal is not None,
            )
        )
        return outcome

    monkeypatch.setattr(trace, "trace_critical_write_checkpoint_policy", policy)
    monkeypatch.setattr(service, "bind_and_mark_dispatch", binding)
    monkeypatch.setattr(service, "persist_request", persisting)
    monkeypatch.setattr(service, "_reconcile_dispatch_write", reconciled)
    return state, outcomes


@pytest.mark.usefixtures("shut_down_console_controllers")
@pytest.mark.parametrize(
    ("case", "read_back"),
    [
        pytest.param("drop", ("durable", True, True), id="drop-project"),
        pytest.param("restore", ("durable", True, False), id="restore-project"),
        pytest.param("drop_no_project", ("durable", False, True), id="drop-no_project"),
        pytest.param("llama", ("durable", True, False), id="llama-project"),
        pytest.param("forged_row", ("unknown", True, True), id="forged_row"),
        pytest.param("forged_head", ("unknown", True, True), id="forged_head"),
    ],
)
async def test_a_committed_slot_change_is_read_back_after_a_post_commit_failure(
    tmp_path, monkeypatch, case, read_back
):
    """TASK-42550 review, Important 1.

    The dispatch-binding transaction commits, then its exit raises. The
    gateway reads the write back (`_reconcile_dispatch_write`) and must find
    it durable. The read-back checked the replacement rows in a fallback's
    write order (replacement first, removal last), so a drop (its removal
    first, the replacement after the marker) and a restore (the slot first,
    the replacement after it) read back as unknown, and the run was blocked
    though its trace was whole. Each case has a replacement (project
    context); `drop-no_project` (a removal only) and `llama-project` (a
    replacement only) are controls. ``read_back`` is (outcome, replacement
    written, removal written).

    The forged cases must still read back as unknown: `forged_row` writes the
    drop's replacement row against the head before the marker (an ancestor,
    which the ledger's lineage trigger allows), and `forged_head` reads the
    write back against another planned head.
    """
    from dataclasses import replace as replaced

    async with _llamacpp_falls_back(
        tmp_path, monkeypatch, tool_first=False, project=case != "drop_no_project", busy=0,
        retire_agent_runs=True,
    ) as (app, groq, hosted):
        if case == "restore":
            hosted.append(True)
        await _send(app, "Hello")
        if case == "restore":
            hosted.clear()
        elif case != "llama":
            hosted.append(True)
        append = type(app.factory.repository).append_surface_replacement

        def appended(self, cursor, *, segment_id, replacement):
            node = self.get_surface_node(cursor, replacement.replacement_node_id)
            if case == "forged_row" and node.component_kind != "surface_removal":
                marker = self.get_surface_node(cursor, replacement.predecessor_head_id)
                replacement = replaced(
                    replacement, predecessor_head_id=marker.predecessor_node_id
                )
            return append(self, cursor, segment_id=segment_id, replacement=replacement)

        def forge(kwargs):
            if case == "forged_head":
                kwargs["surface_delta"] = replaced(
                    kwargs["surface_delta"],
                    predecessor_surface_head_id=kwargs["persisted"].surface_head_id,
                )

        state, outcomes = _fail_after_commit(monkeypatch, forge=forge)
        monkeypatch.setattr(
            type(app.factory.repository), "append_surface_replacement", appended
        )

        if read_back[0] == "unknown":
            await app.controller.submit_draft("And now?", session_id=app.session.id)
            assert app.controller.run_state.status.value == "blocked"
            assert outcomes == [read_back]
            return
        result = await _send(app, "And now?")

        assert not state["armed"]  # the failure was injected after the commit
        assert outcomes == [read_back]
        _calls, rows = _traced(app, result)
        if case in {"restore", "llama"}:
            assert rows[0]["role"] == "system"
        else:
            assert rows[0]["role"] != "system"
            assert [route for _, route in _markers(app.db)] == ["agent_first"]


@pytest.mark.usefixtures("shut_down_console_controllers")
@pytest.mark.parametrize("tool_first", [False, True], ids=["slot_only", "tool_rows"])
@pytest.mark.parametrize("project", [True, False], ids=["project", "no_project"])
async def test_a_committed_fallback_rewrite_is_read_back_after_a_post_commit_failure(
    tmp_path, monkeypatch, tool_first, project
):
    """The read-back of TASK-42525's fallback rewrite, after the review fix.

    A llama.cpp run falls back to groq; the fallback chain's first call
    removes the slot, and with ``tool_first`` also rewrites the run's tool
    rows. It writes the removal last, after its rows: the read-back's other
    write order. Its bind commits, then raises; the write reads back durable
    and the run completes.
    """
    async with _llamacpp_falls_back(
        tmp_path, monkeypatch, tool_first=tool_first, project=project, busy=3,
        retire_agent_runs=True,
    ) as (app, groq, hosted):
        state, outcomes = _fail_after_commit(
            monkeypatch, when=lambda persisted: persisted.removal is not None
        )

        await _send(app, "Use calculator once" if tool_first else "Hello")

        assert not state["armed"]
        assert outcomes == [("durable", tool_first, True)]
        assert [route for _, route in _markers(app.db)] == ["tool_loop"]
        assert groq[0][0]["role"] != "system"


@pytest.mark.asyncio
@pytest.mark.parametrize("harness", ["agent"], indirect=True)
@pytest.mark.parametrize("restart", [False, True], ids=["warm", "restart"])
async def test_compact_now_after_a_drop_keeps_the_next_send(
    harness,  # noqa: F811 - the imported fixture
    monkeypatch,
    restart,
):
    """Turns 1-2 on llama.cpp's wire (system row first), 3-4 on a hosted one.

    The drop at turn 3 leaves its removal marker mid-history. After Compact
    now the next send's rewritten tail starts before it; it is sent, and so
    is the next one back on llama.cpp's wire.
    """
    from tldw_chatbook.Chat.console_context_policy import (
        ConsoleContextPolicyOverrides,
        ContextBudgetMode,
        ContextCompactionMode,
    )

    harness.store.set_session_context_policy_overrides(
        SESSION,
        ConsoleContextPolicyOverrides(
            budget_mode=ContextBudgetMode.CUSTOM,
            custom_budget_tokens=1_800,
            compaction_mode=ContextCompactionMode.OFF,
            summary_max_tokens=100,
        ),
    )
    prepare = harness.gateway.prepare_chat_request
    llama = {"on": True}

    def styled(*args, **kwargs):
        prepared = prepare(*args, **kwargs)
        if not llama["on"]:
            return prepared
        return prepare_provider_request(
            prepared.semantic,
            wire_style="distinct_roles",
            provider=prepared.provider,
            model=prepared.model,
            capacity=prepared.capacity,
        )

    monkeypatch.setattr(harness.gateway, "prepare_chat_request", styled)
    roles = []
    for index in range(4):
        llama["on"] = index < 2
        result, sent, reserved = await _history_send(harness, f"q{index}: {_LONG}")
        _assert_captured(harness, result, sent, reserved)
        roles.append(sent[0][0].get("role"))
    assert roles == ["system", "system", "user", "user"]
    assert [route for _, route in _markers(harness.db)] == ["agent_first"]
    succeeded, _copy = await harness.controller.compact_context_now(SESSION)
    assert succeeded
    if restart:
        _restart_trace_runtime(harness)

    result, sent, reserved = await _history_send(harness, "after compaction")

    _assert_captured(harness, result, sent, reserved)
    assert sent[0][0]["role"] != "system"
    llama["on"] = True
    result, sent, reserved = await _history_send(harness, "back on llama.cpp")
    _assert_captured(harness, result, sent, reserved)
    assert sent[0][0]["role"] == "system"


def _markers(database):
    """Removal markers on the surface, as (sequence, written by route)."""
    with database.transaction() as cursor:
        rows = cursor.execute(
            "SELECT n.sequence, (SELECT c.route_identity FROM console_trace_events b "
            "JOIN console_trace_calls c ON c.call_id = b.call_id "
            "WHERE b.segment_id = e.segment_id AND b.event_type = 'call_boundary' "
            "AND b.sequence < e.sequence ORDER BY b.sequence DESC LIMIT 1) "
            "FROM console_trace_surface_nodes n "
            "JOIN console_trace_surface_replacements r "
            "ON r.replacement_node_id = n.node_id "
            "JOIN console_trace_events e ON e.surface_replacement_id = "
            "r.replacement_id AND e.event_type = 'surface_replace' "
            "WHERE n.component_kind = 'surface_removal' ORDER BY n.sequence"
        ).fetchall()
    return [tuple(row) for row in rows]


@pytest.mark.parametrize(
    "scenario",
    [
        "valid",
        "valid_fresh",
        "keeps_system_row",
        "not_a_slot",
        "tool_loop",
        "retry",
        "fallback",
        "bundled",
        "binding_not_first",
        "binding_no_slot",
        "binding_system_row",
    ],
)
async def test_the_system_slot_is_dropped_only_by_a_new_turns_first_call(
    tmp_path, make_database, make_gateway, monkeypatch, scenario
):
    """ADR-337 amendment 2026-11-26: the checks, one scenario each.

    The first turn went to llama.cpp (the system prompt is its first
    message), so the surface leads with the rendered-system slot. The next
    turn's first call, AGENT_FIRST or FRESH, with no leading system row,
    drops it: its removal marker is written before the call's rows, and
    the trace keeps the call's request as sent. Every other shape keeps the
    existing rules, each pinned by its outcome: a request that still leads
    with a system row keeps the slot (the existing header-owned rule); a
    leading row that is not a slot is never dropped; a tool-loop call, a
    retry and a fallback are refused before anything is reserved, each with
    the code it had before this rule (the fallback's is its own rule's); and
    `bundled`, which also changes a row the existing rules refuse, is refused
    with no marker written. The `binding_*` scenarios forge, at dispatch
    binding only, a call that is not its turn's first, a surface that no
    longer leads with the slot, or a request that leads with a system row:
    the binding transaction refuses to write the removal, and the call stays
    reserved, as after any binding refusal.
    """
    from tldw_chatbook.Chat import console_trace_service as trace
    from tldw_chatbook.Chat.console_trace_native_reader import (
        ConsoleTraceNativeReader,
    )

    database = make_database(tmp_path / "drop.sqlite", "drop")
    conversation = database.add_conversation({"title": "drop"})
    _, hello = _saved_message(database, conversation, "hello")
    policy = FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None)
    factory = ConsoleTraceBoundaryFactory(database)
    boundary, refusals = _recording(factory)
    asked = []
    first_fails = scenario in {"retry", "fallback"}

    def provider(**kwargs):
        asked.append(kwargs["api_endpoint"])
        if first_fails and len(asked) == 1:
            raise ChatRateLimitError("fixture", provider=kwargs["api_endpoint"])
        return {"choices": [{"message": {"content": "answered"}}]}

    gateway = make_gateway(
        chat_api_call_fn=provider,
        trace_call_boundary_factory=boundary,
        safe_error_copy=lambda _provider, _exc: "fixture provider failure",
    )
    prepare = gateway.prepare_chat_request

    def distinct_roles(*args, **kwargs):
        # llama.cpp's shape: a system prompt is sent as the first message.
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

    def system(content="Text tool protocol"):
        return {"role": "system", "content": content}, ProviderArtifactTraceProvenance(
            TraceProvenanceSource.RENDERED_SYSTEM, policy
        )

    async def send(resolution, route, pairs, actor=None, chain=None):
        request = _semantic_request(
            [row for row, _ in pairs],
            [item for _, item in pairs],
            policy,
            route=route,
            actor_id=actor,
            chain_id=chain,
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

    first = [({"role": "user", "content": "hello"}, hello)]
    if scenario != "not_a_slot":
        first.insert(0, system())  # the first turn went to llama.cpp
    actor, chain = new_opaque_id(), new_opaque_id()
    if first_fails:
        with pytest.raises(ChatRateLimitError):
            await send(deepseek, ConsoleRequestRoute.AGENT_FIRST, first, actor, chain)
    else:
        assert await send(
            deepseek, ConsoleRequestRoute.AGENT_FIRST, first, actor, chain
        ) == ["answered"]
    with database.transaction() as cursor:
        (segment_id,) = cursor.execute(
            "SELECT segment_id FROM console_trace_calls LIMIT 1"
        ).fetchone()
        first_tail = factory.repository.get_surface_tail(cursor, segment_id).sequence
    _, answered = _saved_message(database, conversation, "answered", sender="assistant")
    again_id, again = _saved_message(database, conversation, "again")
    history = [
        ({"role": "user", "content": "hello"}, hello),
        ({"role": "assistant", "content": "answered"}, answered),
        ({"role": "user", "content": "again"}, again),
    ]
    route, target, actor_id, chain_id = (
        ConsoleRequestRoute.AGENT_FIRST,
        deepseek,
        new_opaque_id(),
        new_opaque_id(),
    )
    turn = list(history)
    if scenario == "valid_fresh":
        route, actor_id, chain_id = ConsoleRequestRoute.FRESH, None, None
    elif scenario == "keeps_system_row":
        turn = [system("Native tool protocol"), *history]
    elif scenario in {"tool_loop", "retry"}:
        # The turn's next call on its own chain, after an answer or a failure.
        tools, tool_descriptors = _tool_rows(policy, "tool", 2)
        turn = [first[-1], *zip(tools, tool_descriptors, strict=True)]
        route, actor_id, chain_id = ConsoleRequestRoute.TOOL_LOOP, actor, chain
        if scenario == "retry":
            turn = [first[-1]]
    elif scenario == "fallback":
        turn = [first[-1]]
        route, target, actor_id = ConsoleRequestRoute.TOOL_LOOP, _GROQ, actor
    elif scenario == "bundled":
        # The slot may go, but a changed row is still the existing rules'.
        _, changed = _saved_message(database, conversation, "changed")
        turn = [({"role": "user", "content": "changed"}, changed), *history[1:]]
    before = list(asked)
    if scenario.startswith("binding"):
        service_type = trace.ConsoleTraceService
        persist, projection_of, binding_of, binding = (
            service_type.persist_request,
            service_type._surface_projection,
            trace._binding,
            [],
        )

        def forged_projection(self, cursor, segment_id, tail):
            projection = projection_of(self, cursor, segment_id, tail)
            if not binding or scenario != "binding_no_slot":
                return projection
            (sequence, key), *rest = projection.entries
            return trace._SurfaceProjection(
                projection.head_id,
                trace._ProjectionRoot(
                    None, base=((sequence, ("message", *key[1:])), *rest)
                ),
                lineage_domains=projection.lineage_domains,
            )

        def forged_binding(bundle, name):
            found = binding_of(bundle, name)
            if not binding or scenario != "binding_system_row" or name != (
                "messages_payload"
            ):
                return found
            return SimpleNamespace(value=(system()[0], *found.value))

        def bound(self, *args, **kwargs):
            binding.append(True)
            if scenario == "binding_not_first":
                kwargs["reserved_call"] = replace(
                    kwargs["reserved_call"], call_sequence=1
                )
            try:
                return persist(self, *args, **kwargs)
            except Exception as exc:
                refusals.append(str(exc))
                raise
            finally:
                binding.clear()

        monkeypatch.setattr(service_type, "_surface_projection", forged_projection)
        monkeypatch.setattr(trace, "_binding", forged_binding)
        monkeypatch.setattr(service_type, "persist_request", bound)
        with pytest.raises(TraceCallPersistenceError):
            await send(target, route, turn, actor_id, chain_id)
        assert asked == before
        assert refusals == ["rendered_system_slot_unavailable"]
        assert _markers(database) == []
        assert _reserved(database) == 1
        return
    if scenario in {"tool_loop", "retry", "fallback", "bundled"}:
        with pytest.raises(TraceCallPersistenceError):
            await send(target, route, turn, actor_id, chain_id)
        assert asked == before
        assert refusals == [
            # The fallback's own rule needs the failed call's header to
            # carry the slot; a first call's slot is the surface row itself.
            "trace_fallback_rewrite_unavailable"
            if scenario == "fallback"
            else "unsupported_surface_change"
        ]
        assert _markers(database) == []
        assert _reserved(database) == 0
        return
    assert await send(target, route, turn, actor_id, chain_id) == ["answered"]
    assert asked == before + [target.execution_key]
    assert refusals == []
    with database.transaction() as cursor:
        tail = factory.repository.get_surface_tail(cursor, segment_id)
        entries = factory.service._surface_projection(cursor, segment_id, tail).entries
    kinds = [key[0] for _, key in entries]
    (call,) = ConsoleTraceNativeReader(database).read_calls(again_id)
    assert call.capture.request["messages_payload"] == [row for row, _ in turn]
    if scenario in {"keeps_system_row", "not_a_slot"}:
        assert _markers(database) == []
        assert kinds[0] == ("rendered_system" if scenario == "keeps_system_row" else "message")
        assert len(entries) == len(turn)
        return
    # The removal comes first, directly after the tail; the call's rows
    # follow it, and the checkpoint the next call builds on names each row
    # at the sequence it was written.
    assert _markers(database) == [(first_tail + 1, route.value)]
    assert "rendered_system" not in kinds
    assert [s for s, _ in entries] == [1, first_tail + 2, first_tail + 3]
    checkpoint = factory.service.current_surface_checkpoint(
        segment_id, expected_head_id=tail.node_id
    )
    parent = factory.service._parent_capabilities[id(checkpoint)]
    assert [s for s, _ in parent.root.iter_entries()] == [s for s, _ in entries]
    assert [s for s, _ in parent.descriptors.iter_entries()] == [
        s for s, _ in entries
    ]
