"""Edit & resend, Regenerate and Continue are traced (TASK-42560).

ADR-337 amendment 2026-11-26 (owner-approved). These actions used to send
their reply with Capture Off. In a captured chat with project instructions
(AGENTS.md) the next message was then refused ("Trace capture could not
start"). Each action is now a traced call of its turn, and the message after
it is sent and traced.

Every journey below runs the real controller, agent bridge, gateway and
trace factory; only provider HTTP (llama.cpp) or the hosted handler is faked.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager

import pytest

from Tests.Chat.test_console_project_instruction_traces import (
    GUIDANCE,
    _project_console,
)
from Tests.Chat.test_console_trace_slot_drop import _fail_after_commit

# The Console bridge reads the tool gates through config, which keeps the
# collection-time profile only (TASK-41508; RecoveryRequired otherwise).
pytestmark = pytest.mark.bootstrap_profile

ACTIONS = ("edit", "regenerate", "continue")


@pytest.fixture(autouse=True)
def _close_idle_handles(monkeypatch, request):
    """Close the handles each test's databases opened on other threads.

    The lane runs this file under ``--storage-lease-guard`` (TASK-38986); as
    in ``test_console_trace_slot_drop.py``.
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


@asynccontextmanager
async def _console(
    tmp_path,
    monkeypatch,
    *,
    project=True,
    hosted=False,
    plain_path=False,
    mixins=False,
    replies=None,
    response_mode="complete",
):
    """A captured chat; ``app.wire()`` is the last request the provider got.

    ``replies`` (hosted only) is a list the test fills with the handler's next
    replies; an ``Exception`` in it is raised as the provider's failure.
    """
    async with _project_console(
        tmp_path,
        monkeypatch,
        source_name="AGENTS.md" if project else "NOTES.md",
        hosted=hosted,
        response_mode=response_mode,
    ) as app:
        sent = []
        queue = [] if replies is None else replies
        if hosted:

            def handler(*_args, **kwargs):
                reply = queue.pop(0) if queue else None
                if isinstance(reply, Exception):
                    raise reply
                sent.append(kwargs["messages_payload"])
                reply = reply or {
                    "role": "assistant",
                    "content": f"Hosted reply {len(sent)}",
                }
                return {"choices": [{"message": reply}]}

            monkeypatch.setattr(app.gateway, "_chat_api_call_fn", handler)
        app.wire = (
            (lambda: sent[-1])
            if hosted
            else (lambda: app.http_payloads[-1]["messages"])
        )
        app.sent = sent if hosted else app.http_payloads
        if plain_path:
            app.controller._agent_runtime_enabled = False
        if mixins:
            from Tests.Chat.test_console_prompt_mixin_retry import select

            select(app, monkeypatch, session_id=app.session.id)
        yield app


def _calls(app):
    """Every trace call, oldest first: (route, state, turn)."""
    with app.db.transaction() as cursor:
        return [
            tuple(row)
            for row in cursor.execute(
                "SELECT route_identity, state, turn_id FROM console_trace_calls "
                "ORDER BY rowid"
            ).fetchall()
        ]


def _rows(rows):
    """Roles and contents; the project file's row is compared by its marker."""
    return [
        (
            row.get("role"),
            "<project>"
            if GUIDANCE in str(row.get("content"))
            else json.dumps(row.get("content"), sort_keys=True),
            bool(row.get("tool_calls")),
        )
        for row in rows
    ]


def _traced(app):
    """The latest call's rebuilt request rows."""
    from tldw_chatbook.Chat.console_trace_native_reader import (
        ConsoleTraceNativeReader,
    )

    with app.db.transaction() as cursor:
        (turn,) = cursor.execute(
            "SELECT turn_id FROM console_trace_calls ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
    calls = ConsoleTraceNativeReader(app.db).read_calls(turn)
    return calls[-1].capture.request["messages_payload"]


def _last(app, role):
    return [
        row
        for row in app.store.messages_for_session(app.session.id)
        if row.role.value == role
    ][-1]


async def _act(app, action, *, first=False):
    controller = app.controller
    messages = app.store.messages_for_session(app.session.id)
    users = [row for row in messages if row.role.value == "user"]
    replies = [row for row in messages if row.role.value == "assistant"]
    if action == "edit":
        user = users[0] if first else users[-1]
        return await controller.edit_and_resend_message(user.id, "Edited, please")
    if action == "regenerate":
        return await controller.regenerate_message(
            (replies[0] if first else replies[-1]).id
        )
    return await controller.continue_from_message(replies[-1].id)


async def _send(app, text):
    result = await app.controller.submit_draft(text, session_id=app.session.id)
    assert result.accepted
    return result


def _assert_traced_as_sent(app, *, every_request=True):
    """The latest call rebuilds the last request; with ``every_request``,
    each request the provider received has its own trace call."""
    assert app.reservation_errors == []
    assert app.controller.run_state.status.value == "completed"
    assert _rows(_traced(app)) == _rows(app.wire())
    if every_request:
        assert len(_calls(app)) == len(app.sent)


@pytest.mark.usefixtures("shut_down_console_controllers")
@pytest.mark.parametrize("hosted", [False, True], ids=["llama_cpp", "hosted"])
@pytest.mark.parametrize("shape", ["plain", "agents_md", "mixins"])
@pytest.mark.parametrize("action", ACTIONS)
async def test_an_action_is_traced_and_so_is_the_next_message(
    tmp_path, monkeypatch, action, shape, hosted
):
    """The probe's journey: two messages, the action, then one more.

    Before the change the action's reply had no trace call, and with
    AGENTS.md the message after it was refused.
    """
    async with _console(
        tmp_path,
        monkeypatch,
        project=shape == "agents_md",
        hosted=hosted,
        mixins=shape == "mixins",
    ) as app:
        await _send(app, "Hello")
        await _send(app, "And now?")
        before = _calls(app)

        result = await _act(app, action)

        assert result.accepted
        calls = _calls(app)[len(before) :]
        assert [call[:2] for call in calls] == [("agent_first", "complete")]
        _assert_traced_as_sent(app)
        # Edit & resend: the edited user; Regenerate and Continue: the user
        # the replaced or continued reply answers.
        assert calls[0][2] == _last(app, "user").persisted_message_id
        if shape == "agents_md":
            assert GUIDANCE in str(app.wire()[-1]["content"])
        if shape == "mixins":
            assert "body A" in json.dumps(app.wire())
        if action == "continue":
            assert app.wire()[-1 - (shape == "agents_md")]["content"].endswith(
                "Continue and extend the selected message."
            )
        sent = len(app.sent)

        await _send(app, "One more?")

        assert len(app.sent) == sent + 1
        assert [call[:2] for call in _calls(app)[len(before) + 1 :]] == [
            ("agent_first", "complete")
        ]
        _assert_traced_as_sent(app)


@pytest.mark.usefixtures("shut_down_console_controllers")
@pytest.mark.parametrize("action", ACTIONS)
async def test_an_action_on_the_plain_provider_path(tmp_path, monkeypatch, action):
    """Character chats and ``agent_runtime = false``: the call's own route.
    With mix-ins, so the next send restores the action's transformed user."""
    async with _console(tmp_path, monkeypatch, plain_path=True, mixins=True) as app:
        await _send(app, "Hello")
        await _send(app, "And now?")
        before = _calls(app)

        await _act(app, action)

        assert [call[:2] for call in _calls(app)[len(before) :]] == [
            (action, "complete")
        ]
        _assert_traced_as_sent(app)

        await _send(app, "One more?")

        assert [call[:2] for call in _calls(app)[len(before) + 1 :]] == [
            ("fresh", "complete")
        ]
        _assert_traced_as_sent(app)


@pytest.mark.usefixtures("shut_down_console_controllers")
@pytest.mark.parametrize("action", ["edit", "continue"])
async def test_an_action_is_traced_after_a_trace_runtime_restart(
    tmp_path, monkeypatch, action
):
    """A cold trace factory plans the action and the next message from the
    ledger alone (no in-process surface checkpoint)."""
    async with _console(tmp_path, monkeypatch) as app:
        await _send(app, "Hello")
        await _send(app, "And now?")
        app.restart_factory()
        await _act(app, action)
        _assert_traced_as_sent(app)
        app.restart_factory()
        await _send(app, "One more?")
        _assert_traced_as_sent(app)


@pytest.mark.usefixtures("shut_down_console_controllers")
async def test_an_action_is_traced_when_llamacpp_resends_its_stream(
    tmp_path, monkeypatch
):
    """llama.cpp's empty stream is resent without streaming (task-19324): the
    resend carries the action and a Continue's turn, and its reply ends the
    turn."""
    action = "continue"
    async with _console(tmp_path, monkeypatch, response_mode="fallback") as app:
        await _send(app, "Hello")
        await _send(app, "And now?")
        before = _calls(app)

        await _act(app, action)

        assert [call[:2] for call in _calls(app)[len(before) :]] == [
            ("agent_first", "error"),
            ("llama_fallback", "complete"),
        ]
        _assert_traced_as_sent(app)
        await _send(app, "One more?")
        _assert_traced_as_sent(app)


@pytest.mark.usefixtures("shut_down_console_controllers")
async def test_an_action_on_an_earlier_message(tmp_path, monkeypatch):
    """A Regenerate of the first reply replaces the later turns from its own
    user row (the matched prefix ran past it)."""
    action = "regenerate"
    async with _console(tmp_path, monkeypatch) as app:
        await _send(app, "Hello")
        await _send(app, "And now?")
        await _act(app, action, first=True)
        _assert_traced_as_sent(app)
        assert sum("And now?" in json.dumps(row) for row in app.wire()) == 0
        await _send(app, "One more?")
        _assert_traced_as_sent(app)


@pytest.mark.usefixtures("shut_down_console_controllers")
async def test_resend_of_a_turn_without_a_reply_is_traced(tmp_path, monkeypatch):
    """Resend re-runs a broken turn from its user message through Continue's
    body, so its call is traced like a Continue of that user; Resend of a
    failed reply is Retry's and stays out of this test."""
    from tldw_chatbook.Chat.console_turn_resend import resend_turn

    async with _console(tmp_path, monkeypatch) as app:
        await _send(app, "Hello")
        await _send(app, "And now?")
        app.store.delete_message(_last(app, "assistant").id)
        before = _calls(app)

        result = await resend_turn(app.controller, _last(app, "user").id)

        assert result.accepted
        assert [call[:2] for call in _calls(app)[len(before) :]] == [
            ("agent_first", "complete")
        ]
        _assert_traced_as_sent(app)
        await _send(app, "One more?")
        _assert_traced_as_sent(app)


@pytest.mark.usefixtures("shut_down_console_controllers")
@pytest.mark.parametrize(
    "steps",
    [("continue", "continue"), ("continue", "regenerate")],
    ids="-".join,
)
async def test_actions_in_a_row(tmp_path, monkeypatch, steps):
    """A Continue of a continued reply, a Regenerate of one (its request also
    ends with the instruction), and repeated actions."""
    async with _console(tmp_path, monkeypatch, mixins=True) as app:
        await _send(app, "Hello")
        for action in steps:
            before = _calls(app)
            await _act(app, action)
            assert [call[:2] for call in _calls(app)[len(before) :]] == [
                ("agent_first", "complete")
            ]
            _assert_traced_as_sent(app)
        await _send(app, "One more?")
        _assert_traced_as_sent(app)


@pytest.mark.usefixtures("shut_down_console_controllers")
async def test_an_action_that_calls_a_tool(tmp_path, monkeypatch):
    """A Continue's run goes on with a tool-loop call of the same turn, which
    carries the continued reply after the user."""
    action = "continue"
    from Tests.Chat.test_console_failed_call_retry_ownership import (
        _NATIVE_TOOL_CALL,
    )

    replies = []
    async with _console(tmp_path, monkeypatch, hosted=True, replies=replies) as app:
        await _send(app, "Hello")
        before = _calls(app)
        replies.append(_NATIVE_TOOL_CALL)

        await _act(app, action)

        assert [call[:2] for call in _calls(app)[len(before) :]] == [
            ("agent_first", "complete"),
            ("tool_loop", "complete"),
        ]
        assert "tool" in [row["role"] for row in app.wire()]
        _assert_traced_as_sent(app)
        await _send(app, "One more?")
        _assert_traced_as_sent(app)


@pytest.mark.usefixtures("shut_down_console_controllers")
@pytest.mark.parametrize("plain_path", [False, True], ids=["agent", "plain_path"])
@pytest.mark.parametrize("action", ACTIONS)
async def test_after_a_failed_action_the_next_message_is_sent(
    tmp_path, monkeypatch, action, plain_path
):
    """The provider fails the action's reply; the next message still goes out.

    A failed Regenerate leaves the original reply on the path, and a failed
    Continue or Edit & resend leaves its empty failed reply, which the next
    request skips.
    """
    from tldw_chatbook.Provider_Contract import ChatProviderError

    replies = []
    async with _console(
        tmp_path, monkeypatch, hosted=True, plain_path=plain_path, replies=replies
    ) as app:
        await _send(app, "Hello")
        await _send(app, "And now?")
        before = _calls(app)
        outage = ChatProviderError(
            provider="openai", message="fixture outage", status_code=503
        )
        replies.extend([outage] * 3)

        await _act(app, action)

        failed = _calls(app)[len(before) :]
        assert failed and {call[1] for call in failed} == {"error"}
        replies.clear()

        await _send(app, "One more?")

        assert app.reservation_errors == []
        assert _calls(app)[-1][1] == "complete"
        _assert_traced_as_sent(app, every_request=False)


@pytest.mark.usefixtures("shut_down_console_controllers")
async def test_capture_off_actions_stay_untraced(tmp_path, monkeypatch):
    """Under Capture Off nothing changes: the actions send without a call."""
    async with _console(tmp_path, monkeypatch) as app:
        await _send(app, "Hello")
        await _send(app, "And now?")
        before = _calls(app)
        from tldw_chatbook.Chat import console_action_capture
        from tldw_chatbook.Chat.console_trace_provenance import (
            ConsoleTraceCaptureMode,
        )

        monkeypatch.setattr(
            console_action_capture,
            "_capture_mode",
            lambda *_args: (ConsoleTraceCaptureMode.CAPTURE_OFF, None),
        )
        sent = len(app.sent)
        for action in ACTIONS:
            result = await _act(app, action)
            assert result.accepted
        assert len(app.sent) == sent + 3
        assert _calls(app) == before


@pytest.mark.usefixtures("shut_down_console_controllers")
@pytest.mark.parametrize("action", ACTIONS)
async def test_an_action_uses_up_the_one_time_capture_choice(
    tmp_path, monkeypatch, action
):
    """A one-time Capture choice covers one send: the action that honoured it
    uses it up, as a normal send would, so the next plain send is back to the
    chat's own capture mode (here On)."""
    async with _console(tmp_path, monkeypatch, project=False) as app:
        await _send(app, "Hello")
        await _send(app, "And now?")
        controller, session = app.controller, app.session.id
        controller.set_next_trace_privacy(
            session,
            capture_enabled=False,
            pii_redaction_enabled=None,
            expected_policy_revision=controller.capture_policy_snapshot(
                session
            ).policy_revision,
        )
        before = _calls(app)

        await _act(app, action)

        assert _calls(app) == before, "the one-time choice (Off) was honoured"
        assert controller.capture_policy_snapshot(session).next_capture_enabled is None
        await _send(app, "One more?")
        assert [call[:2] for call in _calls(app)[len(before) :]] == [
            ("agent_first", "complete")
        ]


@pytest.mark.usefixtures("shut_down_console_controllers")
async def test_a_temporary_chat_with_capture_on_refuses_before_any_row(
    tmp_path, monkeypatch
):
    """ADR-337 decision 15: no durable call before Save, so each action is
    refused with copy before it creates a reply (or an edited message); only
    the usual refusal note is added."""
    from tldw_chatbook.Chat.console_action_capture import TEMPORARY_CAPTURE_COPY

    async with _console(tmp_path, monkeypatch) as app:
        await _send(app, "Hello")
        monkeypatch.setattr(
            app.store, "session_is_ephemeral", lambda _session_id: True
        )

        def turns():
            return [
                row.id
                for row in app.store.all_messages_for_session(app.session.id)
                if row.role.value != "system"
            ]

        rows = turns()
        sent = len(app.sent)
        for action in ACTIONS:
            result = await _act(app, action)

            assert not result.accepted
            assert TEMPORARY_CAPTURE_COPY in result.visible_copy
            assert len(app.sent) == sent
            assert turns() == rows
            assert _last(app, "system").content == TEMPORARY_CAPTURE_COPY


@pytest.mark.usefixtures("shut_down_console_controllers")
async def test_a_refused_plain_path_action_fails_its_reply(tmp_path, monkeypatch):
    """A refusal before dispatch reaches no provider; the reply fails with
    copy that says why, and the chat can still send."""
    from tldw_chatbook.Chat.console_action_capture import TRACE_REFUSED_COPY
    from tldw_chatbook.Chat.console_trace_errors import TraceCallPersistenceError

    async with _console(tmp_path, monkeypatch, plain_path=True) as app:
        await _send(app, "Hello")
        await _send(app, "And now?")
        sent = len(app.sent)
        stream = app.controller._stream_assistant_response

        async def refuse(**kwargs):
            if kwargs.get("trace_request") is not None:
                raise TraceCallPersistenceError(reservation_status="not_established")
            return await stream(**kwargs)

        monkeypatch.setattr(app.controller, "_stream_assistant_response", refuse)

        result = await _act(app, "edit")

        assert result.visible_copy == TRACE_REFUSED_COPY
        assert len(app.sent) == sent
        assert _last(app, "assistant").status == "failed"
        assert app.controller.run_state.status.value == "failed"
        monkeypatch.setattr(app.controller, "_stream_assistant_response", stream)
        await _send(app, "One more?")
        assert len(app.sent) == sent + 1


# --- Hard negatives: each new rule refuses what it does not name -----------


from Tests.Chat.test_console_trace_runtime import (  # noqa: E402
    _semantic_request,
)
from Tests.Chat.test_console_trace_runtime import (  # noqa: E402
    make_database as _make_database_fixture,
)
from Tests.Chat.test_console_trace_runtime import (  # noqa: E402
    make_gateway as _make_gateway_fixture,
)

make_database = _make_database_fixture
make_gateway = _make_gateway_fixture


def _message(database, conversation, content, *, sender="user", parent=None):
    """Save one message under ``parent`` and return (id, saved descriptor)."""
    from tldw_chatbook.Chat.console_trace_provenance import (
        SavedRevisionTraceProvenance,
    )

    message_id = database.add_message(
        {
            "conversation_id": conversation,
            "sender": sender,
            "content": content,
            "parent_message_id": parent,
        }
    )
    with database.transaction() as cursor:
        (revision,) = cursor.execute(
            "SELECT revision_id FROM console_trace_semantic_revisions "
            "WHERE source_message_id = ? ORDER BY revision_sequence DESC LIMIT 1",
            (message_id,),
        ).fetchone()
    return message_id, SavedRevisionTraceProvenance(str(revision))


@pytest.fixture
def ledger(tmp_path, make_database, make_gateway):
    """A trace factory behind a real gateway; the provider is faked."""
    from types import SimpleNamespace

    from tldw_chatbook.Chat.console_provider_gateway import (
        ConsoleProviderResolution,
    )
    from tldw_chatbook.Chat.console_trace_models import (
        FrozenTracePolicy,
        new_opaque_id,
    )
    from tldw_chatbook.Chat.console_trace_provenance import (
        ConsoleTraceCaptureMode,
        ProviderArtifactTraceProvenance,
        TraceProvenanceSource,
    )
    from tldw_chatbook.Chat.console_trace_runtime import ConsoleTraceBoundaryFactory

    database = make_database(tmp_path / "actions.sqlite", "actions")
    conversation = database.add_conversation({"title": "actions"})
    policy = FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None)
    factory = ConsoleTraceBoundaryFactory(database)
    refusals, asked = [], []

    def boundary(*args, **kwargs):
        try:
            return factory(*args, **kwargs)
        except Exception as exc:
            refusals.append(str(exc))
            raise

    def provider(**kwargs):
        asked.append(kwargs["messages_payload"])
        return {"choices": [{"message": {"content": "answered"}}]}

    gateway = make_gateway(
        chat_api_call_fn=provider,
        trace_call_boundary_factory=boundary,
        safe_error_copy=lambda _provider, _exc: "fixture provider failure",
    )
    resolution = ConsoleProviderResolution(
        ready=True,
        provider="deepseek",
        execution_key="deepseek",
        model="deepseek-chat",
        base_url="https://api.deepseek.com",
        api_key="fixture",
        streaming=False,
    )

    async def send(route, pairs):
        request = _semantic_request(
            [row for row, _ in pairs], [item for _, item in pairs], policy, route=route
        )
        return [
            item
            async for item in gateway.stream_chat(
                resolution,
                request,
                route=route,
                capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
            )
        ]

    async def send_agent(route, pairs, *, action=None):
        """An agent run's first call, naming ``action`` as the bridge does."""
        from tldw_chatbook.Chat.console_prepared_request import build_console_request
        from tldw_chatbook.Chat.console_trace_provenance import (
            request_route_provenance,
        )

        actor, chain = new_opaque_id(), new_opaque_id()
        request = build_console_request(
            [row for row, _ in pairs],
            message_provenance=tuple(item for _, item in pairs),
            memory_provenance=(),
            mandatory_provenance=(),
            tool_provenance=(),
            metadata_provenance=(
                request_route_provenance(
                    route, actor_id=actor, chain_id=chain, action=action
                ),
            ),
            capture_policy=policy,
            capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
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

    def instruction(content="Continue and extend the selected message."):
        return {"role": "user", "content": content}, ProviderArtifactTraceProvenance(
            TraceProvenanceSource.ACTIVE_REQUEST, policy
        )

    return SimpleNamespace(
        database=database,
        conversation=conversation,
        policy=policy,
        send=send,
        send_agent=send_agent,
        refusals=refusals,
        asked=asked,
        instruction=instruction,
        message=lambda *args, **kwargs: _message(database, conversation, *args, **kwargs),
        calls=lambda: [
            row[0]
            for row in database.get_connection()
            .execute("SELECT route_identity FROM console_trace_calls ORDER BY rowid")
            .fetchall()
        ],
    )


@pytest.mark.parametrize(
    "shape",
    [
        "valid",
        "reply_from_another_turn",
        "user_row_as_reply",
        "no_reply_before_instruction",
        "fresh_route",
    ],
)
async def test_a_continue_is_admitted_only_after_its_own_turns_reply(ledger, shape):
    """The Continue rules, one shape each, below the controller.

    A Continue's request ends with an instruction after the saved replies it
    extends, which must lead directly back to the user its call belongs to.
    A reply from another branch, a user row where a reply goes, an
    instruction with no reply before it, or the same rows on a route that
    names no action are refused before the provider is asked.
    """
    from tldw_chatbook.Chat.console_trace_errors import TraceCallPersistenceError
    from tldw_chatbook.Chat.console_trace_provenance import ConsoleRequestRoute

    hello_id, hello = ledger.message("hello")
    assert await ledger.send(
        ConsoleRequestRoute.FRESH, [({"role": "user", "content": "hello"}, hello)]
    ) == ["answered"]
    other_id, _other = ledger.message("elsewhere")
    parent = other_id if shape == "reply_from_another_turn" else hello_id
    reply_role = "user" if shape == "user_row_as_reply" else "assistant"
    _, reply = ledger.message("partial", sender=reply_role, parent=parent)
    rows = [({"role": "user", "content": "hello"}, hello)]
    if shape != "no_reply_before_instruction":
        rows.append(({"role": reply_role, "content": "partial"}, reply))
    rows.append(ledger.instruction())
    route = (
        ConsoleRequestRoute.FRESH if shape == "fresh_route" else ConsoleRequestRoute.CONTINUE
    )
    asked = len(ledger.asked)

    if shape == "valid":
        assert await ledger.send(route, rows) == ["answered"]
        assert ledger.calls() == ["fresh", "continue"]
        return
    with pytest.raises(TraceCallPersistenceError):
        await ledger.send(route, rows)
    assert len(ledger.asked) == asked
    assert ledger.calls() == ["fresh"]


@pytest.mark.parametrize("agent", [False, True], ids=["plain_route", "agent_run"])
@pytest.mark.parametrize("anchored", [True, False], ids=["anchored", "unanchored"])
async def test_an_edit_replaces_only_a_branch_off_its_retained_rows(
    ledger, anchored, agent
):
    """An Edit & resend's call re-records its turn from the edited user, which
    must hang off the last retained row; a one-row replacement needs that
    proof too. On the agent path the earlier calls leave a surface checkpoint
    for the route, so only the proof refuses the stray branch."""
    from tldw_chatbook.Chat.console_trace_errors import TraceCallPersistenceError
    from tldw_chatbook.Chat.console_trace_provenance import ConsoleRequestRoute

    send = ledger.send_agent if agent else ledger.send
    first = ConsoleRequestRoute.AGENT_FIRST if agent else ConsoleRequestRoute.FRESH
    hello_id, hello = ledger.message("hello")
    await send(first, [({"role": "user", "content": "hello"}, hello)])
    answer_id, answer = ledger.message("answered", sender="assistant", parent=hello_id)
    again_id, again = ledger.message("again", parent=answer_id)
    history = [
        ({"role": "user", "content": "hello"}, hello),
        ({"role": "assistant", "content": "answered"}, answer),
    ]
    await send(first, [*history, ({"role": "user", "content": "again"}, again)])
    stray_id, _ = ledger.message("stray")
    _, edited = ledger.message(
        "edited", parent=answer_id if anchored else stray_id
    )
    rows = [*history, ({"role": "user", "content": "edited"}, edited)]
    asked = len(ledger.asked)

    edit = (
        (lambda: send(first, rows, action=ConsoleRequestRoute.EDIT))
        if agent
        else (lambda: send(ConsoleRequestRoute.EDIT, rows))
    )
    if anchored:
        assert await edit() == ["answered"]
        return
    with pytest.raises(TraceCallPersistenceError):
        await edit()
    assert len(ledger.asked) == asked


@pytest.mark.parametrize(
    "fields",
    [
        {"route": "fresh", "action": "edit"},
        {"route": "agent_first", "action": "retry"},
        {"route": "agent_first", "action": "impersonate"},
        {"route": "edit", "action": "edit"},
    ],
    ids=lambda fields: f"{fields['route']}-{fields['action']}",
)
def test_only_an_agent_or_resend_route_names_an_action(fields):
    """The route descriptor names Edit & resend, Regenerate or Continue only,
    and only on a call of an action's run (agent loop or llama.cpp resend)."""
    from tldw_chatbook.Chat.console_trace_models import new_opaque_id
    from tldw_chatbook.Chat.console_trace_provenance import (
        ConsoleRequestRoute,
        TraceProvenanceAlignmentError,
        request_route_provenance,
    )

    route = ConsoleRequestRoute(fields["route"])
    actor = (
        {"actor_id": new_opaque_id(), "chain_id": new_opaque_id()}
        if route is ConsoleRequestRoute.AGENT_FIRST
        else {}
    )
    with pytest.raises(TraceProvenanceAlignmentError):
        request_route_provenance(
            route, action=ConsoleRequestRoute(fields["action"]), **actor
        )
    assert (
        request_route_provenance(
            ConsoleRequestRoute.AGENT_FIRST,
            actor_id=new_opaque_id(),
            chain_id=new_opaque_id(),
            action=ConsoleRequestRoute.CONTINUE,
        ).action
        is ConsoleRequestRoute.CONTINUE
    )


def test_a_rewritten_tail_of_an_ordinary_send_never_ends_with_project_context():
    """Project context in a rewritten tail is only an action's, or the send
    after an action's run: without that evidence the builder refuses."""
    from tldw_chatbook.Chat.console_trace_models import (
        FrozenTracePolicy,
        new_opaque_id,
    )
    from tldw_chatbook.Chat.console_trace_provenance import (
        ProviderArtifactTraceProvenance,
        SavedRevisionTraceProvenance,
        TraceProvenanceSource,
    )
    from tldw_chatbook.Chat.console_trace_rewritten_tail import (
        build_rewritten_tail_witness,
    )

    policy = FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None)
    incoming = (
        SavedRevisionTraceProvenance(new_opaque_id()),
        SavedRevisionTraceProvenance(new_opaque_id()),
        ProviderArtifactTraceProvenance(TraceProvenanceSource.PROJECT_INSTRUCTION, policy),
    )

    class Owner:
        conversation_id = "conversation"

    class Repository:
        def get_owner(self, _cursor, _owner_id):
            return Owner()

    assert (
        build_rewritten_tail_witness(
            None,
            Repository(),
            owner_id="owner",
            retained=(),
            incoming=incoming,
            current_turn_id="user",
            policy_id=policy.policy_id,
        )
        is None
    )


@pytest.mark.usefixtures("shut_down_console_controllers")
async def test_a_reply_the_failed_run_did_not_bear_is_refused(tmp_path, monkeypatch):
    """Retry stays TASK-42555's (AC #4). A Continue's reply fails; a Retry
    (still Capture Off) then writes it. The next message's request carries a
    reply the traced run never bore, so it is refused before the provider is
    asked."""
    from tldw_chatbook.Provider_Contract import ChatProviderError

    replies = []
    async with _console(tmp_path, monkeypatch, hosted=True, replies=replies) as app:
        await _send(app, "Hello")
        replies.extend(
            [ChatProviderError(provider="openai", message="outage", status_code=503)]
            * 3
        )
        await _act(app, "continue")
        replies.clear()
        failed = _last(app, "assistant")
        assert failed.status == "failed"
        await app.controller.retry_message(failed.id)
        assert _last(app, "assistant").content
        sent = len(app.sent)

        await app.controller.submit_draft("One more?", session_id=app.session.id)

        assert len(app.sent) == sent
        assert app.reservation_errors[-1] == "unsupported_surface_change"


from Tests.Chat.test_console_trace_current_turn_transforms import (  # noqa: E402,F401
    trace_harness,
)
from Tests.Chat.test_console_trace_rewritten_history import (  # noqa: E402,F401
    SESSION,
    _assert_captured,
    _four_long_turns,
    harness,
)
from Tests.Chat.test_console_trace_rewritten_history import (  # noqa: E402
    _send as _history_send,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("harness", ["agent"], indirect=True)
async def test_compact_now_after_a_regenerate_keeps_the_next_send(harness):  # noqa: F811
    """The send after an action's run that also drops compacted turns starts
    its replacement before the run's user row: it is planned by the ordinary
    rewritten tail (anchored at the memory), as before the change."""
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
    await _four_long_turns(harness)
    reply = [
        row
        for row in harness.store.messages_for_session(SESSION)
        if row.role.value == "assistant"
    ][-1]
    assert (await harness.controller.regenerate_message(reply.id)).accepted
    assert harness.failures == []
    succeeded, copy = await harness.controller.compact_context_now(SESSION)
    assert succeeded is True, copy

    result, sent, reserved = await _history_send(harness, "after compaction")

    _assert_captured(harness, result, sent, reserved)
    assert harness.rechecks[-1].memory_anchor is True


@pytest.mark.usefixtures("shut_down_console_controllers")
@pytest.mark.parametrize("action", ["regenerate", "continue"])
async def test_an_actions_committed_rewrite_is_read_back_after_a_post_commit_failure(
    tmp_path, monkeypatch, action
):
    """The action's re-record is a replacement plus appends; when the bind's
    commit lands but its exit raises, the gateway reads the write back as
    durable and the reply goes out once, traced as sent."""
    async with _console(tmp_path, monkeypatch) as app:
        await _send(app, "Hello")
        await _send(app, "And now?")
        _state, outcomes = _fail_after_commit(monkeypatch)
        sent = len(app.sent)

        await _act(app, action)

        assert outcomes == [("durable", True, False)]
        assert len(app.sent) == sent + 1
        _assert_traced_as_sent(app)
        await _send(app, "One more?")
        _assert_traced_as_sent(app)
