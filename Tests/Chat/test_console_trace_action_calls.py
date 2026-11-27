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


def _one_time_capture(app, enabled):
    """Arm the one-time Next send capture choice."""
    controller, session = app.controller, app.session.id
    controller.set_next_trace_privacy(
        session,
        capture_enabled=enabled,
        pii_redaction_enabled=None,
        expected_policy_revision=controller.capture_policy_snapshot(
            session
        ).policy_revision,
    )


def _chat_capture_off(app):
    """Turn this chat's own Capture choice off."""
    controller, session = app.controller, app.session.id
    app.store.replace_session_trace_privacy_override(
        session,
        capture_enabled=False,
        pii_redaction_enabled=None,
        expected_policy_revision=controller.capture_policy_snapshot(
            session
        ).policy_revision,
    )


def _next_capture(app):
    return app.controller.capture_policy_snapshot(app.session.id).next_capture_enabled


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
@pytest.mark.parametrize("project", [False, True], ids=["plain", "agents_md"])
async def test_an_edit_and_resend_of_an_earlier_message(tmp_path, monkeypatch, project):
    """An Edit & resend of the first message answers on a new branch off the
    first turn; its call re-records the turn, and the next message is sent."""
    async with _console(tmp_path, monkeypatch, project=project) as app:
        await _send(app, "Hello")
        await _send(app, "And now?")
        before = _calls(app)

        await _act(app, "edit", first=True)

        assert [call[:2] for call in _calls(app)[len(before) :]] == [
            ("agent_first", "complete")
        ]
        _assert_traced_as_sent(app)
        assert sum("And now?" in json.dumps(row) for row in app.wire()) == 0
        await _send(app, "One more?")
        _assert_traced_as_sent(app)


@pytest.mark.usefixtures("shut_down_console_controllers")
@pytest.mark.parametrize("project", [False, True], ids=["plain", "agents_md"])
@pytest.mark.parametrize("target", ["reply", "user"])
async def test_a_continue_of_an_earlier_message_keeps_capture_off(
    tmp_path, monkeypatch, target, project
):
    """A Continue of an earlier message sends only the rows through it, but
    its reply is appended at the end of the chat, so it keeps Capture Off and
    the next message is as before this change: sent without project
    instructions, refused with them (TASK-42586)."""
    async with _console(tmp_path, monkeypatch, project=project) as app:
        await _send(app, "Hello")
        await _send(app, "And now?")
        role = "assistant" if target == "reply" else "user"
        message = next(
            row
            for row in app.store.messages_for_session(app.session.id)
            if row.role.value == role
        )
        before, sent = _calls(app), len(app.sent)

        result = await app.controller.continue_from_message(message.id)

        assert result.accepted
        assert len(app.sent) == sent + 1
        assert _calls(app) == before
        await app.controller.submit_draft("One more?", session_id=app.session.id)
        if project:
            assert app.reservation_errors == ["unsupported_surface_change"]
        else:
            _assert_traced_as_sent(app, every_request=False)


@pytest.mark.usefixtures("shut_down_console_controllers")
async def test_a_continue_of_an_earlier_message_with_mixins_leaves_no_run_open(
    tmp_path, monkeypatch
):
    """The mix-ins have no anchor for a Continue of an earlier user message;
    the reply path refuses it as before, and the chat can still send."""
    from tldw_chatbook.Chat.console_action_capture import MIXIN_UNAVAILABLE_COPY

    async with _console(tmp_path, monkeypatch, project=False, mixins=True) as app:
        await _send(app, "Hello")
        await _send(app, "And now?")
        first = next(
            row
            for row in app.store.messages_for_session(app.session.id)
            if row.role.value == "user"
        )
        sent = len(app.sent)

        result = await app.controller.continue_from_message(first.id)

        assert result.visible_copy == MIXIN_UNAVAILABLE_COPY
        assert len(app.sent) == sent
        assert app.controller.run_state.status.value == "blocked"
        await _send(app, "One more?")
        _assert_traced_as_sent(app, every_request=False)


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
@pytest.mark.parametrize("mixins", [False, True], ids=["", "mixins"])
@pytest.mark.parametrize("plain_path", [False, True], ids=["agent", "plain_path"])
@pytest.mark.parametrize("action", ACTIONS)
async def test_after_a_failed_action_the_next_message_is_sent(
    tmp_path, monkeypatch, action, plain_path, mixins
):
    """The provider fails the action's reply; the next message still goes out.

    A failed Regenerate leaves the original reply on the path, and a failed
    Continue or Edit & resend leaves its empty failed reply, which the next
    request skips. With mix-ins the action's user was a transform, so the
    next request restores its saved source, then skips the failed reply.
    """
    from tldw_chatbook.Provider_Contract import ChatProviderError

    replies = []
    async with _console(
        tmp_path,
        monkeypatch,
        hosted=True,
        plain_path=plain_path,
        mixins=mixins,
        replies=replies,
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
    """In a chat with Capture off nothing changes: the actions send without a
    call."""
    async with _console(tmp_path, monkeypatch) as app:
        _chat_capture_off(app)
        _one_time_capture(app, None)  # the harness arms Capture On for one send
        await _send(app, "Hello")
        await _send(app, "And now?")
        sent = len(app.sent)
        for action in ACTIONS:
            result = await _act(app, action)
            assert result.accepted
        assert len(app.sent) == sent + 3
        assert _calls(app) == []


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
        _one_time_capture(app, False)
        before = _calls(app)

        await _act(app, action)

        assert _calls(app) == before, "the one-time choice (Off) was honoured"
        assert _next_capture(app) is None
        await _send(app, "One more?")
        assert [call[:2] for call in _calls(app)[len(before) :]] == [
            ("agent_first", "complete")
        ]


@pytest.mark.usefixtures("shut_down_console_controllers")
@pytest.mark.parametrize("action", ACTIONS)
async def test_a_one_time_capture_on_choice_traces_an_action_in_a_capture_off_chat(
    tmp_path, monkeypatch, action
):
    """In a chat with Capture off, a one-time Capture On traces the action and
    is used up; the next send is not traced."""
    async with _console(tmp_path, monkeypatch) as app:
        await _send(app, "Hello")
        await _send(app, "And now?")
        _chat_capture_off(app)
        _one_time_capture(app, True)
        before = _calls(app)

        await _act(app, action)

        assert [call[:2] for call in _calls(app)[len(before) :]] == [
            ("agent_first", "complete")
        ]
        assert _next_capture(app) is None
        traced = _calls(app)
        await _send(app, "One more?")
        assert _calls(app) == traced


@pytest.mark.usefixtures("shut_down_console_controllers")
@pytest.mark.parametrize("action", ["edit", "regenerate"])
async def test_an_action_sent_with_capture_off_leaves_a_one_time_capture_on_choice(
    tmp_path, monkeypatch, action
):
    """An action that sends the pinned prefill keeps Capture Off whatever the
    choice (TASK-42583), so a one-time Capture On waits for the next send."""
    async with _console(tmp_path, monkeypatch, project=False) as app:
        await _send(app, "Hello")
        await _send(app, "And now?")
        app.store.set_session_pinned_prefill(app.session.id, "Sure:")
        _chat_capture_off(app)
        _one_time_capture(app, True)
        before, sent = _calls(app), len(app.sent)

        await _act(app, action)

        assert len(app.sent) == sent + 1
        assert _calls(app) == before
        assert _next_capture(app) is True


@pytest.mark.usefixtures("shut_down_console_controllers")
@pytest.mark.parametrize("action", ACTIONS)
async def test_a_capture_off_edit_after_a_traced_action_keeps_the_next_sends(
    tmp_path, monkeypatch, action
):
    """ADR-337: a Capture Off Edit & resend, then a captured send, is
    unchanged. Here the Edit replaces the user row a traced action
    re-recorded, so the next sends prove an ordinary rewritten tail."""
    async with _console(tmp_path, monkeypatch, project=False) as app:
        await _send(app, "Hello")
        await _send(app, "And now?")
        await _act(app, action)
        _one_time_capture(app, False)
        before = _calls(app)

        await _act(app, "edit")

        assert _calls(app) == before
        for text in ("One more?", "And again?"):
            await _send(app, text)
            _assert_traced_as_sent(app, every_request=False)


@pytest.mark.usefixtures("shut_down_console_controllers")
async def test_a_send_after_switching_back_from_a_traced_edit(tmp_path, monkeypatch):
    """Back on the message the Edit & resend replaced, the next send's
    history is the earlier branch: an ordinary rewritten tail."""
    async with _console(tmp_path, monkeypatch, project=False) as app:
        await _send(app, "Hello")
        await _send(app, "And now?")
        original = _last(app, "assistant")
        await _act(app, "edit")

        app.store.set_active_leaf(app.session.id, original.id)

        await _send(app, "One more?")
        _assert_traced_as_sent(app, every_request=False)
        assert "And now?" in json.dumps(app.wire())


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
@pytest.mark.parametrize("action", ["edit", "regenerate", "continue_earlier"])
async def test_a_temporary_chat_sends_an_action_that_keeps_capture_off(
    tmp_path, monkeypatch, action
):
    """Only an action that would be traced asks for a saved chat: one that
    sends the pinned prefill, or a Continue of an earlier message, sends with
    capture off as before."""
    async with _console(tmp_path, monkeypatch, project=False) as app:
        await _send(app, "Hello")
        await _send(app, "And now?")
        app.store.set_session_pinned_prefill(app.session.id, "Sure:")
        monkeypatch.setattr(
            app.store, "session_is_ephemeral", lambda _session_id: True
        )
        before, sent = _calls(app), len(app.sent)

        if action == "continue_earlier":
            first = next(
                row
                for row in app.store.messages_for_session(app.session.id)
                if row.role.value == "assistant"
            )
            result = await app.controller.continue_from_message(first.id)
        else:
            result = await _act(app, action)

        assert result.accepted
        assert len(app.sent) == sent + 1
        assert _calls(app) == before


def _refuse_next_reservation(monkeypatch, *, after=0):
    """The trace service refuses one call's surface, as it does in production;
    ``after`` calls are planned first."""
    from tldw_chatbook.Chat.console_trace_service import ConsoleTraceService

    prepare = ConsoleTraceService.prepare_current_surface_delta
    seen = []

    def refuse(self, *args, **kwargs):
        seen.append(None)
        if len(seen) <= after:
            return prepare(self, *args, **kwargs)
        monkeypatch.setattr(ConsoleTraceService, "prepare_current_surface_delta", prepare)
        raise ValueError("unsupported_surface_change")

    monkeypatch.setattr(ConsoleTraceService, "prepare_current_surface_delta", refuse)


def _transcript(app):
    """The active branch, and every saved row, by id and content."""
    session = app.session.id
    return (
        [(row.id, row.content) for row in app.store.messages_for_session(session)],
        sorted(row.id for row in app.store.all_messages_for_session(session)),
    )


@pytest.mark.usefixtures("shut_down_console_controllers")
@pytest.mark.parametrize(
    ("action", "plain_path", "mixins"),
    [
        pytest.param(
            action,
            plain,
            mixins,
            id=f"{action}-{'plain_path' if plain else 'agent'}"
            + ("-mixins" if mixins else ""),
        )
        for action, plain, mixins in (
            *((action, plain, False) for action in ACTIONS for plain in (False, True)),
            ("edit", False, True),
            ("regenerate", True, True),
        )
    ],
)
async def test_a_refused_action_leaves_the_chat_as_it_was(
    tmp_path, monkeypatch, action, plain_path, mixins
):
    """Owner decision 2026-11-26: the trace refuses the action's own call
    before anything is sent, so the action is undone. Its empty reply (and
    an Edit & resend's edited message) is removed, the branch before it is
    active, the one-time choice is still armed, the copy says why, and the
    next message is sent and traced as after no action."""
    from tldw_chatbook.Chat.console_action_capture import ACTION_NOT_SENT_COPY

    async with _console(
        tmp_path, monkeypatch, plain_path=plain_path, mixins=mixins
    ) as app:
        await _send(app, "Hello")
        await _send(app, "And now?")
        _one_time_capture(app, True)
        before, sent, calls = _transcript(app), len(app.sent), _calls(app)
        _refuse_next_reservation(monkeypatch)

        result = await _act(app, action)

        assert app.reservation_errors == ["unsupported_surface_change"]
        assert not result.accepted
        assert result.visible_copy == ACTION_NOT_SENT_COPY
        assert app.controller.run_state.visible_copy == ACTION_NOT_SENT_COPY
        assert len(app.sent) == sent
        assert _calls(app) == calls
        assert _transcript(app) == before
        assert _next_capture(app) is True
        app.reservation_errors.clear()
        await _send(app, "One more?")
        _assert_traced_as_sent(app)


@pytest.mark.usefixtures("shut_down_console_controllers")
async def test_a_refusal_after_the_run_dispatched_keeps_the_reply(
    tmp_path, monkeypatch
):
    """A Continue's first call is sent and asks for a tool; the trace then
    refuses the tool-loop call. The run reached the provider, so its reply is
    kept and fails with copy that says why."""
    from Tests.Chat.test_console_failed_call_retry_ownership import (
        _NATIVE_TOOL_CALL,
    )
    from tldw_chatbook.Chat.console_action_capture import TRACE_REFUSED_COPY

    replies = []
    async with _console(tmp_path, monkeypatch, hosted=True, replies=replies) as app:
        await _send(app, "Hello")
        replies.append(_NATIVE_TOOL_CALL)
        before, sent = _transcript(app), len(app.sent)
        _refuse_next_reservation(monkeypatch, after=1)

        result = await _act(app, "continue")

        assert len(app.sent) == sent + 1
        assert result.visible_copy == TRACE_REFUSED_COPY
        assert _transcript(app) != before
        assert _last(app, "assistant").status == "failed"


@pytest.mark.usefixtures("shut_down_console_controllers")
@pytest.mark.parametrize("failure", ["mixins", "request"])
async def test_an_action_whose_request_cannot_be_built_fails_visibly(
    tmp_path, monkeypatch, failure
):
    """Before dispatch the action's mix-ins cannot be composed (refused as
    the reply path refuses them), or its trace request cannot be frozen (the
    action is undone): nothing is sent, the copy says why, no run stays
    open, and the next send goes out."""
    from tldw_chatbook.Chat import console_action_capture, console_prepared_request
    from tldw_chatbook.Chat.console_action_capture import (
        ACTION_NOT_SENT_COPY,
        MIXIN_UNAVAILABLE_COPY,
    )
    from tldw_chatbook.Chat.console_prompt_mixin_models import (
        MixinSnapshotUnavailableError,
    )

    async with _console(tmp_path, monkeypatch, mixins=True) as app:
        await _send(app, "Hello")
        module, name, error = (
            (
                console_prepared_request,
                "compose_current_mixin_messages",
                MixinSnapshotUnavailableError("mixin_anchor_unavailable"),
            )
            if failure == "mixins"
            else (console_action_capture, "_trace_request", ValueError("fixture"))
        )
        original = getattr(module, name)

        def fail_once(*_args, **_kwargs):
            monkeypatch.setattr(module, name, original)
            raise error

        monkeypatch.setattr(module, name, fail_once)
        before, sent = _transcript(app), len(app.sent)

        result = await _act(app, "regenerate")

        assert len(app.sent) == sent
        assert result.visible_copy == (
            MIXIN_UNAVAILABLE_COPY if failure == "mixins" else ACTION_NOT_SENT_COPY
        )
        assert app.controller.run_state.status.value == "blocked"
        if failure == "request":
            assert _transcript(app) == before
        await _send(app, "One more?")
        _assert_traced_as_sent(app, every_request=False)


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


@pytest.mark.parametrize(
    ("route", "action", "content", "admitted"),
    [
        ("continue", None, None, True),
        ("regenerate", None, None, True),
        ("agent_first", "continue", None, True),
        ("edit", None, None, False),
        ("agent_first", "edit", None, False),
        ("continue", None, "A question the user never saved", False),
        ("agent_first", "continue", "A question the user never saved", False),
        ("agent_first", "edit", "A question the user never saved", False),
        ("agent_first", None, "A question the user never saved", False),
    ],
    ids=lambda value: str(value),
)
async def test_a_continue_step_back_needs_its_action_and_instruction(
    ledger, route, action, content, admitted
):
    """Only a Continue, or a Regenerate of a continued reply, steps back from
    a trailing instruction to the user its replies answer, and only when that
    row is Console's instruction. An Edit & resend's rows never end that way,
    and on the agent path the action the call names grants nothing by
    itself: an unsaved user row after a saved reply is refused."""
    from tldw_chatbook.Chat.console_trace_errors import TraceCallPersistenceError
    from tldw_chatbook.Chat.console_trace_provenance import ConsoleRequestRoute

    agent = route == "agent_first"
    send = ledger.send_agent if agent else ledger.send
    first = ConsoleRequestRoute.AGENT_FIRST if agent else ConsoleRequestRoute.FRESH
    hello_id, hello = ledger.message("hello")
    await send(first, [({"role": "user", "content": "hello"}, hello)])
    _, reply = ledger.message("partial", sender="assistant", parent=hello_id)
    rows = [
        ({"role": "user", "content": "hello"}, hello),
        ({"role": "assistant", "content": "partial"}, reply),
        ledger.instruction(content) if content else ledger.instruction(),
    ]
    named = None if action is None else ConsoleRequestRoute(action)
    call = (
        (lambda: send(first, rows, action=named))
        if agent
        else (lambda: send(ConsoleRequestRoute(route), rows))
    )
    asked = len(ledger.asked)

    if admitted:
        assert await call() == ["answered"]
        return
    with pytest.raises(TraceCallPersistenceError):
        await call()
    assert len(ledger.asked) == asked


def test_the_trace_rule_names_the_controllers_continue_instruction():
    from tldw_chatbook.Chat.console_chat_controller import (
        CONSOLE_CONTINUE_INSTRUCTION,
    )
    from tldw_chatbook.Chat.console_trace_action_calls import CONTINUE_INSTRUCTION

    assert CONTINUE_INSTRUCTION == CONSOLE_CONTINUE_INSTRUCTION


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
