"""PR 3a-2 Task 5, the safety half: a woken turn grants no new authority.

Spec §3 invariant 5 kept both of the original no-auto-wake ruling's
constraints; this file pins them against the REAL approval machinery
(the test_console_agent_swap harness: real bridge, real controller, real
worker-thread <-> UI-thread ``request_mcp_approvals`` round trip):

1. **The approval floor is unchanged in a woken turn.** A gated
   (ask-state) tool call made by the model DURING a wake turn raises the
   same approval card, through the same round-trip, as the byte-identical
   manual turn (``test_mcp_tool_call_ask_state_routes_through_review_
   hook_and_approves`` is the manual twin) -- and a capture-level pin
   shows the wake dispatches ``run_reply`` under the same authority
   composition (same builtin gate + review hook presence) as a manual
   send, so the risk-tag ask-floor (enforced inside that shared gate,
   pinned by the gate's own suites) binds a woken turn identically.
2. **A wake can never satisfy an approval.** Approval resolution has
   exactly one path -- ``resolve_pending_approval(decisions, round_id)``
   -- and nothing the wake injects flows into it: while a card is
   pending, an arriving completion's wake DEFERS (the session is busy),
   the round's decisions stay empty and its event stays unset for the
   whole intake, and only the explicit UI resolution releases the turn;
   the deferred wake then fires on the terminal transition.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from Tests.Chat.test_console_agent_swap import (
    FakeMCPService,
    _catalog_record,
    _controller,
    _fake_app,
    _fence,
    _tool_dict,
)
from Tests.Chat.test_console_fleet_wake import (
    _drain,
    _survivor,
    _terminal_subagent_run,
)
from Tests.Chat.test_fleet_attention import _AppStub
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleMessageRole,
    ConsoleRunStatus,
)
from tldw_chatbook.Chat.conversation_local_marks_service import (
    ConversationLocalMarksService,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture(autouse=True)
def _mcp_tests_keep_a_small_catalog(monkeypatch):
    """Same autouse guard (and same reasoning, verbatim) as
    ``test_console_agent_swap._mcp_tests_keep_a_small_catalog``: these
    tests exercise the approval gate on a WOKEN turn, not tool
    disclosure. With ``[console] local_tools_enabled`` default-true the
    catalog crosses ``DIRECT_DISCLOSE_THRESHOLD`` and a scripted model
    calling its tool directly is refused at the disclosure gate before
    the permission gate is ever consulted -- diagnosed here the hard way:
    the woken turn's card surfaced but the approved call came back
    "Tool not permitted" until this fixture was copied over."""
    from tldw_chatbook.Chat import console_chat_controller as controller_module

    real_get_cli_setting = controller_module.get_cli_setting

    def _small_catalog(section, key, default=None, *args, **kwargs):
        if section == "console" and key == "local_tools_enabled":
            return False
        return real_get_cli_setting(section, key, default, *args, **kwargs)

    monkeypatch.setattr(controller_module, "get_cli_setting", _small_catalog)


async def _settle(predicate, seconds: float = 30.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return bool(predicate())


async def _quiet(predicate, seconds: float = 0.5) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return False
        await asyncio.sleep(0.02)
    return True


@pytest.mark.asyncio
async def test_a_woken_turns_gated_tool_still_raises_the_approval_card(tmp_path):
    """The floor, end to end on the WAKE path: the wake turn's model calls
    an ask-state tool; the card surfaces through the real round trip; the
    wake's own injected notice resolves NOTHING (the round sits undecided
    until the explicit UI approval); only then does the turn finish."""
    chacha = CharactersRAGDB(str(tmp_path / "chacha.sqlite"), client_id="t")
    try:
        scripts = [
            [_fence("mcp__srv__run", {"x": 1})],
            ["acted on the wake."],
        ]
        controller, store, runs_db = _controller(tmp_path, scripts)
        received: list[dict | None] = []
        service = FakeMCPService(
            catalog_records=[_catalog_record("srv", [_tool_dict("run")])]
        )
        controller.app = _fake_app(service)
        controller.set_pending_approval = received.append
        controller.mcp_approval_timeout_seconds = lambda: 30.0
        app = _AppStub(chacha)
        controller.fleet_wake.wire(app=app)
        session = store.ensure_session()
        _parent, run_id = _terminal_subagent_run(
            runs_db, session.id, result="survivor findings"
        )
        app.conversation_local_marks_service.set_mark(
            session.id, ConversationLocalMarksService.FLEET_UNSEEN
        )

        controller.fleet_wake.on_fleet_drained(
            _drain(session.id, _survivor(run_id, session_id=session.id))
        )
        surfaced = await _settle(lambda: received and received[-1] is not None)
        assert surfaced, (
            "the woken turn's gated tool call never raised an approval card"
        )
        card = received[-1]
        assert card["calls"][0]["llm_name"] == "mcp__srv__run", (
            "same card, same shape, as the manual twin"
        )
        round_id = card["round_id"]

        # The wake notice is not approval: the round sits undecided while
        # the whole machine-injected turn is in flight around it.
        with controller._approval_state_lock:
            round_state = controller._pending_approval_rounds[round_id]
        assert await _quiet(lambda: round_state["event"].is_set()), (
            "NOTHING the wake injected may set the approval event"
        )
        assert round_state["decisions"] == {}, (
            "NOTHING the wake injected may write an approval decision"
        )

        # Only the explicit UI resolution releases the turn.
        controller.resolve_pending_approval(
            {"mcp__srv__run": "approve_once"}, round_id=round_id
        )
        done = await _settle(
            lambda: any(
                m.content == "acted on the wake."
                for m in store.messages_for_session(session.id)
            )
        )
        assert done, "the approved wake turn never finished"
        messages = store.messages_for_session(session.id)
        notice_rows = [
            m
            for m in messages
            if m.role is ConsoleMessageRole.SYSTEM
            and getattr(m.metadata, "origin", "") == "agent_wake"
        ]
        assert len(notice_rows) == 1, "the wake ran as a machine-marked turn"
        assert service.execute_calls == [
            ("local:srv", "run", {"x": 1}, "agent", "approved")
        ], "the tool executed only under the explicit approval"
        assert await _settle(
            lambda: not app.conversation_local_marks_service.has_mark(
                session.id, ConversationLocalMarksService.FLEET_UNSEEN
            ),
            seconds=5.0,
        ), "delivery completed: the mark clears after the accepted wake"
    finally:
        chacha.close()


@pytest.mark.asyncio
async def test_a_wake_defers_behind_a_pending_card_and_cannot_resolve_it(
    tmp_path,
):
    """The other direction: the card was already pending (a MANUAL turn's
    gated call) when the completion arrived. The wake defers -- the busy
    session gate -- and for the whole deferral the round stays exactly as
    the user left it: undecided. The user's explicit approval finishes
    the manual turn, whose terminal transition then (and only then) lets
    the wake deliver."""
    chacha = CharactersRAGDB(str(tmp_path / "chacha.sqlite"), client_id="t")
    try:
        scripts = [
            [_fence("mcp__srv__run", {"x": 1})],
            ["manual turn done."],
            ["woke after approval."],
        ]
        controller, store, runs_db = _controller(tmp_path, scripts)
        received: list[dict | None] = []
        service = FakeMCPService(
            catalog_records=[_catalog_record("srv", [_tool_dict("run")])]
        )
        controller.app = _fake_app(service)
        controller.set_pending_approval = received.append
        controller.mcp_approval_timeout_seconds = lambda: 30.0
        app = _AppStub(chacha)
        controller.fleet_wake.wire(app=app)
        session = store.ensure_session()
        _parent, run_id = _terminal_subagent_run(
            runs_db, session.id, result="finished while you decided"
        )

        send_task = asyncio.ensure_future(
            controller.submit_draft("please run it")
        )
        surfaced = await _settle(lambda: received and received[-1] is not None)
        assert surfaced, "the manual turn's approval card never surfaced"
        round_id = received[-1]["round_id"]
        with controller._approval_state_lock:
            round_state = controller._pending_approval_rounds[round_id]

        # The completion lands NOW, mid-decision.
        controller.fleet_wake.on_fleet_drained(
            _drain(session.id, _survivor(run_id, session_id=session.id))
        )
        assert await _quiet(
            lambda: round_state["event"].is_set() or round_state["decisions"]
        ), "the arriving wake must not decide or release the pending card"
        assert controller.fleet_wake.has_pending(session.id), (
            "the deferred wake keeps its pending bit"
        )
        assert not any(
            m.content == "woke after approval."
            for m in store.messages_for_session(session.id)
        ), "no wake turn may start while its session awaits an approval"

        # The USER decides; the manual turn finishes; the wake follows.
        controller.resolve_pending_approval(
            {"mcp__srv__run": "approve_once"}, round_id=round_id
        )
        result = await send_task
        assert result.accepted is True
        woke = await _settle(
            lambda: any(
                m.content == "woke after approval."
                for m in store.messages_for_session(session.id)
            )
        )
        assert woke, (
            "the deferred wake never fired after the manual turn's "
            "terminal transition"
        )
        # Settled, not sampled: the reply CONTENT lands a beat before the
        # wake turn's own terminal transition, so reading the run state at
        # the content's first appearance races finalization (caught as an
        # interference-only failure in the full battery).
        assert await _settle(
            lambda: controller.run_state_for(session.id).status
            is ConsoleRunStatus.COMPLETED,
            seconds=10.0,
        ), "the wake turn must settle terminal like any turn"
    finally:
        chacha.close()


@pytest.mark.asyncio
async def test_a_wake_dispatches_run_reply_under_the_same_authority_as_manual(
    tmp_path,
):
    """Capture-level twin pin: manual send and wake send through the SAME
    controller hand ``run_reply`` the same authority composition -- a
    per-run builtin gate and a review hook both present (the seam the
    risk-tag ask-floor is enforced behind), the same native-tools flag --
    so nothing about being machine-initiated widens (or bypasses) what
    the turn may do."""
    chacha = CharactersRAGDB(str(tmp_path / "chacha.sqlite"), client_id="t")
    try:
        controller, store, runs_db = _controller(tmp_path, [["m."], ["w."]])
        controller.app = _fake_app()
        captured: list[dict] = []
        real_run_reply = controller._agent_bridge.run_reply

        def capturing_run_reply(**kwargs):
            captured.append(kwargs)
            return real_run_reply(**kwargs)

        controller._agent_bridge.run_reply = capturing_run_reply
        app = _AppStub(chacha)
        controller.fleet_wake.wire(app=app)
        session = store.ensure_session()
        _parent, run_id = _terminal_subagent_run(runs_db, session.id)

        manual = await controller.submit_draft("hello", session_id=session.id)
        assert manual.accepted is True
        controller.fleet_wake.on_fleet_drained(
            _drain(session.id, _survivor(run_id, session_id=session.id))
        )
        assert await _settle(lambda: len(captured) >= 2, seconds=10.0), (
            "the wake turn never reached run_reply"
        )
        manual_kwargs, wake_kwargs = captured[0], captured[1]
        assert (wake_kwargs["builtin_gate"] is None) == (
            manual_kwargs["builtin_gate"] is None
        ), "a woken turn must carry the same per-run builtin gate as manual"
        assert (wake_kwargs["review_tool_calls"] is None) == (
            manual_kwargs["review_tool_calls"] is None
        ), "a woken turn must route tool calls through the same review hook"
        assert wake_kwargs["native_tools_enabled"] == (
            manual_kwargs["native_tools_enabled"]
        )
        # And the payload the wake hands the model ends on its own
        # machine-labelled user-role entry, never a widened toolset.
        assert wake_kwargs["agent_messages"][-1]["role"] == "user"
        assert "not user input" in wake_kwargs["agent_messages"][-1]["content"]
    finally:
        chacha.close()
