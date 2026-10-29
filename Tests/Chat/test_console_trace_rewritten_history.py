"""TASK-33621.7: captured sends after a rewritten history.

Edit & resend changes the provider-visible history without a captured call
that records the change: its own call is dispatched with capture off. The next
captured send therefore carried a rewrite of the recorded surface's tail
together with the new turn. The trace planner admits only an append or a
one-item replacement, so it refused that send with
``unsupported_surface_change``. Every later send in the chat then stopped at
the "Trace capture blocked" card.

These tests drive the real ``ConsoleChatController``, ``ConsoleChatStore``,
SQLite database, ``ConsoleProviderGateway`` request preparation and
``ConsoleTraceBoundaryFactory``. Only the provider adapter at the bottom of the
gateway is a recorder. The agent path is the Console default; the
plain-provider path is what character chats and ``[console] agent_runtime =
false`` take.
"""

from __future__ import annotations

import pytest

from Tests.Chat.test_console_send_diagnostics import sinks  # noqa: F401
from Tests.Chat.test_console_trace_current_turn_transforms import (  # noqa: F401
    make_database,
    make_gateway,
    trace_harness,
)
from tldw_chatbook.Chat import console_trace_rewritten_tail
from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.Chat.console_chat_models import ConsoleRunStatus
from tldw_chatbook.Chat.console_trace_native_reader import ConsoleTraceNativeReader
from tldw_chatbook.Chat.console_trace_runtime import ConsoleTraceBoundaryFactory
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

pytestmark = pytest.mark.bootstrap_profile

SESSION = "session-1"
_LONG = "detail " * 200


@pytest.fixture(params=["agent", "plain"])
def harness(trace_harness, tmp_path, request, monkeypatch):  # noqa: F811
    """``trace_harness`` on the agent path (default) or the plain path.

    ``rechecks`` records each time dispatch binding rechecks a rewritten-tail
    proof.
    """

    rechecks = []
    validate = console_trace_rewritten_tail.validate_rewritten_tail

    def recheck(*args, **kwargs):
        rechecks.append(kwargs["witness"])
        return validate(*args, **kwargs)

    monkeypatch.setattr(
        console_trace_rewritten_tail, "validate_rewritten_tail", recheck
    )
    trace_harness.rechecks = rechecks
    controller = trace_harness.controller
    if request.param == "agent":
        runs = AgentRunsDB(tmp_path / "agent_runs.sqlite", client_id="task33621")
        controller.update_agent_runtime(
            enabled=True,
            bridge=ConsoleAgentBridge(
                agent_runs_db=runs,
                store=trace_harness.store,
                provider_gateway=trace_harness.gateway,
            ),
        )
    trace_harness.path = request.param
    return trace_harness


async def _send(harness, text: str):
    sent_before = len(harness.entries)
    reserved_before = len(harness.boundaries)
    result = await harness.controller.submit_draft(text, session_id=SESSION)
    return result, harness.entries[sent_before:], harness.boundaries[reserved_before:]


def _assert_captured(harness, result, sent, reserved) -> None:
    """The send reached the provider once, with its trace call reserved."""

    assert harness.failures == [], "trace reservation was refused"
    assert result.terminal_status is ConsoleRunStatus.COMPLETED, result
    # No decision card: the turn is not held for a trace-capture choice.
    assert harness.controller.run_state_for(SESSION).status is not (
        ConsoleRunStatus.BLOCKED
    )
    assert harness.store.preparation_for_session(SESSION) is None
    assert len(sent) == 1, "the provider must be contacted exactly once"
    assert len(reserved) == 1, "the send must reserve exactly one trace call"
    user = harness.store.get_message(result.user_message_id)
    calls = ConsoleTraceNativeReader(harness.db).read_calls(user.persisted_message_id)
    assert len(calls) == 1
    # Capture stayed on: the trace holds exactly what the provider received.
    assert calls[0].capture.request["messages_payload"] == sent[0]


def _restart_trace_runtime(harness) -> None:
    """Drop every in-memory surface checkpoint, as an app restart does."""

    harness.factory = ConsoleTraceBoundaryFactory(harness.db)


async def _three_long_turns(harness) -> list:
    results = []
    for index in range(4):
        result, sent, reserved = await _send(harness, f"q{index}: {_LONG}")
        _assert_captured(harness, result, sent, reserved)
        results.append(result)
    return results


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("anchor", "restart"),
    [("last", False), ("middle", False), ("last", True)],
    ids=["last", "middle", "last-restarted"],
)
async def test_captured_sends_after_edit_and_resend_keep_trace_capture(
    harness, anchor, restart
):
    """AC#1/#5 of TASK-33621.7: send, Edit & resend, send again."""

    turns = await _three_long_turns(harness)
    edited = turns[-1] if anchor == "last" else turns[1]

    resent = await harness.controller.edit_and_resend_message(
        edited.user_message_id, "edited question"
    )
    assert resent.accepted is True
    assert harness.entries[-1][-1] == {"role": "user", "content": "edited question"}
    if restart:
        _restart_trace_runtime(harness)

    result, sent, reserved = await _send(harness, "after the edit")
    _assert_captured(harness, result, sent, reserved)
    # Dispatch binding rechecked the branch-anchored proof once.
    assert len(harness.rechecks) == 1
    assert sent[0][-3:] == [
        {"role": "user", "content": "edited question"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "after the edit"},
    ]

    following, sent, reserved = await _send(harness, "and again")
    _assert_captured(harness, following, sent, reserved)
    assert len(harness.rechecks) == 1, "a plain append needs no rewrite proof"
