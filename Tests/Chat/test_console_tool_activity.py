"""Run ownership and interruption regressions for the ephemeral tool display."""

import pytest

from tldw_chatbook.Agents.agent_models import AgentStep
from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.Chat.console_chat_models import ConsoleActivityPresentation
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_tool_activity import ConsoleToolActivity


@pytest.mark.parametrize("cancelled, expected", [(True, "stopped"), (False, "failed")])
def test_unresolved_rows_settle_and_ignore_late_events(cancelled, expected):
    store = ConsoleChatStore()
    session = store.ensure_session()
    activity = ConsoleToolActivity(store, session.id)
    proposed = AgentStep(
        index=1, kind="tool_proposed", tool_name="fs_read", call_id="read"
    )
    activity.observe(proposed, 1)
    activity.observe(
        AgentStep(index=1, kind="tool_execution_started", call_id="read"), 1
    )
    activity.finish(cancelled)
    row = store.messages_for_session(session.id)[-1]
    assert row.activity_presentation.status == expected
    assert row.activity_presentation.started_at_monotonic is None
    assert row.activity_presentation.elapsed_seconds is not None
    assert "before a result" in row.activity_presentation.result_preview
    activity.observe(proposed, 1)
    activity.approval(["read"], True)
    assert not activity.complete(
        AgentStep(
            index=1,
            kind="tool_result",
            tool_name="fs_read",
            call_id="read",
            result="late",
        ),
        ConsoleActivityPresentation("tool", "fs_read", "success"),
        "⚙ fs_read → late",
        None,
        record_trajectory=True,
    )
    assert store.messages_for_session(session.id)[-1] == row
    assert not store._pending_trajectory_tool_rows


def test_approval_projection_is_scoped_to_run_session_and_call():
    store = ConsoleChatStore()
    first = store.ensure_session()
    second = store.create_session()
    bridge = ConsoleAgentBridge(agent_runs_db=None, store=store, provider_gateway=None)
    activity = ConsoleToolActivity(store, first.id)
    bridge._tool_activity_runs["run-a"] = activity
    activity.observe(
        AgentStep(index=1, kind="tool_proposed", tool_name="read", call_id="call-a"), 1
    )
    bridge.set_tool_approval_pending(second.id, "run-a", ["call-a"], True)
    bridge.set_tool_approval_pending(first.id, "wrong-run", ["call-a"], True)
    bridge.set_tool_approval_pending(first.id, "run-a", ["wrong-call"], True)
    assert (
        store.messages_for_session(first.id)[-1].activity_presentation.status
        == "queued"
    )
    bridge.set_tool_approval_pending(first.id, "run-a", ["call-a"], True)
    assert (
        store.messages_for_session(first.id)[-1].activity_presentation.status
        == "awaiting_approval"
    )
    assert not store.messages_for_session(second.id)


@pytest.mark.parametrize(
    "outcome, status",
    [("timeout", "timed_out"), ("cancelled", "stopped"), ("failure", "failed")],
)
def test_terminal_status_uses_execution_outcome_and_keeps_marker_id(outcome, status):
    store = ConsoleChatStore()
    session = store.ensure_session()
    activity = ConsoleToolActivity(store, session.id)
    activity.observe(
        AgentStep(index=1, kind="tool_proposed", tool_name="read", call_id="call"), 1
    )
    marker_id = store.messages_for_session(session.id)[-1].id
    assert activity.complete(
        AgentStep(
            index=1,
            kind="tool_result",
            tool_name="read",
            call_id="call",
            result="failed",
            tool_outcome=outcome,
        ),
        ConsoleActivityPresentation("tool", "read", "failed"),
        "⚙ read → failed",
        None,
        record_trajectory=True,
    )
    activity.finish(False)
    row = store.messages_for_session(session.id)[-1]
    assert row.id == marker_id and row.activity_presentation.status == status


@pytest.mark.parametrize("extra_chars", [-1, 0, 100])
def test_argument_preview_is_bounded_without_truncating_at_or_below_limit(extra_chars):
    import json

    from tldw_chatbook.Chat.console_chat_models import MAX_CONSOLE_TOOL_ARGUMENT_CHARS

    store = ConsoleChatStore()
    session = store.ensure_session()
    activity = ConsoleToolActivity(store, session.id)
    overhead = len(json.dumps({"text": ""}, indent=2))
    args = {"text": "é" * (MAX_CONSOLE_TOOL_ARGUMENT_CHARS - overhead + extra_chars)}
    activity.observe(
        AgentStep(
            index=1, kind="tool_proposed", tool_name="read", call_id="call", args=args
        ),
        1,
    )
    arguments = store.messages_for_session(session.id)[
        -1
    ].activity_presentation.arguments
    original = json.dumps(args, ensure_ascii=False, indent=2)
    if extra_chars <= 0:
        assert arguments == original
    else:
        assert len(arguments) == MAX_CONSOLE_TOOL_ARGUMENT_CHARS
        assert arguments.endswith("\n… arguments truncated")
        assert original.startswith(arguments.removesuffix("\n… arguments truncated"))


def test_partial_text_stays_on_exact_live_row_and_survives_interruption():
    store = ConsoleChatStore()
    session = store.ensure_session()
    activity = ConsoleToolActivity(store, session.id)
    activity.observe(
        AgentStep(index=1, kind="tool_proposed", call_id="reused", tool_name="read"), 1
    )
    activity.observe(
        AgentStep(
            index=2,
            kind="tool_execution_started",
            call_id="reused",
            source_step_index=1,
        ),
        1,
    )
    marker = store.messages_for_session(session.id)[-1].id
    activity.observe(
        AgentStep(
            index=2,
            kind="tool_output",
            call_id="reused",
            source_step_index=1,
            result="stdout\npartial",
        ),
        1,
    )
    row = store.messages_for_session(session.id)[-1]
    assert (
        row.id == marker
        and row.activity_presentation.result_preview == "stdout\npartial"
    )
    assert not store._pending_trajectory_tool_rows
    activity.complete(
        AgentStep(index=3, kind="tool_result", call_id="reused", result="final"),
        ConsoleActivityPresentation("tool", "read", "success"),
        "final",
        None,
        record_trajectory=False,
    )
    activity.observe(
        AgentStep(index=4, kind="tool_proposed", call_id="reused", tool_name="read"), 2
    )
    activity.observe(
        AgentStep(
            index=5,
            kind="tool_execution_started",
            call_id="reused",
            source_step_index=4,
        ),
        2,
    )
    activity.observe(
        AgentStep(
            index=2,
            kind="tool_output",
            call_id="reused",
            source_step_index=1,
            result="stale",
        ),
        1,
    )
    assert (
        store.messages_for_session(session.id)[-1].activity_presentation.result_preview
        is None
    )
    activity.observe(
        AgentStep(
            index=5,
            kind="tool_output",
            call_id="reused",
            source_step_index=4,
            result="current",
        ),
        2,
    )
    activity.finish(True)
    row = store.messages_for_session(session.id)[-1]
    assert row.activity_presentation.status == "stopped"
    assert "current" in row.activity_presentation.result_preview


@pytest.mark.parametrize(
    "outcome, status", [("timeout", "timed_out"), ("cancelled", "stopped")]
)
def test_interrupted_result_keeps_partial_details_out_of_trajectory(outcome, status):
    store = ConsoleChatStore()
    session = store.ensure_session()
    activity = ConsoleToolActivity(store, session.id)
    activity.observe(
        AgentStep(index=1, kind="tool_proposed", call_id="held", tool_name="read"), 1
    )
    activity.observe(
        AgentStep(
            index=2, kind="tool_execution_started", call_id="held", source_step_index=1
        ),
        1,
    )
    activity.observe(
        AgentStep(
            index=2,
            kind="tool_output",
            call_id="held",
            source_step_index=1,
            result="PARTIAL_ONLY",
        ),
        1,
    )
    activity.complete(
        AgentStep(
            index=3,
            kind="tool_result",
            call_id="held",
            result="ERROR timeout",
            tool_outcome=outcome,
        ),
        ConsoleActivityPresentation("tool", "read", "failed"),
        "ERROR timeout",
        None,
        record_trajectory=True,
    )
    activity.finish(False)
    row = store.messages_for_session(session.id)[-1]
    assert row.activity_presentation.status == status
    assert "PARTIAL_ONLY" in row.tool_output_full
    assert "PARTIAL_ONLY" not in str(store._pending_trajectory_tool_rows)
