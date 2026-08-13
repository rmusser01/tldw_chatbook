"""PR 3a-2 Task 4: the teardown notice must stop lying about survivors.

Task 1 A4 (executed): with a cross-turn survivor as a session's only work,
``busy_fleet_session_count()`` at unmount returns 1 (the PR 3a-1 third
leg) and the next mount toasts "1 agent run was cancelled when you left
Console." -- while the run keeps working through ``shutdown()`` and
finishes ``done``. Count right, verb wrong, in the OPPOSITE direction
from the pre-3a-1 defect.

Under test: the split (``ConsoleChatController.fleet_teardown_split``)
that separates sessions teardown genuinely kills (active stream /
pending approval -- the regime where a still-streaming turn's own
children are cancelled with it, pinned by execution in
``Tests/Agents/test_fleet_runtime.py::test_stopping_the_turn_still_stops_
its_children`` and by the shutdown->cancel-event link test below) from
sessions whose only work is survivors, which keep running; and the
next-mount copy that reports each truthfully.
"""
from __future__ import annotations

import asyncio
import threading

import pytest

from Tests.Chat.test_child_run_scope_ordering import _survivor_bridge
from Tests.Chat.test_console_agent_bridge import (
    _fence,
    _join_fleet_threads,
    _run,
)
from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import ConsoleRunState, ConsoleRunStatus
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen


def _screen_with_notify_capture():
    app = _build_test_app()
    screen = ChatScreen(app)
    notifications: list[tuple[str, str]] = []

    def capture(message, *args, severity="information", **kwargs):
        notifications.append((str(message), severity))

    app.notify = capture
    return app, screen, notifications


def _survivor_controller(tmp_path):
    """A controller whose one session's ONLY work is a gated survivor."""
    gate, gateway, db, store, session, aid, bridge = _survivor_bridge(
        tmp_path,
        parent_script=[
            [_fence("spawn_subagent", {"task": "long job"})],
            ["turn final"],
        ],
        needed=1,
    )
    controller = ConsoleChatController(
        store=store, provider_gateway=object(), agent_bridge=bridge
    )
    outcome = _run(bridge, store, session, aid, conversation_id=session.id)
    assert outcome.status == "done"
    assert gateway.entered_event.wait(5), "the child never started"
    return gate, controller, session


@pytest.mark.asyncio
async def test_a_survivor_only_teardown_is_reported_as_continuing_not_cancelled(
    tmp_path,
):
    """The Task 1 A4 lie, fixed at the copy path: a survivor-only teardown
    followed by the next mount's notice must say the work KEPT RUNNING --
    never that it was cancelled -- and the run must in fact finish
    ``done`` on disk after the screen is gone (the ground truth the old
    copy contradicted)."""
    gate, controller, session = _survivor_controller(tmp_path)
    app, screen, notifications = _screen_with_notify_capture()
    try:
        # The exact ChatScreen.on_unmount recording, via its extracted seam.
        await screen._record_console_fleet_teardown(controller)
        screen._notify_console_fleet_teardown_if_any()
    finally:
        gate.set()
    _join_fleet_threads()

    assert notifications, "a busy teardown must produce a next-mount notice"
    assert not any("cancelled" in message for message, _ in notifications), (
        "a survivor-only teardown must never be reported as cancelled: "
        f"{notifications}"
    )
    assert any("kept running" in message for message, _ in notifications), (
        f"the notice must say the work continues: {notifications}"
    )
    # The ground truth: the run the old copy called cancelled finished
    # done, durably, read through a fresh DB handle.
    fresh = AgentRunsDB(tmp_path / "runs.db", client_id="verifier")
    rows = [
        r
        for r in fresh.list_runs(session.id)
        if r["agent_kind"] == "subagent"
    ]
    assert [r["status"] for r in rows] == ["done"], (
        f"the survivor must have completed after teardown: {rows}"
    )
    # One-shot: a second mount with nothing new stays silent.
    notifications.clear()
    screen._notify_console_fleet_teardown_if_any()
    assert notifications == []


@pytest.mark.asyncio
async def test_a_streaming_teardown_keeps_the_cancelled_copy(tmp_path):
    """The genuinely-killed bucket keeps its (true) copy byte-identical:
    a session with an active stream loses that work at shutdown."""
    gate, gateway, db, store, session, aid, bridge = _survivor_bridge(
        tmp_path,
        parent_script=[["only turn"]],
        needed=0,
    )
    controller = ConsoleChatController(
        store=store, provider_gateway=object(), agent_bridge=bridge
    )
    gate.set()
    controller._set_run_state(
        ConsoleRunState(ConsoleRunStatus.STREAMING, "run"),
        session_id=session.id,
    )
    assert controller.in_flight_run_count() == 1
    app, screen, notifications = _screen_with_notify_capture()
    await screen._record_console_fleet_teardown(controller)
    screen._notify_console_fleet_teardown_if_any()
    assert [
        (message, severity) for message, severity in notifications
    ] == [("1 agent run was cancelled when you left Console.", "warning")]


@pytest.mark.asyncio
async def test_a_mixed_teardown_reports_both_truthfully(tmp_path):
    """One session streaming (killed), another whose only work is a
    survivor (continues): the notice must distinguish them rather than
    calling every survivor cancelled."""
    gate, controller, survivor_session = _survivor_controller(tmp_path)
    streaming = controller.store.create_session(title="Streaming one")
    controller._set_run_state(
        ConsoleRunState(ConsoleRunStatus.STREAMING, "run"),
        session_id=streaming.id,
    )
    app, screen, notifications = _screen_with_notify_capture()
    try:
        assert controller.fleet_teardown_split() == (1, 1)
        assert controller.busy_fleet_session_count() == 2, (
            "the split must partition the same union the old count reported"
        )
        await screen._record_console_fleet_teardown(controller)
        screen._notify_console_fleet_teardown_if_any()
    finally:
        gate.set()
    _join_fleet_threads()
    cancelled = [m for m, _ in notifications if "cancelled" in m]
    continuing = [m for m, _ in notifications if "kept running" in m]
    assert cancelled == ["1 agent run was cancelled when you left Console."]
    assert len(continuing) == 1, f"expected one continuing notice: {notifications}"


def test_split_counts_a_survivor_plus_stream_session_as_killed(tmp_path):
    """A session with BOTH an active stream and (prior-turn) survivors is
    counted killed -- its in-flight turn genuinely dies; the stated,
    documented under-report is that its earlier survivors continuing goes
    unmentioned here (their own settle toast still covers them)."""
    gate, controller, session = _survivor_controller(tmp_path)
    try:
        assert controller.fleet_teardown_split() == (0, 1)
        controller._set_run_state(
            ConsoleRunState(ConsoleRunStatus.STREAMING, "run"),
            session_id=session.id,
        )
        assert controller.fleet_teardown_split() == (1, 0)
        assert controller.busy_fleet_session_count() == 1
    finally:
        gate.set()
    _join_fleet_threads()


@pytest.mark.asyncio
async def test_shutdown_sets_the_active_turns_cancel_event(tmp_path):
    """The mid-turn-navigation link, executed (Task 1 left it as labelled
    inference): for a session still in ``_active_stream_tasks``,
    ``shutdown()``'s per-session ``_signal_stop`` fanout sets that run's
    own cancel event -- the same signal whose child-killing consequence
    ``test_stopping_the_turn_still_stops_its_children`` pins by execution
    at the service level (a cancelled turn settles its children
    ``cancelled``; no survivors)."""
    gate, gateway, db, store, session, aid, bridge = _survivor_bridge(
        tmp_path,
        parent_script=[["only turn"]],
        needed=0,
    )
    gate.set()
    controller = ConsoleChatController(
        store=store, provider_gateway=object(), agent_bridge=bridge
    )
    cancel_event = threading.Event()
    controller._active_cancel_events[session.id] = cancel_event
    task = asyncio.create_task(asyncio.sleep(30))
    controller._active_stream_tasks[session.id] = task
    try:
        await controller.shutdown()
        assert cancel_event.is_set(), (
            "shutdown must signal the still-streaming turn's own cancel "
            "event -- the mechanism that makes its children settle "
            "cancelled instead of surviving"
        )
        assert task.cancelled() or task.done()
    finally:
        if not task.done():
            task.cancel()
    _join_fleet_threads()
