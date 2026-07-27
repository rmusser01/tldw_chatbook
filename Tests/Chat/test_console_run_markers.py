"""Run markers: running/needs-approval/finished-unvisited (spec §6)."""

from __future__ import annotations

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleRunMarker,
    ConsoleRunState,
    ConsoleRunStatus,
)


def test_marker_lifecycle(controller_with_two_sessions):
    controller, session_a, session_b = controller_with_two_sessions
    # `controller_with_two_sessions` leaves session_b (not session_a)
    # active (`controller.new_session` activates the session it creates),
    # so session_a is already the NON-active session here -- exactly the
    # case the terminal-stamping rule (Step 3 of the brief) requires this
    # test to exercise honestly.
    assert controller.store.active_session_id == session_b
    assert controller.run_marker_for(session_a) is ConsoleRunMarker.NONE

    controller._set_run_state(
        ConsoleRunState(ConsoleRunStatus.STREAMING, "run"), session_id=session_a
    )
    assert controller.run_marker_for(session_a) is ConsoleRunMarker.RUNNING

    # Brief's `set_pending_approval(session_id, bool)` renamed to
    # `set_run_pending_approval` -- `ConsoleChatController.__init__` already
    # owns a `self.set_pending_approval` INSTANCE ATTRIBUTE (the MCP
    # batch-approval UI callback slot wired by
    # `ChatScreen._ensure_console_chat_controller`, task-5), so a same-named
    # method here would be clobbered by `__init__`'s
    # `self.set_pending_approval = None` assignment. See task-7-report.md.
    controller.set_run_pending_approval(session_a, True)
    assert controller.run_marker_for(session_a) is ConsoleRunMarker.NEEDS_APPROVAL
    controller.set_run_pending_approval(session_a, False)

    controller._set_run_state(
        ConsoleRunState(ConsoleRunStatus.COMPLETED, "done"), session_id=session_a
    )
    assert controller.run_marker_for(session_a) is ConsoleRunMarker.FINISHED_OK

    controller.mark_session_visited(session_a)
    assert controller.run_marker_for(session_a) is ConsoleRunMarker.NONE


def test_failed_marker_and_fleet_counts(controller_with_two_sessions):
    controller, session_a, session_b = controller_with_two_sessions
    # `ConsoleChatStore` has no `set_active_session` method (verified by
    # grep, same finding noted in test_console_run_state_per_session.py) --
    # `switch_session` is the real API. Using the store's own
    # `switch_session` directly (rather than `controller.switch_session`)
    # keeps this test focused on `fleet_summary_counts`' relative-to-active
    # math without also exercising `mark_session_visited`'s side effects.
    controller.store.switch_session(session_a)
    controller._set_run_state(
        ConsoleRunState(ConsoleRunStatus.STREAMING, "run"), session_id=session_b
    )
    running, pending = controller.fleet_summary_counts()
    assert (running, pending) == (1, 0)

    controller._set_run_state(
        ConsoleRunState(ConsoleRunStatus.FAILED, "boom"), session_id=session_b
    )
    assert controller.run_marker_for(session_b) is ConsoleRunMarker.FINISHED_FAILED
