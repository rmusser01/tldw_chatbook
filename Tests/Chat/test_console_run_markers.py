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


def test_closing_active_session_clears_new_active_neighbors_marker(
    controller_with_two_sessions,
):
    """Fix round 1 / IMPORTANT 1: `ConsoleChatStore.close_session` auto-
    activates a neighbor when the ACTIVE session is closed (console_chat_
    store.py ~594-604). That neighbor is now the VIEWED session exactly as
    if `switch_session` had navigated to it, so its stamped unvisited
    outcome must clear -- `close_session` must call `mark_session_visited`
    for the newly-activated neighbor, not just `switch_session`.
    """
    controller, session_a, session_b = controller_with_two_sessions
    assert controller.store.active_session_id == session_b

    # session_a is non-active (session_b is active), so its COMPLETED
    # transition stamps an unvisited FINISHED_OK outcome.
    controller._set_run_state(
        ConsoleRunState(ConsoleRunStatus.COMPLETED, "done"), session_id=session_a
    )
    assert controller.run_marker_for(session_a) is ConsoleRunMarker.FINISHED_OK

    # Closing the ACTIVE session (session_b) leaves session_a as the only
    # remaining session, so the store auto-activates it.
    controller.close_session(session_b)
    assert controller.store.active_session_id == session_a
    assert controller.run_marker_for(session_a) is ConsoleRunMarker.NONE


def test_terminal_transition_clears_leaked_pending_approval_flag(
    controller_with_two_sessions,
):
    """Fix round 1 / IMPORTANT 2: a terminal COMPLETED/FAILED run has no
    live approval left to decide, so `_set_run_state`'s terminal branch
    must discard any leaked `_pending_approvals` entry alongside stamping
    `_unvisited_outcomes` -- otherwise NEEDS_APPROVAL (which outranks a
    stamped outcome in `run_marker_for`) permanently masks FINISHED_OK/
    FINISHED_FAILED.
    """
    controller, session_a, session_b = controller_with_two_sessions
    assert controller.store.active_session_id == session_b

    controller.set_run_pending_approval(session_a, True)
    controller._set_run_state(
        ConsoleRunState(ConsoleRunStatus.FAILED, "boom"), session_id=session_a
    )
    assert controller.run_marker_for(session_a) is ConsoleRunMarker.FINISHED_FAILED

    controller.mark_session_visited(session_a)
    controller.set_run_pending_approval(session_a, True)
    controller._set_run_state(
        ConsoleRunState(ConsoleRunStatus.COMPLETED, "done"), session_id=session_a
    )
    assert controller.run_marker_for(session_a) is ConsoleRunMarker.FINISHED_OK


def test_terminal_transition_clears_pending_approval_flag_on_active_session(
    controller_with_two_sessions,
):
    """PA-T9 finding #2 (deferred from the Task 7 review): the terminal-
    transition discard exercised above lived ONLY inside `_set_run_state`'s
    non-active branch (alongside the unvisited-outcome stamp), so a
    pending-approval flag on the session you were actually LOOKING AT
    survived its own run's termination -- a misleading NEEDS_APPROVAL badge
    with no round left behind it. The discard must apply regardless of
    whether the terminating session is the active one.
    """
    controller, session_a, session_b = controller_with_two_sessions
    controller.store.switch_session(session_a)
    assert controller.store.active_session_id == session_a

    controller.set_run_pending_approval(session_a, True)
    assert controller.run_marker_for(session_a) is ConsoleRunMarker.NEEDS_APPROVAL

    controller._set_run_state(
        ConsoleRunState(ConsoleRunStatus.COMPLETED, "done"), session_id=session_a
    )
    # The active session's own terminal transition is never stamped as an
    # "unvisited" outcome (it is seen live), but the leaked pending-
    # approval flag must not survive it either.
    assert controller.run_marker_for(session_a) is ConsoleRunMarker.NONE

    # Same for a STOPPED transition (user hit Stop mid-approval-wait).
    controller.set_run_pending_approval(session_b, True)
    controller.store.switch_session(session_b)
    controller._set_run_state(
        ConsoleRunState(ConsoleRunStatus.STOPPED, "stopped"), session_id=session_b
    )
    assert controller.run_marker_for(session_b) is ConsoleRunMarker.NONE
