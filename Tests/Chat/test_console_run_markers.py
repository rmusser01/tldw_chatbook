"""Run markers: running/needs-approval/finished-unvisited (spec §6)."""

from __future__ import annotations

import threading

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


def test_approval_state_lock_serializes_snapshot_against_worker_thread_mutation(
    controller_with_two_sessions,
):
    """F2b regression (Qodo wave): ``fleet_summary_counts``'s snapshot of
    ``_pending_approvals`` and ``set_run_pending_approval``'s add/discard
    (reachable from the worker thread via ``request_mcp_approvals``) must
    be mutually exclusive under ``_approval_state_lock`` -- otherwise a
    worker-thread mutation racing the UI-thread's ~0.2s sync-tick
    iteration risks ``RuntimeError: Set changed size during iteration``.

    Deterministically forces the interleaving (never relies on scheduling
    luck, so this cannot flake): a worker thread holds the shared lock
    open for a controlled window, and the main thread's own attempt to
    read ``fleet_summary_counts`` is asserted to BLOCK for as long as the
    lock is held, then complete immediately once it is released.
    """
    controller, session_a, session_b = controller_with_two_sessions

    lock_acquired = threading.Event()
    release_lock = threading.Event()

    def _hold_lock() -> None:
        with controller._approval_state_lock:
            lock_acquired.set()
            release_lock.wait(timeout=2)

    holder = threading.Thread(target=_hold_lock)
    holder.start()
    assert lock_acquired.wait(timeout=1), "worker thread never acquired the lock"

    snapshot_done = threading.Event()
    result: dict[str, tuple[int, int]] = {}

    def _read_fleet_summary() -> None:
        result["counts"] = controller.fleet_summary_counts()
        snapshot_done.set()

    reader = threading.Thread(target=_read_fleet_summary)
    reader.start()
    try:
        # While the worker thread holds the lock, the reader's own
        # snapshot must be blocked -- this is the assertion that actually
        # proves mutual exclusion, not just "no exception happened to
        # occur this run".
        assert not snapshot_done.wait(timeout=0.2), (
            "fleet_summary_counts did not block on the held lock"
        )
    finally:
        release_lock.set()
        holder.join(timeout=2)

    assert snapshot_done.wait(timeout=2), "fleet_summary_counts never completed"
    reader.join(timeout=2)
    assert result["counts"] == (0, 0)


# ---------------------------------------------------------------------------
# TASK-1050: round-keyed pending-approval accounting.
# ---------------------------------------------------------------------------
#
# `_pending_approvals` used to be a plain `set[str]` of session ids
# (`set_run_pending_approval` was the sole writer) -- ANY approval-like
# bridge's teardown for a session cleared the badge for that session
# regardless of whether a SIBLING round (same bridge, or one of the other
# two bridges) was still outstanding. `add_pending_round`/`discard_
# pending_round` replace that with round-id-keyed accounting: a session
# reads as NEEDS_APPROVAL iff it has AT LEAST ONE outstanding round id,
# and discarding one round's id never touches another's. The three
# bridges (`request_mcp_approvals`/`request_skill_install_confirm`/
# `request_skill_script_confirm`) exercise this through real worker
# threads in their own test modules (`test_console_mcp_approval.py`,
# `test_skill_install_concurrent_confirms.py`,
# `test_skill_script_concurrent_confirms.py`) -- these tests pin the
# accounting primitive itself directly, mirroring how this file already
# drives `_set_run_state`/`set_run_pending_approval` as controller seams
# rather than through a live run.


def test_marker_stays_needs_approval_while_any_round_pending_for_session(
    controller_with_two_sessions,
):
    """Two independent approval-like rounds (e.g. one from the MCP bridge,
    one from the skill-install bridge) pending for the SAME session must
    both have to resolve before the badge clears -- discarding the first
    round's id alone must not clear a session that still has a second
    round outstanding.
    """
    controller, session_a, _session_b = controller_with_two_sessions
    assert controller.run_marker_for(session_a) is ConsoleRunMarker.NONE

    controller.add_pending_round(session_a, "round-mcp-1")
    assert controller.run_marker_for(session_a) is ConsoleRunMarker.NEEDS_APPROVAL

    controller.add_pending_round(session_a, "round-skill-install-1")
    assert controller.run_marker_for(session_a) is ConsoleRunMarker.NEEDS_APPROVAL

    # The FIRST round resolving must not clear the badge -- the second is
    # still outstanding.
    controller.discard_pending_round(session_a, "round-mcp-1")
    assert controller.run_marker_for(session_a) is ConsoleRunMarker.NEEDS_APPROVAL
    assert session_a in controller._pending_approvals

    # Only once the LAST round resolves does the badge clear.
    controller.discard_pending_round(session_a, "round-skill-install-1")
    assert controller.run_marker_for(session_a) is ConsoleRunMarker.NONE
    assert session_a not in controller._pending_approvals


def test_add_and_discard_pending_round_is_idempotent(controller_with_two_sessions):
    """Double-adding or double-discarding the SAME round id must not
    corrupt the per-session round-id accounting (no duplicate-clear, no
    raise on a redundant discard)."""
    controller, session_a, _session_b = controller_with_two_sessions

    controller.add_pending_round(session_a, "round-1")
    controller.add_pending_round(session_a, "round-1")  # duplicate add: no-op
    assert controller.run_marker_for(session_a) is ConsoleRunMarker.NEEDS_APPROVAL

    controller.discard_pending_round(session_a, "round-1")
    assert controller.run_marker_for(session_a) is ConsoleRunMarker.NONE

    # A second discard of the same (already-gone) id, and a discard of an
    # id that was never added, must both be safe no-ops.
    controller.discard_pending_round(session_a, "round-1")
    controller.discard_pending_round(session_a, "never-added-round")
    assert controller.run_marker_for(session_a) is ConsoleRunMarker.NONE


def test_idempotent_discard_of_one_round_never_touches_a_sibling_round(
    controller_with_two_sessions,
):
    """Discarding an already-discarded round id twice must not accidentally
    discard a DIFFERENT, still-live round for the same session (guards
    against an implementation that clears the whole set instead of just
    the one id)."""
    controller, session_a, _session_b = controller_with_two_sessions

    controller.add_pending_round(session_a, "round-1")
    controller.add_pending_round(session_a, "round-2")

    controller.discard_pending_round(session_a, "round-1")
    controller.discard_pending_round(session_a, "round-1")  # redundant, must no-op
    assert controller.run_marker_for(session_a) is ConsoleRunMarker.NEEDS_APPROVAL

    controller.discard_pending_round(session_a, "round-2")
    assert controller.run_marker_for(session_a) is ConsoleRunMarker.NONE


def test_legacy_boolean_shim_stacks_with_a_real_round_until_both_clear(
    controller_with_two_sessions,
):
    """Pins the deprecated `set_run_pending_approval` shim's documented
    composition contract: it registers/discards a synthetic sentinel round
    id, so it never clobbers a REAL round id registered separately via
    `add_pending_round` -- and, symmetrically, discarding the real round
    alone does not clear a badge the shim's own `True` call is still
    holding up. This is exactly why `ChatScreen._park_console_approval`
    guards its own shim call with `has_pending_approval_round` first
    (see that method's docstring) rather than calling the shim
    unconditionally.
    """
    controller, session_a, _session_b = controller_with_two_sessions

    controller.add_pending_round(session_a, "real-round-1")
    controller.set_run_pending_approval(session_a, True)  # legacy shim, redundant
    assert controller.run_marker_for(session_a) is ConsoleRunMarker.NEEDS_APPROVAL

    # Discarding only the real round leaves the shim's own sentinel up.
    controller.discard_pending_round(session_a, "real-round-1")
    assert controller.run_marker_for(session_a) is ConsoleRunMarker.NEEDS_APPROVAL

    # The shim's own `False` call clears its sentinel, and only then does
    # the badge fully clear.
    controller.set_run_pending_approval(session_a, False)
    assert controller.run_marker_for(session_a) is ConsoleRunMarker.NONE
