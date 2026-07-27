---
id: TASK-1052
title: >-
  Shutdown-snapshot race: rounds armed before session registration only fail
  closed by timeout
status: Done
assignee: []
created_date: '2026-07-27 14:32'
updated_date: '2026-07-27 22:43'
labels:
  - console
  - approvals
  - shutdown
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Shutdown's per-session cancel fanout, and `close_session`'s deny path, both walk `_active_stream_tasks` to find which sessions have a live round to cancel/deny. A confirm or approval round that gets armed BEFORE its owning session id has been registered into `_active_stream_tasks` is invisible to both of those paths at the moment they run: `shutdown()` will not include it in the fanout, and `close_session` will not deny it either -- the round is simply not iterated.

This does not auto-approve or otherwise mis-decide anything -- the round still fails closed via its own `_MCP_APPROVAL_POLL_SECONDS`/deadline timeout loop (up to the full ~120s configured timeout), same as any other unresolved round. The gap is purely one of promptness: a round caught in this window sits until its own timeout elapses instead of observing the shutdown/close signal immediately like every other in-flight round does.

`Tests/UI/test_skill_install_concurrent_confirms.py::test_bare_shutdown_flag_alone_does_not_deny_a_real_session_round` currently pins this behavior as evidence (a bare `_shutdown_requested` flag with no corresponding `_active_stream_tasks` entry does not, by itself, deny a real armed round) -- that test is the reference point for reproducing and then closing this gap.

**Scope note (added during implementation):** the original AC#2 ("the same promptness holds for `close_session`'s deny path") has been dropped rather than delivered. `close_session`'s deny path is architecturally different from `shutdown()`'s: it is deliberately session-scoped (`_signal_stop(session_id=session_id)`, gated by `_active_stream_belongs_to_session`) and, by design, must never reach for a GLOBAL signal the way `shutdown()`'s dedicated `_shutdown_requested` does -- doing so would let closing one session deny another session's unrelated round. `_active_assistant_message_ids`/`_active_stream_tasks`/`_active_cancel_events` are all registered together, synchronously, in the same call (see `_run_agent_reply`), so `close_session`'s `owns_active_stream` gate and a round's own `_active_cancel_events` entry are always in sync with each other -- there is no independently-fixable promptness gap here of the same shape as shutdown's. Closing this gap for real would need a NEW, always-on, per-session signal that survives arming-before-registration (not "OR in an existing global flag," which is the one seam this task's fix uses) -- a materially different mechanism, left as a candidate for separate future work rather than folded into this fix.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A round armed for a session before that session appears in `_active_stream_tasks` observes `shutdown()`'s cancel fanout promptly (not only via its own timeout).
- [x] #2 A regression test arms a round ahead of session registration, triggers shutdown/close, and asserts the round resolves before its timeout deadline.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm _shutdown_requested is genuinely never reset (grep-audit) before relying on it as a safe global OR-in signal.
2. Fix the seam: _is_session_cancelled's real-session-id branch also checks _shutdown_requested.is_set() (in addition to that session's own _active_cancel_events entry), keeping ONE definition shared by all three bridges.
3. Flip Tests/UI/test_skill_install_concurrent_confirms.py::test_bare_shutdown_flag_alone_does_not_deny_a_real_session_round into the new desired-behavior assertion; grep sibling mcp/script suites for other _shutdown_requested pins (none found targeting the real-session-id gap specifically -- the others already use session_id=None, which already read the flag).
4. Add a TASK-1050 interplay regression test: two rounds armed for different sessions, neither registered in _active_cancel_events, bare shutdown flag set -> both deny within one poll interval, both rounds' round-keyed accounting (pending approvals + parked payloads) clean up.
5. Run the two specified gate suites; confirm only the two known pre-existing failures remain.
6. Re-scope AC#2 (close_session's deny path): architecturally close_session's per-session gate does not consult _shutdown_requested (by design -- closing one session must never globally deny others), and its own owns_active_stream gate is always in sync with _active_cancel_events registration (both set atomically in the same code paths) -- so it has no independently fixable analog to this specific fix without a new, always-on per-session signal, which is a materially different mechanism out of scope here. Document and descope rather than falsely mark done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed the shutdown-snapshot race by widening the single seam all three worker-thread bridges (request_mcp_approvals, request_skill_install_confirm, request_skill_script_confirm) already poll through: `_is_session_cancelled`'s real-session-id branch now ORs in the global, never-reset `_shutdown_requested` Event alongside that session's own `_active_cancel_events` entry, instead of checking only the latter. Verified (grep-audit) `_shutdown_requested` is set exactly once, only inside `shutdown()`, and never `.clear()`-ed anywhere in the module or its tests, so this cannot spuriously deny a live round during normal operation -- it can only ever fire during/after real process teardown.

Flipped the pinned evidence test (`test_bare_shutdown_flag_alone_does_not_deny_a_real_session_round` -> `test_bare_shutdown_flag_alone_denies_a_real_session_round_within_one_poll_interval` in Tests/UI/test_skill_install_concurrent_confirms.py) to assert the new contract: a bare `_shutdown_requested` flag, with no per-session `_signal_stop` fanout at all, now denies a real-session round within one `_MCP_APPROVAL_POLL_SECONDS` poll interval. Used a 30s confirm timeout (far longer than the poll interval) so early resolution can only be attributed to the shutdown signal, not the round's own deadline. Grepped every sibling `_shutdown_requested` reference across the mcp/install/script suites (test_console_mcp_approval.py, test_console_skill_install_confirm.py, test_console_skill_script_confirm.py, test_skill_script_concurrent_confirms.py) -- none of them pin the real-session-id gap this fix closes; they all call the bridges with `session_id=None`, which already read `_shutdown_requested` via `_is_active_session_cancelled`'s pre-existing fallback, so none needed changes.

Added `test_shutdown_flag_alone_denies_both_unregistered_sessions_rounds_and_cleans_accounting` (TASK-1050 interplay): two rounds armed for two DIFFERENT sessions, neither registered in `_active_cancel_events`, bare shutdown flag set -> both deny within one poll interval, and TASK-1050's round-keyed `discard_pending_round` + guarded `_parked_skill_install_payloads` pop clean up both sessions' accounting independently (no cross-talk, no stale entries).

Scoping preserved: `test_request_mcp_approvals_unrelated_session_stop_does_not_cross_cancel` / `test_unrelated_session_stop_does_not_deny` (mcp + install suites) still pass untouched -- an unrelated session's `_signal_stop` still only ever touches ITS OWN `_active_cancel_events` entry, never `_shutdown_requested`; `_stop_requested` remains out of every bridge's poll condition.

Descoped AC#2 (close_session's deny path) -- see the Description's "Scope note" for the architectural reasoning (close_session is deliberately session-scoped and must never reach for a global signal; its own owns_active_stream gate is always in sync with a round's _active_cancel_events registration since both are set atomically together in _run_agent_reply, so it has no independently-fixable gap of the same shape). Not implemented here; would need a new per-session, always-on signal as a separate follow-up.

Gates run: Tests/UI/test_skill_install_concurrent_confirms.py + Tests/Chat/test_skill_script_concurrent_confirms.py + Tests/UI/test_console_mcp_approval.py + Tests/UI/test_console_skill_install_confirm.py = 69 passed, 2 failed (both pre-existing/known: CSS-geometry batch-row zero-size assertion, mcp cancellation execution-log error-message field -- unrelated to this change, present on HEAD before this fix). Tests/UI/test_console_parallel_runs.py + Tests/Chat/test_console_run_state_per_session.py = 35 passed, 0 failed.

Modified files: tldw_chatbook/Chat/console_chat_controller.py (_is_session_cancelled + shutdown() docstring), Tests/UI/test_skill_install_concurrent_confirms.py (flipped test + new interplay test + import).
<!-- SECTION:NOTES:END -->
