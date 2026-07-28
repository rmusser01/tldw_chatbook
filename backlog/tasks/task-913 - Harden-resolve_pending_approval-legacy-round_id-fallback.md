---
id: TASK-913
title: Harden resolve_pending_approval legacy round_id fallback
status: Done
assignee: []
created_date: '2026-07-27 03:55'
updated_date: '2026-07-27 19:18'
labels:
  - console
  - approvals
  - hardening
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
resolve_pending_approval's round_id=None fallback (production-unreachable; kept for legacy direct-call tests) scans _pending_approval_rounds.values() unlocked while a worker thread's finally can pop concurrently, and resolves by active session. Its twin resolve_pending_skill_script fails closed on a missing request_id. Make the fallback fail closed (or snapshot with list()) and migrate the legacy tests to pass round ids.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 No unlocked live-dict iteration remains in the fallback path.
- [x] #2 round_id=None either fails closed like resolve_pending_skill_script or is removed with tests migrated.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
AC #1 done as part of the Qodo-wave PR2 restack (F3b): `resolve_pending_approval`'s
`round_id=None` legacy fallback now takes its `_pending_approval_rounds.values()`
scan under the new `_approval_state_lock` (`with lock: round_states =
list(self._pending_approval_rounds.values())`), snapshotting before iterating
rather than walking the live dict a worker thread's `request_mcp_approvals`
can concurrently register/pop entries in. The `round_id is not None` branch's
`.get()` was guarded too, for consistency with every other access to this map.

(Historical — superseded by the AC#2 pass below.) AC #2 (the
fail-closed-vs-remove behavioral decision for the `round_id=None`
fallback itself) was explicitly out of scope for the fix this pass authorized
(a locking/thread-safety hardening pass, not a behavioral redesign of a
production-unreachable legacy path) -- changing it would also require
migrating every direct-call test that currently relies on the "resolves
whichever round belongs to the active session" fallback, which risks
destabilizing a fully-reviewed, live-smoked branch for a change nobody asked
for in this pass. Left unchecked and the task left in `In Progress` (not
`Done`) rather than silently closing an unimplemented AC. Recommend either a
follow-up task scoped to AC #2 alone, or re-scoping this task's AC list down
to #1 (already satisfied) if AC #2 is not actually wanted.

AC #2 (this pass): removed the round_id=None active-session-scan fallback
entirely from `resolve_pending_approval` and replaced it with a fail-closed
`if round_id is None: return`, mirroring `resolve_pending_skill_script`'s/
`resolve_pending_skill_install`'s identical contract. Verified exhaustively
that production has exactly one emitter/caller path
(`ChatApprovalCard._submit_batch_decisions` -> `ApprovalDecided` ->
`ChatScreen.handle_console_approval_decided` -> `resolve_pending_approval`)
and it always threads the real `round_id` through -- no production caller
relies on the None fallback, so no BLOCKED condition was hit.

Removing the scan makes AC #1's lock/snapshot protection for that branch
moot (the branch itself is gone); the remaining `round_id is not None`
branch's lock-guarded `.get()` is untouched and still guards against the
worker thread's concurrent register/pop.

Migrated every legacy direct-call test that relied on the fallback
(Tests/UI/test_console_mcp_approval.py, Tests/Chat/test_console_agent_swap.py)
to capture the real round_id from the mounted/parked payload
(`received[-1]["round_id"]` / `mounted[-1]["round_id"]`) and pass it
explicitly. `test_resolve_pending_approval_without_active_round_is_a_noop`
(no round armed at all) is kept as a trivial smoke test; a new test,
`test_resolve_pending_approval_without_round_id_fails_closed_and_leaves_round_pending`,
arms a real round for the (only) active session and asserts a None-id
resolve leaves it pending/undecided (TDD: this test fails against the
pre-fix fallback, since with no session ever created both the fallback's
"active" key and the round's "session_id" key default to the same `""`
and would otherwise match).

Files touched: tldw_chatbook/Chat/console_chat_controller.py (
`resolve_pending_approval` body + docstring; two nearby comments
referencing the removed "legacy fallback" cleaned up),
Tests/UI/test_console_mcp_approval.py, Tests/Chat/test_console_agent_swap.py.

Verification: `pytest Tests/UI/test_console_mcp_approval.py
Tests/UI/test_console_parallel_runs.py
Tests/UI/test_skill_install_concurrent_confirms.py
Tests/Chat/test_console_agent_swap.py` -> 102 passed, 2 failed. Both
failures (`test_batch_row_widgets_have_nonzero_geometry_and_do_not_overlap_under_bundled_css`,
`test_request_mcp_approvals_cancellation_records_denied_decision_to_execution_log`)
reproduce identically on the pre-change commit (verified via `git stash`),
so they are pre-existing baseline failures unrelated to this change.
<!-- SECTION:NOTES:END -->
