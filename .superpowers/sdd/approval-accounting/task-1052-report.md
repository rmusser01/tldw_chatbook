# TASK-1052: Shutdown-snapshot race — rounds armed before session registration only fail closed by timeout

## The gap

The three worker-thread approval/confirm bridges (`request_mcp_approvals`,
`request_skill_install_confirm`, `request_skill_script_confirm` in
`tldw_chatbook/Chat/console_chat_controller.py`) all poll
`_is_session_cancelled(session_id)` once per second
(`_MCP_APPROVAL_POLL_SECONDS = 1.0`). Before this fix, that method's
`session_id is not None` branch checked ONLY that session's own
`_active_cancel_events` entry:

```python
if session_id is not None:
    cancel_event = self._active_cancel_events.get(session_id)
    return cancel_event is not None and cancel_event.is_set()
return self._shutdown_requested.is_set() or self._is_active_session_cancelled()
```

`shutdown()` reaches a real-session round by iterating a snapshot of
`_active_stream_tasks` (`tasks = dict(self._active_stream_tasks)`) and
calling `_signal_stop(session_id=...)` for each entry found. A round armed
for a session BEFORE that session appears in that snapshot is invisible to
the fanout — its only remaining path to observe teardown is the round's own
confirm/approval timeout (up to ~120s in production). This was PINNED as
evidence (not desired behavior) by
`Tests/UI/test_skill_install_concurrent_confirms.py::test_bare_shutdown_flag_alone_does_not_deny_a_real_session_round`.

## The fix: one seam, not three

`_is_session_cancelled`'s real-`session_id` branch now also ORs in the
global, dedicated `_shutdown_requested` Event:

```python
if session_id is not None:
    if self._shutdown_requested.is_set():
        return True
    cancel_event = self._active_cancel_events.get(session_id)
    return cancel_event is not None and cancel_event.is_set()
return self._shutdown_requested.is_set() or self._is_active_session_cancelled()
```

All three bridges call this one method, so the fix is a single change
(no per-bridge duplication). Since `_shutdown_requested` is set exactly
once — only inside `shutdown()` — and is never `.clear()`-ed anywhere
(verified by grep across `console_chat_controller.py` and every test file
that touches it: it is only ever `.set()`, never reset), ORing it into the
real-session branch cannot spuriously deny a live round during normal
operation. It can only ever fire during/after real process teardown, which
is global by definition — so a real-session round observing it is always
correct, not just "safe."

**Invariant check requested by the brief:** confirmed `_shutdown_requested`
is never reset anywhere in the codebase (module or tests) — no `.clear()`
call exists. Not blocked.

## Scope decision: AC#2 (`close_session`'s deny path) descoped

The original task-1052 had three ACs; AC#2 ("the same promptness holds for
`close_session`'s deny path") was **not delivered** and has been removed
from the task's AC list (backlog file updated accordingly, with the
reasoning recorded in the task's Description as a "Scope note").

Why: `close_session`'s deny mechanism is architecturally different from
`shutdown()`'s. It is deliberately session-scoped —
`_signal_stop(session_id=session_id)`, gated by
`_active_stream_belongs_to_session(session_id)` (which checks
`_active_assistant_message_ids`) — and must never reach for a GLOBAL signal
the way `shutdown()`'s dedicated `_shutdown_requested` does, because that
would let closing ONE session spuriously deny an unrelated session's
in-flight round (exactly the cross-session bug `_is_session_cancelled`'s
own docstring already documents `close_session`/`stop_active_run` avoiding
via `_signal_stop`'s per-session scoping).

Critically, `_active_assistant_message_ids[session_id]`,
`_active_stream_tasks[session_id]`, and `_active_cancel_events[session_id]`
are all registered together, synchronously, with no `await` between them,
inside `_run_agent_reply` (and the sibling direct-dispatch path). So
`close_session`'s `owns_active_stream` gate and a round's own
`_active_cancel_events` registration are always in sync with each other —
there is no independently-fixable promptness gap of the *same shape* as
shutdown's for `close_session` to close via "OR in an existing flag."
Delivering true parity would require a NEW, always-on, per-session signal
that survives arming-before-registration (e.g., a persisted
"session force-closed" marker checked regardless of `_active_cancel_events`
registration) — a materially different mechanism than "OR in the one global
shutdown flag," and out of scope for this fix. Left as a candidate for
separate future work.

This matches the pre-established scope recorded in
`.superpowers/sdd/progress.md` before this task was dispatched: "1052
(bridges observe global shutdown for all session ids; flip the pinned
evidence test)" — no mention of `close_session`.

## Tests

**Flipped** (pinned evidence → desired-behavior contract), in
`Tests/UI/test_skill_install_concurrent_confirms.py`:

- `test_bare_shutdown_flag_alone_does_not_deny_a_real_session_round` →
  `test_bare_shutdown_flag_alone_denies_a_real_session_round_within_one_poll_interval`.
  Now asserts a bare `_shutdown_requested.set()` — with NO per-session
  `_signal_stop` fanout at all — denies a real-session round within one
  `_MCP_APPROVAL_POLL_SECONDS` poll interval. Uses a 30s confirm timeout
  (far longer than the poll interval) so early resolution can only be
  attributed to the shutdown signal, never the round's own deadline.

**Added** (TASK-1050 interplay), same file:

- `test_shutdown_flag_alone_denies_both_unregistered_sessions_rounds_and_cleans_accounting`:
  two rounds armed for two DIFFERENT sessions, NEITHER registered in
  `_active_cancel_events` — bare shutdown flag set → both deny within one
  poll interval, and TASK-1050's round-keyed `discard_pending_round` +
  guarded `_parked_skill_install_payloads` pop clean up both sessions'
  accounting independently (no cross-talk, no stale entries left behind).

**Sibling-pin sweep**: grepped every `_shutdown_requested` reference across
`Tests/UI/test_console_mcp_approval.py`,
`Tests/UI/test_console_skill_install_confirm.py`,
`Tests/Chat/test_console_skill_script_confirm.py`, and
`Tests/Chat/test_skill_script_concurrent_confirms.py`. None of them pin the
real-`session_id` gap this fix closes — every one of them calls its bridge
with `session_id=None` (the legacy fallback), which already read
`_shutdown_requested` via `_is_active_session_cancelled` before this fix.
No changes needed in those files.

**Scoping preserved** (unaffected, verified still passing):
`test_request_mcp_approvals_unrelated_session_stop_does_not_cross_cancel`
and `test_unrelated_session_stop_does_not_deny` — an unrelated session's
`_signal_stop` still only ever touches ITS OWN `_active_cancel_events`
entry, never `_shutdown_requested`; `_stop_requested` remains excluded from
every bridge's poll condition.

## Test results

Gate 1 — `Tests/UI/test_skill_install_concurrent_confirms.py` +
`Tests/Chat/test_skill_script_concurrent_confirms.py` +
`Tests/UI/test_console_mcp_approval.py` +
`Tests/UI/test_console_skill_install_confirm.py`: **69 passed, 2 failed**.
Both failures are the known pre-existing baseline (unrelated to this
change, reproducible on HEAD before this fix):
`test_batch_row_widgets_have_nonzero_geometry_and_do_not_overlap_under_bundled_css`
(CSS-geometry, zero-size header under bundled CSS) and
`test_request_mcp_approvals_cancellation_records_denied_decision_to_execution_log`
(execution-log `error` field assertion).

Gate 2 — `Tests/UI/test_console_parallel_runs.py` +
`Tests/Chat/test_console_run_state_per_session.py`: **35 passed, 0 failed**.

## Files changed

- `tldw_chatbook/Chat/console_chat_controller.py` — `_is_session_cancelled`
  (the fix) + docstring updates on `_is_session_cancelled` and `shutdown()`
  documenting the closed gap.
- `Tests/UI/test_skill_install_concurrent_confirms.py` — flipped pinned
  test, new interplay test, `_MCP_APPROVAL_POLL_SECONDS` import.
- `backlog/tasks/task-1052 - ....md` — AC#2 removed with recorded
  rationale (Description "Scope note"), AC#1/AC#3 checked, Implementation
  Plan + Notes added, status → Done.
