---
id: TASK-32288
title: Approval deadline countdown never ticks
status: Done
assignee: []
created_date: '2026-09-10 19:16'
updated_date: '2026-09-10 20:57'
labels:
  - console
  - approvals
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The deadline line is rendered once when the batch is set; with [mcp] approval_timeout_seconds configured the card shows a fixed 'Auto-denies in 2:00' for the whole window. Source: live approval-card / MCP-permissions UX review 2026-09-10 against dev 3315241674 (snapshot .impeccable/critique/2026-09-10T17-31-53Z__hatbook-widgets-chat-widgets-chat-approval-card-py.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 With a finite timeout the countdown updates at least once per second and stops when the card hides.
- [x] #2 With the default of 0 nothing is shown.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read the current `set_batch` deadline block and `format_approval_deadline`; grep both test files for `deadline`/`Auto-denies` and for any test pinning the text as static.
2. Add a failing mounted test in `Tests/UI/test_console_mcp_approval.py` (where the `_CardHarnessApp` pilot harness lives) asserting the countdown ticks after `pilot.pause(1.1)`, stays alive across an unchanged-round re-sync, stops on `set_batch([])`, and stays unarmed for `timeout_seconds=0`.
3. Implement: `ChatApprovalCard` gains `_deadline_at`/`_deadline_timer` state, `_render_deadline`/`_tick_deadline`/`_stop_deadline_timer` methods (mirroring `chat_question_card.py`'s existing countdown pattern), and `set_batch` arms/re-arms them from a local `time.monotonic()` deadline.
4. Run both test files, compare the failing-name set to the pre-change baseline.
5. Close the task and commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
ChatApprovalCard now ticks its own deadline instead of freezing the text set_batch painted once. set_batch computes self._deadline_at = time.monotonic() + total (via a new shared _coerce_timeout_total helper that format_approval_deadline also uses, switching from truncation to math.ceil so an instant-of-arm render reads the full 1:30 rather than 1:29 off call-overhead microseconds) and calls the new _render_deadline, which paints #approval-deadline from the remaining time and arms a self.set_interval(1.0, self._tick_deadline) the first time a deadline is live; _tick_deadline re-renders every second and _render_deadline stops the timer once the remaining text goes empty. The countdown never reads the controller's own auto-deny clock, per the module docstring. Re-syncs are unaffected: set_batch's existing unchanged-round guard (same round_id/phase/calls) still early-returns before the deadline block runs, so a resume-state re-sync never resets the clock; every other call that reaches the deadline block (new/changed round, or a clear) unconditionally stops any previous timer and re-arms, and calls=[] forces total=0 so the label blanks/hides and the timer stops even if timeout_seconds is nonzero. Added test_the_deadline_countdown_ticks_and_stops_on_clear_or_timeout in Tests/UI/test_console_mcp_approval.py covering tick-after-pilot.pause(1.1), deadline preserved across a same-round re-sync, timer stopped on set_batch([]), no timer/text for timeout_seconds=0, and re-arm on a new round; Tests/UI/test_chat_approval_card.py needed no changes since its format_approval_deadline assertions are all whole-second ints. No regressions: baseline and post-change runs of both files show the same 4 pre-existing unrelated failures plus the new test passing (121->122 passed, 4->4 failed, same names). Files changed: tldw_chatbook/Widgets/Chat_Widgets/chat_approval_card.py, Tests/UI/test_console_mcp_approval.py.
<!-- SECTION:NOTES:END -->
