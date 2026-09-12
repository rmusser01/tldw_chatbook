---
id: TASK-24602
title: Send-authority Run has no failed state so a failed run reads as Ready
status: Done
assignee:
  - '@claude'
created_date: '2026-08-30 00:53'
updated_date: '2026-08-30 03:14'
labels:
  - console
  - ux
  - inspector
  - critique-2026-08-29
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
project_console_send_authority derives Run from the branches: Inspector data incomplete, Recovery required, Waiting for approval, Blocked, Running, else Ready. There is no failed branch. A turn that returned HTTP 401 left the pinned authority block reading Run: Ready and Provider: ready while the transcript showed the failure, so the one surface pinned above the fold to answer what happens if I send now contradicted the transcript.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A run that ended in provider or transport failure renders a distinct Run state, not Ready
- [x] #2 The failed state names the failure and a specific next action
- [x] #3 A test asserts the projection returns the failed state for a failed run and Ready only when no failure is outstanding
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The pinned 'What happens if I send now?' block had no representation for a failed run, so a turn that returned HTTP 401 left 'Run: Ready' on screen beside a transcript reading 'Agent run failed'.

The signal already existed and nothing consumed it. The agent-run failure path calls _set_run_state(ConsoleRunState(ConsoleRunStatus.FAILED, visible_copy)); _build_console_inspector_state simply never read it, so the projection had no input that could distinguish a failure from an idle rail.

An earlier plan to key off the dispatch-recovery store was abandoned after checking: ConsoleDispatchRecoveryKind is ACCEPTED / DISPATCH_STARTED / REMOTE_* / CONTINUATION / QUARANTINED -- all about DISPATCH, none about a provider error. A 401 produces no recovery owner, so that route would have shipped a fix that never fires for the observed case.

Ordering matters and is stated in the code. 'Failed' sits BELOW incomplete / recovery-required / waiting-for-approval / blocked / running, because all of those describe what the NEXT send will do and that is the question this line asks; a past failure only describes the last one. It sits ABOVE 'Ready', because 'Ready' after a failure is the most misleading thing this line can say. Two tests pin the precedence: an active run and a pending approval each outrank a previous failure.

Tested at BOTH seams deliberately. A projection-only test passes with the screen still unwired -- which is precisely how the defect shipped -- so test_screen_wires_a_failed_run_into_the_pinned_authority_line drives the real ChatScreen, sets the controller's run state through _set_run_state (the only path that mutates the per-session map, per the parallel-agents spec), and asserts the built inspector state carries the failure and the projection renders it.

Modified: tldw_chatbook/Chat/console_display_state.py, tldw_chatbook/Widgets/Console/console_send_authority_summary.py, tldw_chatbook/UI/Screens/chat_screen.py, Tests/UI/test_console_run_inspector.py, Tests/UI/test_console_right_rail.py.
<!-- SECTION:NOTES:END -->
