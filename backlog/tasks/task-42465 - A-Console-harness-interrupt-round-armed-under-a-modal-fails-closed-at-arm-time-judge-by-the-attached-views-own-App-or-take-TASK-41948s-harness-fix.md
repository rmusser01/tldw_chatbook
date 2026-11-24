---
id: TASK-42465
title: "A Console harness interrupt round armed under a modal fails closed at arm time: judge by the attached view's own App, or take TASK-41948's harness fix"
status: Done
assignee:
- '@claude'
created_date: 2026-11-23 06:38
updated_date: 2026-11-24 04:09
labels:
- follow-up
- console
- approvals
- tests
- adr
- needs-owner-decision
priority: low
references:
- tldw_chatbook/Chat/console_interrupt_rounds.py
- tldw_chatbook/Chat/console_runtime.py
- backlog/decisions/314-indefinite-human-approval-waits.md
- Tests/UI/test_console_session_tab_close.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during round 22 of the follow-up burn-down (2026-11-23), by TASK-42424. Checked on the round-22 stack 336d15338b (dev 2f7790873a plus round 22's stacked branches); paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Owner sign-off needed.** Either fix below changes ADR-314's "TASK-41637 clarification (2026-11-18)" section (`backlog/decisions/314-indefinite-human-approval-waits.md`, lines 107-126). That section now says that a harness round whose arm-time check runs while a modal covers the view still fails closed. The owner chooses the approach and approves the new wording before the change lands.

**Why.** `InterruptRoundHost.run_round` (`Chat/console_interrupt_rounds.py:1294-1308`) settles a round as `cancelled` at arm time when `app_gone` is true. That is `not app.is_running` (line 1305), and then, at line 1308, `_approval_view_is_detached()` (line 1623). `_approval_view_is_detached` asks the router's `has_answerable_view()` (`Chat/console_runtime.py:5402`), which is False whenever the attached view is not the displayed screen.

The defect is harness-only. In production `controller.app` is the running App, so `app_gone` is already False before line 1308 consults the view, and a modal cannot cancel a round. Console harnesses instead re-point `controller.app` at a never-run `TldwCli` beside a live view. There the view check decides, so a modal over the view (such as the Close dialog) makes the round fail closed. That was TASK-42424's cause. Its fix is in the tests, which now wait for the round's card or park notice before opening a modal, and its ADR-314 wording records the limit.

**Two approaches.**
- This task's: the view condition asks whether the attached view's own App is running (`view.app.is_running`) rather than whether the view is on screen, so a modal no longer matters in a harness.
- TASK-41948's (open, low, 2026-11-19): the harnesses point `controller.app` at the running host, and the guard then drops the view condition and becomes `not app.is_running` with the AttributeError fallback. That also removes the harness-only branch from product code.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The owner has chosen between this approach and TASK-41948's and approved the matching ADR-314 wording; the choice and its date are recorded in the notes before any code change merges. If TASK-41948's is chosen, this task closes as covered by it
- [x] #2 With the chosen fix, a harness round armed while a modal covers its attached view keeps waiting and can be answered after the modal closes; a test shows it and fails on the current check
- [x] #3 A round armed with no running App still settles as `cancelled` within one poll, and a host without `is_running` still waits: TASK-41637's tests pass unedited (superseded in part, 2026-11-24: one TASK-41637 test pinned the dropped view rule and was rewritten; every other one passes unedited)
- [x] #4 ADR-314's clarification states the chosen rule
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Owner decision, 2026-11-23:** "Fix the harnesses (Recommended)", that is TASK-41948's approach (the harnesses point `controller.app` at the running host and the guard drops its view condition), not this task's (judge by the attached view's own App). The matching ADR-314 wording is in TASK-41948's change, for the owner to review with that PR; the choice itself did not quote the wording.

Closed as covered by TASK-41948. ACs #2-#4 describe the chosen fix's outcome, and TASK-41948 meets them (evidence in its notes):
- #2: `_verify_question_armed_under_a_modal_waits_and_is_answered_after_it`, in `test_session_close_pending_journeys`, arms a question while the Close dialog covers Console. It waits and is answered after Stay. On the old harness and guard it was red 3/3 (`{'answered': False, 'reason': 'cancelled'}`); now it is green.
- #3, superseded in part. One TASK-41637 test, `test_never_run_app_round_stays_pending_while_a_view_can_answer`, pinned the view condition this change drops, so it was rewritten as `test_never_run_app_round_is_refused_even_while_a_view_can_answer`. The rest pass unedited: `test_approval_round_with_an_app_that_never_ran_is_refused_promptly` and the other no-running-App tests settle within one poll, and the `_ThreadApp` host without `is_running` still waits.
- #4: ADR-314's TASK-41637 clarification states the new rule.
<!-- SECTION:NOTES:END -->
