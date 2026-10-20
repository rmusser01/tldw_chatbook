---
id: TASK-35574
title: test_a_probe_settling_mid_poll_still_reaches_the_console passes every run on dev
status: To Do
created_date: 2026-10-19 05:30
dependencies: []
labels:
- follow-up
- tests
- console
priority: low
references:
- Tests/UI/test_console_switcher_local_probe.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in Phase 9's post-rebase interaction check (2026-10-19). `Tests/UI/test_console_switcher_local_probe.py::test_a_probe_settling_mid_poll_still_reaches_the_console` failed serially 2 of 3 times on plain dev 69771c1811, and 3 of 3 on Phase 9 rebased onto it, always with the same assertion: the Console read 'Ready · not tested' where the test expects 'Not ready · refused :9099'. Those runs were at load 30-60; at load about 10 it passed 3 of 3 serial at 917a5ebe85, so the failure depends on load, and Phase 9 is not shown to make it worse. It runs in no PR lane, so no PR has seen it fail. Whether the product or the test is at fault (a probe that settles mid-poll never reaching the Console, or a test that reads before the next poll) is not known yet.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The cause is found and recorded in this task: a product defect (a settled probe that never reaches the Console's readiness) or a test-timing defect
- [ ] #2 The test passes 10 of 10 serial runs under the conditions that reproduce the failure (a stated machine load, or a deterministic reproduction of the mid-poll settle), and still fails when the mid-poll settle is not propagated
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Cause found 2026-10-20 (AC#1), while fixing the same failure in TASK-33005.7's census test (b94aece7d6). A test defect, not a product one: the test's first hand-called poll can run before the Console's attach reconcile has started the view. The poll timer does not exist yet then (the reconcile creates it), so `_stop_console_credential_poll_timer()` is a no-op; the poll returns at its attach gate (`_console_attach_reconciled` false), and the header keeps 'Ready · not tested' unless the reconcile finishes during `pilot.pause()`. Instrumenting every early return of `_poll_console_credential_readiness` in a scratch copy showed it: failing runs logged only the attach-gate return. The lock and pre-gate returns did not fire.

Measured fix, not applied here (the test is dev's and runs in no lane): wait, after `_wait_for_selector`, until `console._console_attach_reconciled and not console._console_attach_reconcile_running` (the file's `_wait` with `tries=500`, which Phase 9 added for TASK-33005.7). With it the test passed 10 of 10 serial runs.

AC#2's control does not hold yet, and the review of b94aece7d6 measured why. With the evidence version dropped from both the pre-gate and the snapshot (scratch copy, the wait added, every poll return logged, 4 runs), the first poll took the full sync and committed its pre-gate, the second returned at the pre-gate, and the header still read 'Ready · not tested' right after the second poll in all 4 runs. 3 of the 4 still passed, because `_sync_native_console_chat_ui` repainted during `pilot.pause()` (a stack logged at the header writer; most likely the `call_after_refresh(self._sync_native_console_chat_ui)` that `_start_console_view_after_reconciliation` queues). With the poll made a no-op, that repaint alone passed the Inspector test in 7 of 10 cases. So a control that bites must exclude that repaint: read `#workbench-header-status` right after the second poll and before any pause, or drain the queued sync before the settle, and land the settle during the build as before TASK-34956.
<!-- SECTION:NOTES:END -->
