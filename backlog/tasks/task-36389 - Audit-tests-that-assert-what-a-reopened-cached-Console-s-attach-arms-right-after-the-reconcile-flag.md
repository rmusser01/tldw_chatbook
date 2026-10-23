---
id: TASK-36389
title: Audit tests that assert what a reopened cached Console's attach arms right after the reconcile flag
status: To Do
created_date: 2026-10-22 23:58
labels:
- follow-up
- tests
- console
priority: low
references:
- Tests/UI/pump_probe.py
- backlog/docs/lessons-testing-evidence.md
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-35833 (PR #3212, merged 2026-10-22). On a reopened, cached Console, `_console_attach_reconciled` can be True before the work the attach queued has run. The reconcile skips the view start, and the transcript poll is armed only through the wake hook's deferred `call_later`. TASK-35833 fixed the two tests in `test_console_runtime_ownership.py` that asserted the poll right after the flag, using `pump_runs(screen, 5.0)` as a same-screen FIFO barrier. TASK-35574's switcher probe test is the same attach-reconcile race in the other direction. Other files were not audited.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every test that reopens a cached Console and then asserts a timer, poll or other deferred effect of the attach is listed in the task notes
- [ ] #2 Each listed test waits for the reconcile to end and for the screen's queue to drain before that assertion, or the notes record why it cannot race
<!-- AC:END -->
