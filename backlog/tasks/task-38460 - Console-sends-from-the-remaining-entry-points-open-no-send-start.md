---
id: TASK-38460
title: 'Console: sends from the remaining entry points open no send start'
status: To Do
assignee: []
created_date: '2026-10-31 09:07'
labels:
  - console
  - performance
dependencies:
  - TASK-36610
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-36610 gave each send its own start window (from its acceptance to its own provider request, 3 s at most), opened at the runtime's acceptance (the composer's sends, a queued prompt's dispatch, an agent wake, a goal iteration) and around an agent's chat start. Some send entry points reach the controller through neither, so they still open no start and their display reads (Library activity rows, the Inspector's registry and appearance reads) go off the loop during it: resumed paused preparations (ConsoleChatController._continue_prepared_submission), the buddy conversation's submission, the mobile web server's submission, and the fleet-wake and goal-run controller fallbacks used when no runtime submitter is bound. This costs first-send time only; no wrong data is shown. The base build before TASK-36610 behaved the same. Found by TASK-36610's review.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each listed entry point opens a send start at its acceptance that ends at its own provider request, or measured evidence shows it need not
- [ ] #2 Deterministic tests fail on the base build for each entry point that gains a start
<!-- AC:END -->
