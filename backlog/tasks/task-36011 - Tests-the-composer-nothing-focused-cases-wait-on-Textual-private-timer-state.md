---
id: TASK-36011
title: >-
  Tests: the composer nothing-focused cases wait on Textual's private timer
  state
status: To Do
assignee: []
created_date: '2026-10-21 07:24'
labels:
  - tests
  - console
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the review of TASK-33620.15.7 (2026-10-21). That task made the typing and paste cases with nothing focused in `Tests/UI/test_console_composer_run_controls.py` wait for the mount's focus-restore timer before clearing focus, which cut their failure rate from 5-12 in 20 runs to 1 in 32. The wait reads Textual private attributes (the screen's timer list, each timer's callback and task). It is acceptable while Textual is pinned below 9, but it breaks silently if those internals change, and it passes at once if the screen has not yet armed the timer when the wait starts, so the race can come back without any test noticing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The two cases wait on something the app exposes (for example the restore having run, or focus having settled), not on Textual private attributes
- [ ] #2 The wait cannot pass before the mount's focus restore has been scheduled
- [ ] #3 Twenty interleaved runs of each case pass on the branch and on dev
<!-- AC:END -->
