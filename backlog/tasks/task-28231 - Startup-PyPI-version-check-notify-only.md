---
id: TASK-28231
title: Startup PyPI version check (notify-only)
status: To Do
assignee: []
created_date: '2026-09-02 06:39'
labels:
  - ops
  - ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deferred row C36's cheap slice, promoted by TASK-26041: no update surface exists at all (hermes invests heavily here; ~20 updater commits in its latest delta). The slice: an async, failure-silent check of PyPI for a newer release, surfacing a one-line notice. No self-update.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 On startup an async check compares the installed version to PyPI and shows a non-blocking notice when newer exists
- [ ] #2 Network failure or slow response is silent and never delays boot (off the boot path per ADR-097)
- [ ] #3 A config knob disables it entirely
<!-- AC:END -->
