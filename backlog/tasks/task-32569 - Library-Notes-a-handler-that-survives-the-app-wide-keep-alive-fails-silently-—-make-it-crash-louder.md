---
id: TASK-32569
title: >-
  Library Notes: a handler that survives the app-wide keep-alive fails silently
  — make it crash louder
status: To Do
assignee: []
created_date: '2026-09-14 22:44'
labels:
  - library
  - notes
  - rider
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
task-32533 shipped an app-level keep-alive: an unhandled exception inside a widget's message pump no longer ends the app, it logs and leaves the pump dead. That is the right trade against a hard exit, but it converts every unguarded failure from a loud crash into a pane that is still painted and no longer responds. The wave already relied on the log grep rather than the screen because the error notification's rendering could not be captured — when the pump that raised IS the screen, the toast may never render at all. A user therefore gets a dead panel and no signal. Decide what the signal should be (an in-pane failure state on the widget that raised, a persistent status line, a Details affordance pointing at the log record) and make it something a live walk can see.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A widget whose pump dies to an unhandled exception is visibly marked as failed, not merely inert
- [ ] #2 The mark survives when the raising pump is the screen itself, where a toast may not render
- [ ] #3 The signal names where the detail is (log record id or file) without leaking paths or user content into the UI
- [ ] #4 One live capture shows the failed state, so the guide can stamp it instead of pointing at a log
<!-- AC:END -->
