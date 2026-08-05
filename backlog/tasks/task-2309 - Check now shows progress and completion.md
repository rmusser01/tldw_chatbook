---
id: TASK-2309
title: Check now shows progress and completion
status: To Do
assignee: []
created_date: '2026-08-04'
labels:
  - watchlists
  - ux
  - uat-2026-08-04
dependencies: []
priority: medium
---

## Description (the why)

UAT: pressing "Check now" produces ~5 seconds of dead air — no progress
indicator, no completion signal, nothing preventing a confused second click
from queuing duplicate work.

UAT finding F19.

## Acceptance Criteria (the what)

- [ ] Triggering a check gives immediate visible acknowledgment, a busy
      state while running, and a completion signal (including the failure
      case).
- [ ] A second activation while a check runs is debounced or explicitly
      queued, never silently duplicated.
